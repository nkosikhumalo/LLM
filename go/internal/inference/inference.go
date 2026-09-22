package inference

import (
	"fmt"
	"github.com/nkosikhumalo/microllm/internal/loader"
	"math"
)

type Model struct{ Export *loader.Export }

func New(e *loader.Export) (*Model, error) {
	if e == nil {
		return nil, fmt.Errorf("nil export")
	}
	if err := loader.Validate(e); err != nil {
		return nil, err
	}
	return &Model{e}, nil
}

// Logits runs a causal, pre-norm Transformer forward pass and returns next-token logits.
func (m *Model) Logits(ids []int) ([]float64, error) {
	c, w := m.Export.Config, m.Export.Weights
	if len(ids) == 0 {
		return nil, fmt.Errorf("prompt is empty")
	}
	if len(ids) > c.MaxSeqLen {
		return nil, fmt.Errorf("context length %d exceeds max_seq_len %d", len(ids), c.MaxSeqLen)
	}
	x := make([][]float64, len(ids))
	for p, id := range ids {
		if id < 0 || id >= c.VocabSize {
			return nil, fmt.Errorf("invalid token ID %d", id)
		}
		x[p] = make([]float64, c.DModel)
		for d := range x[p] {
			x[p][d] = w.TokenEmbedding[id*c.DModel+d] + w.PositionEmbedding[p*c.DModel+d]
		}
	}
	for _, l := range w.Layers {
		norm := make([][]float64, len(x))
		for p := range x {
			norm[p] = rmsNormalize(x[p], l.AttnNorm, c.RMSNormEpsilon)
		}
		q, k, v := make([][]float64, len(x)), make([][]float64, len(x)), make([][]float64, len(x))
		for p := range x {
			q[p] = multiplyRowVector(norm[p], l.Q, c.DModel, c.DModel)
			k[p] = multiplyRowVector(norm[p], l.K, c.DModel, c.DModel)
			v[p] = multiplyRowVector(norm[p], l.V, c.DModel, c.DModel)
		}
		heads, hd := c.NHeads, c.DModel/c.NHeads
		att := make([][]float64, len(x))
		for p := range x {
			joined := make([]float64, c.DModel)
			for h := 0; h < heads; h++ {
				scores := make([]float64, p+1)
				for t := 0; t <= p; t++ {
					for d := 0; d < hd; d++ {
						scores[t] += q[p][h*hd+d] * k[t][h*hd+d]
					}
					scores[t] /= math.Sqrt(float64(hd))
				}
				softmax(scores)
				for t, a := range scores {
					for d := 0; d < hd; d++ {
						joined[h*hd+d] += a * v[t][h*hd+d]
					}
				}
			}
			att[p] = multiplyRowVector(joined, l.O, c.DModel, c.DModel)
			for d := range x[p] {
				x[p][d] += att[p][d]
			}
		}
		for p := range x {
			h := multiplyRowVector(rmsNormalize(x[p], l.FFNNorm, c.RMSNormEpsilon), l.FFNIn, c.DModel, c.DFFN)
			for i := range h {
				h[i] = gelu(h[i])
			}
			out := multiplyRowVector(h, l.FFNOut, c.DFFN, c.DModel)
			for d := range x[p] {
				x[p][d] += out[d]
			}
		}
	}
	return multiplyRowVector(rmsNormalize(x[len(x)-1], w.FinalNorm, c.RMSNormEpsilon), w.Output, c.DModel, c.VocabSize), nil
}

// multiplyRowVector multiplies a row vector by a row-major matrix.
func multiplyRowVector(input, weights []float64, inputSize, outputSize int) []float64 {
	output := make([]float64, outputSize)
	for inputIndex, inputValue := range input {
		for outputIndex := 0; outputIndex < outputSize; outputIndex++ {
			output[outputIndex] += inputValue * weights[inputIndex*outputSize+outputIndex]
		}
	}
	return output
}

func rmsNormalize(values, scaleWeights []float64, epsilon float64) []float64 {
	if epsilon == 0 {
		epsilon = 1e-5
	}
	sumOfSquares := 0.0
	for _, value := range values {
		sumOfSquares += value * value
	}
	normalizer := 1 / math.Sqrt(sumOfSquares/float64(len(values))+epsilon)
	normalized := make([]float64, len(values))
	for index, value := range values {
		normalized[index] = value * normalizer * scaleWeights[index]
	}
	return normalized
}

func softmax(values []float64) {
	maximumValue := values[0]
	for _, value := range values {
		if value > maximumValue {
			maximumValue = value
		}
	}
	probabilitySum := 0.0
	for index, value := range values {
		values[index] = math.Exp(value - maximumValue)
		probabilitySum += values[index]
	}
	for index := range values {
		values[index] /= probabilitySum
	}
}

func gelu(value float64) float64 {
	return 0.5 * value * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(value+0.044715*value*value*value)))
}
