package inference

import (
	"fmt"
	"github.com/nkosikhumalo/microllm/go/internal/loader"
	"math"
)

type QuantizedMatrix struct {
	Shape      []int
	Data       []int8
	Scale      float64
	ZeroPoint  int
	Scales     []float64
	ZeroPoints []int
	Axis       int
}

type Model struct {
	Export    *loader.Export
	quantized map[string]QuantizedMatrix
}

func New(e *loader.Export) (*Model, error) {
	if e == nil {
		return nil, fmt.Errorf("nil export")
	}
	if err := loader.Validate(e); err != nil {
		return nil, err
	}
	return &Model{Export: e}, nil
}

type ActivationRange struct {
	Min float64 `json:"min"`
	Max float64 `json:"max"`
}

// NewQuantized creates an inference model that retains rank-two weights as INT8.
// Small normalization vectors remain float64. Export is validated before its dense
// matrix slices are released, so inference reads those matrices only from INT8 storage.
func NewQuantized(e *loader.Export, weights map[string]QuantizedMatrix) (*Model, error) {
	if err := loader.Validate(e); err != nil {
		return nil, err
	}
	refs := matrixRefs(e)
	if len(weights) != len(refs) {
		return nil, fmt.Errorf("have %d quantized matrices; want %d", len(weights), len(refs))
	}
	for name, values := range refs {
		matrix, ok := weights[name]
		if !ok {
			return nil, fmt.Errorf("missing quantized matrix %s", name)
		}
		if len(matrix.Shape) != 2 || matrix.Shape[0]*matrix.Shape[1] != len(matrix.Data) || len(matrix.Data) != len(*values) {
			return nil, fmt.Errorf("%s has invalid quantized matrix shape", name)
		}
		if len(matrix.Scales) > 0 && (len(matrix.Scales) != len(matrix.ZeroPoints) || (matrix.Axis != 0 && matrix.Axis != 1) || len(matrix.Scales) != matrix.Shape[matrix.Axis]) {
			return nil, fmt.Errorf("%s has invalid per-channel metadata", name)
		}
		if len(matrix.Scales) == 0 && (!(matrix.Scale > 0) || matrix.ZeroPoint != 0) {
			return nil, fmt.Errorf("%s has invalid per-tensor metadata", name)
		}
	}
	for _, values := range refs {
		*values = nil
	}
	return &Model{Export: e, quantized: weights}, nil
}

func matrixRefs(e *loader.Export) map[string]*[]float64 {
	refs := map[string]*[]float64{"token_embedding": &e.Weights.TokenEmbedding, "position_embedding": &e.Weights.PositionEmbedding, "output": &e.Weights.Output}
	for index := range e.Weights.Layers {
		layer := &e.Weights.Layers[index]
		prefix := fmt.Sprintf("layers.%d.", index)
		refs[prefix+"q"] = &layer.Q
		refs[prefix+"k"] = &layer.K
		refs[prefix+"v"] = &layer.V
		refs[prefix+"o"] = &layer.O
		refs[prefix+"ffn_in"] = &layer.FFNIn
		refs[prefix+"ffn_out"] = &layer.FFNOut
	}
	return refs
}

func (m *Model) multiply(name string, input, fallback []float64, inputSize, outputSize int) []float64 {
	matrix, ok := m.quantized[name]
	if !ok {
		return multiplyRowVector(input, fallback, inputSize, outputSize)
	}
	output := make([]float64, outputSize)
	for i, value := range input {
		for j := 0; j < outputSize; j++ {
			index := i*outputSize + j
			scale, zero := matrix.Scale, matrix.ZeroPoint
			if len(matrix.Scales) > 0 {
				channel := i
				if matrix.Axis == 1 {
					channel = j
				}
				scale, zero = matrix.Scales[channel], matrix.ZeroPoints[channel]
			}
			output[j] += value * float64(int(matrix.Data[index])-zero) * scale
		}
	}
	return output
}

func (m *Model) embedding(name string, row, width int, fallback []float64) []float64 {
	matrix, ok := m.quantized[name]
	if !ok {
		return fallback[row*width : (row+1)*width]
	}
	values := make([]float64, width)
	for column := 0; column < width; column++ {
		index := row*width + column
		scale, zero := matrix.Scale, matrix.ZeroPoint
		if len(matrix.Scales) > 0 {
			channel := row
			if matrix.Axis == 1 {
				channel = column
			}
			scale, zero = matrix.Scales[channel], matrix.ZeroPoints[channel]
		}
		values[column] = float64(int(matrix.Data[index])-zero) * scale
	}
	return values
}

// Logits runs a causal, pre-norm Transformer forward pass and returns next-token logits.
func (m *Model) Logits(ids []int) ([]float64, error) { return m.LogitsWithActivationRanges(ids, nil) }

// LogitsWithActivationRanges optionally collects observed ranges for embedding,
// attention, feed-forward, residual, and output values while running inference.
func (m *Model) LogitsWithActivationRanges(ids []int, ranges map[string]ActivationRange) ([]float64, error) {
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
		tokenRow := m.embedding("token_embedding", id, c.DModel, w.TokenEmbedding)
		positionRow := m.embedding("position_embedding", p, c.DModel, w.PositionEmbedding)
		for d := range x[p] {
			x[p][d] = tokenRow[d] + positionRow[d]
		}
	}
	observeMatrix(ranges, "embedding", x)
	for layerIndex, l := range w.Layers {
		norm := make([][]float64, len(x))
		for p := range x {
			norm[p] = rmsNormalize(x[p], l.AttnNorm, c.RMSNormEpsilon)
		}
		observeMatrix(ranges, fmt.Sprintf("layer.%d.attention_norm", layerIndex), norm)
		q, k, v := make([][]float64, len(x)), make([][]float64, len(x)), make([][]float64, len(x))
		for p := range x {
			q[p] = m.multiply(fmt.Sprintf("layers.%d.q", layerIndex), norm[p], l.Q, c.DModel, c.DModel)
			k[p] = m.multiply(fmt.Sprintf("layers.%d.k", layerIndex), norm[p], l.K, c.DModel, c.DModel)
			v[p] = m.multiply(fmt.Sprintf("layers.%d.v", layerIndex), norm[p], l.V, c.DModel, c.DModel)
		}
		observeMatrix(ranges, fmt.Sprintf("layer.%d.q", layerIndex), q)
		observeMatrix(ranges, fmt.Sprintf("layer.%d.k", layerIndex), k)
		observeMatrix(ranges, fmt.Sprintf("layer.%d.v", layerIndex), v)
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
			att[p] = m.multiply(fmt.Sprintf("layers.%d.o", layerIndex), joined, l.O, c.DModel, c.DModel)
			for d := range x[p] {
				x[p][d] += att[p][d]
			}
		}
		observeMatrix(ranges, fmt.Sprintf("layer.%d.attention_residual", layerIndex), x)
		for p := range x {
			h := m.multiply(fmt.Sprintf("layers.%d.ffn_in", layerIndex), rmsNormalize(x[p], l.FFNNorm, c.RMSNormEpsilon), l.FFNIn, c.DModel, c.DFFN)
			for i := range h {
				h[i] = gelu(h[i])
			}
			observeVector(ranges, fmt.Sprintf("layer.%d.feed_forward", layerIndex), h)
			out := m.multiply(fmt.Sprintf("layers.%d.ffn_out", layerIndex), h, l.FFNOut, c.DFFN, c.DModel)
			for d := range x[p] {
				x[p][d] += out[d]
			}
		}
	}
	logits := m.multiply("output", rmsNormalize(x[len(x)-1], w.FinalNorm, c.RMSNormEpsilon), w.Output, c.DModel, c.VocabSize)
	observeVector(ranges, "output_logits", logits)
	return logits, nil
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

func observeVector(ranges map[string]ActivationRange, name string, values []float64) {
	if ranges == nil || len(values) == 0 {
		return
	}
	current, ok := ranges[name]
	if !ok {
		current = ActivationRange{Min: math.Inf(1), Max: math.Inf(-1)}
	}
	for _, v := range values {
		if v < current.Min {
			current.Min = v
		}
		if v > current.Max {
			current.Max = v
		}
	}
	ranges[name] = current
}
func observeMatrix(ranges map[string]ActivationRange, name string, values [][]float64) {
	for _, row := range values {
		observeVector(ranges, name, row)
	}
}
