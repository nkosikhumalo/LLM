package inference

import (
	"fmt"
	"math"
)

// KVCache stores past Key/Value rows for one Transformer layer during decoding.
type KVCache struct {
	Keys   [][]float64
	Values [][]float64
}

// Append stores copies of the newest key and value rows.
func (c *KVCache) Append(key, value []float64) {
	c.Keys = append(c.Keys, append([]float64(nil), key...))
	c.Values = append(c.Values, append([]float64(nil), value...))
}

// Len returns how many cached positions this layer holds.
func (c *KVCache) Len() int {
	return len(c.Keys)
}

// Reset clears cached keys and values.
func (c *KVCache) Reset() {
	c.Keys = nil
	c.Values = nil
}

// Session holds per-layer KV caches for incremental next-token inference.
type Session struct {
	model  *Model
	caches []*KVCache
	length int
}

// NewSession creates an empty decoding session for this model.
func (m *Model) NewSession() *Session {
	caches := make([]*KVCache, m.Export.Config.NLayers)
	for i := range caches {
		caches[i] = &KVCache{}
	}
	return &Session{model: m, caches: caches}
}

// Len is the number of tokens already consumed by the session.
func (s *Session) Len() int {
	return s.length
}

// Reset clears all layer caches and the sequence length.
func (s *Session) Reset() {
	for _, cache := range s.caches {
		cache.Reset()
	}
	s.length = 0
}

// Prefill processes a full prompt and returns logits after the last token.
func (s *Session) Prefill(ids []int) ([]float64, error) {
	config := s.model.Export.Config
	if len(ids) == 0 {
		return nil, fmt.Errorf("prompt is empty")
	}
	if len(ids) > config.MaxSeqLen {
		return nil, fmt.Errorf("context length %d exceeds max_seq_len %d", len(ids), config.MaxSeqLen)
	}
	for _, id := range ids {
		if id < 0 || id >= config.VocabSize {
			return nil, fmt.Errorf("invalid token ID %d", id)
		}
	}

	s.Reset()
	var logits []float64
	var err error
	for _, id := range ids {
		logits, err = s.Step(id)
		if err != nil {
			return nil, err
		}
	}
	return logits, nil
}

// Step appends one token using cached K/V and returns next-token logits.
func (s *Session) Step(tokenID int) ([]float64, error) {
	config := s.model.Export.Config
	weights := s.model.Export.Weights
	if tokenID < 0 || tokenID >= config.VocabSize {
		return nil, fmt.Errorf("invalid token ID %d", tokenID)
	}
	if s.length >= config.MaxSeqLen {
		return nil, fmt.Errorf("context length %d exceeds max_seq_len %d", s.length+1, config.MaxSeqLen)
	}

	position := s.length
	x := make([]float64, config.DModel)
	tokenRow := s.model.embedding("token_embedding", tokenID, config.DModel, weights.TokenEmbedding)
	positionRow := s.model.embedding("position_embedding", position, config.DModel, weights.PositionEmbedding)
	for d := range x {
		x[d] = tokenRow[d] + positionRow[d]
	}

	heads := config.NHeads
	headDim := config.DModel / config.NHeads
	for layerIndex, layer := range weights.Layers {
		normed := rmsNormalize(x, layer.AttnNorm, config.RMSNormEpsilon)
		query := s.model.multiply(fmt.Sprintf("layers.%d.q", layerIndex), normed, layer.Q, config.DModel, config.DModel)
		key := s.model.multiply(fmt.Sprintf("layers.%d.k", layerIndex), normed, layer.K, config.DModel, config.DModel)
		value := s.model.multiply(fmt.Sprintf("layers.%d.v", layerIndex), normed, layer.V, config.DModel, config.DModel)
		cache := s.caches[layerIndex]
		cache.Append(key, value)

		joined := make([]float64, config.DModel)
		past := cache.Len()
		for h := 0; h < heads; h++ {
			scores := make([]float64, past)
			for t := 0; t < past; t++ {
				for d := 0; d < headDim; d++ {
					scores[t] += query[h*headDim+d] * cache.Keys[t][h*headDim+d]
				}
				scores[t] /= math.Sqrt(float64(headDim))
			}
			softmax(scores)
			for t, weight := range scores {
				for d := 0; d < headDim; d++ {
					joined[h*headDim+d] += weight * cache.Values[t][h*headDim+d]
				}
			}
		}
		attnOut := s.model.multiply(fmt.Sprintf("layers.%d.o", layerIndex), joined, layer.O, config.DModel, config.DModel)
		for d := range x {
			x[d] += attnOut[d]
		}

		hidden := s.model.multiply(fmt.Sprintf("layers.%d.ffn_in", layerIndex), rmsNormalize(x, layer.FFNNorm, config.RMSNormEpsilon), layer.FFNIn, config.DModel, config.DFFN)
		for i := range hidden {
			hidden[i] = gelu(hidden[i])
		}
		ffnOut := s.model.multiply(fmt.Sprintf("layers.%d.ffn_out", layerIndex), hidden, layer.FFNOut, config.DFFN, config.DModel)
		for d := range x {
			x[d] += ffnOut[d]
		}
	}

	s.length++
	return s.model.multiply("output", rmsNormalize(x, weights.FinalNorm, config.RMSNormEpsilon), weights.Output, config.DModel, config.VocabSize), nil
}
