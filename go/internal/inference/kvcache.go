package inference

// KVCache is a reusable per-layer cache holder for incremental decoding sessions.
type KVCache struct{ Keys, Values [][]float64 }

func (c *KVCache) Append(k, v []float64) {
	c.Keys = append(c.Keys, append([]float64(nil), k...))
	c.Values = append(c.Values, append([]float64(nil), v...))
}
func (c *KVCache) Reset() { c.Keys = nil; c.Values = nil }
