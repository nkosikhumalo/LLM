// Package loader reads and validates exported MicroLLM model weights.
package loader

import (
	"encoding/json"
	"fmt"
	"os"
)

// Export is the versioned JSON hand-off contract written by Java's WeightExporter.
type Export struct {
	Format  string  `json:"format"`
	Version int     `json:"version"`
	Config  Config  `json:"config"`
	Weights Weights `json:"weights"`
}
type Config struct {
	VocabSize      int     `json:"vocab_size"`
	DModel         int     `json:"d_model"`
	NLayers        int     `json:"n_layers"`
	NHeads         int     `json:"n_heads"`
	DFFN           int     `json:"d_ffn"`
	MaxSeqLen      int     `json:"max_seq_len"`
	RMSNormEpsilon float64 `json:"rms_norm_epsilon"`
}
type Weights struct {
	TokenEmbedding    []float64      `json:"token_embedding"`
	PositionEmbedding []float64      `json:"position_embedding"`
	Layers            []LayerWeights `json:"layers"`
	FinalNorm         []float64      `json:"final_norm"`
	Output            []float64      `json:"output"`
}
type LayerWeights struct {
	AttnNorm []float64 `json:"attn_norm"`
	Q        []float64 `json:"q"`
	K        []float64 `json:"k"`
	V        []float64 `json:"v"`
	O        []float64 `json:"o"`
	FFNNorm  []float64 `json:"ffn_norm"`
	FFNIn    []float64 `json:"ffn_in"`
	FFNOut   []float64 `json:"ffn_out"`
}

func Load(path string) (*Export, error) {
	encodedExport, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var modelExport Export
	if err := json.Unmarshal(encodedExport, &modelExport); err != nil {
		return nil, fmt.Errorf("parse model export: %w", err)
	}
	if modelExport.Format != "microllm" || modelExport.Version != 1 {
		return nil, fmt.Errorf("unsupported export %q version %d", modelExport.Format, modelExport.Version)
	}
	return &modelExport, Validate(&modelExport)
}
func Validate(modelExport *Export) error {
	config := modelExport.Config
	if config.VocabSize < 1 || config.DModel < 1 || config.NLayers < 1 || config.NHeads < 1 || config.DFFN < 1 || config.MaxSeqLen < 1 || config.DModel%config.NHeads != 0 {
		return fmt.Errorf("invalid model config")
	}
	validateLength := func(name string, actual, expected int) error {
		if actual != expected {
			return fmt.Errorf("%s has %d values; want %d", name, actual, expected)
		}
		return nil
	}
	weights := modelExport.Weights
	if err := validateLength("token_embedding", len(weights.TokenEmbedding), config.VocabSize*config.DModel); err != nil {
		return err
	}
	if err := validateLength("position_embedding", len(weights.PositionEmbedding), config.MaxSeqLen*config.DModel); err != nil {
		return err
	}
	if err := validateLength("final_norm", len(weights.FinalNorm), config.DModel); err != nil {
		return err
	}
	if err := validateLength("output", len(weights.Output), config.DModel*config.VocabSize); err != nil {
		return err
	}
	if len(weights.Layers) != config.NLayers {
		return fmt.Errorf("have %d layers; want %d", len(weights.Layers), config.NLayers)
	}
	for layerIndex, layer := range weights.Layers {
		for _, matrix := range []struct {
			name             string
			actual, expected int
		}{{"attn_norm", len(layer.AttnNorm), config.DModel}, {"q", len(layer.Q), config.DModel * config.DModel}, {"k", len(layer.K), config.DModel * config.DModel}, {"v", len(layer.V), config.DModel * config.DModel}, {"o", len(layer.O), config.DModel * config.DModel}, {"ffn_norm", len(layer.FFNNorm), config.DModel}, {"ffn_in", len(layer.FFNIn), config.DModel * config.DFFN}, {"ffn_out", len(layer.FFNOut), config.DFFN * config.DModel}} {
			if err := validateLength(fmt.Sprintf("layer %d %s", layerIndex, matrix.name), matrix.actual, matrix.expected); err != nil {
				return err
			}
		}
	}
	return nil
}
