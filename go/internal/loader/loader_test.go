package loader_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/nkosikhumalo/microllm/go/internal/loader"
)

func TestValidateRejectsBadShape(t *testing.T) {
	export := &loader.Export{
		Format:  "microllm",
		Version: 1,
		Config: loader.Config{
			VocabSize: 2, DModel: 2, NLayers: 1, NHeads: 1, DFFN: 4, MaxSeqLen: 4, RMSNormEpsilon: 1e-5,
		},
		Weights: loader.Weights{
			TokenEmbedding:    make([]float64, 4),
			PositionEmbedding: make([]float64, 8),
			FinalNorm:         make([]float64, 2),
			Output:            make([]float64, 4),
			Layers: []loader.LayerWeights{{
				AttnNorm: make([]float64, 2),
				Q:        make([]float64, 4),
				K:        make([]float64, 4),
				V:        make([]float64, 4),
				O:        make([]float64, 4),
				FFNNorm:  make([]float64, 2),
				FFNIn:    make([]float64, 8),
				FFNOut:   make([]float64, 8),
			}},
		},
	}
	if err := loader.Validate(export); err != nil {
		t.Fatalf("expected valid export: %v", err)
	}
	export.Weights.TokenEmbedding = make([]float64, 3)
	if err := loader.Validate(export); err == nil {
		t.Fatal("expected shape error")
	}
}

func TestLoadRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.json")
	body := `{
  "format": "microllm",
  "version": 1,
  "config": {"vocab_size": 2, "d_model": 2, "n_layers": 1, "n_heads": 1, "d_ffn": 4, "max_seq_len": 4, "rms_norm_epsilon": 0.00001},
  "weights": {
    "token_embedding": [0,0,0,0],
    "position_embedding": [0,0,0,0,0,0,0,0],
    "layers": [{"attn_norm":[1,1],"q":[1,0,0,1],"k":[1,0,0,1],"v":[1,0,0,1],"o":[1,0,0,1],"ffn_norm":[1,1],"ffn_in":[0,0,0,0,0,0,0,0],"ffn_out":[0,0,0,0,0,0,0,0]}],
    "final_norm": [1,1],
    "output": [0,0,0,0]
  }
}`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	export, err := loader.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if export.Config.VocabSize != 2 || export.Config.NLayers != 1 {
		t.Fatalf("unexpected config: %#v", export.Config)
	}
}
