package inference

import (
	"math"
	"testing"

	"github.com/nkosikhumalo/microllm/go/internal/loader"
)

func TestSessionMatchesFullLogits(t *testing.T) {
	model, err := New(tinyExport())
	if err != nil {
		t.Fatal(err)
	}
	ids := []int{0, 1, 2, 1}
	full, err := model.Logits(ids)
	if err != nil {
		t.Fatal(err)
	}
	session := model.NewSession()
	cached, err := session.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	if len(full) != len(cached) {
		t.Fatalf("logit length %d vs %d", len(full), len(cached))
	}
	for i := range full {
		if math.Abs(full[i]-cached[i]) > 1e-9 {
			t.Fatalf("logit %d: full=%g cached=%g", i, full[i], cached[i])
		}
	}
}

func TestPrefillRejectsInvalidPromptWithoutChangingSession(t *testing.T) {
	model, err := New(tinyExport())
	if err != nil {
		t.Fatal(err)
	}
	session := model.NewSession()
	if _, err := session.Prefill([]int{0, 1}); err != nil {
		t.Fatal(err)
	}

	for _, prompt := range [][]int{nil, {0, -1}, {0, 1, 2, 0, 1, 2, 0, 1, 2}} {
		if _, err := session.Prefill(prompt); err == nil {
			t.Fatalf("Prefill(%v) unexpectedly succeeded", prompt)
		}
		if session.Len() != 2 || session.caches[0].Len() != 2 {
			t.Fatalf("invalid prefill changed session: len=%d cache=%d", session.Len(), session.caches[0].Len())
		}
	}
}

func TestMultiLayerSessionMatchesFullLogits(t *testing.T) {
	export := tinyExport()
	layer := export.Weights.Layers[0]
	export.Weights.Layers = append(export.Weights.Layers, layer)
	export.Config.NLayers = 2
	model, err := New(export)
	if err != nil {
		t.Fatal(err)
	}
	ids := []int{2, 0, 1, 2}
	full, err := model.Logits(ids)
	if err != nil {
		t.Fatal(err)
	}
	cached, err := model.NewSession().Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	for i := range full {
		if math.Abs(full[i]-cached[i]) > 1e-9 {
			t.Fatalf("logit %d: full=%g cached=%g", i, full[i], cached[i])
		}
	}
}

func TestStepExtendsPrefill(t *testing.T) {
	model, err := New(tinyExport())
	if err != nil {
		t.Fatal(err)
	}
	session := model.NewSession()
	if _, err := session.Prefill([]int{0, 1}); err != nil {
		t.Fatal(err)
	}
	stepLogits, err := session.Step(2)
	if err != nil {
		t.Fatal(err)
	}
	full, err := model.Logits([]int{0, 1, 2})
	if err != nil {
		t.Fatal(err)
	}
	for i := range full {
		if math.Abs(full[i]-stepLogits[i]) > 1e-9 {
			t.Fatalf("after step logit %d mismatch", i)
		}
	}
	if session.Len() != 3 {
		t.Fatalf("session length = %d", session.Len())
	}
}

func tinyExport() *loader.Export {
	const (
		vocab   = 3
		dModel  = 2
		nLayers = 1
		nHeads  = 1
		dFFN    = 4
		maxSeq  = 8
	)
	ones := func(n int) []float64 {
		values := make([]float64, n)
		for i := range values {
			values[i] = 1
		}
		return values
	}
	fill := func(n int, scale float64) []float64 {
		values := make([]float64, n)
		for i := range values {
			values[i] = scale * float64((i%5)-2) / 2
		}
		return values
	}
	return &loader.Export{
		Format:  "microllm",
		Version: 1,
		Config: loader.Config{
			VocabSize:      vocab,
			DModel:         dModel,
			NLayers:        nLayers,
			NHeads:         nHeads,
			DFFN:           dFFN,
			MaxSeqLen:      maxSeq,
			RMSNormEpsilon: 1e-5,
		},
		Weights: loader.Weights{
			TokenEmbedding:    fill(vocab*dModel, 0.1),
			PositionEmbedding: fill(maxSeq*dModel, 0.05),
			Layers: []loader.LayerWeights{{
				AttnNorm: ones(dModel),
				Q:        fill(dModel*dModel, 0.2),
				K:        fill(dModel*dModel, 0.15),
				V:        fill(dModel*dModel, 0.1),
				O:        fill(dModel*dModel, 0.2),
				FFNNorm:  ones(dModel),
				FFNIn:    fill(dModel*dFFN, 0.1),
				FFNOut:   fill(dFFN*dModel, 0.1),
			}},
			FinalNorm: ones(dModel),
			Output:    fill(dModel*vocab, 0.2),
		},
	}
}

func TestQuantizedMatrixInferenceMatchesDequantizedPath(t *testing.T) {
	fp32Export := tinyExport()
	floatModel, err := New(fp32Export)
	if err != nil {
		t.Fatal(err)
	}
	input := []int{0, 1, 2, 1}
	want, err := floatModel.Logits(input)
	if err != nil {
		t.Fatal(err)
	}

	quantizedExport := tinyExport()
	matrices := make(map[string]QuantizedMatrix)
	for name, values := range matrixRefs(quantizedExport) {
		rows, columns := testMatrixShape(name, quantizedExport.Config)
		maxAbs := 0.0
		for _, value := range *values {
			maxAbs = math.Max(maxAbs, math.Abs(value))
		}
		scale := 1.0
		if maxAbs > 0 {
			scale = maxAbs / 127
		}
		data := make([]int8, len(*values))
		for i, value := range *values {
			data[i] = int8(math.Round(value / scale))
		}
		matrices[name] = QuantizedMatrix{Shape: []int{rows, columns}, Data: data, Scale: scale}
	}
	quantizedModel, err := NewQuantized(quantizedExport, matrices)
	if err != nil {
		t.Fatal(err)
	}
	if quantizedModel.Export.Weights.TokenEmbedding != nil || quantizedModel.Export.Weights.Layers[0].Q != nil {
		t.Fatal("float matrix weights remain allocated after loading quantized model")
	}
	got, err := quantizedModel.Logits(input)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if math.Abs(want[i]-got[i]) > 0.03 {
			t.Fatalf("logit %d: FP32=%g quantized=%g", i, want[i], got[i])
		}
	}
	cached, err := quantizedModel.NewSession().Prefill(input)
	if err != nil {
		t.Fatal(err)
	}
	for i := range got {
		if math.Abs(got[i]-cached[i]) > 1e-9 {
			t.Fatalf("cached quantized logit %d differs", i)
		}
	}
}

func testMatrixShape(name string, c loader.Config) (int, int) {
	switch name {
	case "token_embedding":
		return c.VocabSize, c.DModel
	case "position_embedding":
		return c.MaxSeqLen, c.DModel
	case "output":
		return c.DModel, c.VocabSize
	case "layers.0.q", "layers.0.k", "layers.0.v", "layers.0.o":
		return c.DModel, c.DModel
	case "layers.0.ffn_in":
		return c.DModel, c.DFFN
	case "layers.0.ffn_out":
		return c.DFFN, c.DModel
	default:
		panic("unexpected test matrix " + name)
	}
}
