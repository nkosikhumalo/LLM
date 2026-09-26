package quantization

import (
	"bytes"
	"encoding/binary"
	"testing"

	"github.com/nkosikhumalo/microllm/go/internal/loader"
)

func TestWriteJavaBridgeUsesTrainerTensorOrder(t *testing.T) {
	const vocab, width, layers, ffn = 2, 2, 1, 2
	fill := func(size int, value float64) []float64 {
		values := make([]float64, size)
		for i := range values {
			values[i] = value + float64(i)
		}
		return values
	}
	model := &loader.Export{
		Format: "microllm", Version: 1,
		Config: loader.Config{VocabSize: vocab, DModel: width, NLayers: layers, NHeads: 1, DFFN: ffn, MaxSeqLen: 3, RMSNormEpsilon: 1e-5},
		Weights: loader.Weights{
			TokenEmbedding: fill(vocab*width, 10), PositionEmbedding: fill(3*width, 20),
			FinalNorm: fill(width, 30), Output: fill(width*vocab, 40),
			Layers: []loader.LayerWeights{{AttnNorm: fill(width, 50), Q: fill(width*width, 60), K: fill(width*width, 70), V: fill(width*width, 80), O: fill(width*width, 90), FFNNorm: fill(width, 100), FFNIn: fill(width*ffn, 110), FFNOut: fill(ffn*width, 120)}},
		},
	}
	var encoded bytes.Buffer
	if err := WriteJavaBridge(&encoded, model); err != nil {
		t.Fatal(err)
	}
	if got := string(encoded.Bytes()[:8]); got != "MINIQAT1" {
		t.Fatalf("bridge magic %q", got)
	}
	reader := bytes.NewReader(encoded.Bytes()[8:])
	var firstDimension int32
	if err := binary.Read(reader, binary.BigEndian, &firstDimension); err != nil {
		t.Fatal(err)
	}
	if firstDimension != vocab {
		t.Fatalf("first dimension %d, want vocabulary %d", firstDimension, vocab)
	}
	for i := 0; i < 5; i++ {
		var ignored int32
		if err := binary.Read(reader, binary.BigEndian, &ignored); err != nil {
			t.Fatal(err)
		}
	}
	var epsilon float64
	if err := binary.Read(reader, binary.BigEndian, &epsilon); err != nil {
		t.Fatal(err)
	}
	var tensorCount int32
	if err := binary.Read(reader, binary.BigEndian, &tensorCount); err != nil {
		t.Fatal(err)
	}
	if tensorCount != 12 {
		t.Fatalf("tensor count %d, want 12", tensorCount)
	}
	var firstTensorSize int32
	if err := binary.Read(reader, binary.BigEndian, &firstTensorSize); err != nil {
		t.Fatal(err)
	}
	if firstTensorSize != vocab*width {
		t.Fatalf("first tensor size %d", firstTensorSize)
	}
	var firstValue float64
	if err := binary.Read(reader, binary.BigEndian, &firstValue); err != nil {
		t.Fatal(err)
	}
	if firstValue != 10 {
		t.Fatalf("first tensor value %g, want token embedding 10", firstValue)
	}
}
