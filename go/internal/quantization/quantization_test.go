package quantization

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestTensorRoundTrip(t *testing.T) {
	values := []float64{-2, -1, 0, 0.25, 1, 2}
	encoded, err := QuantizeTensor(values)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DequantizeTensor(encoded)
	if err != nil {
		t.Fatal(err)
	}
	for i, value := range values {
		if math.Abs(value-decoded[i]) > encoded.Scale/2+1e-12 {
			t.Fatalf("value %d: got %g want %g within scale/2", i, decoded[i], value)
		}
	}
}

func TestZeroTensor(t *testing.T) {
	encoded, err := QuantizeTensor([]float64{0, 0})
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DequantizeTensor(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if decoded[0] != 0 || decoded[1] != 0 {
		t.Fatalf("zero tensor changed: %v", decoded)
	}
}

func TestPerChannelRoundTrip(t *testing.T) {
	values := []float64{0.01, 0.02, 0.03, 2, 4, 6}
	encoded, err := quantizeChannels(values, []int{2, 3})
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DequantizeTensor(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if len(encoded.Scales) != 2 || encoded.Axis != 0 {
		t.Fatalf("bad channel metadata: %+v", encoded)
	}
	for i, value := range values {
		if math.Abs(value-decoded[i]) > encoded.Scales[i/3]/2+1e-12 {
			t.Fatalf("value %d: got %g want %g", i, decoded[i], value)
		}
	}
}

func TestSaveCreatesParentDirectory(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "checkpoint.int8.json")
	checkpoint := &Checkpoint{Format: "microllm-int8", Version: 1}
	if err := Save(path, checkpoint); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var decoded Checkpoint
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Format != checkpoint.Format || decoded.Version != checkpoint.Version {
		t.Fatalf("saved checkpoint metadata = %q v%d", decoded.Format, decoded.Version)
	}
}
