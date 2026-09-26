// Package quantization implements symmetric per-tensor and per-channel INT8 checkpoint quantization.
package quantization

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/nkosikhumalo/microllm/go/internal/inference"
	"github.com/nkosikhumalo/microllm/go/internal/loader"
)

type Tensor struct {
	Shape      []int     `json:"shape"`
	Data       string    `json:"data"` // base64 encoded signed int8 bytes
	Scale      float64   `json:"scale"`
	ZeroPoint  int       `json:"zero_point"`
	Scales     []float64 `json:"scales,omitempty"`
	ZeroPoints []int     `json:"zero_points,omitempty"`
	Axis       int       `json:"axis,omitempty"`
}
type Checkpoint struct {
	Format           string                               `json:"format"`
	Version          int                                  `json:"version"`
	Method           string                               `json:"quantization_method,omitempty"`
	QATEpochs        int                                  `json:"qat_epochs,omitempty"`
	Config           loader.Config                        `json:"config"`
	Weights          map[string]Tensor                    `json:"weights"`
	ActivationRanges map[string]inference.ActivationRange `json:"activation_ranges,omitempty"`
}

// QuantizeTensor maps values to [-127,127], reserving -128 and using a symmetric zero point.
func QuantizeTensor(values []float64) (Tensor, error) {
	if len(values) == 0 {
		return Tensor{}, fmt.Errorf("cannot quantize an empty tensor")
	}
	maxAbs := 0.0
	for _, v := range values {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return Tensor{}, fmt.Errorf("tensor contains a non-finite value")
		}
		if math.Abs(v) > maxAbs {
			maxAbs = math.Abs(v)
		}
	}
	scale := 1.0
	if maxAbs > 0 {
		scale = maxAbs / 127
	}
	raw := make([]byte, len(values))
	for i, v := range values {
		q := int(math.Round(v / scale))
		if q > 127 {
			q = 127
		}
		if q < -127 {
			q = -127
		}
		raw[i] = byte(int8(q))
	}
	return Tensor{Shape: []int{len(values)}, Data: base64.StdEncoding.EncodeToString(raw), Scale: scale, ZeroPoint: 0}, nil
}
func DequantizeTensor(t Tensor) ([]float64, error) {
	raw, err := base64.StdEncoding.DecodeString(t.Data)
	if err != nil {
		return nil, err
	}
	n := 1
	for _, d := range t.Shape {
		if d < 0 {
			return nil, fmt.Errorf("negative tensor dimension")
		}
		n *= d
	}
	if n != len(raw) {
		return nil, fmt.Errorf("tensor shape has %d values, data has %d", n, len(raw))
	}
	perChannel := len(t.Scales) > 0
	if perChannel && (len(t.Scales) != len(t.ZeroPoints) || t.Axis < 0 || t.Axis >= len(t.Shape)) {
		return nil, fmt.Errorf("invalid per-channel scale metadata")
	}
	stride := 1
	if perChannel {
		for _, d := range t.Shape[t.Axis+1:] {
			stride *= d
		}
		if t.Shape[t.Axis] != len(t.Scales) {
			return nil, fmt.Errorf("channel count does not match tensor shape")
		}
	}
	out := make([]float64, len(raw))
	for i, b := range raw {
		scale, zp := t.Scale, t.ZeroPoint
		if perChannel {
			ch := (i / stride) % t.Shape[t.Axis]
			scale, zp = t.Scales[ch], t.ZeroPoints[ch]
		}
		if !(scale > 0) || math.IsNaN(scale) || math.IsInf(scale, 0) || zp != 0 {
			return nil, fmt.Errorf("invalid scale or zero point")
		}
		out[i] = float64(int8(b)-int8(zp)) * scale
	}
	return out, nil
}

func quantizeChannels(values []float64, shape []int) (Tensor, error) {
	if len(shape) == 0 || len(values) == 0 {
		return Tensor{}, fmt.Errorf("invalid empty tensor")
	}
	channels, stride := shape[0], 1
	for _, d := range shape[1:] {
		stride *= d
	}
	if channels*stride != len(values) {
		return Tensor{}, fmt.Errorf("shape does not match values")
	}
	data := make([]byte, len(values))
	scales := make([]float64, channels)
	zeros := make([]int, channels)
	for ch := 0; ch < channels; ch++ {
		row := values[ch*stride : (ch+1)*stride]
		maxAbs := 0.0
		for _, v := range row {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return Tensor{}, fmt.Errorf("tensor contains a non-finite value")
			}
			if math.Abs(v) > maxAbs {
				maxAbs = math.Abs(v)
			}
		}
		scale := 1.0
		if maxAbs > 0 {
			scale = maxAbs / 127
		}
		scales[ch] = scale
		for i, v := range row {
			q := int(math.Round(v / scale))
			if q > 127 {
				q = 127
			}
			if q < -127 {
				q = -127
			}
			data[ch*stride+i] = byte(int8(q))
		}
	}
	return Tensor{Shape: append([]int(nil), shape...), Data: base64.StdEncoding.EncodeToString(data), Scales: scales, ZeroPoints: zeros, Axis: 0}, nil
}

func tensors(e *loader.Export) map[string]*[]float64 {
	m := map[string]*[]float64{"token_embedding": &e.Weights.TokenEmbedding, "position_embedding": &e.Weights.PositionEmbedding, "final_norm": &e.Weights.FinalNorm, "output": &e.Weights.Output}
	for i := range e.Weights.Layers {
		l := &e.Weights.Layers[i]
		p := fmt.Sprintf("layers.%d.", i)
		m[p+"attn_norm"] = &l.AttnNorm
		m[p+"q"] = &l.Q
		m[p+"k"] = &l.K
		m[p+"v"] = &l.V
		m[p+"o"] = &l.O
		m[p+"ffn_norm"] = &l.FFNNorm
		m[p+"ffn_in"] = &l.FFNIn
		m[p+"ffn_out"] = &l.FFNOut
	}
	return m
}
func Quantize(e *loader.Export) (*Checkpoint, error) { return QuantizeWithScheme(e, "per-tensor") }

func QuantizeWithScheme(e *loader.Export, scheme string) (*Checkpoint, error) {
	if err := loader.Validate(e); err != nil {
		return nil, err
	}
	if scheme != "per-tensor" && scheme != "per-channel" {
		return nil, fmt.Errorf("unknown quantization scheme %q", scheme)
	}
	c := &Checkpoint{Format: "microllm-int8", Version: 1, Method: "ptq", Config: e.Config, Weights: map[string]Tensor{}}
	for name, p := range tensors(e) {
		shape := tensorShape(e.Config, name)
		var t Tensor
		var err error
		if scheme == "per-channel" {
			t, err = quantizeChannels(*p, shape)
		} else {
			t, err = QuantizeTensor(*p)
		}
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		t.Shape = shape
		c.Weights[name] = t
	}
	return c, nil
}
func tensorShape(c loader.Config, name string) []int {
	switch name {
	case "token_embedding":
		return []int{c.VocabSize, c.DModel}
	case "position_embedding":
		return []int{c.MaxSeqLen, c.DModel}
	case "final_norm":
		return []int{c.DModel}
	case "output":
		return []int{c.DModel, c.VocabSize}
	}
	switch name[strings.LastIndex(name, ".")+1:] {
	case "attn_norm", "ffn_norm":
		return []int{c.DModel}
	case "q", "k", "v", "o":
		return []int{c.DModel, c.DModel}
	case "ffn_in":
		return []int{c.DModel, c.DFFN}
	case "ffn_out":
		return []int{c.DFFN, c.DModel}
	default:
		return nil
	}
}

func equalShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// InferenceWeights decodes matrix weights as signed INT8 without dequantizing them.
func (c *Checkpoint) InferenceWeights() (map[string]inference.QuantizedMatrix, error) {
	weights := make(map[string]inference.QuantizedMatrix)
	for name, tensor := range c.Weights {
		if len(tensor.Shape) != 2 {
			continue
		}
		raw, err := base64.StdEncoding.DecodeString(tensor.Data)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		data := make([]int8, len(raw))
		for i, value := range raw {
			data[i] = int8(value)
		}
		weights[name] = inference.QuantizedMatrix{
			Shape: append([]int(nil), tensor.Shape...), Data: data,
			Scale: tensor.Scale, ZeroPoint: tensor.ZeroPoint,
			Scales:     append([]float64(nil), tensor.Scales...),
			ZeroPoints: append([]int(nil), tensor.ZeroPoints...), Axis: tensor.Axis,
		}
	}
	return weights, nil
}

func (c *Checkpoint) Dequantize() (*loader.Export, error) {
	if c.Format != "microllm-int8" || c.Version != 1 {
		return nil, fmt.Errorf("unsupported quantized format %q version %d", c.Format, c.Version)
	}
	e := &loader.Export{Format: "microllm", Version: 1, Config: c.Config, Weights: loader.Weights{Layers: make([]loader.LayerWeights, c.Config.NLayers)}}
	for name, p := range tensors(e) {
		t, ok := c.Weights[name]
		if !ok {
			return nil, fmt.Errorf("missing tensor %s", name)
		}
		if !equalShape(t.Shape, tensorShape(c.Config, name)) {
			return nil, fmt.Errorf("%s has invalid shape %v", name, t.Shape)
		}
		v, err := DequantizeTensor(t)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		*p = v
	}
	if len(c.Weights) != len(tensors(e)) {
		return nil, fmt.Errorf("unexpected tensor in quantized checkpoint")
	}
	return e, loader.Validate(e)
}
func Save(path string, c *Checkpoint) error {
	b, err := json.Marshal(c)
	if err != nil {
		return err
	}
	if dir := filepath.Dir(path); dir != "." {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return err
		}
	}
	return os.WriteFile(path, b, 0644)
}
func Load(path string) (*Checkpoint, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var c Checkpoint
	if err = json.Unmarshal(b, &c); err != nil {
		return nil, err
	}
	_, err = c.Dequantize()
	return &c, err
}

// LoadInferenceModel retains INT8 matrix weights in their packed representation.
func LoadInferenceModel(path string) (*inference.Model, error) {
	encoded, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var header struct {
		Format string `json:"format"`
	}
	if err := json.Unmarshal(encoded, &header); err != nil {
		return nil, err
	}
	if header.Format != "microllm-int8" {
		export, err := loader.Load(path)
		if err != nil {
			return nil, err
		}
		return inference.New(export)
	}
	var checkpoint Checkpoint
	if err := json.Unmarshal(encoded, &checkpoint); err != nil {
		return nil, err
	}
	export, err := checkpoint.Dequantize()
	if err != nil {
		return nil, err
	}
	weights, err := checkpoint.InferenceWeights()
	if err != nil {
		return nil, err
	}
	return inference.NewQuantized(export, weights)
}

func LoadModel(path string) (*loader.Export, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var h struct {
		Format string `json:"format"`
	}
	if err = json.Unmarshal(b, &h); err != nil {
		return nil, err
	}
	if h.Format == "microllm-int8" {
		var c Checkpoint
		if err = json.Unmarshal(b, &c); err != nil {
			return nil, err
		}
		return c.Dequantize()
	}
	return loader.Load(path)
}
