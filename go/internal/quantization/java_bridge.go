package quantization

import (
	"encoding/binary"
	"fmt"
	"io"

	"github.com/nkosikhumalo/microllm/go/internal/loader"
)

var javaBridgeMagic = []byte("MINIQAT1")

// WriteJavaBridge writes a validated checkpoint in the binary tensor order Java trains.
func WriteJavaBridge(output io.Writer, model *loader.Export) error {
	if err := loader.Validate(model); err != nil {
		return err
	}
	if _, err := output.Write(javaBridgeMagic); err != nil {
		return err
	}
	config := model.Config
	for _, dimension := range []int{config.VocabSize, config.DModel, config.NLayers, config.NHeads, config.DFFN, config.MaxSeqLen} {
		if err := binary.Write(output, binary.BigEndian, int32(dimension)); err != nil {
			return err
		}
	}
	if err := binary.Write(output, binary.BigEndian, config.RMSNormEpsilon); err != nil {
		return err
	}
	tensors := make([][]float64, 0, 4+config.NLayers*8)
	tensors = append(tensors, model.Weights.TokenEmbedding, model.Weights.PositionEmbedding, model.Weights.FinalNorm, model.Weights.Output)
	for _, layer := range model.Weights.Layers {
		tensors = append(tensors, layer.AttnNorm, layer.Q, layer.K, layer.V, layer.O, layer.FFNNorm, layer.FFNIn, layer.FFNOut)
	}
	if err := binary.Write(output, binary.BigEndian, int32(len(tensors))); err != nil {
		return err
	}
	for tensorIndex, tensor := range tensors {
		if err := binary.Write(output, binary.BigEndian, int32(len(tensor))); err != nil {
			return err
		}
		for _, value := range tensor {
			if err := binary.Write(output, binary.BigEndian, value); err != nil {
				return fmt.Errorf("write tensor %d: %w", tensorIndex, err)
			}
		}
	}
	return nil
}
