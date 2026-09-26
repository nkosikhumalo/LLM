package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"

	"github.com/nkosikhumalo/microllm/go/internal/loader"
	"github.com/nkosikhumalo/microllm/go/internal/quantization"
)

func main() {
	input := flag.String("input", "../models/exported/model.json", "trained FP32 checkpoint")
	tokens := flag.String("tokens", "../data/tokenized/tokens.json", "causal fine-tuning token IDs")
	vocab := flag.String("vocab", "../data/tokenized/vocab.json", "model vocabulary JSON")
	output := flag.String("output", "../models/quantized/model.qat.int8.json", "QAT INT8 checkpoint output")
	javaJar := flag.String("java-jar", "../java/target/microllm-train-0.1.0-SNAPSHOT.jar", "built Java trainer JAR")
	epochs := flag.Int("epochs", 2, "QAT fine-tuning epochs")
	learningRate := flag.Float64("learning-rate", 0.00005, "QAT fine-tuning learning rate")
	patience := flag.Int("patience", 2, "early-stopping patience")
	validation := flag.Float64("val-fraction", 0.1, "tail fraction used for validation")
	windows := flag.Int("windows-per-epoch", 0, "maximum training windows per epoch (0 uses all)")
	scheme := flag.String("scheme", "per-channel", "final quantization scheme: per-tensor or per-channel")
	flag.Parse()

	if *epochs < 1 || *learningRate <= 0 || *patience < 1 || *windows < 0 {
		fatal("epochs, learning rate, patience, and window limit must be positive (window limit may be zero)")
	}
	model, err := loader.Load(*input)
	if err != nil {
		fatal("load source checkpoint: %v", err)
	}
	workingDir, err := os.MkdirTemp("", "minillm-qat-")
	if err != nil {
		fatal("create temporary directory: %v", err)
	}
	defer os.RemoveAll(workingDir)
	bridgePath := filepath.Join(workingDir, "initial-weights.bin")
	bridge, err := os.Create(bridgePath)
	if err != nil {
		fatal("create Java weight bridge: %v", err)
	}
	if err := quantization.WriteJavaBridge(bridge, model); err != nil {
		bridge.Close()
		fatal("write Java weight bridge: %v", err)
	}
	if err := bridge.Close(); err != nil {
		fatal("close Java weight bridge: %v", err)
	}
	fineTunedPath := filepath.Join(workingDir, "qat-finetuned.json")
	arguments := []string{
		"-jar", *javaJar,
		"--vocab", *vocab,
		"--tokens", *tokens,
		"--output", fineTunedPath,
		"--init-weights", bridgePath,
		"--fake-quantization", "true",
		"--epochs", strconv.Itoa(*epochs),
		"--learning-rate", strconv.FormatFloat(*learningRate, 'g', -1, 64),
		"--patience", strconv.Itoa(*patience),
		"--val-fraction", strconv.FormatFloat(*validation, 'g', -1, 64),
		"--windows-per-epoch", strconv.Itoa(*windows),
	}
	cmd := exec.Command("java", arguments...)
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	if err := cmd.Run(); err != nil {
		fatal("QAT fine-tuning failed: %v", err)
	}
	fineTuned, err := loader.Load(fineTunedPath)
	if err != nil {
		fatal("load fine-tuned checkpoint: %v", err)
	}
	checkpoint, err := quantization.QuantizeWithScheme(fineTuned, *scheme)
	if err != nil {
		fatal("quantize fine-tuned checkpoint: %v", err)
	}
	checkpoint.Method = "qat"
	checkpoint.QATEpochs = *epochs
	if err := quantization.Save(*output, checkpoint); err != nil {
		fatal("save QAT checkpoint: %v", err)
	}
	fmt.Printf("QAT INT8 checkpoint ready: %s (%s, %d epochs)\n", *output, *scheme, *epochs)
}

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
