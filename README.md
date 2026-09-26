# MiniLLM + Quantization Engine

One local system for training a small Transformer, compressing its checkpoints, and measuring the quality/size tradeoff. It has two quantization paths: post-training quantization (PTQ) and quantization-aware training (QAT).

## System flow

```text
text corpus → tokenizer → Java causal trainer → FP32 checkpoint
                                             ├─ PTQ → PTQ INT8 checkpoint
                                             └─ fake-quant fine-tuning → QAT INT8 checkpoint
FP32 + PTQ + QAT checkpoints → held-out benchmark → JSON + terminal comparison
```

The source model is trained in Java. Go validates the shared checkpoint, implements PTQ and dequantization, orchestrates QAT with the Java trainer, and benchmarks all three model variants.

## Quick start

Requirements: Go 1.22+, Java 21+, and Maven. The end-to-end command accepts a plain UTF-8 text file or a directory containing `.txt` files; it prepares tokens and runs training, PTQ, QAT, and benchmarking in sequence.

```bash
# Interactive: prompts for the dataset path and optional held-out text
./start.sh

# Or provide paths directly
./start.sh run --data /path/to/my-corpus.txt --epochs 100 --qat-epochs 2

# Better evaluation: provide a separate held-out text file
./start.sh run --data /path/to/train.txt --eval /path/to/heldout.txt
```

Artifacts are written under `models/runs/<dataset-name>/`: tokenized data, FP32, PTQ INT8, QAT INT8, and `benchmark.json`. At completion, the launcher prints the exact paths to both compressed checkpoints and the report. The dataset path may be relative to the directory where `start.sh` is invoked. For a directory input, `.txt` files are read in sorted order. Without `--eval`, the benchmark scores the training corpus, so that run is an integration check rather than a generalization measurement.

The lower-level commands remain available:

```bash
# PTQ: quantize an existing checkpoint without training
./start.sh ptq --input ../models/exported/model.json \
  --output ../models/quantized/model.ptq.int8.json --scheme per-channel

# QAT: load that checkpoint, fine-tune with fake-quantized weights, then emit INT8
./start.sh qat --input ../models/exported/model.json \
  --tokens ../data/tokenized/tokens.json --vocab ../data/tokenized/vocab.json \
  --output ../models/quantized/model.qat.int8.json --epochs 2

# Compare all three on the same held-out text, one sequence per line
./start.sh benchmark --fp32 ../models/exported/model.json \
  --ptq ../models/quantized/model.ptq.int8.json \
  --qat ../models/quantized/model.qat.int8.json \
  --eval ../data/eval/heldout.txt --vocab ../data/tokenized/vocab.json \
  --report ../models/quantized/benchmark.json
```

For `run`, dataset and evaluation paths are relative to the caller’s current directory. Other low-level command flags are interpreted from the `go/` directory. `./start.sh` with no arguments prompts for a dataset and runs training, PTQ, QAT, and benchmarking. Use `./start.sh ptq` for per-channel PTQ of an existing checkpoint. `./start.sh test` runs Go and Java tests.

To train a source model from scratch, tokenize a plain text corpus and use the Java trainer:

```bash
./scripts/tokenize.sh
./scripts/train.sh --epochs 100 --val-fraction 0.1 --patience 10
```

## PTQ and QAT

PTQ uses symmetric signed INT8 weights with a zero point of zero. Per-tensor mode uses one scale per tensor; per-channel mode uses one scale per first-dimension channel. Optional calibration stores observed activation ranges, but activation quantization is not implemented.

QAT imports the FP32 checkpoint into the Java model, continues training on causal text windows, and fake-quantizes weights during forward passes. The straight-through estimator sends gradients to the underlying FP32 weights. The resulting fine-tuned weights are then quantized using the same PTQ encoder. QAT artifacts record their method and epoch count in checkpoint metadata.

## Benchmark and current limits

The benchmark compares file size, next-token perplexity, tokens/second, peak Go heap, and dequantized weight size. Use a held-out dataset; a training corpus is only useful as an integration smoke test.

INT8 checkpoints are currently dequantized to float64 before inference. The engine demonstrates checkpoint compression and quality effects; integer matrix kernels, activation quantization, and INT8 inference speedups are not implemented. Its input format is the project’s version 1 JSON export (`format: "microllm"`); ONNX, GGUF, and arbitrary Hugging Face checkpoints are not supported.

## Main code

| Path | Role |
| --- | --- |
| `go/cmd/quantize` | PTQ and optional calibration |
| `go/cmd/qat` | Checkpoint bridge, Java QAT run, final INT8 export |
| `go/cmd/benchmark` | FP32/PTQ/QAT evaluation and JSON report |
| `go/internal/quantization` | PTQ, dequantization, checkpoint format and Java bridge |
| `java/.../model/Transformer.java` | Shared Transformer training and fake-quantized forward path |
| `java/.../train/Trainer.java` | Causal training and QAT fine-tuning loop |

See [Go commands](go/README.md), [architecture](docs/ARCHITECTURE.md), and [status](STATUS.md).
