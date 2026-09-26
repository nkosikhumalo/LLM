# MiniLLM Quantization Engine (Go)

Go validates source checkpoints, implements PTQ, coordinates Java QAT, and evaluates FP32/PTQ/QAT models.

## PTQ

```bash
go run ./cmd/quantize \
  --input ../models/exported/model.json \
  --output ../models/quantized/model.ptq.int8.json \
  --scheme per-channel
```

`--scheme` supports `per-tensor` and `per-channel`. Optional `--calibration TEXT --vocab vocab.json` stores activation ranges as metadata only; it does not quantize activations.

## QAT

Builds/runs the Java trainer through the repository launcher:

```bash
cd ..
./start.sh qat --input ../models/exported/model.json \
  --tokens ../data/tokenized/tokens.json --vocab ../data/tokenized/vocab.json \
  --output ../models/quantized/model.qat.int8.json --epochs 2
```

From this module directory, the same command is `go run ./cmd/qat ...` after building `../java/target/microllm-train-0.1.0-SNAPSHOT.jar`. QAT loads pretrained weights, enables fake quantization during the Java training loop, then writes a real INT8 checkpoint.

## Three-way benchmark

```bash
go run ./cmd/benchmark \
  --fp32 ../models/exported/model.json \
  --ptq ../models/quantized/model.ptq.int8.json \
  --qat ../models/quantized/model.qat.int8.json \
  --eval ../data/eval/heldout.txt --vocab ../data/tokenized/vocab.json \
  --report ../models/quantized/benchmark.json
```

The comparison includes file size, next-token perplexity, tokens/second, sampled peak Go heap, and dequantized weight size. Provide held-out text to make perplexity differences meaningful.

## Implementation limits

INT8 values use symmetric scales and zero point zero. Per-channel mode scales along tensor dimension zero. Calibration records activation ranges only. The current inference path expands all checkpoint types to float64, so reported throughput is not an INT8-kernel performance claim and weight memory is the dequantized weight estimate.

Run `go test ./...` for Go tests. `./start.sh test` runs Go and Java tests.
