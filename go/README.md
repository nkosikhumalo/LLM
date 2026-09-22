# Go inference and data-preparation engine

Repository owner: [nkosikhumalo](https://github.com/nkosikhumalo).

This directory provides the Go half of MicroLLM's pipeline:

- `cmd/tokenize`: builds a deterministic UTF-8 character vocabulary and writes `vocab.json` plus `tokens.json` for Java training.
- `cmd/generate`: loads an exported model, samples next tokens, and writes them immediately to standard output.
- `internal/tokenizer`: stable vocabulary construction, UTF-8 encode/decode, and vocabulary persistence.
- `internal/loader`: validates the model hand-off file before inference.
- `internal/inference`: a causal, pre-RMSNorm Transformer forward pass (multi-head attention, GELU FFN, residuals, and output logits).
- `internal/sampler`: temperature, top-k, top-p, and greedy decoding.
- `internal/cli`: small prompt and streaming helpers.

## Run

Run commands from this directory so their default paths resolve to the repository's `data` and `models` folders:

```bash
go run ./cmd/tokenize -input ../data/raw -output ../data/tokenized
go run ./cmd/generate -model ../models/exported/model.json -vocab ../data/tokenized/vocab.json -prompt "Hello" -tokens 100
```

Use `go test ./...` to compile and check every package.

## Java-to-Go model contract

`cmd/generate` expects `models/exported/model.json`. Java's `WeightExporter` must produce this versioned JSON envelope:

```json
{
  "format": "microllm",
  "version": 1,
  "config": {
    "vocab_size": 256,
    "d_model": 64,
    "n_layers": 2,
    "n_heads": 4,
    "d_ffn": 256,
    "max_seq_len": 128,
    "rms_norm_epsilon": 0.00001
  },
  "weights": {
    "token_embedding": [],
    "position_embedding": [],
    "layers": [{"attn_norm": [], "q": [], "k": [], "v": [], "o": [], "ffn_norm": [], "ffn_in": [], "ffn_out": []}],
    "final_norm": [],
    "output": []
  }
}
```

All matrices are flattened row-major: input dimension first, output dimension second. Required sizes are validated on load:

- embeddings: `vocab_size × d_model` and `max_seq_len × d_model`
- Q/K/V/O: `d_model × d_model`
- FFN input/output: `d_model × d_ffn` and `d_ffn × d_model`
- output head: `d_model × vocab_size`

The vocabulary's token order is part of the model contract: train and generate with the same `vocab.json`.

## Current boundary

Java now exports a deterministic initialized `model.json` that Go loads successfully. The Java training stack is still scaffolded, so its generated text is not meaningful until the Java Transformer and training loop write trained weights using the same export contract. See [`../java/README.md`](../java/README.md).
