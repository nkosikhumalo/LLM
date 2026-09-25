# Go inference and data-preparation engine

Repository owner: [nkosikhumalo](https://github.com/nkosikhumalo).

This directory is the Go half of MicroLLM:

| Path | Role |
|------|------|
| `cmd/tokenize` | Build a greedy wordpiece vocabulary + `tokens.json` from `data/raw/train.txt` by default |
| `cmd/generate` | One-shot generation with **KV-cache** session |
| `cmd/chat` | Multi-turn terminal chat (Human/Bot prefixes) |
| `internal/tokenizer` | Encode / decode / save / load vocab |
| `internal/loader` | Validate and load Java `model.json` |
| `internal/inference` | Full `Logits` + incremental `Session` (KV-cache) |
| `internal/sampler` | Temperature, top-k, top-p, greedy |
| `internal/cli` | Prompt + streaming helpers |

## Run

From this directory (or use `../scripts/*.sh` from the repo root):

```bash
go test ./...

go run ./cmd/tokenize -input ../data/raw/train.txt -output ../data/tokenized

go run ./cmd/generate \
  -model ../models/exported/model.json \
  -vocab ../data/tokenized/vocab.json \
  -prompt "hello " -tokens 64

go run ./cmd/chat \
  -model ../models/exported/model.json \
  -vocab ../data/tokenized/vocab.json
```

Root helpers: `scripts/tokenize.sh`, `scripts/generate.sh`, `scripts/chat.sh`.

## KV-cache

`Model.NewSession()` keeps per-layer Key/Value rows. `Prefill(ids)` warms the cache; each `Step(id)` only runs the new position. `Logits(ids)` still recomputes the full sequence (used in tests to prove cache matches).

## Java-to-Go model contract

`cmd/generate` / `cmd/chat` expect `models/exported/model.json`. Required envelope:

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
  "weights": { "...": "row-major float arrays" }
}
```

Train and generate with the **same** `vocab.json`.

## Tests

```bash
go test ./...
```

Covers tokenizer round-trip, loader validation, sampler greedy/top-k, and KV-cache vs full forward equality.

## Notes

- The default input is `data/raw/train.txt`; pass a `.txt` directory to tokenize multiple files. JSON corpora must be converted first.
- Chat quality depends on the training corpus using the same `Human:` / `Bot:` style you use at inference time.
- See [`../STATUS.md`](../STATUS.md) for the full project status.
