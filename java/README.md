# Java training and Go integration

Repository owner: [nkosikhumalo](https://github.com/nkosikhumalo).

The Java directory owns model configuration and model export. It writes the single hand-off file consumed by Go: `../models/exported/model.json`.

## What is implemented

- `model/ModelConfig.java`: validated immutable Transformer dimensions.
- `export/WeightExporter.java`: deterministic initialized weights and a versioned JSON exporter.
- `train/TrainMain.java`: reads Go's `vocab.json`, trains a correctly sized model from Go's `tokens.json` and exports `model.json`.
- `pom.xml`: Java 21 Maven build, tests, and an executable JAR manifest.

Training includes token/position embeddings, causal-attention Q/K/V/O (future-token mask), GELU FFN (`ffn_in` / `ffn_out`), RMSNorm scales, and the output head — all updated with AdamW. The trainer shuffles causal windows each epoch, can hold out a validation tail (`--val-fraction`), and early-stops on `--patience`.

## Shared contract with Go

Go creates `../data/tokenized/vocab.json`; Java counts its tokens and uses that exact count for `vocab_size`. Java writes this versioned envelope:

```json
{
  "format": "microllm",
  "version": 1,
  "config": {
    "vocab_size": 12,
    "d_model": 16,
    "n_layers": 1,
    "n_heads": 2,
    "d_ffn": 32,
    "max_seq_len": 64,
    "rms_norm_epsilon": 0.00001
  },
  "weights": { "...": "flattened numeric arrays" }
}
```

`WeightExporter` writes all matrices in row-major order, with the input dimension first. The Go loader validates every required shape before inference, including embeddings, attention projections, FFN projections, normalization scales, and output head. This prevents incompatible model files from being used silently.

## Build and run

From the repository root:

```bash
# 1. Tokenize the active Human/Bot corpus with Go.
cd go
go run ./cmd/tokenize -input ../data/raw/train.txt -output ../data/tokenized

# 2. Compile Java, train, and export a compatible model.
cd ../java
mvn test
java -cp target/classes com.microllm.train.TrainMain \
  --vocab ../data/tokenized/vocab.json \
  --tokens ../data/tokenized/tokens.json \
  --output ../models/exported/model.json \
  --epochs 1000 --learning-rate 0.001 \
  --val-fraction 0.1 --patience 50

# 3. Load the Java export and generate with Go.
cd ../go
go run ./cmd/generate \
  -model ../models/exported/model.json \
  -vocab ../data/tokenized/vocab.json \
  -prompt "hello" -tokens 32
```

## Verified integration

The current training corpus is `data/raw/train.txt`; each dialogue ends with `<|end|>`. Answer anchors are optional and disabled by default.

| Stage | Result |
| --- | --- |
| Go tokenization | 27 tokens, 12 vocabulary entries |
| Java build | `mvn test` passed |
| Java export | Created `models/exported/model.json` with `vocab_size=12` |
| Go generation | Loaded the Java export and streamed 12 tokens successfully |

## Next Java work

- Optional multi-layer preset + integration test (`n_layers = 2`).
- Broader finite-difference gradient checks (attention, RMSNorm).
- Checkpoints and `model.metadata.json`.
- Keep the JSON field names, row-major order, and shapes unchanged so Go stays compatible.
