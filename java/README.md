# Optional MiniLLM source-model trainer

This Java 21 module trains and exports the small model format consumed by MiniLLM Quantization Engine. It is a supporting source-checkpoint producer; the main product workflow starts from an existing compatible `model.json` and runs Go quantization.

The export contract is version 1 JSON (`format: "microllm"`) with row-major tensors. The Go loader validates config and tensor shapes before quantization.

## Build and test

```bash
mvn test
```

## Produce a source checkpoint

From the repository root, after tokenization has created `data/tokenized`:

```bash
cd java
mvn package
java -jar target/microllm-train-0.1.0-SNAPSHOT.jar \
  --vocab ../data/tokenized/vocab.json \
  --tokens ../data/tokenized/tokens.json \
  --output ../models/exported/model.json
```

The Java package and JSON format retain their historical `microllm` identifiers for compatibility.
