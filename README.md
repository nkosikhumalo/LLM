<p align="center">
  <img src="image/title-banner.svg" alt="MicroLLM — Go prepares and serves, Java trains and exports" width="880"/>
</p>

<p align="center">
  <a href="https://go.dev/"><img src="https://img.shields.io/badge/Go-00ADD8?style=for-the-badge&amp;logo=go&amp;logoColor=white" alt="Go"/></a>
  <a href="https://www.java.com/"><img src="https://img.shields.io/badge/Java-E76F00?style=for-the-badge&amp;logo=openjdk&amp;logoColor=white" alt="Java"/></a>
  <img src="https://img.shields.io/badge/from%20scratch-1e293b?style=for-the-badge" alt="From scratch"/>
  <img src="https://img.shields.io/badge/no%20Python-334155?style=for-the-badge" alt="No Python"/>
</p>

<p align="center">
  A language model built from scratch — no Python, no cloud, no magic boxes.<br/>
  Just <strong>Go</strong> and <strong>Java</strong> doing everything from raw text to generated output.
</p>

<p align="center">
  <img src="image/architecture.jpeg" alt="Micro-LLM architecture: Go prepares text, Java trains the model, Go writes answers" width="900"/>
</p>

---

<p align="center">
  <img src="image/tagline.svg" alt="Two languages. One pipeline. Zero magic." width="880"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Go-tokenize-00ADD8?style=flat-square&amp;logo=go&amp;logoColor=white" alt="Go tokenize"/>
  <img src="https://img.shields.io/badge/Go-load%20weights-00ADD8?style=flat-square&amp;logo=go&amp;logoColor=white" alt="Go load weights"/>
  <img src="https://img.shields.io/badge/Go-inference-00ADD8?style=flat-square&amp;logo=go&amp;logoColor=white" alt="Go inference"/>
  <img src="https://img.shields.io/badge/Go-stream%20output-00ADD8?style=flat-square&amp;logo=go&amp;logoColor=white" alt="Go stream"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Java-transformer-E76F00?style=flat-square&amp;logo=openjdk&amp;logoColor=white" alt="Java transformer"/>
  <img src="https://img.shields.io/badge/Java-backprop-E76F00?style=flat-square&amp;logo=openjdk&amp;logoColor=white" alt="Java backprop"/>
  <img src="https://img.shields.io/badge/Java-Adam%20%2F%20SGD-E76F00?style=flat-square&amp;logo=openjdk&amp;logoColor=white" alt="Java optimizer"/>
  <img src="https://img.shields.io/badge/Java-export%20weights-E76F00?style=flat-square&amp;logo=openjdk&amp;logoColor=white" alt="Java export"/>
</p>

---

## What this is

You drop a few megabytes of text in a folder. Java trains a small transformer on it — learning which characters tend to follow which. Then you run Go with any prompt and it streams text back to your terminal, one character at a time.

Every part is written by hand. No ML frameworks, no Python, no GPU required. Training takes minutes on a regular laptop.

---

## Colour guide

<p align="center">
  <img src="image/color-guide.svg" alt="Go blue, Java orange, shared slate" width="880"/>
</p>

| Side | Colour | Job |
|------|--------|-----|
| ![Go](https://img.shields.io/badge/Go-00ADD8?style=flat-square&logo=go&logoColor=white) | `#00ADD8` | Tokenize text, load weights, run the model, stream answers |
| ![Java](https://img.shields.io/badge/Java-E76F00?style=flat-square&logo=openjdk&logoColor=white) | `#E76F00` | Tensor math, transformer layers, backprop, optimizer, weight export |
| ![Shared](https://img.shields.io/badge/Shared-64748B?style=flat-square) | `#64748B` | Data files and weight files used by both sides |

---

## The full pipeline

```mermaid
flowchart LR
    A([Raw text]) -->|Go tokenizes| B([Token IDs])
    B -->|Java trains| C([Weight file])
    C -->|Go loads| D([Your prompt])
    D -->|Go generates| E([Output text])

    style A fill:#00ADD8,color:#ffffff,stroke:#007d99
    style B fill:#00ADD8,color:#ffffff,stroke:#007d99
    style C fill:#E76F00,color:#ffffff,stroke:#b45309
    style D fill:#00ADD8,color:#ffffff,stroke:#007d99
    style E fill:#059669,color:#ffffff,stroke:#047857
```

---

## Phase 1 — Training

![Go](https://img.shields.io/badge/Go-data%20prep-00ADD8?style=for-the-badge&logo=go&logoColor=white)
![Java](https://img.shields.io/badge/Java-training-E76F00?style=for-the-badge&logo=openjdk&logoColor=white)

**Go** turns raw text into numbers. **Java** trains the model on those numbers and saves the result.

```mermaid
flowchart TB
    subgraph GO ["Go — Data Prep"]
        A([Raw text files]) --> B([Split into characters])
        B --> C([Assign ID 0–255 to each char])
        C --> D([Save as integer sequences])
    end

    subgraph JV ["Java — Training"]
        E([Load sequences]) --> F([Embed each token])
        F --> G([Multi-head attention])
        G --> H([Feed-forward layer])
        H --> I([Normalize — RMSNorm])
        I --> J([Measure loss])
        J --> K([Backpropagate gradients])
        K --> L([Update weights — AdamW / SGD])
        L --> M([Save weight file])
    end

    D --> E

    style GO fill:#e0f7fc,stroke:#00ADD8,color:#0f172a
    style JV fill:#fff4e5,stroke:#E76F00,color:#0f172a
    style A fill:#00ADD8,color:#ffffff,stroke:#007d99
    style B fill:#00ADD8,color:#ffffff,stroke:#007d99
    style C fill:#00ADD8,color:#ffffff,stroke:#007d99
    style D fill:#00ADD8,color:#ffffff,stroke:#007d99
    style E fill:#E76F00,color:#ffffff,stroke:#b45309
    style F fill:#E76F00,color:#ffffff,stroke:#b45309
    style G fill:#E76F00,color:#ffffff,stroke:#b45309
    style H fill:#E76F00,color:#ffffff,stroke:#b45309
    style I fill:#E76F00,color:#ffffff,stroke:#b45309
    style J fill:#E76F00,color:#ffffff,stroke:#b45309
    style K fill:#E76F00,color:#ffffff,stroke:#b45309
    style L fill:#E76F00,color:#ffffff,stroke:#b45309
    style M fill:#c2410c,color:#ffffff,stroke:#9a3412
```

### What each step means

| Step | Who | Plain meaning |
|------|-----|---------------|
| Character split | ![Go](https://img.shields.io/badge/Go-00ADD8?style=flat-square&logo=go&logoColor=white) | Break the text into individual letters and symbols |
| Assign IDs | ![Go](https://img.shields.io/badge/Go-00ADD8?style=flat-square&logo=go&logoColor=white) | Give every character a number (0 to 255) |
| Save sequences | ![Go](https://img.shields.io/badge/Go-00ADD8?style=flat-square&logo=go&logoColor=white) | Write those numbers to disk so Java can read them |
| Embed tokens | ![Java](https://img.shields.io/badge/Java-E76F00?style=flat-square&logo=openjdk&logoColor=white) | Turn each number into a small list of floats the model can work with |
| Attention | ![Java](https://img.shields.io/badge/Java-E76F00?style=flat-square&logo=openjdk&logoColor=white) | Let each character look at surrounding characters to understand context |
| Feed-forward | ![Java](https://img.shields.io/badge/Java-E76F00?style=flat-square&logo=openjdk&logoColor=white) | Extra processing step after attention — adds depth |
| RMSNorm | ![Java](https://img.shields.io/badge/Java-E76F00?style=flat-square&logo=openjdk&logoColor=white) | Keeps the numbers from getting too big or too small during training |
| Loss | ![Java](https://img.shields.io/badge/Java-E76F00?style=flat-square&logo=openjdk&logoColor=white) | Measures how wrong the model's prediction was |
| Backprop | ![Java](https://img.shields.io/badge/Java-E76F00?style=flat-square&logo=openjdk&logoColor=white) | Figures out which weights caused the mistake |
| Optimizer | ![Java](https://img.shields.io/badge/Java-E76F00?style=flat-square&logo=openjdk&logoColor=white) | Nudges every weight slightly in the right direction |
| Export | ![Java](https://img.shields.io/badge/Java-E76F00?style=flat-square&logo=openjdk&logoColor=white) | Saves everything the model learned into a file |

---

## Phase 2 — Inference

![Go only](https://img.shields.io/badge/Go%20only-inference-00ADD8?style=for-the-badge&logo=go&logoColor=white)

Only **Go** here. No Java, no training loop — just load the weights and generate.

```mermaid
flowchart TB
    A([Your prompt]) --> B([Tokenize — chars to IDs])
    B --> C([Load weight file])
    C --> D([Forward pass through transformer])
    D --> E([KV-cache — skip recomputing old tokens])
    E --> F([Get probabilities for next character])
    F --> G([Sample with temperature + Top-K + Top-P])
    G --> H([Write character to terminal])
    H -->|next character| D

    style A fill:#00ADD8,color:#ffffff,stroke:#007d99
    style B fill:#00ADD8,color:#ffffff,stroke:#007d99
    style C fill:#00ADD8,color:#ffffff,stroke:#007d99
    style D fill:#0891b2,color:#ffffff,stroke:#0e7490
    style E fill:#0891b2,color:#ffffff,stroke:#0e7490
    style F fill:#0891b2,color:#ffffff,stroke:#0e7490
    style G fill:#0891b2,color:#ffffff,stroke:#0e7490
    style H fill:#059669,color:#ffffff,stroke:#047857
```

### The sampler — three knobs

| Knob | What it does |
|------|-------------|
| Temperature | Low = safe and predictable. High = creative and surprising |
| Top-K | Only consider the K most likely next characters |
| Top-P | Only consider characters that together make up P% of the probability mass |

The KV-cache is a speed trick — Go stores the attention results for characters it already processed, so each new character only needs one forward step instead of recomputing everything from scratch.

---

## Module breakdown

```mermaid
flowchart LR
    subgraph GO ["Go packages"]
        G1([tokenizer])
        G2([loader])
        G3([inference + KV-cache])
        G4([sampler])
        G5([cli])
    end

    subgraph JAVA ["Java packages"]
        J1([tensor + ops])
        J2([layers])
        J3([autograd])
        J4([optimizer])
        J5([exporter])
        J6([trainer])
    end

    G1 -->|token IDs| J6
    J5 -->|weight file| G2

    style GO fill:#e0f7fc,stroke:#00ADD8,color:#0f172a
    style JAVA fill:#fff4e5,stroke:#E76F00,color:#0f172a
    style G1 fill:#00ADD8,color:#ffffff,stroke:#007d99
    style G2 fill:#00ADD8,color:#ffffff,stroke:#007d99
    style G3 fill:#00ADD8,color:#ffffff,stroke:#007d99
    style G4 fill:#00ADD8,color:#ffffff,stroke:#007d99
    style G5 fill:#00ADD8,color:#ffffff,stroke:#007d99
    style J1 fill:#E76F00,color:#ffffff,stroke:#b45309
    style J2 fill:#E76F00,color:#ffffff,stroke:#b45309
    style J3 fill:#E76F00,color:#ffffff,stroke:#b45309
    style J4 fill:#E76F00,color:#ffffff,stroke:#b45309
    style J5 fill:#E76F00,color:#ffffff,stroke:#b45309
    style J6 fill:#E76F00,color:#ffffff,stroke:#b45309
```

---

## Project layout

```
micro-llm/
├── go/
│   ├── cmd/
│   │   ├── tokenize/       ← tokenize a raw text corpus
│   │   └── generate/       ← generate text from a prompt
│   └── internal/
│       ├── tokenizer/      ← build vocab, encode, decode
│       ├── loader/         ← read exported weight file
│       ├── inference/      ← forward pass + KV-cache
│       ├── sampler/        ← temperature, top-k, top-p
│       └── cli/            ← wires all packages together
│
├── java/
│   └── src/main/java/com/microllm/
│       ├── tensor/         ← Tensor class and math ops
│       ├── layers/         ← Embedding, Attention, FFN, Norm
│       ├── autograd/       ← Variable graph + backprop engine
│       ├── optim/          ← AdamW and SGD
│       ├── model/          ← Transformer, TransformerBlock, ModelConfig
│       ├── train/          ← Trainer, Loss, TokenDataset
│       └── export/         ← WeightExporter
│
├── data/
│   ├── raw/                ← put your .txt files here
│   └── tokenized/          ← Go writes integer sequences here
│
├── models/
│   └── exported/           ← Java writes weights here, Go reads them
│
├── image/                  ← README banners + architecture diagram
│
└── scripts/
    ├── tokenize.sh
    ├── train.sh
    └── generate.sh
```

---

## Running it

```bash
# step 1 — put your text in data/raw/ then tokenize it
bash scripts/tokenize.sh

# step 2 — train the model (Java)
bash scripts/train.sh

# step 3 — generate from a prompt (Go)
bash scripts/generate.sh
```

---

## Scale — built for a laptop

No cloud, no GPU, no special hardware.

| Setting | Value | Why |
|---------|-------|-----|
| Vocab size | ~256 chars | One ID per printable character — tiny lookup table |
| Layers | 2 – 4 | Deep enough to learn patterns, light enough to run fast |
| Model width | 64 or 128 | Stays within normal CPU cache |
| Attention heads | 2 – 4 | Enough for this scale |
| Parameters | ~1 million | Trains in minutes, not hours |
| Dataset size | 1 – 2 MB text | A play, short stories, or small code files |
| Training RAM | < 20 MB | Laptop stays usable while training |
| Inference RAM | < 10 MB | Answers come back fast |

---

<p align="center">
  <img src="image/pipeline-strip.svg" alt="text → Go tokenizes → Java trains → weights → Go generates" width="880"/>
</p>

<p align="center">
  <a href="docs/ARCHITECTURE.md"><img src="https://img.shields.io/badge/Architecture%20deep--dive-1e293b?style=for-the-badge" alt="Architecture deep-dive"/></a>
</p>
