package main

import (
	"flag"
	"fmt"
	"github.com/nkosikhumalo/microllm/internal/cli"
	"github.com/nkosikhumalo/microllm/internal/inference"
	"github.com/nkosikhumalo/microllm/internal/loader"
	"github.com/nkosikhumalo/microllm/internal/sampler"
	"github.com/nkosikhumalo/microllm/internal/tokenizer"
	"math/rand"
	"os"
	"strings"
	"time"
)

func main() {
	modelPath := flag.String("model", "../models/exported/model.json", "model export JSON")
	vocabPath := flag.String("vocab", "../data/tokenized/vocab.json", "vocabulary JSON")
	prompt := flag.String("prompt", "", "initial text")
	n := flag.Int("tokens", 100, "tokens to generate")
	temp := flag.Float64("temperature", 1, "sampling temperature")
	topK := flag.Int("top-k", 0, "top-k (0 disables)")
	topP := flag.Float64("top-p", 1, "nucleus probability")
	greedy := flag.Bool("greedy", false, "choose most likely token")
	seed := flag.Int64("seed", time.Now().UnixNano(), "random seed")
	flag.Parse()
	e, v := loader.Load(*modelPath)
	if v != nil {
		fatal("load model: %v", v)
	}
	voc, v := tokenizer.Load(*vocabPath)
	if v != nil {
		fatal("load vocab: %v", v)
	}
	if len(voc.Tokens) != e.Config.VocabSize {
		fatal("vocabulary size does not match model")
	}
	m, v := inference.New(e)
	if v != nil {
		fatal("model: %v", v)
	}
	p, v := cli.Prompt(strings.TrimSpace(*prompt))
	if v != nil {
		fatal("prompt: %v", v)
	}
	ids, v := voc.Encode(strings.TrimSpace(p))
	if v != nil {
		fatal("encode prompt: %v", v)
	}
	fmt.Print(p)
	cfg := sampler.Config{Temperature: *temp, TopK: *topK, TopP: *topP, Greedy: *greedy}
	rng := rand.New(rand.NewSource(*seed))
	for i := 0; i < *n; i++ {
		if len(ids) >= e.Config.MaxSeqLen {
			ids = ids[1:]
		}
		logits, v := m.Logits(ids)
		if v != nil {
			fatal("inference: %v", v)
		}
		id, v := sampler.Sample(logits, cfg, rng)
		if v != nil {
			fatal("sample: %v", v)
		}
		text, v := voc.Decode([]int{id})
		if v != nil {
			fatal("decode: %v", v)
		}
		if v = cli.Stream(os.Stdout, text); v != nil {
			fatal("write: %v", v)
		}
		ids = append(ids, id)
	}
	fmt.Println()
}
func fatal(f string, a ...any) { fmt.Fprintf(os.Stderr, f+"\n", a...); os.Exit(1) }
