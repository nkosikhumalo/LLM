package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/nkosikhumalo/microllm/go/internal/cli"
	"github.com/nkosikhumalo/microllm/go/internal/quantization"
	"github.com/nkosikhumalo/microllm/go/internal/sampler"
	"github.com/nkosikhumalo/microllm/go/internal/tokenizer"
)

func main() {
	modelPath := flag.String("model", "../models/exported/model.json", "model export JSON")
	vocabPath := flag.String("vocab", "../data/tokenized/vocab.json", "vocabulary JSON")
	prompt := flag.String("prompt", "", "initial text")
	n := flag.Int("tokens", 100, "tokens to generate")
	temp := flag.Float64("temperature", 0.7, "sampling temperature")
	topKValue := 40
	flag.IntVar(&topKValue, "top-k", 40, "top-k (0 disables)")
	flag.IntVar(&topKValue, "topk", 40, "alias for -top-k")
	topP := flag.Float64("top-p", 1, "nucleus probability")
	greedy := flag.Bool("greedy", false, "choose most likely token")
	repetitionPenalty := flag.Float64("repetition-penalty", 1.15, "penalty for repeating tokens; 1 disables")
	seed := flag.Int64("seed", time.Now().UnixNano(), "random seed")
	flag.Parse()

	model, err := quantization.LoadInferenceModel(*modelPath)
	if err != nil {
		fatal("load model: %v", err)
	}
	modelExport := model.Export
	vocab, err := tokenizer.Load(*vocabPath)
	if err != nil {
		fatal("load vocab: %v", err)
	}
	if len(vocab.Tokens) != modelExport.Config.VocabSize {
		fatal("vocabulary size does not match model")
	}
	promptText, err := cli.Prompt(strings.TrimSpace(*prompt))
	if err != nil {
		fatal("prompt: %v", err)
	}
	promptText = strings.TrimSpace(promptText)
	ids, err := vocab.Encode(promptText)
	if err != nil {
		fatal("encode prompt: %v", err)
	}

	session := model.NewSession()
	cfg := sampler.Config{Temperature: *temp, TopK: topKValue, TopP: *topP, Greedy: *greedy, RepetitionPenalty: *repetitionPenalty}
	rng := rand.New(rand.NewSource(*seed))

	var logits []float64
	if len(ids) > 0 {
		logits, err = session.Prefill(ids)
		if err != nil {
			fatal("prefill: %v", err)
		}
	}

	var generated strings.Builder
	generatedIDs := make([]int, 0, *n)
	for i := 0; i < *n; i++ {
		if session.Len() >= modelExport.Config.MaxSeqLen {
			keep := ids
			if len(keep) >= modelExport.Config.MaxSeqLen {
				keep = keep[len(keep)-modelExport.Config.MaxSeqLen+1:]
			}
			logits, err = session.Prefill(keep)
			if err != nil {
				fatal("re-prefill: %v", err)
			}
			ids = append([]int(nil), keep...)
		}
		if logits == nil {
			fatal("prompt is empty")
		}
		cfg.SeenTokens = generatedIDs
		id, err := sampler.Sample(logits, cfg, rng)
		if err != nil {
			fatal("sample: %v", err)
		}
		text, err := vocab.Decode([]int{id})
		if err != nil {
			fatal("decode: %v", err)
		}
		generated.WriteString(text)
		generatedIDs = append(generatedIDs, id)
		if strings.Contains(generated.String(), "<|end|>") {
			break
		}
		ids = append(ids, id)
		logits, err = session.Step(id)
		if err != nil {
			fatal("step: %v", err)
		}
	}
	response := generated.String()
	if end := strings.Index(response, "<|end|>"); end >= 0 {
		response = response[:end]
	}
	if end := strings.IndexAny(response, "\r\n"); end >= 0 {
		response = response[:end]
	}
	fmt.Println(promptText + response)
}

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
