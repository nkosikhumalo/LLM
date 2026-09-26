package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/nkosikhumalo/microllm/go/internal/tokenizer"
)

func main() {
	input := flag.String("input", "../data/raw/train.txt", "UTF-8 text file or directory of text files")
	output := flag.String("output", "../data/tokenized", "destination directory")
	flag.Parse()
	files := []string{*input}
	if info, err := os.Stat(*input); err == nil && info.IsDir() {
		files, err = filepath.Glob(filepath.Join(*input, "*.txt"))
		if err != nil {
			fatal("list input files: %v", err)
		}
	}
	if len(files) == 0 {
		fatal("no .txt files found in %s", *input)
	}
	sort.Strings(files)
	var corpus strings.Builder
	for _, path := range files {
		data, err := os.ReadFile(path)
		if err != nil {
			fatal("read %s: %v", path, err)
		}
		if corpus.Len() > 0 {
			corpus.WriteByte('\n')
		}
		corpus.Write(data)
	}
	vocab, err := tokenizer.Build(corpus.String())
	if err != nil {
		fatal("build vocab: %v", err)
	}
	ids, err := vocab.Encode(corpus.String())
	if err != nil {
		fatal("encode: %v", err)
	}
	if len(ids) < 2 {
		fatal("dataset must contain at least two tokens; found %d", len(ids))
	}
	if err := os.MkdirAll(*output, 0755); err != nil {
		fatal("create output directory: %v", err)
	}
	if err := os.WriteFile(filepath.Join(*output, "corpus.txt"), []byte(corpus.String()), 0644); err != nil {
		fatal("save normalized corpus: %v", err)
	}
	if err := tokenizer.Save(filepath.Join(*output, "vocab.json"), vocab); err != nil {
		fatal("save vocab: %v", err)
	}
	encoded, err := json.Marshal(ids)
	if err != nil {
		fatal("marshal token IDs: %v", err)
	}
	if err := os.WriteFile(filepath.Join(*output, "tokens.json"), encoded, 0644); err != nil {
		fatal("save token IDs: %v", err)
	}
	fmt.Printf("wrote %d tokens and %d vocabulary entries to %s\n", len(ids), len(vocab.Tokens), *output)
}

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
