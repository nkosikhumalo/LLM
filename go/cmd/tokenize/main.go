package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"github.com/nkosikhumalo/microllm/internal/tokenizer"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

func main() {
	input := flag.String("input", "../data/raw", "raw .txt directory")
	output := flag.String("output", "../data/tokenized", "destination directory")
	flag.Parse()
	files, e := filepath.Glob(filepath.Join(*input, "*.txt"))
	if e != nil || len(files) == 0 {
		fatal("no .txt files found in %s", *input)
	}
	sort.Strings(files)
	var b strings.Builder
	for _, f := range files {
		x, e := os.ReadFile(f)
		if e != nil {
			fatal("read %s: %v", f, e)
		}
		b.Write(x)
	}
	v, e := tokenizer.Build(b.String())
	if e != nil {
		fatal("build vocab: %v", e)
	}
	ids, e := v.Encode(b.String())
	if e != nil {
		fatal("encode: %v", e)
	}
	if e = os.MkdirAll(*output, 0755); e != nil {
		fatal("mkdir: %v", e)
	}
	if e = tokenizer.Save(filepath.Join(*output, "vocab.json"), v); e != nil {
		fatal("save vocab: %v", e)
	}
	out, e := json.Marshal(ids)
	if e != nil {
		fatal("marshal IDs: %v", e)
	}
	if e = os.WriteFile(filepath.Join(*output, "tokens.json"), out, 0644); e != nil {
		fatal("save tokens: %v", e)
	}
	fmt.Printf("wrote %d tokens and %d vocabulary entries to %s\n", len(ids), len(v.Tokens), *output)
}
func fatal(f string, a ...any) { fmt.Fprintf(os.Stderr, f+"\n", a...); os.Exit(1) }
