package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"github.com/nkosikhumalo/microllm/internal/tokenizer"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

func main() {
	input := flag.String("input", "../data/raw/train.txt", "raw .txt file or directory")
	output := flag.String("output", "../data/tokenized", "destination directory")
	anchorsPath := flag.String("anchors", "../data/processed/train_answer_anchors.offsets", "output path for dialogue answer anchors (empty disables)")
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
	if *anchorsPath != "" {
		charOffsets, anchorCount := dialogueAnswerOffsets(b.String())
		if anchorCount > 0 {
			tokenOffsets, err := v.TokenBoundaries(b.String(), charOffsets)
			if err != nil {
				fatal("map answer anchors: %v", err)
			}
			var converted strings.Builder
			for i := 0; i < anchorCount; i++ {
				base := i * 3
				tokenStart, tokenAnswer, tokenEnd := tokenOffsets[base], tokenOffsets[base+1], tokenOffsets[base+2]
				if tokenAnswer <= tokenStart || tokenEnd <= tokenAnswer {
					fatal("answer anchor %d maps to an empty span", i+1)
				}
				if tokenAnswer-tokenStart > 127 {
					tokenStart = tokenAnswer - 127
				}
				if tokenEnd-tokenStart > 128 {
					tokenEnd = tokenStart + 128
				}
				converted.WriteString(strconv.Itoa(tokenStart))
				converted.WriteByte('\t')
				converted.WriteString(strconv.Itoa(tokenAnswer))
				converted.WriteByte('\t')
				converted.WriteString(strconv.Itoa(tokenEnd))
				converted.WriteByte('\n')
			}
			if e = os.MkdirAll(filepath.Dir(*anchorsPath), 0755); e != nil {
				fatal("mkdir answer anchors: %v", e)
			}
			if e = os.WriteFile(*anchorsPath, []byte(converted.String()), 0644); e != nil {
				fatal("save answer anchors: %v", e)
			}
		} else if err := os.Remove(*anchorsPath); err != nil && !os.IsNotExist(err) {
			fatal("remove stale answer anchors: %v", err)
		}
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

// dialogueAnswerOffsets returns rune offsets for each prompt start and answer start.
func dialogueAnswerOffsets(corpus string) ([]int, int) {
	const separator = " | Bot: "
	var offsets []int
	runeOffset := 0
	for _, rawLine := range strings.SplitAfter(corpus, "\n") {
		line := strings.TrimSuffix(strings.TrimSuffix(rawLine, "\n"), "\r")
		if line == "" {
			runeOffset += utf8.RuneCountInString(rawLine)
			continue
		}
		marker := strings.Index(line, separator)
		endMarker := strings.Index(line, "<|end|>")
		if strings.HasPrefix(line, "Human: ") && marker >= 0 && endMarker > marker && strings.HasSuffix(line, "<|end|>") {
			offsets = append(
				offsets,
				runeOffset,
				runeOffset+utf8.RuneCountInString(line[:marker+len(separator)]),
				runeOffset+utf8.RuneCountInString(line[:endMarker+len("<|end|>")]),
			)
		}
		runeOffset += utf8.RuneCountInString(rawLine)
	}
	return offsets, len(offsets) / 3
}
