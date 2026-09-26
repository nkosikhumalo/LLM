package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"

	"github.com/charmbracelet/lipgloss"
	"github.com/nkosikhumalo/microllm/go/internal/inference"
	"github.com/nkosikhumalo/microllm/go/internal/loader"
	"github.com/nkosikhumalo/microllm/go/internal/quantization"
	"github.com/nkosikhumalo/microllm/go/internal/tokenizer"
)

var (
	titleStyle  = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("205")).Padding(0, 1).Border(lipgloss.RoundedBorder()).BorderForeground(lipgloss.Color("63"))
	mutedStyle  = lipgloss.NewStyle().Faint(true)
	resultStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("42")).Padding(0, 1).Border(lipgloss.RoundedBorder()).BorderForeground(lipgloss.Color("42"))
)

func main() {
	in := flag.String("input", "", "fp32 model.json path")
	out := flag.String("output", "", "int8 checkpoint path")
	scheme := flag.String("scheme", "per-tensor", "weight quantization: per-tensor or per-channel")
	calibration := flag.String("calibration", "", "optional text file for activation range calibration")
	vocabPath := flag.String("vocab", "../data/tokenized/vocab.json", "tokenizer vocabulary JSON")
	samples := flag.Int("calibration-samples", 100, "maximum non-empty calibration lines")
	maxLineBytes := flag.Int("max-line-bytes", 64*1024*1024, "maximum bytes in one input text line")
	flag.Parse()
	fmt.Println(titleStyle.Render("MiniLLM Quantization Engine"))
	fmt.Println(mutedStyle.Render("FP32 checkpoint → signed INT8 checkpoint"))
	if *in == "" || *out == "" {
		fmt.Fprintln(os.Stderr, "usage: go run ./cmd/quantize --input model.json --output model.int8.json")
		os.Exit(2)
	}
	fmt.Println(mutedStyle.Render("Loading and validating source checkpoint..."))
	e, err := loader.Load(*in)
	if err != nil {
		fatal(err)
	}
	fmt.Printf("Quantizing %d layers and model tensors (%s)...\n", e.Config.NLayers, *scheme)
	c, err := quantization.QuantizeWithScheme(e, *scheme)
	if err != nil {
		fatal(err)
	}
	if *calibration != "" {
		vocab, loadErr := tokenizer.Load(*vocabPath)
		if loadErr != nil {
			fatal(loadErr)
		}
		if len(vocab.Tokens) != e.Config.VocabSize {
			fatal(fmt.Errorf("vocabulary size does not match model"))
		}
		if *samples < 1 || *maxLineBytes < 1 {
			fatal(fmt.Errorf("calibration-samples and max-line-bytes must be positive"))
		}
		calFile, openErr := os.Open(*calibration)
		if openErr != nil {
			fatal(openErr)
		}
		defer calFile.Close()
		scanner := bufio.NewScanner(calFile)
		scanner.Buffer(make([]byte, 64*1024), *maxLineBytes)
		model, modelErr := inference.New(e)
		if modelErr != nil {
			fatal(modelErr)
		}
		ranges := make(map[string]inference.ActivationRange)
		used := 0
		for scanner.Scan() && used < *samples {
			line := scanner.Text()
			if line == "" {
				continue
			}
			ids, encodeErr := vocab.Encode(line)
			if encodeErr != nil {
				fatal(fmt.Errorf("calibration line %d: %w", used+1, encodeErr))
			}
			if len(ids) > e.Config.MaxSeqLen {
				ids = ids[:e.Config.MaxSeqLen]
			}
			if len(ids) == 0 {
				continue
			}
			if _, runErr := model.LogitsWithActivationRanges(ids, ranges); runErr != nil {
				fatal(runErr)
			}
			used++
		}
		if scanErr := scanner.Err(); scanErr != nil {
			fatal(scanErr)
		}
		if used == 0 {
			fatal(fmt.Errorf("calibration file had no usable lines"))
		}
		c.ActivationRanges = ranges
		fmt.Println(mutedStyle.Render(fmt.Sprintf("Stored activation ranges from %d text sequences (%d tensors).", used, len(ranges))))
	}
	if err = quantization.Save(*out, c); err != nil {
		fatal(err)
	}
	a, _ := os.Stat(*in)
	b, _ := os.Stat(*out)
	if a != nil && b != nil {
		fmt.Println(resultStyle.Render(fmt.Sprintf("INT8 checkpoint ready\n%s\nScheme: %s · %.2fx smaller · %d bytes", *out, *scheme, float64(a.Size())/float64(b.Size()), b.Size())))
	} else {
		fmt.Println(resultStyle.Render(fmt.Sprintf("INT8 checkpoint ready\n%s\nScheme: %s", *out, *scheme)))
	}
}
func fatal(err error) { fmt.Fprintln(os.Stderr, err); os.Exit(1) }
