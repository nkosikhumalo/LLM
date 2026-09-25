// Package tokenizer provides a deterministic, boundary-aware greedy WordPiece tokenizer
// with character fallback. Frequent substrings are learned from the training corpus.
package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

const (
	EndToken      = "<|end|>"
	maxSubwords   = 1200
	maxWordTypes  = 20000
	maxPieceRunes = 12
)

// Vocab associates each token string with a stable integer ID.
type Vocab struct {
	Tokens   []string `json:"tokens"`
	index    map[string]int
	maxRunes int
}

func New(tokens []string) (*Vocab, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("vocabulary is empty")
	}
	v := &Vocab{Tokens: append([]string(nil), tokens...), index: make(map[string]int, len(tokens))}
	for id, token := range v.Tokens {
		if token == "" {
			return nil, fmt.Errorf("token %d is empty", id)
		}
		if _, ok := v.index[token]; ok {
			return nil, fmt.Errorf("duplicate token %q", token)
		}
		v.index[token] = id
		piece := strings.TrimPrefix(token, "##")
		if n := utf8.RuneCountInString(piece); n > v.maxRunes {
			v.maxRunes = n
		}
	}
	return v, nil
}

// Build learns frequent word pieces while keeping each observed character as a fallback token.
func Build(corpus string) (*Vocab, error) {
	chars, words := make(map[string]struct{}), make(map[string]int)
	for _, r := range corpus {
		chars[string(r)] = struct{}{}
	}
	for _, word := range wordRuns(corpus) {
		words[word]++
	}
	orderedWords := make([]string, 0, len(words))
	for word := range words {
		orderedWords = append(orderedWords, word)
	}
	sort.Slice(orderedWords, func(i, j int) bool {
		if words[orderedWords[i]] == words[orderedWords[j]] {
			return orderedWords[i] < orderedWords[j]
		}
		return words[orderedWords[i]] > words[orderedWords[j]]
	})
	if len(orderedWords) > maxWordTypes {
		orderedWords = orderedWords[:maxWordTypes]
	}
	pieceCounts := make(map[string]int)
	for _, word := range orderedWords {
		runes := []rune(word)
		for start := range runes {
			limit := min(len(runes), start+maxPieceRunes)
			for end := start + 2; end <= limit; end++ {
				pieceCounts[string(runes[start:end])] += words[word]
			}
		}
	}
	base := make([]string, 0, len(chars))
	for token := range chars {
		base = append(base, token)
	}
	sort.Strings(base)
	subwords := make([]string, 0, len(pieceCounts))
	for piece := range pieceCounts {
		subwords = append(subwords, piece)
	}
	sort.Slice(subwords, func(i, j int) bool {
		if pieceCounts[subwords[i]] == pieceCounts[subwords[j]] {
			return subwords[i] < subwords[j]
		}
		return pieceCounts[subwords[i]] > pieceCounts[subwords[j]]
	})
	if len(subwords) > maxSubwords {
		subwords = subwords[:maxSubwords]
	}
	tokens := append([]string(nil), base...)
	tokens = append(tokens, subwords...)
	for _, token := range base {
		tokens = append(tokens, "##"+token)
	}
	for _, token := range subwords {
		tokens = append(tokens, "##"+token)
	}
	tokens = append(tokens, EndToken)
	return New(tokens)
}

func wordRuns(text string) []string {
	var words []string
	var current strings.Builder
	flush := func() {
		if current.Len() > 0 {
			words = append(words, current.String())
			current.Reset()
		}
	}
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsNumber(r) || r == '_' {
			current.WriteRune(r)
		} else {
			flush()
		}
	}
	flush()
	return words
}

func (v *Vocab) split(text string) []string {
	runes := []rune(text)
	out := make([]string, 0, len(runes))
	for i := 0; i < len(runes); {
		if i+len([]rune(EndToken)) <= len(runes) && string(runes[i:i+len([]rune(EndToken))]) == EndToken {
			out = append(out, EndToken)
			i += len([]rune(EndToken))
			continue
		}
		if !(unicode.IsLetter(runes[i]) || unicode.IsNumber(runes[i]) || runes[i] == '_') {
			out = append(out, string(runes[i]))
			i++
			continue
		}
		end := i + 1
		for end < len(runes) && (unicode.IsLetter(runes[end]) || unicode.IsNumber(runes[end]) || runes[end] == '_') {
			end++
		}
		atWordStart := true
		for i < end {
			piece := string(runes[i])
			token := piece
			if !atWordStart {
				token = "##" + piece
			}
			bestEnd, best := i+1, token
			limit := min(end, i+v.maxRunes)
			for j := limit; j > i+1; j-- {
				candidate := string(runes[i:j])
				if !atWordStart {
					candidate = "##" + candidate
				}
				if _, ok := v.index[candidate]; ok {
					bestEnd, best = j, candidate
					break
				}
			}
			out = append(out, best)
			i = bestEnd
			atWordStart = false
		}
	}
	return out
}

// Encode converts text to greedy longest-match wordpiece IDs.
func (v *Vocab) Encode(text string) ([]int, error) {
	pieces := v.split(text)
	ids := make([]int, 0, len(pieces))
	for _, piece := range pieces {
		id, ok := v.index[piece]
		if !ok {
			return nil, fmt.Errorf("character %q is not in vocabulary", piece)
		}
		ids = append(ids, id)
	}
	return ids, nil
}

// TokenBoundaries maps ordered rune offsets to token offsets in one corpus pass.
func (v *Vocab) TokenBoundaries(text string, runeOffsets []int) ([]int, error) {
	runeCount := utf8.RuneCountInString(text)
	for i, offset := range runeOffsets {
		if offset < 0 || offset > runeCount || (i > 0 && offset < runeOffsets[i-1]) {
			return nil, fmt.Errorf("rune offsets must be ordered and within the text")
		}
	}
	pieces := v.split(text)
	result := make([]int, len(runeOffsets))
	position, tokenIndex, offsetIndex := 0, 0, 0
	for _, piece := range pieces {
		end := position + utf8.RuneCountInString(strings.TrimPrefix(piece, "##"))
		for offsetIndex < len(runeOffsets) && runeOffsets[offsetIndex] <= end {
			if runeOffsets[offsetIndex] == end {
				result[offsetIndex] = tokenIndex + 1
			} else {
				result[offsetIndex] = tokenIndex
			}
			offsetIndex++
		}
		position = end
		tokenIndex++
	}
	for offsetIndex < len(runeOffsets) {
		result[offsetIndex] = tokenIndex
		offsetIndex++
	}
	return result, nil
}

// TokenBoundary maps a rune offset to the corresponding token offset.
func (v *Vocab) TokenBoundary(text string, runeOffset int) (int, error) {
	boundaries, err := v.TokenBoundaries(text, []int{runeOffset})
	if err != nil {
		return 0, err
	}
	return boundaries[0], nil
}

func (v *Vocab) Decode(ids []int) (string, error) {
	var text strings.Builder
	for _, id := range ids {
		if id < 0 || id >= len(v.Tokens) {
			return "", fmt.Errorf("token ID %d is outside vocabulary", id)
		}
		text.WriteString(strings.TrimPrefix(v.Tokens[id], "##"))
	}
	return text.String(), nil
}

func Save(path string, v *Vocab) error {
	encoded, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return fmt.Errorf("encode vocabulary: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create vocabulary directory: %w", err)
	}
	return os.WriteFile(path, encoded, 0o644)
}

func Load(path string) (*Vocab, error) {
	encoded, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var stored struct {
		Tokens []string `json:"tokens"`
	}
	if err := json.Unmarshal(encoded, &stored); err != nil {
		return nil, fmt.Errorf("parse vocabulary: %w", err)
	}
	return New(stored.Tokens)
}
