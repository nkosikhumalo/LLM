// Package tokenizer provides deterministic character-level tokenization.
package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Vocab associates each UTF-8 character with one stable integer ID.
type Vocab struct {
	Tokens []string `json:"tokens"`
	index  map[string]int
}

// New validates tokens and creates a vocabulary with the supplied ordering.
func New(tokens []string) (*Vocab, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("vocabulary is empty")
	}

	vocabulary := &Vocab{
		Tokens: append([]string(nil), tokens...),
		index:  make(map[string]int, len(tokens)),
	}
	for tokenID, token := range vocabulary.Tokens {
		if token == "" {
			return nil, fmt.Errorf("token %d is empty", tokenID)
		}
		if _, alreadyPresent := vocabulary.index[token]; alreadyPresent {
			return nil, fmt.Errorf("duplicate token %q", token)
		}
		vocabulary.index[token] = tokenID
	}
	return vocabulary, nil
}

// Build creates a deterministic vocabulary by sorting the unique characters.
func Build(corpus string) (*Vocab, error) {
	uniqueTokens := make(map[string]struct{})
	for _, character := range corpus {
		uniqueTokens[string(character)] = struct{}{}
	}

	tokens := make([]string, 0, len(uniqueTokens))
	for token := range uniqueTokens {
		tokens = append(tokens, token)
	}
	sort.Strings(tokens)
	return New(tokens)
}

// Encode converts text to vocabulary IDs.
func (vocabulary *Vocab) Encode(text string) ([]int, error) {
	tokenIDs := make([]int, 0, len([]rune(text)))
	for _, character := range text {
		tokenID, exists := vocabulary.index[string(character)]
		if !exists {
			return nil, fmt.Errorf("character %q is not in vocabulary", character)
		}
		tokenIDs = append(tokenIDs, tokenID)
	}
	return tokenIDs, nil
}

// Decode converts vocabulary IDs back to text.
func (vocabulary *Vocab) Decode(tokenIDs []int) (string, error) {
	var text strings.Builder
	for _, tokenID := range tokenIDs {
		if tokenID < 0 || tokenID >= len(vocabulary.Tokens) {
			return "", fmt.Errorf("token ID %d is outside vocabulary", tokenID)
		}
		text.WriteString(vocabulary.Tokens[tokenID])
	}
	return text.String(), nil
}

// Save writes the public vocabulary representation as JSON.
func Save(path string, vocabulary *Vocab) error {
	encodedVocabulary, err := json.MarshalIndent(vocabulary, "", "  ")
	if err != nil {
		return fmt.Errorf("encode vocabulary: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create vocabulary directory: %w", err)
	}
	return os.WriteFile(path, encodedVocabulary, 0o644)
}

// Load reads a vocabulary written by Save.
func Load(path string) (*Vocab, error) {
	encodedVocabulary, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var storedVocabulary struct {
		Tokens []string `json:"tokens"`
	}
	if err := json.Unmarshal(encodedVocabulary, &storedVocabulary); err != nil {
		return nil, fmt.Errorf("parse vocabulary: %w", err)
	}
	return New(storedVocabulary.Tokens)
}
