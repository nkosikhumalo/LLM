package tokenizer_test

import (
	"path/filepath"
	"testing"

	"github.com/nkosikhumalo/microllm/go/internal/tokenizer"
)

func TestEncodeDecodeRoundTrip(t *testing.T) {
	vocab, err := tokenizer.Build("hello\n")
	if err != nil {
		t.Fatal(err)
	}
	ids, err := vocab.Encode("hello")
	if err != nil {
		t.Fatal(err)
	}
	text, err := vocab.Decode(ids)
	if err != nil {
		t.Fatal(err)
	}
	if text != "hello" {
		t.Fatalf("got %q", text)
	}
}

func TestSaveLoadPreservesOrder(t *testing.T) {
	vocab, err := tokenizer.Build("cab")
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "vocab.json")
	if err := tokenizer.Save(path, vocab); err != nil {
		t.Fatal(err)
	}
	loaded, err := tokenizer.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(loaded.Tokens) != len(vocab.Tokens) {
		t.Fatalf("token count %d vs %d", len(loaded.Tokens), len(vocab.Tokens))
	}
	for i := range vocab.Tokens {
		if loaded.Tokens[i] != vocab.Tokens[i] {
			t.Fatalf("token %d: %q vs %q", i, loaded.Tokens[i], vocab.Tokens[i])
		}
	}
}

func TestContinuationPiecesAreDistinctButDecodeSeamlessly(t *testing.T) {
	vocab, err := tokenizer.Build("abcdefghijklm")
	if err != nil {
		t.Fatal(err)
	}
	ids, err := vocab.Encode("abcdefghijklm")
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) < 2 {
		t.Fatalf("expected a continuation piece for a word longer than the maximum piece length, got %d token", len(ids))
	}
	if got := vocab.Tokens[ids[0]]; len(got) >= 2 && got[:2] == "##" {
		t.Fatalf("word-start token is marked as a continuation: %q", got)
	}
	if got := vocab.Tokens[ids[1]]; len(got) < 2 || got[:2] != "##" {
		t.Fatalf("expected continuation marker on second piece, got %q", got)
	}
	decoded, err := vocab.Decode(ids)
	if err != nil {
		t.Fatal(err)
	}
	if decoded != "abcdefghijklm" {
		t.Fatalf("decoded %q", decoded)
	}
}

func TestEndMarkerIsOneReservedToken(t *testing.T) {
	vocab, err := tokenizer.Build("A small language model <|end|>")
	if err != nil {
		t.Fatal(err)
	}
	ids, err := vocab.Encode("<|end|>")
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 1 {
		t.Fatalf("end marker encoded as %d tokens, want 1", len(ids))
	}
	if vocab.Tokens[ids[0]] != tokenizer.EndToken {
		t.Fatalf("encoded token %q, want %q", vocab.Tokens[ids[0]], tokenizer.EndToken)
	}
	decoded, err := vocab.Decode(ids)
	if err != nil {
		t.Fatal(err)
	}
	if decoded != tokenizer.EndToken {
		t.Fatalf("decoded %q", decoded)
	}
}

func TestUnknownCharacterFails(t *testing.T) {
	vocab, err := tokenizer.Build("ab")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := vocab.Encode("c"); err == nil {
		t.Fatal("expected unknown character error")
	}
}
