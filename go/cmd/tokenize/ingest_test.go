package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCollectInputsRecursivelyFindsFilesRegardlessOfExtension(t *testing.T) {
	root := t.TempDir()
	output := filepath.Join(root, "generated")
	if err := os.MkdirAll(filepath.Join(root, "nested"), 0755); err != nil {
		t.Fatal(err)
	}
	for name, body := range map[string]string{
		"notes.md":            "notes",
		"nested/source.go":    "package sample",
		"generated/old.txt":   "old generated file",
		".hidden/ignored.txt": "hidden file",
	} {
		path := filepath.Join(root, name)
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0644); err != nil {
			t.Fatal(err)
		}
	}
	got, err := collectInputs(root, output)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("collectInputs returned %v, want only the two visible source files", got)
	}
}

func TestExtractFileTextDoesNotRequireTextExtension(t *testing.T) {
	path := filepath.Join(t.TempDir(), "notes.md")
	if err := os.WriteFile(path, []byte("Readable text"), 0644); err != nil {
		t.Fatal(err)
	}
	got, err := extractFileText(path)
	if err != nil || got != "Readable text" {
		t.Fatalf("extractFileText = %q, %v", got, err)
	}
}

func TestHTMLExtractionDropsScriptsAndTags(t *testing.T) {
	path := filepath.Join(t.TempDir(), "page.html")
	if err := os.WriteFile(path, []byte("<html><body>Hello &amp; welcome<script>secret()</script></body></html>"), 0644); err != nil {
		t.Fatal(err)
	}
	got, err := extractFileText(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, "Hello & welcome") || strings.Contains(got, "secret") {
		t.Fatalf("unexpected extracted HTML text: %q", got)
	}
}

func TestMediaRequiresTranscriptAndReadsSidecar(t *testing.T) {
	base := filepath.Join(t.TempDir(), "lecture")
	video := base + ".mp4"
	if err := os.WriteFile(video, []byte{0, 1, 2, 3}, 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := extractFileText(video); err == nil || !strings.Contains(err.Error(), "transcript") {
		t.Fatalf("expected clear transcript guidance, got %v", err)
	}
	if err := os.WriteFile(base+".srt", []byte("A useful transcript"), 0644); err != nil {
		t.Fatal(err)
	}
	got, err := extractFileText(video)
	if err != nil || got != "A useful transcript" {
		t.Fatalf("extractFileText(media) = %q, %v", got, err)
	}
}
