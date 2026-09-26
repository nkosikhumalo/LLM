package main

import (
	"bytes"
	"fmt"
	"html"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"
)

var (
	scriptStyleTags = regexp.MustCompile(`(?is)<script[^>]*>.*?</script>|<style[^>]*>.*?</style>`)
	htmlTags        = regexp.MustCompile(`(?s)<[^>]+>`)
)

// collectInputs accepts one file or recursively gathers every regular file in a directory.
func collectInputs(input, output string) ([]string, error) {
	info, err := os.Stat(input)
	if err != nil {
		return nil, fmt.Errorf("open dataset %q: %w", input, err)
	}
	if !info.IsDir() {
		return []string{input}, nil
	}
	outputAbs, _ := filepath.Abs(output)
	files := make([]string, 0)
	err = filepath.WalkDir(input, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			if path != input && strings.HasPrefix(entry.Name(), ".") {
				return filepath.SkipDir
			}
			pathAbs, _ := filepath.Abs(path)
			if path != input && (pathAbs == outputAbs || strings.HasPrefix(pathAbs, outputAbs+string(filepath.Separator))) {
				return filepath.SkipDir
			}
			return nil
		}
		if !entry.Type().IsRegular() {
			return nil
		}
		files = append(files, path)
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("scan dataset directory: %w", err)
	}
	sort.Strings(files)
	if len(files) == 0 {
		return nil, fmt.Errorf("no regular files found in %q", input)
	}
	return files, nil
}

func extractFileText(path string) (string, error) {
	extension := strings.ToLower(filepath.Ext(path))
	switch extension {
	case ".pdf":
		if _, err := exec.LookPath("pdftotext"); err != nil {
			return "", fmt.Errorf("%s is a PDF; install Poppler (pdftotext) to extract its text", path)
		}
		out, err := exec.Command("pdftotext", "-layout", path, "-").Output()
		if err != nil {
			return "", fmt.Errorf("extract PDF text from %s: %w", path, err)
		}
		return string(out), nil
	case ".doc", ".docx", ".odt", ".rtf", ".ppt", ".pptx", ".odp", ".xls", ".xlsx", ".ods":
		return extractOfficeText(path)
	case ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp":
		if _, err := exec.LookPath("tesseract"); err != nil {
			return "", fmt.Errorf("%s is an image; install Tesseract for OCR. This text model learns recognized text, not image contents", path)
		}
		out, err := exec.Command("tesseract", path, "stdout").Output()
		if err != nil {
			return "", fmt.Errorf("OCR image %s: %w", path, err)
		}
		return string(out), nil
	case ".mp4", ".mov", ".mkv", ".avi", ".webm", ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac":
		return extractMediaTranscript(path)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	if utf8.Valid(data) && !bytes.Contains(data, []byte{0}) {
		text := strings.TrimPrefix(string(data), "\ufeff")
		if extension == ".html" || extension == ".htm" {
			text = html.UnescapeString(htmlTags.ReplaceAllString(scriptStyleTags.ReplaceAllString(text, " "), " "))
		}
		return text, nil
	}
	return "", fmt.Errorf("%s is binary and has no text extractor; this MiniLLM accepts text. For audio/video, add a .txt, .srt, or .vtt transcript beside the media file", path)
}

func extractOfficeText(path string) (string, error) {
	binary, err := exec.LookPath("libreoffice")
	if err != nil {
		return "", fmt.Errorf("%s is an office document; install LibreOffice to extract its text", path)
	}
	directory, err := os.MkdirTemp("", "minillm-office-")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(directory)
	cmd := exec.Command(binary, "--headless", "--convert-to", "txt:Text", "--outdir", directory, path)
	if out, err := cmd.CombinedOutput(); err != nil {
		return "", fmt.Errorf("extract office text from %s: %w (%s)", path, err, strings.TrimSpace(string(out)))
	}
	converted := filepath.Join(directory, strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))+".txt")
	text, err := os.ReadFile(converted)
	if err != nil {
		return "", fmt.Errorf("read extracted text for %s: %w", path, err)
	}
	return string(text), nil
}

func extractMediaTranscript(path string) (string, error) {
	base := strings.TrimSuffix(path, filepath.Ext(path))
	for _, ext := range []string{".txt", ".srt", ".vtt"} {
		candidate := base + ext
		if candidate == path {
			continue
		}
		if data, err := os.ReadFile(candidate); err == nil {
			return string(data), nil
		}
	}
	return "", fmt.Errorf("%s is audio/video; provide a same-name .txt, .srt, or .vtt transcript. This text-only model cannot learn sound or video frames directly", path)
}
