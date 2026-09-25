package cli

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strings"
)

// Prompt returns value when non-empty, otherwise reads one line from stdin.
func Prompt(value string) (string, error) {
	if strings.TrimSpace(value) != "" {
		return value, nil
	}
	fmt.Print("prompt> ")
	line, err := bufio.NewReader(os.Stdin).ReadString('\n')
	if err != nil {
		return "", err
	}
	return line, nil
}

// Stream writes a decoded token to the writer immediately.
func Stream(w io.Writer, token string) error {
	_, err := fmt.Fprint(w, token)
	return err
}

// ReadLine reads a single trimmed line from reader.
func ReadLine(reader *bufio.Reader) (string, error) {
	line, err := reader.ReadString('\n')
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(line), nil
}
