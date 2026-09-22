package cli

import (
	"bufio"
	"fmt"
	"io"
	"os"
)

func Prompt(value string) (string, error) {
	if value != "" {
		return value, nil
	}
	fmt.Print("prompt> ")
	return bufio.NewReader(os.Stdin).ReadString('\n')
}
func Stream(w io.Writer, token string) error { _, e := fmt.Fprint(w, token); return e }
