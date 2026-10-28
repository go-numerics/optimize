// slsqporacle runs SciPy's SLSQP on the Hock–Schittkowski problems of hs.py
// and writes the results to the fixture file the optimize tests replay.
//
// Usage (from the repository root):
//
//	cd tools/slsqporacle && GOWORK=off go run . -o ../../testdata/slsqp.json
//
// It needs uv on PATH; SciPy and NumPy are pinned below. The Python script is
// fed on standard input to an isolated interpreter (python -I), so nothing in
// the working directory is imported.
package main

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"

	"github.com/spf13/cobra"
)

//go:embed hs.py
var script []byte

const (
	scipyPin = "scipy==1.18.1"
	numpyPin = "numpy==2.5.4"
)

// fixture is the part of the output the harness validates before writing.
type fixture struct {
	Scipy string            `json:"scipy"`
	Cases []json.RawMessage `json:"cases"`
}

// osExit is replaced by tests.
var osExit = os.Exit

// uvImage is the container image used with --platform.
const uvImage = "ghcr.io/astral-sh/uv:0.9.5-python3.12-bookworm-slim"

// command returns the oracle command line: uv on the host, or uv inside a
// container of the given platform (e.g. linux/amd64).
func command(platform string) []string {
	uv := []string{"uv", "run", "-q", "--no-project", "--with", scipyPin, "--with", numpyPin, "python", "-I", "-"}
	if platform == "" {
		return uv
	}
	return append([]string{"docker", "run", "--rm", "-i", "--platform", platform, uvImage}, uv...)
}

// runner runs the oracle command; tests replace it.
var runner = func(argv []string, stdin io.Reader, stdout, stderr io.Writer) error {
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = stdin, stdout, stderr
	return cmd.Run()
}

// RootCmd returns the command line of the harness.
func RootCmd() *cobra.Command {
	var out, platform string
	cmd := &cobra.Command{
		Use:          "slsqporacle",
		Short:        "Generate the SciPy SLSQP fixtures (testdata/slsqp.json)",
		SilenceUsage: true,
		Args:         cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			var buf bytes.Buffer
			if err := runner(command(platform), bytes.NewReader(script), &buf, cmd.ErrOrStderr()); err != nil {
				return fmt.Errorf("oracle: %w", err)
			}
			var f fixture
			if err := json.Unmarshal(buf.Bytes(), &f); err != nil {
				return fmt.Errorf("oracle output is not JSON: %w", err)
			}
			if len(f.Cases) == 0 {
				return fmt.Errorf("oracle produced no cases")
			}
			var pretty bytes.Buffer
			if err := json.Indent(&pretty, buf.Bytes(), "", " "); err != nil {
				return err
			}
			pretty.WriteByte('\n')
			if err := os.WriteFile(out, pretty.Bytes(), 0o644); err != nil {
				return err
			}
			fmt.Fprintf(cmd.OutOrStdout(), "wrote %d cases (SciPy %s) to %s\n", len(f.Cases), f.Scipy, out)
			return nil
		},
	}
	cmd.Flags().StringVarP(&out, "output", "o", "testdata/slsqp.json", "fixture file to write")
	cmd.Flags().StringVar(&platform, "platform", "",
		"run SciPy in a container of this platform (e.g. linux/amd64) instead of on the host")
	return cmd
}

func main() {
	if err := RootCmd().Execute(); err != nil {
		osExit(1)
	}
}
