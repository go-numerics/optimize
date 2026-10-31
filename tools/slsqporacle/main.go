// slsqporacle runs SciPy's SLSQP on the Hock–Schittkowski problems of hs.py
// and writes the results to the fixture file the optimize tests replay; its
// diff subcommand compares two such fixtures (for instance SciPy's arm64 and
// x86_64 wheels, or a regenerated fixture against the committed one).
//
// Usage (from tools/slsqporacle):
//
//	GOWORK=off go run . --platform linux/amd64 -o ../../testdata/slsqp.json
//	GOWORK=off go run . diff a.json b.json
//
// Generating needs uv on PATH (and Docker with --platform); SciPy and NumPy
// are pinned below. The Python script is fed on standard input to an
// isolated interpreter (python -I), so nothing in the working directory is
// imported.
package main

import (
	"bytes"
	_ "embed"
	"errors"
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

// ErrDisagree is returned by diff when the fixtures differ.
var ErrDisagree = errors.New("fixtures disagree")

func readFixture(path string) (*Fixture, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	fx, err := Decode(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return fx, nil
}

func diffCmd() *cobra.Command {
	return &cobra.Command{
		Use:          "diff first.json second.json",
		Short:        "Report the cases on which two fixtures disagree (exit 1 if any)",
		SilenceUsage: true,
		Args:         cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			a, err := readFixture(args[0])
			if err != nil {
				return err
			}
			b, err := readFixture(args[1])
			if err != nil {
				return err
			}
			w := cmd.OutOrStdout()
			d := Diff(a, b)
			names := map[string]bool{}
			for _, x := range d {
				fmt.Fprintln(w, x)
				names[x.Case] = true
			}
			fmt.Fprintf(w, "%d cases (%s, %s) against %d (%s, %s): %d disagree\n",
				len(a.Cases), a.Scipy, a.Machine, len(b.Cases), b.Scipy, b.Machine, len(names))
			if len(d) > 0 {
				return ErrDisagree
			}
			return nil
		},
	}
}

// RootCmd returns the command line of the harness.
func RootCmd() *cobra.Command {
	var out, platform, method string
	cmd := &cobra.Command{
		Use:          "slsqporacle",
		Short:        "Generate the SciPy fixtures of a constrained method (testdata/slsqp.json, testdata/trustconstr.json)",
		SilenceUsage: true,
		Args:         cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			if method != "SLSQP" && method != "trust-constr" {
				return fmt.Errorf("--method must be SLSQP or trust-constr, not %q", method)
			}
			src := append([]byte(fmt.Sprintf("METHOD = %q\n", method)), script...)
			var buf bytes.Buffer
			if err := runner(command(platform), bytes.NewReader(src), &buf, cmd.ErrOrStderr()); err != nil {
				return fmt.Errorf("oracle: %w", err)
			}
			f, err := Decode(&buf)
			if err != nil {
				return fmt.Errorf("oracle output: %w", err)
			}
			var enc bytes.Buffer
			if err := Encode(&enc, f); err != nil {
				return err
			}
			if err := os.WriteFile(out, enc.Bytes(), 0o644); err != nil {
				return err
			}
			fmt.Fprintf(cmd.OutOrStdout(), "wrote %d cases (SciPy %s, %s) to %s\n", len(f.Cases), f.Scipy, f.Machine, out)
			return nil
		},
	}
	cmd.Flags().StringVarP(&out, "output", "o", "testdata/slsqp.json", "fixture file to write")
	cmd.Flags().StringVar(&method, "method", "SLSQP", "minimize method: SLSQP or trust-constr")
	cmd.Flags().StringVar(&platform, "platform", "",
		"run SciPy in a container of this platform (e.g. linux/amd64) instead of on the host")
	cmd.AddCommand(diffCmd())
	return cmd
}

func main() {
	if err := RootCmd().Execute(); err != nil {
		osExit(1)
	}
}
