// globaloracle runs SciPy's global optimisers on the problems of global.py and
// writes the results to the fixture files the optimize tests replay
// (testdata/global/<method>.json); its diff subcommand compares two such
// fixtures (SciPy's arm64 and x86_64 wheels, or a regenerated fixture against
// the committed one).
//
// Usage (from tools/globaloracle):
//
//	GOWORK=off go run . gen direct --platform linux/amd64 -o ../../testdata/global/direct.json
//	GOWORK=off go run . diff a.json b.json
//
// Generating needs uv on PATH (and Docker with --platform); SciPy and NumPy
// are pinned below. The Python script is fed on standard input to an
// isolated interpreter (python -I), so nothing in the working directory is
// imported. Fixtures come from a linux/amd64 container because SciPy's
// x86_64 wheel does not fuse multiply-adds, as Python itself does not.
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

//go:embed global.py
var script []byte

const (
	scipyPin = "scipy==1.18.1"
	numpyPin = "numpy==2.5.4"
)

// methods lists the methods global.py can run; the others of the design
// (differential evolution, basin hopping) are added with their ports.
var methods = []string{"direct"}

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

func genCmd() *cobra.Command {
	var out, platform string
	cmd := &cobra.Command{
		Use:          "gen method",
		Short:        "Run SciPy and write the fixture of a method (direct)",
		SilenceUsage: true,
		Args:         cobra.ExactArgs(1),
		ValidArgs:    methods,
		RunE: func(cmd *cobra.Command, args []string) error {
			method := args[0]
			if method != "direct" {
				return fmt.Errorf("unknown method %q (have %v)", method, methods)
			}
			if out == "" {
				out = "testdata/global/" + method + ".json"
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
			if f.Method != method {
				return fmt.Errorf("oracle output: method %q, asked for %q", f.Method, method)
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
	cmd.Flags().StringVarP(&out, "output", "o", "", "fixture file to write (default testdata/global/<method>.json)")
	cmd.Flags().StringVar(&platform, "platform", "",
		"run SciPy in a container of this platform (e.g. linux/amd64) instead of on the host")
	return cmd
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
	cmd := &cobra.Command{
		Use:          "globaloracle",
		Short:        "Generate and compare the SciPy fixtures of the global optimisers (testdata/global/*.json)",
		SilenceUsage: true,
		Args:         cobra.NoArgs,
	}
	cmd.AddCommand(genCmd(), diffCmd())
	return cmd
}

func main() {
	if err := RootCmd().Execute(); err != nil {
		osExit(1)
	}
}
