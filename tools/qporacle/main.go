// qporacle runs the quadratic programs of qp.py through OSQP, Clarabel and
// quadprog and writes the results to the fixture files the optimize tests
// replay (docs/design-qp.md section 6.1); its diff subcommand compares two such
// fixtures (SciPy and OSQP wheels of two platforms, or a regenerated fixture
// against the committed one).
//
// Usage (from tools/qporacle):
//
//	GOWORK=off go run . gen --platform linux/amd64 -o ../../testdata/qp/planted.json
//	GOWORK=off go run . diff a.json b.json
//
// Generating needs uv on PATH (and Docker with --platform). The packages are
// pinned below; the Python script is fed on standard input to an isolated
// interpreter (python -I), so nothing in the working directory is imported.
// Before anything else the script checks that the pinned osqp wheel bundles
// the OSQP v1.0.0 C core, and gen checks it again on what the script reports:
// a wheel with another core makes the harness fail, not the fixture drift.
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

//go:embed qp.py
var script []byte

// The pinned environment. quadprog (GPL) is run, never read.
const (
	osqpPin     = "osqp==1.1.3"
	clarabelPin = "clarabel==0.11.1"
	quadprogPin = "quadprog==0.1.14"
	numpyPin    = "numpy==2.5.4"
	scipyPin    = "scipy==1.18.1"
	// wantCore is the OSQP C core the design targets (docs/design-qp.md 4.0).
	wantCore = "1.0.0"
)

// osExit is replaced by tests.
var osExit = os.Exit

// uvImage is the container image used with --platform.
const uvImage = "ghcr.io/astral-sh/uv:0.9.5-python3.12-bookworm-slim"

// command returns the oracle command line: uv on the host, or uv inside a
// container of the given platform (e.g. linux/amd64).
func command(platform string) []string {
	uv := []string{"uv", "run", "-q", "--no-project",
		"--with", osqpPin, "--with", clarabelPin, "--with", quadprogPin,
		"--with", numpyPin, "--with", scipyPin, "python", "-I", "-"}
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

// ErrCore is returned when the osqp wheel does not bundle OSQP v1.0.0.
var ErrCore = errors.New("the osqp wheel does not bundle the OSQP v1.0.0 C core")

// CheckCore fails unless the fixture was produced by the OSQP v1.0.0 core.
func CheckCore(f *Fixture) error {
	if f.OsqpCore != wantCore {
		return fmt.Errorf("%w: osqp %s reports core %q", ErrCore, f.Osqp, f.OsqpCore)
	}
	return nil
}

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
	var out, platform, set string
	cmd := &cobra.Command{
		Use:          "gen",
		Short:        "Run the planted and hand problems (or the generated sparse set) through OSQP, Clarabel and quadprog",
		SilenceUsage: true,
		Args:         cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			if set != "planted" && set != "sparse" {
				return fmt.Errorf("unknown --set %q: want planted or sparse", set)
			}
			var buf bytes.Buffer
			// The set is chosen by a variable defined ahead of the embedded script.
			in := append([]byte("QP_SET = \""+set+"\"\n"), script...)
			if err := runner(command(platform), bytes.NewReader(in), &buf, cmd.ErrOrStderr()); err != nil {
				return fmt.Errorf("oracle: %w", err)
			}
			f, err := Decode(&buf)
			if err != nil {
				return fmt.Errorf("oracle output: %w", err)
			}
			if err := CheckCore(f); err != nil {
				return err
			}
			var enc bytes.Buffer
			_ = Encode(&enc, f) // cannot fail: f was just decoded from JSON
			if err := os.WriteFile(out, enc.Bytes(), 0o644); err != nil {
				return err
			}
			fmt.Fprintf(cmd.OutOrStdout(), "wrote %d cases (osqp %s, core %s, clarabel %s, quadprog %s, %s) to %s\n",
				len(f.Cases), f.Osqp, f.OsqpCore, f.Clarabel, f.Quadprog, f.Machine, out)
			return nil
		},
	}
	cmd.Flags().StringVarP(&out, "output", "o", "testdata/qp/planted.json", "fixture file to write")
	cmd.Flags().StringVar(&set, "set", "planted",
		"problem set: planted (the hand and planted problems) or sparse (generated sparse problems in the shape of the OSQP paper's random QP, equality QP, Lasso and portfolio classes)")
	cmd.Flags().StringVar(&platform, "platform", "",
		"run in a container of this platform (e.g. linux/amd64) instead of on the host")
	return cmd
}

func diffCmd() *cobra.Command {
	var tol float64
	cmd := &cobra.Command{
		Use:          "diff first.json second.json",
		Short:        "Report the cases and solvers on which two fixtures disagree (exit 1 if any)",
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
			d := Diff(a, b, tol)
			for _, x := range d {
				fmt.Fprintln(w, x)
			}
			fmt.Fprintf(w, "%d cases (osqp %s, %s) against %d (osqp %s, %s): %d disagreements\n",
				len(a.Cases), a.Osqp, a.Machine, len(b.Cases), b.Osqp, b.Machine, len(d))
			if len(d) > 0 {
				return ErrDisagree
			}
			return nil
		},
	}
	cmd.Flags().Float64Var(&tol, "tol", 1e-9, "relative tolerance on x, y and the objective (iteration counts and statuses are compared exactly)")
	return cmd
}

// RootCmd returns the command line of the harness.
func RootCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:          "qporacle",
		Short:        "Oracle fixtures for the quadratic-programming solvers (OSQP, Clarabel, quadprog)",
		SilenceUsage: true,
	}
	cmd.AddCommand(genCmd(), diffCmd(), mmCmd())
	return cmd
}

func main() {
	if err := RootCmd().Execute(); err != nil {
		osExit(1)
	}
}
