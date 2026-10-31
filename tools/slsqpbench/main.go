// slsqpbench times SLSQP against SciPy in interleaved rounds, one thread
// each, and prints the Markdown table of docs/perf.md.
//
// Usage (from the repository root):
//
//	cd tools/slsqpbench && GOWORK=off go run . --repo ../.. --rounds 7
//
// It refuses to measure when the 1-minute load average is above --max-load
// (ratios taken on a loaded host are not meaningful).
package main

import (
	"bufio"
	"bytes"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"
)

//go:embed bench.py
var script []byte

var benchNames = []string{"SLSQPHS71", "SLSQPHS71FD", "SLSQPHS38", "SLSQPHS43", "SLSQPQuad50",
	"TrustConstrHS71", "TrustConstrHS71FD", "TrustConstrHS38", "TrustConstrHS43", "TrustConstrQuad50"}

var osExit = os.Exit

// execRun runs an external command (go, the test binary, uv); tests replace
// it to simulate them.
var execRun = func(c *exec.Cmd) error { return c.Run() }

// now and sleep are replaced by tests of --wait.
var (
	now   = time.Now
	sleep = time.Sleep
)

// loadAvg returns the 1-minute load average.
var loadAvg = func() (float64, error) {
	var raw []byte
	var err error
	if runtime.GOOS == "linux" {
		raw, err = os.ReadFile("/proc/loadavg")
	} else {
		raw, err = exec.Command("sysctl", "-n", "vm.loadavg").Output()
	}
	if err != nil {
		return 0, err
	}
	return parseLoad(string(raw))
}

// parseLoad reads the first number of "0.52 0.48 ..." or "{ 0.52 0.48 ... }".
func parseLoad(s string) (float64, error) {
	for _, f := range strings.Fields(s) {
		if v, err := strconv.ParseFloat(f, 64); err == nil {
			return v, nil
		}
	}
	return 0, fmt.Errorf("no load average in %q", s)
}

// parseGoBench extracts ns/op per benchmark (without "Benchmark" and the
// -N CPU suffix) from go test -bench output.
func parseGoBench(r io.Reader) map[string]float64 {
	out := map[string]float64{}
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		f := strings.Fields(sc.Text())
		if len(f) < 4 || !strings.HasPrefix(f[0], "Benchmark") || f[3] != "ns/op" {
			continue
		}
		name := strings.TrimPrefix(f[0], "Benchmark")
		if i := strings.LastIndex(name, "-"); i > 0 {
			name = name[:i]
		}
		if v, err := strconv.ParseFloat(f[2], 64); err == nil {
			out[name] = v * 1e-9
		}
	}
	return out
}

func median(v []float64) float64 {
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	if len(s) == 0 {
		return 0
	}
	if len(s)%2 == 1 {
		return s[len(s)/2]
	}
	return (s[len(s)/2-1] + s[len(s)/2]) / 2
}

func human(sec float64) string {
	switch {
	case sec >= 1e-3:
		return fmt.Sprintf("%.3g ms", sec*1e3)
	default:
		return fmt.Sprintf("%.3g µs", sec*1e6)
	}
}

// RootCmd returns the command line of the harness.
func RootCmd() *cobra.Command {
	var repo string
	var rounds int
	var maxLoad float64
	var wait time.Duration
	cmd := &cobra.Command{
		Use:          "slsqpbench",
		Short:        "Interleaved SLSQP and trust-constr benchmark rounds against SciPy",
		SilenceUsage: true,
		Args:         cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			w := cmd.OutOrStdout()
			load0, err := loadAvg()
			if err != nil {
				return err
			}
			// --wait polls the load until it drops below --max-load.
			deadline := now().Add(wait)
			for load0 > maxLoad && now().Before(deadline) {
				sleep(30 * time.Second)
				if load0, err = loadAvg(); err != nil {
					return err
				}
			}
			if load0 > maxLoad {
				return fmt.Errorf("1-minute load %.1f > %.1f: measurement deferred", load0, maxLoad)
			}
			tmp, err := os.MkdirTemp("", "slsqpbench")
			if err != nil {
				return err
			}
			defer os.RemoveAll(tmp)
			bin := filepath.Join(tmp, "opt.test")
			build := exec.Command("go", "test", "-c", "-o", bin, ".")
			build.Dir, build.Env = repo, append(os.Environ(), "GOWORK=off")
			build.Stdout, build.Stderr = cmd.ErrOrStderr(), cmd.ErrOrStderr()
			if err := execRun(build); err != nil {
				return fmt.Errorf("build: %w", err)
			}
			goT, pyT := map[string][]float64{}, map[string][]float64{}
			for r := 1; r <= rounds; r++ {
				var gb bytes.Buffer
				g := exec.Command(bin, "-test.run", "^$", "-test.bench", "^Benchmark(SLSQP|TrustConstr)",
					"-test.benchtime", "300ms", "-test.count", "1", "-test.cpu", "1")
				g.Dir, g.Stdout, g.Stderr = repo, &gb, cmd.ErrOrStderr()
				if err := execRun(g); err != nil {
					return fmt.Errorf("go bench: %w", err)
				}
				for k, v := range parseGoBench(&gb) {
					goT[k] = append(goT[k], v)
				}
				var pb bytes.Buffer
				p := exec.Command("uv", "run", "-q", "--no-project", "--with", "scipy==1.18.1",
					"--with", "numpy==2.5.4", "python", "-I", "-")
				p.Stdin, p.Stdout, p.Stderr = bytes.NewReader(script), &pb, cmd.ErrOrStderr()
				p.Env = append(os.Environ(), "OPENBLAS_NUM_THREADS=1", "VECLIB_MAXIMUM_THREADS=1", "OMP_NUM_THREADS=1")
				if err := execRun(p); err != nil {
					return fmt.Errorf("scipy bench: %w", err)
				}
				var py map[string]float64
				if err := json.Unmarshal(pb.Bytes(), &py); err != nil {
					return fmt.Errorf("scipy output: %w", err)
				}
				for k, v := range py {
					pyT[k] = append(pyT[k], v)
				}
				l, _ := loadAvg()
				fmt.Fprintf(cmd.ErrOrStderr(), "round %d/%d done (load %.1f)\n", r, rounds, l)
			}
			load1, _ := loadAvg()
			fmt.Fprintf(w, "Medians of %d interleaved rounds, one thread each; 1-minute load %.1f before, %.1f after.\n\n",
				rounds, load0, load1)
			fmt.Fprintln(w, "| Benchmark | Go | SciPy 1.18.1 | Speed-up |")
			fmt.Fprintln(w, "|---|---|---|---|")
			for _, n := range benchNames {
				g, p := median(goT[n]), median(pyT[n])
				if g == 0 || p == 0 {
					return fmt.Errorf("benchmark %s missing (go %d, scipy %d samples)", n, len(goT[n]), len(pyT[n]))
				}
				fmt.Fprintf(w, "| %s | %s | %s | %.1f× |\n", n, human(g), human(p), p/g)
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&repo, "repo", "../..", "path of the optimize repository")
	cmd.Flags().IntVar(&rounds, "rounds", 7, "interleaved rounds")
	cmd.Flags().Float64Var(&maxLoad, "max-load", 8, "refuse to measure above this 1-minute load average")
	cmd.Flags().DurationVar(&wait, "wait", 0, "wait up to this long for the load to drop below --max-load")
	return cmd
}

func main() {
	if err := RootCmd().Execute(); err != nil {
		osExit(1)
	}
}
