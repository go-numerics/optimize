// globalbench times DifferentialEvolution, Basinhopping and Direct against
// SciPy 1.18.1 in interleaved rounds, one thread each, and prints the Markdown
// table of docs/perf.md (design-global.md section 12).
//
// Usage (from the repository root):
//
//	cd tools/globalbench && GOWORK=off go run . --repo ../.. --rounds 7
//
// The rows are those of BenchmarkGlobal (global_bench_test.go) and bench.py:
// DE on sphere, Rosenbrock and Rastrigin at N = 2, 10, 30 (default options,
// fixed seed, the same random path), DE in SciPy's vectorised form against our
// deferred updating, DE's overhead per trial on a trivial objective, basin
// hopping on Himmelblau with L-BFGS-B, and DIRECT on the documentation
// example and on Rosenbrock and sphere. Where the two sides take the same path
// (arithmetic objectives) nfev is equal and the ratio of the times is the
// comparison; on the others (Rastrigin: libm) the paths diverge and the
// comparison is the time per evaluation.
//
// A round is the Go benchmark followed by the reference run. The 1-minute load
// average is checked before the first round and after each one: a round that
// ends above --max-load is discarded and run again, and the measurement fails
// if the budget of attempts runs out; --wait polls until the load drops.
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
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"
)

//go:embed bench.py
var script []byte

var osExit = os.Exit

// execRun runs an external command (go, the test binary, uv); tests replace it.
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

// Sample is one row of one round: seconds per call and the counts.
type Sample struct {
	Time       float64
	Nfev, Nit  float64
	Seq        int // position in the output, the row order of the table
	hasSeconds bool
}

var rowSuffix = regexp.MustCompile(`-\d+$`)

// parseGoBench reads go test -bench output of BenchmarkGlobal.
func parseGoBench(r io.Reader) map[string]Sample {
	out := map[string]Sample{}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		f := strings.Fields(sc.Text())
		if len(f) < 4 || !strings.HasPrefix(f[0], "BenchmarkGlobal/") {
			continue
		}
		name := rowSuffix.ReplaceAllString(strings.TrimPrefix(f[0], "BenchmarkGlobal/"), "")
		var s Sample
		for k := 2; k+1 < len(f); k += 2 {
			v, err := strconv.ParseFloat(f[k], 64)
			if err != nil {
				continue
			}
			switch f[k+1] {
			case "ns/op":
				s.Time, s.hasSeconds = v*1e-9, true
			case "nfev":
				s.Nfev = v
			case "nit":
				s.Nit = v
			}
		}
		if s.hasSeconds {
			s.Seq = len(out)
			out[name] = s
		}
	}
	return out
}

// RefRow and Ref are the reference side's JSON.
type RefRow struct {
	Time float64
	Nfev int
	Nit  int
	Reps int
}

type Ref struct {
	Meta struct{ Scipy, Numpy string }
	Rows map[string]RefRow
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
	case sec >= 1:
		return fmt.Sprintf("%.3g s", sec)
	case sec >= 1e-3:
		return fmt.Sprintf("%.3g ms", sec*1e3)
	default:
		return fmt.Sprintf("%.3g µs", sec*1e6)
	}
}

type measurement struct {
	goS   map[string][]Sample
	ref   map[string][]RefRow
	scipy string
	numpy string
}

func (m *measurement) add(g map[string]Sample, r Ref) {
	for k, v := range g {
		m.goS[k] = append(m.goS[k], v)
	}
	for k, v := range r.Rows {
		m.ref[k] = append(m.ref[k], v)
	}
	m.scipy, m.numpy = r.Meta.Scipy, r.Meta.Numpy
}

// rowOrder lists the rows both sides measured.
func rowOrder(m *measurement) []string {
	var names []string
	for k := range m.goS {
		if _, ok := m.ref[k]; ok {
			names = append(names, k)
		}
	}
	// The order of BenchmarkGlobal's output, the order of the rows in bench.py.
	sort.Slice(names, func(i, j int) bool { return m.goS[names[i]][0].Seq < m.goS[names[j]][0].Seq })
	return names
}

// report writes the table and returns the number of rows on which we are
// slower than SciPy: by the ratio of total times when both sides took the same
// number of evaluations, by the ratio of times per evaluation otherwise.
func report(w io.Writer, m *measurement, rounds int, load0, load1 float64) (slower int) {
	fmt.Fprintf(w, "Medians of %d interleaved rounds, one thread each; 1-minute load %.1f before, %.1f after.\n", rounds, load0, load1)
	fmt.Fprintf(w, "Reference: SciPy %s, NumPy %s. Ratio = ours / SciPy, above 1 is slower; per-eval compares the time per objective evaluation.\n\n", m.scipy, m.numpy)
	fmt.Fprintln(w, "| Row | nfev ours / SciPy | time ours | time SciPy | ratio | per-eval ratio | |")
	fmt.Fprintln(w, "|---|---|---|---|---|---|---|")
	for _, name := range rowOrder(m) {
		g, r := m.goS[name], m.ref[name]
		gt, rt, gn, rn := make([]float64, len(g)), make([]float64, len(r)), 0.0, 0
		for i := range g {
			gt[i], gn = g[i].Time, g[i].Nfev
		}
		for i := range r {
			rt[i], rn = r[i].Time, r[i].Nfev
		}
		tg, tr := median(gt), median(rt)
		ratio := tg / tr
		perEval := (tg / gn) / (tr / float64(rn))
		verdict := ratio
		if gn != float64(rn) {
			verdict = perEval
		}
		flag := ""
		if verdict > 1.0 {
			flag = "slower"
			slower++
		}
		fmt.Fprintf(w, "| %s | %.0f / %d | %s | %s | %.3f | %.3f | %s |\n", name, gn, rn, human(tg), human(tr), ratio, perEval, flag)
	}
	return slower
}

// RootCmd returns the command line of the harness.
func RootCmd() *cobra.Command {
	var repo string
	var rounds int
	var maxLoad, minTime float64
	var wait time.Duration
	cmd := &cobra.Command{
		Use:          "globalbench",
		Short:        "Interleaved DE / basin hopping / DIRECT benchmark rounds against SciPy",
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
			tmp, err := os.MkdirTemp("", "globalbench")
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
			m := &measurement{goS: map[string][]Sample{}, ref: map[string][]RefRow{}}
			budget, done, discarded := 3*rounds, 0, 0
			for attempt := 1; done < rounds; attempt++ {
				if attempt > budget {
					return fmt.Errorf("only %d of %d rounds ended below load %.1f after %d attempts: measurement abandoned", done, rounds, maxLoad, budget)
				}
				var gb bytes.Buffer
				g := exec.Command(bin, "-test.run", "^$", "-test.bench", "^BenchmarkGlobal$", "-test.benchtime", "200ms",
					"-test.count", "1", "-test.cpu", "1")
				g.Dir, g.Stdout, g.Stderr = repo, &gb, cmd.ErrOrStderr()
				g.Env = append(os.Environ(), "GOMAXPROCS=1")
				if err := execRun(g); err != nil {
					return fmt.Errorf("go bench: %w", err)
				}
				goRes := parseGoBench(&gb)
				var pb bytes.Buffer
				p := exec.Command("uv", "run", "-q", "--no-project", "--with", "numpy==2.5.4", "--with", "scipy==1.18.1", "python", "-I", "-")
				head := fmt.Sprintf("MIN_TIME = %g\n", minTime)
				p.Stdin, p.Stdout, p.Stderr = strings.NewReader(head+string(script)), &pb, cmd.ErrOrStderr()
				p.Env = append(os.Environ(), "OPENBLAS_NUM_THREADS=1", "VECLIB_MAXIMUM_THREADS=1", "OMP_NUM_THREADS=1")
				if err := execRun(p); err != nil {
					return fmt.Errorf("reference bench: %w", err)
				}
				var ref Ref
				if err := json.Unmarshal(pb.Bytes(), &ref); err != nil {
					return fmt.Errorf("reference output: %w", err)
				}
				l, err := loadAvg()
				if err != nil {
					return err
				}
				// The host was quiet at the start; a round that ended on a busy one
				// is not a measurement: discard it and run it again.
				if l > maxLoad {
					discarded++
					fmt.Fprintf(cmd.ErrOrStderr(), "round attempt %d discarded: 1-minute load %.1f > %.1f\n", attempt, l, maxLoad)
					continue
				}
				m.add(goRes, ref)
				done++
				fmt.Fprintf(cmd.ErrOrStderr(), "round %d/%d done (load %.1f)\n", done, rounds, l)
			}
			load1, _ := loadAvg()
			slower := report(w, m, rounds, load0, load1)
			fmt.Fprintf(w, "\n%d discarded round(s) for load; %d row(s) slower than SciPy.\n", discarded, slower)
			return nil
		},
	}
	cmd.Flags().StringVar(&repo, "repo", "../..", "path of the optimize repository")
	cmd.Flags().IntVar(&rounds, "rounds", 7, "interleaved rounds")
	cmd.Flags().Float64Var(&maxLoad, "max-load", 8, "refuse to measure, and discard a round, above this 1-minute load average")
	cmd.Flags().Float64Var(&minTime, "min-time", 0.3, "seconds of repetition per reference measurement")
	cmd.Flags().DurationVar(&wait, "wait", 0, "wait up to this long for the load to drop below --max-load")
	return cmd
}

func main() {
	if err := RootCmd().Execute(); err != nil {
		osExit(1)
	}
}
