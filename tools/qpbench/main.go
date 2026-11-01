// qpbench times Quadprog against OSQP v1.0.0 and the quadprog package in
// interleaved rounds, one thread each, and prints the Markdown tables of
// docs/perf.md.
//
// Usage (from the repository root):
//
//	cd tools/qpbench && GOWORK=off go run . --repo ../.. --rounds 5
//
// What is compared (design-qp.md section 7):
//
//   - ADMM against OSQP on instances of the classes of osqp_benchmarks (random QP,
//     equality QP, portfolio, Lasso, Huber, SVM, control; problem_classes/*.py,
//     opened for this) and on small dense ones: setup time and solve time apart,
//     iteration counts side by side. OSQP's times are its own (info.setup_time,
//     info.solve_time, measured in C), as osqp_benchmarks reports them.
//   - the active set against the quadprog package (Goldfarb–Idnani, Fortran) on
//     dense positive definite instances, n up to 1000: setup plus solve against one
//     call, the call overhead of an empty-sized solve subtracted.
//   - the share of our ADMM time spent in go-sparse's LDLᵀ refactorisation and
//     triangular solves, reported separately from the rest.
//
// A round is the Go benchmark followed by the reference run, on the same
// instance files. The 1-minute load average is checked before the first round and
// after each one: a round that ends above --max-load is discarded and run again,
// and the measurement fails if the budget of attempts runs out.
package main

import (
	"bufio"
	"bytes"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"math"
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

// Sample is what the Go benchmark reports for one instance and method in one
// round, in seconds (iters, obj and status as they are).
type Sample struct {
	Setup, Solve, LDLAnalyse, LDLFactor, LDLSolve float64
	Iters, Obj, Status                            float64
}

var methodSuffix = regexp.MustCompile(`^(admm|activeset)(-\d+)?$`)

// parseGoBench reads go test -bench output of BenchmarkQP: one Sample per
// "instance/method".
func parseGoBench(r io.Reader) map[string]Sample {
	out := map[string]Sample{}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		f := strings.Fields(sc.Text())
		if len(f) < 4 || !strings.HasPrefix(f[0], "BenchmarkQP/") {
			continue
		}
		name := strings.TrimPrefix(f[0], "BenchmarkQP/")
		i := strings.LastIndex(name, "/")
		if i < 0 || !methodSuffix.MatchString(name[i+1:]) {
			continue
		}
		name = name[:i+1] + strings.SplitN(name[i+1:], "-", 2)[0]
		var s Sample
		seen := false
		for k := 2; k+1 < len(f); k += 2 {
			v, err := strconv.ParseFloat(f[k], 64)
			if err != nil {
				continue
			}
			switch f[k+1] {
			case "setup-ns/op":
				s.Setup, seen = v*1e-9, true
			case "solve-ns/op":
				s.Solve = v * 1e-9
			case "ldl-analyse-ns/op":
				s.LDLAnalyse = v * 1e-9
			case "ldl-factor-ns/op":
				s.LDLFactor = v * 1e-9
			case "ldl-solve-ns/op":
				s.LDLSolve = v * 1e-9
			case "iters":
				s.Iters = v
			case "obj":
				s.Obj = v
			case "status":
				s.Status = v
			}
		}
		if seen {
			out[name] = s
		}
	}
	return out
}

// RefOSQP and RefQuadprog are the reference side's JSON.
type RefOSQP struct {
	Setup, Solve float64
	Iter         int
	Status       string
	Obj          *float64
	RhoUpdates   int `json:"rho_updates"`
}

type RefQuadprog struct {
	Total float64
	Obj   float64
}

type Ref struct {
	Meta struct {
		Osqp, Quadprog       string
		QuadprogCallOverhead float64 `json:"quadprog_call_overhead"`
	}
	Osqp     map[string]RefOSQP
	Quadprog map[string]RefQuadprog
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

// measurement is the medians over the accepted rounds.
type measurement struct {
	goS     map[string][]Sample
	osqp    map[string][]RefOSQP
	qp      map[string][]RefQuadprog
	overhd  []float64
	osqpVer string
	qpVer   string
}

func (m *measurement) add(g map[string]Sample, r Ref) {
	for k, v := range g {
		m.goS[k] = append(m.goS[k], v)
	}
	for k, v := range r.Osqp {
		m.osqp[k] = append(m.osqp[k], v)
	}
	for k, v := range r.Quadprog {
		m.qp[k] = append(m.qp[k], v)
	}
	m.overhd = append(m.overhd, r.Meta.QuadprogCallOverhead)
	m.osqpVer, m.qpVer = r.Meta.Osqp, r.Meta.Quadprog
}

func med(n int, f func(i int) float64) float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = f(i)
	}
	return median(v)
}

// instInfo is what the tables print about an instance.
type instInfo struct {
	Name, Class string
	N, M, NNZ   int
}

func readInfo(dir string) ([]instInfo, error) {
	files, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil {
		return nil, err
	}
	var out []instInfo
	for _, f := range files {
		raw, err := os.ReadFile(f)
		if err != nil {
			return nil, err
		}
		var in Instance
		if err := json.Unmarshal(raw, &in); err != nil {
			return nil, fmt.Errorf("%s: %w", f, err)
		}
		out = append(out, instInfo{in.Name, in.Class, in.N, in.M, len(in.P.Data) + len(in.A.Data)})
	}
	return out, nil
}

// report writes the tables and returns the number of rows on which we are slower
// than the reference (setup, solve or total beyond slack).
func report(w io.Writer, m *measurement, info []instInfo, rounds int, load0, load1 float64) (slower int) {
	fmt.Fprintf(w, "Medians of %d interleaved rounds, one thread each; 1-minute load %.1f before, %.1f after.\n", rounds, load0, load1)
	fmt.Fprintf(w, "Reference: OSQP %s (C core v1.0.0), quadprog %s; quadprog call overhead %s subtracted.\n\n",
		m.osqpVer, m.qpVer, human(median(m.overhd)))
	fmt.Fprintln(w, "ADMM against OSQP (eps 1e-3, no polishing; ratio = ours / OSQP, above 1 is slower):")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "| Instance | n | m | nnz | iters ours / OSQP | setup ours | setup OSQP | ratio | solve ours | solve OSQP | ratio | go-sparse AMD + symbolic, share of setup | go-sparse LDLᵀ refactor + solves, share of total | |")
	fmt.Fprintln(w, "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
	classRatios := map[string][]float64{}
	var order []string
	for _, in := range info {
		key := in.Name + "/admm"
		g, o := m.goS[key], m.osqp[in.Name]
		if len(g) == 0 || len(o) == 0 {
			continue
		}
		gs := med(len(g), func(i int) float64 { return g[i].Setup })
		gv := med(len(g), func(i int) float64 { return g[i].Solve })
		ga := med(len(g), func(i int) float64 { return g[i].LDLAnalyse })
		gl := med(len(g), func(i int) float64 { return g[i].LDLFactor + g[i].LDLSolve })
		os_ := med(len(o), func(i int) float64 { return o[i].Setup })
		ov := med(len(o), func(i int) float64 { return o[i].Solve })
		rs, rv, rt := gs/os_, gv/ov, (gs+gv)/(os_+ov)
		flag := ""
		if rt > 1.0 {
			flag = "slower"
			slower++
		}
		if _, ok := classRatios[in.Class]; !ok {
			order = append(order, in.Class)
		}
		classRatios[in.Class] = append(classRatios[in.Class], rt)
		fmt.Fprintf(w, "| %s | %d | %d | %d | %.0f / %d | %s | %s | %.2f | %s | %s | %.2f | %.0f%% | %.0f%% | %s |\n",
			in.Name, in.N, in.M, in.NNZ, g[len(g)-1].Iters, o[len(o)-1].Iter, human(gs), human(os_), rs, human(gv), human(ov), rv,
			100*ga/gs, 100*gl/(gs+gv), flag)
	}
	fmt.Fprintln(w, "\nGeometric mean of (setup+solve) ratio per class:")
	fmt.Fprintln(w)
	for _, c := range order {
		fmt.Fprintf(w, "- %s: %.2f over %d instances\n", c, geomean(classRatios[c]), len(classRatios[c]))
	}
	fmt.Fprintln(w, "\nActive set (Goldfarb–Idnani) against the quadprog package, dense, P positive definite (setup+solve; ratio = ours / quadprog):")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "| Instance | n | m | ours | quadprog | ratio | |")
	fmt.Fprintln(w, "|---|---|---|---|---|---|---|")
	for _, in := range info {
		g, q := m.goS[in.Name+"/activeset"], m.qp[in.Name]
		if len(g) == 0 || len(q) == 0 {
			continue
		}
		gt := med(len(g), func(i int) float64 { return g[i].Setup + g[i].Solve })
		qt := med(len(q), func(i int) float64 { return q[i].Total })
		flag := ""
		if in.N >= 50 && gt/qt > 1.0 { // "no slower from n = 50 up" (design-qp.md section 7)
			flag = "slower"
			slower++
		}
		fmt.Fprintf(w, "| %s | %d | %d | %s | %s | %.2f | %s |\n", in.Name, in.N, in.M, human(gt), human(qt), gt/qt, flag)
	}
	fmt.Fprintln(w, "\nMethod choice (QPAuto): our active set against our ADMM (eps 1e-3, so much less accurate) on the same dense instances, setup+solve; the crossover settles qpAutoLimit (n + m):")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "| Instance | n + m | active set | ADMM | active set / ADMM |")
	fmt.Fprintln(w, "|---|---|---|---|---|")
	for _, in := range info {
		g, a := m.goS[in.Name+"/activeset"], m.goS[in.Name+"/admm"]
		if len(g) == 0 || len(a) == 0 {
			continue
		}
		gt := med(len(g), func(i int) float64 { return g[i].Setup + g[i].Solve })
		at := med(len(a), func(i int) float64 { return a[i].Setup + a[i].Solve })
		fmt.Fprintf(w, "| %s | %d | %s | %s | %.2f |\n", in.Name, in.N+in.M, human(gt), human(at), gt/at)
	}
	return slower
}

// prepareDir makes dir ready for a fresh instance set: it must not lie inside a
// git work tree (tens of megabytes of generated data are one `git add -A` from a
// commit) and may hold nothing but .json files, which are removed; any other
// content is refused, so that a wrong --instances never deletes anything else.
func prepareDir(dir string) error {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return err
	}
	for d := abs; ; d = filepath.Dir(d) {
		if _, err := os.Stat(filepath.Join(d, ".git")); err == nil {
			return fmt.Errorf("%s is inside the work tree %s: choose a directory outside every repository", abs, d)
		}
		if filepath.Dir(d) == d {
			break
		}
	}
	entries, err := os.ReadDir(abs)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".json" {
			return fmt.Errorf("%s holds %q, which is not an instance file: refusing to clear it", abs, e.Name())
		}
	}
	for _, e := range entries {
		if err := os.Remove(filepath.Join(abs, e.Name())); err != nil {
			return err
		}
	}
	return nil
}

func geomean(v []float64) float64 {
	s := 0.0
	for _, x := range v {
		s += math.Log(x)
	}
	return math.Exp(s / float64(len(v)))
}

// RootCmd returns the command line of the harness.
func RootCmd() *cobra.Command {
	var repo, instDir string
	var rounds int
	var maxLoad, eps, minTime float64
	var wait time.Duration
	var quick bool
	var seed uint64
	cmd := &cobra.Command{
		Use:          "qpbench",
		Short:        "Interleaved Quadprog benchmark rounds against OSQP v1.0.0 and quadprog",
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
			if instDir == "" {
				d, err := os.UserCacheDir()
				if err != nil {
					return err
				}
				instDir = filepath.Join(d, "qpbench", "instances")
			}
			if err := prepareDir(instDir); err != nil {
				return err
			}
			if err := writeInstances(instDir, Generate(seed, quick)); err != nil {
				return err
			}
			info, err := readInfo(instDir)
			if err != nil {
				return err
			}
			tmp, err := os.MkdirTemp("", "qpbench")
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
			m := &measurement{goS: map[string][]Sample{}, osqp: map[string][]RefOSQP{}, qp: map[string][]RefQuadprog{}}
			epsS := strconv.FormatFloat(eps, 'g', -1, 64)
			budget, done, discarded := 3*rounds, 0, 0
			for attempt := 1; done < rounds; attempt++ {
				if attempt > budget {
					return fmt.Errorf("only %d of %d rounds ended below load %.1f after %d attempts: measurement abandoned", done, rounds, maxLoad, budget)
				}
				var gb bytes.Buffer
				g := exec.Command(bin, "-test.run", "^$", "-test.bench", "^BenchmarkQP$", "-test.benchtime", "100ms",
					"-test.count", "1", "-test.cpu", "1")
				g.Dir, g.Stdout, g.Stderr = repo, &gb, cmd.ErrOrStderr()
				g.Env = append(os.Environ(), "QPBENCH_DIR="+instDir, "QPBENCH_EPS="+epsS, "GOMAXPROCS=1")
				if err := execRun(g); err != nil {
					return fmt.Errorf("go bench: %w", err)
				}
				goRes := parseGoBench(&gb)
				var pb bytes.Buffer
				p := exec.Command("uv", "run", "-q", "--no-project", "--with", "osqp==1.1.3", "--with", "quadprog==0.1.14",
					"--with", "numpy==2.5.4", "--with", "scipy==1.18.1", "python", "-I", "-")
				head := fmt.Sprintf("QPBENCH_DIR = %q\nMIN_TIME = %g\nEPS = %s\n", instDir, minTime, epsS)
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
			slower := report(w, m, info, rounds, load0, load1)
			fmt.Fprintf(w, "\n%d discarded round(s) for load; %d row(s) slower than the reference.\n", discarded, slower)
			return nil
		},
	}
	cmd.Flags().StringVar(&repo, "repo", "../..", "path of the optimize repository")
	cmd.Flags().StringVar(&instDir, "instances", "", "directory for the generated instances (default: the user cache directory, recreated)")
	cmd.Flags().IntVar(&rounds, "rounds", 5, "interleaved rounds")
	cmd.Flags().Float64Var(&maxLoad, "max-load", 8, "refuse to measure, and discard a round, above this 1-minute load average")
	cmd.Flags().Float64Var(&eps, "eps", 1e-3, "eps_abs = eps_rel of both ADMM solvers")
	cmd.Flags().Float64Var(&minTime, "min-time", 0.3, "seconds of repetition per reference measurement")
	cmd.Flags().DurationVar(&wait, "wait", 0, "wait up to this long for the load to drop below --max-load")
	cmd.Flags().BoolVar(&quick, "quick", false, "the smallest sizes only (a smoke run)")
	cmd.Flags().Uint64Var(&seed, "seed", 20261101, "seed of the instance generator")
	return cmd
}

func main() {
	if err := RootCmd().Execute(); err != nil {
		osExit(1)
	}
}
