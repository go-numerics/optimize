package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// --- instances ---------------------------------------------------------------

func (c CSC) check(t *testing.T, what string) {
	t.Helper()
	if len(c.Indptr) != c.Cols+1 || c.Indptr[0] != 0 || c.Indptr[c.Cols] != len(c.Data) || len(c.Index) != len(c.Data) {
		t.Fatalf("%s: malformed pointers", what)
	}
	for j := 0; j < c.Cols; j++ {
		for k := c.Indptr[j]; k < c.Indptr[j+1]; k++ {
			if c.Index[k] < 0 || c.Index[k] >= c.Rows || (k > c.Indptr[j] && c.Index[k] <= c.Index[k-1]) {
				t.Fatalf("%s: column %d row %d out of range or unsorted", what, j, c.Index[k])
			}
			if c.Data[k] == 0 || math.IsNaN(c.Data[k]) || math.IsInf(c.Data[k], 0) {
				t.Fatalf("%s: bad entry %g", what, c.Data[k])
			}
		}
	}
}

func TestGenerateInstancesAreWellFormed(t *testing.T) {
	for _, quick := range []bool{true, false} {
		if !quick && testing.Short() {
			continue
		}
		set := Generate(1, quick)
		classes := map[string]int{}
		for _, in := range set {
			classes[in.Class]++
			in.P.check(t, in.Name+" P")
			in.A.check(t, in.Name+" A")
			if in.P.Rows != in.N || in.P.Cols != in.N || in.A.Rows != in.M || in.A.Cols != in.N ||
				len(in.Q) != in.N || len(in.L) != in.M || len(in.U) != in.M {
				t.Errorf("%s: dimensions disagree", in.Name)
			}
			for j := 0; j < in.N; j++ {
				for k := in.P.Indptr[j]; k < in.P.Indptr[j+1]; k++ {
					if in.P.Index[k] > j {
						t.Fatalf("%s: P holds an entry below the diagonal", in.Name)
					}
				}
			}
			for i := range in.L {
				if in.L[i] > in.U[i] || in.L[i] >= infBound || in.U[i] <= -infBound {
					t.Errorf("%s: row %d has the bounds [%g, %g]", in.Name, i, in.L[i], in.U[i])
				}
			}
		}
		for _, c := range []string{"random-qp", "eq-qp", "portfolio", "lasso", "huber", "svm", "control", "dense"} {
			if classes[c] == 0 {
				t.Errorf("quick=%v: no %s instance", quick, c)
			}
		}
	}
}

// The classes have the structure of the reference's problem_classes: sizes and
// the equality rows that make them what they are.
func TestGenerateStructure(t *testing.T) {
	g := newGen(7)
	if in := g.randomQP(20); in.N != 20 || in.M != 200 {
		t.Errorf("random QP: n %d m %d", in.N, in.M)
	}
	if in := g.eqQP(50); in.N != 50 || in.M != 25 || in.L[3] != in.U[3] {
		t.Errorf("equality QP: n %d m %d", in.N, in.M)
	}
	if in := g.portfolio(5); in.N != 505 || in.M != 1+5+500 {
		t.Errorf("portfolio: n %d m %d", in.N, in.M)
	}
	if in := g.lasso(5); in.N != 2*5+500 || in.M != 500+10 {
		t.Errorf("lasso: n %d m %d", in.N, in.M)
	}
	if in := g.huber(5); in.N != 5+1500 || in.M != 1500 {
		t.Errorf("huber: n %d m %d", in.N, in.M)
	}
	if in := g.svm(5); in.N != 505 || in.M != 1000 {
		t.Errorf("svm: n %d m %d", in.N, in.M)
	}
	if in := g.control(5); in.N != 11*5+10*2 || in.M != 11*5+in.N {
		t.Errorf("control: n %d m %d", in.N, in.M)
	}
	if in := g.control(1); in.N != 11+10 { // nu is at least 1
		t.Errorf("control(1): n %d", in.N)
	}
}

func TestGenerateIsDeterministic(t *testing.T) {
	enc := func(seed uint64) string {
		b, _ := json.Marshal(Generate(seed, true))
		return string(b)
	}
	if enc(3) != enc(3) {
		t.Error("the same seed gives different instances")
	}
	if enc(3) == enc(4) {
		t.Error("different seeds give the same instances")
	}
}

func TestWriteInstances(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "sub")
	set := Generate(1, true)
	if err := writeInstances(dir, set); err != nil {
		t.Fatal(err)
	}
	info, err := readInfo(dir)
	if err != nil || len(info) != len(set) {
		t.Fatalf("%d instances read back, %v", len(info), err)
	}
	if err := os.WriteFile(filepath.Join(dir, "broken.json"), []byte("{"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := readInfo(dir); err == nil {
		t.Error("a broken instance file is accepted")
	}
	if err := writeInstances(filepath.Join(dir, "broken.json", "x"), set); err == nil {
		t.Error("writing below a file succeeded")
	}
	if _, err := readInfo(filepath.Join(dir, "[")); err == nil {
		t.Error("a malformed glob pattern is accepted")
	}
}

func TestPrepareDir(t *testing.T) {
	root := t.TempDir()
	if err := prepareDir(filepath.Join(root, "absent")); err != nil {
		t.Errorf("absent directory: %v", err)
	}
	dir := filepath.Join(root, "inst")
	_ = os.Mkdir(dir, 0o755)
	_ = os.WriteFile(filepath.Join(dir, "a.json"), []byte("{}"), 0o644)
	if err := prepareDir(dir); err != nil {
		t.Fatal(err)
	}
	if es, _ := os.ReadDir(dir); len(es) != 0 {
		t.Errorf("%d files left", len(es))
	}
	// Anything that is not an instance file stops it, and nothing is removed.
	_ = os.WriteFile(filepath.Join(dir, "a.json"), []byte("{}"), 0o644)
	_ = os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("keep"), 0o644)
	if err := prepareDir(dir); err == nil || !strings.Contains(err.Error(), "notes.txt") {
		t.Errorf("a foreign file: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "a.json")); err != nil {
		t.Error("a file was removed although the directory was refused")
	}
	_ = os.Mkdir(filepath.Join(root, "sub"), 0o755)
	_ = os.Mkdir(filepath.Join(root, "sub", "d"), 0o755)
	if err := prepareDir(filepath.Join(root, "sub")); err == nil {
		t.Error("a subdirectory was accepted")
	}
	// Inside a git work tree: refused, naming the tree.
	tree := filepath.Join(root, "repo")
	_ = os.MkdirAll(filepath.Join(tree, ".git"), 0o755)
	if err := prepareDir(filepath.Join(tree, "out")); err == nil || !strings.Contains(err.Error(), tree) {
		t.Errorf("inside a work tree: %v", err)
	}
	// The file is not a directory.
	f := filepath.Join(root, "file")
	_ = os.WriteFile(f, nil, 0o644)
	if err := prepareDir(f); err == nil {
		t.Error("a regular file was accepted as a directory")
	}
}

// --- parsing -----------------------------------------------------------------

func TestParseGoBench(t *testing.T) {
	in := `goos: darwin
BenchmarkQP/random-qp-n20/admm         	      50	  1000000 ns/op	  200000 setup-ns/op	  800000 solve-ns/op	  150000 ldl-factor-ns/op	  100000 ldl-solve-ns/op	  75.00 iters	  -3.5 obj	  0 status
BenchmarkQP/dense-n50/activeset-8      	      50	  1000000 ns/op	  400000 setup-ns/op	  600000 solve-ns/op	  0 ldl-factor-ns/op	  0 ldl-solve-ns/op	  12.00 iters	  1.5 obj	  0 status
BenchmarkQP/odd-name/other-1  1 5 ns/op 1 setup-ns/op
BenchmarkOther/x/admm  1 5 ns/op 1 setup-ns/op
BenchmarkQP/noMetric/admm  1 5 ns/op
BenchmarkQP/short
PASS
`
	got := parseGoBench(strings.NewReader(in))
	if len(got) != 2 {
		t.Fatalf("%d samples: %v", len(got), got)
	}
	a := got["random-qp-n20/admm"]
	near := func(x, y float64) bool { return math.Abs(x-y) < 1e-12 }
	if !near(a.Setup, 200e-6) || !near(a.Solve, 800e-6) || !near(a.LDLFactor, 150e-6) || !near(a.LDLSolve, 100e-6) || a.Iters != 75 || a.Obj != -3.5 {
		t.Errorf("admm sample %+v", a)
	}
	if got["dense-n50/activeset"].Iters != 12 {
		t.Errorf("the -8 suffix was not stripped: %v", got)
	}
}

func TestParseLoadAndMedian(t *testing.T) {
	for in, want := range map[string]float64{"0.52 0.48 0.40 1/200 1234": 0.52, "{ 3.25 2.00 1.00 }": 3.25} {
		if v, err := parseLoad(in); err != nil || v != want {
			t.Errorf("%q: %v %v", in, v, err)
		}
	}
	if _, err := parseLoad("nothing here"); err == nil {
		t.Error("no number accepted")
	}
	if median(nil) != 0 || median([]float64{3, 1, 2}) != 2 || median([]float64{4, 1, 2, 3}) != 2.5 {
		t.Error("median")
	}
	if h := human(2.5); h != "2.5 s" {
		t.Error(h)
	}
	if h := human(2.5e-3); h != "2.5 ms" {
		t.Error(h)
	}
	if h := human(2.5e-6); h != "2.5 µs" {
		t.Error(h)
	}
	if g := geomean([]float64{1, 4}); math.Abs(g-2) > 1e-12 {
		t.Error(g)
	}
}

// --- the report, with a planted slower row and its control ---------------------

func canned(slow float64) (goOut string, ref string) {
	var gb, rb strings.Builder
	rb.WriteString(`{"meta":{"osqp":"1.1.3","quadprog":"0.1.14","quadprog_call_overhead":2e-6},"osqp":{`)
	first := true
	names := []string{"random-qp-n20", "eq-qp-n50", "portfolio-k5", "lasso-n5", "huber-n5", "svm-n5", "control-nx5", "dense-n10", "dense-n50"}
	for _, n := range names {
		fmt.Fprintf(&gb, "BenchmarkQP/%s/admm \t 10 \t 1 ns/op \t %g setup-ns/op \t %g solve-ns/op \t %g ldl-analyse-ns/op \t %g ldl-factor-ns/op \t %g ldl-solve-ns/op \t 50 iters \t 1 obj \t 0 status\n",
			n, 100e3*slow, 400e3*slow, 30e3*slow, 40e3*slow, 160e3*slow)
		if !first {
			rb.WriteString(",")
		}
		first = false
		fmt.Fprintf(&rb, `%q:{"setup":100e-6,"solve":400e-6,"iter":50,"status":"solved","obj":1,"rho_updates":1}`, n)
	}
	rb.WriteString(`},"quadprog":{`)
	for i, n := range []string{"dense-n10", "dense-n50"} {
		fmt.Fprintf(&gb, "BenchmarkQP/%s/activeset \t 10 \t 1 ns/op \t %g setup-ns/op \t %g solve-ns/op \t 0 ldl-factor-ns/op \t 0 ldl-solve-ns/op \t 9 iters \t 1 obj \t 0 status\n",
			n, 100e3*slow, 400e3*slow)
		if i > 0 {
			rb.WriteString(",")
		}
		fmt.Fprintf(&rb, `%q:{"total":500e-6,"obj":1}`, n)
	}
	rb.WriteString("}}")
	return gb.String(), rb.String()
}

type fakeEnv struct {
	loads    []float64 // returned in turn; the last one repeats
	slow     float64
	failWhat string
	runs     []string
}

func (f *fakeEnv) install(t *testing.T) {
	t.Helper()
	oldRun, oldLoad := execRun, loadAvg
	t.Cleanup(func() { execRun, loadAvg = oldRun, oldLoad })
	loadAvg = func() (float64, error) {
		v := f.loads[0]
		if len(f.loads) > 1 {
			f.loads = f.loads[1:]
		}
		return v, nil
	}
	execRun = func(c *exec.Cmd) error {
		kind := "go"
		switch {
		case strings.HasSuffix(c.Path, "opt.test"):
			kind = "bench"
		case filepath.Base(c.Path) == "uv" || (len(c.Args) > 0 && c.Args[0] == "uv"):
			kind = "uv"
		}
		f.runs = append(f.runs, kind)
		if f.failWhat == kind {
			return errors.New("boom")
		}
		g, r := canned(f.slow)
		switch kind {
		case "bench":
			_, _ = io.WriteString(c.Stdout, g)
		case "uv":
			if f.failWhat == "uvjunk" {
				_, _ = io.WriteString(c.Stdout, "not json")
			} else {
				_, _ = io.WriteString(c.Stdout, r)
			}
		}
		return nil
	}
}

func run(args ...string) (string, string, error) {
	cmd := RootCmd()
	var o, e bytes.Buffer
	cmd.SetOut(&o)
	cmd.SetErr(&e)
	cmd.SetArgs(args)
	err := cmd.Execute()
	return o.String(), e.String(), err
}

func quick(t *testing.T, extra ...string) []string {
	return append([]string{"--quick", "--rounds", "3", "--instances", filepath.Join(t.TempDir(), "i")}, extra...)
}

func TestRunReportsAndFlagsASlowerRow(t *testing.T) {
	// Planted positive: ours is 1.6 times the reference everywhere; every row must
	// be flagged. Control: at 0.5 times, nothing is.
	f := &fakeEnv{loads: []float64{1}, slow: 1.6}
	f.install(t)
	out, _, err := run(quick(t)...)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "10 row(s) slower than the reference") || strings.Count(out, "| slower |") != 10 {
		t.Errorf("the slower rows are not flagged:\n%s", out)
	}
	for _, want := range []string{"Medians of 3 interleaved rounds", "OSQP 1.1.3", "go-sparse AMD + symbolic, share of setup", "random-qp-n20", "dense-n50",
		"| 30% | 40% |", "Method choice (QPAuto)", "| dense-n50 | 155 |", "Geometric mean", "- control: 1.60 over 1 instances", "Goldfarb"} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in\n%s", want, out)
		}
	}

	f2 := &fakeEnv{loads: []float64{1}, slow: 0.5}
	f2.install(t)
	out, _, err = run(quick(t)...)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "0 row(s) slower") || strings.Contains(out, "| slower |") {
		t.Errorf("a row is flagged although ours is twice as fast:\n%s", out)
	}
	// 3 rounds: go build once, then bench and uv per round, interleaved.
	if got := strings.Join(f2.runs, ","); got != "go,bench,uv,bench,uv,bench,uv" {
		t.Errorf("order of the runs: %s", got)
	}
}

// The no-slower-than-quadprog rule starts at n = 50: a slower n = 10 row is only a
// number, a slower n = 50 row is a defect.
func TestQuadprogRuleStartsAtN50(t *testing.T) {
	m := &measurement{goS: map[string][]Sample{}, osqp: map[string][]RefOSQP{}, qp: map[string][]RefQuadprog{}}
	m.goS["dense-n10/activeset"] = []Sample{{Setup: 1e-3, Solve: 1e-3}}
	m.goS["dense-n50/activeset"] = []Sample{{Setup: 1e-3, Solve: 1e-3}}
	m.qp["dense-n10"] = []RefQuadprog{{Total: 1e-4}}
	m.qp["dense-n50"] = []RefQuadprog{{Total: 1e-4}}
	m.overhd = []float64{1e-6}
	var b bytes.Buffer
	slower := report(&b, m, []instInfo{{"dense-n10", "dense", 10, 20, 100}, {"dense-n50", "dense", 50, 100, 100}}, 1, 1, 1)
	if slower != 1 || strings.Count(b.String(), "| slower |") != 1 {
		t.Errorf("slower = %d:\n%s", slower, b.String())
	}
}

func TestRunDiscardsARoundOnALoadedHost(t *testing.T) {
	// load before: 1; after attempt 1: 12 (discarded); then 1 for the rest.
	f := &fakeEnv{loads: []float64{1, 12, 1}, slow: 1}
	f.install(t)
	out, errOut, err := run(quick(t)...)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(errOut, "round attempt 1 discarded") || !strings.Contains(out, "1 discarded round(s)") {
		t.Errorf("the loaded round was not reported:\n%s\n%s", errOut, out)
	}
	if n := strings.Count(strings.Join(f.runs, ","), "bench"); n != 4 {
		t.Errorf("%d bench runs for 3 accepted rounds plus 1 discarded", n)
	}

	// A host that never calms down: the budget runs out and nothing is reported.
	g := &fakeEnv{loads: []float64{1, 12}, slow: 1}
	g.install(t)
	if out, _, err := run(quick(t)...); err == nil || !strings.Contains(err.Error(), "abandoned") || out != "" {
		t.Errorf("a permanently loaded host: %v, output %q", err, out)
	}
}

func TestRunRefusesALoadedHostAtTheStart(t *testing.T) {
	f := &fakeEnv{loads: []float64{20}, slow: 1}
	f.install(t)
	if _, _, err := run(quick(t)...); err == nil || !strings.Contains(err.Error(), "deferred") {
		t.Errorf("%v", err)
	}
	if len(f.runs) != 0 {
		t.Errorf("ran %v on a loaded host", f.runs)
	}

	// --wait polls until the load drops.
	f = &fakeEnv{loads: []float64{20, 20, 1}, slow: 1}
	f.install(t)
	t0 := time.Now()
	oldNow, oldSleep := now, sleep
	defer func() { now, sleep = oldNow, oldSleep }()
	slept := 0
	now = func() time.Time { return t0.Add(time.Duration(slept) * time.Minute) }
	sleep = func(time.Duration) { slept++ }
	if _, _, err := run(quick(t, "--wait", "1h")...); err != nil {
		t.Fatal(err)
	}
	if slept != 2 {
		t.Errorf("slept %d times", slept)
	}
}

func TestRunErrors(t *testing.T) {
	for _, what := range []string{"go", "bench", "uv", "uvjunk"} {
		f := &fakeEnv{loads: []float64{1}, slow: 1, failWhat: what}
		f.install(t)
		if _, _, err := run(quick(t)...); err == nil {
			t.Errorf("%s failing is not reported", what)
		}
	}
	// A load average that cannot be read, at the start and after a round.
	old, oldSleep := loadAvg, sleep
	defer func() { loadAvg, sleep = old, oldSleep }()
	sleep = func(time.Duration) {}
	loadAvg = func() (float64, error) { return 0, errors.New("no sysctl") }
	if _, _, err := run(quick(t)...); err == nil {
		t.Error("an unreadable load average is not reported")
	}
	n := 0
	f := &fakeEnv{loads: []float64{1}, slow: 1}
	f.install(t)
	loadAvg = func() (float64, error) {
		n++
		if n > 1 {
			return 0, errors.New("gone")
		}
		return 1, nil
	}
	if _, _, err := run(quick(t)...); err == nil {
		t.Error("a load average that disappears mid-run is not reported")
	}
	n = 0
	loadAvg = func() (float64, error) {
		n++
		if n == 1 {
			return 30, nil
		}
		return 0, errors.New("gone")
	}
	if _, _, err := run(quick(t, "--wait", "1h")...); err == nil {
		t.Error("an unreadable load average while waiting is not reported")
	}
	// The instance directory is refused inside a work tree and when foreign files are in it.
	f = &fakeEnv{loads: []float64{1}, slow: 1}
	f.install(t)
	dir := t.TempDir()
	_ = os.WriteFile(filepath.Join(dir, "keep.txt"), nil, 0o644)
	if _, _, err := run("--quick", "--instances", dir); err == nil {
		t.Error("a directory with foreign files is cleared")
	}
	if _, _, err := run("--quick", "--instances", filepath.Join(dir, "keep.txt", "x")); err == nil {
		t.Error("an unusable directory is accepted")
	}
	// The default directory is under the user cache directory.
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	t.Setenv("HOME", t.TempDir())
	if _, _, err := run("--quick", "--rounds", "1"); err != nil {
		t.Errorf("default instance directory: %v", err)
	}
}

func TestMainExits(t *testing.T) {
	old, oldArgs := osExit, os.Args
	defer func() { osExit, os.Args = old, oldArgs }()
	code := 0
	osExit = func(c int) { code = c }
	os.Args = []string{"qpbench", "--rounds", "nope"}
	main()
	if code != 1 {
		t.Errorf("exit code %d", code)
	}
	os.Args = []string{"qpbench", "--help"}
	code = 0
	main()
	if code != 0 {
		t.Errorf("--help exit code %d", code)
	}
}

func TestDefaultExecRunAndLoadAvg(t *testing.T) {
	if err := execRun(exec.Command("go", "version")); err != nil {
		t.Skip("go not available")
	}
	if v, err := loadAvg(); err != nil || v < 0 {
		t.Errorf("load average %v %v", v, err)
	}
}

func FuzzParseGoBench(f *testing.F) {
	f.Add("BenchmarkQP/a-b/admm-4 1 2 ns/op 3 setup-ns/op 4 iters")
	f.Add("BenchmarkQP//admm 1 2 ns/op 3 setup-ns/op")
	f.Add(strings.Repeat("BenchmarkQP/x/admm ", 100000))
	f.Fuzz(func(t *testing.T, s string) {
		for name, smp := range parseGoBench(strings.NewReader(s)) {
			if !strings.Contains(name, "/") || smp.Setup == 0 {
				t.Fatalf("bad sample %q %+v", name, smp)
			}
		}
	})
}
