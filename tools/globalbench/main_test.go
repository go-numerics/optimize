package main

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

// The tests need no Python, no Docker, no quiet machine: the external
// commands and the load average are replaced.

const goBenchOutput = `goos: darwin
goarch: arm64
pkg: github.com/go-numerics/optimize
BenchmarkGlobal/de.sphere.2-1         	     100	   1000000 ns/op	      2823 nfev	        93.00 nit
BenchmarkGlobal/de.rastrigin.2-1      	     100	    500000 ns/op	      1953 nfev	        64.00 nit
BenchmarkGlobal/direct.styblinskitang.2-1	     100	   1100000 ns/op	      2011 nfev	       141.0 nit
BenchmarkGlobal/odd         	     garbage
PASS
`

func TestParseGoBench(t *testing.T) {
	got := parseGoBench(strings.NewReader(goBenchOutput))
	if len(got) != 3 {
		t.Fatalf("%d rows: %v", len(got), got)
	}
	s := got["de.sphere.2"]
	if s.Time != 1e-3 || s.Nfev != 2823 || s.Nit != 93 || s.Seq != 0 {
		t.Errorf("de.sphere.2: %+v", s)
	}
	if got["direct.styblinskitang.2"].Seq != 2 {
		t.Errorf("order not kept: %+v", got["direct.styblinskitang.2"])
	}
}

func TestParseLoadAndMedianAndHuman(t *testing.T) {
	for in, want := range map[string]float64{"{ 0.52 0.48 0.40 }": 0.52, "1.5 2 3 1/200 99": 1.5, "42": 42} {
		if v, err := parseLoad(in); err != nil || v != want {
			t.Errorf("parseLoad(%q) = %v, %v", in, v, err)
		}
	}
	if _, err := parseLoad("none here"); err == nil {
		t.Error("no number accepted")
	}
	if median(nil) != 0 || median([]float64{3, 1, 2}) != 2 || median([]float64{4, 1, 2, 3}) != 2.5 {
		t.Error("median")
	}
	for in, want := range map[float64]string{2: "2 s", 0.0025: "2.5 ms", 3e-6: "3 µs"} {
		if got := human(in); got != want {
			t.Errorf("human(%v) = %q, want %q", in, got, want)
		}
	}
}

// canned returns the Go benchmark output and the reference JSON of two rows
// where ours takes `ratio` times SciPy's per call; the rastrigin row has
// different nfev on the two sides (the paths diverged), so it is judged per
// evaluation: ours is `ratio` times per evaluation as well, with 1.25 times
// the evaluations.
func canned(ratio float64) (string, string) {
	g := fmt.Sprintf("BenchmarkGlobal/de.sphere.2-1 10 %g ns/op 1000 nfev 10 nit\nBenchmarkGlobal/de.rastrigin.2-1 10 %g ns/op 1250 nfev 10 nit\n",
		ratio*2e6, ratio*1.25*1e6)
	r := `{"meta":{"scipy":"1.18.1","numpy":"2.5.4"},"rows":{"de.sphere.2":{"time":0.002,"nfev":1000,"nit":10,"reps":3},"de.rastrigin.2":{"time":0.001,"nfev":1000,"nit":10,"reps":3}}}`
	return g, r
}

type fakeEnv struct {
	loads    []float64 // returned in turn; the last one repeats
	ratio    float64
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
		case len(c.Args) > 0 && c.Args[0] == "uv":
			kind = "uv"
		}
		f.runs = append(f.runs, kind)
		if f.failWhat == kind {
			return errors.New("boom")
		}
		g, r := canned(f.ratio)
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

func TestRunReportsAndFlagsASlowerRow(t *testing.T) {
	// Planted positive: ours is 1.6 times SciPy's everywhere; both rows are
	// flagged (the first by total time, the second by time per evaluation).
	f := &fakeEnv{loads: []float64{1}, ratio: 1.6}
	f.install(t)
	out, _, err := run("--rounds", "3")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "2 row(s) slower than SciPy") || strings.Count(out, "| slower |") != 2 {
		t.Errorf("the slower rows are not flagged:\n%s", out)
	}
	for _, want := range []string{"Medians of 3 interleaved rounds", "SciPy 1.18.1, NumPy 2.5.4", "| de.sphere.2 | 1000 / 1000 |", "| de.rastrigin.2 | 1250 / 1000 |"} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in\n%s", want, out)
		}
	}
	// Control: at 0.5 times nothing is flagged. With 1.25 times the evaluations
	// at 0.5 times the time per evaluation, the rastrigin row is faster per
	// evaluation but slower in total: the verdict is the per-evaluation one.
	f2 := &fakeEnv{loads: []float64{1}, ratio: 0.5}
	f2.install(t)
	out, _, err = run("--rounds", "3")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "0 row(s) slower") || strings.Contains(out, "| slower |") {
		t.Errorf("a row is flagged although ours is twice as fast:\n%s", out)
	}
	if got := strings.Join(f2.runs, ","); got != "go,bench,uv,bench,uv,bench,uv" {
		t.Errorf("order of the runs: %s", got)
	}
}

func TestRowOrderKeepsTheBenchmarkOrderAndSkipsUnpairedRows(t *testing.T) {
	m := &measurement{goS: map[string][]Sample{}, ref: map[string][]RefRow{}}
	m.goS["b"] = []Sample{{Seq: 0}}
	m.goS["a"] = []Sample{{Seq: 1}}
	m.goS["onlygo"] = []Sample{{Seq: 2}}
	m.ref["a"] = []RefRow{{}}
	m.ref["b"] = []RefRow{{}}
	m.ref["onlyref"] = []RefRow{{}}
	if got := strings.Join(rowOrder(m), ","); got != "b,a" {
		t.Errorf("rowOrder = %s", got)
	}
}

func TestRunDiscardsARoundOnALoadedHost(t *testing.T) {
	// load before: 1; after attempt 1: 12 (discarded); then 1 for the rest.
	f := &fakeEnv{loads: []float64{1, 12, 1}, ratio: 1}
	f.install(t)
	out, errOut, err := run("--rounds", "3")
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
	g := &fakeEnv{loads: []float64{1, 12}, ratio: 1}
	g.install(t)
	if out, _, err := run("--rounds", "3"); err == nil || !strings.Contains(err.Error(), "abandoned") || out != "" {
		t.Errorf("a permanently loaded host: %v, output %q", err, out)
	}
}

func TestRunRefusesALoadedHostAtTheStart(t *testing.T) {
	f := &fakeEnv{loads: []float64{20}, ratio: 1}
	f.install(t)
	if _, _, err := run("--rounds", "3"); err == nil || !strings.Contains(err.Error(), "deferred") {
		t.Errorf("%v", err)
	}
	if len(f.runs) != 0 {
		t.Errorf("ran %v on a loaded host", f.runs)
	}
	// --wait polls until the load drops.
	f = &fakeEnv{loads: []float64{20, 20, 1}, ratio: 1}
	f.install(t)
	t0 := time.Now()
	oldNow, oldSleep := now, sleep
	defer func() { now, sleep = oldNow, oldSleep }()
	slept := 0
	now = func() time.Time { return t0.Add(time.Duration(slept) * time.Minute) }
	sleep = func(time.Duration) { slept++ }
	if _, _, err := run("--rounds", "1", "--wait", "1h"); err != nil {
		t.Fatal(err)
	}
	if slept != 2 {
		t.Errorf("slept %d times", slept)
	}
}

func TestRunErrors(t *testing.T) {
	for _, what := range []string{"go", "bench", "uv", "uvjunk"} {
		f := &fakeEnv{loads: []float64{1}, ratio: 1, failWhat: what}
		f.install(t)
		if _, _, err := run("--rounds", "1"); err == nil {
			t.Errorf("%s failing is not reported", what)
		}
	}
	// A load average that cannot be read, at the start, after a round and while waiting.
	old, oldSleep := loadAvg, sleep
	defer func() { loadAvg, sleep = old, oldSleep }()
	sleep = func(time.Duration) {}
	loadAvg = func() (float64, error) { return 0, errors.New("no sysctl") }
	if _, _, err := run("--rounds", "1"); err == nil {
		t.Error("an unreadable load average is not reported")
	}
	n := 0
	f := &fakeEnv{loads: []float64{1}, ratio: 1}
	f.install(t)
	loadAvg = func() (float64, error) {
		n++
		if n > 1 {
			return 0, errors.New("gone")
		}
		return 1, nil
	}
	if _, _, err := run("--rounds", "1"); err == nil {
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
	if _, _, err := run("--rounds", "1", "--wait", "1h"); err == nil {
		t.Error("an unreadable load average while waiting is not reported")
	}
}

func TestMainExits(t *testing.T) {
	old, oldArgs := osExit, os.Args
	defer func() { osExit, os.Args = old, oldArgs }()
	code := 0
	osExit = func(c int) { code = c }
	os.Args = []string{"globalbench", "--rounds", "nope"}
	main()
	if code != 1 {
		t.Errorf("exit code %d", code)
	}
	os.Args = []string{"globalbench", "--help"}
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

// The embedded reference script defines every row that BenchmarkGlobal does.
func TestScriptHasTheRows(t *testing.T) {
	for _, row := range []string{`"de.%s.%d"`, `"dev.%s.10"`, "deoverhead.trivial.10", "bh.himmelblau.100", "direct.styblinskitang.2",
		"direct.rosenbrock.2", "direct.rosenbrock.5", "direct.sphere.10"} {
		if !strings.Contains(string(script), strings.Trim(row, `"`)) && !strings.Contains(string(script), row) {
			t.Errorf("bench.py lacks row %s", row)
		}
	}
}

func FuzzParseGoBench(f *testing.F) {
	f.Add("BenchmarkGlobal/de.sphere.2-1 1 2 ns/op 3 nfev")
	f.Add("BenchmarkGlobal// 1 2 ns/op")
	f.Add(strings.Repeat("BenchmarkGlobal/x ", 100000))
	f.Fuzz(func(t *testing.T, s string) {
		for name, smp := range parseGoBench(strings.NewReader(s)) {
			if name == "" && smp.Time < 0 {
				t.Fatalf("bad sample %q %+v", name, smp)
			}
		}
	})
}
