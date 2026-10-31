package main

import (
	"bytes"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

const committed = "../../testdata/qp/planted.json"

func load(t *testing.T) *Fixture {
	t.Helper()
	f, err := readFixture(committed)
	if err != nil {
		t.Fatal(err)
	}
	return f
}

// clone returns a deep copy of f (through its JSON form).
func clone(t *testing.T, f *Fixture) *Fixture {
	t.Helper()
	var buf bytes.Buffer
	if err := Encode(&buf, f); err != nil {
		t.Fatal(err)
	}
	g, err := Decode(&buf)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func find(t *testing.T, f *Fixture, name string) *Case {
	t.Helper()
	for i := range f.Cases {
		if f.Cases[i].Name == name {
			return &f.Cases[i]
		}
	}
	t.Fatalf("no case %q", name)
	return nil
}

func maxAbsDiff(a, b []float64) float64 {
	d := 0.0
	for i := range a {
		d = math.Max(d, math.Abs(a[i]-b[i]))
	}
	return d
}

// The committed fixture is what tools/qporacle gen produced, in the pinned
// container: the OSQP wheel bundles the v1.0.0 core, and the versions match
// the pins of main.go.
func TestCommittedFixtureEnvironment(t *testing.T) {
	f := load(t)
	if err := CheckCore(f); err != nil {
		t.Fatal(err)
	}
	for pin, got := range map[string]string{osqpPin: "osqp==" + f.Osqp, clarabelPin: "clarabel==" + f.Clarabel,
		quadprogPin: "quadprog==" + f.Quadprog, numpyPin: "numpy==" + f.Numpy, scipyPin: "scipy==" + f.Scipy} {
		if pin != got {
			t.Errorf("pin %s, fixture ran %s", pin, got)
		}
	}
	if f.Machine != "x86_64" || len(f.Cases) != 39 {
		t.Errorf("machine %q, %d cases", f.Machine, len(f.Cases))
	}
}

// Known optima, worked out by hand (docs/design-qp.md 6.4): the tight OSQP
// run, Clarabel and quadprog find them; OSQP's default 1e-3 settings come
// within 1e-2.
func TestPlantedKnownOptima(t *testing.T) {
	f := load(t)
	want := map[string][]float64{
		"unconstrained":         {-1.0 / 7, -3.0 / 7},
		"one-bound":             {0.5, 1},
		"one-equality":          {0.5, 0.5},
		"vertex":                {1, 1},
		"degenerate-vertex":     {1, 1},
		"two-sided":             {2, 0},
		"duplicated-rows":       {0.5, 0},
		"feasible-twin":         {1},
		"consistent-equalities": {0.5, 0.5},
	}
	for name, x := range want {
		c := find(t, f, name)
		for solver, tol := range map[string]float64{"osqp_tight": 1e-6, "clarabel": 1e-6, "quadprog": 1e-6, "osqp": 1e-2} {
			r, ok := c.Results[solver]
			if !ok {
				if solver == "quadprog" && name != "unconstrained" {
					t.Errorf("%s: no quadprog result", name)
				}
				continue
			}
			if r.X == nil || maxAbsDiff(r.X, x) > tol {
				t.Errorf("%s [%s]: x = %v, want %v (%v)", name, solver, r.X, x, r.Error)
			}
		}
	}
	// The LP has a whole optimal edge; its objective is 1.
	lp := find(t, f, "lp")
	for _, s := range []string{"osqp_tight", "clarabel"} {
		if o := lp.Results[s].Obj; o == nil || math.Abs(*o-1) > 1e-6 {
			t.Errorf("lp [%s]: objective %v", s, o)
		}
	}
	if _, ok := lp.Results["quadprog"]; ok {
		t.Error("quadprog ran on P = 0")
	}
}

// Infeasible, unbounded and non-convex problems report so, with the feasible
// twin as the control: the detectors do not fire on it.
func TestPlantedStatuses(t *testing.T) {
	f := load(t)
	for _, name := range []string{"infeasible-bounds", "infeasible-equalities"} {
		c := find(t, f, name)
		if c.Results["clarabel"].Status != "PrimalInfeasible" || c.Results["osqp"].Status != "primal infeasible" ||
			c.Results["osqp_tight"].Status != "primal infeasible" || !strings.Contains(c.Results["quadprog"].Error, "inconsistent") {
			t.Errorf("%s: %+v", name, c.Results)
		}
	}
	u := find(t, f, "unbounded")
	if u.Results["clarabel"].Status != "DualInfeasible" || u.Results["osqp"].Status != "dual infeasible" {
		t.Errorf("unbounded: %+v", u.Results)
	}
	// OSQP v1.0.0 refuses a non-convex P at setup (its LDL factorisation fails),
	// it does not return a status.
	nc := find(t, f, "non-convex")
	if !strings.Contains(nc.Results["osqp"].Error, "OSQPException") || nc.Results["osqp"].Status != "" {
		t.Errorf("non-convex: %+v", nc.Results["osqp"])
	}
	twin := find(t, f, "feasible-twin")
	for s, r := range twin.Results {
		if r.Error != "" || !strings.EqualFold(r.Status, "solved") {
			t.Errorf("feasible twin [%s]: %+v", s, r)
		}
	}
}

// The solvers agree with each other on every problem they all solve.
func TestPlantedSolversAgree(t *testing.T) {
	f := load(t)
	agreed := 0
	for _, c := range f.Cases {
		a, b := c.Results["osqp_tight"], c.Results["clarabel"]
		if !strings.EqualFold(a.Status, "solved") || !strings.EqualFold(b.Status, "solved") {
			continue
		}
		agreed++
		if d := maxAbsDiff(a.X, b.X); d > 1e-5 || math.Abs(*a.Obj-*b.Obj) > 1e-7*math.Max(1, math.Abs(*b.Obj)) {
			t.Errorf("%s: osqp_tight and clarabel differ by %g (%g, %g)", c.Name, d, *a.Obj, *b.Obj)
		}
		if q, ok := c.Results["quadprog"]; ok && q.Error == "" && maxAbsDiff(q.X, b.X) > 1e-6 {
			t.Errorf("%s: quadprog %v, clarabel %v", c.Name, q.X, b.X)
		}
	}
	if agreed < 10 {
		t.Errorf("only %d problems solved by both", agreed)
	}
}

// CheckCore is the guard that a wheel with another C core cannot slip past:
// it fires on 0.6.3, on an empty report, and passes on the real one.
func TestCheckCoreFires(t *testing.T) {
	f := load(t)
	for _, core := range []string{"0.6.3", "", "1.0.0-rc1", "1.1.0"} {
		g := clone(t, f)
		g.OsqpCore = core
		if err := CheckCore(g); !errors.Is(err, ErrCore) {
			t.Errorf("core %q accepted: %v", core, err)
		}
	}
	if err := CheckCore(f); err != nil {
		t.Errorf("v1.0.0 refused: %v", err)
	}
}

func TestDiffPlantedPositives(t *testing.T) {
	a := load(t)
	if d := Diff(a, clone(t, a), 1e-9); len(d) != 0 {
		t.Fatalf("identical fixtures disagree: %v", d)
	}
	one := 1
	mods := map[string]func(*Fixture){
		"x differs":         func(f *Fixture) { find(t, f, "vertex").Results["clarabel"].X[0] += 1e-3 },
		"y differs":         func(f *Fixture) { find(t, f, "vertex").Results["osqp"].Y[0] += 1e-3 },
		"objective differs": func(f *Fixture) { *find(t, f, "vertex").Results["clarabel"].Obj += 1e-3 },
		"iteration count":   func(f *Fixture) { *find(t, f, "vertex").Results["osqp"].Iter += 25 },
		"status": func(f *Fixture) {
			r := find(t, f, "vertex").Results["osqp"]
			r.Status = "unsolved"
			find(t, f, "vertex").Results["osqp"] = r
		},
		"error": func(f *Fixture) {
			r := find(t, f, "vertex").Results["osqp"]
			r.Error = "boom"
			find(t, f, "vertex").Results["osqp"] = r
		},
		"x nil against not nil": func(f *Fixture) {
			r := find(t, f, "vertex").Results["osqp"]
			r.X = nil
			find(t, f, "vertex").Results["osqp"] = r
		},
		"obj nil": func(f *Fixture) {
			r := find(t, f, "vertex").Results["osqp"]
			r.Obj = nil
			find(t, f, "vertex").Results["osqp"] = r
		},
		"iter nil": func(f *Fixture) {
			r := find(t, f, "vertex").Results["osqp"]
			r.Iter = nil
			find(t, f, "vertex").Results["osqp"] = r
		},
		"solver only on one side": func(f *Fixture) {
			delete(find(t, f, "vertex").Results, "quadprog")
		},
		"solver only on the other side": func(f *Fixture) {
			find(t, f, "vertex").Results["newsolver"] = Result{Status: "x", Iter: &one}
		},
		"problem data (q)":    func(f *Fixture) { find(t, f, "vertex").Q[0] += 1 },
		"problem data (p)":    func(f *Fixture) { find(t, f, "vertex").P[0][0] += 1 },
		"problem data (a)":    func(f *Fixture) { find(t, f, "vertex").A[0][0] += 1 },
		"problem data (rows)": func(f *Fixture) { c := find(t, f, "vertex"); c.A, c.L, c.U = c.A[:1], c.L[:1], c.U[:1] },
		"case removed":        func(f *Fixture) { f.Cases = f.Cases[1:] },
		"case added":          func(f *Fixture) { c := clone(t, f).Cases[0]; c.Name = "extra"; f.Cases = append(f.Cases, c) },
	}
	for name, mod := range mods {
		b := clone(t, a)
		mod(b)
		if d := Diff(a, b, 1e-9); len(d) == 0 {
			t.Errorf("%s: not reported", name)
		}
	}
	// A difference below the tolerance is not reported, one above is.
	b := clone(t, a)
	find(t, b, "vertex").Results["clarabel"].X[0] += 1e-12
	if d := Diff(a, b, 1e-9); len(d) != 0 {
		t.Errorf("1e-12 reported at tolerance 1e-9: %v", d)
	}
	if d := Diff(a, b, 1e-15); len(d) != 1 {
		t.Errorf("1e-12 not reported at tolerance 1e-15: %v", d)
	}
	// String forms.
	if s := (Disagreement{"c", "", "w"}).String(); s != "c: w" {
		t.Errorf("%q", s)
	}
	if s := (Disagreement{"c", "s", "w"}).String(); s != "c [s]: w" {
		t.Errorf("%q", s)
	}
}

func TestDecodeRejects(t *testing.T) {
	good := `{"cases":[{"name":"a","n":1,"p":[[1]],"q":[0],"a":[],"l":[],"u":[],"results":{}}]}`
	if _, err := Decode(strings.NewReader(good)); err != nil {
		t.Fatalf("minimal fixture: %v", err)
	}
	for name, in := range map[string]string{
		"empty":          ``,
		"truncated":      `{"cases":[`,
		"unknown field":  `{"bogus":1,"cases":[]}`,
		"no cases":       `{"cases":[]}`,
		"trailing data":  good + ` {}`,
		"not an object":  `[]`,
		"empty name":     strings.Replace(good, `"name":"a"`, `"name":""`, 1),
		"n zero":         strings.Replace(good, `"n":1`, `"n":0`, 1),
		"n huge":         strings.Replace(good, `"n":1`, `"n":999999999`, 1),
		"q length":       strings.Replace(good, `"q":[0]`, `"q":[0,0]`, 1),
		"p rows":         strings.Replace(good, `"p":[[1]]`, `"p":[[1],[1]]`, 1),
		"p row length":   strings.Replace(good, `"p":[[1]]`, `"p":[[1,2]]`, 1),
		"l length":       strings.Replace(good, `"l":[]`, `"l":[0]`, 1),
		"a row length":   strings.Replace(strings.Replace(good, `"a":[]`, `"a":[[1,2]]`, 1), `"l":[],"u":[]`, `"l":[0],"u":[1]`, 1),
		"l > u":          strings.Replace(strings.Replace(good, `"a":[]`, `"a":[[1]]`, 1), `"l":[],"u":[]`, `"l":[2],"u":[1]`, 1),
		"l = +inf":       strings.Replace(strings.Replace(good, `"a":[]`, `"a":[[1]]`, 1), `"l":[],"u":[]`, `"l":[1e30],"u":[1e30]`, 1),
		"u = -inf":       strings.Replace(strings.Replace(good, `"a":[]`, `"a":[[1]]`, 1), `"l":[],"u":[]`, `"l":[-1e30],"u":[-1e30]`, 1),
		"x length":       strings.Replace(good, `"results":{}`, `"results":{"s":{"x":[1,2]}}`, 1),
		"y huge":         strings.Replace(good, `"results":{}`, `"results":{"s":{"y":[`+strings.Repeat("0,", 15001)+`0]}}`, 1),
		"unnamed solver": strings.Replace(good, `"results":{}`, `"results":{"":{}}`, 1),
		"NaN":            strings.Replace(good, `"q":[0]`, `"q":[NaN]`, 1),
		"wrong type":     strings.Replace(good, `"n":1`, `"n":"1"`, 1),
	} {
		if _, err := Decode(strings.NewReader(in)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	dup := strings.Replace(good, `]}`, `,{"name":"a","n":1,"p":[[1]],"q":[0],"a":[],"l":[],"u":[],"results":{}}]}`, 1)
	if _, err := Decode(strings.NewReader(dup)); err == nil {
		t.Error("duplicate case name accepted")
	}
	tooMany := `{"cases":[` + strings.Repeat(`{"name":"a","n":1,"p":[[1]],"q":[0],"a":[],"l":[],"u":[],"results":{}},`, maxCases+1) + `{}]}`
	if _, err := Decode(strings.NewReader(tooMany)); err == nil {
		t.Error("more than maxCases accepted")
	}
	// More than maxFixtureBytes: refused without reading it all into memory.
	big := io.MultiReader(strings.NewReader(`{"cases":[{"name":"`), &zeros{n: maxFixtureBytes + 10})
	if _, err := Decode(big); err == nil {
		t.Error("oversized fixture accepted")
	}
	// A valid fixture that is one byte over the limit (trailing blanks).
	padded := io.MultiReader(strings.NewReader(good), &blanks{n: maxFixtureBytes})
	if _, err := Decode(padded); err == nil {
		t.Error("fixture over the byte limit accepted")
	}
	// Encode refuses what JSON cannot hold.
	nan := load(t)
	nan.Cases[0].Q[0] = math.NaN()
	if err := Encode(io.Discard, nan); err == nil {
		t.Error("NaN encoded")
	}
	// Deeply nested arrays: a typed decode refuses them at the first level.
	deep := `{"cases":[{"name":"a","n":1,"p":` + strings.Repeat("[", 100000) + strings.Repeat("]", 100000) + `}]}`
	if _, err := Decode(strings.NewReader(deep)); err == nil {
		t.Error("deeply nested p accepted")
	}
}

// zeros yields n bytes of 'a'.
type zeros struct{ n int64 }

func (z *zeros) Read(p []byte) (int, error) {
	if z.n <= 0 {
		return 0, io.EOF
	}
	k := int64(len(p))
	if k > z.n {
		k = z.n
	}
	for i := int64(0); i < k; i++ {
		p[i] = 'a'
	}
	z.n -= k
	return int(k), nil
}

// blanks yields n spaces.
type blanks struct{ n int64 }

func (b *blanks) Read(p []byte) (int, error) {
	if b.n <= 0 {
		return 0, io.EOF
	}
	k := min(int64(len(p)), b.n)
	for i := int64(0); i < k; i++ {
		p[i] = ' '
	}
	b.n -= k
	return int(k), nil
}

// The default runner runs a real command with the three streams wired.
func TestDefaultRunner(t *testing.T) {
	var out bytes.Buffer
	if err := runner([]string{"go", "version"}, nil, &out, io.Discard); err != nil || !strings.HasPrefix(out.String(), "go version") {
		t.Errorf("%v %q", err, out.String())
	}
	if err := runner([]string{"go", "no-such-subcommand"}, nil, io.Discard, io.Discard); err == nil {
		t.Error("failing command reported success")
	}
}

func run(t *testing.T, args ...string) (string, error) {
	t.Helper()
	cmd := RootCmd()
	var out, errb bytes.Buffer
	cmd.SetArgs(args)
	cmd.SetOut(&out)
	cmd.SetErr(&errb)
	err := cmd.Execute()
	return out.String(), err
}

func TestCommandLine(t *testing.T) {
	if argv := command(""); argv[0] != "uv" || !slices.Contains(argv, osqpPin) || !slices.Contains(argv, "-I") {
		t.Errorf("host command %v", argv)
	}
	if argv := command("linux/amd64"); argv[0] != "docker" || !slices.Contains(argv, "linux/amd64") || !slices.Contains(argv, quadprogPin) {
		t.Errorf("container command %v", argv)
	}
}

func TestGenCommand(t *testing.T) {
	raw, err := os.ReadFile(committed)
	if err != nil {
		t.Fatal(err)
	}
	old := runner
	defer func() { runner = old }()
	var gotArgv []string
	runner = func(argv []string, stdin io.Reader, stdout, _ io.Writer) error {
		gotArgv = argv
		in, _ := io.ReadAll(stdin)
		if !bytes.Contains(in, []byte("osqp_version")) {
			t.Error("the embedded script (with the core check) was not fed on stdin")
		}
		_, err := stdout.Write(raw)
		return err
	}
	dst := filepath.Join(t.TempDir(), "out.json")
	out, err := run(t, "gen", "--platform", "linux/amd64", "-o", dst)
	if err != nil || !strings.Contains(out, "wrote 39 cases") || !strings.Contains(out, "core 1.0.0") || gotArgv[0] != "docker" {
		t.Fatalf("%v %q %v", err, out, gotArgv)
	}
	g, err := readFixture(dst)
	if err != nil || len(Diff(load(t), g, 0)) != 0 {
		t.Fatalf("written fixture differs: %v", err)
	}

	// A wheel with another core: the harness fails and writes nothing.
	bad := clone(t, load(t))
	bad.OsqpCore = "0.6.3"
	var badRaw bytes.Buffer
	if err := Encode(&badRaw, bad); err != nil {
		t.Fatal(err)
	}
	runner = func(_ []string, _ io.Reader, w io.Writer, _ io.Writer) error {
		_, err := w.Write(badRaw.Bytes())
		return err
	}
	dst2 := filepath.Join(t.TempDir(), "bad.json")
	if _, err := run(t, "gen", "-o", dst2); !errors.Is(err, ErrCore) {
		t.Errorf("core 0.6.3: %v", err)
	}
	if _, err := os.Stat(dst2); err == nil {
		t.Error("a fixture was written for the wrong core")
	}

	runner = func([]string, io.Reader, io.Writer, io.Writer) error { return errors.New("boom") }
	if _, err := run(t, "gen", "-o", dst2); err == nil || !strings.Contains(err.Error(), "boom") {
		t.Errorf("runner failure: %v", err)
	}
	runner = func(_ []string, _ io.Reader, w io.Writer, _ io.Writer) error {
		_, err := w.Write([]byte("{"))
		return err
	}
	if _, err := run(t, "gen", "-o", dst2); err == nil {
		t.Error("garbage output accepted")
	}
	runner = func(_ []string, _ io.Reader, w io.Writer, _ io.Writer) error { _, err := w.Write(raw); return err }
	if _, err := run(t, "gen", "-o", filepath.Join(t.TempDir(), "no", "dir.json")); err == nil {
		t.Error("unwritable output accepted")
	}
	if _, err := run(t, "gen", "extra"); err == nil {
		t.Error("extra argument accepted")
	}
}

func TestDiffCommand(t *testing.T) {
	f := load(t)
	dir := t.TempDir()
	write := func(name string, g *Fixture) string {
		var buf bytes.Buffer
		if err := Encode(&buf, g); err != nil {
			t.Fatal(err)
		}
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, buf.Bytes(), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	same := write("a.json", f)
	out, err := run(t, "diff", same, committed)
	if err != nil || !strings.Contains(out, "0 disagreements") {
		t.Fatalf("%v %q", err, out)
	}
	g := clone(t, f)
	find(t, g, "vertex").Results["clarabel"].X[1] += 1e-3
	out, err = run(t, "diff", same, write("b.json", g))
	if !errors.Is(err, ErrDisagree) || !strings.Contains(out, "vertex [clarabel]: x differs") || !strings.Contains(out, "1 disagreements") {
		t.Fatalf("%v %q", err, out)
	}
	out, err = run(t, "diff", "--tol", "1", same, write("b.json", g))
	if err != nil {
		t.Fatalf("tolerance 1: %v %q", err, out)
	}
	if _, err := run(t, "diff", same, filepath.Join(dir, "missing.json")); err == nil {
		t.Error("missing second file accepted")
	}
	if _, err := run(t, "diff", filepath.Join(dir, "missing.json"), same); err == nil {
		t.Error("missing first file accepted")
	}
	bad := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(bad, []byte("{"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := run(t, "diff", same, bad); err == nil {
		t.Error("corrupt file accepted")
	}
	if _, err := run(t, "diff", same); err == nil {
		t.Error("one argument accepted")
	}
}

func TestMainExitsOnError(t *testing.T) {
	var code int
	oldExit, oldArgs := osExit, os.Args
	defer func() { osExit, os.Args = oldExit, oldArgs }()
	osExit = func(c int) { code = c }
	os.Args = []string{"qporacle", "diff", "only-one"}
	main()
	if code != 1 {
		t.Errorf("exit code %d", code)
	}
	os.Args = []string{"qporacle", "--help"}
	code = 0
	main()
	if code != 0 {
		t.Errorf("--help exit code %d", code)
	}
}

func FuzzReadFixture(f *testing.F) {
	raw, err := os.ReadFile(committed)
	if err != nil {
		f.Fatal(err)
	}
	f.Add(raw)
	f.Add([]byte(`{"cases":[{"name":"a","n":1,"p":[[1]],"q":[0],"a":[[1]],"l":[0],"u":[1e30],"results":{"s":{"x":[1],"status":"ok"}}}]}`))
	f.Add([]byte(`{"cases":[{"name":"a","n":2,"p":[[1]],"q":[0]}]}`))
	f.Add([]byte(`{"cases":` + strings.Repeat("[", 5000)))
	f.Add([]byte{})
	f.Fuzz(func(t *testing.T, data []byte) {
		fx, err := Decode(bytes.NewReader(data))
		if err != nil {
			return
		}
		// Whatever decodes is self-consistent and survives a round trip.
		var buf bytes.Buffer
		if err := Encode(&buf, fx); err != nil {
			t.Fatal(err)
		}
		g, err := Decode(&buf)
		if err != nil {
			t.Fatalf("re-decode: %v", err)
		}
		if d := Diff(fx, g, 0); len(d) != 0 {
			t.Fatalf("round trip differs: %v", d)
		}
	})
}
