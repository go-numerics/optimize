package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

const committed = "../../testdata/global/direct.json"

func load(t *testing.T) *Fixture {
	t.Helper()
	f, err := readFixture(committed)
	if err != nil {
		t.Fatal(err)
	}
	return f
}

// cloneFixture deep-copies a fixture through its own encoding.
func cloneFixture(t *testing.T, f *Fixture) *Fixture {
	t.Helper()
	var b bytes.Buffer
	if err := Encode(&b, f); err != nil {
		t.Fatal(err)
	}
	g, err := Decode(&b)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func hexp(v float64) *Hex { h := Hex(v); return &h }

// small is a two-case fixture that never touches Python.
func small() *Fixture {
	mk := func(name string) Case {
		return Case{
			Name: name, Problem: "p", Family: "arith", Dim: 2,
			Lower: []float64{0, 0}, Upper: []float64{1, 1}, Locally: true,
			Options: Options{Eps: 1e-4, Maxfun: 2000, Maxiter: 1000, FMinRtol: 1e-4, VolTol: 1e-16, LenTol: 1e-6},
			X:       []Hex{0.25, 0.75}, Fun: hexp(0.5), Nfev: 11, Nit: 3, Status: 5, Success: true,
			Message: "m", Trace: [][]Hex{{0.5, 0.5}, {0.1, 0.9}}, Ncallbck: 2,
		}
	}
	return &Fixture{Method: "direct", Scipy: "1.18.1", Numpy: "2", Python: "3", Machine: "x86_64", System: "Linux",
		Cases: []Case{mk("a"), mk("b")}}
}

func TestRoundTrip(t *testing.T) {
	f := load(t)
	if len(f.Cases) < 700 || f.Method != "direct" || f.Scipy != "1.18.1" || f.Machine != "x86_64" {
		t.Fatalf("committed fixture: %d cases, %s, SciPy %s, %s", len(f.Cases), f.Method, f.Scipy, f.Machine)
	}
	g := cloneFixture(t, f)
	// (reflect.DeepEqual would fail here: the fixture holds NaN values.)
	if s := small(); !reflect.DeepEqual(s, cloneFixture(t, s)) {
		t.Fatal("Decode(Encode(f)) != f")
	}
	if d := Diff(f, g); len(d) != 0 {
		t.Fatalf("a fixture disagrees with its own copy: %v", d)
	}
	var b bytes.Buffer
	if err := Encode(&b, f); err != nil || !bytes.HasSuffix(b.Bytes(), []byte("}\n")) {
		t.Fatalf("encoding should end with a newline: %v", err)
	}
	// The committed file is the canonical encoding.
	disk, err := os.ReadFile(committed)
	if err != nil || !bytes.Equal(disk, b.Bytes()) {
		t.Fatalf("the committed fixture is not the output of Encode (%v)", err)
	}
}

func TestHex(t *testing.T) {
	for _, v := range []float64{0, math.Copysign(0, -1), 1, -2.5, math.Inf(1), math.Inf(-1), math.NaN(), math.SmallestNonzeroFloat64} {
		b, err := json.Marshal(Hex(v))
		if err != nil {
			t.Fatal(err)
		}
		var h Hex
		if err := json.Unmarshal(b, &h); err != nil || !sameBits(float64(h), v) && !(math.IsNaN(v) && math.IsNaN(float64(h))) {
			t.Errorf("%v -> %s -> %v (%v)", v, b, h, err)
		}
	}
	for _, in := range []string{`1`, `"1"`, `"0x12"`, `"0xzzzzzzzzzzzzzzzz"`, `"1x0000000000000000"`} {
		var h Hex
		if err := json.Unmarshal([]byte(in), &h); err == nil {
			t.Errorf("%s accepted as %v", in, h)
		}
	}
}

func TestDecodeRejects(t *testing.T) {
	ok := small()
	mut := func(f func(c *Case)) string {
		g := cloneFixture(t, ok)
		f(&g.Cases[0])
		var b bytes.Buffer
		if err := Encode(&b, g); err != nil {
			t.Fatal(err)
		}
		return b.String()
	}
	msg := "boom"
	for name, in := range map[string]string{
		"not JSON":          "nope",
		"unknown field":     `{"cases":[{"name":"a"}],"extra":1}`,
		"no cases":          `{"scipy":"1"}`,
		"duplicate":         mut(func(c *Case) { c.Name = "b" }),
		"unnamed":           mut(func(c *Case) { c.Name = "" }),
		"dimension":         mut(func(c *Case) { c.Dim = 3 }),
		"zero dimension":    mut(func(c *Case) { c.Dim = 0; c.Lower, c.Upper = nil, nil }),
		"short upper":       mut(func(c *Case) { c.Upper = c.Upper[:1] }),
		"short x":           mut(func(c *Case) { c.X = c.X[:1] }),
		"missing fun":       mut(func(c *Case) { c.Fun = nil }),
		"short trace entry": mut(func(c *Case) { c.Trace[1] = c.Trace[1][:1] }),
	} {
		if _, err := Decode(strings.NewReader(in)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	// A case SciPy raised on needs no result.
	g := cloneFixture(t, ok)
	g.Cases[0].Error = &msg
	g.Cases[0].X, g.Cases[0].Fun = nil, nil
	var b bytes.Buffer
	if err := Encode(&b, g); err != nil {
		t.Fatal(err)
	}
	if _, err := Decode(&b); err != nil {
		t.Errorf("error case rejected: %v", err)
	}
}

// TestDiffPlantedPositives changes one field of one case at a time and
// requires Diff to report exactly that case.
func TestDiffPlantedPositives(t *testing.T) {
	for _, base := range []*Fixture{small(), load(t)} {
		target := base.Cases[len(base.Cases)/2].Name
		ulp := func(h *Hex) Hex { return Hex(math.Nextafter(float64(*h), math.Inf(1))) }
		fm := 3.0
		for name, plant := range map[string]func(c *Case){
			"status":   func(c *Case) { c.Status++ },
			"success":  func(c *Case) { c.Success = !c.Success },
			"message":  func(c *Case) { c.Message += "." },
			"nfev":     func(c *Case) { c.Nfev++ },
			"nit":      func(c *Case) { c.Nit++ },
			"fun ulp":  func(c *Case) { *c.Fun = ulp(c.Fun) },
			"x ulp":    func(c *Case) { c.X[len(c.X)-1] = ulp(&c.X[len(c.X)-1]) },
			"x length": func(c *Case) { c.X = append(c.X, 0) },
			"trace":    func(c *Case) { c.Trace = append(c.Trace, c.Dimzero()) },
			"trace bit": func(c *Case) {
				if len(c.Trace) == 0 {
					c.Trace = [][]Hex{make([]Hex, c.Dim)}
				}
				c.Trace[0][0] = ulp(&c.Trace[0][0])
			},
			"trace length": func(c *Case) { c.Trace = c.Trace[:0] },
			"maxfun":       func(c *Case) { c.Options.Maxfun++ },
			"eps":          func(c *Case) { c.Options.Eps = ulp(hexp(c.Options.Eps)).f() },
			"f_min":        func(c *Case) { c.Options.FMin = &fm },
			"error":        func(c *Case) { e := "x"; c.Error = &e },
		} {
			other := cloneFixture(t, base)
			for i := range other.Cases {
				if other.Cases[i].Name == target {
					plant(&other.Cases[i])
				}
			}
			d := Diff(base, other)
			if len(d) == 0 && name == "trace length" && len(base.Cases[0].Trace) == 0 {
				continue // nothing to shorten
			}
			if len(d) == 0 || d[0].Case != target {
				t.Errorf("%s planted in %s: Diff reported %v", name, target, d)
			}
			for _, x := range d {
				if x.Case != target {
					t.Errorf("%s planted in %s: Diff blamed %s", name, target, x.Case)
				}
			}
		}
		// A case present on one side only, in each direction.
		other := cloneFixture(t, base)
		other.Cases = other.Cases[1:]
		if d := Diff(base, other); len(d) != 1 || !strings.Contains(d[0].What, "first") {
			t.Errorf("dropped case: %v", d)
		}
		if d := Diff(other, base); len(d) != 1 || !strings.Contains(d[0].What, "second") {
			t.Errorf("added case: %v", d)
		}
	}
}

// Dimzero and f are small helpers of the tests above.
func (c *Case) Dimzero() []Hex { return make([]Hex, c.Dim) }

func (h Hex) f() float64 { return float64(h) }

func TestDiffFMinValues(t *testing.T) {
	a, b := small(), small()
	one, also, two := 1.0, 1.0, 2.0
	a.Cases[0].Options.FMin, b.Cases[0].Options.FMin = &one, &also
	if d := Diff(a, b); len(d) != 0 {
		t.Fatalf("equal f_min in distinct variables disagree: %v", d)
	}
	b.Cases[0].Options.FMin = &two
	if d := Diff(a, b); len(d) != 1 {
		t.Fatalf("different f_min: %v", d)
	}
}

func TestDiffTreatsErrorsAndSignedZeros(t *testing.T) {
	a, b := small(), small()
	msg := "ValueError"
	a.Cases[0].Error, b.Cases[0].Error = &msg, &msg
	if d := Diff(a, b); len(d) != 0 {
		t.Fatalf("identical errors disagree: %v", d)
	}
	other := "TypeError"
	b.Cases[0].Error = &other
	if d := Diff(a, b); len(d) != 1 {
		t.Fatalf("different errors: %v", d)
	}
	b.Cases[0].Error = nil
	if d := Diff(a, b); len(d) != 1 {
		t.Fatalf("error vs result: %v", d)
	}
	// Several disagreements come out sorted by case name.
	a, b = small(), small()
	a.Cases[0], a.Cases[1] = a.Cases[1], a.Cases[0]
	a.Cases[0].Nit++
	a.Cases[1].Nit++
	if d := Diff(a, b); len(d) != 2 || d[0].Case != "a" || d[1].Case != "b" {
		t.Fatalf("%v", d)
	}
	// -0 and +0 are different bit patterns, NaN equals itself bit for bit.
	a, b = small(), small()
	*a.Cases[0].Fun, *b.Cases[0].Fun = 0, Hex(math.Copysign(0, -1))
	if d := Diff(a, b); len(d) != 1 {
		t.Fatalf("signed zeros: %v", d)
	}
	*a.Cases[0].Fun, *b.Cases[0].Fun = Hex(math.NaN()), Hex(math.NaN())
	if d := Diff(a, b); len(d) != 0 {
		t.Fatalf("NaN against NaN: %v", d)
	}
}

func TestCommand(t *testing.T) {
	if got := command(""); got[0] != "uv" || got[len(got)-1] != "-" {
		t.Errorf("host command %v", got)
	}
	got := command("linux/amd64")
	if got[0] != "docker" || !strings.Contains(strings.Join(got, " "), "--platform linux/amd64") || got[len(got)-1] != "-" {
		t.Errorf("container command %v", got)
	}
	if !strings.Contains(strings.Join(got, " "), scipyPin) || !strings.Contains(strings.Join(got, " "), "python -I -") {
		t.Errorf("pins or isolation missing: %v", got)
	}
}

func fakeRunner(t *testing.T, stdout string, err error, gotArgs *[]string, gotScript *string) {
	t.Helper()
	old := runner
	t.Cleanup(func() { runner = old })
	runner = func(argv []string, stdin io.Reader, out, errw io.Writer) error {
		if gotArgs != nil {
			*gotArgs = argv
		}
		if gotScript != nil {
			b, _ := io.ReadAll(stdin)
			*gotScript = string(b)
		}
		io.WriteString(out, stdout)
		return err
	}
}

func encoded(t *testing.T, f *Fixture) string {
	t.Helper()
	var b bytes.Buffer
	if err := Encode(&b, f); err != nil {
		t.Fatal(err)
	}
	return b.String()
}

func run(args ...string) (string, error) {
	cmd := RootCmd()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetErr(&out)
	cmd.SetArgs(args)
	err := cmd.Execute()
	return out.String(), err
}

func TestGen(t *testing.T) {
	dir := t.TempDir()
	out := filepath.Join(dir, "d.json")
	var argv []string
	var script string
	fakeRunner(t, encoded(t, small()), nil, &argv, &script)
	msg, err := run("gen", "direct", "--platform", "linux/amd64", "-o", out)
	if err != nil || !strings.Contains(msg, "wrote 2 cases (SciPy 1.18.1, x86_64)") {
		t.Fatalf("%q, %v", msg, err)
	}
	if argv[0] != "docker" || !strings.HasPrefix(script, "METHOD = \"direct\"\n") || !strings.Contains(script, "def run_direct") {
		t.Errorf("argv %v; script starts %q", argv, script[:40])
	}
	if g, err := readFixture(out); err != nil || len(g.Cases) != 2 {
		t.Fatalf("written fixture: %v", err)
	}

	// Default output path is relative to the working directory.
	wd, _ := os.Getwd()
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(wd)
	if err := os.MkdirAll("testdata/global", 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := run("gen", "direct"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat("testdata/global/direct.json"); err != nil {
		t.Error(err)
	}

	// Failures, each one a different place.
	if _, err := run("gen", "bogus"); err == nil || !strings.Contains(err.Error(), "unknown method") {
		t.Errorf("unknown method: %v", err)
	}
	if _, err := run("gen"); err == nil {
		t.Error("no method accepted")
	}
	wrong := small()
	wrong.Method = "de"
	fakeRunner(t, encoded(t, wrong), nil, nil, nil)
	if _, err := run("gen", "direct", "-o", out); err == nil || !strings.Contains(err.Error(), "asked for") {
		t.Errorf("wrong method in output: %v", err)
	}
	fakeRunner(t, "not json", nil, nil, nil)
	if _, err := run("gen", "direct", "-o", out); err == nil || !strings.Contains(err.Error(), "oracle output") {
		t.Errorf("bad output: %v", err)
	}
	fakeRunner(t, "", errors.New("exit status 1"), nil, nil)
	if _, err := run("gen", "direct", "-o", out); err == nil || !strings.Contains(err.Error(), "oracle:") {
		t.Errorf("runner failure: %v", err)
	}
	fakeRunner(t, encoded(t, small()), nil, nil, nil)
	if _, err := run("gen", "direct", "-o", filepath.Join(dir, "no", "such", "dir", "x.json")); err == nil {
		t.Error("unwritable output accepted")
	}
}

func TestDiffCommand(t *testing.T) {
	dir := t.TempDir()
	a, b := filepath.Join(dir, "a.json"), filepath.Join(dir, "b.json")
	if err := os.WriteFile(a, []byte(encoded(t, small())), 0o644); err != nil {
		t.Fatal(err)
	}
	other := small()
	other.Cases[1].Nfev++
	if err := os.WriteFile(b, []byte(encoded(t, other)), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := run("diff", a, a)
	if err != nil || !strings.Contains(out, "2 cases") || !strings.Contains(out, "0 disagree") {
		t.Fatalf("%q, %v", out, err)
	}
	out, err = run("diff", a, b)
	if !errors.Is(err, ErrDisagree) || !strings.Contains(out, "b: nit/nfev") || !strings.Contains(out, "1 disagree") {
		t.Fatalf("%q, %v", out, err)
	}
	if _, err := run("diff", a, filepath.Join(dir, "missing.json")); err == nil {
		t.Error("missing second file accepted")
	}
	if _, err := run("diff", filepath.Join(dir, "missing.json"), a); err == nil {
		t.Error("missing first file accepted")
	}
	bad := filepath.Join(dir, "bad.json")
	os.WriteFile(bad, []byte("{"), 0o644)
	if _, err := run("diff", bad, a); err == nil || !strings.Contains(err.Error(), "bad.json") {
		t.Errorf("bad file: %v", err)
	}
	if _, err := run("diff", a); err == nil {
		t.Error("one argument accepted")
	}
}

func TestRunnerAndMain(t *testing.T) {
	// The real runner, on a command that exists everywhere Go does.
	var out bytes.Buffer
	if err := runner([]string{os.Args[0], "-test.run=NoSuchTest"}, strings.NewReader(""), &out, io.Discard); err != nil {
		t.Fatal(err)
	}
	if err := runner([]string{filepath.Join(t.TempDir(), "absent")}, nil, &out, io.Discard); err == nil {
		t.Error("absent command ran")
	}

	code := -1
	old := osExit
	osExit = func(c int) { code = c }
	defer func() { osExit = old }()
	oldArgs := os.Args
	defer func() { os.Args = oldArgs }()
	os.Args = []string{"globaloracle", "diff"}
	main()
	if code != 1 {
		t.Errorf("failure exit code %d", code)
	}
	code = -1
	os.Args = []string{"globaloracle", "--help"}
	main()
	if code != -1 {
		t.Errorf("help exited with %d", code)
	}
}

func FuzzDecode(f *testing.F) {
	for _, seed := range []string{"", "{}", `{"cases":[]}`, `{"cases":[{"name":"a"}]}`, encoded2(small())} {
		f.Add([]byte(seed))
	}
	if b, err := os.ReadFile(committed); err == nil {
		f.Add(b[:min(len(b), 4000)])
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		fx, err := Decode(bytes.NewReader(data))
		if err != nil {
			return
		}
		// Whatever Decode accepts, Encode, Decode and Diff must handle.
		var b bytes.Buffer
		if err := Encode(&b, fx); err != nil {
			return // for instance an unrepresentable float
		}
		g, err := Decode(&b)
		if err != nil {
			t.Fatalf("re-decoding an encoded fixture: %v", err)
		}
		if d := Diff(fx, g); len(d) != 0 {
			t.Fatalf("a fixture disagrees with its copy: %v", d)
		}
	})
}

func encoded2(f *Fixture) string {
	var b bytes.Buffer
	if err := Encode(&b, f); err != nil {
		return ""
	}
	return b.String()
}
