package main

import (
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
)

const committed = "../../testdata/slsqp.json"

func load(t *testing.T) *Fixture {
	t.Helper()
	f, err := readFixture(committed)
	if err != nil {
		t.Fatal(err)
	}
	return f
}

// clone deep-copies a fixture through its own encoding.
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

func TestRoundTrip(t *testing.T) {
	f := load(t)
	if len(f.Cases) < 90 {
		t.Fatalf("committed fixture has %d cases", len(f.Cases))
	}
	g := cloneFixture(t, f)
	if !reflect.DeepEqual(f, g) {
		t.Fatal("Decode(Encode(f)) != f")
	}
	if d := Diff(f, g); len(d) != 0 {
		t.Fatalf("a fixture disagrees with its own copy: %v", d)
	}
	var b bytes.Buffer
	if err := Encode(&b, f); err != nil || !bytes.HasSuffix(b.Bytes(), []byte("}\n")) {
		t.Fatalf("encoding should end with a newline: %v", err)
	}
}

func TestDecodeRejects(t *testing.T) {
	for name, in := range map[string]string{
		"not JSON":      "nope",
		"unknown field": `{"cases":[{"name":"a"}],"extra":1}`,
		"no cases":      `{"scipy":"1"}`,
		"duplicate":     `{"cases":[{"name":"a"},{"name":"a"}]}`,
		"unnamed":       `{"cases":[{"problem":"p"}]}`,
	} {
		if _, err := Decode(strings.NewReader(in)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

// TestDiffPlantedPositives changes one field of one case at a time and
// requires Diff to report exactly that case.
func TestDiffPlantedPositives(t *testing.T) {
	base := load(t)
	target := base.Cases[len(base.Cases)/2].Name
	for name, plant := range map[string]func(c *Case){
		"status":      func(c *Case) { c.Status++ },
		"success":     func(c *Case) { c.Success = !c.Success },
		"nfev":        func(c *Case) { c.Nfev++ },
		"fun":         func(c *Case) { c.Fun = c.Fun*(1+1e-6) + 1e-6 },
		"x":           func(c *Case) { c.X[0] += 1e-3 },
		"x length":    func(c *Case) { c.X = append(c.X, 0) },
		"multipliers": func(c *Case) { c.Multipliers = append(c.Multipliers, 1) },
	} {
		other := cloneFixture(t, base)
		for i := range other.Cases {
			if other.Cases[i].Name == target {
				plant(&other.Cases[i])
			}
		}
		d := Diff(base, other)
		if len(d) != 1 || d[0].Case != target {
			t.Errorf("%s planted in %s: Diff reported %v", name, target, d)
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
	// NaN equals NaN, and only NaN.
	if !near(nan(), nan(), 1) || near(nan(), 1, 1) || near(1, nan(), 1) {
		t.Error("near mishandles NaN")
	}
}

func nan() float64 { var z float64; return z / z }

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

func TestGenerateCommand(t *testing.T) {
	raw, err := os.ReadFile(committed)
	if err != nil {
		t.Fatal(err)
	}
	var gotArgv []string
	old := runner
	defer func() { runner = old }()
	runner = func(argv []string, stdin io.Reader, stdout, _ io.Writer) error {
		gotArgv = argv
		in, _ := io.ReadAll(stdin)
		if !bytes.Contains(in, []byte("Hock")) {
			t.Error("the embedded script was not fed on stdin")
		}
		_, err := stdout.Write(raw)
		return err
	}
	dst := filepath.Join(t.TempDir(), "out.json")
	out, err := run(t, "--platform", "linux/amd64", "-o", dst)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "wrote") || gotArgv[0] != "docker" || !slices.Contains(gotArgv, "linux/amd64") {
		t.Errorf("out %q argv %v", out, gotArgv)
	}
	written, err := readFixture(dst)
	if err != nil || len(Diff(load(t), written)) != 0 {
		t.Fatalf("written fixture differs: %v", err)
	}
	if argv := command(""); argv[0] != "uv" {
		t.Errorf("host command %v", argv)
	}

	runner = func([]string, io.Reader, io.Writer, io.Writer) error { return errors.New("boom") }
	if _, err := run(t, "-o", dst); err == nil || !strings.Contains(err.Error(), "boom") {
		t.Errorf("runner failure: %v", err)
	}
	runner = func(_ []string, _ io.Reader, w io.Writer, _ io.Writer) error {
		_, err := w.Write([]byte("{}"))
		return err
	}
	if _, err := run(t, "-o", dst); err == nil {
		t.Error("empty output accepted")
	}
	runner = func(_ []string, _ io.Reader, w io.Writer, _ io.Writer) error { _, err := w.Write(raw); return err }
	if _, err := run(t, "-o", filepath.Join(t.TempDir(), "missing", "x.json")); err == nil {
		t.Error("unwritable output accepted")
	}
}

func TestDiffCommand(t *testing.T) {
	dir := t.TempDir()
	base := load(t)
	write := func(name string, f *Fixture) string {
		var b bytes.Buffer
		if err := Encode(&b, f); err != nil {
			t.Fatal(err)
		}
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, b.Bytes(), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	a := write("a.json", base)
	planted := cloneFixture(t, base)
	planted.Cases[0].Nit += 3
	b := write("b.json", planted)

	out, err := run(t, "diff", a, a)
	if err != nil || !strings.Contains(out, ": 0 disagree") {
		t.Errorf("identical fixtures: %v %q", err, out)
	}
	out, err = run(t, "diff", a, b)
	if !errors.Is(err, ErrDisagree) || !strings.Contains(out, base.Cases[0].Name+": nit/nfev/njev") ||
		!strings.Contains(out, ": 1 disagree") {
		t.Errorf("planted disagreement: %v %q", err, out)
	}
	bad := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(bad, []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, args := range [][]string{
		{"diff", a}, {"diff", filepath.Join(dir, "none"), a}, {"diff", a, filepath.Join(dir, "none")}, {"diff", bad, a},
	} {
		if _, err := run(t, args...); err == nil {
			t.Errorf("%v: no error", args)
		}
	}
}

func TestMainExitsOnError(t *testing.T) {
	code := -1
	oldExit, oldArgs := osExit, os.Args
	defer func() { osExit, os.Args = oldExit, oldArgs }()
	osExit = func(c int) { code = c }
	os.Args = []string{"slsqporacle", "diff", "only-one"}
	main()
	if code != 1 {
		t.Errorf("exit code %d", code)
	}
}
