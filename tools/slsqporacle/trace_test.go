package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func ev(tag string, x ...float64) Event { return Event{tag, x} }

// base is a run of two iterations with one gradient and one constraint.
func base() []Event {
	return []Event{
		ev("f", 1, 2), ev("g", 1, 2), ev("c0", 1, 2), ev("it", 1, 2),
		ev("c0", 3, 4), ev("g", 3, 4), ev("f", 3, 4), ev("it", 3, 4),
	}
}

func TestCompareIdentical(t *testing.T) {
	bit, beyond := Compare(base(), base(), 1e-12)
	if len(bit) != 0 || len(beyond) != 0 {
		t.Fatalf("identical traces diverge: %v %v", bit, beyond)
	}
}

// The order of the gradient and the constraint inside an iteration is not
// part of the comparison (SciPy and Go evaluate them in different orders).
func TestCompareIgnoresOrderAcrossTags(t *testing.T) {
	b := base()
	b[4], b[5] = b[5], b[4] // g before c0 in the second iteration
	if bit, _ := Compare(base(), b, 1e-12); len(bit) != 0 {
		t.Fatalf("reordering across tags reported: %v", bit)
	}
}

// Planted positives: a one-ulp change is reported bitwise but not beyond the
// tolerance; a 1e-6 change is reported beyond it; the iteration is the one in
// which the first differing event happens.
func TestComparePlanted(t *testing.T) {
	ulp := base()
	ulp[6] = ev("f", 3, math.Nextafter(4, 5))
	bit, beyond := Compare(base(), ulp, 1e-12)
	if len(bit) != 1 || bit[0].Tag != "f" || bit[0].Index != 1 || bit[0].Iter != 1 || len(beyond) != 0 {
		t.Fatalf("one-ulp change: %v %v", bit, beyond)
	}
	if bit[0].MaxRel == 0 || bit[0].MaxRel > 1e-15 {
		t.Errorf("relative difference %g", bit[0].MaxRel)
	}
	big := base()
	big[1] = ev("g", 1, 2.000002)
	big[7] = ev("it", 3, 4.0001)
	bit, beyond = Compare(base(), big, 1e-12)
	if len(bit) != 2 || len(beyond) != 2 {
		t.Fatalf("large changes: %v %v", bit, beyond)
	}
	if bit[0].Tag != "g" || bit[0].Iter != 0 || bit[1].Tag != "it" || bit[1].Iter != 1 {
		t.Errorf("not sorted by iteration: %v", bit)
	}
}

// A stream that is shorter or longer, or a tag present on one side only, is a
// divergence at the first missing event, and a different dimension too.
func TestCompareLengths(t *testing.T) {
	short := base()[:6]
	bit, beyond := Compare(base(), short, 1e-12)
	if len(bit) != 2 || len(beyond) != 2 || !math.IsInf(bit[0].MaxRel, 1) {
		t.Fatalf("short second trace: %v %v", bit, beyond)
	}
	if bit, _ := Compare(short, base(), 1e-12); len(bit) != 2 {
		t.Fatalf("short first trace: %v", bit)
	}
	extra := append(base(), ev("j0", 1, 1))
	bit, _ = Compare(base(), extra, 1e-12)
	if len(bit) != 1 || bit[0].Tag != "j0" || bit[0].Index != 0 {
		t.Fatalf("tag on one side only: %v", bit)
	}
	if bit, _ = Compare(extra, base(), 1e-12); len(bit) != 1 || bit[0].Tag != "j0" {
		t.Fatalf("tag on the first side only: %v", bit)
	}
	dim := base()
	dim[0] = ev("f", 1, 2, 3)
	if bit, _ := Compare(base(), dim, 1e-12); len(bit) != 1 || !math.IsInf(bit[0].MaxRel, 1) {
		t.Fatalf("dimension change: %v", bit)
	}
	// A rounding-level difference followed by a missing event: both reports.
	ulpShort := base()[:6]
	ulpShort[0] = ev("f", 1, math.Nextafter(2, 3))
	bit, beyond = Compare(base(), ulpShort, 1e-12)
	if len(bit) != 2 || len(beyond) != 2 || bit[0].Tag != "f" || bit[0].Index != 0 || math.IsInf(bit[0].MaxRel, 1) {
		t.Fatalf("rounding difference and a missing event: %v %v", bit, beyond)
	}
	nan := base()
	nan[0] = ev("f", math.NaN(), 2)
	if bit, _ := Compare(nan, nan, 1e-12); len(bit) != 0 {
		t.Fatalf("NaN against itself: %v", bit)
	}
}

func TestEventJSON(t *testing.T) {
	var e Event
	if err := json.Unmarshal([]byte(`["f",[1,2.5]]`), &e); err != nil || e.Tag != "f" || e.X[1] != 2.5 {
		t.Fatalf("%v %+v", err, e)
	}
	b, err := json.Marshal(e)
	if err != nil || string(b) != `["f",[1,2.5]]` {
		t.Fatalf("%v %s", err, b)
	}
	for _, bad := range []string{`["f"]`, `[1,[2]]`, `["f",3]`, `{}`} {
		if json.Unmarshal([]byte(bad), &e) == nil {
			t.Errorf("accepted %s", bad)
		}
	}
}

func TestDecodeTrace(t *testing.T) {
	if _, err := DecodeTrace(strings.NewReader(`{"case":"x","events":[]}`)); err == nil {
		t.Error("accepted a trace without events")
	}
	if _, err := DecodeTrace(strings.NewReader(`{`)); err == nil {
		t.Error("accepted truncated JSON")
	}
	tr, err := DecodeTrace(strings.NewReader(`{"case":"x","events":[["f",[1]]]}`))
	if err != nil || len(tr.Events) != 1 {
		t.Fatalf("%v %+v", err, tr)
	}
}

func writeTrace(t *testing.T, name string, ev []Event) string {
	t.Helper()
	b, err := json.Marshal(Trace{Case: "HS21/jac", Events: ev})
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestTracediffCommand(t *testing.T) {
	a := writeTrace(t, "a.json", base())
	out, err := run(t, "tracediff", a, a)
	if err != nil || !strings.Contains(out, "identical") {
		t.Fatalf("identical: %v %q", err, out)
	}
	ulp := base()
	ulp[6] = ev("f", 3, math.Nextafter(4, 5))
	out, err = run(t, "tracediff", a, writeTrace(t, "b.json", ulp))
	if !errors.Is(err, ErrDisagree) || !strings.Contains(out, `bitwise: tag "f" event 1 (after 1 iterations)`) ||
		!strings.Contains(out, "no difference above") {
		t.Fatalf("one ulp: %v %q", err, out)
	}
	big := base()
	big[6] = ev("f", 3, 5)
	out, err = run(t, "tracediff", "--tol", "1e-3", "--show", "f", "--index", "1", a, writeTrace(t, "c.json", big))
	if !errors.Is(err, ErrDisagree) || !strings.Contains(out, "above 0.001: tag \"f\"") ||
		strings.Count(out, "f[1] in") != 2 {
		t.Fatalf("large change: %v %q", err, out)
	}
	out, _ = run(t, "tracediff", "--quiet", "--show", "f", "--index", "1", a, writeTrace(t, "d.json", big))
	if strings.Contains(out, "bitwise") || strings.Contains(out, "above") || !strings.Contains(out, "f[1] in") {
		t.Fatalf("quiet: %q", out)
	}
	if _, err := run(t, "tracediff", a, filepath.Join(t.TempDir(), "missing.json")); err == nil {
		t.Error("missing file accepted")
	}
	empty := filepath.Join(t.TempDir(), "empty.json")
	if err := os.WriteFile(empty, []byte(`{"events":[]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := run(t, "tracediff", empty, a); err == nil {
		t.Error("empty trace accepted")
	}
	if _, err := run(t, "tracediff", a); err == nil {
		t.Error("one argument accepted")
	}
}

func TestTraceCommand(t *testing.T) {
	old := runner
	defer func() { runner = old }()
	var src []byte
	var argv []string
	reply := `{"case":"HS21/jac","scipy":"1.18.1","machine":"x86_64","events":[["f",[1,2]]],"result":{"nit":3}}`
	runner = func(a []string, stdin io.Reader, stdout, _ io.Writer) error {
		argv = a
		src, _ = io.ReadAll(stdin)
		_, err := stdout.Write([]byte(reply))
		return err
	}
	dst := filepath.Join(t.TempDir(), "t.json")
	out, err := run(t, "trace", "--platform", "linux/amd64", "--blas-verbose", "-o", dst, "HS21/jac")
	if err != nil || !strings.Contains(out, "wrote 1 events of HS21/jac (SciPy 1.18.1, x86_64)") {
		t.Fatalf("%v %q", err, out)
	}
	for _, want := range []string{`METHOD = "trust-constr"`, `TRACE = "HS21/jac"`, "OPENBLAS_VERBOSE", "Hock"} {
		if !bytes.Contains(src, []byte(want)) {
			t.Errorf("script lacks %q", want)
		}
	}
	if argv[0] != "docker" {
		t.Errorf("argv %v", argv)
	}
	b, err := os.ReadFile(dst)
	if err != nil || !bytes.Contains(b, []byte(`"events":[["f",[1,2]]]`)) {
		t.Fatalf("%v %s", err, b)
	}
	if _, err := run(t, "trace", "--method", "BFGS", "HS21/jac"); err == nil {
		t.Error("bad method accepted")
	}
	runner = func([]string, io.Reader, io.Writer, io.Writer) error { return errors.New("boom") }
	if _, err := run(t, "trace", "-o", dst, "HS21/jac"); err == nil || !strings.Contains(err.Error(), "boom") {
		t.Errorf("runner failure: %v", err)
	}
	runner = func(_ []string, _ io.Reader, w io.Writer, _ io.Writer) error {
		_, err := w.Write([]byte("{"))
		return err
	}
	if _, err := run(t, "trace", "-o", dst, "HS21/jac"); err == nil {
		t.Error("garbage output accepted")
	}
	runner = func(_ []string, _ io.Reader, w io.Writer, _ io.Writer) error {
		_, err := w.Write([]byte(reply))
		return err
	}
	if _, err := run(t, "trace", "-o", filepath.Join(t.TempDir(), "no", "such", "dir.json"), "HS21/jac"); err == nil {
		t.Error("unwritable output accepted")
	}
	if _, err := run(t, "trace"); err == nil {
		t.Error("missing case accepted")
	}
}
