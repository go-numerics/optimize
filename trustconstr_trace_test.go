package optimize

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// traceEvent is one objective, gradient or constraint evaluation ("f", "g",
// "c0", "j0", ...) or one iteration ("it") of a run, in the format that
// tools/slsqporacle trace writes for SciPy, so that `slsqporacle tracediff`
// can find the first place where the two sides part.
type traceEvent struct {
	Tag string
	X   []float64
}

func (e traceEvent) MarshalJSON() ([]byte, error) {
	return json.Marshal([]any{e.Tag, e.X})
}

// traceCase runs the fixture case of that name (e.g. "HS21/jac") with the
// given method and records every evaluation and iteration.
func traceCase(t *testing.T, name string, method Method) []traceEvent {
	t.Helper()
	prob, mode, ok := strings.Cut(name, "/")
	p, known := hsProblems[prob]
	if !ok || !known || (mode != "jac" && mode != "fd") {
		t.Fatalf("trace: unknown case %q", name)
	}
	var ev []traceEvent
	rec := func(tag string, x []float64) { ev = append(ev, traceEvent{tag, clone(x)}) }
	jac := mode == "jac"
	pr := Problem{Func: func(x []float64) float64 { rec("f", x); return p.f(x) }}
	if jac {
		pr.Grad = func(x, g []float64) { rec("g", x); p.g(x, g) }
	}
	var specs []ConstraintSpec
	for i, c := range p.cons {
		c := c
		k := Constraint{Kind: c.kind, M: c.m, Func: func(x, out []float64) { rec(fmt.Sprintf("c%d", i), x); c.fun(x, out) }}
		if jac {
			k.Jac = func(x, j []float64) { rec(fmt.Sprintf("j%d", i), x); c.jac(x, j) }
		}
		specs = append(specs, k)
	}
	tcSpy = rec
	defer func() { tcSpy = nil }()
	o := &Options{Bounds: p.bounds, Constraints: specs,
		Callback: func(x []float64, _ float64) bool { rec("it", x); return false }}
	if _, err := Minimize(pr, p.x0, method, o); err != nil {
		t.Fatalf("trace %s: %v", name, err)
	}
	return ev
}

// TestTraceDump writes the trace of the case named by OPTIMIZE_TRACE_CASE
// (default HS21/jac) with trust-constr to OPTIMIZE_TRACE_OUT (default: a
// temporary file) for `slsqporacle tracediff`.
func TestTraceDump(t *testing.T) {
	name := os.Getenv("OPTIMIZE_TRACE_CASE")
	if name == "" {
		name = "HS21/jac"
	}
	out := os.Getenv("OPTIMIZE_TRACE_OUT")
	if out == "" {
		out = filepath.Join(t.TempDir(), "trace.json")
	}
	ev := traceCase(t, name, TrustConstr)
	if len(ev) == 0 {
		t.Fatal("empty trace")
	}
	b, err := json.Marshal(map[string]any{"case": name, "events": ev})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(out, b, 0o644); err != nil {
		t.Fatal(err)
	}
}
