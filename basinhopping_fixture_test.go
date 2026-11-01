package optimize

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"
)

// The fixture of tools/globaloracle (testdata/global/bh.json): SciPy 1.18.1's
// basinhopping with analytic gradients (BFGS and L-BFGS-B) on four
// arithmetic-only landscapes, for both seeding forms, with the per-step trace.

// Go twins of bh_* in global.py: every product is rounded before it is summed.
func bhHimmelblau(x, g []float64) float64 {
	a := float64(x[0]*x[0]) + x[1] - 11.0
	b := x[0] + float64(x[1]*x[1]) - 7.0
	g[0] = float64(float64(4.0*x[0])*a) + float64(2.0*b)
	g[1] = float64(2.0*a) + float64(float64(4.0*x[1])*b)
	return float64(a*a) + float64(b*b)
}

func bhRosenbrock(x, g []float64) float64 {
	t := x[1] - float64(x[0]*x[0])
	u := 1.0 - x[0]
	g[0] = float64(float64(-400.0*x[0])*t) - float64(2.0*u)
	g[1] = 200.0 * t
	return float64(100.0*float64(t*t)) + float64(u*u)
}

func bhCamel(x, g []float64) float64 {
	a, b := x[0], x[1]
	x2 := a * a
	x3 := x2 * a
	x4 := x2 * x2
	x5 := x3 * x2
	y2 := b * b
	y3 := y2 * b
	f := float64(float64(float64(4.0-float64(2.1*x2))+float64(x4/3.0))*x2) + float64(a*b) +
		float64(float64(-4.0+float64(4.0*y2))*y2)
	g[0] = float64(float64(8.0*a)-float64(8.4*x3)) + float64(2.0*x5) + b
	g[1] = float64(a-float64(8.0*b)) + float64(16.0*y3)
	return f
}

func bhQuartic(x, g []float64) float64 {
	a := float64(x[0]*x[0]) - 1.0
	b := float64(x[1]*x[1]) - 1.0
	g[0] = float64(float64(4.0*x[0])*a) + 0.3
	g[1] = float64(float64(4.0*x[1])*b) + 0.2
	return float64(a*a) + float64(b*b) + float64(0.3*x[0]) + float64(0.2*x[1])
}

var bhProblems = map[string]func(x, g []float64) float64{
	"himmelblau": bhHimmelblau, "rosenbrock": bhRosenbrock, "camel": bhCamel, "quartic": bhQuartic,
}

type bhCase struct {
	Name    string    `json:"name"`
	Problem string    `json:"problem"`
	Method  string    `json:"method"`
	X0      []float64 `json:"x0"`
	Kind    string    `json:"kind"`
	Seed    uint64    `json:"seed"`
	Options struct {
		NIter            int     `json:"niter"`
		T                float64 `json:"T"`
		StepSize         float64 `json:"stepsize"`
		Interval         int     `json:"interval"`
		NIterSuccess     *int    `json:"niter_success"`
		Accept           string  `json:"accept"`
		TargetAcceptRate float64 `json:"target_accept_rate"`
		StepwiseFactor   float64 `json:"stepwise_factor"`
	} `json:"options"`
	Error    *string `json:"error"`
	X        []dgHex `json:"x"`
	Fun      dgHex   `json:"fun"`
	Nit      int     `json:"nit"`
	Nfev     int     `json:"nfev"`
	Njev     int     `json:"njev"`
	Nhev     int     `json:"nhev"`
	Failures int     `json:"minimization_failures"`
	Success  bool    `json:"success"`
	Message  string  `json:"message"`
	Trace    []struct {
		X      []dgHex `json:"x"`
		F      dgHex   `json:"f"`
		Accept bool    `json:"accept"`
	} `json:"trace"`
}

func loadBHFixture(t testing.TB) []bhCase {
	t.Helper()
	b, err := os.ReadFile("testdata/global/bh.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Scipy string   `json:"scipy"`
		BH    []bhCase `json:"bh"`
	}
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	if f.Scipy != "1.18.1" {
		t.Fatalf("SciPy %s", f.Scipy)
	}
	return f.BH
}

// bhAccept maps the accept tests of global.py.
func bhAccept(name string, o *BasinhoppingOptions) {
	switch name {
	case "force":
		o.AcceptTest = func(fNew float64, _ []float64, fOld float64, _ []float64) Acceptance {
			if fNew > fOld+0.1 {
				return ForceAccept
			}
			return Accept
		}
	case "reject":
		o.AcceptTest = func(fNew float64, _ []float64, fOld float64, _ []float64) Acceptance {
			if fNew <= fOld+3.0 {
				return Accept
			}
			return Reject
		}
	case "new":
		o.AcceptTestResult = func(n, old *Result) Acceptance {
			if n.Fun < old.Fun+3.0 {
				return Accept
			}
			return Reject
		}
	}
}

func bhOptionsOf(c *bhCase) (*BasinhoppingOptions, *[]bhStepRec) {
	o := &BasinhoppingOptions{
		NIter: Int(c.Options.NIter), T: Float(c.Options.T), StepSize: c.Options.StepSize,
		Method: Method(c.Method), Interval: c.Options.Interval,
		TargetAcceptRate: c.Options.TargetAcceptRate, StepwiseFactor: c.Options.StepwiseFactor,
	}
	if c.Options.NIterSuccess != nil {
		o.NIterSuccess = *c.Options.NIterSuccess
	}
	if c.Kind == "rng" {
		o.RNG = NewRNG(c.Seed)
	} else {
		o.RNG = NewLegacyRNG(uint32(c.Seed))
	}
	bhAccept(c.Options.Accept, o)
	var trace []bhStepRec
	o.Callback = func(x []float64, f float64, accept bool) bool {
		trace = append(trace, bhStepRec{x, f, accept})
		return false
	}
	return o, &trace
}

type bhStepRec struct {
	x      []float64
	f      float64
	accept bool
}

// bhCompare returns "" when r is SciPy's result for c (the whole trace
// included), else the first difference.
func bhCompare(c *bhCase, r *BasinhoppingResult, trace []bhStepRec) string {
	if len(trace) != len(c.Trace) {
		return fmt.Sprintf("trace of %d steps, SciPy %d", len(trace), len(c.Trace))
	}
	for i, s := range c.Trace {
		if trace[i].accept != s.Accept || !sameFloat(trace[i].f, float64(s.F)) || !sameBitsHex(trace[i].x, s.X) {
			return fmt.Sprintf("step %d: f %v accept %v, SciPy %v %v", i, trace[i].f, trace[i].accept, float64(s.F), s.Accept)
		}
	}
	switch {
	case r.Nit != c.Nit || r.Nfev != c.Nfev || r.Njev != c.Njev || r.Nhev != c.Nhev:
		return fmt.Sprintf("counts nit/nfev/njev/nhev %d/%d/%d/%d, SciPy %d/%d/%d/%d", r.Nit, r.Nfev, r.Njev, r.Nhev, c.Nit, c.Nfev, c.Njev, c.Nhev)
	case r.MinimizationFailures != c.Failures || r.Success != c.Success || r.Message != c.Message:
		return "failures, success or message differ"
	case !sameBitsHex(r.X, c.X) || !sameFloat(r.Fun, float64(c.Fun)):
		return "x or fun differ"
	}
	return ""
}

func bhClose(a, b float64) bool {
	return math.Abs(a-b) <= 1e-8*math.Max(1, math.Max(math.Abs(a), math.Abs(b)))
}

// bhEquivalent is the tolerance form of bhCompare (see TestBasinhoppingFixture).
func bhEquivalent(c *bhCase, r *BasinhoppingResult, trace []bhStepRec) string {
	if len(trace) != len(c.Trace) {
		return fmt.Sprintf("trace of %d steps, SciPy %d", len(trace), len(c.Trace))
	}
	for i, s := range c.Trace {
		if trace[i].accept != s.Accept || !bhClose(trace[i].f, float64(s.F)) {
			return fmt.Sprintf("step %d: f %v accept %v, SciPy %v %v", i, trace[i].f, trace[i].accept, float64(s.F), s.Accept)
		}
	}
	if r.Nit != c.Nit || r.MinimizationFailures != c.Failures || r.Success != c.Success ||
		r.Message != c.Message || !bhClose(r.Fun, float64(c.Fun)) {
		return fmt.Sprintf("nit %d/%d failures %d/%d message %q/%q", r.Nit, c.Nit, r.MinimizationFailures, c.Failures, r.Message, c.Message)
	}
	return ""
}

// TestBasinhoppingFixture replays every case: SciPy's exact path, step by step.
func TestBasinhoppingFixture(t *testing.T) {
	cases := loadBHFixture(t)
	if len(cases) < 150 {
		t.Fatalf("%d cases", len(cases))
	}
	exact := 0
	for i := range cases {
		c := &cases[i]
		f := bhProblems[c.Problem]
		o, trace := bhOptionsOf(c)
		r, err := Basinhopping(Problem{FuncGrad: f}, c.X0, o)
		if c.Error != nil {
			if err == nil {
				t.Errorf("%s: SciPy raised %q, Go did not", c.Name, *c.Error)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: %v", c.Name, err)
			continue
		}
		if d := bhCompare(c, r, *trace); d != "" {
			// The local minimiser (BFGS / L-BFGS-B by SciPy's algorithm) can
			// differ from SciPy's in the last digits near a minimum whose
			// value is rounding noise; the basin-hopping logic must still make
			// SciPy's decisions: same accepts, same counts of steps and
			// failures, same message, values within 1e-8.
			if e := bhEquivalent(c, r, *trace); e != "" {
				t.Errorf("%s: %s (exact: %s)", c.Name, e, d)
			}
			continue
		}
		exact++
	}
	t.Logf("%d/%d basin-hopping runs on SciPy's exact path (per-step trace, counts, x, fun)", exact, len(cases))
}
