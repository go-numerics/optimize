package optimize

import (
	"encoding/json"
	"math"
	"os"
	"runtime"
	"strings"
	"testing"

	"github.com/go-ndarray/ndarray"
)

// slsqpCase is one SciPy run of tools/slsqporacle/hs.py.
type slsqpCase struct {
	Name        string             `json:"name"`
	Problem     string             `json:"problem"`
	Jac         bool               `json:"jac"`
	Options     map[string]float64 `json:"options"`
	Tol         *float64           `json:"tol"`
	X           []float64          `json:"x"`
	Fun         float64            `json:"fun"`
	Nit         int                `json:"nit"`
	Nfev        int                `json:"nfev"`
	Njev        int                `json:"njev"`
	Status      int                `json:"status"`
	Success     bool               `json:"success"`
	Message     string             `json:"message"`
	Multipliers []float64          `json:"multipliers"`
	Maxcv       float64            `json:"maxcv"`
	Fstar       *float64           `json:"fstar"`
}

type slsqpFixture struct {
	Scipy   string      `json:"scipy"`
	Machine string      `json:"machine"`
	Cases   []slsqpCase `json:"cases"`
}

func loadSLSQPFixture(t *testing.T) slsqpFixture {
	t.Helper()
	path := "testdata/slsqp.json"
	if p := os.Getenv("SLSQP_FIXTURE"); p != "" {
		path = p // e.g. a fixture generated with slsqporacle --platform
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var f slsqpFixture
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

// runSLSQPCase replays one fixture case in Go.
func runSLSQPCase(t *testing.T, c slsqpCase) *Result {
	t.Helper()
	p, ok := hsProblems[c.Problem]
	if !ok {
		t.Fatalf("%s: unknown problem %q", c.Name, c.Problem)
	}
	prob := Problem{Func: p.f}
	if c.Jac {
		prob.Grad = p.g
	}
	o := &Options{Bounds: p.bounds, Constraints: p.specs(c.Jac)}
	switch {
	case strings.HasPrefix(c.Name, "HS35-linear"):
		a, _ := ndarray.FromData([]float64{1, 1, 2}, 1, 3)
		o.Constraints = []ConstraintSpec{LinearConstraint{A: a, Ub: []float64{3}}}
	case strings.HasPrefix(c.Name, "HS71-nonlinear"):
		o.Constraints = hs71Nonlinear(c.Jac)
	case strings.HasPrefix(c.Name, "HS43-nonlinear"):
		o.Constraints = hs43Nonlinear()
	}
	if v, ok := c.Options["maxiter"]; ok {
		o.MaxIter = int(v)
	}
	if v, ok := c.Options["ftol"]; ok {
		o.Ftol = v
	}
	if v, ok := c.Options["eps"]; ok {
		o.Eps = v
	}
	if c.Tol != nil {
		o.Tol = *c.Tol
	}
	r, err := Minimize(prob, p.x0, SLSQP, o)
	if err != nil {
		t.Fatalf("%s: %v", c.Name, err)
	}
	return r
}

// hsViolation is hs.py's maxcv.
func hsViolation(p hsProblem, x []float64) float64 {
	v := 0.0
	for _, c := range p.cons {
		out := make([]float64, c.m)
		c.fun(x, out)
		for _, ci := range out {
			if c.kind == Eq {
				v = math.Max(v, math.Abs(ci))
			} else {
				v = math.Max(v, -ci)
			}
		}
	}
	for i, b := range p.bounds {
		v = math.Max(v, math.Max(b.Lo-x[i], x[i]-b.Hi))
	}
	return v
}

// slsqpPathExceptions are the runs that leave the path of the fixture
// (SciPy's x86_64 wheel, whose compiled code has no fused multiply-add).
// On every one of them SciPy's own arm64 and x86_64 wheels also disagree
// with each other (slsqporacle without --platform on Apple silicon), so the
// path is decided by last-bit rounding: HS13 is degenerate (no constraint
// qualification at the optimum), Infeasible stops on a positive directional
// derivative after a rounding-dependent number of augmented steps, and
// HS23, HS29 and HS38 run 6 to 84 iterations near-singular updates.
var slsqpPathExceptions = map[string]bool{
	"HS13/jac": true, "HS13/fd": true, "HS23/jac": true,
	"HS29/jac": true, "HS29/fd": true, "HS38/jac": true, "HS38/fd": true,
	"HS38/eps": true, "Infeasible/jac": true, "Infeasible/fd": true,
}

// slsqpLibm are the problems that call sin, cos, log or pow. Go's math
// package has assembly versions of those on some architectures, so their
// paths are only asserted where they were checked (amd64 and arm64).
var slsqpLibm = map[string]bool{
	"HS4": true, "HS5": true, "HS7": true, "HS9": true, "HS13": true, "HS19": true,
	"HS26": true, "HS39": true, "HS40": true,
}

func TestSLSQPFixtures(t *testing.T) {
	f := loadSLSQPFixture(t)
	var statusOK, pathOK, funOK, total int
	for _, c := range f.Cases {
		total++
		r := runSLSQPCase(t, c)
		p := hsProblems[c.Problem]
		cv := hsViolation(p, r.X)
		same := r.Nit == c.Nit && r.Nfev == c.Nfev && r.Njev == c.Njev
		if r.Status == c.Status {
			statusOK++
		}
		if same {
			pathOK++
		}
		fok := math.Abs(r.Fun-c.Fun) <= 1e-6*(1+math.Abs(c.Fun))
		if fok {
			funOK++
		}
		t.Logf("%-22s status %d/%d nit %d/%d nfev %d/%d njev %d/%d fun %.17g/%.17g cv %.2g/%.2g",
			c.Name, r.Status, c.Status, r.Nit, c.Nit, r.Nfev, c.Nfev, r.Njev, c.Njev, r.Fun, c.Fun, cv, c.Maxcv)
		if r.Status != c.Status || r.Success != c.Success {
			t.Errorf("%s: status %d success %v, SciPy %d %v", c.Name, r.Status, r.Success, c.Status, c.Success)
		}
		if r.Message != slsqpMessages[r.Status] && r.Message != c.Message {
			t.Errorf("%s: message %q, SciPy %q", c.Name, r.Message, c.Message)
		}
		checkPath := !slsqpPathExceptions[c.Name] &&
			(!slsqpLibm[c.Problem] || runtime.GOARCH == "amd64" || runtime.GOARCH == "arm64")
		switch {
		case checkPath:
			if !same {
				t.Errorf("%s: nit/nfev/njev %d/%d/%d, SciPy %d/%d/%d", c.Name, r.Nit, r.Nfev, r.Njev, c.Nit, c.Nfev, c.Njev)
			}
			if math.Abs(r.Fun-c.Fun) > 1e-9*(1+math.Abs(c.Fun)) {
				t.Errorf("%s: fun %.17g, SciPy %.17g", c.Name, r.Fun, c.Fun)
			}
			for i := range c.X {
				if math.Abs(r.X[i]-c.X[i]) > 1e-7*(1+math.Abs(c.X[i])) {
					t.Errorf("%s: x[%d] = %.17g, SciPy %.17g", c.Name, i, r.X[i], c.X[i])
				}
			}
			mtol := 1e-6
			if !c.Jac {
				mtol = 1e-5 // finite-difference Jacobians
			}
			for i := range c.Multipliers {
				if math.Abs(r.Multipliers[i]-c.Multipliers[i]) > mtol*(1+math.Abs(c.Multipliers[i])) {
					t.Errorf("%s: multiplier %d = %.17g, SciPy %.17g", c.Name, i, r.Multipliers[i], c.Multipliers[i])
				}
			}
		case r.Success:
			// Accuracy only: the known optimum within SLSQP's own tolerance.
			if c.Fstar != nil && math.Abs(r.Fun-*c.Fstar) > 2e-3*(1+math.Abs(*c.Fstar)) {
				t.Errorf("%s: fun %.17g, optimum %.17g", c.Name, r.Fun, *c.Fstar)
			}
			if cv > 1e-6 {
				t.Errorf("%s: constraint violation %g", c.Name, cv)
			}
		}
	}
	t.Logf("SciPy %s (%s): %d cases, status %d, exact nit/nfev/njev %d, fun within 1e-6 %d",
		f.Scipy, f.Machine, total, statusOK, pathOK, funOK)
}
