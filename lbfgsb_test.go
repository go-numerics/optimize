package optimize

import (
	_ "embed"
	"encoding/json"
	"errors"
	"math"
	"strconv"
	"testing"
)

//go:embed testdata/lbfgsb.json
var lbfgsbFixtures []byte

type lbCase struct {
	Name     string         `json:"name"`
	Obj      string         `json:"obj"`
	X0       []float64      `json:"x0"`
	Analytic bool           `json:"analytic"`
	Bounds   [][2]*float64  `json:"bounds"`
	Tol      *float64       `json:"tol"`
	Options  map[string]any `json:"options"`
	Params   *lbPolyParams  `json:"params"`
	X        []lbNum        `json:"x"`
	Fun      lbNum          `json:"fun"`
	Jac      []lbNum        `json:"jac"`
	Nit      int            `json:"nit"`
	Nfev     int            `json:"nfev"`
	Njev     int            `json:"njev"`
	Status   int            `json:"status"`
	Success  bool           `json:"success"`
	Message  string         `json:"message"`
	Trace    []lbNum        `json:"trace"`
}

// lbNum is a fixture number: a JSON number, or "NaN", "+Inf", "-Inf".
type lbNum float64

func (v *lbNum) UnmarshalJSON(b []byte) error {
	var s string
	if json.Unmarshal(b, &s) == nil {
		f, err := strconv.ParseFloat(s, 64)
		*v = lbNum(f)
		return err
	}
	var f float64
	err := json.Unmarshal(b, &f)
	*v = lbNum(f)
	return err
}

func lbNums(v []lbNum) []float64 {
	f := make([]float64, len(v))
	for i := range v {
		f[i] = float64(v[i])
	}
	return f
}

// Test objectives besides those of funcs_test.go, the same expressions as
// testdata/gen/lbfgsb.py. The float64() conversions forbid fused
// multiply-adds so that the rounding matches Python's.

func lbQuadB(i int) float64 {
	s := 1.0
	if i%2 == 1 {
		s = -1
	}
	return float64(s * (1 + float64(0.1*float64(i))))
}

func lbQuadAx(x []float64, i int) float64 {
	ax := float64(float64(i+1) * x[i])
	if i > 0 {
		ax += float64(0.5 * x[i-1])
	}
	if i+1 < len(x) {
		ax += float64(0.5 * x[i+1])
	}
	return ax
}

func lbQuad(x []float64) float64 {
	f := 0.0
	for i := range x {
		f += float64(float64(0.5*x[i])*lbQuadAx(x, i)) - float64(lbQuadB(i)*x[i])
	}
	return f
}

func lbQuadG(x, g []float64) {
	for i := range x {
		g[i] = lbQuadAx(x, i) - lbQuadB(i)
	}
}

func lbSphere(x []float64) float64 { return float64(x[0]*x[0]) + float64(x[1]*x[1]) }

func lbSphereBadG(x, g []float64) {
	g[0], g[1] = float64(-2*x[0]), float64(-2*x[1])
}

func lbNaNG(x, g []float64) { g[0], g[1] = math.NaN(), math.NaN() }

func lbTiny(x []float64) float64 { return float64(1e-160 * x[0]) }

func lbTinyG(x, g []float64) { g[0] = 1e-160 }

var lbObjectives = map[string]struct {
	f func([]float64) float64
	g func(x, g []float64)
}{
	"rosen":       {rosen, rosenDer},
	"beale":       {beale, bealeDer},
	"himmelblau":  {himmelblau, himmelblauDer},
	"wood":        {wood, woodDer},
	"powell":      {powellSing, powellSingDer},
	"quad":        {lbQuad, lbQuadG},
	"sphere_badg": {lbSphere, lbSphereBadG},
	"tiny":        {lbTiny, lbTinyG},
	"sphere_nang": {lbSphere, lbNaNG},
}

func lbLoad(t *testing.T) []lbCase {
	t.Helper()
	var fx struct {
		Scipy string   `json:"scipy"`
		Cases []lbCase `json:"cases"`
	}
	if err := json.Unmarshal(lbfgsbFixtures, &fx); err != nil {
		t.Fatal(err)
	}
	t.Logf("fixtures from SciPy %s", fx.Scipy)
	return fx.Cases
}

// lbPolyParams defines f(x) = ½x'Ax - b'x + c·Σx⁴ (testdata/gen/lbfgsb.py
// poly), the family of the branch-coverage fixtures.
type lbPolyParams struct {
	A, B []float64
	C    float64
}

func (p *lbPolyParams) f(x []float64) float64 {
	n := len(x)
	f := 0.0
	for i := range x {
		ax := 0.0
		for j := range x {
			ax += float64(p.A[i*n+j] * x[j])
		}
		f += float64(float64(0.5*x[i])*ax) - float64(p.B[i]*x[i])
		x2 := float64(x[i] * x[i])
		f += float64(p.C * float64(x2*x2))
	}
	return f
}

func (p *lbPolyParams) g(x, g []float64) {
	n := len(x)
	for i := range x {
		ax := 0.0
		for j := range x {
			ax += float64(p.A[i*n+j] * x[j])
		}
		x3 := float64(float64(x[i]*x[i]) * x[i])
		g[i] = ax - p.B[i] + float64(4*p.C*x3)
	}
}

func (c *lbCase) problem() Problem {
	if c.Params != nil {
		return Problem{Func: c.Params.f, Grad: c.Params.g}
	}
	o := lbObjectives[c.Obj]
	p := Problem{Func: o.f}
	if c.Analytic {
		p.Grad = o.g
	}
	return p
}

func (c *lbCase) options(trace *[]float64) *Options {
	o := &Options{Callback: func(x []float64, f float64) bool {
		*trace = append(*trace, f)
		return false
	}}
	if c.Tol != nil {
		o.Tol = *c.Tol
	}
	for k, v := range c.Options {
		f := v.(float64)
		switch k {
		case "maxiter":
			o.MaxIter = int(f)
		case "maxfun":
			o.MaxFev = int(f)
		case "maxls":
			o.MaxLS = int(f)
		case "maxcor":
			o.M = int(f)
		case "gtol":
			o.Gtol = f
		default:
			panic("unknown option " + k)
		}
	}
	if c.Bounds != nil {
		o.Bounds = make([]Bound, len(c.Bounds))
		for i, b := range c.Bounds {
			o.Bounds[i] = Unbounded()
			if b[0] != nil {
				o.Bounds[i].Lo = *b[0]
			}
			if b[1] != nil {
				o.Bounds[i].Hi = *b[1]
			}
		}
	}
	return o
}

// lbSame reports a == b, counting two NaNs as equal.
func lbSame(a, b float64) bool { return a == b || math.IsNaN(a) && math.IsNaN(b) }

// lbMaxDiff is the largest |a_i - b_i|, 0 for two NaNs and +Inf for a NaN
// against a number.
func lbMaxDiff(a, b []float64) float64 {
	d := 0.0
	for i := range a {
		switch {
		case lbSame(a[i], b[i]):
		case math.IsNaN(a[i]) || math.IsNaN(b[i]):
			d = math.Inf(1)
		default:
			d = math.Max(d, math.Abs(a[i]-b[i]))
		}
	}
	return d
}

// lbLongRuns are fixtures whose iterates drift from SciPy's after enough
// iterations: SciPy's wheel computes ddot, dpotrf and dtrtrs with Accelerate
// (blocked, FMA-based kernels) and its C is compiled with FMA contraction,
// while the port uses sequential sums. The port reproduces SciPy's C
// translation bit for bit when both use sequential BLAS without FMA (checked
// with a C harness on these same problems, including rosen100); on a
// 500-iteration run, or one cut by maxfun halfway along the Rosenbrock
// valley, the last-bit differences grow into a different path. For these
// the counts must agree within 5% (converged run) or the status must agree
// and f must have decreased (budget stop).
var lbLongRuns = map[string]bool{"rosen100": true, "rosen100_fd": true}

// lbUnbounded are fixtures whose objective is unbounded below: the run
// stops on the f-reduction test far out along a descent ray, where x is
// only determined by the path, so only the counts and status are compared.
var lbUnbounded = map[string]bool{"poly_formt_fail": true}

func TestLBFGSBFixtures(t *testing.T) {
	for _, c := range lbLoad(t) {
		t.Run(c.Name, func(t *testing.T) {
			var trace []float64
			cx, cfun, ctrace := lbNums(c.X), float64(c.Fun), lbNums(c.Trace)
			r, err := Minimize(c.problem(), c.X0, LBFGSB, c.options(&trace))
			if err != nil {
				t.Fatal(err)
			}
			first := -1
			for i := 0; i < min(len(trace), len(ctrace)); i++ {
				if !lbSame(trace[i], ctrace[i]) {
					first = i
					break
				}
			}
			t.Logf("nit %d/%d nfev %d/%d njev %d/%d (ours/SciPy); |x-x_scipy|=%.2e fun %.17g/%.17g; first differing iterate %d",
				r.Nit, c.Nit, r.Nfev, c.Nfev, r.Njev, c.Njev, lbMaxDiff(r.X, cx), r.Fun, cfun, first)
			if r.Status != c.Status || r.Success != c.Success || r.Message != c.Message {
				t.Errorf("status %d %v %q, SciPy %d %v %q", r.Status, r.Success, r.Message, c.Status, c.Success, c.Message)
			}
			if len(trace) != r.Nit {
				t.Errorf("callback called %d times for %d iterations", len(trace), r.Nit)
			}
			f := c.problem().Func(r.X)
			if !lbSame(f, r.Fun) {
				t.Errorf("Fun %g is not f(X) = %g", r.Fun, f)
			}
			if lbLongRuns[c.Name] {
				if c.Success && math.Abs(float64(r.Nfev-c.Nfev)) > 0.05*float64(c.Nfev) {
					t.Errorf("nfev %d, SciPy %d", r.Nfev, c.Nfev)
				}
				if !c.Success && !(r.Fun < c.problem().Func(c.X0)) {
					t.Errorf("fun %g did not decrease", r.Fun)
				}
				if c.Success && lbMaxDiff(r.X, cx) > 1e-4 {
					t.Errorf("x = %v, SciPy %v", r.X, cx)
				}
				return
			}
			if c.Analytic && (r.Nit != c.Nit || r.Nfev != c.Nfev || r.Njev != c.Njev) {
				t.Errorf("counts differ from SciPy")
			}
			if lbUnbounded[c.Name] {
				return
			}
			if lbMaxDiff(r.X, cx) > 1e-4*(1+lbMaxDiff(cx, make([]float64, len(cx)))) {
				t.Errorf("x = %v, SciPy %v", r.X, cx)
			}
			if !lbSame(r.Fun, cfun) && !(math.Abs(r.Fun-cfun) <= 1e-6*(1+math.Abs(cfun))) && c.Status != 2 {
				t.Errorf("fun = %g, SciPy %g", r.Fun, cfun)
			}
		})
	}
}

func TestLBFGSBInvalid(t *testing.T) {
	p := Problem{Func: rosen, Grad: rosenDer}
	x0 := []float64{-1.2, 1}
	for _, o := range []Options{{M: -1}, {MaxLS: -1}, {Ftol: -1}, {MaxIter: -1}, {MaxFev: -1}} {
		if _, err := Minimize(p, x0, LBFGSB, &o); !errors.Is(err, ErrInput) {
			t.Errorf("%+v: err = %v", o, err)
		}
	}
	// Minimize rejects lb > ub itself; errclb reports it too.
	sf := newScalarFunc(p, LBFGSB, &Options{}, nil, nil)
	if _, err := minimizeLBFGSB(sf, x0, &Options{}, []float64{0, 1}, []float64{1, 0}); !errors.Is(err, ErrBounds) {
		t.Errorf("err = %v", err)
	}
	if err := lbfgsbErrclb(0, 1, 0, nil, nil, nil); !errors.Is(err, ErrInput) {
		t.Errorf("err = %v", err)
	}
}

func TestLBFGSBCallbackStop(t *testing.T) {
	p := Problem{Func: rosen, Grad: rosenDer}
	o := &Options{Callback: func(x []float64, f float64) bool { return true }}
	r, err := Minimize(p, []float64{-1.2, 1}, LBFGSB, o)
	if err != nil {
		t.Fatal(err)
	}
	if r.Nit != 1 || r.Status != 99 || r.Success {
		t.Errorf("%+v", r)
	}
}

func TestLBFGSBCFmaxCFmin(t *testing.T) {
	nan := math.NaN()
	if cfmax(nan, 1) != 1 || cfmax(1, nan) != 1 || cfmax(1, 2) != 2 || cfmax(2, 1) != 2 {
		t.Error("cfmax")
	}
	if cfmin(nan, 1) != 1 || cfmin(1, nan) != 1 || cfmin(1, 2) != 1 || cfmin(2, 1) != 1 {
		t.Error("cfmin")
	}
}

// The reference code guards against a search direction that leaves the box
// from a variable already on its bound, in lnsrlb (stpmx = 0) and in the
// subsm backtrack (alpha = 0). The direction is z - x with z feasible, so
// this happens only through rounding; these tests reach the guards with
// crafted states.
func TestLBFGSBOutwardDirectionGuards(t *testing.T) {
	inf := math.Inf(1)
	l, u := []float64{0, -inf}, []float64{inf, 1}
	s := newLbfgsb(2, 1, l, u, []int{nbdLower, nbdUpper})
	s.cnstnd, s.iter = true, 1
	copy(s.d, []float64{-1, 1})
	x, g := []float64{0, 1}, []float64{1, -1}
	sf := newScalarFunc(Problem{Func: lbSphere, Grad: func(x, g []float64) { g[0], g[1] = 1, -1 }}, LBFGSB, &Options{}, nil, nil)
	if s.lnsrlb(sf, x, g, 1, 1); s.stpmx != 0 {
		t.Errorf("stpmx = %g", s.stpmx)
	}

	for _, tc := range []struct {
		nbd       int
		r, xx, gg float64
	}{{nbdLower, -1, -1, 1}, {nbdUpper, 1, 1, -1}} {
		s := newLbfgsb(1, 1, []float64{0}, []float64{0}, []int{tc.nbd})
		s.col, s.nfree, s.theta = 1, 1, 1
		s.wn[0], s.wn[3] = 1, 1 // identity factor: d stays the reduced gradient
		s.z[0], s.r[0] = 0, tc.r
		s.subsm([]float64{tc.xx}, []float64{tc.gg})
		if s.z[0] != 0 || s.r[0] != 0 {
			t.Errorf("nbd %d: z=%g d=%g", tc.nbd, s.z[0], s.r[0])
		}
	}

	// formk's (1,1) block D + Y'ZZ'Y/θ is positive definite whenever the
	// stored curvatures s'y are positive; a non-positive one fails it.
	s = newLbfgsb(1, 1, []float64{-inf}, []float64{inf}, []int{nbdFree})
	s.col, s.nfree, s.theta, s.ileave = 1, 1, 1, 1
	s.sy[0] = -1
	if s.formk() {
		t.Error("formk succeeded")
	}
}
