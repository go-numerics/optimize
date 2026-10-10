package optimize

import (
	_ "embed"
	"encoding/json"
	"errors"
	"math"
	"strings"
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
	X        []float64      `json:"x"`
	Fun      float64        `json:"fun"`
	Jac      []*float64     `json:"jac"` // null for NaN
	Nit      int            `json:"nit"`
	Nfev     int            `json:"nfev"`
	Njev     int            `json:"njev"`
	Status   int            `json:"status"`
	Success  bool           `json:"success"`
	Message  string         `json:"message"`
	Trace    []float64      `json:"trace"`
}

// Test objectives, the same expressions as testdata/gen/lbfgsb.py.

func lbRosen2(x []float64) float64 {
	return 100*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]) + (1-x[0])*(1-x[0])
}

func lbRosen2G(x, g []float64) {
	g[0] = -400*x[0]*(x[1]-x[0]*x[0]) - 2*(1-x[0])
	g[1] = 200 * (x[1] - x[0]*x[0])
}

func lbExtRosen(x []float64) float64 {
	f := 0.0
	for i := 0; i < len(x); i += 2 {
		t := x[i+1] - x[i]*x[i]
		f += 100*t*t + (1-x[i])*(1-x[i])
	}
	return f
}

func lbExtRosenG(x, g []float64) {
	for i := 0; i < len(x); i += 2 {
		t := x[i+1] - x[i]*x[i]
		g[i] = -400*x[i]*t - 2*(1-x[i])
		g[i+1] = 200 * t
	}
}

func lbBeale(x []float64) float64 {
	a := 1.5 - x[0] + x[0]*x[1]
	b := 2.25 - x[0] + x[0]*x[1]*x[1]
	c := 2.625 - x[0] + x[0]*x[1]*x[1]*x[1]
	return a*a + b*b + c*c
}

func lbBealeG(x, g []float64) {
	a := 1.5 - x[0] + x[0]*x[1]
	b := 2.25 - x[0] + x[0]*x[1]*x[1]
	c := 2.625 - x[0] + x[0]*x[1]*x[1]*x[1]
	g[0] = 2*a*(x[1]-1) + 2*b*(x[1]*x[1]-1) + 2*c*(x[1]*x[1]*x[1]-1)
	g[1] = 2*a*x[0] + 2*b*2*x[0]*x[1] + 2*c*3*x[0]*x[1]*x[1]
}

func lbHimmelblau(x []float64) float64 {
	a := x[0]*x[0] + x[1] - 11
	b := x[0] + x[1]*x[1] - 7
	return a*a + b*b
}

func lbHimmelblauG(x, g []float64) {
	a := x[0]*x[0] + x[1] - 11
	b := x[0] + x[1]*x[1] - 7
	g[0] = 4*a*x[0] + 2*b
	g[1] = 2*a + 4*b*x[1]
}

func lbWood(x []float64) float64 {
	return 100*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]) + (1-x[0])*(1-x[0]) +
		90*(x[3]-x[2]*x[2])*(x[3]-x[2]*x[2]) + (1-x[2])*(1-x[2]) +
		10.1*((x[1]-1)*(x[1]-1)+(x[3]-1)*(x[3]-1)) + 19.8*(x[1]-1)*(x[3]-1)
}

func lbWoodG(x, g []float64) {
	g[0] = -400*x[0]*(x[1]-x[0]*x[0]) - 2*(1-x[0])
	g[1] = 200*(x[1]-x[0]*x[0]) + 20.2*(x[1]-1) + 19.8*(x[3]-1)
	g[2] = -360*x[2]*(x[3]-x[2]*x[2]) - 2*(1-x[2])
	g[3] = 180*(x[3]-x[2]*x[2]) + 20.2*(x[3]-1) + 19.8*(x[1]-1)
}

func lbPowell(x []float64) float64 {
	a := x[0] + 10*x[1]
	b := x[2] - x[3]
	c := x[1] - 2*x[2]
	d := x[0] - x[3]
	return a*a + 5*b*b + c*c*c*c + 10*d*d*d*d
}

func lbPowellG(x, g []float64) {
	a := x[0] + 10*x[1]
	b := x[2] - x[3]
	c := x[1] - 2*x[2]
	d := x[0] - x[3]
	g[0] = 2*a + 40*d*d*d
	g[1] = 20*a + 4*c*c*c
	g[2] = 10*b - 8*c*c*c
	g[3] = -10*b - 40*d*d*d
}

func lbQuadB(i int) float64 {
	s := 1.0
	if i%2 == 1 {
		s = -1
	}
	return s * (1 + 0.1*float64(i))
}

func lbQuadAx(x []float64, i int) float64 {
	ax := float64(i+1) * x[i]
	if i > 0 {
		ax += 0.5 * x[i-1]
	}
	if i+1 < len(x) {
		ax += 0.5 * x[i+1]
	}
	return ax
}

func lbQuad(x []float64) float64 {
	f := 0.0
	for i := range x {
		f += 0.5*x[i]*lbQuadAx(x, i) - lbQuadB(i)*x[i]
	}
	return f
}

func lbQuadG(x, g []float64) {
	for i := range x {
		g[i] = lbQuadAx(x, i) - lbQuadB(i)
	}
}

func lbSphere(x []float64) float64 { return x[0]*x[0] + x[1]*x[1] }

func lbSphereBadG(x, g []float64) {
	g[0], g[1] = -2*x[0], -2*x[1]
}

func lbTiny(x []float64) float64 { return 1e-160 * x[0] }

func lbTinyG(x, g []float64) { g[0] = 1e-160 }

var lbObjectives = map[string]struct {
	f func([]float64) float64
	g func(x, g []float64)
}{
	"rosen2":      {lbRosen2, lbRosen2G},
	"extrosen":    {lbExtRosen, lbExtRosenG},
	"beale":       {lbBeale, lbBealeG},
	"himmelblau":  {lbHimmelblau, lbHimmelblauG},
	"wood":        {lbWood, lbWoodG},
	"powell":      {lbPowell, lbPowellG},
	"quad":        {lbQuad, lbQuadG},
	"sphere_badg": {lbSphere, lbSphereBadG},
	"tiny":        {lbTiny, lbTinyG},
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

func (c *lbCase) problem() Problem {
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

func lbMaxDiff(a, b []float64) float64 {
	d := 0.0
	for i := range a {
		d = math.Max(d, math.Abs(a[i]-b[i]))
	}
	return d
}

func TestLBFGSBFixtures(t *testing.T) {
	for _, c := range lbLoad(t) {
		t.Run(c.Name, func(t *testing.T) {
			var trace []float64
			r, err := Minimize(c.problem(), c.X0, LBFGSB, c.options(&trace))
			if err != nil {
				t.Fatal(err)
			}
			first := -1
			for i := 0; i < min(len(trace), len(c.Trace)); i++ {
				if trace[i] != c.Trace[i] {
					first = i
					break
				}
			}
			t.Logf("nit %d/%d nfev %d/%d njev %d/%d (ours/SciPy); |x-x_scipy|=%.2e fun %.17g/%.17g; first differing iterate %d",
				r.Nit, c.Nit, r.Nfev, c.Nfev, r.Njev, c.Njev, lbMaxDiff(r.X, c.X), r.Fun, c.Fun, first)
			if r.Status != c.Status || r.Success != c.Success || r.Message != c.Message {
				t.Errorf("status %d %v %q, SciPy %d %v %q", r.Status, r.Success, r.Message, c.Status, c.Success, c.Message)
			}
			if len(trace) != r.Nit {
				t.Errorf("callback called %d times for %d iterations", len(trace), r.Nit)
			}
			if c.Analytic && (r.Nit != c.Nit || r.Nfev != c.Nfev || r.Njev != c.Njev) {
				t.Errorf("counts differ from SciPy")
			}
			if lbMaxDiff(r.X, c.X) > 1e-4*(1+lbMaxDiff(c.X, make([]float64, len(c.X)))) {
				t.Errorf("x = %v, SciPy %v", r.X, c.X)
			}
			if math.Abs(r.Fun-c.Fun) > 1e-6*(1+math.Abs(c.Fun)) && c.Status != 2 {
				t.Errorf("fun = %g, SciPy %g", r.Fun, c.Fun)
			}
			f := c.problem().Func(r.X)
			if f != r.Fun {
				t.Errorf("Fun %g is not f(X) = %g", r.Fun, f)
			}
		})
	}
}

func TestLBFGSBInvalid(t *testing.T) {
	p := Problem{Func: lbRosen2, Grad: lbRosen2G}
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
	p := Problem{Func: lbRosen2, Grad: lbRosen2G}
	o := &Options{Callback: func(x []float64, f float64) bool { return true }}
	r, err := Minimize(p, []float64{-1.2, 1}, LBFGSB, o)
	if err != nil {
		t.Fatal(err)
	}
	if r.Nit != 1 || r.Status != 2 || r.Success || r.Message != lbMsgCallback {
		t.Errorf("%+v", r)
	}
	// SciPy lets the iteration limit override the callback's message.
	o.MaxIter = 1
	r, _ = Minimize(p, []float64{-1.2, 1}, LBFGSB, o)
	if r.Status != 1 || !strings.Contains(r.Message, "ITERATIONS") {
		t.Errorf("%+v", r)
	}
}
