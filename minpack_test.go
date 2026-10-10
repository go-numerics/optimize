package optimize

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"math"
	"testing"
)

//go:embed testdata/minpack.json
var minpackJSON []byte

type mpRootCase struct {
	Problem    string         `json:"problem"`
	Jac        bool           `json:"jac"`
	Options    map[string]any `json:"options"`
	X          []float64      `json:"x"`
	Fun        []float64      `json:"fun"`
	Fjac       []float64      `json:"fjac"`
	R          []float64      `json:"r"`
	Qtf        []float64      `json:"qtf"`
	Nfev       int            `json:"nfev"`
	Njev       int            `json:"njev"`
	Status     int            `json:"status"`
	Success    bool           `json:"success"`
	Message    string         `json:"message"`
	FsolveNfev int            `json:"fsolve_nfev"`
	FsolveIer  int            `json:"fsolve_ier"`
	FsolveMesg string         `json:"fsolve_mesg"`
	FsolveFjac []float64      `json:"fsolve_fjac"`
}

type mpLMCase struct {
	Problem    string         `json:"problem"`
	Jac        bool           `json:"jac"`
	Options    map[string]any `json:"options"`
	X          []float64      `json:"x"`
	Cost       float64        `json:"cost"`
	Fun        []float64      `json:"fun"`
	Grad       []float64      `json:"grad"`
	Jacobian   []float64      `json:"jacobian"`
	Optimality float64        `json:"optimality"`
	Nfev       int            `json:"nfev"`
	Njev       int            `json:"njev"`
	Status     int            `json:"status"`
	Success    bool           `json:"success"`
	Message    string         `json:"message"`
}

type mpLeastsqCase struct {
	Problem string         `json:"problem"`
	Jac     bool           `json:"jac"`
	Options map[string]any `json:"options"`
	X       []float64      `json:"x"`
	CovX    []float64      `json:"cov_x"`
	Nfev    int            `json:"nfev"`
	Njev    int            `json:"njev"`
	Fvec    []float64      `json:"fvec"`
	Fjac    []float64      `json:"fjac"`
	Ipvt    []int          `json:"ipvt"`
	Qtf     []float64      `json:"qtf"`
	Ier     int            `json:"ier"`
	Mesg    string         `json:"mesg"`
}

type mpFixture struct {
	Scipy        string          `json:"scipy"`
	Root         []mpRootCase    `json:"root"`
	LeastSquares []mpLMCase      `json:"least_squares"`
	Leastsq      []mpLeastsqCase `json:"leastsq"`
}

func loadMinpack(t *testing.T) *mpFixture {
	t.Helper()
	var f mpFixture
	if err := json.Unmarshal(minpackJSON, &f); err != nil {
		t.Fatal(err)
	}
	return &f
}

func optFloat(o map[string]any, k string) float64 {
	if v, ok := o[k]; ok {
		return v.(float64)
	}
	return 0
}

func optVec(o map[string]any, k string) []float64 {
	v, ok := o[k]
	if !ok {
		return nil
	}
	var out []float64
	for _, e := range v.([]any) {
		out = append(out, e.(float64))
	}
	return out
}

// relDiff is the largest |a-b| / max(1, |b|) over the entries.
func relDiff(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	d := 0.0
	for i := range a {
		d = math.Max(d, math.Abs(a[i]-b[i])/math.Max(1, math.Abs(b[i])))
	}
	return d
}

func caseName(p string, jac bool, o map[string]any) string {
	s := p + "/fd"
	if jac {
		s = p + "/jac"
	}
	if len(o) > 0 {
		s += fmt.Sprintf("/%v", o)
	}
	return s
}

// mpLibm marks the problems whose residuals call exp, sin, cos or atan. Go's math
// package and the platform libm that the fixture's Python used round some of
// those results differently in the last bit, so from the first such
// evaluation on the iterates drift apart by rounding; on every other problem
// the port reproduces SciPy's (FMA-free x86_64 build) iterates to the last
// bit. mpMultiMin marks a problem with several exact minimisers, which the
// two runs may then reach differently.
var (
	mpLibm = map[string]bool{"trigonometric": true, "meyer": true, "box3d": true, "jennrich_sampson": true,
		"brown_dennis": true, "osborne1": true, "biggs_exp6": true, "osborne2": true, "powell_badly_scaled": true, "helical_valley": true}
	mpMultiMin = map[string]bool{"biggs_exp6": true}
)

// mpLooseTol is the tolerance on x for a problem with libm noise: Brown and
// Dennis has large residuals and a flat minimum, so the forward-difference
// runs stop (on ftol) at points 1e-4 apart.
func mpLooseTol(problem string) float64 {
	if problem == "brown_dennis" {
		return 1e-3
	}
	return 1e-4
}

// mpCheck compares one result with its fixture: to the last bits on the
// exact problems, and to the solution's accuracy on the others.
type mpCheck struct {
	t     *testing.T
	exact bool
}

func (c mpCheck) vec(name string, got, want []float64, loose float64) {
	c.t.Helper()
	tol := 1e-12
	if !c.exact {
		if loose < 0 {
			return
		}
		tol = loose
	}
	if d := relDiff(got, want); d > tol {
		c.t.Errorf("%s differs from SciPy by %.3g (> %g)\n got %v\nwant %v", name, d, tol, got, want)
	}
}

func (c mpCheck) counts(nfev, njev, wantNfev, wantNjev int, same *int) {
	c.t.Helper()
	if nfev == wantNfev && njev == wantNjev {
		*same++
		return
	}
	if c.exact {
		c.t.Errorf("counts nfev %d njev %d, SciPy %d %d", nfev, njev, wantNfev, wantNjev)
	}
}

func TestRootFixtures(t *testing.T) {
	fx := loadMinpack(t)
	probs := mpProblems()
	same := 0
	for _, c := range fx.Root {
		t.Run(caseName(c.Problem, c.Jac, c.Options), func(t *testing.T) {
			p := probs[c.Problem]
			sp := SystemProblem{Func: p.f}
			if c.Jac {
				sp.Jac = p.j
			}
			opts := &RootOptions{Xtol: optFloat(c.Options, "xtol"), Maxfev: int(optFloat(c.Options, "maxfev")),
				Eps: optFloat(c.Options, "eps"), Factor: optFloat(c.Options, "factor"), Diag: optVec(c.Options, "diag")}
			r, err := Root(sp, p.x0, opts)
			if err != nil {
				t.Fatal(err)
			}
			// SciPy counts two calls that its wrappers make outside MINPACK
			// (a shape check and the initial evaluation).
			t.Logf("nfev %d (SciPy %d-2)  njev %d (SciPy %d)  status %d (SciPy %d)  |dx| %.3g",
				r.Nfev, c.Nfev, r.Njev, c.Njev, r.Status, c.Status, relDiff(r.X, c.X))
			if r.Status != c.Status || r.Success != c.Success || r.Message != c.Message {
				t.Errorf("status %d %v %q, SciPy %d %v %q", r.Status, r.Success, r.Message, c.Status, c.Success, c.Message)
			}
			ck := mpCheck{t, !mpLibm[c.Problem]}
			ck.counts(r.Nfev+2, r.Njev, c.Nfev, c.Njev, &same)
			loose := -1.0 // an unconverged run is compared only where exact
			if c.Success {
				loose = 1e-8
			}
			ck.vec("x", r.X, c.X, loose)
			ck.vec("fun", r.Fun, c.Fun, loose)
			ck.vec("fjac", r.Fjac, c.Fjac, -1)
			ck.vec("r", r.R, c.R, -1)
			ck.vec("qtf", r.Qtf, c.Qtf, -1)
			f, err := Fsolve(sp, p.x0, opts)
			if err != nil || f.Status != c.FsolveIer || f.Message != c.FsolveMesg || relDiff(f.X, r.X) != 0 {
				t.Errorf("Fsolve: %v %d %q; SciPy %d %q", err, f.Status, f.Message, c.FsolveIer, c.FsolveMesg)
			}
			ck.vec("fsolve fjac", f.Fjac, c.FsolveFjac, -1)
			if ck.exact && f.Nfev+2 != c.FsolveNfev {
				t.Errorf("Fsolve nfev %d, SciPy %d-2", f.Nfev, c.FsolveNfev)
			}
		})
	}
	t.Logf("Root: %d/%d cases with SciPy's exact nfev/njev", same, len(fx.Root))
}

func TestLeastSquaresLMFixtures(t *testing.T) {
	fx := loadMinpack(t)
	probs := mpProblems()
	same := 0
	for _, c := range fx.LeastSquares {
		t.Run(caseName(c.Problem, c.Jac, c.Options), func(t *testing.T) {
			p := probs[c.Problem]
			lp := LSProblem{Func: p.f, M: p.m}
			if c.Jac {
				lp.Jac = p.j
			}
			opts := &LSOptions{Ftol: optFloat(c.Options, "ftol"), Xtol: optFloat(c.Options, "xtol"),
				Gtol: optFloat(c.Options, "gtol"), MaxNfev: int(optFloat(c.Options, "max_nfev")),
				DiffStep: optFloat(c.Options, "diff_step"), XScale: optVec(c.Options, "x_scale")}
			r, err := LeastSquares(lp, p.x0, LM, opts)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("nfev %d (SciPy %d)  njev %d (SciPy %d)  status %d (SciPy %d)  cost %.10g (SciPy %.10g)  |dx| %.3g",
				r.Nfev, c.Nfev, r.Njev, c.Njev, r.Status, c.Status, r.Cost, c.Cost, relDiff(r.X, c.X))
			ck := mpCheck{t, !mpLibm[c.Problem]}
			if r.Success != c.Success || (ck.exact && (r.Status != c.Status || r.Message != c.Message)) {
				t.Errorf("status %d %v %q, SciPy %d %v %q", r.Status, r.Success, r.Message, c.Status, c.Success, c.Message)
			}
			ck.counts(r.Nfev, r.Njev, c.Nfev, c.Njev, &same)
			if math.Abs(r.Cost-c.Cost) > 1e-7*math.Max(1, c.Cost) {
				t.Errorf("cost %v, SciPy %v", r.Cost, c.Cost)
			}
			loose := mpLooseTol(c.Problem)
			if !c.Success || mpMultiMin[c.Problem] {
				loose = -1
			}
			ck.vec("x", r.X, c.X, loose)
			ck.vec("fun", r.Fun, c.Fun, loose)
			ck.vec("jac", r.Jac, c.Jacobian, loose)
			ck.vec("grad", r.Grad, c.Grad, -1)
			if ck.exact && math.Abs(r.Optimality-c.Optimality) > 1e-12*math.Max(1, c.Optimality) {
				t.Errorf("optimality %v, SciPy %v", r.Optimality, c.Optimality)
			}
			if fmt.Sprint(r.ActiveMask) != fmt.Sprint(make([]int, len(r.X))) {
				t.Errorf("active mask %v", r.ActiveMask)
			}
		})
	}
	t.Logf("least_squares lm: %d/%d cases with SciPy's exact nfev/njev", same, len(fx.LeastSquares))
}

func TestLeastsqFixtures(t *testing.T) {
	fx := loadMinpack(t)
	probs := mpProblems()
	same := 0
	for _, c := range fx.Leastsq {
		t.Run(caseName(c.Problem, c.Jac, c.Options), func(t *testing.T) {
			p := probs[c.Problem]
			lp := LSProblem{Func: p.f, M: p.m}
			if c.Jac {
				lp.Jac = p.j
			}
			opts := &LeastsqOptions{Ftol: optFloat(c.Options, "ftol"), Xtol: optFloat(c.Options, "xtol"),
				Gtol: optFloat(c.Options, "gtol"), Maxfev: int(optFloat(c.Options, "maxfev")),
				Epsfcn: optFloat(c.Options, "epsfcn"), Factor: optFloat(c.Options, "factor"), Diag: optVec(c.Options, "diag")}
			// An explicit 0 tolerance in the fixture is a negative one here.
			if v, ok := c.Options["ftol"]; ok && v.(float64) == 0 {
				opts.Ftol = -1
			}
			if v, ok := c.Options["xtol"]; ok && v.(float64) == 0 {
				opts.Xtol = -1
			}
			r, err := Leastsq(lp, p.x0, opts)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("nfev %d (SciPy %d)  njev %d (SciPy %d)  ier %d (SciPy %d)  |dx| %.3g",
				r.Nfev, c.Nfev, r.Njev, c.Njev, r.Ier, c.Ier, relDiff(r.X, c.X))
			ck := mpCheck{t, !mpLibm[c.Problem]}
			success := c.Ier >= 1 && c.Ier <= 4
			if r.Success() != success || (ck.exact && (r.Ier != c.Ier || r.Mesg != c.Mesg)) {
				t.Errorf("ier %d %q, SciPy %d %q", r.Ier, r.Mesg, c.Ier, c.Mesg)
			}
			ck.counts(r.Nfev, r.Njev, c.Nfev, c.Njev, &same)
			loose := mpLooseTol(c.Problem)
			if !success || mpMultiMin[c.Problem] {
				loose = -1
			}
			ck.vec("x", r.X, c.X, loose)
			ck.vec("fvec", r.Fvec, c.Fvec, loose)
			if (r.CovX == nil) != (c.CovX == nil) {
				t.Errorf("cov_x nil %v, SciPy nil %v", r.CovX == nil, c.CovX == nil)
			} else if c.CovX != nil {
				tol := 1e-12
				if !ck.exact {
					tol = 1e-2
				}
				if d := relCov(r.CovX, c.CovX); d > tol && loose > 0 {
					t.Errorf("cov_x differs by %.3g", d)
				}
			}
			if ck.exact && fmt.Sprint(r.Ipvt) != fmt.Sprint(c.Ipvt) {
				t.Errorf("ipvt %v, SciPy %v", r.Ipvt, c.Ipvt)
			}
			ck.vec("fjac", r.Fjac, c.Fjac, -1)
			ck.vec("qtf", r.Qtf, c.Qtf, -1)
		})
	}
	t.Logf("leastsq: %d/%d cases with SciPy's exact nfev/njev", same, len(fx.Leastsq))
}

// relCov compares covariance matrices relative to their largest entry.
func relCov(a, b []float64) float64 {
	s := 0.0
	for _, v := range b {
		s = math.Max(s, math.Abs(v))
	}
	d := 0.0
	for i := range a {
		d = math.Max(d, math.Abs(a[i]-b[i]))
	}
	return d / s
}
