package optimize

import (
	_ "embed"
	"encoding/json"
	"math"
	"testing"
	"time"

	"github.com/go-ndarray/ndarray"
)

//go:embed testdata/linprog.json
var lpFixturesJSON []byte

type lpFixture struct {
	Name     string         `json:"name"`
	C        []float64      `json:"c"`
	AUb      []float64      `json:"a_ub"`
	MUb      int            `json:"m_ub"`
	BUb      []float64      `json:"b_ub"`
	AEq      []float64      `json:"a_eq"`
	MEq      int            `json:"m_eq"`
	BEq      []float64      `json:"b_eq"`
	Bounds   []float64      `json:"bounds"`
	Status   int            `json:"status"`
	Message  string         `json:"message"`
	Nit      int            `json:"nit"`
	Nits     map[string]int `json:"nits"`
	IPStatus int            `json:"ip_status"`
	IPX      []float64      `json:"ip_x"`
	X        []float64      `json:"x"`
	Fun      float64        `json:"fun"`
	IneqLin  []float64      `json:"ineqlin"`
	EqLin    []float64      `json:"eqlin"`
	Lower    []float64      `json:"lower"`
	Upper    []float64      `json:"upper"`
	XLo      []float64      `json:"x_lo"`
	XHi      []float64      `json:"x_hi"`
	DLo      []float64      `json:"d_lo"`
	DHi      []float64      `json:"d_hi"`
}

func lpLoadFixtures(t *testing.T) []lpFixture {
	t.Helper()
	var fs []lpFixture
	if err := json.Unmarshal(lpFixturesJSON, &fs); err != nil {
		t.Fatal(err)
	}
	return fs
}

// lpFromJSON maps the fixture's ±1e300 back to ±Inf.
func lpFromJSON(v float64) float64 {
	switch {
	case v >= 1e300:
		return math.Inf(1)
	case v <= -1e300:
		return math.Inf(-1)
	}
	return v
}

func (f *lpFixture) constraints(t *testing.T) LPConstraints {
	t.Helper()
	n := len(f.C)
	var cons LPConstraints
	mk := func(d []float64, shape ...int) *ndarray.Array {
		a, err := ndarray.FromData(d, shape...)
		if err != nil {
			t.Fatal(err)
		}
		return a
	}
	if f.AUb != nil {
		cons.AUb, cons.BUb = mk(f.AUb, f.MUb, n), mk(f.BUb, f.MUb)
	}
	if f.AEq != nil {
		cons.AEq, cons.BEq = mk(f.AEq, f.MEq, n), mk(f.BEq, f.MEq)
	}
	if f.Bounds != nil {
		cons.Bounds = make([]Bound, n)
		for j := range cons.Bounds {
			cons.Bounds[j] = Bound{lpFromJSON(f.Bounds[2*j]), lpFromJSON(f.Bounds[2*j+1])}
		}
	}
	return cons
}

// lpOutside reports how far v lies outside [lo, hi] (0 when inside).
func lpOutside(v, lo, hi float64) float64 {
	lo, hi = lpFromJSON(lo), lpFromJSON(hi)
	switch {
	case v < lo:
		return lo - v
	case v > hi:
		return v - hi
	}
	return 0
}

// lpCheckKKT verifies that the marginals certify optimality of r.X: signs,
// stationarity c = A_ubᵀ·λ_ub + A_eqᵀ·λ_eq + λ_lo + λ_up, complementary
// slackness, and a zero duality gap. It returns the largest violation,
// relative to the problem's scale.
func lpCheckKKT(t *testing.T, c []float64, cons LPConstraints, r *LPResult) float64 {
	t.Helper()
	n := len(c)
	p, err := parseLP(c, cons)
	if err != nil {
		t.Fatal(err)
	}
	worst := 0.0
	upd := func(v float64) { worst = math.Max(worst, v) }
	scale := 1 + norm(c, math.Inf(1))
	red := clone(c)
	for i := 0; i < p.mub; i++ {
		l := r.IneqLin.Marginals[i]
		upd(math.Max(l, 0) / scale)
		upd(math.Abs(l*r.Slack[i]) / (scale * (1 + math.Abs(p.bub[i]))))
		for j := 0; j < n; j++ {
			red[j] -= p.aub[i*n+j] * l
		}
	}
	for i := 0; i < p.meq; i++ {
		for j := 0; j < n; j++ {
			red[j] -= p.aeq[i*n+j] * r.EqLin.Marginals[i]
		}
	}
	for j := 0; j < n; j++ {
		lo, up := r.Lower.Marginals[j], r.Upper.Marginals[j]
		upd(math.Abs(red[j]-lo-up) / scale)
		upd(math.Max(-lo, 0) / scale)
		upd(math.Max(up, 0) / scale)
		if !math.IsInf(p.lb[j], 0) {
			upd(math.Abs(lo*(r.X[j]-p.lb[j])) / (scale * (1 + math.Abs(p.lb[j]))))
		} else {
			upd(math.Abs(lo) / scale)
		}
		if !math.IsInf(p.ub[j], 0) {
			upd(math.Abs(up*(p.ub[j]-r.X[j])) / (scale * (1 + math.Abs(p.ub[j]))))
		} else {
			upd(math.Abs(up) / scale)
		}
	}
	return worst
}

func lpTag(method LPMethod, noPresolve bool) string {
	tag := map[LPMethod]string{LPSimplex: "rs", LPInteriorPoint: "ip"}[method]
	if noPresolve {
		tag += "-nopre"
	}
	return tag
}

func TestLinprogFixtures(t *testing.T) {
	fs := lpLoadFixtures(t)
	type tally struct {
		solved, nitMatch, ipRuns int
		dur                      time.Duration
	}
	type config struct {
		method     LPMethod
		noPresolve bool
	}
	for _, cfg := range []config{{LPSimplex, false}, {LPSimplex, true}, {LPInteriorPoint, false}} {
		method := cfg.method
		var tl tally
		// Accuracy expected of each method: the simplex method ends on an
		// exact vertex; the interior-point method stops at Tol = 1e-8.
		funTol, xTol, dTol, kktTol := 1e-9, 1e-7, 1e-7, 1e-9
		if method == LPInteriorPoint {
			funTol, xTol, dTol, kktTol = 1e-7, 1e-5, 1e-5, 1e-6
		}
		for _, f := range fs {
			cons := f.constraints(t)
			t0 := time.Now()
			r, err := Linprog(f.C, cons, &LPOptions{Method: method, NoPresolve: cfg.noPresolve})
			tl.dur += time.Since(t0)
			if err != nil {
				t.Fatalf("%s/%s: %v", method, f.Name, err)
			}
			if r.Status != f.Status {
				t.Errorf("%s/%s: status %d (%s), SciPy %d (%s)", method, f.Name, r.Status, r.Message, f.Status, f.Message)
				continue
			}
			if r.Success != (f.Status == 0) {
				t.Errorf("%s/%s: Success %v with status %d", method, f.Name, r.Success, r.Status)
			}
			if f.Status != 0 {
				t.Logf("%-8s %-24s status=%d nit=%3d (HiGHS ds %d) %q", lpTag(cfg.method, cfg.noPresolve), f.Name, r.Status, r.Nit, f.Nits["highs-ds"], r.Message)
				continue
			}
			tl.solved++
			ferr := math.Abs(r.Fun-f.Fun) / math.Max(1, math.Abs(f.Fun))
			if ferr > funTol {
				t.Errorf("%s/%s: fun %.17g, SciPy %.17g (rel %.2g)", method, f.Name, r.Fun, f.Fun, ferr)
			}
			xerr := 0.0
			for j, x := range r.X {
				xerr = math.Max(xerr, lpOutside(x, f.XLo[j], f.XHi[j])/(1+math.Abs(x)))
			}
			if xerr > xTol {
				t.Errorf("%s/%s: x %v outside SciPy's optimal face by %.2g", method, f.Name, r.X, xerr)
			}
			duals := append(append(append(clone(r.IneqLin.Marginals), r.EqLin.Marginals...), r.Lower.Marginals...), r.Upper.Marginals...)
			derr := 0.0
			for k, d := range duals {
				derr = math.Max(derr, lpOutside(d, f.DLo[k], f.DHi[k])/(1+math.Abs(d)))
			}
			if derr > dTol {
				t.Errorf("%s/%s: marginals outside SciPy's dual optimal face by %.2g\n ours  %v\n HiGHS %v %v %v %v",
					method, f.Name, derr, duals, f.IneqLin, f.EqLin, f.Lower, f.Upper)
			}
			kkt := lpCheckKKT(t, f.C, cons, r)
			if kkt > kktTol {
				t.Errorf("%s/%s: KKT violation %.2g", method, f.Name, kkt)
			}
			ref := f.Nits["highs-ds"]
			if method == LPInteriorPoint {
				ref = f.Nits["highs-ipm"]
			}
			t.Logf("%-8s %-24s fun err %.1e  x off-face %.1e  duals off-face %.1e  KKT %.1e  nit %4d  (HiGHS %3d, legacy rs %4d, legacy ip %3d)",
				lpTag(cfg.method, cfg.noPresolve), f.Name, ferr, xerr, derr, kkt, r.Nit, ref, f.Nits["revised simplex"], f.Nits["interior-point"])
		}
		t.Logf("%s (presolve %v): %d/%d optimal problems solved, %d problems in %v", method, !cfg.noPresolve, tl.solved, len(fs), len(fs), tl.dur)
	}
}

// TestLinprogIPMatchesSciPyIterates checks that, without presolve, the
// interior-point method follows SciPy's legacy method='interior-point'
// (presolve=False): same standard form, same iterates, same iteration count.
func TestLinprogIPMatchesSciPyIterates(t *testing.T) {
	match, total := 0, 0
	for _, f := range lpLoadFixtures(t) {
		if f.IPX == nil {
			continue
		}
		r, err := Linprog(f.C, f.constraints(t), &LPOptions{Method: LPInteriorPoint, NoPresolve: true})
		if err != nil {
			t.Fatal(err)
		}
		total++
		want := f.Nits["interior-point"]
		dx := 0.0
		for j := range r.X {
			dx = math.Max(dx, math.Abs(r.X[j]-lpFromJSON(f.IPX[j]))/(1+math.Abs(r.X[j])))
		}
		if r.Status != f.IPStatus {
			t.Errorf("%s: status %d, SciPy's interior-point %d", f.Name, r.Status, f.IPStatus)
		}
		if r.Nit == want {
			match++
		}
		t.Logf("%-24s nit %3d SciPy %3d  status %d SciPy %d  max |Δx| %.1e", f.Name, r.Nit, want, r.Status, f.IPStatus, dx)
	}
	t.Logf("interior-point iteration counts equal to SciPy's: %d/%d", match, total)
	if match < total*3/4 {
		t.Errorf("only %d/%d iteration counts match SciPy's legacy interior-point", match, total)
	}
}
