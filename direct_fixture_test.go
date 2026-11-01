package optimize

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"
)

// The fixture of tools/globaloracle (testdata/global/direct.json): SciPy
// 1.18.1's scipy.optimize.direct on the problems of global.py, generated in a
// linux/amd64 container (no fused multiply-add).

type dgHex float64

func (h *dgHex) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	u, err := strconv.ParseUint(strings.TrimPrefix(s, "0x"), 16, 64)
	if err != nil {
		return err
	}
	*h = dgHex(math.Float64frombits(u))
	return nil
}

type dgCase struct {
	Name     string    `json:"name"`
	Problem  string    `json:"problem"`
	Family   string    `json:"family"`
	Dim      int       `json:"dim"`
	Lower    []float64 `json:"lower"`
	Upper    []float64 `json:"upper"`
	Locally  bool      `json:"locally_biased"`
	Options  dgOptions `json:"options"`
	Error    *string   `json:"error"`
	X        []dgHex   `json:"x"`
	Fun      dgHex     `json:"fun"`
	Nfev     int       `json:"nfev"`
	Nit      int       `json:"nit"`
	Status   int       `json:"status"`
	Success  bool      `json:"success"`
	Message  string    `json:"message"`
	Trace    [][]dgHex `json:"trace"`
	Ncallbck int       `json:"ncallbacks"`
}

type dgOptions struct {
	Eps      float64  `json:"eps"`
	Maxfun   int      `json:"maxfun"`
	Maxiter  int      `json:"maxiter"`
	FMin     *float64 `json:"f_min"`
	FMinRtol float64  `json:"f_min_rtol"`
	VolTol   float64  `json:"vol_tol"`
	LenTol   float64  `json:"len_tol"`
}

type dgFixture struct {
	Scipy string   `json:"scipy"`
	Cases []dgCase `json:"cases"`
}

func loadDirectFixture(t testing.TB) *dgFixture {
	t.Helper()
	b, err := os.ReadFile("testdata/global/direct.json")
	if err != nil {
		t.Fatal(err)
	}
	var f dgFixture
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	return &f
}

// dgOptionsOf maps SciPy's keyword arguments to DirectOptions: SciPy's 0
// for the two tolerances means "off", which is a negative value here.
func dgOptionsOf(c dgCase) *DirectOptions {
	off := func(v float64) float64 {
		if v == 0 {
			return -1
		}
		return v
	}
	return &DirectOptions{
		Eps: c.Options.Eps, MaxFev: c.Options.Maxfun, MaxIter: c.Options.Maxiter,
		Original: !c.Locally, FMin: c.Options.FMin, FMinRtol: c.Options.FMinRtol,
		VolTol: off(c.Options.VolTol), LenTol: off(c.Options.LenTol),
	}
}

// sameFloat is bit equality, except that any NaN equals any NaN (the payload
// of a NaN depends on who produced it).
func sameFloat(a, b float64) bool {
	if math.IsNaN(a) && math.IsNaN(b) {
		return true
	}
	return math.Float64bits(a) == math.Float64bits(b)
}

// TestDirectFixture replays every case of the fixture. On the "arith"
// family (products and sums, no libm) the path must be SciPy's exactly:
// status, message, nfev, nit, the bits of x and fun, and the per-iteration
// last-evaluated points. The "trans" family (cos, exp) must agree on the
// status and, where the paths agree, on fun.
func TestDirectFixture(t *testing.T) {
	fx := loadDirectFixture(t)
	if fx.Scipy != "1.18.1" || len(fx.Cases) < 300 {
		t.Fatalf("unexpected fixture: SciPy %s, %d cases", fx.Scipy, len(fx.Cases))
	}
	var exact, arith, transSame, trans int
	var diverged []string
	for _, c := range fx.Cases {
		f := dgProblems[c.Problem]
		if f == nil {
			t.Fatalf("%s: no Go objective for problem %q", c.Name, c.Problem)
		}
		if c.Error != nil {
			t.Fatalf("%s: SciPy raised %s", c.Name, *c.Error)
		}
		bounds := make([]Bound, c.Dim)
		for i := range bounds {
			bounds[i] = Bound{c.Lower[i], c.Upper[i]}
		}
		o := dgOptionsOf(c)
		var trace [][]float64
		ncb := 0
		o.Callback = func(xl, _ []float64, _ float64) bool {
			ncb++
			if len(trace) < 25 {
				trace = append(trace, xl)
			}
			return false
		}
		r, err := Direct(f, bounds, o)
		if err != nil {
			t.Errorf("%s: %v", c.Name, err)
			continue
		}
		same := r.Status == c.Status && r.Nfev == c.Nfev && r.Nit == c.Nit && r.Message == c.Message &&
			r.Success == c.Success && sameFloat(r.Fun, float64(c.Fun)) && ncb == c.Ncallbck
		for i := range c.X {
			same = same && sameFloat(r.X[i], float64(c.X[i]))
		}
		tracesSame := len(trace) == len(c.Trace)
		for i := 0; tracesSame && i < len(trace); i++ {
			for j := range trace[i] {
				tracesSame = tracesSame && sameFloat(trace[i][j], float64(c.Trace[i][j]))
			}
		}
		same = same && tracesSame
		switch c.Family {
		case "arith":
			arith++
			if same {
				exact++
			} else {
				t.Errorf("%s: status %d/%d nfev %d/%d nit %d/%d fun %v/%v (Go/SciPy), traces equal %v",
					c.Name, r.Status, c.Status, r.Nfev, c.Nfev, r.Nit, c.Nit, r.Fun, float64(c.Fun), tracesSame)
			}
		default:
			trans++
			if same {
				exact++
				transSame++
			} else {
				diverged = append(diverged, fmt.Sprintf("%s: status %d/%d nfev %d/%d fun %v/%v",
					c.Name, r.Status, c.Status, r.Nfev, c.Nfev, r.Fun, float64(c.Fun)))
				if r.Status != c.Status {
					t.Errorf("%s: status %d, SciPy %d", c.Name, r.Status, c.Status)
				}
				if r.Nfev == c.Nfev && r.Nit == c.Nit &&
					math.Abs(r.Fun-float64(c.Fun)) > 1e-10*math.Max(1, math.Abs(float64(c.Fun))) {
					t.Errorf("%s: same path but fun %v, SciPy %v", c.Name, r.Fun, float64(c.Fun))
				}
				if c.Problem == "docstring_pow" {
					if r.Nfev != c.Nfev || math.Abs(r.Fun-float64(c.Fun)) > 1e-12 {
						t.Errorf("%s: the documented SciPy example must agree: %d/%d %v/%v",
							c.Name, r.Nfev, c.Nfev, r.Fun, float64(c.Fun))
					}
				}
			}
		}
	}
	t.Logf("exact path: %d/%d cases (arith %d/%d, trans %d/%d)", exact, len(fx.Cases), exact-transSame, arith, transSame, trans)
	for _, d := range diverged {
		t.Logf("trans case off the exact path: %s", d)
	}
}
