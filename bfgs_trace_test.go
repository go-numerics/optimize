package optimize

import (
	"encoding/json"
	"math"
	"os"
	"strconv"
	"testing"
)

// testdata/bfgs_trace.json (testdata/gen/bfgs_trace.py, SciPy 1.18.1 on
// linux/amd64): every evaluation (x, f, g as hex floats) that
// minimize(rosenbrock, [-1.2, 1], jac=True) makes with BFGS and L-BFGS-B.
// BFGS must reproduce the whole trace bit for bit; L-BFGS-B must reproduce
// the iteration and evaluation counts and stay within 1e-6 relative on every
// evaluation, its first differing evaluation being logged (SciPy's BLAS and
// LAPACK kernels sum in orders this port reproduces only in part).
func TestBFGSLBFGSBTraceAgainstSciPy(t *testing.T) {
	b, err := os.ReadFile("testdata/bfgs_trace.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref map[string]struct {
		Ev   [][]string `json:"ev"`
		Fun  string     `json:"fun"`
		Nit  int        `json:"nit"`
		Nfev int        `json:"nfev"`
	}
	if err := json.Unmarshal(b, &ref); err != nil {
		t.Fatal(err)
	}
	hexf := func(s string) float64 {
		v, err := strconv.ParseFloat(s, 64)
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	for _, m := range []Method{BFGS, LBFGSB} {
		want := ref[string(m)]
		var got [][5]float64
		f := func(x, g []float64) float64 {
			v := bhRosenbrock(x, g)
			got = append(got, [5]float64{x[0], x[1], v, g[0], g[1]})
			return v
		}
		r, err := Minimize(Problem{FuncGrad: f}, []float64{-1.2, 1}, m, nil)
		if err != nil {
			t.Fatal(err)
		}
		if r.Nit != want.Nit || r.Nfev != want.Nfev || len(got) != len(want.Ev) {
			t.Fatalf("%s: nit %d nfev %d evals %d, SciPy %d %d %d", m, r.Nit, r.Nfev, len(got), want.Nit, want.Nfev, len(want.Ev))
		}
		first := -1
		for i, e := range want.Ev {
			for k := range 5 {
				w := hexf(e[k])
				if math.Float64bits(got[i][k]) == math.Float64bits(w) {
					continue
				}
				if first < 0 {
					first = i
				}
				if m == BFGS {
					t.Fatalf("BFGS: evaluation %d field %d: %x, SciPy %x", i, k, got[i][k], w)
				}
				if d := math.Abs(got[i][k] - w); d > 1e-6*math.Max(1, math.Abs(w)) {
					t.Fatalf("L-BFGS-B: evaluation %d field %d: %x, SciPy %x", i, k, got[i][k], w)
				}
			}
		}
		t.Logf("%s: %d evaluations, first differing evaluation %d (-1: none), f %v, SciPy %v", m, len(got), first, r.Fun, hexf(want.Fun))
	}
}
