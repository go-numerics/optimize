package optimize

import (
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/go-ndarray/ndarray"
)

func lpMat(t *testing.T, d []float64, shape ...int) *ndarray.Array {
	t.Helper()
	a, err := ndarray.FromData(d, shape...)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func TestLinprogInputErrors(t *testing.T) {
	a := lpMat(t, []float64{1, 1}, 1, 2)
	b := lpMat(t, []float64{1}, 1)
	nan := math.NaN()
	cases := []struct {
		name string
		c    []float64
		cons LPConstraints
		opts *LPOptions
		want error
	}{
		{"method", []float64{1, 1}, LPConstraints{}, &LPOptions{Method: "highs"}, ErrMethod},
		{"empty c", nil, LPConstraints{}, nil, ErrInput},
		{"nan c", []float64{nan, 1}, LPConstraints{}, nil, ErrInput},
		{"A without b", []float64{1, 1}, LPConstraints{AUb: a}, nil, ErrDimension},
		{"A 1-D", []float64{1, 1}, LPConstraints{AEq: lpMat(t, []float64{1, 1}, 2), BEq: b}, nil, ErrDimension},
		{"A columns", []float64{1, 1, 1}, LPConstraints{AUb: a, BUb: b}, nil, ErrDimension},
		{"b length", []float64{1, 1}, LPConstraints{AUb: a, BUb: lpMat(t, []float64{1, 2}, 2)}, nil, ErrDimension},
		{"inf b", []float64{1, 1}, LPConstraints{AUb: a, BUb: lpMat(t, []float64{math.Inf(1)}, 1)}, nil, ErrInput},
		{"bounds length", []float64{1, 1}, LPConstraints{Bounds: []Bound{{0, 1}}}, nil, ErrDimension},
		{"nan bound", []float64{1, 1}, LPConstraints{Bounds: []Bound{{0, 1}, {nan, 1}}}, nil, ErrBounds},
	}
	for _, tc := range cases {
		if _, err := Linprog(tc.c, tc.cons, tc.opts); !errors.Is(err, tc.want) {
			t.Errorf("%s: error %v, want %v", tc.name, err, tc.want)
		}
	}
}

func TestLinprogOptionsAndResult(t *testing.T) {
	// The docs example, with integer (non-float64) arrays and explicit
	// tolerances.
	a := lpMat(t, []float64{-3, 1, 1, 2}, 2, 2).AsType(ndarray.Int64)
	b := lpMat(t, []float64{6, 4}, 2).AsType(ndarray.Int64)
	cons := LPConstraints{AUb: a, BUb: b, Bounds: []Bound{Unbounded(), {-3, math.Inf(1)}}}
	for _, m := range []LPMethod{LPSimplex, LPInteriorPoint} {
		r, err := Linprog([]float64{-1, 4}, cons, &LPOptions{Method: m, Tol: 1e-10, MaxIter: 50})
		if err != nil {
			t.Fatal(err)
		}
		x := r.XArray().Shape()
		if !r.Success || math.Abs(r.Fun+22) > 1e-7 || len(x) != 1 || x[0] != 2 {
			t.Errorf("%s: %+v", m, r)
		}
		// HiGHS: ineqlin marginals [-0, -1], lower [0, 6], upper [0, 0].
		if math.Abs(r.IneqLin.Marginals[1]+1) > 1e-7 || math.Abs(r.Lower.Marginals[1]-6) > 1e-7 {
			t.Errorf("%s: marginals %+v %+v", m, r.IneqLin, r.Lower)
		}
	}
	var r LPResult
	if r.XArray() != nil {
		t.Error("XArray of nil X")
	}
}

func TestLinprogIterationLimit(t *testing.T) {
	// Klee–Minty n = 6 needs 63 Dantzig pivots in phase two.
	n := 6
	c := make([]float64, n)
	a := make([]float64, n*n)
	b := make([]float64, n)
	for i := 0; i < n; i++ {
		c[i] = -math.Pow(2, float64(n-1-i))
		b[i] = math.Pow(5, float64(i+1))
		for j := 0; j < i; j++ {
			a[i*n+j] = math.Pow(2, float64(i-j+1))
		}
		a[i*n+i] = 1
	}
	km := LPConstraints{AUb: lpMat(t, a, n, n), BUb: lpMat(t, b, n)}
	// An equality system needs phase-one pivots.
	eq := LPConstraints{AEq: lpMat(t, []float64{1, 1, 1, 1, 2, 3}, 2, 3), BEq: lpMat(t, []float64{3, 6}, 2)}
	for _, tc := range []struct {
		name string
		cons LPConstraints
		opts LPOptions
	}{
		{"simplex phase two", km, LPOptions{MaxIter: 10}},
		{"simplex phase one", eq, LPOptions{MaxIter: 1}},
		{"interior point", km, LPOptions{Method: LPInteriorPoint, MaxIter: 2}},
	} {
		cc := c
		if tc.cons.AEq != nil {
			cc = []float64{1, 1, 1}
		}
		r, err := Linprog(cc, tc.cons, &tc.opts)
		if err != nil {
			t.Fatal(err)
		}
		if r.Status != LPIterLimit || r.Success || lpHasMarginals(r) {
			t.Errorf("%s: status %d %q", tc.name, r.Status, r.Message)
		}
	}
}

// lpHasMarginals reports whether any row marginals were set.
func lpHasMarginals(r *LPResult) bool { return r.IneqLin.Marginals != nil || r.EqLin.Marginals != nil }

func TestLinprogCheckResult(t *testing.T) {
	// Equalities inconsistent by 1e-6: without presolve the simplex method
	// reports infeasibility, but the point it holds satisfies the
	// constraints within 10·√Tol, so the result is downgraded to status 4
	// as SciPy's _check_result does.
	cons := LPConstraints{AEq: lpMat(t, []float64{1, 1, 1, 1}, 2, 2), BEq: lpMat(t, []float64{1, 1 + 1e-6}, 2)}
	r, err := Linprog([]float64{1, 2}, cons, &LPOptions{NoPresolve: true})
	if err != nil {
		t.Fatal(err)
	}
	if r.Status != LPNumericalFail || !strings.Contains(r.Message, "feasible, but") {
		t.Errorf("status %d %q", r.Status, r.Message)
	}
	// With a loose tolerance, phase one accepts a residual of 5 (below
	// Tol·(1+‖b‖∞) = 10.6) that the final audit (10·√Tol = 3.2) rejects.
	cons = LPConstraints{AEq: lpMat(t, []float64{1, 1, 1, 1}, 2, 2), BEq: lpMat(t, []float64{100, 105}, 2)}
	r, err = Linprog([]float64{1, 2}, cons, &LPOptions{Tol: 0.1, NoPresolve: true})
	if err != nil {
		t.Fatal(err)
	}
	if r.Status != LPNumericalFail || !strings.Contains(r.Message, "does not satisfy") {
		t.Errorf("status %d %q %v", r.Status, r.Message, r.Con)
	}
}

func TestLinprogSingularBasis(t *testing.T) {
	var f lpFactor
	if f.factor([]float64{1, 2, 2, 4}, 2, 1e-11) {
		t.Error("singular matrix factored")
	}
	p, err := parseLP([]float64{1, 1}, LPConstraints{AUb: lpMat(t, []float64{1, 2, 2, 4}, 2, 2), BUb: lpMat(t, []float64{1, 1}, 2)})
	if err != nil {
		t.Fatal(err)
	}
	s := newSimplex(p, 1e-9)
	s.head = []int{0, 1} // dependent columns
	if st := s.run(10); st != LPNumericalFail {
		t.Errorf("status %d", st)
	}
	if !strings.Contains(lpSimplexMessages[LPNumericalFail], "singular") {
		t.Error(lpSimplexMessages[LPNumericalFail])
	}
}
