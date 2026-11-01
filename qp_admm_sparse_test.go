package optimize

import (
	"math"
	"testing"

	"github.com/go-sparse/sparse"
)

// denseToCSC stores the nonzeros of a dense matrix (rows of cols values) as a
// CSC, keeping only the upper triangle when upper is set.
func denseToCSC(t testing.TB, rows, cols int, d [][]float64, upper bool) *sparse.CSC[float64] {
	t.Helper()
	ptr := []int{0}
	var idx []int32
	var val []float64
	for j := 0; j < cols; j++ {
		for i := 0; i < rows; i++ {
			if upper && i > j {
				continue
			}
			if v := d[i][j]; v != 0 {
				idx, val = append(idx, int32(i)), append(val, v)
			}
		}
		ptr = append(ptr, len(idx))
	}
	return qpCSC(t, rows, cols, ptr, idx, val)
}

// sparseFixtureProblem builds a case with PSparse and ASparse (bounds as L, U).
func sparseFixtureProblem(t testing.TB, c qpFixFull) QPProblem {
	n := len(c.Q)
	clamp := func(v []float64) []float64 {
		o := clone(v)
		for i := range o {
			if math.Abs(o[i]) >= 1e20 {
				o[i] = math.Copysign(math.Inf(1), o[i])
			}
		}
		return o
	}
	pr := QPProblem{PSparse: denseToCSC(t, n, n, c.P, true), Q: c.Q}
	if len(c.A) > 0 {
		pr.Cons.ASparse = denseToCSC(t, len(c.A), n, c.A, false)
		pr.Cons.L, pr.Cons.U = clamp(c.L), clamp(c.U)
	}
	return pr
}

// The generated sparse set (tools/qporacle gen --set sparse: random QP, equality
// QP, Lasso, portfolio), with sparse P and A: the status, the iteration count
// and the ρ updates of OSQP v1.0.0 at its default settings, and Clarabel's
// solution at 1e-9 settings and polishing.
func TestADMMSparseAgainstOSQP(t *testing.T) {
	cases := loadQPFixtureFile(t, "testdata/qp/sparse.json")
	if len(cases) != 12 {
		t.Fatalf("%d cases", len(cases))
	}
	same, near25 := 0, 0
	for _, c := range cases {
		ref, osq := c.Results["clarabel"], c.Results["osqp"]
		r, err := Quadprog(sparseFixtureProblem(t, c), &QPOptions{Method: QPADMM})
		if err != nil || r.Status != QPOptimal || osq.Status != "solved" {
			t.Errorf("%s: %v %+v (osqp %s)", c.Name, err, r, osq.Status)
			continue
		}
		if osq.Obj != nil && math.Abs(r.Fun-*osq.Obj) > 10*(1e-3+1e-3*math.Abs(*osq.Obj)) {
			t.Errorf("%s: f %g, OSQP %g", c.Name, r.Fun, *osq.Obj)
		}
		if r.Nit == osq.Iter && r.RhoUpdates == osq.RhoUpdates {
			same++
		}
		if d := r.Nit - osq.Iter; d >= -25 && d <= 25 {
			near25++
		} else {
			t.Logf("%s: nit %d (rho updates %d), OSQP %d (%d)", c.Name, r.Nit, r.RhoUpdates, osq.Iter, osq.RhoUpdates)
		}
		// 1e-9 settings against Clarabel (1e-10): the same minimiser.
		tight, err := Quadprog(sparseFixtureProblem(t, c), &QPOptions{Method: QPADMM, EpsAbs: 1e-9, EpsRel: 1e-9, MaxIter: 200000, Polish: true})
		if err != nil || tight.Status != QPOptimal {
			t.Errorf("%s: tight %v %+v", c.Name, err, tight)
			continue
		}
		if math.Abs(tight.Fun-*ref.Obj) > 1e-6*math.Max(1, math.Abs(*ref.Obj)) {
			t.Errorf("%s: tight f %g, clarabel %g", c.Name, tight.Fun, *ref.Obj)
		}
	}
	t.Logf("12 sparse cases: identical (nit, rho updates) %d, nit within ±25 %d", same, near25)
	if same < 11 || near25 < 11 {
		t.Errorf("identical on %d, within ±25 on %d, of 12", same, near25)
	}
}
