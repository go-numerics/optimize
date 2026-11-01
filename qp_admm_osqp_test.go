package optimize

import (
	"math"
	"testing"
)

// The planted fixture was solved by OSQP v1.0.0 with its default settings
// (iteration-based ρ adaptation, so deterministic). The ADMM solver follows the
// same iteration: same status, an objective within 10·(ε_abs + ε_rel·|f|), and
// the iteration count within one check interval (design-qp.md §6.3).
func TestADMMAgainstOSQP(t *testing.T) {
	status := map[string]int{"solved": QPOptimal, "solved inaccurate": QPInaccurate, "primal infeasible": QPInfeasible,
		"dual infeasible": QPUnbounded, "maximum iterations reached": QPIterLimit}
	same, near25, total := 0, 0, 0
	for _, c := range loadQPFixture(t) {
		ref := c.Results["osqp"]
		r, err := Quadprog(fixtureProblem(t, c), &QPOptions{Method: QPADMM})
		if err != nil {
			t.Errorf("%s: %v", c.Name, err)
			continue
		}
		if ref.Error != "" { // OSQP refuses a non-convex P at setup
			if r.Status != QPNonConvex {
				t.Errorf("%s: OSQP error %q, ours status %d", c.Name, ref.Error, r.Status)
			}
			continue
		}
		total++
		want, ok := status[ref.Status]
		if !ok || r.Status != want {
			t.Errorf("%s: status %d, OSQP %q", c.Name, r.Status, ref.Status)
			continue
		}
		if r.Status == QPOptimal && ref.Obj != nil {
			tol := 10 * (1e-3 + 1e-3*math.Abs(*ref.Obj))
			if math.Abs(r.Fun-*ref.Obj) > tol {
				t.Errorf("%s: f %g, OSQP %g", c.Name, r.Fun, *ref.Obj)
			}
		}
		if r.Nit == ref.Iter && r.RhoUpdates == ref.RhoUpdates {
			same++
		}
		if d := r.Nit - ref.Iter; d >= -25 && d <= 25 {
			near25++
		} else {
			t.Logf("%s: nit %d (rho updates %d), OSQP %d (%d)", c.Name, r.Nit, r.RhoUpdates, ref.Iter, ref.RhoUpdates)
		}
	}
	t.Logf("%d cases: identical (nit, rho updates) %d, nit within ±25 %d", total, same, near25)
	// 38 of 38 are identical on amd64 and arm64 at the time of writing; the
	// fused multiply-adds of go-simd differ from OSQP's plain ones in the last
	// bits, so only 90% are required.
	if near25*10 < total*9 || same*10 < total*9 {
		t.Errorf("nit within ±25 on %d, identical on %d, of %d cases", near25, same, total)
	}
}
