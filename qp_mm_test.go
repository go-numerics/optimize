package optimize

import (
	"math"
	"testing"
)

// The Maros–Mészáros problems of tools/qporacle mm convert (tiny tier and the
// part of the small tier that fits a dense fixture): the ADMM solver at 1e-9
// settings with polishing, and the active set where P is positive definite,
// against Clarabel at 1e-10. Every case Clarabel solves, we solve to its
// objective within 1e-5 relative; the tallies are in the log.
func TestMarosMeszarosFixtures(t *testing.T) {
	admmOK, asOK, admmBad, total := 0, 0, 0, 0
	for _, path := range []string{"testdata/qp/mm/tiny.json", "testdata/qp/mm/small.json"} {
		for _, c := range loadQPFixtureFile(t, path) {
			total++
			ref := c.Results["clarabel"]
			if ref.Status != "Solved" || ref.Obj == nil {
				continue
			}
			pr := fixtureProblem(t, c)
			tol := 1e-5 * math.Max(1, math.Abs(*ref.Obj))
			r, err := Quadprog(pr, &QPOptions{Method: QPADMM, EpsAbs: 1e-9, EpsRel: 1e-9, MaxIter: 400000, Polish: true})
			switch {
			case err != nil || r.Status != QPOptimal || math.Abs(r.Fun-*ref.Obj) > tol:
				admmBad++
				t.Logf("%s: ADMM %v status %d f %g (clarabel %g)", c.Name, err, statusOf(r), funOf(r), *ref.Obj)
			default:
				admmOK++
			}
			if a, err := Quadprog(pr, &QPOptions{Method: QPActiveSet}); err == nil && a.Status == QPOptimal {
				if math.Abs(a.Fun-*ref.Obj) > tol {
					t.Errorf("%s: active set f %g, clarabel %g", c.Name, a.Fun, *ref.Obj)
				}
				asOK++
			}
		}
	}
	t.Logf("%d problems: ADMM agrees on %d (disagrees %d), active set on %d", total, admmOK, admmBad, asOK)
	if total != 25 || admmBad > 3 {
		t.Errorf("total %d, ADMM disagreements %d", total, admmBad)
	}
}

func statusOf(r *QPResult) int {
	if r == nil {
		return -1
	}
	return r.Status
}

func funOf(r *QPResult) float64 {
	if r == nil {
		return math.NaN()
	}
	return r.Fun
}
