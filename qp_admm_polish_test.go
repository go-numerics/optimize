package optimize

import (
	"errors"
	"math"
	"testing"

	"github.com/go-sparse/sparse"
)

// Polishing turns the 1e-3 ADMM answer into a high-accuracy one wherever the
// active set is identified: x within 1e-6 of Clarabel's (1e-10) on the solved
// fixture cases, as OSQP's own polish does at 1e-9 settings.
func TestADMMPolish(t *testing.T) {
	polished, total := 0, 0
	for _, c := range loadQPFixture(t) {
		ref := c.Results["clarabel"]
		if ref.Status != "Solved" || c.Name == "non-convex" {
			continue
		}
		total++
		r, err := Quadprog(fixtureProblem(t, c), &QPOptions{Method: QPADMM, Polish: true})
		if err != nil || r.Status != QPOptimal {
			t.Fatalf("%s: %v %+v", c.Name, err, r)
		}
		if !r.Polished {
			continue
		}
		polished++
		if d := maxAbsDiff(r.X, ref.X); d > 1e-6*math.Max(1, vnormInf(ref.X)) {
			t.Errorf("%s: polished x off by %g (nit %d)", c.Name, d, r.Nit)
		}
	}
	t.Logf("polished %d of %d", polished, total)
	if polished*2 < total {
		t.Errorf("polished only %d of %d", polished, total)
	}
}

// No active row: polishing is skipped, not failed. A rejected polish leaves the
// ADMM iterate and its information unchanged. A polish that cannot factor or
// order its system leaves them too.
func TestADMMPolishEdges(t *testing.T) {
	free := QPProblem{P: qpArr(t, []float64{2, 0, 0, 4}, 2, 2), Q: []float64{-2, -4}, Cons: QPConstraints{Bounds: []Bound{{-10, 10}, {-10, 10}}}}
	r, err := Quadprog(free, &QPOptions{Method: QPADMM, Polish: true})
	if err != nil || r.Status != QPOptimal || r.Polished {
		t.Fatalf("no active set: %v %+v", err, r)
	}
	pr := QPProblem{P: qpArr(t, []float64{2, 0.5, 0.5, 1}, 2, 2), Q: []float64{-1, -3}, Cons: QPConstraints{
		AUb: qpArr(t, []float64{1, 1}, 1, 2), BUb: qpArr(t, []float64{1}, 1), Bounds: []Bound{{0, 4}, {0, 4}}}}
	solve := func(mod func(*admm)) (*admm, *QPResult) {
		s, err := NewQPSolver(pr, &QPOptions{Method: QPADMM})
		if err != nil {
			t.Fatal(err)
		}
		mod(s.ad)
		res, err := s.Solve()
		if err != nil {
			t.Fatal(err)
		}
		return s.ad, res
	}
	// A heavily regularised polish without refinement is worse: rejected.
	a, res := solve(func(a *admm) { a.set.polish, a.set.polishDelta, a.set.polishRefine = true, 1e3, 0 })
	if res.Polished || res.Status != QPOptimal {
		t.Errorf("rejected polish: %+v", res)
	}
	plain, _ := solve(func(*admm) {})
	if maxAbsDiff(a.x, plain.x) != 0 || a.primRes != plain.primRes || a.dualRes != plain.dualRes {
		t.Error("a rejected polish changed the iterate or its information")
	}
	// A system that cannot be ordered or factored: no polish, no error.
	s, err := NewQPSolver(pr, &QPOptions{Method: QPADMM})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Solve(); err != nil {
		t.Fatal(err)
	}
	save := qpAMD
	qpAMD = func(*sparse.CSR[float64]) ([]int, error) { return nil, errors.New("boom") }
	s.ad.polish()
	qpAMD = save
	if s.ad.polished {
		t.Error("polished with a failed ordering")
	}
	for i := range s.ad.pu.Data {
		s.ad.pu.Data[i] = math.NaN()
	}
	s.ad.polish()
	if s.ad.polished {
		t.Error("polished with a NaN matrix")
	}
}
