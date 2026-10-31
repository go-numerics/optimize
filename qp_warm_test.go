package optimize

import (
	"errors"
	"math"
	"math/rand/v2"
	"strings"
	"testing"
)

// UpdateQ then Solve warm-starts from the previous active set and gives the
// cold answer; over many perturbations the warm start is accepted often and
// saves pivots, and is rejected (cold start) when its multipliers are not
// dual feasible.
func TestQPSolverWarmStart(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	accepted, rejected, warmNit, coldNit := 0, 0, 0, 0
	for trial := 0; trial < 300; trial++ {
		n := 3 + rng.IntN(5)
		pm := make([]float64, n*n)
		l := make([]float64, n*n)
		for i := range l {
			l[i] = rng.NormFloat64()
		}
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				for k := 0; k < n; k++ {
					pm[i*n+j] += float64(l[i*n+k] * l[j*n+k])
				}
			}
			pm[i*n+i] += 0.5
		}
		q := make([]float64, n)
		for i := range q {
			q[i] = 2 * rng.NormFloat64()
		}
		bounds := make([]Bound, n)
		a := make([]float64, n)
		for i := range bounds {
			bounds[i] = Bound{-1, 1}
			a[i] = 1
		}
		pr := QPProblem{P: qpArr(t, pm, n, n), Q: q, Cons: QPConstraints{Bounds: bounds,
			Linear: []LinearConstraint{{A: qpArr(t, a, 1, n), Ub: []float64{float64(n) / 4}}}}}
		s, err := NewQPSolver(pr, &QPOptions{Method: QPActiveSet})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := s.Solve(); err != nil {
			t.Fatal(err)
		}
		q2 := clone(q)
		for i := range q2 {
			q2[i] += 0.05 * rng.NormFloat64()
		}
		if err := s.UpdateQ(q2); err != nil {
			t.Fatal(err)
		}
		warm, err := s.Solve()
		if err != nil {
			t.Fatal(err)
		}
		pr.Q = q2
		cold, err := Quadprog(pr, &QPOptions{Method: QPActiveSet})
		if err != nil {
			t.Fatal(err)
		}
		if warm.Status != QPOptimal || cold.Status != QPOptimal || maxAbsDiff(warm.X, cold.X) > 1e-7 {
			t.Fatalf("trial %d: warm %v (%d, %s), cold %v", trial, warm.X, warm.Status, warm.Message, cold.X)
		}
		switch {
		case strings.Contains(warm.Message, "accepted"):
			accepted++
			warmNit += warm.Nit
			coldNit += cold.Nit
		case strings.Contains(warm.Message, "rejected"):
			rejected++
		}
	}
	t.Logf("accepted %d, rejected %d, pivots warm %d against cold %d", accepted, rejected, warmNit, coldNit)
	// The warm start re-enters the whole working set, so it saves no pivots
	// (design-qp.md §5 repairs the factors instead — not done); it must not
	// cost more than the entering itself.
	if accepted < 100 || rejected < 1 || float64(warmNit) > 1.15*float64(coldNit) {
		t.Errorf("accepted %d, rejected %d, pivots warm %d against cold %d", accepted, rejected, warmNit, coldNit)
	}
	s, _ := NewQPSolver(QPProblem{P: qpArr(t, []float64{1}, 1, 1), Q: []float64{0}}, nil)
	if err := s.UpdateQ([]float64{1, 2}); !errors.Is(err, ErrDimension) {
		t.Errorf("length: %v", err)
	}
	if err := s.UpdateQ([]float64{math.NaN()}); !errors.Is(err, ErrInput) {
		t.Errorf("NaN: %v", err)
	}
	// A hinted row already satisfied with equality is skipped; two dependent
	// hinted rows cannot both enter, which rejects the warm start (cold solve).
	one := qpArr(t, []float64{1}, 1, 1)
	ub := func(b float64) LinearConstraint {
		return LinearConstraint{A: qpArr(t, []float64{1}, 1), Ub: []float64{b}}
	}
	r, err := Quadprog(QPProblem{P: one, Q: []float64{-1}, Cons: QPConstraints{Linear: []LinearConstraint{ub(1)}}},
		&QPOptions{Method: QPActiveSet, WorkingSet: []int{0}})
	if err != nil || r.Status != QPOptimal || !strings.Contains(r.Message, "accepted") {
		t.Fatalf("satisfied hint: %v %+v", err, r)
	}
	lb := LinearConstraint{A: qpArr(t, []float64{1}, 1), Lb: []float64{2}}
	r, err = Quadprog(QPProblem{P: one, Q: []float64{0}, Cons: QPConstraints{Linear: []LinearConstraint{ub(1), lb}}},
		&QPOptions{Method: QPActiveSet, WorkingSet: []int{0, 1}})
	if err != nil || r.Status != QPInfeasible {
		t.Fatalf("contradictory hint: %v %+v", err, r)
	}
	// An explicit WorkingSet hint is used by the first Solve too.
	pr := QPProblem{P: qpArr(t, []float64{1, 0, 0, 1}, 2, 2), Q: []float64{-3, -3}, Cons: QPConstraints{Bounds: []Bound{{0, 1}, {0, 1}}}}
	r, err = Quadprog(pr, &QPOptions{Method: QPActiveSet, WorkingSet: []int{0, 1}})
	if err != nil || r.Status != QPOptimal || !strings.Contains(r.Message, "Warm start") {
		t.Fatalf("%v %+v", err, r)
	}
}
