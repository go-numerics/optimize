package optimize

import (
	"errors"
	"fmt"
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
	for _, sigma := range []float64{0.05, 0.5, 2} {
		t.Run(fmt.Sprint("sigma", sigma), func(t *testing.T) { warmBench(t, sigma) })
	}
	warmEdges(t)
}

func warmBench(t *testing.T, sigma float64) {
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
			q2[i] += sigma * rng.NormFloat64()
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
		if !strings.Contains(warm.Message, "recovered") {
			t.Fatalf("trial %d: not resumed: %s", trial, warm.Message)
		}
		accepted++
		if strings.Contains(warm.Message, ", 0 dropped") {
			rejected++ // here: resumed with no drop at all
		}
		warmNit += warm.Nit
		coldNit += cold.Nit
	}
	t.Logf("resumed %d (%d without a drop), pivots warm %d against cold %d: ratio %.3f", accepted, rejected, warmNit, coldNit, float64(warmNit)/float64(coldNit))
	// Repair (Prop. 8.1–8.2) must save pivots: the measured ratio is the one in
	// docs/design-qp.md.
	limit := 0.8
	if sigma > 1 { // a perturbation of the size of q itself: the old set is nearly useless
		limit = 1
	}
	if accepted != 300 || rejected < 1 || float64(warmNit) > limit*float64(coldNit) {
		t.Errorf("resumed %d, no-drop %d, pivots warm %d against cold %d", accepted, rejected, warmNit, coldNit)
	}
}

func warmEdges(t *testing.T) {
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
	if err != nil || r.Status != QPOptimal || !strings.Contains(r.Message, "1 of 1 hinted rows entered") {
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
	warmUpdates(t)
}

// UpdateBounds, UpdateMatrixValues and WarmStart: each gives the cold answer
// of the changed problem, and rejects what it cannot take.
func warmUpdates(t *testing.T) {
	rng := rand.New(rand.NewPCG(8, 9))
	for trial := 0; trial < 100; trial++ {
		n := 3 + rng.IntN(4)
		pm := make([]float64, n*n)
		for i := 0; i < n; i++ {
			pm[i*n+i] = 1 + rng.Float64()
			for j := 0; j < i; j++ {
				pm[i*n+j] = 0.2 * rng.NormFloat64()
				pm[j*n+i] = pm[i*n+j]
			}
		}
		q := make([]float64, n)
		bounds := make([]Bound, n)
		for i := range q {
			q[i] = 3 * rng.NormFloat64()
			bounds[i] = Bound{-1, 1}
		}
		eq := make([]float64, n)
		for i := range eq {
			eq[i] = rng.NormFloat64()
		}
		mk := func(P []float64, ub float64, bds []Bound) QPProblem {
			return QPProblem{P: qpArr(t, clone(P), n, n), Q: q, Cons: QPConstraints{Bounds: bds,
				AEq: qpArr(t, eq, 1, n), BEq: qpArr(t, []float64{0.1}, 1),
				Linear: []LinearConstraint{{A: qpArr(t, eq, 1, n), Ub: []float64{ub}}}}}
		}
		s, err := NewQPSolver(mk(pm, 0.5, bounds), &QPOptions{Method: QPActiveSet})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := s.Solve(); err != nil {
			t.Fatal(err)
		}
		// New bounds: the canonical rows are AEq, the Linear row, then n bound rows.
		lo, hi := make([]float64, n+2), make([]float64, n+2)
		lo[0], hi[0] = 0.3, 0.3
		lo[1], hi[1] = math.Inf(-1), 0.2+rng.Float64()
		bds2 := make([]Bound, n)
		for i := 0; i < n; i++ {
			w := 0.3 + rng.Float64()
			lo[2+i], hi[2+i] = -w, w
			bds2[i] = Bound{-w, w}
		}
		if err := s.UpdateBounds(lo, hi); err != nil {
			t.Fatal(err)
		}
		warm, err := s.Solve()
		if err != nil {
			t.Fatal(err)
		}
		p2 := mk(pm, hi[1], bds2)
		p2.Cons.BEq = qpArr(t, []float64{0.3}, 1)
		cold, err := Quadprog(p2, &QPOptions{Method: QPActiveSet})
		if err != nil {
			t.Fatal(err)
		}
		if warm.Status != cold.Status || (cold.Status == QPOptimal && maxAbsDiff(warm.X, cold.X) > 1e-7) {
			t.Fatalf("trial %d bounds: warm %v (%d) cold %v (%d)", trial, warm.X, warm.Status, cold.X, cold.Status)
		}
		// New matrix values: P, and the equality row.
		pm2 := clone(pm)
		for i := 0; i < n; i++ {
			pm2[i*n+i] += rng.Float64()
		}
		eq2 := make([]float64, n)
		for i := range eq2 {
			eq2[i] = rng.NormFloat64()
		}
		// dense rows: the equality row, then the Linear row.
		if err := s.UpdateMatrixValues(pm2, append(clone(eq2), eq2...)); err != nil {
			t.Fatal(err)
		}
		warm, err = s.Solve()
		if err != nil {
			t.Fatal(err)
		}
		p3 := p2
		p3.P = qpArr(t, clone(pm2), n, n)
		p3.Cons.AEq = qpArr(t, eq2, 1, n)
		p3.Cons.Linear = []LinearConstraint{{A: qpArr(t, eq2, 1, n), Ub: []float64{hi[1]}}}
		cold, err = Quadprog(p3, &QPOptions{Method: QPActiveSet})
		if err != nil {
			t.Fatal(err)
		}
		if warm.Status != cold.Status || (cold.Status == QPOptimal && maxAbsDiff(warm.X, cold.X) > 1e-7) {
			t.Fatalf("trial %d values: warm %v (%d) cold %v (%d)", trial, warm.X, warm.Status, cold.X, cold.Status)
		}
	}
	// Rejections.
	one := qpArr(t, []float64{1}, 1, 1)
	pr := QPProblem{P: one, Q: []float64{0}, Cons: QPConstraints{Linear: []LinearConstraint{
		{A: qpArr(t, []float64{1}, 1), Ub: []float64{1}}}, Bounds: []Bound{{0, 2}}}}
	s, err := NewQPSolver(pr, &QPOptions{Method: QPActiveSet})
	if err != nil {
		t.Fatal(err)
	}
	inf := math.Inf(1)
	for name, f := range map[string]func() error{
		"bounds length":   func() error { return s.UpdateBounds([]float64{0}, []float64{1}) },
		"bounds NaN":      func() error { return s.UpdateBounds([]float64{math.NaN(), 0}, []float64{1, 2}) },
		"bounds order":    func() error { return s.UpdateBounds([]float64{-inf, 3}, []float64{1, 2}) },
		"bounds kind":     func() error { return s.UpdateBounds([]float64{-1, 0}, []float64{1, 2}) },
		"bounds equality": func() error { return s.UpdateBounds([]float64{-inf, 1}, []float64{1, 1}) },
		"p length":        func() error { return s.UpdateMatrixValues([]float64{1, 2}, nil) },
		"p NaN":           func() error { return s.UpdateMatrixValues([]float64{math.NaN()}, nil) },
		"p indefinite":    func() error { return s.UpdateMatrixValues([]float64{-1}, nil) },
		"a length":        func() error { return s.UpdateMatrixValues(nil, []float64{1, 2, 3}) },
		"a NaN":           func() error { return s.UpdateMatrixValues(nil, []float64{math.Inf(1)}) },
		"x length":        func() error { return s.WarmStart([]float64{1, 2}, nil) },
		"y length":        func() error { return s.WarmStart(nil, []float64{1}) },
		"y NaN":           func() error { return s.WarmStart(nil, []float64{math.NaN(), 0}) },
	} {
		if err := f(); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	if err := s.UpdateMatrixValues([]float64{4}, nil); err != nil { // the failed updates left it usable
		t.Fatal(err)
	}
	if err := s.WarmStart([]float64{1}, []float64{0, 0}); err != nil {
		t.Fatal(err)
	}
	if r, err := s.Solve(); err != nil || r.Status != QPOptimal || r.X[0] != 0 {
		t.Fatalf("%v %+v", err, r)
	}
	// P = 0 has no values; a sparse P updates its entries; ASparse values do not update.
	s0, err := NewQPSolver(QPProblem{Q: []float64{1}, Cons: QPConstraints{Bounds: []Bound{{0, 1}}}}, &QPOptions{Method: QPADMM})
	if err != nil {
		t.Fatal(err)
	}
	if err := s0.UpdateMatrixValues([]float64{1}, nil); !errors.Is(err, ErrDimension) {
		t.Errorf("P = 0 has no values: %v", err)
	}
	ps := qpCSC(t, 2, 2, []int{0, 1, 3}, []int32{0, 0, 1}, []float64{2, 1, 3})
	sp, err := NewQPSolver(QPProblem{PSparse: ps, Q: []float64{-1, -1}}, &QPOptions{Method: QPActiveSet})
	if err != nil {
		t.Fatal(err)
	}
	if err := sp.UpdateMatrixValues([]float64{4, 1, 3}, nil); err != nil {
		t.Fatal(err)
	}
	if err := sp.UpdateMatrixValues([]float64{-4, 1, 3}, nil); err == nil || ps.Data[0] != 2 {
		t.Fatalf("indefinite sparse P accepted or user data touched: %v %v", err, ps.Data)
	}
	if r, err := sp.Solve(); err != nil || r.Status != QPOptimal {
		t.Fatalf("%v %+v", err, r)
	}
	as := qpCSC(t, 1, 2, []int{0, 1, 2}, []int32{0, 0}, []float64{1, 1})
	sa, err := NewQPSolver(QPProblem{Q: []float64{1, 1}, Cons: QPConstraints{ASparse: as, U: []float64{1}}}, &QPOptions{Method: QPADMM})
	if err != nil {
		t.Fatal(err)
	}
	if err := sa.UpdateMatrixValues(nil, []float64{1, 1}); err != nil {
		t.Errorf("ASparse values: %v", err)
	}
}
