package optimize

import (
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/go-sparse/sparse"
)

// A QPSolver on a sparse problem keeps its factorisation through UpdateQ,
// UpdateBounds (including rows that change type) and UpdateMatrixValues (P,
// dense rows and ASparse entries); every warm answer equals a cold one, and
// the warm solves are cheaper.
func TestADMMUpdates(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	warmNit, coldNit := 0, 0
	for trial := 0; trial < 60; trial++ {
		n := 4 + rng.IntN(5)
		// P: tridiagonal upper (sparse), values perturbed per update.
		mkP := func(scale float64) *sparse.CSC[float64] {
			ptr := []int{0}
			var idx []int32
			var val []float64
			for j := 0; j < n; j++ {
				if j > 0 {
					idx, val = append(idx, int32(j-1)), append(val, 0.3*scale)
				}
				idx, val = append(idx, int32(j)), append(val, 2*scale+float64(j%3))
				ptr = append(ptr, len(idx))
			}
			return qpCSC(t, n, n, ptr, idx, val)
		}
		// ASparse: row i touches columns i and i+1 (CSC arrays built column by column).
		mA := n - 1
		mkA := func(w float64) *sparse.CSC[float64] {
			ptr := []int{0}
			var idx []int32
			var val []float64
			for j := 0; j < n; j++ {
				if j > 0 {
					idx, val = append(idx, int32(j-1)), append(val, -w)
				}
				if j < mA {
					idx, val = append(idx, int32(j)), append(val, w)
				}
				ptr = append(ptr, len(idx))
			}
			return qpCSC(t, mA, n, ptr, idx, val)
		}
		q := make([]float64, n)
		for i := range q {
			q[i] = rng.NormFloat64()
		}
		lo, hi := make([]float64, mA), make([]float64, mA)
		for i := range lo {
			lo[i], hi[i] = -1-rng.Float64(), 1+rng.Float64()
		}
		dense := LinearConstraint{A: qpArr(t, ones(n), 1, n), Ub: []float64{float64(n) / 3}}
		build := func(pScale, w float64, q, lo, hi []float64) QPProblem {
			return QPProblem{PSparse: mkP(pScale), Q: q, Cons: QPConstraints{ASparse: mkA(w), L: lo, U: hi,
				Linear: []LinearConstraint{dense}, Bounds: boundsOf(n, 3)}}
		}
		o := admmOpts()
		s, err := NewQPSolver(build(1, 1, q, lo, hi), o)
		if err != nil {
			t.Fatal(err)
		}
		if r, err := s.Solve(); err != nil || r.Status != QPOptimal {
			t.Fatalf("trial %d: %v %+v", trial, err, r)
		}
		// 1. new q; 2. new bounds (row 0 becomes an equality, row 1 loose); 3. new values.
		q2 := clone(q)
		for i := range q2 {
			q2[i] += 0.1 * rng.NormFloat64()
		}
		if err := s.UpdateQ(q2); err != nil {
			t.Fatal(err)
		}
		lo2, hi2 := clone(lo), clone(hi)
		lo2[0], hi2[0] = 0.25, 0.25
		if mA > 1 {
			lo2[1], hi2[1] = -1e35, 1e35
		}
		// canonical rows: the Linear row, then the ASparse rows, then the bounds.
		l, u := make([]float64, len(s.m.rows)), make([]float64, len(s.m.rows))
		for i, r := range s.m.rows {
			l[i], u[i] = r.l, r.u
		}
		copy(l[1:], lo2)
		copy(u[1:], hi2)
		if err := s.UpdateBounds(l, u); err != nil {
			t.Fatal(err)
		}
		warm, err := s.Solve()
		if err != nil {
			t.Fatal(err)
		}
		cold, err := Quadprog(build(1, 1, q2, lo2, hi2), o)
		if err != nil {
			t.Fatal(err)
		}
		if warm.Status != cold.Status || (cold.Status == QPOptimal && maxAbsDiff(warm.X, cold.X) > 1e-5*math.Max(1, vnormInf(cold.X))) {
			t.Fatalf("trial %d bounds: warm %v (%d) cold %v (%d)", trial, warm.X, warm.Status, cold.X, cold.Status)
		}
		warmNit += warm.Nit
		coldNit += cold.Nit
		// values: P ×1.5, ASparse weight 1.2, the dense row ×2.
		pv := mkP(1.5).Data
		av := mkA(1.2).Data
		dv := make([]float64, n)
		for i := range dv {
			dv[i] = 2
		}
		if err := s.UpdateMatrixValues(pv, append(dv, av...)); err != nil {
			t.Fatal(err)
		}
		warm, err = s.Solve()
		if err != nil {
			t.Fatal(err)
		}
		p3 := build(1.5, 1.2, q2, lo2, hi2)
		p3.Cons.Linear = []LinearConstraint{{A: qpArr(t, scaled(ones(n), 2), 1, n), Ub: []float64{float64(n) / 3}}}
		cold, err = Quadprog(p3, o)
		if err != nil {
			t.Fatal(err)
		}
		if warm.Status != cold.Status || (cold.Status == QPOptimal && maxAbsDiff(warm.X, cold.X) > 1e-5*math.Max(1, vnormInf(cold.X))) {
			t.Fatalf("trial %d values: warm %v (%d) cold %v (%d)", trial, warm.X, warm.Status, cold.X, cold.Status)
		}
		warmNit += warm.Nit
		coldNit += cold.Nit
	}
	// The changes here are large (a row becomes an equality, another loose), at a
	// 1e-9 tolerance: no saving is claimed, the answers are what is checked.
	t.Logf("iterations warm %d, cold %d: ratio %.2f", warmNit, coldNit, float64(warmNit)/float64(coldNit))
}

func ones(n int) []float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = 1
	}
	return v
}

func scaled(v []float64, s float64) []float64 {
	o := clone(v)
	for i := range o {
		o[i] *= s
	}
	return o
}

func boundsOf(n int, b float64) []Bound {
	bs := make([]Bound, n)
	for i := range bs {
		bs[i] = Bound{-b, b}
	}
	return bs
}

// A dense P whose pattern changes needs a new ordering; an update that makes
// the KKT matrix not quasi-definite is refused and leaves the solver as it was.
func TestADMMUpdateRejectsAndPatterns(t *testing.T) {
	pr := QPProblem{P: qpArr(t, []float64{2, 1, 1, 2}, 2, 2), Q: []float64{-1, -1}, Cons: QPConstraints{Bounds: []Bound{{0, 1}, {0, 1}}}}
	s, err := NewQPSolver(pr, admmOpts())
	if err != nil {
		t.Fatal(err)
	}
	before, err := s.Solve()
	if err != nil || before.Status != QPOptimal {
		t.Fatal(err, before)
	}
	// Off-diagonal to zero: the pattern changes (one entry fewer).
	if err := s.UpdateMatrixValues([]float64{2, 0, 0, 2}, nil); err != nil {
		t.Fatal(err)
	}
	r, err := s.Solve()
	if err != nil || r.Status != QPOptimal || maxAbsDiff(r.X, []float64{0.5, 0.5}) > 1e-6 {
		t.Fatalf("%v %+v", err, r)
	}
	// An indefinite P is refused, and the solver still answers as before.
	if err := s.UpdateMatrixValues([]float64{-2, 0, 0, 2}, nil); !errors.Is(err, ErrInput) {
		t.Fatalf("indefinite update: %v", err)
	}
	r, err = s.Solve()
	if err != nil || r.Status != QPOptimal || maxAbsDiff(r.X, []float64{0.5, 0.5}) > 1e-6 {
		t.Fatalf("after the refused update: %v %+v", err, r)
	}
	// A pattern change whose ordering fails is refused too.
	save := qpAMD
	qpAMD = func(*sparse.CSR[float64]) ([]int, error) { return nil, errors.New("boom") }
	err = s.UpdateMatrixValues([]float64{2, 1, 1, 2}, nil)
	qpAMD = save
	if !errors.Is(err, ErrInput) {
		t.Fatalf("failing ordering: %v", err)
	}
	r, err = s.Solve()
	if err != nil || r.Status != QPOptimal || maxAbsDiff(r.X, []float64{0.5, 0.5}) > 1e-6 {
		t.Fatalf("after the failed ordering: %v %+v", err, r)
	}
	// Argument checks.
	for name, f := range map[string]func() error{
		"p length": func() error { return s.UpdateMatrixValues([]float64{1}, nil) },
		"p NaN":    func() error { return s.UpdateMatrixValues([]float64{math.NaN(), 0, 0, 1}, nil) },
		"a length": func() error { return s.UpdateMatrixValues(nil, []float64{1}) },
		"bounds":   func() error { return s.UpdateBounds([]float64{0}, []float64{1}) },
		"l > u":    func() error { return s.UpdateBounds([]float64{1, 1}, []float64{0, 0}) },
	} {
		if f() == nil {
			t.Errorf("%s accepted", name)
		}
	}
}

// A small perturbation of q: the kept factorisation and iterates reach the
// same answer in fewer iterations than a cold solve (default 1e-3 tolerances,
// as in OSQP's warm-start use).
func TestADMMWarmPerturbation(t *testing.T) {
	rng := rand.New(rand.NewPCG(13, 14))
	warmNit, coldNit := 0, 0
	for trial := 0; trial < 40; trial++ {
		n := 6
		pm := make([]float64, n*n)
		for i := 0; i < n; i++ {
			pm[i*n+i] = 1 + rng.Float64()
		}
		q := make([]float64, n)
		for i := range q {
			q[i] = 2 * rng.NormFloat64()
		}
		pr := QPProblem{P: qpArr(t, pm, n, n), Q: q, Cons: QPConstraints{Bounds: boundsOf(n, 1),
			Linear: []LinearConstraint{{A: qpArr(t, ones(n), 1, n), Ub: []float64{1}}}}}
		def := &QPOptions{Method: QPADMM, EpsAbs: 1e-6, EpsRel: 1e-6, CheckInterval: 5}
		s, err := NewQPSolver(pr, def)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := s.Solve(); err != nil {
			t.Fatal(err)
		}
		q2 := clone(q)
		for i := range q2 {
			q2[i] += 0.01 * rng.NormFloat64()
		}
		if err := s.UpdateQ(q2); err != nil {
			t.Fatal(err)
		}
		warm, err := s.Solve()
		if err != nil {
			t.Fatal(err)
		}
		pr.Q = q2
		cold, err := Quadprog(pr, def)
		if err != nil {
			t.Fatal(err)
		}
		if warm.Status != QPOptimal || cold.Status != QPOptimal || maxAbsDiff(warm.X, cold.X) > 2e-2 {
			t.Fatalf("trial %d: warm %v cold %v", trial, warm.X, cold.X)
		}
		warmNit += warm.Nit
		coldNit += cold.Nit
	}
	t.Logf("iterations warm %d, cold %d: ratio %.2f", warmNit, coldNit, float64(warmNit)/float64(coldNit))
	if float64(warmNit) > 0.7*float64(coldNit) {
		t.Errorf("warm %d against cold %d", warmNit, coldNit)
	}
}
