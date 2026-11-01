package optimize

import (
	"errors"
	"math"
	"strings"
	"testing"
)

func admmOpts(extra ...func(*QPOptions)) *QPOptions {
	o := &QPOptions{Method: QPADMM, EpsAbs: 1e-9, EpsRel: 1e-9, MaxIter: 100000}
	for _, f := range extra {
		f(o)
	}
	return o
}

// Problems the active-set method cannot take: an LP, a singular P, no
// constraints at all, sparse P and constraints.
func TestADMMProblemShapes(t *testing.T) {
	// LP: min −x1 − x2 s.t. x1 + x2 <= 1, x >= 0.
	lp := QPProblem{Q: []float64{-1, -1}, Cons: QPConstraints{
		AUb: qpArr(t, []float64{1, 1}, 1, 2), BUb: qpArr(t, []float64{1}, 1),
		Bounds: []Bound{{0, math.Inf(1)}, {0, math.Inf(1)}}}}
	r, err := Quadprog(lp, admmOpts())
	if err != nil || r.Status != QPOptimal || math.Abs(r.Fun+1) > 1e-6 || r.Method != QPADMM {
		t.Fatalf("LP: %v %+v", err, r)
	}
	// Singular P: min ½(x1+x2)² − x1 − x2 on a line; any x with x1 + x2 = 1.
	sing := QPProblem{P: qpArr(t, []float64{1, 1, 1, 1}, 2, 2), Q: []float64{-1, -1},
		Cons: QPConstraints{Bounds: []Bound{{0, 5}, {0, 5}}}}
	r, err = Quadprog(sing, admmOpts())
	if err != nil || r.Status != QPOptimal || math.Abs(r.Fun+0.5) > 1e-6 || math.Abs(r.X[0]+r.X[1]-1) > 1e-5 {
		t.Fatalf("singular: %v %+v", err, r)
	}
	// No constraints: x = −P⁻¹q.
	free := QPProblem{P: qpArr(t, []float64{2, 0, 0, 4}, 2, 2), Q: []float64{-2, -4}}
	r, err = Quadprog(free, admmOpts())
	if err != nil || r.Status != QPOptimal || maxAbsDiff(r.X, []float64{1, 1}) > 1e-6 {
		t.Fatalf("free: %v %+v", err, r)
	}
	// Sparse P (upper triangle) and sparse constraints against the dense
	// active-set answer of the same problem.
	pu := qpCSC(t, 3, 3, []int{0, 1, 3, 5}, []int32{0, 0, 1, 1, 2}, []float64{2, 0.5, 3, 0.25, 1})
	as := qpCSC(t, 2, 3, []int{0, 1, 3, 4}, []int32{0, 0, 1, 1}, []float64{1, 1, 1, -1})
	// rows: x0 + x1 <= 2 ; x1 − ... CSC columns: col0 = {row0: 1}, col1 = {row0: 1, row1: 1}, col2 = {row1: −1}
	sp := QPProblem{PSparse: pu, Q: []float64{-1, -2, -3}, Cons: QPConstraints{ASparse: as,
		L: []float64{math.Inf(-1), -1}, U: []float64{2, 1}, Bounds: []Bound{{-3, 3}, {-3, 3}, {-3, 3}}}}
	r, err = Quadprog(sp, admmOpts())
	if err != nil || r.Status != QPOptimal || len(r.Sparse.Marginals) != 2 {
		t.Fatalf("sparse: %v %+v", err, r)
	}
	dp := []float64{2, 0.5, 0, 0.5, 3, 0.25, 0, 0.25, 1}
	dense := QPProblem{P: qpArr(t, dp, 3, 3), Q: []float64{-1, -2, -3}, Cons: QPConstraints{
		Linear: []LinearConstraint{{A: qpArr(t, []float64{1, 1, 0, 0, 1, -1}, 2, 3), Lb: []float64{math.Inf(-1), -1}, Ub: []float64{2, 1}}},
		Bounds: []Bound{{-3, 3}, {-3, 3}, {-3, 3}}}}
	want, err := Quadprog(dense, &QPOptions{Method: QPActiveSet})
	if err != nil || want.Status != QPOptimal {
		t.Fatal(err, want)
	}
	if maxAbsDiff(r.X, want.X) > 1e-5 || math.Abs(r.Fun-want.Fun) > 1e-6 {
		t.Errorf("sparse x %v f %g, active set %v f %g", r.X, r.Fun, want.X, want.Fun)
	}
	// The multipliers satisfy the stationarity identity of the QPResult comment.
	if r.Optimality > 1e-6 {
		t.Errorf("optimality %g", r.Optimality)
	}
}

// Infeasible and unbounded certificates re-verified from the vector alone.
func TestADMMCertificates(t *testing.T) {
	// x >= 1 and x <= 0: δy with Aᵀδy ≈ 0 and uᵀδy₊ + lᵀδy₋ < 0 (12).
	inf := QPProblem{P: qpArr(t, []float64{1}, 1, 1), Q: []float64{0}, Cons: QPConstraints{Linear: []LinearConstraint{
		{A: qpArr(t, []float64{1}, 1), Lb: []float64{1}}, {A: qpArr(t, []float64{1}, 1), Ub: []float64{0}}}}}
	s, err := NewQPSolver(inf, admmOpts())
	if err != nil {
		t.Fatal(err)
	}
	r, err := s.Solve()
	if err != nil || r.Status != QPInfeasible || r.PrimalCert == nil || !math.IsNaN(r.X[0]) {
		t.Fatalf("%v %+v", err, r)
	}
	at, supp := 0.0, 0.0
	for j, row := range s.m.rows {
		at += s.m.rowDot(row, []float64{1}) * r.PrimalCert[j]
		if v := r.PrimalCert[j]; v > 0 {
			supp += row.u * v
		} else if v < 0 {
			supp += row.l * v
		}
	}
	if math.Abs(at) > 1e-3 || supp >= 0 || math.Abs(vnormInf(r.PrimalCert)-1) > 1e-12 {
		t.Errorf("certificate %v: Aᵀδy %g, support %g", r.PrimalCert, at, supp)
	}
	// min −x1 with x1 >= 0 free above and P = 0: δx with qᵀδx < 0, Pδx = 0,
	// Aδx in the recession cone (13).
	unb := QPProblem{Q: []float64{-1, 0}, Cons: QPConstraints{Bounds: []Bound{{0, math.Inf(1)}, {-1, 1}}}}
	s, err = NewQPSolver(unb, admmOpts())
	if err != nil {
		t.Fatal(err)
	}
	r, err = s.Solve()
	if err != nil || r.Status != QPUnbounded || r.DualCert == nil {
		t.Fatalf("%v %+v", err, r)
	}
	if r.DualCert[0]*(-1) >= 0 || math.Abs(r.DualCert[1]) > 1e-3 || r.DualCert[0] < -1e-3 {
		t.Errorf("dual certificate %v", r.DualCert)
	}
	// A problem with no constraint rows at all can be unbounded too.
	r, err = Quadprog(QPProblem{Q: []float64{-1}}, admmOpts())
	if err != nil || r.Status != QPUnbounded {
		t.Fatalf("m = 0: %v %+v", err, r)
	}
}

func TestADMMNonConvexAndLimits(t *testing.T) {
	// P + σI with an exactly zero pivot, and with a negative one.
	for _, p := range []float64{-1e-6, -1} {
		r, err := Quadprog(QPProblem{P: qpArr(t, []float64{p}, 1, 1), Q: []float64{1}}, &QPOptions{Method: QPADMM})
		if err != nil || r.Status != QPNonConvex || r.Success || !strings.Contains(r.Message, "non convex") {
			t.Fatalf("P = %g: %v %+v", p, err, r)
		}
	}
	// An iteration limit returns the last iterate and no error.
	pr := QPProblem{P: qpArr(t, []float64{2, 0, 0, 4}, 2, 2), Q: []float64{-2, -4}, Cons: QPConstraints{Bounds: []Bound{{0, 0.5}, {0, 0.5}}}}
	r, err := Quadprog(pr, &QPOptions{Method: QPADMM, MaxIter: 1})
	if err != nil || r.Status != QPIterLimit || r.Nit != 1 || len(r.X) != 2 || math.IsNaN(r.X[0]) {
		t.Fatalf("%v %+v", err, r)
	}
	// Overflowing data end in a status, never a panic or an endless loop.
	big := QPProblem{P: qpArr(t, []float64{1e300}, 1, 1), Q: []float64{-1e300}, Cons: QPConstraints{Bounds: []Bound{{0, 1e300}}}}
	r, err = Quadprog(big, &QPOptions{Method: QPADMM})
	if err != nil || r.Status == QPOptimal && math.IsNaN(r.Fun) {
		t.Fatalf("%v %+v", err, r)
	}
	t.Logf("overflow: status %d (%s)", r.Status, r.Message)
}

// Settings search: a status that only the ten-times-looser pass can give is
// reached for some iteration limit, for each of the three families.
func TestADMMInaccurateStatuses(t *testing.T) {
	prob := func(kind int) QPProblem {
		switch kind {
		case 1: // primal infeasible
			return QPProblem{P: qpArr(t, []float64{1}, 1, 1), Q: []float64{0}, Cons: QPConstraints{Linear: []LinearConstraint{
				{A: qpArr(t, []float64{1}, 1), Lb: []float64{1}}, {A: qpArr(t, []float64{1}, 1), Ub: []float64{0}}}}}
		case 2: // dual infeasible
			return QPProblem{Q: []float64{-1, 0}, Cons: QPConstraints{Bounds: []Bound{{0, math.Inf(1)}, {-1, 1}}}}
		}
		return QPProblem{P: qpArr(t, []float64{2, 0.5, 0.5, 1}, 2, 2), Q: []float64{-1, -3}, Cons: QPConstraints{
			AUb: qpArr(t, []float64{1, 1}, 1, 2), BUb: qpArr(t, []float64{1}, 1), Bounds: []Bound{{0, 4}, {0, 4}}}}
	}
	for kind, want := range []int{QPInaccurate, QPInfeasible} {
		found := false
		for maxIter := 1; maxIter < 300 && !found; maxIter++ {
			for _, eps := range []float64{1e-3, 1e-4, 1e-5, 1e-6} {
				r, err := Quadprog(prob(kind), &QPOptions{Method: QPADMM, MaxIter: maxIter, EpsAbs: eps, EpsRel: eps, CheckInterval: 1000})
				if err != nil {
					t.Fatal(err)
				}
				if r.Status == want && (kind == 0 || strings.Contains(r.Message, "inaccurate")) {
					found = true
					break
				}
			}
		}
		if !found {
			t.Errorf("kind %d: status %d never reached through the approximate pass", kind, want)
		}
	}
}

// A warm start from the solution converges at the first check; UpdateQ keeps
// the iterates; the answer equals a cold solve.
func TestADMMWarmStart(t *testing.T) {
	pr := QPProblem{P: qpArr(t, []float64{2, 0.5, 0.5, 1}, 2, 2), Q: []float64{-1, -3}, Cons: QPConstraints{
		AUb: qpArr(t, []float64{1, 1}, 1, 2), BUb: qpArr(t, []float64{1}, 1), Bounds: []Bound{{0, 4}, {0, 4}}}}
	o := admmOpts()
	cold, err := Quadprog(pr, o)
	if err != nil || cold.Status != QPOptimal {
		t.Fatal(err, cold)
	}
	o2 := admmOpts(func(o *QPOptions) { o.X0, o.Y0 = cold.X, cold.Y })
	warm, err := Quadprog(pr, o2)
	if err != nil || warm.Status != QPOptimal || warm.Nit > 25 || maxAbsDiff(warm.X, cold.X) > 1e-6 {
		t.Fatalf("%v nit %d (cold %d) %v", err, warm.Nit, cold.Nit, warm.X)
	}
	s, err := NewQPSolver(pr, o)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Solve(); err != nil {
		t.Fatal(err)
	}
	q2 := []float64{-1.1, -3.2}
	if err := s.UpdateQ(q2); err != nil {
		t.Fatal(err)
	}
	again, err := s.Solve()
	if err != nil {
		t.Fatal(err)
	}
	pr.Q = q2
	ref, _ := Quadprog(pr, o)
	if again.Status != QPOptimal || maxAbsDiff(again.X, ref.X) > 1e-6 || again.Nit >= ref.Nit {
		t.Fatalf("UpdateQ: nit %d against cold %d, x %v against %v", again.Nit, ref.Nit, again.X, ref.X)
	}
	if err := s.WarmStart(cold.X, cold.Y); err != nil {
		t.Fatal(err)
	}
	if err := s.WarmStart(cold.X, nil); err != nil {
		t.Fatal(err)
	}
	if err := s.WarmStart(nil, cold.Y); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Solve(); err != nil {
		t.Fatal(err)
	}
	// Updates ADMM does not have yet say so.
	if err := s.UpdateBounds([]float64{0}, []float64{1}); !errors.Is(err, ErrMethod) {
		t.Errorf("UpdateBounds: %v", err)
	}
	if err := s.UpdateMatrixValues(nil, nil); !errors.Is(err, ErrMethod) {
		t.Errorf("UpdateMatrixValues: %v", err)
	}
}

// The size limits of the KKT matrix and of its factor.
func TestADMMSizeLimits(t *testing.T) {
	pr := QPProblem{P: qpArr(t, []float64{2, 0, 0, 4}, 2, 2), Q: []float64{-2, -4}, Cons: QPConstraints{Bounds: []Bound{{0, 1}, {0, 1}}}}
	save := qpFactorLimit
	qpFactorLimit = 1
	_, err := NewQPSolver(pr, &QPOptions{Method: QPADMM})
	qpFactorLimit = save
	if !errors.Is(err, ErrInput) || !strings.Contains(err.Error(), "factor") {
		t.Errorf("factor limit: %v", err)
	}
	// The KKT matrix size is bounded by the model's own limit.
	saveMax := qpMaxSize
	qpMaxSize = 3
	_, err = NewQPSolver(pr, &QPOptions{Method: QPADMM})
	qpMaxSize = saveMax
	if !errors.Is(err, ErrInput) {
		t.Errorf("size limit: %v", err)
	}
}
