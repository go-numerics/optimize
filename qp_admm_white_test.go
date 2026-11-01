package optimize

import (
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/go-sparse/sparse"
	"github.com/go-sparse/sparse/ldl"
)

// The fixture with scaling off, ρ adaptation off, and the duality gap check
// off: the unscaled branches of the residual and tolerance code.
func TestADMMUnscaledAndSettings(t *testing.T) {
	for _, set := range []func(*QPOptions){
		func(o *QPOptions) { o.ScalingIter = -1 },
		func(o *QPOptions) { o.AdaptiveRhoInterval = -1 },
		func(o *QPOptions) { o.NoDualGapCheck = true },
		func(o *QPOptions) {
			o.ScalingIter, o.AdaptiveRhoInterval, o.Rho, o.Alpha, o.Sigma = -1, 7, 1, 1.2, 1e-4
		},
	} {
		ok := 0
		for _, c := range loadQPFixture(t) {
			ref := c.Results["clarabel"]
			if ref.Status != "Solved" || c.Name == "non-convex" {
				continue
			}
			r, err := Quadprog(fixtureProblem(t, c), admmOpts(set))
			if err != nil || r.Status != QPOptimal || maxAbsDiff(r.X, ref.X) > 1e-4*math.Max(1, vnormInf(ref.X)) {
				t.Errorf("%s: %v %+v", c.Name, err, r)
				continue
			}
			ok++
		}
		if ok != 35 {
			t.Errorf("settings: %d solved, want 35", ok)
		}
	}
}

// Rows whose finite sides are beyond 1e30 are the "loose" rows of OSQP (ρ_min)
// and fall into the recession-cone code both ways.
func TestADMMLooseRows(t *testing.T) {
	pr := QPProblem{P: qpArr(t, []float64{2, 0, 0, 2}, 2, 2), Q: []float64{-2, -4}, Cons: QPConstraints{Linear: []LinearConstraint{
		{A: qpArr(t, []float64{1, 1}, 1, 2), Lb: []float64{-1e35}, Ub: []float64{1e35}},
		{A: qpArr(t, []float64{1, -1}, 1, 2), Ub: []float64{0.5}}}}}
	r, err := Quadprog(pr, admmOpts())
	if err != nil || r.Status != QPOptimal || r.Linear.Marginals[0] != 0 && math.Abs(r.Linear.Marginals[0]) > 1e-6 {
		t.Fatalf("%v %+v", err, r)
	}
	s, err := NewQPSolver(pr, admmOpts())
	if err != nil {
		t.Fatal(err)
	}
	loose := 0
	for _, c := range s.ad.ctype {
		if c == -1 {
			loose++
		}
	}
	if loose != 1 {
		t.Errorf("ctype %v, want one loose row", s.ad.ctype)
	}
	y := []float64{3, -3, -3}
	l, u := []float64{-1e31, -1e31, 0}, []float64{1e31, 1, 1e31}
	projectPolarRecCone(y, l, u)
	if y[0] != 0 || y[1] != 0 || y[2] != -3 {
		t.Errorf("projection %v", y)
	}
	// y2 has an infinite upper side only: positive y is cut; y1 has an infinite lower side only.
	y = []float64{3, -3}
	projectPolarRecCone(y, []float64{0, -1e31}, []float64{1e31, 1})
	if y[0] != 0 || y[1] != 0 {
		t.Errorf("projection %v", y)
	}
	if !inRecCone([]float64{5, -5}, []float64{-1e31, -1e31}, []float64{1e31, 1e31}, 0) ||
		inRecCone([]float64{5}, []float64{0}, []float64{1}, 1) || inRecCone([]float64{-5}, []float64{0}, []float64{1e31}, 1) {
		t.Error("inRecCone")
	}
}

func TestADMMInjectedFailures(t *testing.T) {
	pr := QPProblem{P: qpArr(t, []float64{2, 0, 0, 2}, 2, 2), Q: []float64{-2, -4}, Cons: QPConstraints{Bounds: []Bound{{0, 1}, {0, 1}}}}
	// The ordering and the analysis fail: reported as ErrInput.
	saveA, saveP := qpAMD, qpNewPermuted
	defer func() { qpAMD, qpNewPermuted = saveA, saveP }()
	qpAMD = func(*sparse.CSR[float64]) ([]int, error) { return nil, errors.New("boom") }
	if _, err := NewQPSolver(pr, &QPOptions{Method: QPADMM}); !errors.Is(err, ErrInput) || !strings.Contains(err.Error(), "ordering") {
		t.Errorf("AMD: %v", err)
	}
	qpAMD = saveA
	qpNewPermuted = func(*sparse.CSC[float64], []int) (*ldl.Permuted, error) { return nil, errors.New("boom") }
	if _, err := NewQPSolver(pr, &QPOptions{Method: QPADMM}); !errors.Is(err, ErrInput) || !strings.Contains(err.Error(), "analysing") {
		t.Errorf("NewPermuted: %v", err)
	}
	qpNewPermuted = saveP

	// A NaN residual is a numerical failure, never an endless loop.
	s, err := NewQPSolver(pr, &QPOptions{Method: QPADMM})
	if err != nil {
		t.Fatal(err)
	}
	s.ad.primRes = math.NaN()
	if !s.ad.checkTermination(false) || s.ad.status != QPNumericalFail {
		t.Errorf("NaN residual: status %d", s.ad.status)
	}
	if r := s.ad.result(); r.Status != QPNumericalFail || r.Success || !math.IsNaN(r.X[0]) {
		t.Errorf("%+v", r)
	}
	// The C-style min and max take the second argument on NaN and on ties.
	if cMin(2, 1) != 1 || cMin(1, 2) != 1 || cMax(math.NaN(), 3) != 3 || cMax(4, 3) != 4 {
		t.Error("cMin/cMax")
	}
	// A refactorisation that fails when ρ adapts stops the solve.
	failed := false
	for _, c := range loadQPFixture(t) {
		if c.Name == "non-convex" || c.Results["clarabel"].Status != "Solved" {
			continue
		}
		s, err := NewQPSolver(fixtureProblem(t, c), &QPOptions{Method: QPADMM, AdaptiveRhoInterval: 1, ScalingIter: -1, CheckInterval: 1000})
		if err != nil {
			t.Fatal(err)
		}
		for _, p := range s.ad.kkt.pDiag {
			s.ad.kkt.mat.Data[p] = math.NaN()
		}
		if r, err := s.Solve(); err == nil && r.Status == QPNumericalFail {
			failed = true
			break
		}
	}
	if !failed {
		t.Error("no case adapted ρ against a broken factor")
	}
}

func TestADMMRhoEstimate(t *testing.T) {
	pr := QPProblem{P: qpArr(t, []float64{2, 0, 0, 2}, 2, 2), Q: []float64{-2, -4}, Cons: QPConstraints{Bounds: []Bound{{0, 1}, {0, 1}}}}
	s, err := NewQPSolver(pr, &QPOptions{Method: QPADMM})
	if err != nil {
		t.Fatal(err)
	}
	a := s.ad
	a.scPrim, a.scDual = 1, 0 // an exactly zero dual residual: the estimate saturates at ρ_max
	if got := a.rhoEstimate(); got != admmRhoMax {
		t.Errorf("estimate %g", got)
	}
	a.scPrim, a.scDual = 0, 0 // 0/0: C's min/max give ρ_min
	if got := a.rhoEstimate(); got != admmRhoMin {
		t.Errorf("estimate %g", got)
	}
}
