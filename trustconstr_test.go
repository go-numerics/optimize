package optimize

import (
	"errors"
	"math"
	"testing"

	"github.com/go-ndarray/ndarray"
)

func TestBarrierValue(t *testing.T) {
	if v := barrierValue(1, 0.5, []float64{1, math.E}); math.Abs(v-0.5) > 1e-15 {
		t.Errorf("barrierValue = %v, want 0.5", v)
	}
	if v := barrierValue(1, 0.5, []float64{1, 0}); !math.IsInf(v, 1) {
		t.Errorf("non-positive slack gave %v", v)
	}
}

func TestIntersections(t *testing.T) {
	z, d := []float64{0, 0}, []float64{1, 0}
	if ta, tb, ok := sphereIntersections(z, d, 2, true); !ok || ta != -2 || tb != 2 {
		t.Errorf("entire line: %v %v %v", ta, tb, ok)
	}
	if _, _, ok := sphereIntersections(z, []float64{0, 0}, 1, false); ok {
		t.Error("zero direction intersects")
	}
	if _, _, ok := sphereIntersections([]float64{5, 5}, d, 1, false); ok {
		t.Error("distant line intersects")
	}
	if _, _, ok := sphereIntersections([]float64{3, 0}, d, 1, false); ok {
		t.Error("segment beyond the ball intersects")
	}
	lb, ub := []float64{-1, -1}, []float64{1, 1}
	if _, _, ok := boxIntersections([]float64{0, 2}, d, lb, ub, false); ok {
		t.Error("segment outside the box in a fixed coordinate intersects")
	}
	if _, _, ok := boxIntersections(z, []float64{0, 0}, lb, ub, false); ok {
		t.Error("zero direction intersects the box")
	}
	if ta, tb, ok := boxIntersections(z, d, lb, ub, true); !ok || ta != -1 || tb != 1 {
		t.Errorf("box entire line: %v %v %v", ta, tb, ok)
	}
	if _, _, ok := boxIntersections([]float64{5, 0}, d, lb, ub, false); ok {
		t.Error("segment beyond the box intersects")
	}
}

func TestProjectedCGEdges(t *testing.T) {
	p, _ := newProjector(0, 2, nil)
	lb, ub := []float64{-10, -10}, []float64{10, 10}
	neg := func(v, out []float64) { out[0], out[1] = -v[0], -v[1] }
	x, info := projectedCG(neg, []float64{1, 0}, p, 1, lb, ub)
	if info.stopCond != 3 || math.Abs(vnorm(x)-1) > 1e-12 {
		t.Errorf("negative curvature: %v %+v", x, info)
	}
	if _, info := projectedCG(neg, []float64{1, 0}, p, 0, lb, ub); info.stopCond != 2 || info.niter != 0 {
		t.Errorf("zero radius: %+v", info)
	}
	// Negative curvature along a direction leaving the box at once.
	tight := []float64{0, 0}
	x, info = projectedCG(neg, []float64{1, 0}, p, 1, []float64{0, 0}, tight)
	if info.stopCond != 3 || x[0] != 0 {
		t.Errorf("box-limited negative curvature: %v %+v", x, info)
	}
}

func TestProjectorSVDFailure(t *testing.T) {
	old := svdFunc
	defer func() { svdFunc = old }()
	svdFunc = func(*ndarray.Array, bool) (u, s, vh *ndarray.Array, err error) {
		return nil, nil, nil, errors.New("no convergence")
	}
	p, ok := newProjector(2, 2, []float64{1, 1, 1, 1})
	if !ok || !p.svd || p.k != 0 {
		t.Fatalf("ok %v svd %v k %d", ok, p.svd, p.k)
	}
	z := make([]float64, 2)
	p.nullSpace([]float64{1, 2}, z)
	if z[0] != 1 || z[1] != 2 {
		t.Errorf("null space with no direction: %v", z)
	}
}

// TestBFGSIndefiniteReset: a matrix that rounding made indefinite (sᵀBs <= 0)
// is reset to a scaled identity before the update, as in SciPy. Which
// fixtures reach this depends on the architecture, hence a direct test.
func TestBFGSIndefiniteReset(t *testing.T) {
	h := newBFGSHess(2)
	h.first = false
	h.b = []float64{-1, 0, 0, -1}
	h.update([]float64{1, 0}, []float64{2, 0})
	if !(h.b[0] > 0) || !(h.b[3] > 0) {
		t.Errorf("B = %v, want positive definite", h.b)
	}
}

// TestProjectorZeroRow: a zero row keeps a zero partial norm (no downdate)
// and makes the Jacobian singular (SVD fallback).
func TestProjectorZeroRow(t *testing.T) {
	p, ok := newProjector(2, 3, []float64{1, 2, 0, 0, 0, 0})
	if !ok || !p.svd || p.k != 1 {
		t.Fatalf("ok %v svd %v k %d", ok, p.svd, p.k)
	}
}

func TestBFGSAutoScaleDegenerate(t *testing.T) {
	if s := bfgsAutoScale([]float64{1, 0}, []float64{0, 1}); s != 1 {
		t.Errorf("orthogonal s, y: scale %v", s)
	}
}

// tcQuad is sum (i+1)(x_i-1)² with its gradient and Hessian.
func tcQuad() Problem {
	return Problem{Func: quad, Grad: quadGrad,
		Hess: func(x, h []float64) {
			n := len(x)
			clear(h)
			for i := range n {
				h[i*n+i] = float64(2 * (i + 1))
			}
		}}
}

func TestTrustConstrDerivativeForms(t *testing.T) {
	cons := []ConstraintSpec{Constraint{Kind: Ineq, M: 1, Func: func(x, c []float64) { c[0] = 1 - x[0] - x[1] }}}
	base := tcQuad()
	hessp := Problem{Func: quad, Grad: quadGrad, HessP: func(x, p, hp []float64) {
		for i := range p {
			hp[i] = float64(2*(i+1)) * p[i]
		}
	}}
	fg := Problem{FuncGrad: func(x, g []float64) float64 { quadGrad(x, g); return quad(x) }}
	for name, p := range map[string]Problem{"Hess": base, "HessP": hessp, "FuncGrad": fg} {
		r, err := Minimize(p, []float64{0, 0}, TrustConstr, &Options{Constraints: cons, Bounds: []Bound{{-5, 5}, {-5, 5}},
			Gtol: 1e-9, Xtol: 1e-10, BarrierTol: 1e-10, InitialTrRadius: 2, InitialConstrPenalty: 2,
			InitialBarrierParameter: 0.2, InitialBarrierTolerance: 0.2})
		if err != nil || !r.Success || math.Abs(r.X[0]-1.0/3) > 1e-5 || math.Abs(r.X[1]-2.0/3) > 1e-5 {
			t.Errorf("%s: %v %+v", name, err, r)
		}
		if name != "FuncGrad" && r.Nhev == 0 {
			t.Errorf("%s: no Hessian evaluation counted", name)
		}
		// SciPy's sign: v = -λ for a lower-bounded (c >= 0) constraint.
		if len(r.Multipliers) != 1 || math.Abs(r.Multipliers[0]+4.0/3) > 1e-6 {
			t.Errorf("%s: multipliers %v", name, r.Multipliers)
		}
	}
}

func TestTrustConstrCallbackAndErrors(t *testing.T) {
	calls := 0
	r, err := Minimize(tcQuad(), []float64{0, 0}, TrustConstr, &Options{
		Callback: func(x []float64, f float64) bool { calls++; return calls == 3 }})
	if err != nil || r.Status != 99 || calls != 3 {
		t.Errorf("callback stop: %v %+v", err, r)
	}
	// More equalities than variables with an inequality (interior point).
	f := func(x, c []float64) { c[0], c[1] = x[0]-1, x[0]+1 }
	_, err = Minimize(Problem{Func: quad}, []float64{0}, TrustConstr, &Options{Constraints: []ConstraintSpec{
		Constraint{Kind: Eq, M: 2, Func: f},
		Constraint{Kind: Ineq, M: 1, Func: func(x, c []float64) { c[0] = x[0] }}}})
	if !errors.Is(err, ErrInput) {
		t.Errorf("overdetermined interior point: %v", err)
	}
	if _, err := Minimize(Problem{Func: quad}, []float64{0}, TrustConstr,
		&Options{Constraints: []ConstraintSpec{Constraint{M: 1}}}); err == nil {
		t.Error("invalid constraint accepted")
	}
	if _, err := Minimize(Problem{Func: quad}, []float64{0}, TrustConstr,
		&Options{Constraints: []ConstraintSpec{nil}}); err == nil {
		t.Error("nil constraint accepted")
	}
	if _, err := Minimize(Problem{Func: quad}, []float64{0}, TrustConstr,
		&Options{Constraints: []ConstraintSpec{NonlinearConstraint{M: 1}}}); err == nil {
		t.Error("nonlinear constraint without Func accepted")
	}
	if _, err := Minimize(Problem{Func: quad}, []float64{0}, TrustConstr,
		&Options{Constraints: []ConstraintSpec{LinearConstraint{}}}); err == nil {
		t.Error("linear constraint without A accepted")
	}
}
