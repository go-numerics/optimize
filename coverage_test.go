package optimize

import (
	"errors"
	"math"
	"testing"

	"github.com/go-ndarray/ndarray"
)

// Edge cases and failure modes of Minimize and its helpers that the SciPy
// fixtures do not reach.

func quad2(x []float64) float64 { return float64(x[0]*x[0]) + float64(4*float64(x[1]*x[1])) }

func quad2Grad(x, g []float64) { g[0], g[1] = 2*x[0], 8*x[1] }

func TestMinimizeInputErrors(t *testing.T) {
	p := Problem{Func: quad2}
	cases := []struct {
		p  Problem
		x0 []float64
		m  Method
		o  *Options
	}{
		{Problem{}, []float64{1, 1}, BFGS, nil},
		{p, nil, BFGS, nil},
		{p, []float64{1, 1}, "nope", nil},
		{p, []float64{1, 1}, BFGS, &Options{Bounds: []Bound{{0, 1}}}},
		{p, []float64{1, 1}, BFGS, &Options{Bounds: []Bound{{0, 1}, {0, 1}}}},
		{p, []float64{1, 1}, NewtonCG, nil},
		{p, []float64{1, 1}, BFGS, &Options{C1: 0.95, C2: 0.9}},
		{p, []float64{1, 1}, CG, &Options{C1: 0.5, C2: 0.4}},
		{Problem{Func: quad2, Grad: quad2Grad}, []float64{1, 1}, NewtonCG, &Options{C2: 2}},
		{p, []float64{1, 1}, BFGS, &Options{HessInv0: []float64{1}}},
		{p, []float64{1, 1}, NelderMead, &Options{InitialSimplex: [][]float64{{0, 0}}}},
		{p, []float64{1, 1}, NelderMead, &Options{InitialSimplex: [][]float64{{0, 0}, {1}, {0, 1}}}},
		{p, []float64{1, 1}, Powell, &Options{Direc: [][]float64{{1, 0}}}},
		{p, []float64{1, 1}, Powell, &Options{Direc: [][]float64{{1, 0}, {1}}}},
		{p, []float64{1, 1}, NelderMead, &Options{Bounds: []Bound{{1, 0}, {0, 1}}}},
	}
	for i, c := range cases {
		if _, err := Minimize(c.p, c.x0, c.m, c.o); err == nil {
			t.Errorf("case %d: no error", i)
		}
	}
	if _, _, err := splitBounds([]Bound{{0, 1}}, 2); !errors.Is(err, ErrDimension) {
		t.Error(err)
	}
}

func TestMinimizeOptionsAndStops(t *testing.T) {
	x0 := []float64{1.5, -0.7}
	grad := Problem{Func: quad2, Grad: quad2Grad}
	// FuncGrad only, every gradient method.
	fg := Problem{FuncGrad: func(x, g []float64) float64 { quad2Grad(x, g); return quad2(x) }}
	for _, m := range []Method{NelderMead, Powell, CG, BFGS, NewtonCG} {
		r, err := Minimize(fg, x0, m, &Options{Tol: 1e-10})
		if err != nil || !r.Success || r.Fun > 1e-8 {
			t.Errorf("%s FuncGrad: %v %+v", m, err, r)
		}
	}
	// Finite-difference variants.
	for _, o := range []*Options{
		{FiniteDiffRelStep: 1e-7}, {Eps: 1e-7}, {FDMethod: FD3Point}, {FDMethod: FD3Point, Eps: 1e-6},
	} {
		if r, err := Minimize(Problem{Func: quad2}, x0, BFGS, o); err != nil || r.Fun > 1e-8 {
			t.Errorf("FD %+v: %v %+v", o, err, r)
		}
	}
	// BFGS with an initial inverse Hessian and a relative step stop.
	r, err := Minimize(grad, x0, BFGS, &Options{HessInv0: []float64{0.5, 0, 0, 0.125}})
	if err != nil || !r.Success || r.HessInvArray() == nil || r.XArray().Size() != 2 {
		t.Errorf("HessInv0: %v %+v", err, r)
	}
	if r, _ := Minimize(grad, x0, BFGS, &Options{XRtol: 10}); r.Nit != 1 || !r.Success {
		t.Errorf("XRtol: %+v", r)
	}
	if (&Result{}).HessInvArray() != nil {
		t.Error("nil HessInv")
	}
	// Callback stops every method with status 99.
	for _, m := range []Method{NelderMead, Powell, CG, BFGS, NewtonCG} {
		n := 0
		r, err := Minimize(grad, x0, m, &Options{Callback: func(x []float64, f float64) bool { n++; return n == 2 }})
		if err != nil || r.Status != 99 || r.Success || n != 2 {
			t.Errorf("%s callback: %v %+v", m, err, r)
		}
	}
	// Iteration limits.
	if r, _ := Minimize(grad, []float64{-1.2, 1}, NewtonCG, &Options{MaxIter: 1, Tol: 1e-30}); r.Status != 1 {
		t.Errorf("Newton-CG maxiter: %+v", r)
	}
	ros := Problem{Func: rosen}
	if r, _ := Minimize(ros, []float64{-1.2, 1}, NelderMead, &Options{MaxIter: 5}); r.Status != 2 {
		t.Errorf("NM maxiter: %+v", r)
	}
	if r, _ := Minimize(ros, []float64{-1.2, 1}, NelderMead, &Options{MaxIter: 500, MaxFev: 40}); r.Status != 1 {
		t.Errorf("NM both limits: %+v", r)
	}
	if r, _ := Minimize(ros, []float64{-1.2, 1}, Powell, &Options{MaxFev: 30}); r.Status != 1 {
		t.Errorf("Powell maxfev: %+v", r)
	}
	if r, _ := Minimize(ros, []float64{-1.2, 1}, Powell, &Options{MaxFev: 1}); r.Status != 1 {
		t.Errorf("Powell maxfev 1: %+v", r)
	}
	if r, _ := Minimize(Problem{FuncGrad: func(x, g []float64) float64 { return rosen(x) }}, []float64{-1.2, 1}, NelderMead, &Options{MaxFev: 30}); r.Status != 1 {
		t.Errorf("NM FuncGrad maxfev: %+v", r)
	}
	// Nelder–Mead with an explicit simplex, bounds that reflect it, and
	// the adaptive coefficients shrinking.
	sim := [][]float64{{2, 2}, {2.6, 2}, {2, 2.4}}
	if r, _ := Minimize(Problem{Func: quad2}, []float64{2, 2}, NelderMead, &Options{InitialSimplex: sim, Bounds: []Bound{{-1, 2.5}, {-1, 2.5}}}); !r.Success || r.Fun > 1e-6 {
		t.Errorf("NM simplex+bounds: %+v", r)
	}
	abs := Problem{Func: func(x []float64) float64 { return math.Abs(x[0]) + math.Abs(x[1]-1) }}
	if r, _ := Minimize(abs, []float64{3, -2}, NelderMead, &Options{Adaptive: true}); !r.Success || r.Fun > 1e-3 {
		t.Errorf("NM abs: %+v", r)
	}
	// Powell: given directions, half-bounded line searches, zero
	// directions, NaN objective.
	if r, _ := Minimize(Problem{Func: quad2}, x0, Powell, &Options{Direc: [][]float64{{1, 1}, {1, -1}}}); !r.Success {
		t.Errorf("Powell direc: %+v", r)
	}
	if r, _ := Minimize(Problem{Func: quad2}, []float64{3, 3}, Powell, &Options{Bounds: []Bound{{1, math.Inf(1)}, Unbounded()}}); !r.Success || math.Abs(r.X[0]-1) > 1e-4 {
		t.Errorf("Powell half bound: %+v", r)
	}
	if r, _ := Minimize(Problem{Func: quad2}, []float64{3, 3}, Powell, &Options{Direc: [][]float64{{0, 0}, {0, 1}}}); r.X[0] != 3 {
		t.Errorf("Powell zero direction: %+v", r)
	}
	nan := Problem{Func: func([]float64) float64 { return math.NaN() }}
	for _, m := range []Method{Powell, BFGS, CG} {
		if r, _ := Minimize(nan, x0, m, nil); r.Success {
			t.Errorf("%s NaN: %+v", m, r)
		}
	}
	// A gradient inconsistent with the function: the line searches fail.
	liar := Problem{Func: quad2, Grad: func(x, g []float64) { g[0], g[1] = -2*x[0], -8*x[1] }}
	for _, m := range []Method{BFGS, CG, NewtonCG} {
		if r, _ := Minimize(liar, x0, m, nil); r.Success || r.Status != 2 {
			t.Errorf("%s liar: %+v", m, r)
		}
	}
	// A function that turns NaN away from x0.
	cliff := Problem{Func: func(x []float64) float64 {
		if math.Abs(x[0]) < 1 {
			return math.NaN()
		}
		return quad2(x)
	}, Grad: quad2Grad}
	for _, m := range []Method{BFGS, CG, NewtonCG} {
		if r, _ := Minimize(cliff, []float64{3, 1}, m, nil); r.Success {
			t.Errorf("%s cliff: %+v", m, r)
		}
	}
	// Newton-CG: negative curvature (first and later CG steps) and CG
	// running out of iterations on an indefinite Hessian.
	saddle := Problem{Func: func(x []float64) float64 { return float64(x[0]*x[0]) - float64(x[1]*x[1]) + math.Pow(x[1], 4) },
		Grad: func(x, g []float64) { g[0], g[1] = 2*x[0], -2*x[1]+4*math.Pow(x[1], 3) }}
	if r, _ := Minimize(saddle, []float64{1, 0.1}, NewtonCG, nil); !r.Success || math.Abs(math.Abs(r.X[1])-math.Sqrt(0.5)) > 1e-3 {
		t.Errorf("Newton-CG saddle: %+v", r)
	}
	if r, _ := Minimize(saddle, []float64{1, 0.1}, NewtonCG, &Options{Tol: 1e-12}); !r.Success {
		t.Errorf("Newton-CG saddle tol: %+v", r)
	}
}

func TestDcsrchEdges(t *testing.T) {
	for _, d := range []dcsrch{
		{ftol: 1e-3, gtol: 0.9, xtol: 0.1, stpmin: 1, stpmax: 0},
		{ftol: 1e-3, gtol: 0.9, xtol: 0.1, stpmin: -1, stpmax: 1},
		{ftol: 1e-3, gtol: 0.9, xtol: -1, stpmax: 1},
		{ftol: 1e-3, gtol: -1, xtol: 0.1, stpmax: 1},
		{ftol: -1, gtol: 0.9, xtol: 0.1, stpmax: 1},
	} {
		if _, task := d.iterate(0.5, 0, -1, lsStart); task != lsError || d.msg == "" {
			t.Errorf("%+v: %v", d, task)
		}
	}
	d := dcsrch{ftol: 1e-3, gtol: 0.9, xtol: 0.1, stpmin: 0.1, stpmax: 1}
	for _, c := range []struct{ stp, g float64 }{{0.5, 1}, {2, -1}, {0.01, -1}} {
		if _, task := d.iterate(c.stp, 0, c.g, lsStart); task != lsError {
			t.Errorf("start %+v: %v", c, task)
		}
	}
	// stp = stpmax with sufficient decrease and a still-negative slope.
	d = dcsrch{ftol: 1e-3, gtol: 0.1, xtol: 0.1, stpmin: 0, stpmax: 1}
	stp, task := d.iterate(1, 0, -1, lsStart)
	if _, task = d.iterate(stp, -1, -0.9, task); task != lsWarn || d.msg != "WARNING: STP = STPMAX" {
		t.Errorf("stpmax: %v %q", task, d.msg)
	}
	// stp = stpmin without decrease.
	d = dcsrch{ftol: 1e-3, gtol: 0.1, xtol: 0.1, stpmin: 0.5, stpmax: 1}
	stp, task = d.iterate(0.5, 0, -1, lsStart)
	if _, task = d.iterate(stp, 1, 1, task); task != lsWarn || d.msg != "WARNING: STP = STPMIN" {
		t.Errorf("stpmin: %v %q", task, d.msg)
	}
	// A wide xtol ends the search once the minimiser is bracketed.
	d = dcsrch{ftol: 1e-3, gtol: 1e-6, xtol: 0.99, stpmin: 0, stpmax: 10}
	f := func(s float64) (float64, float64) { return math.Pow(s-1, 4), 4 * math.Pow(s-1, 3) }
	stp, task = d.iterate(4, 1, -2, lsStart)
	for range 20 {
		fv, gv := f(stp)
		if stp, task = d.iterate(stp, fv, gv, task); task != lsFG {
			break
		}
	}
	if task != lsWarn || d.msg != "WARNING: XTOL TEST SATISFIED" {
		t.Errorf("xtol: %v %q", task, d.msg)
	}
	if sign(0) != 0 || sign(-2) != -1 {
		t.Error("sign")
	}
	// dcstep branches the searches above do not visit.
	stx, fx, dx, sty, fy, dy := 0.0, 0.0, -1.0, 0.0, 0.0, -1.0
	br := false
	// Case 3, not bracketed, cubic without a minimiser below stx.
	if s := dcstep(&stx, &fx, &dx, &sty, &fy, &dy, -1, -0.5, -0.9, &br, -5, 5); s != -5 && s != 5 {
		t.Logf("case 3 step %g", s)
	}
	stx, fx, dx, sty, fy, dy, br = 1, 0, 1, 3, 1, 2, true
	dcstep(&stx, &fx, &dx, &sty, &fy, &dy, 0.5, -0.2, 0.5, &br, 0, 5)
	stx, fx, dx, sty, fy, dy, br = 1, 0, -1, 3, 1, 2, true
	dcstep(&stx, &fx, &dx, &sty, &fy, &dy, 4, -0.5, -1, &br, 0, 5)
	stx, fx, dx, br = 1, 0, -1, false
	if s := dcstep(&stx, &fx, &dx, &sty, &fy, &dy, 0.5, -0.5, -1, &br, 0.1, 5); s != 0.1 {
		t.Errorf("case 4 backwards: %g", s)
	}
	stx, fx, dx, br = 1, 0, -1, false
	if s := dcstep(&stx, &fx, &dx, &sty, &fy, &dy, 0.5, -0.1, -0.5, &br, 0.1, 5); s != 0.1 {
		t.Logf("case 3 backwards: %g", s)
	}
}

func TestLineSearchEdges(t *testing.T) {
	if a := initialStep(1, 0, 1, true); a != 1 {
		t.Errorf("initialStep %g", a)
	}
	if _, ok := cubicMin(0, 0, -1, 1, 1, 1, 2); ok {
		t.Error("cubic with a repeated point")
	}
	if _, ok := cubicMin(0, 0, -1, 1, 0, 2, 4); !ok {
		t.Error("cubic")
	}
	if _, ok := cubicMin(0, 0, 1, 1, 1, 2, 6); ok {
		t.Error("cubic without a real minimiser")
	}
	if _, ok := cubicMin(0, math.NaN(), -1, 1, 1, 2, 4); ok {
		t.Error("cubic overflow")
	}
	if _, ok := quadMin(0, 0, -1, 0, 1); ok {
		t.Error("quad degenerate")
	}
	if _, ok := quadMin(0, 0, -1, 1, -1); ok {
		t.Error("quad flat")
	}
	if _, ok := quadMin(0, math.NaN(), -1, 1, 1); ok {
		t.Error("quad overflow")
	}
	// wolfe2 on its own: expansion steps, then the bracket, a step that
	// cannot grow (amax reached), and the 10-step limit.
	sf := newScalarFunc(Problem{Func: func(x []float64) float64 { return -x[0] + float64(1e-3*float64(x[0]*x[0])) },
		Grad: func(x, g []float64) { g[0] = -1 + 2e-3*x[0] }}, BFGS, &Options{}, nil, nil)
	ls := newLineSearch(sf, 1)
	ls.xk, ls.pk = []float64{0}, []float64{1}
	if a, _, ok := ls.wolfe2(0, 0, -1, false, 1e-4, 0.1, 1e10); !ok || a < 100 {
		t.Errorf("wolfe2 expansion: %g %v", a, ok)
	}
	if _, _, ok := ls.wolfe2(0, 0, -1, false, 1e-4, 0.1, 2); ok {
		t.Error("wolfe2 amax")
	}
	lin := newScalarFunc(Problem{Func: func(x []float64) float64 { return -x[0] },
		Grad: func(x, g []float64) { g[0] = -1 }}, BFGS, &Options{}, nil, nil)
	ll := newLineSearch(lin, 1)
	ll.xk, ll.pk = []float64{0}, []float64{1}
	if a, _, ok := ll.wolfe2(0, 0, -1, false, 1e-4, 0.5, 1e300); !ok || a != 1024 {
		t.Errorf("wolfe2 limit: %g %v", a, ok)
	}
	if _, _, ok := ll.wolfe1(0, 0, -1, false, 1e-4, 0.9, 1e-100, 1e300); ok {
		t.Error("wolfe1 unbounded")
	}
	// wolfe1 meeting a non-finite step.
	nanf := newScalarFunc(Problem{Func: func(x []float64) float64 {
		if x[0] > 2 {
			return math.NaN()
		}
		return -x[0]
	}, Grad: func(x, g []float64) {
		g[0] = -1
		if x[0] > 2 {
			g[0] = math.NaN()
		}
	}}, BFGS, &Options{}, nil, nil)
	ln := newLineSearch(nanf, 1)
	ln.xk, ln.pk = []float64{0}, []float64{1}
	if _, _, ok := ln.wolfe1(0, 0, -1, false, 1e-4, 0.9, 1e-100, 1e300); ok {
		t.Error("wolfe1 NaN")
	}
	// zoom: phi rises everywhere, the interpolants fail, the limit hits.
	wall := newScalarFunc(Problem{Func: func(x []float64) float64 { return math.Abs(x[0] - 0.3) },
		Grad: func(x, g []float64) { g[0] = sign(x[0] - 0.3) }}, BFGS, &Options{}, nil, nil)
	lz := newLineSearch(wall, 1)
	lz.xk, lz.pk = []float64{0}, []float64{1}
	lz.zoom(0, 1, 0.3, 0.7, -1, 0.3, -1, 1e-4, 1e-9, func(float64, float64) bool { return true })
	lz.zoom(1, 0, 0.7, 0.3, 1, 0.3, -1, 1e-4, 1e-9, func(float64, float64) bool { return true })
}

func TestNumdiffEdges(t *testing.T) {
	f := func(x []float64) float64 { return float64(x[0]*x[0]*x[0]) + x[1] }
	want := []float64{3, 1}
	x := []float64{1, 2}
	for _, s := range []fdStep{
		{method: FD3Point},
		{method: FD3Point, lb: []float64{1, 0}, ub: []float64{3, 3}},  // one-sided forward
		{method: FD3Point, lb: []float64{-3, 0}, ub: []float64{1, 3}}, // one-sided backward
		{method: FD3Point, lb: []float64{1 - 1e-9, 0}, ub: []float64{1 + 2e-9, 3}},
		{method: FD2Point, lb: []float64{0, 0}, ub: []float64{1, 3}}, // flipped
		{method: FD2Point, lb: []float64{1 - 1e-12, 0}, ub: []float64{1 + 2e-12, 3}},
		{method: FD2Point, lb: []float64{1 - 2e-12, 0}, ub: []float64{1 + 1e-12, 3}},
		{method: FD2Point, rel: 1e-8},
		{method: FD2Point, abs: 1e-300},
		{method: FD2Point, lb: []float64{math.Inf(-1), math.Inf(-1)}, ub: []float64{math.Inf(1), math.Inf(1)}},
	} {
		g := make([]float64, 2)
		s.gradFD(f, x, f(x), g)
		for i := range g {
			if math.Abs(g[i]-want[i]) > 1e-3 {
				t.Errorf("%+v: grad %v", s, g)
				break
			}
		}
		J := make([]float64, 2)
		s.jacFD(func(x, out []float64) { out[0] = f(x) }, x, []float64{f(x)}, J)
	}
	g := ApproxFprime(f, x, 0)
	if math.Abs(g[0]-3) > 1e-6 {
		t.Errorf("ApproxFprime %v", g)
	}
	if e := CheckGrad(f, func(x, g []float64) { g[0], g[1] = 3*x[0]*x[0], 1 }, x, 1e-7); e > 1e-5 {
		t.Errorf("CheckGrad %g", e)
	}
}

func TestTypesHelpers(t *testing.T) {
	v := []float64{3, -4}
	if norm(v, math.Inf(1)) != 4 || norm(v, math.Inf(-1)) != 3 || norm(v, 2) != 5 || math.Abs(norm(v, 1)-7) > 1e-15 {
		t.Error("norm")
	}
	if !math.IsInf(enorm([]float64{1, math.Inf(1)}), 1) || enorm([]float64{0, 0}) != 0 {
		t.Error("enorm")
	}
	if vecArray(nil) != nil || matArray(nil, 1, 1) != nil || arrayData(nil) != nil {
		t.Error("nil helpers")
	}
	ints, _ := ndarray.FromSlice([]int64{1, 2}, 2)
	if d := arrayData(ints); d[1] != 2 {
		t.Error("arrayData int")
	}
	if (&LSResult{X: []float64{1}, Fun: []float64{1, 2}, Jac: []float64{1, 2}}).JacArray().Size() != 2 ||
		(&LSResult{X: []float64{1}}).XArray().Size() != 1 {
		t.Error("LSResult accessors")
	}
}

func TestLeastSquaresInputErrors(t *testing.T) {
	p := LSProblem{Func: func(x, f []float64) { f[0], f[1] = x[0]-1, x[1]-2 }, M: 2}
	x0 := []float64{0, 0}
	bad := []struct {
		p  LSProblem
		x0 []float64
		m  LSMethod
		o  *LSOptions
	}{
		{LSProblem{}, x0, "", nil},
		{p, x0, "", &LSOptions{FScale: -1}},
		{p, x0, "", &LSOptions{XScale: []float64{1}}},
		{p, x0, "", &LSOptions{XScale: []float64{1, 0}}},
		{p, x0, "", &LSOptions{Bounds: []Bound{{0, 1}}}},
		{p, x0, "", &LSOptions{Bounds: []Bound{{0, 0}, {0, 1}}}},
		{p, x0, "", &LSOptions{Bounds: []Bound{{1, 2}, {0, 1}}}},
		{p, x0, LM, &LSOptions{Bounds: []Bound{{-1, 2}, {-1, 1}}}},
		{p, x0, LM, &LSOptions{Loss: LossHuber}},
		{LSProblem{Func: p.Func, M: 1}, x0, LM, nil},
		{p, x0, "dogbox", nil},
	}
	for i, c := range bad {
		if _, err := LeastSquares(c.p, c.x0, c.m, c.o); err == nil {
			t.Errorf("case %d: no error", i)
		}
	}
	if r, err := LeastSquares(p, x0, "", &LSOptions{Ftol: -1}); err != nil || math.Abs(r.X[1]-2) > 1e-8 {
		t.Errorf("default method: %v %+v", err, r)
	}
}

func TestMoreEdges(t *testing.T) {
	// BFGS at a NaN start where f and ∇f stay finite.
	flat := Problem{Func: func([]float64) float64 { return 0 }, Grad: func(_, g []float64) { g[0] = 0 }}
	if r, _ := Minimize(flat, []float64{math.NaN()}, BFGS, nil); r.Status != 3 {
		t.Errorf("BFGS NaN x: %+v", r)
	}
	if _, err := Minimize(flat, []float64{1}, BFGS, &Options{MaxIter: -1}); err == nil {
		t.Error("negative MaxIter")
	}
	if a := initialStep(1, 0, -1, true); a != 1 {
		t.Errorf("initialStep rising: %g", a)
	}
	// wolfe2 with no room to move.
	ls := newLineSearch(newScalarFunc(Problem{Func: quad2, Grad: quad2Grad}, BFGS, &Options{}, nil, nil), 2)
	ls.xk, ls.pk = []float64{1, 1}, []float64{-1, -1}
	if _, _, ok := ls.wolfe2(5, 0, -10, false, 1e-4, 0.9, 0); ok {
		t.Error("wolfe2 amax 0")
	}
	// wolfe1 where f jumps to +Inf: the cubic step becomes NaN.
	inf := newScalarFunc(Problem{Func: func(x []float64) float64 {
		if x[0] > 0.5 {
			return math.Inf(1)
		}
		return float64(x[0]*x[0]) - 2*x[0]
	}, Grad: func(x, g []float64) { g[0] = 2*x[0] - 2 }}, BFGS, &Options{}, nil, nil)
	li := newLineSearch(inf, 1)
	li.xk, li.pk = []float64{0}, []float64{1}
	if _, _, ok := li.wolfe1(0, 0, -2, false, 1e-4, 0.9, 1e-100, 1e300); ok {
		t.Error("wolfe1 NaN step")
	}
	// L-BFGS-B with every variable fixed, and an option error on the
	// reduced problem.
	if r, err := Minimize(Problem{Func: quad2}, []float64{0, 0}, LBFGSB, &Options{Bounds: []Bound{{1, 1}, {2, 2}}}); err != nil || r.Fun != 17 || !r.Success {
		t.Errorf("all fixed: %v %+v", err, r)
	}
	if _, err := Minimize(Problem{Func: quad2}, []float64{0, 0}, LBFGSB, &Options{Bounds: []Bound{{1, 1}, {-2, 2}}, M: -1}); err == nil {
		t.Error("reduced problem error")
	}
	// A panic other than the budget's passes through.
	func() {
		defer func() {
			if recover() != "boom" {
				t.Error("panic swallowed")
			}
		}()
		Minimize(Problem{Func: func([]float64) float64 { panic("boom") }}, []float64{1}, NelderMead, nil)
	}()
	// Nelder–Mead shrink steps.
	for _, f := range []func([]float64) float64{
		func(x []float64) float64 { return math.Abs(x[0]) + math.Abs(x[1]) },
		func(x []float64) float64 { return math.Max(math.Abs(x[0]), math.Abs(x[1])) },
		func(x []float64) float64 { return math.Floor(4*x[0])*math.Floor(4*x[0]) + math.Abs(x[1]) },
		func(x []float64) float64 { return math.Sin(1000*x[0]) * math.Cos(1000*x[1]) },
	} {
		Minimize(Problem{Func: f}, []float64{1, 1}, NelderMead, &Options{Bounds: []Bound{{-3, 3}, {-3, 3}}})
		Minimize(Problem{Func: f}, []float64{0.3, -0.7}, NelderMead, nil)
	}
	if lo, hi := lineForSearch([]float64{5, 0}, []float64{1, 1}, []float64{0, 0}, []float64{1, 1}); lo != 0 || hi != 0 {
		t.Errorf("empty line: %g %g", lo, hi)
	}
	// Bracket: the parabolic point lands between xb and xc above f(xb).
	spike := func(x float64) float64 {
		switch {
		case x == 0:
			return 2
		case x == 1:
			return 1
		case math.Abs(x-2.618034) < 1e-9:
			return 0.5
		}
		return 10
	}
	if x, _, _, err := Bracket(spike, 0, 1); err != nil || x[2] < 2.3 || x[2] > 2.5 {
		t.Errorf("bracket spike: %v %v", x, err)
	}
	// Brent on a function that turns NaN inside its bracket.
	holey := func(x float64) float64 {
		if x == 0 || x == 1 || x == 2 {
			return (x - 1) * (x - 1)
		}
		return math.NaN()
	}
	if r, _ := MinimizeScalar(holey, Brent, &ScalarOptions{Bracket: []float64{0, 1, 2}}); r.Success || r.Message != msgNaN {
		t.Errorf("brent NaN inside: %+v", r)
	}
	// Curve fitting: a redundant parameter (rank-deficient Jacobian) and an
	// invalid evaluation budget.
	x, _ := ndarray.Linspace(0, 1, 10)
	y := x.MulScalar(2)
	red := func(x float64, p []float64) float64 { return (p[0] + p[1]) * x }
	if r, err := CurveFit(red, x, y, []float64{1, 1}, &CurveFitOptions{Method: TRF}); err != nil || math.Abs(r.Popt[0]+r.Popt[1]-2) > 1e-6 {
		t.Errorf("redundant: %v %+v", err, r)
	}
	if _, err := CurveFit(red, x, y, []float64{1, 1}, &CurveFitOptions{MaxFev: -1}); err == nil {
		t.Error("LM negative MaxFev")
	}
	if _, err := CurveFit(red, x, y, []float64{1, 1}, &CurveFitOptions{MaxFev: -1, Method: TRF}); err == nil {
		t.Error("TRF negative MaxFev")
	}
	// Parallel model evaluation gives the same fit, with and without an
	// analytic gradient; the tall-Jacobian QR path of TRF and its
	// covariance agree with the direct SVD.
	xl, _ := ndarray.Linspace(0, 4, 3000)
	yl := xl.Map(func(v float64) float64 { return 2.5*math.Exp(-1.3*v) + 0.5 + 0.01*math.Sin(97*v) })
	for _, mod := range []Model{{F: cfExp}, {F: cfExp, Grad: cfExpGrad}} {
		var ref *CurveFitResult
		for _, w := range []int{1, 4} {
			r, err := CurveFitModel(mod, xl, yl, []float64{1, 1, 1}, &CurveFitOptions{Method: TRF, Workers: w})
			if err != nil {
				t.Fatal(err)
			}
			if ref == nil {
				ref = r
				continue
			}
			for i := range r.Popt {
				if r.Popt[i] != ref.Popt[i] || r.Pcov[i] != ref.Pcov[i] {
					t.Errorf("workers changed the fit: %v %v", r.Popt, ref.Popt)
				}
			}
		}
		saved := trfQRMin
		trfQRMin = 1 << 30
		d, err := CurveFitModel(mod, xl, yl, []float64{1, 1, 1}, &CurveFitOptions{Method: TRF})
		trfQRMin = saved
		if err != nil {
			t.Fatal(err)
		}
		for i := range d.Popt {
			if math.Abs(d.Popt[i]-ref.Popt[i]) > 1e-9 || math.Abs(d.Pcov[i*4]-ref.Pcov[i*4]) > 1e-6*math.Abs(ref.Pcov[i*4]) {
				t.Errorf("QR path %v %v, direct %v %v", ref.Popt, ref.Pcov, d.Popt, d.Pcov)
			}
		}
	}
	tall := make([]float64, 2*trfQRMin)
	tall[3] = math.Inf(1)
	if _, _, _, err := trfSVDf(tall, make([]float64, trfQRMin), trfQRMin, 2, nil); err == nil {
		t.Error("non-finite tall Jacobian")
	}
	nan := []float64{math.NaN(), 1, 1, 1}
	if pinvNormal(nan, 2, 2) != nil {
		t.Error("SVD of a NaN matrix")
	}
}
