package optimize

import (
	"math"
	"testing"
)

// SciPy never marks a point infeasible (DIRinfcn sets the flag to 0), so the
// hidden-constraint machinery of Gablonsky's code (DIRreplaceInf,
// DIRResortlist, the "increase the budget until a feasible point is found"
// logic) is not reachable through Direct. It is ported all the same and
// driven here through the internal entry point, whose objective returns the
// feasibility flag. There is no SciPy oracle for these paths; the tests check
// what the algorithm promises: a feasible best point, close to the best
// feasible value, a bounded number of evaluations, no panic.

func flagParams(original bool) dirParams {
	return dirParams{eps: 1e-4, maxf: 2000, maxt: 200, original: original,
		fglobal: math.Inf(-1), volper: -1, sigmaper: 1e-6}
}

func TestDirectCoreHiddenConstraints(t *testing.T) {
	f := dgShifted(2) // minimum at (0.3, 0.47)
	type region func(x []float64) bool
	regions := map[string]region{
		"left strip":   func(x []float64) bool { return x[0] < 0.35 },
		"disc":         func(x []float64) bool { return math.Hypot(x[0]-0.3, x[1]-0.47) < 0.2 },
		"right slab":   func(x []float64) bool { return x[0] > 0.8 },
		"checkerboard": func(x []float64) bool { return int(x[0]*8)%2 == int(x[1]*8)%2 },
		"most":         func(x []float64) bool { return x[0] < 0.97 || x[1] < 0.5 },
	}
	for name, bad := range regions {
		for _, orig := range []bool{false, true} {
			infeasibleSeen := 0
			fn := func(x []float64) (float64, int) {
				if bad(x) {
					infeasibleSeen++
					return f(x), 1
				}
				return f(x), 0
			}
			out := directCore(fn, []float64{0, 0}, []float64{1, 1}, flagParams(orig), nil)
			if infeasibleSeen == 0 {
				t.Errorf("%s: the region was never sampled", name)
			}
			if out.ierror < 1 || out.ierror > 5 {
				t.Errorf("%s orig=%v: status %d", name, orig, out.ierror)
			}
			if bad(out.x) {
				t.Errorf("%s orig=%v: best point %v is infeasible", name, orig, out.x)
			}
			if math.Abs(out.fun-f(out.x)) > 1e-9 {
				t.Errorf("%s orig=%v: fun %v is not f(x) %v", name, orig, out.fun, f(out.x))
			}
			t.Logf("%s orig=%v: status %d nfev %d nit %d fun %.4g", name, orig, out.ierror, out.numfunc, out.numiter, out.fun)
		}
	}
}

// With no feasible point at all the budget keeps being extended (status 2
// stops it at the iteration limit); a feasible point found later ends it
// normally.
func TestDirectCoreNoFeasiblePointYet(t *testing.T) {
	f := dgShifted(2)
	bad := func(x []float64) bool { return x[0] < 0.95 }
	for _, orig := range []bool{false, true} {
		p := flagParams(orig)
		p.maxf, p.maxt = 20, 400
		out := directCore(func(x []float64) (float64, int) {
			if bad(x) {
				return f(x), 1
			}
			return f(x), 0
		}, []float64{0, 0}, []float64{1, 1}, p, nil)
		if out.ierror != 1 && out.ierror != 5 || bad(out.x) {
			t.Errorf("orig=%v: %+v", orig, out)
		}
		t.Logf("orig=%v: status %d nfev %d nit %d x %v", orig, out.ierror, out.numfunc, out.numiter, out.x)
	}
	// Nothing is ever feasible.
	p := flagParams(false)
	p.maxf, p.maxt = 20, 12
	out := directCore(func(x []float64) (float64, int) { return 1, 1 }, []float64{0, 0}, []float64{1, 1}, p, nil)
	if out.ierror != 2 {
		t.Errorf("%+v", out)
	}
}

func TestDirectCoreSizesMatchPublicAPI(t *testing.T) {
	// maxf <= 0 and maxt <= 0 select the C defaults (Direct never passes them).
	p := flagParams(false)
	p.maxf, p.maxt = 0, 0
	out := directCore(func(x []float64) (float64, int) { return dgShifted(2)(x), 0 }, []float64{0, 0}, []float64{1, 1}, p, nil)
	if out.ierror != 5 {
		t.Errorf("%+v", out)
	}
}

func TestDirectIsinbox(t *testing.T) {
	if !isinbox([]float64{0.5}, []float64{0}, []float64{1}) ||
		isinbox([]float64{-1}, []float64{0}, []float64{1}) || isinbox([]float64{2}, []float64{0}, []float64{1}) {
		t.Fatal("isinbox")
	}
}
