package optimize

import "math"

// lineSearch evaluates φ(α) = f(xk + α·pk) and φ'(α) = ∇f(xk + α·pk)·pk
// through a scalarFunc, keeping the gradient of the last φ' call (which is
// the gradient at the accepted step when the search succeeds).
type lineSearch struct {
	sf     *scalarFunc
	xk, pk []float64
	x, g   []float64 // scratch point and last gradient
	// extra is an optional additional acceptance test on (α, x, f, g),
	// SciPy's extra_condition (used by CG).
	extra func(alpha float64, x []float64, f float64, g []float64) bool
}

func newLineSearch(sf *scalarFunc, n int) *lineSearch {
	return &lineSearch{sf: sf, x: make([]float64, n), g: make([]float64, n)}
}

func (ls *lineSearch) point(alpha float64) []float64 {
	for i := range ls.x {
		ls.x[i] = ls.xk[i] + alpha*ls.pk[i]
	}
	return ls.x
}

func (ls *lineSearch) phi(alpha float64) float64 { return ls.sf.fun(ls.point(alpha)) }

func (ls *lineSearch) derphi(alpha float64) float64 {
	ls.sf.grad(ls.point(alpha), ls.g)
	return dot(ls.g, ls.pk)
}

// initialStep is the first trial step of SciPy's scalar searches, based on
// the decrease of the previous iteration.
func initialStep(phi0, oldPhi0, derphi0 float64, haveOld bool) float64 {
	if haveOld && derphi0 != 0 {
		a := math.Min(1, 1.01*2*(phi0-oldPhi0)/derphi0)
		if a < 0 {
			a = 1
		}
		return a
	}
	return 1
}

// wolfe12 is SciPy's _line_search_wolfe12: the Moré–Thuente search
// (line_search_wolfe1) and, if it fails or the extra condition rejects its
// step, the bracketing/zoom search of Nocedal & Wright (line_search_wolfe2).
// gfk is ∇f(xk) and fk = f(xk). It returns the step, f at the step, and ok;
// on success ls.g holds the gradient at the step. amin and amax bound the
// first search, amax2 the second (SciPy passes Newton-CG's amax to the first
// only).
func (ls *lineSearch) wolfe12(xk, pk, gfk []float64, fk, oldOld float64, haveOld bool, c1, c2, amin, amax, amax2 float64) (alpha, fNew float64, ok bool) {
	ls.xk, ls.pk = xk, pk
	derphi0 := dot(gfk, pk)
	alpha, fNew, ok = ls.wolfe1(fk, oldOld, derphi0, haveOld, c1, c2, amin, amax)
	if ok && ls.extra != nil && !ls.extra(alpha, ls.point(alpha), fNew, ls.g) {
		ok = false
	}
	if !ok {
		alpha, fNew, ok = ls.wolfe2(fk, oldOld, derphi0, haveOld, c1, c2, amax2)
	}
	return alpha, fNew, ok
}

// wolfe1 is scalar_search_wolfe1: dcsrch driven for at most 100 steps.
func (ls *lineSearch) wolfe1(phi0, oldPhi0, derphi0 float64, haveOld bool, c1, c2, amin, amax float64) (float64, float64, bool) {
	stp := initialStep(phi0, oldPhi0, derphi0, haveOld)
	d := dcsrch{ftol: c1, gtol: c2, xtol: 1e-14, stpmin: amin, stpmax: amax}
	f, g := phi0, derphi0
	task := lsStart
	for range 100 {
		stp, task = d.iterate(stp, f, g, task)
		if math.IsNaN(stp) || math.IsInf(stp, 0) {
			return 0, f, false
		}
		if task != lsFG {
			return stp, f, task == lsConv
		}
		f = ls.phi(stp)
		g = ls.derphi(stp)
	}
	return 0, f, false
}

// wolfe2 is scalar_search_wolfe2 (Nocedal & Wright, Algorithm 3.5) with at
// most 10 bracketing steps.
func (ls *lineSearch) wolfe2(phi0, oldPhi0, derphi0 float64, haveOld bool, c1, c2, amax float64) (float64, float64, bool) {
	alpha0 := 0.0
	alpha1 := math.Min(initialStep(phi0, oldPhi0, derphi0, haveOld), amax)
	phiA1 := ls.phi(alpha1)
	phiA0 := phi0
	derphiA0 := derphi0
	accept := func(alpha, f float64) bool {
		return ls.extra == nil || ls.extra(alpha, ls.point(alpha), f, ls.g)
	}
	for i := range 10 {
		if alpha1 == 0 || alpha0 > amax {
			return 0, phi0, false
		}
		if phiA1 > phi0+c1*alpha1*derphi0 || (phiA1 >= phiA0 && i > 0) {
			return ls.zoom(alpha0, alpha1, phiA0, phiA1, derphiA0, phi0, derphi0, c1, c2, accept)
		}
		derphiA1 := ls.derphi(alpha1)
		if math.Abs(derphiA1) <= -c2*derphi0 && accept(alpha1, phiA1) {
			return alpha1, phiA1, true
		}
		if derphiA1 >= 0 {
			return ls.zoom(alpha1, alpha0, phiA1, phiA0, derphiA1, phi0, derphi0, c1, c2, accept)
		}
		alpha2 := math.Min(2*alpha1, amax)
		alpha0, alpha1 = alpha1, alpha2
		phiA0 = phiA1
		phiA1 = ls.phi(alpha1)
		derphiA0 = derphiA1
	}
	// SciPy returns the last step with a warning but no derivative; the
	// caller then treats it as a failure only through derphi_star, which
	// BFGS and CG do not inspect: the step is used.
	ls.derphi(alpha1)
	return alpha1, phiA1, true
}

// zoom is SciPy's _zoom: cubic, then quadratic, then bisection
// interpolation inside [aLo, aHi], at most 11 trials.
func (ls *lineSearch) zoom(aLo, aHi, phiLo, phiHi, derphiLo, phi0, derphi0, c1, c2 float64, accept func(a, f float64) bool) (float64, float64, bool) {
	const delta1, delta2 = 0.2, 0.1
	phiRec, aRec := phi0, 0.0
	for i := 0; i <= 10; i++ {
		dalpha := aHi - aLo
		a, b := aLo, aHi
		if dalpha < 0 {
			a, b = aHi, aLo
		}
		var aj float64
		ok := false
		if i > 0 {
			cchk := delta1 * dalpha
			aj, ok = cubicMin(aLo, phiLo, derphiLo, aHi, phiHi, aRec, phiRec)
			ok = ok && aj <= b-cchk && aj >= a+cchk
		}
		if !ok {
			qchk := delta2 * dalpha
			aj, ok = quadMin(aLo, phiLo, derphiLo, aHi, phiHi)
			if !ok || aj > b-qchk || aj < a+qchk {
				aj = aLo + 0.5*dalpha
			}
		}
		phiAj := ls.phi(aj)
		if phiAj > phi0+c1*aj*derphi0 || phiAj >= phiLo {
			phiRec, aRec = phiHi, aHi
			aHi, phiHi = aj, phiAj
			continue
		}
		derphiAj := ls.derphi(aj)
		if math.Abs(derphiAj) <= -c2*derphi0 && accept(aj, phiAj) {
			return aj, phiAj, true
		}
		if derphiAj*(aHi-aLo) >= 0 {
			phiRec, aRec = phiHi, aHi
			aHi, phiHi = aLo, phiLo
		} else {
			phiRec, aRec = phiLo, aLo
		}
		aLo, phiLo, derphiLo = aj, phiAj, derphiAj
	}
	return 0, 0, false
}

// cubicMin returns the minimiser of the cubic through (a,fa) with slope fpa
// at a and through (b,fb), (c,fc); ok is false when it does not exist or is
// not finite (SciPy's _cubicmin, which treats any floating-point exception as
// failure).
func cubicMin(a, fa, fpa, b, fb, c, fc float64) (float64, bool) {
	C := fpa
	db := b - a
	dc := c - a
	denom := (db * dc) * (db * dc) * (db - dc)
	if denom == 0 {
		return 0, false
	}
	u := fb - fa - C*db
	v := fc - fa - C*dc
	A := (dc*dc*u - db*db*v) / denom
	B := (-dc*dc*dc*u + db*db*db*v) / denom
	radical := B*B - 3*A*C
	if radical < 0 || A == 0 {
		return 0, false
	}
	x := a + (-B+math.Sqrt(radical))/(3*A)
	if math.IsNaN(x) || math.IsInf(x, 0) {
		return 0, false
	}
	return x, true
}

// quadMin returns the minimiser of the quadratic through (a,fa) with slope
// fpa at a and through (b,fb) (SciPy's _quadmin).
func quadMin(a, fa, fpa, b, fb float64) (float64, bool) {
	db := b - a
	if db == 0 {
		return 0, false
	}
	B := (fb - fa - fpa*db) / (db * db)
	if B == 0 {
		return 0, false
	}
	x := a - fpa/(2*B)
	if math.IsNaN(x) || math.IsInf(x, 0) {
		return 0, false
	}
	return x, true
}
