package optimize

import (
	"errors"
	"fmt"
	"math"
)

// lineForSearch returns the range [lmin, lmax] of α for which x0 + α·d stays
// within the bounds (SciPy's _line_for_search); (0, 0) if empty.
func lineForSearch(x0, d, lb, ub []float64) (lmin, lmax float64) {
	lmin, lmax = math.Inf(-1), math.Inf(1)
	for i, a := range d {
		if a == 0 {
			continue
		}
		low := (lb[i] - x0[i]) / a
		high := (ub[i] - x0[i]) / a
		if a > 0 {
			lmin = math.Max(lmin, low)
			lmax = math.Min(lmax, high)
		} else {
			lmin = math.Max(lmin, high)
			lmax = math.Min(lmax, low)
		}
	}
	if lmax >= lmin {
		return lmin, lmax
	}
	return 0, 0
}

// powellLineSearch minimises f along p + α·xi with Brent's method (or the
// bounded method when bounds restrict α), returning the new value, point and
// step vector, as SciPy's _linesearch_powell.
func powellLineSearch(f func([]float64) float64, p, xi []float64, tol float64, lb, ub []float64, fval float64) (float64, []float64, []float64) {
	nonzero := false
	for _, v := range xi {
		if v != 0 {
			nonzero = true
			break
		}
	}
	if !nonzero {
		return fval, p, xi
	}
	x := make([]float64, len(p))
	along := func(alpha float64) float64 {
		for i := range x {
			x[i] = p[i] + alpha*xi[i]
		}
		return f(x)
	}
	step := func(alpha, fun float64) (float64, []float64, []float64) {
		nx := make([]float64, len(p))
		nxi := make([]float64, len(p))
		for i := range p {
			nxi[i] = alpha * xi[i]
			nx[i] = p[i] + nxi[i]
		}
		return fun, nx, nxi
	}
	lo, hi := math.Inf(-1), math.Inf(1)
	if lb != nil {
		lo, hi = lineForSearch(p, xi, lb, ub)
	}
	switch {
	case math.IsInf(lo, -1) && math.IsInf(hi, 1):
		r, err := minimizeScalarBrent(along, nil, tol, 500)
		var be *BracketError
		if errors.As(err, &be) {
			r = be.result()
		}
		return step(r.X, r.Fun)
	case !math.IsInf(lo, 0) && !math.IsInf(hi, 0):
		r, _ := minimizeScalarBounded(along, lo, hi, tol/100, 500)
		return step(r.X, r.Fun)
	default:
		// Half-infinite range: search over arctan(α) instead.
		r, _ := minimizeScalarBounded(func(t float64) float64 { return along(math.Tan(t)) },
			math.Atan(lo), math.Atan(hi), tol/100, 500)
		return step(math.Tan(r.X), r.Fun)
	}
}

// minimizePowell is Powell's (1964) conjugate-direction method with the
// direction-replacement test of Press et al. and bound handling by restricted
// line searches, following SciPy's _minimize_powell.
func minimizePowell(sf *scalarFunc, x0 []float64, o *Options, lb, ub []float64) (*Result, error) {
	n := len(x0)
	xtol, ftol := o.Xtol, o.Ftol
	if xtol == 0 {
		xtol = o.Tol
	}
	if ftol == 0 {
		ftol = o.Tol
	}
	if xtol == 0 {
		xtol = 1e-4
	}
	if ftol == 0 {
		ftol = 1e-4
	}
	maxiter, maxfun := iterLimits(o, n, 1000)
	direc := make([][]float64, n)
	if o.Direc != nil {
		if len(o.Direc) != n {
			return nil, fmt.Errorf("%w: Direc must hold n directions", ErrDimension)
		}
		for i, d := range o.Direc {
			if len(d) != n {
				return nil, fmt.Errorf("%w: Direc vectors must have length n", ErrDimension)
			}
			direc[i] = clone(d)
		}
	} else {
		for i := range direc {
			direc[i] = make([]float64, n)
			direc[i][i] = 1
		}
	}
	bud := &budget{sf: sf, maxfun: maxfun}
	x := clone(x0)
	var fval float64
	if guard(func() { fval = bud.f(x) }) {
		fval = math.NaN()
	}
	x1 := clone(x)
	iter := 0
	for {
		stop := false
		cut := guard(func() {
			fx := fval
			bigind := 0
			delta := 0.0
			for i := range n {
				fx2 := fval
				fval, x, _ = powellLineSearch(bud.f, x, direc[i], xtol*100, lb, ub, fval)
				if fx2-fval > delta {
					delta = fx2 - fval
					bigind = i
				}
			}
			iter++
			if o.Callback != nil && o.Callback(clone(x), fval) {
				stop = true
				return
			}
			bnd := ftol*(math.Abs(fx)+math.Abs(fval)) + 1e-20
			if 2*(fx-fval) <= bnd || bud.ncalls >= maxfun || iter >= maxiter ||
				(math.IsNaN(fx) && math.IsNaN(fval)) {
				stop = true
				return
			}
			direc1 := make([]float64, n)
			for i := range x {
				direc1[i] = x[i] - x1[i]
			}
			x1 = clone(x)
			lmax := 1.0
			if lb != nil {
				_, lmax = lineForSearch(x, direc1, lb, ub)
			}
			x2 := make([]float64, n)
			s := math.Min(lmax, 1)
			for i := range x2 {
				x2[i] = x[i] + s*direc1[i]
			}
			fx2 := bud.f(x2)
			if fx > fx2 {
				t := 2 * (fx + fx2 - 2*fval)
				temp := fx - fval - delta
				t *= temp * temp
				temp = fx - fx2
				t -= delta * temp * temp
				if t < 0 {
					fval, x, direc1 = powellLineSearch(bud.f, x, direc1, xtol*100, lb, ub, fval)
					nonzero := false
					for _, v := range direc1 {
						nonzero = nonzero || v != 0
					}
					if nonzero {
						direc[bigind] = direc[n-1]
						direc[n-1] = direc1
					}
				}
			}
		})
		if cut || stop {
			break
		}
	}
	r := &Result{X: x, Fun: fval, Nit: iter, Message: msgSuccess}
	outside := false
	if lb != nil {
		for i := range x {
			outside = outside || x[i] < lb[i] || x[i] > ub[i]
		}
	}
	switch {
	case outside:
		r.Status, r.Message = 4, "The result is outside of the provided bounds."
	case bud.ncalls >= maxfun:
		r.Status, r.Message = 1, msgMaxFev
	case iter >= maxiter:
		r.Status, r.Message = 2, msgMaxIter
	case math.IsNaN(fval) || hasNaN(x):
		r.Status, r.Message = 3, msgNaN
	}
	r.Success = r.Status == 0
	return sf.result(r), nil
}
