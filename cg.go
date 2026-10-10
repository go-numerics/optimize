package optimize

import (
	"fmt"
	"math"
)

var errC1C2 = fmt.Errorf("%w: C1 and C2 must satisfy 0 < C1 < C2 < 1", ErrInput)

// minimizeCG is the nonlinear conjugate-gradient method of Polak and Ribière
// with the non-negative β of Powell (PR+), and a strong-Wolfe line search
// whose step must also yield a sufficient-descent next direction (Nocedal &
// Wright, §5.2), following SciPy's _minimize_cg.
func minimizeCG(sf *scalarFunc, x0 []float64, o *Options) (*Result, error) {
	n := len(x0)
	gtol := o.Gtol
	if gtol == 0 {
		gtol = 1e-5
	}
	ord := o.Norm
	if ord == 0 {
		ord = math.Inf(1)
	}
	maxiter := o.MaxIter
	if maxiter == 0 {
		maxiter = 200 * n
	}
	c1, c2 := o.C1, o.C2
	if c1 == 0 {
		c1 = 1e-4
	}
	if c2 == 0 {
		c2 = 0.4
	}
	if err := checkC1C2(c1, c2); err != nil {
		return nil, err
	}
	const sigma3 = 0.01

	xk := clone(x0)
	gfk := make([]float64, n)
	oldF := sf.funGrad(xk, gfk)
	oldOld := oldF + enorm(gfk)/2
	pk := make([]float64, n)
	for i := range pk {
		pk[i] = -gfk[i]
	}
	gnorm := norm(gfk, ord)

	// The next direction computed for the accepted step, cached by the
	// descent condition so that it is not recomputed.
	pNext := make([]float64, n)
	var cachedAlpha float64
	var cachedGnorm float64
	cached := false
	var deltak float64
	prp := func(g []float64) float64 {
		beta := 0.0
		for i := range g {
			beta += (g[i] - gfk[i]) * g[i]
		}
		beta = math.Max(0, beta/deltak)
		for i := range g {
			pNext[i] = -g[i] + beta*pk[i]
		}
		return norm(g, ord)
	}

	ls := newLineSearch(sf, n)
	ls.extra = func(alpha float64, _ []float64, _ float64, g []float64) bool {
		cachedAlpha, cachedGnorm, cached = alpha, prp(g), true
		if cachedGnorm <= gtol {
			return true
		}
		return dot(pNext, g) <= -sigma3*dot(g, g)
	}
	k, warn := 0, 0
	for gnorm > gtol && k < maxiter {
		deltak = dot(gfk, gfk)
		cached = false
		alpha, fNew, ok := ls.wolfe12(xk, pk, gfk, oldF, oldOld, true, c1, c2, 1e-100, 1e100, 1e100)
		if !ok {
			warn = 2
			break
		}
		oldOld, oldF = oldF, fNew
		if !cached || cachedAlpha != alpha {
			cachedGnorm = prp(ls.g)
		}
		gnorm = cachedGnorm
		for i := range xk {
			xk[i] += alpha * pk[i]
		}
		copy(gfk, ls.g)
		pk, pNext = pNext, pk
		k++
		if o.Callback != nil && o.Callback(clone(xk), oldF) {
			break
		}
	}
	r := &Result{X: xk, Fun: oldF, Jac: gfk, Nit: k, Status: warn}
	switch {
	case warn == 2:
		r.Message = msgPrLoss
	case k >= maxiter:
		r.Status, r.Message = 1, msgMaxIter
	case math.IsNaN(gnorm) || math.IsNaN(oldF) || hasNaN(xk):
		r.Status, r.Message = 3, msgNaN
	default:
		r.Message = msgSuccess
	}
	r.Success = r.Status == 0
	return sf.result(r), nil
}
