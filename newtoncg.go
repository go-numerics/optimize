package optimize

import "math"

// minimizeNewtonCG is the line-search Newton–CG method (truncated Newton,
// Nocedal & Wright, Algorithm 7.1): each step solves the Newton system
// approximately by conjugate gradients, stopped by the forcing term
// η = min(0.5, √‖g‖₁) or on negative curvature, then takes a strong-Wolfe
// line search along it. It follows SciPy's _minimize_newtoncg, including the
// Hessian sources: Problem.Hess (evaluated once per outer iteration),
// Problem.HessP, or a forward difference of the gradient,
// (∇f(x+εp) − ∇f(x))/ε (∇f(x) first, so that the cached value is reused) with ε = Options.Eps (√eps by default).
func minimizeNewtonCG(sf *scalarFunc, x0 []float64, o *Options) (*Result, error) {
	n := len(x0)
	avextol := o.Xtol
	if avextol == 0 {
		avextol = 1e-5
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
		c2 = 0.9
	}
	if err := checkC1C2(c1, c2); err != nil {
		return nil, err
	}
	eps := o.Eps
	if eps == 0 {
		eps = math.Sqrt(epsmch)
	}
	cgMaxiter := 20 * n
	xtol := float64(n) * avextol
	update := math.MaxFloat64

	xk := clone(x0)
	b := make([]float64, n)
	gfk := make([]float64, n)
	xsupi := make([]float64, n)
	ri := make([]float64, n)
	psupi := make([]float64, n)
	ap := make([]float64, n)
	var A []float64
	if sf.p.Hess != nil {
		A = make([]float64, n*n)
	}
	xe := make([]float64, n)
	g1 := make([]float64, n)
	g2 := make([]float64, n)
	ls := newLineSearch(sf, n)

	oldF := sf.fun(xk)
	oldOld, haveOld := 0.0, false
	hcalls, k := 0, 0
	r := &Result{}
	finish := func(status int, msg string) (*Result, error) {
		r.X, r.Fun, r.Jac, r.Nit = xk, oldF, gfk, k
		r.Status, r.Success, r.Message = status, status == 0, msg
		sf.result(r)
		r.Nhev = hcalls
		return r, nil
	}
	for update > xtol {
		if k >= maxiter {
			return finish(1, "Warning: "+msgMaxIter)
		}
		sf.grad(xk, gfk)
		maggrad := 0.0
		for i := range b {
			b[i] = -gfk[i]
			maggrad += math.Abs(b[i])
		}
		eta := math.Min(0.5, math.Sqrt(maggrad))
		termcond := eta * maggrad
		for i := range xsupi {
			xsupi[i] = 0
			ri[i] = gfk[i]
			psupi[i] = -gfk[i]
		}
		dri0 := dot(ri, ri)
		if A != nil {
			sf.p.Hess(xk, A)
			hcalls++
		}
		converged := false
		for i := 0; i < cgMaxiter; i++ {
			s := 0.0
			for _, v := range ri {
				s += math.Abs(v)
			}
			if s <= termcond {
				converged = true
				break
			}
			switch {
			case A != nil:
				for p := range n {
					ap[p] = dot(A[p*n:p*n+n], psupi)
				}
			case sf.p.HessP != nil:
				sf.p.HessP(xk, psupi, ap)
				hcalls++
			default:
				for p := range n {
					xe[p] = xk[p] + eps*psupi[p]
				}
				sf.grad(xk, g1)
				sf.grad(xe, g2)
				for p := range n {
					ap[p] = (g2[p] - g1[p]) / eps
				}
			}
			curv := dot(psupi, ap)
			if 0 <= curv && curv <= 3*epsmch {
				converged = true
				break
			}
			if curv < 0 {
				if i == 0 {
					// Negative curvature on the first step: use the
					// steepest-descent direction scaled to the CG model.
					for p := range n {
						xsupi[p] = dri0 / (-curv) * b[p]
					}
				}
				converged = true
				break
			}
			alphai := dri0 / curv
			for p := range n {
				xsupi[p] += alphai * psupi[p]
				ri[p] += alphai * ap[p]
			}
			dri1 := dot(ri, ri)
			betai := dri1 / dri0
			for p := range n {
				psupi[p] = -ri[p] + betai*psupi[p]
			}
			dri0 = dri1
		}
		if !converged {
			return finish(3, "Warning: CG iterations didn't converge. The Hessian is not positive definite.")
		}
		pk := clone(xsupi)
		alpha, fNew, ok := ls.wolfe12(xk, pk, gfk, oldF, oldOld, haveOld, c1, c2, 1e-8, 50, math.Inf(1))
		if !ok {
			return finish(2, "Warning: "+msgPrLoss)
		}
		oldOld, oldF, haveOld = oldF, fNew, true
		update = 0
		for p := range n {
			u := alpha * pk[p]
			xk[p] += u
			update += math.Abs(u)
		}
		k++
		if o.Callback != nil && o.Callback(clone(xk), oldF) {
			return finish(5, "")
		}
	}
	if math.IsNaN(oldF) || math.IsNaN(update) {
		return finish(3, msgNaN)
	}
	return finish(0, msgSuccess)
}
