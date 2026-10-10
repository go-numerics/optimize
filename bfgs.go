package optimize

import "math"

// Status messages shared by the methods ported from SciPy's _optimize.py.
const (
	msgSuccess = "Optimization terminated successfully."
	msgMaxFev  = "Maximum number of function evaluations has been exceeded."
	msgMaxIter = "Maximum number of iterations has been exceeded."
	msgPrLoss  = "Desired error not necessarily achieved due to precision loss."
	msgNaN     = "NaN result encountered."
)

func hasNaN(v []float64) bool {
	for _, x := range v {
		if math.IsNaN(x) {
			return true
		}
	}
	return false
}

// minimizeBFGS is the quasi-Newton method of Broyden, Fletcher, Goldfarb and
// Shanno with the inverse-Hessian update and a strong-Wolfe line search
// (Nocedal & Wright, Algorithm 6.1), following SciPy's _minimize_bfgs step by
// step. The update is applied in O(n²) as a rank-two correction instead of
// SciPy's two dense matrix products, which is the same formula in exact
// arithmetic.
func minimizeBFGS(sf *scalarFunc, x0 []float64, o *Options) (*Result, error) {
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
		c2 = 0.9
	}
	if err := checkC1C2(c1, c2); err != nil {
		return nil, err
	}
	H := make([]float64, n*n)
	if o.HessInv0 != nil {
		if len(o.HessInv0) != n*n {
			return nil, ErrDimension
		}
		copy(H, o.HessInv0)
	} else {
		for i := range n {
			H[i*n+i] = 1
		}
	}

	xk := clone(x0)
	gfk := make([]float64, n)
	oldF := sf.funGrad(xk, gfk)
	oldOld := oldF + enorm(gfk)/2
	pk := make([]float64, n)
	sk := make([]float64, n)
	yk := make([]float64, n)
	hy := make([]float64, n) // H·y
	yh := make([]float64, n) // yᵀ·H
	ls := newLineSearch(sf, n)
	k, warn := 0, 0
	gnorm := norm(gfk, ord)
	for gnorm > gtol && k < maxiter {
		for i := range n {
			s := 0.0
			row := H[i*n : i*n+n]
			for j, h := range row {
				s += float64(h * gfk[j])
			}
			pk[i] = -s
		}
		alpha, fNew, ok := ls.wolfe12(xk, pk, gfk, oldF, oldOld, true, c1, c2, 1e-100, 1e100, 1e100)
		if !ok {
			warn = 2
			break
		}
		oldOld, oldF = oldF, fNew
		for i := range n {
			sk[i] = float64(alpha * pk[i])
			xk[i] += sk[i]
			yk[i] = ls.g[i] - gfk[i]
		}
		copy(gfk, ls.g)
		k++
		if o.Callback != nil && o.Callback(clone(xk), oldF) {
			break
		}
		gnorm = norm(gfk, ord)
		if gnorm <= gtol {
			break
		}
		if alpha*enorm(pk) <= o.XRtol*(o.XRtol+enorm(xk)) {
			break
		}
		if math.IsNaN(oldF) || math.IsInf(oldF, 0) {
			warn = 2
			break
		}
		rhoInv := dot(yk, sk)
		rho := 1000.0
		if rhoInv != 0 {
			rho = 1 / rhoInv
		}
		// H ← (I − ρ s yᵀ) H (I − ρ y sᵀ) + ρ s sᵀ
		//   = H − ρ s (yᵀH) − ρ (Hy) sᵀ + (ρ² yᵀHy + ρ) s sᵀ.
		for i := range n {
			hy[i], yh[i] = 0, 0
		}
		for i := range n {
			row := H[i*n : i*n+n]
			yi := yk[i]
			s := 0.0
			for j, h := range row {
				s += float64(h * yk[j])
				yh[j] += float64(yi * h)
			}
			hy[i] = s
		}
		c := float64(rho*rho*dot(yk, hy)) + rho
		for i := range n {
			row := H[i*n : i*n+n]
			si, hyi := sk[i], hy[i]
			for j := range row {
				row[j] += float64(-rho*si*yh[j]) - float64(rho*hyi*sk[j]) + float64(c*si*sk[j])
			}
		}
	}
	r := &Result{X: xk, Fun: oldF, Jac: gfk, HessInv: H, Nit: k, Status: warn}
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

func checkC1C2(c1, c2 float64) error {
	if !(0 < c1 && c1 < c2 && c2 < 1) {
		return errC1C2
	}
	return nil
}
