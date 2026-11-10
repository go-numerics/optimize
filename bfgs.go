package optimize

import (
	"math"

	"github.com/go-simd/floats"
)

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
// step. The inverse-Hessian update is the O(n²) rank-two form by default
// (validated by tolerance); with Options.SciPyExactUpdate it is SciPy's two
// dense n×n products, O(n³) per iteration, bit for bit: the rank-two form
// rounds differently and leaves SciPy's path after a few iterations
// (docs/design-global.md, "BFGS and L-BFGS-B last digits").
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
	// The O(n²) update needs a symmetric H; an asymmetric HessInv0 takes the
	// dense update, which does not.
	exact := o.SciPyExactUpdate || !symmetric(H, n)
	var a1, a2, tmp, hnew, u []float64
	if exact {
		a1, a2 = make([]float64, n*n), make([]float64, n*n)
		tmp, hnew = make([]float64, n*n), make([]float64, n*n)
	} else {
		u = make([]float64, n)
	}
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
		if !exact {
			updateHessInvRankTwo(H, sk, yk, u, rho, n)
			continue
		}
		// H ← A1 (H A2) + ρ s sᵀ with A1 = I − (s yᵀ)ρ and A2 = I − (y sᵀ)ρ,
		// SciPy's expression evaluated in its order (two dense products, each
		// entry a left-to-right sum of rounded products, as reference BLAS
		// and OpenBLAS's non-FMA kernels do for these sizes).
		for i := range n {
			for j := range n {
				id := 0.0
				if i == j {
					id = 1
				}
				a1[i*n+j] = id - float64(float64(sk[i]*yk[j])*rho)
				a2[i*n+j] = id - float64(float64(yk[i]*sk[j])*rho)
			}
		}
		matmul(tmp, H, a2, n)
		matmul(hnew, a1, tmp, n)
		for i := range n {
			rs := float64(rho * sk[i])
			for j := range n {
				H[i*n+j] = hnew[i*n+j] + float64(rs*sk[j])
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

func symmetric(h []float64, n int) bool {
	for i := range n {
		for j := range i {
			if h[i*n+j] != h[j*n+i] {
				return false
			}
		}
	}
	return true
}

// updateHessInvRankTwo is the BFGS inverse-Hessian update in O(n²):
//
//	H ← (I − ρ s yᵀ) H (I − ρ y sᵀ) + ρ s sᵀ
//	  = H − ρ (s uᵀ + u sᵀ) + (ρ² yᵀu + ρ) s sᵀ,   u = H y  (H symmetric),
//
// the same matrix as SciPy's two dense products, rounded differently (it is
// validated by tolerance, not bit for bit). Only the upper triangle is
// computed, with the FMA kernels, and mirrored, so H stays exactly symmetric.
// u is scratch of length n.
func updateHessInvRankTwo(H, s, y, u []float64, rho float64, n int) {
	for i := range n {
		u[i] = floats.Dot(H[i*n:i*n+n], y)
	}
	c := rho*rho*floats.Dot(y, u) + rho
	for i := range n {
		row := H[i*n+i : i*n+n]
		floats.Axpy(-rho*s[i], u[i:], row)
		floats.Axpy(c*s[i]-rho*u[i], s[i:], row)
	}
	for i := range n {
		for j := i + 1; j < n; j++ {
			H[j*n+i] = H[i*n+j]
		}
	}
}

// matmul sets c = a·b (n×n, row-major); every entry is the sum over k in
// increasing order of rounded products, never fused. The loops run in i-k-j
// order: row i of c accumulates a[i][k]·(row k of b) for k = 0, 1, …, so
// each entry c[i][j] still receives its products in increasing k, each
// rounded before the addition (floats.AxpyRowsNoFMA, element-wise and
// unfused): the same bits as the i-j-k dot products, with rows of b read
// contiguously, the inner loop on SIMD lanes and a block of row i of c held
// in registers across all n values of k.
func matmul(c, a, b []float64, n int) {
	for i := range n {
		ci := c[i*n : i*n+n]
		clear(ci)
		floats.AxpyRowsNoFMA(a[i*n:i*n+n], b, n, ci)
	}
}

func checkC1C2(c1, c2 float64) error {
	if !(0 < c1 && c1 < c2 && c2 < 1) {
		return errC1C2
	}
	return nil
}
