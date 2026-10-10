package optimize

import "math"

// hybrdConfig holds the inputs of hybrd/hybrj beyond the functions.
type hybrdConfig struct {
	xtol   float64
	maxfev int
	epsfcn float64   // forward-difference step parameter (hybrd only)
	diag   []float64 // nil selects MINPACK's internal scaling (mode 1)
	factor float64
}

// hybrdOut is what hybrd/hybrj return.
type hybrdOut struct {
	fvec []float64
	fjac []float64 // the n×n orthogonal Q, column-major
	r    []float64 // upper triangle of R packed by rows, n(n+1)/2 entries
	qtf  []float64
	info int
	nfev int
	njev int
}

// hybrd is MINPACK's hybrd (jac == nil, forward-difference Jacobian) and
// hybrj (analytic Jacobian): Powell's hybrid method for the n×n system
// fcn(x) = 0. It overwrites x with the final estimate. jac writes the n×n
// Jacobian at x column-major, given fvec = fcn(x). The inputs are assumed
// valid (the callers check them), so info is never 0.
//
// Products are wrapped in float64() wherever they meet an addition, which
// forbids Go from fusing them into FMA instructions (as it does on arm64):
// the port then rounds exactly like the reference C compiled without
// contraction, on every architecture.
//
// info: 1 the relative error between two consecutive iterates is at most
// xtol; 2 maxfev calls reached; 3 xtol too small; 4 no progress over the last
// five Jacobian evaluations; 5 no progress over the last ten iterations.
func hybrd(fcn func(x, f []float64), jac func(x, fvec, fjac []float64), x []float64, c *hybrdConfig) *hybrdOut {
	n := len(x)
	o := &hybrdOut{
		fvec: make([]float64, n),
		fjac: make([]float64, n*n),
		r:    make([]float64, n*(n+1)/2),
		qtf:  make([]float64, n),
	}
	fvec, fjac, r, qtf := o.fvec, o.fjac, o.r, o.qtf
	ldfjac := n
	diag := make([]float64, n)
	internal := c.diag == nil
	if !internal {
		copy(diag, c.diag)
	}
	wa1 := make([]float64, n)
	wa2 := make([]float64, n)
	wa3 := make([]float64, n)
	wa4 := make([]float64, n)

	fcn(x, fvec)
	o.nfev = 1
	fnorm := mpEnorm(fvec)
	iter := 1
	ncsuc, ncfail, nslow1, nslow2 := 0, 0, 0, 0
	var delta, xnorm float64
	for {
		// The Jacobian, then its QR factorisation.
		jeval := true
		if jac == nil {
			fdjac(fcn, n, n, x, fvec, fjac, c.epsfcn, wa1)
			o.nfev += n
		} else {
			jac(x, fvec, fjac)
			o.njev++
		}
		qrfac(n, n, fjac, ldfjac, false, nil, wa1, wa2, wa3)
		// On the first iteration, scale by the column norms (mode 1) and
		// set the step bound delta.
		if iter == 1 {
			if internal {
				for j := 0; j < n; j++ {
					diag[j] = wa2[j]
					if wa2[j] == 0 {
						diag[j] = 1
					}
				}
			}
			for j := 0; j < n; j++ {
				wa3[j] = diag[j] * x[j]
			}
			xnorm = mpEnorm(wa3)
			delta = initialStepBound(c.factor, xnorm)
		}
		// Qᵀ·fvec into qtf.
		copy(qtf, fvec)
		for j := 0; j < n; j++ {
			if fjac[j+ldfjac*j] != 0 {
				sum := 0.0
				for i := j; i < n; i++ {
					sum += float64(fjac[i+ldfjac*j] * qtf[i])
				}
				temp := -sum / fjac[j+ldfjac*j]
				for i := j; i < n; i++ {
					qtf[i] += float64(fjac[i+ldfjac*j] * temp)
				}
			}
		}
		// Copy the triangular factor into r, packed by rows.
		for j := 0; j < n; j++ {
			l := j
			for i := 0; i < j; i++ {
				r[l] = fjac[i+ldfjac*j]
				l += n - i - 1
			}
			r[l] = wa1[j]
		}
		// Accumulate Q.
		qform(n, fjac, ldfjac, wa1)
		if internal {
			for j := 0; j < n; j++ {
				diag[j] = math.Max(diag[j], wa2[j])
			}
		}
		for {
			// The dogleg direction, and the trial point x + p.
			dogleg(n, r, diag, qtf, delta, wa1, wa2, wa3)
			for j := 0; j < n; j++ {
				wa1[j] = -wa1[j]
				wa2[j] = x[j] + wa1[j]
				wa3[j] = diag[j] * wa1[j]
			}
			pnorm := mpEnorm(wa3)
			if iter == 1 {
				delta = math.Min(delta, pnorm)
			}
			fcn(wa2, wa4)
			o.nfev++
			fnorm1 := mpEnorm(wa4)
			// Actual and predicted relative reductions.
			actred := -1.0
			if fnorm1 < fnorm {
				t := fnorm1 / fnorm
				actred = 1 - float64(t*t)
			}
			l := 0
			for i := 0; i < n; i++ {
				sum := 0.0
				for j := i; j < n; j++ {
					sum += float64(r[l] * wa1[j])
					l++
				}
				wa3[i] = qtf[i] + sum
			}
			temp := mpEnorm(wa3)
			prered := 0.0
			if temp < fnorm {
				t := temp / fnorm
				prered = 1 - float64(t*t)
			}
			ratio := 0.0
			if prered > 0 {
				ratio = actred / prered
			}
			// Update the step bound.
			if ratio < 0.1 {
				ncsuc = 0
				ncfail++
				delta = 0.5 * delta
			} else {
				ncfail = 0
				ncsuc++
				if ratio >= 0.5 || ncsuc > 1 {
					delta = math.Max(delta, pnorm/0.5)
				}
				if math.Abs(ratio-1) <= 0.1 {
					delta = pnorm / 0.5
				}
			}
			// Accept a successful step.
			if ratio >= 0.0001 {
				for j := 0; j < n; j++ {
					x[j] = wa2[j]
					wa2[j] = diag[j] * x[j]
					fvec[j] = wa4[j]
				}
				xnorm = mpEnorm(wa2)
				fnorm = fnorm1
				iter++
			}
			// Progress counters.
			nslow1++
			if actred >= 0.001 {
				nslow1 = 0
			}
			if jeval {
				nslow2++
			}
			if actred >= 0.1 {
				nslow2 = 0
			}
			// Convergence and termination tests.
			if delta <= c.xtol*xnorm || fnorm == 0 {
				o.info = 1
				return o
			}
			if o.nfev >= c.maxfev {
				o.info = 2
			}
			if 0.1*math.Max(0.1*delta, pnorm) <= epsmch*xnorm {
				o.info = 3
			}
			if nslow2 == 5 {
				o.info = 4
			}
			if nslow1 == 10 {
				o.info = 5
			}
			if o.info != 0 {
				return o
			}
			// Two consecutive failures: recompute the Jacobian.
			if ncfail == 2 {
				break
			}
			// Otherwise Broyden's rank-1 update of R and Q.
			for j := 0; j < n; j++ {
				sum := 0.0
				for i := 0; i < n; i++ {
					sum += float64(fjac[i+ldfjac*j] * wa4[i])
				}
				wa2[j] = (sum - wa3[j]) / pnorm
				wa1[j] = diag[j] * ((diag[j] * wa1[j]) / pnorm)
				if ratio >= 0.0001 {
					qtf[j] = sum
				}
			}
			r1updt(n, n, r, wa1, wa2, wa3)
			r1mpyq(n, n, fjac, ldfjac, wa2, wa3)
			r1mpyq(1, n, qtf, 1, wa2, wa3)
			jeval = false
		}
	}
}

// initialStepBound returns MINPACK's initial step bound factor·xnorm, or
// factor when that is zero.
func initialStepBound(f, xnorm float64) float64 {
	if d := f * xnorm; d != 0 {
		return d
	}
	return f
}
