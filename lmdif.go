package optimize

import (
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
)

// lmConfig holds the inputs of lmdif/lmder beyond the functions.
type lmConfig struct {
	m          int
	ftol, xtol float64
	gtol       float64
	maxfev     int
	epsfcn     float64   // forward-difference step parameter (lmdif only)
	diag       []float64 // nil selects MINPACK's internal scaling (mode 1)
	factor     float64
}

// lmOut is what lmdif/lmder return.
type lmOut struct {
	fvec []float64
	// fjac is column-major m×n: its upper n×n triangle holds R of the final
	// pivoted QR factorisation, the rest the Householder vectors.
	fjac []float64
	ipvt []int
	qtf  []float64
	info int
	nfev int
	njev int
}

// lmdif is MINPACK's lmdif (jac == nil, forward-difference Jacobian) and
// lmder (analytic Jacobian): the Levenberg–Marquardt method for minimising
// the sum of squares of m residuals in n <= m variables. It overwrites x
// with the final estimate. jac writes the m×n Jacobian at x column-major,
// given the residuals fvec at x. The
// inputs are assumed valid (the callers check them), so info is never 0.
//
// info: 1 ftol (actual and predicted reductions); 2 xtol; 3 both; 4 gtol
// (orthogonality of fvec and the Jacobian columns); 5 maxfev calls reached;
// 6, 7, 8 ftol, xtol, gtol too small to make further progress.
func lmdif(fcn func(x, f []float64), jac func(x, fvec, fjac []float64), x []float64, c *lmConfig) *lmOut {
	m, n := c.m, len(x)
	o := &lmOut{
		fvec: make([]float64, m),
		fjac: make([]float64, m*n),
		ipvt: make([]int, n),
		qtf:  make([]float64, n),
	}
	fvec, fjac, ipvt, qtf := o.fvec, o.fjac, o.ipvt, o.qtf
	ldfjac := m
	diag := make([]float64, n)
	internal := c.diag == nil
	if !internal {
		copy(diag, c.diag)
	}
	wa1 := make([]float64, n)
	wa2 := make([]float64, n)
	wa3 := make([]float64, n)
	wa4 := make([]float64, m)

	fcn(x, fvec)
	o.nfev = 1
	fnorm := mpEnorm(fvec)
	par := 0.0
	iter := 1
	var delta, xnorm float64
	for {
		// The Jacobian, then its pivoted QR factorisation.
		if jac == nil {
			fdjac(fcn, m, n, x, fvec, fjac, c.epsfcn, wa4)
			o.nfev += n
		} else {
			jac(x, fvec, fjac)
			o.njev++
		}
		qrfac(m, n, fjac, ldfjac, true, ipvt, wa1, wa2, wa3)
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
		// Qᵀ·fvec, keeping its first n components in qtf; R's diagonal
		// goes back into fjac.
		copy(wa4, fvec)
		for j := 0; j < n; j++ {
			if fjac[j+ldfjac*j] != 0 {
				sum := 0.0
				for i := j; i < m; i++ {
					sum += float64(fjac[i+ldfjac*j] * wa4[i])
				}
				temp := -sum / fjac[j+ldfjac*j]
				for i := j; i < m; i++ {
					wa4[i] += float64(fjac[i+ldfjac*j] * temp)
				}
			}
			fjac[j+ldfjac*j] = wa1[j]
			qtf[j] = wa4[j]
		}
		// The norm of the scaled gradient.
		gnorm := 0.0
		if fnorm != 0 {
			for j := 0; j < n; j++ {
				l := ipvt[j]
				if wa2[l] != 0 {
					sum := 0.0
					for i := 0; i <= j; i++ {
						sum += float64(fjac[i+ldfjac*j] * (qtf[i] / fnorm))
					}
					gnorm = math.Max(gnorm, math.Abs(sum/wa2[l]))
				}
			}
		}
		if gnorm <= c.gtol {
			o.info = 4
			return o
		}
		if internal {
			for j := 0; j < n; j++ {
				diag[j] = math.Max(diag[j], wa2[j])
			}
		}
		for {
			// The Levenberg–Marquardt parameter and direction, and the
			// trial point x + p.
			par = lmpar(n, fjac, ldfjac, ipvt, diag, qtf, delta, par, wa1, wa2, wa3, wa4)
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
			// Actual and predicted relative reductions, and the
			// directional derivative.
			actred := -1.0
			if 0.1*fnorm1 < fnorm {
				t := fnorm1 / fnorm
				actred = 1 - float64(t*t)
			}
			for j := 0; j < n; j++ {
				wa3[j] = 0
				temp := wa1[ipvt[j]]
				for i := 0; i <= j; i++ {
					wa3[i] += float64(fjac[i+ldfjac*j] * temp)
				}
			}
			temp1 := mpEnorm(wa3) / fnorm
			temp2 := (math.Sqrt(par) * pnorm) / fnorm
			prered := float64(temp1*temp1) + temp2*temp2/0.5
			dirder := -(float64(temp1*temp1) + float64(temp2*temp2))
			ratio := 0.0
			if prered != 0 {
				ratio = actred / prered
			}
			// Update the step bound.
			if ratio <= 0.25 {
				temp := 0.5
				if actred < 0 {
					temp = (0.5 * dirder) / (dirder + float64(0.5*actred))
				}
				if 0.1*fnorm1 >= fnorm || temp < 0.1 {
					temp = 0.1
				}
				delta = temp * math.Min(delta, pnorm/0.1)
				par /= temp
			} else if par == 0 || ratio >= 0.75 {
				delta = pnorm / 0.5
				par *= 0.5
			}
			// Accept a successful step.
			if ratio >= 0.0001 {
				for j := 0; j < n; j++ {
					x[j] = wa2[j]
					wa2[j] = diag[j] * x[j]
				}
				copy(fvec, wa4)
				xnorm = mpEnorm(wa2)
				fnorm = fnorm1
				iter++
			}
			// Convergence tests.
			fconv := math.Abs(actred) <= c.ftol && prered <= c.ftol && 0.5*ratio <= 1
			if fconv {
				o.info = 1
			}
			if delta <= c.xtol*xnorm {
				o.info = 2
				if fconv {
					o.info = 3
				}
			}
			if o.info != 0 {
				return o
			}
			// Termination tests.
			if o.nfev >= c.maxfev {
				o.info = 5
			}
			if math.Abs(actred) <= epsmch && prered <= epsmch && 0.5*ratio <= 1 {
				o.info = 6
			}
			if delta <= epsmch*xnorm {
				o.info = 7
			}
			if gnorm <= epsmch {
				o.info = 8
			}
			if o.info != 0 {
				return o
			}
			if ratio >= 0.0001 {
				break
			}
		}
	}
}

// lmStatus maps MINPACK's lmdif/lmder info to the status of SciPy's
// least_squares (FROM_MINPACK_TO_COMMON); 6, 7 and 8 cannot occur there
// because every tolerance is at least the machine epsilon.
var lmStatus = [...]int{-1, 2, 3, 4, 1, 0}

// lsMessages are the termination messages of SciPy's least_squares.
var lsMessages = map[int]string{
	-1: "Improper input parameters status returned from `leastsq`",
	0:  "The maximum number of function evaluations is exceeded.",
	1:  "`gtol` termination condition is satisfied.",
	2:  "`ftol` termination condition is satisfied.",
	3:  "`xtol` termination condition is satisfied.",
	4:  "Both `ftol` and `xtol` termination conditions are satisfied.",
}

// rowToCol returns a function that evaluates jac (row-major m×n) into a
// column-major buffer, as SciPy's wrapper transposes a Dfun that does not
// set col_deriv.
func rowToCol(jac func(x, fvec, j []float64), m, n int) func(x, fvec, fjac []float64) {
	buf := make([]float64, m*n)
	return func(x, fvec, fjac []float64) {
		jac(x, fvec, buf)
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				fjac[i+m*j] = buf[i*n+j]
			}
		}
	}
}

// lsLM is LeastSquares with method LM, i.e. SciPy's least_squares(...,
// method='lm'), which since SciPy 1.16 always calls lmder: without an
// analytic Jacobian, the Jacobian is the '2-point' finite difference of
// approx_derivative (relative step DiffStep), and those evaluations are not
// counted in Nfev, which is MINPACK's count; Njev is MINPACK's count with an
// analytic Jacobian and 0 otherwise (SciPy reports None). XScale nil or
// XScaleJac select MINPACK's internal scaling (SciPy's default x_scale='jac'
// for this method); otherwise diag = 1/XScale.
func lsLM(p LSProblem, x0 []float64, o *LSOptions) (*LSResult, error) {
	m, n := p.M, len(x0)
	if o.Ftol < epsmch || o.Xtol < epsmch || o.Gtol < epsmch {
		return nil, fmt.Errorf("%w: all tolerances must be at least the machine epsilon for method lm", ErrInput)
	}
	c := &lmConfig{m: m, ftol: o.Ftol, xtol: o.Xtol, gtol: o.Gtol, maxfev: o.MaxNfev, factor: 100}
	if c.maxfev == 0 {
		c.maxfev = 100 * n
	}
	if c.maxfev < 0 {
		return nil, fmt.Errorf("%w: MaxNfev must be positive", ErrInput)
	}
	if o.XScale != nil && !o.XScaleJac {
		c.diag = make([]float64, n)
		for i, s := range o.XScale {
			c.diag[i] = 1 / s
		}
	}
	// The Jacobian as SciPy's VectorFunction supplies it: the user's, or a
	// '2-point' difference from the residuals at x.
	jacRow := func(x, _, j []float64) { p.Jac(x, j) }
	if p.Jac == nil {
		fd := fdStep{method: FD2Point, rel: o.DiffStep}
		jacRow = func(x, fvec, j []float64) { fd.jacFD(p.Func, x, fvec, j) }
	}
	x := x0
	out := lmdif(p.Func, rowToCol(jacRow, m, n), x, c)
	res := &LSResult{X: x, Fun: out.fvec, Nfev: out.nfev, ActiveMask: make([]int, n)}
	if p.Jac != nil {
		res.Njev = out.njev
	}
	res.Jac = make([]float64, m*n)
	jacRow(x, out.fvec, res.Jac)
	res.Cost = 0.5 * dot(res.Fun, res.Fun)
	res.Grad = make([]float64, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			res.Grad[j] += float64(res.Jac[i*n+j] * res.Fun[i])
		}
	}
	res.Optimality = norm(res.Grad, math.Inf(1))
	res.Status = lmStatus[out.info]
	res.Success = res.Status > 0
	res.Message = lsMessages[res.Status]
	return res, nil
}

// LeastsqOptions configures Leastsq. Zero values select the defaults of
// scipy.optimize.leastsq.
type LeastsqOptions struct {
	// Ftol is the relative error desired in the sum of squares and Xtol the
	// relative error desired in the solution (1.49012e-8 each; a negative
	// value means exactly 0). Gtol is the orthogonality desired between the
	// residuals and the Jacobian columns (0).
	Ftol, Xtol, Gtol float64
	// Maxfev bounds the calls to Func: 0 selects 100·(n+1) with a Jacobian
	// and 200·(n+1) without.
	Maxfev int
	// Epsfcn sets the forward-difference step sqrt(Epsfcn)·|x_j| when there
	// is no Jacobian; 0 selects the machine epsilon.
	Epsfcn float64
	// Factor sets the initial step bound Factor·|diag·x0| (100).
	Factor float64
	// Diag holds n positive scale factors for the variables; nil lets
	// MINPACK scale by the Jacobian column norms.
	Diag []float64
}

// LeastsqResult is the full output of Leastsq: SciPy's tuple
// (x, cov_x, infodict, mesg, ier).
type LeastsqResult struct {
	// X is the solution (or the last iterate).
	X []float64
	// CovX is inv(JᵀJ) at X, n×n row-major, from the final R and
	// permutation; nil when R is singular or the run did not converge
	// (Ier not in 1..4). Multiply it by the residual variance to obtain the
	// covariance of X.
	CovX []float64
	// Nfev counts calls to Func, including those of the finite-difference
	// Jacobian; Njev counts calls to Jac (0 without one).
	Nfev, Njev int
	// Fvec holds the residuals at X.
	Fvec []float64
	// Fjac is SciPy's infodict['fjac']: an n×m row-major array whose rows
	// are the columns of MINPACK's final QR workspace, so that the upper
	// triangle of its transpose's first n rows is R with Jac·P = Q·R.
	Fjac []float64
	// Ipvt is the permutation P: column j of P is column Ipvt[j]
	// (zero-based) of the identity.
	Ipvt []int
	// Qtf is the first n components of Qᵀ·Fvec.
	Qtf []float64
	// Ier is MINPACK's info: 1 to 4 mean a solution was found, 5 to 8 that
	// it was not.
	Ier int
	// Mesg is SciPy's message for Ier.
	Mesg string
}

// Success reports whether Ier is 1, 2, 3 or 4, as SciPy's leastsq judges it.
func (r *LeastsqResult) Success() bool { return r.Ier >= 1 && r.Ier <= 4 }

// XArray returns X as a 1-D ndarray.
func (r *LeastsqResult) XArray() *ndarray.Array { return vecArray(r.X) }

// CovXArray returns CovX as an n×n ndarray, or nil when absent.
func (r *LeastsqResult) CovXArray() *ndarray.Array { return matArray(r.CovX, len(r.X), len(r.X)) }

// FjacArray returns Fjac as an n×m ndarray.
func (r *LeastsqResult) FjacArray() *ndarray.Array { return matArray(r.Fjac, len(r.X), len(r.Fvec)) }

// Leastsq minimises the sum of squares of p.M residuals with MINPACK's
// Levenberg–Marquardt method; it is scipy.optimize.leastsq(func, x0,
// Dfun=p.Jac, full_output=True, ftol, xtol, gtol, maxfev, epsfcn, factor,
// diag). Without p.Jac it runs lmdif (forward differences), otherwise
// lmder. Non-convergence is reported through Ier and Mesg.
func Leastsq(p LSProblem, x0 []float64, opts *LeastsqOptions) (*LeastsqResult, error) {
	n, m := len(x0), p.M
	if p.Func == nil || m <= 0 || n == 0 {
		return nil, fmt.Errorf("%w: LSProblem needs Func, M > 0 and a non-empty x0", ErrInput)
	}
	if n > m {
		return nil, fmt.Errorf("%w: %d variables exceed %d residuals", ErrDimension, n, m)
	}
	var o LeastsqOptions
	if opts != nil {
		o = *opts
	}
	c := &lmConfig{m: m, ftol: tolDefault(o.Ftol), xtol: tolDefault(o.Xtol), gtol: math.Max(o.Gtol, 0),
		maxfev: o.Maxfev, epsfcn: o.Epsfcn, factor: o.Factor}
	if c.maxfev == 0 {
		c.maxfev = 200 * (n + 1)
		if p.Jac != nil {
			c.maxfev = 100 * (n + 1)
		}
	}
	if c.factor == 0 {
		c.factor = 100
	}
	if err := checkMinpack(c.maxfev, c.factor, o.Diag, n); err != nil {
		return nil, err
	}
	c.diag = o.Diag
	var jac func(x, fvec, fjac []float64)
	if p.Jac != nil {
		jac = rowToCol(func(x, _, j []float64) { p.Jac(x, j) }, m, n)
	}
	x := clone(x0)
	out := lmdif(p.Func, jac, x, c)
	res := &LeastsqResult{X: x, Nfev: out.nfev, Njev: out.njev, Fvec: out.fvec, Fjac: out.fjac,
		Ipvt: out.ipvt, Qtf: out.qtf, Ier: out.info, Mesg: leastsqMessage(out.info, c)}
	if res.Success() {
		res.CovX = leastsqCov(out.fjac, out.ipvt, m, n)
	}
	return res, nil
}

// tolDefault returns MINPACK's default tolerance 1.49012e-8 for 0, and 0
// for a negative value.
func tolDefault(t float64) float64 {
	if t == 0 {
		return 1.49012e-8
	}
	return math.Max(t, 0)
}

// checkMinpack rejects the inputs for which MINPACK returns info = 0
// ("improper input parameters").
func checkMinpack(maxfev int, factor float64, diag []float64, n int) error {
	if maxfev < 0 {
		return fmt.Errorf("%w: Maxfev must be positive", ErrInput)
	}
	if !(factor > 0) {
		return fmt.Errorf("%w: Factor must be positive", ErrInput)
	}
	if diag == nil {
		return nil
	}
	if len(diag) != n {
		return fmt.Errorf("%w: Diag has %d entries for %d variables", ErrDimension, len(diag), n)
	}
	for _, d := range diag {
		if !(d > 0) {
			return fmt.Errorf("%w: Diag entries must be positive", ErrInput)
		}
	}
	return nil
}

// leastsqMessage is SciPy's leastsq message for info (Python's {:f} is
// Go's %f).
func leastsqMessage(info int, c *lmConfig) string {
	switch info {
	case 1:
		return fmt.Sprintf("Both actual and predicted relative reductions in the sum of squares\n  are at most %f", c.ftol)
	case 2:
		return fmt.Sprintf("The relative error between two consecutive iterates is at most %f", c.xtol)
	case 3:
		return fmt.Sprintf("Both actual and predicted relative reductions in the sum of squares\n  are at most %f and the relative error between two consecutive iterates is at \n  most %f", c.ftol, c.xtol)
	case 4:
		return fmt.Sprintf("The cosine of the angle between func(x) and any column of the\n  Jacobian is at most %f in absolute value", c.gtol)
	case 5:
		return fmt.Sprintf("Number of calls to function has reached maxfev = %d.", c.maxfev)
	case 6:
		return fmt.Sprintf("ftol=%f is too small, no further reduction in the sum of squares\n  is possible.", c.ftol)
	case 7:
		return fmt.Sprintf("xtol=%f is too small, no further improvement in the approximate\n  solution is possible.", c.xtol)
	}
	return fmt.Sprintf("gtol=%f is too small, func(x) is orthogonal to the columns of\n  the Jacobian to machine precision.", c.gtol)
}

// leastsqCov computes SciPy's cov_x from lmdif's output: R is the upper
// triangle of the first n rows of the column-major m×n fjac; its inverse is
// formed as LAPACK's dtrti2 does, its rows are permuted by ipvt
// (invR[ipvt] = invR), and cov_x = invR·invRᵀ. It returns nil when R has a
// zero diagonal entry (trtri's info > 0).
func leastsqCov(fjac []float64, ipvt []int, m, n int) []float64 {
	// a is R, column-major n×n.
	a := make([]float64, n*n)
	for j := 0; j < n; j++ {
		for i := 0; i <= j; i++ {
			a[i+n*j] = fjac[i+m*j]
		}
		if a[j+n*j] == 0 {
			return nil
		}
	}
	// dtrti2 (upper, non-unit): column j of the inverse is
	// -inv(R[j,j]) · inv(R[:j,:j]) · R[:j,j].
	for j := 0; j < n; j++ {
		a[j+n*j] = 1 / a[j+n*j]
		ajj := -a[j+n*j]
		col := a[n*j : n*j+j]
		// dtrmv: col := T·col with T the inverted leading j×j block.
		for jj := 0; jj < j; jj++ {
			if col[jj] != 0 {
				temp := col[jj]
				for i := 0; i < jj; i++ {
					col[i] += float64(temp * a[i+n*jj])
				}
				col[jj] *= a[jj+n*jj]
			}
		}
		for i := range col {
			col[i] *= ajj
		}
	}
	// inv[ipvt[i], :] = invR[i, :], row-major.
	inv := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			inv[ipvt[i]*n+j] = a[i+n*j]
		}
	}
	cov := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0
			for k := 0; k < n; k++ {
				sum += float64(inv[i*n+k] * inv[j*n+k])
			}
			cov[i*n+j] = sum
		}
	}
	return cov
}
