package optimize

import (
	"errors"
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/linalg"
)

// ErrNotConverged is returned by CurveFit when the least-squares solver did
// not converge (SciPy raises RuntimeError("Optimal parameters not found")).
// The partial result is returned alongside it.
var ErrNotConverged = errors.New("optimize: optimal parameters not found")

// Model is a curve-fitting model y = f(x; p) evaluated at one sample. For
// 1-D xdata x has one element; for xdata of shape (k, m) — k independent
// variables, m samples, SciPy's layout — x holds the k values of one sample.
type Model struct {
	// F returns the model value at sample x for parameters p. Required.
	F func(x, p []float64) float64
	// Grad writes ∂f/∂p at sample x into grad (len(p) values). Optional:
	// finite differences are used when it is nil (SciPy's jac=callable).
	Grad func(x, p, grad []float64)
}

// CurveFitOptions configures CurveFit. Zero values select SciPy's defaults.
type CurveFitOptions struct {
	// Sigma is the uncertainty of ydata: a 1-D array of standard
	// deviations (the residuals are divided by it) or a 2-D covariance
	// matrix (the residuals are whitened by its Cholesky factor). nil means
	// unit weights.
	Sigma *ndarray.Array
	// AbsoluteSigma uses Sigma in an absolute sense: Pcov is not scaled by
	// the reduced chi-square.
	AbsoluteSigma bool
	// Bounds on the parameters; bounds select TRF.
	Bounds []Bound
	// Method is LM (the default without bounds) or TRF (with bounds).
	Method LSMethod
	// MaxFev, Ftol, Xtol and Gtol are passed to the solver (SciPy's maxfev
	// for LM, max_nfev for TRF; tolerances 1.49012e-8 for LM, 1e-8 for TRF).
	MaxFev           int
	Ftol, Xtol, Gtol float64
	// Loss and FScale select a robust loss (TRF only), as SciPy forwards
	// loss= and f_scale= to least_squares.
	Loss   Loss
	FScale float64
}

// CurveFitResult is the outcome of CurveFit: SciPy's (popt, pcov) plus its
// full_output information.
type CurveFitResult struct {
	// Popt holds the fitted parameters.
	Popt []float64
	// Pcov is the n×n covariance of Popt, row-major. When it cannot be
	// estimated every entry is +Inf and CovOK is false (SciPy warns).
	Pcov  []float64
	CovOK bool
	// Fvec holds the (weighted) residuals at Popt and Nfev the solver's
	// count of residual evaluations (SciPy's infodict['nfev']: MINPACK's
	// count including finite differences for LM, least_squares' count
	// excluding them for TRF).
	Fvec []float64
	Nfev int
	// Status and Message are the solver's (SciPy's ier and mesg).
	Status  int
	Message string
}

// PoptArray returns Popt as a 1-D ndarray.
func (r *CurveFitResult) PoptArray() *ndarray.Array { return vecArray(r.Popt) }

// PcovArray returns Pcov as an n×n ndarray.
func (r *CurveFitResult) PcovArray() *ndarray.Array {
	return matArray(r.Pcov, len(r.Popt), len(r.Popt))
}

// Perr returns the one-standard-deviation errors √diag(Pcov).
func (r *CurveFitResult) Perr() []float64 {
	n := len(r.Popt)
	e := make([]float64, n)
	for i := range e {
		e[i] = math.Sqrt(r.Pcov[i*n+i])
	}
	return e
}

// CurveFit fits a 1-D model f(x; p) to (xdata, ydata) by nonlinear least
// squares; it is scipy.optimize.curve_fit(f, xdata, ydata, p0, sigma=...,
// absolute_sigma=..., bounds=..., method=...). See CurveFitModel for models
// of several independent variables or with analytic derivatives.
func CurveFit(f func(x float64, p []float64) float64, xdata, ydata *ndarray.Array, p0 []float64, o *CurveFitOptions) (*CurveFitResult, error) {
	return CurveFitModel(Model{F: func(x, p []float64) float64 { return f(x[0], p) }}, xdata, ydata, p0, o)
}

// CurveFitModel is CurveFit for a general Model. xdata is 1-D (m samples)
// or 2-D (k, m); ydata is 1-D with m elements. p0 sets the number of
// parameters; nil starts from SciPy's default (ones, or a feasible point
// inside the bounds) when Bounds gives the count.
//
// Without bounds the default method is MINPACK's Levenberg–Marquardt and the
// covariance is SciPy leastsq's cov_x; with TRF it is the pseudo-inverse of
// JᵀJ from the SVD of the final Jacobian, discarding singular values below
// eps·max(m, n)·s₀. Unless AbsoluteSigma is set the covariance is scaled by
// the reduced chi-square Σr²/(m − n).
func CurveFitModel(model Model, xdata, ydata *ndarray.Array, p0 []float64, o *CurveFitOptions) (*CurveFitResult, error) {
	var opt CurveFitOptions
	if o != nil {
		opt = *o
	}
	if model.F == nil || xdata == nil || ydata == nil {
		return nil, fmt.Errorf("%w: CurveFit needs a model, xdata and ydata", ErrInput)
	}
	y := arrayData(ydata)
	m := len(y)
	if m == 0 {
		return nil, fmt.Errorf("%w: ydata must not be empty", ErrInput)
	}
	xs := arrayData(xdata)
	k := 1
	switch xdata.Ndim() {
	case 1:
	case 2:
		k = xdata.Shape()[0]
	default:
		return nil, fmt.Errorf("%w: xdata must be 1-D or 2-D", ErrDimension)
	}
	if len(xs) != k*m {
		return nil, fmt.Errorf("%w: xdata has %d samples, ydata %d", ErrDimension, len(xs)/k, m)
	}
	for _, v := range append(append([]float64(nil), xs...), y...) {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("%w: xdata and ydata must be finite", ErrInput)
		}
	}
	// Sample j as a k-vector (columns of a (k, m) xdata).
	samples := make([][]float64, m)
	for j := range samples {
		s := make([]float64, k)
		for i := range s {
			s[i] = xs[i*m+j]
		}
		samples[j] = s
	}
	n := len(p0)
	if p0 == nil {
		if opt.Bounds == nil {
			return nil, fmt.Errorf("%w: p0 or Bounds is needed to know the number of parameters", ErrInput)
		}
		n = len(opt.Bounds)
	}
	lb, ub, err := splitBounds(opt.Bounds, n)
	if err != nil {
		return nil, err
	}
	if p0 == nil {
		p0 = initializeFeasible(lb, ub)
	}
	method := opt.Method
	if method == "" {
		method = LM
		if opt.Bounds != nil {
			method = TRF
		}
	}
	if method == LM && opt.Bounds != nil {
		return nil, fmt.Errorf("%w: method lm only works for unconstrained problems; use TRF", ErrMethod)
	}
	// Weighting: r = W (f − y), with W = diag(1/σ) or L⁻¹ for C = L·Lᵀ.
	var invSigma, chol []float64
	if s := opt.Sigma; s != nil {
		sd := arrayData(s)
		switch {
		case len(sd) == 1 || (s.Ndim() == 1 && len(sd) == m):
			invSigma = make([]float64, m)
			for i := range invSigma {
				invSigma[i] = 1 / sd[min(i, len(sd)-1)]
			}
		case s.Ndim() == 2 && s.Shape()[0] == m && s.Shape()[1] == m:
			L, err := linalg.Cholesky(s, linalg.Lower)
			if err != nil {
				return nil, fmt.Errorf("%w: sigma must be positive definite", ErrInput)
			}
			chol = arrayData(L)
		default:
			return nil, fmt.Errorf("%w: sigma has incorrect shape", ErrDimension)
		}
	}
	weigh := func(r []float64, stride int) {
		// r holds m rows of stride values each.
		switch {
		case invSigma != nil:
			for i := range m {
				for c := range stride {
					r[i*stride+c] *= invSigma[i]
				}
			}
		case chol != nil:
			for c := range stride {
				for i := range m {
					s := r[i*stride+c]
					for j := range i {
						s -= float64(chol[i*m+j] * r[j*stride+c])
					}
					r[i*stride+c] = s / chol[i*m+i]
				}
			}
		}
	}
	resid := func(p, f []float64) {
		for j, x := range samples {
			f[j] = model.F(x, p) - y[j]
		}
		weigh(f, 1)
	}
	var jac func(p, J []float64)
	if model.Grad != nil {
		jac = func(p, J []float64) {
			for j, x := range samples {
				model.Grad(x, p, J[j*n:j*n+n])
			}
			weigh(J, n)
		}
	}

	res := &CurveFitResult{}
	var pcov []float64
	var cost float64
	if method == LM {
		if m != 1 && n > m {
			return nil, fmt.Errorf("%w: the number of parameters %d must not exceed the number of data points %d", ErrInput, n, m)
		}
		lo := &LeastsqOptions{Ftol: opt.Ftol, Xtol: opt.Xtol, Gtol: opt.Gtol, MaxFev: opt.MaxFev}
		lr, err := Leastsq(resid, jac, m, p0, lo)
		if err != nil {
			return nil, err
		}
		res.Popt, res.Fvec, res.Status, res.Message, res.Nfev = lr.X, lr.Fvec, lr.Ier, lr.Mesg, lr.Nfev
		cost = dot(lr.Fvec, lr.Fvec)
		pcov = lr.CovX
		if lr.Ier < 1 || lr.Ier > 4 {
			return res, fmt.Errorf("%w: %s", ErrNotConverged, lr.Mesg)
		}
	} else {
		lp := LSProblem{Func: resid, M: m}
		if jac != nil {
			lp.Jac = jac
		}
		lo := &LSOptions{Bounds: opt.Bounds, Ftol: opt.Ftol, Xtol: opt.Xtol, Gtol: opt.Gtol,
			MaxNfev: opt.MaxFev, Loss: opt.Loss, FScale: opt.FScale}
		lr, err := LeastSquares(lp, p0, method, lo)
		if err != nil {
			return nil, err
		}
		res.Popt, res.Fvec, res.Status, res.Message, res.Nfev = lr.X, lr.Fun, lr.Status, lr.Message, lr.Nfev
		if !lr.Success {
			return res, fmt.Errorf("%w: %s", ErrNotConverged, lr.Message)
		}
		cost = 2 * lr.Cost
		pcov, err = pinvNormal(lr.Jac, m, n)
		if err != nil {
			return nil, err
		}
	}
	res.CovOK = pcov != nil && !hasNaN(pcov)
	if res.CovOK && !opt.AbsoluteSigma {
		if m > n {
			s2 := cost / float64(m-n)
			for i := range pcov {
				pcov[i] *= s2
			}
		} else {
			res.CovOK = false
		}
	}
	if !res.CovOK {
		pcov = make([]float64, n*n)
		for i := range pcov {
			pcov[i] = math.Inf(1)
		}
	}
	res.Pcov = pcov
	return res, nil
}

// pinvNormal returns (JᵀJ)⁺ = V·diag(1/s²)·Vᵀ from the thin SVD of the m×n
// Jacobian J, dropping singular values at or below eps·max(m,n)·s₀, exactly
// as curve_fit does for the TRF method.
func pinvNormal(J []float64, m, n int) ([]float64, error) {
	_, s, vt, err := linalg.SVD(matArray(J, m, n), false)
	if err != nil {
		return nil, err
	}
	sv := arrayData(s)
	v := arrayData(vt) // k×n
	threshold := epsmch * float64(max(m, n)) * sv[0]
	pc := make([]float64, n*n)
	for r, sr := range sv {
		if !(sr > threshold) {
			break
		}
		w := 1 / (sr * sr)
		row := v[r*n : r*n+n]
		for i := range n {
			a := row[i] * w
			for j := range n {
				pc[i*n+j] += float64(a * row[j])
			}
		}
	}
	return pc, nil
}

// initializeFeasible is SciPy's _initialize_feasible: 1 for a free
// variable, the midpoint of a finite interval, and one unit inside a
// half-bounded one.
func initializeFeasible(lb, ub []float64) []float64 {
	p := make([]float64, len(lb))
	for i := range p {
		lo, hi := !math.IsInf(lb[i], 0), !math.IsInf(ub[i], 0)
		switch {
		case lo && hi:
			p[i] = 0.5 * (lb[i] + ub[i])
		case lo:
			p[i] = lb[i] + 1
		case hi:
			p[i] = ub[i] - 1
		default:
			p[i] = 1
		}
	}
	return p
}
