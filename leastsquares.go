package optimize

import (
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
)

// LSProblem is a nonlinear least-squares problem: minimise
// ½·Σ ρ(f_i(x)²) over x, with m residuals f_i and n variables.
type LSProblem struct {
	// Func writes the m residuals at x into f. Required.
	Func func(x, f []float64)
	// Jac writes the m×n Jacobian at x into jac, row-major. Optional:
	// finite differences are used when it is nil.
	Jac func(x, jac []float64)
	// M is the number of residuals. Required.
	M int
}

// LSMethod names a least-squares algorithm; the values are SciPy's
// least_squares method strings.
type LSMethod string

// Least-squares methods.
const (
	// TRF is the trust-region reflective method (SciPy's default), which
	// supports bounds and robust loss functions.
	TRF LSMethod = "trf"
	// LM is the Levenberg–Marquardt method of MINPACK (lmdif/lmder); it
	// needs m >= n and supports neither bounds nor robust losses.
	LM LSMethod = "lm"
)

// Loss names a robust loss ρ for LeastSquares (SciPy's loss argument).
type Loss string

// Loss functions, with z = (f/FScale)².
const (
	LossLinear Loss = "linear"  // ρ(z) = z
	LossSoftL1 Loss = "soft_l1" // ρ(z) = 2((1+z)^½ − 1)
	LossHuber  Loss = "huber"   // ρ(z) = z if z <= 1 else 2z^½ − 1
	LossCauchy Loss = "cauchy"  // ρ(z) = ln(1+z)
	LossArctan Loss = "arctan"  // ρ(z) = arctan(z)
)

// LSOptions configures LeastSquares. Zero values select SciPy's defaults.
type LSOptions struct {
	// Bounds constrains the variables (TRF only). nil means unbounded.
	Bounds []Bound
	// Ftol, Xtol and Gtol are the termination tolerances (1e-8 each).
	// SciPy lets None disable one; here a negative value disables it.
	Ftol, Xtol, Gtol float64
	// MaxNfev bounds function evaluations (100·n; for LM it bounds
	// MINPACK's count, which leaves out finite-difference evaluations).
	MaxNfev int
	// XScale is the characteristic scale of each variable. nil means 1
	// (TRF) or 'jac' (LM, SciPy's default for that method); XScaleJac
	// selects SciPy's x_scale='jac'.
	XScale    []float64
	XScaleJac bool
	// Loss is the robust loss (TRF only; LossLinear by default) and FScale
	// its soft margin between inliers and outliers (1).
	Loss   Loss
	FScale float64
	// DiffStep is the relative finite-difference step (SciPy diff_step);
	// 0 selects the default of the scheme.
	DiffStep float64
	// FDMethod is the finite-difference scheme for TRF (FD2Point). LM uses
	// MINPACK's own forward differences.
	FDMethod FDMethod
}

// LSResult is the outcome of LeastSquares, mirroring the OptimizeResult of
// scipy.optimize.least_squares.
type LSResult struct {
	// X is the solution.
	X []float64
	// Cost is ½·Σρ(f_i²) at X.
	Cost float64
	// Fun holds the m residuals at X.
	Fun []float64
	// Jac is the m×n Jacobian at X, row-major (modified by the loss, as in
	// SciPy).
	Jac []float64
	// Grad is the gradient of the cost at X.
	Grad []float64
	// Optimality is the first-order optimality measure (max-norm of the
	// scaled gradient, as SciPy defines it per method).
	Optimality float64
	// ActiveMask is -1 or +1 for a variable at its lower or upper bound,
	// 0 otherwise.
	ActiveMask []int
	// Nfev and Njev count residual and Jacobian evaluations.
	Nfev, Njev int
	// Status follows SciPy: -1 improper input, 0 max_nfev reached, 1 gtol,
	// 2 ftol, 3 xtol, 4 both ftol and xtol satisfied.
	Status int
	// Success is Status > 0.
	Success bool
	// Message describes the termination.
	Message string
}

// XArray returns X as a 1-D ndarray.
func (r *LSResult) XArray() *ndarray.Array { return vecArray(r.X) }

// JacArray returns Jac as an m×n ndarray.
func (r *LSResult) JacArray() *ndarray.Array { return matArray(r.Jac, len(r.Fun), len(r.X)) }

// LeastSquares solves a nonlinear least-squares problem from x0; it is
// scipy.optimize.least_squares(fun, x0, jac=..., bounds=..., method=...,
// ftol=..., xtol=..., gtol=..., x_scale=..., loss=..., f_scale=...,
// diff_step=..., max_nfev=...). method "" selects TRF.
func LeastSquares(p LSProblem, x0 []float64, method LSMethod, opts *LSOptions) (*LSResult, error) {
	if p.Func == nil || p.M <= 0 || len(x0) == 0 {
		return nil, fmt.Errorf("%w: LSProblem needs Func, M > 0 and a non-empty x0", ErrInput)
	}
	var o LSOptions
	if opts != nil {
		o = *opts
	}
	n := len(x0)
	if method == "" {
		method = TRF
	}
	if o.Loss == "" {
		o.Loss = LossLinear
	}
	if o.FScale == 0 {
		o.FScale = 1
	}
	if o.MaxNfev < 0 {
		return nil, fmt.Errorf("%w: MaxNfev must not be negative", ErrInput)
	}
	if o.FScale < 0 {
		return nil, fmt.Errorf("%w: FScale must be positive", ErrInput)
	}
	for _, t := range []*float64{&o.Ftol, &o.Xtol, &o.Gtol} {
		if *t == 0 {
			*t = 1e-8
		} else if *t < 0 {
			*t = 0
		}
	}
	if o.XScale != nil {
		if len(o.XScale) != n {
			return nil, ErrDimension
		}
		for _, s := range o.XScale {
			if !(s > 0) || math.IsInf(s, 1) {
				return nil, fmt.Errorf("%w: XScale must be positive and finite", ErrInput)
			}
		}
	}
	lb, ub, err := splitBounds(o.Bounds, n)
	if err != nil {
		return nil, err
	}
	for i := range x0 {
		if lb[i] >= ub[i] {
			return nil, fmt.Errorf("%w: each lower bound must be strictly less than its upper bound", ErrBounds)
		}
		if x0[i] < lb[i] || x0[i] > ub[i] {
			return nil, fmt.Errorf("%w: x0 is infeasible", ErrBounds)
		}
	}
	switch method {
	case LM:
		if o.Bounds != nil {
			return nil, fmt.Errorf("%w: method lm does not support bounds", ErrMethod)
		}
		if o.Loss != LossLinear {
			return nil, fmt.Errorf("%w: method lm supports only the linear loss", ErrMethod)
		}
		if p.M < n {
			return nil, fmt.Errorf("%w: method lm needs at least as many residuals as variables", ErrMethod)
		}
		return lsLM(p, clone(x0), &o)
	case TRF:
		switch o.Loss {
		case LossLinear, LossSoftL1, LossHuber, LossCauchy, LossArctan:
		default:
			return nil, fmt.Errorf("%w: unknown loss %q", ErrInput, o.Loss)
		}
		return lsTRF(p, clone(x0), lb, ub, &o)
	}
	return nil, fmt.Errorf("%w: %q", ErrMethod, method)
}

// lsMessages are SciPy least_squares' TERMINATION_MESSAGES.
var lsMessages = map[int]string{
	-1: "Improper input parameters status returned from `leastsq`",
	0:  "The maximum number of function evaluations is exceeded.",
	1:  "`gtol` termination condition is satisfied.",
	2:  "`ftol` termination condition is satisfied.",
	3:  "`xtol` termination condition is satisfied.",
	4:  "Both `ftol` and `xtol` termination conditions are satisfied.",
}
