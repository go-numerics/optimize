package optimize

import (
	"errors"
	"math"

	"github.com/go-ndarray/ndarray"
)

// Errors returned for invalid input. Failure to converge is not an error: it
// is reported through the Success, Status and Message fields of a result, as
// SciPy does, so the caller still receives the best point found.
var (
	// ErrDimension reports slices whose lengths disagree.
	ErrDimension = errors.New("optimize: dimension mismatch")
	// ErrMethod reports an unknown or unsupported method.
	ErrMethod = errors.New("optimize: unknown method")
	// ErrBounds reports inconsistent bounds or a starting point outside them.
	ErrBounds = errors.New("optimize: invalid bounds")
	// ErrBracket reports a scalar root bracket whose ends do not change sign.
	ErrBracket = errors.New("optimize: f(a) and f(b) must have different signs")
	// ErrInput reports any other invalid argument.
	ErrInput = errors.New("optimize: invalid input")
)

// Bound is a closed interval [Lo, Hi] for one variable. Use math.Inf(-1) and
// math.Inf(1) for a missing side; the zero value Bound{} pins the variable to
// zero, so use Unbounded() for a free variable.
type Bound struct {
	Lo, Hi float64
}

// Unbounded returns the free interval (-Inf, +Inf).
func Unbounded() Bound { return Bound{math.Inf(-1), math.Inf(1)} }

// splitBounds turns bounds into lower and upper vectors of length n. A nil
// slice means no bounds; otherwise len(b) must be n and Lo <= Hi.
func splitBounds(b []Bound, n int) (lb, ub []float64, err error) {
	lb = make([]float64, n)
	ub = make([]float64, n)
	if b == nil {
		for i := range lb {
			lb[i], ub[i] = math.Inf(-1), math.Inf(1)
		}
		return lb, ub, nil
	}
	if len(b) != n {
		return nil, nil, ErrDimension
	}
	for i, bi := range b {
		if math.IsNaN(bi.Lo) || math.IsNaN(bi.Hi) || bi.Lo > bi.Hi {
			return nil, nil, ErrBounds
		}
		lb[i], ub[i] = bi.Lo, bi.Hi
	}
	return lb, ub, nil
}

// Result is the outcome of Minimize, the counterpart of SciPy's
// OptimizeResult for scipy.optimize.minimize.
type Result struct {
	// X is the solution.
	X []float64
	// Fun is the objective value at X.
	Fun float64
	// Jac is the gradient at X, when the method computes one.
	Jac []float64
	// HessInv is the final inverse-Hessian approximation as an n*n row-major
	// matrix (BFGS only, nil otherwise).
	HessInv []float64
	// Nit, Nfev, Njev and Nhev count iterations and evaluations of the
	// objective, the gradient and the Hessian. Function evaluations spent on
	// finite-difference gradients are counted in Nfev, and each such gradient
	// also counts once in Njev, as in SciPy.
	Nit, Nfev, Njev, Nhev int
	// Status is a method-specific termination code; 0 means success.
	Status int
	// Success reports whether the method met its convergence criteria.
	Success bool
	// Message describes the reason for termination.
	Message string
}

// XArray returns X as a 1-D ndarray.
func (r *Result) XArray() *ndarray.Array { return vecArray(r.X) }

// HessInvArray returns HessInv as an n×n ndarray, or nil when absent.
func (r *Result) HessInvArray() *ndarray.Array {
	if r.HessInv == nil {
		return nil
	}
	n := len(r.X)
	a, _ := ndarray.FromData(r.HessInv, n, n)
	return a
}

// vecArray wraps a vector as a 1-D ndarray (copying it).
func vecArray(v []float64) *ndarray.Array {
	if v == nil {
		return nil
	}
	a, _ := ndarray.FromData(v, len(v))
	return a
}

// matArray wraps an r×c row-major matrix as an ndarray (copying it).
func matArray(m []float64, r, c int) *ndarray.Array {
	if m == nil {
		return nil
	}
	a, _ := ndarray.FromData(m, r, c)
	return a
}

// arrayData returns the elements of a as a float64 slice in row-major order,
// converting other dtypes. It never aliases a.
func arrayData(a *ndarray.Array) []float64 {
	if a == nil {
		return nil
	}
	if a.DType() != ndarray.Float64 {
		a = a.AsType(ndarray.Float64)
	}
	d, _ := ndarray.Data[float64](a)
	return append([]float64(nil), d...)
}

// Fortran-free small vector helpers shared by the methods.

func dot(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += float64(a[i] * b[i])
	}
	return s
}

// norm returns the vector norm of order ord: math.Inf(1) for the maximum
// absolute value, math.Inf(-1) for the minimum, and the p-norm otherwise
// (numpy.linalg.norm semantics).
func norm(v []float64, ord float64) float64 {
	switch {
	case math.IsInf(ord, 1):
		m := 0.0
		for _, x := range v {
			m = math.Max(m, math.Abs(x))
		}
		return m
	case math.IsInf(ord, -1):
		m := math.Inf(1)
		for _, x := range v {
			m = math.Min(m, math.Abs(x))
		}
		return m
	case ord == 2:
		return enorm(v)
	}
	s := 0.0
	for _, x := range v {
		s += math.Pow(math.Abs(x), ord)
	}
	return math.Pow(s, 1/ord)
}

// enorm is the overflow- and underflow-safe Euclidean norm.
func enorm(v []float64) float64 {
	scale, ssq := 0.0, 1.0
	for _, x := range v {
		if x == 0 {
			continue
		}
		ax := math.Abs(x)
		if scale < ax {
			ssq = 1 + float64(ssq*(scale/ax)*(scale/ax))
			scale = ax
		} else {
			ssq += float64((ax / scale) * (ax / scale))
		}
	}
	if math.IsInf(scale, 1) {
		return scale
	}
	return float64(scale * math.Sqrt(ssq))
}

func clone(v []float64) []float64 { return append([]float64(nil), v...) }

const epsmch = 2.220446049250313e-16 // float64 machine epsilon
