package optimize

import (
	"fmt"
	"math"
)

// Problem describes a scalar objective for Minimize. Func is required unless
// FuncGrad is set. Derivatives are optional: when neither Grad nor FuncGrad is
// given, gradient-based methods use finite differences with SciPy's step
// rules (see Options.Eps, Options.FiniteDiffRelStep, Options.FDMethod).
//
// Every callback receives the point first and writes its outputs into the
// slices that follow; it must not retain x.
type Problem struct {
	// Func returns f(x).
	Func func(x []float64) float64
	// Grad writes ∇f(x) into grad (SciPy's jac=callable).
	Grad func(x, grad []float64)
	// FuncGrad returns f(x) and writes ∇f(x) into grad in one call
	// (SciPy's jac=True). It takes precedence over Func and Grad.
	FuncGrad func(x, grad []float64) float64
	// Hess writes the n×n Hessian at x into hess, row-major (SciPy's hess).
	Hess func(x, hess []float64)
	// HessP writes the Hessian at x times p into hp (SciPy's hessp).
	HessP func(x, p, hp []float64)
}

// Method names a minimisation algorithm. The values are the strings SciPy's
// minimize accepts for its method argument.
type Method string

// Methods implemented by Minimize.
const (
	NelderMead Method = "Nelder-Mead"
	Powell     Method = "Powell"
	CG         Method = "CG"
	BFGS       Method = "BFGS"
	NewtonCG   Method = "Newton-CG"
	LBFGSB     Method = "L-BFGS-B"
)

// Options configures Minimize. Zero values select the SciPy defaults of the
// chosen method, listed with each field.
type Options struct {
	// Bounds constrains the variables (L-BFGS-B, Nelder-Mead and Powell, as in
	// SciPy). nil means unbounded.
	Bounds []Bound
	// Tol is minimize's tol argument: it sets the method's main tolerance
	// the way SciPy does (Nelder-Mead: Xatol and Fatol; Powell: Xtol and
	// Ftol; CG and BFGS: Gtol; Newton-CG: Xtol; L-BFGS-B: Ftol and Gtol)
	// when those fields are zero.
	Tol float64
	// MaxIter bounds the iterations (Nelder-Mead and Powell: 200·n by
	// default unless MaxFev is set; CG/BFGS: 200·n; Newton-CG: 200·n;
	// L-BFGS-B: 15000).
	MaxIter int
	// MaxFev bounds function evaluations (Nelder-Mead/Powell: 200·n by
	// default unless MaxIter is set; L-BFGS-B: 15000).
	MaxFev int
	// Gtol is the gradient tolerance: CG and BFGS stop when the Norm-norm
	// of the gradient is below it (1e-5); L-BFGS-B stops when the max-norm
	// of the projected gradient is below it (SciPy pgtol, 1e-5).
	Gtol float64
	// Ftol is the relative reduction tolerance: Powell (1e-4),
	// L-BFGS-B ((f_k - f_{k+1})/max(|f_k|,|f_{k+1}|,1) <= Ftol, 2.22e-9).
	Ftol float64
	// Xtol is the step tolerance: Powell (1e-4), Newton-CG (1e-5).
	Xtol float64
	// Xatol and Fatol are Nelder-Mead's absolute tolerances (1e-4 each).
	Xatol, Fatol float64
	// Adaptive makes Nelder-Mead adapt its coefficients to the dimension.
	Adaptive bool
	// InitialSimplex is Nelder-Mead's initial simplex, n+1 points.
	InitialSimplex [][]float64
	// Direc is Powell's initial set of directions, n vectors.
	Direc [][]float64
	// Norm is the order of the gradient norm for CG and BFGS (+Inf).
	Norm float64
	// Eps is the absolute finite-difference step for gradient-based
	// methods when no gradient is given (sqrt(eps) ≈ 1.49e-8 for CG and
	// BFGS, 1e-8 for L-BFGS-B). It is ignored when FiniteDiffRelStep is set.
	Eps float64
	// FiniteDiffRelStep selects a relative finite-difference step instead
	// of the absolute Eps (SciPy finite_diff_rel_step with jac='2-point').
	FiniteDiffRelStep float64
	// FDMethod is the finite-difference scheme (FD2Point).
	FDMethod FDMethod
	// M is the number of L-BFGS-B correction pairs (10).
	M int
	// MaxLS bounds L-BFGS-B line-search steps per iteration (20).
	MaxLS int
	// C1 and C2 are the Wolfe constants for CG (1e-4, 0.4) and BFGS
	// (1e-4, 0.9).
	C1, C2 float64
	// HessInv0 is BFGS's initial inverse Hessian, n*n row-major (identity).
	HessInv0 []float64
	// XRtol is BFGS's relative step tolerance (0: disabled).
	XRtol float64
	// Callback is called after each iteration with the current point and
	// objective value; returning true stops the minimisation, which then
	// reports Success=false with a "stopped by callback" message.
	Callback func(x []float64, f float64) (stop bool)
}

// Minimize minimises p from x0 with the given method; it is
// scipy.optimize.minimize(fun, x0, method=..., jac=..., hess=..., bounds=...,
// tol=..., options=...).
//
// Invalid input returns an error; non-convergence returns a Result with
// Success false.
func Minimize(p Problem, x0 []float64, method Method, opts *Options) (*Result, error) {
	if p.Func == nil && p.FuncGrad == nil {
		return nil, fmt.Errorf("%w: Problem.Func or Problem.FuncGrad is required", ErrInput)
	}
	if len(x0) == 0 {
		return nil, fmt.Errorf("%w: x0 is empty", ErrInput)
	}
	var o Options
	if opts != nil {
		o = *opts
	}
	n := len(x0)
	lb, ub, err := splitBounds(o.Bounds, n)
	if err != nil {
		return nil, err
	}
	if o.Bounds == nil {
		lb, ub = nil, nil
	}
	x0 = clone(x0)
	switch method {
	case NelderMead, Powell, LBFGSB:
		if lb != nil {
			// SciPy clips x0 into the bounds for these methods.
			for i := range x0 {
				x0[i] = math.Min(math.Max(x0[i], lb[i]), ub[i])
			}
		}
	case CG, BFGS, NewtonCG:
		if lb != nil {
			return nil, fmt.Errorf("%w: method %s cannot handle bounds", ErrMethod, method)
		}
	default:
		return nil, fmt.Errorf("%w: %q", ErrMethod, method)
	}
	sf := newScalarFunc(p, method, &o, lb, ub)
	switch method {
	case NelderMead:
		return minimizeNelderMead(sf, x0, &o, lb, ub)
	case Powell:
		return minimizePowell(sf, x0, &o, lb, ub)
	case CG:
		return minimizeCG(sf, x0, &o)
	case BFGS:
		return minimizeBFGS(sf, x0, &o)
	case NewtonCG:
		if p.Grad == nil && p.FuncGrad == nil {
			return nil, fmt.Errorf("%w: Newton-CG requires a gradient", ErrInput)
		}
		return minimizeNewtonCG(sf, x0, &o)
	default: // LBFGSB
		return minimizeLBFGSB(sf, x0, &o, lb, ub)
	}
}

// scalarFunc wraps a Problem with evaluation counters, a one-point cache and
// finite-difference gradients, the counterpart of SciPy's ScalarFunction: the
// objective and gradient are each computed (and counted) at most once per
// point, so a line search that asks for f and then ∇f at the same x pays once
// for each, and a finite-difference gradient reuses the cached f(x).
type scalarFunc struct {
	p                Problem
	fd               fdStep
	nfev, njev, nhev int

	x        []float64 // point of the cache
	f        float64
	g        []float64
	fOK, gOK bool // f and g are valid (and counted) at x
	rawOK    bool // FuncGrad has filled f and g at x
}

func newScalarFunc(p Problem, method Method, o *Options, lb, ub []float64) *scalarFunc {
	fd := fdStep{method: o.FDMethod, lb: lb, ub: ub}
	switch {
	case o.FiniteDiffRelStep != 0:
		fd.rel = o.FiniteDiffRelStep
	case o.Eps != 0:
		fd.abs = o.Eps
	case o.FDMethod == FD3Point:
		// SciPy passes its default epsilon only for the '2-point' scheme.
	case method == LBFGSB:
		fd.abs = 1e-8
	default:
		fd.abs = math.Sqrt(epsmch)
	}
	return &scalarFunc{p: p, fd: fd}
}

// hasGrad reports whether the problem provides an analytic gradient.
func (s *scalarFunc) hasGrad() bool { return s.p.Grad != nil || s.p.FuncGrad != nil }

// moveTo makes x the cached point, invalidating the cache if it moved.
func (s *scalarFunc) moveTo(x []float64) {
	if s.x != nil {
		same := true
		for i := range x {
			if x[i] != s.x[i] {
				same = false
				break
			}
		}
		if same {
			return
		}
		copy(s.x, x)
	} else {
		s.x = clone(x)
		s.g = make([]float64, len(x))
	}
	s.fOK, s.gOK, s.rawOK = false, false, false
}

// raw fills f and g from FuncGrad at the cached point.
func (s *scalarFunc) raw() {
	if !s.rawOK {
		s.f = s.p.FuncGrad(s.x, s.g)
		s.rawOK = true
	}
}

// fun returns f(x).
func (s *scalarFunc) fun(x []float64) float64 {
	s.moveTo(x)
	if !s.fOK {
		s.nfev++
		if s.p.FuncGrad != nil {
			s.raw()
		} else {
			s.f = s.p.Func(s.x)
		}
		s.fOK = true
	}
	return s.f
}

// grad writes ∇f(x) into g.
func (s *scalarFunc) grad(x, g []float64) {
	s.moveTo(x)
	if !s.gOK {
		s.njev++
		switch {
		case s.p.FuncGrad != nil:
			s.raw()
		case s.p.Grad != nil:
			s.p.Grad(s.x, s.g)
		default:
			f0 := s.fun(s.x)
			s.nfev += s.fd.gradFD(s.p.Func, s.x, f0, s.g)
		}
		s.gOK = true
	}
	copy(g, s.g)
}

// funGrad evaluates f(x) and writes ∇f(x) into g.
func (s *scalarFunc) funGrad(x, g []float64) float64 {
	f := s.fun(x)
	s.grad(x, g)
	return f
}

// result fills the evaluation counters of r.
func (s *scalarFunc) result(r *Result) *Result {
	r.Nfev, r.Njev, r.Nhev = s.nfev, s.njev, s.nhev
	return r
}
