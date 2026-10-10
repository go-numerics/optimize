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
	// MaxIter bounds the iterations (Nelder-Mead 200·n, Powell 1000·n,
	// CG, BFGS and Newton-CG 200·n, L-BFGS-B 15000). For Nelder-Mead and
	// Powell, setting only one of MaxIter and MaxFev leaves the other
	// unlimited, as in SciPy.
	MaxIter int
	// MaxFev bounds function evaluations (Nelder-Mead 200·n, Powell 1000·n,
	// L-BFGS-B 15000).
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
	// Callback is called after each iteration with a copy of the current
	// point and the objective value; returning true stops the minimisation,
	// which then reports Success false and Status 99 (SciPy's callback
	// raising StopIteration).
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
	if method == NewtonCG && p.Grad == nil && p.FuncGrad == nil {
		return nil, fmt.Errorf("%w: Newton-CG requires a gradient", ErrInput)
	}
	stopped := false
	if cb := o.Callback; cb != nil {
		o.Callback = func(x []float64, f float64) bool {
			stopped = cb(x, f)
			return stopped
		}
	}
	var r *Result
	if method == LBFGSB && lb != nil {
		r, err = minimizeLBFGSBFixed(p, x0, &o, lb, ub)
	} else {
		r, err = dispatch(p, x0, method, &o, lb, ub)
	}
	if err == nil && stopped {
		r.Success, r.Status, r.Message = false, 99, "`Callback` requested a stop."
	}
	return r, err
}

func dispatch(p Problem, x0 []float64, method Method, o *Options, lb, ub []float64) (*Result, error) {
	sf := newScalarFunc(p, method, o, lb, ub)
	switch method {
	case NelderMead:
		return minimizeNelderMead(sf, x0, o, lb, ub)
	case Powell:
		return minimizePowell(sf, x0, o, lb, ub)
	case CG:
		return minimizeCG(sf, x0, o)
	case BFGS:
		return minimizeBFGS(sf, x0, o)
	case NewtonCG:
		return minimizeNewtonCG(sf, x0, o)
	default: // LBFGSB
		return minimizeLBFGSB(sf, x0, o, lb, ub)
	}
}

// minimizeLBFGSBFixed handles variables fixed by equal bounds the way SciPy's
// minimize does for L-BFGS-B: if all are fixed the objective is evaluated
// once at the bounds; if some are fixed and the gradient is computed by
// finite differences (which could not step off a fixed variable), they are
// removed from the problem and restored in the result, with NaN gradient
// entries.
func minimizeLBFGSBFixed(p Problem, x0 []float64, o *Options, lb, ub []float64) (*Result, error) {
	n := len(x0)
	var free []int
	for i := range n {
		if lb[i] != ub[i] {
			free = append(free, i)
		}
	}
	if len(free) == 0 {
		x := clone(lb)
		sf := newScalarFunc(p, LBFGSB, o, nil, nil)
		f := sf.fun(x)
		return &Result{X: x, Fun: f, Nfev: 1, Success: true,
			Message: "All independent variables were fixed by bounds."}, nil
	}
	if len(free) == n || p.Grad != nil || p.FuncGrad != nil {
		return dispatch(p, x0, LBFGSB, o, lb, ub)
	}
	full := clone(lb) // fixed entries hold their value
	expand := func(xr []float64) []float64 {
		for k, i := range free {
			full[i] = xr[k]
		}
		return full
	}
	m := len(free)
	xr, lr, ur := make([]float64, m), make([]float64, m), make([]float64, m)
	for k, i := range free {
		xr[k], lr[k], ur[k] = x0[i], lb[i], ub[i]
	}
	f := p.Func
	rp := Problem{Func: func(x []float64) float64 { return f(expand(x)) }}
	ro := *o
	if cb := o.Callback; cb != nil {
		ro.Callback = func(x []float64, fx float64) bool { return cb(clone(expand(x)), fx) }
	}
	r, err := dispatch(rp, xr, LBFGSB, &ro, lr, ur)
	if err != nil {
		return nil, err
	}
	x := clone(lb)
	jac := make([]float64, n)
	for i := range jac {
		jac[i] = math.NaN()
	}
	for k, i := range free {
		x[i] = r.X[k]
		if r.Jac != nil {
			jac[i] = r.Jac[k]
		}
	}
	r.X, r.Jac = x, jac
	return r, nil
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
