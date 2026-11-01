package optimize

import (
	"fmt"
	"math"
)

// DirectOptions configure Direct. The zero value (or a nil pointer) selects
// SciPy's defaults of scipy.optimize.direct: DIRECT-L, eps 1e-4, 1000*N
// evaluations, 1000 iterations.
type DirectOptions struct {
	// Eps is Jones' epsilon, which keeps the search from over-refining
	// near the current best. 0 selects 1e-4; a negative value selects
	// Jones' update eps = max(1e-4*|minf|, |Eps|) after each iteration (the
	// C code's convention, which SciPy passes through). SciPy's eps=0 is
	// not expressible (0 means the default here).
	Eps float64
	// MaxFev is an approximate upper bound on the number of objective
	// evaluations: the budget is checked once per iteration, after all its
	// divisions, so Nfev can exceed it by one iteration's samples. 0 selects
	// 1000*N. SciPy's maxfun=0 (which the C code turns into 100000) is not
	// expressible. Odd budgets are rounded down by the C code's sizing
	// arithmetic (MaxFev 1001 becomes 1000).
	MaxFev int
	// MaxIter is the maximum number of iterations; the C loop runs at most
	// MaxIter-2 of them. 0 selects 1000.
	MaxIter int
	// Original selects Jones' original DIRECT; false selects the locally
	// biased DIRECT-L (SciPy's locally_biased=True, the default).
	Original bool
	// FMin is the known global minimum; nil means unknown.
	FMin *float64
	// FMinRtol stops the search when (fun - FMin)/max(1, |FMin|) <= FMinRtol
	// (FMin != 0 and unknown divides by |FMin|, as the C code does). 0
	// selects 1e-4; negative disables the test. Must be <= 1.
	FMinRtol float64
	// VolTol stops the search when the volume of the hyperrectangle that
	// holds the best point falls below this fraction of the box. 0 selects
	// 1e-16; negative disables the test. Must be <= 1.
	VolTol float64
	// LenTol stops the search when the side-length measure of the
	// hyperrectangle that holds the best point falls below it (half the
	// longest side for DIRECT-L, half the diagonal for the original). 0
	// selects 1e-6; negative disables the test. Must be <= 1.
	LenTol float64
	// Workers evaluates the objective at the points sampled inside one
	// rectangle concurrently (2*k points for k longest sides). The result
	// does not depend on the worker count. The objective must then be safe
	// for concurrent use and must not retain its argument. 0 or 1 is serial.
	Workers int
	// Callback is called after each iteration with a copy of the last
	// evaluated point (SciPy's callback receives this point, although its
	// docstring says "best"), a copy of the best point and its value.
	// Returning true stops the search with status -102, "Forced stop".
	Callback func(xLast, xBest []float64, fBest float64) (stop bool)
}

// DirectResult is the outcome of Direct, the counterpart of the
// OptimizeResult of scipy.optimize.direct.
type DirectResult struct {
	// X is the best point found; Fun its value.
	X   []float64
	Fun float64
	// Nfev is the number of objective evaluations, Nit the iteration at which
	// the search ended.
	Nfev, Nit int
	// Status is SciPy's ret_code: 1 (MaxFev), 2 (MaxIter), 3 (FMin reached),
	// 4 (VolTol), 5 (LenTol), or negative for an error (-6: maximum number of
	// levels reached; -102: stopped by the callback).
	Status int
	// Success is Status > 2.
	Success bool
	// Message describes the reason for termination.
	Message string
}

// Direct finds the global minimum of f over the box bounds with the DIRECT
// algorithm of Jones, Perttunen and Stuckman (1993) or its locally biased
// variant DIRECT-L of Gablonsky and Kelley (2001), following SciPy 1.18.1's
// scipy.optimize.direct, which wraps Gablonsky's C code. The search is
// deterministic and takes SciPy's path: same points, same Nfev and Nit.
//
// Every bound must be finite with Lo < Hi. f must not retain or modify its
// argument. A NaN objective value is stored like any other value, as in
// SciPy: it is never smaller than anything, so such points are never the
// best. Invalid input returns an error wrapping ErrInput or ErrBounds before
// any objective call.
func Direct(f func(x []float64) float64, bounds []Bound, o *DirectOptions) (*DirectResult, error) {
	if f == nil {
		return nil, fmt.Errorf("%w: nil objective", ErrInput)
	}
	if o == nil {
		o = &DirectOptions{}
	}
	n := len(bounds)
	if n < 1 {
		return nil, fmt.Errorf("%w: DIRECT requires at least one dimension", ErrInput)
	}
	if n > 1<<20 {
		return nil, fmt.Errorf("%w: too many dimensions (%d)", ErrInput, n)
	}
	lb := make([]float64, n)
	ub := make([]float64, n)
	for i, b := range bounds {
		if !(b.Lo < b.Hi) {
			return nil, fmt.Errorf("%w: Bounds are not consistent min < max", ErrBounds)
		}
		lb[i], ub[i] = b.Lo, b.Hi
	}
	for i := range lb {
		if math.IsInf(lb[i], 0) || math.IsInf(ub[i], 0) {
			return nil, fmt.Errorf("%w: Bounds must not be inf", ErrBounds)
		}
		if math.IsInf(ub[i]-lb[i], 0) {
			return nil, fmt.Errorf("%w: bound %d spans more than the largest float", ErrBounds, i)
		}
	}
	eps := o.Eps
	if eps == 0 {
		eps = 1e-4
	}
	if math.IsNaN(eps) || math.IsInf(eps, 0) {
		return nil, fmt.Errorf("%w: Eps must be finite", ErrInput)
	}
	volTol, lenTol, rtol := dirTol(o.VolTol, 1e-16), dirTol(o.LenTol, 1e-6), dirTol(o.FMinRtol, 1e-4)
	for _, tc := range []struct {
		v    float64
		name string
	}{{volTol, "VolTol"}, {lenTol, "LenTol"}, {rtol, "FMinRtol"}} {
		if tc.v > 1 || math.IsNaN(tc.v) {
			return nil, fmt.Errorf("%w: %s must be between 0 and 1", ErrInput, tc.name)
		}
	}
	if o.MaxFev < 0 || o.MaxIter < 0 {
		return nil, fmt.Errorf("%w: MaxFev and MaxIter must not be negative", ErrInput)
	}
	maxfun, maxiter := o.MaxFev, o.MaxIter
	if maxfun == 0 {
		maxfun = 1000 * n
	}
	if maxiter == 0 {
		maxiter = 1000
	}
	// The C code stores both in an int and adds 1000 (and 500%) to them.
	if int64(maxfun) > math.MaxInt32-2000 || int64(maxiter) > math.MaxInt32-2000 {
		return nil, fmt.Errorf("%w: MaxFev or MaxIter too large", ErrInput)
	}
	fglobal := math.Inf(-1)
	if o.FMin != nil {
		fglobal = *o.FMin
	}
	// direct_optimize: volper, sigmaper <= 0 are ignored, and fglper is
	// meaningless without a known minimum.
	fglper := rtol
	if math.IsInf(fglobal, -1) {
		fglper = 0
	} else if rtol < 0 {
		fglper = math.Inf(-1)
	}
	vol, sig := volTol, lenTol
	if vol <= 0 {
		vol = -1
	}
	if sig <= 0 {
		sig = -1
	}
	cb := o.Callback
	out := directCore(func(x []float64) (float64, int) { return f(x), 0 }, lb, ub, dirParams{
		eps: eps, maxf: int64(maxfun), maxt: int64(maxiter), original: o.Original,
		fglobal: fglobal, fglper: fglper, volper: vol, sigmaper: sig, workers: o.Workers,
	}, cb)
	return &DirectResult{
		X: out.x, Fun: out.fun, Nfev: out.numfunc, Nit: out.numiter,
		Status:  out.ierror,
		Success: out.ierror > 2,
		Message: directMessage(out.ierror, maxfun, maxiter, rtol, volTol, lenTol),
	}, nil
}

// dirTol applies the 0-selects-default convention: negative stays negative
// (disabled).
func dirTol(v, def float64) float64 {
	if v == 0 {
		return def
	}
	return v
}

var directErrorMessages = [...]string{
	"Number of function evaluations done is larger than maxfun=%s",
	"Number of iterations is larger than maxiter=%s",
	"u[i] < l[i] for some i",
	"maxfun is too large",
	"Initialization failed",
	"There was an error in the creation of the sample points",
	"An error occurred while the function was sampled",
	"Maximum number of levels has been reached.",
	"Forced stop",
	"Invalid arguments",
	"Out of memory",
}

var directSuccessMessages = [...]string{
	"The best function value found is within a relative error=%s of the (known) global optimum f_min",
	"The volume of the hyperrectangle containing the lowest function value found is below vol_tol=%s",
	"The side length measure of the hyperrectangle containing the lowest function value found is below len_tol=%s",
}

// directMessage is the message selection of _direct_py.py (lines 267-276):
// success messages for codes above 2, error messages indexed as in Python,
// including its negative index for codes -100 and below.
func directMessage(code, maxfun, maxiter int, rtol, vol, length float64) string {
	val := [...]string{fmt.Sprint(maxfun), fmt.Sprint(maxiter), pyFloat(rtol).String(), pyFloat(vol).String(), pyFloat(length).String()}
	switch {
	case code > 2:
		return fmt.Sprintf(directSuccessMessages[code-3], val[code-1])
	case code > 0:
		return fmt.Sprintf(directErrorMessages[code-1], val[code-1])
	case code > -100:
		return directErrorMessages[-code+1]
	}
	i := code + 99
	if i < 0 {
		i += len(directErrorMessages)
	}
	return directErrorMessages[i]
}
