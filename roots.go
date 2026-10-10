package optimize

import (
	"fmt"
	"math"
)

// Defaults of SciPy's bracketing root finders.
const (
	defaultRootXtol = 2e-12
	defaultRootRtol = 4 * epsmch
	defaultRootIter = 100
)

// Root-finder flags, as SciPy's RootResults.flag strings.
const (
	FlagConverged = "converged"
	FlagConvErr   = "convergence error"
	FlagSignErr   = "sign error"
)

// RootResult describes a scalar root search, like SciPy's RootResults.
type RootResult struct {
	// Root is the estimated root.
	Root float64
	// Iterations and FunctionCalls count the work done (derivative
	// evaluations count as function calls for Newton and Halley).
	Iterations, FunctionCalls int
	// Converged reports whether the tolerance was met.
	Converged bool
	// Flag is "converged", "convergence error" or a message describing the
	// failure.
	Flag string
	// Method is the algorithm used.
	Method string
}

// BracketOptions configures the bracketing root finders. Zero values select
// SciPy's defaults: Xtol 2e-12, Rtol 4·eps (the smallest allowed),
// MaxIter 100.
type BracketOptions struct {
	Xtol, Rtol float64
	MaxIter    int
}

type bracketSolver func(f func(float64) float64, a, b, xtol, rtol float64, iter int, st *RootResult) float64

// nanError reports a NaN function value, which stops every root finder.
type nanError struct {
	x     float64
	calls int
}

func (e nanError) Error() string {
	return fmt.Sprintf("optimize: the function value at x=%s is NaN; solver cannot continue", pyRepr(e.x))
}

func solveBracket(name string, solver bracketSolver, f func(float64) float64, a, b float64, o *BracketOptions) (*RootResult, error) {
	var opt BracketOptions
	if o != nil {
		opt = *o
	}
	xtol, rtol, iter := opt.Xtol, opt.Rtol, opt.MaxIter
	if xtol == 0 {
		xtol = defaultRootXtol
	}
	if rtol == 0 {
		rtol = defaultRootRtol
	}
	if iter == 0 {
		iter = defaultRootIter
	}
	if xtol <= 0 {
		return nil, fmt.Errorf("%w: xtol too small (%g <= 0)", ErrInput, xtol)
	}
	if rtol < defaultRootRtol {
		return nil, fmt.Errorf("%w: rtol too small (%g < %g)", ErrInput, rtol, defaultRootRtol)
	}
	if iter < 0 {
		return nil, fmt.Errorf("%w: maxiter must be non-negative", ErrInput)
	}
	calls := 0
	var bad *nanError
	g := func(x float64) float64 {
		if bad != nil {
			return 0
		}
		fx := f(x)
		calls++
		if math.IsNaN(fx) {
			bad = &nanError{x, calls}
		}
		return fx
	}
	st := &RootResult{Method: name}
	root := solver(g, a, b, xtol, rtol, iter, st)
	if bad != nil {
		return &RootResult{Root: bad.x, FunctionCalls: bad.calls, Flag: bad.Error(), Method: name}, *bad
	}
	if st.Flag == FlagSignErr {
		return nil, ErrBracket
	}
	st.Root = root
	return st, nil
}

// Brentq finds a root of f in [a, b], where f(a) and f(b) have opposite
// signs, with Brent's (1973) method using inverse quadratic extrapolation;
// it is scipy.optimize.brentq(f, a, b, xtol, rtol, maxiter,
// full_output=True). The returned root x satisfies |x − x₀| <= 4·(Xtol +
// Rtol·|x₀|) for the exact root x₀.
//
// A bracket without a sign change returns ErrBracket; a NaN function value
// returns an error together with the result describing where it happened;
// running out of iterations is reported by Converged false.
func Brentq(f func(float64) float64, a, b float64, o *BracketOptions) (*RootResult, error) {
	return solveBracket("brentq", brentq, f, a, b, o)
}

// Brenth is Brentq with hyperbolic instead of inverse quadratic
// extrapolation (Bus & Dekker 1975); it is scipy.optimize.brenth.
func Brenth(f func(float64) float64, a, b float64, o *BracketOptions) (*RootResult, error) {
	return solveBracket("brenth", brenth, f, a, b, o)
}

// Ridder finds a root of f in [a, b] with Ridders' (1979) method; it is
// scipy.optimize.ridder.
func Ridder(f func(float64) float64, a, b float64, o *BracketOptions) (*RootResult, error) {
	return solveBracket("ridder", ridder, f, a, b, o)
}

// Bisect finds a root of f in [a, b] by bisection; it is
// scipy.optimize.bisect.
func Bisect(f func(float64) float64, a, b float64, o *BracketOptions) (*RootResult, error) {
	return solveBracket("bisect", bisect, f, a, b, o)
}

// The four bracketing solvers below follow SciPy's C implementations
// (scipy/optimize/Zeros/*.c, written by Charles Harris, BSD licence)
// statement for statement, so that iterates and call counts agree.

func sameSign(x, y float64) bool { return math.Signbit(x) == math.Signbit(y) }

// bracketStart evaluates both ends and settles the trivial cases.
func bracketStart(f func(float64) float64, a, b float64, st *RootResult) (fa, fb, root float64, done bool) {
	fa = f(a)
	fb = f(b)
	st.FunctionCalls = 2
	switch {
	case fa == 0:
		st.Converged, st.Flag = true, FlagConverged
		return fa, fb, a, true
	case fb == 0:
		st.Converged, st.Flag = true, FlagConverged
		return fa, fb, b, true
	case sameSign(fa, fb):
		st.Flag = FlagSignErr
		return fa, fb, 0, true
	}
	return fa, fb, 0, false
}

func converged(st *RootResult, x float64) float64 {
	st.Converged, st.Flag = true, FlagConverged
	return x
}

func brentSolver(hyperbolic bool) bracketSolver {
	return func(f func(float64) float64, xa, xb, xtol, rtol float64, iter int, st *RootResult) float64 {
		xpre, xcur := xa, xb
		var xblk, fblk, spre, scur float64
		fpre, fcur, root, done := bracketStart(f, xpre, xcur, st)
		if done {
			return root
		}
		for range iter {
			st.Iterations++
			if fpre != 0 && fcur != 0 && !sameSign(fpre, fcur) {
				xblk, fblk = xpre, fpre
				spre = xcur - xpre
				scur = spre
			}
			if math.Abs(fblk) < math.Abs(fcur) {
				xpre, xcur, xblk = xcur, xblk, xcur
				fpre, fcur, fblk = fcur, fblk, fcur
			}
			delta := (xtol + rtol*math.Abs(xcur)) / 2
			sbis := (xblk - xcur) / 2
			if fcur == 0 || math.Abs(sbis) < delta {
				return converged(st, xcur)
			}
			if math.Abs(spre) > delta && math.Abs(fcur) < math.Abs(fpre) {
				var stry float64
				if xpre == xblk {
					// Interpolate (secant).
					stry = -fcur * (xcur - xpre) / (fcur - fpre)
				} else {
					// Extrapolate.
					dpre := (fpre - fcur) / (xpre - xcur)
					dblk := (fblk - fcur) / (xblk - xcur)
					if hyperbolic {
						stry = -fcur * (fblk - fpre) / (fblk*dpre - fpre*dblk)
					} else {
						stry = -fcur * (fblk*dblk - fpre*dpre) / (dblk * dpre * (fblk - fpre))
					}
				}
				if 2*math.Abs(stry) < math.Min(math.Abs(spre), 3*math.Abs(sbis)-delta) {
					spre, scur = scur, stry
				} else {
					spre, scur = sbis, sbis
				}
			} else {
				spre, scur = sbis, sbis
			}
			xpre, fpre = xcur, fcur
			if math.Abs(scur) > delta {
				xcur += scur
			} else if sbis > 0 {
				xcur += delta
			} else {
				xcur -= delta
			}
			fcur = f(xcur)
			st.FunctionCalls++
		}
		st.Flag = FlagConvErr
		return xcur
	}
}

var (
	brentq = brentSolver(false)
	brenth = brentSolver(true)
)

func ridder(f func(float64) float64, xa, xb, xtol, rtol float64, iter int, st *RootResult) float64 {
	tol := xtol + rtol*math.Min(math.Abs(xa), math.Abs(xb))
	fa, fb, root, done := bracketStart(f, xa, xb, st)
	if done {
		return root
	}
	xn := 0.0
	for range iter {
		st.Iterations++
		dm := 0.5 * (xb - xa)
		xm := xa + dm
		fm := f(xm)
		st.FunctionCalls++
		if fm == 0 {
			return converged(st, xm)
		}
		// Ridders' update, normalised by fa to avoid underflow in fm².
		ratio := fm / fa
		dn := dm * ratio / math.Sqrt(ratio*ratio-fb/fa)
		s := -1.0
		if dn > 0 {
			s = 1
		}
		xn = xm + s*math.Min(math.Abs(dn), math.Abs(dm)-0.5*tol)
		fn := f(xn)
		st.FunctionCalls++
		switch {
		case !sameSign(fn, fm):
			xa, fa, xb, fb = xn, fn, xm, fm
		case !sameSign(fn, fa):
			xb, fb = xn, fn
		default:
			xa, fa = xn, fn
		}
		tol = xtol + rtol*xn
		if fn == 0 || math.Abs(xb-xa) < tol {
			return converged(st, xn)
		}
	}
	st.Flag = FlagConvErr
	return xn
}

func bisect(f func(float64) float64, xa, xb, xtol, rtol float64, iter int, st *RootResult) float64 {
	fa, _, root, done := bracketStart(f, xa, xb, st)
	if done {
		return root
	}
	dm := xb - xa
	for range iter {
		st.Iterations++
		dm *= 0.5
		xm := xa + dm
		fm := f(xm)
		st.FunctionCalls++
		if sameSign(fm, fa) {
			xa = xm
		}
		if fm == 0 || math.Abs(dm) < xtol+rtol*math.Abs(xm) {
			return converged(st, xm)
		}
	}
	st.Flag = FlagConvErr
	return xa
}

// NewtonOptions configures Newton. Zero values select SciPy's defaults:
// Tol 1.48e-8, Rtol 0, MaxIter 50.
type NewtonOptions struct {
	// Fprime is the derivative; nil selects the secant method.
	Fprime func(float64) float64
	// Fprime2 is the second derivative; with Fprime it selects Halley's
	// method.
	Fprime2 func(float64) float64
	// X1 is the second starting point of the secant method; 0 means
	// x0·(1+1e-4) ± 1e-4, as in SciPy. Set X1Set to use an explicit 0.
	X1    float64
	X1Set bool
	// Tol and Rtol are the absolute and relative tolerances on successive
	// iterates (numpy.isclose semantics).
	Tol, Rtol float64
	MaxIter   int
}

// Newton finds a root of f near x0 with the Newton–Raphson method (given
// Fprime), Halley's method (given Fprime and Fprime2) or the secant method
// (neither); it is scipy.optimize.newton(func, x0, fprime, tol=1.48e-8,
// maxiter=50, fprime2, x1, rtol=0, full_output=True, disp=False).
//
// Failure to converge (including a zero derivative) is reported by
// Converged false and Flag "convergence error", not as an error.
func Newton(f func(float64) float64, x0 float64, o *NewtonOptions) (*RootResult, error) {
	var opt NewtonOptions
	if o != nil {
		opt = *o
	}
	tol, maxiter := opt.Tol, opt.MaxIter
	if tol == 0 {
		tol = 1.48e-8
	}
	if maxiter == 0 {
		maxiter = 50
	}
	if tol <= 0 {
		return nil, fmt.Errorf("%w: tol too small (%g <= 0)", ErrInput, tol)
	}
	if maxiter < 1 {
		return nil, fmt.Errorf("%w: maxiter must be greater than 0", ErrInput)
	}
	isclose := func(a, b float64) bool { return math.Abs(a-b) <= tol+opt.Rtol*math.Abs(b) }
	res := func(x float64, calls, it int, ok bool, method string) *RootResult {
		r := &RootResult{Root: x, FunctionCalls: calls, Iterations: it, Converged: ok, Method: method, Flag: FlagConvErr}
		if ok {
			r.Flag = FlagConverged
		}
		return r
	}
	p0 := x0
	calls := 0
	if opt.Fprime != nil {
		method := "newton"
		var p float64
		for itr := range maxiter {
			fval := f(p0)
			calls++
			if fval == 0 {
				return res(p0, calls, itr, true, method), nil
			}
			fder := opt.Fprime(p0)
			calls++
			if fder == 0 {
				return res(p0, calls, itr+1, false, method), nil
			}
			step := fval / fder
			if opt.Fprime2 != nil {
				fder2 := opt.Fprime2(p0)
				calls++
				method = "halley"
				adj := step * fder2 / fder / 2
				if math.Abs(adj) < 1 {
					step /= 1.0 - adj
				}
			}
			p = p0 - step
			if isclose(p, p0) {
				return res(p, calls, itr+1, true, method), nil
			}
			p0 = p
		}
		return res(p, calls, maxiter, false, method), nil
	}
	var p1 float64
	if opt.X1Set || opt.X1 != 0 {
		if opt.X1 == x0 {
			return nil, fmt.Errorf("%w: x1 and x0 must be different", ErrInput)
		}
		p1 = opt.X1
	} else {
		const eps = 1e-4
		p1 = x0 * (1 + eps)
		if p1 >= 0 {
			p1 += eps
		} else {
			p1 -= eps
		}
	}
	q0 := f(p0)
	q1 := f(p1)
	calls = 2
	if math.Abs(q1) < math.Abs(q0) {
		p0, p1, q0, q1 = p1, p0, q1, q0
	}
	var p float64
	for itr := range maxiter {
		if q1 == q0 {
			return res((p1+p0)/2, calls, itr+1, false, "secant"), nil
		}
		if math.Abs(q1) > math.Abs(q0) {
			p = (-q0/q1*p1 + p0) / (1 - q0/q1)
		} else {
			p = (-q1/q0*p0 + p1) / (1 - q1/q0)
		}
		if isclose(p, p1) {
			return res(p, calls, itr+1, true, "secant"), nil
		}
		p0, q0 = p1, q1
		p1 = p
		q1 = f(p1)
		calls++
	}
	return res(p, calls, maxiter, false, "secant"), nil
}

// RootScalarOptions configures RootScalar.
type RootScalarOptions struct {
	// Bracket [a, b] for the bracketing methods.
	Bracket []float64
	// Fprime and Fprime2 are the first and second derivatives.
	Fprime, Fprime2 func(float64) float64
	// X0 and X1 are starting points for Newton, Halley and the secant
	// method; X0Set/X1Set mark them as given (0 is a valid start).
	X0, X1       float64
	X0Set, X1Set bool
	// Xtol, Rtol and MaxIter override the method's defaults.
	Xtol, Rtol float64
	MaxIter    int
}

// RootScalar finds a root of a scalar function; it is
// scipy.optimize.root_scalar(f, method, bracket, fprime, fprime2, x0, x1,
// xtol, rtol, maxiter). method is one of "brentq", "brenth", "ridder",
// "bisect", "newton", "halley", "secant", or "" to choose as SciPy does: a
// bracket selects brentq; otherwise x0 with Fprime selects newton (halley
// with Fprime2), x0 with x1 the secant method, and x0 alone Newton with a
// finite-difference derivative.
func RootScalar(f func(float64) float64, method string, o *RootScalarOptions) (*RootResult, error) {
	var opt RootScalarOptions
	if o != nil {
		opt = *o
	}
	hasX0 := opt.X0Set || opt.X0 != 0
	hasX1 := opt.X1Set || opt.X1 != 0
	if method == "" {
		switch {
		case opt.Bracket != nil:
			method = "brentq"
		case hasX0 && opt.Fprime != nil && opt.Fprime2 != nil:
			method = "halley"
		case hasX0 && opt.Fprime != nil:
			method = "newton"
		case hasX0 && hasX1:
			method = "secant"
		case hasX0:
			method = "newton"
		default:
			return nil, fmt.Errorf("%w: unable to select a solver as neither bracket nor starting point provided", ErrInput)
		}
	}
	bo := &BracketOptions{Xtol: opt.Xtol, Rtol: opt.Rtol, MaxIter: opt.MaxIter}
	no := &NewtonOptions{Tol: opt.Xtol, Rtol: opt.Rtol, MaxIter: opt.MaxIter}
	var solver func(func(float64) float64, float64, float64, *BracketOptions) (*RootResult, error)
	switch method {
	case "brentq":
		solver = Brentq
	case "brenth":
		solver = Brenth
	case "ridder":
		solver = Ridder
	case "bisect":
		solver = Bisect
	case "secant", "newton", "halley":
		if !hasX0 {
			return nil, fmt.Errorf("%w: x0 must be given for %s", ErrInput, method)
		}
	default:
		return nil, fmt.Errorf("%w: %q", ErrMethod, method)
	}
	if solver != nil {
		if len(opt.Bracket) < 2 {
			return nil, fmt.Errorf("%w: bracket needed for %s", ErrInput, method)
		}
		r, err := solver(f, opt.Bracket[0], opt.Bracket[1], bo)
		if _, isNaN := err.(nanError); isNaN {
			r.Iterations = -1 // SciPy reports NaN iterations here
			return r, nil
		}
		return r, err
	}
	switch method {
	case "secant":
		no.X1, no.X1Set = opt.X1, hasX1
	case "newton":
		no.Fprime = opt.Fprime
		if no.Fprime == nil {
			// SciPy's root_scalar differentiates numerically with a
			// forward difference and its default relative step.
			fd := fdStep{method: FD2Point}
			g := make([]float64, 1)
			no.Fprime = func(x float64) float64 {
				xs := []float64{x}
				fd.gradFD(func(v []float64) float64 { return f(v[0]) }, xs, f(x), g)
				return g[0]
			}
		}
	case "halley":
		if opt.Fprime == nil || opt.Fprime2 == nil {
			return nil, fmt.Errorf("%w: halley needs Fprime and Fprime2", ErrInput)
		}
		no.Fprime, no.Fprime2 = opt.Fprime, opt.Fprime2
	}
	return Newton(f, opt.X0, no)
}
