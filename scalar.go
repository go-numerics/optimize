package optimize

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// ScalarMethod names a scalar minimisation algorithm; the values are the
// method strings of scipy.optimize.minimize_scalar.
type ScalarMethod string

// Scalar minimisation methods.
const (
	// Brent is Brent's method: parabolic interpolation safeguarded by
	// golden-section steps, from a bracket.
	Brent ScalarMethod = "brent"
	// Bounded is Brent's method restricted to a finite interval (the
	// algorithm of fminbound).
	Bounded ScalarMethod = "bounded"
	// Golden is the golden-section search, from a bracket.
	Golden ScalarMethod = "golden"
)

// ScalarOptions configures MinimizeScalar. Zero values select SciPy's
// defaults.
type ScalarOptions struct {
	// Bracket is a bracketing interval for Brent and Golden: two points
	// (xa, xb) from which Bracket searches downhill, or three points
	// (xa, xb, xc) with xa < xb < xc and f(xb) < f(xa), f(xc). nil starts
	// the downhill search from (0, 1).
	Bracket []float64
	// Bounds is the interval for the Bounded method; setting it selects
	// Bounded when the method is "".
	Bounds *Bound
	// Tol is minimize_scalar's tol: it sets Xtol (Brent, Golden) or Xatol
	// (Bounded) when those are zero.
	Tol float64
	// Xtol is the relative tolerance of Brent (1.48e-8) and Golden
	// (√eps ≈ 1.49e-8).
	Xtol float64
	// Xatol is the absolute tolerance of Bounded (1e-5).
	Xatol float64
	// MaxIter bounds iterations (Brent 500, Golden 5000) or function
	// evaluations (Bounded 500).
	MaxIter int
}

// ScalarResult is the outcome of MinimizeScalar, the OptimizeResult of
// scipy.optimize.minimize_scalar.
type ScalarResult struct {
	X, Fun    float64
	Nit, Nfev int
	Status    int // Bounded only: 0 solution found, 1 maxfun, 2 NaN
	Success   bool
	Message   string
}

// MinimizeScalar minimises a function of one variable; it is
// scipy.optimize.minimize_scalar(fun, bracket=..., bounds=..., method=...,
// tol=..., options={'xtol'/'xatol', 'maxiter'}). method "" selects Brent, or
// Bounded when o.Bounds is set.
//
// A failed bracket search is not an error: as in SciPy, the result then
// holds the best of the three points and Success is false.
func MinimizeScalar(f func(float64) float64, method ScalarMethod, o *ScalarOptions) (*ScalarResult, error) {
	var opt ScalarOptions
	if o != nil {
		opt = *o
	}
	if method == "" {
		method = Brent
		if opt.Bounds != nil {
			method = Bounded
		}
	}
	switch method {
	case Brent, Golden:
		if opt.Bounds != nil {
			return nil, fmt.Errorf("%w: use of Bounds is incompatible with method %s", ErrMethod, method)
		}
		xtol := opt.Xtol
		if xtol == 0 {
			xtol = opt.Tol
		}
		if xtol < 0 {
			return nil, fmt.Errorf("%w: tolerance should be >= 0, got %g", ErrInput, xtol)
		}
		var r *ScalarResult
		var err error
		if method == Brent {
			if xtol == 0 {
				xtol = 1.48e-8
			}
			r, err = minimizeScalarBrent(f, opt.Bracket, xtol, opt.MaxIter)
		} else {
			if xtol == 0 {
				xtol = math.Sqrt(epsmch)
			}
			r, err = minimizeScalarGolden(f, opt.Bracket, xtol, opt.MaxIter)
		}
		var be *BracketError
		if errors.As(err, &be) {
			return be.result(), nil
		}
		return r, err
	case Bounded:
		if opt.Bounds == nil {
			return nil, fmt.Errorf("%w: the bounded method requires Bounds", ErrInput)
		}
		xatol := opt.Xatol
		if xatol == 0 {
			xatol = opt.Tol
		}
		if xatol == 0 {
			xatol = 1e-5
		}
		maxfun := opt.MaxIter
		if maxfun == 0 {
			maxfun = 500
		}
		return minimizeScalarBounded(f, opt.Bounds.Lo, opt.Bounds.Hi, xatol, maxfun)
	}
	return nil, fmt.Errorf("%w: %q", ErrMethod, method)
}

// Fminbound minimises f on [x1, x2] with absolute tolerance xtol (1e-5 when
// 0) and at most maxfun evaluations (500 when 0); it is
// scipy.optimize.fminbound(func, x1, x2, xtol, maxfun, full_output=True).
func Fminbound(f func(float64) float64, x1, x2, xtol float64, maxfun int) (*ScalarResult, error) {
	return MinimizeScalar(f, Bounded, &ScalarOptions{Bounds: &Bound{x1, x2}, Xatol: xtol, MaxIter: maxfun})
}

// BracketError reports that Bracket found no valid bracket. It carries the
// last three points, as SciPy's BracketError does.
type BracketError struct {
	Xa, Xb, Xc, Fa, Fb, Fc float64
	Nfev                   int
	Msg                    string
}

func (e *BracketError) Error() string { return "optimize: " + e.Msg }

func (e *BracketError) result() *ScalarResult {
	xs := [3]float64{e.Xa, e.Xb, e.Xc}
	fs := [3]float64{e.Fa, e.Fb, e.Fc}
	r := &ScalarResult{Nfev: e.Nfev, Message: e.Msg, X: math.NaN(), Fun: math.NaN()}
	for i := range 3 {
		if math.IsNaN(xs[i]) || math.IsNaN(fs[i]) {
			return r
		}
	}
	imin := 0
	for i := 1; i < 3; i++ {
		if fs[i] < fs[imin] {
			imin = i
		}
	}
	r.X, r.Fun = xs[imin], fs[imin]
	return r
}

// Bracket searches downhill from xa, xb (0 and 1 when both are zero) for
// three points xa, xb, xc with f(xb) below f(xa) and f(xc); it is
// scipy.optimize.bracket(func, xa, xb, grow_limit=110, maxiter=1000). It
// returns the points, their values and the number of evaluations, or a
// *BracketError.
func Bracket(f func(float64) float64, xa, xb float64) (x [3]float64, fx [3]float64, nfev int, err error) {
	if xa == 0 && xb == 0 {
		xb = 1
	}
	return bracket(f, xa, xb, 110, 1000)
}

func bracket(f func(float64) float64, xa, xb, growLimit float64, maxiter int) (x [3]float64, fx [3]float64, nfev int, err error) {
	const gold = 1.618034
	const verySmall = 1e-21
	fa := f(xa)
	fb := f(xb)
	if fa < fb {
		xa, xb = xb, xa
		fa, fb = fb, fa
	}
	xc := xb + float64(gold*(xb-xa))
	fc := f(xc)
	nfev = 3
	iter := 0
	for fc < fb {
		tmp1 := float64((xb - xa) * (fb - fc))
		tmp2 := float64((xb - xc) * (fb - fa))
		val := tmp2 - tmp1
		denom := float64(2 * val)
		if math.Abs(val) < verySmall {
			denom = float64(2 * verySmall)
		}
		w := xb - (float64((xb-xc)*tmp2)-float64((xb-xa)*tmp1))/denom
		wlim := xb + float64(growLimit*(xc-xb))
		if iter > maxiter {
			return x, fx, nfev, &BracketError{xa, xb, xc, fa, fb, fc, nfev,
				"No valid bracket was found before the iteration limit was reached. " +
					"Consider trying different initial points or increasing `maxiter`."}
		}
		iter++
		var fw float64
		switch {
		case (w-xc)*(xb-w) > 0:
			fw = f(w)
			nfev++
			if fw < fc {
				xa, xb = xb, w
				fa, fb = fb, fw
				goto done
			} else if fw > fb {
				xc, fc = w, fw
				goto done
			}
			w = xc + float64(gold*(xc-xb))
			fw = f(w)
			nfev++
		case (w-wlim)*(wlim-xc) >= 0:
			w = wlim
			fw = f(w)
			nfev++
		case (w-wlim)*(xc-w) > 0:
			fw = f(w)
			nfev++
			if fw < fc {
				xb, xc = xc, w
				w = xc + float64(gold*(xc-xb))
				fb, fc = fc, fw
				fw = f(w)
				nfev++
			}
		default:
			w = xc + float64(gold*(xc-xb))
			fw = f(w)
			nfev++
		}
		xa, xb, xc = xb, xc, w
		fa, fb, fc = fb, fc, fw
	}
done:
	cond1 := (fb < fc && fb <= fa) || (fb < fa && fb <= fc)
	cond2 := (xa < xb && xb < xc) || (xc < xb && xb < xa)
	cond3 := !math.IsInf(xa, 0) && !math.IsNaN(xa) && !math.IsInf(xb, 0) && !math.IsNaN(xb) &&
		!math.IsInf(xc, 0) && !math.IsNaN(xc)
	if !(cond1 && cond2 && cond3) {
		return x, fx, nfev, &BracketError{xa, xb, xc, fa, fb, fc, nfev,
			"The algorithm terminated without finding a valid bracket. Consider trying different initial points."}
	}
	return [3]float64{xa, xb, xc}, [3]float64{fa, fb, fc}, nfev, nil
}

// bracketInfo resolves a user bracket the way SciPy's Brent and golden do.
func bracketInfo(f func(float64) float64, brack []float64) (x [3]float64, fx [3]float64, nfev int, err error) {
	switch len(brack) {
	case 0:
		return bracket(f, 0, 1, 110, 1000)
	case 2:
		return bracket(f, brack[0], brack[1], 110, 1000)
	case 3:
		xa, xb, xc := brack[0], brack[1], brack[2]
		if xa > xc {
			xa, xc = xc, xa
		}
		if !(xa < xb && xb < xc) {
			return x, fx, 0, fmt.Errorf("%w: bracketing values (xa, xb, xc) do not fulfill this requirement: (xa < xb) and (xb < xc)", ErrInput)
		}
		fa, fb, fc := f(xa), f(xb), f(xc)
		if !(fb < fa && fb < fc) {
			return x, fx, 3, fmt.Errorf("%w: bracketing values (xa, xb, xc) do not fulfill this requirement: (f(xb) < f(xa)) and (f(xb) < f(xc))", ErrInput)
		}
		return [3]float64{xa, xb, xc}, [3]float64{fa, fb, fc}, 3, nil
	}
	return x, fx, 0, fmt.Errorf("%w: bracketing interval must be a length 2 or 3 sequence", ErrInput)
}

func scalarMessage(nit, maxiter int, x, fval, xtol float64) (bool, string) {
	if math.IsNaN(x) || math.IsNaN(fval) {
		return false, msgNaN
	}
	if nit >= maxiter {
		return false, "\nMaximum number of iterations exceeded"
	}
	return true, fmt.Sprintf("\nOptimization terminated successfully;\nThe returned value satisfies the termination criteria\n(using xtol = %v )", pyFloat(xtol))
}

// pyFloat formats like Python's repr of a float, so messages match SciPy's.
type pyFloat float64

func (p pyFloat) String() string {
	return pyRepr(float64(p))
}

// The scalar routines below are statement-for-statement ports of SciPy's
// Python code. Python never fuses a multiply with an add; Go does on arm64,
// ppc64le, s390x, riscv64 and loong64, so the products that feed a sum are
// wrapped in float64() (which forbids fusion) to keep the iterates identical
// on every architecture.

// minimizeScalarBrent is Brent's (1973) method as in SciPy's Brent class.
func minimizeScalarBrent(f func(float64) float64, brack []float64, tol float64, maxiter int) (*ScalarResult, error) {
	if maxiter == 0 {
		maxiter = 500
	}
	const mintol = 1.0e-11
	const cg = 0.3819660
	xs, fs, funcalls, err := bracketInfo(f, brack)
	if err != nil {
		return nil, err
	}
	xa, xb, xc := xs[0], xs[1], xs[2]
	x, w, v := xb, xb, xb
	fx, fw, fv := fs[1], fs[1], fs[1]
	a, b := xc, xa
	if xa < xc {
		a, b = xa, xc
	}
	deltax, rat := 0.0, 0.0
	iter := 0
	for iter < maxiter {
		tol1 := float64(tol*math.Abs(x)) + mintol
		tol2 := float64(2 * tol1)
		xmid := float64(0.5 * (a + b))
		if math.Abs(x-xmid) < tol2-float64(0.5*(b-a)) {
			break
		}
		if math.Abs(deltax) <= tol1 {
			if x >= xmid {
				deltax = a - x
			} else {
				deltax = b - x
			}
			rat = float64(cg * deltax)
		} else {
			tmp1 := float64((x - w) * (fx - fv))
			tmp2 := float64((x - v) * (fx - fw))
			p := float64((x-v)*tmp2) - float64((x-w)*tmp1)
			tmp2 = float64(2 * (tmp2 - tmp1))
			if tmp2 > 0 {
				p = -p
			}
			tmp2 = math.Abs(tmp2)
			dxTemp := deltax
			deltax = rat
			if p > tmp2*(a-x) && p < tmp2*(b-x) && math.Abs(p) < math.Abs(0.5*tmp2*dxTemp) {
				rat = float64(p * 1.0 / tmp2)
				u := x + rat
				if u-a < tol2 || b-u < tol2 {
					if xmid-x >= 0 {
						rat = tol1
					} else {
						rat = -tol1
					}
				}
			} else {
				if x >= xmid {
					deltax = a - x
				} else {
					deltax = b - x
				}
				rat = float64(cg * deltax)
			}
		}
		var u float64
		if math.Abs(rat) < tol1 {
			if rat >= 0 {
				u = x + tol1
			} else {
				u = x - tol1
			}
		} else {
			u = x + rat
		}
		fu := f(u)
		funcalls++
		if fu > fx {
			if u < x {
				a = u
			} else {
				b = u
			}
			if fu <= fw || w == x {
				v, w = w, u
				fv, fw = fw, fu
			} else if fu <= fv || v == x || v == w {
				v, fv = u, fu
			}
		} else {
			if u >= x {
				a = x
			} else {
				b = x
			}
			v, w, x = w, x, u
			fv, fw, fx = fw, fx, fu
		}
		iter++
	}
	ok, msg := scalarMessage(iter, maxiter, x, fx, tol)
	return &ScalarResult{X: x, Fun: fx, Nit: iter, Nfev: funcalls, Success: ok, Message: msg}, nil
}

// minimizeScalarGolden is the golden-section search as in SciPy's golden.
func minimizeScalarGolden(f func(float64) float64, brack []float64, tol float64, maxiter int) (*ScalarResult, error) {
	if maxiter == 0 {
		maxiter = 5000
	}
	xs, _, funcalls, err := bracketInfo(f, brack)
	if err != nil {
		return nil, err
	}
	xa, xb, xc := xs[0], xs[1], xs[2]
	// SciPy computes 1 − gR in float64 at run time; a Go constant
	// expression would be exact and round differently.
	gR := 0.61803399
	gC := 1.0 - gR
	x3, x0 := xc, xa
	var x1, x2 float64
	if math.Abs(xc-xb) > math.Abs(xb-xa) {
		x1 = xb
		x2 = xb + float64(gC*(xc-xb))
	} else {
		x2 = xb
		x1 = xb - float64(gC*(xb-xa))
	}
	f1, f2 := f(x1), f(x2)
	funcalls += 2
	nit := 0
	for range maxiter {
		if math.Abs(x3-x0) <= tol*(math.Abs(x1)+math.Abs(x2)) {
			break
		}
		if f2 < f1 {
			x0, x1 = x1, x2
			x2 = float64(gR*x1) + float64(gC*x3)
			f1 = f2
			f2 = f(x2)
		} else {
			x3, x2 = x2, x1
			x1 = float64(gR*x2) + float64(gC*x0)
			f2 = f1
			f1 = f(x1)
		}
		funcalls++
		nit++
	}
	xmin, fval := x2, f2
	if f1 < f2 {
		xmin, fval = x1, f1
	}
	ok, msg := scalarMessage(nit, maxiter, xmin, fval, tol)
	return &ScalarResult{X: xmin, Fun: fval, Nit: nit, Nfev: funcalls, Success: ok, Message: msg}, nil
}

// sign1 is SciPy's np.sign(v) + (v == 0): the sign of v, with +1 for zero.
// A difference or quotient here is never −0 (x − x is +0, and (p+0.0)/q
// clears a negative zero), so Copysign is equivalent.
func sign1(v float64) float64 { return math.Copysign(1, v) }

// minimizeScalarBounded is the bounded Brent method of fminbound (Forsythe,
// Malcolm & Moler's FMIN), as in SciPy's _minimize_scalar_bounded.
func minimizeScalarBounded(f func(float64) float64, x1, x2, xatol float64, maxfun int) (*ScalarResult, error) {
	if math.IsNaN(x1) || math.IsNaN(x2) || math.IsInf(x1, 0) || math.IsInf(x2, 0) {
		return nil, fmt.Errorf("%w: optimization bounds must be finite scalars", ErrBounds)
	}
	if x1 > x2 {
		return nil, fmt.Errorf("%w: the lower bound exceeds the upper bound", ErrBounds)
	}
	flag := 0
	sqrtEps := math.Sqrt(2.2e-16)
	goldenMean := float64(0.5 * (3.0 - math.Sqrt(5.0)))
	a, b := x1, x2
	fulc := a + float64(goldenMean*(b-a))
	nfc, xf := fulc, fulc
	rat, e := 0.0, 0.0
	x := xf
	fx := f(x)
	num := 1
	fu := math.Inf(1)
	ffulc, fnfc := fx, fx
	xm := float64(0.5 * (a + b))
	tol1 := float64(sqrtEps*math.Abs(xf)) + xatol/3.0
	tol2 := float64(2.0 * tol1)
	for math.Abs(xf-xm) > tol2-float64(0.5*(b-a)) {
		golden := true
		if math.Abs(e) > tol1 {
			golden = false
			r := float64((xf - nfc) * (fx - ffulc))
			q := float64((xf - fulc) * (fx - fnfc))
			p := float64((xf-fulc)*q) - float64((xf-nfc)*r)
			q = float64(2.0 * (q - r))
			if q > 0 {
				p = -p
			}
			q = math.Abs(q)
			r = e
			e = rat
			if math.Abs(p) < math.Abs(0.5*q*r) && p > q*(a-xf) && p < q*(b-xf) {
				rat = (p + 0.0) / q
				x = xf + rat
				if x-a < tol2 || b-x < tol2 {
					rat = float64(tol1 * sign1(xm-xf))
				}
			} else {
				golden = true
			}
		}
		if golden {
			if xf >= xm {
				e = a - xf
			} else {
				e = b - xf
			}
			rat = float64(goldenMean * e)
		}
		x = xf + float64(sign1(rat)*math.Max(math.Abs(rat), tol1))
		fu = f(x)
		num++
		if fu <= fx {
			if x >= xf {
				a = xf
			} else {
				b = xf
			}
			fulc, ffulc = nfc, fnfc
			nfc, fnfc = xf, fx
			xf, fx = x, fu
		} else {
			if x < xf {
				a = x
			} else {
				b = x
			}
			if fu <= fnfc || nfc == xf {
				fulc, ffulc = nfc, fnfc
				nfc, fnfc = x, fu
			} else if fu <= ffulc || fulc == xf || fulc == nfc {
				fulc, ffulc = x, fu
			}
		}
		xm = float64(0.5 * (a + b))
		tol1 = float64(sqrtEps*math.Abs(xf)) + xatol/3.0
		tol2 = float64(2.0 * tol1)
		if num >= maxfun {
			flag = 1
			break
		}
	}
	if math.IsNaN(xf) || math.IsNaN(fx) || math.IsNaN(fu) {
		flag = 2
	}
	msg := [...]string{"Solution found.", "Maximum number of function calls reached.", msgNaN}[flag]
	return &ScalarResult{X: xf, Fun: fx, Nit: num, Nfev: num, Status: flag, Success: flag == 0, Message: msg}, nil
}

// pyRepr formats v as Python's repr(float) does: the shortest round-trip
// digits, positional for decimal exponents in [-4, 16), scientific otherwise.
func pyRepr(v float64) string {
	switch {
	case math.IsNaN(v):
		return "nan"
	case math.IsInf(v, 1):
		return "inf"
	case math.IsInf(v, -1):
		return "-inf"
	case v == 0:
		if math.Signbit(v) {
			return "-0.0"
		}
		return "0.0"
	}
	exp := int(math.Floor(math.Log10(math.Abs(v))))
	// Log10 can be off by one near powers of ten; the scientific form tells.
	s := strconv.FormatFloat(v, 'e', -1, 64)
	if i := strings.IndexByte(s, 'e'); i >= 0 {
		exp, _ = strconv.Atoi(s[i+1:])
	}
	if exp < -4 || exp >= 16 {
		return s
	}
	s = strconv.FormatFloat(v, 'f', -1, 64)
	if !strings.Contains(s, ".") {
		s += ".0"
	}
	return s
}
