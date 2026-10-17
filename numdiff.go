package optimize

import "math"

// FDMethod selects a finite-difference scheme.
type FDMethod int

const (
	// FD2Point is the forward difference (SciPy '2-point'), one evaluation
	// per variable, error O(h).
	FD2Point FDMethod = iota
	// FD3Point is the central difference (SciPy '3-point'), two evaluations
	// per variable, error O(h²); it falls back to a one-sided second-order
	// formula next to a bound.
	FD3Point
)

// fdStep holds the step rules of SciPy's approx_derivative: a relative step
// (rel, 0 for the default sqrt(eps) or eps^(1/3)) or an absolute step (abs,
// used when non-zero), and the bounds the steps must respect.
type fdStep struct {
	method FDMethod
	rel    float64
	abs    float64
	lb, ub []float64 // nil means unbounded
	// buf, when set, keeps jacFD's residual buffers from one call to the
	// next instead of allocating them every time.
	buf *fdBuf
}

// fdBuf holds jacFD's m-sized work vectors.
type fdBuf struct{ f1, f2 []float64 }

// vectors returns jacFD's work vectors of length m (f2 only for 3-point).
func (s fdStep) vectors(m int) (f1, f2 []float64) {
	b := s.buf
	if b == nil {
		b = &fdBuf{}
	}
	if len(b.f1) != m {
		b.f1 = make([]float64, m)
	}
	if s.method != FD2Point && len(b.f2) != m {
		b.f2 = make([]float64, m)
	}
	return b.f1, b.f2
}

// steps computes the signed step for each variable at x0 and whether the
// 3-point scheme must go one-sided there. It follows
// scipy.optimize._numdiff: h = rel*sign(x)*max(1,|x|) by default,
// h = rel*sign(x)*|x| for a user relative step, h = abs for an absolute step,
// and a zero effective step falls back to the default; then the scheme is
// adjusted so that every evaluation point stays inside the bounds.
func (s fdStep) steps(x0 []float64) (h []float64, oneSided []bool) {
	n := len(x0)
	h = make([]float64, n)
	oneSided = make([]bool, n)
	def := math.Sqrt(epsmch)
	if s.method == FD3Point {
		def = math.Cbrt(epsmch)
	}
	for i, x := range x0 {
		sign := 1.0
		if x < 0 {
			sign = -1
		}
		dflt := float64(def * sign * math.Max(1, math.Abs(x)))
		switch {
		case s.abs != 0:
			h[i] = s.abs
		case s.rel != 0:
			h[i] = float64(s.rel * sign * math.Abs(x))
		default:
			h[i] = dflt
		}
		if (x+h[i])-x == 0 {
			h[i] = dflt
		}
	}
	if s.lb == nil {
		return h, oneSided
	}
	free := true
	for i := range x0 {
		if !math.IsInf(s.lb[i], -1) || !math.IsInf(s.ub[i], 1) {
			free = false
			break
		}
	}
	if free {
		return h, oneSided
	}
	for i, x := range x0 {
		lo, up := x-s.lb[i], s.ub[i]-x
		if s.method == FD2Point {
			xh := x + h[i]
			violated := xh < s.lb[i] || xh > s.ub[i]
			fitting := math.Abs(h[i]) <= math.Max(lo, up)
			switch {
			case violated && fitting:
				h[i] = -h[i]
			case !fitting && up >= lo:
				h[i] = up
			case !fitting:
				h[i] = -lo
			}
			continue
		}
		hi := math.Abs(h[i])
		h[i] = hi
		central := lo >= hi && up >= hi
		if !central {
			if up >= lo {
				h[i] = math.Min(hi, 0.5*up)
			} else {
				h[i] = -math.Min(hi, 0.5*lo)
			}
			oneSided[i] = true
			if md := math.Min(up, lo); math.Abs(h[i]) <= md {
				h[i] = md
				oneSided[i] = false
			}
		}
	}
	return h, oneSided
}

// gradFD estimates the gradient of f at x0 into g, given f0 = f(x0). It
// returns the number of evaluations of f.
func (s fdStep) gradFD(f func([]float64) float64, x0 []float64, f0 float64, g []float64) int {
	h, one := s.steps(x0)
	x := clone(x0)
	nfev := 0
	for i := range x0 {
		xi := x0[i]
		if s.method == FD2Point {
			x[i] = xi + h[i]
			dx := x[i] - xi
			g[i] = (f(x) - f0) / dx
			nfev++
		} else if one[i] {
			x[i] = xi + h[i]
			f1 := f(x)
			x[i] = xi + float64(2*h[i])
			f2 := f(x)
			dx := x[i] - xi
			g[i] = (float64(-3*f0) + float64(4*f1) - f2) / dx
			nfev += 2
		} else {
			x[i] = xi - h[i]
			f1 := f(x)
			x[i] = xi + h[i]
			f2 := f(x)
			dx := x[i] - (xi - h[i])
			g[i] = (f2 - f1) / dx
			nfev += 2
		}
		x[i] = xi
	}
	return nfev
}

// jacFD estimates the m×n Jacobian of f at x0 into jac (row-major), given
// f0 = f(x0) of length m. f writes its m outputs into its second argument.
// It returns the number of evaluations of f.
func (s fdStep) jacFD(f func(x, out []float64), x0, f0, jac []float64) int {
	n, m := len(x0), len(f0)
	h, one := s.steps(x0)
	x := clone(x0)
	f1, f2 := s.vectors(m)
	nfev := 0
	for j := range x0 {
		xj := x0[j]
		if s.method == FD2Point {
			x[j] = xj + h[j]
			dx := x[j] - xj
			f(x, f1)
			for i := 0; i < m; i++ {
				jac[i*n+j] = (f1[i] - f0[i]) / dx
			}
			nfev++
		} else if one[j] {
			x[j] = xj + h[j]
			f(x, f1)
			x[j] = xj + float64(2*h[j])
			f(x, f2)
			dx := x[j] - xj
			for i := 0; i < m; i++ {
				jac[i*n+j] = (float64(-3*f0[i]) + float64(4*f1[i]) - f2[i]) / dx
			}
			nfev += 2
		} else {
			x[j] = xj - h[j]
			f(x, f1)
			x[j] = xj + h[j]
			f(x, f2)
			dx := x[j] - (xj - h[j])
			for i := 0; i < m; i++ {
				jac[i*n+j] = (f2[i] - f1[i]) / dx
			}
			nfev += 2
		}
		x[j] = xj
	}
	return nfev
}

// ApproxFprime returns the forward-difference gradient of f at x with the
// absolute step epsilon (0 selects SciPy's default sqrt(eps) ≈ 1.49e-8), as
// scipy.optimize.approx_fprime(xk, f, epsilon) does.
func ApproxFprime(f func([]float64) float64, x []float64, epsilon float64) []float64 {
	if epsilon == 0 {
		epsilon = math.Sqrt(epsmch)
	}
	g := make([]float64, len(x))
	fdStep{method: FD2Point, abs: epsilon}.gradFD(f, x, f(x), g)
	return g
}

// CheckGrad returns the Euclidean norm of the difference between grad(x) and
// a forward-difference gradient of f with absolute step epsilon (0 for the
// default), like scipy.optimize.check_grad(func, grad, x0, epsilon=...).
// grad writes the gradient at x into its second argument.
func CheckGrad(f func([]float64) float64, grad func(x, g []float64), x []float64, epsilon float64) float64 {
	g := make([]float64, len(x))
	grad(x, g)
	a := ApproxFprime(f, x, epsilon)
	for i := range g {
		g[i] -= a[i]
	}
	return enorm(g)
}
