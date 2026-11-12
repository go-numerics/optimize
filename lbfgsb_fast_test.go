package optimize

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestGatherKernels: gdot, gdotD, gaxpy against plain loops on permuted
// indices of every length mod 4.
func TestGatherKernels(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 7))
	for n := 0; n < 12; n++ {
		a, b, d := make([]float64, 12), make([]float64, 12), make([]float64, n)
		for i := range a {
			a[i], b[i] = r.NormFloat64(), r.NormFloat64()
		}
		for i := range d {
			d[i] = r.NormFloat64()
		}
		idx := r.Perm(12)[:n]
		var w1, w2 float64
		for k, i := range idx {
			w1 += a[i] * b[i]
			w2 += a[i] * d[k]
		}
		if g := gdot(a, b, idx); math.Abs(g-w1) > 1e-13 {
			t.Fatalf("gdot %g want %g", g, w1)
		}
		if g := gdotD(a, idx, d); math.Abs(g-w2) > 1e-13 {
			t.Fatalf("gdotD %g want %g", g, w2)
		}
		e := append([]float64(nil), d...)
		gaxpy(2, a, idx, e)
		for k, i := range idx {
			if math.Abs(e[k]-(d[k]+2*a[i])) > 1e-14 {
				t.Fatal("gaxpy")
			}
		}
	}
}

// lbQuadProblem is f(x) = ½ dᵀA d, d = x − xstar, A tridiagonal (4 on the
// diagonal, −1 beside it): symmetric positive definite, minimiser xstar
// whatever the box, as long as xstar lies inside it.
func lbQuadProblem(xstar []float64) Problem {
	n := len(xstar)
	ad := func(x, out []float64) {
		for i := range x {
			v := 4 * (x[i] - xstar[i])
			if i > 0 {
				v -= x[i-1] - xstar[i-1]
			}
			if i < n-1 {
				v -= x[i+1] - xstar[i+1]
			}
			out[i] = v
		}
	}
	return Problem{
		Func: func(x []float64) float64 {
			g := make([]float64, n)
			ad(x, g)
			s := 0.0
			for i := range x {
				s += (x[i] - xstar[i]) * g[i]
			}
			return s / 2
		},
		Grad: func(x, g []float64) { ad(x, g) },
	}
}

// TestAccuracyBoundLBFGSB: the accuracy contract on the default (FMA, SIMD)
// kernels, against optima that are known exactly, never against another
// float64 run. Unbounded Rosenbrock (x* = 1, f* = 0) for n = 2, 10, 40 and
// 100; and the bounded problems, the only paths through the gather kernels
// (gdot, gdotD, gaxpy) and the free-set bookkeeping: a coupled quadratic whose
// exact minimiser lies inside a box that the first Cauchy steps hit (start at
// the corner), a separable ill-scaled quadratic whose box clamps half of the
// variables (x*_i = clamp(c_i, lo, hi) exactly), and Rosenbrock inside a box
// that contains x* = 1 but is active along the way. The bounds are those of
// TestAccuracyBoundMinimize (1e-3 / 1e-6).
func TestAccuracyBoundLBFGSB(t *testing.T) {
	check := func(name string, p Problem, x0, want []float64, o *Options, fstar float64) {
		t.Helper()
		r, err := Minimize(p, x0, LBFGSB, o)
		if err != nil {
			t.Fatal(name, err)
		}
		dx, df := maxDist(r.X, want), math.Abs(r.Fun-fstar)
		t.Logf("%-28s nit %4d |x-x*|=%.2e |f-f*|=%.2e", name, r.Nit, dx, df)
		// The f bound is relative to max(1, f*): L-BFGS-B stops on a relative
		// reduction (ftol 2.2e-9), and the separable problem has f* ~ 2e3.
		if fb := 1e-6 * math.Max(1, fstar); dx > 1e-3 || df > fb {
			t.Errorf("%s: |x-x*| = %g (bound 1e-3), |f-f*| = %g (bound %g)", name, dx, df, fb)
		}
	}
	for _, n := range []int{2, 10, 40, 100} {
		ones := make([]float64, n)
		for i := range ones {
			ones[i] = 1
		}
		check("rosenbrock", Problem{Func: rosen, Grad: rosenDer}, altStart(n), ones, nil, 0)
		o := &Options{Bounds: rep(Bound{Lo: -2, Hi: 1.5}, n)}
		check("rosenbrock boxed", Problem{Func: rosen, Grad: rosenDer}, altStart(n), ones, o, 0)
	}
	for _, n := range []int{8, 40} {
		xstar, x0 := make([]float64, n), make([]float64, n)
		for i := range xstar {
			xstar[i] = float64((i*7)%9-4) / 5
			x0[i] = 1
		}
		o := &Options{Bounds: rep(Bound{Lo: -1, Hi: 1}, n)}
		check("coupled quadratic, box", lbQuadProblem(xstar), x0, xstar, o, 0)
	}
	// Separable, ill scaled; c_i outside [-1, 1] for odd i, so the optimum is
	// on the bound there: x*_i = clamp(c_i).
	n := 40
	c, want, w := make([]float64, n), make([]float64, n), make([]float64, n)
	for i := range c {
		w[i] = math.Pow(10, float64(i%4))
		c[i] = float64(i%5-2) / 3
		if i%2 == 1 {
			c[i] = 3 * float64(1-2*(i%4/2)) // ±3
		}
		want[i] = math.Max(-1, math.Min(1, c[i]))
	}
	p := Problem{
		Func: func(x []float64) float64 {
			s := 0.0
			for i := range x {
				s += w[i] * (x[i] - c[i]) * (x[i] - c[i])
			}
			return s
		},
		Grad: func(x, g []float64) {
			for i := range x {
				g[i] = 2 * w[i] * (x[i] - c[i])
			}
		},
	}
	fstar := 0.0
	for i := range want {
		fstar += w[i] * (want[i] - c[i]) * (want[i] - c[i])
	}
	check("separable, clamped", p, make([]float64, n), want, &Options{Bounds: rep(Bound{Lo: -1, Hi: 1}, n)}, fstar)
}
