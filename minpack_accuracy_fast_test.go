package optimize

import (
	"math"
	"math/big"
	"math/rand/v2"
	"testing"

	"github.com/go-ndarray/ndarray"
)

// TestAccuracyBoundMinpackFast: the accuracy contract on the default kernels,
// which run from 32 equations or residuals (TestAccuracyBoundRoot has n = 2
// and TestAccuracyBoundLeastSquares m = 6, both on MINPACK's order). Every
// optimum is exact: integers, a 300-bit square root, or the big.Rat normal
// equations; never another float64 run. Bounds are the README's: 1e-7 for
// Root and Fsolve (hybrd stops at xtol 1.49e-8 relative), 1e-6 for the least
// squares solvers.
func TestAccuracyBoundMinpackFast(t *testing.T) {
	// 1. A coupled cubic system whose root is a vector of small integers:
	//    f_i = x_i^3 + x_i + (x_{i+1} - x_{i-1})/2 - c_i, c_i evaluated at
	//    x* exactly (integers and halves).
	coupled := func(n int) (SystemProblem, []float64, []float64) {
		xs, x0 := make([]float64, n), make([]float64, n)
		for i := range xs {
			xs[i] = float64(i%5 + 1)
			x0[i] = 1
		}
		at := func(x []float64, i int) float64 {
			v := x[i]*x[i]*x[i] + x[i]
			if i > 0 {
				v -= x[i-1] / 2
			}
			if i < n-1 {
				v += x[i+1] / 2
			}
			return v
		}
		c := make([]float64, n)
		for i := range c {
			c[i] = at(xs, i)
		}
		return SystemProblem{Func: func(x, f []float64) {
			for i := range f {
				f[i] = at(x, i) - c[i]
			}
		}}, x0, xs
	}
	// 2. Decoupled squares: x_i^2 = i + 2, root sqrt(i + 2) in 300 bits.
	squares := func(n int) (SystemProblem, []float64, []*big.Float) {
		x0, want := make([]float64, n), make([]*big.Float, n)
		for i := range x0 {
			x0[i] = float64(i + 2) // above the positive root: Newton-like steps stay on its side
			want[i] = new(big.Float).SetPrec(bigPrec).SetInt64(int64(i + 2))
			want[i].Sqrt(want[i])
		}
		return SystemProblem{Func: func(x, f []float64) {
			for i := range f {
				f[i] = x[i]*x[i] - float64(i+2)
			}
		}}, x0, want
	}
	for _, n := range []int{32, 40, 100} {
		for _, solve := range []struct {
			name string
			f    func(SystemProblem, []float64, *RootOptions) (*RootResult, error)
		}{{"Root", Root}, {"Fsolve", Fsolve}} {
			p, x0, want := coupled(n)
			r, err := solve.f(p, x0, nil)
			if err != nil || !r.Success {
				t.Fatal(solve.name, n, err, r)
			}
			if d := maxDist(r.X, want); d > 1e-7 {
				t.Errorf("%s coupled n=%d: error %g, bound 1e-7", solve.name, n, d)
			}
			sp, sx0, swant := squares(n)
			r, err = solve.f(sp, sx0, nil)
			if err != nil || !r.Success {
				t.Fatal(solve.name, n, err, r)
			}
			d := 0.0
			for i := range swant {
				d = math.Max(d, bigDist(r.X[i], swant[i]))
			}
			if d > 1e-7 {
				t.Errorf("%s squares n=%d: error %g, bound 1e-7", solve.name, n, d)
			}
		}
	}
	// 3. The line fitted to m >= 32 integer points against the exact big.Rat
	//    normal-equation solution: LM, Leastsq, CurveFit.
	rng := rand.New(rand.NewPCG(8, 9))
	for _, m := range []int{32, 40, 300} {
		ys := make([]int64, m)
		for i := range ys {
			ys[i] = int64(rng.IntN(21) - 10)
		}
		a, b := linearFit(ys)
		resid := func(x, f []float64) {
			for i, y := range ys {
				f[i] = x[0] + x[1]*float64(i) - float64(y)
			}
		}
		check := func(name string, x []float64) {
			if d := math.Max(ratDist(x[0], a), ratDist(x[1], b)); d > 1e-6 {
				t.Errorf("%s m=%d: error %g, bound 1e-6 (x = %v)", name, m, d, x)
			}
		}
		p := LSProblem{M: m, Func: resid}
		lr, err := LeastSquares(p, []float64{0, 0}, LM, nil)
		if err != nil {
			t.Fatal(err)
		}
		check("least_squares LM", lr.X)
		qr, err := Leastsq(p, []float64{0, 0}, nil)
		if err != nil {
			t.Fatal(err)
		}
		check("leastsq", qr.X)
		ts, yd := make([]float64, m), make([]float64, m)
		for i, y := range ys {
			ts[i], yd[i] = float64(i), float64(y)
		}
		xa, _ := ndarray.FromData(ts, m)
		ya, _ := ndarray.FromData(yd, m)
		cr, err := CurveFit(func(x float64, q []float64) float64 { return q[0] + q[1]*x }, xa, ya, []float64{0, 0}, nil)
		if err != nil {
			t.Fatal(err)
		}
		check("curve_fit", cr.Popt)
	}
	// 4. Noise-free exponential with exact parameters (2.5, 1.3, 0.5): the
	//    optimum is the generating point (the data carry only the rounding of
	//    math.Exp, about 1e-16).
	m := 300
	xs, ys := make([]float64, m), make([]float64, m)
	for i := range xs {
		xs[i] = 4 * float64(i) / float64(m-1)
		ys[i] = 2.5*math.Exp(-1.3*xs[i]) + 0.5
	}
	xa, _ := ndarray.FromData(xs, m)
	ya, _ := ndarray.FromData(ys, m)
	cr, err := CurveFit(func(x float64, p []float64) float64 { return p[0]*math.Exp(-p[1]*x) + p[2] }, xa, ya, []float64{1, 1, 1}, &CurveFitOptions{Method: LM})
	if err != nil {
		t.Fatal(err)
	}
	if d := maxDist(cr.Popt, []float64{2.5, 1.3, 0.5}); d > 1e-6 {
		t.Errorf("curve_fit exponential: error %g, bound 1e-6 (popt %v)", d, cr.Popt)
	}
}
