package optimize

import (
	"math"
	"math/big"
	"math/rand"
	"testing"

	"github.com/go-ndarray/ndarray"
)

// The accuracy contract (README, "Accuracy contract"): the solution of every
// solver family is within a stated bound of the optimum, and the optimum is
// computed exactly or in extended precision (math/big), never taken from
// another float64 implementation. These tests assert the bounds.

const bigPrec = 300

// bigDist is |got - want| for an oracle held as a big.Float.
func bigDist(got float64, want *big.Float) float64 {
	d := new(big.Float).SetPrec(bigPrec).SetFloat64(got)
	d.Sub(d, want)
	f, _ := d.Abs(d).Float64()
	return f
}

// bigCubeRoot returns the cube root of a positive integer to bigPrec bits by
// Newton's iteration in big.Float.
func bigCubeRoot(a int64) *big.Float {
	A := new(big.Float).SetPrec(bigPrec).SetInt64(a)
	x := new(big.Float).SetPrec(bigPrec).SetFloat64(math.Cbrt(float64(a)))
	three := new(big.Float).SetPrec(bigPrec).SetInt64(3)
	for i := 0; i < 12; i++ {
		x2 := new(big.Float).SetPrec(bigPrec).Mul(x, x)
		q := new(big.Float).SetPrec(bigPrec).Quo(A, x2)
		x.Add(x, x)
		x.Add(x, q)
		x.Quo(x, three)
	}
	return x
}

func maxDist(got, want []float64) float64 {
	m := 0.0
	for i := range want {
		m = math.Max(m, math.Abs(got[i]-want[i]))
	}
	return m
}

// TestAccuracyBoundMinimize: the minimiser and the optimal value of Rosenbrock
// (x* = (1, 1), f* = 0) and Himmelblau from (0, 0) (x* = (3, 2), f* = 0), both
// exact, for every Minimize method at its default tolerances.
func TestAccuracyBoundMinimize(t *testing.T) {
	type prob struct {
		name string
		p    Problem
		x0   []float64
		want []float64
	}
	probs := []prob{
		{"rosenbrock", Problem{Func: rosen, Grad: rosenDer, Hess: rosenHess}, []float64{-1.2, 1}, []float64{1, 1}},
		{"himmelblau", Problem{Func: himmelblau, Grad: himmelblauDer}, []float64{0, 0}, []float64{3, 2}},
	}
	// xBound bounds max|x - x*|, fBound bounds |f - f*|; the defaults of each
	// method stop at its own tolerance, so the bounds differ by method.
	cases := []struct {
		m      Method
		o      *Options
		xBound float64
		fBound float64
	}{
		{NelderMead, nil, 1e-3, 1e-6},
		{Powell, nil, 1e-3, 1e-6},
		{CG, nil, 1e-3, 1e-6},
		{BFGS, nil, 1e-3, 1e-6},
		{BFGS, &Options{SciPyExactUpdate: true}, 1e-3, 1e-6},
		{NewtonCG, nil, 1e-3, 1e-6},
		{LBFGSB, nil, 1e-3, 1e-6},
		{SLSQP, nil, 1e-3, 1e-6},
		{TrustConstr, nil, 1e-3, 1e-6},
	}
	for _, c := range cases {
		for _, pr := range probs {
			p := pr.p
			if c.m == NewtonCG && pr.name != "rosenbrock" {
				continue // Himmelblau has no Hessian here
			}
			r, err := Minimize(p, pr.x0, c.m, c.o)
			if err != nil {
				t.Fatalf("%s %s: %v", c.m, pr.name, err)
			}
			dx, df := maxDist(r.X, pr.want), math.Abs(r.Fun)
			t.Logf("%-12s %-10s exact=%-5v |x-x*|=%.2e |f-f*|=%.2e", c.m, pr.name, c.o != nil, dx, df)
			if dx > c.xBound || df > c.fBound {
				t.Errorf("%s on %s: |x-x*| = %g (bound %g), |f-f*| = %g (bound %g)", c.m, pr.name, dx, c.xBound, df, c.fBound)
			}
		}
	}
}

// TestAccuracyBoundScalarRoots: the root of x^3 - 2 against the cube root of 2
// in 300-bit arithmetic; the bound is the default tolerance 2e-12 + 4*eps*|x|
// of the bracketing methods.
func TestAccuracyBoundScalarRoots(t *testing.T) {
	want := bigCubeRoot(2)
	f := func(x float64) float64 { return x*x*x - 2 }
	fp := func(x float64) float64 { return 3 * x * x }
	const bound = 2e-12 + 4*epsmch*1.26
	for _, c := range []struct {
		name string
		f    func(func(float64) float64, float64, float64, *BracketOptions) (*RootScalarResult, error)
	}{{"brentq", Brentq}, {"brenth", Brenth}, {"ridder", Ridder}, {"bisect", Bisect}} {
		r, err := c.f(f, 1, 2, nil)
		if err != nil {
			t.Fatal(err)
		}
		if d := bigDist(r.Root, want); d > bound {
			t.Errorf("%s: |root - 2^(1/3)| = %g, bound %g", c.name, d, bound)
		}
	}
	r, err := Newton(f, 1.5, &NewtonOptions{Fprime: fp})
	if err != nil {
		t.Fatal(err)
	}
	if d := bigDist(r.Root, want); d > 1.48e-8 {
		t.Errorf("newton: |root - 2^(1/3)| = %g, bound 1.48e-8", d)
	}
}

// TestAccuracyBoundScalarMinimum: the minimiser of f(x) = x^4 - 4x + 5 is
// exactly 1; the achievable accuracy of a minimum found from function values
// is sqrt(eps), and the bound for Brent and Golden is 1e-6.
func TestAccuracyBoundScalarMinimum(t *testing.T) {
	want := new(big.Float).SetPrec(bigPrec).SetInt64(1)
	// d/dx (x^4 - 4x + 5) = 4x^3 - 4 = 0 at x = 1.
	f := func(x float64) float64 { return x*x*x*x - 4*x + 5 }
	for _, m := range []ScalarMethod{Brent, Golden} {
		r, err := MinimizeScalar(f, m, &ScalarOptions{Bracket: []float64{0, 2}})
		if err != nil {
			t.Fatal(err)
		}
		if d := bigDist(r.X, want); d > 1e-6 {
			t.Errorf("%s: |x - x*| = %g, bound 1e-6", m, d)
		}
	}
	r, err := Fminbound(f, 0, 3, 1e-5, 500)
	if err != nil {
		t.Fatal(err)
	}
	if d := bigDist(r.X, want); d > 1e-5+1e-6 {
		t.Errorf("fminbound: |x - x*| = %g, bound 1.1e-5", d)
	}
}

// TestAccuracyBoundRoot: x0^2 = 2, x1^3 = 3 (hybr, MINPACK), against the
// square root of 2 and the cube root of 3 in 300-bit arithmetic. The bound is
// 1e-7: hybrd stops at xtol 1.49e-8 relative.
func TestAccuracyBoundRoot(t *testing.T) {
	w0 := new(big.Float).SetPrec(bigPrec).SetInt64(2)
	w0.Sqrt(w0)
	w1 := bigCubeRoot(3)
	p := SystemProblem{Func: func(x, f []float64) {
		f[0] = x[0]*x[0] - 2
		f[1] = x[1]*x[1]*x[1] - 3
	}}
	for _, solve := range []func(SystemProblem, []float64, *RootOptions) (*RootResult, error){Root, Fsolve} {
		r, err := solve(p, []float64{1, 1}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if d := math.Max(bigDist(r.X[0], w0), bigDist(r.X[1], w1)); d > 1e-7 {
			t.Errorf("root: error %g, bound 1e-7 (x = %v)", d, r.X)
		}
	}
}

// linearFit is the exact least-squares line through (t, y) = (0..5, ys): the
// normal equations solved in big.Rat.
func linearFit(ys []int64) (a, b *big.Rat) {
	n := int64(len(ys))
	var st, stt, sy, sty int64
	for i, y := range ys {
		ti := int64(i)
		st += ti
		stt += ti * ti
		sy += y
		sty += ti * y
	}
	det := big.NewRat(n*stt-st*st, 1)
	a = new(big.Rat).Quo(big.NewRat(sy*stt-st*sty, 1), det)
	b = new(big.Rat).Quo(big.NewRat(n*sty-st*sy, 1), det)
	return a, b
}

func ratDist(got float64, want *big.Rat) float64 {
	g := new(big.Rat).SetFloat64(got)
	d, _ := g.Sub(g, want).Float64()
	return math.Abs(d)
}

// TestAccuracyBoundLeastSquares: the line fitted to six integer points
// against the exact (big.Rat) normal-equation solution, for LM, TRF, Leastsq
// and CurveFit; the bound is 1e-6 (the 1.49e-8 relative xtol times the
// condition of the problem).
func TestAccuracyBoundLeastSquares(t *testing.T) {
	ys := []int64{1, 3, 2, 5, 4, 7}
	a, b := linearFit(ys)
	resid := func(x, f []float64) {
		for i, y := range ys {
			f[i] = x[0] + x[1]*float64(i) - float64(y)
		}
	}
	check := func(name string, x []float64) {
		if d := math.Max(ratDist(x[0], a), ratDist(x[1], b)); d > 1e-6 {
			t.Errorf("%s: error %g, bound 1e-6 (x = %v)", name, d, x)
		}
	}
	p := LSProblem{M: len(ys), Func: resid}
	for _, m := range []LSMethod{LM, TRF} {
		r, err := LeastSquares(p, []float64{0, 0}, m, nil)
		if err != nil {
			t.Fatal(err)
		}
		check("least_squares "+string(m), r.X)
	}
	lr, err := Leastsq(p, []float64{0, 0}, nil)
	if err != nil {
		t.Fatal(err)
	}
	check("leastsq", lr.X)
	ts := make([]float64, len(ys))
	yd := make([]float64, len(ys))
	for i, y := range ys {
		ts[i], yd[i] = float64(i), float64(y)
	}
	xa, _ := ndarray.FromData(ts, len(ts))
	ya, _ := ndarray.FromData(yd, len(yd))
	cr, err := CurveFit(func(x float64, q []float64) float64 { return q[0] + q[1]*x }, xa, ya, []float64{0, 0}, nil)
	if err != nil {
		t.Fatal(err)
	}
	check("curve_fit", cr.Popt)
}

// TestAccuracyBoundLinear: a linear program and a quadratic program whose
// optima are exact (vertex (10, -3), value -22; minimiser (1, 1), value -3).
func TestAccuracyBoundLinear(t *testing.T) {
	A, _ := ndarray.FromData([]float64{-3, 1, 1, 2}, 2, 2)
	b, _ := ndarray.FromData([]float64{6, 4}, 2)
	r, err := Linprog([]float64{-1, 4}, LPConstraints{AUb: A, BUb: b,
		Bounds: []Bound{Unbounded(), {Lo: -3, Hi: math.Inf(1)}}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if d := math.Max(maxDist(r.X, []float64{10, -3}), math.Abs(r.Fun+22)); d > 1e-9 {
		t.Errorf("linprog: error %g, bound 1e-9", d)
	}
	P, _ := ndarray.FromData([]float64{2, 0, 0, 4}, 2, 2)
	q, err := Quadprog(QPProblem{P: P, Q: []float64{-2, -4}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if d := math.Max(maxDist(q.X, []float64{1, 1}), math.Abs(q.Fun+3)); d > 1e-12 {
		t.Errorf("quadprog: error %g, bound 1e-12", d)
	}
}

// TestAccuracyBoundEnorm: the two Euclidean norms of the package, enorm (a
// scaled sum) and mpEnorm (MINPACK's three accumulators, which the Root and
// LM ports use) against a 300-bit sum of squares and square root, over inputs
// from 2^-1060 (subnormal) to 2^1000 and mixed magnitudes: the error stays
// within n/2 + 2 ulps of the result, the worst case of a recursive sum, and the
// norm neither overflows nor underflows spuriously.
func TestAccuracyBoundEnorm(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	exps := []int{0, -40, -600, -1060, 500, 1000}
	for _, n := range []int{1, 3, 10, 100, 1000} {
		bound := float64(n)/2 + 2
		for _, e := range exps {
			for rep := 0; rep < 20; rep++ {
				v := make([]float64, n)
				for i := range v {
					v[i] = math.Ldexp(r.NormFloat64(), e)
				}
				checkEnorm(t, v, bound)
			}
		}
		for rep := 0; rep < 20; rep++ { // mixed magnitudes
			v := make([]float64, n)
			for i := range v {
				v[i] = math.Ldexp(r.NormFloat64(), r.Intn(1100)-600)
			}
			checkEnorm(t, v, bound)
		}
	}
}

func checkEnorm(t *testing.T, v []float64, bound float64) {
	t.Helper()
	want := bigNorm(v)
	for name, f := range map[string]func([]float64) float64{"enorm": enorm, "mpEnorm": mpEnorm} {
		if e := ulpErr(f(v), want); !(e <= bound) {
			t.Fatalf("%s: n=%d error %g ulp, bound %g", name, len(v), e, bound)
		}
	}
}
