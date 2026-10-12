package optimize

import (
	"math"
	"testing"

	"github.com/go-ndarray/ndarray"
)

// Benchmarks mirrored by benchmarks/bench_scipy.py; benchmarks/run.sh runs
// both interleaved and reports medians (docs/perf.md).

func benchMinimize(b *testing.B, n int, m Method, withGrad bool) {
	x0 := altStart(n)
	p := Problem{Func: rosen}
	if withGrad {
		p.Grad = rosenDer
	}
	b.ReportAllocs()
	for b.Loop() {
		r, err := Minimize(p, x0, m, nil)
		if err != nil || r.Fun > 1e-6 && n == 2 {
			b.Fatal(err, r)
		}
	}
}

func BenchmarkRosen2BFGS(b *testing.B)       { benchMinimize(b, 2, BFGS, true) }
func BenchmarkRosen100BFGS(b *testing.B)     { benchMinimize(b, 100, BFGS, true) }
func BenchmarkRosen2LBFGSB(b *testing.B)     { benchMinimize(b, 2, LBFGSB, true) }
func BenchmarkRosen100LBFGSB(b *testing.B)   { benchMinimize(b, 100, LBFGSB, true) }
func BenchmarkRosen2BFGSFD(b *testing.B)     { benchMinimize(b, 2, BFGS, false) }
func BenchmarkRosen100LBFGSBFD(b *testing.B) { benchMinimize(b, 100, LBFGSB, false) }
func BenchmarkRosen2NelderMead(b *testing.B) { benchMinimize(b, 2, NelderMead, false) }

// curveFitData is a decaying exponential with deterministic pseudo-noise
// (benchmarks/bench_scipy.py builds the same data).
func curveFitData(m int) (x, y *ndarray.Array) {
	xs := make([]float64, m)
	ys := make([]float64, m)
	for i := range xs {
		xs[i] = 4 * float64(i) / float64(m-1)
		noise := 0.2 * math.Sin(float64(i)*12.9898)
		ys[i] = 2.5*math.Exp(-1.3*xs[i]) + 0.5 + noise
	}
	x, _ = ndarray.FromData(xs, m)
	y, _ = ndarray.FromData(ys, m)
	return x, y
}

func benchCurveFit(b *testing.B, m int, method LSMethod, workers int) {
	x, y := curveFitData(m)
	f := func(x float64, p []float64) float64 { return p[0]*math.Exp(-p[1]*x) + p[2] }
	b.ReportAllocs()
	for b.Loop() {
		r, err := CurveFit(f, x, y, []float64{1, 1, 1}, &CurveFitOptions{Method: method, Workers: workers})
		if err != nil || math.Abs(r.Popt[1]-1.3) > 0.1 {
			b.Fatal(err, r)
		}
	}
}

func BenchmarkCurveFit1e4LM(b *testing.B)   { benchCurveFit(b, 10000, LM, 1) }
func BenchmarkCurveFit1e4TRF(b *testing.B)  { benchCurveFit(b, 10000, TRF, 1) }
func BenchmarkCurveFit1e4LMW4(b *testing.B) { benchCurveFit(b, 10000, LM, 4) }
func BenchmarkCurveFit1e4TRFW4(b *testing.B) {
	benchCurveFit(b, 10000, TRF, 4)
}

func BenchmarkBrentq(b *testing.B) {
	f := func(x float64) float64 { return x*x*x - 2*x - 5 }
	b.ReportAllocs()
	for b.Loop() {
		r, err := Brentq(f, 2, 3, nil)
		if err != nil || !r.Converged {
			b.Fatal(err)
		}
	}
}

// The rest of the public operations, each on a problem that
// benchmarks/bench_scipy.py poses to SciPy in the same form.

func BenchmarkRosen2Powell(b *testing.B)   { benchMinimize(b, 2, Powell, false) }
func BenchmarkRosen2CG(b *testing.B)       { benchMinimize(b, 2, CG, true) }
func BenchmarkRosen100CG(b *testing.B)     { benchMinimize(b, 100, CG, true) }
func BenchmarkRosen100BFGSFD(b *testing.B) { benchMinimize(b, 100, BFGS, false) }

// rosenHessProd is scipy.optimize.rosen_hess_prod: the Hessian of rosen
// times p, in O(n).
func rosenHessProd(x, p, hp []float64) {
	n := len(x)
	hp[0] = float64(float64(1200*x[0]*x[0])-float64(400*x[1])+2)*p[0] - float64(400*x[0]*p[1])
	for i := 1; i < n-1; i++ {
		hp[i] = float64(-400*x[i-1]*p[i-1]) +
			float64(float64(202+float64(1200*x[i]*x[i])-float64(400*x[i+1]))*p[i]) -
			float64(400*x[i]*p[i+1])
	}
	hp[n-1] = float64(-400*x[n-2]*p[n-2]) + float64(200*p[n-1])
}

func benchNewtonCG(b *testing.B, n int, hess bool) {
	x0 := altStart(n)
	p := Problem{Func: rosen, Grad: rosenDer}
	if hess {
		p.Hess = rosenHess
	} else {
		p.HessP = rosenHessProd
	}
	b.ReportAllocs()
	for b.Loop() {
		if r, err := Minimize(p, x0, NewtonCG, nil); err != nil || r.Fun > 1e-3 {
			b.Fatal(err, r)
		}
	}
}

func BenchmarkRosen2NewtonCGHess(b *testing.B)    { benchNewtonCG(b, 2, true) }
func BenchmarkRosen100NewtonCGHessP(b *testing.B) { benchNewtonCG(b, 100, false) }

// rosenResid is Rosenbrock's function as two residuals.
func rosenResid(x, f []float64) {
	f[0] = 10 * (x[1] - float64(x[0]*x[0]))
	f[1] = 1 - x[0]
}

func BenchmarkLeastSquaresRosenTRF(b *testing.B) { benchLeastSquares(b, TRF) }
func BenchmarkLeastSquaresRosenLM(b *testing.B)  { benchLeastSquares(b, LM) }

func benchLeastSquares(b *testing.B, m LSMethod) {
	b.ReportAllocs()
	for b.Loop() {
		r, err := LeastSquares(LSProblem{Func: rosenResid, M: 2}, []float64{-1.2, 1}, m, nil)
		if err != nil || r.Cost > 1e-12 {
			b.Fatal(err, r)
		}
	}
}

func BenchmarkLeastsqRosen(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		r, err := Leastsq(LSProblem{Func: rosenResid, M: 2}, []float64{-1.2, 1}, nil)
		if err != nil || !r.Success() {
			b.Fatal(err, r)
		}
	}
}

func BenchmarkCurveFit1e4TRFBounds(b *testing.B) {
	x, y := curveFitData(10000)
	f := func(x float64, p []float64) float64 { return p[0]*math.Exp(-p[1]*x) + p[2] }
	o := &CurveFitOptions{Bounds: []Bound{{0, 10}, {0, 3}, {-1, 1}}}
	b.ReportAllocs()
	for b.Loop() {
		r, err := CurveFit(f, x, y, []float64{1, 1, 0}, o)
		if err != nil || math.Abs(r.Popt[1]-1.3) > 0.1 {
			b.Fatal(err, r)
		}
	}
}

// rootFun is the system of SciPy's root documentation example.
func rootFun(x, f []float64) {
	d := x[0] - x[1]
	f[0] = x[0] + float64(0.5*d*d*d) - 1
	f[1] = float64(0.5*-d*-d*-d) + x[1]
}

// broydenTri is Broyden's tridiagonal system (Moré, Garbow, Hillstrom).
func broydenTri(x, f []float64) {
	n := len(x)
	for i := range n {
		v := float64((3-2*x[i])*x[i]) + 1
		if i > 0 {
			v -= x[i-1]
		}
		if i < n-1 {
			v -= 2 * x[i+1]
		}
		f[i] = v
	}
}

func benchRoot(b *testing.B, f func(x, f []float64), x0 []float64, fsolve bool) {
	b.ReportAllocs()
	for b.Loop() {
		var r *RootResult
		var err error
		if fsolve {
			r, err = Fsolve(SystemProblem{Func: f}, x0, nil)
		} else {
			r, err = Root(SystemProblem{Func: f}, x0, nil)
		}
		if err != nil || !r.Success {
			b.Fatal(err, r)
		}
	}
}

func BenchmarkRootHybr(b *testing.B)   { benchRoot(b, rootFun, []float64{0, 0}, false) }
func BenchmarkFsolveHybr(b *testing.B) { benchRoot(b, rootFun, []float64{0, 0}, true) }
func BenchmarkRootBroyden100(b *testing.B) {
	x0 := make([]float64, 100)
	for i := range x0 {
		x0[i] = -1
	}
	benchRoot(b, broydenTri, x0, false)
}

// lpDense is a feasible, bounded LP with m inequality rows on n variables
// whose data come from the same deterministic formulas as in
// benchmarks/bench_scipy.py.
func lpDense(m, n int) ([]float64, LPConstraints) {
	A := make([]float64, m*n)
	for i := range m {
		for j := range n {
			A[i*n+j] = 1 + math.Sin(float64(i*n+j)*12.9898)*0.5
		}
	}
	bv := make([]float64, m)
	for i := range bv {
		bv[i] = float64(n) * (1 + 0.5*math.Sin(float64(i)*78.233))
	}
	c := make([]float64, n)
	for j := range c {
		c[j] = -1 - math.Sin(float64(j)*4.1414)*0.5
	}
	Aa, _ := ndarray.FromData(A, m, n)
	ba, _ := ndarray.FromData(bv, m)
	return c, LPConstraints{AUb: Aa, BUb: ba}
}

func benchLinprog(b *testing.B, c []float64, cons LPConstraints, method LPMethod) {
	b.ReportAllocs()
	for b.Loop() {
		r, err := Linprog(c, cons, &LPOptions{Method: method})
		if err != nil || r.Status != 0 {
			b.Fatal(err, r)
		}
	}
}

func BenchmarkLinprogSmall(b *testing.B) {
	A, _ := ndarray.FromData([]float64{-3, 1, 1, 2}, 2, 2)
	bv, _ := ndarray.FromData([]float64{6, 4}, 2)
	cons := LPConstraints{AUb: A, BUb: bv, Bounds: []Bound{Unbounded(), {-3, math.Inf(1)}}}
	benchLinprog(b, []float64{-1, 4}, cons, LPSimplex)
}

func BenchmarkLinprogDense30x50(b *testing.B) {
	c, cons := lpDense(30, 50)
	benchLinprog(b, c, cons, LPSimplex)
}

func BenchmarkLinprogDense30x50IP(b *testing.B) {
	c, cons := lpDense(30, 50)
	benchLinprog(b, c, cons, LPInteriorPoint)
}

func cubic(x float64) float64 { return x*x*x - 2*x - 5 }

func benchBracketRoot(b *testing.B, root func(func(float64) float64, float64, float64, *BracketOptions) (*RootScalarResult, error)) {
	b.ReportAllocs()
	for b.Loop() {
		r, err := root(cubic, 2, 3, nil)
		if err != nil || !r.Converged {
			b.Fatal(err)
		}
	}
}

func BenchmarkBrenth(b *testing.B) { benchBracketRoot(b, Brenth) }
func BenchmarkRidder(b *testing.B) { benchBracketRoot(b, Ridder) }
func BenchmarkBisect(b *testing.B) { benchBracketRoot(b, Bisect) }

func benchNewton(b *testing.B, o *NewtonOptions) {
	b.ReportAllocs()
	for b.Loop() {
		r, err := Newton(cubic, 2, o)
		if err != nil || !r.Converged {
			b.Fatal(err)
		}
	}
}

func BenchmarkNewton(b *testing.B) {
	benchNewton(b, &NewtonOptions{Fprime: func(x float64) float64 { return 3*x*x - 2 }})
}
func BenchmarkSecant(b *testing.B) { benchNewton(b, nil) }
func BenchmarkHalley(b *testing.B) {
	benchNewton(b, &NewtonOptions{Fprime: func(x float64) float64 { return 3*x*x - 2 },
		Fprime2: func(x float64) float64 { return 6 * x }})
}

// scalarF is the function of SciPy's minimize_scalar documentation.
func scalarF(x float64) float64 { return (x - 2) * x * (x + 2) * (x + 2) }

func benchMinScalar(b *testing.B, m ScalarMethod, o *ScalarOptions) {
	b.ReportAllocs()
	for b.Loop() {
		if _, err := MinimizeScalar(scalarF, m, o); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMinScalarBrent(b *testing.B)  { benchMinScalar(b, Brent, nil) }
func BenchmarkMinScalarGolden(b *testing.B) { benchMinScalar(b, Golden, nil) }
func BenchmarkMinScalarBounded(b *testing.B) {
	benchMinScalar(b, Bounded, &ScalarOptions{Bounds: &Bound{-3, -1}})
}

func BenchmarkFminbound(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		if _, err := Fminbound(math.Cos, 3, 4, 0, 0); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkBracket(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		if _, _, _, err := Bracket(scalarF, 0, 1); err != nil {
			b.Fatal(err)
		}
	}
}

func himmelblau2(x []float64) float64 {
	a := x[0]*x[0] + x[1] - 11
	c := x[0] + x[1]*x[1] - 7
	return a*a + c*c
}

func BenchmarkBrute2D(b *testing.B) {
	rg := []Range{{-4, 4, 0}, {-4, 4, 0}}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := Brute(himmelblau2, rg, &BruteOptions{Ns: 50, NoFinish: true}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkBrute2DFinish(b *testing.B) {
	rg := []Range{{-4, 4, 0}, {-4, 4, 0}}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := Brute(himmelblau2, rg, &BruteOptions{Ns: 20}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkApproxFprime100(b *testing.B) {
	x := altStart(100)
	b.ReportAllocs()
	for b.Loop() {
		ApproxFprime(rosen, x, 0)
	}
}

func BenchmarkCheckGrad100(b *testing.B) {
	x := altStart(100)
	b.ReportAllocs()
	for b.Loop() {
		CheckGrad(rosen, rosenDer, x, 0)
	}
}
