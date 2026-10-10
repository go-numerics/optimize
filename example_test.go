package optimize_test

import (
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-numerics/optimize"
)

// scipy.optimize.minimize(rosen, x0, jac=rosen_der, method='BFGS')
func ExampleMinimize() {
	rosen := func(x []float64) float64 {
		return 100*math.Pow(x[1]-x[0]*x[0], 2) + math.Pow(1-x[0], 2)
	}
	grad := func(x, g []float64) {
		g[0] = -400*x[0]*(x[1]-x[0]*x[0]) - 2*(1-x[0])
		g[1] = 200 * (x[1] - x[0]*x[0])
	}
	r, err := optimize.Minimize(optimize.Problem{Func: rosen, Grad: grad}, []float64{-1.2, 1}, optimize.BFGS, nil)
	if err != nil {
		panic(err)
	}
	fmt.Printf("x = [%.6f %.6f], nit = %d, %s\n", r.X[0], r.X[1], r.Nit, r.Message)
	// Output: x = [1.000000 1.000000], nit = 32, Optimization terminated successfully.
}

// scipy.optimize.minimize(f, x0, method='L-BFGS-B', bounds=[(-2, 0.5), (None, None)])
func ExampleMinimize_bounds() {
	f := func(x []float64) float64 { return (x[0]-1)*(x[0]-1) + (x[1]+2)*(x[1]+2) }
	opts := &optimize.Options{Bounds: []optimize.Bound{{Lo: -2, Hi: 0.5}, optimize.Unbounded()}}
	r, _ := optimize.Minimize(optimize.Problem{Func: f}, []float64{0, 0}, optimize.LBFGSB, opts)
	fmt.Printf("x = [%.4f %.4f]\n", r.X[0], r.X[1])
	// Output: x = [0.5000 -2.0000]
}

// popt, pcov = scipy.optimize.curve_fit(lambda x, a, b, c: a*np.exp(-b*x) + c, x, y)
func ExampleCurveFit() {
	x, _ := ndarray.Linspace(0, 4, 50)
	y := x.Map(func(v float64) float64 { return 2.5*math.Exp(-1.3*v) + 0.5 + 0.01*math.Sin(37*v) })
	model := func(x float64, p []float64) float64 { return p[0]*math.Exp(-p[1]*x) + p[2] }
	r, err := optimize.CurveFit(model, x, y, []float64{1, 1, 1}, nil)
	if err != nil {
		panic(err)
	}
	perr := r.Perr()
	fmt.Printf("a = %.3f ± %.3f, b = %.3f ± %.3f, c = %.3f ± %.3f\n",
		r.Popt[0], perr[0], r.Popt[1], perr[1], r.Popt[2], perr[2])
	// Output: a = 2.500 ± 0.004, b = 1.300 ± 0.005, c = 0.500 ± 0.002
}

// scipy.optimize.brentq(lambda x: x**3 - 2*x - 5, 2, 3)
func ExampleBrentq() {
	r, _ := optimize.Brentq(func(x float64) float64 { return x*x*x - 2*x - 5 }, 2, 3, nil)
	fmt.Printf("%.12f after %d iterations\n", r.Root, r.Iterations)
	// Output: 2.094551481542 after 7 iterations
}

// scipy.optimize.root(fun, [0, 0], method='hybr')
func ExampleRoot() {
	fun := func(x, f []float64) {
		f[0] = x[0] + 0.5*math.Pow(x[0]-x[1], 3) - 1
		f[1] = 0.5*math.Pow(x[1]-x[0], 3) + x[1]
	}
	r, _ := optimize.Root(optimize.SystemProblem{Func: fun}, []float64{0, 0}, nil)
	fmt.Printf("x = [%.6f %.6f] %v\n", r.X[0], r.X[1], r.Success)
	// Output: x = [0.841164 0.158836] true
}

// scipy.optimize.least_squares(fun, x0, bounds=([0, 0], [2, 0.5]), loss='soft_l1')
func ExampleLeastSquares() {
	t := []float64{0, 1, 2, 3, 4, 5}
	y := []float64{1.0, 0.62, 0.36, 0.23, 0.13, 3.0} // the last point is an outlier
	p := optimize.LSProblem{M: len(t), Func: func(x, f []float64) {
		for i := range t {
			f[i] = x[0]*math.Exp(-x[1]*t[i]) - y[i]
		}
	}}
	r, _ := optimize.LeastSquares(p, []float64{1, 0.1}, optimize.TRF, &optimize.LSOptions{
		Bounds: []optimize.Bound{{Lo: 0, Hi: 2}, {Lo: 0, Hi: 0.5}}, Loss: optimize.LossSoftL1, FScale: 0.1})
	fmt.Printf("x = [%.3f %.3f] active %v\n", r.X[0], r.X[1], r.ActiveMask)
	// Output: x = [0.990 0.466] active [0 0]
}

// scipy.optimize.linprog(c, A_ub=A, b_ub=b, bounds=[(None, None), (-3, None)])
func ExampleLinprog() {
	A, _ := ndarray.FromData([]float64{-3, 1, 1, 2}, 2, 2)
	b, _ := ndarray.FromData([]float64{6, 4}, 2)
	r, _ := optimize.Linprog([]float64{-1, 4}, optimize.LPConstraints{AUb: A, BUb: b,
		Bounds: []optimize.Bound{optimize.Unbounded(), {Lo: -3, Hi: math.Inf(1)}}}, nil)
	fmt.Printf("x = [%.1f %.1f], fun = %.1f, %s\n", r.X[0], r.X[1], r.Fun, r.Message)
	// Output: x = [10.0 -3.0], fun = -22.0, Optimization terminated successfully.
}
