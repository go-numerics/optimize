package bench

import (
	"bytes"
	"context"
	"crypto/sha256"
	_ "embed"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/go-ndarray/ndarray"
	opt "github.com/go-numerics/optimize"
	"github.com/go-scikit/refbench"
	"github.com/go-scikit/refbench/rbtest"
)

// The speed comparison with scipy.optimize (docs/perf.md).
//
//	cd bench
//	go test -run TestPerf .                          CI mode: allocations, table, inventory
//	go test -run TestPerf -cpu 1 -perf.run .         measure (refuses at a 1-minute load of 8 or more)
//	go test -run TestPerf -perf.record .             allocation counts only (a deferred report)
//	go test -run 'TestDataParity|TestCounts' -perf.parity .   same bytes, same nfev/nit (needs uv)
//
// What is compared. One row is one call of the solver, from the same starting
// point, with the same tolerances (the defaults of both sides) and the same
// problem. The SciPy objective is SciPy's own vectorised rosen / rosen_der or
// the cheapest NumPy expression of the function, so that its Python callbacks
// are as small as they can be; they still dominate small problems (docs/perf.md
// quantifies the share), so a row on a small problem times the solver overhead
// plus a Python call per evaluation on one side and a Go call on the other.
// A row is only comparable when both sides take the same path: TestCounts
// compares nfev and nit and names every row where they differ.

var (
	genFlag    = flag.Bool("perf.gen", false, "TestGenData: regenerate testdata/data.f64")
	parityFlag = flag.Bool("perf.parity", false, "TestDataParity and TestCounts: start the SciPy worker through uv")
	sink       float64
)

// row is one solver call posed to both sides.
type row struct {
	name, fn string
	n        int    // the size n of the Python setup expression
	setup    string // Python: the argument list
	data     []any  // the same arguments on the Go side: []float64 or float64
	call     string // Python: a callable of the arguments; %FO% marks where TestCounts adds full_output=True
	ref      string // the SciPy function, for the table
	run      func() (nfev, nit int)
	nocount  string // non-empty: why nfev/nit are not compared
	extra    int    // SciPy's nfev minus ours on the same path (root counts its wrapper's two calls)
}

// size is the n of the Python setup (the worker requires one).
func (r row) size() int { return max(r.n, 1) }

const soImport = `__import__("scipy.optimize", fromlist=["_"])`

// py builds a callable expression: params are the arguments of setup, body may
// use so (scipy.optimize) and the hoisted definitions defs ("name", "source", …),
// which are evaluated once, outside the timed loop.
func py(params, body string, defs ...string) string {
	names, srcs := "", ""
	for i := 0; i < len(defs); i += 2 {
		names += ", " + defs[i]
		srcs += ", " + defs[i+1]
	}
	return "(lambda so" + names + ": lambda " + params + ": " + body + ")(" + soImport + srcs + ")"
}

func alt(n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = 1
		if i%2 == 0 {
			x[i] = -1.2
		}
	}
	return x
}

const altSetup = "[np.array([-1.2 if i % 2 == 0 else 1.0 for i in range(n)])]"

// The objectives. float64() forbids a fused multiply-add, so that the roundings
// are NumPy's (the same forms as the package's own tests).
func rosen(x []float64) float64 {
	s := 0.0
	for i := 0; i+1 < len(x); i++ {
		a := x[i+1] - float64(x[i]*x[i])
		b := 1 - x[i]
		s += float64(100*float64(a*a)) + float64(b*b)
	}
	return s
}

func rosenDer(x, g []float64) {
	n := len(x)
	for i := range g {
		g[i] = 0
	}
	for i := 0; i+1 < n; i++ {
		a := x[i+1] - float64(x[i]*x[i])
		g[i] += float64(-400*x[i]*a) - float64(2*(1-x[i]))
		g[i+1] += float64(200 * a)
	}
}

func rosenHess(x, h []float64) {
	n := len(x)
	for i := range h {
		h[i] = 0
	}
	for i := 0; i+1 < n; i++ {
		h[i*n+i] += float64(1200*x[i]*x[i]) - float64(400*x[i+1]) + 2
		h[i*n+i+1] += float64(-400 * x[i])
		h[(i+1)*n+i] += float64(-400 * x[i])
		h[(i+1)*n+i+1] += 200
	}
}

// rosenHessProd is scipy.optimize.rosen_hess_prod.
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

func rosenResid(x, f []float64) {
	f[0] = 10 * (x[1] - float64(x[0]*x[0]))
	f[1] = 1 - x[0]
}

const pyRosenResid = `lambda x: np.array([10 * (x[1] - x[0] ** 2), 1 - x[0]])`

func rootFun(x, f []float64) {
	d := x[0] - x[1]
	f[0] = x[0] + float64(0.5*d*d*d) - 1
	f[1] = float64(0.5*-d*-d*-d) + x[1]
}

const pyRootFun = `lambda x: [x[0] + 0.5 * (x[0] - x[1]) ** 3 - 1.0, 0.5 * (x[1] - x[0]) ** 3 + x[1]]`

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

const pyBroyden = `lambda x: (lambda f: (f.__setitem__(slice(1, None), f[1:] - x[:-1]), f.__setitem__(slice(None, -1), f[:-1] - 2 * x[1:]), f)[2])((3 - 2 * x) * x + 1)`

func cubic(x float64) float64 { return x*x*x - 2*x - 5 }

const pyCubic = `lambda x: x * x * x - 2 * x - 5`

func scalarF(x float64) float64 { return (x - 2) * x * (x + 2) * (x + 2) }

const pyScalarF = `lambda x: (x - 2) * x * (x + 2) ** 2`

func himmelblau2(x []float64) float64 {
	a := x[0]*x[0] + x[1] - 11
	c := x[0] + x[1]*x[1] - 7
	return a*a + c*c
}

const pyHimmelblau = `lambda p: (p[0] * p[0] + p[1] - 11) ** 2 + (p[0] + p[1] * p[1] - 7) ** 2`

// curveFitData is a decaying exponential with deterministic pseudo-noise.
func genCurve(m int) (xs, ys []float64) {
	xs = make([]float64, m)
	ys = make([]float64, m)
	for i := range xs {
		xs[i] = float64(4*float64(i)) / float64(m-1)
		noise := float64(0.2 * math.Sin(float64(i)*12.9898))
		ys[i] = float64(float64(2.5*math.Exp(float64(-1.3*xs[i])))+0.5) + noise
	}
	return xs, ys
}

// The data of the curve-fit and LP rows are generated with Go's math.Exp and
// math.Sin (TestGenData, -perf.gen) into testdata/data.f64 and read from that
// file by both sides: NumPy's sin and exp differ from Go's in the last bit, and a
// row is only a comparison when the bytes are the same. Layout (little-endian
// float64): x[10000], y[10000], c[50], A[30*50], b[30].
//
//go:embed testdata/data.f64
var dataBytes []byte

func dataF64() []float64 {
	d := make([]float64, len(dataBytes)/8)
	for i := range d {
		d[i] = math.Float64frombits(binary.LittleEndian.Uint64(dataBytes[8*i:]))
	}
	return d
}

func curveFitData(m int) (xs, ys []float64) {
	d := dataF64()
	return d[:m], d[m : 2*m]
}

const dataLoad = `np.fromfile("testdata/data.f64", dtype="<f8")`
const curveSetup = `(lambda d: [d[:n], d[n:2 * n]])(` + dataLoad + `)`

func TestGenData(t *testing.T) {
	if !*genFlag {
		t.Skip("set -perf.gen to regenerate testdata/data.f64")
	}
	xs, ys := genCurve(10000)
	c, A, b := genLP(30, 50)
	var buf bytes.Buffer
	for _, s := range [][]float64{xs, ys, c, A, b} {
		for _, v := range s {
			var w [8]byte
			binary.LittleEndian.PutUint64(w[:], math.Float64bits(v))
			buf.Write(w[:])
		}
	}
	if err := os.WriteFile("testdata/data.f64", buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

const pyModel = `lambda x, a, b, c: a * np.exp(-b * x) + c`

func arr(v []float64) *ndarray.Array {
	a, err := ndarray.FromData(append([]float64(nil), v...), len(v))
	if err != nil {
		panic(err)
	}
	return a
}

func must[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

func lpDense(m, n int) ([]float64, []float64, []float64) {
	d := dataF64()[20000:]
	return d[:n], d[n : n+m*n], d[n+m*n : n+m*n+m]
}

func genLP(m, n int) ([]float64, []float64, []float64) {
	A := make([]float64, m*n)
	for i := range A {
		A[i] = 1 + float64(math.Sin(float64(i)*12.9898)*0.5)
	}
	bv := make([]float64, m)
	for i := range bv {
		bv[i] = float64(n) * (1 + float64(0.5*math.Sin(float64(i)*78.233)))
	}
	c := make([]float64, n)
	for j := range c {
		c[j] = -1 - float64(math.Sin(float64(j)*4.1414)*0.5)
	}
	return c, A, bv
}

const lpSetup = `(lambda d: [d[20000:20050], d[20050:21550].reshape(30, 50), d[21550:21580]])(` + dataLoad + `)`

func box(n int, lo, hi float64) []opt.Bound {
	b := make([]opt.Bound, n)
	for i := range b {
		b[i] = opt.Bound{Lo: lo, Hi: hi}
	}
	return b
}

func sphere(x []float64) float64 {
	s := 0.0
	for _, v := range x {
		s += float64(v * v)
	}
	return s
}

func rosenbrockND(x []float64) float64 { return rosen(x) }

func himmelblauFG(x, g []float64) float64 {
	a := float64(x[0]*x[0]) + x[1] - 11.0
	b := x[0] + float64(x[1]*x[1]) - 7.0
	g[0] = float64(float64(4.0*x[0])*a) + float64(2.0*b)
	g[1] = float64(2.0*a) + float64(float64(4.0*x[1])*b)
	return float64(a*a) + float64(b*b)
}

// mini is a Minimize row on Rosenbrock.
func mini(name string, n int, m opt.Method, scipy, kw string, p opt.Problem, o *opt.Options) row {
	x0 := alt(n)
	return row{
		name: name, fn: "Minimize", n: n, setup: altSetup, data: []any{x0},
		call: py("x0", "so.minimize(so.rosen, x0, method='"+scipy+"'"+kw+")"),
		ref:  "scipy.optimize.minimize(method='" + scipy + "')",
		run: func() (int, int) {
			r, err := opt.Minimize(p, x0, m, o)
			if err != nil {
				panic(err)
			}
			sink = r.Fun
			return r.Nfev, r.Nit
		},
	}
}

func pf(f func([]float64) float64, g func(x, g []float64)) opt.Problem {
	return opt.Problem{Func: f, Grad: g}
}

func scalarRow(name, fn, ref, pyBody string, go1 func() *opt.RootScalarResult) row {
	return row{name: name, fn: fn, setup: "[2.0, 3.0]", data: []any{2.0, 3.0},
		call: py("a, b", pyBody, "f", pyCubic), ref: ref,
		run: func() (int, int) {
			r := go1()
			sink = r.Root
			return r.FunctionCalls, r.Iterations
		}}
}

func bracketRow(name, fn, scipy string, g func(func(float64) float64, float64, float64, *opt.BracketOptions) (*opt.RootScalarResult, error)) row {
	return scalarRow(name, fn, "scipy.optimize."+scipy, "so."+scipy+"(f, a, b%FO%)", func() *opt.RootScalarResult {
		return must(g(cubic, 2, 3, nil))
	})
}

func newtonRow(name string, o *opt.NewtonOptions, scipyKW string) row {
	r := row{name: name, fn: "Newton", setup: "[2.0]", data: []any{2.0},
		call: py("x0", "so.newton(f, x0"+scipyKW+"%FO%)", "f", pyCubic, "f1", "lambda x: 3 * x * x - 2", "f2", "lambda x: 6 * x"),
		ref:  "scipy.optimize.newton"}
	r.run = func() (int, int) {
		res := must(opt.Newton(cubic, 2, o))
		sink = res.Root
		return res.FunctionCalls, res.Iterations
	}
	return r
}

func scalarMin(name string, m opt.ScalarMethod, o *opt.ScalarOptions, kw string) row {
	return row{name: name, fn: "MinimizeScalar", setup: "[]", data: nil,
		call: py("", "so.minimize_scalar(f"+kw+")", "f", pyScalarF),
		ref:  "scipy.optimize.minimize_scalar",
		run: func() (int, int) {
			r := must(opt.MinimizeScalar(scalarF, m, o))
			sink = r.X
			return r.Nfev, r.Nit
		}}
}

func rows() []row {
	var rs []row
	f1 := func(f func([]float64) float64) opt.Problem { return opt.Problem{Func: f} }
	_ = f1
	rs = append(rs,
		mini("Minimize/BFGS/rosen2/grad", 2, opt.BFGS, "BFGS", ", jac=so.rosen_der", pf(rosen, rosenDer), nil),
		mini("Minimize/BFGS/rosen100/grad", 100, opt.BFGS, "BFGS", ", jac=so.rosen_der", pf(rosen, rosenDer), nil),
		mini("Minimize/LBFGSB/rosen2/grad", 2, opt.LBFGSB, "L-BFGS-B", ", jac=so.rosen_der", pf(rosen, rosenDer), nil),
		mini("Minimize/LBFGSB/rosen100/grad", 100, opt.LBFGSB, "L-BFGS-B", ", jac=so.rosen_der", pf(rosen, rosenDer), nil),
		mini("Minimize/BFGS/rosen2/fd", 2, opt.BFGS, "BFGS", "", f1(rosen), nil),
		mini("Minimize/BFGS/rosen100/fd", 100, opt.BFGS, "BFGS", "", f1(rosen), nil),
		mini("Minimize/LBFGSB/rosen100/fd", 100, opt.LBFGSB, "L-BFGS-B", "", f1(rosen), nil),
		mini("Minimize/NelderMead/rosen2", 2, opt.NelderMead, "Nelder-Mead", "", f1(rosen), nil),
		mini("Minimize/Powell/rosen2", 2, opt.Powell, "Powell", "", f1(rosen), nil),
		mini("Minimize/CG/rosen2/grad", 2, opt.CG, "CG", ", jac=so.rosen_der", pf(rosen, rosenDer), nil),
		mini("Minimize/CG/rosen100/grad", 100, opt.CG, "CG", ", jac=so.rosen_der", pf(rosen, rosenDer), nil),
		mini("Minimize/NewtonCG/rosen2/hess", 2, opt.NewtonCG, "Newton-CG", ", jac=so.rosen_der, hess=so.rosen_hess",
			opt.Problem{Func: rosen, Grad: rosenDer, Hess: rosenHess}, nil),
		mini("Minimize/NewtonCG/rosen100/hessp", 100, opt.NewtonCG, "Newton-CG", ", jac=so.rosen_der, hessp=so.rosen_hess_prod",
			opt.Problem{Func: rosen, Grad: rosenDer, HessP: rosenHessProd}, nil),
		mini("Minimize/SLSQP/rosen2/grad", 2, opt.SLSQP, "SLSQP", ", jac=so.rosen_der", pf(rosen, rosenDer), nil),
		mini("Minimize/TrustConstr/rosen2/hess", 2, opt.TrustConstr, "trust-constr", ", jac=so.rosen_der, hess=so.rosen_hess",
			opt.Problem{Func: rosen, Grad: rosenDer, Hess: rosenHess}, nil),
	)

	// Least squares and curve fitting.
	x0r := []float64{-1.2, 1}
	lsq := func(name string, m opt.LSMethod, scipy string) row {
		return row{name: name, fn: "LeastSquares", setup: "[np.array([-1.2, 1.0])]", data: []any{x0r},
			call: py("x0", "so.least_squares(r, x0, method='"+scipy+"')", "r", pyRosenResid),
			ref:  "scipy.optimize.least_squares(method='" + scipy + "')",
			run: func() (int, int) {
				r := must(opt.LeastSquares(opt.LSProblem{Func: rosenResid, M: 2}, x0r, m, nil))
				sink = r.Cost
				return r.Nfev, -1
			}}
	}
	rs = append(rs, lsq("LeastSquares/TRF/rosen", opt.TRF, "trf"), lsq("LeastSquares/LM/rosen", opt.LM, "lm"))
	rs = append(rs, row{name: "Leastsq/rosen", fn: "Leastsq", setup: "[np.array([-1.2, 1.0])]", data: []any{x0r},
		call: py("x0", "so.leastsq(r, x0, full_output=True)", "r", pyRosenResid), ref: "scipy.optimize.leastsq",
		run: func() (int, int) {
			r := must(opt.Leastsq(opt.LSProblem{Func: rosenResid, M: 2}, x0r, nil))
			sink = r.X[0]
			return r.Nfev, -1
		}})

	const m = 10000
	xs, ys := curveFitData(m)
	xa, ya := arr(xs), arr(ys)
	expf := func(x float64, p []float64) float64 { return float64(p[0]*math.Exp(float64(-p[1]*x))) + p[2] }
	cf := func(name string, o *opt.CurveFitOptions, p0 []float64, kw, p0s string) row {
		return row{name: name, fn: "CurveFit", n: m, setup: curveSetup, data: []any{xs, ys},
			call: py("x, y", "so.curve_fit(model, x, y, p0="+p0s+kw+"%FO%)", "model", pyModel), ref: "scipy.optimize.curve_fit",
			run: func() (int, int) {
				r := must(opt.CurveFit(expf, xa, ya, p0, o))
				sink = r.Popt[0]
				return r.Nfev, -1
			}}
	}
	rs = append(rs,
		cf("CurveFit/LM/1e4", &opt.CurveFitOptions{Method: opt.LM}, []float64{1, 1, 1}, ", method='lm'", "[1, 1, 1]"),
		cf("CurveFit/TRF/1e4", &opt.CurveFitOptions{Method: opt.TRF}, []float64{1, 1, 1}, ", method='trf'", "[1, 1, 1]"),
		cf("CurveFit/TRF/1e4/bounds", &opt.CurveFitOptions{Bounds: []opt.Bound{{Lo: 0, Hi: 10}, {Lo: 0, Hi: 3}, {Lo: -1, Hi: 1}}},
			[]float64{1, 1, 0}, ", bounds=([0, 0, -1], [10, 3, 1])", "[1, 1, 0]"),
	)
	model := opt.Model{
		F: func(x, p []float64) float64 { return expf(x[0], p) },
		Grad: func(x, p, g []float64) {
			e := math.Exp(float64(-p[1] * x[0]))
			g[0] = e
			g[1] = float64(float64(-p[0]*x[0]) * e)
			g[2] = 1
		},
	}
	rs = append(rs, row{name: "CurveFitModel/LM/1e4/jac", fn: "CurveFitModel", n: m, setup: curveSetup, data: []any{xs, ys},
		call: py("x, y", "so.curve_fit(model, x, y, p0=[1, 1, 1], method='lm', jac=jac%FO%)", "model", pyModel,
			"jac", "lambda x, a, b, c: np.column_stack((np.exp(-b * x), -a * x * np.exp(-b * x), np.ones_like(x)))"),
		ref: "scipy.optimize.curve_fit(jac=callable)",
		run: func() (int, int) {
			r := must(opt.CurveFitModel(model, xa, ya, []float64{1, 1, 1}, &opt.CurveFitOptions{Method: opt.LM}))
			sink = r.Popt[0]
			return r.Nfev, -1
		}})

	// Systems of equations.
	rootRow := func(name, fn, scipy, body string, n int, setup string, x0 []float64, f func(x, f []float64), pyf string) row {
		extra := 2 // SciPy's root counts the wrapper's initial and final evaluations (docs/perf.md)
		return row{name: name, fn: fn, n: n, setup: setup, data: []any{x0}, extra: extra,
			call: py("x0", body, "f", pyf), ref: scipy,
			run: func() (int, int) {
				var r *opt.RootResult
				if fn == "Fsolve" {
					r = must(opt.Fsolve(opt.SystemProblem{Func: f}, x0, nil))
				} else {
					r = must(opt.Root(opt.SystemProblem{Func: f}, x0, nil))
				}
				sink = r.X[0]
				return r.Nfev, -1
			}}
	}
	ones := make([]float64, 100)
	for i := range ones {
		ones[i] = -1
	}
	rs = append(rs,
		rootRow("Root/hybr/2", "Root", "scipy.optimize.root(method='hybr')", "so.root(f, x0, method='hybr')", 0, "[np.zeros(2)]", []float64{0, 0}, rootFun, pyRootFun),
		rootRow("Fsolve/2", "Fsolve", "scipy.optimize.fsolve", "so.fsolve(f, x0%FO%)", 0, "[np.zeros(2)]", []float64{0, 0}, rootFun, pyRootFun),
		rootRow("Root/hybr/broyden100", "Root", "scipy.optimize.root(method='hybr')", "so.root(f, x0, method='hybr')", 100, "[-np.ones(n)]", ones, broydenTri, pyBroyden),
	)

	// Linear programming (SciPy has HiGHS only; the Go side its own simplex and
	// interior point: the rows compare the same LP solved to optimality).
	lpc, lpA, lpb := lpDense(30, 50)
	lpAa := must(ndarray.FromData(append([]float64(nil), lpA...), 30, 50))
	lpba := arr(lpb)
	lp := func(name string, meth opt.LPMethod, scipy string) row {
		return row{name: name, fn: "Linprog", setup: lpSetup, data: []any{lpc, lpA, lpb},
			call: py("c, A, b", "so.linprog(c, A_ub=A, b_ub=b, method='"+scipy+"')"), ref: "scipy.optimize.linprog(method='" + scipy + "')",
			nocount: "different algorithms (own simplex / interior point against HiGHS)",
			run: func() (int, int) {
				r := must(opt.Linprog(lpc, opt.LPConstraints{AUb: lpAa, BUb: lpba}, &opt.LPOptions{Method: meth}))
				sink = r.Fun
				return -1, -1
			}}
	}
	small := must(ndarray.FromData([]float64{-3, 1, 1, 2}, 2, 2))
	smallb := arr([]float64{6, 4})
	rs = append(rs,
		lp("Linprog/simplex/30x50", opt.LPSimplex, "highs-ds"),
		lp("Linprog/interior/30x50", opt.LPInteriorPoint, "highs-ipm"),
		row{name: "Linprog/simplex/2", fn: "Linprog", setup: "[]", data: nil,
			call: py("", "so.linprog([-1, 4], A_ub=[[-3, 1], [1, 2]], b_ub=[6, 4], bounds=[(None, None), (-3, None)])"), ref: "scipy.optimize.linprog(highs)",
			nocount: "different algorithms (own simplex against HiGHS)",
			run: func() (int, int) {
				r := must(opt.Linprog([]float64{-1, 4}, opt.LPConstraints{AUb: small, BUb: smallb,
					Bounds: []opt.Bound{opt.Unbounded(), {Lo: -3, Hi: math.Inf(1)}}}, &opt.LPOptions{Method: opt.LPSimplex}))
				sink = r.Fun
				return -1, -1
			}},
	)

	// Scalar roots and minimisation.
	rs = append(rs,
		bracketRow("Brentq/cubic", "Brentq", "brentq", opt.Brentq),
		bracketRow("Brenth/cubic", "Brenth", "brenth", opt.Brenth),
		bracketRow("Ridder/cubic", "Ridder", "ridder", opt.Ridder),
		bracketRow("Bisect/cubic", "Bisect", "bisect", opt.Bisect),
		newtonRow("Newton/fprime", &opt.NewtonOptions{Fprime: func(x float64) float64 { return 3*x*x - 2 }}, ", fprime=f1"),
		newtonRow("Newton/secant", nil, ""),
		newtonRow("Newton/halley", &opt.NewtonOptions{Fprime: func(x float64) float64 { return 3*x*x - 2 },
			Fprime2: func(x float64) float64 { return 6 * x }}, ", fprime=f1, fprime2=f2"),
		scalarRow("RootScalar/brentq", "RootScalar", "scipy.optimize.root_scalar(method='brentq')",
			"so.root_scalar(f, method='brentq', bracket=[a, b])", func() *opt.RootScalarResult {
				return must(opt.RootScalar(cubic, "brentq", &opt.RootScalarOptions{Bracket: []float64{2, 3}}))
			}),
		scalarMin("MinimizeScalar/brent", opt.Brent, nil, ", method='brent'"),
		scalarMin("MinimizeScalar/golden", opt.Golden, nil, ", method='golden'"),
		scalarMin("MinimizeScalar/bounded", opt.Bounded, &opt.ScalarOptions{Bounds: &opt.Bound{Lo: -3, Hi: -1}}, ", bounds=(-3, -1), method='bounded'"),
		row{name: "Fminbound/cos", fn: "Fminbound", setup: "[3.0, 4.0]", data: []any{3.0, 4.0},
			call: py("a, b", "so.fminbound(f, a, b%FO%)", "f", `__import__("math").cos`), ref: "scipy.optimize.fminbound",
			run: func() (int, int) {
				r := must(opt.Fminbound(math.Cos, 3, 4, 0, 0))
				sink = r.X
				return r.Nfev, -1
			}},
		row{name: "Bracket/scalarF", fn: "Bracket", setup: "[0.0, 1.0]", data: []any{0.0, 1.0},
			call: py("a, b", "so.bracket(f, a, b)", "f", pyScalarF), ref: "scipy.optimize.bracket",
			run: func() (int, int) {
				x, _, nfev, err := opt.Bracket(scalarF, 0, 1)
				if err != nil {
					panic(err)
				}
				sink = x[0]
				return nfev, -1
			}},
	)

	// Grid search and derivative checks.
	rg := []opt.Range{{Lo: -4, Hi: 4}, {Lo: -4, Hi: 4}}
	brute := func(name string, ns int, finish bool) row {
		kw := ", finish=None"
		if finish {
			kw = ""
		}
		return row{name: name, fn: "Brute", setup: "[]", data: nil,
			call: py("", "so.brute(h, ((-4, 4), (-4, 4)), Ns="+strconv.Itoa(ns)+kw+")", "h", pyHimmelblau), ref: "scipy.optimize.brute",
			nocount: "brute reports no counts",
			run: func() (int, int) {
				r := must(opt.Brute(himmelblau2, rg, &opt.BruteOptions{Ns: ns, NoFinish: !finish}))
				sink = r.Fun
				return -1, -1
			}}
	}
	x100 := alt(100)
	rs = append(rs,
		brute("Brute/50x50/nofinish", 50, false),
		brute("Brute/20x20/finish", 20, true),
		row{name: "ApproxFprime/rosen100", fn: "ApproxFprime", n: 100, setup: altSetup, data: []any{x100},
			call: py("x", "so.approx_fprime(x, so.rosen)"), ref: "scipy.optimize.approx_fprime", nocount: "no solver",
			run: func() (int, int) { sink = opt.ApproxFprime(rosen, x100, 0)[0]; return -1, -1 }},
		row{name: "CheckGrad/rosen100", fn: "CheckGrad", n: 100, setup: altSetup, data: []any{x100},
			call: py("x", "so.check_grad(so.rosen, so.rosen_der, x)"), ref: "scipy.optimize.check_grad", nocount: "no solver",
			run: func() (int, int) { sink = opt.CheckGrad(rosen, rosenDer, x100, 0); return -1, -1 }},
	)

	// Global optimisation (the small rows of tools/globalbench; the large ones
	// take seconds and stay there).
	de := func(name, fname string, f func([]float64) float64, pyf string, dim int) row {
		return row{name: name, fn: "DifferentialEvolution", setup: "[]", data: nil,
			call: py("", "so.differential_evolution(f, [(-5, 5)] * "+strconv.Itoa(dim)+", rng=1)", "f", pyf),
			ref:  "scipy.optimize.differential_evolution(rng=1)",
			run: func() (int, int) {
				r := must(opt.DifferentialEvolution(f, box(dim, -5, 5), &opt.DEOptions{RNG: opt.NewRNG(1)}))
				sink = r.Fun
				return r.Nfev, r.Nit
			}}
	}
	_ = fmt.Sprint
	rs = append(rs,
		de("DifferentialEvolution/sphere/2", "sphere", sphere, `lambda x: float(np.sum(x * x))`, 2),
		de("DifferentialEvolution/rosenbrock/2", "rosenbrock", rosenbrockND,
			`lambda x: float(np.sum(100.0 * ((x[1:] - x[:-1] * x[:-1]) ** 2) + (1 - x[:-1]) ** 2))`, 2),
		row{name: "Basinhopping/himmelblau/100", fn: "Basinhopping", setup: "[]", data: nil,
			call: py("", "so.basinhopping(h, np.array([0.0, 0.0]), niter=100, rng=1, minimizer_kwargs=dict(method='L-BFGS-B', jac=True))",
				"h", `lambda v: (lambda x, y: ((x * x + y - 11.0) ** 2 + (x + y * y - 7.0) ** 2, np.array([4.0 * x * (x * x + y - 11.0) + 2.0 * (x + y * y - 7.0), 2.0 * (x * x + y - 11.0) + 4.0 * y * (x + y * y - 7.0)])))(v[0], v[1])`),
			ref: "scipy.optimize.basinhopping(rng=1)",
			run: func() (int, int) {
				r := must(opt.Basinhopping(opt.Problem{FuncGrad: himmelblauFG}, []float64{0, 0},
					&opt.BasinhoppingOptions{NIter: opt.Int(100), RNG: opt.NewRNG(1), Method: opt.LBFGSB}))
				sink = r.Fun
				return r.Nfev, r.Nit
			}},
		row{name: "Direct/rosenbrock/2", fn: "Direct", setup: "[]", data: nil,
			call: py("", "so.direct(f, [(-5, 5)] * 2)", "f", `lambda x: float(np.sum(100.0 * ((x[1:] - x[:-1] * x[:-1]) ** 2) + (1 - x[:-1]) ** 2))`),
			ref:  "scipy.optimize.direct",
			run: func() (int, int) {
				r := must(opt.Direct(rosen, box(2, -5, 5), nil))
				sink = r.Fun
				return r.Nfev, r.Nit
			}},
		row{name: "Direct/sphere/10", fn: "Direct", setup: "[]", data: nil,
			call: py("", "so.direct(f, [(-5, 5)] * 10)", "f", `lambda x: float(np.sum(x * x))`), ref: "scipy.optimize.direct",
			run: func() (int, int) {
				r := must(opt.Direct(sphere, box(10, -5, 5), nil))
				sink = r.Fun
				return r.Nfev, r.Nit
			}},
	)
	return rs
}

func timedCall(call string) string { return strings.ReplaceAll(call, "%FO%", "") }

// worker is the SciPy process, started once per test binary run that times.
func startWorker(t testing.TB) *refbench.Worker {
	t.Helper()
	uv, err := exec.LookPath("uv")
	if err != nil {
		home, _ := os.UserHomeDir()
		uv = filepath.Join(home, ".local", "bin", "uv")
	}
	refs, err := filepath.Abs("refs.txt")
	if err != nil {
		t.Fatal(err)
	}
	w, err := refbench.StartPython(context.Background(), uv, refs)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { w.Close() })
	return w
}

func cases(w *refbench.Worker) []refbench.Case {
	var out []refbench.Case
	for _, r := range rows() {
		var ref refbench.Timer
		if w != nil {
			ref = w.Timer(r.setup, timedCall(r.call), r.size())
		}
		out = append(out, refbench.Case{
			Name:      r.name,
			Func:      r.fn,
			Reference: r.ref + " (Python objective, one core)",
			Parity:    refbench.ParityNone, // nfev/nit: TestCounts; the results: the package's SciPy fixtures
			Go: func(iters int) {
				for ; iters > 0; iters-- {
					r.run()
				}
			},
			Ref: ref,
		})
	}
	return out
}

func running() bool {
	f := flag.Lookup("perf.run")
	return f != nil && f.Value.String() == "true"
}

func TestPerf(t *testing.T) {
	var w *refbench.Worker
	if running() {
		w = startWorker(t)
	}
	o := refbench.Options{Repo: "go-numerics/optimize"}
	if w != nil {
		v, err := w.RefVersions()
		if err != nil {
			t.Fatal(err)
		}
		o.RefVersions = v
	}
	rbtest.Perf(t, refbench.CIConfig{Root: "..", Cases: cases(w)}, o)
}

// pyRepr is the repr of a Python float (what the worker hashes for a scalar).
func pyRepr(v float64) string {
	s := strconv.FormatFloat(v, 'g', -1, 64)
	if !strings.ContainsAny(s, ".eEnN") {
		s += ".0"
	}
	return s
}

func digest(data []any) string {
	h := sha256.New()
	for _, d := range data {
		switch v := d.(type) {
		case []float64:
			for _, x := range v {
				var b [8]byte
				binary.LittleEndian.PutUint64(b[:], math.Float64bits(x))
				h.Write(b[:])
			}
		case float64:
			h.Write([]byte(pyRepr(v)))
		default:
			panic(fmt.Sprintf("unsupported data %T", d))
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

// TestDataParity checks, for every row, that the Go arguments and NumPy's are
// the same bytes: a ratio between two solves of different data means nothing.
func TestDataParity(t *testing.T) {
	if !*parityFlag {
		t.Skip("set -perf.parity (starts the SciPy worker through uv)")
	}
	w := startWorker(t)
	for _, r := range rows() {
		got, err := w.DataHash(r.setup, r.size())
		if err != nil {
			t.Fatalf("%s: %v", r.name, err)
		}
		if want := digest(r.data); got != want {
			t.Errorf("%s: NumPy data %s, Go data %s", r.name, got, want)
		}
	}
}

func TestRowsRun(t *testing.T) {
	seen := map[string]bool{}
	for _, r := range rows() {
		if seen[r.name] {
			t.Fatalf("duplicate row %s", r.name)
		}
		seen[r.name] = true
		r.run()
		if got := digest(r.data); len(got) != 64 {
			t.Fatal(got)
		}
	}
}

// divergent names the rows where nfev or nit differ from SciPy's, with the
// reason. A difference not listed here fails TestCounts: an optimiser that
// converges in a different number of evaluations is a different result, and a
// timing ratio on it compares two different computations.
var divergent = map[string]string{
	"Minimize/BFGS/rosen100/grad":      reduction,
	"Minimize/LBFGSB/rosen100/grad":    reduction,
	"Minimize/BFGS/rosen100/fd":        reduction,
	"Minimize/LBFGSB/rosen100/fd":      reduction + "; both stop at the 15000-evaluation limit",
	"Minimize/CG/rosen100/grad":        reduction,
	"Minimize/NewtonCG/rosen100/hessp": reduction,
}

// reduction: the n = 100 rows take different paths (counts within 11%) while
// every n = 2 row is equal. The cause is not isolated; it is consistent with the
// last bit of a length-100 reduction (NumPy sums pairwise and calls BLAS; the Go
// side sums in its own order), which Rosenbrock's narrow valley amplifies. A
// timing ratio on these rows compares two different iteration counts: read it
// per evaluation (docs/perf.md).
const reduction = "n = 100: different path, cause not isolated (suspected last-bit reduction order)"

// TestCounts runs every row once on both sides and compares nfev and nit (-1:
// not reported by that side or not comparable).
func TestCounts(t *testing.T) {
	if !*parityFlag {
		t.Skip("set -perf.parity (runs counts.py through uv)")
	}
	type req struct {
		Name, Setup, Call string
		N                 int
	}
	var reqs []req
	rs := rows()
	for _, r := range rs {
		if r.nocount == "" {
			reqs = append(reqs, req{r.name, r.setup, strings.ReplaceAll(r.call, "%FO%", ", full_output=True"), r.size()})
		}
	}
	in, _ := json.Marshal(reqs)
	uv, err := exec.LookPath("uv")
	if err != nil {
		home, _ := os.UserHomeDir()
		uv = filepath.Join(home, ".local", "bin", "uv")
	}
	refs, _ := filepath.Abs("refs.txt")
	cmd := exec.Command(uv, "run", "--no-project", "--python", "3.12.12", "--with-requirements", refs, "python", "-I", "counts.py")
	cmd.Stdin = bytes.NewReader(in)
	cmd.Env = append(os.Environ(), "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1")
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("counts.py: %v\n%s", err, stderr.String())
	}
	var got map[string]struct {
		Nfev, Nit *int
		Error     string
	}
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("%v\n%s", err, stdout.String())
	}
	for _, r := range rs {
		if r.nocount != "" {
			continue
		}
		ref := got[r.name]
		if ref.Error != "" {
			t.Errorf("%s: SciPy: %s", r.name, ref.Error)
			continue
		}
		nfev, nit := r.run()
		same := true
		line := fmt.Sprintf("%-40s nfev %6d / %-6s nit %5d / %-5s", r.name, nfev, ptr(ref.Nfev), nit, ptr(ref.Nit))
		if ref.Nfev != nil && nfev >= 0 && *ref.Nfev != nfev+r.extra {
			same = false
		}
		if ref.Nit != nil && nit >= 0 && *ref.Nit != nit {
			same = false
		}
		if same {
			t.Log("same      " + line)
		} else if why, ok := divergent[r.name]; ok {
			t.Log("DIVERGENT " + line + " (" + why + ")")
		} else {
			t.Error("DIFFERENT " + line)
		}
	}
}

func ptr(p *int) string {
	if p == nil {
		return "-"
	}
	return strconv.Itoa(*p)
}
