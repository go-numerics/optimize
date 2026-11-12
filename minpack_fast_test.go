package optimize

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/go-ndarray/ndarray"
)

func relClose(t *testing.T, what string, got, want []float64, tol float64) {
	t.Helper()
	scale := 1.0
	for _, w := range want {
		scale = math.Max(scale, math.Abs(w))
	}
	for i := range want {
		if d := math.Abs(got[i] - want[i]); !(d <= tol*scale) {
			t.Fatalf("%s[%d] = %g, want %g (tol %g)", what, i, got[i], want[i], tol*scale)
		}
	}
}

// TestFastEnormBound: fastEnorm against a 300-bit sum of squares over the
// magnitudes of TestAccuracyBoundEnorm, same bound (n/2 + 2 ulp); NaN and Inf
// propagate.
func TestFastEnormBound(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 9))
	for _, n := range []int{1, 3, 10, 100, 1000} {
		for _, e := range []int{0, -40, -600, -1060, 500, 1000} {
			for rep := 0; rep < 20; rep++ {
				v := make([]float64, n)
				for i := range v {
					v[i] = math.Ldexp(r.NormFloat64(), e)
				}
				if e := ulpErr(fastEnorm(v), bigNorm(v)); !(e <= float64(n)/2+2) {
					t.Fatalf("n=%d error %g ulp", n, e)
				}
			}
		}
	}
	if !math.IsNaN(fastEnorm([]float64{1, math.NaN()})) || !math.IsInf(fastEnorm([]float64{1, math.Inf(1)}), 1) {
		t.Fatal("NaN / Inf not propagated")
	}
}

func randMat(r *rand.Rand, m, n int) []float64 {
	a := make([]float64, m*n)
	for i := range a {
		a[i] = r.NormFloat64()
	}
	return a
}

// TestQrfacFastAgainstPlain: the factorisation, the column norms, the
// permutation and Qᵀ·qtb of qrfacFast against MINPACK's order, on random
// matrices with a zero column, a column of norm 1e-300 (the division
// fallback) and a nearly dependent one (the norm recomputation).
func TestQrfacFastAgainstPlain(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, m := range []int{40, 200} {
		for _, n := range []int{1, 3, 8, 30} {
			for _, pivot := range []bool{true, false} {
				a := randMat(r, m, n)
				if n >= 8 {
					for i := 0; i < m; i++ {
						a[i+m*2] = 0
						a[i+m*3] *= 1e-300
						a[i+m*5] = a[i+m*4] * (1 + 1e-9*r.NormFloat64())
					}
				}
				qtb := randMat(r, m, 1)
				run := func(f func(int, int, []float64, int, bool, []int, []float64, []float64, []float64, []float64)) (aa []float64, ip []int, rd, ac, qq []float64) {
					aa, qq = append([]float64(nil), a...), append([]float64(nil), qtb...)
					ip, rd, ac = make([]int, n), make([]float64, n), make([]float64, n)
					f(m, n, aa, m, pivot, ip, rd, ac, make([]float64, n), qq)
					return
				}
				wa, wip, wrd, wac, wqq := run(qrfacPlain)
				ga, gip, grd, gac, gqq := run(qrfacFast)
				if pivot {
					for j := range wip {
						if wip[j] != gip[j] {
							t.Fatalf("m=%d n=%d: pivot order differs: %v / %v", m, n, gip, wip)
						}
					}
				}
				tol := 1e-8
				if n >= 8 {
					tol = 1e-5 // the nearly dependent column amplifies rounding
				}
				relClose(t, "rdiag", grd, wrd, tol)
				relClose(t, "acnorm", gac, wac, 1e-12)
				relClose(t, "qtb", gqq, wqq, tol)
				if n < 8 {
					relClose(t, "a", ga, wa, 1e-9)
				}
			}
		}
	}
}

// TestQformFastAgainstPlain: the accumulated Q of qformFast against qformPlain.
func TestQformFastAgainstPlain(t *testing.T) {
	r := rand.New(rand.NewPCG(4, 5))
	for _, n := range []int{2, 40, 100} {
		a := randMat(r, n, n)
		if n > 10 {
			for i := 0; i < n; i++ {
				a[i+n*5] = 0 // a zero Householder vector
			}
		}
		rd, ac, wa := make([]float64, n), make([]float64, n), make([]float64, n)
		qrfacPlain(n, n, a, n, false, nil, rd, ac, wa, make([]float64, n))
		q1, q2 := append([]float64(nil), a...), append([]float64(nil), a...)
		for j := 1; j < n; j++ {
			for i := 0; i < j; i++ {
				q2[i+j*n] = 0
			}
		}
		qformPlain(n, q2, n, make([]float64, n))
		qformFast(n, q1, n, make([]float64, n))
		relClose(t, "q", q1, q2, 1e-10)
	}
}

// TestR1mpyqFastAgainstPlain: the Givens sweeps of r1mpyqFast against r1mpyq,
// with rotation parameters both inside and outside [-1, 1].
func TestR1mpyqFastAgainstPlain(t *testing.T) {
	r := rand.New(rand.NewPCG(6, 7))
	for _, m := range []int{5, 100} {
		n := m
		a := randMat(r, m, n)
		v, w := make([]float64, n), make([]float64, n)
		for i := range v {
			v[i], w[i] = 2*r.Float64()-1, 3*r.Float64()-1.5
		}
		a1, a2 := append([]float64(nil), a...), append([]float64(nil), a...)
		r1mpyqFast(m, n, a1, m, v, w, make([]float64, m))
		r1mpyq(m, n, a2, m, v, w)
		relClose(t, "a", a1, a2, 1e-10)
	}
}

// TestMinpackFastAgainstExact: Root, LeastSquares LM, Leastsq and CurveFit on
// problems with 32 or more residuals, default kernels against SciPyExact:
// the same solution within the accuracy contract (1e-6, the 1.49e-8 xtol
// times the conditioning), the evaluation counts logged.
func TestMinpackFastAgainstExact(t *testing.T) {
	broyden := func(x, f []float64) {
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
	for _, n := range []int{40, 100} {
		x0 := make([]float64, n)
		for i := range x0 {
			x0[i] = -1
		}
		e, _ := Root(SystemProblem{Func: broyden}, x0, &RootOptions{SciPyExact: true})
		f, _ := Root(SystemProblem{Func: broyden}, x0, nil)
		t.Logf("root n=%d: nfev exact %d fast %d", n, e.Nfev, f.Nfev)
		if !f.Success || maxDist(f.X, e.X) > 1e-6 {
			t.Errorf("root n=%d: fast %v exact %v", n, f.Status, e.Status)
		}
	}
	m := 300
	xa, ya := curveFitData(m)
	xs, _ := ndarray.Data[float64](xa)
	ys, _ := ndarray.Data[float64](ya)
	model := func(x float64, p []float64) float64 { return p[0]*math.Exp(-p[1]*x) + p[2] }
	var popt [2][]float64
	for k, exact := range []bool{true, false} {
		cr, err := CurveFit(model, xa, ya, []float64{1, 1, 1}, &CurveFitOptions{Method: LM, SciPyExact: exact})
		if err != nil {
			t.Fatal(err)
		}
		popt[k] = cr.Popt
		t.Logf("curve_fit LM m=%d exact=%v: nfev %d popt %v", m, exact, cr.Nfev, cr.Popt)
	}
	if d := maxDist(popt[1], popt[0]); d > 1e-6 {
		t.Errorf("curve_fit: |dp| = %g", d)
	}
	resid := func(x, f []float64) {
		for i := range f {
			f[i] = model(xs[i], x) - ys[i]
		}
	}
	p := LSProblem{M: m, Func: resid}
	l1, _ := LeastSquares(p, []float64{1, 1, 1}, LM, &LSOptions{SciPyExact: true})
	l2, _ := LeastSquares(p, []float64{1, 1, 1}, LM, nil)
	q1, _ := Leastsq(p, []float64{1, 1, 1}, &LeastsqOptions{SciPyExact: true})
	q2, _ := Leastsq(p, []float64{1, 1, 1}, nil)
	if maxDist(l1.X, l2.X) > 1e-6 || maxDist(q1.X, q2.X) > 1e-6 {
		t.Errorf("least squares: %v %v / %v %v", l1.X, l2.X, q1.X, q2.X)
	}
}
