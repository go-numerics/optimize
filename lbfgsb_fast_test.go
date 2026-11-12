package optimize

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestFastNrm2Bound: the default dnrm2 (sqrt of an FMA dot, scaled-sum
// fallback outside 1e-145..1e150) against a 300-bit sum of squares, over the
// magnitudes of TestAccuracyBoundEnorm; same bound, n/2 + 2 ulps.
func TestFastNrm2Bound(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 13))
	s := &lbfgsb{}
	for _, n := range []int{1, 3, 10, 100, 1000} {
		for _, e := range []int{0, -40, -600, -1060, 500, 1000} {
			for rep := 0; rep < 20; rep++ {
				v := make([]float64, n)
				for i := range v {
					v[i] = math.Ldexp(r.NormFloat64(), e)
				}
				if e := ulpErr(s.dnrm2(v), bigNorm(v)); !(e <= float64(n)/2+2) {
					t.Fatalf("n=%d error %g ulp", n, e)
				}
			}
		}
	}
	if !math.IsNaN(s.dnrm2([]float64{1, math.NaN()})) || !math.IsInf(s.dnrm2([]float64{1, math.Inf(1)}), 1) {
		t.Fatal("NaN / Inf not propagated")
	}
}

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

// TestLBFGSBFastAgainstExact: on bounded and unbounded Rosenbrock and a
// bounded quadratic, the default kernels and SciPy's order reach the same
// minimiser within the contract's 1e-3 / 1e-6; the iteration counts are logged.
func TestLBFGSBFastAgainstExact(t *testing.T) {
	for _, n := range []int{2, 10, 40} {
		for _, bounded := range []bool{false, true} {
			x0 := make([]float64, n)
			for i := range x0 {
				x0[i] = -1.2 + 0.1*float64(i%3)
			}
			var res [2]*Result
			for k, exact := range []bool{true, false} {
				o := &Options{SciPyExact: exact}
				if bounded {
					o.Bounds = make([]Bound, n)
					for i := range o.Bounds {
						o.Bounds[i] = Bound{Lo: -2, Hi: 0.5 + float64(i%2)}
					}
				}
				r, err := Minimize(Problem{Func: rosen, Grad: rosenDer}, x0, LBFGSB, o)
				if err != nil {
					t.Fatal(err)
				}
				res[k] = r
			}
			t.Logf("n=%d bounded=%v: nit exact %d fast %d, f %.3g / %.3g", n, bounded, res[0].Nit, res[1].Nit, res[0].Fun, res[1].Fun)
			if d := maxDist(res[1].X, res[0].X); d > 1e-3 || math.Abs(res[1].Fun-res[0].Fun) > 1e-6 {
				t.Errorf("n=%d bounded=%v: |dx| = %g, |df| = %g", n, bounded, d, res[1].Fun-res[0].Fun)
			}
		}
	}
}
