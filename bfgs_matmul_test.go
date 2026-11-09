package optimize

import (
	"math"
	"math/rand/v2"
	"testing"
)

// matmulRef is BFGS's product as i-j-k dot products (each entry a
// left-to-right sum of rounded products), the form SciPy's update is
// matched against; matmul must give the same bits.
func matmulRef(c, a, b []float64, n int) {
	for i := range n {
		for j := range n {
			s := 0.0
			for k := range n {
				s += float64(a[i*n+k] * b[k*n+j])
			}
			c[i*n+j] = s
		}
	}
}

// TestNormInfMatchesMaxLoop: the infinity norm through floats.MaxAbs is
// numpy.linalg.norm(v, inf) = max(abs(v)): NaN as soon as one element is
// NaN. The loop of math.Max(m, |x|) it replaced returned +Inf for a vector
// holding both +Inf and NaN (math.Max(x, +Inf) is +Inf even for a NaN x);
// every other input gives the same bits.
func TestNormInfMatchesMaxLoop(t *testing.T) {
	ref := func(v []float64) float64 {
		m := 0.0
		for _, x := range v {
			if math.IsNaN(x) {
				return math.NaN()
			}
			m = math.Max(m, math.Abs(x))
		}
		return m
	}
	nz := math.Copysign(0, -1)
	for _, v := range [][]float64{nil, {nz}, {nz, 0}, {-3, 2, 1}, {1, math.Inf(-1)},
		{1, math.NaN(), 5}, {math.Inf(1), math.NaN()}, {-1e-310, 1e-320}} {
		got, want := norm(v, math.Inf(1)), ref(v)
		if math.Float64bits(got) != math.Float64bits(want) && !(math.IsNaN(got) && math.IsNaN(want)) {
			t.Errorf("norm(%v, inf) = %v, loop %v", v, got, want)
		}
	}
	r := rand.New(rand.NewPCG(3, 4))
	for n := range 300 {
		v := make([]float64, n)
		for i := range v {
			v[i] = (r.Float64() - 0.5) * math.Ldexp(1, r.IntN(80)-40)
		}
		if got, want := norm(v, math.Inf(1)), ref(v); got != want {
			t.Fatalf("n=%d: %v vs %v", n, got, want)
		}
	}
}

func TestMatmulSameBitsAsDotProducts(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, n := range []int{1, 2, 3, 5, 8, 17, 64, 100, 129} {
		for trial := range 4 {
			a, b := make([]float64, n*n), make([]float64, n*n)
			for i := range a {
				// Mixed magnitudes and signs, so that rounding and
				// cancellation depend on the order.
				a[i] = (r.Float64() - 0.5) * math.Ldexp(1, r.IntN(40)-20)
				b[i] = (r.Float64() - 0.5) * math.Ldexp(1, r.IntN(40)-20)
			}
			switch trial {
			case 1: // signed zeros: 0 + (−0) must stay +0 in both
				for i := range a {
					if i%3 == 0 {
						a[i] = math.Copysign(0, -1)
					}
				}
			case 2:
				a[0], b[n*n-1] = math.NaN(), math.Inf(1)
			}
			got, want := make([]float64, n*n), make([]float64, n*n)
			for i := range got {
				got[i] = 7 // matmul must overwrite, not accumulate
			}
			matmul(got, a, b, n)
			matmulRef(want, a, b, n)
			if !sameBits(got, want) {
				t.Fatalf("n=%d trial %d: matmul differs from the dot-product order", n, trial)
			}
		}
	}
}
