package optimize

import (
	"math"
	"math/rand/v2"
	"testing"
)

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
