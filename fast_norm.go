package optimize

import (
	"math"

	"github.com/go-simd/floats"
)

// fastEnorm is the Euclidean norm as sqrt(x·x) on an FMA dot (go-simd/floats,
// lane-blocked sum), the default norm of L-BFGS-B and of the MINPACK solvers
// from 32 rows. It falls back to the scaled sum enorm when the square could
// overflow or lose its leading bits to underflow (norm outside 1e-145 ..
// 1e150). Within n/2 + 2 ulp of the exact norm (TestFastEnormBound); NaN
// propagates and an infinite component gives +Inf.
func fastEnorm(x []float64) float64 {
	r := math.Sqrt(floats.Dot(x, x))
	if math.IsNaN(r) || r > 1e-145 && r < 1e150 {
		return r
	}
	return enorm(x)
}
