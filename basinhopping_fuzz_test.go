package optimize

import (
	"errors"
	"math"
	"testing"
)

// FuzzBasinhopping drives basin hopping with arbitrary step size, temperature,
// iteration count, interval and an objective whose values come from the fuzz
// bytes (NaN and +-Inf included). It asserts: no panic, only the package's
// input errors, and bounded work (at most NIter+1 local minimisations).
func FuzzBasinhopping(f *testing.F) {
	f.Add(0.5, 1.0, uint8(3), uint8(2), uint8(0), uint64(1), []byte{0, 1, 2, 3})
	f.Add(math.NaN(), 0.0, uint8(5), uint8(0), uint8(1), uint64(2), []byte{1})
	f.Add(1e300, -1.0, uint8(2), uint8(255), uint8(2), uint64(3), []byte{})
	f.Add(0.0, math.Inf(1), uint8(4), uint8(1), uint8(3), uint64(4), []byte{2, 3, 2})
	f.Fuzz(func(t *testing.T, stepsize, temp float64, niter, interval, variant uint8, seed uint64, vals []byte) {
		calls, k := 0, 0
		obj := func(x, g []float64) float64 {
			calls++
			fv := bhQuad(x, g)
			if len(vals) == 0 {
				return fv
			}
			v := vals[k%len(vals)]
			k++
			switch v & 3 {
			case 1:
				return math.NaN()
			case 2:
				return math.Inf(1)
			case 3:
				return math.Inf(-1)
			}
			return fv
		}
		o := &BasinhoppingOptions{
			NIter: Int(int(niter) % 6), T: Float(temp), StepSize: stepsize, Interval: int(interval) % 8,
			RNG: NewRNG(seed), MinimizeOptions: &Options{MaxIter: 20},
		}
		if variant%2 == 1 {
			o.RNG = NewLegacyRNG(uint32(seed))
		}
		if variant%4 >= 2 {
			o.Method = NelderMead
		}
		r, err := Basinhopping(Problem{FuncGrad: obj}, []float64{1, -1}, o)
		if err != nil {
			if !errors.Is(err, ErrInput) && !errors.Is(err, ErrMethod) && !errors.Is(err, ErrBounds) && !errors.Is(err, ErrDimension) {
				t.Fatalf("unexpected error %v", err)
			}
			return
		}
		if r.Nit > *o.NIter+1 || r.Lowest == nil || len(r.X) != 2 {
			t.Fatalf("%+v", r)
		}
	})
}
