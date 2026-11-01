package optimize

import (
	"errors"
	"math"
	"testing"
)

// FuzzDirect drives Direct with arbitrary dimensions, bounds, tolerances,
// budgets (up to the largest the C code accepts) and an objective whose
// values come from the fuzz bytes, including NaN and ±Inf. A callback ends
// every run after a few iterations, so that the huge budgets are exercised
// without running for hours: what is checked is that nothing panics or
// allocates in proportion to the budget, that valid input never returns an
// error, that Nfev stays within the C code's slot limit, and that the result
// is a finite point inside the box.
func FuzzDirect(f *testing.F) {
	f.Add(uint8(2), -4.0, 4.0, 1e-4, int32(0), int32(0), false, 1e-4, []byte{4, 9, 1, 7, 3})
	f.Add(uint8(3), 0.0, 1.0, -1e-4, int32(1001), int32(5), true, 0.0, []byte{0, 1, 2, 3})
	f.Add(uint8(1), -1.0, 3.0, 0.0, int32(math.MaxInt32-2001), int32(math.MaxInt32-2001), false, 1.0, []byte{5})
	f.Add(uint8(12), -1e300, 1e300, 1e300, int32(1), int32(1), true, 0.5, []byte{})
	f.Add(uint8(7), 0.0, 1e-300, math.NaN(), int32(100), int32(100), false, 0.0, []byte{1, 1, 1, 1, 1, 1})
	f.Fuzz(func(t *testing.T, n uint8, lo, hi, eps float64, maxfev, maxiter int32, original bool, rtol float64, data []byte) {
		dim := int(n)%12 + 1
		calls := 0
		obj := func(x []float64) float64 {
			calls++
			if len(data) == 0 {
				return x[0]
			}
			b := data[(calls-1)%len(data)]
			switch b % 8 {
			case 0:
				return math.NaN()
			case 1:
				return math.Inf(1)
			case 2:
				return math.Inf(-1)
			case 3:
				return 0
			}
			return float64(b) - 100 + x[0]
		}
		bounds := make([]Bound, dim)
		for i := range bounds {
			bounds[i] = Bound{lo + float64(i), hi + float64(i)}
		}
		fm := -1.0
		o := &DirectOptions{Eps: eps, MaxFev: int(maxfev), MaxIter: int(maxiter), Original: original,
			FMin: &fm, FMinRtol: rtol}
		iters := 0
		o.Callback = func(xl, xb []float64, fb float64) bool {
			iters++
			return iters >= 40
		}
		r, err := Direct(obj, bounds, o)
		if err != nil {
			if !errors.Is(err, ErrInput) && !errors.Is(err, ErrBounds) {
				t.Fatalf("unexpected error %v", err)
			}
			if calls != 0 {
				t.Fatalf("%d objective calls before the error %v", calls, err)
			}
			return
		}
		if len(r.X) != dim || r.Message == "" {
			t.Fatalf("malformed result %+v", r)
		}
		for i, x := range r.X {
			if math.IsNaN(x) || math.IsInf(x, 0) || x < bounds[i].Lo || x > bounds[i].Hi {
				t.Fatalf("x[%d] = %v outside [%v, %v]", i, x, bounds[i].Lo, bounds[i].Hi)
			}
		}
		maxf := int64(maxfev)
		if maxf == 0 {
			maxf = int64(1000 * dim)
		}
		maxfunc, _, _, _ := dirSizes(dim, maxf, int64(maxiter))
		if int64(r.Nfev) > maxfunc || r.Nfev != calls {
			t.Fatalf("Nfev %d, calls %d, slot limit %d", r.Nfev, calls, maxfunc)
		}
		switch r.Status {
		case 1, 2, 3, 4, 5, -4, -6, -102:
		default:
			t.Fatalf("status %d", r.Status)
		}
	})
}
