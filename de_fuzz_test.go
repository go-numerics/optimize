package optimize

import (
	"errors"
	"math"
	"testing"
)

// FuzzDifferentialEvolution drives DE with arbitrary bounds, sizes, mutation,
// recombination, strategy, updating and an objective whose values come from
// the fuzz bytes (NaN and +-Inf included). It asserts: no panic, only
// ErrInput/ErrBounds/ErrDimension errors, bounded evaluations and memory
// (a huge popsize is rejected), and a finite X inside the bounds.
func FuzzDifferentialEvolution(f *testing.F) {
	f.Add(uint8(2), uint8(3), uint8(0), uint8(0), uint8(0), uint8(10), 1.0, 0.5, 0.7, uint64(1), []byte{1, 2, 3, 200})
	f.Add(uint8(1), uint8(0), uint8(7), uint8(1), uint8(1), uint8(3), -3.0, 0.0, 1.0, uint64(9), []byte{0xff, 0xf8, 0x7f})
	f.Add(uint8(4), uint8(200), uint8(11), uint8(2), uint8(2), uint8(2), 0.0, 1.99, 0.0, uint64(3), []byte{})
	f.Add(uint8(3), uint8(255), uint8(5), uint8(1), uint8(0), uint8(255), 1e300, math.NaN(), math.Inf(1), uint64(0), []byte{0x7f, 0xf0})
	names := []DEStrategy{Best1Bin, Best1Exp, Rand1Bin, Rand1Exp, RandToBest1Bin, RandToBest1Exp,
		CurrentToBest1Bin, CurrentToBest1Exp, Best2Bin, Best2Exp, Rand2Bin, Rand2Exp}
	f.Fuzz(func(t *testing.T, n, pop, strat, upd, workers, maxiter uint8, lo, mut, cr float64, seed uint64, vals []byte) {
		dim := int(n)%5 + 1
		bounds := make([]Bound, dim)
		for i := range bounds {
			bounds[i] = Bound{lo + float64(i), lo + float64(i) + math.Abs(mut)*float64(i%2)*3}
		}
		popsize := int(pop)
		if pop == 255 {
			popsize = math.MaxInt // must be rejected, not allocated
		} else {
			popsize = popsize%12 + 1
		}
		o := &DEOptions{
			Strategy: names[int(strat)%len(names)], PopSize: popsize, Mutation: []float64{mut},
			Recombination: Float(cr), RNG: NewRNG(seed), NoPolish: true,
			MaxIter: Int(int(maxiter) % 8), Updating: DEUpdating(int(upd) % 3), Workers: int(workers)%3 + 1,
		}
		if strat%3 == 0 {
			o.Integrality = make([]bool, dim)
			o.Integrality[0] = true
		}
		if pop%2 == 1 && pop != 255 {
			o.Constraints = []ConstraintSpec{Constraint{Kind: Ineq, M: 1, Func: func(x, c []float64) {
				c[0] = x[0] - cr
			}}}
		}
		if upd%2 == 1 {
			o.RNG = NewLegacyRNG(uint32(seed))
		}
		calls := 0
		k := 0
		obj := func(x []float64) float64 {
			if o.Workers > 1 { // values must not depend on the order
				return deSphere(x)
			}
			calls++
			if len(vals) == 0 {
				return deSphere(x)
			}
			// a byte stream chooses between finite, NaN, +Inf, -Inf
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
			return deSphere(x)
		}
		r, err := DifferentialEvolution(obj, bounds, o)
		if err != nil {
			if !errors.Is(err, ErrInput) && !errors.Is(err, ErrBounds) && !errors.Is(err, ErrDimension) {
				t.Fatalf("unexpected error %v", err)
			}
			return
		}
		s := len(r.Population)
		// while every energy is infinite the initial block is evaluated again
		// at each generation (SciPy's behaviour): at most 2·S per generation
		if o.Workers == 1 && calls > (2**o.MaxIter+1)*s+1 {
			t.Fatalf("%d evaluations for %d generations of %d", calls, *o.MaxIter, s)
		}
		for i, x := range r.X {
			slack := 1e-9 * (math.Abs(bounds[i].Lo) + math.Abs(bounds[i].Hi) + 1)
			if !finite(x) || x < bounds[i].Lo-slack || x > bounds[i].Hi+slack {
				t.Fatalf("x[%d] = %v outside [%v, %v]", i, x, bounds[i].Lo, bounds[i].Hi)
			}
		}
	})
}
