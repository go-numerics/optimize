package optimize

import (
	"fmt"

	"github.com/go-ndarray/ndarray/random"
)

// RandomSource is the random stream of a stochastic solver. Exactly one of a
// NumPy Generator or a legacy RandomState is behind it, so the solvers can
// reproduce SciPy's draws for both seeding forms (SciPy's rng=int and
// seed=int). A RandomSource is not safe for concurrent use; the solvers draw
// only on the calling goroutine.
type RandomSource struct {
	g *random.Generator
	r *random.RandomState
}

// NewRNG is SciPy's rng=seed: Generator(PCG64(SeedSequence(seed))).
func NewRNG(seed uint64) *RandomSource {
	return &RandomSource{g: random.DefaultRNG(seed)}
}

// NewLegacyRNG is SciPy's seed=seed: RandomState(seed), MT19937 with the
// legacy seeding.
func NewLegacyRNG(seed uint32) *RandomSource {
	return &RandomSource{r: random.NewRandomState(seed)}
}

// RNGFromGenerator wraps a NumPy-compatible Generator (a nil g is an
// entropy-seeded PCG64).
func RNGFromGenerator(g *random.Generator) *RandomSource {
	if g == nil {
		return newEntropyRNG()
	}
	return &RandomSource{g: g}
}

// RNGFromRandomState wraps a legacy RandomState (a nil r is an
// entropy-seeded one).
func RNGFromRandomState(r *random.RandomState) *RandomSource {
	if r == nil {
		return &RandomSource{r: random.NewRandomStateEntropy()}
	}
	return &RandomSource{r: r}
}

// newEntropyRNG is the stream used when the options carry no RandomSource:
// an entropy-seeded PCG64, not reproducible.
func newEntropyRNG() *RandomSource {
	return &RandomSource{g: random.DefaultRNG()}
}

// orEntropy returns s, or an entropy-seeded source when s is nil.
func (s *RandomSource) orEntropy() *RandomSource {
	if s == nil || (s.g == nil && s.r == nil) {
		return newEntropyRNG()
	}
	return s
}

// Float64 is next_double: one draw in [0, 1).
func (s *RandomSource) Float64() float64 {
	if s.g != nil {
		return s.g.Float64()
	}
	return s.r.Float64()
}

// Uniform is uniform(low, high) with size=None: low + (high-low)·U, the
// product rounded before the sum.
func (s *RandomSource) Uniform(low, high float64) float64 {
	return float64((high-low)*s.Float64()) + low
}

// FillUniform is uniform(low, high, size=len(dst)), in order. It draws
// nothing and returns an error when high-low is NaN or infinite (and, for a
// Generator, negative).
func (s *RandomSource) FillUniform(dst []float64, low, high float64) error {
	var err error
	if s.g != nil {
		err = s.g.FillUniform(dst, low, high)
	} else {
		err = s.r.FillUniform(dst, low, high)
	}
	if err != nil {
		return fmt.Errorf("%w: %v", ErrInput, err)
	}
	return nil
}

// fillUnit is uniform(size=len(dst)): the range [0, 1) is always valid, so
// the error of FillUniform cannot happen.
func (s *RandomSource) fillUnit(dst []float64) {
	if s.g != nil {
		_ = s.g.FillUniform(dst, 0, 1)
		return
	}
	_ = s.r.FillUniform(dst, 0, 1)
}

// fillBelow is integers(0, high, size=len(dst)) for a high >= 1.
func (s *RandomSource) fillBelow(dst []int64, high int64) {
	if s.g != nil {
		_ = s.g.FillIntN(dst, high)
		return
	}
	_ = s.r.FillIntN(dst, high)
}

// IntN is integers(0, high) with size=None; it panics when high <= 0.
func (s *RandomSource) IntN(high int64) int64 {
	if s.g != nil {
		return s.g.IntN(high)
	}
	return s.r.IntN(high)
}

// FillIntN is integers(0, high, size=len(dst)).
func (s *RandomSource) FillIntN(dst []int64, high int64) error {
	var err error
	if s.g != nil {
		err = s.g.FillIntN(dst, high)
	} else {
		err = s.r.FillIntN(dst, high)
	}
	if err != nil {
		return fmt.Errorf("%w: %v", ErrInput, err)
	}
	return nil
}

// ShuffleInt64 is shuffle of a one-dimensional int64 array.
func (s *RandomSource) ShuffleInt64(d []int64) {
	if s.g != nil {
		s.g.ShuffleInt64(d)
		return
	}
	s.r.ShuffleInt64(d)
}
