package optimize

import (
	"fmt"
	"math"
	"math/big"
	"math/rand"
	"testing"
)

// Candidates for the norm of the MINPACK ports, measured against mpEnorm.

// enormPlain1 is a plain left-to-right sum of squares (the same bits as
// mpEnorm whenever every component is in (rdwarf, agiant)), falling back to
// mpEnorm when the sum cannot be trusted (overflow, or squares that underflow).
func enormPlain1(x []float64) float64 {
	s := 0.0
	for _, xi := range x {
		s += float64(xi * xi)
	}
	if s > 1e-250 && s < 1e290 {
		return math.Sqrt(s)
	}
	return mpEnorm(x)
}

// enormPlain4 uses four independent accumulators.
func enormPlain4(x []float64) float64 {
	var s0, s1, s2, s3 float64
	i := 0
	for ; i+4 <= len(x); i += 4 {
		s0 += float64(x[i] * x[i])
		s1 += float64(x[i+1] * x[i+1])
		s2 += float64(x[i+2] * x[i+2])
		s3 += float64(x[i+3] * x[i+3])
	}
	for ; i < len(x); i++ {
		s0 += float64(x[i] * x[i])
	}
	s := (s0 + s1) + (s2 + s3)
	if s > 1e-250 && s < 1e290 {
		return math.Sqrt(s)
	}
	return mpEnorm(x)
}

// enormTwoPass scales by the largest magnitude first.
func enormTwoPass(x []float64) float64 {
	m := 0.0
	for _, xi := range x {
		if a := math.Abs(xi); a > m {
			m = a
		}
	}
	if m == 0 || math.IsInf(m, 0) || m != m {
		return m
	}
	inv := 1 / m
	s := 0.0
	for _, xi := range x {
		t := xi * inv
		s += float64(t * t)
	}
	return m * math.Sqrt(s)
}

// bigNorm is the oracle: sum of squares at 300 bits, square root, rounded.
func bigNorm(x []float64) *big.Float {
	acc := new(big.Float).SetPrec(300)
	sq := new(big.Float).SetPrec(300)
	for _, v := range x {
		sq.SetFloat64(v)
		sq.Mul(sq, sq)
		acc.Add(acc, sq)
	}
	return acc.Sqrt(acc)
}

// ulpErr is |got - want| in units of the spacing of float64 at want.
func ulpErr(got float64, want *big.Float) float64 {
	w, _ := want.Float64()
	if w == 0 || math.IsInf(w, 0) {
		if got == w {
			return 0
		}
		return math.Inf(1)
	}
	if math.IsInf(got, 0) || got != got {
		return math.Inf(1)
	}
	ulp := math.Nextafter(math.Abs(w), math.Inf(1)) - math.Abs(w)
	if ulp == 0 || math.IsInf(ulp, 0) {
		ulp = math.SmallestNonzeroFloat64
	}
	d := new(big.Float).SetPrec(300).SetFloat64(got)
	d.Sub(d, want)
	d.Abs(d)
	r, _ := new(big.Float).SetPrec(300).Quo(d, new(big.Float).SetFloat64(ulp)).Float64()
	return r
}

type enormCase struct {
	name string
	gen  func(r *rand.Rand, n int) []float64
}

func enormCases() []enormCase {
	scaled := func(e int) func(r *rand.Rand, n int) []float64 {
		return func(r *rand.Rand, n int) []float64 {
			v := make([]float64, n)
			for i := range v {
				v[i] = math.Ldexp(r.NormFloat64(), e)
			}
			return v
		}
	}
	return []enormCase{
		{"1", scaled(0)},
		{"2^-40", scaled(-40)},
		{"2^-600", scaled(-600)},
		{"2^-1060 (subnormal)", scaled(-1060)},
		{"2^500", scaled(500)},
		{"2^1020", scaled(1020)},
		{"mixed 2^-600..2^500", func(r *rand.Rand, n int) []float64 {
			v := make([]float64, n)
			for i := range v {
				v[i] = math.Ldexp(r.NormFloat64(), r.Intn(1100)-600)
			}
			return v
		}},
	}
}

// TestEnormVariantsError prints the worst error in ulps of each candidate.
func TestEnormVariantsError(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	fns := []struct {
		name string
		f    func([]float64) float64
	}{{"mpEnorm", mpEnorm}, {"enorm", enorm}, {"plain1", enormPlain1}, {"plain4", enormPlain4}, {"twopass", enormTwoPass}}
	for _, c := range enormCases() {
		for _, n := range []int{3, 10, 100, 1000} {
			worst := make([]float64, len(fns))
			for rep := 0; rep < 200; rep++ {
				v := c.gen(r, n)
				want := bigNorm(v)
				for k, fn := range fns {
					if e := ulpErr(fn.f(v), want); e > worst[k] {
						worst[k] = e
					}
				}
			}
			line := fmt.Sprintf("%-22s n=%-5d", c.name, n)
			for k, fn := range fns {
				line += fmt.Sprintf(" %s=%.2f", fn.name, worst[k])
			}
			t.Log(line)
		}
	}
}

func benchEnorm(b *testing.B, f func([]float64) float64, n int) {
	r := rand.New(rand.NewSource(2))
	v := make([]float64, n)
	for i := range v {
		v[i] = r.NormFloat64()
	}
	b.ResetTimer()
	sink := 0.0
	for i := 0; i < b.N; i++ {
		sink += f(v)
	}
	_ = sink
}

func BenchmarkEnormVariants(b *testing.B) {
	for _, n := range []int{4, 16, 100, 1000} {
		for _, fn := range []struct {
			name string
			f    func([]float64) float64
		}{{"mpEnorm", mpEnorm}, {"plain1", enormPlain1}, {"plain4", enormPlain4}, {"twopass", enormTwoPass}} {
			b.Run(fmt.Sprintf("%s/n=%d", fn.name, n), func(b *testing.B) { benchEnorm(b, fn.f, n) })
		}
	}
}
