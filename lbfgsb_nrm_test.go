package optimize

import (
	"math"
	"math/big"
	"math/rand/v2"
	"testing"
)

// dnrm2Big is the previous dnrm2, the oracle: big.Float at precision 64 for the
// squares, the sum and the root.
func dnrm2Big(v []float64) float64 {
	acc, sq := new(big.Float).SetPrec(64), new(big.Float).SetPrec(64)
	for _, x := range v {
		if math.IsInf(x, 0) || math.IsNaN(x) {
			return math.Sqrt(dot(v, v))
		}
		sq.SetFloat64(x)
		sq.Mul(sq, sq)
		acc.Add(acc, sq)
	}
	acc.Sqrt(acc)
	r, _ := acc.Float64()
	return r
}

func TestDnrm2MatchesBigFloat(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	s := &lbfgsb{}
	check := func(v []float64) {
		t.Helper()
		got, want := s.dnrm2(v), dnrm2Big(v)
		if math.Float64bits(got) != math.Float64bits(want) && !(math.IsNaN(got) && math.IsNaN(want)) {
			t.Fatalf("dnrm2(%v) = %v (%#x), big.Float gives %v (%#x)", v, got, math.Float64bits(got), want, math.Float64bits(want))
		}
	}
	check(nil)
	check([]float64{0, 0, -0.0})
	check([]float64{math.SmallestNonzeroFloat64})
	check([]float64{math.MaxFloat64, math.MaxFloat64}) // beyond double: the extended sum holds it, the root fits
	check([]float64{1, math.Inf(1)})
	check([]float64{1, math.NaN()})
	for range 20000 {
		n := 1 + rng.IntN(130)
		v := make([]float64, n)
		scale := math.Ldexp(1, rng.IntN(80)-40)
		for i := range v {
			switch rng.IntN(6) {
			case 0:
				v[i] = 0
			case 1: // wide range of magnitudes in one vector
				v[i] = rng.NormFloat64() * math.Ldexp(1, rng.IntN(300)-150)
			case 2: // subnormal and near-underflow
				v[i] = math.Float64frombits(rng.Uint64() & 0x001fffffffffffff)
			case 3: // exact tie candidates: few significant bits
				v[i] = float64(rng.IntN(1<<10)) * math.Ldexp(1, rng.IntN(20)-10)
			default:
				v[i] = rng.NormFloat64() * scale
			}
			if rng.IntN(2) == 0 {
				v[i] = -v[i]
			}
		}
		check(v)
	}
	// Equal exponents and cancellation-free carries: many identical entries.
	for n := 1; n <= 300; n++ {
		v := make([]float64, n)
		for i := range v {
			v[i] = 1.0000000000000002
		}
		check(v)
		for i := range v {
			v[i] = math.Nextafter(2, 1)
		}
		check(v)
	}
}

func TestDnrm2DoesNotAllocatePerElement(t *testing.T) {
	s := &lbfgsb{}
	v := altStart(100)
	if a := testing.AllocsPerRun(50, func() { s.dnrm2(v) }); a != 0 {
		t.Fatalf("dnrm2 on 100 elements allocates %v times, want 0 (was 105 with big.Float)", a)
	}
}

// TestExt64SqrtMatchesBigFloat compares the integer root and the two roundings
// with big.Float (precision 64 Sqrt, then Float64) on random mantissas and
// exponents of both parities, and on mantissas next to a rounding boundary.
func TestExt64SqrtMatchesBigFloat(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 5))
	acc, sq := new(big.Float).SetPrec(64), new(big.Float).SetPrec(64)
	n := 400000
	if testing.Short() {
		n = 20000
	}
	checked := 0
	for i := range n {
		a := ext64{m: rng.Uint64() | 1<<63, e: rng.IntN(1500) - 750}
		switch i % 4 {
		case 1:
			a.m = 1 << 63
		case 2:
			a.m = ^uint64(0) - uint64(rng.IntN(4))
		case 3: // few significant bits: sparse mantissas, exact and near-exact roots
			a.m = uint64(rng.Uint32()|1<<31) << 32
			a.e &^= 1
		}
		got, ok := a.sqrtFloat64()
		if !ok {
			continue
		}
		checked++
		acc.SetMantExp(sq.SetUint64(a.m), a.e)
		acc.Sqrt(acc)
		want, _ := acc.Float64()
		if math.Float64bits(got) != math.Float64bits(want) {
			t.Fatalf("sqrt(%#x·2^%d) = %v, big.Float gives %v", a.m, a.e, got, want)
		}
	}
	if checked < n/2 {
		t.Fatalf("only %d of %d inputs reached the integer path", checked, n)
	}
}

// TestExt64Carries pins the rounding that carries out of the mantissa: a tie
// whose upper neighbour is odd rounds up and the mantissa wraps to 2^63.
func TestExt64Carries(t *testing.T) {
	a := ext64{m: ^uint64(0), e: 0}
	b := ext64{m: 1 << 63, e: -64} // half a unit in the last place of a
	if got, want := a.add(b), (ext64{m: 1 << 63, e: 1}); got != want {
		t.Fatalf("(2^64-1) + 1/2 = %+v, want %+v", got, want)
	}
	if got := a.add(ext64{}); got != a {
		t.Fatalf("a + 0 = %+v", got)
	}
	if got := (ext64{}).add(a); got != a {
		t.Fatalf("0 + a = %+v", got)
	}
	// Equal operands: the sum carries out of 64 bits exactly.
	if got, want := a.add(a), (ext64{m: ^uint64(0), e: 1}); got != want {
		t.Fatalf("a + a = %+v, want %+v", got, want)
	}
	// An operand more than 2^128 times smaller is only a sticky bit: below half
	// a unit, the sum does not move.
	c := ext64{m: 1 << 63, e: -130}
	if got := (ext64{m: 1<<63 | 1, e: 0}).add(c); got != (ext64{m: 1<<63 | 1, e: 0}) {
		t.Fatalf("a + tiny = %+v", got)
	}
}
