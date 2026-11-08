package optimize

import (
	"math"
	"math/bits"
)

// ext64 is a non-negative number m·2^e with a 64-bit mantissa (m == 0, or the
// top bit of m set) and an unbounded exponent: the value an x87 register or a
// big.Float of precision 64 holds. Its operations round to nearest even at 64
// bits, which is what dnrm2 needs to reproduce OpenBLAS's x87 kernel.
type ext64 struct {
	m uint64
	e int
}

// round64 returns (hi·2^64 + lo + ε)·2^e rounded to 64 bits, where ε is a
// positive amount below one unit of lo when sticky is set. hi is not zero (a
// square of a 53-bit mantissa has 104 bits or more; a sum has the top bit of its
// larger operand).
func round64(hi, lo uint64, sticky bool, e int) ext64 {
	if lz := bits.LeadingZeros64(hi); lz > 0 {
		hi = hi<<lz | lo>>(64-lz)
		lo <<= lz
		e -= lz
	}
	if lo>>63 == 1 && (lo<<1 != 0 || sticky || hi&1 == 1) {
		if hi++; hi == 0 {
			hi = 1 << 63
			e++
		}
	}
	return ext64{hi, e + 64}
}

// square64 returns x·x rounded to 64 bits for a finite x (exactly: the 106-bit
// product, then one rounding).
func square64(x float64) ext64 {
	if x == 0 {
		return ext64{}
	}
	fr, ex := math.Frexp(math.Abs(x))
	m := uint64(fr * (1 << 53)) // exact: fr has at most 53 significant bits
	hi, lo := bits.Mul64(m, m)
	return round64(hi, lo, false, 2*(ex-53))
}

// add returns a+b rounded to 64 bits.
func (a ext64) add(b ext64) ext64 {
	if a.m == 0 {
		return b
	}
	if b.m == 0 {
		return a
	}
	if a.e < b.e {
		a, b = b, a
	}
	// b's mantissa, aligned under a's in a 128-bit window; what falls off the
	// window is a sticky bit.
	var bh, bl uint64
	sticky := false
	switch d := uint(a.e - b.e); {
	case d == 0:
		bh = b.m
	case d < 64:
		bh, bl = b.m>>d, b.m<<(64-d)
	case d < 128:
		bl = b.m >> (d - 64)
		sticky = b.m<<(128-d) != 0
	default:
		sticky = true
	}
	hi, carry := bits.Add64(a.m, bh, 0)
	lo := bl
	e := a.e - 64
	if carry == 1 {
		sticky = sticky || lo&1 == 1
		lo = lo>>1 | hi<<63
		hi = hi>>1 | 1<<63
		e++
	}
	return round64(hi, lo, sticky, e)
}

// u128 is an unsigned 128-bit integer.
type u128 struct{ hi, lo uint64 }

func (a u128) add(b u128) u128 {
	lo, c := bits.Add64(a.lo, b.lo, 0)
	hi, _ := bits.Add64(a.hi, b.hi, c)
	return u128{hi, lo}
}

func (a u128) sub(b u128) u128 {
	lo, c := bits.Sub64(a.lo, b.lo, 0)
	hi, _ := bits.Sub64(a.hi, b.hi, c)
	return u128{hi, lo}
}

func (a u128) geq(b u128) bool { return a.hi > b.hi || a.hi == b.hi && a.lo >= b.lo }

func (a u128) shr1() u128 { return u128{a.hi >> 1, a.lo>>1 | a.hi<<63} }

func (a u128) shr2() u128 { return u128{a.hi >> 2, a.lo>>2 | a.hi<<62} }

// isqrt128 returns floor(√M) and M - floor(√M)² for 2^126 <= M < 2^128 (the
// digit-by-digit square root: the root has 64 bits).
func isqrt128(m u128) (root uint64, rem u128) {
	var res u128
	bit := u128{hi: 1 << 62}
	for bit != (u128{}) {
		t := res.add(bit)
		if m.geq(t) {
			m = m.sub(t)
			res = res.shr1().add(bit)
		} else {
			res = res.shr1()
		}
		bit = bit.shr2()
	}
	return res.lo, m
}

// sqrtFloat64 returns the square root of a, rounded to 64 bits and then to 53,
// as big.Float's Sqrt at precision 64 followed by Float64 did, when the result
// is a normal float64; ok is false otherwise. The 64-bit root is the correctly
// rounded one (the root of a 64-bit number is never half way between two
// 64-bit numbers), which is also what an x87 FSQRT delivers.
func (a ext64) sqrtFloat64() (r float64, ok bool) {
	// a = m·2^e = M·2^e' with M = m·2^64 (e' = e-64, even) or m·2^63 (e' = e-63).
	var m u128
	e := a.e
	if e%2 == 0 {
		m = u128{a.m, 0}
		e -= 64
	} else {
		m = u128{a.m >> 1, a.m << 63}
		e -= 63
	}
	s, rem := isqrt128(m)
	// Round up when √M is above s + ½, that is M > s² + s (no tie exists).
	// s + 1 cannot wrap: at s = 2^64-1 (M = 2^128-2^64) the remainder is s itself.
	e /= 2
	if rem.hi != 0 || rem.lo > s {
		s++
	}
	// The value is s·2^e, 2^63 <= s < 2^64; round it to 53 bits (nearest even).
	if p := e + 63; p < -1021 || p > 1022 {
		return 0, false
	}
	q, low := s>>11, s&0x7ff
	if low > 0x400 || low == 0x400 && q&1 == 1 {
		q++ // may reach 2^53, which float64 holds exactly
	}
	return math.Ldexp(float64(q), e+11), true
}
