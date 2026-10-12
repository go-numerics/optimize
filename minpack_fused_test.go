package optimize

import (
	"math"
	"math/rand/v2"
	"testing"
)

// The reference kernels below are MINPACK's qrfac and enorm as ported before
// the fused passes of minpack.go, followed by the Qᵀ·b loop that lmdif and
// hybrd ran after the factorisation. The fused kernels must reproduce them to
// the last bit, NaN payloads and signed zeros included.

func mpEnormRef(x []float64) float64 {
	var s1, s2, s3, x1max, x3max float64
	agiant := rgiant / float64(len(x))
	for _, xi := range x {
		xabs := math.Abs(xi)
		switch {
		case xabs > rdwarf && xabs < agiant:
			s2 += float64(xabs * xabs)
		case xabs > rdwarf:
			if xabs > x1max {
				t := x1max / xabs
				s1 = 1 + float64(s1*(t*t))
				x1max = xabs
			} else {
				t := xabs / x1max
				s1 += float64(t * t)
			}
		case xabs > x3max:
			t := x3max / xabs
			s3 = 1 + float64(s3*(t*t))
			x3max = xabs
		case xabs != 0:
			t := xabs / x3max
			s3 += float64(t * t)
		}
	}
	if s1 != 0 {
		return x1max * math.Sqrt(s1+(s2/x1max)/x1max)
	}
	if s2 != 0 {
		if s2 >= x3max {
			return math.Sqrt(s2 * (1 + float64((x3max/s2)*(x3max*s3))))
		}
		return math.Sqrt(x3max * ((s2 / x3max) + float64(x3max*s3)))
	}
	return x3max * math.Sqrt(s3)
}

func qrfacRef(m, n int, a []float64, lda int, pivot bool, ipvt []int, rdiag, acnorm, wa, qtb []float64) {
	for j := 0; j < n; j++ {
		acnorm[j] = mpEnormRef(a[j*lda : j*lda+m])
		rdiag[j] = acnorm[j]
		wa[j] = rdiag[j]
		if pivot {
			ipvt[j] = j
		}
	}
	minmn := min(m, n)
	for j := 0; j < minmn; j++ {
		if pivot {
			kmax := j
			for k := j; k < n; k++ {
				if rdiag[k] > rdiag[kmax] {
					kmax = k
				}
			}
			if kmax != j {
				for i := 0; i < m; i++ {
					a[i+lda*j], a[i+lda*kmax] = a[i+lda*kmax], a[i+lda*j]
				}
				rdiag[kmax] = rdiag[j]
				wa[kmax] = wa[j]
				ipvt[j], ipvt[kmax] = ipvt[kmax], ipvt[j]
			}
		}
		ajnorm := mpEnormRef(a[lda*j+j : lda*j+m])
		if ajnorm == 0 {
			rdiag[j] = -ajnorm
			continue
		}
		if a[lda*j+j] < 0 {
			ajnorm = -ajnorm
		}
		for i := j; i < m; i++ {
			a[i+lda*j] /= ajnorm
		}
		a[j+lda*j] += 1
		for k := j + 1; k < n; k++ {
			sum := 0.0
			for i := j; i < m; i++ {
				sum += float64(a[i+lda*j] * a[i+lda*k])
			}
			temp := sum / a[j+lda*j]
			for i := j; i < m; i++ {
				a[i+lda*k] -= float64(temp * a[i+lda*j])
			}
			if pivot && rdiag[k] != 0 {
				temp = a[j+lda*k] / rdiag[k]
				rdiag[k] *= math.Sqrt(math.Max(0, 1-float64(temp*temp)))
				if q := rdiag[k] / wa[k]; 0.05*(q*q) <= epsmch {
					rdiag[k] = mpEnormRef(a[(j+1)+lda*k : m+lda*k])
					wa[k] = rdiag[k]
				}
			}
		}
		rdiag[j] = -ajnorm
	}
	// lmdif's (and hybrd's) Qᵀ·b after the factorisation.
	for j := 0; j < minmn; j++ {
		if a[j+lda*j] != 0 {
			sum := 0.0
			for i := j; i < m; i++ {
				sum += float64(a[i+lda*j] * qtb[i])
			}
			temp := -sum / a[j+lda*j]
			for i := j; i < m; i++ {
				qtb[i] += float64(a[i+lda*j] * temp)
			}
		}
	}
}

// fusedEntry draws a matrix entry from every range mpEnorm distinguishes.
func fusedEntry(r *rand.Rand, m int) float64 {
	s := 1.0
	if r.IntN(2) == 0 {
		s = -1
	}
	switch r.IntN(12) {
	case 0:
		return 0
	case 1:
		return s * 1e-25 * r.Float64() // below rdwarf
	case 2:
		return s * 1e19 * float64(m) * (1 + r.Float64()) // above agiant
	case 3:
		return math.Copysign(0, s)
	}
	return s * r.Float64()
}

func sameBits(a, b []float64) bool {
	for i := range a {
		if math.Float64bits(a[i]) != math.Float64bits(b[i]) {
			return false
		}
	}
	return len(a) == len(b)
}

func TestQrfacMatchesReference(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	cases := 0
	for _, m := range []int{1, 2, 3, 4, 5, 8, 13, 40, 300} {
		for n := 1; n <= 10; n++ {
			for trial := range 12 {
				lda := m + trial%2
				a := make([]float64, lda*n)
				kind := trial % 4
				for j := range n {
					for i := range m {
						switch kind {
						case 0: // well scaled
							a[i+lda*j] = r.NormFloat64()
						case 1: // every magnitude range
							a[i+lda*j] = fusedEntry(r, m)
						case 2: // nearly dependent columns: norm downdates lose accuracy
							a[i+lda*j] = a[i] + 1e-9*r.NormFloat64()*float64(j)
							if j == 0 {
								a[i] = r.NormFloat64()
							}
						default: // a zero column and repeated columns
							if j%3 == 1 {
								a[i+lda*j] = 0
							} else {
								a[i+lda*j] = float64(i%3) + float64(j%2)
							}
						}
					}
				}
				if trial == 11 && m > 2 {
					a[1] = math.NaN()
				}
				b := make([]float64, m)
				for i := range b {
					b[i] = fusedEntry(r, m)
				}
				for _, pivot := range []bool{false, true} {
					a1, a2 := clone(a), clone(a)
					b1, b2 := clone(b), clone(b)
					p1, p2 := make([]int, n), make([]int, n)
					d1, d2 := make([]float64, n), make([]float64, n)
					c1, c2 := make([]float64, n), make([]float64, n)
					w1, w2 := make([]float64, n), make([]float64, n)
					qrfacRef(m, n, a1, lda, pivot, p1, d1, c1, w1, b1)
					qrfac(m, n, a2, lda, pivot, p2, d2, c2, w2, b2)
					if !sameBits(a1, a2) || !sameBits(b1, b2) || !sameBits(d1, d2) || !sameBits(c1, c2) || !sameBits(w1, w2) {
						t.Fatalf("m=%d n=%d trial=%d pivot=%v: fused qrfac differs from the reference", m, n, trial, pivot)
					}
					for j := range p1 {
						if p1[j] != p2[j] {
							t.Fatalf("m=%d n=%d trial=%d: ipvt %v, reference %v", m, n, trial, p2, p1)
						}
					}
					cases++
				}
			}
		}
	}
	t.Logf("%d factorisations identical to the reference", cases)
}

func TestMpEnormMatchesReference(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	for _, m := range []int{0, 1, 2, 3, 4, 5, 7, 64} {
		for range 50 {
			v := make([][]float64, 4)
			for k := range v {
				v[k] = make([]float64, m)
				for i := range v[k] {
					v[k][i] = fusedEntry(r, m)
				}
			}
			n0, n1, n2, n3 := mpEnorm4(v[0], v[1], v[2], v[3])
			got := []float64{mpEnorm(v[0]), n0, n1, n2, n3}
			want := []float64{mpEnormRef(v[0]), mpEnormRef(v[0]), mpEnormRef(v[1]), mpEnormRef(v[2]), mpEnormRef(v[3])}
			if !sameBits(got, want) {
				t.Fatalf("m=%d: %v, reference %v", m, got, want)
			}
			for nc := 1; nc <= 9; nc++ {
				a := make([]float64, nc*m)
				for i := range a {
					a[i] = fusedEntry(r, m)
				}
				out := make([]float64, nc)
				colEnorms(a, m, 0, m, out)
				for k := range out {
					if w := mpEnormRef(a[k*m : k*m+m]); math.Float64bits(w) != math.Float64bits(out[k]) {
						t.Fatalf("colEnorms m=%d column %d of %d: %v, reference %v", m, k, nc, out[k], w)
					}
				}
			}
		}
	}
}
