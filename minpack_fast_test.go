package optimize

import (
	"math"
	"math/rand/v2"
	"testing"
)

func relClose(t *testing.T, what string, got, want []float64, tol float64) {
	t.Helper()
	scale := 1.0
	for _, w := range want {
		scale = math.Max(scale, math.Abs(w))
	}
	for i := range want {
		if d := math.Abs(got[i] - want[i]); !(d <= tol*scale) {
			t.Fatalf("%s[%d] = %g, want %g (tol %g)", what, i, got[i], want[i], tol*scale)
		}
	}
}

func randMat(r *rand.Rand, m, n int) []float64 {
	a := make([]float64, m*n)
	for i := range a {
		a[i] = r.NormFloat64()
	}
	return a
}

// TestQrfacFastAgainstPlain: the factorisation, the column norms, the
// permutation and Qᵀ·qtb of qrfacFast against MINPACK's order, on random
// matrices with a zero column, a column of norm 1e-300 (the division
// fallback) and a nearly dependent one (the norm recomputation).
func TestQrfacFastAgainstPlain(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, m := range []int{40, 200} {
		for _, n := range []int{1, 3, 8, 30} {
			for _, pivot := range []bool{true, false} {
				a := randMat(r, m, n)
				if n >= 8 {
					for i := 0; i < m; i++ {
						a[i+m*2] = 0
						a[i+m*3] *= 1e-300
						a[i+m*5] = a[i+m*4] * (1 + 1e-9*r.NormFloat64())
					}
				}
				qtb := randMat(r, m, 1)
				run := func(f func(int, int, []float64, int, bool, []int, []float64, []float64, []float64, []float64)) (aa []float64, ip []int, rd, ac, qq []float64) {
					aa, qq = append([]float64(nil), a...), append([]float64(nil), qtb...)
					ip, rd, ac = make([]int, n), make([]float64, n), make([]float64, n)
					f(m, n, aa, m, pivot, ip, rd, ac, make([]float64, n), qq)
					return
				}
				wa, wip, wrd, wac, wqq := run(qrfacPlain)
				ga, gip, grd, gac, gqq := run(qrfacFast)
				if pivot {
					for j := range wip {
						if wip[j] != gip[j] {
							t.Fatalf("m=%d n=%d: pivot order differs: %v / %v", m, n, gip, wip)
						}
					}
				}
				tol := 1e-8
				if n >= 8 {
					tol = 1e-5 // the nearly dependent column amplifies rounding
				}
				relClose(t, "rdiag", grd, wrd, tol)
				relClose(t, "acnorm", gac, wac, 1e-12)
				relClose(t, "qtb", gqq, wqq, tol)
				if n < 8 {
					relClose(t, "a", ga, wa, 1e-9)
				}
			}
		}
	}
}

// TestQformFastAgainstPlain: the accumulated Q of qformFast against qformPlain.
func TestQformFastAgainstPlain(t *testing.T) {
	r := rand.New(rand.NewPCG(4, 5))
	for _, n := range []int{2, 40, 100} {
		a := randMat(r, n, n)
		if n > 10 {
			for i := 0; i < n; i++ {
				a[i+n*5] = 0 // a zero Householder vector
			}
		}
		rd, ac, wa := make([]float64, n), make([]float64, n), make([]float64, n)
		qrfacPlain(n, n, a, n, false, nil, rd, ac, wa, make([]float64, n))
		q1, q2 := append([]float64(nil), a...), append([]float64(nil), a...)
		for j := 1; j < n; j++ {
			for i := 0; i < j; i++ {
				q2[i+j*n] = 0
			}
		}
		qformPlain(n, q2, n, make([]float64, n))
		qformFast(n, q1, n, make([]float64, n))
		relClose(t, "q", q1, q2, 1e-10)
	}
}

// TestR1mpyqFastAgainstPlain: the Givens sweeps of r1mpyqFast against r1mpyq,
// with rotation parameters both inside and outside [-1, 1].
func TestR1mpyqFastAgainstPlain(t *testing.T) {
	r := rand.New(rand.NewPCG(6, 7))
	for _, m := range []int{5, 100} {
		n := m
		a := randMat(r, m, n)
		v, w := make([]float64, n), make([]float64, n)
		for i := range v {
			v[i], w[i] = 2*r.Float64()-1, 3*r.Float64()-1.5
		}
		a1, a2 := append([]float64(nil), a...), append([]float64(nil), a...)
		r1mpyqFast(m, n, a1, m, v, w, make([]float64, m))
		r1mpyq(m, n, a2, m, v, w)
		relClose(t, "a", a1, a2, 1e-10)
	}
}
