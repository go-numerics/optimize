package optimize

import (
	"math"

	"github.com/go-simd/floats"
)

// The default MINPACK kernels for vectors of fuseMin or more components. They
// are the same Householder and Givens algebra as qrfac, qform and r1mpyq, with
// every sum on go-simd/floats (FMA, lane-blocked order) instead of MINPACK's
// sequential chain of rounded products. The results differ from MINPACK's in
// the last bits, within the accuracy contract; the SciPy-exact kernels
// (minpack.go) run when SciPyExact is set and for shorter vectors, where the
// call overhead of the kernels exceeds their gain.

// fastEnorm is the Euclidean norm as sqrt(x·x) on an FMA dot, falling back to
// MINPACK's three-accumulator enorm when the square could overflow or lose its
// leading bits to underflow (norm outside 1e-145 .. 1e150).
func fastEnorm(x []float64) float64 {
	r := math.Sqrt(floats.Dot(x, x))
	if math.IsNaN(r) || r > 1e-145 && r < 1e150 {
		return r
	}
	return mpEnorm(x)
}

// qrfacFast is qrfac (including the Qᵀ·qtb product) on floats kernels.
func qrfacFast(m, n int, a []float64, lda int, pivot bool, ipvt []int, rdiag, acnorm, wa, qtb []float64) {
	for j := 0; j < n; j++ {
		acnorm[j] = fastEnorm(a[j*lda : j*lda+m])
		rdiag[j] = acnorm[j]
		wa[j] = rdiag[j]
		if pivot {
			ipvt[j] = j
		}
	}
	for j := 0; j < min(m, n); j++ {
		if pivot {
			kmax := j
			for k := j; k < n; k++ {
				if rdiag[k] > rdiag[kmax] {
					kmax = k
				}
			}
			if kmax != j {
				cj, ck := a[lda*j:lda*j+m], a[lda*kmax:lda*kmax+m]
				for i := range cj {
					cj[i], ck[i] = ck[i], cj[i]
				}
				rdiag[kmax] = rdiag[j]
				wa[kmax] = wa[j]
				ipvt[j], ipvt[kmax] = ipvt[kmax], ipvt[j]
			}
		}
		v := a[lda*j+j : lda*j+m]
		ajnorm := fastEnorm(v)
		if ajnorm == 0 {
			rdiag[j] = -ajnorm
			continue
		}
		if v[0] < 0 {
			ajnorm = -ajnorm
		}
		if inv := 1 / ajnorm; math.Abs(inv) > 1e-290 && math.Abs(inv) < 1e290 {
			floats.Scale(inv, v)
		} else {
			for i := range v {
				v[i] /= ajnorm
			}
		}
		v[0]++
		for k := j + 1; k < n; k++ {
			ck := a[lda*k+j : lda*k+m]
			floats.Axpy(-floats.Dot(v, ck)/v[0], v, ck)
			if pivot && rdiag[k] != 0 {
				temp := a[j+lda*k] / rdiag[k]
				rdiag[k] *= math.Sqrt(math.Max(0, 1-temp*temp))
				if q := rdiag[k] / wa[k]; 0.05*(q*q) <= epsmch {
					rdiag[k] = fastEnorm(a[(j+1)+lda*k : m+lda*k])
					wa[k] = rdiag[k]
				}
			}
		}
		rdiag[j] = -ajnorm
	}
	for j := 0; j < min(m, n); j++ {
		if v := a[lda*j+j : lda*j+m]; v[0] != 0 {
			floats.Axpy(-floats.Dot(v, qtb[j:m])/v[0], v, qtb[j:m])
		}
	}
}

// qformFast is qform on floats kernels.
func qformFast(n int, q []float64, ldq int, wa []float64) {
	for j := 1; j < n; j++ {
		for i := 0; i < j; i++ {
			q[i+j*ldq] = 0
		}
	}
	for k := n - 1; k >= 0; k-- {
		for i := k; i < n; i++ {
			wa[i] = q[i+ldq*k]
			q[i+ldq*k] = 0
		}
		q[k+ldq*k] = 1
		if wa[k] == 0 {
			continue
		}
		v := wa[k:n]
		for j := k; j < n; j++ {
			cj := q[k+ldq*j : n+ldq*j]
			floats.Axpy(-floats.Dot(cj, v)/v[0], v, cj)
		}
	}
}

// colDotsFast sets sums[j] = a_j · v for the n columns of the n×n matrix a.
func colDotsFast(a []float64, lda int, v, sums []float64) {
	for j := range v {
		sums[j] = floats.Dot(a[lda*j:lda*j+len(v)], v)
	}
}

// r1mpyqFast is r1mpyq for an m×n matrix with rotations as FMA kernels on
// whole columns; scratch has length m.
func r1mpyqFast(m, n int, a []float64, lda int, v, w, scratch []float64) {
	rot := func(s float64) (cos, sin float64) {
		if math.Abs(s) > 1 {
			cos = 1 / s
			return cos, math.Sqrt(1 - cos*cos)
		}
		return math.Sqrt(1 - s*s), s
	}
	last := a[(n-1)*lda : (n-1)*lda+m]
	for j := n - 2; j >= 0; j-- {
		cos, sin := rot(v[j])
		cj := a[j*lda : j*lda+m]
		floats.Axpby3(scratch, cos, cj, -sin, last, 0, last)
		floats.Axpby3(last, sin, cj, cos, last, 0, last)
		copy(cj, scratch)
	}
	for j := 0; j < n-1; j++ {
		cos, sin := rot(w[j])
		cj := a[j*lda : j*lda+m]
		floats.Axpby3(scratch, cos, cj, sin, last, 0, last)
		floats.Axpby3(last, -sin, cj, cos, last, 0, last)
		copy(cj, scratch)
	}
}
