package optimize

import "github.com/go-simd/floats"

// trsmRUNFast is trsmRUN (X·U = B in place, U upper n×n) with every column's
// updates b_j -= u_kj·b_k applied in one pass on floats.AxpyRowsNoFMA: a block
// of b_j stays in registers across all k (SIMD), where the reference loop
// makes one memory pass per four coefficients. neg (length >= n) is scratch
// for the negated coefficients. Every product is still rounded before its
// subtraction and the k are taken in increasing order, so the values are the
// reference's; what differs is that zero coefficients are not skipped (a
// signed zero, or an Inf/NaN in B meeting a zero coefficient, may differ).
// Options.SciPyExact keeps the reference loop, which skips them.
func trsmRUNFast(m, n int, a []float64, lda int, b []float64, ldb int, neg []float64) {
	if m == 0 {
		return
	}
	for j := 0; j < n; j++ {
		bj := b[j*ldb : j*ldb+m]
		for k := 0; k < j; k++ {
			neg[k] = -a[k+j*lda]
		}
		if j > 0 {
			floats.AxpyRowsNoFMA(neg[:j], b[:(j-1)*ldb+m], ldb, bj) // columns < j only: no overlap with bj
		}
		floats.Scale(1/a[j+j*lda], bj)
	}
}
