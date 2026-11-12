package optimize

import (
	"math"

	"github.com/go-simd/floats"
)

// The default L-BFGS-B kernels. SciPy's C code (and Options.SciPyExact) sums
// each product rounded before it is added, in one sequential chain, and takes
// the 2-norm in x87 extended precision. Here the sums run on FMA with several
// accumulators (go-simd/floats when the free set is the whole vector, a
// four-lane gather loop otherwise) and the norm is fastEnorm. The results
// differ from SciPy's in the last bits and may take a different number of
// iterations; TestLBFGSBFixturesFast and TestAccuracyBoundLBFGSB hold them to
// the accuracy contract instead of to SciPy's bits.

// The three gather kernels below (gdot, gdotD, gaxpy) belong in
// github.com/go-simd/floats, the lowest shared library, which has no gather
// kernel as of v0.15.0 (the request is go-simd/floats#50: DotIdx, DotGather,
// AxpyGather). They stay here until a go-simd/floats release carries them,
// then this file calls it.

// gdot returns Σ a[idx[k]]·b[idx[k]] on four FMA accumulators.
func gdot(a, b []float64, idx []int) float64 {
	var s0, s1, s2, s3 float64
	k := 0
	for ; k+4 <= len(idx); k += 4 {
		i0, i1, i2, i3 := idx[k], idx[k+1], idx[k+2], idx[k+3]
		s0 = math.FMA(a[i0], b[i0], s0)
		s1 = math.FMA(a[i1], b[i1], s1)
		s2 = math.FMA(a[i2], b[i2], s2)
		s3 = math.FMA(a[i3], b[i3], s3)
	}
	for ; k < len(idx); k++ {
		i := idx[k]
		s0 = math.FMA(a[i], b[i], s0)
	}
	return (s0 + s1) + (s2 + s3)
}

// gdotD returns Σ a[idx[k]]·d[k] on four FMA accumulators.
func gdotD(a []float64, idx []int, d []float64) float64 {
	var s0, s1, s2, s3 float64
	k := 0
	for ; k+4 <= len(idx); k += 4 {
		s0 = math.FMA(a[idx[k]], d[k], s0)
		s1 = math.FMA(a[idx[k+1]], d[k+1], s1)
		s2 = math.FMA(a[idx[k+2]], d[k+2], s2)
		s3 = math.FMA(a[idx[k+3]], d[k+3], s3)
	}
	for ; k < len(idx); k++ {
		s0 = math.FMA(a[idx[k]], d[k], s0)
	}
	return (s0 + s1) + (s2 + s3)
}

// gaxpy does d[k] += α·a[idx[k]].
func gaxpy(alpha float64, a []float64, idx []int, d []float64) {
	for k, i := range idx {
		d[k] = math.FMA(alpha, a[i], d[k])
	}
}

// freeDot returns Σ_{k<nsub} a[index[k]]·b[index[k]]: a contiguous dot when
// nothing is bounded (index is then the identity), a gather otherwise.
func (s *lbfgsb) freeDot(a, b []float64, nsub int) float64 {
	if !s.cnstnd {
		return floats.Dot(a[:nsub], b[:nsub])
	}
	return gdot(a, b, s.index[:nsub])
}

// freeDotD returns Σ_{k<nsub} a[index[k]]·d[k].
func (s *lbfgsb) freeDotD(a, d []float64, nsub int) float64 {
	if !s.cnstnd {
		return floats.Dot(a[:nsub], d[:nsub])
	}
	return gdotD(a, s.index[:nsub], d)
}

// freeAxpy does d[k] += α·a[index[k]] for k < nsub.
func (s *lbfgsb) freeAxpy(alpha float64, a, d []float64, nsub int) {
	if !s.cnstnd {
		floats.Axpy(alpha, a[:nsub], d[:nsub])
		return
	}
	gaxpy(alpha, a, s.index[:nsub], d)
}

// subsmFast is the first half of subsm (wv = W'Zd, wv := K⁻¹wv, d = (1/θ)d +
// (1/θ²)Z'W wv) on the FMA kernels.
func (s *lbfgsb) subsmFast(d, wv []float64, nsub int) {
	n, m, col, head, theta := s.n, s.m, s.col, s.head, s.theta
	ws, wy := s.ws, s.wy
	m2 := 2 * m
	col2 := 2 * col
	pointr := head
	for i := 0; i < col; i++ {
		wv[i] = s.freeDotD(wy[n*pointr:n*pointr+n], d, nsub)
		wv[col+i] = theta * s.freeDotD(ws[n*pointr:n*pointr+n], d, nsub)
		pointr = (pointr + 1) % m
	}
	dtrsl(s.wn, m2, col2, wv, true)
	for i := 0; i < col; i++ {
		wv[i] = -wv[i]
	}
	dtrsl(s.wn, m2, col2, wv, false)
	pointr = head
	for jy := 0; jy < col; jy++ {
		s.freeAxpy(wv[jy]/theta, wy[n*pointr:n*pointr+n], d, nsub)
		s.freeAxpy(wv[col+jy], ws[n*pointr:n*pointr+n], d, nsub)
		pointr = (pointr + 1) % m
	}
	floats.Scale(1/theta, d[:nsub])
}
