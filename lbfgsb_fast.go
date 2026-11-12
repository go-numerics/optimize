package optimize

import "github.com/go-simd/floats"

// The default L-BFGS-B kernels. SciPy's C code (and Options.SciPyExact) sums
// each product rounded before it is added, in one sequential chain, and takes
// the 2-norm in x87 extended precision. Here the sums run on go-simd/floats
// (FMA, several accumulators): floats.Dot and Axpy when the free set is the
// whole vector, the gather kernels floats.DotIdx, DotGather and AxpyGather
// (v0.17.0) otherwise, and the norm is fastEnorm. The results differ from
// SciPy's in the last bits and may take a different number of iterations;
// TestLBFGSBFixturesFast and TestAccuracyBoundLBFGSB hold them to the accuracy
// contract instead of to SciPy's bits.

// freeDot returns Σ_{k<nsub} a[index[k]]·b[index[k]]: a contiguous dot when
// nothing is bounded (index is then the identity), a gather otherwise.
func (s *lbfgsb) freeDot(a, b []float64, nsub int) float64 {
	if !s.cnstnd {
		return floats.Dot(a[:nsub], b[:nsub])
	}
	return floats.DotIdx(a, b, s.index[:nsub])
}

// freeDotD returns Σ_{k<nsub} a[index[k]]·d[k].
func (s *lbfgsb) freeDotD(a, d []float64, nsub int) float64 {
	if !s.cnstnd {
		return floats.Dot(a[:nsub], d[:nsub])
	}
	return floats.DotGather(a, s.index[:nsub], d)
}

// freeAxpy does d[k] += α·a[index[k]] for k < nsub.
func (s *lbfgsb) freeAxpy(alpha float64, a, d []float64, nsub int) {
	if !s.cnstnd {
		floats.Axpy(alpha, a[:nsub], d[:nsub])
		return
	}
	floats.AxpyGather(alpha, a, s.index[:nsub], d)
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
