package optimize

import "math"

// This file holds the shared kernels of MINPACK (Moré, Garbow and Hillstrom,
// Argonne National Laboratory, 1980), ported from the C translation that
// SciPy compiles (scipy/optimize/src/minpack.c, itself a line-by-line
// translation of the netlib Fortran). Matrices are kept column-major, as in
// MINPACK, so that a(i,j) is a[i+lda*j] and the operation order of every
// sum matches the reference; indices are zero-based. Every product that
// meets an addition is wrapped in float64(), which forbids Go from fusing it
// into an FMA (Go does on arm64): the port rounds like the reference built
// without contraction, identically on every architecture.

// Machine constants of MINPACK's dpmpar for IEEE double precision.
const (
	mpDwarf = 2.2250738585072014e-308 // smallest positive normal
	mpGiant = math.MaxFloat64         // largest finite magnitude
)

// The thresholds of mpEnorm: components below rdwarf or above rgiant/n are
// accumulated with scaling.
var (
	rgiant = math.Pow(2, 63.5)
	rdwarf = 0.5 / rgiant
)

// mpEnorm is MINPACK's enorm: the Euclidean norm computed with three
// accumulators for small, intermediate and large components, so that it
// neither overflows nor underflows. For the intermediate range it is a plain
// sum of squares, which is what makes the iterates match the reference to the
// last bit; the package's enorm (a scaled sum) rounds differently.
func mpEnorm(x []float64) float64 {
	var s2 float64
	var t enormTail
	agiant := rgiant / float64(len(x))
	for _, xi := range x {
		xabs := math.Abs(xi)
		if xabs > rdwarf && xabs < agiant {
			s2 += float64(xabs * xabs)
		} else {
			t.add(xabs)
		}
	}
	return t.norm(s2)
}

// enormTail holds the accumulators of mpEnorm for the components outside the
// intermediate range; the intermediate sum of squares s2 is kept apart so
// that the hot loops hold it in a register.
type enormTail struct{ s1, s3, x1max, x3max float64 }

// add accumulates a component that is not in the intermediate range
// (rdwarf, agiant), exactly as the remaining cases of MINPACK's enorm.
func (e *enormTail) add(xabs float64) {
	switch {
	case xabs > rdwarf:
		if xabs > e.x1max {
			t := e.x1max / xabs
			e.s1 = 1 + float64(e.s1*(t*t))
			e.x1max = xabs
		} else {
			t := xabs / e.x1max
			e.s1 += float64(t * t)
		}
	case xabs > e.x3max:
		t := e.x3max / xabs
		e.s3 = 1 + float64(e.s3*(t*t))
		e.x3max = xabs
	case xabs != 0:
		t := xabs / e.x3max
		e.s3 += float64(t * t)
	}
}

// norm combines the accumulators into enorm's result.
func (e *enormTail) norm(s2 float64) float64 {
	s1, s3, x1max, x3max := e.s1, e.s3, e.x1max, e.x3max
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

// mpEnorm4 returns mpEnorm of four vectors of equal length (the same vector
// may be passed more than once). Each norm is accumulated in its own order,
// so the results are those of four mpEnorm calls to the last bit, but the
// four sums advance together and a long vector costs about one pass of
// floating-point latency instead of four.
func mpEnorm4(c0, c1, c2, c3 []float64) (n0, n1, n2, n3 float64) {
	n := len(c0)
	c1, c2, c3 = c1[:n], c2[:n], c3[:n]
	agiant := rgiant / float64(n)
	// A magnitude a lies in the open interval (rdwarf, agiant) exactly when
	// bits(a) − bits(rdwarf) − 1 < bits(agiant) − bits(rdwarf) − 1 as
	// unsigned integers (non-negative floats order as their bit patterns;
	// NaN's bits lie above +Inf's). One test then covers four components.
	// (agiant >= rgiant/MaxInt ≫ rdwarf, so the interval is never empty.)
	lo := math.Float64bits(rdwarf) + 1
	span := math.Float64bits(agiant) - lo
	var s0, s1, s2, s3 float64
	var t0, t1, t2, t3 enormTail
	for i := range c0 {
		a0, a1, a2, a3 := math.Abs(c0[i]), math.Abs(c1[i]), math.Abs(c2[i]), math.Abs(c3[i])
		u := max(math.Float64bits(a0)-lo, math.Float64bits(a1)-lo, math.Float64bits(a2)-lo, math.Float64bits(a3)-lo)
		if u < span {
			s0 += float64(a0 * a0)
			s1 += float64(a1 * a1)
			s2 += float64(a2 * a2)
			s3 += float64(a3 * a3)
			continue
		}
		if a0 > rdwarf && a0 < agiant {
			s0 += float64(a0 * a0)
		} else {
			t0.add(a0)
		}
		if a1 > rdwarf && a1 < agiant {
			s1 += float64(a1 * a1)
		} else {
			t1.add(a1)
		}
		if a2 > rdwarf && a2 < agiant {
			s2 += float64(a2 * a2)
		} else {
			t2.add(a2)
		}
		if a3 > rdwarf && a3 < agiant {
			s3 += float64(a3 * a3)
		} else {
			t3.add(a3)
		}
	}
	return t0.norm(s0), t1.norm(s1), t2.norm(s2), t3.norm(s3)
}

// colEnorms sets out[k] = mpEnorm(a[off+lda*k : end+lda*k]) for every k in
// [0, len(out)), four columns per pass.
func colEnorms(a []float64, lda, off, end int, out []float64) {
	col := func(k int) []float64 { return a[off+lda*k : end+lda*k] }
	k := 0
	for ; k+4 <= len(out); k += 4 {
		out[k], out[k+1], out[k+2], out[k+3] = mpEnorm4(col(k), col(k+1), col(k+2), col(k+3))
	}
	switch len(out) - k {
	case 1:
		out[k] = mpEnorm(col(k))
	case 2:
		out[k], out[k+1], _, _ = mpEnorm4(col(k), col(k+1), col(k), col(k+1))
	case 3:
		out[k], out[k+1], out[k+2], _ = mpEnorm4(col(k), col(k+1), col(k+2), col(k+2))
	}
}

// fdjac is the dense forward-difference Jacobian of MINPACK (fdjac2, and
// fdjac1 without a band): column j is (f(x+h·e_j) − fvec)/h with
// h = sqrt(max(epsfcn, eps))·|x_j| (or that root when x_j is 0). fjac is
// column-major m×n with leading dimension m; wa has length m.
func fdjac(fcn func(x, f []float64), m, n int, x, fvec, fjac []float64, epsfcn float64, wa []float64) {
	eps := math.Sqrt(math.Max(epsfcn, epsmch))
	for j := 0; j < n; j++ {
		temp := x[j]
		h := float64(eps * math.Abs(temp))
		if h == 0 {
			h = eps
		}
		x[j] = temp + h
		fcn(x, wa)
		x[j] = temp
		for i := 0; i < m; i++ {
			fjac[i+m*j] = (wa[i] - fvec[i]) / h
		}
	}
}

// qrfac computes the Householder QR factorisation of the m×n column-major
// matrix a (leading dimension lda), with column pivoting when pivot is set:
// a·P = Q·R. On return the lower trapezoid of a holds the Householder vectors
// (Q in factored form), its strict upper triangle holds R off the diagonal,
// rdiag the diagonal of R, acnorm the norms of the columns of the input, and
// ipvt the permutation (column j of P is column ipvt[j] of the identity).
// wa has length n.
//
// The reflections are also applied to qtb (length m) as they are formed,
// leaving Qᵀ·qtb: MINPACK's lmdif and hybrd compute this product after the
// factorisation, with the same sums in the same order, so the result is
// identical to the last bit.
//
// The operation order of every sum is MINPACK's. The sums that MINPACK
// computes one after the other but that do not depend on one another (the
// column norms, and the products of a reflection with the remaining
// columns) are accumulated in the same pass, which overlaps their latency.
func qrfac(m, n int, a []float64, lda int, pivot bool, ipvt []int, rdiag, acnorm, wa, qtb []float64) {
	var colStack [8][]float64
	var tStack [8]float64
	var renormStack [8]bool
	cols, temps, renorm := colStack[:0], tStack[:], renormStack[:]
	if n+1 > len(tStack) {
		temps, renorm = make([]float64, n+1), make([]bool, n)
	}
	colEnorms(a, lda, 0, m, acnorm[:n])
	for j := 0; j < n; j++ {
		rdiag[j] = acnorm[j]
		wa[j] = rdiag[j]
		if pivot {
			ipvt[j] = j
		}
	}
	minmn := min(m, n)
	// next, when nextOK, is the norm of rows j..m of the column that the
	// pivoting below brings into position j, accumulated by the previous
	// step while it updated that column.
	var next float64
	nextOK := false
	for j := 0; j < minmn; j++ {
		orig := j
		if pivot {
			// Bring the column of largest remaining norm into the pivot
			// position.
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
			orig = ipvt[j]
		}
		// The Householder transformation that reduces column j to a multiple
		// of the j-th unit vector. For j = 0 the column is an input column
		// whose norm acnorm already holds.
		var ajnorm float64
		switch {
		case j == 0:
			ajnorm = acnorm[orig]
		case nextOK:
			ajnorm = next
		default:
			ajnorm = mpEnorm(a[lda*j+j : lda*j+m])
		}
		nextOK = false
		if ajnorm == 0 {
			rdiag[j] = -ajnorm
			continue
		}
		if a[lda*j+j] < 0 {
			ajnorm = -ajnorm
		}
		// v ← v/ajnorm, v[0] += 1, and the sums vᵀc of the reflection for
		// the remaining columns (and qtb), in one pass.
		v := a[lda*j+j : lda*j+m]
		cols = cols[:0]
		for k := j + 1; k < n; k++ {
			cols = append(cols, a[lda*k+j:lda*k+m])
		}
		cols = append(cols, qtb[j:m])
		scaleDots(v, ajnorm, cols, temps)
		for c := range cols {
			temps[c] /= v[0]
		}
		// The norm downdates need only row j of the updated columns, so they
		// and the next pivot are known before the update pass, which then
		// also accumulates the norm the next step needs. A column whose norm
		// must be recomputed (the downdate lost too much accuracy) gets it
		// after the pass, and the next step computes its own.
		nc, anyRenorm := -1, false
		for k := j + 1; k < n; k++ {
			renorm[k] = false
			if pivot && rdiag[k] != 0 {
				ajk := a[j+lda*k] - float64(temps[k-j-1]*v[0])
				temp := ajk / rdiag[k]
				rdiag[k] *= math.Sqrt(math.Max(0, 1-float64(temp*temp)))
				if q := rdiag[k] / wa[k]; 0.05*(q*q) <= epsmch {
					renorm[k], anyRenorm = true, true
				}
			}
		}
		if j+1 < minmn && !anyRenorm {
			nc = j + 1
			if pivot {
				for k := j + 1; k < n; k++ {
					if rdiag[k] > rdiag[nc] {
						nc = k
					}
				}
			}
			nc -= j + 1 // index in cols
		}
		next, nextOK = applyReflection(v, cols, temps, nc)
		if anyRenorm {
			for k := j + 1; k < n; k++ {
				if renorm[k] {
					rdiag[k] = mpEnorm(a[(j+1)+lda*k : m+lda*k])
					wa[k] = rdiag[k]
				}
			}
		}
		rdiag[j] = -ajnorm
	}
}

// scaleDots divides v by ajnorm and adds 1 to v[0], as qrfac forms its
// Householder vector, and sets sums[c] = Σ v[i]·cols[c][i] (index order,
// from 0.0) with the scaled v, up to four columns per pass. cols is not
// empty (qrfac always reflects qtb).
func scaleDots(v []float64, ajnorm float64, cols [][]float64, sums []float64) {
	n := len(v)
	v[0] = v[0]/ajnorm + 1
	for g := 0; g < len(cols); g += 4 {
		// Pad a short group with repeated columns: their sums are dropped.
		var c [4][]float64
		for k := range c {
			c[k] = cols[min(g+k, len(cols)-1)][:n]
		}
		c0, c1, c2, c3 := c[0], c[1], c[2], c[3]
		var s0, s1, s2, s3 float64
		v0 := v[0]
		s0 += float64(v0 * c0[0])
		s1 += float64(v0 * c1[0])
		s2 += float64(v0 * c2[0])
		s3 += float64(v0 * c3[0])
		if g == 0 {
			for i := 1; i < n; i++ {
				vi := v[i] / ajnorm
				v[i] = vi
				s0 += float64(vi * c0[i])
				s1 += float64(vi * c1[i])
				s2 += float64(vi * c2[i])
				s3 += float64(vi * c3[i])
			}
		} else {
			for i := 1; i < n; i++ {
				vi := v[i]
				s0 += float64(vi * c0[i])
				s1 += float64(vi * c1[i])
				s2 += float64(vi * c2[i])
				s3 += float64(vi * c3[i])
			}
		}
		s := [4]float64{s0, s1, s2, s3}
		copy(sums[g:min(g+4, len(cols))], s[:])
	}
}

// applyReflection sets c −= temps[k]·v for each column c = cols[k], up to
// four columns per pass. When nc is a column index it also returns mpEnorm
// of the updated cols[nc][1:], accumulated in the first pass (ok reports
// that it did).
func applyReflection(v []float64, cols [][]float64, temps []float64, nc int) (norm float64, ok bool) {
	var others [3]int
	cnt := 0
	if nc >= 0 {
		for k := 0; k < len(cols) && cnt < 3; k++ {
			if k != nc {
				others[cnt] = k
				cnt++
			}
		}
		norm, ok = axpyNorm(v, cols, temps, nc, others[:cnt]), true
	}
	// The columns not done above, four at a time.
	var idx [4]int
	n := 0
	for k := range cols {
		if k == nc || (cnt > 0 && k <= others[cnt-1]) {
			continue
		}
		idx[n] = k
		n++
		if n == 4 {
			axpyCols(v, cols, temps, idx[:n])
			n = 0
		}
	}
	if n > 0 {
		axpyCols(v, cols, temps, idx[:n])
	}
	return norm, ok
}

// axpyCols sets c −= temps[k]·v for the columns cols[k], k in ks (1 to 4
// indices), in one pass.
func axpyCols(v []float64, cols [][]float64, temps []float64, ks []int) {
	n := len(v)
	switch len(ks) {
	case 1:
		c0, t0 := cols[ks[0]][:n], temps[ks[0]]
		for i, vi := range v {
			c0[i] -= float64(t0 * vi)
		}
	case 2:
		c0, t0 := cols[ks[0]][:n], temps[ks[0]]
		c1, t1 := cols[ks[1]][:n], temps[ks[1]]
		for i, vi := range v {
			c0[i] -= float64(t0 * vi)
			c1[i] -= float64(t1 * vi)
		}
	case 3:
		c0, t0 := cols[ks[0]][:n], temps[ks[0]]
		c1, t1 := cols[ks[1]][:n], temps[ks[1]]
		c2, t2 := cols[ks[2]][:n], temps[ks[2]]
		for i, vi := range v {
			c0[i] -= float64(t0 * vi)
			c1[i] -= float64(t1 * vi)
			c2[i] -= float64(t2 * vi)
		}
	default:
		c0, t0 := cols[ks[0]][:n], temps[ks[0]]
		c1, t1 := cols[ks[1]][:n], temps[ks[1]]
		c2, t2 := cols[ks[2]][:n], temps[ks[2]]
		c3, t3 := cols[ks[3]][:n], temps[ks[3]]
		for i, vi := range v {
			c0[i] -= float64(t0 * vi)
			c1[i] -= float64(t1 * vi)
			c2[i] -= float64(t2 * vi)
			c3[i] -= float64(t3 * vi)
		}
	}
}

// axpyNorm updates cols[nc] and the columns cols[k], k in others (at most
// three, at least one: qrfac always reflects qtb besides the column whose
// norm it wants), as axpyCols does, and returns mpEnorm of the updated
// cols[nc][1:].
func axpyNorm(v []float64, cols [][]float64, temps []float64, nc int, others []int) float64 {
	n := len(v)
	cn, tn := cols[nc][:n], temps[nc]
	var c [3][]float64
	var t [3]float64
	for k, o := range others {
		c[k], t[k] = cols[o][:n], temps[o]
	}
	cn[0] -= float64(tn * v[0])
	for k := range others {
		c[k][0] -= float64(t[k] * v[0])
	}
	var s float64
	var tail enormTail
	agiant := rgiant / float64(n-1)
	switch len(others) {
	case 1:
		c0, t0 := c[0], t[0]
		for i := 1; i < n; i++ {
			vi := v[i]
			c0[i] -= float64(t0 * vi)
			x := cn[i] - float64(tn*vi)
			cn[i] = x
			if xabs := math.Abs(x); xabs > rdwarf && xabs < agiant {
				s += float64(xabs * xabs)
			} else {
				tail.add(xabs)
			}
		}
	case 2:
		c0, t0, c1, t1 := c[0], t[0], c[1], t[1]
		for i := 1; i < n; i++ {
			vi := v[i]
			c0[i] -= float64(t0 * vi)
			c1[i] -= float64(t1 * vi)
			x := cn[i] - float64(tn*vi)
			cn[i] = x
			if xabs := math.Abs(x); xabs > rdwarf && xabs < agiant {
				s += float64(xabs * xabs)
			} else {
				tail.add(xabs)
			}
		}
	default:
		c0, t0, c1, t1, c2, t2 := c[0], t[0], c[1], t[1], c[2], t[2]
		for i := 1; i < n; i++ {
			vi := v[i]
			c0[i] -= float64(t0 * vi)
			c1[i] -= float64(t1 * vi)
			c2[i] -= float64(t2 * vi)
			x := cn[i] - float64(tn*vi)
			cn[i] = x
			if xabs := math.Abs(x); xabs > rdwarf && xabs < agiant {
				s += float64(xabs * xabs)
			} else {
				tail.add(xabs)
			}
		}
	}
	return tail.norm(s)
}

// qrsolv solves, in the least-squares sense, A·x = b together with D·x = 0,
// given the pivoted QR factorisation A·P = Q·R of qrfac: r holds R in its
// upper triangle (leading dimension ldr), qtb is Qᵀb. The full upper triangle
// of r is preserved; its strict lower triangle receives the transpose of the
// triangular factor S of Pᵀ(AᵀA + DD)P = SᵀS, whose diagonal goes to sdiag.
func qrsolv(n int, r []float64, ldr int, ipvt []int, diag, qtb, x, sdiag, wa []float64) {
	// Copy R and Qᵀb to preserve input and initialise S; save the diagonal
	// of R in x.
	for j := 0; j < n; j++ {
		for i := j; i < n; i++ {
			r[i+j*ldr] = r[j+i*ldr]
		}
		x[j] = r[j+j*ldr]
		wa[j] = qtb[j]
	}
	// Eliminate the diagonal matrix D with Givens rotations.
	for j := 0; j < n; j++ {
		l := ipvt[j]
		if diag[l] != 0 {
			for k := j; k < n; k++ {
				sdiag[k] = 0
			}
			sdiag[j] = diag[l]
			qtbpj := 0.0
			for k := j; k < n; k++ {
				if sdiag[k] == 0 {
					continue
				}
				var cos, sin float64
				if math.Abs(r[ldr*k+k]) < math.Abs(sdiag[k]) {
					cotan := r[ldr*k+k] / sdiag[k]
					sin = 0.5 / math.Sqrt(0.25+float64(0.25*(cotan*cotan)))
					cos = sin * cotan
				} else {
					tan := sdiag[k] / r[ldr*k+k]
					cos = 0.5 / math.Sqrt(0.25+float64(0.25*(tan*tan)))
					sin = cos * tan
				}
				r[ldr*k+k] = float64(r[ldr*k+k] * cos)
				r[ldr*k+k] += float64(sin * sdiag[k])
				temp := float64(cos*wa[k]) + float64(sin*qtbpj)
				qtbpj = float64(-sin*wa[k]) + float64(cos*qtbpj)
				wa[k] = temp
				for i := k + 1; i < n; i++ {
					temp = float64(cos*r[i+ldr*k]) + float64(sin*sdiag[i])
					sdiag[i] = float64(-sin*r[i+ldr*k]) + float64(cos*sdiag[i])
					r[i+ldr*k] = temp
				}
			}
		}
		sdiag[j] = r[ldr*j+j]
		r[ldr*j+j] = x[j]
	}
	// Solve the triangular system; a singular S gives a least-squares
	// solution.
	nsing := n
	for j := 0; j < n; j++ {
		if sdiag[j] == 0 && nsing == n {
			nsing = j
		}
		if nsing < n {
			wa[j] = 0
		}
	}
	for j := nsing - 1; j >= 0; j-- {
		sum := 0.0
		for i := j + 1; i < nsing; i++ {
			sum += float64(r[i+j*ldr] * wa[i])
		}
		wa[j] = (wa[j] - sum) / sdiag[j]
	}
	for j := 0; j < n; j++ {
		x[ipvt[j]] = wa[j]
	}
}

// lmpar determines the Levenberg–Marquardt parameter par such that the
// solution x of A·x = b, sqrt(par)·D·x = 0 (least squares) satisfies
// |D·x| ≈ delta (within 10%), or par = 0 when the Gauss–Newton step is
// already inside the trust region. r, ipvt and qtb come from qrfac; the
// strict lower triangle of r and sdiag receive the factor S of qrsolv. It
// returns the new par, starting the search from the given one.
func lmpar(n int, r []float64, ldr int, ipvt []int, diag, qtb []float64, delta, par float64, x, sdiag, wa1, wa2 []float64) float64 {
	// The Gauss–Newton direction, a least-squares solution when R is
	// singular.
	nsing := n
	for j := 0; j < n; j++ {
		wa1[j] = qtb[j]
		if r[j+j*ldr] == 0 && nsing == n {
			nsing = j
		}
		if nsing < n {
			wa1[j] = 0
		}
	}
	for j := nsing - 1; j >= 0; j-- {
		wa1[j] /= r[j+j*ldr]
		temp := wa1[j]
		for i := 0; i < j; i++ {
			wa1[i] -= float64(r[i+j*ldr] * temp)
		}
	}
	for j := 0; j < n; j++ {
		x[ipvt[j]] = wa1[j]
	}
	// Accept it if it is within the trust region.
	iter := 0
	for j := 0; j < n; j++ {
		wa2[j] = diag[j] * x[j]
	}
	dxnorm := mpEnorm(wa2[:n])
	fp := dxnorm - delta
	if fp <= 0.1*delta {
		return 0
	}
	// A lower bound parl for the zero of the function, from the Newton step
	// (only when R is nonsingular).
	parl := 0.0
	if nsing == n {
		for j := 0; j < n; j++ {
			l := ipvt[j]
			wa1[j] = diag[l] * (wa2[l] / dxnorm)
		}
		for j := 0; j < n; j++ {
			sum := 0.0
			for i := 0; i < j; i++ {
				sum += float64(r[i+j*ldr] * wa1[i])
			}
			wa1[j] = (wa1[j] - sum) / r[j+j*ldr]
		}
		temp := mpEnorm(wa1[:n])
		parl = ((fp / delta) / temp) / temp
	}
	// An upper bound paru.
	for j := 0; j < n; j++ {
		sum := 0.0
		for i := 0; i <= j; i++ {
			sum += float64(r[i+j*ldr] * qtb[i])
		}
		wa1[j] = sum / diag[ipvt[j]]
	}
	gnorm := mpEnorm(wa1[:n])
	paru := gnorm / delta
	if paru == 0 {
		paru = mpDwarf / math.Min(delta, 0.1)
	}
	// Clamp the input par into [parl, paru]; a zero par starts from the
	// gradient ratio.
	par = math.Min(math.Max(par, parl), paru)
	if par == 0 {
		par = gnorm / dxnorm
	}
	for {
		iter++
		if par == 0 {
			par = math.Max(mpDwarf, paru*0.001)
		}
		temp := math.Sqrt(par)
		for j := 0; j < n; j++ {
			wa1[j] = temp * diag[j]
		}
		qrsolv(n, r, ldr, ipvt, wa1, qtb, x, sdiag, wa2)
		for j := 0; j < n; j++ {
			wa2[j] = diag[j] * x[j]
		}
		dxnorm = mpEnorm(wa2[:n])
		temp = fp
		fp = dxnorm - delta
		// Stop when |D·x| is close to delta, when parl is zero and the
		// function decreases while below delta, or after ten iterations.
		if math.Abs(fp) <= 0.1*delta || (parl == 0 && fp <= temp && temp < 0) || iter == 10 {
			return par
		}
		// The Newton correction.
		for j := 0; j < n; j++ {
			l := ipvt[j]
			wa1[j] = diag[l] * (wa2[l] / dxnorm)
		}
		for j := 0; j < n; j++ {
			wa1[j] /= sdiag[j]
			temp = wa1[j]
			for i := j + 1; i < n; i++ {
				wa1[i] -= float64(r[i+j*ldr] * temp)
			}
		}
		temp = mpEnorm(wa1[:n])
		parc := ((fp / delta) / temp) / temp
		if fp > 0 {
			parl = math.Max(parl, par)
		}
		if fp < 0 {
			paru = math.Min(paru, par)
		}
		par = math.Max(parl, par+parc)
	}
}

// dogleg computes the convex combination x of the Gauss–Newton and scaled
// gradient directions that minimises |A·x − b| within |D·x| <= delta, given
// the QR factorisation A = Q·R with R packed by rows in r (n(n+1)/2 entries)
// and qtb = Qᵀb.
func dogleg(n int, r, diag, qtb []float64, delta float64, x, wa1, wa2 []float64) {
	// The Gauss–Newton direction.
	jj := n * (n + 1) / 2
	for k := 0; k < n; k++ {
		j := n - k - 1
		jj -= k + 1
		l := jj + 1
		sum := 0.0
		for i := j + 1; i < n; i++ {
			sum += float64(r[l] * x[i])
			l++
		}
		temp := r[jj]
		if temp == 0 {
			// A zero pivot: replace it by eps times the largest entry of its
			// column.
			l = j
			for i := 0; i <= j; i++ {
				temp = math.Max(temp, math.Abs(r[l]))
				l += n - i - 1
			}
			temp = epsmch * temp
			if temp == 0 {
				temp = epsmch
			}
		}
		x[j] = (qtb[j] - sum) / temp
	}
	// Accept it when it lies inside the trust region.
	for j := 0; j < n; j++ {
		wa1[j] = 0
		wa2[j] = diag[j] * x[j]
	}
	qnorm := mpEnorm(wa2[:n])
	if qnorm <= delta {
		return
	}
	// Otherwise the scaled gradient direction.
	l := 0
	for j := 0; j < n; j++ {
		temp := qtb[j]
		for i := j; i < n; i++ {
			wa1[i] += float64(r[l] * temp)
			l++
		}
		wa1[j] /= diag[j]
	}
	// The norm of the scaled gradient, and the minimiser along it (sgnorm).
	gnorm := mpEnorm(wa1[:n])
	sgnorm := 0.0
	alpha := delta / qnorm
	if gnorm != 0 {
		for j := 0; j < n; j++ {
			wa1[j] /= gnorm
			wa1[j] /= diag[j]
		}
		l = 0
		for j := 0; j < n; j++ {
			sum := 0.0
			for i := j; i < n; i++ {
				sum += float64(r[l] * wa1[i])
				l++
			}
			wa2[j] = sum
		}
		temp := mpEnorm(wa2[:n])
		sgnorm = (gnorm / temp) / temp
		// The scaled gradient step lies inside the region: walk along the
		// dogleg to its boundary.
		alpha = 0
		if sgnorm < delta {
			bnorm := mpEnorm(qtb[:n])
			sd := sgnorm / delta
			dq := delta / qnorm
			temp = (bnorm / gnorm) * (bnorm / qnorm) * sd
			temp = temp - float64(dq*(sd*sd)) + math.Sqrt(float64((temp-dq)*(temp-dq))+float64((1-float64(dq*dq))*(1-float64(sd*sd))))
			alpha = (dq * (1 - float64(sd*sd))) / temp
		}
	}
	// The convex combination of the two directions.
	temp := (1 - alpha) * math.Min(sgnorm, delta)
	for j := 0; j < n; j++ {
		x[j] = float64(temp*wa1[j]) + float64(alpha*x[j])
	}
}

// qform accumulates the n×n orthogonal matrix Q from the Householder vectors
// that qrfac left in the lower triangle of q (column-major, leading dimension
// ldq).
func qform(n int, q []float64, ldq int, wa []float64) {
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
		for j := k; j < n; j++ {
			sum := 0.0
			for i := k; i < n; i++ {
				sum += float64(q[i+ldq*j] * wa[i])
			}
			temp := sum / wa[k]
			for i := k; i < n; i++ {
				q[i+ldq*j] -= float64(temp * wa[i])
			}
		}
	}
}

// r1mpyq computes A·Qᵀ·... for the m×n matrix a (column-major, leading
// dimension lda): it applies the 2(n−1) Givens rotations, encoded in v and
// w as r1updt produced them, that update an orthogonal factor after a rank-1
// change.
func r1mpyq(m, n int, a []float64, lda int, v, w []float64) {
	rot := func(s float64) (cos, sin float64) {
		if math.Abs(s) > 1 {
			cos = 1 / s
			return cos, math.Sqrt(1 - float64(cos*cos))
		}
		return math.Sqrt(1 - float64(s*s)), s
	}
	for j := n - 2; j >= 0; j-- {
		cos, sin := rot(v[j])
		for i := 0; i < m; i++ {
			temp := float64(cos*a[i+j*lda]) - float64(sin*a[i+(n-1)*lda])
			a[i+(n-1)*lda] = float64(sin*a[i+j*lda]) + float64(cos*a[i+(n-1)*lda])
			a[i+j*lda] = temp
		}
	}
	for j := 0; j < n-1; j++ {
		cos, sin := rot(w[j])
		for i := 0; i < m; i++ {
			temp := float64(cos*a[i+lda*j]) + float64(sin*a[i+(n-1)*lda])
			a[i+(n-1)*lda] = float64(-sin*a[i+j*lda]) + float64(cos*a[i+(n-1)*lda])
			a[i+lda*j] = temp
		}
	}
}

// r1updt updates the lower-trapezoidal m×n matrix S, stored by columns in
// packed form in s, to S' with S + u·vᵀ = Q·S' (Broyden's rank-1 update of
// the triangular factor). v and w receive the encoded rotations for r1mpyq,
// and it reports whether S' has a zero diagonal entry.
func r1updt(m, n int, s, u, v, w []float64) (sing bool) {
	// Here i, j, jj and l are one-based, as in the reference.
	jj := (n*(2*m-n+1))/2 - (m - n)
	l := jj
	for i := n; i <= m; i++ {
		w[i-1] = s[l-1]
		l++
	}
	// Rotate v into a multiple of the n-th unit vector, turning the last
	// column of S into a spike.
	for j := n - 1; j >= 1; j-- {
		jj -= m - j + 1
		w[j-1] = 0
		if v[j-1] == 0 {
			continue
		}
		var cos, sin, tau float64
		if math.Abs(v[n-1]) < math.Abs(v[j-1]) {
			cotan := v[n-1] / v[j-1]
			sin = 0.5 / math.Sqrt(0.25+float64(0.25*(cotan*cotan)))
			cos = sin * cotan
			tau = 1
			if math.Abs(cos)*mpGiant > 1 {
				tau = 1 / cos
			}
		} else {
			tan := v[j-1] / v[n-1]
			cos = 0.5 / math.Sqrt(0.25+float64(0.25*(tan*tan)))
			sin = cos * tan
			tau = sin
		}
		v[n-1] = float64(sin*v[j-1]) + float64(cos*v[n-1])
		v[j-1] = tau
		l = jj
		for i := j; i <= m; i++ {
			temp := float64(cos*s[l-1]) - float64(sin*w[i-1])
			w[i-1] = float64(sin*s[l-1]) + float64(cos*w[i-1])
			s[l-1] = temp
			l++
		}
	}
	// Add the spike from the rank-1 update.
	for i := 0; i < m; i++ {
		w[i] += float64(v[n-1] * u[i])
	}
	// Eliminate the spike.
	for j := 1; j <= n-1; j++ {
		if w[j-1] != 0 {
			var cos, sin, tau float64
			if math.Abs(s[jj-1]) < math.Abs(w[j-1]) {
				cotan := s[jj-1] / w[j-1]
				sin = 0.5 / math.Sqrt(0.25+float64(0.25*(cotan*cotan)))
				cos = sin * cotan
				tau = 1
				if math.Abs(cos)*mpGiant > 1 {
					tau = 1 / cos
				}
			} else {
				tan := w[j-1] / s[jj-1]
				cos = 0.5 / math.Sqrt(0.25+float64(0.25*(tan*tan)))
				sin = cos * tan
				tau = sin
			}
			l = jj
			for i := j; i <= m; i++ {
				temp := float64(cos*s[l-1]) + float64(sin*w[i-1])
				w[i-1] = float64(-sin*s[l-1]) + float64(cos*w[i-1])
				s[l-1] = temp
				l++
			}
			w[j-1] = tau
		}
		if s[jj-1] == 0 {
			sing = true
		}
		jj += m - j + 1
	}
	// Store the last column.
	l = jj
	for i := n; i <= m; i++ {
		s[l-1] = w[i-1]
		l++
	}
	if s[jj-1] == 0 {
		sing = true
	}
	return sing
}
