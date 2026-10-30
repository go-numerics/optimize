package optimize

import "math"

// Small dense LAPACK/BLAS kernels used by SLSQP. They follow Reference-LAPACK
// v3.12.1 (dlarfg.f, dlarfgp.f, dlartgp.f, dgerq2.f, dormr2.f, dgeqr2.f,
// dorm2r.f) and the reference BLAS loop orders statement by statement, so the
// SLSQP port performs the same operations in the same order as SciPy's C code.
// Matrices are column-major with an explicit leading dimension, as in LAPACK.
// Every product that feeds an addition is rounded explicitly so that Go does
// not contract it into a fused multiply-add on some architectures.
//
// The go-simd/floats kernels are deliberately not used here: they reduce in a
// lane-blocked FMA order, which would change SLSQP's path both against SciPy
// and between architectures, and the vectors involved are a few dozen
// elements long.

const (
	lapackEps    = 1.1102230246251565e-16 // DLAMCH('E'): eps/2, rounding
	lapackPrec   = 2.220446049250313e-16  // DLAMCH('P'): eps*base
	lapackSafmin = 2.2250738585072014e-308
	// lapackSmlnum is DLAMCH('S')/DLAMCH('E') = 2^-969.
	lapackSmlnum = lapackSafmin / lapackEps
)

// nrm2 is the Euclidean norm of n elements of x with stride inc.
func nrm2(n int, x []float64, inc int) float64 {
	scale, ssq := 0.0, 1.0
	for i := 0; i < n; i++ {
		v := x[i*inc]
		if v == 0 {
			continue
		}
		av := math.Abs(v)
		if scale < av {
			ssq = 1 + float64(ssq*(scale/av)*(scale/av))
			scale = av
		} else {
			ssq += float64((av / scale) * (av / scale))
		}
	}
	return float64(scale * math.Sqrt(ssq))
}

// dlapy2 is sqrt(x²+y²) as LAPACK's DLAPY2 computes it. math.Hypot is not
// used: its internal 1+q*q may be fused on some architectures.
func dlapy2(x, y float64) float64 {
	if math.IsNaN(x) {
		return x
	}
	if math.IsNaN(y) {
		return y
	}
	w := math.Max(math.Abs(x), math.Abs(y))
	z := math.Min(math.Abs(x), math.Abs(y))
	if z == 0 || w > math.MaxFloat64 {
		return w
	}
	q := z / w
	return float64(w * math.Sqrt(1+float64(q*q)))
}

// dlarfg generates an elementary reflector H with H*(alpha;x) = (beta;0),
// overwriting alpha with beta and x with v(2:n); it returns tau.
func dlarfg(n int, alpha *float64, x []float64, incx int) float64 {
	if n <= 1 {
		return 0
	}
	xnorm := nrm2(n-1, x, incx)
	if xnorm == 0 {
		return 0
	}
	beta := -math.Copysign(dlapy2(*alpha, xnorm), *alpha)
	knt := 0
	if math.Abs(beta) < lapackSmlnum {
		rsafmn := 1 / lapackSmlnum
		for {
			knt++
			scal(n-1, rsafmn, x, incx)
			beta *= rsafmn
			*alpha *= rsafmn
			if !(math.Abs(beta) < lapackSmlnum && knt < 20) {
				break
			}
		}
		xnorm = nrm2(n-1, x, incx)
		beta = -math.Copysign(dlapy2(*alpha, xnorm), *alpha)
	}
	tau := (beta - *alpha) / beta
	scal(n-1, 1/(*alpha-beta), x, incx)
	for j := 0; j < knt; j++ {
		beta *= lapackSmlnum
	}
	*alpha = beta
	return tau
}

// dlarfgp is dlarfg with a non-negative beta.
func dlarfgp(n int, alpha *float64, x []float64, incx int) float64 {
	if n <= 0 {
		return 0
	}
	xnorm := nrm2(n-1, x, incx)
	if xnorm <= float64(lapackPrec*math.Abs(*alpha)) {
		if *alpha >= 0 {
			return 0
		}
		for j := 0; j < n-1; j++ {
			x[j*incx] = 0
		}
		*alpha = -*alpha
		return 2
	}
	beta := math.Copysign(dlapy2(*alpha, xnorm), *alpha)
	knt := 0
	if math.Abs(beta) < lapackSmlnum {
		bignum := 1 / lapackSmlnum
		for {
			knt++
			scal(n-1, bignum, x, incx)
			beta *= bignum
			*alpha *= bignum
			if !(math.Abs(beta) < lapackSmlnum && knt < 20) {
				break
			}
		}
		xnorm = nrm2(n-1, x, incx)
		beta = math.Copysign(dlapy2(*alpha, xnorm), *alpha)
	}
	*alpha += beta
	var tau float64
	if beta < 0 {
		beta = -beta
		tau = -*alpha / beta
	} else {
		*alpha = float64(xnorm * (xnorm / *alpha))
		tau = *alpha / beta
		*alpha = -*alpha
	}
	// LAPACK flushes a subnormal tau to zero here. It cannot happen after
	// the xnorm <= eps*|alpha| exit above: tau is in [1, 2] when beta < 0
	// and above eps²/2 otherwise (or NaN), so that branch is omitted.
	scal(n-1, 1 / *alpha, x, incx)
	for j := 0; j < knt; j++ {
		beta *= lapackSmlnum
	}
	*alpha = beta
	return tau
}

// dlartgp generates a plane rotation with a non-negative r:
// [cs sn; -sn cs] * [f; g] = [r; 0].
func dlartgp(f, g float64) (cs, sn, r float64) {
	const safmn2 = 0x1p-484 // 2^int(log2(safmin/eps)/2)
	const safmx2 = 1 / safmn2
	switch {
	case g == 0:
		return math.Copysign(1, f), 0, math.Abs(f)
	case f == 0:
		return 0, math.Copysign(1, g), math.Abs(g)
	}
	f1, g1 := f, g
	scale := math.Max(math.Abs(f1), math.Abs(g1))
	count := 0
	switch {
	case scale >= safmx2:
		for {
			count++
			f1 *= safmn2
			g1 *= safmn2
			scale = math.Max(math.Abs(f1), math.Abs(g1))
			if !(scale >= safmx2 && count < 20) {
				break
			}
		}
		r = math.Sqrt(float64(f1*f1) + float64(g1*g1))
		cs, sn = f1/r, g1/r
		for i := 0; i < count; i++ {
			r *= safmx2
		}
	case scale <= safmn2:
		for {
			count++
			f1 *= safmx2
			g1 *= safmx2
			scale = math.Max(math.Abs(f1), math.Abs(g1))
			// Reference LAPACK has no count limit here; a NaN-free finite
			// non-zero input leaves the loop after at most 3 rounds.
			if !(scale <= safmn2 && count < 20) {
				break
			}
		}
		r = math.Sqrt(float64(f1*f1) + float64(g1*g1))
		cs, sn = f1/r, g1/r
		for i := 0; i < count; i++ {
			r *= safmn2
		}
	default:
		r = math.Sqrt(float64(f1*f1) + float64(g1*g1))
		cs, sn = f1/r, g1/r
	}
	// LAPACK's final sign flip for r < 0 is omitted: r is a square root.
	return cs, sn, r
}

func scal(n int, a float64, x []float64, inc int) {
	for i := 0; i < n; i++ {
		x[i*inc] *= a
	}
}

// dlarfLeft applies H = I - tau*v*v' from the left to the m×n matrix C
// (leading dimension ldc); v has stride incv.
func dlarfLeft(m, n int, v []float64, incv int, tau float64, c []float64, ldc int) {
	if tau == 0 {
		return
	}
	for j := 0; j < n; j++ {
		col := c[j*ldc:]
		w := 0.0
		for i := 0; i < m; i++ {
			w += float64(col[i] * v[i*incv])
		}
		t := float64(-tau * w)
		for i := 0; i < m; i++ {
			col[i] += float64(v[i*incv] * t)
		}
	}
}

// dlarfRight applies H = I - tau*v*v' from the right to the m×n matrix C.
func dlarfRight(m, n int, v []float64, incv int, tau float64, c []float64, ldc int, work []float64) {
	if tau == 0 {
		return
	}
	w := work[:m]
	for i := range w {
		w[i] = 0
	}
	for j := 0; j < n; j++ {
		t := v[j*incv]
		col := c[j*ldc:]
		for i := 0; i < m; i++ {
			w[i] += float64(t * col[i])
		}
	}
	for j := 0; j < n; j++ {
		t := float64(-tau * v[j*incv])
		col := c[j*ldc:]
		for i := 0; i < m; i++ {
			col[i] += float64(w[i] * t)
		}
	}
}

// dgeqr2 computes the QR factorisation of the m×n matrix A.
func dgeqr2(m, n int, a []float64, lda int, tau []float64) {
	k := min(m, n)
	for i := 0; i < k; i++ {
		tau[i] = dlarfg(m-i, &a[i+i*lda], a[min(i+1, m-1)+i*lda:], 1)
		if i < n-1 {
			aii := a[i+i*lda]
			a[i+i*lda] = 1
			dlarfLeft(m-i, n-i-1, a[i+i*lda:], 1, tau[i], a[i+(i+1)*lda:], lda)
			a[i+i*lda] = aii
		}
	}
}

// dorm2rLT overwrites the m×n matrix C with Q'*C, Q from dgeqr2 (k reflectors).
func dorm2rLT(m, n, k int, a []float64, lda int, tau, c []float64, ldc int) {
	for i := 0; i < k; i++ {
		aii := a[i+i*lda]
		a[i+i*lda] = 1
		dlarfLeft(m-i, n, a[i+i*lda:], 1, tau[i], c[i:], ldc)
		a[i+i*lda] = aii
	}
}

// dgerq2 computes the RQ factorisation of the m×n matrix A.
func dgerq2(m, n int, a []float64, lda int, tau, work []float64) {
	k := min(m, n)
	for i := k - 1; i >= 0; i-- {
		r, c := m-k+i, n-k+i
		tau[i] = dlarfg(c+1, &a[r+c*lda], a[r:], lda)
		aii := a[r+c*lda]
		a[r+c*lda] = 1
		dlarfRight(r, c+1, a[r:], lda, tau[i], a, lda, work)
		a[r+c*lda] = aii
	}
}

// dormr2RT overwrites the m×n matrix C with C*Q', Q from dgerq2 of the k×n
// matrix A.
func dormr2RT(m, n, k int, a []float64, lda int, tau, c []float64, ldc int, work []float64) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	for i := k - 1; i >= 0; i-- {
		ni := n - k + i + 1
		p := i + (ni-1)*lda
		aii := a[p]
		a[p] = 1
		dlarfRight(m, ni, a[i:], lda, tau[i], c, ldc, work)
		a[p] = aii
	}
}

// dormr2LTvec overwrites the n-vector c with Q'*c, Q from dgerq2 of the k×n
// matrix A.
func dormr2LTvec(n, k int, a []float64, lda int, tau, c []float64) {
	for i := 0; i < k; i++ {
		mi := n - k + i + 1
		p := i + (mi-1)*lda
		aii := a[p]
		a[p] = 1
		dlarfLeft(mi, 1, a[i:], lda, tau[i], c, n)
		a[p] = aii
	}
}

// gemvN computes y = alpha*A*x + beta*y for the m×n matrix A (reference BLAS
// column order).
func gemvN(m, n int, alpha float64, a []float64, lda int, x []float64, beta float64, y []float64) {
	if m == 0 || n == 0 {
		return
	}
	// SLSQP only calls it with beta = 1 or -1.
	if beta != 1 {
		for i := 0; i < m; i++ {
			y[i] *= beta
		}
	}
	for j := 0; j < n; j++ {
		t := float64(alpha * x[j])
		col := a[j*lda:]
		for i := 0; i < m; i++ {
			y[i] += float64(t * col[i])
		}
	}
}

// gemvT computes y = alpha*A'*x + beta*y for the m×n matrix A.
func gemvT(m, n int, alpha float64, a []float64, lda int, x []float64, beta float64, y []float64) {
	if m == 0 || n == 0 {
		return
	}
	// SLSQP only calls it with beta = 0 or 1.
	for j := 0; j < n; j++ {
		if beta == 0 {
			y[j] = 0
		}
		t := 0.0
		col := a[j*lda:]
		for i := 0; i < m; i++ {
			t += float64(col[i] * x[i])
		}
		y[j] += float64(alpha * t)
	}
}

// trsvUN solves U*x = b in place, U upper triangular n×n.
func trsvUN(n int, a []float64, lda int, x []float64) {
	for j := n - 1; j >= 0; j-- {
		if x[j] != 0 {
			x[j] /= a[j+j*lda]
			t := x[j]
			for i := j - 1; i >= 0; i-- {
				x[i] -= float64(t * a[i+j*lda])
			}
		}
	}
}

// trsvUT solves U'*x = b in place.
func trsvUT(n int, a []float64, lda int, x []float64) {
	for j := 0; j < n; j++ {
		t := x[j]
		for i := 0; i < j; i++ {
			t -= float64(a[i+j*lda] * x[i])
		}
		x[j] = t / a[j+j*lda]
	}
}

// trsmRUN solves X*U = B in place for the m×n matrix B, U upper n×n.
func trsmRUN(m, n int, a []float64, lda int, b []float64, ldb int) {
	for j := 0; j < n; j++ {
		for k := 0; k < j; k++ {
			if akj := a[k+j*lda]; akj != 0 {
				for i := 0; i < m; i++ {
					b[i+j*ldb] -= float64(akj * b[i+k*ldb])
				}
			}
		}
		t := 1 / a[j+j*lda]
		for i := 0; i < m; i++ {
			b[i+j*ldb] = float64(t * b[i+j*ldb])
		}
	}
}

// tpsvLNU solves L*x = b in place, L unit lower triangular in packed storage.
func tpsvLNU(n int, ap, x []float64) {
	kk := 0
	for j := 0; j < n; j++ {
		if x[j] != 0 {
			t := x[j]
			k := kk + 1
			for i := j + 1; i < n; i++ {
				x[i] -= float64(t * ap[k])
				k++
			}
		}
		kk += n - j
	}
}

// tpmvLTU computes x = L'*x, L unit lower triangular packed.
func tpmvLTU(n int, ap, x []float64) {
	kk := 0
	for j := 0; j < n; j++ {
		t := x[j]
		k := kk + 1
		for i := j + 1; i < n; i++ {
			t += float64(ap[k] * x[i])
			k++
		}
		x[j] = t
		kk += n - j
	}
}

// tpmvLNU computes x = L*x, L unit lower triangular packed.
func tpmvLNU(n int, ap, x []float64) {
	kk := n*(n+1)/2 - 1
	for j := n - 1; j >= 0; j-- {
		if x[j] != 0 {
			t := x[j]
			k := kk
			for i := n - 1; i > j; i-- {
				x[i] += float64(t * ap[k])
				k--
			}
		}
		kk -= n - j
	}
}

// lstsqPivoted solves min |A*x - b| for the m×n matrix A (overwritten) by
// Householder QR with column pivoting, the factorisation LAPACK's dgelsy
// starts with. The rank is the number of leading diagonal entries of R with
// |R(k,k)| > rcond*|R(0,0)|, a simplification of dgelsy's incremental
// condition estimate; in SLSQP A is a factor of the positive-definite BFGS
// matrix, so the rank test only fires on numerically singular problems. On
// return b[:n] holds the basic solution (zero in the rank-deficient part).
func lstsqPivoted(m, n int, a []float64, b []float64, rcond float64, work []float64) (rank int) {
	k := min(m, n)
	jpvt := make([]int, n)
	norms := work[:n]
	tau := work[n : n+k]
	for j := 0; j < n; j++ {
		jpvt[j] = j
		norms[j] = nrm2(m, a[j*m:], 1)
	}
	for i := 0; i < k; i++ {
		p := i
		for j := i + 1; j < n; j++ {
			if norms[j] > norms[p] {
				p = j
			}
		}
		if p != i {
			for r := 0; r < m; r++ {
				a[r+p*m], a[r+i*m] = a[r+i*m], a[r+p*m]
			}
			jpvt[p], jpvt[i] = jpvt[i], jpvt[p]
			norms[p] = norms[i]
		}
		tau[i] = dlarfg(m-i, &a[i+i*m], a[min(i+1, m-1)+i*m:], 1)
		aii := a[i+i*m]
		a[i+i*m] = 1
		if i < n-1 {
			dlarfLeft(m-i, n-i-1, a[i+i*m:], 1, tau[i], a[i+(i+1)*m:], m)
		}
		dlarfLeft(m-i, 1, a[i+i*m:], 1, tau[i], b[i:], m)
		a[i+i*m] = aii
		for j := i + 1; j < n; j++ {
			norms[j] = nrm2(m-i-1, a[i+1+j*m:], 1)
		}
	}
	for rank < k && math.Abs(a[rank+rank*m]) > float64(rcond*math.Abs(a[0])) {
		rank++
	}
	z := make([]float64, n)
	copy(z, b[:rank])
	trsvUN(rank, a, m, z)
	for j := 0; j < n; j++ {
		b[jpvt[j]] = z[j]
	}
	return rank
}
