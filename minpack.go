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
func qrfac(m, n int, a []float64, lda int, pivot bool, ipvt []int, rdiag, acnorm, wa []float64) {
	for j := 0; j < n; j++ {
		acnorm[j] = mpEnorm(a[j*lda : j*lda+m])
		rdiag[j] = acnorm[j]
		wa[j] = rdiag[j]
		if pivot {
			ipvt[j] = j
		}
	}
	minmn := min(m, n)
	for j := 0; j < minmn; j++ {
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
		}
		// The Householder transformation that reduces column j to a multiple
		// of the j-th unit vector.
		ajnorm := mpEnorm(a[lda*j+j : lda*j+m])
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
		// Apply it to the remaining columns and update their norms.
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
					rdiag[k] = mpEnorm(a[(j+1)+lda*k : m+lda*k])
					wa[k] = rdiag[k]
				}
			}
		}
		rdiag[j] = -ajnorm
	}
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
