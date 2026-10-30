package optimize

import "math"

// The least-squares subproblem solvers of SLSQP, ported from SciPy v1.18.1
// scipy/optimize/src/slsqp.c (lsq, lsei, lsi, ldp, ldl_update) and
// scipy/optimize/src/nnls.c (__nnls), BSD-3-Clause. Those are SciPy's C
// rewrites of Dieter Kraft's SLSQP (DFVLR-FB 88-28, 1988) and of the
// Lawson–Hanson NNLS/LSEI routines it builds on. Matrices are column-major.

// lsq solves the SQP search-direction subproblem
//
//	min |A*x - b|  s.t.  E*x = f,  G*x >= h,  xl <= x <= xu
//
// where A and b are recovered from the packed LDL' factor lf of the BFGS
// matrix and the gradient (A = sqrt(D) L', b = -inv(L sqrt(D)) gradx), E and f
// are the first meq rows of the m×n constraint matrix c and of d, and G and h
// its other rows and the finite bounds. NaN bounds are absent. With augment
// the problem gains the variable of Kraft's §2.2.3 for inconsistent
// linearisations, weighted by augWeight; xl, xu and x must then have n+1
// entries. x receives the step and y the multipliers (m + 2(n+1) entries).
// It returns SLSQP's mode.
func lsq(ws *arena, m, meq, n int, augment bool, augWeight float64, lf, gradx, c []float64, ldc int,
	d, xl, xu, x, y []float64) int {
	ws.off = 0
	orign := n
	mineq := m - meq
	ld := n
	if augment {
		ld = n + 1
		x[n] = 1
		xl[n] = 0
		xu[n] = 1
	}
	wA := ws.take(ld * ld)
	wb := ws.take(ld)
	cursor := 0
	for j := 0; j < n; j++ {
		diag := math.Sqrt(lf[cursor])
		cursor++
		wA[j+j*ld] = diag
		for i := j + 1; i < n; i++ {
			wA[j+i*ld] = float64(lf[cursor] * diag)
			cursor++
		}
	}
	copy(wb, gradx[:n])
	tpsvLNU(n, lf, wb)
	cursor = 0
	for i := 0; i < n; i++ {
		wb[i] /= -math.Sqrt(lf[cursor])
		cursor += n - i
	}
	if augment {
		wA[ld*ld-1] = augWeight
		n++
	}

	wE := ws.take(n * meq)
	wf := ws.take(meq)
	for j := 0; j < orign; j++ {
		for i := 0; i < meq; i++ {
			wE[i+j*meq] = c[i+j*ldc]
		}
	}
	if augment {
		for i := 0; i < meq; i++ {
			wE[i+(n-1)*meq] = -d[i]
		}
	}
	for i := 0; i < meq; i++ {
		wf[i] = -d[i]
	}

	wh := ws.take(mineq + 2*n)
	nrow := mineq
	for i := 0; i < mineq; i++ {
		wh[i] = -d[meq+i]
	}
	for i := 0; i < n; i++ {
		if !math.IsNaN(xl[i]) {
			wh[nrow] = xl[i]
			nrow++
		}
	}
	for i := 0; i < n; i++ {
		if !math.IsNaN(xu[i]) {
			wh[nrow] = -xu[i]
			nrow++
		}
	}
	nG := nrow
	wh = wh[:nG]
	wG := ws.take(nG * n)
	for j := 0; j < orign; j++ {
		for i := 0; i < mineq; i++ {
			wG[i+j*nG] = c[meq+i+j*ldc]
		}
	}
	if augment {
		for i := 0; i < mineq; i++ {
			wG[i+orign*nG] = math.Max(-d[meq+i], 0)
		}
	}
	nrow = mineq
	for i := 0; i < n; i++ {
		if !math.IsNaN(xl[i]) {
			wG[nrow+i*nG] = 1
			nrow++
		}
	}
	for i := 0; i < n; i++ {
		if !math.IsNaN(xu[i]) {
			wG[nrow+i*nG] = -1
			nrow++
		}
	}

	mode, emults, gmults := lsei(ws, ld, meq, nG, n, wA, wb, wE, wf, wG, wh, x)
	if mode == 1 {
		copy(y[:meq], emults)
		copy(y[meq:meq+mineq], gmults[:mineq])
		for i := 0; i < 2*n; i++ {
			y[m+i] = math.NaN()
		}
	}
	clampNaNBounds(x[:n], xl, xu)
	return mode
}

// arena hands out zeroed scratch vectors from one reusable buffer, so that
// the subproblem solvers allocate nothing once it has grown; lsq resets it.
// Growing replaces the buffer, which leaves earlier slices valid.
type arena struct {
	buf []float64
	off int
}

func (a *arena) take(n int) []float64 {
	if a.off+n > len(a.buf) {
		a.buf = make([]float64, max(2*len(a.buf), a.off+n, 64))
		a.off = 0
	}
	s := a.buf[a.off : a.off+n : a.off+n]
	a.off += n
	clear(s)
	return s
}

// clampNaNBounds clips x into [xl, xu], where a NaN bound is absent.
func clampNaNBounds(x, xl, xu []float64) {
	for i := range x {
		if !math.IsNaN(xl[i]) && x[i] < xl[i] {
			x[i] = xl[i]
		} else if !math.IsNaN(xu[i]) && x[i] > xu[i] {
			x[i] = xu[i]
		}
	}
}

// lsei solves min |A*x - b| s.t. E*x = f, G*x >= h for A ma×n, E me×n and G
// mg×n (all column-major, overwritten, as are b, f and h). It returns the
// mode and, when the mode is 1, the multipliers of E and of G.
func lsei(ws *arena, ma, me, mg, n int, a, b, e, f, g, h, x []float64) (mode int, emults, gmults []float64) {
	for i := 0; i < n; i++ {
		x[i] = 0
	}
	if me > n {
		return 2, nil, nil
	}
	nvars := n - me
	lde, ldg := max(me, 1), max(mg, 1)
	tau := ws.take(me)
	work := ws.take(max(ma, mg, me, 1))
	dgerq2(me, n, e, lde, tau, work)
	dormr2RT(ma, n, me, e, lde, tau, a, ma, work)
	dormr2RT(mg, n, me, e, lde, tau, g, ldg, work)
	for i := 0; i < me; i++ {
		if !(math.Abs(e[i+(nvars+i)*me]) >= epsmch) {
			return 6, nil, nil
		}
	}
	copy(x[nvars:nvars+me], f[:me])
	if me > 0 {
		trsvUN(me, e[nvars*me:], lde, x[nvars:])
	}
	mode = 1
	gmults = ws.take(mg)
	if me != n {
		wb := ws.take(ma)
		copy(wb, b[:ma])
		gemvN(ma, me, -1, a[ma*nvars:], ma, x[nvars:], 1, wb)
		a2 := ws.take(ma * nvars)
		copy(a2, a[:ma*nvars])
		if mg == 0 {
			// SciPy calls dgelsy here and also computes the residual norm,
			// which SLSQP never reads; only the rank decides the mode.
			rank := lstsqPivoted(ma, nvars, a2, wb, math.Sqrt(epsmch), ws.take(2*nvars))
			copy(x[:nvars], wb[:nvars])
			if rank < nvars {
				return 7, nil, nil
			}
		} else {
			g2 := ws.take(mg * nvars)
			copy(g2, g[:mg*nvars])
			gemvN(mg, me, -1, g[mg*nvars:], ldg, x[nvars:], 1, h)
			var lm []float64
			mode, lm = lsi(ws, ma, mg, nvars, a2, wb, g2, h, x)
			if mode != 1 {
				return mode, nil, nil
			}
			copy(gmults, lm)
			if me == 0 {
				return 1, nil, gmults
			}
		}
	}
	// Convert the solution and multipliers to the original basis.
	gemvN(ma, n, 1, a, ma, x, -1, b)
	gemvT(ma, me, 1, a[nvars*ma:], ma, b, 0, f)
	gemvT(mg, me, -1, g[nvars*mg:], ldg, gmults, 1, f)
	dormr2LTvec(n, me, e, lde, tau, x)
	emults = ws.take(me)
	copy(emults, f[:me])
	trsvUT(me, e[(n-me)*me:], lde, emults)
	return 1, emults, gmults
}

// lsi solves min |A*x - b| s.t. G*x >= h for A ma×n and G mg×n, returning the
// mode and, when it is 1, the multipliers of G.
func lsi(ws *arena, ma, mg, n int, a, b, g, h, x []float64) (mode int, mults []float64) {
	k := min(ma, n)
	tau := ws.take(k)
	dgeqr2(ma, n, a, ma, tau)
	dorm2rLT(ma, 1, k, a, ma, tau, b, ma)
	for i := 0; i < k; i++ {
		if !(math.Abs(a[i+i*ma]) >= epsmch) {
			return 5, nil
		}
	}
	trsmRUN(mg, n, a, ma, g, mg)
	gemvN(mg, n, -1, g, mg, b, 1, h)
	mode, mults = ldp(ws, mg, n, g, h, x)
	if mode != 1 {
		return mode, nil
	}
	for i := 0; i < n; i++ {
		x[i] += b[i]
	}
	trsvUN(n, a, ma, x)
	return 1, mults
}

// ldpMaxIter*m bounds the NNLS inner iterations of ldp (3m in SciPy); tests
// lower it to reach the iteration-limit exit.
var ldpMaxIter = 3

// ldp solves the least-distance problem min |x| s.t. G*x >= h through the
// NNLS dual of Lawson and Hanson. It returns the mode and, when it is 1, the
// multipliers.
func ldp(ws *arena, m, n int, g, h, x []float64) (mode int, mults []float64) {
	// lsei only reaches ldp with m >= 1 (m == 0 takes its least-squares
	// branch), so SciPy's m == 0 and n <= 0 early returns are not needed.
	for i := 0; i < n; i++ {
		x[i] = 0
	}
	n1 := n + 1
	a := ws.take(n1 * m)
	for j := 0; j < m; j++ {
		for i := 0; i < n; i++ {
			a[i+j*n1] = g[j+i*m]
		}
		a[n+j*n1] = h[j]
	}
	b := ws.take(n1)
	b[n] = 1
	y := ws.take(m)
	rnorm, info := nnls(ws, n1, m, a, b, y, ldpMaxIter*m)
	if info != 1 {
		return info, nil
	}
	if rnorm <= 0 {
		return 4, nil
	}
	fac := 1 - dot(h[:m], y)
	if !((1+fac)-1 > 0) {
		return 4, nil
	}
	fac = 1 / fac
	gemvT(m, n, fac, g, m, y, 0, x)
	mults = ws.take(m)
	for i := range mults {
		mults[i] = float64(fac * y[i])
	}
	return 1, mults
}

// nnls solves min |A*x - b| s.t. x >= 0 for the m×n matrix A (overwritten,
// as is b) by the Lawson–Hanson active-set method, SciPy's __nnls. It
// returns the residual norm and info: 1 on success, 3 when maxiter inner
// iterations were exceeded.
func nnls(ws *arena, m, n int, a, b, x []float64, maxiter int) (rnorm float64, info int) {
	info = 1
	w := ws.take(n)
	zz := ws.take(m)
	indices := make([]int, n)
	for i := 0; i < n; i++ {
		indices[i] = i
		x[i] = 0
	}
	var (
		indz, iteration, iz, izmax, j, jj int
		tau, pivot                        float64
	)
	for indz < min(m, n) {
		for i := indz; i < n; i++ {
			j = indices[i]
			w[j] = dot(a[indz+j*m:(j+1)*m], b[indz:m])
		}
		for {
			wmax := 0.0
			for k := indz; k < n; k++ {
				j = indices[k]
				if w[j] > wmax {
					wmax, izmax = w[j], k
				}
			}
			if wmax <= 0 {
				goto END
			}
			iz = izmax
			j = indices[iz]
			pivot = a[indz+j*m]
			tau = dlarfgp(m-indz, &pivot, a[min(indz+1, m-1)+j*m:], 1)
			unorm := 0.0
			if indz > 0 {
				unorm = nrm2(indz, a[j*m:], 1)
			}
			spacing := 0.0
			if unorm > 0 {
				spacing = math.Nextafter(unorm, 2*unorm) - unorm
			}
			if math.Abs(pivot) > float64(100*spacing) {
				copy(zz, b[:m])
				pivot2 := a[indz+j*m]
				a[indz+j*m] = 1
				dlarfLeft(m-indz, 1, a[indz+j*m:], 1, tau, zz[indz:], m-indz)
				if zz[indz]/pivot > 0 {
					break
				}
				a[indz+j*m] = pivot2
			}
			w[j] = 0
		}
		copy(b[:m], zz)
		indices[iz] = indices[indz]
		indices[indz] = j
		indz++
		for k := indz; k < n; k++ {
			jj = indices[k]
			dlarfLeft(m-indz+1, 1, a[indz-1+j*m:], 1, tau, a[indz-1+jj*m:], m-indz+1)
		}
		a[indz-1+j*m] = pivot
		for i := indz; i < m; i++ {
			a[j*m+i] = 0
		}
		w[j] = 0
		nnlsSolve(m, indz, a, indices, zz)

		for {
			iteration++
			if iteration >= maxiter {
				info = 3
				goto END
			}
			alpha := 2.0
			for ip := 0; ip < indz; ip++ {
				k := indices[ip]
				if zz[ip] <= 0 {
					t := -x[k] / (zz[ip] - x[k])
					if alpha > t {
						alpha, jj = t, ip
					}
				}
			}
			if alpha == 2 {
				break
			}
			for ip := 0; ip < indz; ip++ {
				k := indices[ip]
				x[k] += float64(alpha * (zz[ip] - x[k]))
			}
			i := indices[jj]
			for {
				x[i] = 0
				if jj != indz-1 {
					jj++
					for jx := jj; jx < indz; jx++ {
						ii := indices[jx]
						indices[jx-1] = ii
						cc, ss, r := dlartgp(a[jx-1+ii*m], a[jx+ii*m])
						a[jx-1+ii*m] = r
						a[jx+ii*m] = 0
						for k := 0; k < n; k++ {
							if k != ii {
								t := a[jx-1+k*m]
								a[jx-1+k*m] = float64(cc*t) + float64(ss*a[jx+k*m])
								a[jx+k*m] = float64(-ss*t) + float64(cc*a[jx+k*m])
							}
						}
						t := b[jx-1]
						b[jx-1] = float64(cc*t) + float64(ss*b[jx])
						b[jx] = float64(-ss*t) + float64(cc*b[jx])
					}
				}
				indz--
				indices[indz] = i
				feasible := indz > 0
				for jj = 0; jj < indz; jj++ {
					i = indices[jj]
					if x[i] <= 0 {
						feasible = false
						break
					}
				}
				// SciPy's loop would index out of range once the passive set
				// is empty; it is left here instead (Go would panic).
				if feasible || indz == 0 {
					break
				}
			}
			copy(zz, b[:m])
			nnlsSolve(m, indz, a, indices, zz)
		}
		for k := 0; k < indz; k++ {
			x[indices[k]] = zz[k]
		}
	}
END:
	if indz < m {
		rnorm = nrm2(m-indz, b[indz:], 1)
	}
	return rnorm, info
}

// nnlsSolve solves the permuted triangular system of the passive set into zz.
func nnlsSolve(m, indz int, a []float64, indices []int, zz []float64) {
	jj := 0
	for k := 0; k < indz; k++ {
		ip := indz - 1 - k
		if k != 0 {
			for i := 0; i <= ip; i++ {
				zz[i] -= float64(a[i+jj*m] * zz[ip+1])
			}
		}
		jj = indices[ip]
		zz[ip] /= a[ip+jj*m]
	}
}

// ldlUpdate updates the packed LDL' factors of a by the rank-one matrix
// sigma*z*z' (Fletcher and Powell's composite-t method, as in Kraft's code);
// w is workspace used when sigma < 0. z is overwritten.
func ldlUpdate(n int, a, z []float64, sigma float64, w []float64) {
	// The C code returns early for sigma == 0; SLSQP never passes it (it
	// resets the matrix when h1 or h2 is zero), so that test is omitted.
	ij := 0
	t := 1 / sigma
	if sigma < 0 {
		copy(w[:n], z[:n])
		for i := 0; i < n; i++ {
			v := w[i]
			t += float64(v*v) / a[ij]
			for j := i + 1; j < n; j++ {
				ij++
				w[j] -= float64(v * a[ij])
			}
			ij++
		}
		if t >= 0 {
			t = epsmch / sigma
		}
		for i := 0; i < n; i++ {
			j := n - i - 1
			ij -= i + 1
			u := w[j]
			w[j] = t
			t -= float64(u*u) / a[ij]
		}
	}
	for i := 0; i < n; i++ {
		v := z[i]
		delta := v / a[ij]
		var tp float64
		if sigma < 0 {
			tp = w[i]
		} else {
			tp = t + float64(delta*v)
		}
		alpha := tp / t
		a[ij] = float64(alpha * a[ij])
		if i == n-1 {
			return
		}
		beta := delta / tp
		if alpha <= 4 {
			for j := i + 1; j < n; j++ {
				ij++
				z[j] -= float64(v * a[ij])
				a[ij] += float64(beta * z[j])
			}
		} else {
			gamma := t / tp
			for j := i + 1; j < n; j++ {
				ij++
				u := a[ij]
				a[ij] = float64(gamma*u) + float64(beta*z[j])
				z[j] -= float64(v * u)
			}
		}
		ij++
		t = tp
	}
}
