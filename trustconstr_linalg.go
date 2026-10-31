package optimize

import (
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/linalg"
	"github.com/go-simd/floats"
)

// Dense kernels of the trust-constr method, after SciPy v1.18.1
// scipy/optimize/_trustregion_constr/{projections,qp_subproblem}.py and
// _hessian_update_strategy.py (BSD-3-Clause). Matrices are row-major. The
// element-wise updates use go-simd/floats (exact fused multiply-adds, the same
// on every architecture); reductions use floats.Dot, whose lane-blocked order
// is not SciPy's — trust-constr is validated against SciPy on solutions and
// statuses, not on bit-identical iterates.

func vnorm(x []float64) float64 { return math.Sqrt(floats.Dot(x, x)) }

// sqNorm is SciPy's norm(x)**2: the square of the rounded norm, which is not
// the dot product x·x (projected_cg's rt_g, the trust radius of the
// tangential step).
func sqNorm(x []float64) float64 {
	n := vnorm(x)
	return float64(n * n)
}

func vnormInf(x []float64) float64 {
	m := 0.0
	for _, v := range x {
		m = math.Max(m, math.Abs(v))
	}
	return m
}

// matvec writes A·x into out for the m×n row-major matrix A.
func matvec(m, n int, a, x, out []float64) {
	for i := 0; i < m; i++ {
		out[i] = floats.Dot(a[i*n:(i+1)*n], x)
	}
}

// matTvec writes Aᵀ·v into out (length n).
func matTvec(m, n int, a, v, out []float64) {
	clear(out[:n])
	for i := 0; i < m; i++ {
		if v[i] != 0 {
			floats.Axpy(v[i], a[i*n:(i+1)*n], out[:n])
		}
	}
}

// projector holds the three operators of projections(A): the null-space
// projection Z = I - Aᵀ(AAᵀ)⁻¹A, the least-squares operator LS = (AAᵀ)⁻¹A and
// the row-space operator Y = Aᵀ(AAᵀ)⁻¹, for the m×n matrix A (m <= n).
type projector struct {
	m, n int
	a    []float64
	// QR factorisation with column pivoting of Aᵀ: Householder vectors in
	// the columns of qr (n×m, column-major, unit diagonal implied), R in
	// its upper triangle, permutation perm.
	qr   []float64
	tau  []float64
	perm []int
	// SVD fallback (rank-deficient A): U (m×k), s (k), Vt (k×n), row-major.
	svd      bool
	u, s, vt []float64
	k        int
	normA    float64
	work     []float64
}

const (
	projOrthTol  = 1e-12
	projMaxRefin = 3
	projTol      = 1e-15
)

// newProjector factorises A. It reports false when A has more rows than
// columns, which SciPy rejects ("more equality constraints than
// independent variables").
func newProjector(m, n int, a []float64) (*projector, bool) {
	p := &projector{m: m, n: n, a: a, work: make([]float64, max(m, n))}
	if m == 0 {
		return p, true
	}
	if m > n {
		return nil, false
	}
	p.normA = vnorm(a)
	// Aᵀ is n×m; store it column-major: column j of Aᵀ is row j of A.
	p.qr = append([]float64(nil), a...)
	p.tau = make([]float64, m)
	p.perm = make([]int, m)
	// Householder QR with column pivoting, LAPACK dlaqp2's algorithm: the
	// partial column norms are downdated (LAPACK Working Note 176) and
	// recomputed when cancellation makes the downdate unreliable.
	vn1, vn2 := make([]float64, m), make([]float64, m)
	tol3z := math.Sqrt(epsmch)
	for j := range m {
		p.perm[j] = j
		vn1[j] = vnorm(p.qr[j*n : (j+1)*n])
		vn2[j] = vn1[j]
	}
	for i := 0; i < m; i++ {
		piv := i
		for j := i + 1; j < m; j++ {
			if vn1[j] > vn1[piv] {
				piv = j
			}
		}
		if piv != i {
			ci, cp := p.qr[i*n:(i+1)*n], p.qr[piv*n:(piv+1)*n]
			for r := range ci {
				ci[r], cp[r] = cp[r], ci[r]
			}
			p.perm[i], p.perm[piv] = p.perm[piv], p.perm[i]
			vn1[piv], vn2[piv] = vn1[i], vn2[i]
		}
		col := p.qr[i*n : (i+1)*n]
		p.tau[i] = dlarfg(n-i, &col[i], col[min(i+1, n-1):], 1)
		aii := col[i]
		col[i] = 1
		for j := i + 1; j < m; j++ {
			householder(col[i:], p.tau[i], p.qr[j*n+i:(j+1)*n])
		}
		col[i] = aii
		for j := i + 1; j < m; j++ {
			if vn1[j] == 0 {
				continue
			}
			r := math.Abs(p.qr[j*n+i]) / vn1[j]
			temp := math.Max(1-r*r, 0)
			q := vn1[j] / vn2[j]
			if temp*q*q <= tol3z {
				vn1[j] = vnorm(p.qr[j*n+i+1 : (j+1)*n])
				vn2[j] = vn1[j]
			} else {
				vn1[j] *= math.Sqrt(temp)
			}
		}
	}
	// Singular Jacobian (the last row of R, whose only entry is R(m-1,m-1),
	// below 1e-15): SciPy warns and switches to the SVD.
	if !(math.Abs(p.rAt(m-1, m-1)) >= projTol) {
		p.initSVD()
	}
	return p, true
}

// householder applies I - tau·v·vᵀ to c with go-simd/floats kernels (the
// projections need not round like SciPy's, unlike SLSQP's dlarfLeft).
func householder(v []float64, tau float64, c []float64) {
	if tau == 0 {
		return
	}
	floats.Axpy(-tau*floats.Dot(v, c), v, c)
}

// rAt returns R(i, j) for i <= j (the callers never ask below the diagonal).
func (p *projector) rAt(i, j int) float64 { return p.qr[j*p.n+i] }

// svdFunc is go-ndarray's SVD; tests replace it to reach the failure path.
var svdFunc = linalg.SVD

func (p *projector) initSVD() {
	p.svd = true
	arr, _ := ndarray.FromData(append([]float64(nil), p.a...), p.m, p.n)
	u, s, vt, err := svdFunc(arr, false)
	if err != nil {
		// A non-converging SVD keeps no direction: Z = I, LS = Y = 0.
		p.k = 0
		return
	}
	ud, sd, vd := arrayData(u), arrayData(s), arrayData(vt)
	kk := len(sd)
	var keep []int
	for j := range kk {
		if sd[j] > projTol {
			keep = append(keep, j)
			p.s = append(p.s, sd[j])
			p.vt = append(p.vt, vd[j*p.n:(j+1)*p.n]...)
		}
	}
	p.k = len(keep)
	p.u = make([]float64, p.m*p.k) // m×k, row-major
	for c, j := range keep {
		for i := 0; i < p.m; i++ {
			p.u[i*p.k+c] = ud[i*kk+j]
		}
	}
}

// qtx writes the first m entries of Qᵀx into out.
func (p *projector) qtx(x, out []float64) {
	w := p.work[:p.n]
	copy(w, x)
	for i := 0; i < p.m; i++ {
		col := p.qr[i*p.n : (i+1)*p.n]
		aii := col[i]
		col[i] = 1
		householder(col[i:], p.tau[i], w[i:])
		col[i] = aii
	}
	copy(out, w[:p.m])
}

// qy writes Q·[y; 0] (length n) into out.
func (p *projector) qy(y, out []float64) {
	clear(out[:p.n])
	copy(out, y[:p.m])
	for i := p.m - 1; i >= 0; i-- {
		col := p.qr[i*p.n : (i+1)*p.n]
		aii := col[i]
		col[i] = 1
		householder(col[i:], p.tau[i], out[i:p.n])
		col[i] = aii
	}
}

// solveR solves R·z = y in place (trans: Rᵀ·z = y).
func (p *projector) solveR(y []float64, trans bool) {
	m := p.m
	if !trans {
		for i := m - 1; i >= 0; i-- {
			t := y[i]
			for j := i + 1; j < m; j++ {
				t -= float64(p.rAt(i, j) * y[j])
			}
			y[i] = t / p.rAt(i, i)
		}
		return
	}
	for i := 0; i < m; i++ {
		t := y[i]
		for j := 0; j < i; j++ {
			t -= float64(p.rAt(j, i) * y[j])
		}
		y[i] = t / p.rAt(i, i)
	}
}

// leastSquares writes (AAᵀ)⁻¹A·x (length m) into out.
func (p *projector) leastSquares(x, out []float64) {
	if p.m == 0 {
		return
	}
	if p.svd {
		aux := make([]float64, p.k)
		matvec(p.k, p.n, p.vt, x, aux)
		for i := range aux {
			aux[i] /= p.s[i]
		}
		matvec(p.m, p.k, p.u, aux, out)
		return
	}
	aux := make([]float64, p.m)
	p.qtx(x, aux)
	p.solveR(aux, false)
	for i, j := range p.perm {
		out[j] = aux[i]
	}
}

// rowSpace writes Aᵀ(AAᵀ)⁻¹·x (length n) into out.
func (p *projector) rowSpace(x, out []float64) {
	if p.m == 0 {
		clear(out[:p.n])
		return
	}
	if p.svd {
		aux := make([]float64, p.k)
		for j := range p.k {
			t := 0.0
			for i := 0; i < p.m; i++ {
				t += float64(p.u[i*p.k+j] * x[i])
			}
			aux[j] = t / p.s[j]
		}
		matTvec(p.k, p.n, p.vt, aux, out)
		return
	}
	aux := make([]float64, p.m)
	for i, j := range p.perm {
		aux[i] = x[j]
	}
	p.solveR(aux, true)
	p.qy(aux, out)
}

// orthogonality is |A z| / (|A|_F |z|), zero when either norm is.
func (p *projector) orthogonality(z []float64) float64 {
	nz := vnorm(z)
	if nz == 0 || p.normA == 0 {
		return 0
	}
	az := make([]float64, p.m)
	matvec(p.m, p.n, p.a, z, az)
	return vnorm(az) / (p.normA * nz)
}

// nullSpace writes (I - Aᵀ(AAᵀ)⁻¹A)·x into out, with up to three steps of
// iterative refinement.
func (p *projector) nullSpace(x, out []float64) {
	copy(out, x[:p.n])
	if p.m == 0 {
		return
	}
	v := make([]float64, p.m)
	atv := make([]float64, p.n)
	step := func(src []float64) {
		p.leastSquares(src, v)
		matTvec(p.m, p.n, p.a, v, atv)
		for i := range out[:p.n] {
			out[i] = src[i] - atv[i]
		}
	}
	step(x)
	for k := 0; p.orthogonality(out) > projOrthTol && k < projMaxRefin; k++ {
		z := append([]float64(nil), out[:p.n]...)
		step(z)
	}
}

// sphereIntersections is sphere_intersections: the interval [ta, tb] of t
// for which z + t·d lies in the ball of radius delta (t in [0, 1] unless
// entireLine).
func sphereIntersections(z, d []float64, delta float64, entireLine bool) (ta, tb float64, ok bool) {
	// SciPy also handles an infinite radius; trust-constr's radii are
	// always finite, so that branch is omitted.
	if vnorm(d) == 0 {
		return 0, 0, false
	}
	a := floats.Dot(d, d)
	b := 2 * floats.Dot(z, d)
	c := floats.Dot(z, z) - delta*delta
	disc := b*b - 4*a*c
	if disc < 0 {
		return 0, 0, false
	}
	sq := math.Sqrt(disc)
	aux := b + math.Copysign(sq, b)
	ta, tb = -aux/(2*a), -2*c/aux
	if ta > tb {
		ta, tb = tb, ta
	}
	if entireLine {
		return ta, tb, true
	}
	if tb < 0 || ta > 1 {
		return 0, 0, false
	}
	return math.Max(0, ta), math.Min(1, tb), true
}

// boxIntersections is box_intersections for lb <= z + t·d <= ub.
func boxIntersections(z, d, lb, ub []float64, entireLine bool) (ta, tb float64, ok bool) {
	if vnorm(d) == 0 {
		return 0, 0, false
	}
	ta, tb = math.Inf(-1), math.Inf(1)
	for i := range z {
		if d[i] == 0 {
			if z[i] < lb[i] || z[i] > ub[i] {
				return 0, 0, false
			}
			continue
		}
		tl, tu := (lb[i]-z[i])/d[i], (ub[i]-z[i])/d[i]
		ta = math.Max(ta, math.Min(tl, tu))
		tb = math.Min(tb, math.Max(tl, tu))
	}
	ok = ta <= tb
	if !entireLine {
		if tb < 0 || ta > 1 {
			return 0, 0, false
		}
		ta, tb = math.Max(0, ta), math.Min(1, tb)
	}
	return ta, tb, ok
}

func boxSphereIntersections(z, d, lb, ub []float64, delta float64, entireLine bool) (ta, tb float64, ok bool) {
	tab, tbb, okb := boxIntersections(z, d, lb, ub, entireLine)
	tas, tbs, oks := sphereIntersections(z, d, delta, entireLine)
	ta, tb = math.Max(tab, tas), math.Min(tbb, tbs)
	return ta, tb, okb && oks && ta <= tb
}

func insideBox(x, lb, ub []float64) bool {
	for i := range x {
		if !(lb[i] <= x[i]) || !(x[i] <= ub[i]) {
			return false
		}
	}
	return true
}

func clipBox(x, lb, ub []float64) {
	for i := range x {
		x[i] = math.Min(math.Max(x[i], lb[i]), ub[i])
	}
}

// modifiedDogleg approximately minimises |A x + b|²/2 in the ball of radius
// delta and the box [lb, ub] (Byrd, Hribar and Nocedal 1999, pp. 885–886, as
// SciPy implements it).
func modifiedDogleg(p *projector, b []float64, delta float64, lb, ub []float64) []float64 {
	m, n := p.m, p.n
	newton := make([]float64, n)
	nb := make([]float64, m)
	for i := range b {
		nb[i] = -b[i]
	}
	p.rowSpace(nb, newton)
	if insideBox(newton, lb, ub) && vnorm(newton) <= delta {
		return newton
	}
	g := make([]float64, n)
	matTvec(m, n, p.a, b, g)
	ag := make([]float64, m)
	matvec(m, n, p.a, g, ag)
	cauchy := append([]float64(nil), g...)
	floats.Scale(-floats.Dot(g, g)/floats.Dot(ag, ag), cauchy)
	origin := make([]float64, n)

	pdir := make([]float64, n)
	for i := range pdir {
		pdir[i] = newton[i] - cauchy[i]
	}
	x1 := make([]float64, n)
	if _, alpha, ok := boxSphereIntersections(cauchy, pdir, lb, ub, delta, false); ok {
		copy(x1, cauchy)
		floats.Axpy(alpha, pdir, x1)
	} else {
		_, alpha, _ := boxSphereIntersections(origin, cauchy, lb, ub, delta, false)
		floats.Axpy(alpha, cauchy, x1)
	}
	_, alpha, _ := boxSphereIntersections(origin, newton, lb, ub, delta, false)
	x2 := make([]float64, n)
	floats.Axpy(alpha, newton, x2)

	r := func(x []float64) float64 {
		ax := make([]float64, m)
		matvec(m, n, p.a, x, ax)
		for i := range ax {
			ax[i] += b[i]
		}
		return vnorm(ax)
	}
	if r(x1) < r(x2) {
		return x1
	}
	return x2
}

// cgInfo is projected_cg's info dictionary.
type cgInfo struct {
	niter, stopCond int
	hitsBoundary    bool
}

// projectedCG solves min ½xᵀHx + cᵀx s.t. A x = 0, |x| <= delta,
// lb <= x <= ub approximately by projected CG (Gould, Hribar and Nocedal
// 2001, Algorithm 6.2, as in SciPy). b is zero in every call SLSQP's
// trust-constr makes, so the initial point is 0.
func projectedCG(hmul func(p, out []float64), c []float64, p *projector, delta float64, lb, ub []float64) ([]float64, cgInfo) {
	const closeToZero = 1e-25
	n, m := len(c), p.m
	x := make([]float64, n)
	hx := make([]float64, n)
	hmul(x, hx)
	for i := range hx {
		hx[i] += c[i]
	}
	r := make([]float64, n)
	p.nullSpace(hx, r)
	g := make([]float64, n)
	p.nullSpace(r, g)
	pd := make([]float64, n)
	for i := range g {
		pd[i] = -g[i]
	}
	hp := make([]float64, n)
	hmul(pd, hp)
	rtg := sqNorm(g)
	if delta-vnorm(x) < closeToZero {
		return x, cgInfo{0, 2, true}
	}
	tol := math.Max(math.Min(0.01*math.Sqrt(rtg), 0.1*rtg), closeToZero)
	maxIter := n - m
	hits, stop, counter, k := false, 1, 0, 0
	lastFeasible := make([]float64, n)
	xNext := make([]float64, n)
	ap := make([]float64, n)
	negG := make([]float64, n)
	for range max(maxIter, 0) {
		if rtg < tol {
			stop = 4
			break
		}
		k++
		pHp := floats.Dot(hp, pd)
		if pHp <= 0 {
			if _, alpha, ok := boxSphereIntersections(x, pd, lb, ub, delta, true); ok {
				floats.Axpy(alpha, pd, x)
			}
			clipBox(x, lb, ub)
			stop, hits = 3, true
			break
		}
		alpha := rtg / pHp
		copy(xNext, x)
		floats.Axpy(alpha, pd, xNext)
		if vnorm(xNext) >= delta {
			copy(ap, pd)
			floats.Scale(alpha, ap)
			if _, theta, ok := boxSphereIntersections(x, ap, lb, ub, delta, false); ok {
				floats.Axpy(theta*alpha, pd, x) // SciPy: x + theta*alpha*p
			}
			clipBox(x, lb, ub)
			stop, hits = 2, true
			break
		}
		if insideBox(xNext, lb, ub) {
			counter = 0
		} else {
			counter++
		}
		if counter > 0 {
			copy(ap, pd)
			floats.Scale(alpha, ap)
			if _, theta, ok := boxSphereIntersections(x, ap, lb, ub, delta, false); ok {
				copy(lastFeasible, x)
				floats.Axpy(theta*alpha, pd, lastFeasible)
				clipBox(lastFeasible, lb, ub)
				counter = 0
			}
		}
		// SciPy stops once counter exceeds max_infeasible_iter (n - m by
		// default). counter cannot exceed the iteration count, itself at
		// most n - m, so that test never fires and is omitted.
		floats.Axpy(alpha, hp, r)
		p.nullSpace(r, g)
		rtgNext := sqNorm(g)
		beta := rtgNext / rtg
		for i := range negG {
			negG[i] = -g[i]
		}
		floats.Xpby(negG, beta, pd) // p = -g_next + beta·p
		copy(x, xNext)
		copy(r, g)
		rtg = rtgNext
		hmul(pd, hp)
	}
	if !insideBox(x, lb, ub) {
		x = lastFeasible
		hits = true
	}
	return x, cgInfo{k, stop, hits}
}

// bfgsHess is SciPy's BFGS HessianUpdateStrategy with approx_type 'hess',
// exception_strategy 'skip_update' (min_curvature 1e-8) and init_scale
// 'auto'. B is kept full and symmetric.
type bfgsHess struct {
	n     int
	b     []float64
	first bool
	mw    []float64
}

func newBFGSHess(n int) *bfgsHess {
	h := &bfgsHess{n: n, b: make([]float64, n*n), first: true, mw: make([]float64, n)}
	for i := range n {
		h.b[i*n+i] = 1
	}
	return h
}

func (h *bfgsHess) mul(p, out []float64) { matvec(h.n, h.n, h.b, p, out) }

func bfgsAutoScale(dx, dg []float64) float64 {
	s2, y2, ys := floats.Dot(dx, dx), floats.Dot(dg, dg), math.Abs(floats.Dot(dg, dx))
	if ys == 0 || y2 == 0 || s2 == 0 {
		return 1
	}
	return y2 / ys
}

func allZero(v []float64) bool {
	for _, x := range v {
		if x != 0 {
			return false
		}
	}
	return true
}

func (h *bfgsHess) update(dx, dg []float64) {
	if allZero(dx) || allZero(dg) {
		return
	}
	n := h.n
	if h.first {
		floats.Scale(bfgsAutoScale(dx, dg), h.b)
		h.first = false
	}
	wz := floats.Dot(dx, dg)
	h.mul(dx, h.mw)
	wMw := floats.Dot(h.mw, dx)
	if wMw <= 0 {
		sc := bfgsAutoScale(dx, dg)
		clear(h.b)
		for i := range n {
			h.b[i*n+i] = sc
		}
		h.mul(dx, h.mw)
		wMw = floats.Dot(h.mw, dx)
	}
	if wz <= 1e-8*wMw {
		return
	}
	// B += y yᵀ/wz - Mw Mwᵀ/wMw: two BLAS dsyr updates of the upper
	// triangle, A(i,j) += x(i)·(alpha·x(j)), mirrored to keep B symmetric.
	a1, a2 := 1/wz, -1/wMw
	for j := range n {
		t1, t2 := float64(a1*dg[j]), float64(a2*h.mw[j])
		for i := 0; i <= j; i++ {
			v := h.b[i*n+j] + float64(dg[i]*t1)
			v += float64(h.mw[i] * t2)
			h.b[i*n+j], h.b[j*n+i] = v, v
		}
	}
}
