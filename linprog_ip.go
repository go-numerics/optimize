package optimize

import "math"

// Primal–dual interior-point method on the homogeneous self-dual embedding
// of Andersen & Andersen ("The MOSEK interior point optimizer for linear
// programming: an implementation of the homogeneous algorithm", 2000), with
// Mehrotra's predictor–corrector, following the structure of SciPy's legacy
// method='interior-point' (pc=True, ip=False, alpha0=0.99995, dense
// Cholesky). The problem is first put in SciPy's standard form (_get_Abc)
// with the same row and column order, so the iterates match SciPy's up to
// rounding.

// lpStandard is min c·x, A·x = b, x >= 0 with the maps back to the reduced
// problem.
type lpStandard struct {
	m, nn int       // rows and columns
	a     []float64 // row-major m×nn
	b, c  []float64
	nb    int // number of upper-bound rows
	src   *lp
	free  []int // structural index of each split (negative-part) column
}

// standardForm mirrors SciPy's _get_Abc: variables bounded only above are
// negated, finite upper bounds become rows after the A_ub rows, free
// variables are split, slacks are appended and finite lower bounds shifted
// out. Row order is A_ub, bound rows, A_eq.
func standardForm(p *lp) *lpStandard {
	n := p.n
	lb, ub, c := clone(p.lb), clone(p.ub), clone(p.c)
	sign := make([]float64, n)
	var boundCols, free []int
	for j := 0; j < n; j++ {
		sign[j] = 1
		if math.IsInf(lb[j], -1) && !math.IsInf(ub[j], 1) {
			sign[j], c[j] = -1, -c[j]
			lb[j], ub[j] = -ub[j], math.Inf(1)
		}
	}
	for j := 0; j < n; j++ {
		if !math.IsInf(ub[j], 1) {
			boundCols = append(boundCols, j)
		}
		if math.IsInf(lb[j], -1) {
			free = append(free, j)
		}
	}
	nb := len(boundCols)
	mu := p.mub + nb
	m := mu + p.meq
	nn := n + len(free) + mu
	s := &lpStandard{m: m, nn: nn, a: make([]float64, m*nn), b: make([]float64, m),
		c: make([]float64, nn), nb: nb, src: p, free: free}
	row := func(i int) []float64 { return s.a[i*nn : (i+1)*nn] }
	for i := 0; i < p.mub; i++ {
		for j := 0; j < n; j++ {
			row(i)[j] = sign[j] * p.aub[i*n+j]
		}
		s.b[i] = p.bub[i]
	}
	for k, j := range boundCols {
		row(p.mub + k)[j] = 1
		s.b[p.mub+k] = ub[j]
	}
	for i := 0; i < p.meq; i++ {
		for j := 0; j < n; j++ {
			row(mu + i)[j] = sign[j] * p.aeq[i*n+j]
		}
		s.b[mu+i] = p.beq[i]
	}
	copy(s.c, c)
	for k, j := range free {
		s.c[n+k] = -c[j]
		for i := 0; i < m; i++ {
			row(i)[n+k] = -row(i)[j]
		}
	}
	for i := 0; i < mu; i++ {
		row(i)[n+len(free)+i] = 1
	}
	for j := 0; j < n; j++ {
		if !math.IsInf(lb[j], -1) && lb[j] != 0 {
			for i := 0; i < m; i++ {
				s.b[i] -= row(i)[j] * lb[j]
			}
		}
	}
	return s
}

// recover maps a standard-form point back to the reduced variables
// (SciPy's _postsolve).
func (s *lpStandard) recover(xs []float64) []float64 {
	p := s.src
	x := clone(xs[:p.n])
	k := 0
	for j := range x {
		switch {
		case math.IsInf(p.lb[j], -1) && math.IsInf(p.ub[j], 1):
			x[j] -= xs[p.n+k]
			k++
		case math.IsInf(p.lb[j], -1):
			x[j] = p.ub[j] - x[j]
		default:
			x[j] += p.lb[j]
		}
	}
	return x
}

// lpCholesky factors the symmetric positive semidefinite m×m matrix a in
// place (lower triangle). A pivot that is not positive, or tiny relative to
// the original diagonal, marks a (numerically) dependent row: it is
// replaced by a huge value so that the corresponding solution component
// vanishes, the rule of Wright's modified Cholesky used by PCx and LIPSOL.
// (SciPy instead retries with LU and then least squares.)
func lpCholesky(a []float64, m int) {
	for j := 0; j < m; j++ {
		d := a[j*m+j]
		orig := d
		for k := 0; k < j; k++ {
			d -= a[j*m+k] * a[j*m+k]
		}
		if !(d > 1e-14*orig) {
			a[j*m+j] = 1e64
			for i := j + 1; i < m; i++ {
				a[i*m+j] = 0
			}
			continue
		}
		d = math.Sqrt(d)
		a[j*m+j] = d
		for i := j + 1; i < m; i++ {
			s := a[i*m+j]
			for k := 0; k < j; k++ {
				s -= a[i*m+k] * a[j*m+k]
			}
			a[i*m+j] = s / d
		}
	}
}

// lpCholSolve solves L·Lᵀ·v = r with the factor from lpCholesky.
func lpCholSolve(l []float64, m int, r []float64) []float64 {
	v := clone(r)
	for i := 0; i < m; i++ {
		s := v[i]
		for k := 0; k < i; k++ {
			s -= l[i*m+k] * v[k]
		}
		v[i] = s / l[i*m+i]
	}
	for i := m - 1; i >= 0; i-- {
		s := v[i]
		for k := i + 1; k < m; k++ {
			s -= l[k*m+i] * v[k]
		}
		v[i] = s / l[i*m+i]
	}
	return v
}

func (s *lpStandard) mulA(x []float64) []float64 {
	out := make([]float64, s.m)
	for i := range out {
		out[i] = dot(s.a[i*s.nn:(i+1)*s.nn], x)
	}
	return out
}

func (s *lpStandard) mulAT(y []float64) []float64 {
	out := make([]float64, s.nn)
	for i, yi := range y {
		if yi != 0 {
			for j, v := range s.a[i*s.nn : (i+1)*s.nn] {
				out[j] += v * yi
			}
		}
	}
	return out
}

// lpHSD holds an iterate of the homogeneous self-dual method.
type lpHSD struct {
	s             *lpStandard
	x, y, z       []float64
	tau, kappa    float64
	rp0, rd0, rg0 float64 // residual norms at the blind start
}

// indicators returns SciPy's ρ_p, ρ_d, ρ_A, ρ_g and ρ_μ ([4] §4.5).
func (h *lpHSD) indicators() (rhoP, rhoD, rhoA, rhoG, rhoMu float64) {
	s := h.s
	ax := s.mulA(h.x)
	rp := make([]float64, s.m)
	for i := range rp {
		rp[i] = s.b[i]*h.tau - ax[i]
	}
	aty := s.mulAT(h.y)
	rd := make([]float64, s.nn)
	for j := range rd {
		rd[j] = s.c[j]*h.tau - aty[j] - h.z[j]
	}
	cx, by := dot(s.c, h.x), dot(s.b, h.y)
	rg := h.kappa + cx - by
	mu := (dot(h.x, h.z) + h.tau*h.kappa) / float64(s.nn+1)
	rhoA = math.Abs(cx-by) / (h.tau + math.Abs(by))
	rhoP = enorm(rp) / math.Max(1, h.rp0)
	rhoD = enorm(rd) / math.Max(1, h.rd0)
	rhoG = math.Abs(rg) / math.Max(1, h.rg0)
	return rhoP, rhoD, rhoA, rhoG, mu
}

// direction computes Mehrotra's predictor–corrector search direction
// ([4] equations 8.8–8.32; SciPy's _get_delta with pc=True).
func (h *lpHSD) direction() (dx, dy, dz []float64, dtau, dkappa float64) {
	s := h.s
	m, nn := s.m, s.nn
	x, y, z, tau, kappa := h.x, h.y, h.z, h.tau, h.kappa
	ax, aty := s.mulA(x), s.mulAT(y)
	rP := make([]float64, m)
	for i := range rP {
		rP[i] = s.b[i]*tau - ax[i]
	}
	rD := make([]float64, nn)
	for j := range rD {
		rD[j] = s.c[j]*tau - aty[j] - z[j]
	}
	rG := dot(s.c, x) - dot(s.b, y) + kappa
	mu := (dot(x, z) + tau*kappa) / float64(nn+1)
	dinv := make([]float64, nn)
	for j := range dinv {
		dinv[j] = x[j] / z[j]
	}
	mm := make([]float64, m*m)
	for i := 0; i < m; i++ {
		ai := s.a[i*nn : (i+1)*nn]
		for k := 0; k <= i; k++ {
			ak := s.a[k*nn : (k+1)*nn]
			v := 0.0
			for j, d := range dinv {
				v += ai[j] * d * ak[j]
			}
			mm[i*m+k], mm[k*m+i] = v, v
		}
	}
	lpCholesky(mm, m)
	symSolve := func(r1, r2 []float64) (u, v []float64) {
		t := make([]float64, nn)
		for j := range t {
			t[j] = dinv[j] * r1[j]
		}
		r := s.mulA(t)
		for i := range r {
			r[i] += r2[i]
		}
		v = lpCholSolve(mm, m, r)
		u = s.mulAT(v)
		for j := range u {
			u[j] = dinv[j] * (u[j] - r1[j])
		}
		return u, v
	}
	p, q := symSolve(s.c, s.b)
	cp, bq := dot(s.c, p), dot(s.b, q)
	gamma := 0.0
	rhatxs := make([]float64, nn)
	r1 := make([]float64, nn)
	rhatp := make([]float64, m)
	for pass := 0; pass < 2; pass++ {
		eta := 1 - gamma
		for i := range rhatp {
			rhatp[i] = eta * rP[i]
		}
		rhatg := eta * rG
		rhattk := gamma*mu - tau*kappa
		for j := range rhatxs {
			rhatxs[j] = gamma*mu - x[j]*z[j]
			if pass == 1 {
				rhatxs[j] -= dx[j] * dz[j]
			}
		}
		if pass == 1 {
			rhattk -= dtau * dkappa
		}
		for j := range r1 {
			r1[j] = eta*rD[j] - rhatxs[j]/x[j]
		}
		u, v := symSolve(r1, rhatp)
		dtau = (rhatg + rhattk/tau - (-dot(s.c, u) + dot(s.b, v))) / (kappa/tau + (-cp + bq))
		dx = make([]float64, nn)
		dz = make([]float64, nn)
		dy = make([]float64, m)
		for j := range dx {
			dx[j] = u[j] + p[j]*dtau
			dz[j] = (rhatxs[j] - z[j]*dx[j]) / x[j]
		}
		for i := range dy {
			dy[i] = v[i] + q[i]*dtau
		}
		dkappa = (rhattk - kappa*dtau) / tau
		alpha := h.step(dx, dz, dtau, dkappa, 1)
		gamma = (1 - alpha) * (1 - alpha) * math.Min(0.1, 1-alpha)
	}
	return dx, dy, dz, dtau, dkappa
}

// step is the largest step, damped by alpha0, keeping x, z, τ, κ positive
// ([4] equation 8.21).
func (h *lpHSD) step(dx, dz []float64, dtau, dkappa, alpha0 float64) float64 {
	alpha := 1.0
	for j := range dx {
		if dx[j] < 0 {
			alpha = math.Min(alpha, alpha0*h.x[j]/-dx[j])
		}
		if dz[j] < 0 {
			alpha = math.Min(alpha, alpha0*h.z[j]/-dz[j])
		}
	}
	if dtau < 0 {
		alpha = math.Min(alpha, alpha0*h.tau/-dtau)
	}
	if dkappa < 0 {
		alpha = math.Min(alpha, alpha0*h.kappa/-dkappa)
	}
	return alpha
}

// interiorPointLP solves p by the homogeneous self-dual interior-point
// method.
func interiorPointLP(p *lp, tol float64, maxIter int) lpSolution {
	s := standardForm(p)
	h := &lpHSD{s: s, x: lpOnes(s.nn), y: make([]float64, s.m), z: lpOnes(s.nn), tau: 1, kappa: 1}
	// Residuals at the blind start x = z = 1, y = 0, τ = κ = 1.
	h.rp0 = enorm(lpSub(s.b, s.mulA(h.x)))
	h.rd0 = enorm(lpSub(s.c, h.z))
	h.rg0 = math.Abs(1 + dot(s.c, h.x))
	rhoP, rhoD, rhoA, _, _ := h.indicators()
	sol := lpSolution{status: LPOptimal, message: lpIPMessages[0]}
	for rhoP > tol || rhoD > tol || rhoA > tol {
		sol.nit++
		dx, dy, dz, dtau, dkappa := h.direction()
		if !lpAllFinite(dx) || !lpAllFinite(dy) || !lpAllFinite(dz) || !lpAllFinite([]float64{dtau, dkappa}) {
			sol.status = LPNumericalFail
			break
		}
		alpha := h.step(dx, dz, dtau, dkappa, 0.99995)
		for j := range h.x {
			h.x[j] += alpha * dx[j]
			h.z[j] += alpha * dz[j]
		}
		for i := range h.y {
			h.y[i] += alpha * dy[i]
		}
		h.tau += alpha * dtau
		h.kappa += alpha * dkappa
		var rhoG, rhoMu float64
		rhoP, rhoD, rhoA, rhoG, rhoMu = h.indicators()
		inf1 := rhoP < tol && rhoD < tol && rhoG < tol && h.tau < tol*math.Max(1, h.kappa)
		inf2 := rhoMu < tol && h.tau < tol*math.Min(1, h.kappa)
		if inf1 || inf2 {
			sol.status = LPUnbounded
			if dot(s.b, h.y) > tol {
				sol.status = LPInfeasible
			}
			break
		}
		if sol.nit >= maxIter {
			sol.status = LPIterLimit
			break
		}
	}
	sol.message = lpIPMessages[sol.status]
	xs := make([]float64, s.nn)
	for j := range xs {
		xs[j] = h.x[j] / h.tau
	}
	sol.x = s.recover(xs)
	if sol.status == LPOptimal {
		mu := p.mub
		sol.yub = make([]float64, mu)
		sol.yeq = make([]float64, p.meq)
		for i := range sol.yub {
			sol.yub[i] = h.y[i] / h.tau
		}
		for i := range sol.yeq {
			sol.yeq[i] = h.y[mu+s.nb+i] / h.tau
		}
	}
	return sol
}

func lpOnes(n int) []float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = 1
	}
	return v
}

func lpSub(a, b []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] - b[i]
	}
	return out
}

var lpIPMessages = [...]string{
	"Optimization terminated successfully.",
	"The iteration limit was reached before the algorithm converged.",
	"The algorithm terminated successfully and determined that the problem is infeasible.",
	"The algorithm terminated successfully and determined that the problem is unbounded.",
	"Numerical difficulties were encountered before the problem converged. " +
		"Please check your problem formulation for errors, independence of " +
		"linear equality constraints, and reasonable scaling and matrix " +
		"condition numbers.",
}
