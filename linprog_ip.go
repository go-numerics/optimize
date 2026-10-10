package optimize

import (
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/linalg"
)

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
	src   *lpProblem
	free  []int // structural index of each split (negative-part) column
}

// standardForm mirrors SciPy's _get_Abc: variables bounded only above are
// negated, finite upper bounds become rows after the A_ub rows, free
// variables are split, slacks are appended and finite lower bounds shifted
// out. Row order is A_ub, bound rows, A_eq.
func standardForm(p *lpProblem) *lpStandard {
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
			row(i)[j] = float64(sign[j] * p.aub[i*n+j])
		}
		s.b[i] = p.bub[i]
	}
	for k, j := range boundCols {
		row(p.mub + k)[j] = 1
		s.b[p.mub+k] = ub[j]
	}
	for i := 0; i < p.meq; i++ {
		for j := 0; j < n; j++ {
			row(mu + i)[j] = float64(sign[j] * p.aeq[i*n+j])
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
				s.b[i] -= float64(row(i)[j] * lb[j])
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

// lpCholesky factors the symmetric m×m matrix a in place (lower
// triangle) and reports false, like LAPACK's potrf, when a pivot is not
// positive.
func lpCholesky(a []float64, m int) bool {
	for j := 0; j < m; j++ {
		d := a[j*m+j]
		for k := 0; k < j; k++ {
			d -= float64(a[j*m+k] * a[j*m+k])
		}
		if !(d > 0) {
			return false
		}
		d = math.Sqrt(d)
		a[j*m+j] = d
		for i := j + 1; i < m; i++ {
			s := a[i*m+j]
			for k := 0; k < j; k++ {
				s -= float64(a[i*m+k] * a[j*m+k])
			}
			a[i*m+j] = s / d
		}
	}
	return true
}

// lpNormal solves the normal equations M·v = r of one interior-point
// iteration the way SciPy's _get_solver does for dense problems: Cholesky
// first; if that fails, LU with partial pivoting; if that meets an exactly
// zero pivot, least squares (gelsd with rcond = eps, which is what
// scipy.linalg.lstsq uses by default).
type lpNormal struct {
	m     int
	mm    []float64
	stage int // 0 Cholesky, 1 LU, 2 least squares
	chol  []float64
	lu    lpFactor
}

func newLPNormal(mm []float64, m int) *lpNormal {
	ns := &lpNormal{m: m, mm: mm}
	ns.chol = clone(mm)
	if lpCholesky(ns.chol, m) {
		return ns
	}
	ns.stage = 1
	if ns.lu.factor(clone(mm), m, 0) {
		return ns
	}
	ns.stage = 2
	return ns
}

func (ns *lpNormal) solve(r []float64) []float64 {
	switch ns.stage {
	case 0:
		return lpCholSolve(ns.chol, ns.m, r)
	case 1:
		return ns.lu.ftran(r)
	}
	a, _ := ndarray.FromData(ns.mm, ns.m, ns.m)
	b, _ := ndarray.FromData(r, ns.m)
	// gelsd only fails on non-finite input, which the factorisations
	// above would already have turned into NaN; leaving v NaN then ends the
	// solve with status 4 like SciPy's LinAlgError.
	res, _ := linalg.LstsqRcond(a, b, epsmch)
	v := make([]float64, ns.m)
	for i := range v {
		v[i] = math.NaN()
	}
	copy(v, arrayData(res.X))
	return v
}

// lpCholSolve solves L·Lᵀ·v = r with the factor from lpCholesky.
func lpCholSolve(l []float64, m int, r []float64) []float64 {
	v := clone(r)
	for i := 0; i < m; i++ {
		s := v[i]
		for k := 0; k < i; k++ {
			s -= float64(l[i*m+k] * v[k])
		}
		v[i] = s / l[i*m+i]
	}
	for i := m - 1; i >= 0; i-- {
		s := v[i]
		for k := i + 1; k < m; k++ {
			s -= float64(l[k*m+i] * v[k])
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
				out[j] += float64(v * yi)
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
		rp[i] = float64(s.b[i]*h.tau) - ax[i]
	}
	aty := s.mulAT(h.y)
	rd := make([]float64, s.nn)
	for j := range rd {
		rd[j] = float64(s.c[j]*h.tau) - aty[j] - h.z[j]
	}
	cx, by := dot(s.c, h.x), dot(s.b, h.y)
	rg := h.kappa + cx - by
	mu := (dot(h.x, h.z) + float64(h.tau*h.kappa)) / float64(s.nn+1)
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
		rP[i] = float64(s.b[i]*tau) - ax[i]
	}
	rD := make([]float64, nn)
	for j := range rD {
		rD[j] = float64(s.c[j]*tau) - aty[j] - z[j]
	}
	rG := dot(s.c, x) - dot(s.b, y) + kappa
	mu := (dot(x, z) + float64(tau*kappa)) / float64(nn+1)
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
				v += float64(ai[j] * d * ak[j])
			}
			mm[i*m+k], mm[k*m+i] = v, v
		}
	}
	ns := newLPNormal(mm, m)
	symSolve := func(r1, r2 []float64) (u, v []float64) {
		t := make([]float64, nn)
		for j := range t {
			t[j] = float64(dinv[j] * r1[j])
		}
		r := s.mulA(t)
		for i := range r {
			r[i] += r2[i]
		}
		v = ns.solve(r)
		u = s.mulAT(v)
		for j := range u {
			u[j] = float64(dinv[j] * (u[j] - r1[j]))
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
			rhatp[i] = float64(eta * rP[i])
		}
		rhatg := float64(eta * rG)
		rhattk := float64(gamma*mu) - float64(tau*kappa)
		for j := range rhatxs {
			rhatxs[j] = float64(gamma*mu) - float64(x[j]*z[j])
			if pass == 1 {
				rhatxs[j] -= float64(dx[j] * dz[j])
			}
		}
		if pass == 1 {
			rhattk -= float64(dtau * dkappa)
		}
		for j := range r1 {
			r1[j] = float64(eta*rD[j]) - rhatxs[j]/x[j]
		}
		u, v := symSolve(r1, rhatp)
		dtau = (rhatg + rhattk/tau - (-dot(s.c, u) + dot(s.b, v))) / (kappa/tau + (-cp + bq))
		dx = make([]float64, nn)
		dz = make([]float64, nn)
		dy = make([]float64, m)
		for j := range dx {
			dx[j] = u[j] + float64(p[j]*dtau)
			dz[j] = (rhatxs[j] - float64(z[j]*dx[j])) / x[j]
		}
		for i := range dy {
			dy[i] = v[i] + float64(q[i]*dtau)
		}
		dkappa = (rhattk - float64(kappa*dtau)) / tau
		alpha := h.step(dx, dz, dtau, dkappa, 1)
		gamma = float64((1 - alpha) * (1 - alpha) * math.Min(0.1, 1-alpha))
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
func interiorPointLP(p *lpProblem, tol float64, maxIter int) lpSolution {
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
			h.x[j] += float64(alpha * dx[j])
			h.z[j] += float64(alpha * dz[j])
		}
		for i := range h.y {
			h.y[i] += float64(alpha * dy[i])
		}
		h.tau += float64(alpha * dtau)
		h.kappa += float64(alpha * dkappa)
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
