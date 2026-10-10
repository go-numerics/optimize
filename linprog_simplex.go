package optimize

import (
	"fmt"
	"math"
)

// Bounded-variable revised simplex method.
//
// The reduced problem min c·x, A_ub·x + s = b_ub, A_eq·x = b_eq,
// lb <= x <= ub, s >= 0 gets one artificial column ±e_i per row. Phase one
// starts from the basis of slacks (where the initial residual allows) and
// artificials, every structural variable nonbasic at a finite bound (or at
// zero when free), and minimises the sum of the artificials; phase two
// fixes the artificials to [0, 0] and minimises c·x. An artificial still
// basic at zero after phase one marks a redundant row and is harmless.
//
// Pricing is Dantzig's rule (most negative reduced cost); after
// lpMaxDegenerate consecutive degenerate pivots the method switches to
// Bland's rule (smallest eligible index, ties in the ratio test broken by
// the smallest index) until a pivot makes progress, which rules out
// cycling. The ratio test is Harris's two-pass test with bound flips.
// The basis is held as a dense LU factorisation with partial pivoting plus
// a product-form eta file, refactorised every lpRefactor updates.

const (
	lpRefactor      = 64 // eta updates between refactorisations
	lpMaxDegenerate = 50 // degenerate pivots before switching to Bland's rule
)

// lpFactor is the basis B = P⁻¹·L·U·E₁⋯E_k.
type lpFactor struct {
	m    int
	lu   []float64 // L (unit, strictly below) and U, row-major m×m
	perm []int     // row i of P·B is row perm[i] of B
	etaR []int
	etaV [][]float64
}

// factor computes the LU factorisation of the m×m row-major b (which it
// takes over) and clears the eta file. It reports false when b is
// numerically singular: a pivot column whose largest candidate is at most
// rel times its largest entry (or zero).
func (f *lpFactor) factor(b []float64, m int, rel float64) bool {
	f.m, f.lu, f.perm, f.etaR, f.etaV = m, b, lpIota(m), nil, nil
	lu := b
	for k := 0; k < m; k++ {
		colmax := 0.0
		for i := 0; i < m; i++ {
			colmax = math.Max(colmax, math.Abs(lu[i*m+k]))
		}
		p, big := k, math.Abs(lu[k*m+k])
		for i := k + 1; i < m; i++ {
			if v := math.Abs(lu[i*m+k]); v > big {
				p, big = i, v
			}
		}
		if big <= rel*colmax || big == 0 {
			return false
		}
		if p != k {
			f.perm[p], f.perm[k] = f.perm[k], f.perm[p]
			for j := 0; j < m; j++ {
				lu[p*m+j], lu[k*m+j] = lu[k*m+j], lu[p*m+j]
			}
		}
		for i := k + 1; i < m; i++ {
			l := lu[i*m+k] / lu[k*m+k]
			lu[i*m+k] = l
			if l != 0 {
				for j := k + 1; j < m; j++ {
					lu[i*m+j] -= l * lu[k*m+j]
				}
			}
		}
	}
	return true
}

// ftran returns B⁻¹·v.
func (f *lpFactor) ftran(v []float64) []float64 {
	m, lu := f.m, f.lu
	w := make([]float64, m)
	for i := range w {
		w[i] = v[f.perm[i]]
	}
	for i := 0; i < m; i++ {
		s := w[i]
		for k := 0; k < i; k++ {
			s -= lu[i*m+k] * w[k]
		}
		w[i] = s
	}
	for i := m - 1; i >= 0; i-- {
		s := w[i]
		for k := i + 1; k < m; k++ {
			s -= lu[i*m+k] * w[k]
		}
		w[i] = s / lu[i*m+i]
	}
	for e, r := range f.etaR {
		a := f.etaV[e]
		xr := w[r] / a[r]
		for i := range w {
			w[i] -= a[i] * xr
		}
		w[r] = xr
	}
	return w
}

// btran returns B⁻ᵀ·v.
func (f *lpFactor) btran(v []float64) []float64 {
	m, lu := f.m, f.lu
	w := clone(v)
	for e := len(f.etaR) - 1; e >= 0; e-- {
		r, a := f.etaR[e], f.etaV[e]
		s := w[r]
		for i := range w {
			if i != r {
				s -= a[i] * w[i]
			}
		}
		w[r] = s / a[r]
	}
	for i := 0; i < m; i++ {
		s := w[i]
		for k := 0; k < i; k++ {
			s -= lu[k*m+i] * w[k]
		}
		w[i] = s / lu[i*m+i]
	}
	for i := m - 1; i >= 0; i-- {
		s := w[i]
		for k := i + 1; k < m; k++ {
			s -= lu[k*m+i] * w[k]
		}
		w[i] = s
	}
	y := make([]float64, m)
	for i := range w {
		y[f.perm[i]] = w[i]
	}
	return y
}

// update appends the eta matrix for column r of the basis replaced by the
// column whose ftran is alpha.
func (f *lpFactor) update(r int, alpha []float64) {
	f.etaR = append(f.etaR, r)
	f.etaV = append(f.etaV, alpha)
}

type lpSimplex struct {
	m, n, mub int
	ncol      int         // n structurals, mub slacks, m artificials
	cols      [][]float64 // dense columns, length m each
	b         []float64
	lo, hi, x []float64
	cost      []float64
	head      []int // basic variable in each basis position
	pos       []int // basis position of each variable, or -1
	f         lpFactor
	refactor  bool
	degen     int
	bland     bool
	y         []float64 // simplex multipliers of the last pricing
	tol       float64
	nit       int
}

// newSimplex sets up phase one of the bounded simplex for p.
func newSimplex(p *lpProblem, tol float64) *lpSimplex {
	m := p.mub + p.meq
	s := &lpSimplex{m: m, n: p.n, mub: p.mub, ncol: p.n + p.mub + m, tol: tol, refactor: true}
	s.cols = make([][]float64, s.ncol)
	s.b = append(clone(p.bub), p.beq...)
	for j := 0; j < p.n; j++ {
		col := make([]float64, m)
		for i := 0; i < p.mub; i++ {
			col[i] = p.aub[i*p.n+j]
		}
		for i := 0; i < p.meq; i++ {
			col[p.mub+i] = p.aeq[i*p.n+j]
		}
		s.cols[j] = col
	}
	s.lo = make([]float64, s.ncol)
	s.hi = make([]float64, s.ncol)
	s.x = make([]float64, s.ncol)
	s.cost = make([]float64, s.ncol)
	s.pos = make([]int, s.ncol)
	for j := 0; j < p.n; j++ {
		s.lo[j], s.hi[j] = p.lb[j], p.ub[j]
		switch {
		case !math.IsInf(p.lb[j], 0):
			s.x[j] = p.lb[j]
		case !math.IsInf(p.ub[j], 0):
			s.x[j] = p.ub[j]
		}
		s.pos[j] = -1
	}
	// Residual of the rows at the starting point of the structurals.
	r := clone(s.b)
	for j := 0; j < p.n; j++ {
		for i := range r {
			r[i] -= s.cols[j][i] * s.x[j]
		}
	}
	s.head = make([]int, m)
	for i := 0; i < m; i++ {
		sl, art := p.n+i, p.n+p.mub+i
		if i < p.mub {
			s.cols[sl] = lpUnit(m, i, 1)
			s.hi[sl] = math.Inf(1)
			s.pos[sl] = -1
		}
		sign := 1.0
		if r[i] < 0 {
			sign = -1
		}
		s.cols[art] = lpUnit(m, i, sign)
		s.pos[art] = -1
		if i < p.mub && r[i] >= 0 {
			s.head[i], s.pos[sl], s.x[sl] = sl, i, r[i]
			continue
		}
		s.head[i], s.pos[art], s.x[art] = art, i, math.Abs(r[i])
		s.hi[art], s.cost[art] = math.Inf(1), 1
	}
	return s
}

func lpUnit(m, i int, v float64) []float64 {
	e := make([]float64, m)
	e[i] = v
	return e
}

// refactorise rebuilds the LU factors of the basis and recomputes the
// basic variables from the nonbasic ones.
func (s *lpSimplex) refactorise() bool {
	m := s.m
	bm := make([]float64, m*m)
	for k, j := range s.head {
		for i := 0; i < m; i++ {
			bm[i*m+k] = s.cols[j][i]
		}
	}
	if !s.f.factor(bm, m, 1e-11) {
		return false
	}
	rhs := clone(s.b)
	for j := 0; j < s.ncol; j++ {
		if s.pos[j] < 0 && s.x[j] != 0 {
			for i := range rhs {
				rhs[i] -= s.cols[j][i] * s.x[j]
			}
		}
	}
	for k, v := range s.f.ftran(rhs) {
		s.x[s.head[k]] = v
	}
	s.refactor = false
	return true
}

// run iterates from the current basis until optimality (LPOptimal),
// unboundedness (LPUnbounded), the iteration limit (LPIterLimit) or a
// singular basis (LPNumericalFail).
func (s *lpSimplex) run(maxIter int) int {
	m := s.m
	ctol := s.tol * math.Max(1, norm(s.cost, math.Inf(1)))
	cb := make([]float64, m)
	for {
		if s.refactor && !s.refactorise() {
			return LPNumericalFail
		}
		for k, j := range s.head {
			cb[k] = s.cost[j]
		}
		s.y = s.f.btran(cb)
		// Pricing.
		q, dir, best := -1, 0.0, 0.0
		for j := 0; j < s.ncol; j++ {
			if s.pos[j] >= 0 || s.lo[j] == s.hi[j] {
				continue
			}
			d := s.cost[j] - dot(s.y, s.cols[j])
			var dj, score float64
			switch {
			case d < -ctol && s.x[j] < s.hi[j]:
				dj, score = 1, -d
			case d > ctol && s.x[j] > s.lo[j]:
				dj, score = -1, d
			default:
				continue
			}
			if score > best {
				q, dir, best = j, dj, score
				if s.bland {
					break
				}
			}
		}
		if q < 0 {
			return LPOptimal
		}
		if s.nit >= maxIter {
			return LPIterLimit
		}
		alpha := s.f.ftran(s.cols[q])
		r, t, ok := s.ratio(q, dir, alpha)
		if !ok {
			return LPUnbounded
		}
		s.nit++
		s.x[q] += dir * t
		for i, j := range s.head {
			s.x[j] -= dir * t * alpha[i]
		}
		if r < 0 { // bound flip
			s.x[q] = s.hi[q]
			if dir < 0 {
				s.x[q] = s.lo[q]
			}
		} else {
			l := s.head[r]
			if l >= s.n+s.mub { // an artificial never re-enters
				s.hi[l] = 0
			}
			s.x[l] = s.lo[l]
			if dir*alpha[r] < 0 {
				s.x[l] = s.hi[l]
			}
			s.head[r], s.pos[q], s.pos[l] = q, r, -1
			s.f.update(r, alpha)
			s.refactor = len(s.f.etaR) >= lpRefactor
		}
		if t <= 1e-12 {
			s.degen++
			s.bland = s.degen > lpMaxDegenerate
		} else {
			s.degen, s.bland = 0, false
		}
	}
}

// ratio performs the ratio test for variable q moving in direction dir
// with B⁻¹·a_q = alpha. It returns the leaving basis position (-1 for a
// bound flip of q) and the step; ok is false when nothing limits the step.
func (s *lpSimplex) ratio(q int, dir float64, alpha []float64) (r int, t float64, ok bool) {
	flip := s.hi[q] - s.lo[q]
	amax := norm(alpha, math.Inf(1))
	pt := 1e-9 * math.Max(1, amax)
	ftol := s.tol
	// rate[i] is d x_B[i] / d t; limit returns the distance to the bound
	// it heads for, relaxed by slack.
	limit := func(i int, slack float64) float64 {
		j := s.head[i]
		a := -dir * alpha[i]
		switch {
		case a < -pt:
			return (s.x[j] - s.lo[j] + slack) / -a
		case a > pt:
			return (s.hi[j] - s.x[j] + slack) / a
		}
		return math.Inf(1)
	}
	r = -1
	if s.bland {
		t = math.Inf(1)
		for i := range alpha {
			ri := math.Max(limit(i, 0), 0)
			if ri < t-1e-12 || (ri <= t+1e-12 && r >= 0 && s.head[i] < s.head[r]) {
				r, t = i, ri
			}
		}
	} else {
		tmax := math.Inf(1)
		for i := range alpha {
			tmax = math.Min(tmax, limit(i, ftol))
		}
		big := 0.0
		for i := range alpha {
			if limit(i, 0) <= tmax && math.Abs(alpha[i]) > big {
				r, big = i, math.Abs(alpha[i])
			}
		}
		if r >= 0 {
			t = math.Max(limit(r, 0), 0)
		}
	}
	if r < 0 || flip <= t {
		if math.IsInf(flip, 1) {
			return -1, 0, false
		}
		return -1, flip, true
	}
	return r, t, true
}

// simplexLP solves p by the two-phase bounded revised simplex method.
func simplexLP(p *lpProblem, tol float64, maxIter int) lpSolution {
	s := newSimplex(p, tol)
	sol := lpSolution{}
	finish := func(status int, msg string) lpSolution {
		sol.x = clone(s.x[:s.n])
		sol.status, sol.message, sol.nit = status, msg, s.nit
		return sol
	}
	st := s.run(maxIter)
	switch st {
	case LPIterLimit:
		return finish(st, "Iteration limit reached.")
	case LPNumericalFail:
		return finish(st, lpSingularMsg)
	}
	infeas := 0.0
	for j := s.n + s.mub; j < s.ncol; j++ {
		infeas += math.Abs(s.x[j])
	}
	if lim := tol * (1 + norm(s.b, math.Inf(1))); infeas > lim {
		return finish(LPInfeasible, fmt.Sprintf("The problem appears infeasible, as the "+
			"phase one auxiliary problem terminated successfully with a residual of "+
			"%.1e, greater than the tolerance %.1e required for the solution to be "+
			"considered feasible. Consider increasing the tolerance to be greater "+
			"than %.1e. If this tolerance is unacceptably large, the problem is "+
			"likely infeasible.", infeas, lim, infeas))
	}
	for j := s.n + s.mub; j < s.ncol; j++ {
		s.hi[j], s.cost[j] = 0, 0
	}
	copy(s.cost, p.c)
	s.degen, s.bland = 0, false
	switch st = s.run(maxIter); st {
	case LPOptimal:
		sol.yub, sol.yeq = s.y[:s.mub], s.y[s.mub:]
		return finish(st, "Optimization terminated successfully.")
	case LPUnbounded:
		return finish(st, "The problem is unbounded, as the simplex algorithm found "+
			"a basic feasible solution from which there is a direction with negative "+
			"reduced cost along which the objective decreases without bound.")
	case LPIterLimit:
		return finish(st, "Iteration limit reached.")
	}
	return finish(st, lpSingularMsg)
}

const lpSingularMsg = "Numerical difficulties encountered: the basis matrix " +
	"became singular; consider trying method LPInteriorPoint."
