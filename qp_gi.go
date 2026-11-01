package optimize

import (
	"fmt"
	"math"

	"github.com/go-simd/floats"
)

// The Goldfarb–Idnani dual active-set method (docs/design-qp.md §3), after
// Schmelzer & Stoll, arXiv:2608.30933 (Algorithm 1, (15), (19), (23)–(26),
// (30)). With P = L·Lᵀ, J = L⁻ᵀ so that JᵀPJ = I; the working-set normals A
// satisfy JᵀA = [R; 0] with R upper triangular. Difference from the design: R
// is a dense n×n array, not packed by columns.

// giColumn is one constraint cᵀx >= b of the dual method: a canonical row
// with its sign (+1 for the lower or equality side, -1 for the upper side).
type giColumn struct {
	row  int
	sign float64
	b    float64
	eq   bool
	norm float64
}

type giState struct {
	md     *qpModel
	n, k   int
	j      []float64 // n×n, row-major
	r      []float64 // n×n, row-major; the leading k×k block is upper triangular
	cols   []giColumn
	ws     []int     // working set: indices into cols, in insertion order
	wsSign []float64 // effective sign of each working column (-1 for an equality entered negated)
	u      []float64 // multipliers of the working set
	x      []float64
	f      float64
	nit    int
}

// vdot is dot(vec(c, sgn), x) without building the vector (the scan of the
// violated constraints calls it for every column at every pass).
func (g *giState) vdot(c giColumn, sgn float64, x []float64) float64 {
	return sgn * c.sign * g.md.rowDot(g.md.rows[c.row], x)
}

func (g *giState) vec(c giColumn, sgn float64) []float64 {
	v := make([]float64, g.n)
	g.md.rowAdd(g.md.rows[c.row], sgn*c.sign, v)
	return v
}

// giStep is how one entering constraint ended.
const (
	giInserted = iota
	giSetAside
	giInfeasible
	giIterLimit
	giNumerical
)

// newGI builds the columns of the dual method from the canonical rows
// (equalities first); the factors are set by start.
func (md *qpModel) newGI() *giState {
	n := md.n
	g := &giState{md: md, n: n, j: make([]float64, n*n), r: make([]float64, n*n)}
	for pass := 0; pass < 2; pass++ { // equalities first, then the inequality sides
		for i, row := range md.rows {
			isEq := row.l == row.u
			if (pass == 0) != isEq {
				continue
			}
			add := func(sign, b float64) {
				c := giColumn{row: i, sign: sign, b: b, eq: isEq}
				c.norm = vnorm(g.vec(c, 1))
				g.cols = append(g.cols, c)
			}
			switch {
			case isEq:
				add(1, row.l)
			default:
				if !math.IsInf(row.l, -1) {
					add(1, row.l)
				}
				if !math.IsInf(row.u, 1) {
					add(-1, -row.u)
				}
			}
		}
	}
	return g
}

// solve runs the walk. With resume the state of the previous solve (factors,
// working set) is repaired for the changed data (Prop. 8.1–8.2); otherwise it
// starts from the unconstrained minimiser, entering the hinted rows first.
func (g *giState) solve(maxIter int, feasTol float64, hint []int, resume bool) *QPResult {
	md, n := g.md, g.n
	if maxIter == 0 {
		maxIter = 10 * (n + len(md.rows))
	}
	warmNote := ""
	switch {
	case resume:
		g.nit = 0
		kept := g.k
		dropped := g.repair()
		warmNote = fmt.Sprintf(" Warm start: working set of %d rows recovered, %d dropped.", kept, dropped)
	default:
		g.start()
		if len(hint) > 0 {
			warmNote = fmt.Sprintf(" Warm start: %d of %d hinted rows entered, %d dropped.", g.enterHint(hint), len(hint), g.repair())
		}
	}
	status, msg, cert := g.walk(maxIter, feasTol)

	marg := make([]float64, len(md.rows))
	active := make([]int, 0, g.k)
	for w, ci := range g.ws {
		c := g.cols[ci]
		marg[c.row] += g.u[w] * g.wsSign[w] * c.sign
		active = append(active, c.row)
	}
	res := md.assemble(g.x, marg, QPActiveSet, feasTol)
	res.ActiveSet, res.Nit = active, g.nit
	if status != QPOptimal {
		res.Status, res.Success, res.Message, res.PrimalCert = status, false, msg, cert
		return res
	}
	if res.Success {
		res.Message = "Optimal solution found (Goldfarb–Idnani dual active set)." + warmNote
		// design-qp.md §3.2 step 6: the running objective must match.
		if !(math.Abs(g.f-res.Fun) <= 1e-10*math.Max(1, math.Abs(res.Fun))) {
			res.Status, res.Success = QPNumericalFail, false
			res.Message = fmt.Sprintf("Numerical difficulties: the running objective %.12g drifted from %.12g.", g.f, res.Fun)
		}
	}
	return res
}

// start (re)sets the state to the unconstrained minimiser: J = L⁻ᵀ column by
// column, x = P⁻¹(−q), f = −½ aᵀx with a = −q, an empty working set.
func (g *giState) start() {
	n, md := g.n, g.md
	clear(g.j)
	clear(g.r)
	g.k, g.nit, g.ws, g.wsSign, g.u = 0, 0, nil, nil, nil
	for c := 0; c < n; c++ {
		e := make([]float64, n)
		e[c] = 1
		for i, v := range backSolve(md.chol, n, e) {
			g.j[i*n+c] = v
		}
	}
	mq := make([]float64, n)
	for i := range mq {
		mq[i] = -md.q[i]
	}
	g.x = backSolve(md.chol, n, forwardSolve(md.chol, n, mq))
	g.f = -0.5 * floats.Dot(mq, g.x)
}

// enterHint inserts the hinted canonical rows (e.g. the previous solve's
// active set) into the working set in order, one Householder insertion each
// (O(n²) apiece, no step taken): for a two-sided row the side nearer to the
// unconstrained minimiser, and a row whose normal depends on those already held
// is skipped. It returns the number entered; repair then recovers x and u.
func (g *giState) enterHint(hint []int) int {
	entered := 0
	for _, row := range hint {
		best, bs := -1, math.Inf(1)
		for i, c := range g.cols {
			if c.row == row && !c.eq {
				if s := math.Abs(g.vdot(c, 1, g.x) - c.b); s < bs {
					best, bs = i, s
				}
			}
		}
		if best < 0 { // an equality: the walk enters it
			continue
		}
		d := g.dvec(g.vec(g.cols[best], 1))
		if g.k >= g.n || vnorm(d[g.k:]) <= 1e3*0x1p-52*vnorm(d) {
			continue
		}
		g.insert(d, best, 1, 0)
		entered++
	}
	return entered
}

// dvec returns Jᵀv.
func (g *giState) dvec(v []float64) []float64 {
	d := make([]float64, g.n)
	for r := 0; r < g.n; r++ {
		if v[r] != 0 {
			floats.Axpy(v[r], g.j[r*g.n:(r+1)*g.n], d)
		}
	}
	return d
}

// recover computes the minimiser x and the multipliers u of the working-set
// subproblem from the factors alone (Schmelzer & Stoll, Prop. 8.1, (35)):
// x_u = J(Jᵀa), ρ = b_𝒜 − Aᵀx_u, Rᵀy = ρ, x = x_u + J₁y, Ru = y, with a = −q.
// O(n² + nk + k²); no constraint outside the working set is looked at.
func (g *giState) recover() {
	n, k, md := g.n, g.k, g.md
	mq := make([]float64, n)
	for i := range mq {
		mq[i] = -md.q[i]
	}
	d := g.dvec(mq) // Jᵀa
	xu := make([]float64, n)
	for r := 0; r < n; r++ {
		xu[r] = floats.Dot(g.j[r*n:(r+1)*n], d)
	}
	y := make([]float64, k) // Rᵀy = ρ, forward
	for i := 0; i < k; i++ {
		c := g.cols[g.ws[i]]
		t := g.wsSign[i]*c.b - g.vdot(c, g.wsSign[i], xu)
		for l := 0; l < i; l++ {
			t -= float64(g.r[l*n+i] * y[l])
		}
		y[i] = t / g.r[i*n+i]
	}
	g.x = xu
	for r := 0; r < n; r++ {
		g.x[r] += floats.Dot(g.j[r*n:r*n+k], y)
	}
	g.u = make([]float64, k)
	for i := k - 1; i >= 0; i-- { // Ru = y, backward
		t := y[i]
		for l := i + 1; l < k; l++ {
			t -= float64(g.r[i*n+l] * g.u[l])
		}
		g.u[i] = t / g.r[i*n+i]
	}
	g.f = 0.5*floats.Dot(md.px(g.x), g.x) + floats.Dot(md.q, g.x)
}

// repair is Proposition 8.2: recover (x, u) from the cached factors, and while
// an inequality of the working set has a negative multiplier drop the most
// negative one (downdating J and R) and recover again. It stops after at most
// k drops in a state satisfying the dual method's invariant (u ≥ 0). It returns
// the number of drops (each one counts in nit).
func (g *giState) repair() int {
	dropped := 0
	for {
		g.recover()
		p, worst := -1, 0.0
		for w, ci := range g.ws {
			if !g.cols[ci].eq && g.u[w] < worst {
				p, worst = w, g.u[w]
			}
		}
		if p < 0 {
			return dropped
		}
		g.remove(p)
		dropped++
	}
}

// walk runs the outer loop and reports how it ended.
func (g *giState) walk(maxIter int, feasTol float64) (int, string, []float64) {
	setAside := map[int]bool{}
	inWS := make([]bool, len(g.cols))
	for _, ci := range g.ws { // a warm start has entered some already
		inWS[ci] = true
	}
	for {
		istar, s, best := -1, 0.0, 0.0
		for i, c := range g.cols {
			if inWS[i] || setAside[i] {
				continue
			}
			si := g.vdot(c, 1, g.x) - c.b
			if c.eq {
				if math.Abs(si) <= feasTol*(1+math.Abs(c.b)) {
					continue // satisfied: nothing to enter (a redundant or already-met equality)
				}
				istar, s = i, si
				break
			}
			if c.norm == 0 {
				if c.b > feasTol {
					return QPInfeasible, "The problem is infeasible: a row 0 >= b with b > 0.", nil
				}
				continue
			}
			if v := -si / c.norm; -si > feasTol*(1+math.Abs(c.b)) && v > best {
				istar, s, best = i, si, v
			}
		}
		if istar < 0 {
			return QPOptimal, "", nil
		}
		how, msg, cert := g.enter(istar, s, maxIter, feasTol, inWS)
		switch how {
		case giSetAside:
			setAside[istar] = true
		case giInfeasible, giIterLimit, giNumerical:
			return map[int]int{giInfeasible: QPInfeasible, giIterLimit: QPIterLimit, giNumerical: QPNumericalFail}[how], msg, cert
		default:
			clear(setAside)
		}
	}
}

// enter brings column istar (slack s < 0 for an inequality; either sign for
// an equality) into the working set, with partial steps that drop columns.
func (g *giState) enter(istar int, s float64, maxIter int, feasTol float64, inWS []bool) (int, string, []float64) {
	n := g.n
	c := g.cols[istar]
	sgn := 1.0
	if c.eq && s > 0 {
		sgn = -1 // Remark 4.3: enter the negated row, so that s < 0
	}
	nstar := g.vec(c, sgn)
	s *= sgn
	bstar := sgn * c.b
	ustar := 0.0
	for {
		if g.nit >= maxIter {
			return giIterLimit, "The iteration limit was reached.", nil
		}
		k := g.k
		d := make([]float64, n) // Jᵀ n*
		for r := 0; r < n; r++ {
			if v := nstar[r]; v != 0 {
				floats.Axpy(v, g.j[r*n:(r+1)*n], d)
			}
		}
		d2 := d[k:]
		nd2 := vnorm(d2)
		z := make([]float64, n)
		if nd2 > 1e3*0x1p-52*vnorm(d) {
			for r := 0; r < n; r++ {
				z[r] = floats.Dot(g.j[r*n+k:(r+1)*n], d2)
			}
		} else {
			nd2 = 0
		}
		rr := make([]float64, k) // R⁻¹ d₁
		for i := k - 1; i >= 0; i-- {
			t := d[i]
			for jx := i + 1; jx < k; jx++ {
				t -= float64(g.r[i*n+jx] * rr[jx])
			}
			rr[i] = t / g.r[i*n+i]
		}
		t1, idel := math.Inf(1), -1
		for w := range g.ws {
			if !g.cols[g.ws[w]].eq && rr[w] > 0 {
				if t := g.u[w] / rr[w]; t < t1 {
					t1, idel = t, w
				}
			}
		}
		t2 := math.Inf(1)
		if nd2 > 0 {
			t2 = -s / float64(nd2*nd2)
		}
		if math.IsNaN(s) || math.IsNaN(t1) || math.IsNaN(t2) {
			return giNumerical, "Numerical difficulties: a step length is not a number (overflow in the data).", nil
		}
		if math.IsInf(t1, 1) && math.IsInf(t2, 1) {
			// Stuck (§3.5): a rounding-level violation is set aside, a
			// macroscopic one is infeasibility with the Farkas certificate rr.
			scale := 1 + math.Abs(bstar)
			if c.norm > 0 {
				scale = 1 + math.Abs(bstar)/c.norm
			}
			if math.Abs(s) <= 1e3*feasTol*scale {
				return giSetAside, "", nil
			}
			cert := make([]float64, len(g.md.rows))
			lhs := -bstar
			for w, ci := range g.ws {
				cw := g.cols[ci]
				cert[cw.row] += rr[w] * g.wsSign[w] * cw.sign
				lhs += rr[w] * g.wsSign[w] * cw.b
			}
			cert[c.row] -= sgn * c.sign
			// §3.5: verify the certificate before reporting. A macroscopic
			// violation (the set-aside test above) makes lhs = rᵀb − b* =
			// −s·(…) clearly negative, so the failed-check outcome is a
			// defensive one that no input is known to reach; it is selected
			// without a branch, and QPNumericalFail would be its status.
			return infeasibleOutcome(lhs < -feasTol*(1+math.Abs(bstar)), cert)
		}
		t := math.Min(t1, t2)
		if nd2 > 0 {
			floats.Axpy(t, z, g.x)
			g.f += float64(t*(t/2+ustar)) * float64(nd2*nd2)
		}
		for w := range g.u {
			g.u[w] -= float64(t * rr[w])
		}
		ustar += t
		if t2 <= t1 {
			g.insert(d, istar, sgn, ustar)
			inWS[istar] = true
			return giInserted, "", nil
		}
		inWS[g.ws[idel]] = false
		g.remove(idel)
		s = floats.Dot(nstar, g.x) - bstar
	}
}

// infeasibleOutcome is the end of a stuck walk: giInfeasible with the
// certificate when it verified, otherwise a message saying it did not and no
// certificate (design-qp.md §3.5 would report QPNumericalFail; the status stays
// QPInfeasible with the message, since no input is known to reach this).
func infeasibleOutcome(verified bool, cert []float64) (int, string, []float64) {
	if verified {
		return giInfeasible, "The problem is infeasible.", cert
	}
	return giInfeasible, "Numerical difficulties: the infeasibility certificate failed its check.", nil
}

// insert adds the entering column with d = Jᵀn*: one Householder reflection
// of the trailing columns of J with Hd₂ = αe₁, and the column [d₁; α] of R.
func (g *giState) insert(d []float64, ci int, sgn, ustar float64) {
	n, k := g.n, g.k
	d2 := d[k:]
	nd2 := vnorm(d2)
	alpha := -nd2
	if d2[0] < 0 {
		alpha = nd2
	}
	// v = d₂ − α e₁ (no cancellation: α has the sign opposite to d₂₁).
	v := clone(d2)
	v[0] -= alpha
	vv := floats.Dot(v, v)
	if vv > 0 {
		for r := 0; r < n; r++ {
			row := g.j[r*n+k : (r+1)*n]
			floats.Axpy(-2*floats.Dot(row, v)/vv, v, row)
		}
	}
	for i := 0; i < k; i++ {
		g.r[i*n+k] = d[i]
	}
	g.r[k*n+k] = alpha
	g.ws = append(g.ws, ci)
	g.wsSign = append(g.wsSign, sgn)
	g.u = append(g.u, ustar)
	g.k++
	g.nit++
}

// remove deletes working-set column p: R loses its column p and the
// sub-diagonal is chased with symmetric 2×2 reflections applied to rows of R
// and columns of J ((30)).
func (g *giState) remove(p int) {
	n, k := g.n, g.k
	for i := 0; i < k; i++ {
		copy(g.r[i*n+p:i*n+k-1], g.r[i*n+p+1:i*n+k])
		g.r[i*n+k-1] = 0
	}
	for j := p; j < k-1; j++ {
		x, y := g.r[j*n+j], g.r[(j+1)*n+j]
		h := math.Hypot(x, y) // > 0: R's diagonal is never zero (Prop. 6.1)
		cc, ss := x/h, y/h    // h >= 0: with x = 0 this is the plain exchange (Remark 6.3)
		for col := 0; col < k-1; col++ {
			a, b := g.r[j*n+col], g.r[(j+1)*n+col]
			g.r[j*n+col], g.r[(j+1)*n+col] = cc*a+ss*b, ss*a-cc*b
		}
		for r := 0; r < n; r++ {
			a, b := g.j[r*n+j], g.j[r*n+j+1]
			g.j[r*n+j], g.j[r*n+j+1] = cc*a+ss*b, ss*a-cc*b
		}
	}
	for col := 0; col < k; col++ {
		g.r[(k-1)*n+col] = 0
	}
	g.ws = append(g.ws[:p], g.ws[p+1:]...)
	g.wsSign = append(g.wsSign[:p], g.wsSign[p+1:]...)
	g.u = append(g.u[:p], g.u[p+1:]...)
	g.k--
	g.nit++
}
