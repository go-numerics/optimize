package optimize

import (
	"fmt"
	"math"
	"slices"

	"github.com/go-simd/floats"
)

// TrustConstr is the trust-region interior-point method of
// scipy.optimize.minimize(method='trust-constr') (MATLAB fmincon's
// 'interior-point'): Byrd–Omojokun trust-region SQP for equality-constrained
// problems, and a barrier method around it (Byrd, Hribar and Nocedal 1999)
// when there are inequalities or bounds. The port follows SciPy v1.18.1's
// scipy/optimize/_trustregion_constr (BSD-3-Clause) and the
// ScalarFunction/VectorFunction/BFGS machinery it relies on, including when
// gradients and quasi-Newton updates are evaluated, so that the evaluation
// counts are comparable.

var trustConstrMessages = map[int]string{
	0: "The maximum number of function evaluations is exceeded.",
	1: "`gtol` termination condition is satisfied.",
	2: "`xtol` termination condition is satisfied.",
	3: "`callback` raised `StopIteration`.",
	4: "Constraint violation exceeds 'gtol'",
}

// tcObjective is SciPy's ScalarFunction for trust-constr: a one-point cache
// of f, ∇f and the Hessian, with a BFGS approximation (updated each time
// the point moves, rejected trial points included) unless Problem.Hess or
// Problem.HessP is given.
type tcObjective struct {
	p             Problem
	n             int
	fd            fdStep
	x, g          []float64
	f             float64
	fOK, gOK, hOK bool
	nfev, ngev    int
	nhev          int
	bfgs          *bfgsHess
	xPrev, gPrev  []float64
	hmat          []float64
	dx, dg        []float64
}

func newTCObjective(p Problem, x0 []float64, o *Options) *tcObjective {
	n := len(x0)
	if p.FuncGrad != nil {
		// SciPy's MemoizeJac: one call fills both, cached at its point.
		fg := p.FuncGrad
		var mx, mg []float64
		var mf float64
		eval := func(x []float64) {
			if mx == nil || !slices.Equal(mx, x) {
				mx, mg = clone(x), make([]float64, n)
				mf = fg(mx, mg)
			}
		}
		p.Func = func(x []float64) float64 { eval(x); return mf }
		p.Grad = func(x, g []float64) { eval(x); copy(g, mg) }
	}
	ob := &tcObjective{p: p, n: n, x: clone(x0), g: make([]float64, n),
		fd: fdStep{method: o.FDMethod, rel: o.FiniteDiffRelStep},
		dx: make([]float64, n), dg: make([]float64, n)}
	ob.updateFun()
	ob.updateGrad()
	switch {
	case p.Hess != nil:
		ob.hmat = make([]float64, n*n)
		ob.updateHess()
	case p.HessP != nil:
		ob.updateHess()
	default:
		ob.bfgs = newBFGSHess(n)
		ob.hOK = true
	}
	return ob
}

func (ob *tcObjective) updateFun() {
	if !ob.fOK {
		ob.f = ob.p.Func(clone(ob.x))
		ob.nfev++
		ob.fOK = true
	}
}

func (ob *tcObjective) updateGrad() {
	if !ob.gOK {
		if ob.p.Grad != nil {
			ob.p.Grad(clone(ob.x), ob.g)
		} else {
			ob.updateFun()
			ob.nfev += ob.fd.gradFD(ob.p.Func, ob.x, ob.f, ob.g)
		}
		ob.ngev++
		ob.gOK = true
	}
}

func (ob *tcObjective) updateHess() {
	if ob.hOK {
		return
	}
	switch {
	case ob.bfgs != nil:
		ob.updateGrad()
		for i := range ob.dx {
			ob.dx[i] = ob.x[i] - ob.xPrev[i]
			ob.dg[i] = ob.g[i] - ob.gPrev[i]
		}
		ob.bfgs.update(ob.dx, ob.dg)
	case ob.hmat != nil:
		ob.p.Hess(clone(ob.x), ob.hmat)
		ob.nhev++
	default:
		ob.nhev++
	}
	ob.hOK = true
}

func (ob *tcObjective) moveTo(x []float64) {
	if slices.Equal(x, ob.x) {
		return
	}
	if ob.bfgs != nil {
		ob.updateGrad()
		ob.xPrev, ob.gPrev = clone(ob.x), clone(ob.g)
		copy(ob.x, x)
		ob.fOK, ob.gOK, ob.hOK = false, false, false
		ob.updateHess()
		return
	}
	copy(ob.x, x)
	ob.fOK, ob.gOK, ob.hOK = false, false, false
}

func (ob *tcObjective) fun(x []float64) float64 {
	ob.moveTo(x)
	ob.updateFun()
	return ob.f
}

func (ob *tcObjective) grad(x []float64) []float64 {
	ob.moveTo(x)
	ob.updateGrad()
	return ob.g
}

// hess returns the Hessian at x as an operator.
func (ob *tcObjective) hess(x []float64) func(p, out []float64) {
	ob.moveTo(x)
	ob.updateHess()
	switch {
	case ob.bfgs != nil:
		return ob.bfgs.mul
	case ob.hmat != nil:
		return func(p, out []float64) { matvec(ob.n, ob.n, ob.hmat, p, out) }
	}
	xh := clone(ob.x)
	return func(p, out []float64) { ob.p.HessP(xh, p, out) }
}

// tcVecFun is SciPy's VectorFunction (nonlinear constraints, BFGS Hessian
// of vᵀc), LinearVectorFunction or IdentityVectorFunction (bounds), with
// their bounds lb <= f(x) <= ub.
type tcVecFun struct {
	gc            genConstraint
	n, m          int
	identity      bool
	x, f, j       []float64
	fOK, jOK, hOK bool
	v             []float64
	bfgs          *bfgsHess
	xPrev, jPrev  []float64
	nfev, njev    int
	fd            fdStep
	lb, ub        []float64
	jtv, jtvPrev  []float64
	eq, less, grt []int
	interval      []int
	dx, dg, vfull []float64
}

func newTCVecFun(gc genConstraint, x0 []float64, identity bool, fd fdStep) *tcVecFun {
	n := len(x0)
	c := &tcVecFun{gc: gc, n: n, m: gc.m, identity: identity, x: clone(x0), fd: fd,
		f: make([]float64, gc.m), j: make([]float64, gc.m*n), v: make([]float64, gc.m),
		lb: gc.lb, ub: gc.ub, jtv: make([]float64, n), jtvPrev: make([]float64, n),
		dx: make([]float64, n), dg: make([]float64, n), vfull: make([]float64, gc.m)}
	switch {
	case identity:
		for i := range n {
			c.j[i*n+i] = 1
		}
		copy(c.f, x0)
	case gc.linear:
		copy(c.j, gc.a)
		matvec(c.m, n, c.j, c.x, c.f)
	default:
		c.gc.fun(clone(c.x), c.f)
		c.nfev++
		c.evalJac()
		c.bfgs = newBFGSHess(n)
	}
	c.fOK, c.jOK, c.hOK = true, true, true
	for i := range c.m {
		lbInf, ubInf := math.IsInf(c.lb[i], -1), math.IsInf(c.ub[i], 1)
		switch {
		case c.lb[i] == c.ub[i]:
			c.eq = append(c.eq, i)
		case lbInf && !ubInf:
			c.less = append(c.less, i)
		case ubInf && !lbInf:
			c.grt = append(c.grt, i)
		case !lbInf && !ubInf:
			c.interval = append(c.interval, i)
		}
	}
	return c
}

func (c *tcVecFun) nonlinear() bool { return c.bfgs != nil }

func (c *tcVecFun) evalJac() {
	if c.gc.jac != nil {
		c.gc.jac(clone(c.x), c.j)
		c.njev++
		return
	}
	c.nfev += c.fd.jacFD(c.gc.fun, c.x, c.f, c.j)
}

func (c *tcVecFun) updateFun() {
	if !c.fOK {
		if c.nonlinear() {
			c.gc.fun(clone(c.x), c.f)
			c.nfev++
		} else {
			matvec(c.m, c.n, c.j, c.x, c.f)
		}
		c.fOK = true
	}
}

func (c *tcVecFun) updateJac() {
	if !c.jOK {
		if c.gc.jac == nil {
			c.updateFun()
		}
		c.evalJac()
		c.jOK = true
	}
}

func (c *tcVecFun) updateHess() {
	if c.hOK {
		return
	}
	c.updateJac()
	if c.xPrev != nil {
		matTvec(c.m, c.n, c.j, c.v, c.jtv)
		matTvec(c.m, c.n, c.jPrev, c.v, c.jtvPrev)
		for i := range c.dx {
			c.dx[i] = c.x[i] - c.xPrev[i]
			c.dg[i] = c.jtv[i] - c.jtvPrev[i]
		}
		c.bfgs.update(c.dx, c.dg)
	}
	c.hOK = true
}

func (c *tcVecFun) moveTo(x []float64) {
	if slices.Equal(x, c.x) {
		return
	}
	if c.nonlinear() {
		c.updateJac()
		c.xPrev, c.jPrev = clone(c.x), clone(c.j)
		copy(c.x, x)
		c.fOK, c.jOK, c.hOK = false, false, false
		c.updateHess()
		return
	}
	copy(c.x, x)
	c.fOK = false
}

func (c *tcVecFun) fun(x []float64) []float64 {
	c.moveTo(x)
	c.updateFun()
	return c.f
}

func (c *tcVecFun) jac(x []float64) []float64 {
	c.moveTo(x)
	if c.nonlinear() {
		c.updateJac()
	}
	return c.j
}

// hess records the multipliers v and returns the Hessian of vᵀc at x as an
// operator (nil for linear constraints and bounds, whose Hessian is zero).
func (c *tcVecFun) hess(x, v []float64) func(p, out []float64) {
	if !slices.Equal(v, c.v) {
		copy(c.v, v)
		c.hOK = false
	}
	if !c.nonlinear() {
		c.moveTo(x)
		return nil
	}
	c.moveTo(x)
	c.updateHess()
	return c.bfgs.mul
}

// nEq and nIneq are the canonical counts of the constraint.
func (c *tcVecFun) nEq() int   { return len(c.eq) }
func (c *tcVecFun) nIneq() int { return len(c.less) + len(c.grt) + 2*len(c.interval) }

// canonical writes the canonical values of f (equalities f - lb; then
// inequalities <= 0: f - ub, lb - f, f - ub, lb - f for the less, greater
// and both sides of interval components).
func (c *tcVecFun) canonical(f, eq, ineq []float64) {
	for k, i := range c.eq {
		eq[k] = f[i] - c.lb[i]
	}
	k := 0
	for _, i := range c.less {
		ineq[k] = f[i] - c.ub[i]
		k++
	}
	for _, i := range c.grt {
		ineq[k] = c.lb[i] - f[i]
		k++
	}
	for _, i := range c.interval {
		ineq[k] = f[i] - c.ub[i]
		k++
	}
	for _, i := range c.interval {
		ineq[k] = c.lb[i] - f[i]
		k++
	}
}

// canonicalJac writes the rows of the canonical Jacobians.
func (c *tcVecFun) canonicalJac(j, jeq, jineq []float64) {
	n := c.n
	row := func(dst []float64, r int, src int, sign float64) {
		d := dst[r*n : (r+1)*n]
		s := j[src*n : (src+1)*n]
		for t := range d {
			d[t] = sign * s[t]
		}
	}
	for k, i := range c.eq {
		row(jeq, k, i, 1)
	}
	k := 0
	for _, i := range c.less {
		row(jineq, k, i, 1)
		k++
	}
	for _, i := range c.grt {
		row(jineq, k, i, -1)
		k++
	}
	for _, i := range c.interval {
		row(jineq, k, i, 1)
		k++
	}
	for _, i := range c.interval {
		row(jineq, k, i, -1)
		k++
	}
}

// canonicalV maps canonical multipliers back to the constraint's own.
func (c *tcVecFun) canonicalV(veq, vineq []float64) []float64 {
	v := c.vfull
	clear(v)
	for k, i := range c.eq {
		v[i] = veq[k]
	}
	nl, ng, ni := len(c.less), len(c.grt), len(c.interval)
	for k, i := range c.less {
		v[i] = vineq[k]
	}
	for k, i := range c.grt {
		v[i] = -vineq[nl+k]
	}
	for k, i := range c.interval {
		v[i] = vineq[nl+ng+k] - vineq[nl+ng+ni+k]
	}
	return v
}

// tcProblem gathers the objective and the prepared constraints (bounds last)
// in canonical form: c_eq(x) = 0, c_ineq(x) <= 0.
type tcProblem struct {
	n         int
	obj       *tcObjective
	cons      []*tcVecFun
	nEq, nIne int
	hsum      []float64
}

func (tp *tcProblem) constr(x []float64) (ceq, cineq []float64) {
	ceq, cineq = make([]float64, tp.nEq), make([]float64, tp.nIne)
	ke, ki := 0, 0
	for _, c := range tp.cons {
		f := c.fun(x)
		c.canonical(f, ceq[ke:ke+c.nEq()], cineq[ki:ki+c.nIneq()])
		ke += c.nEq()
		ki += c.nIneq()
	}
	return ceq, cineq
}

func (tp *tcProblem) jacs(x []float64) (jeq, jineq []float64) {
	n := tp.n
	jeq, jineq = make([]float64, tp.nEq*n), make([]float64, tp.nIne*n)
	ke, ki := 0, 0
	for _, c := range tp.cons {
		j := c.jac(x)
		c.canonicalJac(j, jeq[ke*n:(ke+c.nEq())*n], jineq[ki*n:(ki+c.nIneq())*n])
		ke += c.nEq()
		ki += c.nIneq()
	}
	return jeq, jineq
}

// initialCanonical is initial_constraints_as_canonical: the stored values.
func (tp *tcProblem) initialCanonical() (ceq, cineq, jeq, jineq []float64) {
	n := tp.n
	ceq, cineq = make([]float64, tp.nEq), make([]float64, tp.nIne)
	jeq, jineq = make([]float64, tp.nEq*n), make([]float64, tp.nIne*n)
	ke, ki := 0, 0
	for _, c := range tp.cons {
		c.canonical(c.f, ceq[ke:ke+c.nEq()], cineq[ki:ki+c.nIneq()])
		c.canonicalJac(c.j, jeq[ke*n:(ke+c.nEq())*n], jineq[ki*n:(ki+c.nIneq())*n])
		ke += c.nEq()
		ki += c.nIneq()
	}
	return
}

// lagrHess is LagrangianHessian: ∇²f(x) + Σ ∇²(vᵀc)(x) as an operator.
func (tp *tcProblem) lagrHess(x, veq, vineq []float64) func(p, out []float64) {
	hobj := tp.obj.hess(x)
	var hs []func(p, out []float64)
	ke, ki := 0, 0
	for _, c := range tp.cons {
		v := c.canonicalV(veq[ke:ke+c.nEq()], vineq[ki:ki+c.nIneq()])
		if h := c.hess(x, v); h != nil {
			hs = append(hs, h)
		}
		ke += c.nEq()
		ki += c.nIneq()
	}
	n := tp.n
	tmp := make([]float64, n)
	return func(p, out []float64) {
		hobj(p, out)
		for _, h := range hs {
			h(p, tmp)
			for i := range out[:n] {
				out[i] += tmp[i]
			}
		}
	}
}

// tcState is the OptimizeResult that trust-constr updates at each stop test.
type tcState struct {
	nit, nfev, njev, nhev int
	x, grad               []float64
	fun                   float64
	optimality, cv        float64
	trRadius, penalty     float64
	cgNiter, cgStop       int
	barrier, barrierTol   float64
	status                int
	v                     [][]float64
}

func (tp *tcProblem) updateState(st *tcState, x []float64, lastFailed bool, tr, penalty float64, cg cgInfo) {
	st.nit++
	st.nfev, st.njev, st.nhev = tp.obj.nfev, tp.obj.ngev, tp.obj.nhev
	if !lastFailed {
		st.x = clone(x)
		st.fun = tp.obj.f
		st.grad = clone(tp.obj.g)
		lg := clone(tp.obj.g)
		st.v = st.v[:0]
		st.cv = 0
		for _, c := range tp.cons {
			st.v = append(st.v, clone(c.v))
			tmp := make([]float64, tp.n)
			matTvec(c.m, c.n, c.j, c.v, tmp)
			for i := range lg {
				lg[i] += tmp[i]
			}
			for i := range c.m {
				st.cv = math.Max(st.cv, math.Max(c.lb[i]-c.f[i], c.f[i]-c.ub[i]))
			}
		}
		st.optimality = vnormInf(lg)
	}
	st.trRadius, st.penalty = tr, penalty
	st.cgNiter += cg.niter
	st.cgStop = cg.stopCond
}

// tcOpts are the trust-constr options with SciPy's defaults filled in.
type tcOpts struct {
	gtol, xtol, barrierTol         float64
	maxiter                        int
	penalty0, tr0, barrier0, btol0 float64
	callback                       func(x []float64, f float64) bool
}

// sqpFuncs are the callbacks of equality_constrained_sqp.
type sqpFuncs struct {
	funAndConstr func(x []float64) (float64, []float64)
	gradAndJac   func(x []float64) ([]float64, []float64)
	lagrHess     func(x, v []float64) func(p, out []float64)
	stop         func(x []float64, lastFailed bool, optimality, cv, tr, penalty float64, cg cgInfo) bool
	scaling      func(x []float64) []float64 // diagonal of the scaling, nil for identity
}

// equalityConstrainedSQP is equality_constrained_sqp (Byrd–Omojokun
// trust-region SQP; Lalee, Nocedal and Plantenga 1998). It returns the final
// point and its trust radius.
func equalityConstrainedSQP(fs sqpFuncs, x0 []float64, f0 float64, c0, b0, a0 []float64,
	penalty0, tr0 float64, trustLb, trustUb []float64) ([]float64, float64, error) {
	const (
		penaltyFactor  = 0.3
		largeRatio     = 0.9
		interRatio     = 0.3
		suffRatio      = 1e-8
		enlargeL       = 7.0
		enlargeS       = 2.0
		maxTrReduction = 0.5
		minTrReduction = 0.1
		socThreshold   = 0.1
		trFactor       = 0.8
		boxFactor      = 0.5
	)
	n, m := len(x0), len(b0)
	x := clone(x0)
	tr, penalty := tr0, penalty0
	f, c, b, a := f0, clone(c0), clone(b0), clone(a0)
	scale := func(x, d []float64) []float64 {
		out := clone(d)
		if fs.scaling != nil {
			s := fs.scaling(x)
			for i := range out {
				out[i] *= s[i]
			}
		}
		return out
	}
	proj, ok := newProjector(m, n, a)
	if !ok {
		return nil, 0, fmt.Errorf("%w: more equality constraints than independent variables", ErrInput)
	}
	v := make([]float64, m)
	proj.leastSquares(c, v)
	for i := range v {
		v[i] = -v[i]
	}
	h := fs.lagrHess(x, v)
	optim := func() float64 {
		t := make([]float64, n)
		matTvec(m, n, a, v, t)
		for i := range t {
			t[i] += c[i]
		}
		return vnormInf(t)
	}
	optimality := optim()
	cv := vnormInf(b)
	cg := cgInfo{}
	lastFailed := false
	lbBox, ubBox := make([]float64, n), make([]float64, n)
	for i := range n {
		lbBox[i], ubBox[i] = boxFactor*trustLb[i], boxFactor*trustUb[i]
	}
	hd, ad := make([]float64, n), make([]float64, m)
	for !fs.stop(x, lastFailed, optimality, cv, tr, penalty, cg) {
		dn := modifiedDogleg(proj, b, trFactor*tr, lbBox, ubBox)
		h(dn, hd)
		ct := make([]float64, n)
		for i := range ct {
			ct[i] = hd[i] + c[i]
		}
		trT := math.Sqrt(tr*tr - floats.Dot(dn, dn))
		lbT, ubT := make([]float64, n), make([]float64, n)
		for i := range n {
			lbT[i], ubT[i] = trustLb[i]-dn[i], trustUb[i]-dn[i]
		}
		var dt []float64
		dt, cg = projectedCG(h, ct, proj, trT, lbT, ubT)
		d := make([]float64, n)
		for i := range d {
			d[i] = dn[i] + dt[i]
		}
		h(d, hd)
		quad := 0.5*floats.Dot(hd, d) + floats.Dot(c, d)
		matvec(m, n, a, d, ad)
		for i := range ad {
			ad[i] += b[i]
		}
		vpred := math.Max(1e-16, vnorm(b)-vnorm(ad))
		prevPenalty := penalty
		if quad > 0 {
			penalty = math.Max(penalty, quad/((1-penaltyFactor)*vpred))
		}
		pred := -quad + penalty*vpred
		merit := f + penalty*vnorm(b)
		xNext := clone(x)
		floats.Axpy(1, scale(x, d), xNext)
		fNext, bNext := fs.funAndConstr(xNext)
		ratio := (merit - (fNext + penalty*vnorm(bNext))) / pred
		if ratio < suffRatio && vnorm(dn) <= socThreshold*vnorm(dt) {
			y := make([]float64, n)
			nb := make([]float64, m)
			for i := range nb {
				nb[i] = -bNext[i]
			}
			proj.rowSpace(nb, y)
			_, t, intersect := boxIntersections(d, y, trustLb, trustUb, false)
			dsoc := clone(d)
			floats.Axpy(t, y, dsoc)
			xSoc := clone(x)
			floats.Axpy(1, scale(x, dsoc), xSoc)
			fSoc, bSoc := fs.funAndConstr(xSoc)
			ratioSoc := (merit - (fSoc + penalty*vnorm(bSoc))) / pred
			if intersect && ratioSoc >= suffRatio {
				xNext, fNext, bNext, ratio = xSoc, fSoc, bSoc, ratioSoc
			}
		}
		nd := vnorm(d)
		switch {
		case ratio >= largeRatio:
			tr = math.Max(enlargeL*nd, tr)
		case ratio >= interRatio:
			tr = math.Max(enlargeS*nd, tr)
		case ratio < suffRatio:
			newTr := (1 - suffRatio) / (1 - ratio) * nd
			switch {
			case newTr >= maxTrReduction*tr:
				tr *= maxTrReduction
			case newTr >= minTrReduction*tr:
				tr = newTr
			default:
				tr *= minTrReduction
			}
		}
		if ratio >= suffRatio {
			x, f, b = xNext, fNext, bNext
			c, a = fs.gradAndJac(x)
			// m <= n was checked on the first factorisation and the shape
			// does not change.
			proj, _ = newProjector(m, n, a)
			proj.leastSquares(c, v)
			for i := range v {
				v[i] = -v[i]
			}
			h = fs.lagrHess(x, v)
			lastFailed = false
			optimality = optim()
			cv = vnormInf(b)
		} else {
			penalty = prevPenalty
			lastFailed = true
		}
	}
	return x, tr, nil
}

// minimizeTrustConstr is _minimize_trustregion_constr.
func minimizeTrustConstr(p Problem, x0 []float64, o *Options, lb, ub []float64, gcs []genConstraint) (*Result, error) {
	n := len(x0)
	opt := tcOpts{gtol: 1e-8, xtol: 1e-8, barrierTol: 1e-8, maxiter: 1000,
		penalty0: 1, tr0: 1, barrier0: 0.1, btol0: 0.1, callback: o.Callback}
	if o.Tol != 0 {
		opt.gtol, opt.xtol, opt.barrierTol = o.Tol, o.Tol, o.Tol
	}
	if o.Gtol != 0 {
		opt.gtol = o.Gtol
	}
	if o.Xtol != 0 {
		opt.xtol = o.Xtol
	}
	if o.BarrierTol != 0 {
		opt.barrierTol = o.BarrierTol
	}
	if o.MaxIter != 0 {
		opt.maxiter = o.MaxIter
	}
	if o.InitialTrRadius != 0 {
		opt.tr0 = o.InitialTrRadius
	}
	if o.InitialConstrPenalty != 0 {
		opt.penalty0 = o.InitialConstrPenalty
	}
	if o.InitialBarrierParameter != 0 {
		opt.barrier0 = o.InitialBarrierParameter
	}
	if o.InitialBarrierTolerance != 0 {
		opt.btol0 = o.InitialBarrierTolerance
	}
	fd := fdStep{method: o.FDMethod, rel: o.FiniteDiffRelStep}
	tp := &tcProblem{n: n, obj: newTCObjective(p, x0, o)}
	for _, gc := range gcs {
		tp.cons = append(tp.cons, newTCVecFun(gc, x0, false, fd))
	}
	nUser := len(tp.cons)
	if lb != nil {
		// SciPy widens finite bounds by one ulp before using them.
		blb, bub := clone(lb), clone(ub)
		for i := range n {
			if !math.IsInf(blb[i], 0) {
				blb[i] = math.Nextafter(blb[i], math.Inf(-1))
			}
			if !math.IsInf(bub[i], 0) {
				bub[i] = math.Nextafter(bub[i], math.Inf(1))
			}
		}
		tp.cons = append(tp.cons, newTCVecFun(genConstraint{m: n, lb: blb, ub: bub}, x0, true, fd))
	}
	for _, c := range tp.cons {
		tp.nEq += c.nEq()
		tp.nIne += c.nIneq()
	}
	ceq0, cineq0, jeq0, jineq0 := tp.initialCanonical()

	st := &tcState{status: -1, fun: tp.obj.f, grad: clone(tp.obj.g), x: clone(x0)}
	for _, c := range tp.cons {
		st.v = append(st.v, clone(c.v))
	}
	global := func(x []float64, lastFailed bool, tr, penalty float64, cg cgInfo, barrier, btol float64, ip bool) bool {
		tp.updateState(st, x, lastFailed, tr, penalty, cg)
		st.barrier, st.barrierTol = barrier, btol
		st.status = -1
		if opt.callback != nil && opt.callback(clone(st.x), st.fun) {
			st.status = 3
			return true
		}
		switch {
		case st.optimality < opt.gtol && st.cv < opt.gtol:
			st.status = 1
		case st.trRadius < opt.xtol && (!ip || st.barrier < opt.barrierTol):
			st.status = 2
		case st.nit >= opt.maxiter:
			st.status = 0
		}
		return st.status >= 0
	}

	var err error
	if tp.nIne == 0 {
		fs := sqpFuncs{
			funAndConstr: func(x []float64) (float64, []float64) {
				f := tp.obj.fun(x)
				ceq, _ := tp.constr(x)
				return f, ceq
			},
			gradAndJac: func(x []float64) ([]float64, []float64) {
				g := clone(tp.obj.grad(x))
				jeq, _ := tp.jacs(x)
				return g, jeq
			},
			lagrHess: func(x, v []float64) func(p, out []float64) { return tp.lagrHess(x, v, nil) },
			stop: func(x []float64, lastFailed bool, _, _, tr, penalty float64, cg cgInfo) bool {
				return global(x, lastFailed, tr, penalty, cg, 0, 0, false)
			},
		}
		lbT, ubT := make([]float64, n), make([]float64, n)
		for i := range n {
			lbT[i], ubT[i] = math.Inf(-1), math.Inf(1)
		}
		_, _, err = equalityConstrainedSQP(fs, x0, tp.obj.f, clone(tp.obj.g), ceq0, jeq0, opt.penalty0, opt.tr0, lbT, ubT)
	} else {
		err = trInteriorPoint(tp, st, opt, x0, ceq0, cineq0, jeq0, jineq0, global)
	}
	if err != nil {
		return nil, err
	}
	if (st.status == 1 || st.status == 2) && st.cv > opt.gtol {
		st.status = 4
	}
	var mult []float64
	for _, v := range st.v[:nUser] {
		mult = append(mult, v...)
	}
	r := &Result{X: st.x, Fun: st.fun, Jac: st.grad, Nit: st.nit, Nfev: st.nfev, Njev: st.njev,
		Nhev: st.nhev, Status: st.status, Success: st.status == 1 || st.status == 2,
		Message: trustConstrMessages[st.status], Multipliers: mult,
		Optimality: st.optimality, ConstrViolation: st.cv}
	return r, nil
}

// barrierValue is f - μ Σ log s, with log s = -Inf for s <= 0 (SciPy's
// _compute_function).
func barrierValue(f, mu float64, s []float64) float64 {
	sum := 0.0
	for _, si := range s {
		if si > 0 {
			sum += math.Log(si)
		} else {
			sum += math.Inf(-1)
		}
	}
	return f - mu*sum
}

// trInteriorPoint is tr_interior_point: a sequence of barrier problems
//
//	min f(x) - μ Σ log s  s.t.  c_eq(x) = 0, c_ineq(x) + s = 0
//
// each solved by equalityConstrainedSQP in the scaled variables z = (x, s).
func trInteriorPoint(tp *tcProblem, st *tcState, opt tcOpts, x0, ceq0, cineq0, jeq0, jineq0 []float64,
	global func(x []float64, lastFailed bool, tr, penalty float64, cg cgInfo, barrier, btol float64, ip bool) bool) error {
	const (
		boundary = 0.995
		decay    = 0.2
		enlarge  = 5.0
	)
	n, ni, ne := tp.n, tp.nIne, tp.nEq
	nz := n + ni
	mu, tol, tr := opt.barrier0, opt.btol0, opt.tr0
	s0 := make([]float64, ni)
	for i := range s0 {
		s0[i] = math.Max(-1.5*cineq0[i], 1)
	}
	barrierF := func(f float64, s []float64) float64 { return barrierValue(f, mu, s) }
	barrierC := func(ceq, cineq, s []float64) []float64 {
		out := make([]float64, ne+ni)
		copy(out, ceq)
		for i := range ni {
			out[ne+i] = cineq[i] + s[i]
		}
		return out
	}
	barrierG := func(g []float64) []float64 {
		out := make([]float64, nz)
		copy(out, g)
		for i := n; i < nz; i++ {
			out[i] = -mu
		}
		return out
	}
	barrierJ := func(jeq, jineq, s []float64) []float64 {
		out := make([]float64, (ne+ni)*nz)
		for r := range ne {
			copy(out[r*nz:r*nz+n], jeq[r*n:(r+1)*n])
		}
		for r := range ni {
			copy(out[(ne+r)*nz:(ne+r)*nz+n], jineq[r*n:(r+1)*n])
			out[(ne+r)*nz+n+r] = s[r]
		}
		return out
	}
	terminate := false
	fs := sqpFuncs{
		funAndConstr: func(z []float64) (float64, []float64) {
			x, s := z[:n], z[n:]
			ceq, cineq := tp.constr(x)
			f := tp.obj.fun(x)
			return barrierF(f, s), barrierC(ceq, cineq, s)
		},
		gradAndJac: func(z []float64) ([]float64, []float64) {
			x, s := z[:n], z[n:]
			g := barrierG(tp.obj.grad(x))
			jeq, jineq := tp.jacs(x)
			return g, barrierJ(jeq, jineq, s)
		},
		lagrHess: func(z, v []float64) func(p, out []float64) {
			x, s := z[:n], z[n:]
			hx := tp.lagrHess(x, v[:ne], v[ne:ne+ni])
			shs := make([]float64, ni)
			for i := range ni {
				if vi := v[ne+i]; vi > 0 {
					shs[i] = vi * s[i]
				} else {
					shs[i] = mu
				}
			}
			return func(p, out []float64) {
				hx(p[:n], out[:n])
				for i := range ni {
					out[n+i] = shs[i] * p[n+i]
				}
			}
		},
		stop: func(z []float64, lastFailed bool, optimality, cv, trr, penalty float64, cg cgInfo) bool {
			if global(z[:n], lastFailed, trr, penalty, cg, mu, tol, true) {
				terminate = true
				return true
			}
			return (optimality < tol && cv < tol) || trr < opt.xtol
		},
		scaling: func(z []float64) []float64 {
			d := make([]float64, nz)
			for i := range n {
				d[i] = 1
			}
			copy(d[n:], z[n:])
			return d
		},
	}
	z := make([]float64, nz)
	copy(z, x0)
	copy(z[n:], s0)
	f0 := barrierF(tp.obj.f, s0)
	c0 := barrierC(ceq0, cineq0, s0)
	g0 := barrierG(tp.obj.g)
	j0 := barrierJ(jeq0, jineq0, s0)
	trustLb, trustUb := make([]float64, nz), make([]float64, nz)
	for i := range nz {
		trustLb[i], trustUb[i] = math.Inf(-1), math.Inf(1)
		if i >= n {
			trustLb[i] = -boundary
		}
	}
	for {
		var err error
		z, _, err = equalityConstrainedSQP(fs, z, f0, g0, c0, j0, opt.penalty0, tr, trustLb, trustUb)
		if err != nil {
			return err
		}
		if terminate {
			return nil
		}
		// state.tr_radius is the radius of the last stop test.
		tr = math.Max(opt.tr0, enlarge*st.trRadius)
		mu *= decay
		tol *= decay
		f0, c0 = fs.funAndConstr(z)
		g0, j0 = fs.gradAndJac(z)
	}
}
