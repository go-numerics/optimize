package optimize

import (
	"fmt"
	"math"
)

// SLSQP is Kraft's sequential least-squares quadratic programming method
// (scipy.optimize.minimize(method='SLSQP'), MATLAB fmincon's 'sqp'). It
// handles bounds and the equality and inequality constraints of
// Options.Constraints. The port follows SciPy v1.18.1's C implementation
// (scipy/optimize/src/slsqp.c and _slsqp_py.py, BSD-3-Clause) statement by
// statement.

// slsqpMessages are SciPy's exit_modes.
var slsqpMessages = map[int]string{
	0: "Optimization terminated successfully",
	2: "More equality constraints than independent variables",
	3: "More than 3*n iterations in LSQ subproblem",
	4: "Inequality constraints incompatible",
	5: "Singular matrix E in LSQ subproblem",
	6: "Singular matrix C in LSQ subproblem",
	7: "Rank-deficient equality constraint subproblem HFTI",
	8: "Positive directional derivative for linesearch",
	9: "Iteration limit reached",
}

// slsqpState is SciPy's struct SLSQP_vars plus the persistent part of its
// work buffer.
type slsqpState struct {
	acc, alpha, f0, gs, h1, h2, h3, h4, t, t0, tol float64
	inconsistent, reset, iter, itermax, line, mode int
	m, meq, n                                      int
	bfgs, x0, mu, s, u, v                          []float64
	ws                                             arena // scratch of the subproblem solvers
}

func newSLSQPState(n, m, meq int, acc float64, itermax int) *slsqpState {
	n1 := n + 1
	return &slsqpState{
		acc: acc, tol: 10 * acc, itermax: itermax, m: m, meq: meq, n: n,
		bfgs: make([]float64, n1*n/2), x0: make([]float64, n), mu: make([]float64, m),
		s: make([]float64, n1), u: make([]float64, n1), v: make([]float64, n1),
	}
}

// eqPart is d[j] for an equality and 0 for an inequality, the reference
// value of SLSQP's constraint violation max(-d[j], eqPart).
func (S *slsqpState) eqPart(d []float64, j int) float64 {
	if j < S.meq {
		return d[j]
	}
	return 0
}

// violation is sum_j max(-d[j], eqPart(j)), weighted by w when w != nil.
func (S *slsqpState) violation(d, w []float64) float64 {
	h := 0.0
	for j := 0; j < S.m; j++ {
		if w == nil {
			h += cfmax(-d[j], S.eqPart(d, j))
		} else {
			h += float64(w[j] * cfmax(-d[j], S.eqPart(d, j)))
		}
	}
	return h
}

// body is __slsqp_body: one reverse-communication step. On return S.mode
// is 1 (evaluate f and the constraints at sol), -1 (evaluate the gradient
// and the constraint Jacobian at sol), 0 (converged) or an error mode. c is
// the m×n constraint Jacobian, column-major with leading dimension ldc.
func (S *slsqpState) body(funx float64, gradx, c []float64, ldc int, d, sol, mult, xl, xu []float64) {
	const alfmin = 0.1
	n, m := S.n, S.m
	bfgs, x0, mu, s, u, v := S.bfgs, S.x0, S.mu, S.s, S.u, S.v
	// badlin is a local of the C function, reset on every call.
	badlin := false
	var j int
	var rho float64
	switch S.mode {
	case -1:
		goto MODEM1
	case 1:
		goto MODE1
	}

	// MODE0: start.
	S.acc = math.Abs(S.acc)
	S.tol = float64(10 * S.acc)
	S.iter = 0
	S.reset = 0
	for i := 0; i < n; i++ {
		s[i] = 0
	}
	for i := 0; i < m; i++ {
		mu[i] = 0
	}

RESETBFGS:
	S.reset++
	if S.reset > 5 {
		goto LABEL255
	}
	for i := range bfgs {
		bfgs[i] = 0
	}
	j = 0
	for i := 0; i < n; i++ {
		bfgs[j] = 1
		j += n - i
	}

ITERSTART:
	S.mode = 9
	if S.iter >= S.itermax {
		return
	}
	S.iter++
	for i := 0; i < n; i++ {
		u[i] = -sol[i] + xl[i]
		v[i] = -sol[i] + xu[i]
	}
	S.h4 = 1
	S.mode = lsq(&S.ws, m, S.meq, n, false, 0, bfgs, gradx, c, ldc, d, u, v, s, mult)
	badlin = false
	if S.mode == 6 && n == S.meq {
		S.mode = 4
	}
	if S.mode == 4 {
		badlin = true
		for i := 0; i < n; i++ {
			s[i] = 0
		}
		S.h3 = 0
		rho = 100
		S.inconsistent = 0
		for {
			S.mode = lsq(&S.ws, m, S.meq, n, true, rho, bfgs, gradx, c, ldc, d, u, v, s, mult)
			S.h4 = 1 - s[n]
			if S.mode == 4 {
				rho *= 10
				S.inconsistent++
				if S.inconsistent > 5 {
					return
				}
				continue
			}
			if S.mode != 1 {
				return
			}
			break
		}
	} else if S.mode != 1 {
		return
	}

	// Multipliers for the L1 test.
	copy(v[:n], gradx[:n])
	gemvT(m, n, -1, c, ldc, mult, 1, v)
	S.f0 = funx
	copy(x0, sol[:n])
	S.gs = dot(gradx[:n], s[:n])
	S.h1 = math.Abs(S.gs)
	S.h2 = 0
	for j := 0; j < m; j++ {
		S.h3 = S.eqPart(d, j)
		S.h2 += cfmax(-d[j], S.h3)
		S.h3 = math.Abs(mult[j])
		mu[j] = cfmax(S.h3, (mu[j]+S.h3)/2)
		S.h1 += float64(S.h3 * math.Abs(d[j]))
	}
	S.mode = 0
	if S.h1 < S.acc && S.h2 < S.acc && !badlin && funx == funx {
		return
	}
	S.h1 = S.violation(d, mu)
	S.t0 = funx + S.h1
	S.h3 = S.gs - float64(S.h1*S.h4)
	S.mode = 8
	if S.h3 >= 0 {
		goto RESETBFGS
	}
	S.line = 0
	S.alpha = 1

LINESEARCH:
	S.line++
	S.h3 = float64(S.alpha * S.h3)
	scal(n, S.alpha, s, 1)
	for i := 0; i < n; i++ {
		sol[i] = x0[i] + s[i]
	}
	S.mode = 1
	return

MODE1:
	S.t = funx + S.violation(d, mu)
	S.h1 = S.t - S.t0
	if !(S.h1 <= S.h3/10 || S.line > 10) {
		S.alpha = cfmax(S.h3/(2*(S.h3-S.h1)), alfmin)
		goto LINESEARCH
	}
	S.h3 = S.violation(d, nil)
	if (math.Abs(funx-S.f0) < S.acc || nrm2(n, s, 1) < S.acc) && S.h3 < S.acc && !badlin && funx == funx {
		S.mode = 0
	} else {
		S.mode = -1
	}
	return

LABEL255:
	S.h3 = S.violation(d, nil)
	if (math.Abs(funx-S.f0) < S.tol || nrm2(n, s, 1) < S.tol) && S.h3 < S.tol && !badlin && funx == funx {
		S.mode = 0
	} else {
		S.mode = 8
	}
	return

MODEM1:
	// BFGS update of the LDL' factors with Powell's modification.
	copy(u[:n], gradx[:n])
	gemvT(m, n, -1, c, ldc, mult, 1, u)
	for i := 0; i < n; i++ {
		u[i] -= v[i]
	}
	copy(v[:n], s[:n])
	tpmvLTU(n, bfgs, v)
	j = 0
	for i := 0; i < n; i++ {
		v[i] = float64(bfgs[j] * v[i])
		j += n - i
	}
	tpmvLNU(n, bfgs, v)
	S.h1 = dot(s[:n], u[:n])
	S.h2 = dot(s[:n], v[:n])
	S.h3 = float64(0.2 * S.h2)
	if S.h1 < S.h3 {
		S.h4 = (S.h2 - S.h3) / (S.h2 - S.h1)
		S.h1 = S.h3
		scal(n, S.h4, u, 1)
		t := 1 - S.h4
		for i := 0; i < n; i++ {
			u[i] += float64(t * v[i])
		}
	}
	if S.h1 == 0 || S.h2 == 0 {
		goto RESETBFGS
	}
	ldlUpdate(n, bfgs, u, 1/S.h1, v)
	ldlUpdate(n, bfgs, v, -1/S.h2, u)
	goto ITERSTART
}

// slsqpCons evaluates the standardised constraints: equalities first, then
// inequalities, each in the order given, as _eval_constraint does.
type slsqpCons struct {
	blocks []stdConstraint // equality blocks, then inequality blocks
	m, meq int
	fd     fdStep
	buf    []float64 // block values
	jbuf   []float64 // block Jacobian, row-major
	f0     []float64
}

func newSLSQPCons(cons []stdConstraint, n int, fd fdStep) *slsqpCons {
	sc := &slsqpCons{fd: fd}
	maxm := 0
	for pass := 0; pass < 2; pass++ {
		for _, c := range cons {
			if c.eq == (pass == 0) {
				sc.blocks = append(sc.blocks, c)
				sc.m += c.m
				if c.eq {
					sc.meq += c.m
				}
				maxm = max(maxm, c.m)
			}
		}
	}
	sc.buf = make([]float64, maxm)
	sc.f0 = make([]float64, maxm)
	sc.jbuf = make([]float64, maxm*n)
	return sc
}

// eval writes the constraint values at x into d.
func (sc *slsqpCons) eval(x, d []float64) {
	row := 0
	for _, c := range sc.blocks {
		c.fun(x, d[row:row+c.m])
		row += c.m
	}
}

// jac writes the constraint Jacobian at x into the column-major m×n matrix
// cm with leading dimension ldc.
func (sc *slsqpCons) jac(x, cm []float64, ldc int) {
	n := len(x)
	row := 0
	for _, c := range sc.blocks {
		jb := sc.jbuf[:c.m*n]
		if c.jac != nil {
			c.jac(x, jb)
		} else {
			f0 := sc.f0[:c.m]
			c.fun(x, f0)
			sc.fd.jacFD(c.fun, x, f0, jb)
		}
		for i := 0; i < c.m; i++ {
			for k := 0; k < n; k++ {
				cm[row+i+k*ldc] = jb[i*n+k]
			}
		}
		row += c.m
	}
}

// minimizeSLSQP is _minimize_slsqp. lb and ub are nil without bounds; x0 is
// already clipped into them.
func minimizeSLSQP(p Problem, x0 []float64, o *Options, lb, ub []float64, cons []stdConstraint) *Result {
	n := len(x0)
	acc := o.Ftol
	if acc == 0 {
		acc = o.Tol
	}
	if acc == 0 {
		acc = 1e-6
	}
	maxiter := o.MaxIter
	if maxiter == 0 {
		maxiter = 100
	}
	xl, xu := make([]float64, n), make([]float64, n)
	for i := range xl {
		xl[i], xu[i] = math.NaN(), math.NaN()
		if lb != nil {
			if !math.IsInf(lb[i], 0) {
				xl[i] = lb[i]
			}
			if !math.IsInf(ub[i], 0) {
				xu[i] = ub[i]
			}
		}
	}
	sf := newScalarFunc(p, SLSQP, o, lb, ub)
	cfd := fdStep{method: FD2Point, abs: sf.fd.abs, lb: lb, ub: ub}
	if o.FDMethod == FD3Point || o.FiniteDiffRelStep != 0 {
		cfd = fdStep{method: o.FDMethod, rel: o.FiniteDiffRelStep, lb: lb, ub: ub}
	}
	sc := newSLSQPCons(cons, n, cfd)
	m, meq := sc.m, sc.meq

	x := clone(x0)
	g := make([]float64, n)
	fx := sf.fun(x)
	sf.grad(x, g)
	ldc := max(m, 1)
	cm := make([]float64, ldc*n)
	d := make([]float64, max(m, 1))
	mult := make([]float64, max(1, m+2*n+2))
	sc.jac(x, cm, ldc)
	sc.eval(x, d)

	st := newSLSQPState(n, m, meq, acc, maxiter)
	iterPrev := 0
	for {
		st.body(fx, g, cm, ldc, d, x, mult, xl, xu)
		if st.mode == 1 || st.mode == -1 {
			// _slsqpmodule.c: remove the few-ulp bound violations of the
			// line-search point before evaluating there.
			clampNaNBounds(x, xl, xu)
		}
		if st.mode == 1 {
			fx = sf.fun(x)
			sc.eval(x, d)
		}
		if st.mode == -1 {
			sf.grad(x, g)
			sc.jac(x, cm, ldc)
		}
		if st.iter > iterPrev && o.Callback != nil && o.Callback(clone(x), fx) {
			break
		}
		if st.mode != 1 && st.mode != -1 {
			break
		}
		iterPrev = st.iter
	}
	r := &Result{
		X: x, Fun: fx, Jac: g, Nit: st.iter, Status: st.mode,
		Message: slsqpMessages[st.mode], Success: st.mode == 0,
		Multipliers: clone(mult[:m]),
	}
	return sf.result(r)
}

// minimizeSLSQPFixed handles variables fixed by equal bounds as SciPy's
// minimize does for SLSQP: when every variable is fixed the objective is
// evaluated once and the constraints are checked; when some are fixed and a
// finite-difference derivative is needed anywhere, they are removed from the
// problem and restored in the result (with NaN gradient entries).
func minimizeSLSQPFixed(p Problem, x0 []float64, o *Options, lb, ub []float64, cons []stdConstraint) (*Result, error) {
	n := len(x0)
	if lb == nil {
		return minimizeSLSQP(p, x0, o, lb, ub, cons), nil
	}
	var free []int
	for i := range n {
		if lb[i] != ub[i] {
			free = append(free, i)
		}
	}
	if len(free) == 0 {
		return slsqpAllFixed(p, lb, o, cons), nil
	}
	fdNeeded := p.Grad == nil && p.FuncGrad == nil
	for _, c := range cons {
		fdNeeded = fdNeeded || c.jac == nil
	}
	if len(free) == n || !fdNeeded {
		return minimizeSLSQP(p, x0, o, lb, ub, cons), nil
	}
	nr := len(free)
	full := clone(lb)
	expand := func(xr []float64) []float64 {
		for k, i := range free {
			full[i] = xr[k]
		}
		return full
	}
	gfull := make([]float64, n)
	compress := func(gr []float64) {
		for k, i := range free {
			gr[k] = gfull[i]
		}
	}
	var rp Problem
	switch {
	case p.FuncGrad != nil:
		rp.FuncGrad = func(x, g []float64) float64 {
			f := p.FuncGrad(expand(x), gfull)
			compress(g)
			return f
		}
	case p.Func != nil:
		rp.Func = func(x []float64) float64 { return p.Func(expand(x)) }
		if p.Grad != nil {
			rp.Grad = func(x, g []float64) {
				p.Grad(expand(x), gfull)
				compress(g)
			}
		}
	}
	rcons := make([]stdConstraint, len(cons))
	for ci, c := range cons {
		c := c
		rc := stdConstraint{eq: c.eq, m: c.m}
		rc.fun = func(x, out []float64) { c.fun(expand(x), out) }
		if c.jac != nil {
			jf := make([]float64, c.m*n)
			rc.jac = func(x, j []float64) {
				c.jac(expand(x), jf)
				for r := 0; r < c.m; r++ {
					for k, i := range free {
						j[r*nr+k] = jf[r*n+i]
					}
				}
			}
		}
		rcons[ci] = rc
	}
	xr, lr, ur := make([]float64, nr), make([]float64, nr), make([]float64, nr)
	for k, i := range free {
		xr[k], lr[k], ur[k] = x0[i], lb[i], ub[i]
	}
	ro := *o
	if cb := o.Callback; cb != nil {
		ro.Callback = func(x []float64, f float64) bool { return cb(clone(expand(x)), f) }
	}
	r := minimizeSLSQP(rp, xr, &ro, lr, ur, rcons)
	x := clone(lb)
	jac := make([]float64, n)
	for i := range jac {
		jac[i] = math.NaN()
	}
	for k, i := range free {
		x[i] = r.X[k]
		jac[i] = r.Jac[k]
	}
	r.X, r.Jac = x, jac
	return r, nil
}

// slsqpAllFixed is _optimize_result_for_equal_bounds.
func slsqpAllFixed(p Problem, x []float64, o *Options, cons []stdConstraint) *Result {
	x = clone(x)
	sf := newScalarFunc(p, SLSQP, o, nil, nil)
	r := &Result{X: x, Fun: sf.fun(x), Nfev: 1, Success: true,
		Message: "All independent variables were fixed by bounds."}
	if len(cons) == 0 {
		return r
	}
	r.Message = "All independent variables were fixed by bounds at values that satisfy the constraints."
	maxcv := 0.0
	for _, c := range cons {
		v := make([]float64, c.m)
		c.fun(x, v)
		sum := 0.0
		for _, ci := range v {
			ex := math.Max(-ci, 0) // ineq: 0 <= c(x)
			if c.eq {
				ex += math.Max(ci, 0) // eq: 0 <= c(x) <= 0
			}
			sum += ex
			maxcv = math.Max(maxcv, ex)
		}
		if sum != 0 {
			r.Success = false
			r.Message = fmt.Sprintf("All independent variables were fixed by bounds, but the "+
				"independent variables do not satisfy the constraints exactly. (Maximum violation: %s).",
				pyRepr(maxcv))
		}
	}
	return r
}
