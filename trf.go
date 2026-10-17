package optimize

import (
	"errors"
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/linalg"
)

// This file is the trust-region reflective (TRF) method of LeastSquares, a
// port of scipy.optimize.least_squares(method='trf', tr_solver='exact')
// (scipy/optimize/_lsq/trf.py and _lsq/common.py), after Branch, Coleman
// and Li, "A Subspace, Interior, and Conjugate Gradient Method for
// Large-Scale Bound-Constrained Minimization Problems", SIAM J. Sci. Comput.
// 21(1), 1999, with the trust-region subproblem solved exactly from an SVD as
// in Moré, "The Levenberg–Marquardt Algorithm: Implementation and Theory",
// 1978. Without bounds it reduces to a classical trust-region
// Levenberg–Marquardt iteration.

// noStatus stands for SciPy's termination_status = None.
const noStatus = -100

// lsRobust is a robust loss with its soft margin fs: construct_loss_function.
// rho0, rho1 and rho2 hold ρ and its first two derivatives per residual.
type lsRobust struct {
	kind             Loss
	fs               float64
	rho0, rho1, rho2 []float64
}

// rho evaluates the raw loss and its derivatives at z = (f/fs)². SciPy
// skips the derivatives when only the cost is wanted; computing them anyway
// changes no value.
func (l *lsRobust) rho(f []float64) {
	for i, fi := range f {
		t := fi / l.fs
		z := float64(t * t)
		switch l.kind {
		case LossHuber:
			if z <= 1 {
				l.rho0[i], l.rho1[i], l.rho2[i] = z, 1, 0
			} else {
				l.rho0[i] = float64(2*math.Sqrt(z)) - 1
				l.rho1[i] = math.Pow(z, -0.5)
				l.rho2[i] = float64(-0.5 * math.Pow(z, -1.5))
			}
		case LossSoftL1:
			t := 1 + z
			l.rho0[i] = float64(2 * (math.Sqrt(t) - 1))
			l.rho1[i] = math.Pow(t, -0.5)
			l.rho2[i] = float64(-0.5 * math.Pow(t, -1.5))
		case LossCauchy:
			l.rho0[i] = math.Log1p(z)
			t := 1 + z
			l.rho1[i] = 1 / t
			l.rho2[i] = -1 / (t * t)
		default: // LossArctan
			l.rho0[i] = math.Atan(z)
			t := 1 + float64(z*z)
			l.rho1[i] = 1 / t
			l.rho2[i] = -2 * z / (t * t)
		}
	}
}

// cost is loss_function(f, cost_only=True): ½·fs²·Σρ(z).
func (l *lsRobust) cost(f []float64) float64 {
	l.rho(f)
	return float64(0.5 * (l.fs * l.fs) * trfSum(l.rho0))
}

// eval is loss_function(f): ρ scaled by fs² and the second derivative by 1/fs².
func (l *lsRobust) eval(f []float64) {
	l.rho(f)
	fs2 := float64(l.fs * l.fs)
	for i := range l.rho0 {
		l.rho0[i] *= fs2
		l.rho2[i] /= fs2
	}
}

// scale is scale_for_robust_loss_function: it rescales the m×n Jacobian j and
// the residuals f in place so that the Gauss–Newton model of the robust cost
// becomes an ordinary least-squares one.
func (l *lsRobust) scale(j, f []float64, n int) {
	for i, fi := range f {
		s := l.rho1[i] + float64(2*l.rho2[i]*fi*fi)
		if s < epsmch {
			s = epsmch
		}
		s = math.Sqrt(s)
		f[i] = float64(fi * (l.rho1[i] / s))
		row := j[i*n : (i+1)*n]
		for k := range row {
			row[k] *= s
		}
	}
}

func trfSum(v []float64) float64 {
	s := 0.0
	for _, x := range v {
		s += x
	}
	return s
}

// norm2 is numpy.linalg.norm of a vector, sqrt(x·x).
func trfNorm2(v []float64) float64 { return math.Sqrt(dot(v, v)) }

// lsTRF runs the trust-region reflective method; x0 is feasible and lb, ub
// have length n (±Inf for missing bounds). LeastSquares has validated the
// input and filled in the defaults.
func lsTRF(p LSProblem, x0, lb, ub []float64, o *LSOptions) (*LSResult, error) {
	n, m := len(x0), p.M
	bounded := false
	for i := range lb {
		if !math.IsInf(lb[i], -1) || !math.IsInf(ub[i], 1) {
			bounded = true
		}
	}
	if bounded {
		x0 = makeStrictlyFeasible(x0, lb, ub, 1e-10)
	}
	fd := fdStep{method: o.FDMethod, rel: o.DiffStep, lb: lb, ub: ub, buf: &fdBuf{}}
	jac := func(x, f, j []float64) error {
		if p.Jac != nil {
			p.Jac(x, j)
		} else {
			fd.jacFD(p.Func, x, f, j)
		}
		if !trfFinite(j) {
			return fmt.Errorf("%w: the Jacobian is not finite", ErrInput)
		}
		return nil
	}

	x := clone(x0)
	fTrue := make([]float64, m)
	p.Func(x, fTrue)
	if !trfFinite(fTrue) {
		return nil, fmt.Errorf("%w: residuals are not finite in the initial point", ErrInput)
	}
	J := make([]float64, m*n)
	if err := jac(x, fTrue, J); err != nil {
		return nil, err
	}
	nfev, njev := 1, 1
	f := clone(fTrue)

	var loss *lsRobust
	var cost float64
	if o.Loss != LossLinear {
		loss = &lsRobust{kind: o.Loss, fs: o.FScale,
			rho0: make([]float64, m), rho1: make([]float64, m), rho2: make([]float64, m)}
		loss.eval(f)
		cost = float64(0.5 * trfSum(loss.rho0))
		loss.scale(J, f, n)
	} else {
		cost = float64(0.5 * dot(f, f))
	}
	g := make([]float64, n)
	gradTF(J, f, g)

	var scale, scaleInv []float64
	if o.XScaleJac {
		scale, scaleInv = jacScale(J, m, n, nil)
	} else {
		scale, scaleInv = make([]float64, n), make([]float64, n)
		for i := range scale {
			scale[i] = 1
			if o.XScale != nil {
				scale[i] = o.XScale[i]
			}
			scaleInv[i] = 1 / scale[i]
		}
	}

	v := make([]float64, n)
	dv := make([]float64, n)
	tmp := make([]float64, n)
	if bounded {
		clScaling(x, g, lb, ub, v, dv)
		for i := range v {
			if dv[i] != 0 {
				v[i] *= scaleInv[i]
			}
			tmp[i] = x0[i] * scaleInv[i] / math.Sqrt(v[i])
		}
	} else {
		for i := range tmp {
			tmp[i] = float64(x0[i] * scaleInv[i])
		}
	}
	Delta := trfNorm2(tmp)
	if Delta == 0 {
		Delta = 1
	}

	maxNfev := o.MaxNfev
	if maxNfev == 0 {
		maxNfev = 100 * n
	}

	rows := m
	if bounded {
		rows = m + n
	}
	Jaug := make([]float64, rows*n) // J_h, then diag(diag_h^½) when bounded
	faug := make([]float64, rows)
	d := make([]float64, n)
	gh := make([]float64, n)
	diagH := make([]float64, n)
	xNew := make([]float64, n)
	fNew := make([]float64, m)
	jtmp := make([]float64, m)
	var hwork []float64 // householderR's workspace, for tall Jacobians
	if rows >= trfQRMin {
		hwork = make([]float64, rows*n+rows)
	}

	alpha := 0.0
	status := noStatus
	var gNorm, costNew float64
	for {
		if bounded {
			clScaling(x, g, lb, ub, v, dv)
			for i := range tmp {
				tmp[i] = float64(g[i] * v[i])
			}
			gNorm = norm(tmp, math.Inf(1))
		} else {
			gNorm = norm(g, math.Inf(1))
		}
		if gNorm < o.Gtol {
			status = 1
		}
		if status != noStatus || nfev == maxNfev {
			break
		}

		for i := range d {
			if bounded {
				if dv[i] != 0 {
					v[i] *= scaleInv[i]
				}
				d[i] = float64(math.Sqrt(v[i]) * scale[i])
				diagH[i] = float64(g[i] * dv[i] * scale[i])
			} else {
				d[i] = scale[i]
			}
			gh[i] = float64(d[i] * g[i])
		}
		for i := 0; i < m; i++ {
			for k := 0; k < n; k++ {
				Jaug[i*n+k] = float64(J[i*n+k] * d[k])
			}
		}
		copy(faug, f)
		if bounded {
			for i := 0; i < n; i++ {
				row := Jaug[(m+i)*n : (m+i+1)*n]
				for k := range row {
					row[k] = 0
				}
				row[i] = math.Sqrt(diagH[i])
			}
		}
		Jh := Jaug[:m*n]
		uf, s, V, err := trfSVDf(Jaug, faug, rows, n, hwork)
		if err != nil {
			return nil, err
		}

		theta := math.Max(0.995, 1-gNorm)
		actualReduction := -1.0
		for actualReduction <= 0 && nfev < maxNfev {
			var ph []float64
			ph, alpha = solveLsqTrustRegion(n, m, uf, s, V, Delta, alpha)
			var step, stepH []float64
			var predicted float64
			if bounded {
				pp := make([]float64, n)
				for i := range pp {
					pp[i] = float64(d[i] * ph[i])
				}
				step, stepH, predicted = selectStep(x, Jh, m, diagH, gh, pp, ph, d, Delta, lb, ub, theta)
				for i := range xNew {
					xNew[i] = x[i] + step[i]
				}
				copy(xNew, makeStrictlyFeasible(xNew, lb, ub, 0))
			} else {
				predicted = -evaluateQuadratic(Jh, m, gh, ph, nil, jtmp)
				step = make([]float64, n)
				for i := range step {
					step[i] = float64(d[i] * ph[i])
					xNew[i] = x[i] + step[i]
				}
				stepH = ph
			}
			p.Func(xNew, fNew)
			nfev++
			stepHNorm := trfNorm2(stepH)
			var ff float64
			finite := true
			if loss != nil {
				finite = trfFinite(fNew)
			} else {
				ff, finite = trfFiniteDot(fNew)
			}
			if !finite {
				Delta = float64(0.25 * stepHNorm)
				continue
			}
			if loss != nil {
				costNew = loss.cost(fNew)
			} else {
				costNew = float64(0.5 * ff)
			}
			actualReduction = cost - costNew
			DeltaNew, ratio := updateTrRadius(Delta, actualReduction, predicted,
				stepHNorm, stepHNorm > 0.95*Delta)
			status = checkTermination(actualReduction, cost, trfNorm2(step), trfNorm2(x), ratio, o.Ftol, o.Xtol)
			if status != noStatus {
				break
			}
			alpha *= Delta / DeltaNew
			Delta = DeltaNew
		}

		if actualReduction > 0 {
			x, xNew = xNew, x
			copy(fTrue, fNew)
			copy(f, fNew)
			cost = costNew
			if err := jac(x, fTrue, J); err != nil {
				return nil, err
			}
			njev++
			if loss != nil {
				loss.eval(f)
				loss.scale(J, f, n)
			}
			gradTF(J, f, g)
			if o.XScaleJac {
				scale, scaleInv = jacScale(J, m, n, scaleInv)
			}
		}
	}
	if status == noStatus {
		status = 0
	}
	active := make([]int, n)
	if bounded {
		active = findActiveConstraints(x, lb, ub, o.Xtol)
	}
	return &LSResult{
		X: x, Cost: cost, Fun: fTrue, Jac: J, Grad: g, Optimality: gNorm,
		ActiveMask: active, Nfev: nfev, Njev: njev,
		Status: status, Success: status > 0, Message: lsMessages[status],
	}, nil
}

func trfFinite(v []float64) bool {
	// A float is NaN or ±Inf exactly when its exponent bits are all set.
	const exp = 0x7ff0000000000000
	for _, x := range v {
		if math.Float64bits(x)&exp == exp {
			return false
		}
	}
	return true
}

// trfFiniteDot returns v·v (summed in index order, as dot) and true when
// every element of v is finite, and false (with no sum) otherwise, in one
// pass.
func trfFiniteDot(v []float64) (float64, bool) {
	const exp = 0x7ff0000000000000
	s := 0.0
	for _, x := range v {
		if math.Float64bits(x)&exp == exp {
			return 0, false
		}
		s += float64(x * x)
	}
	return s, true
}

// gradTF is compute_grad: g = Jᵀf.
func gradTF(J, f, g []float64) {
	n := len(g)
	for k := range g {
		g[k] = 0
	}
	for i, fi := range f {
		row := J[i*n : (i+1)*n]
		for k, jik := range row {
			g[k] += float64(jik * fi)
		}
	}
}

// jacScale is compute_jac_scale: the column norms of J (zeros replaced by
// one at the start, never decreasing afterwards) and their inverses.
func jacScale(J []float64, m, n int, old []float64) (scale, scaleInv []float64) {
	scaleInv = make([]float64, n)
	for k := range scaleInv {
		s := 0.0
		for i := 0; i < m; i++ {
			s += float64(J[i*n+k] * J[i*n+k])
		}
		scaleInv[k] = math.Sqrt(s)
		if old == nil {
			if scaleInv[k] == 0 {
				scaleInv[k] = 1
			}
		} else {
			scaleInv[k] = math.Max(scaleInv[k], old[k])
		}
	}
	scale = make([]float64, n)
	for k := range scale {
		scale[k] = 1 / scaleInv[k]
	}
	return scale, scaleInv
}

// clScaling is CL_scaling_vector, the Coleman–Li scaling: v holds the
// distance to the bound the anti-gradient points to (1 if none) and dv its
// derivative sign.
func clScaling(x, g, lb, ub, v, dv []float64) {
	for i := range x {
		v[i], dv[i] = 1, 0
		if g[i] < 0 && !math.IsInf(ub[i], 0) {
			v[i], dv[i] = ub[i]-x[i], -1
		}
		if g[i] > 0 && !math.IsInf(lb[i], 0) {
			v[i], dv[i] = x[i]-lb[i], 1
		}
	}
}

// checkTermination is check_termination.
func checkTermination(dF, F, dxNorm, xNorm, ratio, ftol, xtol float64) int {
	fOK := dF < ftol*F && ratio > 0.25
	xOK := dxNorm < xtol*(xtol+xNorm)
	switch {
	case fOK && xOK:
		return 4
	case fOK:
		return 2
	case xOK:
		return 3
	}
	return noStatus
}

// updateTrRadius is update_tr_radius.
func updateTrRadius(Delta, actual, predicted, stepNorm float64, boundHit bool) (float64, float64) {
	var ratio float64
	switch {
	case predicted > 0:
		ratio = actual / predicted
	case predicted == actual && actual == 0:
		ratio = 1
	}
	if ratio < 0.25 {
		Delta = float64(0.25 * stepNorm)
	} else if ratio > 0.75 && boundHit {
		Delta *= 2
	}
	return Delta, ratio
}

// solveLsqTrustRegion is solve_lsq_trust_region: it minimises
// ½‖J p + f‖² subject to ‖p‖ <= Delta given the thin SVD J = U·diag(s)·Vᵀ
// and uf = Uᵀf, by Moré's safeguarded Newton iteration on the
// Levenberg–Marquardt parameter alpha. V is n×k row-major. It returns the
// step and the final alpha.
func solveLsqTrustRegion(n, m int, uf, s, V []float64, Delta, initialAlpha float64) ([]float64, float64) {
	k := len(s)
	suf := make([]float64, k)
	for i := range suf {
		suf[i] = float64(s[i] * uf[i])
	}
	w := make([]float64, k)
	phiAndDerivative := func(alpha float64) (float64, float64) {
		for i := range w {
			w[i] = suf[i] / (float64(s[i]*s[i]) + alpha)
		}
		pNorm := trfNorm2(w)
		acc := 0.0
		for i := range w {
			den := float64(s[i]*s[i]) + alpha
			acc += suf[i] * suf[i] / (den * den * den)
		}
		return pNorm - Delta, -acc / pNorm
	}
	vmul := func(w []float64) []float64 {
		p := make([]float64, n)
		for r := 0; r < n; r++ {
			acc := 0.0
			for c := 0; c < k; c++ {
				acc += float64(V[r*k+c] * w[c])
			}
			p[r] = -acc
		}
		return p
	}
	fullRank := false
	if m >= n {
		fullRank = s[k-1] > epsmch*float64(m)*s[0]
	}
	if fullRank {
		for i := range w {
			w[i] = uf[i] / s[i]
		}
		p := vmul(w)
		if trfNorm2(p) <= Delta {
			return p, 0
		}
	}
	alphaUpper := trfNorm2(suf) / Delta
	alphaLower := 0.0
	if fullRank {
		phi, phiPrime := phiAndDerivative(0)
		alphaLower = -phi / phiPrime
	}
	alpha := initialAlpha
	if !fullRank && alpha == 0 {
		alpha = math.Max(0.001*alphaUpper, math.Sqrt(alphaLower*alphaUpper))
	}
	for it := 0; it < 10; it++ {
		if alpha < alphaLower || alpha > alphaUpper {
			alpha = math.Max(0.001*alphaUpper, math.Sqrt(alphaLower*alphaUpper))
		}
		phi, phiPrime := phiAndDerivative(alpha)
		if phi < 0 {
			alphaUpper = alpha
		}
		ratio := phi / phiPrime
		alphaLower = math.Max(alphaLower, alpha-ratio)
		alpha -= (phi + Delta) * ratio / Delta
		if math.Abs(phi) < 0.01*Delta {
			break
		}
	}
	for i := range w {
		w[i] = suf[i] / (float64(s[i]*s[i]) + alpha)
	}
	p := vmul(w)
	f := Delta / trfNorm2(p)
	for i := range p {
		p[i] *= f
	}
	return p, alpha
}

// evaluateQuadratic is evaluate_quadratic: ½‖J s‖² + ½ sᵀdiag(diag)s + gᵀs
// for an m×n J; js is scratch of length m.
func evaluateQuadratic(J []float64, m int, g, s, diag, js []float64) float64 {
	n := len(s)
	q := 0.0 // dot(js, js), accumulated as js is formed
	for i := 0; i < m; i++ {
		v := dot(J[i*n:(i+1)*n], s)
		js[i] = v
		q += float64(v * v)
	}
	if diag != nil {
		q += diagQuad(s, diag, s)
	}
	return float64(0.5*q) + dot(s, g)
}

// diagQuad is np.dot(a * diag, b).
func diagQuad(a, diag, b []float64) float64 {
	r := 0.0
	for i := range a {
		r += float64(a[i] * diag[i] * b[i])
	}
	return r
}

// buildQuadratic1D is build_quadratic_1d: the coefficients of
// f(t) = a t² + b t + c, the quadratic model along s0 + t·s.
func buildQuadratic1D(J []float64, m int, g, s, diag, s0 []float64) (a, b, c float64) {
	n := len(s)
	v := make([]float64, m)
	for i := 0; i < m; i++ {
		v[i] = dot(J[i*n:(i+1)*n], s)
	}
	a = dot(v, v)
	a += diagQuad(s, diag, s)
	a *= 0.5
	b = dot(g, s)
	if s0 != nil {
		u := make([]float64, m)
		for i := 0; i < m; i++ {
			u[i] = dot(J[i*n:(i+1)*n], s0)
		}
		b += dot(u, v)
		c = float64(0.5*dot(u, u)) + dot(g, s0)
		b += diagQuad(s0, diag, s)
		c += float64(0.5 * diagQuad(s0, diag, s0))
	}
	return a, b, c
}

// minimizeQuadratic1D is minimize_quadratic_1d: the minimiser of
// a t² + b t + c over [lb, ub] and its value.
func minimizeQuadratic1D(a, b, lb, ub, c float64) (float64, float64) {
	ts := []float64{lb, ub}
	if a != 0 {
		if e := -0.5 * b / a; lb < e && e < ub {
			ts = append(ts, e)
		}
	}
	bt, by := ts[0], float64(ts[0]*(float64(a*ts[0])+b))+c
	for _, t := range ts[1:] {
		if y := float64(t*(float64(a*t)+b)) + c; y < by {
			bt, by = t, y
		}
	}
	return bt, by
}

// intersectTrustRegion is intersect_trust_region: the values of t for which
// ‖x + t·s‖ = Delta, smaller first.
func intersectTrustRegion(x, s []float64, Delta float64) (float64, float64) {
	a := dot(s, s)
	b := dot(x, s)
	c := dot(x, x) - float64(Delta*Delta)
	d := math.Sqrt(float64(b*b) - float64(a*c))
	q := -(b + math.Copysign(d, b))
	t1, t2 := q/a, c/q
	if t1 < t2 {
		return t1, t2
	}
	return t2, t1
}

// inBounds is in_bounds.
func trfInBounds(x, lb, ub []float64) bool {
	for i := range x {
		if !(x[i] >= lb[i] && x[i] <= ub[i]) {
			return false
		}
	}
	return true
}

// stepSizeToBound is step_size_to_bound: the largest t with x + t·s
// feasible, and which variables reach a bound there (sign of s, 0 otherwise).
func stepSizeToBound(x, s, lb, ub []float64) (float64, []int) {
	steps := make([]float64, len(x))
	minStep := math.Inf(1)
	for i := range x {
		steps[i] = math.Inf(1)
		if s[i] != 0 {
			steps[i] = math.Max((lb[i]-x[i])/s[i], (ub[i]-x[i])/s[i])
		}
		minStep = math.Min(minStep, steps[i])
	}
	hits := make([]int, len(x))
	for i := range x {
		if steps[i] == minStep {
			switch {
			case s[i] > 0:
				hits[i] = 1
			case s[i] < 0:
				hits[i] = -1
			}
		}
	}
	return minStep, hits
}

// findActiveConstraints is find_active_constraints.
func findActiveConstraints(x, lb, ub []float64, rtol float64) []int {
	active := make([]int, len(x))
	for i := range x {
		if rtol == 0 {
			if x[i] <= lb[i] {
				active[i] = -1
			}
			if x[i] >= ub[i] {
				active[i] = 1
			}
			continue
		}
		lo, up := x[i]-lb[i], ub[i]-x[i]
		loT := float64(rtol * math.Max(1, math.Abs(lb[i])))
		upT := float64(rtol * math.Max(1, math.Abs(ub[i])))
		if !math.IsInf(lb[i], 0) && lo <= math.Min(up, loT) {
			active[i] = -1
		}
		if !math.IsInf(ub[i], 0) && up <= math.Min(lo, upT) {
			active[i] = 1
		}
	}
	return active
}

// makeStrictlyFeasible is make_strictly_feasible: it moves the variables
// within rstep (relative) of a bound inside, or to the next float when
// rstep is 0, and returns a new slice.
func makeStrictlyFeasible(x, lb, ub []float64, rstep float64) []float64 {
	out := clone(x)
	active := findActiveConstraints(x, lb, ub, rstep)
	for i, a := range active {
		switch {
		case a == -1 && rstep == 0:
			out[i] = math.Nextafter(lb[i], ub[i])
		case a == 1 && rstep == 0:
			out[i] = math.Nextafter(ub[i], lb[i])
		case a == -1:
			out[i] = lb[i] + float64(rstep*math.Max(1, math.Abs(lb[i])))
		case a == 1:
			out[i] = ub[i] - float64(rstep*math.Max(1, math.Abs(ub[i])))
		}
		if out[i] < lb[i] || out[i] > ub[i] {
			out[i] = float64(0.5 * (lb[i] + ub[i]))
		}
	}
	return out
}

// selectStep is trf.select_step: the best of the trust-region step clipped
// at the first bound it crosses, the step reflected off that bound, and the
// constrained Cauchy step. Jh is m×n; p and ph are the trust-region step in
// the original and scaled variables and are modified.
func selectStep(x, Jh []float64, m int, diagH, gh, p, ph, d []float64, Delta float64,
	lb, ub []float64, theta float64) (step, stepH []float64, predicted float64) {
	n := len(x)
	js := make([]float64, m)
	xp := make([]float64, n)
	for i := range xp {
		xp[i] = x[i] + p[i]
	}
	if trfInBounds(xp, lb, ub) {
		return p, ph, -evaluateQuadratic(Jh, m, gh, ph, diagH, js)
	}
	pStride, hits := stepSizeToBound(x, p, lb, ub)

	// The reflected direction.
	rh := clone(ph)
	r := make([]float64, n)
	xOnBound := make([]float64, n)
	for i := range rh {
		if hits[i] != 0 {
			rh[i] = -rh[i]
		}
		r[i] = float64(d[i] * rh[i])
		p[i] *= pStride
		ph[i] *= pStride
		xOnBound[i] = x[i] + p[i]
	}
	_, toTr := intersectTrustRegion(ph, rh, Delta)
	toBound, _ := stepSizeToBound(xOnBound, r, lb, ub)
	rStride := math.Min(toBound, toTr)
	var rStrideL, rStrideU float64
	if rStride > 0 {
		rStrideL = (1 - theta) * pStride / rStride
		if rStride == toBound {
			rStrideU = float64(theta * toBound)
		} else {
			rStrideU = toTr
		}
	} else {
		rStrideL, rStrideU = 0, -1
	}
	rValue := math.Inf(1)
	if rStrideL <= rStrideU {
		a, b, c := buildQuadratic1D(Jh, m, gh, rh, diagH, ph)
		rStride, rValue = minimizeQuadratic1D(a, b, rStrideL, rStrideU, c)
		for i := range rh {
			rh[i] = float64(rh[i]*rStride) + ph[i]
			r[i] = float64(rh[i] * d[i])
		}
	}

	// The trust-region step, stepped back from the bound.
	for i := range p {
		p[i] *= theta
		ph[i] *= theta
	}
	pValue := evaluateQuadratic(Jh, m, gh, ph, diagH, js)

	// The constrained Cauchy step along the scaled anti-gradient.
	agh := make([]float64, n)
	ag := make([]float64, n)
	for i := range agh {
		agh[i] = -gh[i]
		ag[i] = float64(d[i] * agh[i])
	}
	toTr = Delta / trfNorm2(agh)
	toBound, _ = stepSizeToBound(x, ag, lb, ub)
	agStride := toTr
	if toBound < toTr {
		agStride = float64(theta * toBound)
	}
	a, b, _ := buildQuadratic1D(Jh, m, gh, agh, diagH, nil)
	agStride, agValue := minimizeQuadratic1D(a, b, 0, agStride, 0)
	for i := range agh {
		agh[i] *= agStride
		ag[i] *= agStride
	}

	switch {
	case pValue < rValue && pValue < agValue:
		return p, ph, -pValue
	case rValue < pValue && rValue < agValue:
		return r, rh, -rValue
	}
	return ag, agh, -agValue
}

// trfSVD is scipy.linalg.svd(a, full_matrices=False) for the r×c row-major
// matrix a: U is r×k and V is c×k (both row-major, k = min(r, c)) and s is
// decreasing. It uses go-ndarray's Golub–Kahan bidiagonalisation with
// implicit-shift QR, LAPACK's gesvd algorithm. A one-sided Jacobi SVD is 2–4
// times faster on the tiny matrices typical here, but it returns exactly zero
// for an exactly rank-deficient Jacobian where LAPACK returns a singular value
// at rounding level, and the trust-region solver turns that rounding-level
// value into a full-length step along the null space; matching SciPy's
// iterates needs the LAPACK-like behaviour.
// trfSVDf returns uf = Uᵀf, s and V of the thin SVD of the r×c matrix a.
// For a tall matrix (r >= trfQRRatio·c, the regime where LAPACK's gesdd
// also factors QR first) it computes a = Q·R by Householder reflections,
// applies them to f, and takes the SVD of the c×c R only: Uᵀf = U_Rᵀ(Qᵀf).
// This is O(r·c²) with no r×c U, the dominant saving for curve fits of many
// points.
func trfSVDf(a, f []float64, r, c int, work []float64) (uf, s, V []float64, err error) {
	if r < trfQRMin || 6*r < 11*c {
		U, s, V, err := trfSVD(a, r, c)
		if err != nil {
			return nil, nil, nil, err
		}
		k := len(s)
		uf = make([]float64, k)
		for j := 0; j < k; j++ {
			acc := 0.0
			for i := 0; i < r; i++ {
				acc += float64(U[i*k+j] * f[i])
			}
			uf[j] = acc
		}
		return uf, s, V, nil
	}
	R, qf := householderR(a, f, r, c, work) // a non-finite a gives a non-finite R
	U, s, V, err := trfSVD(R, c, c)
	if err != nil {
		return nil, nil, nil, err
	}
	uf = make([]float64, c)
	for j := 0; j < c; j++ {
		acc := 0.0
		for i := 0; i < c; i++ {
			acc += float64(U[i*c+j] * qf[i])
		}
		uf[j] = acc
	}
	return uf, s, V, nil
}

// householderR factors the r×c row-major matrix a = Q·R by Householder
// reflections and returns the c×c upper-triangular R (row-major) and Qᵀf
// (nil f gives nil).
//
// Each sum is accumulated in the order of the plain algorithm (norm of the
// column, then vᵀv and vᵀc for the remaining columns, then the updates), so
// the result is that algorithm's to the last bit; but the sums that do not
// depend on one another advance in the same pass: vᵀv with every vᵀc, and
// the norm of the next column with the updates that produce it.
//
// work, if it holds at least r·c+r elements, is used for the column-major
// copy and for Qᵀf (which then aliases it); otherwise they are allocated.
func householderR(a, f []float64, r, c int, work []float64) (R, qf []float64) {
	if len(work) < r*c+r {
		work = make([]float64, r*c+r)
	}
	// Column-major copy for cache-friendly Householder updates; the norm of
	// the first column is accumulated on the way.
	q := work[:r*c]
	scale, ssq := 0.0, 1.0
	for i := 0; i < r; i++ {
		row := a[i*c : i*c+c]
		x := row[0]
		q[i] = x
		scale, ssq = enormStep(scale, ssq, x)
		for j := 1; j < c; j++ {
			q[j*r+i] = row[j]
		}
	}
	next, nextOK := enormEnd(scale, ssq), true
	if f != nil {
		qf = work[r*c : r*c+r]
		copy(qf, f)
	}
	R = make([]float64, c*c)
	var colStack [8][]float64
	var dStack [8]float64
	cols, d := colStack[:0], dStack[:]
	if c+1 > len(dStack) {
		d = make([]float64, c+1)
	}
	for j := 0; j < c; j++ {
		col := q[j*r : (j+1)*r]
		alpha := next
		if !nextOK {
			alpha = enorm(col[j:])
		}
		nextOK = false
		if col[j] > 0 {
			alpha = -alpha
		}
		if alpha != 0 {
			// v = x − α e₁ stored in col[j:], τ = 2/‖v‖².
			col[j] -= alpha
			v := col[j:]
			cols = cols[:0]
			for k := j + 1; k < c; k++ {
				cols = append(cols, q[k*r+j:(k+1)*r])
			}
			if qf != nil {
				cols = append(cols, qf[j:])
			}
			vv := householderDots(v, cols, d)
			for k := range cols {
				d[k] = 2 * d[k] / vv
			}
			nextOK = j+1 < c
			next = householderApply(v, cols, d, nextOK)
		}
		R[j*c+j] = alpha
		for k := j + 1; k < c; k++ {
			R[j*c+k] = q[k*r+j]
		}
	}
	return R, qf
}

// enormStep advances the package's enorm by one component: (scale, ssq)
// starts at (0, 1), and the norm is enormEnd(scale, ssq). It returns the new
// state instead of updating a struct, so that the state stays in registers.
func enormStep(scale, ssq, x float64) (float64, float64) {
	if x == 0 {
		return scale, ssq
	}
	ax := math.Abs(x)
	if scale < ax {
		return ax, 1 + float64(ssq*(scale/ax)*(scale/ax))
	}
	return scale, ssq + float64((ax/scale)*(ax/scale))
}

func enormEnd(scale, ssq float64) float64 {
	if math.IsInf(scale, 1) {
		return scale
	}
	return float64(scale * math.Sqrt(ssq))
}

// householderDots returns vᵀv and sets d[k] = vᵀcols[k], each summed in
// index order from 0.0, three columns per pass beside vᵀv.
func householderDots(v []float64, cols [][]float64, d []float64) (vv float64) {
	n := len(v)
	for g := 0; g == 0 || g < len(cols); g += 3 {
		var c [3][]float64
		cnt := max(0, min(3, len(cols)-g))
		for k := range c {
			c[k] = v // padding: its sums are dropped
			if k < cnt {
				c[k] = cols[g+k][:n]
			}
		}
		c0, c1, c2 := c[0], c[1], c[2]
		var s, s0, s1, s2 float64
		for i, vi := range v {
			s += float64(vi * vi)
			s0 += float64(vi * c0[i])
			s1 += float64(vi * c1[i])
			s2 += float64(vi * c2[i])
		}
		if g == 0 {
			vv = s
		}
		sums := [3]float64{s0, s1, s2}
		copy(d[g:g+cnt], sums[:cnt])
	}
	return vv
}

// householderApply sets cols[k] −= d[k]·v for every k. With norm set it
// also returns the package's enorm of the updated cols[0][1:], accumulated
// in the same pass.
func householderApply(v []float64, cols [][]float64, d []float64, norm bool) float64 {
	n := len(v)
	done := 0
	var nrm float64
	if norm {
		// cols[0] with its norm, and up to two more columns.
		c0, d0 := cols[0][:n], d[0]
		c0[0] -= float64(d0 * v[0])
		scale, ssq := 0.0, 1.0
		switch len(cols) {
		case 1:
			for i := 1; i < n; i++ {
				x := c0[i] - float64(d0*v[i])
				c0[i] = x
				scale, ssq = enormStep(scale, ssq, x)
			}
			done = 1
		case 2:
			c1, d1 := cols[1][:n], d[1]
			c1[0] -= float64(d1 * v[0])
			for i := 1; i < n; i++ {
				vi := v[i]
				c1[i] -= float64(d1 * vi)
				x := c0[i] - float64(d0*vi)
				c0[i] = x
				scale, ssq = enormStep(scale, ssq, x)
			}
			done = 2
		default:
			c1, d1 := cols[1][:n], d[1]
			c2, d2 := cols[2][:n], d[2]
			c1[0] -= float64(d1 * v[0])
			c2[0] -= float64(d2 * v[0])
			for i := 1; i < n; i++ {
				vi := v[i]
				c1[i] -= float64(d1 * vi)
				c2[i] -= float64(d2 * vi)
				x := c0[i] - float64(d0*vi)
				c0[i] = x
				scale, ssq = enormStep(scale, ssq, x)
			}
			done = 3
		}
		nrm = enormEnd(scale, ssq)
	}
	for k := done; k < len(cols); k++ {
		ck, dk := cols[k][:n], d[k]
		for i, vi := range v {
			ck[i] -= float64(dk * vi)
		}
	}
	return nrm
}

// trfQRMin is the row count from which trfSVDf factors QR first.
var trfQRMin = 1000

func trfSVD(a []float64, r, c int) (U, s, V []float64, err error) {
	// scipy.linalg.svd checks its input first (check_finite=True).
	var u, sv, vh *ndarray.Array
	err = errors.New("array must not contain infs or NaNs")
	if trfFinite(a) {
		u, sv, vh, err = linalg.SVD(matArray(a, r, c), false)
	}
	if err != nil {
		return nil, nil, nil, fmt.Errorf("%w: SVD of the scaled Jacobian: %v", ErrInput, err)
	}
	U, s = arrayData(u), arrayData(sv)
	vt := arrayData(vh)
	k := len(s)
	V = make([]float64, c*k)
	for i := 0; i < k; i++ {
		for j := 0; j < c; j++ {
			V[j*k+i] = vt[i*c+j]
		}
	}
	return U, s, V, nil
}
