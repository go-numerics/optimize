package optimize

import (
	"math"

	"github.com/go-simd/floats"
)

// Residuals, termination and certificates (design-qp.md §4.3), a transcription
// of OSQP v1.0.0 update_info, check_termination, compute_*_res/tol,
// is_primal_infeasible and is_dual_infeasible (src/auxil.c) with the default
// scaled_termination = 0: every test is made on the unscaled quantities.

// scaledNormInf is ‖s ⊙ v‖∞, with tmp as scratch of the length of v.
func scaledNormInf(s, v, tmp []float64) float64 {
	floats.Mul(tmp, s, v)
	return floats.MaxAbs(tmp)
}

// updateInfo computes the residuals, the objective and the duality gap at the
// current iterate (OSQP's update_info for the non-polishing case).
func (a *admm) updateInfo(iter int) {
	a.iter = iter
	// primal residual ‖E⁻¹(Ax − z)‖∞
	if a.m == 0 {
		a.primRes, a.scPrim = 0, 0
	} else {
		a.mulA(a.ax, a.x)
		for i := range a.tmpM {
			a.tmpM[i] = a.ax[i] - a.z[i]
		}
		a.scPrim = floats.MaxAbs(a.tmpM)
		a.primRes = a.scPrim
		if a.scaled {
			a.primRes = scaledNormInf(a.einv, a.tmpM, a.zPrev) // z_prev is free until the next swap, as in OSQP
		}
	}
	// dual residual c⁻¹‖D⁻¹(Px + q + Aᵀy)‖∞
	a.pu.MulVecSymUpper(a.px, a.x)
	for i := range a.tmpN {
		a.tmpN[i] = a.q[i] + a.px[i]
	}
	if a.m > 0 {
		a.mulAT(a.aty, a.y)
		for i := range a.tmpN {
			a.tmpN[i] += a.aty[i]
		}
	}
	a.scDual = floats.MaxAbs(a.tmpN)
	a.dualRes = a.scDual
	if a.scaled {
		a.dualRes = a.cinv * scaledNormInf(a.dinv, a.tmpN, a.xPrev)
	}
	// objective and duality gap: SC(y) on y projected on the polar of the
	// recession cone of [l, u], with the dead zone.
	a.xPx = floats.Dot(a.px, a.x)
	a.qx = floats.Dot(a.q, a.x)
	copy(a.tmpM, a.y)
	projectPolarRecCone(a.tmpM, a.l, a.u)
	for i, v := range a.tmpM {
		if math.Abs(v) < admmZeroDead {
			a.tmpM[i] = 0
		}
	}
	a.sc = 0
	for i, v := range a.tmpM {
		a.sc += float64(a.u[i] * math.Max(v, 0))
	}
	for i, v := range a.tmpM {
		a.sc += float64(a.l[i] * math.Min(v, 0))
	}
	a.scGap = a.xPx + a.qx + a.sc
	a.objVal = 0.5*a.xPx + a.qx
	a.gap = a.scGap
	if a.scaled {
		a.objVal *= a.cinv
		a.gap = a.cinv * a.scGap
	}
}

// projectPolarRecCone projects y onto the polar of the recession cone of the
// box [l, u]: 0 where both bounds are infinite, min(y, 0) where only the upper
// one is, max(y, 0) where only the lower one is.
func projectPolarRecCone(y, l, u []float64) {
	const inf = admmInfty * admmMinScaling
	for i := range y {
		switch {
		case u[i] > inf && l[i] < -inf:
			y[i] = 0
		case u[i] > inf:
			y[i] = math.Min(y[i], 0)
		case l[i] < -inf:
			y[i] = math.Max(y[i], 0)
		}
	}
}

// inRecCone reports whether y lies in the recession cone of [l, u] within tol.
func inRecCone(y, l, u []float64, tol float64) bool {
	const inf = admmInfty * admmMinScaling
	for i, v := range y {
		if (u[i] < inf && v > tol) || (l[i] > -inf && v < -tol) {
			return false
		}
	}
	return true
}

// primTol is ε_abs + ε_rel·max(‖E⁻¹z‖∞, ‖E⁻¹Ax‖∞) (compute_prim_tol).
func (a *admm) primTol(epsAbs, epsRel float64) float64 {
	var mx float64
	if a.scaled {
		mx = math.Max(scaledNormInf(a.einv, a.z, a.tmpM), scaledNormInf(a.einv, a.ax, a.tmpM))
	} else {
		mx = math.Max(floats.MaxAbs(a.z), floats.MaxAbs(a.ax))
	}
	return epsAbs + epsRel*mx
}

// dualTol is ε_abs + ε_rel·c⁻¹·max(‖D⁻¹q‖∞, ‖D⁻¹Aᵀy‖∞, ‖D⁻¹Px‖∞).
func (a *admm) dualTol(epsAbs, epsRel float64) float64 {
	var mx float64
	if a.scaled {
		mx = scaledNormInf(a.dinv, a.q, a.tmpN)
		mx = math.Max(mx, scaledNormInf(a.dinv, a.aty, a.tmpN))
		mx = math.Max(mx, scaledNormInf(a.dinv, a.px, a.tmpN))
		mx *= a.cinv
	} else {
		mx = math.Max(floats.MaxAbs(a.q), math.Max(floats.MaxAbs(a.aty), floats.MaxAbs(a.px)))
	}
	return epsAbs + epsRel*mx
}

// gapTol is ε_abs + ε_rel·c⁻¹·max(|xᵀPx|, |qᵀx|, |SC(y)|).
func (a *admm) gapTol(epsAbs, epsRel float64) float64 {
	mx := math.Max(math.Abs(a.xPx), math.Max(math.Abs(a.qx), math.Abs(a.sc)))
	if a.scaled {
		mx *= a.cinv
	}
	return epsAbs + epsRel*mx
}

// primalInfeasible is is_primal_infeasible on the last δy (projected in place).
func (a *admm) primalInfeasible(epsPrimInf float64) bool {
	projectPolarRecCone(a.dy, a.l, a.u)
	var norm float64
	if a.scaled {
		norm = scaledNormInf(a.e, a.dy, a.tmpM)
	} else {
		norm = floats.MaxAbs(a.dy)
	}
	if norm > admmDivisionTol {
		lhs := 0.0
		for i, v := range a.dy {
			lhs += float64(a.u[i] * math.Max(v, 0))
		}
		for i, v := range a.dy {
			lhs += float64(a.l[i] * math.Min(v, 0))
		}
		if lhs < 0 {
			a.mulAT(a.atdy, a.dy)
			if a.scaled {
				floats.Mul(a.atdy, a.atdy, a.dinv)
			}
			return floats.MaxAbs(a.atdy) < epsPrimInf*norm
		}
	}
	return false
}

// dualInfeasible is is_dual_infeasible on the last δx.
func (a *admm) dualInfeasible(epsDualInf float64) bool {
	norm, cost := floats.MaxAbs(a.dx), 1.0
	if a.scaled {
		norm, cost = scaledNormInf(a.d, a.dx, a.tmpN), a.c
	}
	if norm > admmDivisionTol && floats.Dot(a.q, a.dx) < 0 {
		a.pu.MulVecSymUpper(a.tmpN, a.dx)
		if a.scaled {
			floats.Mul(a.tmpN, a.tmpN, a.dinv)
		}
		if floats.MaxAbs(a.tmpN) < cost*epsDualInf*norm {
			a.mulA(a.tmpM, a.dx)
			if a.scaled {
				floats.Mul(a.tmpM, a.tmpM, a.einv)
			}
			return inRecCone(a.tmpM, a.l, a.u, epsDualInf*norm)
		}
	}
	return false
}

// checkTermination is OSQP's check_termination: it sets a.status and returns
// true when the iteration should stop. With approximate every tolerance is ten
// times larger and the status is the "inaccurate" variant (reported through
// a.inaccurate).
func (a *admm) checkTermination(approximate bool) bool {
	s := &a.set
	if math.IsNaN(a.primRes) || math.IsNaN(a.dualRes) {
		a.status = QPNumericalFail
		return true
	}
	if a.primRes > admmInfty || a.dualRes > admmInfty {
		a.status = QPNonConvex
		return true
	}
	epsAbs, epsRel, epsPrimInf, epsDualInf := s.epsAbs, s.epsRel, s.epsPrimInf, s.epsDualInf
	if approximate {
		epsAbs, epsRel, epsPrimInf, epsDualInf = 10*epsAbs, 10*epsRel, 10*epsPrimInf, 10*epsDualInf
	}
	var primOK, dualOK, gapOK, primInf, dualInf bool
	if a.m == 0 {
		primOK = true
	} else if a.primRes < a.primTol(epsAbs, epsRel) {
		primOK = true
	} else {
		primInf = a.primalInfeasible(epsPrimInf)
	}
	if a.dualRes < a.dualTol(epsAbs, epsRel) {
		dualOK = true
	} else {
		dualInf = a.dualInfeasible(epsDualInf)
	}
	gapOK = !s.dualGap || math.Abs(a.gap) < a.gapTol(epsAbs, epsRel)
	switch {
	case primOK && dualOK && gapOK:
		a.status = QPOptimal
		if approximate {
			a.status, a.inaccurate = QPInaccurate, true
		}
	case primInf:
		a.status, a.inaccurate = QPInfeasible, approximate
		if a.scaled {
			floats.Mul(a.dy, a.dy, a.e)
		}
	case dualInf:
		a.status, a.inaccurate = QPUnbounded, approximate
		if a.scaled {
			floats.Mul(a.dx, a.dx, a.d)
		}
	default:
		return false
	}
	return true
}
