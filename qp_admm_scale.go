package optimize

import (
	"math"

	"github.com/go-simd/floats"
)

// Ruiz equilibration and ρ adaptation (design-qp.md §4.1.2 and §4.4), a
// transcription of OSQP v1.0.0 src/scaling.c scale_data and src/auxil.c
// compute_rho_estimate and adapt_rho.

// limitScaling is OSQP's limit_scaling_scalar: values below 1e-4 become 1,
// values above 1e4 become 1e4.
func limitScaling(v float64) float64 {
	if v < admmMinScaling {
		v = 1
	}
	if v > admmMaxScaling {
		v = admmMaxScaling
	}
	return v
}

// cMax and cMin are C's (a > b ? a : b) and (a < b ? a : b), which OSQP's
// c_max and c_min are: with a NaN argument they return b.
func cMax(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func cMin(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

// colNormsP fills out with the ∞-norms of the columns of the symmetric matrix
// whose upper triangle is pu.
func (a *admm) colNormsP(out []float64) {
	clear(out)
	for j := 0; j < a.n; j++ {
		for p := a.pu.Indptr[j]; p < a.pu.Indptr[j+1]; p++ {
			v := math.Abs(a.pu.Data[p])
			i := int(a.pu.Indices[p])
			out[j] = math.Max(out[j], v)
			if i != j {
				out[i] = math.Max(out[i], v)
			}
		}
	}
}

// scaleData runs ScalingIter passes of the modified Ruiz algorithm on the
// working copies of P, q, A, l, u and sets D, E, c (all 1 when scaling is off).
func (a *admm) scaleData() {
	if a.set.scalingIter < 0 {
		return
	}
	n, m := a.n, a.m
	dTmp, dTmpA, eTmp := make([]float64, n), make([]float64, n), make([]float64, m)
	for it := 0; it < a.set.scalingIter; it++ {
		// ∞-norms of the columns of [P Aᵀ; A 0], without forming it.
		a.colNormsP(dTmp)
		clear(dTmpA)
		for i := 0; i < m; i++ {
			eTmp[i] = 0
			for p := a.a.ptr[i]; p < a.a.ptr[i+1]; p++ {
				v := math.Abs(a.a.val[p])
				eTmp[i] = math.Max(eTmp[i], v)
				dTmpA[a.a.idx[p]] = math.Max(dTmpA[a.a.idx[p]], v)
			}
		}
		for i := range dTmp {
			dTmp[i] = 1 / math.Sqrt(limitScaling(math.Max(dTmp[i], dTmpA[i])))
		}
		for i := range eTmp {
			eTmp[i] = 1 / math.Sqrt(limitScaling(eTmp[i]))
		}
		// P ← DPD, A ← EAD, q ← Dq, then D, E accumulate.
		for j := 0; j < n; j++ {
			for p := a.pu.Indptr[j]; p < a.pu.Indptr[j+1]; p++ {
				a.pu.Data[p] = float64(float64(dTmp[a.pu.Indices[p]]*a.pu.Data[p]) * dTmp[j])
			}
		}
		for i := 0; i < m; i++ {
			for p := a.a.ptr[i]; p < a.a.ptr[i+1]; p++ {
				a.a.val[p] = float64(float64(eTmp[i]*a.a.val[p]) * dTmp[a.a.idx[p]])
			}
		}
		floats.Mul(a.q, a.q, dTmp)
		floats.Mul(a.d, a.d, dTmp)
		floats.Mul(a.e, a.e, eTmp)
		// cost normalisation: the mean column norm of the scaled P against ‖q‖∞.
		a.colNormsP(dTmp)
		avg := 0.0
		for _, v := range dTmp {
			avg += v
		}
		avg /= float64(n)
		cs := 1 / limitScaling(math.Max(avg, limitScaling(floats.MaxAbs(a.q))))
		for p := range a.pu.Data {
			a.pu.Data[p] *= cs
		}
		for i := range a.q {
			a.q[i] *= cs
		}
		a.c *= cs
	}
	a.cinv = 1 / a.c
	for i := range a.d {
		a.dinv[i] = 1 / a.d[i]
	}
	for i := range a.e {
		a.einv[i] = 1 / a.e[i]
	}
	floats.Mul(a.l, a.l, a.e)
	floats.Mul(a.u, a.u, a.e)
	a.scaled = a.set.scalingIter > 0
}

// rhoEstimate is compute_rho_estimate on the scaled residuals of the last
// updateInfo.
func (a *admm) rhoEstimate() float64 {
	prim := a.scPrim / (math.Max(floats.MaxAbs(a.z), floats.MaxAbs(a.ax)) + admmDivisionTol)
	dn := cMax(floats.MaxAbs(a.q), floats.MaxAbs(a.aty))
	dn = cMax(dn, floats.MaxAbs(a.px))
	dual := a.scDual / (dn + admmDivisionTol)
	return cMin(cMax(a.rho*math.Sqrt(prim/dual), admmRhoMin), admmRhoMax)
}

// adaptRho is adapt_rho: it takes the estimate only when it differs from ρ by
// more than the factor 5, then refactors numerically (same pattern).
func (a *admm) adaptRho() {
	rho := a.rhoEstimate()
	if rho > a.rho*admmRhoAdaptTol || rho < a.rho/admmRhoAdaptTol {
		a.setRho(rho) // a failure (impossible: only the diagonal changed sign-preservingly) shows as a.nonConvex
		a.rhoUpdates++
	}
}
