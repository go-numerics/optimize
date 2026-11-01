package optimize

import (
	"github.com/go-simd/floats"
)

// Solution polishing (design-qp.md §4.5), a transcription of OSQP v1.0.0
// src/polish.c: guess the active rows from the ADMM iterate, solve the reduced
// KKT system [P+δI Aredᵀ; Ared −δI] with iterative refinement against the
// unregularised matrix, and keep the result when it reduces the residuals.
// (OSQP's qdldl interface regularises the polishing system with σ rather than δ;
// the defaults are both 1e-6, and δ = PolishDelta is used here, as the design says.)

// mul computes dst = A·x for the rows of c.
func (c *admmCSR) mul(dst, x []float64) {
	for i := range dst {
		s := 0.0
		for p := c.ptr[i]; p < c.ptr[i+1]; p++ {
			s += float64(c.val[p] * x[c.idx[p]])
		}
		dst[i] = s
	}
}

// mulTAdd adds s·Aᵀ·y to dst.
func (c *admmCSR) mulTAdd(dst, y []float64, s float64) {
	for i, yi := range y {
		if yi != 0 {
			for p := c.ptr[i]; p < c.ptr[i+1]; p++ {
				dst[c.idx[p]] += float64(s * float64(c.val[p]*yi))
			}
		}
	}
}

// polish tries to improve the current (optimal) iterate; a.polished says
// whether the polished point replaced it.
func (a *admm) polish() {
	n, m := a.n, a.m
	// active rows: lower (or equality) = −1, upper = +1
	flags := make([]int8, m)
	var red admmCSR
	red.ptr = []int{0}
	var rhs []float64 // [−q; l_j or u_j]
	rhs = append(rhs, a.q...)
	for i := range rhs {
		rhs[i] = -rhs[i]
	}
	for j := 0; j < m; j++ {
		switch {
		case a.z[j]-a.l[j] < -a.y[j] || a.l[j] == a.u[j]:
			flags[j] = -1
			rhs = append(rhs, a.l[j])
		case a.u[j]-a.z[j] < a.y[j]:
			flags[j] = 1
			rhs = append(rhs, a.u[j])
		default:
			continue
		}
		red.idx = append(red.idx, a.a.idx[a.a.ptr[j]:a.a.ptr[j+1]]...)
		red.val = append(red.val, a.a.val[a.a.ptr[j]:a.a.ptr[j+1]]...)
		red.ptr = append(red.ptr, len(red.idx))
	}
	mred := len(red.ptr) - 1
	if mred == 0 { // no active set: polishing is skipped, not failed
		return
	}
	delta := a.set.polishDelta
	dvec := make([]float64, mred)
	for i := range dvec {
		dvec[i] = delta
	}
	k := buildKKT(a.pu, &red, n, mred, delta, dvec)
	if k.analyse() != nil || k.factor() != nil {
		return
	}
	sol := clone(rhs)
	k.solve(sol)
	// iterative refinement: solve (K+dK)·dz = b − K·z with the regularised factor
	r := make([]float64, n+mred)
	tmpN := make([]float64, n)
	tmpM := make([]float64, mred)
	for it := 0; it < a.set.polishRefine; it++ {
		copy(r, rhs)
		a.pu.MulVecSymUpper(tmpN, sol[:n])
		red.mulTAdd(tmpN, sol[n:], 1)
		red.mul(tmpM, sol[:n])
		for i := 0; i < n; i++ {
			r[i] -= tmpN[i]
		}
		for i := 0; i < mred; i++ {
			r[n+i] -= tmpM[i]
		}
		k.solve(r)
		for i := range sol {
			sol[i] += r[i]
		}
	}
	// the polished (x, z, y): z = Ax, y scattered, then y ← y + z, z ← Π(y), y ← y − z
	px, pz, py := clone(sol[:n]), make([]float64, m), make([]float64, m)
	a.mulA(pz, px)
	for j, c := 0, 0; j < m; j++ {
		if flags[j] != 0 {
			py[j] = sol[n+c]
			c++
		}
	}
	for j := range py {
		py[j] += pz[j]
	}
	floats.Clamp(pz, py, a.l, a.u)
	for j := range py {
		py[j] -= pz[j]
	}
	// residuals at the polished point, accepted by OSQP's rule
	old := a.snapshotInfo()
	oldX, oldZ, oldY := a.x, a.z, a.y
	a.x, a.z, a.y = px, pz, py
	a.updateInfo(a.iter)
	pr, dr := a.primRes, a.dualRes
	ok := (pr < old.primRes && dr < old.dualRes) ||
		(pr < old.primRes && old.dualRes < 1e-10) ||
		(dr < old.dualRes && old.primRes < 1e-10)
	if ok {
		a.polished = true
		return
	}
	a.x, a.z, a.y = oldX, oldZ, oldY
	a.restoreInfo(old)
}

// admmInfo is the part of the solver's information a failed polish restores.
type admmInfo struct {
	primRes, dualRes, scPrim, scDual, gap, scGap, objVal, xPx, qx, sc float64
}

func (a *admm) snapshotInfo() admmInfo {
	return admmInfo{a.primRes, a.dualRes, a.scPrim, a.scDual, a.gap, a.scGap, a.objVal, a.xPx, a.qx, a.sc}
}

func (a *admm) restoreInfo(i admmInfo) {
	a.primRes, a.dualRes, a.scPrim, a.scDual, a.gap, a.scGap, a.objVal, a.xPx, a.qx, a.sc =
		i.primRes, i.dualRes, i.scPrim, i.scDual, i.gap, i.scGap, i.objVal, i.xPx, i.qx, i.sc
}
