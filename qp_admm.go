package optimize

import (
	"math"

	"github.com/go-simd/floats"
	"github.com/go-sparse/sparse"
)

// The ADMM method in the shape of OSQP v1.0.0 (docs/design-qp.md §4; the C
// source osqp/osqp tag v1.0.0, src/auxil.c, src/osqp_api.c, algebra/_common/
// lin_sys/qdldl/qdldl_interface.c, and Stellato et al., arXiv:1711.08013): the
// splitting of Algorithm 1 with a vector ρ, the direct KKT solve of qp_kkt.go,
// the termination order of check_termination and the certificates of
// is_primal_infeasible and is_dual_infeasible.

// OSQP v1.0.0 constants (include/public/osqp_api_constants.h).
const (
	admmInfty       = 1e30  // OSQP_INFTY
	admmMinScaling  = 1e-4  // OSQP_MIN_SCALING
	admmMaxScaling  = 1e4   // OSQP_MAX_SCALING
	admmDivisionTol = 1e-30 // OSQP_DIVISION_TOL = 1/OSQP_INFTY
	admmZeroDead    = 1e-15 // OSQP_ZERO_DEADZONE
	admmRhoMin      = 1e-6
	admmRhoMax      = 1e6
	admmRhoTol      = 1e-4 // OSQP_RHO_TOL, the width below which a row is an equality
	admmRhoEqRatio  = 1e3  // OSQP_RHO_EQ_OVER_RHO_INEQ
	admmRhoAdaptTol = 5.0  // OSQP_ADAPTIVE_RHO_TOLERANCE
)

// admmSettings are the resolved QPOptions: the zero value of each ADMM field is
// OSQP's default.
type admmSettings struct {
	maxIter, checkInterval, adaptInterval, scalingIter int
	epsAbs, epsRel, epsPrimInf, epsDualInf             float64
	rho, sigma, alpha                                  float64
	dualGap                                            bool
	polish                                             bool
	polishDelta                                        float64
	polishRefine                                       int
}

func (o *QPOptions) admmSettings() admmSettings {
	pick := func(v, def float64) float64 {
		if v == 0 {
			return def
		}
		return v
	}
	pickI := func(v, def int) int {
		if v == 0 {
			return def
		}
		return v
	}
	s := admmSettings{
		maxIter: pickI(o.MaxIter, 4000), checkInterval: pickI(o.CheckInterval, 25),
		adaptInterval: pickI(o.AdaptiveRhoInterval, 50), scalingIter: pickI(o.ScalingIter, 10),
		epsAbs: pick(o.EpsAbs, 1e-3), epsRel: pick(o.EpsRel, 1e-3),
		epsPrimInf: pick(o.EpsPrimInf, 1e-4), epsDualInf: pick(o.EpsDualInf, 1e-4),
		rho: pick(o.Rho, 0.1), sigma: pick(o.Sigma, 1e-6), alpha: pick(o.Alpha, 1.6),
		dualGap: !o.NoDualGapCheck, polish: o.Polish,
		polishDelta: pick(o.PolishDelta, 1e-6), polishRefine: pickI(o.PolishRefine, 3),
	}
	s.rho = math.Min(math.Max(s.rho, admmRhoMin), admmRhoMax)
	return s
}

// admm is the solver state for one problem. The data in P, q, A, l, u are the
// scaled ones (the original when no scaling is done); orig keeps the data as
// given so that scaling can be redone.
type admm struct {
	md   *qpModel
	n, m int
	set  admmSettings
	orig admmData

	pu   *sparse.CSC[float64]
	q    []float64
	a    admmCSR
	l, u []float64

	scaled           bool
	d, dinv, e, einv []float64
	c, cinv          float64

	rho            float64
	rhoVec, rhoInv []float64
	ctype          []int8 // -1 loose, 0 inequality, 1 equality (OSQP's constr_type)
	kkt            *admmKKT
	nonConvex      error

	x, z, y, xPrev, zPrev, xt, zt, dx, dy []float64
	rhs                                   []float64
	ax, px, aty, atdy, tmpN, tmpM         []float64

	// information of the last update (OSQP's OSQPInfo and the work values).
	status             int // QP status codes; admmUnsolved while running
	iter, rhoUpdates   int
	primRes, dualRes   float64
	scPrim, scDual     float64 // scaled residuals
	gap, objVal        float64 // unscaled
	scGap              float64
	xPx, qx, sc        float64
	inaccurate         bool
	warmX, warmY       []float64
	infCert            []float64
	polished           bool
	primRes0, dualRes0 float64
}

const admmUnsolved = -1

// admmData is the problem as ADMM takes it, before scaling.
type admmData struct {
	pu   *sparse.CSC[float64]
	q    []float64
	a    admmCSR
	l, u []float64
}

// admmFromModel converts the canonical rows of md: P as an upper-triangular CSC
// (nonzeros of a dense P), A by rows, ±Inf replaced by ±1e30 (OSQP_INFTY).
func (md *qpModel) admmFromModel() admmData {
	n := md.n
	var d admmData
	switch {
	case md.p != nil:
		// The arrays are built sorted and in range, so the CSC is valid by
		// construction (no NewCSC validation to fail).
		pu := &sparse.CSC[float64]{Rows: n, Cols: n, Indptr: make([]int, 1, n+1), Indices: []int32{}, Data: []float64{}}
		for j := 0; j < n; j++ {
			for i := 0; i <= j; i++ {
				if v := md.p[i*n+j]; v != 0 {
					pu.Indices = append(pu.Indices, int32(i))
					pu.Data = append(pu.Data, v)
				}
			}
			pu.Indptr = append(pu.Indptr, len(pu.Indices))
		}
		d.pu = pu
	case md.pUpper != nil:
		d.pu = md.pUpper.Clone()
	default:
		d.pu = &sparse.CSC[float64]{Rows: n, Cols: n, Indptr: make([]int, n+1), Indices: []int32{}, Data: []float64{}}
	}
	d.q = clone(md.q)
	m := len(md.rows)
	d.a.ptr = make([]int, 1, m+1)
	d.l, d.u = make([]float64, m), make([]float64, m)
	for i, r := range md.rows {
		switch r.kind {
		case rowDense:
			for j, v := range md.dense[r.ref*n : (r.ref+1)*n] {
				if v != 0 {
					d.a.idx = append(d.a.idx, int32(j))
					d.a.val = append(d.a.val, v)
				}
			}
		case rowSparse:
			lo, hi := md.sp.Indptr[r.ref], md.sp.Indptr[r.ref+1]
			d.a.idx = append(d.a.idx, md.sp.Indices[lo:hi]...)
			d.a.val = append(d.a.val, md.sp.Data[lo:hi]...)
		default:
			d.a.idx = append(d.a.idx, int32(r.ref))
			d.a.val = append(d.a.val, 1)
		}
		d.a.ptr = append(d.a.ptr, len(d.a.idx))
		d.l[i], d.u[i] = math.Max(r.l, -admmInfty), math.Min(r.u, admmInfty)
	}
	return d
}

// newADMM sets the solver up: scaling (design-qp.md §4.1.2), the ρ vector, the
// KKT matrix, its ordering, symbolic and numeric factorisation. A KKT matrix
// that is not quasi-definite is not an error here: it is remembered, and Solve
// reports QPNonConvex (OSQP raises an exception at setup instead).
func newADMM(md *qpModel, set admmSettings) (*admm, error) {
	a := &admm{md: md, n: md.n, m: len(md.rows), set: set, orig: md.admmFromModel(), status: admmUnsolved}
	a.alloc()
	a.loadData()
	if err := a.setRho(set.rho); err != nil {
		return nil, err
	}
	return a, nil
}

func (a *admm) alloc() {
	n, m := a.n, a.m
	mk := func(k int) []float64 { return make([]float64, k) }
	a.x, a.xPrev, a.xt, a.dx, a.px, a.tmpN = mk(n), mk(n), mk(n), mk(n), mk(n), mk(n)
	a.aty, a.atdy = mk(n), mk(n)
	a.z, a.zPrev, a.zt, a.dy, a.y, a.ax, a.tmpM = mk(m), mk(m), mk(m), mk(m), mk(m), mk(m), mk(m)
	a.rhs = mk(n + m)
	a.rhoVec, a.rhoInv = mk(m), mk(m)
	a.ctype = make([]int8, m)
	a.d, a.dinv, a.e, a.einv = mk(n), mk(n), mk(m), mk(m)
}

// loadData copies the original data into the working arrays and scales them.
func (a *admm) loadData() {
	o := a.orig
	a.pu = o.pu.Clone()
	a.q = clone(o.q)
	a.a = admmCSR{ptr: o.a.ptr, idx: o.a.idx, val: clone(o.a.val)}
	a.l, a.u = clone(o.l), clone(o.u)
	for i := range a.d {
		a.d[i], a.dinv[i] = 1, 1
	}
	for i := range a.e {
		a.e[i], a.einv[i] = 1, 1
	}
	a.c, a.cinv, a.scaled = 1, 1, false
	a.scaleData()
}

// setRho (re)classifies the rows, rebuilds the ρ vector for the scalar ρ and
// updates the KKT matrix: it assembles and factors it the first time, rewrites
// its diagonal afterwards. A KKT matrix that is not quasi-definite sets
// a.nonConvex.
func (a *admm) setRho(rho float64) error {
	a.rho = math.Min(math.Max(rho, admmRhoMin), admmRhoMax)
	for i := 0; i < a.m; i++ {
		switch {
		case a.l[i] < -admmInfty*admmMinScaling && a.u[i] > admmInfty*admmMinScaling:
			a.ctype[i] = -1
			a.rhoVec[i] = admmRhoMin
		case a.u[i]-a.l[i] < admmRhoTol:
			a.ctype[i] = 1
			a.rhoVec[i] = admmRhoEqRatio * a.rho
		default:
			a.ctype[i] = 0
			a.rhoVec[i] = a.rho
		}
		a.rhoInv[i] = 1 / a.rhoVec[i]
	}
	a.nonConvex = nil
	if a.kkt == nil {
		k := buildKKT(a.pu, &a.a, a.n, a.m, a.set.sigma, a.rhoInv)
		if err := k.analyse(); err != nil {
			return err
		}
		a.kkt = k
	} else {
		for i, p := range a.kkt.rhoDiag {
			a.kkt.mat.Data[p] = -a.rhoInv[i]
		}
	}
	if err := a.kkt.factor(); err != nil {
		a.nonConvex = err
	}
	return nil
}

// setWarm records a warm start (x, y in original units, either may be nil).
func (a *admm) setWarm(x, y []float64) {
	a.warmX, a.warmY = clone(x), clone(y)
}

// applyWarm loads the warm start into the iterates, scaled as OSQP's
// osqp_warm_start: x ← D⁻¹x, y ← c·E⁻¹y, z ← A·x.
func (a *admm) applyWarm() {
	if a.warmX != nil {
		floats.Mul(a.x, a.warmX, a.dinv)
		a.mulA(a.z, a.x)
	}
	if a.warmY != nil {
		floats.Mul(a.y, a.warmY, a.einv)
		for i := range a.y {
			a.y[i] *= a.c
		}
	}
	a.warmX, a.warmY = nil, nil
}

// coldStart zeroes the iterates.
func (a *admm) coldStart() {
	clear(a.x)
	clear(a.z)
	clear(a.y)
}

// mulA computes dst = A·x (rows of the scaled A).
func (a *admm) mulA(dst, x []float64) {
	for i := 0; i < a.m; i++ {
		s := 0.0
		for p := a.a.ptr[i]; p < a.a.ptr[i+1]; p++ {
			s += float64(a.a.val[p] * x[a.a.idx[p]])
		}
		dst[i] = s
	}
}

// mulAT computes dst = Aᵀ·y.
func (a *admm) mulAT(dst, y []float64) {
	clear(dst)
	for i := 0; i < a.m; i++ {
		if yi := y[i]; yi != 0 {
			for p := a.a.ptr[i]; p < a.a.ptr[i+1]; p++ {
				dst[a.a.idx[p]] += float64(a.a.val[p] * yi)
			}
		}
	}
}

// step is one ADMM iteration (Algorithm 1; OSQP's update_xz_tilde, update_x,
// update_z, update_y).
func (a *admm) step() {
	n, m, s := a.n, a.m, &a.set
	a.x, a.xPrev = a.xPrev, a.x
	a.z, a.zPrev = a.zPrev, a.z
	// right-hand side: x-part σx_prev − q; ν-part z_prev − ρ⁻¹y (kept in z̃).
	for i := 0; i < n; i++ {
		a.rhs[i] = float64(s.sigma*a.xPrev[i]) - a.q[i]
	}
	for i := 0; i < m; i++ {
		t := float64(a.rhoInv[i] * a.y[i])
		a.zt[i] = a.zPrev[i] - t
		a.rhs[n+i] = a.zt[i]
	}
	a.kkt.solve(a.rhs)
	copy(a.xt, a.rhs[:n])
	for i := 0; i < m; i++ {
		a.zt[i] += float64(a.rhoInv[i] * a.rhs[n+i])
	}
	// x = αx̃ + (1−α)x_prev
	for i := 0; i < n; i++ {
		a.x[i] = float64(s.alpha*a.xt[i]) + float64((1-s.alpha)*a.xPrev[i])
		a.dx[i] = a.x[i] - a.xPrev[i]
	}
	// z = Π(ρ⁻¹y + αz̃ + (1−α)z_prev)
	floats.Mul(a.z, a.rhoInv, a.y)
	floats.Axpby3(a.z, 1, a.z, s.alpha, a.zt, 1-s.alpha, a.zPrev)
	floats.Clamp(a.z, a.z, a.l, a.u)
	// δy = ρ(αz̃ + (1−α)z_prev − z); y += δy
	floats.Axpby3(a.dy, s.alpha, a.zt, 1-s.alpha, a.zPrev, -1, a.z)
	floats.Mul(a.dy, a.dy, a.rhoVec)
	for i := 0; i < m; i++ {
		a.y[i] += a.dy[i]
	}
}

// setQ replaces the linear term (original units), scaled as osqp_update_data_vec.
func (a *admm) setQ(q []float64) {
	copy(a.orig.q, q)
	for i, v := range q {
		a.q[i] = float64(v*a.d[i]) * a.c
	}
}

var admmMessages = map[int]string{
	QPOptimal:       "Optimal solution found (ADMM, OSQP v1.0.0 iteration).",
	QPInaccurate:    "Solved to ten times the requested tolerances only (inaccurate).",
	QPIterLimit:     "The iteration limit was reached.",
	QPInfeasible:    "The problem is primal infeasible; PrimalCert holds the certificate.",
	QPUnbounded:     "The problem is dual infeasible (unbounded); DualCert holds the certificate.",
	QPNonConvex:     "The problem is non convex: the residuals diverged.",
	QPNumericalFail: "Numerical difficulties: a residual is not a number.",
}

// result builds the QPResult from the final iterate: the unscaled solution
// x = Dx̂, y = c⁻¹Eŷ and the marginals −y, or NaN and a certificate when the
// status has no solution (OSQP's store_solution).
func (a *admm) result() *QPResult {
	md, n, m := a.md, a.n, a.m
	var res *QPResult
	switch a.status {
	case QPInfeasible, QPUnbounded, QPNonConvex, QPNumericalFail:
		x := make([]float64, n)
		for i := range x {
			x[i] = math.NaN()
		}
		res, _ = md.fill(x, make([]float64, m), QPADMM)
		res.ActiveSet = nil
		for i := range res.Y {
			res.Y[i] = math.NaN()
		}
		res.Fun, res.Optimality, res.ConstrViolation = math.NaN(), math.NaN(), math.NaN()
		switch a.status {
		case QPInfeasible:
			res.PrimalCert = unitInf(a.dy)
		case QPUnbounded:
			res.DualCert = unitInf(a.dx)
		}
		a.coldStart()
	default:
		x := make([]float64, n)
		floats.Mul(x, a.x, a.d)
		marg := make([]float64, m)
		floats.Mul(marg, a.y, a.e)
		for i := range marg {
			marg[i] *= -a.cinv
		}
		res, _ = md.fill(x, marg, QPADMM)
	}
	res.Status, res.Success = a.status, a.status == QPOptimal
	res.Message = admmMessages[a.status]
	if a.inaccurate && (a.status == QPInfeasible || a.status == QPUnbounded) {
		res.Message += " (inaccurate: tolerances ×10)"
	}
	res.Nit, res.RhoUpdates, res.DualityGap, res.Polished = a.iter, a.rhoUpdates, a.gap, a.polished
	return res
}

// unitInf returns v divided by its ∞-norm, as OSQP's store_solution does for
// the infeasibility certificates.
func unitInf(v []float64) []float64 {
	out := clone(v)
	if nrm := floats.MaxAbs(out); nrm > 0 {
		for i := range out {
			out[i] /= nrm
		}
	}
	return out
}

// solve is osqp_solve: the loop, then the result.
func (a *admm) solve() *QPResult {
	if a.nonConvex != nil {
		a.status, a.iter = QPNonConvex, 0
		res := a.result()
		res.Message = "The problem is non convex: " + a.nonConvex.Error() + "."
		return res
	}
	a.run()
	return a.result()
}

// run executes the loop of osqp_solve and returns the status; the iterates
// hold the solution or the certificates.
func (a *admm) run() {
	s := &a.set
	a.status, a.inaccurate, a.rhoUpdates, a.polished = admmUnsolved, false, 0, false
	a.applyWarm()
	canCheck := false
	iter := 1
	for ; iter <= s.maxIter; iter++ {
		a.step()
		canCheck = iter%s.checkInterval == 0
		canAdapt := s.adaptInterval > 0 && iter%s.adaptInterval == 0
		if canCheck || canAdapt || iter == 1 {
			a.updateInfo(iter)
		}
		if canCheck && a.checkTermination(false) {
			break
		}
		if canAdapt {
			a.adaptRho()
			if a.nonConvex != nil { // the refactorisation failed: nothing sound to iterate on
				a.status = QPNumericalFail
				return
			}
		}
	}
	if !canCheck {
		a.updateInfo(iter - 1)
		a.checkTermination(false)
	}
	if a.status == admmUnsolved && !a.checkTermination(true) {
		a.status = QPIterLimit
	}
	if s.polish && a.status == QPOptimal {
		a.polish()
	}
}
