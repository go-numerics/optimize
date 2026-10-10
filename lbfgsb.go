package optimize

import (
	"fmt"
	"math"
)

// This file is a port of L-BFGS-B 3.0 (Byrd, Lu, Nocedal and Zhu 1995; Zhu,
// Byrd, Lu and Nocedal 1997; Morales and Nocedal 2011), the limited-memory
// quasi-Newton method for bound-constrained minimisation, following the
// structure of the reference Fortran subroutines (mainlb, active, projgr,
// cauchy, hpsolb, bmv, formk, formt, freev, cmprlb, matupd, subsm, lnsrlb,
// errclb, dpofa, dtrsl) with 0-based indices. Matrices are stored column-major
// with the reference leading dimensions: ws and wy are n×m, sy, ss and wt are
// m×m, wn and snd are 2m×2m.

// Messages of scipy.optimize.minimize(method='L-BFGS-B').
const (
	lbMsgConvGrad = "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL"
	lbMsgConvF    = "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH"
	lbMsgMaxFun   = "STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT"
	lbMsgMaxIter  = "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT"
	lbMsgCallback = "STOP: CALLBACK REQUESTED HALT"
	lbMsgAbnormal = "ABNORMAL: "
)

// Bound types of the reference code (nbd).
const (
	nbdFree  = 0 // unbounded
	nbdLower = 1 // lower bound only
	nbdBoth  = 2 // lower and upper bounds
	nbdUpper = 3 // upper bound only
)

// lbfgsb holds the state of mainlb: the problem bounds, the limited-memory
// matrices and the bookkeeping of the active set.
type lbfgsb struct {
	n, m   int
	l, u   []float64
	nbd    []int
	cnstnd bool // some variable is bounded
	boxed  bool // every variable has two bounds

	ws, wy     []float64 // n×m: the S and Y corrections, a ring from head
	sy, ss, wt []float64 // m×m: S'Y, S'S and the Cholesky factor of θS'S+LD⁻¹L'
	wn, snd    []float64 // 2m×2m: the middle matrix of formk and its parts
	z, r, d    []float64 // Cauchy/subspace point, reduced gradient, direction
	t, xp      []float64 // line-search start point, subsm backup of z
	wa         []float64 // 8m work array (p, c, wbp, v of cauchy)

	index  []int // free variables first (nfree), then the active ones
	iwhere []int // per-variable state (-1 free, 0 free and bounded, 1/2 at l/u, 3 fixed)
	indx2  []int // entering variables first (nenter), leaving ones from ileave

	theta                 float64
	col, head, itail      int
	iupdat                int
	updatd                bool
	nfree, nenter, ileave int
	iter                  int
	sbgnrm                float64 // max-norm of the projected gradient

	// Line-search quantities of lnsrlb.
	stp, gd, gdold, dnorm, dtd, stpmx float64
	ls                                dcsrch
}

// lbfgsbErrclb checks the arguments the way the reference errclb does and
// returns an error for each condition it reports.
func lbfgsbErrclb(n, m int, factr float64, l, u []float64, nbd []int) error {
	switch {
	case n <= 0:
		return fmt.Errorf("%w: L-BFGS-B: N <= 0", ErrInput)
	case m <= 0:
		return fmt.Errorf("%w: L-BFGS-B: M <= 0", ErrInput)
	case factr < 0:
		return fmt.Errorf("%w: L-BFGS-B: FTOL < 0", ErrInput)
	}
	for i := range nbd {
		if nbd[i] == nbdBoth && l[i] > u[i] {
			return fmt.Errorf("%w: L-BFGS-B: NO FEASIBLE SOLUTION", ErrBounds)
		}
	}
	return nil
}

// minimizeLBFGSB is scipy.optimize.minimize(method='L-BFGS-B'): the driver
// loop of _minimize_lbfgsb around mainlb. lb and ub are nil when the problem
// is unbounded; x0 is already inside the bounds.
func minimizeLBFGSB(sf *scalarFunc, x0 []float64, o *Options, lb, ub []float64) (*Result, error) {
	n := len(x0)
	m := o.M
	if m == 0 {
		m = 10
	}
	ftol, gtol := o.Ftol, o.Gtol
	if ftol == 0 {
		ftol = 2.2204460492503131e-09
		if o.Tol != 0 {
			ftol = o.Tol
		}
	}
	if gtol == 0 {
		gtol = 1e-5
		if o.Tol != 0 {
			gtol = o.Tol
		}
	}
	maxiter, maxfun, maxls := o.MaxIter, o.MaxFev, o.MaxLS
	if maxiter == 0 {
		maxiter = 15000
	}
	if maxfun == 0 {
		maxfun = 15000
	}
	if maxls == 0 {
		maxls = 20
	}
	if maxls < 0 {
		return nil, fmt.Errorf("%w: L-BFGS-B: maxls must be positive", ErrInput)
	}
	if maxiter < 0 || maxfun < 0 {
		return nil, fmt.Errorf("%w: L-BFGS-B: maxiter and maxfun must be positive", ErrInput)
	}
	factr := ftol / epsmch
	pgtol := gtol

	l := make([]float64, n)
	u := make([]float64, n)
	nbd := make([]int, n)
	if lb != nil {
		for i := range nbd {
			lo, hi := !math.IsInf(lb[i], 0), !math.IsInf(ub[i], 0)
			if lo {
				l[i] = lb[i]
			}
			if hi {
				u[i] = ub[i]
			}
			switch {
			case lo && hi:
				nbd[i] = nbdBoth
			case lo:
				nbd[i] = nbdLower
			case hi:
				nbd[i] = nbdUpper
			}
		}
	}
	if err := lbfgsbErrclb(n, m, factr, l, u, nbd); err != nil {
		return nil, err
	}

	s := newLbfgsb(n, m, l, u, nbd)
	x := clone(x0)
	g := make([]float64, n)
	s.active(x)
	tol := float64(factr * epsmch)

	nit := 0
	var msg string
	f := sf.funGrad(x, g)
	s.sbgnrm = s.projgr(x, g)
	if s.sbgnrm <= pgtol {
		msg = lbMsgConvGrad
	} else {
		var fold float64
		for {
			var ok bool
			f, fold, ok = s.iterate(sf, x, g, f, maxls)
			if !ok {
				msg = lbMsgAbnormal
				break
			}
			// NEW_X: one more iteration for the driver.
			nit++
			if o.Callback != nil && o.Callback(clone(x), f) {
				msg = lbMsgCallback
			}
			if nit >= maxiter {
				msg = lbMsgMaxIter
			} else if sf.nfev > maxfun {
				msg = lbMsgMaxFun
			}
			if msg != "" {
				break
			}
			s.sbgnrm = s.projgr(x, g)
			if s.sbgnrm <= pgtol {
				msg = lbMsgConvGrad
				break
			}
			ddum := cfmax(cfmax(math.Abs(fold), math.Abs(f)), 1)
			if fold-f <= tol*ddum {
				msg = lbMsgConvF
				break
			}
			s.update(g)
		}
	}

	status := 0
	switch {
	case msg == lbMsgConvGrad || msg == lbMsgConvF:
	case sf.nfev > maxfun || nit >= maxiter:
		status = 1
	default:
		status = 2
	}
	return sf.result(&Result{
		X:       x,
		Fun:     f,
		Jac:     g,
		Nit:     nit,
		Status:  status,
		Success: status == 0,
		Message: msg,
	}), nil
}

func newLbfgsb(n, m int, l, u []float64, nbd []int) *lbfgsb {
	return &lbfgsb{
		n: n, m: m, l: l, u: u, nbd: nbd,
		ws: make([]float64, n*m), wy: make([]float64, n*m),
		sy: make([]float64, m*m), ss: make([]float64, m*m), wt: make([]float64, m*m),
		wn: make([]float64, 4*m*m), snd: make([]float64, 4*m*m),
		z: make([]float64, n), r: make([]float64, n), d: make([]float64, n),
		t: make([]float64, n), xp: make([]float64, n),
		wa:    make([]float64, 8*m),
		index: make([]int, n), iwhere: make([]int, n), indx2: make([]int, n),
		theta: 1,
		nfree: n,
	}
}

// reset discards the limited-memory matrix (the reference code's response
// to a failed factorisation or line search).
func (s *lbfgsb) reset() {
	s.col, s.head, s.theta, s.iupdat, s.updatd = 0, 0, 1, 0, false
}

// iterate runs one iteration of mainlb from the point x with f and g at x:
// generalized Cauchy point, subspace minimisation and line search. It
// returns the new f, the previous one, and false when the line search failed
// with an empty limited-memory matrix (ABNORMAL), in which case x, g and the
// returned f are those at the start of the iteration.
func (s *lbfgsb) iterate(sf *scalarFunc, x, g []float64, f float64, maxls int) (fnew, fold float64, ok bool) {
	n := s.n
	for {
		if s.cnstnd || s.col == 0 {
			s.cauchy(x, g)
			s.freev()
		} else {
			// Unconstrained with a non-empty memory: the Cauchy step is
			// skipped and z = x; no variable enters or leaves (freev keeps
			// nenter = 0 and ileave = n when nothing is bounded).
			copy(s.z, x)
		}
		if s.nfree != 0 && s.col != 0 {
			wrk := s.ileave < n || s.nenter > 0 || s.updatd
			if wrk && !s.formk() {
				s.reset()
				continue
			}
			s.cmprlb(x, g)
			s.subsm(x, g)
		}
		for i := range s.d {
			s.d[i] = s.z[i] - x[i]
		}
		fnew, ok := s.lnsrlb(sf, x, g, f, maxls)
		if ok {
			s.iter++
			return fnew, f, true
		}
		copy(x, s.t)
		copy(g, s.r)
		if s.col == 0 {
			s.iter++
			return f, f, false
		}
		s.reset()
	}
}

// update forms the new correction pair from the step and the gradient change
// and refreshes the limited-memory matrices (mainlb after NEW_X). It is
// called with g at the new point; s.r holds the old gradient, s.d the
// direction and s.stp, s.gd, s.gdold, s.dtd the line-search quantities.
func (s *lbfgsb) update(g []float64) {
	for i := range s.r {
		s.r[i] = g[i] - s.r[i]
	}
	rr := s.dnrm2(s.r)
	rr *= rr
	var dr, ddum float64
	if s.stp == 1 {
		dr = s.gd - s.gdold
		ddum = -s.gdold
	} else {
		dr = float64((s.gd - s.gdold) * s.stp)
		for i := range s.d {
			s.d[i] *= s.stp
		}
		ddum = float64(-s.gdold * s.stp)
	}
	if dr <= epsmch*ddum {
		// Skip the update: the curvature condition fails.
		s.updatd = false
		return
	}
	s.updatd = true
	s.iupdat++
	s.matupd(rr, dr)
	if !s.formt() {
		s.reset()
	}
}
