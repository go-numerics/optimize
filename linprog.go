package optimize

import (
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
)

// LPMethod names a linear-programming algorithm. The values are the strings
// SciPy's linprog accepts for the matching legacy methods.
type LPMethod string

// Linear-programming methods implemented by Linprog.
const (
	// LPSimplex is a dense bounded-variable revised simplex method with an
	// LU-factorised basis and product-form updates. It is the default: it
	// ends on a vertex, its duals are exact for the final basis, and it
	// certifies infeasibility and unboundedness directly.
	LPSimplex LPMethod = "revised simplex"
	// LPInteriorPoint is the homogeneous self-dual primal–dual
	// interior-point method of Andersen & Andersen with Mehrotra's
	// predictor–corrector, solving dense normal equations by Cholesky. Its
	// solution is interior to the optimal face, not a vertex.
	LPInteriorPoint LPMethod = "interior-point"
)

// LP status codes, as scipy.optimize.linprog reports them in status.
const (
	LPOptimal       = 0 // optimisation terminated successfully
	LPIterLimit     = 1 // iteration limit reached
	LPInfeasible    = 2 // problem appears to be infeasible
	LPUnbounded     = 3 // problem appears to be unbounded
	LPNumericalFail = 4 // numerical difficulties encountered
)

// LPConstraints holds the constraints of a linear program
//
//	A_ub·x <= b_ub,  A_eq·x == b_eq,  Bounds[i].Lo <= x[i] <= Bounds[i].Hi,
//
// in the shape of linprog's A_ub, b_ub, A_eq, b_eq and bounds arguments.
// Any of the matrices may be nil (no such constraints); a non-nil matrix
// must be 2-D with len(c) columns and its right-hand side 1-D with one entry
// per row.
type LPConstraints struct {
	// AUb and BUb are the inequality constraints A_ub·x <= b_ub.
	AUb, BUb *ndarray.Array
	// AEq and BEq are the equality constraints A_eq·x == b_eq.
	AEq, BEq *ndarray.Array
	// Bounds gives one interval per variable. nil means linprog's default
	// (0, +Inf) for every variable — note that this differs from Minimize,
	// where nil means unbounded. Use Unbounded() for a free variable.
	Bounds []Bound
}

// LPOptions configures Linprog. Zero values select the defaults listed with
// each field.
type LPOptions struct {
	// Method selects the algorithm; the default is LPSimplex.
	Method LPMethod
	// Tol is the primal and dual feasibility tolerance. Default 1e-9 for
	// LPSimplex and 1e-8 for LPInteriorPoint (SciPy's interior-point
	// default). As in SciPy, the final feasibility check uses 10·√Tol.
	Tol float64
	// MaxIter bounds the iterations (simplex pivots and bound flips over
	// both phases, or interior-point iterations). Default 5000 for
	// LPSimplex and 1000 for LPInteriorPoint, as in SciPy.
	MaxIter int
	// NoPresolve disables the presolve pass (linprog's presolve=False).
	NoPresolve bool
}

// LPDual pairs the residuals of one family of constraints with their
// marginals (dual values), like the ineqlin, eqlin, lower and upper fields
// of SciPy's HiGHS result.
type LPDual struct {
	// Residual is the slack of each constraint: b_ub − A_ub·x for
	// inequalities, b_eq − A_eq·x for equalities, x − lb and ub − x for
	// the bounds (±Inf for a missing bound).
	Residual []float64
	// Marginals is the sensitivity of the optimal objective to the
	// constraint's right-hand side (∂fun/∂b), with SciPy's HiGHS sign
	// convention: ineqlin and upper marginals are <= 0, lower marginals
	// are >= 0, eqlin marginals have any sign. They satisfy
	// c = A_ubᵀ·ineqlin + A_eqᵀ·eqlin + lower + upper. nil unless Status
	// is 0.
	Marginals []float64
}

// LPResult is the outcome of Linprog, the counterpart of the OptimizeResult
// that scipy.optimize.linprog returns.
type LPResult struct {
	// X is the solution. When Status is not 0 it is the last point the
	// solver held (it may violate the constraints), or nil when presolve
	// stopped before any solve.
	X []float64
	// Fun is c·X.
	Fun float64
	// Slack is b_ub − A_ub·X and Con is b_eq − A_eq·X.
	Slack, Con []float64
	// IneqLin, EqLin, Lower and Upper hold residuals and marginals of the
	// inequality rows, equality rows, lower bounds and upper bounds.
	IneqLin, EqLin, Lower, Upper LPDual
	// Status is SciPy's code: 0 optimal, 1 iteration limit, 2 infeasible,
	// 3 unbounded, 4 numerical difficulties.
	Status int
	// Success reports Status == 0.
	Success bool
	// Message describes the reason for termination.
	Message string
	// Nit counts iterations over all phases.
	Nit int
}

// XArray returns X as a 1-D ndarray (nil when X is nil).
func (r *LPResult) XArray() *ndarray.Array { return vecArray(r.X) }

// lpProblem is a dense linear program min c·x s.t. aub·x <= bub, aeq·x == beq,
// lb <= x <= ub, with row-major matrices.
type lpProblem struct {
	n, mub, meq int
	c           []float64
	aub, bub    []float64
	aeq, beq    []float64
	lb, ub      []float64
}

// lpSolution is what a solver returns for the reduced problem.
type lpSolution struct {
	x        []float64
	yub, yeq []float64 // row duals (∂fun/∂b), set when status is 0
	status   int
	message  string
	nit      int
}

// Linprog minimises c·x subject to cons; it is the counterpart of
// scipy.optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=...).
// SciPy's default HiGHS solvers are not available; opts.Method selects a
// dense revised simplex (default) or a dense interior-point method, after
// the presolve SciPy applies to its own legacy methods (see Linprog's
// package documentation in the result's Message for presolve outcomes).
//
// Invalid input (shapes, NaN or Inf coefficients, NaN bounds, unknown
// method) is an error wrapping ErrDimension, ErrInput, ErrBounds or
// ErrMethod. Infeasible, unbounded or unsolved problems are not errors:
// they are reported through Status, Success and Message.
func Linprog(c []float64, cons LPConstraints, opts *LPOptions) (*LPResult, error) {
	var o LPOptions
	if opts != nil {
		o = *opts
	}
	switch o.Method {
	case "":
		o.Method = LPSimplex
	case LPSimplex, LPInteriorPoint:
	default:
		return nil, fmt.Errorf("%w: %q", ErrMethod, o.Method)
	}
	if o.Tol <= 0 {
		o.Tol = 1e-9
		if o.Method == LPInteriorPoint {
			o.Tol = 1e-8
		}
	}
	if o.MaxIter <= 0 {
		o.MaxIter = 5000
		if o.Method == LPInteriorPoint {
			o.MaxIter = 1000
		}
	}
	p, err := parseLP(c, cons)
	if err != nil {
		return nil, err
	}
	res := &LPResult{}
	for i := range p.lb {
		if p.lb[i] > p.ub[i] || math.IsInf(p.lb[i], 1) || math.IsInf(p.ub[i], -1) {
			res.Status = LPInfeasible
			res.Message = "The problem is (trivially) infeasible since one or more " +
				"upper bounds are smaller than the corresponding lower bounds, " +
				"a lower bound is +Inf or an upper bound is -Inf."
			return finishLP(p, res, nil, nil, o.Tol), nil
		}
	}
	pre := presolveIdentity(p)
	if !o.NoPresolve {
		pre = presolveLP(p, o.Tol)
		if pre.done {
			// Every row left is empty, so the row duals are zero.
			res.Status, res.Message = pre.status, pre.message
			zero := &lpSolution{yub: make([]float64, p.mub), yeq: make([]float64, p.meq)}
			return finishLP(p, res, pre.x, zero, o.Tol), nil
		}
	}
	var sol lpSolution
	if o.Method == LPSimplex {
		sol = simplexLP(pre.red, o.Tol, o.MaxIter)
	} else {
		sol = interiorPointLP(pre.red, o.Tol, o.MaxIter)
	}
	res.Status, res.Message, res.Nit = sol.status, sol.message, sol.nit
	x := pre.restore(sol.x)
	var duals *lpSolution
	if sol.status == LPOptimal {
		duals = &lpSolution{yub: make([]float64, p.mub), yeq: make([]float64, p.meq)}
		for k, i := range pre.ubRows {
			duals.yub[i] = sol.yub[k]
		}
		for k, i := range pre.eqRows {
			duals.yeq[i] = sol.yeq[k]
		}
	}
	return finishLP(p, res, x, duals, o.Tol), nil
}

// parseLP validates the inputs and returns the dense problem.
func parseLP(c []float64, cons LPConstraints) (*lpProblem, error) {
	n := len(c)
	if n == 0 {
		return nil, fmt.Errorf("%w: c must not be empty", ErrInput)
	}
	if !lpAllFinite(c) {
		return nil, fmt.Errorf("%w: c must not contain Inf or NaN", ErrInput)
	}
	p := &lpProblem{n: n, c: clone(c)}
	var err error
	if p.aub, p.bub, p.mub, err = parseRows("A_ub", "b_ub", cons.AUb, cons.BUb, n); err != nil {
		return nil, err
	}
	if p.aeq, p.beq, p.meq, err = parseRows("A_eq", "b_eq", cons.AEq, cons.BEq, n); err != nil {
		return nil, err
	}
	p.lb = make([]float64, n)
	p.ub = make([]float64, n)
	if cons.Bounds == nil {
		for i := range p.ub {
			p.ub[i] = math.Inf(1)
		}
		return p, nil
	}
	if len(cons.Bounds) != n {
		return nil, fmt.Errorf("%w: %d bounds for %d variables", ErrDimension, len(cons.Bounds), n)
	}
	for i, b := range cons.Bounds {
		if math.IsNaN(b.Lo) || math.IsNaN(b.Hi) {
			return nil, fmt.Errorf("%w: bound %d is NaN", ErrBounds, i)
		}
		p.lb[i], p.ub[i] = b.Lo, b.Hi
	}
	return p, nil
}

// parseRows checks one (A, b) pair: A 2-D with n columns, b 1-D with one
// entry per row, all finite. Both nil means no rows.
func parseRows(an, bn string, a, b *ndarray.Array, n int) ([]float64, []float64, int, error) {
	if a == nil && b == nil {
		return nil, nil, 0, nil
	}
	if a == nil || b == nil {
		return nil, nil, 0, fmt.Errorf("%w: %s and %s must be given together", ErrDimension, an, bn)
	}
	as, bs := a.Shape(), b.Shape()
	if len(as) != 2 || as[1] != n {
		return nil, nil, 0, fmt.Errorf("%w: %s must be 2-D with %d columns, got shape %v", ErrDimension, an, n, as)
	}
	if len(bs) != 1 || bs[0] != as[0] {
		return nil, nil, 0, fmt.Errorf("%w: %s must be 1-D with %d entries, got shape %v", ErrDimension, bn, as[0], bs)
	}
	ad, bd := arrayData(a), arrayData(b)
	if !lpAllFinite(ad) || !lpAllFinite(bd) {
		return nil, nil, 0, fmt.Errorf("%w: %s and %s must not contain Inf or NaN", ErrInput, an, bn)
	}
	return ad, bd, as[0], nil
}

func lpAllFinite(v []float64) bool {
	for _, x := range v {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return false
		}
	}
	return true
}

// finishLP fills X, Fun, Slack, Con and the duals of res from the full
// solution x of p, then applies SciPy's _check_result feasibility audit.
func finishLP(p *lpProblem, res *LPResult, x []float64, d *lpSolution, tol float64) *LPResult {
	res.X = x
	if x != nil {
		res.Fun = dot(p.c, x)
		res.Slack = lpRowResidual(p.aub, p.bub, p.mub, p.n, x)
		res.Con = lpRowResidual(p.aeq, p.beq, p.meq, p.n, x)
		res.IneqLin.Residual, res.EqLin.Residual = res.Slack, res.Con
		res.Lower.Residual = make([]float64, p.n)
		res.Upper.Residual = make([]float64, p.n)
		for j := range x {
			res.Lower.Residual[j] = x[j] - p.lb[j]
			res.Upper.Residual[j] = p.ub[j] - x[j]
		}
		res.Status, res.Message = checkLP(p, res, tol)
	}
	if d != nil && res.Status == LPOptimal {
		// Reduced costs c − A_ubᵀ·y_ub − A_eqᵀ·y_eq split by sign into the
		// lower-bound (>= 0) and upper-bound (<= 0) marginals.
		red := clone(p.c)
		for i := 0; i < p.mub; i++ {
			for j := 0; j < p.n; j++ {
				red[j] -= p.aub[i*p.n+j] * d.yub[i]
			}
		}
		for i := 0; i < p.meq; i++ {
			for j := 0; j < p.n; j++ {
				red[j] -= p.aeq[i*p.n+j] * d.yeq[i]
			}
		}
		res.Lower.Marginals = make([]float64, p.n)
		res.Upper.Marginals = make([]float64, p.n)
		for j, r := range red {
			if r > 0 {
				res.Lower.Marginals[j] = r
			} else {
				res.Upper.Marginals[j] = r
			}
		}
		res.IneqLin.Marginals, res.EqLin.Marginals = d.yub, d.yeq
	}
	res.Success = res.Status == LPOptimal
	return res
}

// lpRowResidual returns b − A·x for an m×n row-major A.
func lpRowResidual(a, b []float64, m, n int, x []float64) []float64 {
	r := make([]float64, m)
	for i := range r {
		r[i] = b[i] - dot(a[i*n:(i+1)*n], x)
	}
	return r
}

// checkLP is SciPy's _check_result: a point claimed optimal must satisfy the
// bounds and constraints to 10·√tol, and a point claimed infeasible must not.
func checkLP(p *lpProblem, res *LPResult, tol float64) (int, string) {
	tol = math.Sqrt(tol) * 10
	feasible := !math.IsNaN(res.Fun)
	for j, x := range res.X {
		if !(x >= p.lb[j]-tol && x <= p.ub[j]+tol) {
			feasible = false
		}
	}
	if res.Status != LPUnbounded {
		for _, s := range res.Slack {
			if !(s >= -tol) {
				feasible = false
			}
		}
		for _, s := range res.Con {
			if !(math.Abs(s) <= tol) {
				feasible = false
			}
		}
	}
	switch {
	case res.Status == LPOptimal && !feasible:
		return LPNumericalFail, fmt.Sprintf("The solution does not satisfy the "+
			"constraints within the required tolerance of %.2E, yet no errors "+
			"were raised and there is no certificate of infeasibility or "+
			"unboundedness. Check whether the slack and constraint residuals "+
			"are acceptable; if not, consider enabling presolve, adjusting the "+
			"tolerance, and/or using a different method.", tol)
	case res.Status == LPInfeasible && feasible:
		return LPNumericalFail, "The solution is feasible, but the solver did " +
			"not report that the solution was optimal. Please try a different method."
	}
	return res.Status, res.Message
}
