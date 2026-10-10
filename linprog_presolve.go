package optimize

import "math"

// presolved is a reduced linear program and the bookkeeping that maps its
// solution back to the original problem.
type presolved struct {
	red    *lp
	varIdx []int     // original index of each reduced variable
	xfix   []float64 // full-length vector holding the removed variables
	ubRows []int     // original index of each reduced inequality row
	eqRows []int     // original index of each reduced equality row
	// done reports that presolve settled the problem; status, message and
	// x (nil unless status is 0) are then final.
	done    bool
	status  int
	message string
	x       []float64
}

// presolveIdentity wraps p without reducing it (presolve disabled).
func presolveIdentity(p *lp) *presolved {
	return &presolved{red: p, varIdx: lpIota(p.n), xfix: make([]float64, p.n),
		ubRows: lpIota(p.mub), eqRows: lpIota(p.meq)}
}

func lpIota(n int) []int {
	s := make([]int, n)
	for i := range s {
		s[i] = i
	}
	return s
}

// restore expands a reduced solution to the original variables.
func (pr *presolved) restore(xr []float64) []float64 {
	x := clone(pr.xfix)
	for k, j := range pr.varIdx {
		x[j] = xr[k]
	}
	return x
}

func lpZeroRow(a []float64, i, n int) bool {
	for _, v := range a[i*n : (i+1)*n] {
		if v != 0 {
			return false
		}
	}
	return true
}

// presolveLP is a light version of SciPy's _presolve (after Andersen &
// Andersen, "Presolving in linear programming", 1995): it removes empty
// rows, settles empty columns at the bound the objective prefers, removes
// fixed variables, detects the trivial infeasibilities and unboundedness
// these steps reveal, and drops linearly dependent equality rows (or
// reports them inconsistent). It does not, unlike SciPy, turn singleton
// rows into bounds.
func presolveLP(p *lp, tol float64) *presolved {
	n := p.n
	pr := &presolved{}
	stop := func(status int, msg string) *presolved {
		pr.done, pr.status, pr.message = true, status, msg
		return pr
	}
	var eqRows, ubRows []int
	for i := 0; i < p.meq; i++ {
		if !lpZeroRow(p.aeq, i, n) {
			eqRows = append(eqRows, i)
		} else if math.Abs(p.beq[i]) > tol {
			return stop(LPInfeasible, "The problem is (trivially) infeasible due to a row "+
				"of zeros in the equality constraint matrix with a nonzero "+
				"corresponding constraint value.")
		}
	}
	for i := 0; i < p.mub; i++ {
		if !lpZeroRow(p.aub, i, n) {
			ubRows = append(ubRows, i)
		} else if p.bub[i] < -tol {
			return stop(LPInfeasible, "The problem is (trivially) infeasible due to a row "+
				"of zeros in the inequality constraint matrix with a negative "+
				"corresponding constraint value.")
		}
	}
	lb, ub := clone(p.lb), clone(p.ub)
	for j := 0; j < n; j++ {
		if !lpZeroColumn(p, j) {
			continue
		}
		// An unconstrained variable sits at the bound its cost prefers.
		var v float64
		switch {
		case p.c[j] > 0:
			v = lb[j]
		case p.c[j] < 0:
			v = ub[j]
		case !math.IsInf(lb[j], 0):
			v = lb[j]
		case !math.IsInf(ub[j], 0):
			v = ub[j]
		}
		if math.IsInf(v, 0) {
			return stop(LPUnbounded, "If feasible, the problem is (trivially) unbounded "+
				"due to a zero column in the constraint matrices. If you wish to "+
				"check whether the problem is infeasible, turn presolve off.")
		}
		lb[j], ub[j] = v, v
	}
	pr.xfix = make([]float64, n)
	for j := 0; j < n; j++ {
		if math.Abs(ub[j]-lb[j]) < tol {
			pr.xfix[j] = lb[j]
		} else {
			pr.varIdx = append(pr.varIdx, j)
		}
	}
	red := &lp{n: len(pr.varIdx), lb: lpPick(lb, pr.varIdx), ub: lpPick(ub, pr.varIdx), c: lpPick(p.c, pr.varIdx)}
	reduceRows := func(a, b []float64, rows []int) ([]float64, []float64) {
		var ra, rb []float64
		for _, i := range rows {
			row := a[i*n : (i+1)*n]
			ra = append(ra, lpPick(row, pr.varIdx)...)
			rb = append(rb, b[i]-dot(row, pr.xfix))
		}
		return ra, rb
	}
	red.aub, red.bub = reduceRows(p.aub, p.bub, ubRows)
	red.aeq, red.beq = reduceRows(p.aeq, p.beq, eqRows)
	if red.n == 0 {
		// Every variable is fixed: the constraints must hold as they are.
		for _, s := range red.bub {
			if s < -tol*(1+math.Abs(s)) {
				return stop(LPInfeasible, infeasibleFixedMsg)
			}
		}
		for _, s := range red.beq {
			if math.Abs(s) > tol*(1+math.Abs(s)) {
				return stop(LPInfeasible, infeasibleFixedMsg)
			}
		}
		pr.x = clone(pr.xfix)
		return stop(LPOptimal, "The solution was determined in presolve as there "+
			"are no non-trivial constraints.")
	}
	keep, ok := lpIndependentRows(red.aeq, red.beq, len(eqRows), red.n, tol)
	if !ok {
		return stop(LPInfeasible, "There is a linear combination of rows of A_eq that "+
			"results in zero, suggesting a redundant constraint. However the same "+
			"linear combination of b_eq is nonzero, suggesting that the constraints "+
			"conflict and the problem is infeasible.")
	}
	var aeq, beq []float64
	for _, k := range keep {
		aeq = append(aeq, red.aeq[k*red.n:(k+1)*red.n]...)
		beq = append(beq, red.beq[k])
		pr.eqRows = append(pr.eqRows, eqRows[k])
	}
	red.aeq, red.beq = aeq, beq
	red.mub, red.meq = len(ubRows), len(keep)
	pr.ubRows = ubRows
	pr.red = red
	return pr
}

const infeasibleFixedMsg = "The problem is (trivially) infeasible because the " +
	"bounds fix all variables to values inconsistent with the constraints."

func lpZeroColumn(p *lp, j int) bool {
	for i := 0; i < p.mub; i++ {
		if p.aub[i*p.n+j] != 0 {
			return false
		}
	}
	for i := 0; i < p.meq; i++ {
		if p.aeq[i*p.n+j] != 0 {
			return false
		}
	}
	return true
}

func lpPick(v []float64, idx []int) []float64 {
	out := make([]float64, len(idx))
	for k, j := range idx {
		out[k] = v[j]
	}
	return out
}

// lpIndependentRows returns the indices of a maximal set of linearly
// independent rows of the m×n matrix a, scanning in order with modified
// Gram–Schmidt (two passes). A dependent row whose right-hand side does not
// follow the same combination of b to within tol·(1+‖b‖∞) makes the system
// inconsistent (ok false).
func lpIndependentRows(a, b []float64, m, n int, tol float64) (keep []int, ok bool) {
	bnorm := norm(b, math.Inf(1))
	var qs [][]float64
	var betas []float64
	for i := 0; i < m; i++ {
		v := clone(a[i*n : (i+1)*n])
		bb := b[i]
		anorm := enorm(v)
		for pass := 0; pass < 2; pass++ {
			for k, q := range qs {
				r := dot(q, v)
				for j := range v {
					v[j] -= r * q[j]
				}
				bb -= r * betas[k]
			}
		}
		vn := enorm(v)
		if vn <= 1e-9*anorm {
			if math.Abs(bb) > tol*(1+bnorm) {
				return nil, false
			}
			continue
		}
		for j := range v {
			v[j] /= vn
		}
		qs = append(qs, v)
		betas = append(betas, bb/vn)
		keep = append(keep, i)
	}
	return keep, true
}
