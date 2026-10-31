package optimize

import (
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-sparse/sparse"
)

// Quadratic programming: the API of docs/design-qp.md (§2). This file holds
// the types, the validation (§2.1), the canonical rows (§2.2), the method
// choice (§2.3) and the result assembly. The Goldfarb–Idnani and ADMM solvers
// of §3 and §4 are not written yet: until they land, Quadprog solves the
// problems whose canonical rows are all equalities (including none) and whose
// P is positive definite, directly, and returns ErrMethod for every other
// problem.

// QPMethod names a quadratic-programming algorithm.
type QPMethod string

const (
	// QPAuto picks the method: QPActiveSet when P is given dense and positive
	// definite, no ASparse is given and n + m <= 2000, QPADMM otherwise.
	QPAuto QPMethod = ""
	// QPActiveSet is the Goldfarb–Idnani dual active-set method: dense,
	// P positive definite.
	QPActiveSet QPMethod = "active-set"
	// QPADMM is OSQP's ADMM method: dense or sparse, P positive semidefinite.
	QPADMM QPMethod = "admm"
)

// qpAutoLimit is the n + m up to which QPAuto chooses the active-set method
// (provisional, design-qp.md §2.3).
const qpAutoLimit = 2000

// qpDirectLimit bounds the number of entries of each dense matrix the direct
// equality-constrained solve allocates (n·n for the Cholesky factor of P and
// m·n for the transformed constraints): 2^26 entries, 512 MB. qpMaxSize is the
// limit on the sizes derived from the input (nnz of the KKT matrix, rows, n):
// go-sparse stores row indices as int32, so nothing larger can be factored,
// and no int is overflowed on a 32-bit target. Both are variables so that the
// tests can lower them instead of allocating gigabytes.
var (
	qpDirectLimit int64 = 1 << 26
	qpMaxSize     int64 = math.MaxInt32
)

// QPProblem is min ½xᵀPx + qᵀx subject to the constraints. Give P either
// dense (P) or sparse (PSparse, upper triangle only, as OSQP takes it), never
// both; nil for both means P = 0 (an LP, ADMM only).
type QPProblem struct {
	// P is n×n and is symmetrised as (P+Pᵀ)/2, as MATLAB's quadprog does.
	P *ndarray.Array
	// PSparse is n×n and holds the entries with row <= column only.
	PSparse *sparse.CSC[float64]
	// Q has length n and defines n.
	Q    []float64
	Cons QPConstraints
}

// QPConstraints mirrors quadprog's (A, b, Aeq, beq, lb, ub) and adds the
// two-sided rows of OSQP. Every part is optional.
type QPConstraints struct {
	// AUb·x <= BUb. AUb is m×n (or one row of length n); BUb has one entry
	// per row, +Inf for a vacuous row.
	AUb, BUb *ndarray.Array
	// AEq·x = BEq, with finite right-hand sides.
	AEq, BEq *ndarray.Array
	// Linear holds blocks Lb <= A·x <= Ub, the type of Minimize's constraints.
	Linear []LinearConstraint
	// ASparse holds rows l <= ASparse·x <= u (QPADMM only); L and U have one
	// entry per row, or one entry broadcast, or nil for -Inf and +Inf.
	ASparse *sparse.CSC[float64]
	L, U    []float64
	// Bounds are the variable bounds; nil means free, as in Minimize (not
	// Linprog's (0, +Inf)).
	Bounds []Bound
}

// QPOptions tunes Quadprog. The zero value selects every default.
type QPOptions struct {
	Method QPMethod
	// MaxIter is the iteration limit: 10·(n + m) for the active set, 4000 for
	// ADMM when zero.
	MaxIter int
	// FeasTol is the active set's primal feasibility tolerance (default 1e-9).
	FeasTol float64
	// ADMM settings: the zero value of each selects OSQP v1.0.0's default
	// (EpsAbs and EpsRel 1e-3, EpsPrimInf and EpsDualInf 1e-4, Rho 0.1,
	// Sigma 1e-6, Alpha 1.6, ScalingIter 10, AdaptiveRhoInterval 50,
	// CheckInterval 25, PolishDelta 1e-6, PolishRefine 3). A negative
	// ScalingIter or AdaptiveRhoInterval disables that step.
	EpsAbs, EpsRel         float64
	EpsPrimInf, EpsDualInf float64
	Rho, Sigma, Alpha      float64
	ScalingIter            int
	AdaptiveRhoInterval    int
	CheckInterval          int
	Polish                 bool
	PolishDelta            float64
	PolishRefine           int
	NoDualGapCheck         bool
	// X0 and Y0 (lengths n and the number of canonical rows) and WorkingSet
	// (canonical row indices) are validated; the solvers that use them are
	// not written yet.
	X0, Y0     []float64
	WorkingSet []int
}

// QP status codes. 0–4 have Linprog's meaning; 5 and 6 are QP-specific.
const (
	QPOptimal       = 0 // KKT conditions hold within tolerance
	QPIterLimit     = 1
	QPInfeasible    = 2 // with a certificate in PrimalCert
	QPUnbounded     = 3 // with a certificate in DualCert (ADMM only)
	QPNumericalFail = 4
	QPNonConvex     = 5 // ADMM residuals above 1e30 (OSQP_NON_CVX)
	QPInaccurate    = 6 // ADMM met the tolerances ×10 only
)

// QPResult is the outcome of Quadprog.
type QPResult struct {
	X   []float64
	Fun float64
	// Residuals and marginals per family of constraints, one entry per input
	// row (a row unbounded on both sides has residual +Inf and marginal 0).
	// The marginal of a row a is the coefficient with which it enters
	// P·x + q = Σ marginal·a, so that
	//   P·x + q = AUbᵀ·IneqLin + AEqᵀ·EqLin + Aᵀ·Linear + A_sparseᵀ·Sparse
	//             + Lower + Upper,
	// the sign convention of Linprog (HiGHS). The residual of a two-sided row
	// (Linear, Sparse) is the slack of its nearer finite side. Lower and Upper
	// are indexed by variable; a variable pinned by lb == ub puts its marginal
	// in Lower when it is >= 0 and in Upper otherwise.
	IneqLin, EqLin, Linear, Sparse, Lower, Upper LPDual
	// Y holds OSQP-form multipliers (P·x + q + Aᵀy = 0, so y = −marginal) of
	// the canonical rows (design-qp.md §2.2: AEq rows, AUb rows, the Linear
	// blocks, the ASparse rows, then the finite bounds, rows unbounded on both
	// sides dropped).
	Y []float64
	// ActiveSet lists the canonical rows in the final working set.
	ActiveSet []int
	// PrimalCert and DualCert are the certificates of infeasibility and
	// unboundedness (nil until the solvers that produce them exist).
	PrimalCert, DualCert []float64
	// Optimality is ‖P·x + q − Σ marginal·a‖∞ recomputed from scratch,
	// ConstrViolation the largest violation of a canonical row, DualityGap the
	// gap between the primal and dual objectives (0 for a direct solve).
	Optimality, ConstrViolation, DualityGap float64
	Status                                  int
	// Success reports Status == QPOptimal.
	Success bool
	Message string
	// Method is the method that ran (QPAuto resolved).
	Method     QPMethod
	Nit        int
	RhoUpdates int
	Polished   bool
}

// Row kinds of the canonical list.
const (
	rowDense  = iota // a row of qpModel.dense
	rowSparse        // a row of qpModel.sp
	rowBound         // the unit row of one variable
)

// Blocks of the result a canonical row reports into.
const (
	blkEq = iota
	blkUb
	blkLinear
	blkSparse
	blkLower // a bound row; the marginal's sign picks Lower or Upper
)

// qpRow is one canonical row l <= a·x <= u.
type qpRow struct {
	l, u float64
	kind int
	ref  int // dense: row of qpModel.dense; sparse: row of qpModel.sp; bound: variable
	blk  int
	src  int // position in the input family (row of AEq, AUb, the Linear blocks, ASparse, or variable)
}

// qpModel is the validated problem: dense symmetric P (or its upper
// triangle), q, and the canonical rows of design-qp.md §2.2.
type qpModel struct {
	n      int
	p      []float64 // n×n symmetric, row-major; nil when P = 0
	pUpper *sparse.CSC[float64]
	chol   []float64 // lower Cholesky factor of P, set by positiveDefinite
	q      []float64
	dense  []float64 // dense canonical rows, row-major, one per rowDense
	sp     *sparse.CSR[float64]
	rows   []qpRow
	// sizes of the input families, for the result blocks.
	nEq, nUb, nLin, nSp int
}

// QPSolver holds a validated problem. It is not safe for concurrent use. The
// vector and matrix updates and warm starts of design-qp.md §5 arrive with
// the solvers that need them.
type QPSolver struct {
	m    *qpModel
	opts QPOptions
	meth QPMethod
}

// Quadprog solves p once: NewQPSolver followed by Solve.
func Quadprog(p QPProblem, opts *QPOptions) (*QPResult, error) {
	s, err := NewQPSolver(p, opts)
	if err != nil {
		return nil, err
	}
	return s.Solve()
}

// NewQPSolver validates p and builds the canonical rows. It reports every
// invalid input as an error (ErrDimension, ErrInput, ErrBounds or ErrMethod)
// before allocating anything proportional to the data it did not receive.
func NewQPSolver(p QPProblem, opts *QPOptions) (*QPSolver, error) {
	var o QPOptions
	if opts != nil {
		o = *opts
	}
	if len(p.Q) == 0 {
		return nil, fmt.Errorf("%w: Q must have n >= 1 entries", ErrDimension)
	}
	m, err := newQPModel(p)
	if err != nil {
		return nil, err
	}
	if err := o.validate(m.n, len(m.rows)); err != nil {
		return nil, err
	}
	s := &QPSolver{m: m, opts: o}
	if s.meth, err = m.chooseMethod(o.Method); err != nil {
		return nil, err
	}
	return s, nil
}

// Solve runs the solver chosen at construction.
func (s *QPSolver) Solve() (*QPResult, error) {
	m := s.m
	if s.meth == QPADMM {
		return nil, fmt.Errorf("%w: the ADMM solver is not implemented yet (design-qp.md task 8)", ErrMethod)
	}
	for _, r := range m.rows {
		if r.l != r.u {
			return nil, fmt.Errorf("%w: the active-set solver is not implemented yet (design-qp.md task 6); only equality constraints are solved", ErrMethod)
		}
	}
	feasTol := s.opts.FeasTol
	if feasTol == 0 {
		feasTol = 1e-9
	}
	return m.solveEqualities(feasTol)
}

// validate checks the options against n variables and mRows canonical rows.
func (o *QPOptions) validate(n, mRows int) error {
	switch o.Method {
	case QPAuto, QPActiveSet, QPADMM:
	default:
		return fmt.Errorf("%w: %q", ErrMethod, o.Method)
	}
	if o.MaxIter < 0 || o.PolishRefine < 0 || o.CheckInterval < 0 {
		return fmt.Errorf("%w: MaxIter, PolishRefine and CheckInterval must not be negative", ErrInput)
	}
	for _, f := range []float64{o.FeasTol, o.EpsAbs, o.EpsRel, o.EpsPrimInf, o.EpsDualInf,
		o.Rho, o.Sigma, o.Alpha, o.PolishDelta} {
		if math.IsNaN(f) || math.IsInf(f, 0) || f < 0 {
			return fmt.Errorf("%w: tolerances and ADMM parameters must be finite and not negative", ErrInput)
		}
	}
	if o.Alpha >= 2 {
		return fmt.Errorf("%w: Alpha must be in (0, 2)", ErrInput)
	}
	if o.X0 != nil && len(o.X0) != n {
		return fmt.Errorf("%w: X0 has %d entries, want %d", ErrDimension, len(o.X0), n)
	}
	if o.Y0 != nil && len(o.Y0) != mRows {
		return fmt.Errorf("%w: Y0 has %d entries, want %d (canonical rows)", ErrDimension, len(o.Y0), mRows)
	}
	if !allFinite(o.X0) || !allFinite(o.Y0) {
		return fmt.Errorf("%w: X0 and Y0 must be finite", ErrInput)
	}
	seen := make(map[int]bool, len(o.WorkingSet))
	for _, w := range o.WorkingSet {
		if w < 0 || w >= mRows || seen[w] {
			return fmt.Errorf("%w: WorkingSet entry %d is out of range or repeated (%d canonical rows)", ErrInput, w, mRows)
		}
		seen[w] = true
	}
	return nil
}

func allFinite(v []float64) bool {
	for _, x := range v {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return false
		}
	}
	return true
}

// checkSides rejects NaN, a +Inf lower side, a -Inf upper side and l > u.
func checkSides(l, u []float64, what string) error {
	for i := range l {
		if math.IsNaN(l[i]) || math.IsNaN(u[i]) || math.IsInf(l[i], 1) || math.IsInf(u[i], -1) || l[i] > u[i] {
			return fmt.Errorf("%w: %s %d: invalid interval [%g, %g]", ErrBounds, what, i, l[i], u[i])
		}
	}
	return nil
}

// qpMatrix reads an optional dense matrix with n columns (a 1-D array of
// length n is one row) and returns its rows and row-major copy.
func qpMatrix(a *ndarray.Array, n int, what string) ([]float64, int, error) {
	if a == nil {
		return nil, 0, nil
	}
	sh := a.Shape()
	rows := 1
	switch {
	case len(sh) == 1 && sh[0] == n:
	case len(sh) == 2 && sh[1] == n:
		rows = sh[0]
	default:
		return nil, 0, fmt.Errorf("%w: %s has shape %v, want (m, %d)", ErrDimension, what, sh, n)
	}
	d := arrayData(a)
	if !allFinite(d) {
		return nil, 0, fmt.Errorf("%w: %s contains NaN or Inf", ErrInput, what)
	}
	return d, rows, nil
}

// qpVector reads the right-hand side of a dense family with rows rows.
func qpVector(b *ndarray.Array, rows int, what string) ([]float64, error) {
	if b == nil {
		if rows != 0 {
			return nil, fmt.Errorf("%w: %s is required with its matrix", ErrDimension, what)
		}
		return nil, nil
	}
	if rows == 0 {
		return nil, fmt.Errorf("%w: %s given without its matrix", ErrDimension, what)
	}
	d := arrayData(b)
	if len(b.Shape()) != 1 || len(d) != rows {
		return nil, fmt.Errorf("%w: %s must be a vector of %d entries", ErrDimension, what, rows)
	}
	return d, nil
}

// qpSparse revalidates a user-built CSC (its fields are exported) and
// checks the values.
func qpSparse(a *sparse.CSC[float64], rows, cols int, what string) (*sparse.CSC[float64], error) {
	b, err := sparse.NewCSC(a.Rows, a.Cols, a.Indptr, a.Indices, a.Data)
	if err != nil {
		return nil, fmt.Errorf("%w: %s: %v", ErrInput, what, err)
	}
	if (rows >= 0 && b.Rows != rows) || b.Cols != cols {
		return nil, fmt.Errorf("%w: %s is %d×%d, want %d columns", ErrDimension, what, b.Rows, b.Cols, cols)
	}
	if !allFinite(b.Data) {
		return nil, fmt.Errorf("%w: %s contains NaN or Inf", ErrInput, what)
	}
	return b, nil
}

// qpSizes accumulates the sizes derived from the input in 64 bits and
// reports the first that exceeds math.MaxInt32, which bounds go-sparse's
// int32 row indices, and so any 32-bit int.
type qpSizes struct{ nnz, rows int64 }

func (z *qpSizes) add(nnz, rows int64) error {
	z.nnz += nnz
	z.rows += rows
	if z.nnz > qpMaxSize || z.rows > qpMaxSize {
		return fmt.Errorf("%w: problem too large: nnz(KKT) = nnz(P) + nnz(A) + n + m and the number of rows are limited to %d (go-sparse stores row indices as int32)",
			ErrInput, qpMaxSize)
	}
	return nil
}

// newQPModel validates p and builds the canonical row list.
func newQPModel(p QPProblem) (*qpModel, error) {
	n := len(p.Q)
	if int64(n) > qpMaxSize {
		return nil, fmt.Errorf("%w: n = %d exceeds %d", ErrInput, n, qpMaxSize)
	}
	if !allFinite(p.Q) {
		return nil, fmt.Errorf("%w: Q contains NaN or Inf", ErrInput)
	}
	md := &qpModel{n: n, q: append([]float64(nil), p.Q...)}
	z := qpSizes{nnz: int64(n), rows: int64(n)} // the identity block of the KKT matrix; n <= qpMaxSize
	if err := md.setP(p, &z); err != nil {
		return nil, err
	}
	c := p.Cons
	if err := md.addDenseFamily(c.AEq, c.BEq, "AEq", &z); err != nil {
		return nil, err
	}
	if err := md.addDenseFamily(c.AUb, c.BUb, "AUb", &z); err != nil {
		return nil, err
	}
	for i, lc := range c.Linear {
		if err := md.addLinear(lc, i, &z); err != nil {
			return nil, err
		}
	}
	if c.ASparse != nil {
		if err := md.addSparse(c.ASparse, c.L, c.U, &z); err != nil {
			return nil, err
		}
	} else if c.L != nil || c.U != nil {
		return nil, fmt.Errorf("%w: L and U given without ASparse", ErrDimension)
	}
	if err := md.addBounds(c.Bounds, &z); err != nil {
		return nil, err
	}
	return md, nil
}

// setP reads P (dense, symmetrised) or PSparse (upper triangle).
func (md *qpModel) setP(p QPProblem, z *qpSizes) error {
	n := md.n
	switch {
	case p.P != nil && p.PSparse != nil:
		return fmt.Errorf("%w: give P or PSparse, not both", ErrInput)
	case p.P != nil:
		sh := p.P.Shape()
		if len(sh) != 2 || sh[0] != n || sh[1] != n {
			return fmt.Errorf("%w: P has shape %v, want (%d, %d)", ErrDimension, sh, n, n)
		}
		d := arrayData(p.P)
		if !allFinite(d) {
			return fmt.Errorf("%w: P contains NaN or Inf", ErrInput)
		}
		for i := 0; i < n; i++ {
			for j := 0; j < i; j++ {
				v := 0.5 * (d[i*n+j] + d[j*n+i])
				d[i*n+j], d[j*n+i] = v, v
			}
		}
		md.p = d
		return z.add(int64(n)*int64(n+1)/2, 0)
	case p.PSparse != nil:
		u, err := qpSparse(p.PSparse, n, n, "PSparse")
		if err != nil {
			return err
		}
		for j := 0; j < n; j++ {
			for k := u.Indptr[j]; k < u.Indptr[j+1]; k++ {
				if int(u.Indices[k]) > j {
					return fmt.Errorf("%w: PSparse has an entry below the diagonal (row %d, column %d); give the upper triangle only", ErrInput, u.Indices[k], j)
				}
			}
		}
		md.pUpper = u
		return z.add(int64(u.NNZ()), 0)
	}
	return nil
}

// symmetricP returns P as a dense symmetric n×n matrix (a copy of the model's
// for a dense P, expanded from the upper triangle for PSparse, zeros for
// P = 0), or an error if it would exceed qpDirectLimit entries.
func (md *qpModel) symmetricP() ([]float64, error) {
	n := md.n
	if int64(n)*int64(n) > qpDirectLimit {
		return nil, fmt.Errorf("%w: n = %d: a dense P would exceed %d entries", ErrInput, n, qpDirectLimit)
	}
	d := make([]float64, n*n)
	switch {
	case md.p != nil:
		copy(d, md.p)
	case md.pUpper != nil:
		u := md.pUpper
		for j := 0; j < n; j++ {
			for k := u.Indptr[j]; k < u.Indptr[j+1]; k++ {
				i := int(u.Indices[k])
				d[i*n+j] += u.Data[k]
				if i != j {
					d[j*n+i] += u.Data[k]
				}
			}
		}
	}
	return d, nil
}

// addRow appends a canonical row unless it is unbounded on both sides.
func (md *qpModel) addRow(r qpRow) {
	if math.IsInf(r.l, -1) && math.IsInf(r.u, 1) {
		return
	}
	md.rows = append(md.rows, r)
}

// addDenseFamily adds AEq (equalities) or AUb (upper-bounded) rows.
func (md *qpModel) addDenseFamily(a, b *ndarray.Array, what string, z *qpSizes) error {
	d, rows, err := qpMatrix(a, md.n, what)
	if err != nil {
		return err
	}
	bn := "BEq"
	if what == "AUb" {
		bn = "BUb"
	}
	bv, err := qpVector(b, rows, bn)
	if err != nil {
		return err
	}
	if err := z.add(int64(rows)*int64(md.n), int64(rows)); err != nil {
		return err
	}
	eq := what == "AEq"
	for i, v := range bv {
		switch {
		case math.IsNaN(v):
			return fmt.Errorf("%w: %s[%d] is NaN", ErrBounds, bn, i)
		case eq && math.IsInf(v, 0):
			return fmt.Errorf("%w: %s[%d] must be finite", ErrInput, bn, i)
		case !eq && math.IsInf(v, -1):
			return fmt.Errorf("%w: %s[%d] is -Inf: the row cannot be satisfied", ErrBounds, bn, i)
		}
	}
	if eq {
		md.nEq = rows
	} else {
		md.nUb = rows
	}
	for i := 0; i < rows; i++ {
		r := qpRow{l: math.Inf(-1), u: bv[i], kind: rowDense, ref: len(md.dense) / md.n, blk: blkUb, src: i}
		if eq {
			r.l, r.blk = bv[i], blkEq
		}
		if math.IsInf(r.l, -1) && math.IsInf(r.u, 1) {
			continue // BUb = +Inf: vacuous row, dropped
		}
		md.dense = append(md.dense, d[i*md.n:(i+1)*md.n]...)
		md.rows = append(md.rows, r)
	}
	return nil
}

// addLinear adds one Lb <= A·x <= Ub block.
func (md *qpModel) addLinear(lc LinearConstraint, idx int, z *qpSizes) error {
	what := fmt.Sprintf("Linear[%d].A", idx)
	if lc.A == nil {
		return fmt.Errorf("%w: %s is required", ErrInput, what)
	}
	d, rows, err := qpMatrix(lc.A, md.n, what)
	if err != nil {
		return err
	}
	if rows == 0 {
		return fmt.Errorf("%w: %s has no rows", ErrInput, what)
	}
	l, err := boundVector(lc.Lb, rows, math.Inf(-1))
	if err != nil {
		return fmt.Errorf("%w: Linear[%d].Lb has %d entries, want 0, 1 or %d", err, idx, len(lc.Lb), rows)
	}
	u, err := boundVector(lc.Ub, rows, math.Inf(1))
	if err != nil {
		return fmt.Errorf("%w: Linear[%d].Ub has %d entries, want 0, 1 or %d", err, idx, len(lc.Ub), rows)
	}
	if err := checkSides(l, u, fmt.Sprintf("Linear[%d] row", idx)); err != nil {
		return err
	}
	if err := z.add(int64(rows)*int64(md.n), int64(rows)); err != nil {
		return err
	}
	for i := 0; i < rows; i++ {
		if math.IsInf(l[i], -1) && math.IsInf(u[i], 1) {
			continue
		}
		md.rows = append(md.rows, qpRow{l: l[i], u: u[i], kind: rowDense, ref: len(md.dense) / md.n, blk: blkLinear, src: md.nLin + i})
		md.dense = append(md.dense, d[i*md.n:(i+1)*md.n]...)
	}
	md.nLin += rows
	return nil
}

// addSparse adds the rows l <= ASparse·x <= u.
func (md *qpModel) addSparse(a *sparse.CSC[float64], lIn, uIn []float64, z *qpSizes) error {
	b, err := qpSparse(a, -1, md.n, "ASparse")
	if err != nil {
		return err
	}
	// A CSC can declare 2^31 empty rows in a few bytes: allocate nothing
	// proportional to Rows unless the data (entries and bounds) can account
	// for the rows.
	if int64(b.Rows) > int64(b.NNZ())+int64(len(lIn))+int64(len(uIn))+int64(md.n) {
		return fmt.Errorf("%w: ASparse declares %d rows but has %d entries and %d+%d bounds", ErrInput, b.Rows, b.NNZ(), len(lIn), len(uIn))
	}
	l, err := boundVector(lIn, b.Rows, math.Inf(-1))
	if err != nil {
		return fmt.Errorf("%w: L has %d entries, want 0, 1 or %d", err, len(lIn), b.Rows)
	}
	u, err := boundVector(uIn, b.Rows, math.Inf(1))
	if err != nil {
		return fmt.Errorf("%w: U has %d entries, want 0, 1 or %d", err, len(uIn), b.Rows)
	}
	if err := checkSides(l, u, "ASparse row"); err != nil {
		return err
	}
	if err := z.add(int64(b.NNZ()), int64(b.Rows)); err != nil {
		return err
	}
	md.sp = b.ToCSR()
	md.nSp = b.Rows
	for i := 0; i < b.Rows; i++ {
		md.addRow(qpRow{l: l[i], u: u[i], kind: rowSparse, ref: i, blk: blkSparse, src: i})
	}
	return nil
}

// addBounds adds the finite variable bounds as unit rows.
func (md *qpModel) addBounds(bs []Bound, z *qpSizes) error {
	if bs == nil {
		return nil
	}
	lb, ub, err := splitBounds(bs, md.n)
	if err != nil {
		return err
	}
	if err := checkSides(lb, ub, "bound"); err != nil {
		return err
	}
	if err := z.add(0, int64(md.n)); err != nil {
		return err
	}
	for i := 0; i < md.n; i++ {
		md.addRow(qpRow{l: lb[i], u: ub[i], kind: rowBound, ref: i, blk: blkLower, src: i})
	}
	return nil
}

// chooseMethod resolves QPAuto (design-qp.md §2.3) and checks what the
// requested method needs.
func (md *qpModel) chooseMethod(want QPMethod) (QPMethod, error) {
	switch want {
	case QPActiveSet:
		if md.sp != nil {
			return "", fmt.Errorf("%w: the active-set method takes dense constraints; use QPADMM for ASparse", ErrInput)
		}
		if !md.positiveDefinite() {
			return "", fmt.Errorf("%w: active-set needs a positive definite P; use QPADMM", ErrInput)
		}
		return QPActiveSet, nil
	case QPADMM:
		return QPADMM, nil
	}
	if md.p != nil && md.sp == nil && md.n+len(md.rows) <= qpAutoLimit && md.positiveDefinite() {
		return QPActiveSet, nil
	}
	return QPADMM, nil
}

// positiveDefinite reports whether the Cholesky factorisation of P succeeds
// (our definition of "positive definite"); P = 0 is not.
func (md *qpModel) positiveDefinite() bool {
	if md.p == nil && md.pUpper == nil {
		return false
	}
	d, err := md.symmetricP()
	if err != nil || !lpCholesky(d, md.n) {
		return false
	}
	md.chol = d
	return true
}

// rowDot returns a·x for canonical row r.
func (md *qpModel) rowDot(r qpRow, x []float64) float64 {
	switch r.kind {
	case rowDense:
		return dot(md.dense[r.ref*md.n:(r.ref+1)*md.n], x)
	case rowSparse:
		s := 0.0
		for k := md.sp.Indptr[r.ref]; k < md.sp.Indptr[r.ref+1]; k++ {
			s += float64(md.sp.Data[k] * x[md.sp.Indices[k]])
		}
		return s
	}
	return x[r.ref]
}

// rowAdd adds s·a to out for canonical row r.
func (md *qpModel) rowAdd(r qpRow, s float64, out []float64) {
	switch r.kind {
	case rowDense:
		for j, v := range md.dense[r.ref*md.n : (r.ref+1)*md.n] {
			out[j] += float64(s * v)
		}
	case rowSparse:
		for k := md.sp.Indptr[r.ref]; k < md.sp.Indptr[r.ref+1]; k++ {
			out[md.sp.Indices[k]] += float64(s * md.sp.Data[k])
		}
	default:
		out[r.ref] += s
	}
}

// solveEqualities solves min ½xᵀPx + qᵀx s.t. a_j·x = b_j for every canonical
// row (all equalities), P positive definite, by the range-space method:
// with P = LLᵀ, W = L⁻¹Aᵀ and S = WᵀW, the multipliers solve S·λ = b + Wᵀ·L⁻¹q
// and x = L⁻ᵀ(Wλ − L⁻¹q).
func (md *qpModel) solveEqualities(feasTol float64) (*QPResult, error) {
	n, mr := md.n, len(md.rows)
	if int64(mr)*int64(n) > qpDirectLimit {
		return nil, fmt.Errorf("%w: %d equality rows × %d variables exceed the %d entries of the direct solve", ErrInput, mr, n, qpDirectLimit)
	}
	// Solve is reached with the active-set method only, which chooseMethod
	// allows for a positive definite P: md.chol is its Cholesky factor.
	chol := md.chol
	z := forwardSolve(chol, n, md.q) // L⁻¹q
	w := make([][]float64, mr)       // rows of W: L⁻¹a_j
	rhs := make([]float64, mr)
	for j, r := range md.rows {
		a := make([]float64, n)
		md.rowAdd(r, 1, a)
		w[j] = forwardSolve(chol, n, a)
		rhs[j] = r.l + dot(w[j], z)
	}
	lambda := make([]float64, mr)
	if mr > 0 {
		s := make([]float64, mr*mr)
		for i := 0; i < mr; i++ {
			for j := 0; j <= i; j++ {
				v := dot(w[i], w[j])
				s[i*mr+j], s[j*mr+i] = v, v
			}
		}
		if !choleskyRel(s, mr, 1e-12) {
			return nil, fmt.Errorf("%w: linearly dependent equality rows need the Goldfarb–Idnani solver (design-qp.md task 6)", ErrMethod)
		}
		lambda = lpCholSolve(s, mr, rhs)
	}
	t := make([]float64, n) // Wλ − L⁻¹q
	for i := range t {
		t[i] = -z[i]
	}
	for j := range w {
		for i := range t {
			t[i] += float64(lambda[j] * w[j][i])
		}
	}
	x := backSolve(chol, n, t)
	return md.assemble(x, lambda, QPActiveSet, feasTol), nil
}

// choleskyRel is lpCholesky with a relative pivot test: it also reports false
// when a pivot is below rel times the diagonal entry it was taken from, which
// is how the linear dependence of the rows of A shows in S = A·P⁻¹·Aᵀ (an
// exact zero pivot is rare after rounding).
func choleskyRel(a []float64, m int, rel float64) bool {
	for j := 0; j < m; j++ {
		d := a[j*m+j]
		d0 := d
		for k := 0; k < j; k++ {
			d -= float64(a[j*m+k] * a[j*m+k])
		}
		if !(d > rel*d0) {
			return false
		}
		d = math.Sqrt(d)
		a[j*m+j] = d
		for i := j + 1; i < m; i++ {
			s := a[i*m+j]
			for k := 0; k < j; k++ {
				s -= float64(a[i*m+k] * a[j*m+k])
			}
			a[i*m+j] = s / d
		}
	}
	return true
}

// forwardSolve solves L·y = b for the lower-triangular factor of lpCholesky.
func forwardSolve(l []float64, n int, b []float64) []float64 {
	y := clone(b)
	for i := 0; i < n; i++ {
		s := y[i]
		for k := 0; k < i; k++ {
			s -= float64(l[i*n+k] * y[k])
		}
		y[i] = s / l[i*n+i]
	}
	return y
}

// backSolve solves Lᵀ·x = y.
func backSolve(l []float64, n int, y []float64) []float64 {
	x := clone(y)
	for i := n - 1; i >= 0; i-- {
		s := x[i]
		for k := i + 1; k < n; k++ {
			s -= float64(l[k*n+i] * x[k])
		}
		x[i] = s / l[i*n+i]
	}
	return x
}

// px returns P·x (zeros for P = 0).
func (md *qpModel) px(x []float64) []float64 {
	out := make([]float64, md.n)
	switch {
	case md.p != nil:
		for i := range out {
			out[i] = dot(md.p[i*md.n:(i+1)*md.n], x)
		}
	case md.pUpper != nil:
		u := md.pUpper
		for j := 0; j < md.n; j++ {
			for k := u.Indptr[j]; k < u.Indptr[j+1]; k++ {
				i := int(u.Indices[k])
				out[i] += float64(u.Data[k] * x[j])
				if i != j {
					out[j] += float64(u.Data[k] * x[i])
				}
			}
		}
	}
	return out
}

// assemble builds the result for the point x and marginals marg of the
// canonical rows (P·x + q = Σ marg_j·a_j).
func (md *qpModel) assemble(x, marg []float64, method QPMethod, feasTol float64) *QPResult {
	n := md.n
	grad := md.px(x)
	fun := 0.5*dot(grad, x) + dot(md.q, x)
	res := make([]float64, n) // P·x + q − Σ marg·a
	for i := range res {
		res[i] = grad[i] + md.q[i]
	}
	r := &QPResult{X: x, Fun: fun, Method: method, Status: QPOptimal, Success: true,
		Message: "Optimal solution found (direct solve of the equality-constrained KKT system).",
		Y:       make([]float64, len(md.rows)),
		IneqLin: md.block(md.nUb), EqLin: md.block(md.nEq), Linear: md.block(md.nLin),
		Sparse: md.block(md.nSp), Lower: md.block(n), Upper: md.block(n)}
	for j, row := range md.rows {
		md.rowAdd(row, -marg[j], res)
		ax := md.rowDot(row, x)
		r.ConstrViolation = math.Max(r.ConstrViolation, math.Max(row.l-ax, ax-row.u))
		r.Y[j] = -marg[j]
		if row.l == row.u {
			r.ActiveSet = append(r.ActiveSet, j)
		}
		md.report(r, row, ax, marg[j])
	}
	for _, v := range res {
		r.Optimality = math.Max(r.Optimality, math.Abs(v))
	}
	// design-qp.md §3.6: optimality is checked against (34), and the status
	// drops to QPNumericalFail when the recomputed residuals say the point is
	// not a solution (NaN fails both tests).
	scale := math.Max(1, math.Max(vnormInf(md.q), vnormInf(grad)))
	// An infinite scale (an overflowing P·x) would accept any residual.
	if !(scale <= math.MaxFloat64) || !(r.Optimality <= 1e-8*scale) || !(r.ConstrViolation <= feasTol*(1+md.maxSide())) {
		r.Status, r.Success = QPNumericalFail, false
		r.Message = fmt.Sprintf("Numerical difficulties: stationarity residual %.3g, constraint violation %.3g.", r.Optimality, r.ConstrViolation)
	}
	return r
}

// maxSide is the largest finite |l| or |u| of the canonical rows, the scale
// of the constraint violation test.
func (md *qpModel) maxSide() float64 {
	m := 0.0
	for _, r := range md.rows {
		for _, v := range []float64{r.l, r.u} {
			if !math.IsInf(v, 0) {
				m = math.Max(m, math.Abs(v))
			}
		}
	}
	return m
}

// block returns an LPDual whose residuals are +Inf and marginals 0: the
// state of a row that is unbounded on both sides.
func (md *qpModel) block(rows int) LPDual {
	d := LPDual{Residual: make([]float64, rows), Marginals: make([]float64, rows)}
	for i := range d.Residual {
		d.Residual[i] = math.Inf(1)
	}
	return d
}

// report writes the residual and marginal of canonical row row into its block.
func (md *qpModel) report(r *QPResult, row qpRow, ax, marg float64) {
	switch row.blk {
	case blkEq:
		r.EqLin.Residual[row.src], r.EqLin.Marginals[row.src] = row.l-ax, marg
	case blkUb:
		r.IneqLin.Residual[row.src], r.IneqLin.Marginals[row.src] = row.u-ax, marg
	case blkLinear:
		r.Linear.Residual[row.src], r.Linear.Marginals[row.src] = nearest(row, ax), marg
	case blkSparse:
		r.Sparse.Residual[row.src], r.Sparse.Marginals[row.src] = nearest(row, ax), marg
	default:
		// A bound row: both sides are reported for the variable; the
		// marginal goes to Lower when it is >= 0 and to Upper otherwise.
		r.Lower.Residual[row.src], r.Upper.Residual[row.src] = ax-row.l, row.u-ax
		if marg >= 0 {
			r.Lower.Marginals[row.src] = marg
		} else {
			r.Upper.Marginals[row.src] = marg
		}
	}
}

// nearest is the slack of the nearer finite side of a two-sided row.
func nearest(row qpRow, ax float64) float64 {
	return math.Min(ax-row.l, row.u-ax)
}
