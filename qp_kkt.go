package optimize

import (
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/go-sparse/sparse"
	"github.com/go-sparse/sparse/ldl"
	"github.com/go-sparse/sparse/order"
)

// The KKT matrix of the ADMM iteration (design-qp.md §4.1, OSQP's
// algebra/_common/kkt.c form_KKT, v1.0.0):
//
//	K = [ P + σI    Aᵀ        ]
//	    [ A         −diag(1/ρ) ]
//
// stored as its upper triangle in CSC (the form go-sparse's ldl takes). Column
// j < n holds the upper part of column j of P with its diagonal always present;
// column n+i holds row i of A followed by the diagonal −1/ρ_i. The diagonal
// positions are remembered so that a new ρ rewrites 2·m values and refactors.

// qpFactorLimit bounds the entries of L the ADMM solver will allocate (2²⁸ is
// 2 GiB of values plus 1 GiB of indices). A variable so that tests can lower it.
var qpFactorLimit int64 = 1 << 28

// qpTimeKKT turns on the accounting of the time spent in go-sparse: the AMD
// ordering and the symbolic analysis (qpKKTAnalyseNS), the numeric
// refactorisations (qpKKTFactorNS) and the triangular solves (qpKKTSolveNS), in
// nanoseconds. Only tools/qpbench's benchmark sets it: it splits the ADMM time into
// the part that belongs to go-sparse and the rest. Not safe for concurrent solves;
// off, it costs one branch per call.
var (
	qpTimeKKT                                   bool
	qpKKTAnalyseNS, qpKKTFactorNS, qpKKTSolveNS int64
)

// The ordering and the symbolic analysis only fail on malformed input, which
// buildKKT excludes; they are variables so that a test can make them fail.
var (
	qpAMD         = amdPooled
	qpNewPermuted = ldl.NewPermuted
)

// amdWorkspaces holds the scratch memory of order.AMD between solves: the
// ordering of a KKT matrix allocates only the permutation it returns.
var amdWorkspaces = sync.Pool{New: func() any { return new(order.Workspace) }}

func amdPooled(a *sparse.CSR[float64]) ([]int, error) {
	w := amdWorkspaces.Get().(*order.Workspace)
	defer amdWorkspaces.Put(w)
	return order.AMDWith(w, nil, a)
}

// admmCSR is the constraint matrix A by rows, the canonical row order.
type admmCSR struct {
	ptr []int
	idx []int32
	val []float64
}

// admmKKT is the assembled and factored KKT matrix.
type admmKKT struct {
	n, m    int
	mat     *sparse.CSC[float64]
	pDiag   []int // position of the diagonal of column j < n
	rhoDiag []int // position of the diagonal −1/ρ_i
	perm    []int
	fac     *ldl.Permuted
	pos     int // positive pivots of the last factorisation
}

// errNonConvex is returned when the KKT matrix is not quasi-definite: a zero
// pivot, or fewer than n positive pivots, which means P is not positive
// semidefinite (OSQP's OSQP_NONCVX_ERROR).
var errNonConvex = errors.New("the KKT matrix is not quasi-definite: P is not positive semidefinite")

// buildKKT assembles the upper triangle for P (upper CSC, any pattern), A
// (rows) and the given σ and 1/ρ.
//
// The sizes are bounded by newQPModel (nnz(KKT) ≤ qpMaxSize ≤ MaxInt32, design-
// qp.md §2.1) and every column is sorted by construction (P's and A's rows are
// validated sorted and in range), so the CSC is valid and nothing here fails.
func buildKKT(pu *sparse.CSC[float64], a *admmCSR, n, m int, sigma float64, rhoInv []float64) *admmKKT {
	k := &admmKKT{n: n, m: m, pDiag: make([]int, n), rhoDiag: make([]int, m)}
	nnz := pu.NNZ() + n + len(a.val) + m
	ptr := make([]int, 0, n+m+1)
	idx := make([]int32, 0, nnz)
	val := make([]float64, 0, nnz)
	ptr = append(ptr, 0)
	for j := 0; j < n; j++ {
		hasDiag := false
		for p := pu.Indptr[j]; p < pu.Indptr[j+1]; p++ {
			i := int(pu.Indices[p])
			if i == j {
				hasDiag = true
				k.pDiag[j] = len(idx)
				idx = append(idx, int32(i))
				val = append(val, pu.Data[p]+sigma)
				continue
			}
			idx = append(idx, int32(i))
			val = append(val, pu.Data[p])
		}
		if !hasDiag { // entries of a column are sorted and all rows < j: the diagonal goes last
			k.pDiag[j] = len(idx)
			idx = append(idx, int32(j))
			val = append(val, sigma)
		}
		ptr = append(ptr, len(idx))
	}
	for i := 0; i < m; i++ {
		for p := a.ptr[i]; p < a.ptr[i+1]; p++ {
			idx = append(idx, a.idx[p])
			val = append(val, a.val[p])
		}
		k.rhoDiag[i] = len(idx)
		idx = append(idx, int32(n+i))
		val = append(val, -rhoInv[i])
		ptr = append(ptr, len(idx))
	}
	k.mat = &sparse.CSC[float64]{Rows: n + m, Cols: n + m, Indptr: ptr, Indices: idx, Data: val}
	return k
}

// analyse orders K with AMD and runs the symbolic phase once; the factor it
// would need is refused above qpFactorLimit entries.
func (k *admmKKT) analyse() error {
	if qpTimeKKT {
		defer func(t time.Time) { qpKKTAnalyseNS += int64(time.Since(t)) }(time.Now())
	}
	perm, err := qpAMD(k.mat.ToCSR())
	if err != nil {
		return fmt.Errorf("%w: ordering the KKT matrix: %v", ErrInput, err)
	}
	k.perm = perm
	k.fac, err = qpNewPermuted(k.mat, perm)
	if err != nil {
		return fmt.Errorf("%w: analysing the KKT matrix: %v", ErrInput, err)
	}
	if int64(k.fac.NNZL()) > qpFactorLimit {
		return fmt.Errorf("%w: the LDLᵀ factor of the KKT matrix would have %d entries, over the limit %d", ErrInput, k.fac.NNZL(), qpFactorLimit)
	}
	return nil
}

// factor runs the numeric phase on the current values. A zero or non-finite
// pivot, or fewer than n positive pivots, is errNonConvex.
func (k *admmKKT) factor() error {
	if qpTimeKKT {
		defer func(t time.Time) { qpKKTFactorNS += int64(time.Since(t)) }(time.Now())
	}
	if err := k.fac.Refactor(k.mat); err != nil {
		return fmt.Errorf("%w: %v", errNonConvex, err)
	}
	if k.pos = k.fac.NumPositive(); k.pos != k.n {
		return fmt.Errorf("%w: %d positive pivots, want %d", errNonConvex, k.pos, k.n)
	}
	return nil
}

// solve overwrites b (length n+m) with K⁻¹b.
func (k *admmKKT) solve(b []float64) {
	if qpTimeKKT {
		defer func(t time.Time) { qpKKTSolveNS += int64(time.Since(t)) }(time.Now())
	}
	// The only errors are a wrong length or a missing factorisation, both
	// excluded by construction (factor succeeded before any solve).
	_ = k.fac.Solve(b)
}
