# Design: quadratic programming (`Quadprog`)

Status: **tasks 4, 5, 6 and 7 implemented**. Task 7 (`qp_solver.go`, `qp_gi.go`):
`QPSolver` keeps J, R, the working set and the multipliers between solves;
`UpdateQ`, `UpdateBounds` (same row kinds only) and `WorkingSet` recover x and u
from the factors by (35) and repair by dropping the most negative inequality
multiplier until none is negative (Prop. 8.1–8.2, read from arXiv:2608.30933
pp. 18–19); `UpdateMatrixValues` rebuilds the factors seeded with the previous
active set (ASparse values: `ErrMethod`); `WarmStart` validates and keeps x, y
for ADMM. Measured on 300 random box+row problems, pivots warm / cold:
14/493 = 0.03 (σ = 0.05 perturbation of q), 150/503 = 0.30 (σ = 0.5),
520/647 = 0.80 (σ = 2), each answer equal to the cold one to 1e-7 (this
supersedes the 506/480 of the first version below). The text that follows
describes the earlier state; the simplified-7 sentences are obsolete. Task 6
(`qp_gi.go`): the Goldfarb–Idnani walk of §3 for a positive definite dense P,
with equalities first, Householder insertion and 2×2-reflection deletion,
infeasibility certificates, the set-aside rule and the objective drift check;
checked against Clarabel (x to 1e-7) and quadprog (1e-8) on the 39-case oracle
fixture, and against `lsei` on 600 random problems (a witness that declines or
stops early on rank-deficient equalities: only a feasible lsei point with a
lower objective would fail the test). Differences from §3: R is a dense n×n
array, not packed; the slacks are recomputed each pass instead of updated
incrementally; the redundant-equality skip is a satisfied-row test. Task 7 is
simplified: `UpdateQ` and `QPOptions.WorkingSet` enter the previous active set
as equalities and use it only if dual feasible, else start cold (always the
cold answer; it saves selection scans, not pivots: measured 506 against 480
inserts and deletes over 245 accepted warm starts), not the factor repair of
Prop. 8.1–8.2; `UpdateBounds`, `UpdateMatrixValues` and `WarmStart` are not
written. Perf against quadprog is deferred (no quiet-machine measurement yet),
and the design's release plan (v0.5.0 at task 14) is kept. Earlier: task 4
(`quadprog.go`: the API of §2, validation, canonical rows, result blocks) and
**task 5 implemented** (`tools/qporacle gen` and `diff`, `testdata/qp/planted.json`);
the solvers of §3 and §4 are not written. Findings while implementing task 5:
the `osqp==1.1.3` wheel bundles the OSQP **v1.0.0** C core on x86_64 and arm64,
but has no Python call for it, so the harness loads the extension module with
`ctypes` and calls `osqp_version()` (and fails, before solving anything, if it
is not `1.0.0`); OSQP v1.0.0 refuses a non-convex P at setup (an
`OSQPException` from the failed LDLᵀ factorisation), it does not return a
status, which the ADMM solver's `QPNonConvex` test (§6.4) must account for;
Clarabel 0.11.1 accepts an indefinite P and returns "Solved", so it is no
oracle for non-convexity; `mm fetch/convert` stay in task 12.
Differences from §2 made while implementing: `QPResult.Sparse` holds the
residuals and marginals of the `ASparse` rows (the design had no block for
them); the `QPSolver` update and warm-start methods are left to tasks 7 and
11; an `ASparse` that declares more rows than its entries and bounds can
account for is rejected (a CSC can declare 2³¹ empty rows in a few bytes);
`QPAuto` takes the active set for a dense P only, so a sparse P needs
`QPActiveSet` explicitly. This document is item 5 of the go-scikit
parity plan (go-scikit/roadmap `PLAN.md`, wave 2). It specifies a dense dual
active-set solver (Goldfarb–Idnani) first, then an ADMM solver in the shape
of OSQP for large and sparse problems. It is written so that the
implementation can proceed without reopening the sources. Where the sources
were not available, that is stated, and nothing here depends on them.

The problem is

    minimise    ½ xᵀPx + qᵀx
    subject to  A_ub·x ≤ b_ub,  A_eq·x = b_eq,  l ≤ A·x ≤ u,  lb ≤ x ≤ ub

with P symmetric positive semidefinite (positive definite for the active-set
method).

## 1. Sources

### Opened

| Source | Licence | What it gives this design |
|---|---|---|
| go-scikit/roadmap `bib/optimization-learning-autodiff.md` (main, 2026-10-31) | — | QP gap row (MATLAB `quadprog`, Mathematica `QuadraticOptimization`), reference algorithms (Goldfarb–Idnani, OSQP, Mehrotra), the licence table (OSQP Apache-2.0 port; ECOS GPL oracle), the Maros–Mészáros corpus row |
| Stellato, Banjac, Goulart, Bemporad, Boyd, *OSQP: an operator splitting solver for quadratic programs*, Math. Prog. Comp. 12:637 (2020), doi:10.1007/s12532-020-00179-2; read as arXiv:1711.08013, pp. 4–17 | open access | §2 optimality conditions (4)–(9) and certificates (12)–(13); §3 the ADMM splitting (15)–(19), the KKT system (24), Algorithm 1; §3.4 termination (26) and infeasibility tests; §4 polishing (27)–(31); §5 Ruiz equilibration (Algorithm 2), unscaled termination, ρ selection and adaptation; §6 warm start and factor reuse; §7 defaults |
| OSQP C source, tag **v1.0.0** (`include/public/osqp_api_constants.h`, `src/auxil.c`, `src/osqp_api.c` solve loop, `src/scaling.c`, `src/polish.c`, `algebra/builtin/vector.c`) | Apache-2.0 | the exact defaults, the termination order, the duality-gap test, ρ-vector classification, the recession-cone projections, the polish acceptance rule — where v1.0.0 differs from the 2020 paper, the code wins (§4.0) |
| QDLDL, tag **v0.1.9** (`include/qdldl.h`) | Apache-2.0 | the quasi-definite LDLᵀ interface: elimination tree on the upper triangle, numeric factor returning the count of positive pivots and failing on an exact zero pivot, the three solves |
| Schmelzer & Stoll, *Goldfarb–Idnani Revisited: Invariants, Certificates, and the Limits of Guessing*, arXiv:2608.30933v1 (31 Aug 2026), pp. 1–20 | CC BY 4.0 | a complete restatement of the Goldfarb–Idnani method: invariants (10)–(11), directions (15), selection rule (19), step limits (23)–(24), objective increment (25), Algorithm 1, finite termination (Prop. 5.1–5.2), the Farkas certificate (Prop. 5.3), insertion by one Householder reflection (26)–(29), deletion by a chase of 2×2 symmetric reflections (30), recovery and repair for a new linear term (Prop. 8.1–8.2) |
| TU Darmstadt numerical analysis page *The dual QP solver of Goldfarb and Idnani* (numawww.mathematik.tu-darmstadt.de) | — | the problem form with equalities first, equalities never leaving once added, the Cholesky and rank checks — consistent with the note above |
| qpsolvers/maros_meszaros_qpbenchmark (main, pushed 2026-10-01): `README.md`, `data/README.md`, the `data/` listing, `maros_meszaros.py`, `maros_meszaros_dense_posdef.py`, `results/maros_meszaros_dense_posdef.md` | Apache-2.0 | the 138 problems as MAT files with their sizes, the form ½xᵀPx + qᵀx + r, l ≤ Ax ≤ u with `A = [C; I]` and ±1e20 as infinity, the dense (62) / dense positive-definite (19) / sparse (76) subsets, the qpbenchmark tolerance tiers (1e-3, 1e-6, 1e-9) |
| MathWorks `quadprog` documentation page | proprietary (documentation) | the argument order, the three algorithms, exit flags, `lambda` fields, default tolerances (active-set 1e-8, `MaxIterations` 10·(n + m)), `optimwarmstart` |
| This repository: `README.md`, `docs/design.md`, `types.go`, `constraints.go`, `linprog.go`, `slsqp_lsq.go`, `tools/slsqporacle/main.go`, `go.mod` | BSD-3-Clause | conventions reused below |
| go-sparse/sparse v0.1.0: file tree, `csc.go`, `order/amd.go`; go-ndarray v0.28.0 `linalg/` exported functions; go-simd/floats `floats.go` exported functions | BSD-3-Clause | what exists and what must be added (§9) |

### Not obtained

- **Goldfarb & Idnani 1983**, Math. Program. 27:1, doi:10.1007/BF02591962:
  behind Springer's login. The method is taken from Schmelzer & Stoll, who
  restate it with proofs.
- **Nocedal & Wright, *Numerical Optimization*, 2nd ed., ch. 16**: not
  available. Nothing here cites it directly.
- **Maros & Mészáros 1999**, Optim. Methods Softw. 11:671: not opened. The
  reference objective values therefore come from the oracles (§6), not from
  the paper's table.
- **quadprog** (R/Python, GPL), **qpOASES** (LGPL), **CVXOPT** (GPL): not
  opened, on purpose. They are run as black-box oracles and never read.
  **DAQP** (Arnström, Bemporad & Axehill, arXiv:2103.16236) was found but not
  opened. It is the reference for a later semidefinite extension (§10, task 15).
- **osqp_benchmarks** (the problem classes behind the paper's §8): not
  opened. The perf plan names it as the source to open before task 13.

## 2. API

QP is a separate entry point, like `Linprog`, not a `Minimize` method. The
data are matrices, not callbacks; the result carries constraint duals in the
`LPDual` shape; and a reusable solver object is needed for warm start and
factor reuse, which `Minimize` has no place for.

```go
// QPMethod names a quadratic-programming algorithm.
type QPMethod string

const (
	QPAuto      QPMethod = ""           // see "Method choice"
	QPActiveSet QPMethod = "active-set" // Goldfarb–Idnani dual active set, dense, P ≻ 0
	QPADMM      QPMethod = "admm"       // OSQP's ADMM, dense or sparse, P ⪰ 0
)

// QPProblem is min ½xᵀPx + qᵀx subject to the constraints. Give P either
// dense (P) or sparse (PSparse, upper triangle only, as OSQP takes it), never
// both; nil for both means P = 0 (an LP, ADMM only).
type QPProblem struct {
	P       *ndarray.Array        // n×n; symmetrised as (P+Pᵀ)/2, as quadprog does
	PSparse *sparse.CSC[float64]  // n×n, entries with row ≤ col only
	Q       []float64             // length n; defines n
	Cons    QPConstraints
}

// QPConstraints mirrors quadprog's (A, b, Aeq, beq, lb, ub) and adds the
// two-sided rows of OSQP. Every part is optional.
type QPConstraints struct {
	AUb, BUb *ndarray.Array        // A_ub·x ≤ b_ub
	AEq, BEq *ndarray.Array        // A_eq·x = b_eq
	Linear   []LinearConstraint    // Lb ≤ A·x ≤ Ub, the existing type, reused
	ASparse  *sparse.CSC[float64]  // l ≤ ASparse·x ≤ L/U below (ADMM only)
	L, U     []float64
	Bounds   []Bound               // nil = free, as in Minimize (NOT linprog's (0, +Inf))
}

type QPOptions struct {
	Method   QPMethod
	MaxIter  int     // active set: 10·(n + m) (quadprog); ADMM: 4000 (OSQP)
	// Active set.
	FeasTol  float64 // primal feasibility tolerance, default 1e-9 (§3.4)
	// ADMM — the zero value selects OSQP v1.0.0's default for each field.
	EpsAbs, EpsRel     float64 // 1e-3, 1e-3
	EpsPrimInf, EpsDualInf float64 // 1e-4, 1e-4
	Rho, Sigma, Alpha  float64 // 0.1, 1e-6, 1.6
	ScalingIter        int     // 10; negative disables scaling
	AdaptiveRhoInterval int    // 50; negative disables adaptation
	CheckInterval      int     // 25 (OSQP check_termination)
	Polish             bool    // false (OSQP v1.0.0 default)
	PolishDelta        float64 // 1e-6
	PolishRefine       int     // 3
	NoDualGapCheck     bool    // OSQP check_dualgap = 1 by default
	// Warm start (both methods; see §5).
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
	QPInaccurate    = 6 // ADMM met the tolerances ×10 only (OSQP *_INACCURATE)
)

type QPResult struct {
	X       []float64
	Fun     float64
	// Constraint residuals and marginals with Linprog's (HiGHS) sign
	// convention, so that P·x + q = A_ubᵀ·ineqlin + A_eqᵀ·eqlin + Aᵀ·linear
	// + lower + upper.
	IneqLin, EqLin, Linear, Lower, Upper LPDual
	Y          []float64 // OSQP-form multipliers of the canonical rows (§2.2)
	ActiveSet  []int     // canonical row indices in the final working set
	PrimalCert []float64 // δy satisfying (12) when Status == QPInfeasible
	DualCert   []float64 // δx satisfying (13) when Status == QPUnbounded
	Optimality, ConstrViolation, DualityGap float64 // ∞-norms, unscaled
	Status  int
	Success bool // Status == QPOptimal
	Message string
	Method  QPMethod // the method that ran (resolves QPAuto)
	Nit, RhoUpdates int
	Polished bool
}

// Quadprog solves p once: NewQPSolver followed by Solve.
func Quadprog(p QPProblem, opts *QPOptions) (*QPResult, error)

// QPSolver keeps the factorisations between solves (OSQP's setup/solve/update).
type QPSolver struct{ /* unexported */ }

func NewQPSolver(p QPProblem, opts *QPOptions) (*QPSolver, error)
func (s *QPSolver) Solve() (*QPResult, error)
func (s *QPSolver) UpdateQ(q []float64) error           // keeps every factor
func (s *QPSolver) UpdateBounds(l, u []float64) error   // canonical rows; keeps factors unless an equality flips (ADMM ρ)
func (s *QPSolver) UpdateMatrixValues(p, a []float64) error // same patterns; ADMM refactors numerically, active set re-Cholesky
func (s *QPSolver) WarmStart(x, y []float64) error
```

The name `Quadprog` follows `Linprog` and MATLAB's `quadprog`; there is no
SciPy counterpart. go-sparse becomes a direct dependency of this module
(its licence is to be confirmed BSD-3-Clause before task 4).

### 2.1 Validation (errors, never panics)

All checks happen in `NewQPSolver`, before any allocation proportional to
the data:

| Input | Outcome |
|---|---|
| `len(Q) == 0`, or a matrix with the wrong shape, or both `P` and `PSparse` | `ErrDimension` / `ErrInput` |
| NaN or ±Inf in P, q or any constraint matrix; NaN in a bound; `+Inf` lower or `-Inf` upper | `ErrInput` / `ErrBounds` |
| lb > ub or l > u | `ErrBounds` (as `splitBounds` and OSQP's `validate_data`; quadprog reports infeasibility, Minimize errors — we side with Minimize) |
| sparse: unsorted or duplicate indices, `PSparse` with an entry below the diagonal | `ErrInput` (go-sparse `NewCSC` validates the arrays) |
| sizes: `n·n` (dense P), `m·n` (dense A), and nnz(KKT) = nnz(P_upper) + nnz(A) + n + m computed with overflow checks against `math.MaxInt` **and** `math.MaxInt32` (go-sparse stores row indices as `int32`) | `ErrInput` naming the limit |
| `QPActiveSet` and P not positive definite (the Cholesky of P fails) | `ErrInput`: "active-set needs a positive definite P; use QPADMM" |
| unknown method | `ErrMethod` |

Non-convexity is not detectable before solving in the semidefinite case:
ADMM reports it as status 5 (OSQP's divergence test). Infeasibility and
unboundedness are statuses, never errors, as in `Linprog`.

### 2.2 Canonical forms

`NewQPSolver` builds one row list, in this order: `AEq` rows, `AUb` rows,
each `Linear` block, the `ASparse` rows, then the finite bounds. Each
canonical row `j` has `(a_j, l_j, u_j)`: equality rows have `l = u`; `A_ub`
rows have `l = -Inf`; bounds rows are `e_i` with `(lb_i, ub_i)`. Rows that are
unbounded on both sides are dropped. `QPResult.Y` and `ActiveSet` index this
list, and the `LPDual` blocks are scattered back from it.

- **Active set** consumes it as `Cᵀx ≥ b` with equalities first
  (Schmelzer & Stoll (2), P = G, a = −q): an equality row gives one column
  `c = a_j`, `b = l_j`; a one-sided row gives `c = a_j, b = l_j` (lower) or
  `c = −a_j, b = −u_j` (upper); a two-sided row gives both columns. Bound
  rows are flagged so that `d = Jᵀe_i` is read as row `i` of J (O(n) instead
  of O(n²)).
- **ADMM** consumes it as OSQP's `l ≤ Ax ≤ u`, with ±Inf replaced by
  ±1e30 (`OSQP_INFTY`), and treats a bound as infinite when its magnitude
  exceeds `OSQP_INFTY·OSQP_MIN_SCALING = 1e26` after scaling, exactly as
  v1.0.0 does.

Marginal signs. GI's λ ≥ 0 maps to `lower = +λ`, `upper = −λ`, `ineqlin = −λ`,
`eqlin = λ`. OSQP's y (P x + q + Aᵀy = 0, y > 0 upper-active, y < 0
lower-active) maps to marginal `= −y`. Both satisfy the stationarity identity
in the `QPResult` comment, which a test asserts on every fixture.

### 2.3 Method choice (`QPAuto`)

`QPActiveSet` when P is given dense, its Cholesky succeeds, no `ASparse` is
given and `n + m ≤ 2000`; otherwise `QPADMM`. The threshold is provisional
and is set from the perf measurements of task 13. The accuracies differ:
ADMM's defaults are OSQP's 1e-3. `QPResult.Method` says which ran, and the
README must say so in the first line describing `Quadprog`.

### 2.4 The existing NNLS/LSEI code

`slsqp_lsq.go` holds SciPy's `lsei → lsi → ldp → nnls` chain. With
P = LLᵀ, `½xᵀPx + qᵀx = ½‖Lᵀx + L⁻¹q‖² − const`, so `lsei` already solves a
strictly convex QP. It is **not** exposed and **not** used as the QP solver:

1. it carries no warm start and no factor reuse;
2. LDP via NNLS rebuilds its least-squares factors at each NNLS step, which
   is O(n³) per change where GI's updates are O(n²);
3. SLSQP must keep calling it unchanged, because SLSQP's path parity with
   SciPy depends on it.

It **is** used in tests as an independent witness: on every positive
definite fixture, `Cholesky + lsei` and GI must agree on x to 1e-9 (relative
∞-norm). The two are different algorithms on different factorisations, so
agreement is evidence that neither shares a bug with the other.

## 3. Active set: Goldfarb–Idnani

Notation and numbering follow Schmelzer & Stoll. G = P ≻ 0, a = −q, C holds
the canonical columns, the first `meq` being equalities.

### 3.1 State

- `J ∈ ℝ^{n×n}` dense row-major and `R` upper triangular k×k packed by
  columns (entry (i, j), i ≤ j, at offset j(j+1)/2 + i, Remark 6.4), with the
  invariants (10) `JᵀGJ = I` and (11) `JᵀA = [R; 0]` for the working-set
  normals A.
- The working set 𝒜 (canonical column indices, in insertion order), its
  multipliers u, the iterate x and the running objective f.

### 3.2 Algorithm (their Algorithm 1)

0. `R_G = chol(G)` (go-ndarray `linalg.Cholesky`, upper). `J ← R_G⁻¹`
   (triangular inverse, local helper). `x ← G⁻¹a = J(Jᵀa)`;
   `f ← −½ aᵀx`; 𝒜 ← ∅; k ← 0.
1. **Equalities first.** For each equality column in order, run steps 3–5
   with that column forced as the entering one (Darmstadt page: equalities
   are added one at a time and never leave). If an equality cannot enter
   because z = 0 (step 3) it is linearly dependent on the ones already held:
   if its residual `|s_i| ≤ FeasTol·(1 + |b_i|)` it is redundant and is
   skipped (recorded in the message); otherwise the equalities are
   inconsistent: `QPInfeasible` with certificate r.
2. **Select** (19): `s_i = c_iᵀx − b_i`; `v_i = |s_i|` for equalities not
   in 𝒜, `v_i = −s_i` for inequalities not in 𝒜 and not set aside (§3.4).
   Choose `i* = argmax v_i/‖c_i‖₂`, first index on ties. Stop with
   `QPOptimal` when `max v_i/‖c_i‖ ≤ FeasTol·(1 + |b_i|/‖c_i‖)` (§3.4). A
   column with `‖c_i‖ = 0` is skipped when `b_i ≤ 0` (vacuous) and is
   `QPInfeasible` when `b_i > FeasTol` (§5.1 of the note). `‖c_i‖` is
   computed once at setup.
3. **Directions** (15): `d = Jᵀn*`, `z = J₂d₂`, `r = R⁻¹d₁` (one packed
   triangular solve). By Lemma 4.1(ii), `n*ᵀz = ‖d₂‖²`; take z = 0 when
   `‖d₂‖₂ ≤ 1e3·ε·‖d‖₂` (ε = 2⁻⁵²).
4. **Step limits.** (23) `t₁ = min{u_i/r_i : i ∈ 𝒜 inequality, r_i > 0}`,
   `i_del` its argmin (the earliest-inserted on ties), +∞ if the set is
   empty. (24) `t₂ = |s_{i*}|/‖d₂‖²`, +∞ when z = 0. For an equality violated
   from above (`s > 0`, Remark 4.3) the step is taken along −z: t is
   negative, the ratio test runs on −r, and u* accumulates negatively.
5. **Step.** If both limits are infinite: infeasible (§3.5). Else
   `t = min(t₁, t₂)` (in magnitude); `x ← x + t z`;
   `f ← f + t(t/2 + u*) ‖d₂‖²` (25); `u ← u − t r`; `u* ← u* + t`.
   - **Full step** (`t₂ ≤ t₁`): insert i* (§3.3) with multiplier u*; go to 2.
   - **Partial step**: clamp `u_{i_del} ← 0`, delete i_del (§3.3), recompute
     `s_{i*} = n*ᵀx − b*` and go to 3 with the same i*.
6. After termination, recompute f from x directly and compare it with the
   running total: a relative difference above 1e-10·max(1, |f|) is a
   `QPNumericalFail` (it means the invariants drifted).

### 3.3 Factor updates

- **Insert** (26)–(29): one Householder reflection W₂ on the trailing n − k
  columns of J with `W₂ᵀd₂ = αe₁`. Use the cancellation-free form (29):
  `τ = ‖(d₂)_{2:}‖²`, `ν = √((d₂)₁² + τ)`,
  `v₁ = −sign((d₂)₁)·τ/(|(d₂)₁| + ν)`, `v_{2:} = (d₂)_{2:}`, `β = τ + v₁²`,
  `α = sign((d₂)₁)·ν`. Append column `[d₁; α]` to R. Proposition 6.1
  guarantees α ≠ 0 after a full step, so no rank test is needed.
- **Delete** (30): remove column p of R, shift the rest left, and chase the
  sub-diagonal for j = p … k−2 with symmetric 2×2 reflections
  `[[c, s], [s, −c]]`, `h = sign(x)·√(x² + y²)` (computed by `math.Hypot`),
  `c = x/h`, `s = y/h`, applied to rows j, j+1 of R and columns j, j+1 of J;
  when x = 0 use the plain exchange (Remark 6.3).

By Proposition 3.2 the iterates do not depend on these choices (Givens
versus reflection), only the rounding does.

### 3.4 Tolerances, degeneracy and cycling

- `FeasTol` (default 1e-9; MATLAB's active-set uses 1e-8) is the only
  user tolerance. The selection test is relative to each constraint's own
  scale, which keeps the walk invariant under row scaling (Prop. 4.1); a
  test asserts it (§6.3).
- **Set-aside rule** (Remark 5.2): when the iteration is stuck (z = 0 and
  t₁ = ∞) on a constraint whose violation is below `1e3·FeasTol·(1 + |b|/‖c‖)`,
  the violation is rounding, not infeasibility: mark i* as set aside, and
  clear all marks as soon as x moves. Infeasibility is declared only on a
  macroscopic violation.
- **Cycling.** Finite termination holds when every full step has t₂ > 0
  (Prop. 5.2). Zero-length full steps cannot occur, because a constraint is
  selected only when its violation exceeds FeasTol. Zero-length partial
  steps (u_i = 0 in 𝒜) can; ties are broken by the least-index rules in
  steps 2 and 4 (the classical remedy named in Remark 5.1), and the
  iteration counter (every insert and every delete counts as one) stops the
  walk at `MaxIter` with `QPIterLimit`. Linear dependence of the working set
  is unreachable (Prop. 6.1), so no rank test exists in the loop.
- **Bounded work.** Each pass is O(n²) (one `Jᵀn*` and one J update) plus
  O(nm) for the slacks; the slacks are updated incrementally,
  `s ← s + t·(Cᵀz)`, with a full recomputation every 50 steps and at the end
  to stop drift. Memory: J (n²), R (≤ n(n+1)/2), C (n·m dense or bound
  flags), all allocated once in `NewQPSolver`.

### 3.5 Infeasibility

Stuck with a macroscopic violation: z = 0 and `r_i ≤ 0` for every
inequality in 𝒜 is the hypothesis of Proposition 5.3, and r is a Farkas
certificate. `PrimalCert` holds r scattered to canonical rows (zeros
elsewhere, sign-adjusted for upper-side columns). Before reporting, verify
it: `rᵀb_𝒜 − b* < −FeasTol·(1 + |b*|)`; if the check fails, report
`QPNumericalFail` instead. GI never reports `QPUnbounded`: a positive
definite G makes the problem bounded whenever it is feasible.

### 3.6 Results

x, f, u (scattered and sign-mapped to the `LPDual` blocks, §2.2),
`ActiveSet` = 𝒜, `Nit` = number of inserts + deletes, and `Optimality` =
‖Px + q − Σλ_i c_i‖∞ recomputed from scratch, `ConstrViolation` = max
canonical-row violation. Optimality is checked against (34) of the note:
`Optimality ≤ 1e-8·max(1, ‖q‖∞, ‖Px‖∞)` or the status drops to
`QPNumericalFail`.

## 4. ADMM: OSQP

### 4.0 Fidelity target

The solver follows OSQP **v1.0.0** step by step, using `rho_is_vec = 1`, the
direct linear-system solver, and `adaptive_rho = ITERATIONS`. Two places
where v1.0.0 differs from the 2020 paper, and the code is followed:

1. the termination test also requires the **duality gap** (`check_dualgap`
   on by default);
2. ρ adaptation fires on a **fixed interval** of 50 iterations, not on the
   paper's 40%-of-setup-time rule (which v1.0.0 keeps as an option; timing
   rules make results depend on machine load and are not implemented).

The iterates cannot be bit-identical to OSQP: the fill-reducing ordering
(go-sparse AMD against SuiteSparse AMD in QDLDL) and the summation order
differ. The oracle criteria (§6) are set accordingly.

### 4.1 Setup

1. Canonical data (§2.2): `P` upper triangle in CSC, `A` in CSC, `l`, `u`.
2. **Scaling** (`scaling.c`, Algorithm 2), `ScalingIter = 10` passes, no
   convergence test:
   - `δ_D, δ_E` ← ∞-norms of the columns of `M = [P Aᵀ; A 0]` (P's full
     symmetric columns from the upper triangle);
   - limit each: `v < 1e-4 → 1`, `v > 1e4 → 1e4`; then `δ ← 1/√δ`;
   - `P ← D P D`, `A ← E A D`, `q ← D q`; `D ← D·δ_D`, `E ← E·δ_E`;
   - cost: `γ = mean_j ‖P_{:,j}‖∞` (= ‖col-norms‖₁/n), `‖q‖∞` limited as
     above, `γ ← max(γ, ‖q‖∞)` limited, `c_step = 1/γ`; `P ← c_step P`,
     `q ← c_step q`, `c ← c·c_step`;
   - then `l ← E l`, `u ← E u` (bounds beyond 1e26 stay infinite).
3. **ρ vector** (`set_rho_vec`): ρ clamped to [1e-6, 1e6]; per row:
   `ρ_i = 1e-6` (ρ_min) if `l_i < −1e26` and `u_i > 1e26` (loose), `1e3·ρ`
   if `u_i − l_i < 1e-4` (equality, `OSQP_RHO_TOL`), else ρ.
4. **KKT** (24): assemble the upper triangle of
   `K = [P + σI, Aᵀ; A, −diag(1/ρ)]` in CSC, order it once with
   `order.AMD`, compute the symbolic factorisation once, and the numeric
   LDLᵀ (§9, go-sparse). K is quasi-definite, so the factor exists for any
   P ⪰ 0 and any A; a zero pivot or a positive-pivot count different from n
   means P is not PSD: status `QPNonConvex` (QDLDL's return value gives the
   count).

### 4.2 Iteration (Algorithm 1 of the paper; `osqp_solve`)

For k = 1 … MaxIter (x, z, y from the warm start or zero):

1. swap (x, x_prev), (z, z_prev);
2. right-hand side: `x̃-part = σ x_prev − q`, `ν-part = z_prev − y ⊘ ρ`;
   solve K, giving `(x̃, ν)`; `z̃ = z_prev + (ν − y) ⊘ ρ`
   (in v1.0.0 the solver returns z̃ directly in `xz_tilde`; same formula);
3. `x = α x̃ + (1−α) x_prev`, `δx = x − x_prev`;
4. `z = Π_[l,u](α z̃ + (1−α) z_prev + y ⊘ ρ)`;
5. `δy = ρ ⊙ (α z̃ + (1−α) z_prev − z)`; `y ← y + δy`;
6. if `k % CheckInterval == 0`, or `k % AdaptiveRhoInterval == 0`, or
   k = 1: update residuals (§4.3);
7. if `k % CheckInterval == 0` and the termination test (§4.3) holds: stop;
8. if `k % AdaptiveRhoInterval == 0`: adapt ρ (§4.4).

After the loop: if the last iteration was not a check iteration, update and
check once; if still unsolved, check with all tolerances ×10 and report
`QPInaccurate` (or the inaccurate infeasibility statuses, which map to
`QPInfeasible`/`QPUnbounded` with `Message` saying "inaccurate") or
`QPIterLimit`. Then polish if requested and solved.

### 4.3 Residuals and termination (`check_termination`, unscaled)

With D, E, c from scaling (all 1 when scaling is off), `Ax`, `Px`, `Aᵀy`
evaluated on the scaled data:

- `r_prim = ‖E⁻¹(Ax − z)‖∞`, `ε_prim = ε_abs + ε_rel·max(‖E⁻¹z‖∞, ‖E⁻¹Ax‖∞)`;
- `r_dual = c⁻¹‖D⁻¹(Px + q + Aᵀy)‖∞`,
  `ε_dual = ε_abs + ε_rel·c⁻¹·max(‖D⁻¹q‖∞, ‖D⁻¹Aᵀy‖∞, ‖D⁻¹Px‖∞)`;
- duality gap: `SC(y) = uᵀ max(y,0) + lᵀ min(y,0)` after projecting a copy
  of y on the polar of the recession cone of [l,u] (`y_i ← 0` if both bounds
  infinite, `min(y_i,0)` if only u infinite, `max(y_i,0)` if only l
  infinite) and zeroing `|y_i| ≤ 1e-15`; `gap = c⁻¹(xᵀPx + qᵀx + SC)`,
  `ε_gap = ε_abs + ε_rel·c⁻¹·max(|xᵀPx|, |qᵀx|, |SC|)`.
- **Non-convex**: if `r_prim > 1e30` or `r_dual > 1e30`, stop with
  `QPNonConvex`.
- **Order** (exactly v1.0.0): m = 0 makes the primal check pass; else
  primal passes if `r_prim < ε_prim`, otherwise test primal infeasibility;
  dual passes if `r_dual < ε_dual`, otherwise test dual infeasibility; gap
  passes if `|gap| < ε_gap` (or the check is off). All three pass →
  `QPOptimal`; else primal-infeasible → `QPInfeasible`; else dual-infeasible
  → `QPUnbounded`.
- **Primal infeasibility** (`is_primal_infeasible`): project δy on the polar
  recession cone; `‖Eδy‖∞ > 1e-30`, and `uᵀmax(δy,0) + lᵀmin(δy,0) < 0`,
  and `‖D⁻¹Aᵀδy‖∞ < ε_pinf·‖Eδy‖∞`. Certificate: `E δy`.
- **Dual infeasibility** (`is_dual_infeasible`): `‖Dδx‖∞ > 1e-30`,
  `qᵀδx < 0`, `‖D⁻¹Pδx‖∞ < c·ε_dinf·‖Dδx‖∞`, and `E⁻¹Aδx` lies in the
  recession cone of [l,u] within `ε_dinf·‖Dδx‖∞` (component in
  [−ε, ε]·‖Dδx‖ for two finite bounds, ≥ −ε‖Dδx‖ when u = ∞, ≤ ε‖Dδx‖
  when l = −∞). Certificate: `D δx`.

`Optimality`, `ConstrViolation` and `DualityGap` report these unscaled
quantities.

### 4.4 ρ adaptation (`compute_rho_estimate`, `adapt_rho`)

On the scaled residuals,
`ρ_new = ρ·√((r_p/(max(‖z‖∞,‖Ax‖∞) + 1e-30)) / (r_d/(max(‖q‖∞,‖Aᵀy‖∞,‖Px‖∞) + 1e-30)))`,
clamped to [1e-6, 1e6]. Accept it only if `ρ_new > 5ρ` or `ρ_new < ρ/5`;
then rebuild the ρ vector (§4.1.3) and refactor **numerically** (same
pattern, same symbolic factor). `RhoUpdates` counts them.

### 4.5 Polishing (`polish.c`, paper §4)

On the unscaled-to-scaled final iterate: lower-active rows
`z_j − l_j < −y_j` or `l_j = u_j`; upper-active `u_j − z_j < y_j`. Solve the
regularised reduced system (30) with δ = 1e-6 (its own quasi-definite LDLᵀ),
then `PolishRefine = 3` passes of iterative refinement (31) on the
unregularised K. Accept the polished point when
`(r_p' < r_p and r_d' < r_d) or (r_p' < r_p and r_d < 1e-10) or
(r_d' < r_d and r_p < 1e-10)`; set `Polished`. No active row at all: polish
is skipped, not failed.

### 4.6 Bounded work

Per iteration: one K solve (two triangular sweeps and a diagonal scale),
two SpMVs only on check iterations, O(n + m) vector work. Memory: K and L
nnz computed by the symbolic pass before allocation, with the overflow checks
of §2.1. `MaxIter` caps time; there is no wall-clock limit (it would make
results load-dependent).

## 5. Warm start

| | Active set | ADMM |
|---|---|---|
| `X0`/`Y0` | ignored (as quadprog's interior-point ignores x0) | (x⁰, z⁰, y⁰) ← (X0, A X0, Y0) in scaled form (paper §6) |
| `WorkingSet` | rebuild J, R by inserting the listed columns in order (k insertions, O(kn²)) from the cold start, then repair (Prop. 8.2) | ignored |
| `QPSolver.UpdateQ` | keep J, R, 𝒜; recover x, u by (35) in O(n² + nk + k²); if every inequality multiplier is ≥ 0, resume at step 2; otherwise drop one negative-multiplier constraint at a time (Prop. 8.2) and resume | keep the factorisation; rescale q; keep x, z, y as the warm start |
| `UpdateBounds` | keep J, R; drop working-set columns whose b changed sign of slack, then repair | rescale; recompute the ρ vector; refactor numerically only if a row's type (loose/ineq/eq) changed (`update_rho_vec`) |
| `UpdateMatrixValues` | recompute everything (G changed) | rescale with the stored D, E, c is **not** valid — rerun scaling, then numeric refactorisation on the stored symbolic factor |

`Nit` counts the work actually done, so warm and cold counts are not
comparable (Remark 8.1). Warm starting never changes the answer beyond the
tolerances; a test asserts it (§6.3).

## 6. Test plan

### 6.1 Oracle harness: `tools/qporacle`

Same shape as `tools/slsqporacle`: its own `go.mod`, cobra
(`RootCmd()`, `SilenceUsage`, `RunE`, `osExit` variable, output through
`cmd.OutOrStdout()`), Go driving Python through `uv run -q --no-project
--with … python -I -` with the script embedded, and `--platform linux/amd64`
to run inside the pinned uv container, because the macOS arm64 wheels may
contract `a*b + c` into FMA. Pins: `osqp==1.1.3` (the harness records
`osqp.__version__` and the C core version it reports, and the test fails if
the core is not v1.0.0 — check this first, before generating anything),
`clarabel` (Apache-2.0, interior point: the high-accuracy reference),
`quadprog` (GPL: run only, never read), `numpy`, `scipy` (for `loadmat`
only).

Subcommands:

- `qporacle gen [--platform] -o testdata/qp/<set>.json` — runs the planted
  and hand problems through OSQP (defaults, and `eps_abs=eps_rel=1e-9` with
  polish), Clarabel (1e-10) and quadprog (positive definite only), and
  writes status, x, y, objective, iterations, ρ updates, residuals.
- `qporacle mm fetch` — downloads the MAT files of
  qpsolvers/maros_meszaros_qpbenchmark **at a pinned commit SHA** into
  `os.UserCacheDir()/qporacle/mm` (never into a repository), verifying a
  SHA-256 list committed in the tool; `--max-bytes` bounds each download.
- `qporacle mm convert --tier tiny|small|medium|all` — converts MAT to our
  JSON (P upper CSC, q, r, A CSC, l, u, with ±1e20 → ±Inf as the
  benchmark's loader does), records n, m, nnz, whether `cholesky(P)`
  succeeds (our definition of "positive definite"), and the oracle results.
  Tiny and small tiers are written to `testdata/qp/mm/`, the others to the
  cache directory.
- `qporacle diff a.json b.json` — lists the cases on which two fixtures
  disagree, exit status 1 if any (how wheels and platforms are compared).

The harness has its own tests in CI that need neither Python, Docker nor
network (fake runner, as `slsqporacle` does), and a fuzz target on its JSON
reader.

### 6.2 Maros–Mészáros subset

Sizes below are the MAT file sizes from the benchmark's `data/` listing
(bytes); n, m and nnz are recorded by `mm convert` into the fixture and the
fidelity table of `docs/design.md`, not quoted here.

| Tier | Problems | Where it runs |
|---|---|---|
| tiny (≤ 1.2 KB) | GENHS28 681, HS118 772, HS21 567, HS268 748, HS35 574, HS35MOD 588, HS51 616, HS52 610, HS53 591, HS76 619, LOTSCHD 787, QPTEST 588, S268 748, TAME 551, ZECEVIC2 551, QAFIRO 1116 | CI, every push, both methods (GI where P is positive definite) |
| small (≤ 6 KB) | CVXQP1_S 3386, CVXQP2_S 3182, CVXQP3_S 3548, DUALC1 5533, DUALC2 4672, DUALC5 5312, PRIMALC2 4889, PRIMALC5 5425, QADLITTL 3039, QPCBLEND 3748, QRECIPE 2701, QSC205 2531, QSCAGR7 2256, QSHARE1B 5700, QSHARE2B 3007, QSCTAP1 4966, VALUES 5789 | CI, ADMM; GI where positive definite |
| medium (≤ 40 KB) | the remaining files up to 40 KB (about 60) | `go test -run MarosMeszaros` with `QPORACLE_MM_DIR` set; not CI |
| all 138 | including BOYD1 (2.4 MB), EXDATA (18.8 MB), CONT-300, the AUG and LISWET families | perf harness only (task 13) |

The converted tiny and small fixtures are committed with the benchmark's
Apache-2.0 `NOTICE` alongside. The 19-problem "dense positive definite"
subset of qpbenchmark is reproduced by our own Cholesky criterion and its
membership is asserted equal to the benchmark's report, as a check that the
conversion is right.

### 6.3 Criteria

| Comparison | Criterion |
|---|---|
| GI vs Clarabel 1e-10 and vs quadprog | same status; ‖x − x_ref‖∞ ≤ 1e-8·max(1, ‖x_ref‖∞); \|f − f_ref\| ≤ 1e-10·max(1, \|f_ref\|); marginals to 1e-6 where the reference is nondegenerate; our own (34)-style KKT residual ≤ 1e-9 scaled |
| GI vs `Cholesky + lsei` (internal witness, §2.4) | x to 1e-9 on every positive definite fixture |
| ADMM vs OSQP, same settings | identical status; \|f − f_osqp\| ≤ 10·(ε_abs + ε_rel·\|f\|); `Nit` within one check interval (±25) on ≥ 90% of the cases, every exception listed in the fixture test with the reason (as `slsqp_fixture_test.go` does); our unscaled residuals meet our own tolerances |
| ADMM 1e-9 + polish vs Clarabel | f to 1e-7 relative, x to 1e-6 |
| every solved fixture | the stationarity identity with the `LPDual` blocks (§2.2) holds to 1e-8 scaled |

### 6.4 Planted positives and controls

Each detector is shown to fire on a case built to trigger it, and not to
fire on its twin:

- **Known optima**: unconstrained (x = −P⁻¹q); one active bound; one active
  equality; a vertex with n active constraints; a degenerate vertex with
  n + 1 active constraints (zero-length partial steps); an LP (P = 0, ADMM).
- **Infeasible**: `x ≥ 1, x ≤ 0` and an inconsistent pair of equalities —
  GI returns r and the test re-verifies Proposition 5.3 from r alone; ADMM
  returns δy and the test re-verifies (12). Twin: the same with `x ≤ 2`,
  which must solve.
- **Unbounded**: free variable with q < 0 and P = 0 on it (ADMM) — the test
  re-verifies (13) from δx. GI is asserted never to return `QPUnbounded`.
- **Non-convex**: P with a negative eigenvalue — ADMM `QPNonConvex`, GI
  `ErrInput`.
- **Linear dependence**: duplicated constraint columns (the note's §9.1
  family that breaks guessing methods) — GI must solve it (Prop. 6.1), with
  the walk's count recorded.
- **Scale invariance** (Prop. 4.1, Remark 4.1): scaling rows by
  γ ∈ {1e-3, 1e3} leaves GI's insert/delete sequence identical and x within
  1e-12.
- **Invariants**: a test-only hook checks `‖JᵀGJ − I‖_max ≤ 1e-12·n` and
  `‖JᵀA − [R;0]‖_max ≤ 1e-12·n·‖A‖` after every update on all fixtures; a
  control flips one sign in the packed R indexing and the hook must catch it
  (the note's §9 warns this error is otherwise silent).
- **Running objective**: the incremental f of (25) equals f(x) to 1e-12 on
  every fixture; a control drops the `u*` term and must be caught.
- **Warm start**: perturb q by 1%; warm `UpdateQ` gives the cold answer to
  the method's tolerance with fewer inserts (GI) or iterations (ADMM).
- **Limits**: `MaxIter = 1` returns `QPIterLimit` with the last x, never an
  error.

### 6.5 Fuzz targets (corpus committed under `testdata/fuzz/`)

- `FuzzQuadprogDense`: bytes → n ≤ 8, m ≤ 12, P = LLᵀ + δI (sometimes
  singular or indefinite), random constraints, injected NaN/±Inf/huge
  values. Properties: never panics; returns an error or a result; if
  `Success`, the unscaled KKT residuals meet the method's tolerance ×10;
  `Nit ≤ MaxIter`; GI and ADMM never disagree on feasibility when both
  succeed.
- `FuzzQuadprogSparse`: malformed `Indptr`/`Indices` (unsorted, duplicates,
  out of range, lower-triangle entries in `PSparse`) → error, never panic.
- `FuzzQPSolverUpdate`: a random sequence of `UpdateQ`/`UpdateBounds`/
  `WarmStart`/`Solve` calls with wrong lengths and NaN → error or a valid
  result.
- `qporacle`: `FuzzReadFixture` on the JSON reader (size limits, depth).

### 6.6 Coverage

100% of statements in every new file (`coverage_test.go` already enforces
it for the package). Unreachable branches are documented in the code, as
the SLSQP and trust-constr ports do, not left uncovered.

## 7. Performance plan

Reference: OSQP v1.0.0 through the pinned Python wheel, single thread by
construction (QDLDL is sequential). Measured with a `tools/qpbench`
command in the shape of `tools/slsqpbench`: setup time and solve time
separately, interleaved rounds (ours, OSQP, ours, …), median of 5 rounds,
one core (`GOMAXPROCS=1`), never with a 1-minute load above 8 (the tool
refuses and says so). Python call overhead is measured with an empty solve
and subtracted, and reported.

Sets: the Maros–Mészáros medium and full tiers; random dense QPs
n ∈ {10, 50, 200, 1000} for GI vs quadprog (the GPL Goldfarb–Idnani, run
only) and vs OSQP; the OSQP benchmark classes (random QP, equality QP,
portfolio, control, Lasso, Huber, SVM — open `osqp/osqp_benchmarks` first)
at three sizes for ADMM.

Target: no slower than OSQP on one core, per row, for setup and for solve;
a slower row is an open defect listed in `docs/perf.md`. GI against
quadprog: no slower from n = 50 up.

Kernels, in the order they matter:

1. the LDLᵀ solves (go-sparse, §9) — the ADMM hot loop;
2. CSC SpMV and SpMVᵀ, and the symmetric upper-triangle SpMV for P
   (go-sparse);
3. vector kernels for steps 3–5 of §4.2 and the residuals (go-simd/floats:
   `Axpy`, `Xpby` exist; the element-wise product, the 3-term combination
   `αa + βb + γc`, the box clamp `min(max(v, l), u)`, and the ∞-norm are to
   be added, §9);
4. GI: `Jᵀn*` and the rank-one Householder update of J (`Dot`, `Axpy` from
   go-simd/floats; bound columns read one row of J).

Determinism: like trust-constr, the QP solvers use go-simd kernels, so
results are deterministic per architecture but not bit-identical across
architectures. The tests assert tolerances, not bits.

## 8. Security and robustness checklist (audited in each PR)

- All sizes derived from the input are checked for overflow on 32-bit
  targets before allocation (§2.1); go-sparse's `int32` row indices bound
  `n + m` and nnz.
- NaN/Inf rejected at the boundary (§2.1); inside the solvers every
  division has a guarded denominator (1e-30 as OSQP's `OSQP_DIVISION_TOL`,
  the z = 0 test of §3.2 step 3).
- Every loop is bounded: `MaxIter`, the inner GI loop by k + 1 passes
  (Prop. 5.1), scaling by `ScalingIter`, refinement by `PolishRefine`.
- No user callback, so no re-entrancy; `QPSolver` is not safe for
  concurrent use and says so.
- The oracle tool downloads only from one pinned commit, checks SHA-256,
  bounds sizes, writes only under `os.UserCacheDir()`, and runs Python with
  `-I`.
- `govulncheck` already runs in CI; the new go-sparse dependency is covered
  by it.

## 9. Dependencies to add elsewhere

| Where | What | Why |
|---|---|---|
| **go-sparse/sparse** | a `factor` (or `ldl`) package: elimination tree and column counts on the upper triangle, symbolic factor, numeric LDLᵀ for **quasi-definite** matrices returning the number of positive pivots and failing on an exact zero pivot, `Solve`, `LSolve`, `LTSolve`, a permuted wrapper taking an `order.AMD` permutation, and numeric refactorisation on a stored symbolic factor. QDLDL v0.1.9 (Apache-2.0) is the reference to port; Davis's LDL is LGPL and must not be read. | ADMM KKT (24), polishing (30); go-sparse v0.1.0 has orderings and Krylov solvers but no factorisation |
| go-sparse/sparse | `MulVecSymUpper` (y = Px from the upper triangle only) | OSQP stores P upper-triangular; avoids storing both halves |
| **go-simd/floats** | element-wise `Mul(dst, a, b)`, `AxpBy3`-style 3-term combination, `Clamp(dst, v, lo, hi)`, `MaxAbs(v)` (∞-norm) — via go-asmgen like the existing kernels | ADMM steps 3–5 and residuals; useful beyond QP |
| go-ndarray (optional) | `linalg.SolveTriangular` / `ChoSolve` | GI's setup needs a triangular inverse and solves; a local helper in `lapack.go` suffices until then |

## 10. File layout

```
quadprog.go            API types, validation (§2.1), canonical rows (§2.2), method choice, result assembly
qp_solver.go           QPSolver, updates, warm start (§5)
qp_gi.go               Goldfarb–Idnani walk (§3.2–3.6)
qp_gi_factor.go        J/R insert (Householder) and delete (2×2 chase) (§3.3)
qp_admm.go             OSQP iteration (§4.2), ρ adaptation (§4.4)
qp_admm_scale.go       Ruiz scaling (§4.1.2)
qp_admm_check.go       residuals, gap, certificates, termination (§4.3)
qp_admm_polish.go      polishing (§4.5)
qp_kkt.go              KKT assembly and the go-sparse LDLᵀ wrapper
quadprog_test.go, qp_gi_test.go, qp_admm_test.go, qp_solver_test.go,
qp_fixture_test.go, qp_mm_test.go, qp_fuzz_test.go, qp_bench_test.go
testdata/qp/*.json, testdata/qp/mm/*.json (+ NOTICE), testdata/fuzz/Fuzz*/
tools/qporacle/        cobra harness (own go.mod), embedded Python, SHA-256 list
tools/qpbench/         perf harness (own go.mod)
docs/design-qp.md      this file; the fidelity results go to docs/design.md
```

## 11. Implementation tasks, in order (one PR each)

1. **go-sparse: quasi-definite sparse LDLᵀ** (etree, symbolic, numeric,
   solves, permuted wrapper, refactor; fuzz on malformed input; tests
   against dense reconstruction; minor release).
2. **go-sparse: symmetric upper-triangle SpMV** (with the release of 1 if
   convenient).
3. **go-simd/floats: Mul, 3-term combination, Clamp, MaxAbs** (go-asmgen
   kernels, all architectures, fuzz against the reference; minor release).
4. **optimize: `Quadprog` API, validation, canonical rows, result
   assembly** — with the unconstrained and equality-only cases solved
   directly (Cholesky / KKT), so the PR is useful and covered; other cases
   return `ErrMethod` until 6 and 8 land.
5. **tools/qporacle: harness skeleton** (cobra, gen, diff, fake-runner
   tests, the OSQP core-version check) and the hand/planted fixtures.
6. **optimize: Goldfarb–Idnani solver** (§3, invariants hook, certificates,
   lsei witness, fixtures vs Clarabel and quadprog).
7. **optimize: QPSolver for the active set** (`UpdateQ` recovery and repair,
   `WorkingSet` warm start).
8. **optimize: ADMM core** (KKT via task 1, fixed ρ, no scaling, §4.2–4.3
   termination and certificates, non-convex detection).
9. **optimize: ADMM scaling and adaptive ρ** (§4.1.2, §4.4; the OSQP
   default-settings fixtures become comparable here).
10. **optimize: ADMM polishing** (§4.5).
11. **optimize: QPSolver for ADMM** (vector and matrix-value updates, warm
    start).
12. **Maros–Mészáros fixtures**: `qporacle mm fetch/convert`, tiny and small
    tiers in CI, the fidelity table in `docs/design.md`.
13. **Performance**: `tools/qpbench`, `docs/perf.md` rows against OSQP and
    quadprog, fix every slower row (kernels to go-simd/go-sparse), settle
    the `QPAuto` threshold.
14. **Release v0.5.0** (minor: new feature): README section, org landing
    (landing kit) and docs site (hextra fork) updated and the published
    pages checked.
15. Later, separate slice: semidefinite P in the active-set method by outer
    proximal-point iterations (DAQP; open arXiv:2103.16236 first), and
    OSQP's indirect (PCG) linear-system solver for problems whose KKT factor
    does not fit.
