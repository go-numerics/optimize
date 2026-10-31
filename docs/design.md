# Design: algorithms, sources and fidelity

This package is an independent Go implementation of the parts of
`scipy.optimize` that scientific notebooks use most. A census of 339 notebooks
set the order: `minimize`, `minimize_scalar`, scalar roots, `root`/`fsolve`,
`least_squares`/`curve_fit`, `linprog`, `brute`, `check_grad` and
`approx_fprime`. SciPy, MINPACK and L-BFGS-B are cited below only as
references and as test oracles.

## Fidelity policy

Each routine is a port of a published algorithm. Wherever SciPy's
implementation of that algorithm is itself a reference, the port follows it
statement by statement, so that on the same problem it takes the same
iterates. The test suite replays SciPy 1.18.1 on fixtures that
`testdata/gen/*.py` generates. It checks solutions, objective values, status
codes and messages. It also compares iteration and evaluation counts; where
counts are not asserted equal, they are logged side by side.

Three things make floating-point agreement possible, or explain where it
stops:

1. **No fused multiply-add.** Python never fuses `a*b + c`. Go fuses it on
   arm64, ppc64le, s390x, riscv64 and loong64, which changes the last bit and,
   over many iterations, the path. Every product that feeds an addition in the
   ported code is wrapped in `float64()`. The Go specification makes that
   conversion a rounding point that forbids fusion. The wrapping was applied
   mechanically by a small `go/ast` rewriter. As a result, every algorithm
   gives bit-identical iterates on amd64 and arm64.
2. **The oracle must not fuse either.** SciPy's macOS arm64 wheel is built
   with Apple clang's default `-ffp-contract=on`, and its compiled kernels
   (MINPACK, L-BFGS-B, the C root finders) contain fused instructions. The
   MINPACK fixtures were therefore generated with the x86_64 wheel under
   Rosetta, which has no FMA. The L-BFGS-B port was checked against SciPy's
   own `lbfgsb.c`, compiled without contraction and linked to a sequential
   BLAS: per-iteration f values agree bit for bit on 11 problems, including
   513 iterations of Rosenbrock n=100. Against the wheel itself, Accelerate's
   blocked BLAS kernels round differently. With an analytic gradient this
   only shows on long runs: Rosenbrock n=100 takes 513 iterations here and
   514 in SciPy.
3. **libm.** Go's `math.Exp`, `math.Cos` and `math.Pow` can differ from the
   platform C library in the last ulp. On test functions built from + − × ÷
   and `sqrt` only, the paths are identical, and the tests assert that. On
   functions calling `exp`, `sin`, `cos` or `pow`, a last-ulp difference can
   move an iterate across a decision. There the tests assert accuracy only.

## Minimize

| Method | Algorithm | Reference followed |
|---|---|---|
| `NelderMead` | Nelder & Mead (1965) simplex; adaptive coefficients of Gao & Han (2012); bounds by reflection of the initial simplex and clipping | SciPy `_minimize_neldermead` (stable ordering, `maxfev` unwinding exactly where SciPy raises) |
| `Powell` | Powell (1964) conjugate directions with the Press et al. replacement test; Brent line searches, bounded line searches restricted to the box, `arctan` mapping for half-infinite ranges | SciPy `_minimize_powell`, `_linesearch_powell`, `_line_for_search` |
| `CG` | Polak–Ribière with Powell's β ≥ 0 (PR+), strong-Wolfe search plus the sufficient-descent condition on the next direction (Nocedal & Wright §5.2) | SciPy `_minimize_cg` |
| `BFGS` | Inverse-Hessian BFGS (Nocedal & Wright Alg. 6.1), strong-Wolfe line search | SciPy `_minimize_bfgs`; **deviation:** the update is applied in O(n²) as a rank-two correction instead of two dense n×n products (O(n³)). It is the same formula in exact arithmetic but rounds differently. It is the main reason BFGS is 10–20× faster at n=100. |
| `NewtonCG` | Truncated Newton (Nocedal & Wright Alg. 7.1): CG on the Newton system with forcing term min(0.5, √‖g‖₁), stopped on negative curvature | SciPy `_minimize_newtoncg`, including its three Hessian sources and the order of the two gradient calls in its finite-difference Hessian-vector product, which decides how often the cache hits |
| `LBFGSB` | L-BFGS-B 3.0 (Byrd, Lu, Nocedal & Zhu 1995; Zhu et al. 1997; Morales & Nocedal 2011): generalized Cauchy point with a breakpoint heap, compact limited-memory matrices, subspace minimisation with the 3.0 projection and backtrack, Moré–Thuente line search | SciPy 1.18.1's C translation `lbfgsb.c`, one subroutine per method (`cauchy`, `hpsolb`, `bmv`, `formt`, `formk`, `cmprlb`, `subsm`, `lnsrlb`, `matupd`, LINPACK `dpofa`/`dtrsl`) |

Iterate parity is asserted on the problems built from arithmetic only
(Rosenbrock n=2, Beale, Himmelblau): 36 of the 38 runs there take SciPy's
exact path. The two exceptions:

- BFGS with finite differences on Rosenbrock, because of the O(n²) update
  above.
- Newton-CG on Beale, which differs by one gradient call (52 against 53) with
  the same iterations and function evaluations.

Shared machinery:

- **Line searches** (`dcsrch.go`, `linesearch.go`). The Moré–Thuente search
  (MINPACK-2 `dcsrch`/`dcstep`, as in SciPy's `_dcsrch.py`) is used by
  BFGS, CG, Newton-CG and L-BFGS-B. Its fallback is the Nocedal–Wright
  bracketing/zoom search (SciPy's `scalar_search_wolfe2`, `_zoom`,
  `_cubicmin`, `_quadmin`).
- **Evaluation counting** (`scalarFunc`). It reproduces SciPy's
  `ScalarFunction`. There is a one-point cache, so f and ∇f are each computed
  and counted at most once per x. A finite-difference gradient reuses the
  cached f(x) and adds its evaluations to `Nfev`, and every gradient counts
  once in `Njev`.
- **Finite differences** (`numdiff.go`). This is SciPy's `approx_derivative`:
  2-point and 3-point schemes, relative or absolute steps with the same
  defaults (√eps·max(1,|x|), or SciPy's absolute 1.49e-8 for BFGS/CG and
  1e-8 for L-BFGS-B), and steps adjusted to stay inside bounds, going
  one-sided when needed.
- **Wrapper semantics.** These come from `scipy.optimize.minimize`: the `tol`
  mapping to each method's tolerance, clipping x0 into the bounds, removing
  variables fixed by equal bounds for L-BFGS-B with finite differences, and
  status 99 when the callback stops the run.

## SLSQP (constrained minimisation)

`Minimize(…, SLSQP, …)` is Dieter Kraft's sequential least-squares quadratic
programming method (DFVLR-FB 88-28, 1988), as rewritten in C by SciPy. Sources
opened for the port, all BSD-3-Clause or BSD-style:

- SciPy **v1.18.1** `scipy/optimize/src/slsqp.c` (`__slsqp_body`, `lsq`,
  `lsei`, `lsi`, `ldp`, `ldl_update`) and `src/nnls.c` (Lawson–Hanson NNLS),
  followed statement by statement, including the reverse-communication state
  machine (modes 0/1/−1), the augmented problem for inconsistent
  linearisations (weight 100, ×10 up to five times), the five BFGS resets and
  the relaxed test with `tol = 10·acc`;
- SciPy v1.18.1 `_slsqp_py.py`, `_slsqpmodule.c`, `_minimize.py` and
  `_constraints.py` for the driver: evaluation order and counting, clipping
  the line-search point into the bounds, the removal of fixed variables, the
  all-fixed shortcut and `new_constraint_to_old` for `NonlinearConstraint`
  and `LinearConstraint`;
- Reference-LAPACK **v3.12.1** `dlarfg`, `dlarfgp`, `dlartgp`, `dlapy2`,
  `dgerq2`, `dormr2` (and the reference BLAS loop orders) for the dense
  kernels in `lapack.go`.

Kraft's 1988 report itself was not available to us; the method is cited
through SciPy's code, which references its §2.2.3.

**Deviations.** `dgelsy` (least squares with only equality constraints) is
replaced by Householder QR with column pivoting and the rank test
|R(k,k)| > √ε·|R(0,0)|; SLSQP only reads the rank, and on these
problems the matrix is a factor of the positive-definite BFGS matrix. Three
LAPACK branches that cannot be reached here are omitted and documented in
the code (the subnormal-τ flush of `dlarfgp`, the sign flip of `dlartgp`,
`ldl_update` with σ = 0). Where SciPy's NNLS would index out of range on an
empty passive set, the loop stops. The go-simd/floats kernels are not used:
they reduce in a lane-blocked FMA order that would change the path both
against SciPy and between architectures, on vectors of a few dozen entries.

**Constraint types.** `Options.Constraints` takes `Constraint` (SciPy's
dictionary form), `NonlinearConstraint` (lb ≤ f(x) ≤ ub) and
`LinearConstraint` (lb ≤ A·x ≤ ub). All three reduce to equality and
inequality blocks (`stdConstraint`), the form later constrained methods will
share. Unlike SciPy, which warns, other methods return `ErrMethod` when
constraints are given.

**Fidelity.** Fixtures come from SciPy's **x86_64** wheel run in a
`linux/amd64` container (`tools/slsqporacle --platform linux/amd64`), because
SciPy's macOS arm64 wheel contracts `a*b + c` into fused multiply-adds. On
100 runs (36 Hock–Schittkowski problems and 10 stress cases, each with and
without derivatives, plus option and new-style-constraint variants): every
status agrees, **88 take SciPy's exact path** (identical nit, nfev, njev,
with x to 1e-7 and f to 1e-9 relative), and 96 reach the same objective to
1e-6. The 12 others are among the runs listed in `slsqp_fixture_test.go`
(which asserts the path on all remaining ones); on each of them
SciPy's own arm64 and x86_64 wheels also disagree with each other, so the
path is decided by last-bit rounding (HS13 is degenerate; HS23, HS29, HS38
run near-singular updates; Infeasible, SingularEq, SingularEq3 hinge on a
pivot that is zero in exact arithmetic). Go results are bit-identical on
amd64 and arm64 (checked by running the amd64 test binary in a container).
`slsqporacle diff a.json b.json` lists the cases on which two fixtures
disagree (exit status 1 if any), which is how the wheels were compared; the
harness and `tools/slsqpbench` have their own tests, run in CI, which need
neither Python, Docker nor a quiet machine.

## trust-constr (interior point)

`Minimize(…, TrustConstr, …)` is SciPy's `trust-constr`: the Byrd–Omojokun
trust-region SQP method for equality-constrained problems (Lalee, Nocedal and
Plantenga 1998), and, as soon as there is an inequality or a bound, the
barrier (interior-point) method of Byrd, Hribar and Nocedal (1999) around it.
Sources opened for the port, BSD-3-Clause:

- SciPy **v1.18.1** `scipy/optimize/_trustregion_constr/`:
  `minimize_trustregion_constr.py` (driver, state, stop tests, statuses 0–4,
  bounds widened by one ulp), `tr_interior_point.py` (barrier subproblem,
  slack scaling, τ = 0.995, μ and tolerance ×0.2, trust radius ×5),
  `equality_constrained_sqp.py` (normal and tangential steps, merit penalty,
  second-order correction, radius updates), `qp_subproblem.py`
  (`modified_dogleg`, `projected_cg`, box and sphere intersections),
  `projections.py` (null-space, least-squares and row-space operators by QR
  with column pivoting, SVD when the Jacobian is singular),
  `canonical_constraint.py` (eq/less/greater/interval classification);
- SciPy v1.18.1 `_hessian_update_strategy.py` (BFGS, `skip_update`,
  `init_scale='auto'`) and `_differentiable_functions.py` (ScalarFunction and
  VectorFunction): when gradients and BFGS updates happen — at **every** new
  point, rejected trial points included, and again for a constraint when its
  multipliers change — which is what makes the evaluation counts comparable;
- SciPy v1.18.1 `_constraints.py` and `_minimize.py` (dictionary constraints
  become `NonlinearConstraint(fun, 0, 0 or inf)` with a BFGS Hessian each; no
  clipping of x0; `tol` sets gtol, xtol and barrier_tol);
- Reference-LAPACK v3.12.1 `dlaqp2` (pivoted QR with norm downdating).

The two papers themselves were not available to us; they are cited through
SciPy's code.

**Kernels.** The projections, CG and BFGS products use go-simd/floats
(`Dot`, `Axpy`, `Xpby`, `Scale`). That is allowed here because SciPy's own
iterates depend on its BLAS and on SuperLU (bounds-only problems use the
sparse augmented system), so bit-identity is out of reach anyway; it means
trust-constr's results are not bit-identical across architectures, unlike
SLSQP's. The projections always use dense QR with column pivoting.

**Fidelity.** `tools/slsqporacle --method trust-constr --platform
linux/amd64` replays the 100 SLSQP runs with `trust-constr`. Success or
failure agrees with SciPy on all 100 (two raise in both: more equalities than
variables, an `ErrInput` here), the status itself on 95 (the others are `gtol`
against `xtol`, both successes), and every successful objective is within
1e-5 of SciPy's or 1e-4 of the known optimum. 59 runs take SciPy's exact path
on arm64 and 68 on amd64. SciPy's own arm64 and x86_64 wheels disagree with
each other on 35 runs (`slsqporacle diff`), almost all with finite-difference
derivatives; counts are asserted on every other run except five listed in
`trustconstr_fixture_test.go` (HS21/jac, HS35/jac, HS48/fd and the
rank-deficient SingularEq3), on amd64 and arm64.

**Omitted.** `keep_feasible` (strict bounds), sparse Jacobians and the
NormalEquation/AugmentedSystem projections, the SR1 strategy and
user-supplied constraint Hessians, `verbose` reports. Branches of SciPy's
code that cannot be reached with these options (an infinite trust radius,
the infeasible-iteration counter of projected CG) are documented in the code.

## MinimizeScalar, Bracket, Fminbound

These are Brent's (1973) parabolic/golden method, the golden-section search,
the bounded variant of Forsythe, Malcolm & Moler's FMIN (fminbound), and the
downhill bracket search. All four are ported from SciPy's Python
implementations (`Brent`, `_minimize_scalar_golden`,
`_minimize_scalar_bounded`, `bracket`), messages included. A failed bracket
search returns the best of its three points, as SciPy's
`_recover_from_bracket_error` does. On arithmetic-only test functions the
iterates are identical to SciPy's. One subtlety: SciPy computes the golden
constant 1 − 0.61803399 at run time in float64, while a Go constant
expression would be exact and round differently.

## Scalar roots

- **`Brentq`, `Brenth`, `Ridder`, `Bisect`** follow SciPy's C
  implementations (`scipy/optimize/Zeros/*.c`, by Charles Harris, BSD
  licence): Brent with inverse quadratic or hyperbolic extrapolation (Bus &
  Dekker 1975), Ridders (1979) with SciPy 1.18's underflow-safe update, and
  bisection. Defaults are xtol 2e-12, rtol 4·eps and 100 iterations.
  - C's `MIN` macro returns b when a is NaN, unlike `math.Min`. Ridder relies
    on this after an infinite midpoint value, so it is reproduced (`cmin`).
  - A NaN function value stops the search with an error, as SciPy raises.
    Through `RootScalar` it becomes a result instead, as SciPy's root_scalar
    does.
- **`Newton`** covers Newton–Raphson, Halley and the secant method, ported
  from SciPy's `newton` (scalar path).
- **`RootScalar`** selects the method by SciPy's rules.

## Root / Fsolve (MINPACK hybrd/hybrj)

Powell's hybrid method, ported from SciPy 1.18.1's C translation of MINPACK
(Moré, Garbow & Hillstrom 1980):

- **Kernels:** `dogleg`, `qrfac` with column pivoting, `qform`, `r1updt`,
  `r1mpyq`, and the forward-difference `fdjac1`.
- **Layout:** storage is column-major, as in MINPACK, so every sum keeps the
  reference order.
- **Norm:** MINPACK's three-accumulator `enorm` is kept, because a different
  norm alone breaks bit parity.
- **Defaults:** as SciPy's `_root_hybr`: maxfev 200(n+1) without a Jacobian
  and 100(n+1) with one, factor 100, xtol 1.49012e-8. Messages are SciPy's.
- **Deviation:** `Nfev` excludes the two extra calls SciPy's Python wrappers
  make (a shape check and an initial evaluation). SciPy's reported nfev is
  always ours + 2.
- **Not ported:** `band` and `col_deriv`.

## Least squares

### `Leastsq` and `LeastSquares(LM)`: MINPACK lmdif/lmder

The Levenberg–Marquardt method with Moré's (1978) trust region, ported from
the same C MINPACK: `lmpar`, `qrsolv`, `qrfac`, `fdjac2`.

- **`Leastsq`** mirrors `scipy.optimize.leastsq`. `cov_x` is computed as
  SciPy does: R is inverted in the style of LAPACK `dtrti2`, the rows are
  permuted, and the result is invR·invRᵀ.
- **`LeastSquares` with method LM** follows `least_squares`'
  `call_minpack`:
  - It always calls lmder, using `approx_derivative` for a missing
    Jacobian.
  - The default budget is max_nfev = 100·n.
  - `x_scale='jac'` maps to MINPACK's mode 1; an explicit scale gives
    diag = 1/x_scale.
  - Statuses are mapped to least_squares' codes.

### `LeastSquares(TRF)`

The trust-region reflective method of Branch, Coleman & Li (1999), ported
from SciPy's `_lsq/trf.py` and `_lsq/common.py` (dense, `tr_solver='exact'`):

- **Bounds:** Coleman–Li scaling, and iterates kept strictly feasible.
- **Step choice:** the best of three candidates: the trust-region step
  clipped at the first bound and stepped back by θ, its reflection off that
  bound, and the constrained Cauchy step.
- **Subproblem:** solved exactly from a thin SVD, with Moré's safeguarded
  Newton iteration for the Levenberg–Marquardt parameter.
- **Loss and scaling:** the robust losses (soft_l1, huber, cauchy, arctan)
  rescale J and f as SciPy does, and `x_scale='jac'` scaling never decreases.
- **Counting:** `Nfev` excludes finite-difference evaluations, as in
  least_squares.
- **SVD:** it comes from go-ndarray's `linalg.SVD`, the Golub–Kahan/QR
  algorithm of LAPACK gesvd. A faster one-sided Jacobi SVD was measured and
  rejected: on exactly rank-deficient Jacobians it returns exact zeros where
  LAPACK returns tiny values, and SciPy's path depends on those tiny values.
- **Tall matrices** (m ≥ 1000 and m ≥ 11n/6, the regime where LAPACK's
  gesdd also factors QR first): the Jacobian is reduced by Householder QR
  and only the n×n R is decomposed. Uᵀf is obtained as U_Rᵀ(Qᵀf), with no
  m×n U ever formed. The fixtures, all smaller than 1000 rows, take the
  direct path.

### `CurveFit` / `CurveFitModel`

`scipy.optimize.curve_fit`:

- **Method:** LM through `Leastsq` without bounds, TRF through
  `LeastSquares` with bounds.
- **`sigma`:** a 1-D array divides the residuals; a 2-D covariance whitens
  them with its Cholesky factor (go-ndarray `linalg.Cholesky`, then forward
  substitution).
- **Covariance:**
  - With LM it is `cov_x`.
  - With TRF it is the pseudo-inverse of JᵀJ from the SVD of J. Singular
    values below eps·max(m,n)·s₀ are discarded. For a tall J the SVD is
    taken of R, since JᵀJ = RᵀR.
  - It is scaled by the reduced χ² unless `AbsoluteSigma` is set.
  - It is +Inf (`CovOK` false) when it cannot be estimated.
- **Errors:** non-convergence returns `ErrNotConverged` together with the
  partial result, where SciPy raises RuntimeError.
- **Parallelism:** `Workers` evaluates the model over the samples in
  parallel, from 2048 samples up. The result is bit-identical to the serial
  run.

## Linprog

HiGHS, SciPy's default, is out of reach. Two dense solvers are provided
instead, with documented limits.

- **Revised simplex** (default):
  - Two phases with bounded variables. Phase one starts from slack and
    artificial variables; phase two fixes the artificials at zero, so any
    left in the basis stand for redundant rows.
  - Dantzig pricing, switching to Bland's rule after 50 degenerate pivots.
  - Harris's two-pass ratio test with bound flips.
  - Dense LU with product-form updates, refactorised every 64 updates.
- **Interior point**: the homogeneous self-dual method of Andersen &
  Andersen (2000) with Mehrotra predictor–corrector, ported from SciPy's
  legacy `_linprog_ip.py` with its standard form and settings. The normal
  equations are solved by Cholesky, then LU, then least squares, in that
  fallback order.
- **Presolve:** after SciPy's `_presolve`: it removes empty rows and columns
  and fixed variables, and detects linearly dependent equality rows.
- **Duals:** they follow HiGHS's sign convention.
- **Final check:** SciPy's `_check_result` downgrades a status that is
  inconsistent with the returned point.
- **Not done:** singleton-row presolve, crossover, scaling, warm start,
  sparse matrices.

## Brute, CheckGrad, ApproxFprime

- **`Brute`** evaluates `numpy.mgrid`'s grid with the same point formula,
  takes the first minimum as `argmin` does (NaN counts as the minimum), and
  polishes it with Nelder–Mead, as `fmin` does. `Workers` evaluates the grid
  in parallel.
- **`CheckGrad`** and **`ApproxFprime`** are `check_grad` and
  `approx_fprime`, with SciPy's default absolute step of 1.49e-8.

## API conventions

- **Callbacks** take the point first and write their outputs into the slices
  that follow: `Func(x)`, `Grad(x, g)`, `FuncGrad(x, g)`, `Hess(x, h)`,
  `Jac(x, J)`. Matrices are row-major `[]float64`.
- **ndarray:** `*ndarray.Array` appears where arrays are natural (curve-fit
  data, `sigma`, LP constraint matrices), and results have `…Array()`
  accessors.
- **Options** are structs whose zero values select SciPy's defaults.
- **Errors:** invalid input returns an error wrapping `ErrInput`,
  `ErrDimension`, `ErrBounds` or `ErrMethod`. Non-convergence is reported in
  `Success`/`Status`/`Message`, as SciPy reports it, not as an error.
