# Coverage against SciPy and MATLAB

Inventory of what `go-numerics/optimize` (main, after v0.5.0) implements
against `scipy.optimize` v1.18.1 and the MATLAB Optimization and Global
Optimization Toolboxes. Status: **done** (same semantics and options that
matter), **partial** (what is missing is named), **missing**. "Dep" is what a
missing item needs beyond this package.

Sources. SciPy: `scipy/optimize/__init__.py` at tag v1.18.1 (the function and
method list below follows it line by line). MATLAB: the MathWorks function
lists could not be read by the tool used (the pages are rendered client-side),
so the MATLAB column is built from the toolbox's documented function families
and solver names and must be re-checked against the MathWorks reference lists
before a slice is started. Go side: README "Not done", `go doc`, and the
sources on main.

## Totals

| | done | partial | missing | rows |
|---|---|---|---|---|
| SciPy v1.18.1 | 27 | 14 | 44 | 85 |
| MATLAB (both toolboxes) | 5 | 8 | 17 | 30 |

## Local minimisation

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 1 | `minimize_scalar` brent | `fminbnd` (parabolic+golden) | `MinimizeScalar(Brent)` | done | |
| 2 | `minimize_scalar` bounded, `fminbound` | `fminbnd` | `MinimizeScalar(Bounded)`, `Fminbound` | done | |
| 3 | `minimize_scalar` golden | | `MinimizeScalar(Golden)` | done | |
| 4 | `minimize` Nelder-Mead | `fminsearch` | `Minimize(NelderMead)` | done | |
| 5 | `minimize` Powell | | `Minimize(Powell)` | done | |
| 6 | `minimize` CG | | `Minimize(CG)` | done | |
| 7 | `minimize` BFGS | `fminunc` quasi-newton | `Minimize(BFGS)` | done | |
| 8 | `minimize` Newton-CG | `fminunc` trust-region (Hess) | `Minimize(NewtonCG)` | done | |
| 9 | `minimize` L-BFGS-B | `fmincon` bounds | `Minimize(LBFGSB)` | partial | `hess_inv` operator (`LbfgsInvHessProduct`) not returned |
| 10 | `minimize` TNC, `fmin_tnc` | | | missing | none (port of TNC C code) |
| 11 | `minimize` COBYLA, `fmin_cobyla` | | | missing | none (derivative-free, constraints) |
| 12 | `minimize` COBYQA | | | missing | none (model-based, needs a small QP/trust-region solver; `Quadprog` can serve) |
| 13 | `minimize` SLSQP | `fmincon` sqp | `Minimize(SLSQP)` | done | |
| 14 | `minimize` trust-constr | `fmincon` interior-point | `Minimize(TrustConstr)` | partial | `keep_feasible`, sparse Jacobians, SR1 Hessian, constraint Hessians (`hess` of `NonlinearConstraint`); dep for sparse: go-sparse |
| 15 | `minimize` dogleg | `fminunc` / `fsolve` trust-region | | missing | none |
| 16 | `minimize` trust-ncg | | | missing | none |
| 17 | `minimize` trust-krylov | | | missing | Lanczos / GLTR (go-sparse or own) |
| 18 | `minimize` trust-exact | | | missing | dense Cholesky (have it) |
| 19 | `NonlinearConstraint` | `fmincon` nonlcon | `NonlinearConstraint` | partial | `keep_feasible`, `hess`, finite-difference `rel_step` forms |
| 20 | `LinearConstraint` | `fmincon` A,b,Aeq,beq | `LinearConstraint` | partial | `keep_feasible`; sparse A |
| 21 | `Bounds` | lb, ub | `Bound`, `Unbounded` | partial | `keep_feasible` |
| 22 | `BFGS` strategy | | internal | done | |
| 23 | `SR1` strategy | | | missing | none |
| 24 | `HessianUpdateStrategy` interface | | | missing | none (public interface) |
| 25 | `LbfgsInvHessProduct` | | | missing | linear-operator type (go-ndarray or own) |
| 26 | | `fmincon` active-set, trust-region-reflective, sqp-legacy | | missing | none (active-set is the `lsei` chain of SLSQP; TRR needs sparse for large scale) |
| 27 | | `fseminf` | | missing | SQP + semi-infinite constraint sampling |
| 28 | | `fminimax`, `fgoalattain` | | missing | SQP (have it) |
| 29 | `check_grad`, `approx_fprime` | `checkGradients` | `CheckGrad`, `ApproxFprime` | done / partial | Hessian and Jacobian checks (MATLAB) |
| 30 | `line_search`, `bracket` | | `Bracket`; line search internal (`dcsrch`) | partial | export `LineSearch` (strong Wolfe) |
| 31 | `rosen`, `rosen_der`, `rosen_hess`, `rosen_hess_prod` | | tests only | missing | none (trivial) |
| 32 | `fmin`, `fmin_powell`, `fmin_cg`, `fmin_bfgs`, `fmin_ncg`, `fmin_l_bfgs_b`, `fmin_slsqp` | | `Minimize` | partial | legacy wrappers not provided (methods are) |

## Global optimisation

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 33 | `basinhopping` | | `Basinhopping` | done | |
| 34 | `brute` | | `Brute` | done | |
| 35 | `differential_evolution` | `ga` (nearest) | `DifferentialEvolution` | partial | `init='sobol'`/`'halton'`: QMC engine |
| 36 | `direct` | `patternsearch` (nearest) | `Direct` | partial | `maxfun=0`, `eps=0`; status -6 on candidate-list overflow instead of SciPy's out-of-bounds write |
| 37 | `dual_annealing` | `simulannealbnd` | | missing | none (SciPy uses GSA + local search, port exactly) |
| 38 | `shgo` | | | missing | Delaunay triangulation, QMC (sobol/halton) |
| 39 | | `MultiStart`, `createOptimProblem` | | missing | none (runs `Minimize` from start points; start-point sets need QMC for parity) |
| 40 | | `GlobalSearch` | | missing | scatter search + `fmincon` |
| 41 | | `ga` (integer, mixed, constraints) | | missing | none |
| 42 | | `gamultiobj`, `paretosearch` | | missing | none |
| 43 | | `particleswarm` | | missing | none |
| 44 | | `patternsearch` (GPS/GSS/MADS) | | missing | none |
| 45 | | `surrogateopt` | | missing | RBF interpolation, QMC |

## Least squares and curve fitting

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 46 | `least_squares` (trf, lm; losses; bounds) | `lsqnonlin` | `LeastSquares(TRF, LM)` | partial | `method='dogbox'`, `tr_solver='lsmr'`, sparse Jacobians (`jac_sparsity`), `x_scale='jac'` check; dep: go-sparse + LSMR |
| 47 | `leastsq` | | `Leastsq` | done | `band`/`col_deriv` |
| 48 | `curve_fit` | `lsqcurvefit` | `CurveFit`, `CurveFitModel` | partial | `method='dogbox'`, `nan_policy`, `full_output` extras |
| 49 | `nnls` | `lsqnonneg` | | missing | none (Lawson-Hanson) |
| 50 | `lsq_linear` | `lsqlin` | | missing | BVLS + TRF (have TRF), LSMR for sparse |
| 51 | `isotonic_regression` | | | missing | none (PAVA) |

## Root finding

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 52 | `brentq`, `brenth`, `ridder`, `bisect` | `fzero` | `Brentq`, `Brenth`, `Ridder`, `Bisect` | done | MATLAB `fzero` start-point bracket search not provided |
| 53 | `newton` (secant, Halley), `root_scalar` | | `Newton`, `RootScalar` | partial | `root_scalar` method `toms748`; array / complex forms of `newton` |
| 54 | `toms748` | | | missing | none |
| 55 | `fixed_point` | | | missing | none |
| 56 | `RootResults` | | `RootScalarResult` | done | |
| 57 | `root` hybr, `fsolve` | `fsolve` trust-region-dogleg | `Root`, `Fsolve` | partial | `band`, `col_deriv`; MATLAB `trust-region` (sparse) and `levenberg-marquardt` options |
| 58 | `root` lm | `fsolve` levenberg-marquardt | (`LeastSquares(LM)` is the engine) | missing | wire `root(method='lm')` to MINPACK lmdif |
| 59 | `root` broyden1, broyden2, `broyden1/2` | | | missing | none |
| 60 | `root` anderson | | | missing | none |
| 61 | `root` linearmixing, diagbroyden, excitingmixing | | | missing | none |
| 62 | `root` krylov, `newton_krylov`, `KrylovJacobian` | | | missing | GMRES / LGMRES / MINRES (iterative solvers, go-sparse or new) |
| 63 | `root` df-sane | | | missing | none |
| 64 | `optimize.elementwise`: `find_root`, `find_minimum`, `bracket_root`, `bracket_minimum` | | | missing | vectorised (array-API) form; scalar kernels exist (`Brentq`, `Bracket`) |

## Linear, quadratic, conic, integer programming; assignment

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 65 | `linprog` revised-simplex | `linprog` dual-simplex-legacy | `Linprog(LPSimplex)` | done | |
| 66 | `linprog` interior-point | `linprog` interior-point | `Linprog(LPInteriorPoint)` | done | |
| 67 | `linprog` (front end: A_ub/A_eq/bounds, duals) | `linprog` | `Linprog` | partial | sparse matrices, presolve beyond the basics, crossover, HiGHS-class speed |
| 68 | `linprog` simplex (legacy tableau) | | | missing | none (removed upstream in later releases) |
| 69 | `linprog` highs, highs-ds, highs-ipm | `linprog` dual-simplex-highs | | missing | HiGHS-class dual simplex + sparse LU (go-sparse) |
| 70 | `milp` | `intlinprog` | | missing | branch and bound over `Linprog`, cuts, presolve |
| 71 | `linprog_verbose_callback` | | | missing | none (trivial) |
| 72 | | `quadprog` interior-point-convex | `Quadprog`, `QPAuto`, `QPADMM` | partial | PSD P in the active set; OSQP PCG solver; trust-region-reflective option |
| 73 | | `optimwarmstart` | `QPSolver` (`UpdateQ`, `WarmStart`) | partial | LP / general warm-start object |
| 74 | | `coneprog` | | missing | SOCP interior point (extend `linprog_ip`) or ADMM |
| 75 | | `optimoptions`, `optimset`, `optimget` | `Options`, `LSOptions`, ... | done | |
| 76 | | problem-based (`optimvar`, `optimproblem`, `solve`, `prob2struct`) | | missing | expression graph + autodiff; outside the numerics core |
| 77 | `linear_sum_assignment` | | | missing | none (Jonker-Volgenant) |
| 78 | `quadratic_assignment` faq, 2opt | | | missing | LSA (row 77) |

The totals count individual items: 85 SciPy entries of `__init__.py` (each
`minimize`, `root` and `linprog` method, each function and class; legacy
`fmin_*` folded into one entry) and 30 MATLAB functions or families (21
Optimization Toolbox, 9 Global Optimization Toolbox). Table rows group some of
them, so the row count differs.

## Proposed slices

1. **Roots II** (rows 54, 55, 58-61, 63): `toms748`, `fixed_point`, `root`
   lm / broyden / anderson / mixing / df-sane. No dependency.
2. **Trust-region family** (15, 16, 18, 23, 24, 14 partial): dogleg,
   trust-ncg, trust-exact, SR1, `keep_feasible`, constraint Hessians. No
   dependency. trust-krylov (17) and sparse Jacobians need go-sparse plus a
   Lanczos/GLTR solver.
3. **Derivative-free** (10-12): TNC, COBYLA, COBYQA. No dependency.
4. **Linear least squares** (49-51, 46 dogbox): `nnls`, `lsq_linear`,
   `isotonic_regression`, dogbox. No dependency; LSMR for the sparse path.
5. **Global III** (37, 39, 41-44): `dual_annealing`, MultiStart, `ga`,
   `particleswarm`, `patternsearch`, `paretosearch`. No dependency.
6. **QMC + shgo** (35 sobol/halton, 38, 45): QMC engine (Sobol, Halton),
   Delaunay triangulation, RBF interpolation for `surrogateopt`.
7. **Mixed-integer and conic** (69-71, 74): `milp`, HiGHS-class simplex
   (sparse LU), SOCP.
8. **Krylov roots** (62): GMRES/LGMRES iterative solvers.
9. **Assignment** (77, 78): LSA, then QAP.
10. **MATLAB SQP derivatives** (26-28): `fminimax`, `fgoalattain`,
    `fseminf`, `fmincon` active-set variants, on top of SLSQP.
11. **Utilities** (9, 25, 30-32, 64): exported line search, `LbfgsInvHessProduct`,
    `rosen*`, legacy `fmin_*` wrappers, elementwise forms.
