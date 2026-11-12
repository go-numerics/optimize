# Roadmap: what other reference libraries do better

This file lists what the reference optimisation libraries of Julia, R and
GSL do better than this package (v0.8.0). The SciPy and MATLAB gaps are
already listed in [coverage.md](coverage.md) and are not repeated here. The
point of this list is to bring each improvement into go-numerics, so that every
binding gets it: Ruby (rbgo), Julia, R and C. Those libraries are sources of
algorithms, defaults, test problems and edge cases, not only APIs to imitate.

Inventory taken 2026-11-11 from source. Every reference below was opened at
the path cited:

- **Julia (MIT):** Optim, LineSearches, Roots, LsqFit and NonlinearSolve, from a
  Julia depot.
- **R 4.6.2 (GPL-2+):** `src/library/stats`.
- **GSL 2.8 (GPL-3+):** `multimin`, `roots`, `min` and `multifit_nlinear`.

Absence in this package was checked with `git grep` and against README "Not
done" and coverage.md. Nothing was implemented.

## Licence rule

R and GSL were read only to understand them. Implementations must come from:

- the papers;
- the permissive originals those libraries wrap (netlib PORT, MINPACK);
- MIT Julia packages;
- Boost.

They must never come from R's C/R code or from GSL. Each entry names its
permissive source.

## Oracle rule

This follows the fleet rule of 2026-11-11: bit identity is a porting step,
never the goal.

- **Acceptance.** Each item is accepted against a stated error bound, asserted
  against an extended-precision oracle: double-double, `math/big` or mpmath.
  - For a root, the root is computed at extended precision, and the bound is on
    |x − x*| or on the bracket width.
  - For a minimiser, the minimum is computed at extended precision, and the
    bound is on x and on f.
  - For least squares, the bound is against the NIST certified values, which
    are given to 11–15 digits.
- **Speed.** The item is measured against the reference implementation.
- **Porting step.** While porting, reproducing the reference's iterates and
  evaluation counts checks the port; it is not the acceptance test.
- **Defaults.** A faster method that stays within the bound may become the
  default. A gain is never kept opt-in only to preserve parity with SciPy.
  SciPy-exact paths stay available as an option where they exist today.

The "Oracle" column below gives the bound and its oracle first, then the
reference used for the porting step.

## Ranked by gain / cost

Sizes: S < 300 lines, M 300–1500, L > 1500.

### New algorithms

| # | Item | Reference (licence), evidence | What is missing here | Gain for the fleet | Implementation source | Oracle | Size |
|---|---|---|---|---|---|---|---|
| 1 | TOMS 748 bracketing (Alefeld–Potra–Shi, A42) and Chandrupatla | Roots.jl (MIT) `src/Bracketing/alefeld_potra_shi.jl:237,297`, `chandrupatlu.jl:13` | coverage.md row 54: `toms748` missing | SciPy `toms748`; Roots.jl's bracketing method; fewer calls than Brent on smooth roots | The paper: Alefeld, Potra & Shi, "Algorithm 748: enclosing zeros of continuous functions", ACM TOMS 21 (1995). The code: Roots.jl (MIT). The TOMS-published Fortran is under the ACM software licence and is not a source | Final bracket within the requested `xtol + rtol·\|x*\|` of a root computed with `math/big` (or mpmath) on the APS Table 1 problems, in no more calls than Brent. Porting step: the iterates and calls of `find_zero(f, (a,b), A42())` and `scipy.optimize.toms748` (Roots `test/test_bracketing.jl`) | S |
| 2 | ITP bracketing | Roots.jl `src/Bracketing/itp.jl:24,70` | no ITP | worst case never worse than bisection, superlinear on average: a safe default for binders | Oliveira & Takahashi, ACM TOMS 47(1) (2020); Roots.jl (MIT) | Final bracket within the tolerance of a `math/big` root, in at most ⌈log₂((b−a)/2ε)⌉ + n₀ calls (the paper's worst-case bound). Porting step: the iterates of `find_zero(f, (a,b), ITP())` with the same κ₁, κ₂, n₀ | S |
| 3 | Hager–Zhang line search (approximate Wolfe, `secant2`), initial-step strategies (`InitialQuadratic`, `InitialHagerZhang`), backtracking; CG_DESCENT | LineSearches (MIT) `hagerzhang.jl:88,385`, `initialguess.jl:71,183`, `backtracking.jl:13`; Optim `cg.jl:61-65,223-252` | `linesearch.go` holds only SciPy's wolfe1/wolfe2/zoom; `cg.go` is PR+; coverage.md row 30: no exported line search | Optim parity for Julia; a pluggable, exported line search for every gradient method | The papers: Hager & Zhang, "A new conjugate gradient method with guaranteed descent and an efficient line search", SIAM J. Optim. 16(1) (2005), and "Algorithm 851: CG_DESCENT", ACM TOMS 32 (2006). The code: LineSearches.jl and Optim.jl (MIT). The TOMS-published C code is under the ACM software licence and is not a source | Minimiser and minimum within a stated bound (‖x − x*‖ ≤ 1e-8·max(1, ‖x*‖), f within 1e-12 relative) of the known optimum, or of an extended-precision Newton refinement, on OptimTestProblems and MGH. Porting step: the iterates and f/g counts of Optim `ConjugateGradient()` | M |
| 4 | Geodesic acceleration and Nielsen damping update in Levenberg–Marquardt | LsqFit (MIT) `geodesic.jl:47`, `levenberg_marquardt.jl:96,203`; GSL `multifit_nlinear/trust.c:61,545`, `nielsen.c:22-98` (read only) | LM is MINPACK only; no geodesic or Nielsen code | far fewer iterations on valley-shaped curve fits; LsqFit parity | Transtrum & Sethna, arXiv:1201.5885 (2012); Nielsen (1999) IMM report; LsqFit (MIT) | Parameters within a stated number of the NIST StRD certified digits (at least 6 on the "higher difficulty" set, from both NIST start points). Porting step: the iterates and counts of LsqFit `curve_fit(…; avv! = …)` | M |
| 5 | Exact trust-region Newton (Moré–Sorensen, with the hard case) and Steihaug–Krylov trust region | Optim `newton_trust_region.jl:18,97,321`, `krylov_trust_region.jl:1` | coverage.md rows for `trust-exact`, `trust-ncg`, `dogleg` | SciPy and Optim parity; robust second-order minimisation | Moré & Sorensen, SIAM J. Sci. Stat. Comput. 4 (1983); Nocedal & Wright Alg. 4.3 and 7.2; Optim.jl (MIT) | Minimiser within a stated bound of the known MGH optimum, refined in double-double; the hard case is solved on a constructed problem with a known answer. Porting step: SciPy `trust-exact`/`trust-ncg` counts and Optim counts | M |
| 6 | Quasi-Newton and spectral system solvers: Broyden (good/bad), Klement, limited-memory Broyden, DF-SANE, pseudo-transient continuation | NonlinearSolveQuasiNewton (MIT) `broyden.jl:34,78,85`, `klement.jl:29`, `lbroyden.jl:20`; SimpleNonlinearSolve `dfsane.jl:53`; NonlinearSolveFirstOrder `pseudo_transient.jl:40` | only `hybr` (README "Not done"; coverage.md rows 59–63) | SciPy `root(method=…)`; NonlinearSolve parity; derivative-free large systems | Broyden 1965; Klement 2014; La Cruz, Martínez & Raydan, Math. Comp. 75 (2006); Kelley & Keyes, SIAM J. Numer. Anal. 35 (1998); NonlinearSolve (MIT) | ‖F(x)‖ below a stated tolerance, and x within a stated bound of the root computed by Newton in `math/big`, on the MGH systems. Porting step: SciPy `root(method='broyden1'/'df-sane')` counts and SimpleNonlinearSolve iterates | M |
| 7 | PORT `nlminb`: bound-constrained Newton/quasi-Newton with reverse communication | R `stats/src/port.c:345-362` calls `drmnfb`/`drmngb`/`drmnhb` (read only) | no nlminb | R `nlminb` and `nls(algorithm="port")` parity | PORT library, netlib (D. Gay): `dmnfb`/`drmnfb`, `dmngb`/`drmngb`, `dmnhb`/`drmnhb` (check the netlib licence notice before porting) | Minimiser within a stated bound of the known optimum (MGH, Hock–Schittkowski bound-constrained problems) refined in double-double, with the bounds respected exactly. Porting step: netlib PORT compiled with gfortran (iterates and counts), and R's `nlminb` examples run as a black box | L |
| 8 | Variable projection for partially linear least squares (`nls` "plinear") and a plain Gauss–Newton `nls` | R `stats/R/nls.R:39,516,618` (read only) | no variable projection or plain Gauss–Newton | R `nls` parity; fewer parameters in separable fits | Golub & Pereyra, SIAM J. Numer. Anal. 10 (1973); Kaufman, BIT 15 (1975) | Parameters within a stated number of the NIST StRD certified digits. Porting step: R `nls(…, algorithm="plinear")` run as a black box | M |
| 9 | Fminbox (log-barrier wrapper around any optimiser), N-GMRES/O-ACCEL acceleration, SAMIN (bounded simulated annealing), particle swarm, momentum/accelerated gradient descent | Optim `fminbox.jl:154`, `ngmres.jl:11,21`, `samin.jl:42`, `particle_swarm.jl:1` | none present; coverage.md lists MATLAB `particleswarm` | Optim parity for Julia | Optim.jl (MIT) and the papers it cites (Goffe 1996 for SAMIN; Riseth 2019 for O-ACCEL) | Deterministic methods: minimiser within a stated bound of the known optimum. Stochastic methods: success rate and median evaluations over seeded runs, as `docs/design-global.md` already does for DE. Porting step: Optim's iterates | S–M each |
| 10 | Higher-order derivative-free scalar methods: Roots' `Order0` default (secant plus bracketing), Steffensen, Order5/8/16, Thukral, King, Esser | Roots.jl `src/DerivativeFree/order16.jl:22-23`, `src/find_zero.jl:227` | only Newton, secant and Halley | Roots.jl parity | Roots.jl (MIT) and its cited papers | Root within a stated bound of a `math/big` root on the APS and Roots.jl problem sets. Porting step: Roots.jl counts | S each |
| 11 | Quad-golden safeguarded scalar minimiser (safeguarded parabolic step with approximate golden section; contributed to GSL by James Howse, GSL `min/ChangeLog:3-4`) | GSL `min/quad_golden.c:46-54,87-118` (read only) | absent | small | The two works it cites: Brent, *Algorithms for Minimization without Derivatives*, Prentice-Hall (1973); Gill & Murray, "Safeguarded Steplength Algorithms for Optimization using Descent Methods", NPL Report NAC 37 (1974). Not GSL code | Minimiser within a stated bound of the minimiser computed in `math/big`, in no more calls than `Fminbound` | S |

### Better defaults and robustness

| # | Item | Evidence | Gain | Source / oracle | Size |
|---|---|---|---|---|---|
| D1 | Bisection on the float bit pattern: it reaches an exact zero, or a sign change between adjacent floats. Roots.jl's default for float brackets. | Roots.jl `src/Bracketing/bisection.jl:113-142`, `find_zero.jl:227-230` | no "loose tolerance" failures; deterministic last-ulp brackets for every binder | Roots.jl (MIT). Oracle: the final bracket is two adjacent floats (or an exact zero) that contain the root computed in `math/big`; that is a bound of 1 ulp, stated and asserted. Porting step: the same result as `find_zero(f, (a,b), Bisection())` | S |
| D2 | Bracket extension (`extendInt = "yes"/"upX"/"downX"`) for scalar roots | R `stats/R/nlm.R:53-73` (read only); coverage.md row 52 notes MATLAB `fzero`'s bracket search is missing | R `uniroot` parity; MATLAB `fzero` | Implemented from R's documented behaviour, not its code. Oracle: the extended bracket contains a sign change and a root, checked in `math/big`, within a stated number of expansions. Porting step: the status and interval of R `uniroot`, run as a black box | S |

### Test problems to import

| # | Suite | Where | Gain | Size |
|---|---|---|---|---|
| T1 | Alefeld–Potra–Shi and Roots.jl's bracketing problem sets (problems1/2/3/oo) | Roots.jl `test/benchmark-bracketing-methods.jl:242-354`; APS 1995 Table 1 | today's scalar-root suite has 11 functions (README "How close to SciPy") | S |
| T2 | NIST StRD nonlinear regression (boxbod, eckerle, hahn1, kirby2, rat42, rat43, thurber, enso, …) with certified values | public domain, NIST; also used by GSL `multifit_nlinear/test_*.c` (transcribe from NIST, not from GSL) | least squares is tested on MGH only | S–M |
| T3 | Optim's multivariate test problems (OptimTestProblems) | Optim test tree | oracle set for items 3, 5 and 9 | S |

## Already present (dropped)

These were found in the reference libraries but are already here:

- Moré–Thuente and zoom line searches. Fletcher's line search in GSL
  `vector_bfgs2` is covered by the strong-Wolfe search.
- MINPACK dogleg (`hybrd.go`) and MINPACK LM.
- BFGS, L-BFGS-B, Nelder–Mead, Brent and golden section.
- Optim's IPNewton, which is covered by `TrustConstr`.

## Not read

Not read in this inventory. These are unknowns, not absences:

- Optim: IPNewton internals, manifold support and AcceleratedGradientDescent
  details.
- NonlinearSolve: the default polyalgorithm (`poly_algs.jl`), the trust-region
  radius schemes, dogbox and multistart.
- LineSearches: how `MoreThuente`/`StrongWolfe` differ from SciPy's.
- R: `uncmin.c` (`nlm`), `optim.R` (SANN, CG variants) and `src/appl`.
- GSL: `multilarge_nlinear`, `multifit_nlinear` (subspace2D, cgst, ddogleg,
  scaling), `vector_bfgs2.c`, `simplex2.c` and `multiroots` (hybrids,
  dnewton).
- The netlib PORT licence text.
