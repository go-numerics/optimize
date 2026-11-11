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

The reference is the equality oracle. That means the same iterates and
evaluation counts as the named implementation, on the named problem set, or a
stated tolerance where no bit-exact path is possible.

## Ranked by gain / cost

Sizes: S < 300 lines, M 300–1500, L > 1500.

### New algorithms

| # | Item | Reference (licence), evidence | What is missing here | Gain for the fleet | Implementation source | Oracle | Size |
|---|---|---|---|---|---|---|---|
| 1 | TOMS 748 bracketing (Alefeld–Potra–Shi, A42) and Chandrupatla | Roots.jl (MIT) `src/Bracketing/alefeld_potra_shi.jl:237,297`, `chandrupatlu.jl:13` | coverage.md row 54: `toms748` missing | SciPy `toms748`; Roots.jl's bracketing method; fewer calls than Brent on smooth roots | Alefeld, Potra & Shi, ACM TOMS 21 (1995), Alg. 748; Roots.jl (MIT) | Same root, iterations and calls as `scipy.optimize.toms748` and `find_zero(f, (a,b), A42())` on the APS Table 1 problems (Roots `test/test_bracketing.jl`) | S |
| 2 | ITP bracketing | Roots.jl `src/Bracketing/itp.jl:24,70` | no ITP | worst case never worse than bisection, superlinear on average: a safe default for binders | Oliveira & Takahashi, ACM TOMS 47(1) (2020); Roots.jl (MIT) | Same iterates as `find_zero(f, (a,b), ITP())` with the same κ₁, κ₂, n₀ | S |
| 3 | Hager–Zhang line search (approximate Wolfe, `secant2`), initial-step strategies (`InitialQuadratic`, `InitialHagerZhang`), backtracking; CG_DESCENT | LineSearches (MIT) `hagerzhang.jl:88,385`, `initialguess.jl:71,183`, `backtracking.jl:13`; Optim `cg.jl:61-65,223-252` | `linesearch.go` holds only SciPy's wolfe1/wolfe2/zoom; `cg.go` is PR+; coverage.md row 30: no exported line search | Optim parity for Julia; a pluggable, exported line search for every gradient method | Hager & Zhang, SIAM J. Optim. 16(1) (2005), ACM TOMS 32 (2006) Alg. 851; Optim.jl / LineSearches (MIT) | Same iterates and f/g counts as Optim `ConjugateGradient()` on OptimTestProblems' multivariate set | M |
| 4 | Geodesic acceleration and Nielsen damping update in Levenberg–Marquardt | LsqFit (MIT) `geodesic.jl:47`, `levenberg_marquardt.jl:96,203`; GSL `multifit_nlinear/trust.c:61,545`, `nielsen.c:22-98` (read only) | LM is MINPACK only; no geodesic or Nielsen code | far fewer iterations on valley-shaped curve fits; LsqFit parity | Transtrum & Sethna, arXiv:1201.5885 (2012); Nielsen (1999) IMM report; LsqFit (MIT) | Same iterates and counts as LsqFit `curve_fit(…; avv! = …)`; NIST StRD certified values | M |
| 5 | Exact trust-region Newton (Moré–Sorensen, with the hard case) and Steihaug–Krylov trust region | Optim `newton_trust_region.jl:18,97,321`, `krylov_trust_region.jl:1` | coverage.md rows for `trust-exact`, `trust-ncg`, `dogleg` | SciPy and Optim parity; robust second-order minimisation | Moré & Sorensen, SIAM J. Sci. Stat. Comput. 4 (1983); Nocedal & Wright Alg. 4.3 and 7.2; Optim.jl (MIT) | SciPy `trust-exact`/`trust-ncg` counts and Optim counts on MGH | M |
| 6 | Quasi-Newton and spectral system solvers: Broyden (good/bad), Klement, limited-memory Broyden, DF-SANE, pseudo-transient continuation | NonlinearSolveQuasiNewton (MIT) `broyden.jl:34,78,85`, `klement.jl:29`, `lbroyden.jl:20`; SimpleNonlinearSolve `dfsane.jl:53`; NonlinearSolveFirstOrder `pseudo_transient.jl:40` | only `hybr` (README "Not done"; coverage.md rows 59–63) | SciPy `root(method=…)`; NonlinearSolve parity; derivative-free large systems | Broyden 1965; Klement 2014; La Cruz, Martínez & Raydan, Math. Comp. 75 (2006); Kelley & Keyes, SIAM J. Numer. Anal. 35 (1998); NonlinearSolve (MIT) | SciPy `root(method='broyden1'/'df-sane')` counts; SimpleNonlinearSolve iterates on MGH systems | M |
| 7 | PORT `nlminb`: bound-constrained Newton/quasi-Newton with reverse communication | R `stats/src/port.c:345-362` calls `drmnfb`/`drmngb`/`drmnhb` (read only) | no nlminb | R `nlminb` and `nls(algorithm="port")` parity | PORT library, netlib (D. Gay): `dmnfb`/`drmnfb`, `dmngb`/`drmngb`, `dmnhb`/`drmnhb` (check the netlib licence notice before porting) | netlib PORT compiled with gfortran: same iterates and counts; R's `nlminb` examples, run as a black box | L |
| 8 | Variable projection for partially linear least squares (`nls` "plinear") and a plain Gauss–Newton `nls` | R `stats/R/nls.R:39,516,618` (read only) | no variable projection or plain Gauss–Newton | R `nls` parity; fewer parameters in separable fits | Golub & Pereyra, SIAM J. Numer. Anal. 10 (1973); Kaufman, BIT 15 (1975) | NIST certified values to tolerance; R `nls(…, algorithm="plinear")` as a black box | M |
| 9 | Fminbox (log-barrier wrapper around any optimiser), N-GMRES/O-ACCEL acceleration, SAMIN (bounded simulated annealing), particle swarm, momentum/accelerated gradient descent | Optim `fminbox.jl:154`, `ngmres.jl:11,21`, `samin.jl:42`, `particle_swarm.jl:1` | none present; coverage.md lists MATLAB `particleswarm` | Optim parity for Julia | Optim.jl (MIT) and the papers it cites (Goffe 1996 for SAMIN; Riseth 2019 for O-ACCEL) | Optim's iterates; stochastic methods need random-stream parity | S–M each |
| 10 | Higher-order derivative-free scalar methods: Roots' `Order0` default (secant plus bracketing), Steffensen, Order5/8/16, Thukral, King, Esser | Roots.jl `src/DerivativeFree/order16.jl:22-23`, `src/find_zero.jl:227` | only Newton, secant and Halley | Roots.jl parity | Roots.jl (MIT) and its cited papers | Roots.jl counts | S each |
| 11 | Quad-golden safeguarded scalar minimiser (Gilbert–Brent) | GSL `min/quad_golden.c:87-118` (read only) | absent | small | P. Gilbert's description (see GSL ChangeLog); not GSL code | tolerance against the known minimiser | S |

### Better defaults and robustness

| # | Item | Evidence | Gain | Source / oracle | Size |
|---|---|---|---|---|---|
| D1 | Bisection on the float bit pattern: it reaches an exact zero, or a sign change between adjacent floats. Roots.jl's default for float brackets. | Roots.jl `src/Bracketing/bisection.jl:113-142`, `find_zero.jl:227-230` | no "loose tolerance" failures; deterministic last-ulp brackets for every binder | Roots.jl (MIT). Oracle: bit-exact root against `find_zero(f, (a,b), Bisection())` | S |
| D2 | Bracket extension (`extendInt = "yes"/"upX"/"downX"`) for scalar roots | R `stats/R/nlm.R:53-73` (read only); coverage.md row 52 notes MATLAB `fzero`'s bracket search is missing | R `uniroot` parity; MATLAB `fzero` | Implemented from R's documented behaviour, not its code. Oracle: status and interval of R `uniroot`, run as a black box | S |

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
