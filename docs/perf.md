# Performance against SciPy

## Method

The benchmarks pair `bench_test.go` with `benchmarks/bench_scipy.py`, which
runs the same problems from the same starting points. `benchmarks/run.sh`
alternates the two: each round runs every Go benchmark once
(`-benchtime=300ms`, on one thread: `-test.cpu 1`, as SciPy runs), then the
whole SciPy script (each case a median of 3 batches of at least 0.2 s).
Interleaving spreads the load changes of a shared host evenly over both
sides. `benchmarks/summarize.py` reports the median over rounds. The raw
numbers are in `benchmarks/results.json`. `benchmarks/run.sh 5 'regexp'`
runs only the matching benchmarks on both sides.

Setup of the run below:

- 5 rounds on 2026-10-17 on an Apple M4 Max (arm64), Go 1.27.2, SciPy
  1.18.1, NumPy 2.5.4 (both pinned by `run.sh`), CPython 3.12.
- The host is shared: the load average was 11–13 during the run (16
  cores). Ratios move by about ±10–20% from run to run.
- **The SciPy objectives are SciPy's own vectorised `rosen`/`rosen_der`,
  and the curve-fit model is a vectorised NumPy expression.** That is the
  fastest form a notebook would write, not a straw man.

## Results (median of 5 interleaved rounds)

| Benchmark | Go | SciPy | SciPy time in Python callbacks | Speed-up | Speed-up on SciPy's time minus its callbacks |
|---|---|---|---|---|---|
| Rosenbrock n=2, BFGS, analytic gradient | 3.03 µs | 1.26 ms | 382 µs | **416×** | 290× |
| Rosenbrock n=100, BFGS, analytic gradient | 7.69 ms | 49.1 ms | 6.6 ms | **6.38×** | 5.52× |
| Rosenbrock n=2, L-BFGS-B, analytic gradient | 68 µs | 763 µs | 431 µs | **11×** | 4.88× |
| Rosenbrock n=100, L-BFGS-B, analytic gradient | 3.28 ms | 12.8 ms | 6.4 ms | **3.89×** | 1.94× |
| Rosenbrock n=2, BFGS, finite differences | 9.79 µs | 2.56 ms | 708 µs | **261×** | 189× |
| Rosenbrock n=100, L-BFGS-B, finite differences (stops at 15000 evaluations) | 1.66 ms | 121 ms | 67.3 ms | **73×** | 32× |
| Rosenbrock n=2, Nelder–Mead | 10.4 µs | 1.4 ms | 674 µs | **134×** | 70× |
| Rosenbrock n=2, Powell | 25.2 µs | 4.31 ms | 2.56 ms | **171×** | 70× |
| Rosenbrock n=2, CG, analytic gradient | 5.96 µs | 2.03 ms | 749 µs | **341×** | 215× |
| Rosenbrock n=100, CG, analytic gradient | 2.59 ms | 53.2 ms | 20.3 ms | **21×** | 13× |
| Rosenbrock n=100, BFGS, finite differences | 12.3 ms | 576 ms | 293 ms | **47×** | 23× |
| Rosenbrock n=2, Newton-CG, Hessian | 10.1 µs | 4.69 ms | 1.02 ms | **464×** | 363× |
| Rosenbrock n=100, Newton-CG, Hessian-vector product | 792 µs | 26.3 ms | 2.59 ms | **33×** | 30× |
| `curve_fit`, 10⁴ points, a·e^(−bx)+c, LM | 1.37 ms | 1.47 ms | — | **1.07×** | — |
| `curve_fit`, 10⁴ points, TRF | 2.03 ms | 2.58 ms | — | **1.27×** | — |
| `curve_fit`, 10⁴ points, TRF with bounds | 2.61 ms | 3.88 ms | — | **1.49×** | — |
| `least_squares`, Rosenbrock residuals, TRF | 24.9 µs | 1.36 ms | — | **55×** | — |
| `least_squares`, Rosenbrock residuals, LM | 7.8 µs | 521 µs | — | **67×** | — |
| `leastsq`, Rosenbrock residuals | 6.89 µs | 40.1 µs | — | **5.82×** | — |
| `root(method='hybr')`, 2 equations | 1.16 µs | 14.5 µs | — | **13×** | — |
| `fsolve`, 2 equations | 1.15 µs | 15.6 µs | — | **14×** | — |
| `root(method='hybr')`, Broyden tridiagonal, 100 equations | 649 µs | 825 µs | — | **1.27×** | — |
| `linprog`, 2 variables (Go simplex, SciPy HiGHS) | 1.21 µs | 405 µs | — | **333×** | — |
| `linprog`, dense 30×50 (Go simplex, SciPy HiGHS dual simplex) | 42 µs | 771 µs | — | **18×** | — |
| `linprog`, dense 30×50 (Go interior point, SciPy HiGHS IPM) | 322 µs | 1.35 ms | — | **4.20×** | — |
| `brentq`, x³−2x−5 on [2, 3] | 133 ns | 4.06 µs | — | **31×** | — |
| `brenth`, same | 145 ns | 3.88 µs | — | **27×** | — |
| `ridder`, same | 166 ns | 6.34 µs | — | **38×** | — |
| `bisect`, same | 160 ns | 16.5 µs | — | **103×** | — |
| `newton` with f′, from 2 | 29 ns | 24.2 µs | — | **838×** | — |
| `newton` (secant), from 2 | 38 ns | 30 µs | — | **791×** | — |
| `newton` with f′ and f″ (Halley), from 2 | 38 ns | 19.7 µs | — | **525×** | — |
| `minimize_scalar`, Brent | 316 ns | 22.6 µs | — | **71×** | — |
| `minimize_scalar`, golden | 340 ns | 25.9 µs | — | **76×** | — |
| `minimize_scalar`, bounded | 194 ns | 28.3 µs | — | **146×** | — |
| `fminbound(cos, 3, 4)` | 194 ns | 17.7 µs | — | **91×** | — |
| `bracket` | 12 ns | 2.57 µs | — | **217×** | — |
| `brute`, 50×50 grid, no finish | 11.1 µs | 1.57 ms | — | **141×** | — |
| `brute`, 20×20 grid, Nelder–Mead finish | 4.87 µs | 539 µs | — | **111×** | — |
| `approx_fprime`, Rosenbrock n=100 | 6.44 µs | 792 µs | — | **123×** | — |
| `check_grad`, Rosenbrock n=100 | 7.04 µs | 765 µs | — | **109×** | — |

How to read the columns:

- **"SciPy time in Python callbacks"** is the time to make SciPy's own
  `nfev` objective calls and `njev` gradient calls, outside the optimiser.
- **The last column** divides SciPy's time minus those callbacks by ours.
  It is the honest comparison of the algorithms' own code, because Go's
  callbacks cost almost nothing.
- `linprog`: SciPy 1.18 only has HiGHS; this package has a revised simplex
  and the interior-point method SciPy used to have. The rows compare the
  same LP solved to optimality, not the same algorithm.

Every operation is at least as fast as SciPy. The closest are the large
least-squares problems, where SciPy's model and its kernels are compiled
code; they are discussed below.

## Before and after this round of work

The rows that were slower than SciPy, and those this work changed.
"Before" is the code of 2026-10-12 (`origin/main` then): the `curve_fit`
rows are interleaved Go/SciPy rounds of that day; for the other rows the
old and new Go builds were alternated against each other on 2026-10-17 and
divided into this run's SciPy time.

| Benchmark | before | after |
|---|---:|---:|
| `curve_fit`, 10⁴ points, LM | **0.66×** (1.99 ms vs 1.32 ms) | 1.07× |
| `curve_fit`, 10⁴ points, TRF | **0.83×** (2.89 ms vs 2.39 ms) | 1.27× |
| `curve_fit`, 10⁴ points, TRF with bounds (new row) | **0.94×** (4.11 ms) | 1.49× |
| `root`, Broyden tridiagonal, 100 equations (new row) | **0.78×** (1.06 ms) | 1.27× |
| `root` / `fsolve`, 2 equations | 13× | 13× (unchanged) |
| `leastsq`, `least_squares`, 2 residuals | 5.8×, 55–67× | unchanged |

The Go side of `curve_fit` became 1.55× faster (LM 2.10 → 1.35 ms, TRF
3.12 → 1.96 ms, bounded TRF 4.11 → 2.65 ms in the alternated runs) and
`root` on 100 equations 1.76× (1.06 ms → 0.60 ms).

### Same iterations as SciPy

The slow rows were not doing more work: Go and SciPy take the same steps
(`curve_fit` LM: 21 function evaluations each, TRF: 6 Jacobians, bounded
TRF: 8; `root` on Broyden: 111 evaluations plus SciPy's 2 wrapper calls).
The time went into the per-iteration linear algebra.

### What changed

All of it keeps every sum in its original order, so **every result is
unchanged to the last bit** (see "Evidence" below).

- **Independent sums advance in the same pass.** MINPACK's QR
  factorisation (`qrfac`) forms, one after the other, sums that do not
  depend on one another: the column norms, and at each step the products
  of the Householder vector with every remaining column. Each such sum is
  latency-bound (one floating-point add per element, waiting for the
  previous one). They are now accumulated side by side, four per pass, each
  in its own order. The same applies to `Qᵀ·fvec`, which lmdif and hybrd
  formed after the factorisation and which is now formed with it; to the
  norm of the next pivot column, accumulated while the update produces it;
  to the exchange of the pivot column, done in the scaling pass; to
  `qform`'s columns; to hybrd's Broyden update and predicted reduction and
  dogleg's `R·g` (four rows of a packed triangle at a time); and to TRF's
  `householderR` (`vᵀv` with every `vᵀc`, the next column's norm with the
  update). On 10⁴×3, qrfac went from about 100 µs to 55 µs and
  householderR from 162 µs to 67 µs; on 100×100, qform from 347 µs to
  144 µs.
- **One range test for four components.** The column norms use MINPACK's
  three-accumulator `enorm`; testing the four components of a row for the
  intermediate range is one unsigned comparison of their bit patterns.
- **Fewer allocations and copies.** TRF keeps its work buffers between
  iterations, `curve_fit` reads `xdata` and `ydata` in place and calls a
  scalar model directly instead of through a k-vector wrapper, and
  finiteness tests read exponent bits.
- **Small problems keep MINPACK's plain loops.** Below 32 rows the
  bookkeeping of the fused passes costs more than it saves (it had made
  `root` on 2 equations 5–10% slower), so `qrfac`, `qform` and the packed
  sums take the original loops there. Both paths give the same bits.

### Evidence that results are unchanged

- A dump of every result of `curve_fit` (10 option sets on 4 data sizes,
  with and without sigma, bounds, robust losses, 2-D xdata, analytic and
  numerical Jacobians), `leastsq`, `least_squares` and `root` on all
  MINPACK test problems is identical, bit for bit, before and after, on
  arm64 (with fused multiply-add) and on amd64 (without).
- `minpack_fused_test.go` compares every fused kernel with the original
  code on random matrices covering every magnitude range of `enorm`, its
  exact bounds, zero and repeated columns, NaN and signed zeros: 4320 QR
  factorisations, 350 `householderR` factorisations, `qform`,
  `packedRowDots`, and `root` on 40 equations through both paths.
  Mutations of the fused code (a dropped term, a shifted bound) make them
  fail.
- The SciPy fixture tests (bit-exact on MINPACK's iterates and counts
  where they were before) pass unchanged; no tolerance was changed.

### What remains close, and why

`curve_fit` LM is 1.07× and TRF 1.27×. Of LM's 1.37 ms, about 0.95 ms is
the 21 evaluations of the model over 10⁴ points: a Go closure calling
`math.Exp` costs about 45 µs per sweep, against 29 µs for NumPy's
vectorised `a*np.exp(-b*x)+c`. The solver itself now takes about 0.4 ms
against about 0.8 ms for SciPy's compiled MINPACK and wrappers; what is
left of it is the minimum number of latency-bound passes MINPACK's order
of operations allows (each sum must stay sequential to keep the results
bit-identical). `CurveFitOptions.Workers` evaluates the model in parallel
when that is acceptable; the table is single-threaded.

`root` on 100 equations is 1.27×: SciPy's 0.83 ms is about 0.31 ms of
Python callbacks and 0.5 ms of compiled MINPACK, ours 0.6 ms of solver.

## SLSQP

Measured with `tools/slsqpbench` (Go), which builds the test binary, runs
`BenchmarkSLSQP*` with `-test.cpu 1` and SciPy's `minimize(method='SLSQP')`
on the same problems with one BLAS thread, in interleaved rounds, and
refuses to start when the 1-minute load average is above 8. Apple M4 Max,
SciPy 1.18.1 (arm64 wheel), 2026-10-30.

Medians of 7 interleaved rounds; load 4.6 before, 5.4 after.

| Benchmark | Go | SciPy 1.18.1 | Speed-up |
|---|---|---|---|
| HS71, analytic derivatives | 5.59 µs | 114 µs | 20.4× |
| HS71, finite differences | 6.90 µs | 648 µs | 93.9× |
| HS38 (bounds, 76 iterations) | 37.0 µs | 936 µs | 25.3× |
| HS43 (3 nonlinear inequalities) | 6.83 µs | 163 µs | 23.8× |
| Quad50 (n = 50, 100 bounds, 1 linear) | 3.35 ms | 4.59 ms | 1.4× |

On the Hock–Schittkowski problems SciPy's time is mostly Python around the
callbacks and the per-iteration call into C. At n = 50 both run the same
O(m·n²) subproblem per iteration in compiled code; there `trsm` (X·R = G
for the 101×50 constraint block) dominated the Go profile and the row was at
parity (4.66 ms against 4.65 ms, same host and method, load 5.5). `trsmRUN`
now applies four columns per pass over the block, bit-identical to the
reference loop (`TestTrsmMatchesReference`), which halves its time in
`BenchmarkTrsm` and gives the 1.4× above. The subproblem scratch is reused
across iterations (0.7 MB allocated per Quad50 solve instead of 11 MB).

## trust-constr

`tools/slsqpbench` times `BenchmarkTrustConstr*` against SciPy's
`minimize(method='trust-constr')` on the same five problems, in the same
interleaved rounds as SLSQP.

Apple M4 Max, 2026-10-31 (v0.4.1), medians of 7 interleaved rounds, one
thread each, `slsqpbench --wait 150m`. The 1-minute load was 7.1 before the
first round and 7.3 after the last, and 6.4–7.7 after each round in between:
below the tool's limit of 8 throughout, though not an idle machine (another
session's work). An earlier run of the same day, which began at 7.3 and ended
at 21.9, was discarded: its Go times were 2.5 times too slow (HS71 SLSQP 14 µs
instead of 5.6 µs), and the tool now discards such a run itself (it reads the
load after every round and stops when it exceeds the limit).

| Benchmark | Go | SciPy 1.18.1 | Speed-up |
|---|---|---|---|
| HS71, analytic derivatives | 496 µs | 32.7 ms | 66.0× |
| HS71, finite differences | 438 µs | 34.1 ms | 77.8× |
| HS38 (bounds) | 277 µs | 32.1 ms | 116× |
| HS43 (3 nonlinear inequalities) | 62.4 µs | 7.15 ms | 115× |
| Quad50 (n = 50, 100 bounds, 1 linear) | 27.8 ms | 69.9 ms | 2.5× |

This replaces the table taken on the same day while the load rose from 7.4 to
10–11 (64.7×, 76.6×, 130×, 116×, 2.7×): the ratios moved by 2–11%, the
reading did not change. SciPy's trust-constr is pure Python around
NumPy/LAPACK calls, so on small problems its time is interpreter overhead. At
n = 50 each iteration factorises the 101×151 barrier Jacobian; the pivoted QR
(dlaqp2's norm downdating, go-simd reflectors) cut Go's time from about 130 ms
to 28 ms. The SLSQP rows measured in the same rounds agree with the table above
(5.57 µs, 6.78 µs, 36.2 µs, 6.53 µs and 3.31 ms against 113 µs, 640 µs,
936 µs, 163 µs and 4.58 ms: 20.4×, 94.4×, 25.8×, 25.0×, 1.4×).

## Quadprog

`tools/qpbench` times `Quadprog` against OSQP v1.0.0 (the C core of the `osqp`
1.1.3 wheel, which `qporacle` checks) and the `quadprog` 0.1.14 package
(Goldfarb–Idnani, Fortran) on instances it generates into a cache directory
outside any repository. The classes are those of `osqp_benchmarks`
(`problem_classes/*.py`, opened for this: random QP, equality QP, portfolio,
Lasso, Huber, SVM, control) at three sizes, with Go's random numbers instead of
numpy's, control with Q_N = 10 Q (the alternative in the reference's own
comments), and dense positive definite problems (P = MMᵀ/n + I, 2n inequalities,
n/10 equalities) at n = 10 to 1000. OSQP's setup and solve times are its own
(`info.setup_time`, `info.solve_time`, measured in C, as `osqp_benchmarks`
reports them); `quadprog` has no timer, so the call is timed and the 4.2 µs of an
empty-sized call subtracted. ε_abs = ε_rel = 1e-3, no polishing, OSQP's other
defaults on both sides. Our *iteration counts equal OSQP's on all 25 instances*.

Apple M4 Max, 2026-11-01, medians of 5 interleaved rounds (the Go benchmark, then
the reference, on the same files), one thread each, 1-minute load 2.5 before and
2.4 after (2.4–2.7 after each round; no round discarded). Ratio = ours / reference,
above 1 is slower.

ADMM against OSQP. The last two columns are the share of *our* time inside
go-sparse v0.2.1: AMD ordering plus symbolic analysis in the setup, numeric
refactorisation plus the triangular solves in setup + solve (reported apart, as
go-sparse is a separate package whose speed is its own open defect, below).

| Instance | n | m | setup ours | OSQP | ratio | solve ours | OSQP | ratio | AMD + symbolic / setup | LDLᵀ / total |
|---|---|---|---|---|---|---|---|---|---|---|
| control-nx5 | 75 | 130 | 95.7 µs | 29.6 µs | 3.24 | 293 µs | 235 µs | 1.25 | 49% | 62% |
| control-nx10 | 160 | 270 | 299 µs | 89 µs | 3.36 | 166 µs | 142 µs | 1.17 | 47% | 34% |
| control-nx20 | 320 | 540 | 1.1 ms | 314 µs | 3.50 | 482 µs | 425 µs | 1.13 | 44% | 32% |
| eq-qp-n50 | 50 | 25 | 212 µs | 45.2 µs | 4.69 | 96.5 µs | 93.1 µs | 1.04 | 36% | 31% |
| eq-qp-n100 | 100 | 50 | 1.11 ms | 237 µs | 4.67 | 374 µs | 345 µs | 1.08 | 34% | 29% |
| eq-qp-n200 | 200 | 100 | 4.37 ms | 1.1 ms | 3.97 | 1.41 ms | 1.34 ms | 1.05 | 19% | 33% |
| huber-n5 | 1505 | 1500 | 834 µs | 262 µs | 3.18 | 2.07 ms | 1.39 ms | 1.49 | 54% | 49% |
| huber-n10 | 3010 | 3000 | 1.98 ms | 636 µs | 3.11 | 4.31 ms | 3.02 ms | 1.43 | 57% | 48% |
| lasso-n5 | 510 | 510 | 290 µs | 96.8 µs | 3.00 | 576 µs | 401 µs | 1.43 | 51% | 46% |
| lasso-n10 | 1020 | 1020 | 898 µs | 275 µs | 3.26 | 1.33 ms | 1 ms | 1.33 | 62% | 39% |
| lasso-n20 | 2040 | 2040 | 4.57 ms | 1.82 ms | 2.51 | 3.55 ms | 3.04 ms | 1.17 | 80% | 30% |
| portfolio-k5 | 505 | 506 | 741 µs | 225 µs | 3.29 | 2.98 ms | 2.34 ms | 1.27 | 63% | 60% |
| portfolio-k10 | 1010 | 1011 | 1.14 ms | 363 µs | 3.15 | 17.3 ms | 20.1 ms | **0.86** | 33% | 73% |
| portfolio-k20 | 2020 | 2021 | 3.6 ms | 1.14 ms | 3.16 | 20.6 ms | 26.5 ms | **0.78** | 32% | 68% |
| random-qp-n20 | 20 | 200 | 163 µs | 41.8 µs | 3.91 | 708 µs | 572 µs | 1.24 | 63% | 58% |
| random-qp-n50 | 50 | 500 | 1.58 ms | 603 µs | 2.63 | 1.62 ms | 1.42 ms | 1.14 | 75% | 41% |
| random-qp-n100 | 100 | 1000 | 9.58 ms | 4.28 ms | 2.24 | 16.1 ms | 14.7 ms | 1.09 | 80% | 54% |
| svm-n5 | 505 | 1000 | 397 µs | 123 µs | 3.22 | 1.83 ms | 1.33 ms | 1.37 | 57% | 53% |
| svm-n10 | 1010 | 2000 | 1.1 ms | 321 µs | 3.44 | 5.34 ms | 4.08 ms | 1.31 | 64% | 53% |
| svm-n20 | 2020 | 4000 | 4.91 ms | 1.89 ms | 2.59 | 11.8 ms | 9.87 ms | 1.19 | 79% | 46% |
| dense-n10 | 10 | 21 | 40.1 µs | 10.2 µs | 3.92 | 79.6 µs | 71.1 µs | 1.12 | 41% | 54% |
| dense-n50 | 50 | 105 | 864 µs | 194 µs | 4.45 | 481 µs | 446 µs | 1.08 | 27% | 38% |
| dense-n200 | 200 | 420 | 17.1 ms | 6.39 ms | 2.68 | 10.8 ms | 10.3 ms | 1.05 | 19% | 52% |
| dense-n500 | 500 | 1050 | 145 ms | 222 ms | **0.65** | 90 ms | 166 ms | **0.54** | 17% | 62% |
| dense-n1000 | 1000 | 2100 | 832 ms | 1.67 s | **0.50** | 511 ms | 918 ms | **0.56** | 15% | 71% |

Geometric mean of the (setup + solve) ratio per class: portfolio 1.05, dense 1.10,
random QP 1.45, SVM 1.47, Lasso 1.72, Huber 1.74, control 1.85, equality QP 2.38.

*Reading.* The iterations are the same, so the differences are per-iteration cost
and setup. On the larger problems (dense n ≥ 500, portfolio k ≥ 10) ADMM is
faster than OSQP, by up to 2× on the dense ones. The solve phase of the sparse
problems is 1.04–1.49× OSQP's and the setup 2.2–4.9×. The cause is go-sparse,
not the ADMM code: the LDLᵀ solves alone (`Permuted.Solve`: permute, L, D, Lᵀ,
un-permute) take 29–73% of our solve phase, e.g. 1.0 ms of the 2.07 ms of
`huber-n5`, where OSQP's complete solve phase (QDLDL solves included) is 1.39 ms;
and AMD plus the symbolic analysis are 17–80% of our setup and 70% of the bytes a
setup allocates (`order.symGraph` 23%, `order.newEngine` 14%, `ldl.NewPermuted`
20%: 140 KB for the 75×75 KKT matrix of `eq-qp-n50`), so that on macOS the Go
runtime spends more time in GC and `madvise` than in the ordering. What belongs
to this package was fixed in this round: loops without bounds checks in the ADMM
step (huber-n5 solve 1.70 → 1.47 ms), one allocation for the rows of the model.

Active set (Goldfarb–Idnani) against the `quadprog` package, dense, P positive
definite (setup + solve):

| Instance | n | m | ours | quadprog | ratio |
|---|---|---|---|---|---|
| dense-n10 | 10 | 21 | 7.98 µs | 1.75 µs | 4.56 |
| dense-n50 | 50 | 105 | 270 µs | 134 µs | 2.01 |
| dense-n200 | 200 | 420 | 12.2 ms | 11.3 ms | 1.09 |
| dense-n500 | 500 | 1050 | 177 ms | 231 ms | **0.76** |
| dense-n1000 | 1000 | 2100 | 1.46 s | 2.15 s | **0.68** |

Before this round the same rows were 5.87, 3.41, 2.41, 1.88 and 1.68 (a first
run at load 2.7): the constraint scan built a vector per column per pass
(`giState.vec`, 4040 allocations and 2 MB for n = 50) and the scalar `dot` was
used where `go-simd/floats.Dot` exists; the scan now dots the stored row, the dense
rows are stored with one allocation, and the dots and axpys are go-simd's.

*QPAuto threshold.* Our active set against our ADMM on the same dense instances
(the ADMM at 1e-3, so far less accurate than the active set's answer):

| Instance | n + m | active set | ADMM | active set / ADMM |
|---|---|---|---|---|
| dense-n10 | 31 | 7.98 µs | 120 µs | 0.07 |
| dense-n50 | 155 | 270 µs | 1.35 ms | 0.20 |
| dense-n200 | 620 | 12.2 ms | 28.2 ms | 0.43 |
| dense-n500 | 1550 | 177 ms | 236 ms | 0.75 |
| dense-n1000 | 3100 | 1.46 s | 1.34 s | 1.09 |

The ratio crosses 1 near n + m = 2600 (log-linear between 1550 and 3100). Since the
active set is exact and ADMM is not, `qpAutoLimit` is set to **2500**, below the
crossover (it was provisionally 2000).

### Open defects

- **ADMM setup 2.2–4.9× OSQP's, solve 1.04–1.49× on sparse problems below
  n + m ≈ 2000 (21 of the 25 ADMM rows are flagged slower by the tool; the other
  four are the largest problems): go-sparse.** `order.AMD` and
  `ldl.NewPermuted` allocate heavily (symGraph per node slices), `Permuted.Solve`
  costs about one whole OSQP iteration, and `Refactor` is the known 11–21% behind
  QDLDL. They are owned by go-sparse/sparse and are *not* worked around here;
  suggested there: a reusable workspace for AMD and the symbolic phase, the
  permutation fused into the first and last triangular sweep, allocation-free
  solves. Re-measure with `tools/qpbench` after each go-sparse release. (An issue
  could not be opened from this work; this paragraph is its text.)
- **Active set 2.0× `quadprog` at n = 50 and 4.6× at n = 10** (7.98 vs 1.75 µs,
  263 µs vs 134 µs): the per-call setup (Cholesky of P, J = L⁻ᵀ, canonical rows,
  result assembly with the LPDual blocks and the from-scratch KKT residuals) is
  fixed cost that quadprog does not pay; at n = 200 the gap is 9% and from
  n = 500 we are faster. Not yet profiled below 1 ms.
- ADMM setup allocations (about 390 KB for a 75-dimensional problem) are mostly
  go-sparse; ours (the model, `admm.alloc`, `buildKKT`) are about a quarter.
- Not measured: the Maros–Mészáros medium and full tiers (they need the MAT
  conversion for 138 problems, `qporacle mm`, and a harness mode that reads them),
  OSQP at ε = 1e-5, and polishing.

## Reproducing

```sh
cd tools/qpbench && GOWORK=off go run . --repo ../.. --rounds 5   # needs uv and a load below 8
```

Everything above the line reproduces the SciPy numbers:

```sh
benchmarks/run.sh 5          # needs uv (https://docs.astral.sh/uv/) for SciPy
GOWORK=off go test -run '^$' -bench . -benchtime=1s -cpu 1   # Go side only
```
