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

TRUSTCONSTR_TABLE

## Reproducing

```sh
benchmarks/run.sh 5          # needs uv (https://docs.astral.sh/uv/) for SciPy
GOWORK=off go test -run '^$' -bench . -benchtime=1s -cpu 1   # Go side only
```
