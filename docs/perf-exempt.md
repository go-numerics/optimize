# Public functions without a performance case

Read by `refbench` in the CI mode (`cd bench && go test -run TestPerf .`): every
public function of the module has a case in `bench/perf_test.go` or is named here
with its reason. The method and the table are in [perf.md](perf.md). A row here
is a debt or a deliberate absence, not a verdict.

## Result accessors and constructors (no solver, no SciPy counterpart)

`OptimizeResult` attributes are plain NumPy arrays in SciPy; here the slices are
the fields and these methods copy one into an `ndarray`, once per call.

- `CurveFitResult.PcovArray`: copies `Pcov` into an n×n ndarray.
- `CurveFitResult.Perr`: square roots of the covariance diagonal (`np.sqrt(np.diag(pcov))`, a one-line NumPy expression).
- `CurveFitResult.PoptArray`: copies `Popt` into an ndarray.
- `LPResult.XArray`: copies `X` into an ndarray.
- `LSResult.JacArray`: copies `Jac` into an ndarray.
- `LSResult.XArray`: copies `X` into an ndarray.
- `LeastsqResult.CovXArray`: copies `CovX` into an ndarray.
- `LeastsqResult.FjacArray`: copies `Fjac` into an ndarray.
- `LeastsqResult.Success`: compares `Ier` with 1..4.
- `LeastsqResult.XArray`: copies `X` into an ndarray.
- `Result.HessInvArray`: copies `HessInv` into an ndarray.
- `Result.XArray`: copies `X` into an ndarray.
- `RootResult.FjacArray`: copies `Fjac` into an ndarray.
- `RootResult.XArray`: copies `X` into an ndarray.
- `Float`: pointer constructor for an optional option.
- `Int`: pointer constructor for an optional option.
- `Unbounded`: returns the free interval.

## Quadratic programming (SciPy has no QP solver)

Measured against OSQP and the `quadprog` package by `tools/qpbench`
(docs/perf.md, "Quadprog"); SciPy 1.18.1 offers `linprog` and `minimize` only,
so the reference of this campaign has nothing to time. Allocation counts of the
solver are covered by `qp_bench_test.go` (`-benchmem`).

- `Quadprog`: no scipy.optimize reference; see above.
- `NewQPSolver`: as `Quadprog`.
- `QPSolver.Solve`: as `Quadprog`.
- `QPSolver.UpdateQ`: as `Quadprog` (warm-start API, no SciPy equivalent).
- `QPSolver.UpdateBounds`: as `QPSolver.UpdateQ`.
- `QPSolver.UpdateMatrixValues`: as `QPSolver.UpdateQ`.
- `QPSolver.WarmStart`: as `QPSolver.UpdateQ`.

## Random sources (a port of NumPy's generators, not a scipy.optimize function)

The reference of these is `numpy.random.Generator` / `RandomState`, outside the
scipy.optimize row set; the streams are verified bit for bit by the package's
fixtures (`rng.go`), and every global optimiser row above draws from them.
A NumPy-vs-Go row for the generators is a follow-up.

- `NewRNG`: constructor.
- `NewLegacyRNG`: constructor.
- `RNGFromGenerator`: constructor.
- `RNGFromRandomState`: constructor.
- `RandomSource.Float64`: scalar draw, `Generator.random()`.
- `RandomSource.Uniform`: scalar draw, `Generator.uniform()`.
- `RandomSource.FillUniform`: array draw, `Generator.uniform(size=)`.
- `RandomSource.IntN`: scalar draw, `Generator.integers()`.
- `RandomSource.FillIntN`: array draw, `Generator.integers(size=)`.
- `RandomSource.ShuffleInt64`: `Generator.shuffle` of an int64 array.
