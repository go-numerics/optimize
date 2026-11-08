# Evaluation counts against SciPy

Output of `cd bench && go test -run TestCounts -v -perf.parity .` (SciPy 1.18.1,
NumPy 2.5.4, 2026-11-08): every comparable row of `bench/perf_test.go` run once on
both sides, from the same starting point and with the defaults of both. A timing
ratio is only a comparison of the solvers where the two sides took the same path,
that is where these counts are equal. `-` is a count that side does not report;
rows without counts (`Linprog` against HiGHS, `Brute`, `ApproxFprime`,
`CheckGrad`) are named in the harness with the reason.

SciPy's `root` and `fsolve` report two evaluations more than ours on the same
path (the wrapper's initial and final calls; the `+2` is `row.extra`).

| Row | nfev ours / SciPy | nit ours / SciPy | |
|---|---|---|---|
| Minimize/BFGS/rosen2/grad | 39 / 39 | 32 / 32 | same |
| Minimize/BFGS/rosen100/grad | 644 / 645 | 544 / 538 | **different path** |
| Minimize/LBFGSB/rosen2/grad | 44 / 44 | 36 / 36 | same |
| Minimize/LBFGSB/rosen100/grad | 622 / 610 | 518 / 514 | **different path** |
| Minimize/BFGS/rosen2/fd | 117 / 117 | 32 / 32 | same |
| Minimize/BFGS/rosen100/fd | 65650 / 65852 | 544 / 565 | **different path** |
| Minimize/LBFGSB/rosen100/fd | 15049 / 15049 | 120 / 124 | **different path** (both stop at the 15000-evaluation limit) |
| Minimize/NelderMead/rosen2 | 159 / 159 | 85 / 85 | same |
| Minimize/Powell/rosen2 | 607 / 607 | 23 / 23 | same |
| Minimize/CG/rosen2/grad | 78 / 78 | 36 / 36 | same |
| Minimize/CG/rosen100/grad | 1781 / 1991 | 1134 / 1311 | **different path** |
| Minimize/NewtonCG/rosen2/hess | 105 / 105 | 83 / 83 | same |
| Minimize/NewtonCG/rosen100/hessp | 246 / 252 | 216 / 219 | **different path** |
| Minimize/SLSQP/rosen2/grad | 47 / 47 | 34 / 34 | same |
| Minimize/TrustConstr/rosen2/hess | 44 / 44 | 44 / 44 | same |
| LeastSquares/TRF/rosen | 25 / 25 | - | same |
| LeastSquares/LM/rosen | 21 / 21 | - | same |
| Leastsq/rosen | 54 / 54 | - | same |
| CurveFit/LM/1e4 | 21 / 21 | - | same |
| CurveFit/TRF/1e4 | 6 / 6 | - | same |
| CurveFit/TRF/1e4/bounds | 8 / 8 | - | same |
| CurveFitModel/LM/1e4/jac | 6 / 6 | - | same |
| Root/hybr/2 | 12 / 14 | - | same (+2) |
| Fsolve/2 | 12 / 14 | - | same (+2) |
| Root/hybr/broyden100 | 111 / 113 | - | same (+2) |
| Brentq, Brenth, Ridder, Bisect (cubic) | 8, 8, 14, 41 / equal | 7, 7, 6, 39 / equal | same |
| Newton (f′), secant, Halley | 8, 6, 9 / equal | 4, 5, 3 / equal | same |
| RootScalar/brentq | 8 / 8 | 7 / 7 | same |
| MinimizeScalar brent, golden, bounded | 14, 43, 12 / equal | 11, 38, 12 / equal | same |
| Fminbound/cos, Bracket | 8, 3 / equal | - | same |
| DifferentialEvolution sphere/2, rosenbrock/2 | 2823, 3903 / equal | 93, 129 / equal | same |
| Basinhopping/himmelblau/100 | 871 / 871 | 100 / 100 | same |
| Direct rosenbrock/2, sphere/10 | 835, 181 / equal | 44, 4 / equal | same |

Of the 43 rows compared, 37 take the same path. The six that do not are all
`n = 100` Rosenbrock rows, and every `n = 2` row is equal. The cause of the
divergence is not isolated; it is consistent with the last bit of a length-100
reduction (NumPy sums pairwise and calls BLAS, the Go side sums in its own
order), which the narrow valley amplifies over a few hundred iterations. For
these rows the time per evaluation, not the time per call, is the comparison of
the solvers.
