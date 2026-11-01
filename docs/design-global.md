# Design: global optimisation (`DifferentialEvolution`, `Basinhopping`, `Direct`)

Status: **implemented in v0.6.0** (tasks 3 to 10 of section 13; tasks 1 and 2
are go-ndarray and go-simd releases; the comparison of times, task 9, is built
but not yet measured, see docs/perf.md; sections 14 and 15 record what was
found on the way). This is the last slice of item 5 of the
go-scikit parity plan (`go-scikit/roadmap` PLAN.md, wave 2: "global methods
(differential evolution, basin hopping, DIRECT)"). It covers
`scipy.optimize.differential_evolution`, `scipy.optimize.basinhopping` and
`scipy.optimize.direct` (DIRECT and its locally biased variant DIRECT-L), and
says where `dual_annealing` and `shgo` go (§9). Everything below is written
so that the implementation can be done from this document and the cited
sources without re-deriving the design.

The fidelity policy of [design.md](design.md) applies unchanged: each routine
follows SciPy 1.18.1 statement by statement, every product that feeds a sum
is wrapped in `float64()` so Go cannot fuse it, and the oracle fixtures come
from SciPy's linux/amd64 wheel (no FMA), as for `tools/slsqporacle` and
`tools/qporacle`.

## 1. Sources

### Opened

| Source | What it gives |
|---|---|
| SciPy v1.18.1 `scipy/optimize/_differentialevolution.py` | `differential_evolution` and `DifferentialEvolutionSolver`: every step of §3, the draw order, the Lampinen selection, polishing, messages |
| SciPy v1.18.1 `scipy/optimize/_basinhopping.py` | `basinhopping`, `BasinHoppingRunner`, `AdaptiveStepsize`, `RandomDisplacement`, `Metropolis`, `Storage`: §4 |
| SciPy v1.18.1 `scipy/optimize/_direct_py.py`, `_directmodule.c`, `_directmodule.h` | the Python wrapper of DIRECT: validation, defaults, return codes and the message tables of §5 |
| SciPy v1.18.1 `scipy/optimize/_direct/` (`COPYING`, `README`, `AUTHORS`, `DIRect.c`, `DIRsubrout.c`, `DIRserial.c`, `direct_wrap.c`, `direct-internal.h`) | the C DIRECT code (Gablonsky's Fortran DIRECT 2.0.4, f2c-converted and cleaned up by S. G. Johnson for NLopt). **Licence: MIT** (Copyright 1999–2001 North Carolina State University, `COPYING`), compatible with this package's BSD-3-Clause; a port must carry the MIT notice (§10) |
| SciPy v1.18.1 `scipy/_lib/_util.py` | `check_random_state`, `_transition_to_rng`, `rng_integers`, `MapWrapper`: the seeding rules of §2.1 |
| SciPy v1.18.1 `scipy/optimize/_dual_annealing.py`, `scipy/optimize/_shgo.py` (structure and dependencies only) | the scoping decision of §9 |
| SciPy v1.18.1 `benchmarks/benchmarks/optimize.py` (`_BenchOptimizers.run_*`, `bench_run_global`) and `benchmarks/benchmarks/go_benchmark_functions/go_benchmark.py` (`Benchmark.success`) | the success criterion and the settings of SciPy's own global benchmark (§7.3) |
| NumPy v2.5.3 `numpy/random/_generator.pyx` (`_shuffle_raw`, `shuffle`, `permutation`, `uniform`, `integers`) and `numpy/random/src/distributions/distributions.c` (`random_uniform`, `random_interval`) | the draws SciPy makes (§2.2) |
| NumPy v2.5.3 `numpy/_core/src/umath/loops_utils.h.src` (`pairwise_sum`) | the summation order of `np.mean`/`np.std`/`np.sum` that DE's termination test depends on (§3.7) |
| NumPy v2.5.3 `numpy/_core/function_base.py` (`linspace`) | the Latin hypercube offsets (§3.3) |
| NumPy v2.5.3 `numpy/_core/_methods.py` (`_mean`, `_var`, `_std`) | `mean = sum/n`; `var = sum(square(x − mean))/n`; `std = sqrt(var)` (§3.7) |
| go-ndarray/ndarray `random/` (merged in #45, in every tag since v0.10.0; read at `main`): `doc.go`, `bitgen.go`, `bounded.go`, `generator.go`, `legacy.go` | PCG64, MT19937, SeedSequence, `Generator`, `RandomState` already port NumPy bit for bit; what is missing for this slice is listed in §2.3 |
| go-ndarray/ndarray `internal/kernels/kernels.go` (`Sum`) and `ndarray.go` (`Sum`, `Mean`) | the float64 sum is a SIMD/parallel sum, not NumPy's pairwise order (§3.7) |
| go-scikit/roadmap `bib/optimization-learning-autodiff.md` | the reference list (Storn & Price 1997, Wales & Doye 1997, Jones et al. 1993, Endres et al. 2018, Kirkpatrick et al. 1983), the MATLAB/Mathematica inventory, the "statistical: success rate over N seeds" validation rule for heuristics |
| arXiv cond-mat/9803344 (abstract page) | Wales & Doye, "Global Optimization by Basin-Hopping and the Lowest Energy Structures of Lennard-Jones Clusters Containing up to 110 Atoms", J. Phys. Chem. A 101, 5111–5116 (1997): the transformation that basin hopping performs |
| arXiv 1308.4008 (abstract page) | Jamil & Yang, "A literature survey of benchmark functions for global optimization problems", IJMMNO 4(2):150–194 (2013): 175 unconstrained test functions |
| MathWorks, `GlobalSearch` reference page | scatter-search start points (Ugray et al.) feeding `fmincon` (§11) |
| This repository at 48b89e7 (v0.5.0 + docs): `README.md`, `docs/design.md`, `docs/design-qp.md`, `minimize.go`, `types.go`, `constraints.go`, `brute.go`, `lbfgsb.go` | the conventions this design reuses (§2) |

### Not obtained

- **Storn & Price 1997** (J. Glob. Optim. 11:341) and the ICSI technical
  report TR-95-012: the ICSI server refused the connection. SciPy's code is
  the reference that is ported; the paper is cited through SciPy's docstring.
- **Jones, Perttunen & Stuckman 1993** (JOTA 79:157) and **Gablonsky & Kelley
  2001** (J. Glob. Optim. 21:27): paywalled; not opened. The C code, its
  comments and its README are the reference ported. Gablonsky's DIRECT 2.0.4
  user guide (`userguide.pdf`, mentioned in the README) is not in SciPy's tree
  and was not obtained.
- **Lampinen 2002** (the constraint-handling rule cited by
  `_accept_trial`): not opened; SciPy's implementation is what is specified.
- **Wales & Doye 1997** full text: only the abstract page was opened.
- **CEC benchmark suites**: not opened; not used (§7.3 uses SciPy's own set).
- NumPy's `mtrand.pyx` (legacy `RandomState.uniform`, `randint`, `shuffle`)
  was **not** opened in this step; go-ndarray's `legacy.go` was. Task 1
  opens it before writing the legacy scalar draws.
- MATLAB's `ga`, `particleswarm`, `simulannealbnd`, `MultiStart` pages were
  not opened; §11 relies on the roadmap inventory for their names only.

## 2. Shared design decisions

### 2.1 Seeding: what SciPy actually does

`differential_evolution` and `basinhopping` are wrapped by
`_transition_to_rng("seed", ...)` (`_util.py`). With no `end_version` the
decorator emits no warning, and:

| Call | What the solver receives | Bit generator |
|---|---|---|
| `rng=42` (keyword) | `np.random.default_rng(42)` → `Generator(PCG64(SeedSequence(42)))` | PCG64 |
| `seed=42` (old keyword) or positional | `42`, then `check_random_state(42)` → `np.random.RandomState(42)` | MT19937, legacy seeding |
| `rng=None` / nothing | `check_random_state(None)` → NumPy's global `RandomState` singleton | MT19937, unseeded |
| a `Generator` or `RandomState` | used as is | either |

Both seeded forms are common in notebooks (`seed=1` in older code, `rng=1` in
newer code), and they give different numbers. `basinhopping` calls
`check_random_state(rng)` once (line 659) and passes the **same** object to
`RandomDisplacement` and `Metropolis`, so its displacements and its
acceptance draws interleave on one stream in both cases.

### 2.2 RNG decision: reproduce SciPy's path exactly

**Decision: DE and basin hopping reproduce SciPy's exact random stream, for
both PCG64 (`rng=int`) and MT19937 (`seed=int`). DIRECT is deterministic and
needs no RNG.** Justification:

1. **The bit generators already exist.** go-ndarray/ndarray's `random`
   package ports SeedSequence, PCG64, MT19937 (with legacy seeding),
   `Generator` and `RandomState`, and matches NumPy 2.5.3 bit for bit on
   integers and arithmetic-only samplers (its `doc.go`, `docs/random.md`). The
   cost of exactness is therefore a few scalar entry points (§2.3), not a
   PRNG port.
2. **Every draw these two solvers make is arithmetic-only**: `uniform`
   (`low + range*next_double`, `distributions.c:394`), `integers` (Lemire's
   bounded method for `Generator`, masked rejection for `RandomState`),
   `shuffle`/`permutation` (`random_interval`, masked rejection,
   `_generator.pyx:75-105`, `distributions.c:1084`). No `log`/`exp`, so no
   libm ulp can break the stream.
3. **Exactness turns a statistical test into an equality test.** With the
   same stream, DE on an arithmetic-only objective must give SciPy's final
   population bit for bit — a check that detects a wrong draw order, a wrong
   mutation formula or a wrong termination test in one fixture, where a
   success-rate comparison needs hundreds of runs to see a small bias.
4. **Users can replay a notebook.** `differential_evolution(f, b, seed=1)`
   ported to Go gives the same answer, not a statistically equivalent one.

Statistical validation is kept as a second, independent layer (§7.2), for
objectives with transcendental functions (where a last-ulp libm difference
can flip one comparison and the path then diverges legitimately) and to
guard against the case where SciPy and we share a bug.

The draws, in the order SciPy makes them (this order is part of the
specification; a test plants a swap and must fail, §7.4):

| Where | SciPy call | NumPy primitive |
|---|---|---|
| DE LHS init | `rng.uniform(size=(S, N))`, then for each `j < N`: `rng.permutation(range(S))` | S·N × `next_double`; N Fisher–Yates shuffles with `random_interval` |
| DE random init | `rng.uniform(size=(S, N))` | S·N × `next_double` |
| DE dither, once per generation | `rng.uniform(dither[0], dither[1])` | 1 × `next_double` |
| DE immediate, per candidate | `rng_integers(rng, N)` (fill point) **then** `_select_samples` = `rng.shuffle(idx)` of `arange(S)` **then** `rng.uniform(size=N)` (crossover) **then**, in `_ensure_constraint`, `rng.uniform(size=oob)` | Lemire (Generator) / masked (RandomState) bounded int; shuffle; N doubles; oob doubles |
| DE deferred, per generation | for each candidate in order: `rng.shuffle(idx)`; then `rng_integers(rng, N, size=S)`; then `rng.uniform(size=(S, N))`; then `rng.uniform(size=oob)` over the whole trial population (row-major mask order) | as above, batched |
| BH, per step | `rng.uniform(-stepsize, stepsize, n)` (default step), local minimisation, then `rng.uniform()` in `Metropolis` — **only if** no earlier accept test returned `"force accept"` | doubles |

Note the immediate/deferred difference: immediate draws the fill point
*before* the shuffle; deferred shuffles for all candidates first.

### 2.3 Prerequisite in go-ndarray: allocation-free scalar draws

go-ndarray's `Generator` and `RandomState` return `*ndarray.Array` from every
sampler; the bounded and interval helpers (`boundedUint64`, `interval`,
`uint64ToDouble`) are unexported. DE draws a few values per candidate, so an
array allocation per draw would dominate a cheap objective. Per the fleet
rule (optimisations go to the lowest shared library), task 1 adds to
`go-ndarray/ndarray/random`, for both `*Generator` and `*RandomState`:

```go
func (g *Generator) Float64() float64                         // random() scalar: next_double
func (g *Generator) FillUniform(dst []float64, low, high float64) error // uniform(low, high, size=len(dst)), row-major
func (g *Generator) IntN(high int64) int64                     // integers(0, high) scalar, dtype int64 (rng_integers(rng, high))
func (g *Generator) FillIntN(dst []int64, high int64) error    // integers(0, high, size=len(dst))
func (g *Generator) ShuffleInt64(d []int64)                    // shuffle of a 1-D int64 array: _shuffle_raw with random_interval
// and the same five on *RandomState, with its legacy semantics
// (randint's masked rejection; legacy uniform accepts high < low).
```

Each is tested against NumPy 2.5.3 fixtures in go-ndarray's own
`random/testdata/cases.py`, including the PCG64 half-word buffering of
`next_uint32` across mixed `IntN`/`Float64` calls (the 32-bit Lemire path
draws half of a 64-bit output and keeps the other half).

### 2.4 Seeding API

```go
// RandomSource is the random stream of a stochastic solver. Exactly one of
// a NumPy Generator or a legacy RandomState is behind it.
type RandomSource struct{ /* unexported: g *random.Generator; r *random.RandomState */ }

func NewRNG(seed uint64) *RandomSource          // SciPy rng=seed: Generator(PCG64(SeedSequence(seed)))
func NewLegacyRNG(seed uint32) *RandomSource    // SciPy seed=seed: RandomState(seed), MT19937
func RNGFromGenerator(g *random.Generator) *RandomSource
func RNGFromRandomState(r *random.RandomState) *RandomSource
```

`RNG: nil` in the options means an entropy-seeded PCG64 (`random.DefaultRNG()`
with no seed); it is documented as non-reproducible (SciPy's `None` uses the
global `RandomState`, which is not reproducible either unless the caller
seeded it). A `RandomSource` is not safe for concurrent use; the solvers draw
only on the calling goroutine (§2.5).

### 2.5 Parallel evaluation, deterministic for any worker count

One helper, `evalBatch(f, points [][]float64, out []float64, workers int)`,
in `parallel.go`:

- `workers <= 1`: a plain loop, in index order.
- `workers > 1`: `workers` goroutines take indices from an atomic counter and
  write `out[i] = f(points[i])`. Each result lands in its own slot, so the
  values do not depend on scheduling. All **random draws and all
  comparisons happen on the calling goroutine** before and after the batch.
- A panic in the objective is recovered in the worker, the batch is drained,
  and the panic is re-raised **on the calling goroutine** with the original
  value, so a panicking objective behaves the same with 1 or 8 workers
  (a panic in a bare goroutine would kill the process, uncatchable by the
  caller).
- The objective must be safe for concurrent use when `Workers > 1`
  (documented, as `BruteOptions.Workers` already says).

Where each method uses it:

- **DE**: only with deferred updating, where a generation's trials are
  independent. As in SciPy, `Workers > 1` with the default updating switches
  to deferred (SciPy warns and overrides `updating='immediate'`). With
  `Updating: DEDeferred` given explicitly, **results are bit-identical for
  every `Workers` value** — asserted by a test (§7.4). Constraint functions are
  evaluated serially (SciPy evaluates them in a Python loop too).
- **DIRECT**: within one hyperrectangle, `DIRsamplef` evaluates all `2·maxi`
  new points first and only then scans them for a new minimum in list order
  (`DIRserial.c:74-155`). The `2·maxi` evaluations can therefore run in
  parallel without changing any decision. Batching across the rectangles of
  one iteration is *not* done in this slice (it would need a proof that
  `DIRdivide`/`DIRinsertlist` of rectangle j cannot affect rectangle j+1's
  sample points).
- **Basin hopping**: strictly sequential (each step starts from the last
  accepted point); no `Workers` (SciPy has none).
- **Brute** already evaluates its grid with `Workers` goroutines writing by
  index (brute.go at 48b89e7). The 2026-10-17 note that it ignored `workers`
  no longer applies to the current code. Task 4 moves it onto `evalBatch` so
  it also gets the panic propagation; no behaviour change otherwise.

### 2.6 Option conventions

Zero values select SciPy's defaults, as everywhere in the package. Where 0 is
a meaningful SciPy value *and* the default is not 0 (DE `tol`, `recombination`;
basin hopping `niter`, `T`), the field is a pointer, with helpers
`optimize.Float(v) *float64` and `optimize.Int(v) *int`. This is new to the
package; the alternative (a negative sentinel, as `LSOptions` uses for a
disabled tolerance) cannot express `T = 0` (monotonic basin hopping) or
`niter = 0` without inventing a meaning for negative values that SciPy gives
another meaning to (`T < 0` is accepted by SciPy).

## 3. Differential evolution

### 3.1 API

```go
type DEStrategy string // the 12 SciPy names
const (
	Best1Bin DEStrategy = "best1bin" // default
	Best1Exp, Rand1Bin, Rand1Exp, RandToBest1Bin, RandToBest1Exp,
	CurrentToBest1Bin, CurrentToBest1Exp, Best2Bin, Best2Exp, Rand2Bin, Rand2Exp
)

type DEInit int
const (
	DEInitLatinHypercube DEInit = iota // default ('latinhypercube')
	DEInitRandom                       // 'random'
	// 'sobol' and 'halton' need scipy.stats.qmc; they return ErrInput
	// "not implemented" until a QMC engine exists in the fleet (§9).
)

type DEUpdating int
const (
	DEUpdatingDefault DEUpdating = iota // immediate, or deferred when Workers > 1 (SciPy)
	DEImmediate
	DEDeferred
)

type DEOptions struct {
	Strategy       DEStrategy
	// CustomStrategy is SciPy's callable strategy: it writes the trial (in
	// the problem's units) for candidate into trial. population is S×N in
	// the problem's units; it must not be retained.
	CustomStrategy func(candidate int, population [][]float64, rng *RandomSource, trial []float64)
	MaxIter        int      // 1000
	MaxFev         int      // 0: unlimited (DifferentialEvolutionSolver's maxfun; the public SciPy function does not expose it)
	PopSize        int      // 15
	Tol            *float64 // 0.01
	Atol           float64  // 0
	Mutation       []float64 // nil: (0.5, 1) dithered; one value: constant; two: dither range (sorted)
	Recombination  *float64 // 0.7
	RNG            *RandomSource
	Init           DEInit
	InitPopulation [][]float64 // SciPy init=array: S×N, S >= 5; clipped to the bounds
	X0             []float64
	Updating       DEUpdating
	Workers        int
	Constraints    []ConstraintSpec // Constraint, NonlinearConstraint, LinearConstraint
	Integrality    []bool           // nil: none; length N
	NoPolish       bool             // SciPy polish=False
	// Polish replaces the default polisher (L-BFGS-B, or trust-constr with
	// constraints) — SciPy's callable polish. It receives the bounds the
	// default would use.
	Polish   func(f func([]float64) float64, x0 []float64, bounds []Bound, cons []ConstraintSpec) (*Result, error)
	Callback func(r *DEResult) (stop bool) // after each generation; r.Convergence = tol/(convergence+eps)
}

type DEResult struct {
	X                  []float64
	Fun                float64
	Jac                []float64 // from the polisher, when it improved the point
	Nfev, Nit          int
	Status             int // 0 converged, 1 maxiter, 2 maxfev, 3 callback, 4 infeasible (Go-defined; SciPy has no status)
	Success            bool
	Message            string
	Population         [][]float64 // S×N, problem units
	PopulationEnergies []float64
	Constr             [][]float64 // per constraint, its violation at X (only with constraints)
	ConstrViolation    float64     // max violation (SciPy constr_violation and maxcv)
	Convergence        float64     // set only in callbacks
}

func DifferentialEvolution(f func(x []float64) float64, bounds []Bound, o *DEOptions) (*DEResult, error)
```

Not provided: `vectorized` (it exists to amortise Python call overhead; it
also changes `nfev` to count calls instead of points), `disp`, `args`.
SciPy accepts only `NonlinearConstraint`, `LinearConstraint` and `Bounds` as
DE constraints; the Go function also accepts the dict-style `Constraint`
through its existing `general()` form (Eq → lb = ub = 0, Ineq → [0, ∞)). This
is an extension; it changes nothing for the other two types.

### 3.2 Validation (errors wrapping `ErrInput`/`ErrBounds`, never panics)

In SciPy's order (`__init__`, lines 813–1026):

1. strategy known (or `CustomStrategy` set);
2. every mutation value finite, in `[0, 2)`; two values are sorted (`dither`);
3. bounds: `len(bounds) = N >= 1`, every `Lo`, `Hi` finite (`np.isfinite`),
   `Lo <= Hi`;
4. `MaxIter >= 0`, `MaxFev >= 0`, `PopSize >= 1`, `0 <= recombination <= 1`
   (SciPy does not check recombination; a value outside [0, 1] only changes
   the `<` comparison — accepted for parity, documented);
5. integrality: `lb = ceil(Lo)`, `ub = floor(Hi)` on integral axes; error
   "One of the integrality constraints does not have any possible integer
   values between the lower/upper bounds." if `lb > ub`; then the limits of
   integral axes become `nextafter(lb − 0.5, +∞)` and `nextafter(ub + 0.5, −∞)`;
6. population size, with overflow checks (§8):
   `S = max(5, popsize · max(1, N − eb))`, `eb` = count of axes with
   `Lo == Hi` (after the integrality change);
7. `InitPopulation`: `S_init >= 5` rows of length N, else
   "The population supplied needs to have shape (S, len(x)), where S > 4.";
8. `X0`: length N; scaled `x0_scaled = (x0 − m)·r + 0.5` must lie in [0, 1],
   else "Some entries in x0 lay outside the specified bounds".

### 3.3 State and scaling

The population is held in the unit cube, `pop[S][N]`. With
`m = 0.5·(Lo + Hi)`, `w = |Lo − Hi|` and `r = 1/w` (`r = 0` where `w = 0`,
lines 912–919):

- scale (unit → problem): `x = m + (t − 0.5)·w`, then `round` (NumPy
  round-half-to-even, Go `math.RoundToEven`) on integral axes;
- unscale (problem → unit): `t = (x − m)·r + 0.5`.

Both products are rounded explicitly (`float64(...)`).

Initialisation (each sets energies to `+Inf` and `nfev = 0`):

- **LHS** (`init_population_lhs`): `seg = 1.0/S`;
  `samples[i][j] = float64(seg·u[i][j]) + y_i` with `u` drawn row-major by
  `uniform(size=(S, N))` and `y_i = float64(i)·(1.0/S) + 0.0` (NumPy
  `linspace(0, 1, S, endpoint=False)`: `step = delta/div`, `y *= step`,
  `y += start`); then for `j = 0..N−1`: `order = permutation(S)` and
  `pop[:, j] = samples[order, j]`.
- **random**: `pop = uniform(size=(S, N))`.
- **array**: `pop = clip(unscale(InitPopulation), 0, 1)`; S becomes its row
  count.
- `X0` then overwrites `pop[0]`.

### 3.4 Solve loop (`solve`, lines 1185–1341)

```
if every energy is ±Inf:                       # np.all(np.isinf(...))
    feasible, cv = population_feasibilities(pop)
    energies[feasible] = population_energies(pop[feasible])
    promote_lowest()
warning = false; msg = "Optimization terminated successfully."
for nit = 1 .. MaxIter:
    if not next_generation():                  # StopIteration (maxfev)
        warning = true
        msg = (nfev > MaxFev) ? "Maximum number of function evaluations has been exceeded."
            : (nfev == MaxFev) ? "Maximum number of function evaluations has been reached."
            : msg                                # (unreachable in practice)
        break
    if Callback != nil:
        res = intermediate result (nit, "in progress"), res.Convergence = tol/(convergence()+eps)
        if Callback(res): warning = true; msg = "callback function requested stop early"
    if warning or converged(): break
else (loop ran to MaxIter without break):
    msg = "Maximum number of iterations has been exceeded."; warning = true
result = intermediate(nit, msg, success = !warning)
polish (§3.6); constraint report (§3.5)
```

`nit` is the loop variable at exit (with `MaxIter = 0`, SciPy's `nit` stays 0
and the `else` branch runs: message "Maximum number of iterations has been
exceeded.", success false). `next_generation` also re-runs the initial
evaluation if all energies are still infinite (lines 1614–1624) — a second
copy of the block above, kept because `MaxIter = 0` never enters the loop.

### 3.5 One generation (`__next__`, lines 1601–1724)

First, with dithering: `scale = uniform(dither[0], dither[1])` (one draw per
generation, before any candidate).

**Immediate** (default with one worker), for `c = 0 .. S−1` in order:

1. `if nfev > MaxFev: stop` (strict `>`: up to `MaxFev + 1` evaluations);
2. `trial = mutate(c)` (§3.5.1); `ensure_constraint(trial)`: every component
   `> 1` or `< 0`, in index order, is replaced by one `uniform()` draw
   (one `uniform(size=oob)` call — same doubles in the same order);
3. `x = scale(trial)`;
4. with constraints: `cv = violations(x)` (§3.5.2);
   `feasible = !(sum(cv) > 0)` (so a NaN violation counts as **feasible**,
   parity); energy is evaluated only when feasible (`nfev += 1`), else `+Inf`;
   without constraints: `cv = [0]`, evaluate;
5. if `accept(trial_vs_c)` (§3.5.3): replace member c (vector, energy,
   feasibility, cv); and if additionally `accept(trial_vs_0)`, run
   `promote_lowest()`. Later candidates of the same generation see the
   updated population (in particular a new `pop[0]`).

**Deferred** (or `Workers > 1`):

1. `if nfev >= MaxFev: stop` (note `>=` here);
2. `trials = mutate_many(0..S−1)` (§3.5.1, batched draws);
3. `ensure_constraint(trials)` over the S×N block, row-major;
4. `feasible, cv = population_feasibilities(trials)`;
5. energies of feasible trials via `population_energies` (§3.5.4), the
   others `+Inf`;
6. `loc[i] = accept(trial_i vs member_i)` for every i, then replace all
   accepted members at once; then `promote_lowest()`.

#### 3.5.1 Mutation and crossover

`select_samples(c, 5)`: shuffle the persistent index array `idx` (initially
`0..S−1`; it is **not** reset between calls — the permutation state carries
over, parity); take `idx[:6]`, drop `c` if present, keep the first 5.

Mutant `b` (all arithmetic elementwise, left to right, each product rounded):

| Strategy family | `b` |
|---|---|
| best1 | `p[0] + F·(p[r0] − p[r1])` |
| rand1 | `p[r0] + F·(p[r1] − p[r2])` |
| randtobest1 | `b = p[r0]; b += F·(p[0] − b); b += F·(p[r1] − p[r2])` |
| currenttobest1 | `p[c] + F·(((p[0] − p[c]) + p[r0]) − p[r1])` |
| best2 | `p[0] + F·(((p[r0] + p[r1]) − p[r2]) − p[r3])` |
| rand2 | `p[r0] + F·(((p[r1] + p[r2]) − p[r3]) − p[r4])` |

In deferred mode `p[0]` is the generation's start value for all candidates.

Crossover, with `cross[j] = (u_j < CR)` from `uniform(size=N)`:

- **binomial**: `cross[fill] = true`; `trial[j] = cross[j] ? b[j] : p[c][j]`;
- **exponential**: `cross[0] = true` (index 0, *not* `fill` — parity); then
  `i = 0; k = fill; while i < N && cross[i]: trial[k] = b[k]; k = (k+1) % N; i++`.

Draw order (immediate): `fill = IntN(N)`, then the shuffle, then the N
crossover doubles; deferred: S shuffles, then `FillIntN(S)`, then S·N doubles.

`CustomStrategy` receives the population in problem units and must return a
trial in problem units, which is unscaled; a wrong length is an error (SciPy:
RuntimeError "strategy must have signature …").

#### 3.5.2 Constraint violations (`_ConstraintWrapper.violation`)

Every constraint, in general form `lb <= g(x) <= ub` (the package's
`genConstraint`): `v_k = max(lb_k − g_k, 0) + max(g_k − ub_k, 0)`, with
`max` as NumPy's `np.maximum` (NaN-propagating: Go's builtin `max`, not
`math.Max`). The per-member violation row concatenates all constraints'
components (`total_constraints` columns). `feasible = !(sum(row) > 0)`.

#### 3.5.3 Selection (Lampinen, `_accept_trial`)

```
if feasible_orig && feasible_trial:   accept = energy_trial <= energy_orig
elif feasible_trial && !feasible_orig: accept = true
elif !feasible_trial && all(cv_trial <= cv_orig): accept = true
else: accept = false
```

A NaN energy is never accepted against a feasible member (`<=` is false). The
third case accepts an infeasible trial over a **feasible** original when every
component of its violation is `<=` the original's (all zeros) — impossible
for a truly infeasible trial unless a component is NaN; ported as written.

`promote_lowest()`: among feasible members, `l = argmin(energies)` with
NumPy's `argmin` rule (first minimum; **the first NaN wins**, as `Brute`
already implements); if none is feasible, `l = argmin(sum of cv per row)`.
Swap member 0 and l (vector, energy, feasibility, cv row).

#### 3.5.4 Energies with an evaluation budget

`population_energies(P)`: `k = min(len(P), MaxFev − nfev)` (unlimited:
`len(P)`); evaluate the first k points (`evalBatch`), the rest are `+Inf`;
`nfev += k`.

### 3.6 Polishing

If not `NoPolish` and not every axis is integral:

- integral axes are pinned: `limits[0][i] = limits[1][i] = result.X[i]`
  (SciPy modifies its limits in place; replicated on a copy — the solver is
  not reused after `solve`, so the effect is the same);
- method: `LBFGSB`, or `TrustConstr` when there are constraints (SciPy warns
  when the best point is infeasible; the Go result's `Message` is left as
  the solver set it, the warning text is not reproduced);
- call `Minimize(Problem{Func: f}, copy(result.X), method,
  &Options{Bounds: limits, Constraints: cons})` — no gradient, so finite
  differences with SciPy's step rules, as SciPy's `minimize` does;
- `nfev += r.Nfev` (always, even if the polish is rejected);
- accept only if `r.Fun < result.Fun && r.Success && Lo <= r.X <= Hi`
  (componentwise, limits as modified); then `X, Fun, Jac` from `r`, and
  `pop[0] = unscale(r.X)`, `energies[0] = r.Fun`.

Then, with constraints: `Constr[k] = violation_k(X)`,
`ConstrViolation = max over all components`; if `> 0`: `Success = false`,
`Status = 4`, `Message = "The solution does not satisfy the constraints,
MAXCV = " + pyFloat(maxcv)` (the package's `pyFloat` gives Python's float
repr).

### 3.7 Convergence test, exactly

```
converged:   if any energy is ±Inf → false
             else std(E) <= atol + tol·|mean(E)|
convergence: if any energy is ±Inf → +Inf
             else std(E) / (|mean(E)| + eps)          (eps = 2.220446049250313e-16)
```

`mean` and `std` are NumPy's: `mean = S_pw(E)/n`; `std = sqrt(S_pw(d·d)/n)`
with `d_i = E_i − mean` (`np.std`, `ddof = 0`, via `_methods._var`), where
`S_pw` is NumPy's **pairwise summation** (`loops_utils.h.src:81-144`): for
`n < 8` a left-to-right sum starting at `−0.0`; for `n <= 128` eight
accumulators over blocks of 8, combined as
`((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))`, then the remainder added left to right;
above 128, split at `n2 = n/2 − (n/2 mod 8)` and recurse. The termination
decision compares two rounded quantities; a different summation order moves
it by one generation on some runs and the whole path then differs. go-ndarray's
float64 `Sum` is a SIMD/parallel sum, not this order, so task 2 adds
`SumPairwise(a []float64) float64` (NumPy's order) to **go-simd/floats**: the
eight-accumulator block maps onto 2×4- or 4×2-lane vectors without changing
the order (NumPy's own comment says the unrolling was chosen for that).
The per-row `sum(cv)` uses it too.

`_nfev`, `nit`, messages and `success` follow §3.4 exactly.

## 4. Basin hopping

### 4.1 API

```go
type Acceptance int
const (Reject Acceptance = iota; Accept; ForceAccept)

// StepTaker displaces x in place. If it also implements StepSizer, it is
// wrapped by the adaptive step-size controller (SciPy wraps any take_step
// that has a stepsize attribute).
type StepTaker interface{ TakeStep(x []float64) }
type StepSizer interface{ StepSize() float64; SetStepSize(float64) }

type BasinhoppingOptions struct {
	NIter            *int     // 100
	T                *float64 // 1.0; 0 is monotonic basin hopping (beta = +Inf)
	StepSize         float64  // 0.5
	Method           Method   // local minimiser; "" = minimize's default: BFGS, L-BFGS-B with Bounds, SLSQP with Constraints
	MinimizeOptions  *Options // SciPy minimizer_kwargs (Bounds, Constraints, tolerances ...)
	TakeStep         StepTaker
	// AcceptTest is SciPy's accept_test(f_new, x_new, f_old, x_old);
	// AcceptTestResult its new form accept_test(res_new, res_old). At most one.
	AcceptTest       func(fNew float64, xNew []float64, fOld float64, xOld []float64) Acceptance
	AcceptTestResult func(resNew, resOld *Result) Acceptance
	Callback         func(x []float64, f float64, accepted bool) (stop bool)
	Interval         int      // 50
	NIterSuccess     int      // 0: niter + 2 (never triggers)
	RNG              *RandomSource
	TargetAcceptRate float64  // 0.5, open interval (0, 1)
	StepwiseFactor   float64  // 0.9, open interval (0, 1)
}

type BasinhoppingResult struct {
	X                    []float64
	Fun                  float64
	Nit                  int
	Nfev, Njev, Nhev     int
	MinimizationFailures int
	Lowest               *Result // SciPy lowest_optimization_result
	Success              bool
	Message              string
}

func Basinhopping(p Problem, x0 []float64, o *BasinhoppingOptions) (*BasinhoppingResult, error)
```

### 4.2 Algorithm (`basinhopping`, `BasinHoppingRunner`)

Validation: `TargetAcceptRate`, `StepwiseFactor` in (0, 1) (SciPy's messages);
`Interval >= 1` (SciPy divides by it: `nstep % interval`, a ZeroDivisionError
at 0); `StepSize >= 0` (NumPy's `Generator.uniform` rejects a negative range
at the first step, after the initial minimisation — the Go function refuses
it up front; documented difference); `NIter >= 0`.

```
minres = minimise(x0); failures += !minres.Success
x, energy = minres.X, minres.Fun; incumbent = minres; storage = copy(minres)
nfev, njev, nhev = minres counts
callback(storage.X, storage.Fun, true)               # its return value is ignored here (SciPy)
count = 0; i = 0; msg = "requested number of basinhopping iterations completed successfully"
for i = 0 .. niter−1:
    nstep += 1
    # Monte-Carlo step
    xs = copy(x); step(xs)                           # adaptive wrapper, §4.3
    minres = minimise(xs); failures += !minres.Success; add counts
    accept = true
    for test in [user test (if any), metropolis]:
        r = test(...)
        if r == ForceAccept: accept = true; break    # Metropolis not called → no draw
        if r == Reject: accept = false               # keep going: Metropolis still draws
    stepper.report(accept)
    if accept:
        energy = minres.Fun; x = copy(minres.X); incumbent = minres
        newGlobal = storage.update(minres)           # minres.Success && (minres.Fun < storage.Fun || !storage.Success)
    if callback != nil && callback(minres.X, minres.Fun, accept):
        msg = "callback function requested stop early byreturning True"   # SciPy's missing space, kept
        break
    count += 1
    if newGlobal: count = 0
    elif count > niterSuccess: msg = "success condition satisfied"; break
res.X, res.Fun = storage; res.Nit = i + 1; res.Success = storage.Success
```

`newGlobal` is false on steps that are not accepted. `Nit = i + 1` holds the
Python loop variable: with `niter = 0` SciPy reports `nit = 1`; replicated.

**Metropolis** (`accept_reject`): `beta = 1/T` (`+Inf` when `T == 0`);
`prod = −(fNew − fOld)·beta`; `w = exp(min(0, prod))` with **Python's `min`**,
which returns its first argument when the comparison is false: `prod` NaN
(`0·Inf` when `T = 0` and the energies are equal) gives `w = exp(0) = 1`. In Go:
`if prod < 0 { w = math.Exp(prod) } else { w = 1 }`. Then one draw
`u = Float64()` (always drawn when Metropolis is reached) and
`accept = w >= u && (resNew.Success || !resOld.Success)`. `resOld` is the
incumbent (last accepted) result, not the stored lowest.

`math.Exp` vs the C library: the only transcendental call in basin hopping's
own logic; a last-ulp difference flips `w >= u` with probability ~1e-16 per
step. Logged, not engineered around.

### 4.3 Steps

- Default step: `RandomDisplacement`: `x += uniform(−s, s, n)` elementwise,
  where each value is `−s + (2s)·next_double` (`low + range·d`, range computed
  once as `high − low`).
- `AdaptiveStepsize` wraps the default step and any `StepTaker` that is also a
  `StepSizer`: before each step `nstep += 1`; if `nstep % interval == 0`:
  `rate = naccept/nstep` (**cumulative** since the start, not per window);
  `rate > target ? s /= factor : s *= factor`. `report(accept)` increments
  `naccept`. The verbose print is not reproduced.

### 4.4 Local minimiser

`Minimize(p, xs, method, MinimizeOptions)` with SciPy's default method rule
when `Method == ""`: SLSQP if `MinimizeOptions.Constraints` is set, else
L-BFGS-B if `Bounds` is set, else BFGS (`scipy.optimize.minimize`). A
`Minimize` error (invalid input) aborts and is returned; non-convergence
counts as a minimisation failure, as in SciPy.

## 5. DIRECT and DIRECT-L

### 5.1 API

```go
type DirectOptions struct {
	Eps      float64  // 1e-4 (0 selects it); negative: Jones' update eps = max(1e-4·|minf|, |Eps|) after each iteration (the C code's convention, which SciPy passes through)
	MaxFev   int      // 0: 1000·N
	MaxIter  int      // 0: 1000
	Original bool     // false: DIRECT-L (SciPy locally_biased=True, the default); true: Jones' DIRECT
	FMin     *float64 // known global minimum; nil: -Inf (criterion off)
	FMinRtol float64  // 1e-4; negative: off
	VolTol   float64  // 1e-16; negative: off
	LenTol   float64  // 1e-6; negative: off
	Workers  int
	Callback func(xLast, xBest []float64, fBest float64) (stop bool) // stop → status -102 "Forced stop"
}

type DirectResult struct {
	X       []float64
	Fun     float64
	Nfev    int
	Nit     int
	Status  int // SciPy's ret_code: 1..5, or negative
	Success bool // Status > 2
	Message string
}

func Direct(f func(x []float64) float64, bounds []Bound, o *DirectOptions) (*DirectResult, error)
```

SciPy's callback receives `x_seq`, which holds the **last evaluated point**,
not the best (`DIRsubrout.c:1085-1088`, `DIRect.c:775-784`), although its
docstring says "best". The Go callback receives both. SciPy's `maxfun=0`
(which the C code turns into 100 000) and `maxiter=0` are not reachable
through the Go options (0 selects the default); documented.

### 5.2 Validation (`_direct_py.py:202-245`)

`N >= 1` ("DIRECT requires at least one dimension"); every `Lo < Hi`
("Bounds are not consistent min < max"); no infinite bound ("Bounds must not
be inf."); `VolTol`, `LenTol`, `FMinRtol` in [0, 1] when given; `MaxFev`,
`MaxIter` non-negative. NaN bounds fail `Lo < Hi`.

### 5.3 Port: what to follow

Port `DIRect.c`, `DIRsubrout.c` and `DIRserial.c` statement by statement,
keeping the 1-based index arithmetic in helper accessors rather than
rewriting it (that is where a port goes wrong). `algmethod = !Original`
(`jones = 1` is DIRECT-L/Gablonsky, `0` is Jones' original). The arguments
the Python layer passes: `eps`, `maxf = maxfun`, `maxT = maxiter`,
`fglobal = f_min` (−∞ default), `fglper = f_min_rtol`, `volper = vol_tol`,
`sigmaper = len_tol`, `magic_eps_abs = 0`, and `direct_optimize` turns
`volper <= 0`, `sigmaper <= 0` into −1 (disabled) and, when
`fglobal == −∞`, `fglper` into 0.

Points the port must keep, each with a fixture or a unit test:

1. **Sizes and the evaluation budget** (`DIRect.c:94-145`):
   `MAXFUNC = maxf <= 0 ? 101000 : maxf + 1000 + maxf/2` (integer division),
   capped so `MAXFUNC·(N·(8+4) + 8·2 + 4) <= 2^30` bytes; then
   `maxf = maxf <= 0 ? MAXFUNC − 1000 : 2·(MAXFUNC − 1000)/3`. **This rounds
   some budgets down by one** (`maxfun = 1001` becomes 1000). `MAXDEEP =
   maxT <= 0 ? MAXFUNC/5 : maxT + 1000` (capped the same way), and
   `maxT = MAXDEEP − 1000`. `MAXDIV = 5000`. Compute all of it in `int64`
   (the C code overflows `int` for `maxfun > ~1.43e9`).
2. **`DIRheader`**: `eps < 0` → `iepschange = 1`, `epsfix = eps = −eps`;
   else `epsfix = 1e100`. `maxf + 20 > MAXFUNC` → error −2.
3. **Scaling** (`DIRpreprc`): `xs1 = u − l`, `xs2 = l/(u − l)`; points are
   evaluated at `(c + xs2)·xs1` (`DIRinfcn`), but the **returned x** is
   `c·xs1 + xs1·xs2` (`DIRect.c:805`) — a different rounding; both kept, each
   product rounded explicitly.
4. **Initialisation** (`DIRinit`): `thirds[i] = 1/3^i` (`help2 *= 3`
   repeatedly); `levels` = `thirds` for DIRECT-L, and for the original
   `levels[(i−1)·n + j] = w_j / 3^(i−1)` with `w_j = 0.5·sqrt(n − j + j/9.0)`.
   Centre evaluation, then `2·maxi` samples, `DIRdivide`, `DIRinsertlist`.
   `numfunc = 2·maxi + 1`.
5. **Main loop**: `for t = 2; t <= maxT − 1; t++` — so `maxiter = 1000` runs at
   most 998 division rounds; `Nit` is `t` at exit (`maxT` when the loop ends,
   status 2). The budget is checked **once per iteration, after all its
   divisions** (`numfunc > maxf`), so `Nfev` can exceed `MaxFev` by a whole
   iteration's samples (SciPy behaves the same; the docstring says
   "approximate upper bound").
6. **Termination order after each iteration**: volume (`thirds[level_jones0(minpos)]
   <= volper` → 4), side measure (`levels[level(minpos)] <= sigmaper` → 5),
   known minimum (`(minf − fglobal)/divfactor <= fglper` → 3, `divfactor =
   |fglobal|` or 1 when `fglobal == 0`; with `fglobal = −∞` this is `NaN <= 0`,
   false), then infeasible-point replacement, the Jones eps update, the budget
   (→ 1, or a budget extension while no feasible point exists), then the
   callback. Maximum depth (`actdeep + 1 >= mdeep`) → −6.
7. **`DIRchoose`** is the convex-hull selection; port its two loops and the
   `goto L60` exits exactly, including the comparison
   `f − K·thirds[level] > min(minf − eps·|minf|, minf − epsabs)`.
8. **NaN and Inf objectives.** SciPy's `DIRinfcn` always sets the feasibility
   flag to 0 (`DIRsubrout.c:1081`), so Gablonsky's hidden-constraint machinery
   (`DIRreplaceinf`, flag 1/2 points) is **never triggered from SciPy**: a NaN
   value is stored as a feasible value and takes part in comparisons, where it
   is never `<` anything. `fmax = MAX(f, fmax)` uses the C macro
   `((a) > (b) ? (a) : (b))`, which keeps `fmax` when `f` is NaN — Go's
   `math.Max` and builtin `max` propagate NaN and must **not** be used for the
   `MAX`/`MIN` macros; port them as the ternaries. The hidden-constraint code
   paths are still ported (they are reachable through the flag), and a later
   option can map NaN to the flag (Gablonsky's design) — not in this slice.
9. **Array bounds the C code does not check**: `DIRchoose` writes `s[k]` for
   every non-empty level up to `MAXDEEP` with capacity `MAXDIV = 5000`. With
   the original algorithm (`level = k·n + p`) and large n this can exceed 5000
   in C (an out-of-bounds write). The Go port checks and returns status −6
   ("Maximum number of levels has been reached.") instead of panicking; a
   fuzz target (§8) aims at it.
10. **Messages** (`_direct_py.py`): `ret_code > 2` → `SUCCESS_MESSAGES[code−3]`
    formatted with `(maxfun, maxiter, f_min_rtol, vol_tol, len_tol)[code−1]`
    (Python's `str.format` of a float is its repr → `pyFloat`; `maxfun` is
    the user's value or `1000·N`, not the C-adjusted one); `1, 2` →
    `ERROR_MESSAGES[code−1]` formatted the same way; `−1 .. −99` →
    `ERROR_MESSAGES[|code|+1]`; `−100 ..` → `ERROR_MESSAGES[code+99]`.

The MIT notice of `COPYING` goes at the top of each ported file.

## 6. Results, status and errors (all three)

- Invalid input returns an error wrapping `ErrInput`, `ErrBounds` or
  `ErrDimension`, before any objective call; never a panic.
- Non-convergence is `Success = false` with SciPy's message.
- A nil objective is an `ErrInput`.
- Every slice in a result is freshly allocated (callers may keep them).

## 7. Validation plan

### 7.1 Oracle harness: `tools/globaloracle`

A cobra tool in the `tools/slsqporacle` / `tools/qporacle` shape
(`RootCmd()`, `SilenceUsage`, `RunE`, `cmd.OutOrStdout()`, `osExit`), driving
`uv run --with scipy==1.18.1 --with numpy==2.5.3 python` on an embedded
`global.py`, in a **linux/amd64 container** (no FMA in the wheel, §1 of
design.md). Subcommands `de`, `bh`, `direct`, each writing
`testdata/global/<method>.json` with, per case: the problem name, the inputs,
the seed form (`rng=` or `seed=`), and SciPy's `x`, `fun`, `nfev`, `nit`,
`message`, `success`, plus for DE the final `population` and
`population_energies` (bit patterns as hex, so the comparison is exact), for
basin hopping `minimization_failures`, `lowest_optimization_result`, and a
per-step trace (`x_trial`, `f_trial`, `accept`) through the callback, for
DIRECT the callback's per-iteration `x` (to locate a first divergence).
MaxFev cases call `DifferentialEvolutionSolver(..., maxfun=...)` directly.

### 7.2 Criteria

| Method | Exact-path cases | Criterion | Statistical layer |
|---|---|---|---|
| DIRECT | all (deterministic) | on arithmetic-only objectives (+ − × ÷, sqrt: sphere, shifted sphere n = 1..10, Rosenbrock, Goldstein–Price, six-hump camel, Styblinski–Tang, Shekel 5/7/10, Booth, Matyas, Beale): **identical** `nfev`, `nit`, `status`, `message`, and `x`, `fun` bit-identical, for both variants, each termination reason, `MaxFev`/`MaxIter` limits incl. the odd-budget rounding, `FMin` stop, NaN/Inf regions; on transcendental ones (Branin, Hartmann 3/6, Ackley, Rastrigin): same `status`, `fun` within 1e-10·max(1, abs(f)) of SciPy's, counts logged side by side. The SciPy docstring example (written with `**4`, i.e. `pow`) must give SciPy's `nfev = 2011`, `fun = -78.3323279095383` to 1e-12 (and `nfev = 207` with `len_tol=1e-3`) | none needed |
| DE | arithmetic-only objectives, both seed forms, every strategy, both updatings, LHS/random/array init, `X0`, dithered and constant mutation, `Atol`, `MaxFev`, constraints (linear and polynomial nonlinear), integrality, polish on/off | **identical** `nit`, `nfev`, `message`, `success`; final population and energies bit-identical; `x`, `fun` bit-identical when polishing is off; with polishing (L-BFGS-B by finite differences, which takes SciPy's exact path on most arithmetic problems, README) bit-identical where the polish path matches, else `fun` within the polisher's tolerance and the divergence logged | 100 seeds × the benchmark subset (§7.3) on transcendental objectives: Go's success count must not be lower than SciPy's at one-sided Fisher exact p < 0.01; median `nfev` within ±5 % (it should in fact be equal on most seeds; the count of exact-path seeds is logged) |
| Basin hopping | BFGS / L-BFGS-B with analytic gradients on polynomial landscapes (Himmelblau, six-hump camel, Rosenbrock, a 2-D quartic with four minima), `T` = 1 and 0, adaptive interval hit, `NIterSuccess`, both accept-test forms incl. `ForceAccept` (the skipped Metropolis draw) | identical per-step trace, `nit`, counts, `minimization_failures`, `x`/`fun` bit-identical | the SciPy docstring examples (cosine landscapes) and the benchmark subset over 100 seeds, same rule as DE |

"Arithmetic-only" means written with products on both sides: `global.py`
spells every integer power as repeated multiplication (`x*x*x*x`, never
`x**4`, which NumPy evaluates with `pow`), and the Go test functions do the
same, so no libm call enters either side.

The number of exact-path cases is reported in the README the way the other
methods report it ("n/m on SciPy's exact path"), with every exception
explained.

### 7.3 Benchmark set for the statistical layer

SciPy's own: `benchmarks/benchmarks/go_benchmark_functions` (the Jamil & Yang
2013 collection, 175 functions in the paper), with SciPy's success rule
`Benchmark.success`: `|f(x) − fglob| < 1e-5`, `x` inside the bounds. Settings
as SciPy's `_BenchOptimizers`: DE with `popsize=20`, basin hopping with
`L-BFGS-B`, DIRECT with defaults. The functions used are re-implemented in Go
under `internal/gofuncs` with their `fglob` and bounds (a subset of ~30
spanning separable, multimodal, plateau and deceptive shapes, listed in the
task PR), each checked against SciPy's value at three points by the oracle.

### 7.4 Planted positives and negative controls

Each break below must make named fixtures fail, then be reverted (recorded
in the README table, as for the existing methods):

- DE: swap the fill-point draw and the shuffle in immediate mode; reset `idx`
  before each shuffle; use `math.Max` in `violation`; plain sum instead of
  `SumPairwise` in `converged`; `cross[fill]` instead of `cross[0]` for
  exponential crossover; `<` instead of `<=` in `_accept_trial`.
- Basin hopping: draw the Metropolis number even after `ForceAccept`; use a
  windowed acceptance rate; `math.Exp(math.Min(0, prod))`.
- DIRECT: `math.Max` for the `MAX` macro; check the budget inside the
  division loop; return x as `(c + xs2)·xs1`; skip the `maxf` rounding.
- Workers: DE deferred with `Workers` = 1, 2, 7 must give bit-identical
  results (positive); a planted shared-scratch race in `evalBatch` must fail
  under `-race`.

### 7.5 Coverage

100 % statement coverage of the new files, as the rest of the package
(the coverage gate in `coverage_test.go`), with the unreachable C branches
(e.g. logfile-only code) not ported rather than excluded.

## 8. Security and robustness (audited in each PR)

- **Bounded work.** DE: at most `MaxIter·S` objective calls plus the polish
  (and `MaxFev` when set); basin hopping: `(NIter + 1)` local minimisations,
  each bounded by its method's limits; DIRECT: `MaxFev` plus one iteration's
  samples, `MaxIter` iterations, memory capped at 1 GiB as in the C code.
- **NaN/Inf objectives** never panic and never loop: DE never accepts a NaN
  trial against a feasible member and its convergence test is false while
  any energy is infinite or NaN, so it stops at `MaxIter`; DIRECT stores NaN
  as a feasible value (§5.3 item 8); basin hopping's Metropolis maps NaN
  `prod` to `w = 1` (parity) and `Storage.update` never takes a NaN as lower.
  Fixtures pin these behaviours to SciPy's.
- **Huge sizes and 32-bit overflow.** `S = max(5, popsize·max(1, N − eb))`,
  `S·N` floats, `MaxIter·S` evaluations, DIRECT's `MAXFUNC·N` are computed with
  overflow checks (`math/bits.Mul64` or `int64` with explicit limits) and
  rejected with `ErrInput` when they exceed `math.MaxInt` or a population
  cap of 2^27 elements (1 GiB of float64) — SciPy would raise MemoryError.
  Tests run on `GOARCH=386` in CI (the package's cross-build matrix) with a
  case that would overflow a 32-bit `int`.
- **Array bounds in the DIRECT port** (§5.3 item 9): checked, status −6.
- **Callbacks** receive copies; the objective's `x` must not be retained
  (documented, as for `Minimize`).
- **Panics** in user code under `Workers > 1` are re-raised on the calling
  goroutine (§2.5).
- **Fuzz targets** (corpus committed under `testdata/fuzz/`):
  `FuzzDifferentialEvolution` (bounds, popsize, mutation, recombination,
  strategy, updating, integrality, an objective whose values come from the
  fuzz bytes incl. NaN/±Inf), `FuzzBasinhopping` (stepsize, T, niter ≤ 5,
  interval, a fuzzed objective), `FuzzDirect` (dimension ≤ 12, bounds, eps
  sign, budgets, `Original`, fuzzed objective). Each asserts: no panic, no
  error after a successful validation, `Nfev` within the documented bound,
  finite `X` inside the bounds (DE, DIRECT).
- **Chaos** (once, recorded in the PR): an objective that panics on the k-th
  call with 1 and 8 workers; an objective that sleeps randomly per call
  under 8 workers (results must stay bit-identical); a callback that stops at
  every possible generation.
- `govulncheck` runs in CI already.

## 9. dual_annealing and shgo: later slices

- **`dual_annealing`** — *next slice (wave 2b), not this one.* It is
  generalised simulated annealing (Tsallis visiting distribution) with a
  local search (`_dual_annealing.py`); its draws are `normal` (go-ndarray's
  ziggurat is exact) and `uniform`, but its visiting step calls `exp`,
  `pow` and `gammaln` (`scipy.special`), so libm differences make an exact
  path fragile: validate statistically first, exact path logged. It needs a
  `gammaln` (go-numerics availability not checked here) and reuses
  `RandomSource`, `evalBatch` and the benchmark set of this slice. It is the
  closest SciPy analogue of MATLAB's `simulannealbnd`.
- **`shgo`** — *later (wave 3).* It builds its sampling on
  `scipy.spatial.Delaunay` (Qhull) and `scipy.stats.qmc.Sobol` and on its own
  simplicial-complex library (`scipy.optimize._shgo_lib._complex`), all of
  which would have to exist in Go first; it is a different size of work.
- DE's `init='sobol'`/`'halton'` wait for the same QMC engine.

## 10. File layout

```
rng.go                    RandomSource, NewRNG, NewLegacyRNG, adapters over go-ndarray random
parallel.go               evalBatch (workers, deterministic, panic propagation)
options_ptr.go            Float, Int helpers
de.go                     DEOptions, DEResult, DifferentialEvolution, validation
de_solver.go              init, generation (immediate, deferred), mutation, crossover, selection, convergence
de_constraints.go         violations (general form), promote_lowest
de_polish.go              polishing
basinhopping.go           API, runner, Metropolis, adaptive step, displacement
direct.go                 API, validation, messages            (MIT notice)
direct_core.go            DIRect.c port                        (MIT notice)
direct_sub.go             DIRsubrout.c + DIRserial.c port      (MIT notice)
*_test.go, *_fixture_test.go, *_fuzz_test.go
internal/gofuncs/         benchmark functions (Go), with fglob and bounds
testdata/global/          de.json, bh.json, direct.json (from tools/globaloracle)
testdata/fuzz/Fuzz*/      committed corpora
tools/globaloracle/       cobra; global.py; linux/amd64 container recipe in its README
tools/globalbench/        cobra; bench.py; per-round load check
docs/design-global.md     this document
```

## 11. MATLAB Global Optimization Toolbox coverage

| MATLAB | Covered by this slice | Notes |
|---|---|---|
| `ga` | partly: `DifferentialEvolution` | both population-based evolutionary methods with bounds, nonlinear constraints and integer variables (DE `Integrality`); `ga`'s own operators (selection, crossover and mutation functions, `gamultiobj`) are not reproduced |
| `GlobalSearch` | no | scatter-search start points feeding `fmincon` (MathWorks page); a later `MultiStart`-style helper over `Minimize` with LHS/DE-style start points and `evalBatch` is a small follow-up |
| `MultiStart` | no | same follow-up; trivially parallel over `Minimize` |
| `particleswarm` | no | not in SciPy; would be a new port (later) |
| `simulannealbnd` | no | `dual_annealing` (§9) is the analogue; basin hopping's Metropolis acceptance is related but different |
| `patternsearch`, `surrogateopt`, `paretosearch`, `gamultiobj` | no | out of scope of item 5 |
| — | `Direct` | DIRECT has no MATLAB toolbox counterpart; Mathematica and NLopt have it |

## 12. Performance plan

Reference: SciPy 1.18.1 single-thread, `tools/globalbench` in the
`tools/qpbench` shape: interleaved rounds Go/SciPy, medians of 7, a per-round
1-minute load check (rounds above load 8 are discarded and the run deferred),
`OMP_NUM_THREADS=1`, `workers=1`.

| Row | Go | SciPy |
|---|---|---|
| DE, sphere/Rosenbrock/Rastrigin, N = 2, 10, 30, default options, fixed seed | same seed, same path | `differential_evolution(..., rng=seed)`; also `vectorized=True, updating='deferred'` with a NumPy-vectorised objective (SciPy's fastest form; an honest row) |
| DE overhead | ns per trial with a trivial objective | same |
| Basin hopping, Himmelblau, L-BFGS-B, 100 steps | | |
| DIRECT, the SciPy docstring example; Rosenbrock N = 2, 5; sphere N = 10 | | compiled C core; Python only in the objective |

Targets: no row slower than SciPy on one core; a slower row is an open
defect recorded in docs/perf.md. Built in from the start:

- no allocation per trial or per generation (scratch slices sized at setup;
  `testing.AllocsPerRun` asserts 0 in the steady state of a generation, and
  per DIRECT iteration once the arrays are sized);
- the scalar RNG entry points of §2.3 (no `*ndarray.Array` in the loop);
- `SumPairwise` in go-simd/floats with SIMD kernels that keep NumPy's order;
- DIRECT's list arrays allocated once at the C code's sizes (or grown
  geometrically up to them).

The risk row is DE against SciPy's vectorised mode on a cheap objective,
where SciPy evaluates S points in one NumPy call; our per-trial overhead must
stay below SciPy's per-point cost there.

## 13. Implementation tasks, in order (one PR each)

1. **go-ndarray/ndarray: allocation-free scalar draws** (`Float64`,
   `FillUniform`, `IntN`, `FillIntN`, `ShuffleInt64` on `Generator` and
   `RandomState`), NumPy 2.5.3 fixtures incl. mixed-call buffering; open
   `mtrand.pyx` for the legacy forms first; minor release of go-ndarray.
2. **go-simd/floats: `SumPairwise`** (NumPy's pairwise order, generic and
   SIMD kernels on the package's six targets, fixtures from NumPy incl.
   n = 0..300 and signed zeros, fuzz), minor release.
3. **optimize: DIRECT and DIRECT-L** — `direct*.go`, `tools/globaloracle`
   with the `direct` subcommand and container recipe, `testdata/global/direct.json`,
   exact-path tests, planted breaks, `FuzzDirect`, `Workers` within a
   rectangle. (Independent of tasks 1–2; may go first.)
4. **optimize: `RandomSource`, `evalBatch`, option pointer helpers**, dependency
   bumps (go-ndarray, go-simd/floats); `Brute` moved onto `evalBatch`.
5. **optimize: differential evolution, unconstrained** — all strategies,
   init forms, `X0`, immediate/deferred, dither, tolerances, `MaxIter`/`MaxFev`,
   callback, workers, L-BFGS-B polish; `globaloracle de`; exact-path fixtures
   for both seed forms; workers determinism test; `FuzzDifferentialEvolution`.
6. **optimize: DE constraints and integrality** — Lampinen selection,
   violations, trust-constr polish, integrality limits and rounding, infeasible
   result report; fixtures.
7. **optimize: basin hopping** — `globaloracle bh`, per-step trace fixtures,
   accept-test forms, adaptive step, `FuzzBasinhopping`.
8. **optimize: benchmark function set and statistical suite** —
   `internal/gofuncs`, success-rate tests for the three methods against SciPy
   over 100 seeds.
9. **optimize: `tools/globalbench`** and the rows of docs/perf.md; open
   defects for any slower row.
10. **Docs and release** — README tables ("What is there", "How close to
    SciPy", negative controls, "Not done" updated to dual_annealing/shgo),
    docs/design.md status line, the docs site (hextra fork) and the org
    landing (landing kit), each in its own PR; then the release tag (a MINOR
    bump, v0.6.0) by the release owner.

Later slices: `dual_annealing` (wave 2b), a `MultiStart`/`GlobalSearch`-style
helper, `shgo` and QMC initialisation (wave 3), Gablonsky's hidden-constraint
option for DIRECT, batching DIRECT's evaluations across rectangles.

## 14. BFGS and L-BFGS-B last digits (v0.6.0 finding)

Cause of the basin-hopping and DE-polish mismatches of the first global slice
(Rosenbrock from (-1.2, 1): f 2.5353058e-15 against SciPy's 2.5353090e-15).
Method: every evaluation (x, f, g) of `minimize(..., jac=True)` was recorded
on SciPy 1.18.1 (linux/amd64 container, `testdata/gen/bfgs_trace.py`,
`testdata/bfgs_trace.json`) and compared bit for bit; for L-BFGS-B SciPy's
own workspace (`scipy.optimize._lbfgsb.setulb`, the `ws`, `wy`, `sy`, `ss`,
`wt`, `wn` arrays and `theta`, `dnorm`, `dtd`, `stp`) was dumped at every
function request and compared field by field.

- **BFGS: our defect.** The inverse-Hessian update used the O(n²) rank-two
  form, equal to SciPy's `A1 @ (Hk @ A2) + rho s s'` only in exact
  arithmetic. First divergence: evaluation 3. Ported SciPy's expression in
  its order (`A1 = I - (s y')rho`, two dense products, `(rho s) s'`): the
  39-evaluation trace is now bit-identical.
- **L-BFGS-B: BLAS/LAPACK summation order, not FMA.** SciPy 1.18.1 ships a C
  translation of L-BFGS-B 3.0 (`scipy/optimize/src/lbfgsb.c`, BSD-3) that
  calls `dpotrf`, `dtrtrs`, `dnrm2`, `ddot` from the wheel's OpenBLAS; the
  port followed the Fortran with LINPACK `dpofa`/`dtrsl`. Differences found,
  each the first divergence at its time: (1) `dpotrf` scales a row by the
  reciprocal of the pivot (`x * (1/r)`), not by division; (2) `dtrtrs` with
  several right-hand sides (formk's L⁻¹(-L_aᵀ+R_zᵀ)) goes through TRSM: the
  reciprocal diagonal, rows in blocks of two, the rows above a block applied
  as one accumulated product subtracted from the right-hand side; (3) `dnrm2`
  sums the squares and takes the root in x87 extended precision (a 64-bit
  mantissa) before rounding to double. Each was accepted only because the
  first divergence moved later and the aggregate counts rose (exact basin
  hopping 95 → 136 of 224; polished DE 56 → 57 of 82; `poly_formt_fail`, a
  runaway whose end depends on a Cholesky failure, still passes).
- **Open.** The first remaining divergence is at the ninth function request of
  the Rosenbrock run: a 5-term `ddot` inside `dpotrf` (an OpenBLAS kernel
  with a lane/blocking order that no sequential summation order reproduces;
  a four-lane fold matched one element and made the aggregate worse, so it
  was not kept). Matching it needs the kernel itself; until then L-BFGS-B
  results agree with SciPy's to the tolerance of the termination test and the
  basin-hopping decisions are SciPy's on every fixture.
