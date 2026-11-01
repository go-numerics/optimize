package optimize

import (
	"fmt"
	"math"
)

// DEStrategy names one of SciPy's twelve differential-evolution strategies.
type DEStrategy string

// The strategies of scipy.optimize.differential_evolution.
const (
	Best1Bin          DEStrategy = "best1bin" // the default
	Best1Exp          DEStrategy = "best1exp"
	Rand1Bin          DEStrategy = "rand1bin"
	Rand1Exp          DEStrategy = "rand1exp"
	RandToBest1Bin    DEStrategy = "randtobest1bin"
	RandToBest1Exp    DEStrategy = "randtobest1exp"
	CurrentToBest1Bin DEStrategy = "currenttobest1bin"
	CurrentToBest1Exp DEStrategy = "currenttobest1exp"
	Best2Bin          DEStrategy = "best2bin"
	Best2Exp          DEStrategy = "best2exp"
	Rand2Bin          DEStrategy = "rand2bin"
	Rand2Exp          DEStrategy = "rand2exp"
)

// DEInit selects how the initial population is drawn.
type DEInit int

const (
	// DEInitLatinHypercube is SciPy's init='latinhypercube' (default).
	DEInitLatinHypercube DEInit = iota
	// DEInitRandom is init='random'.
	DEInitRandom
	// 'sobol' and 'halton' need scipy.stats.qmc; they are not available until
	// a QMC engine exists in the fleet.
)

// DEUpdating selects when the best member is updated.
type DEUpdating int

const (
	// DEUpdatingDefault is immediate updating, or deferred when Workers > 1
	// (SciPy overrides updating='immediate' the same way).
	DEUpdatingDefault DEUpdating = iota
	// DEImmediate updates the population after each candidate.
	DEImmediate
	// DEDeferred updates it once per generation; the trials of a generation
	// are independent, so they can be evaluated with several workers and the
	// result is the same for every worker count.
	DEDeferred
)

// DEOptions configures DifferentialEvolution. Zero values select SciPy's
// defaults; the fields where 0 is meaningful are pointers (see Float, Int).
type DEOptions struct {
	// Strategy is one of the twelve strategies (Best1Bin).
	Strategy DEStrategy
	// CustomStrategy is SciPy's callable strategy: it writes the trial (in
	// the problem's units) for candidate into trial. population is S×N in the
	// problem's units; it must not be retained. The draws it makes on rng
	// are interleaved with the solver's own.
	CustomStrategy func(candidate int, population [][]float64, rng *RandomSource, trial []float64)
	// MaxIter is the maximum number of generations (nil: 1000). Zero is
	// allowed: the initial population is evaluated and nothing else.
	MaxIter *int
	// MaxFev limits the objective evaluations (0: unlimited), as the
	// maxfun of SciPy's DifferentialEvolutionSolver.
	MaxFev int
	// PopSize is the population multiplier (15): S = max(5, PopSize·N), N
	// counting only the axes whose bounds differ.
	PopSize int
	// Tol is the relative tolerance of the convergence test (0.01).
	Tol *float64
	// Atol is its absolute tolerance (0).
	Atol float64
	// Mutation: nil is the dither (0.5, 1); one value is a constant; two
	// values (sorted) are the dither range.
	Mutation []float64
	// Recombination is the crossover probability (0.7).
	Recombination *float64
	// RNG is the random stream; nil is an entropy-seeded PCG64
	// (not reproducible). NewRNG(seed) is SciPy's rng=seed and
	// NewLegacyRNG(seed) is seed=seed.
	RNG *RandomSource
	// Init selects the initial population when InitPopulation is nil.
	Init DEInit
	// InitPopulation is SciPy's init=array: S×N with S >= 5, clipped to the
	// bounds.
	InitPopulation [][]float64
	// X0 replaces the first member of the initial population.
	X0 []float64
	// Updating selects immediate or deferred updating.
	Updating DEUpdating
	// Workers evaluates the trials of a deferred generation (and the initial
	// population) with this many goroutines. The objective must then be safe
	// for concurrent use.
	Workers int
	// NoPolish skips the final L-BFGS-B polish (SciPy polish=False).
	NoPolish bool
	// Polish replaces the default polisher (SciPy's callable polish). It
	// receives the bounds the default would use.
	Polish func(f func([]float64) float64, x0 []float64, bounds []Bound) (*Result, error)
	// Callback is called after each generation; returning true stops the
	// run. The result it receives is a copy with Convergence set.
	Callback func(r *DEResult) (stop bool)
}

// DEResult is the outcome of DifferentialEvolution (SciPy's OptimizeResult).
type DEResult struct {
	X    []float64
	Fun  float64
	Jac  []float64 // from the polisher, when it improved the point
	Nfev int
	Nit  int
	// Status: 0 converged, 1 maxiter, 2 maxfev, 3 callback stop. (SciPy has
	// no status; these codes are Go-defined.)
	Status             int
	Success            bool
	Message            string
	Population         [][]float64 // S×N, problem units
	PopulationEnergies []float64
	// Convergence is set only in the result handed to a callback:
	// Tol/(convergence+eps).
	Convergence float64
}

// Statuses of a DEResult.
const (
	DEConverged = 0
	DEMaxIter   = 1
	DEMaxFev    = 2
	DEStopped   = 3
)

// dePopCap bounds S·N (2^27 float64, 1 GiB); SciPy would raise MemoryError.
const dePopCap = 1 << 27

const (
	deMsgSuccess = "Optimization terminated successfully."
	deMsgMaxFev  = "Maximum number of function evaluations has been exceeded."
	deMsgReached = "Maximum number of function evaluations has been reached."
	deMsgMaxIter = "Maximum number of iterations has been exceeded."
	deMsgStopped = "callback function requested stop early"
)

// strategy kinds
const (
	deBest1 = iota
	deRand1
	deRandToBest1
	deCurrentToBest1
	deBest2
	deRand2
)

type deStrategyInfo struct {
	kind int
	exp  bool
}

var deStrategies = map[DEStrategy]deStrategyInfo{
	Best1Bin: {deBest1, false}, Best1Exp: {deBest1, true},
	Rand1Bin: {deRand1, false}, Rand1Exp: {deRand1, true},
	RandToBest1Bin: {deRandToBest1, false}, RandToBest1Exp: {deRandToBest1, true},
	CurrentToBest1Bin: {deCurrentToBest1, false}, CurrentToBest1Exp: {deCurrentToBest1, true},
	Best2Bin: {deBest2, false}, Best2Exp: {deBest2, true},
	Rand2Bin: {deRand2, false}, Rand2Exp: {deRand2, true},
}

// DifferentialEvolution minimises f over the box bounds; it is
// scipy.optimize.differential_evolution (SciPy 1.18.1) statement by
// statement, so that with the same seed (NewRNG for rng=int, NewLegacyRNG
// for seed=int) and an objective built from IEEE operations only it follows
// SciPy's path bit for bit.
func DifferentialEvolution(f func(x []float64) float64, bounds []Bound, o *DEOptions) (*DEResult, error) {
	var opt DEOptions
	if o != nil {
		opt = *o
	}
	s, err := newDESolver(f, bounds, &opt)
	if err != nil {
		return nil, err
	}
	return s.solve()
}

func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

// newDESolver validates the input in SciPy's order and draws the initial
// population.
func newDESolver(f func([]float64) float64, bounds []Bound, o *DEOptions) (*deSolver, error) {
	if f == nil {
		return nil, fmt.Errorf("%w: nil objective", ErrInput)
	}
	s := &deSolver{f: f, o: o}
	// 1. strategy
	if o.CustomStrategy == nil {
		name := o.Strategy
		if name == "" {
			name = Best1Bin
		}
		info, ok := deStrategies[name]
		if !ok {
			return nil, fmt.Errorf("%w: unknown mutation strategy %q", ErrInput, o.Strategy)
		}
		s.strat = info
	}
	// 2. mutation
	mut := o.Mutation
	if mut == nil {
		mut = []float64{0.5, 1}
	}
	if len(mut) == 0 {
		return nil, fmt.Errorf("%w: empty mutation", ErrInput)
	}
	for _, m := range mut {
		if !finite(m) || m >= 2 || m < 0 {
			return nil, fmt.Errorf("%w: the mutation constant must be in [0, 2), or a (min, max) pair in [0, 2)", ErrInput)
		}
	}
	s.scale = mut[0]
	if len(mut) > 1 {
		s.dither = true
		s.d0, s.d1 = min(mut[0], mut[1]), max(mut[0], mut[1])
	}
	// 3. bounds
	n := len(bounds)
	if n < 1 {
		return nil, fmt.Errorf("%w: bounds must hold a (min, max) pair for each variable", ErrBounds)
	}
	s.n = n
	s.lo, s.hi = make([]float64, n), make([]float64, n)
	for i, b := range bounds {
		if !finite(b.Lo) || !finite(b.Hi) || b.Lo > b.Hi {
			return nil, fmt.Errorf("%w: bounds should be finite (min, max) pairs with min <= max", ErrBounds)
		}
		s.lo[i], s.hi[i] = b.Lo, b.Hi
	}
	// 4. counts
	s.maxiter = 1000
	if o.MaxIter != nil {
		s.maxiter = *o.MaxIter
	}
	if s.maxiter < 0 || o.MaxFev < 0 || o.PopSize < 0 {
		return nil, fmt.Errorf("%w: MaxIter, MaxFev and PopSize must be non-negative", ErrInput)
	}
	s.maxfev = math.MaxInt
	if o.MaxFev > 0 {
		s.maxfev = o.MaxFev
	}
	s.tol, s.atol = 0.01, o.Atol
	if o.Tol != nil {
		s.tol = *o.Tol
	}
	s.cr = 0.7
	if o.Recombination != nil {
		s.cr = *o.Recombination
	}
	s.workers = o.Workers
	switch {
	case o.Updating == DEImmediate && o.Workers <= 1:
	case o.Updating == DEDeferred || o.Workers > 1:
		s.deferred = true
	}
	// scaling constants
	s.arg1, s.arg2, s.recip = make([]float64, n), make([]float64, n), make([]float64, n)
	eb := 0
	for i := range n {
		s.arg1[i] = 0.5 * (s.lo[i] + s.hi[i])
		s.arg2[i] = math.Abs(s.lo[i] - s.hi[i])
		if r := 1 / s.arg2[i]; finite(r) {
			s.recip[i] = r
		}
		if s.lo[i] == s.hi[i] {
			eb++
		}
	}
	// 6. population size, with overflow checks
	pops := o.PopSize
	if pops == 0 {
		pops = 15
	}
	free := max(1, n-eb)
	if pops > dePopCap/free {
		return nil, fmt.Errorf("%w: the population is too large", ErrInput)
	}
	s.s = max(5, pops*free)
	if s.s > dePopCap/n {
		return nil, fmt.Errorf("%w: the population is too large", ErrInput)
	}
	// 7. user population
	if o.InitPopulation != nil {
		if len(o.InitPopulation) < 5 {
			return nil, fmt.Errorf("%w: The population supplied needs to have shape (S, len(x)), where S > 4.", ErrInput)
		}
		for _, row := range o.InitPopulation {
			if len(row) != n {
				return nil, fmt.Errorf("%w: The population supplied needs to have shape (S, len(x)), where S > 4.", ErrDimension)
			}
		}
		// the caller already holds S·N values: no cap beyond that.
		s.s = len(o.InitPopulation)
	}
	// SciPy fails at the first mutation (not enough values to unpack)
	if o.CustomStrategy == nil && s.strat.kind == deRand2 && s.s < 6 {
		return nil, fmt.Errorf("%w: rand2 strategies need a population of at least 6", ErrInput)
	}
	if o.X0 != nil && len(o.X0) != n {
		return nil, fmt.Errorf("%w: x0 must have %d entries", ErrDimension, n)
	}
	s.alloc()
	s.rng = o.RNG.orEntropy()
	// initial population
	switch {
	case o.InitPopulation != nil:
		for i, row := range o.InitPopulation {
			s.unscale(s.pop[i], row)
			for j, v := range s.pop[i] {
				s.pop[i][j] = min(max(v, 0), 1) // np.clip; NaN stays NaN
			}
		}
	case o.Init == DEInitRandom:
		s.rng.fillUnit(s.flat)
	default:
		s.initLHS()
	}
	if o.X0 != nil {
		s.unscale(s.pop[0], o.X0)
		for _, v := range s.pop[0] {
			if v > 1 || v < 0 {
				return nil, fmt.Errorf("%w: Some entries in x0 lay outside the specified bounds", ErrBounds)
			}
		}
	}
	return s, nil
}
