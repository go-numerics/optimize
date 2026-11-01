package optimize

import (
	"fmt"
	"math"
)

// Acceptance is the verdict of a user accept test.
type Acceptance int

const (
	// Reject rejects the step (the Metropolis test still draws its number).
	Reject Acceptance = iota
	// Accept lets the later tests decide.
	Accept
	// ForceAccept accepts the step and skips the remaining tests, including
	// the Metropolis draw (SciPy's "force accept").
	ForceAccept
)

// StepTaker displaces x in place. If it also implements StepSizer it is
// wrapped by the adaptive step-size controller, as SciPy does for any
// take_step that has a stepsize attribute.
type StepTaker interface{ TakeStep(x []float64) }

// StepSizer is a StepTaker with an adjustable step size.
type StepSizer interface {
	StepTaker
	StepSize() float64
	SetStepSize(float64)
}

// BasinhoppingOptions configures Basinhopping.
type BasinhoppingOptions struct {
	// NIter is the number of basin-hopping iterations (nil: 100).
	NIter *int
	// T is the Metropolis temperature (nil: 1); 0 is monotonic basin hopping.
	T *float64
	// StepSize is the initial maximum displacement of the default step
	// (0 selects 0.5).
	StepSize float64
	// Method is the local minimiser; "" follows scipy.optimize.minimize:
	// SLSQP with MinimizeOptions.Constraints, else L-BFGS-B with Bounds,
	// else BFGS.
	Method Method
	// MinimizeOptions are SciPy's minimizer_kwargs (bounds, constraints,
	// tolerances ...).
	MinimizeOptions *Options
	// TakeStep replaces the default random displacement.
	TakeStep StepTaker
	// AcceptTest is SciPy's accept_test(f_new, x_new, f_old, x_old);
	// AcceptTestResult its newer form accept_test(res_new, res_old). At most
	// one may be set. The old point is the last accepted one.
	AcceptTest       func(fNew float64, xNew []float64, fOld float64, xOld []float64) Acceptance
	AcceptTestResult func(resNew, resOld *Result) Acceptance
	// Callback is called with the initial minimum and after each step with a
	// copy of the step's minimum; returning true stops the run (the return
	// value of the initial call is ignored, as in SciPy).
	Callback func(x []float64, f float64, accepted bool) (stop bool)
	// Interval is how often the step size is adapted (0 selects 50).
	Interval int
	// NIterSuccess stops after that many steps without a new global minimum
	// (0: NIter + 2, never).
	NIterSuccess int
	// RNG is the random stream, shared by the displacement and the
	// Metropolis draws as in SciPy; nil is an entropy-seeded PCG64.
	RNG *RandomSource
	// TargetAcceptRate (0.5) and StepwiseFactor (0.9) lie in (0, 1).
	TargetAcceptRate float64
	StepwiseFactor   float64
}

// BasinhoppingResult is the outcome of Basinhopping (SciPy's OptimizeResult).
type BasinhoppingResult struct {
	X                    []float64
	Fun                  float64
	Nit                  int
	Nfev, Njev, Nhev     int
	MinimizationFailures int
	// Lowest is the lowest local minimisation found (SciPy's
	// lowest_optimization_result).
	Lowest  *Result
	Success bool
	Message string
}

const (
	bhMsgDone     = "requested number of basinhopping iterations completed successfully"
	bhMsgCallback = "callback function requested stop early byreturning True"
	bhMsgSuccess  = "success condition satisfied"
)

// bhMinimize is the local minimiser; tests replace it to fail a later call.
var bhMinimize = Minimize

// randomDisplacement is the default step: x += uniform(-s, s, n).
type randomDisplacement struct {
	stepsize float64
	rng      *RandomSource
	buf      []float64
}

func (d *randomDisplacement) TakeStep(x []float64) {
	// the range -s..s is valid by construction (s >= 0 and finite)
	_ = d.rng.FillUniform(d.buf, -d.stepsize, d.stepsize)
	for i := range x {
		x[i] += d.buf[i]
	}
}

func (d *randomDisplacement) StepSize() float64     { return d.stepsize }
func (d *randomDisplacement) SetStepSize(s float64) { d.stepsize = s }

// adaptiveStep is AdaptiveStepsize: every interval steps the step size is
// multiplied or divided by factor according to the cumulative acceptance rate.
type adaptiveStep struct {
	inner                  StepSizer
	target, factor         float64
	interval, nstep, naccp int
}

func (a *adaptiveStep) takeStep(x []float64) {
	a.nstep++
	if a.nstep%a.interval == 0 {
		rate := float64(a.naccp) / float64(a.nstep)
		if rate > a.target {
			a.inner.SetStepSize(a.inner.StepSize() / a.factor)
		} else {
			a.inner.SetStepSize(a.inner.StepSize() * a.factor)
		}
	}
	a.inner.TakeStep(x)
}

// Basinhopping finds a global minimum of p by basin hopping; it is
// scipy.optimize.basinhopping (SciPy 1.18.1) statement by statement. With the
// same seed, a local minimiser that follows SciPy's path and an objective
// built from IEEE operations, the sequence of steps is SciPy's.
func Basinhopping(p Problem, x0 []float64, o *BasinhoppingOptions) (*BasinhoppingResult, error) {
	var opt BasinhoppingOptions
	if o != nil {
		opt = *o
	}
	if opt.TargetAcceptRate == 0 {
		opt.TargetAcceptRate = 0.5
	}
	if opt.StepwiseFactor == 0 {
		opt.StepwiseFactor = 0.9
	}
	if !(opt.TargetAcceptRate > 0 && opt.TargetAcceptRate < 1) {
		return nil, fmt.Errorf("%w: target_accept_rate has to be in range (0, 1)", ErrInput)
	}
	if !(opt.StepwiseFactor > 0 && opt.StepwiseFactor < 1) {
		return nil, fmt.Errorf("%w: stepwise_factor has to be in range (0, 1)", ErrInput)
	}
	niter := 100
	if opt.NIter != nil {
		niter = *opt.NIter
	}
	if niter < 0 || opt.Interval < 0 || opt.NIterSuccess < 0 {
		return nil, fmt.Errorf("%w: NIter, Interval and NIterSuccess must not be negative", ErrInput)
	}
	if opt.Interval == 0 {
		opt.Interval = 50
	}
	if !(opt.StepSize >= 0) || math.IsInf(opt.StepSize, 0) { // also NaN
		return nil, fmt.Errorf("%w: StepSize must be finite and non-negative", ErrInput)
	}
	if opt.StepSize == 0 {
		opt.StepSize = 0.5
	}
	temp := 1.0
	if opt.T != nil {
		temp = *opt.T
	}
	if opt.AcceptTest != nil && opt.AcceptTestResult != nil {
		return nil, fmt.Errorf("%w: set at most one of AcceptTest and AcceptTestResult", ErrInput)
	}
	nSuccess := opt.NIterSuccess
	if nSuccess == 0 {
		nSuccess = math.MaxInt
		if niter < math.MaxInt-2 {
			nSuccess = niter + 2
		}
	}
	method := opt.Method
	if method == "" {
		switch mo := opt.MinimizeOptions; {
		case mo != nil && mo.Constraints != nil:
			method = SLSQP
		case mo != nil && mo.Bounds != nil:
			method = LBFGSB
		default:
			method = BFGS
		}
	}
	rng := opt.RNG.orEntropy()
	beta := math.Inf(1)
	if temp != 0 {
		beta = 1 / temp
	}

	// the step taker
	var step func(x []float64)
	var ad *adaptiveStep
	switch t := opt.TakeStep.(type) {
	case nil:
		ad = &adaptiveStep{inner: &randomDisplacement{stepsize: opt.StepSize, rng: rng, buf: make([]float64, len(x0))}}
	case StepSizer:
		ad = &adaptiveStep{inner: t}
	default:
		step = t.TakeStep
	}
	if ad != nil {
		ad.target, ad.factor, ad.interval = opt.TargetAcceptRate, opt.StepwiseFactor, opt.Interval
		step = ad.takeStep
	}

	minimize := func(x []float64) (*Result, error) {
		return bhMinimize(p, x, method, opt.MinimizeOptions)
	}
	res := &BasinhoppingResult{}
	minres, err := minimize(x0)
	if err != nil {
		return nil, err
	}
	if !minres.Success {
		res.MinimizationFailures++
	}
	x, energy, incumbent, lowest := clone(minres.X), minres.Fun, minres, minres
	res.Nfev, res.Njev, res.Nhev = minres.Nfev, minres.Njev, minres.Nhev
	if opt.Callback != nil {
		opt.Callback(clone(lowest.X), lowest.Fun, true)
	}

	count, i := 0, 0
	msg := bhMsgDone
	for it := range niter {
		i = it // Python's loop variable: niter-1 after a full run, 0 for niter = 0
		// one cycle: a Monte-Carlo step
		xs := clone(x)
		step(xs)
		minres, err := minimize(xs)
		if err != nil {
			return nil, err
		}
		if !minres.Success {
			res.MinimizationFailures++
		}
		res.Nfev += minres.Nfev
		res.Njev += minres.Njev
		res.Nhev += minres.Nhev
		accept := true
		var user Acceptance = Accept
		switch {
		case opt.AcceptTest != nil:
			user = opt.AcceptTest(minres.Fun, clone(minres.X), energy, clone(x))
		case opt.AcceptTestResult != nil:
			user = opt.AcceptTestResult(minres, incumbent)
		}
		switch user {
		case ForceAccept:
			// the remaining tests, Metropolis included, are skipped
		default:
			if user == Reject {
				accept = false
			}
			// Metropolis: one draw whenever it is reached
			prod := -(minres.Fun - incumbent.Fun) * beta
			w := 1.0 // Python's min(0, prod) keeps 0 for a NaN prod
			if prod < 0 {
				w = math.Exp(prod)
			}
			rand := rng.Float64()
			if !(w >= rand && (minres.Success || !incumbent.Success)) {
				accept = false
			}
		}
		if ad != nil && accept {
			ad.naccp++
		}
		newGlobal := false
		if accept {
			energy, x, incumbent = minres.Fun, clone(minres.X), minres
			if minres.Success && (minres.Fun < lowest.Fun || !lowest.Success) {
				lowest, newGlobal = minres, true
			}
		}
		if opt.Callback != nil && opt.Callback(clone(minres.X), minres.Fun, accept) {
			msg = bhMsgCallback
			break
		}
		count++
		if newGlobal {
			count = 0
		} else if count > nSuccess {
			msg = bhMsgSuccess
			break
		}
	}
	res.Lowest = lowest
	res.X, res.Fun = clone(lowest.X), lowest.Fun
	res.Message = msg
	res.Nit = i + 1
	res.Success = lowest.Success
	return res, nil
}
