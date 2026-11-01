package optimize

import (
	"fmt"
	"math"
)

// bruteMaxPoints bounds the grid (2^27 values, 1 GiB of float64) so that its
// size can neither overflow an int, even on 32-bit targets, nor exhaust
// memory; SciPy would raise MemoryError.
const bruteMaxPoints = 1 << 27

// Range is one axis of a Brute grid. With Step 0 the axis holds Ns points
// from Lo to Hi inclusive (SciPy's (low, high) tuple, i.e. slice(low, high,
// Ns*1j)); with Step > 0 it holds Lo, Lo+Step, … below Hi (slice(low, high,
// step)).
type Range struct {
	Lo, Hi, Step float64
}

// BruteOptions configures Brute.
type BruteOptions struct {
	// Ns is the number of grid points per axis when Range.Step is 0 (20).
	Ns int
	// NoFinish skips the final local minimisation (SciPy's finish=None).
	NoFinish bool
	// Finish is the local method polishing the best grid point
	// (Nelder-Mead, SciPy's fmin). Finish options come from FinishOptions.
	Finish        Method
	FinishOptions *Options
	// Workers evaluates the grid with this many goroutines (1). The
	// objective must then be safe for concurrent use.
	Workers int
}

// BruteResult is Brute's full output (SciPy's full_output=True).
type BruteResult struct {
	// X and Fun are the minimiser and its value (after the finish step).
	X   []float64
	Fun float64
	// Success is false when the finish step did not converge.
	Success bool
	// Axes holds the coordinates of each grid axis; Jout the objective on
	// the grid, row-major over the axes (shape len(Axes[0]) × …).
	Axes [][]float64
	Jout []float64
}

// Brute minimises f by evaluating it on a grid and polishing the best point
// with a local method; it is scipy.optimize.brute(func, ranges, Ns=20,
// full_output=True, finish=fmin, workers=...). The finish step may leave
// the grid's box, as in SciPy.
func Brute(f func([]float64) float64, ranges []Range, o *BruteOptions) (*BruteResult, error) {
	var opt BruteOptions
	if o != nil {
		opt = *o
	}
	n := len(ranges)
	if n == 0 {
		return nil, fmt.Errorf("%w: no ranges", ErrInput)
	}
	if n > 40 {
		return nil, fmt.Errorf("%w: brute force not possible with more than 40 variables", ErrInput)
	}
	ns := opt.Ns
	if ns == 0 {
		ns = 20
	}
	if ns < 1 || ns > bruteMaxPoints {
		return nil, fmt.Errorf("%w: Ns must be in [1, %d]", ErrInput, int64(bruteMaxPoints))
	}
	axes := make([][]float64, n)
	total := 1
	for k, r := range ranges {
		var ax []float64
		if r.Step == 0 {
			step := 0.0
			if ns != 1 {
				step = (r.Hi - r.Lo) / float64(ns-1)
			}
			ax = make([]float64, ns)
			for i := range ax {
				ax[i] = float64(float64(i)*step) + r.Lo
			}
		} else {
			cnt := math.Ceil((r.Hi - r.Lo) / r.Step)
			if !(cnt > 0) || math.IsInf(cnt, 0) {
				return nil, fmt.Errorf("%w: range %d holds no points", ErrInput, k)
			}
			if cnt > bruteMaxPoints {
				return nil, fmt.Errorf("%w: range %d holds too many points", ErrInput, k)
			}
			ax = make([]float64, int(cnt))
			for i := range ax {
				ax[i] = float64(float64(i)*r.Step) + r.Lo
			}
		}
		axes[k] = ax
		if total > bruteMaxPoints/len(ax) {
			return nil, fmt.Errorf("%w: the grid holds more than %d points", ErrInput, int64(bruteMaxPoints))
		}
		total *= len(ax)
	}
	jout := make([]float64, total)
	point := func(idx int, x []float64) {
		for k := n - 1; k >= 0; k-- {
			m := len(axes[k])
			x[k] = axes[k][idx%m]
			idx /= m
		}
	}
	scratch := make([][]float64, effectiveWorkers(total, opt.Workers))
	for w := range scratch {
		scratch[w] = make([]float64, n)
	}
	parallelFor(total, opt.Workers, func(w, i int) {
		x := scratch[w]
		point(i, x)
		jout[i] = f(x)
	})
	// numpy.argmin: the first minimum, NaN counting as the minimum.
	best := 0
	for i, v := range jout {
		if math.IsNaN(v) {
			best = i
			break
		}
		if v < jout[best] {
			best = i
		}
	}
	x := make([]float64, n)
	point(best, x)
	res := &BruteResult{X: x, Fun: jout[best], Success: true, Axes: axes, Jout: jout}
	if opt.NoFinish {
		return res, nil
	}
	method := opt.Finish
	if method == "" {
		method = NelderMead
	}
	r, err := Minimize(Problem{Func: f}, x, method, opt.FinishOptions)
	if err != nil {
		return nil, err
	}
	res.X, res.Fun, res.Success = r.X, r.Fun, r.Success
	return res, nil
}
