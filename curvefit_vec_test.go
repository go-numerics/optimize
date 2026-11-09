package optimize

import (
	"encoding/json"
	"errors"
	"math"
	"testing"

	"github.com/go-simd/floats"
)

// vecOf turns a per-sample model of k variables into a Model.Vec over the
// (k, m) row-major layout, with the same arithmetic per sample.
func vecOf(f func(x, p []float64) float64, k int) func(xs, p, out []float64) {
	return func(xs, p, out []float64) {
		m := len(out)
		s := make([]float64, k)
		for j := range out {
			for i := range k {
				s[i] = xs[i*m+j]
			}
			out[j] = f(s, p)
		}
	}
}

// expVec is a·exp(−b·x)+c written over arrays as NumPy evaluates it: −b·x,
// exp, a·e, +c, each rounded in turn, with go-simd/floats' vector exp.
func expVec(xs, p, out []float64) {
	nb := -p[1]
	for j, x := range xs {
		out[j] = float64(nb * x)
	}
	floats.Exp(out, out)
	for j, e := range out {
		out[j] = float64(p[0]*e) + p[2]
	}
}

func sameFit(a, b *CurveFitResult) bool {
	return sameBits(a.Popt, b.Popt) && sameBits(a.Pcov, b.Pcov) && sameBits(a.Fvec, b.Fvec) &&
		a.Nfev == b.Nfev && a.Status == b.Status && a.CovOK == b.CovOK
}

// TestCurveFitVecSameBits: with the same arithmetic per sample, a model
// given as Vec gives the fit of the same model given as F, to the last bit,
// on every SciPy fixture (1-D and (2, m) xdata, sigma, bounds, LM and TRF).
func TestCurveFitVecSameBits(t *testing.T) {
	var cs []cfCase
	if err := json.Unmarshal(curvefitJSON, &cs); err != nil {
		t.Fatal(err)
	}
	for _, c := range cs {
		t.Run(c.Name, func(t *testing.T) {
			o := &CurveFitOptions{AbsoluteSigma: c.AbsoluteSigma, Method: LSMethod(c.Method)}
			if c.Sigma != nil {
				o.Sigma = cfArray(c.Sigma)
			}
			for _, b := range c.Bounds {
				o.Bounds = append(o.Bounds, Bound{b[0], b[1]})
			}
			xd := cfArray(c.X)
			k := 1
			if xd.Ndim() == 2 {
				k = xd.Shape()[0]
			}
			want, err1 := CurveFitModel(cfModels[c.Model], xd, cfArray(c.Y), c.P0, o)
			got, err2 := CurveFitModel(Model{Vec: vecOf(cfModels[c.Model].F, k)}, xd, cfArray(c.Y), c.P0, o)
			if (err1 == nil) != (err2 == nil) || !sameFit(got, want) {
				t.Errorf("Vec %v %v, F %v %v", got, err2, want, err1)
			}
		})
	}
}

// TestCurveFitVecPaths covers Vec with workers, with an analytic gradient
// (which Vec does not replace), and a vector exp against SciPy's result.
func TestCurveFitVecPaths(t *testing.T) {
	x, y := curveFitData(5000)
	for _, o := range []CurveFitOptions{{}, {Method: TRF}, {Bounds: []Bound{{0, 10}, {0, 3}, {-1, 1}}}} {
		want, err := CurveFitModel(Model{F: cfExp}, x, y, []float64{1, 1, 0}, &o)
		if err != nil {
			t.Fatal(err)
		}
		vec := Model{Vec: vecOf(cfExp, 1)}
		for _, w := range []int{1, 4} {
			ow := o
			ow.Workers = w
			got, err := CurveFitModel(vec, x, y, []float64{1, 1, 0}, &ow)
			if err != nil || !sameFit(got, want) {
				t.Errorf("%+v: Vec %v %v, F %v", ow, got.Popt, err, want.Popt)
			}
		}
		// Grad is used with Vec as with F.
		wantG, err1 := CurveFitModel(Model{F: cfExp, Grad: cfExpGrad}, x, y, []float64{1, 1, 0}, &o)
		gotG, err2 := CurveFitModel(Model{Vec: vecOf(cfExp, 1), Grad: cfExpGrad}, x, y, []float64{1, 1, 0}, &o)
		if err1 != nil || err2 != nil || !sameFit(gotG, wantG) {
			t.Errorf("%+v Grad: Vec %v %v, F %v %v", o, gotG, err2, wantG, err1)
		}
		// The vector exp differs from math.Exp in the last bit only: the
		// fit agrees within the tolerance of TestCurveFitAgainstSciPy.
		fast, err := CurveFitModel(Model{Vec: expVec}, x, y, []float64{1, 1, 0}, &o)
		if err != nil {
			t.Fatal(err)
		}
		for i := range want.Popt {
			if math.Abs(fast.Popt[i]-want.Popt[i]) > 1e-6*math.Max(1, math.Abs(want.Popt[i])) {
				t.Errorf("%+v: vector exp popt %v, math.Exp %v", o, fast.Popt, want.Popt)
			}
		}
	}
	// Vec never sees the caller's xdata: modifying xs does not change it.
	before := clone(arrayView(x))
	scribble := func(xs, p, out []float64) {
		expVec(xs, p, out)
		xs[0] = math.NaN()
	}
	if _, err := CurveFitModel(Model{Vec: scribble}, x, y, []float64{1, 1, 0}, nil); err != nil && !errors.Is(err, ErrNotConverged) {
		t.Fatal(err)
	}
	if !sameBits(arrayView(x), before) {
		t.Error("Vec modified the caller's xdata")
	}
	// Neither F nor Vec.
	if _, err := CurveFitModel(Model{}, x, y, []float64{1, 1, 0}, nil); !errors.Is(err, ErrInput) {
		t.Errorf("empty model: %v", err)
	}
}

// TestCurveFitEvaluationAllocs: an evaluation of the residuals allocates
// nothing, so the allocations of a fit do not grow with its evaluations.
func TestCurveFitEvaluationAllocs(t *testing.T) {
	if raceEnabled {
		t.Skip("the race detector's instrumentation allocates on its own schedule")
	}
	x, y := curveFitData(64)
	// Both budgets stop the fit before convergence, on the same path.
	run := func(maxfev int) float64 {
		return testing.AllocsPerRun(5, func() {
			if _, err := CurveFitModel(Model{Vec: expVec}, x, y, []float64{1, 1, 0}, &CurveFitOptions{MaxFev: maxfev}); !errors.Is(err, ErrNotConverged) {
				t.Fatalf("maxfev %d: %v", maxfev, err)
			}
		})
	}
	if a, b := run(6), run(14); a != b {
		t.Errorf("allocations grow with the evaluations: %v at maxfev 6, %v at 14", a, b)
	}
}
