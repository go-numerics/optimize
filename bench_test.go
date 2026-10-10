package optimize

import (
	"math"
	"testing"

	"github.com/go-ndarray/ndarray"
)

// Benchmarks mirrored by benchmarks/bench_scipy.py; benchmarks/run.sh runs
// both interleaved and reports medians (docs/perf.md).

func benchMinimize(b *testing.B, n int, m Method, withGrad bool) {
	x0 := altStart(n)
	p := Problem{Func: rosen}
	if withGrad {
		p.Grad = rosenDer
	}
	b.ReportAllocs()
	for b.Loop() {
		r, err := Minimize(p, x0, m, nil)
		if err != nil || r.Fun > 1e-6 && n == 2 {
			b.Fatal(err, r)
		}
	}
}

func BenchmarkRosen2BFGS(b *testing.B)       { benchMinimize(b, 2, BFGS, true) }
func BenchmarkRosen100BFGS(b *testing.B)     { benchMinimize(b, 100, BFGS, true) }
func BenchmarkRosen2LBFGSB(b *testing.B)     { benchMinimize(b, 2, LBFGSB, true) }
func BenchmarkRosen100LBFGSB(b *testing.B)   { benchMinimize(b, 100, LBFGSB, true) }
func BenchmarkRosen2BFGSFD(b *testing.B)     { benchMinimize(b, 2, BFGS, false) }
func BenchmarkRosen100LBFGSBFD(b *testing.B) { benchMinimize(b, 100, LBFGSB, false) }
func BenchmarkRosen2NelderMead(b *testing.B) { benchMinimize(b, 2, NelderMead, false) }

// curveFitData is a decaying exponential with deterministic pseudo-noise
// (benchmarks/bench_scipy.py builds the same data).
func curveFitData(m int) (x, y *ndarray.Array) {
	xs := make([]float64, m)
	ys := make([]float64, m)
	for i := range xs {
		xs[i] = 4 * float64(i) / float64(m-1)
		noise := 0.2 * math.Sin(float64(i)*12.9898)
		ys[i] = 2.5*math.Exp(-1.3*xs[i]) + 0.5 + noise
	}
	x, _ = ndarray.FromData(xs, m)
	y, _ = ndarray.FromData(ys, m)
	return x, y
}

func benchCurveFit(b *testing.B, m int, method LSMethod, workers int) {
	x, y := curveFitData(m)
	f := func(x float64, p []float64) float64 { return p[0]*math.Exp(-p[1]*x) + p[2] }
	b.ReportAllocs()
	for b.Loop() {
		r, err := CurveFit(f, x, y, []float64{1, 1, 1}, &CurveFitOptions{Method: method, Workers: workers})
		if err != nil || math.Abs(r.Popt[1]-1.3) > 0.1 {
			b.Fatal(err, r)
		}
	}
}

func BenchmarkCurveFit1e4LM(b *testing.B)   { benchCurveFit(b, 10000, LM, 1) }
func BenchmarkCurveFit1e4TRF(b *testing.B)  { benchCurveFit(b, 10000, TRF, 1) }
func BenchmarkCurveFit1e4LMW4(b *testing.B) { benchCurveFit(b, 10000, LM, 4) }
func BenchmarkCurveFit1e4TRFW4(b *testing.B) {
	benchCurveFit(b, 10000, TRF, 4)
}

func BenchmarkBrentq(b *testing.B) {
	f := func(x float64) float64 { return x*x*x - 2*x - 5 }
	b.ReportAllocs()
	for b.Loop() {
		r, err := Brentq(f, 2, 3, nil)
		if err != nil || !r.Converged {
			b.Fatal(err)
		}
	}
}
