// Package gofuncs holds Go twins of functions from the Jamil & Yang (2013)
// benchmark collection as SciPy ships it (benchmarks/benchmarks/
// go_benchmark_functions at v1.18.1, BSD-3): their bounds, their known global
// minimum fglob, and SciPy's success rule. It is the benchmark set of the
// statistical layer of the global optimisers (docs/design-global.md 7.3);
// tools/globaloracle (METHOD "funcs") checks every function against SciPy's
// own class at three points.
//
// The functions are written as NumPy evaluates them (x**2 is x*x, a sum is a
// left-to-right loop); they call libm-equivalents (math.Cos, math.Exp), so
// they agree with SciPy to a few ulps, not bit for bit.
package gofuncs

import (
	"fmt"
	"math"
)

// Func is one benchmark problem.
type Func struct {
	// Name is "Class/N", e.g. "Rastrigin/5".
	Name string
	// Lo and Hi are the bounds.
	Lo, Hi []float64
	// FGlob is the known global minimum value (SciPy's constant, which for a
	// few functions is rounded: success is |f - FGlob| < 1e-5).
	FGlob float64
	// XGlob is one location of the global minimum.
	XGlob []float64
	// F evaluates the function; len(x) == len(Lo).
	F func(x []float64) float64
}

// Dim returns the number of variables.
func (f *Func) Dim() int { return len(f.Lo) }

// Success is SciPy's Benchmark.success with tol = 1e-5: x is a success when
// |f(x) - fglob| < tol; otherwise it is not (a point outside the bounds is
// a failure, and a value below fglob is reported as an error by SciPy and
// counted as a failure here).
func (f *Func) Success(x []float64) bool {
	if len(x) != len(f.Lo) {
		return false
	}
	if math.Abs(f.F(x)-f.FGlob) < 1e-5 {
		return true
	}
	return false
}

// m is a product rounded before it is used, so that no compiler fuses it
// into a following addition (arm64, ppc64le, s390x): NumPy's does not.
func m(a, b float64) float64 { return float64(a * b) }

func sq(v float64) float64 { return m(v, v) }

func uniform(n int, lo, hi float64) (l, h []float64) {
	l, h = make([]float64, n), make([]float64, n)
	for i := range l {
		l[i], h[i] = lo, hi
	}
	return l, h
}

func filled(n int, v float64) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = v
	}
	return x
}

func ackley01(x []float64) float64 {
	var u, v float64
	for _, t := range x {
		u += m(t, t)
		v += math.Cos(m(2*math.Pi, t))
	}
	n := float64(len(x))
	return m(-20., math.Exp(m(-0.2, math.Sqrt(u/n)))) - math.Exp(v/n) + 20. + math.E
}

func alpine01(x []float64) float64 {
	var s float64
	for _, t := range x {
		s += math.Abs(m(t, math.Sin(t)) + m(0.1, t))
	}
	return s
}

func bohachevsky1(x []float64) float64 {
	return sq(x[0]) + m(2, sq(x[1])) - m(0.3, math.Cos(m(3*math.Pi, x[0]))) - m(0.4, math.Cos(m(4*math.Pi, x[1]))) + 0.7
}

func branin01(x []float64) float64 {
	return sq(x[1]-m(5.1/(4*sq(math.Pi)), sq(x[0]))+m(5, x[0])/math.Pi-6) +
		m(10*(1-1/(8*math.Pi)), math.Cos(x[0])) + 10
}

func bukin06(x []float64) float64 {
	return m(100, math.Sqrt(math.Abs(x[1]-m(0.01, sq(x[0]))))) + m(0.01, math.Abs(x[0]+10))
}

func crossInTray(x []float64) float64 {
	return m(-0.0001, math.Pow(math.Abs(m(m(math.Sin(x[0]), math.Sin(x[1])),
		math.Exp(math.Abs(100-math.Sqrt(sq(x[0])+sq(x[1]))/math.Pi))))+1, 0.1))
}

func deb01(x []float64) float64 {
	var s float64
	for _, t := range x {
		s += math.Pow(math.Sin(m(5*math.Pi, t)), 6)
	}
	return m(-(1.0 / float64(len(x))), s)
}

func dropWave(x []float64) float64 {
	var n float64
	for _, t := range x {
		n += m(t, t)
	}
	return -(1 + math.Cos(m(12, math.Sqrt(n)))) / (m(0.5, n) + 2)
}

func eggHolder(x []float64) float64 {
	var s float64
	for i := 0; i+1 < len(x); i++ {
		a, b := x[i], x[i+1]
		s += m(-(b+47), math.Sin(math.Sqrt(math.Abs(b+a/2.+47)))) - m(a, math.Sin(math.Sqrt(math.Abs(a-(b+47)))))
	}
	return s
}

func griewank(x []float64) float64 {
	var s float64
	p := 1.0
	for i, t := range x {
		s += m(t, t) / 4000
		p *= math.Cos(t / math.Sqrt(float64(i+1)))
	}
	return s - p + 1
}

func himmelBlau(x []float64) float64 {
	return sq(sq(x[0])+x[1]-11) + sq(x[0]+sq(x[1])-7)
}

func rastrigin(x []float64) float64 {
	s := 10.0 * float64(len(x))
	for _, t := range x {
		s += m(t, t) - m(10.0, math.Cos(m(2.0*math.Pi, t)))
	}
	return s
}

func rosenbrock(x []float64) float64 {
	var s float64
	for i := 0; i+1 < len(x); i++ {
		s += m(100.0, sq(x[i+1]-m(x[i], x[i]))) + sq(1-x[i])
	}
	return s
}

func salomon(x []float64) float64 {
	var n float64
	for _, t := range x {
		n += m(t, t)
	}
	u := math.Sqrt(n)
	return 1 - math.Cos(m(2*math.Pi, u)) + m(0.1, u)
}

func schwefel26(x []float64) float64 {
	s := m(418.982887, float64(len(x)))
	for _, t := range x {
		s -= m(t, math.Sin(math.Sqrt(math.Abs(t))))
	}
	return s
}

func sixHumpCamel(x []float64) float64 {
	return m(4-m(2.1, sq(x[0]))+math.Pow(x[0], 4)/3, sq(x[0])) + m(x[0], x[1]) + m(m(4, sq(x[1]))-4, sq(x[1]))
}

func sphere(x []float64) float64 {
	var s float64
	for _, t := range x {
		s += m(t, t)
	}
	return s
}

func styblinskiTang(x []float64) float64 {
	var s float64
	for _, t := range x {
		s += math.Pow(t, 4) - m(16, sq(t)) + m(5, t)
	}
	return s / 2
}

func wavy(x []float64) float64 {
	var s float64
	for _, t := range x {
		s += m(math.Cos(m(10, t)), math.Exp(-sq(t)/2.0))
	}
	return 1.0 - m(1.0/float64(len(x)), s)
}

// All returns the benchmark set, in the order of tools/globaloracle/gob.py.
func All() []Func {
	f := func(cls string, n int, lo, hi []float64, fglob float64, xg []float64, fn func([]float64) float64) Func {
		return Func{Name: fmt.Sprintf("%s/%d", cls, n), Lo: lo, Hi: hi, FGlob: fglob, XGlob: xg, F: fn}
	}
	box := func(n int, lo, hi float64) ([]float64, []float64) { return uniform(n, lo, hi) }
	var out []Func
	add := func(cls string, n int, lo, hi float64, fglob float64, xg []float64, fn func([]float64) float64) {
		l, h := box(n, lo, hi)
		out = append(out, f(cls, n, l, h, fglob, xg, fn))
	}
	add("Ackley01", 2, -35, 35, 0, filled(2, 0), ackley01)
	add("Ackley01", 5, -35, 35, 0, filled(5, 0), ackley01)
	add("Alpine01", 2, -10, 10, 0, filled(2, 0), alpine01)
	add("Bohachevsky1", 2, -100, 100, 0, filled(2, 0), bohachevsky1)
	out = append(out, f("Branin01", 2, []float64{-5, 0}, []float64{10, 15}, 0.39788735772973816,
		[]float64{-math.Pi, 12.275}, branin01))
	out = append(out, f("Bukin06", 2, []float64{-15, -3}, []float64{-5, 3}, 0, []float64{-10, 1}, bukin06))
	add("CrossInTray", 2, -10, 10, -2.062611870822739, []float64{1.349406685353340, 1.349406608602084}, crossInTray)
	add("Deb01", 2, -1, 1, -1, []float64{0.3, -0.3}, deb01)
	add("DropWave", 2, -5.12, 5.12, -1, filled(2, 0), dropWave)
	out = append(out, f("EggHolder", 2, filled(2, -512.1), filled(2, 512), -959.640662711,
		[]float64{512, 404.2319}, eggHolder))
	add("Griewank", 2, -100, 100, 0, filled(2, 0), griewank)
	add("HimmelBlau", 2, -5, 5, 0, []float64{3, 2}, himmelBlau)
	add("Rastrigin", 2, -5.12, 5.12, 0, filled(2, 0), rastrigin)
	add("Rastrigin", 5, -5.12, 5.12, 0, filled(5, 0), rastrigin)
	add("Rosenbrock", 2, -30, 30, 0, filled(2, 1), rosenbrock)
	add("Salomon", 2, -100, 100, 0, filled(2, 0), salomon)
	add("Schwefel26", 2, -500, 500, 0, filled(2, 420.968746), schwefel26)
	add("SixHumpCamel", 2, -5, 5, -1.031628, []float64{0.08984201368301331, -0.7126564032704135}, sixHumpCamel)
	add("Sphere", 2, -5.12, 5.12, 0, filled(2, 0), sphere)
	add("StyblinskiTang", 2, -5, 5, -39.16616570377142*2, filled(2, -2.903534018185960), styblinskiTang)
	add("Wavy", 2, -math.Pi, math.Pi, 0, filled(2, 0), wavy)
	return out
}

// ByName returns the function called name.
func ByName(name string) (*Func, bool) {
	for _, f := range All() {
		if f.Name == name {
			return &f, true
		}
	}
	return nil, false
}
