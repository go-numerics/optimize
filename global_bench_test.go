package optimize

import (
	"math"
	"strconv"
	"strings"
	"testing"
)

// BenchmarkGlobal is the Go side of tools/globalbench: one sub-benchmark per
// row of docs/perf.md "Global optimisation", reporting nfev (and nit) beside
// ns/op so that the reference run (scipy 1.18.1, tools/globalbench/bench.py)
// can be compared per evaluation as well as per call. Row names are the
// reference's, with "." for "/".

func gbSphere(x []float64) float64 {
	s := 0.0
	for _, v := range x {
		s += float64(v * v)
	}
	return s
}

func gbRosenbrock(x []float64) float64 {
	s := 0.0
	for i := 0; i+1 < len(x); i++ {
		t := x[i+1] - float64(x[i]*x[i])
		u := 1 - x[i]
		s += float64(100*float64(t*t)) + float64(u*u)
	}
	return s
}

func gbRastrigin(x []float64) float64 {
	s := 10.0 * float64(len(x))
	for _, v := range x {
		s += float64(v*v) - 10*math.Cos(float64(2*math.Pi*v))
	}
	return s
}

// gbStyblinskiTang is the objective of SciPy's direct documentation example,
// written with products so that no libm call enters.
func gbStyblinskiTang(x []float64) float64 {
	s := 0.0
	for _, v := range x {
		v2 := float64(v * v)
		s += float64(v2*v2) - float64(16*v2) + float64(5*v)
	}
	return s / 2
}

func gbBounds(n int, lo, hi float64) []Bound {
	b := make([]Bound, n)
	for i := range b {
		b[i] = Bound{lo, hi}
	}
	return b
}

var gbObjectives = map[string]func([]float64) float64{
	"sphere": gbSphere, "rosenbrock": gbRosenbrock, "rastrigin": gbRastrigin,
}

type gbRow struct {
	name string
	run  func() (nfev, nit int)
}

func gbRows() []gbRow {
	var rows []gbRow
	for _, fn := range []string{"sphere", "rosenbrock", "rastrigin"} {
		for _, n := range []int{2, 10, 30} {
			f, n := gbObjectives[fn], n
			rows = append(rows, gbRow{"de." + fn + "." + strconv.Itoa(n), func() (int, int) {
				r, err := DifferentialEvolution(f, gbBounds(n, -5, 5), &DEOptions{RNG: NewRNG(1)})
				if err != nil {
					panic(err)
				}
				return r.Nfev, r.Nit
			}})
		}
	}
	for _, fn := range []string{"sphere", "rosenbrock", "rastrigin"} {
		f := gbObjectives[fn]
		rows = append(rows, gbRow{"dev." + fn + ".10", func() (int, int) {
			r, err := DifferentialEvolution(f, gbBounds(10, -5, 5), &DEOptions{RNG: NewRNG(1), Updating: DEDeferred})
			if err != nil {
				panic(err)
			}
			return r.Nfev, r.Nit
		}})
	}
	// Overhead per trial: a trivial objective, 50 generations, never converged.
	rows = append(rows, gbRow{"deoverhead.trivial.10", func() (int, int) {
		r, err := DifferentialEvolution(func(x []float64) float64 { return x[0] }, gbBounds(10, -5, 5),
			&DEOptions{RNG: NewRNG(1), MaxIter: Int(50), Tol: Float(-1), NoPolish: true})
		if err != nil {
			panic(err)
		}
		return r.Nfev, r.Nit
	}})
	rows = append(rows, gbRow{"bh.himmelblau.100", func() (int, int) {
		r, err := Basinhopping(Problem{FuncGrad: bhHimmelblau}, []float64{0, 0}, &BasinhoppingOptions{
			NIter: Int(100), RNG: NewRNG(1), Method: LBFGSB})
		if err != nil {
			panic(err)
		}
		return r.Nfev, r.Nit
	}})
	direct := func(name string, f func([]float64) float64, b []Bound) {
		rows = append(rows, gbRow{name, func() (int, int) {
			r, err := Direct(f, b, nil)
			if err != nil {
				panic(err)
			}
			return r.Nfev, r.Nit
		}})
	}
	direct("direct.styblinskitang.2", gbStyblinskiTang, gbBounds(2, -4, 4))
	direct("direct.rosenbrock.2", gbRosenbrock, gbBounds(2, -5, 5))
	direct("direct.rosenbrock.5", gbRosenbrock, gbBounds(5, -5, 5))
	direct("direct.sphere.10", gbSphere, gbBounds(10, -5, 5))
	return rows
}

func BenchmarkGlobal(b *testing.B) {
	for _, row := range gbRows() {
		b.Run(row.name, func(b *testing.B) {
			var nfev, nit int
			for i := 0; i < b.N; i++ {
				nfev, nit = row.run()
			}
			b.ReportMetric(float64(nfev), "nfev")
			b.ReportMetric(float64(nit), "nit")
		})
	}
}

// TestGlobalBenchRowsRun runs every row once so that a row that breaks fails
// the ordinary test run, not only the benchmark.
func TestGlobalBenchRowsRun(t *testing.T) {
	if testing.Short() {
		t.Skip("the rows take seconds")
	}
	seen := map[string]bool{}
	for _, row := range gbRows() {
		if seen[row.name] {
			t.Fatalf("duplicate row %s", row.name)
		}
		seen[row.name] = true
		if strings.HasSuffix(row.name, ".30") {
			continue // seconds each; the benchmark runs them
		}
		if nfev, _ := row.run(); nfev <= 0 {
			t.Errorf("%s: nfev %d", row.name, nfev)
		}
	}
}
