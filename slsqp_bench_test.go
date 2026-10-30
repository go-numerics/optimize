package optimize

import (
	"testing"

	"github.com/go-ndarray/ndarray"
)

// SLSQP benchmarks; tools/slsqpbench/bench.py times the same problems in
// SciPy and tools/slsqpbench interleaves the two.

func benchHS(b *testing.B, name string, jac bool) {
	p := hsProblems[name]
	prob := Problem{Func: p.f}
	if jac {
		prob.Grad = p.g
	}
	o := &Options{Bounds: p.bounds, Constraints: p.specs(jac)}
	b.ReportAllocs()
	for b.Loop() {
		if r, err := Minimize(prob, p.x0, SLSQP, o); err != nil || !r.Success {
			b.Fatal(err, r)
		}
	}
}

func BenchmarkSLSQPHS71(b *testing.B)   { benchHS(b, "HS71", true) }
func BenchmarkSLSQPHS71FD(b *testing.B) { benchHS(b, "HS71", false) }
func BenchmarkSLSQPHS38(b *testing.B)   { benchHS(b, "HS38", true) }
func BenchmarkSLSQPHS43(b *testing.B)   { benchHS(b, "HS43", true) }

// BenchmarkSLSQPQuad50: min sum (i+1)(x_i-1)^2 s.t. sum x <= 25,
// -10 <= x_i <= 10, from 0 (the subproblems are 50×50).
func BenchmarkSLSQPQuad50(b *testing.B) {
	const n = 50
	ones := make([]float64, n)
	for i := range ones {
		ones[i] = 1
	}
	a, _ := ndarray.FromData(ones, 1, n)
	o := &Options{Bounds: rep(Bound{-10, 10}, n),
		Constraints: []ConstraintSpec{LinearConstraint{A: a, Ub: []float64{25}}}}
	x0 := make([]float64, n)
	b.ReportAllocs()
	for b.Loop() {
		if r, err := Minimize(Problem{Func: quad, Grad: quadGrad}, x0, SLSQP, o); err != nil || !r.Success {
			b.Fatal(err, r)
		}
	}
}
