package optimize

import (
	"math"
	"math/rand/v2"
	"testing"
)

// FuzzQuadprogADMM: small random problems (P sometimes singular or indefinite,
// rows sometimes infeasible or unbounded, values sometimes NaN, ±Inf or huge)
// from a seed and a mask. The solver returns an error or a result, never
// panics, never exceeds MaxIter, and a Success result is finite and nearly
// feasible.
func FuzzQuadprogADMM(f *testing.F) {
	for _, s := range []uint64{0, 1, 2, 7, 12345, 1 << 40} {
		f.Add(s, uint8(0))
		f.Add(s, uint8(0xff))
	}
	f.Fuzz(func(t *testing.T, seed uint64, mask uint8) {
		rng := rand.New(rand.NewPCG(seed, ^seed))
		n := 1 + rng.IntN(6)
		m := rng.IntN(7)
		val := func() float64 {
			switch rng.IntN(40) {
			case 0:
				if mask&1 != 0 {
					return math.NaN()
				}
			case 1:
				if mask&2 != 0 {
					return math.Inf(1 - 2*rng.IntN(2))
				}
			case 2:
				if mask&4 != 0 {
					return 1e300 * (rng.Float64() - 0.5)
				}
			}
			return float64(rng.IntN(9)-4) / 2
		}
		p := make([]float64, n*n)
		for i := range p {
			p[i] = val()
		}
		if mask&8 == 0 { // positive semidefinite: LLᵀ with some zero rows
			l := make([]float64, n*n)
			for i := range l {
				l[i] = float64(rng.IntN(5) - 2)
			}
			for i := 0; i < n; i++ {
				for j := 0; j < n; j++ {
					p[i*n+j] = 0
					for k := 0; k < n; k++ {
						p[i*n+j] += float64(l[i*n+k] * l[j*n+k])
					}
				}
			}
		}
		q := make([]float64, n)
		for i := range q {
			q[i] = val()
		}
		var lin []LinearConstraint
		for r := 0; r < m; r++ {
			a := make([]float64, n)
			for i := range a {
				a[i] = val()
			}
			lb, ub := val(), val()
			if rng.IntN(2) == 0 {
				lb = math.Inf(-1)
			}
			if rng.IntN(3) == 0 {
				ub = math.Inf(1)
			}
			if rng.IntN(5) == 0 {
				ub = lb
			}
			lin = append(lin, LinearConstraint{A: qpArr(t, a, 1, n), Lb: []float64{lb}, Ub: []float64{ub}})
		}
		pr := QPProblem{P: qpArr(t, p, n, n), Q: q, Cons: QPConstraints{Linear: lin}}
		const maxIter = 300
		res, err := Quadprog(pr, &QPOptions{Method: QPADMM, MaxIter: maxIter})
		if err != nil {
			return
		}
		if res.Nit > maxIter {
			t.Fatalf("nit %d", res.Nit)
		}
		if res.Success {
			scale := 1 + vnormInf(res.X)
			for _, v := range res.X {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					t.Fatalf("non-finite solution %v", res.X)
				}
			}
			if !(res.ConstrViolation <= 0.5*scale*(1+float64(n))) && !math.IsInf(scale, 0) {
				t.Fatalf("violation %g, scale %g, x %v", res.ConstrViolation, scale, res.X)
			}
		}
	})
}
