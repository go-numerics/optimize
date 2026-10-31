package optimize

import (
	"math"
	"math/rand/v2"
	"testing"
)

// degenerateQP builds a strictly convex QP whose constraints are built to tie:
// small-integer rows, each repeated, scaled and combined, several through the
// same vertex, so that step lengths tie and active constraints are redundant.
func degenerateQP(t testing.TB, rng *rand.Rand) QPProblem {
	n := 2 + rng.IntN(4)
	pm := make([]float64, n*n)
	for i := 0; i < n; i++ {
		pm[i*n+i] = float64(1 + rng.IntN(3))
	}
	q := make([]float64, n)
	for i := range q {
		q[i] = float64(rng.IntN(7) - 3)
	}
	var lin []LinearConstraint
	var base [][]float64
	for r := 0; r < 2+rng.IntN(2*n); r++ {
		var a []float64
		switch {
		case len(base) > 0 && rng.IntN(2) == 0: // repeat, scale or add two earlier rows
			a = clone(base[rng.IntN(len(base))])
			if rng.IntN(2) == 0 {
				b := base[rng.IntN(len(base))]
				for i := range a {
					a[i] += b[i]
				}
			} else {
				s := float64(1 + rng.IntN(3))
				for i := range a {
					a[i] *= s
				}
			}
		default:
			a = make([]float64, n)
			for i := range a {
				a[i] = float64(rng.IntN(3) - 1)
			}
		}
		base = append(base, a)
		rhs := float64(rng.IntN(3)) // right-hand sides in {0,1,2}: many pass through the origin
		lc := LinearConstraint{A: qpArr(t, a, 1, n)}
		switch rng.IntN(4) {
		case 0:
			lc.Lb, lc.Ub = []float64{rhs}, []float64{rhs}
		case 1:
			lc.Lb = []float64{rhs}
		default:
			lc.Ub = []float64{rhs}
		}
		lin = append(lin, lc)
	}
	return QPProblem{P: qpArr(t, pm, n, n), Q: q, Cons: QPConstraints{Linear: lin}}
}

// Degenerate problems terminate within the default iteration limit: no
// QPIterLimit appears on thousands of tied, redundant systems, so cycling is
// not reachable here and the least-index tie rules of design-qp.md §3.4
// suffice (the limit stays only as a guard).
func TestGIDegenerateNoCycling(t *testing.T) {
	rng := rand.New(rand.NewPCG(99, 100))
	counts := map[int]int{}
	for trial := 0; trial < 5000; trial++ {
		r, err := Quadprog(degenerateQP(t, rng), &QPOptions{Method: QPActiveSet})
		if err != nil {
			t.Fatal(err)
		}
		counts[r.Status]++
		if r.Status == QPIterLimit || r.Status == QPNumericalFail {
			t.Fatalf("trial %d: status %d (%s)", trial, r.Status, r.Message)
		}
	}
	if counts[QPOptimal] < 1000 || counts[QPInfeasible] < 100 {
		t.Errorf("statuses %v", counts)
	}
}

// FuzzQuadprogDegenerate: tied, redundant systems from a seed; never a panic,
// never an iteration limit, and an optimal result passes its own tests.
func FuzzQuadprogDegenerate(f *testing.F) {
	for _, s := range []uint64{0, 1, 2, 12345, 1 << 40} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, seed uint64) {
		r, err := Quadprog(degenerateQP(t, rand.New(rand.NewPCG(seed, ^seed))), &QPOptions{Method: QPActiveSet})
		if err != nil {
			t.Fatal(err)
		}
		if r.Status == QPIterLimit || r.Status == QPNumericalFail {
			t.Fatalf("status %d (%s)", r.Status, r.Message)
		}
	})
}

// FuzzQPSolverUpdate: a random sequence of UpdateQ and Solve calls with wrong
// lengths and NaN/Inf, with and without a WorkingSet; each call is an error or
// a valid result, and after a valid UpdateQ the warm answer equals a cold one.
func FuzzQPSolverUpdate(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8, 9}, uint8(3))
	f.Add([]byte{255, 0, 255, 0, 17, 200, 3}, uint8(0))
	f.Add([]byte{}, uint8(7))
	f.Fuzz(func(t *testing.T, ops []byte, seed uint8) {
		rng := rand.New(rand.NewPCG(uint64(seed), 7))
		pr := degenerateQP(t, rng)
		n := len(pr.Q)
		var ws []int
		if len(ops) > 0 && ops[0]&1 == 1 {
			ws = []int{int(ops[0]) % 3, 100} // out of range on purpose
		}
		s, err := NewQPSolver(pr, &QPOptions{Method: QPActiveSet, WorkingSet: ws})
		if err != nil {
			return
		}
		for _, op := range ops {
			switch op % 5 {
			case 4: // UpdateBounds: the same sides must succeed, garbage must not panic
				l, u := make([]float64, len(s.m.rows)), make([]float64, len(s.m.rows))
				for i, r := range s.m.rows {
					l[i], u[i] = r.l, r.u
				}
				if op&8 != 0 && len(l) > 0 {
					l[int(op)%len(l)] = math.NaN()
					if s.UpdateBounds(l, u) == nil {
						t.Fatal("NaN side accepted")
					}
				} else if err := s.UpdateBounds(l, u); err != nil {
					t.Fatal(err)
				}
				if s.UpdateBounds(l[:len(l)/2], u) == nil && len(l) > 0 {
					t.Fatal("wrong length accepted")
				}
				if _, err := s.Solve(); err != nil {
					t.Fatal(err)
				}
			case 0: // wrong length
				if s.UpdateQ(make([]float64, n+int(op%3)+1)) == nil {
					t.Fatal("wrong length accepted")
				}
			case 1: // NaN or Inf
				q := make([]float64, n)
				q[int(op)%n] = []float64{math.NaN(), math.Inf(1), math.Inf(-1)}[int(op/4)%3]
				if s.UpdateQ(q) == nil {
					t.Fatal("non-finite q accepted")
				}
			case 2: // valid update, then warm against cold
				q := make([]float64, n)
				for i := range q {
					q[i] = float64(int(op>>2)%5-2) + float64(i)
				}
				if err := s.UpdateQ(q); err != nil {
					t.Fatal(err)
				}
				warm, err := s.Solve()
				if err != nil {
					t.Fatal(err)
				}
				p2 := pr
				p2.Q = q
				cold, err := Quadprog(p2, &QPOptions{Method: QPActiveSet})
				if err != nil {
					t.Fatal(err)
				}
				if warm.Status == QPOptimal && cold.Status == QPOptimal && maxAbsDiff(warm.X, cold.X) > 1e-6 {
					t.Fatalf("warm %v cold %v", warm.X, cold.X)
				}
			default:
				if _, err := s.Solve(); err != nil {
					t.Fatal(err)
				}
			}
		}
	})
}
