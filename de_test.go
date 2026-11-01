package optimize

import (
	"errors"
	"math"
	"sync/atomic"
	"testing"
)

func deSphere(x []float64) float64 {
	s := 0.0
	for _, v := range x {
		s += float64(v * v)
	}
	return s
}

func box(n int, lo, hi float64) []Bound {
	b := make([]Bound, n)
	for i := range b {
		b[i] = Bound{lo, hi}
	}
	return b
}

func TestDEFindsMinimum(t *testing.T) {
	r, err := DifferentialEvolution(deSphere, box(3, -5, 5), &DEOptions{RNG: NewRNG(1)})
	if err != nil {
		t.Fatal(err)
	}
	if !r.Success || r.Fun > 1e-12 || r.Status != DEConverged || r.Message != deMsgSuccess {
		t.Errorf("%+v", r)
	}
	if len(r.Population) != 45 || len(r.PopulationEnergies) != 45 {
		t.Error("population size")
	}
}

func TestDEInputErrors(t *testing.T) {
	ok := box(2, -1, 1)
	pop := func(rows, cols int) [][]float64 {
		p := make([][]float64, rows)
		for i := range p {
			p[i] = make([]float64, cols)
		}
		return p
	}
	cases := []struct {
		name string
		f    func([]float64) float64
		b    []Bound
		o    *DEOptions
		is   error
	}{
		{"nil f", nil, ok, nil, ErrInput},
		{"strategy", deSphere, ok, &DEOptions{Strategy: "nope"}, ErrInput},
		{"empty mutation", deSphere, ok, &DEOptions{Mutation: []float64{}}, ErrInput},
		{"mutation NaN", deSphere, ok, &DEOptions{Mutation: []float64{math.NaN()}}, ErrInput},
		{"mutation 2", deSphere, ok, &DEOptions{Mutation: []float64{0.5, 2}}, ErrInput},
		{"mutation neg", deSphere, ok, &DEOptions{Mutation: []float64{-0.1}}, ErrInput},
		{"no bounds", deSphere, nil, nil, ErrBounds},
		{"inf bound", deSphere, []Bound{{0, math.Inf(1)}}, nil, ErrBounds},
		{"nan bound", deSphere, []Bound{{math.NaN(), 1}}, nil, ErrBounds},
		{"lo>hi", deSphere, []Bound{{1, 0}}, nil, ErrBounds},
		{"maxiter<0", deSphere, ok, &DEOptions{MaxIter: Int(-1)}, ErrInput},
		{"maxfev<0", deSphere, ok, &DEOptions{MaxFev: -1}, ErrInput},
		{"popsize<0", deSphere, ok, &DEOptions{PopSize: -1}, ErrInput},
		{"popsize huge", deSphere, ok, &DEOptions{PopSize: math.MaxInt}, ErrInput},
		{"population cap", deSphere, box(1000, 0, 1), &DEOptions{PopSize: 100000}, ErrInput},
		{"rand2 small", deSphere, []Bound{{0, 1}}, &DEOptions{Strategy: Rand2Bin, PopSize: 1}, ErrInput},
		{"init pop rows", deSphere, ok, &DEOptions{InitPopulation: pop(4, 2)}, ErrInput},
		{"init pop cols", deSphere, ok, &DEOptions{InitPopulation: pop(5, 3)}, ErrDimension},
		{"x0 length", deSphere, ok, &DEOptions{X0: []float64{0}}, ErrDimension},
		{"x0 outside", deSphere, ok, &DEOptions{X0: []float64{0, 2}}, ErrBounds},
	}
	for _, c := range cases {
		if _, err := DifferentialEvolution(c.f, c.b, c.o); !errors.Is(err, c.is) {
			t.Errorf("%s: %v", c.name, err)
		}
	}
	// a huge user population is bounded too
	big := &DEOptions{InitPopulation: make([][]float64, 5)}
	for i := range big.InitPopulation {
		big.InitPopulation[i] = make([]float64, 1)
	}
	if _, err := DifferentialEvolution(deSphere, ok[:1], big); err != nil {
		t.Error(err)
	}
}

func TestDEDegenerateBounds(t *testing.T) {
	// all axes pinned: S = max(5, 15·1), the answer is the point itself
	r, err := DifferentialEvolution(deSphere, []Bound{{1, 1}, {2, 2}}, &DEOptions{RNG: NewRNG(1), NoPolish: true})
	if err != nil {
		t.Fatal(err)
	}
	if r.X[0] != 1 || r.X[1] != 2 || len(r.Population) != 15 {
		t.Errorf("%+v", r)
	}
	// one pinned axis out of two: population 15
	r, err = DifferentialEvolution(deSphere, []Bound{{-1, 1}, {2, 2}}, &DEOptions{RNG: NewRNG(1), NoPolish: true})
	if err != nil || len(r.Population) != 15 || r.X[1] != 2 {
		t.Errorf("%v %+v", err, r)
	}
}

func TestDEMaxIterMaxFev(t *testing.T) {
	r, _ := DifferentialEvolution(deSphere, box(2, -5, 5), &DEOptions{RNG: NewRNG(1), MaxIter: Int(0), NoPolish: true})
	if r.Success || r.Status != DEMaxIter || r.Nit != 0 || r.Message != deMsgMaxIter || r.Nfev != 30 {
		t.Errorf("%+v", r)
	}
	r, _ = DifferentialEvolution(deSphere, box(2, -5, 5), &DEOptions{RNG: NewRNG(1), MaxIter: Int(2), NoPolish: true, Tol: Float(0)})
	if r.Status != DEMaxIter || r.Nit != 2 {
		t.Errorf("%+v", r)
	}
	// immediate: exceeds the budget by one
	r, _ = DifferentialEvolution(deSphere, box(2, -5, 5), &DEOptions{RNG: NewRNG(1), MaxFev: 40, NoPolish: true})
	if r.Status != DEMaxFev || r.Message != deMsgMaxFev || r.Nfev != 41 {
		t.Errorf("%+v", r)
	}
	// deferred: reaches it exactly
	r, _ = DifferentialEvolution(deSphere, box(2, -5, 5), &DEOptions{RNG: NewRNG(1), MaxFev: 60, Updating: DEDeferred, NoPolish: true})
	if r.Status != DEMaxFev || r.Message != deMsgReached || r.Nfev != 60 {
		t.Errorf("%+v", r)
	}
	// a budget smaller than the population: the rest of the energies stay +Inf
	// (and the first candidate of the next generation is still evaluated)
	r, _ = DifferentialEvolution(deSphere, box(2, -5, 5), &DEOptions{RNG: NewRNG(1), MaxFev: 10, NoPolish: true})
	if r.Nfev != 11 || !math.IsInf(r.PopulationEnergies[20], 1) {
		t.Errorf("%+v", r)
	}
	// MaxFev reached with a budget the loop cannot classify (never evaluated)
	r, _ = DifferentialEvolution(deSphere, box(2, -5, 5), &DEOptions{RNG: NewRNG(1), MaxFev: 1, Updating: DEDeferred, NoPolish: true})
	if r.Status != DEMaxFev {
		t.Errorf("%+v", r)
	}
}

func TestDECallback(t *testing.T) {
	calls := 0
	r, _ := DifferentialEvolution(deSphere, box(2, -5, 5), &DEOptions{RNG: NewRNG(1), NoPolish: true, Callback: func(r *DEResult) bool {
		calls++
		if r.Message != "in progress" || r.Nit != calls || !(r.Convergence > 0) {
			t.Errorf("callback result %+v", r)
		}
		r.X[0] = 99 // a copy
		return calls == 3
	}})
	if calls != 3 || r.Status != DEStopped || r.Success || r.Message != deMsgStopped || r.X[0] == 99 {
		t.Errorf("%d %+v", calls, r)
	}
	// a callback that never stops sees the run converge
	n := 0
	r, _ = DifferentialEvolution(deSphere, box(2, -5, 5), &DEOptions{RNG: NewRNG(1), Callback: func(*DEResult) bool { n++; return false }})
	if !r.Success || n != r.Nit {
		t.Errorf("%d %+v", n, r)
	}
}

func TestDECustomStrategy(t *testing.T) {
	for _, upd := range []DEUpdating{DEImmediate, DEDeferred} {
		calls := 0
		strat := func(c int, pop [][]float64, rng *RandomSource, trial []float64) {
			calls++
			for j := range trial {
				trial[j] = pop[0][j] + float64(0.3*(pop[(c+1)%len(pop)][j]-pop[(c+2)%len(pop)][j]))
			}
			if c == 0 {
				trial[0] = 100 // out of bounds: redrawn
			}
		}
		r, err := DifferentialEvolution(deSphere, box(2, -5, 5), &DEOptions{RNG: NewRNG(1), CustomStrategy: strat, Updating: upd, MaxIter: Int(20), NoPolish: true})
		if err != nil || calls == 0 || r.Fun > 20 {
			t.Errorf("%v %d %+v", err, calls, r)
		}
	}
}

func TestDEPolish(t *testing.T) {
	// the default polisher improves a loosely converged run
	r, err := DifferentialEvolution(deSphere, box(2, -5, 5), &DEOptions{RNG: NewRNG(2), MaxIter: Int(3)})
	if err != nil {
		t.Fatal(err)
	}
	if r.Fun > 1e-12 || r.Jac == nil {
		t.Errorf("%+v", r)
	}
	// a custom polisher: rejected when worse, error and bad results reported
	worse := func(f func([]float64) float64, x0 []float64, b []Bound) (*Result, error) {
		return &Result{X: x0, Fun: f(x0) + 1, Success: true, Nfev: 3}, nil
	}
	r, _ = DifferentialEvolution(deSphere, box(2, -5, 5), &DEOptions{RNG: NewRNG(2), MaxIter: Int(3), Polish: worse})
	if r.Jac != nil {
		t.Error("worse polish accepted")
	}
	boom := errors.New("boom")
	if _, err := DifferentialEvolution(deSphere, box(2, -5, 5), &DEOptions{RNG: NewRNG(2), MaxIter: Int(1),
		Polish: func(func([]float64) float64, []float64, []Bound) (*Result, error) { return nil, boom }}); !errors.Is(err, boom) {
		t.Error(err)
	}
	for _, res := range []*Result{nil, {X: []float64{1}}} {
		if _, err := DifferentialEvolution(deSphere, box(2, -5, 5), &DEOptions{RNG: NewRNG(2), MaxIter: Int(1),
			Polish: func(func([]float64) float64, []float64, []Bound) (*Result, error) { return res, nil }}); !errors.Is(err, ErrInput) {
			t.Error(err)
		}
	}
	// outside the bounds: rejected
	out := func(f func([]float64) float64, x0 []float64, b []Bound) (*Result, error) {
		return &Result{X: []float64{9, 9}, Fun: -1, Success: true}, nil
	}
	r, _ = DifferentialEvolution(deSphere, box(2, -5, 5), &DEOptions{RNG: NewRNG(2), MaxIter: Int(1), Polish: out})
	if r.Fun < 0 {
		t.Error("out-of-bounds polish accepted")
	}
	if within([]float64{math.NaN()}, []float64{0}, []float64{1}) {
		t.Error("NaN within")
	}
}

func TestDENaNObjective(t *testing.T) {
	// NaN everywhere: never accepted, never converges, stops at MaxIter
	nan := func([]float64) float64 { return math.NaN() }
	r, err := DifferentialEvolution(nan, box(2, -1, 1), &DEOptions{RNG: NewRNG(3), MaxIter: Int(5), Workers: 3})
	if err != nil || r.Status != DEMaxIter || r.Nfev < 5*30 {
		t.Errorf("%v %+v", err, r)
	}
	// Inf everywhere: the all-infinite block reruns every generation
	inf := func([]float64) float64 { return math.Inf(1) }
	r, err = DifferentialEvolution(inf, box(2, -1, 1), &DEOptions{RNG: NewRNG(3), MaxIter: Int(3), NoPolish: true})
	if err != nil || r.Status != DEMaxIter {
		t.Errorf("%v %+v", err, r)
	}
}

func TestDEInfiniteEnergiesNeverConverge(t *testing.T) {
	half := func(x []float64) float64 {
		if x[0] > 0 {
			return math.Inf(1)
		}
		return deSphere(x)
	}
	sawInf := false
	r, err := DifferentialEvolution(half, box(2, -1, 1), &DEOptions{RNG: NewRNG(5), MaxIter: Int(4), NoPolish: true,
		Callback: func(r *DEResult) bool {
			sawInf = sawInf || r.Convergence == 0 // tol/(Inf+eps) = 0
			return false
		}})
	if err != nil || !sawInf || r.Status == DEConverged {
		t.Errorf("%v %v %+v", err, sawInf, r)
	}
}

func TestDEInitFormsAndX0(t *testing.T) {
	b := box(2, -5, 5)
	for _, o := range []*DEOptions{
		{Init: DEInitRandom},
		{X0: []float64{1, 2}},
		{InitPopulation: [][]float64{{0, 0}, {1, 1}, {-1, 1}, {9, 9}, {-9, -9}}}, // clipped
		{Mutation: []float64{0.4}, Recombination: Float(1)},
		{Strategy: Best1Exp, Recombination: Float(0)},
	} {
		o.RNG = NewRNG(4)
		o.NoPolish = true
		o.MaxIter = Int(5)
		if _, err := DifferentialEvolution(deSphere, b, o); err != nil {
			t.Error(err)
		}
	}
	r, _ := DifferentialEvolution(deSphere, b, &DEOptions{X0: []float64{0.5, 0.5}, MaxIter: Int(0), NoPolish: true, RNG: NewRNG(1)})
	if r.Fun > 0.5 {
		t.Errorf("x0 is not in the population: %v", r.Fun)
	}
}

func TestDEEntropyRNG(t *testing.T) {
	if _, err := DifferentialEvolution(deSphere, box(2, -1, 1), &DEOptions{MaxIter: Int(2), NoPolish: true}); err != nil {
		t.Error(err)
	}
}

func TestDEWorkersIdentical(t *testing.T) {
	var ref *DEResult
	for _, w := range []int{1, 2, 7} {
		r, err := DifferentialEvolution(deSphere, box(4, -5, 5), &DEOptions{RNG: NewRNG(9), Updating: DEDeferred, Workers: w, NoPolish: true})
		if err != nil {
			t.Fatal(err)
		}
		if ref == nil {
			ref = r
			continue
		}
		if d := sameResults(ref, r); d != "" {
			t.Errorf("workers %d: %s", w, d)
		}
	}
	// Workers > 1 with the default updating is deferred (SciPy overrides)
	a, _ := DifferentialEvolution(deSphere, box(4, -5, 5), &DEOptions{RNG: NewRNG(9), Workers: 3, NoPolish: true})
	if d := sameResults(ref, a); d != "" {
		t.Error(d)
	}
	// ... and so is an explicit immediate request
	a, _ = DifferentialEvolution(deSphere, box(4, -5, 5), &DEOptions{RNG: NewRNG(9), Updating: DEImmediate, Workers: 3, NoPolish: true})
	if d := sameResults(ref, a); d != "" {
		t.Error(d)
	}
}

func sameResults(a, b *DEResult) string {
	if a.Nfev != b.Nfev || a.Nit != b.Nit || a.Message != b.Message || !sameFloat(a.Fun, b.Fun) {
		return "counts differ"
	}
	for i := range a.Population {
		for j := range a.Population[i] {
			if !sameFloat(a.Population[i][j], b.Population[i][j]) {
				return "population differs"
			}
		}
	}
	return ""
}

func TestDEPanicPropagates(t *testing.T) {
	for _, w := range []int{1, 4} {
		func() {
			defer func() {
				if v := recover(); v != "objective boom" {
					t.Errorf("workers %d: %v", w, v)
				}
			}()
			var n atomic.Int64
			_, _ = DifferentialEvolution(func(x []float64) float64 {
				if n.Add(1) > 40 {
					panic("objective boom")
				}
				return deSphere(x)
			}, box(2, -1, 1), &DEOptions{RNG: NewRNG(1), Updating: DEDeferred, Workers: w, NoPolish: true})
		}()
	}
}

func TestDEHelpers(t *testing.T) {
	if argmin([]float64{3, 1, 1, 2}) != 1 || argmin([]float64{3, math.NaN(), 0, math.NaN()}) != 1 {
		t.Error("argmin")
	}
	if !allInf([]float64{math.Inf(1), math.Inf(-1)}) || allInf([]float64{math.NaN()}) || !anyInf([]float64{1, math.Inf(-1)}) {
		t.Error("allInf/anyInf")
	}
}

// the steady state of a generation allocates nothing (immediate updating).
func TestDEGenerationAllocs(t *testing.T) {
	for _, upd := range []DEUpdating{DEImmediate, DEDeferred} {
		o := &DEOptions{RNG: NewRNG(1), Updating: upd}
		s, err := newDESolver(deSphere, box(5, -5, 5), o)
		if err != nil {
			t.Fatal(err)
		}
		s.initEnergies()
		s.generation()
		a := testing.AllocsPerRun(20, func() { s.generation() })
		if a != 0 {
			t.Errorf("updating %v: %v allocations per generation", upd, a)
		}
	}
}
