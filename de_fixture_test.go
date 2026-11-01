package optimize

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/go-ndarray/ndarray"
)

// The fixture of tools/globaloracle (testdata/global/de.json): SciPy 1.18.1's
// DifferentialEvolutionSolver on the arithmetic-only problems of global.py,
// generated in a linux/amd64 container (no fused multiply-add), for both
// seeding forms: default_rng(seed) (PCG64) and RandomState(seed) (MT19937).

type deCase struct {
	Name    string    `json:"name"`
	Problem string    `json:"problem"`
	Dim     int       `json:"dim"`
	Lower   []float64 `json:"lower"`
	Upper   []float64 `json:"upper"`
	Kind    string    `json:"kind"`
	Seed    uint64    `json:"seed"`
	Options struct {
		Strategy      string      `json:"strategy"`
		Maxiter       int         `json:"maxiter"`
		Popsize       int         `json:"popsize"`
		Tol           float64     `json:"tol"`
		Atol          float64     `json:"atol"`
		Mutation      []float64   `json:"mutation"`
		Recombination float64     `json:"recombination"`
		Init          string      `json:"init"`
		InitPop       [][]float64 `json:"init_pop"`
		X0            []float64   `json:"x0"`
		Updating      string      `json:"updating"`
		Polish        bool        `json:"polish"`
		Maxfun        *int        `json:"maxfun"`
		Constraints   []struct {
			Type string      `json:"type"`
			Name string      `json:"name"`
			A    [][]float64 `json:"A"`
			Lb   []*float64  `json:"lb"`
			Ub   []*float64  `json:"ub"`
		} `json:"constraints"`
		Integrality []bool `json:"integrality"`
	} `json:"options"`
	ConstrViolation *dgHex `json:"constr_violation"`
	Error      *string   `json:"error"`
	X          []dgHex   `json:"x"`
	Fun        dgHex     `json:"fun"`
	Nfev       int       `json:"nfev"`
	Nit        int       `json:"nit"`
	Success    bool      `json:"success"`
	Message    string    `json:"message"`
	Population [][]dgHex `json:"population"`
	Energies   []dgHex   `json:"population_energies"`
}

type deFixture struct {
	Scipy      string   `json:"scipy"`
	DE         []deCase `json:"de"`
	Reductions []struct {
		Name   string  `json:"name"`
		Values []dgHex `json:"values"`
		Mean   dgHex   `json:"mean"`
		Std    dgHex   `json:"std"`
	} `json:"reductions"`
}

// TestDEMeanStd: the termination test's mean and std are np.mean and np.std
// to the bit, on arrays where a plain left-to-right sum gives other bits.
func TestDEMeanStd(t *testing.T) {
	fx := loadDEFixture(t)
	if len(fx.Reductions) < 40 {
		t.Fatalf("%d reductions", len(fx.Reductions))
	}
	for _, r := range fx.Reductions {
		e := make([]float64, len(r.Values))
		for i, v := range r.Values {
			e[i] = float64(v)
		}
		s := &deSolver{energies: e, dev: make([]float64, len(e))}
		mean, std := s.meanStd()
		if !sameFloat(mean, float64(r.Mean)) || !sameFloat(std, float64(r.Std)) {
			t.Errorf("%s: mean %v std %v, NumPy %v %v", r.Name, mean, std, float64(r.Mean), float64(r.Std))
		}
	}
}

func loadDEFixture(t testing.TB) *deFixture {
	t.Helper()
	b, err := os.ReadFile("testdata/global/de.json")
	if err != nil {
		t.Fatal(err)
	}
	var f deFixture
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	return &f
}

// deNLConstraints are the Go twins of NL_CONSTRAINTS in global.py; every
// product is rounded before it is summed.
var deNLConstraints = map[string]struct {
	m int
	f func(x, c []float64)
}{
	"circle": {1, func(x, c []float64) { c[0] = float64(x[0]*x[0]) + float64(x[1]*x[1]) }},
	"hyper": {2, func(x, c []float64) {
		c[0] = x[0] * x[1]
		c[1] = x[0] - float64(float64(x[1]*x[1])*x[1])
	}},
	"impossible": {1, func(x, c []float64) { c[0] = float64(float64(x[0]*x[0])+float64(x[1]*x[1])) + 1.0 }},
}

var deStrategyOf =map[string]DEStrategy{
	"best1bin": Best1Bin, "best1exp": Best1Exp, "rand1bin": Rand1Bin, "rand1exp": Rand1Exp,
	"randtobest1bin": RandToBest1Bin, "randtobest1exp": RandToBest1Exp,
	"currenttobest1bin": CurrentToBest1Bin, "currenttobest1exp": CurrentToBest1Exp,
	"best2bin": Best2Bin, "best2exp": Best2Exp, "rand2bin": Rand2Bin, "rand2exp": Rand2Exp,
}

// deOptionsOf maps a fixture case to DEOptions.
func deOptionsOf(c *deCase) (*DEOptions, []Bound) {
	o := &DEOptions{
		Strategy: deStrategyOf[c.Options.Strategy], MaxIter: Int(c.Options.Maxiter),
		PopSize: c.Options.Popsize, Tol: Float(c.Options.Tol), Atol: c.Options.Atol,
		Mutation: c.Options.Mutation, Recombination: Float(c.Options.Recombination),
		InitPopulation: c.Options.InitPop, X0: c.Options.X0, NoPolish: !c.Options.Polish,
	}
	if c.Options.Init == "random" {
		o.Init = DEInitRandom
	}
	if c.Options.Updating == "deferred" {
		o.Updating = DEDeferred
	} else {
		o.Updating = DEImmediate
	}
	if c.Options.Maxfun != nil {
		o.MaxFev = *c.Options.Maxfun
	}
	o.Integrality = c.Options.Integrality
	for _, k := range c.Options.Constraints {
		lim := func(v []*float64, inf float64) []float64 {
			out := make([]float64, len(v))
			for i, p := range v {
				out[i] = inf
				if p != nil {
					out[i] = *p
				}
			}
			return out
		}
		lb, ub := lim(k.Lb, math.Inf(-1)), lim(k.Ub, math.Inf(1))
		if k.Type == "linear" {
			flat := []float64{}
			for _, row := range k.A {
				flat = append(flat, row...)
			}
			a, _ := ndarray.FromData(flat, len(k.A), len(k.A[0]))
			o.Constraints = append(o.Constraints, LinearConstraint{A: a, Lb: lb, Ub: ub})
			continue
		}
		fn := deNLConstraints[k.Name]
		o.Constraints = append(o.Constraints, NonlinearConstraint{M: fn.m, Func: fn.f, Lb: lb, Ub: ub})
	}
	if c.Kind == "rng" {
		o.RNG = NewRNG(c.Seed)
	} else {
		o.RNG = NewLegacyRNG(uint32(c.Seed))
	}
	b := make([]Bound, c.Dim)
	for i := range b {
		b[i] = Bound{c.Lower[i], c.Upper[i]}
	}
	return o, b
}

func sameBitsHex(a []float64, b []dgHex) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !sameFloat(a[i], float64(b[i])) {
			return false
		}
	}
	return true
}

// deCompare returns "" when r is SciPy's result for c bit for bit, else the
// first difference. Without polishing the final population is compared too.
func deCompare(c *deCase, r *DEResult) string {
	if r.Nfev != c.Nfev || r.Nit != c.Nit {
		return fmt.Sprintf("nfev/nit %d/%d, SciPy %d/%d", r.Nfev, r.Nit, c.Nfev, c.Nit)
	}
	if r.Success != c.Success || r.Message != c.Message {
		return fmt.Sprintf("success %v %q, SciPy %v %q", r.Success, r.Message, c.Success, c.Message)
	}
	if (c.ConstrViolation == nil) != (len(r.Constr) == 0) ||
		(c.ConstrViolation != nil && !sameFloat(r.ConstrViolation, float64(*c.ConstrViolation))) {
		return fmt.Sprintf("constraint violation %v, SciPy %v", r.ConstrViolation, c.ConstrViolation)
	}
	if !sameBitsHex(r.X, c.X) || !sameFloat(r.Fun, float64(c.Fun)) {
		return fmt.Sprintf("x %v fun %v, SciPy %v %v", r.X, r.Fun, c.X, c.Fun)
	}
	if !sameBitsHex(r.PopulationEnergies, c.Energies) {
		return "population energies differ"
	}
	if len(r.Population) != len(c.Population) {
		return "population size differs"
	}
	for i := range r.Population {
		if !sameBitsHex(r.Population[i], c.Population[i]) {
			return fmt.Sprintf("population member %d differs", i)
		}
	}
	return ""
}

// TestDEFixture replays every case. Cases without polishing must be SciPy's
// path bit for bit: the final population, its energies, nfev, nit, message.
// With polishing the DE part is the same and the polisher is L-BFGS-B by
// finite differences, which follows SciPy's path on most problems; the cases
// where it does not are listed and bounded below.
func TestDEFixture(t *testing.T) {
	fx := loadDEFixture(t)
	if fx.Scipy != "1.18.1" || len(fx.DE) < 250 {
		t.Fatalf("unexpected fixture: SciPy %s, %d cases", fx.Scipy, len(fx.DE))
	}
	var exact, total, polished, polishedExact int
	var bad []string
	for i := range fx.DE {
		c := &fx.DE[i]
		f := dgProblems[c.Problem]
		if f == nil {
			t.Fatalf("%s: unknown problem %q", c.Name, c.Problem)
		}
		o, b := deOptionsOf(c)
		r, err := DifferentialEvolution(f, b, o)
		if c.Error != nil {
			if err == nil {
				t.Errorf("%s: SciPy raised %q, Go did not", c.Name, *c.Error)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: %v", c.Name, err)
			continue
		}
		total++
		diff := deCompare(c, r)
		if c.Options.Polish {
			polished++
			if diff == "" {
				polishedExact++
			} else {
				t.Logf("polished, diverges from SciPy: %s: %s", c.Name, diff)
			}
			continue
		}
		if diff != "" {
			bad = append(bad, c.Name+": "+diff)
			continue
		}
		exact++
	}
	for _, s := range bad {
		t.Error(s)
	}
	t.Logf("%d/%d unpolished cases on SciPy's exact path (both seed forms); %d/%d polished cases bit-identical",
		exact, total-polished, polishedExact, polished)
	if polishedExact*10 < polished*8 {
		t.Errorf("only %d of %d polished cases are bit-identical", polishedExact, polished)
	}
}

// TestDEFixtureWorkers: deferred updating gives SciPy's population for any
// number of workers.
func TestDEFixtureWorkers(t *testing.T) {
	fx := loadDEFixture(t)
	n := 0
	for i := range fx.DE {
		c := &fx.DE[i]
		if c.Error != nil || c.Options.Updating != "deferred" || c.Options.Polish || n >= 40 {
			continue
		}
		n++
		for _, w := range []int{1, 2, 7} {
			o, b := deOptionsOf(c)
			o.Workers = w
			r, err := DifferentialEvolution(dgProblems[c.Problem], b, o)
			if err != nil {
				t.Fatal(err)
			}
			if d := deCompare(c, r); d != "" {
				t.Errorf("%s, %d workers: %s", c.Name, w, d)
			}
		}
	}
	if n == 0 {
		t.Fatal("no deferred case")
	}
}
