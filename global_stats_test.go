package optimize

import (
	"encoding/json"
	"fmt"
	"math"
	"math/big"
	"os"
	"slices"
	"testing"

	"github.com/go-numerics/optimize/internal/gofuncs"
)

// The statistical layer of the global optimisers (docs/design-global.md 7.2,
// 7.3): on transcendental objectives libm differs by an ulp from SciPy's, a
// run leaves SciPy's exact path, and what must still agree is the
// distribution of outcomes. testdata/global/stats.json holds SciPy 1.18.1's
// 100 seeded runs of differential_evolution (popsize 20) and of
// basinhopping (L-BFGS-B with bounds, 30 steps, start points given) on
// functions of internal/gofuncs, and one run of direct; the criteria are
//
//   - the number of successes (SciPy's Benchmark.success) of Go is not lower
//     than SciPy's at a one-sided Fisher exact p < 0.01;
//   - the median nfev of Go is within 5 % of SciPy's (basin hopping: the
//     mean, see compareRuns);
//
// and the number of seeds that stay on SciPy's exact path (same success and
// same nfev) is logged.

// fisherLower returns P(X <= a), X hypergeometric: the one-sided p-value of
// "group 1 has fewer successes than group 2" for a and b successes out of n
// runs each (Fisher's exact test, alternative "less"). It is computed in
// log space.
func fisherLower(a, b, n int) float64 {
	t := a + b
	lg := func(k int) float64 { v, _ := math.Lgamma(float64(k) + 1); return v }
	lchoose := func(m, k int) float64 { return lg(m) - lg(k) - lg(m-k) }
	denom := lchoose(2*n, t)
	p := 0.0
	for x := max(0, t-n); x <= a; x++ {
		p += math.Exp(lchoose(n, x) + lchoose(n, t-x) - denom)
	}
	return math.Min(p, 1)
}

// fisherLowerExact is the same p-value with big.Int binomials.
func fisherLowerExact(a, b, n int) float64 {
	t := a + b
	binom := func(m, k int) *big.Int { return new(big.Int).Binomial(int64(m), int64(k)) }
	num := new(big.Int)
	for x := max(0, t-n); x <= a; x++ {
		if t-x > n {
			continue
		}
		num.Add(num, new(big.Int).Mul(binom(n, x), binom(n, t-x)))
	}
	r := new(big.Rat).SetFrac(num, binom(2*n, t))
	f, _ := r.Float64()
	return f
}

func TestFisherLower(t *testing.T) {
	for _, c := range [][3]int{{1, 11, 12}, {10, 90, 100}, {50, 50, 100}, {0, 0, 100}, {100, 100, 100}, {37, 41, 100}, {95, 100, 100}, {0, 100, 100}} {
		a, b, n := c[0], c[1], c[2]
		got, want := fisherLower(a, b, n), fisherLowerExact(a, b, n)
		if math.Abs(got-want) > 1e-12*math.Max(want, 1e-300)+1e-15 {
			t.Errorf("fisherLower(%d, %d, %d) = %g, exact %g", a, b, n, got, want)
		}
	}
	// A planted difference is detected, equal rates are not.
	if p := fisherLower(40, 90, 100); !(p < 0.01) {
		t.Errorf("40 against 90 successes: p = %g, want < 0.01", p)
	}
	if p := fisherLower(88, 90, 100); !(p >= 0.01) {
		t.Errorf("88 against 90 successes: p = %g, want >= 0.01", p)
	}
	if p := fisherLower(95, 90, 100); p < 0.5 {
		t.Errorf("more successes than SciPy: p = %g, want large", p)
	}
}

type statRun struct {
	Seed    uint64    `json:"seed"`
	Success bool      `json:"success"`
	Nfev    int       `json:"nfev"`
	Nit     int       `json:"nit"`
	Fun     dgHex     `json:"fun"`
	X       []dgHex   `json:"x"`
	X0      []dgHex   `json:"x0"`
	Status  int       `json:"status"`
	_       [0]func() // not comparable
}

type statCase struct {
	Name   string    `json:"name"`
	Method string    `json:"method"`
	Runs   []statRun `json:"runs"`
}

func loadStats(t testing.TB) []statCase {
	t.Helper()
	b, err := os.ReadFile("testdata/global/stats.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Scipy string     `json:"scipy"`
		Stats []statCase `json:"stats"`
	}
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	if f.Scipy != "1.18.1" {
		t.Fatalf("SciPy %s", f.Scipy)
	}
	return f.Stats
}

func medianInt(v []int) float64 {
	s := slices.Clone(v)
	slices.Sort(s)
	n := len(s)
	if n%2 == 1 {
		return float64(s[n/2])
	}
	return float64(s[n/2-1]+s[n/2]) / 2
}

// funcBounds returns the bounds of f as Bounds.
func funcBounds(f *gofuncs.Func) []Bound {
	b := make([]Bound, f.Dim())
	for i := range b {
		b[i] = Bound{f.Lo[i], f.Hi[i]}
	}
	return b
}

// compareRuns applies the criteria to Go's (success, nfev) series against
// SciPy's and returns a verdict and a log line.
func compareRuns(name string, goOK []bool, goNfev []int, c *statCase) (string, string) {
	n := len(c.Runs)
	var sg, ss, exact int
	scipyNfev := make([]int, n)
	for i, r := range c.Runs {
		if goOK[i] {
			sg++
		}
		if r.Success {
			ss++
		}
		scipyNfev[i] = r.Nfev
		if goOK[i] == r.Success && goNfev[i] == r.Nfev {
			exact++
		}
	}
	p := fisherLower(sg, ss, n)
	mg, ms := medianInt(goNfev), medianInt(scipyNfev)
	ag, as := meanInt(goNfev), meanInt(scipyNfev)
	line := fmt.Sprintf("%-18s %-2s success Go %3d SciPy %3d (p %.3f)  nfev median Go %7.0f SciPy %7.0f, mean Go %8.1f SciPy %8.1f  exact seeds %3d/%d",
		name, c.Method, sg, ss, p, mg, ms, ag, as, exact, n)
	switch {
	case p < 0.01:
		return fmt.Sprintf("%s: Go succeeds less often than SciPy (p = %.4g)", name, p), line
	case c.Method == "de" && math.Abs(mg-ms) > 0.05*ms:
		return fmt.Sprintf("%s: median nfev Go %v, SciPy %v (more than 5 %% apart)", name, mg, ms), line
	case c.Method == "bh" && math.Abs(ag-as) > 0.05*as:
		// Basin hopping's nfev is bimodal (runs that stay on a plateau against
		// runs that descend), so its median sits on a cliff between the two
		// clusters (Ackley01/2: 3584 against 3210 with means within 1 %); the
		// mean is the stable summary there.
		return fmt.Sprintf("%s: mean nfev Go %v, SciPy %v (more than 5 %% apart)", name, ag, as), line
	}
	return "", line
}

func meanInt(v []int) float64 {
	s := 0
	for _, x := range v {
		s += x
	}
	return float64(s) / float64(len(v))
}

func TestStatisticsDE(t *testing.T) {
	if testing.Short() {
		t.Skip("100 seeds per function")
	}
	n := 0
	for _, c := range loadStats(t) {
		if c.Method != "de" {
			continue
		}
		f, ok := gofuncs.ByName(c.Name)
		if !ok {
			t.Fatalf("unknown function %s", c.Name)
		}
		goOK := make([]bool, len(c.Runs))
		goNfev := make([]int, len(c.Runs))
		for i, r := range c.Runs {
			res, err := DifferentialEvolution(f.F, funcBounds(f), &DEOptions{PopSize: 20, RNG: NewRNG(r.Seed)})
			if err != nil {
				t.Fatalf("%s seed %d: %v", c.Name, r.Seed, err)
			}
			goOK[i], goNfev[i] = f.Success(res.X), res.Nfev
		}
		verdict, line := compareRuns(c.Name, goOK, goNfev, &c)
		t.Log(line)
		if verdict != "" {
			t.Error(verdict)
		}
		n++
	}
	if n < 10 {
		t.Fatalf("only %d differential-evolution functions in the fixture", n)
	}
}

func TestStatisticsBasinhopping(t *testing.T) {
	if testing.Short() {
		t.Skip("100 seeds per function")
	}
	n := 0
	for _, c := range loadStats(t) {
		if c.Method != "bh" {
			continue
		}
		f, ok := gofuncs.ByName(c.Name)
		if !ok {
			t.Fatalf("unknown function %s", c.Name)
		}
		goOK := make([]bool, len(c.Runs))
		goNfev := make([]int, len(c.Runs))
		for i, r := range c.Runs {
			x0 := make([]float64, len(r.X0))
			for j, v := range r.X0 {
				x0[j] = float64(v)
			}
			res, err := Basinhopping(Problem{Func: f.F}, x0, &BasinhoppingOptions{
				NIter: Int(30), RNG: NewRNG(r.Seed), Method: LBFGSB,
				MinimizeOptions: &Options{Bounds: funcBounds(f)},
			})
			if err != nil {
				t.Fatalf("%s seed %d: %v", c.Name, r.Seed, err)
			}
			goOK[i], goNfev[i] = f.Success(res.X), res.Nfev
		}
		verdict, line := compareRuns(c.Name, goOK, goNfev, &c)
		t.Log(line)
		if verdict != "" {
			t.Error(verdict)
		}
		n++
	}
	if n < 6 {
		t.Fatalf("only %d basin-hopping functions in the fixture", n)
	}
}

// TestStatisticsDirect: DIRECT is deterministic, so one run per function
// must give SciPy's success and, on every function, SciPy's status; the
// counts and values are logged (the transcendental ones may differ by an ulp
// of libm, which can move a rectangle).
func TestStatisticsDirect(t *testing.T) {
	n, sameN := 0, 0
	for _, c := range loadStats(t) {
		if c.Method != "direct" {
			continue
		}
		f, ok := gofuncs.ByName(c.Name)
		if !ok {
			t.Fatalf("unknown function %s", c.Name)
		}
		r := c.Runs[0]
		res, err := Direct(f.F, funcBounds(f), nil)
		if err != nil {
			t.Fatalf("%s: %v", c.Name, err)
		}
		n++
		if res.Status != r.Status {
			t.Errorf("%s: status %d, SciPy %d", c.Name, res.Status, r.Status)
		}
		if f.Success(res.X) != r.Success {
			t.Errorf("%s: success %v, SciPy %v", c.Name, f.Success(res.X), r.Success)
		}
		if res.Nfev == r.Nfev && res.Nit == r.Nit {
			sameN++
		} else {
			t.Logf("%s: nfev %d nit %d, SciPy %d %d (fun %.12g, SciPy %.12g)", c.Name, res.Nfev, res.Nit, r.Nfev, r.Nit, res.Fun, float64(r.Fun))
		}
	}
	if n < 15 {
		t.Fatalf("only %d DIRECT functions in the fixture", n)
	}
	t.Logf("%d/%d DIRECT runs with SciPy's nfev and nit", sameN, n)
}

// TestStatisticsNegativeControl: a planted weakness (DE stopped after 15
// generations, far from converged) must be caught by the criteria it is
// meant to test, on a function where SciPy succeeds on every seed.
func TestStatisticsNegativeControl(t *testing.T) {
	for _, c := range loadStats(t) {
		if c.Method != "de" || c.Name != "Rastrigin/5" {
			continue
		}
		f, _ := gofuncs.ByName(c.Name)
		goOK := make([]bool, len(c.Runs))
		goNfev := make([]int, len(c.Runs))
		for i, r := range c.Runs {
			res, err := DifferentialEvolution(f.F, funcBounds(f), &DEOptions{PopSize: 20, RNG: NewRNG(r.Seed), MaxIter: Int(15), NoPolish: true})
			if err != nil {
				t.Fatal(err)
			}
			goOK[i], goNfev[i] = f.Success(res.X), res.Nfev
		}
		verdict, line := compareRuns(c.Name, goOK, goNfev, &c)
		if verdict == "" {
			t.Fatalf("the weakened run was not detected: %s", line)
		}
		t.Log("detected: " + verdict)
		return
	}
	t.Fatal("Rastrigin/5 not in the fixture")
}
