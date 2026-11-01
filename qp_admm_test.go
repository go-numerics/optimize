package optimize

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/go-sparse/sparse"
)

type qpFixFull struct {
	Name string      `json:"name"`
	Note string      `json:"note"`
	P    [][]float64 `json:"p"`
	Q    []float64   `json:"q"`
	A    [][]float64 `json:"a"`
	// The sparse form (tools/qporacle fixture.go): m rows, P full symmetric.
	M       int       `json:"m"`
	PCSC    *qpFixCSC `json:"p_csc"`
	ACSC    *qpFixCSC `json:"a_csc"`
	L       []float64 `json:"l"`
	U       []float64 `json:"u"`
	Results map[string]struct {
		Status     string    `json:"status"`
		X          []float64 `json:"x"`
		Y          []float64 `json:"y"`
		Obj        *float64  `json:"obj"`
		Iter       int       `json:"iter"`
		RhoUpdates int       `json:"rho_updates"`
		Error      string    `json:"error"`
	} `json:"results"`
}

type qpFixCSC struct {
	Indptr  []int     `json:"indptr"`
	Indices []int32   `json:"indices"`
	Data    []float64 `json:"data"`
}

func (s *qpFixCSC) dense(rows, cols int) [][]float64 {
	d := make([][]float64, rows)
	for i := range d {
		d[i] = make([]float64, cols)
	}
	for j := 0; j < cols; j++ {
		for k := s.Indptr[j]; k < s.Indptr[j+1]; k++ {
			d[s.Indices[k]][j] = s.Data[k]
		}
	}
	return d
}

// csc returns the matrix as a sparse.CSC; upper keeps the entries with row <= column.
func (s *qpFixCSC) csc(t testing.TB, rows, cols int, upper bool) *sparse.CSC[float64] {
	t.Helper()
	ptr := []int{0}
	var idx []int32
	var val []float64
	for j := 0; j < cols; j++ {
		for k := s.Indptr[j]; k < s.Indptr[j+1]; k++ {
			if !upper || int(s.Indices[k]) <= j {
				idx = append(idx, s.Indices[k])
				val = append(val, s.Data[k])
			}
		}
		ptr = append(ptr, len(idx))
	}
	m, err := sparse.NewCSC(rows, cols, ptr, idx, val)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// fixtureProblemSparse rebuilds a sparse-form case through PSparse and ASparse.
func fixtureProblemSparse(t testing.TB, c qpFixFull) QPProblem {
	t.Helper()
	n := len(c.Q)
	return QPProblem{PSparse: c.PCSC.csc(t, n, n, true), Q: c.Q,
		Cons: QPConstraints{ASparse: c.ACSC.csc(t, c.M, n, false), L: qpInf(c.L), U: qpInf(c.U)}}
}

func qpInf(v []float64) []float64 {
	o := clone(v)
	for i := range o {
		if math.Abs(o[i]) >= 1e20 {
			o[i] = math.Copysign(math.Inf(1), o[i])
		}
	}
	return o
}

func loadQPFixture(t testing.TB) []qpFixFull { return loadQPFixtureFile(t, "testdata/qp/planted.json") }

func loadQPFixtureFile(t testing.TB, path string) []qpFixFull {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var fx struct {
		Cases []qpFixFull `json:"cases"`
	}
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatal(err)
	}
	return fx.Cases
}

// fixtureProblem rebuilds a case as a QPProblem with its constraints as one
// Linear block (the shape the oracle fixture has).
func fixtureProblem(t testing.TB, c qpFixFull) QPProblem {
	t.Helper()
	n := len(c.Q)
	if c.PCSC != nil { // the dense form of a sparse case, for the active set
		c.P = c.PCSC.dense(n, n)
	}
	if c.ACSC != nil {
		c.A = c.ACSC.dense(c.M, n)
	}
	p := make([]float64, 0, n*n)
	for _, r := range c.P {
		p = append(p, r...)
	}
	inf := func(v []float64) []float64 {
		o := clone(v)
		for i := range o {
			if math.Abs(o[i]) >= 1e20 {
				o[i] = math.Copysign(math.Inf(1), o[i])
			}
		}
		return o
	}
	pr := QPProblem{P: qpArr(t, p, n, n), Q: c.Q}
	if len(c.A) > 0 {
		var a []float64
		for _, r := range c.A {
			a = append(a, r...)
		}
		pr.Cons.Linear = []LinearConstraint{{A: qpArr(t, a, len(c.A), n), Lb: inf(c.L), Ub: inf(c.U)}}
	}
	return pr
}

// ADMM at tight tolerances finds Clarabel's solution on every solved fixture
// case, and reports the infeasible ones with a certificate that verifies (12).
func TestADMMAgainstClarabel(t *testing.T) {
	solved, infeasible, other := 0, 0, 0
	for _, c := range loadQPFixture(t) {
		ref := c.Results["clarabel"]
		pr := fixtureProblem(t, c)
		r, err := Quadprog(pr, &QPOptions{Method: QPADMM, EpsAbs: 1e-9, EpsRel: 1e-9, MaxIter: 200000})
		if err != nil {
			t.Errorf("%s: %v", c.Name, err)
			continue
		}
		switch {
		case c.Name == "non-convex":
			// Clarabel accepts an indefinite P and returns "Solved"; OSQP refuses it
			// at setup; here the KKT matrix has fewer than n positive pivots.
			other++
			if r.Status != QPNonConvex || !math.IsNaN(r.X[0]) {
				t.Errorf("%s: status %d (%s), want non convex", c.Name, r.Status, r.Message)
			}
			continue
		case ref.Status == "DualInfeasible":
			other++
			if r.Status != QPUnbounded || r.DualCert == nil {
				t.Errorf("%s: status %d (%s), want unbounded", c.Name, r.Status, r.Message)
			}
			continue
		}
		switch ref.Status {
		case "Solved":
			solved++
			if r.Status != QPOptimal {
				t.Errorf("%s: status %d (%s)", c.Name, r.Status, r.Message)
				continue
			}
			if d := maxAbsDiff(r.X, ref.X); d > 1e-5*math.Max(1, vnormInf(ref.X)) {
				t.Errorf("%s: x %v, clarabel %v (diff %g)", c.Name, r.X, ref.X, d)
			}
		case "PrimalInfeasible":
			infeasible++
			if r.Status != QPInfeasible || r.PrimalCert == nil {
				t.Errorf("%s: status %d (%s), want infeasible", c.Name, r.Status, r.Message)
			}
		default:
			t.Errorf("%s: unexpected clarabel status %s", c.Name, ref.Status)
		}
	}
	if solved != 35 || infeasible != 2 || other != 2 {
		t.Errorf("solved %d, infeasible %d, other %d", solved, infeasible, other)
	}
}
