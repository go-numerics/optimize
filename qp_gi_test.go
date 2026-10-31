package optimize

import (
	"encoding/json"
	"math"
	"math/rand/v2"
	"os"
	"strings"
	"testing"
)

type qpFixCase struct {
	Name    string `json:"name"`
	N       int
	P       [][]float64 `json:"p"`
	Q       []float64   `json:"q"`
	A       [][]float64 `json:"a"`
	L       []float64   `json:"l"`
	U       []float64   `json:"u"`
	Results map[string]struct {
		Status string    `json:"status"`
		X      []float64 `json:"x"`
		Error  string    `json:"error"`
	} `json:"results"`
}

func qpFixtureProblem(t *testing.T, c qpFixCase) QPProblem {
	t.Helper()
	n := len(c.Q)
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
		a := make([]float64, 0)
		for _, r := range c.A {
			a = append(a, r...)
		}
		pr.Cons.Linear = []LinearConstraint{{A: qpArr(t, a, len(c.A), n), Lb: inf(c.L), Ub: inf(c.U)}}
	}
	return pr
}

// The committed oracle fixture (OSQP, Clarabel, quadprog on the planted
// problems): wherever P is positive definite, Goldfarb–Idnani finds the same
// x as the references, and reports the infeasible problems infeasible.
func TestGIAgainstOracleFixture(t *testing.T) {
	raw, err := os.ReadFile("testdata/qp/planted.json")
	if err != nil {
		t.Fatal(err)
	}
	var fx struct {
		Cases []qpFixCase `json:"cases"`
	}
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatal(err)
	}
	solved, infeasible := 0, 0
	for _, c := range fx.Cases {
		pr := qpFixtureProblem(t, c)
		r, err := Quadprog(pr, &QPOptions{Method: QPActiveSet})
		if err != nil { // P singular or indefinite: not for the active set
			if !strings.Contains(err.Error(), "positive definite") {
				t.Errorf("%s: %v", c.Name, err)
			}
			continue
		}
		ref := c.Results["clarabel"]
		switch ref.Status {
		case "Solved":
			solved++
			// design-qp.md §6.3: x to 1e-8·max(1, ‖x‖∞) against the references (Clarabel
			// at 1e-10 carries about 1e-9 itself).
			if r.Status != QPOptimal || maxAbsDiff(r.X, ref.X) > 1e-7*math.Max(1, vnormInf(ref.X)) {
				t.Errorf("%s: status %d x %v, clarabel %v", c.Name, r.Status, r.X, ref.X)
			}
			if q, ok := c.Results["quadprog"]; ok && q.Error == "" && maxAbsDiff(r.X, q.X) > 1e-8*math.Max(1, vnormInf(q.X)) {
				t.Errorf("%s: x %v, quadprog %v", c.Name, r.X, q.X)
			}
		case "PrimalInfeasible":
			infeasible++
			if r.Status != QPInfeasible || r.PrimalCert == nil {
				t.Errorf("%s: status %d (%s), want infeasible with a certificate", c.Name, r.Status, r.Message)
			}
		}
	}
	if solved < 33 || infeasible != 2 {
		t.Errorf("solved %d, infeasible %d", solved, infeasible)
	}
}

func maxAbsDiff(a, b []float64) float64 {
	d := 0.0
	for i := range a {
		d = math.Max(d, math.Abs(a[i]-b[i]))
	}
	return d
}

// kktOK checks the KKT conditions of a QPOptimal result from scratch: primal
// feasibility, stationarity (Optimality), marginal signs and complementarity.
func kktOK(t *testing.T, md *qpModel, r *QPResult, tol float64) {
	t.Helper()
	if r.ConstrViolation > tol || r.Optimality > tol {
		t.Fatalf("violation %g, optimality %g", r.ConstrViolation, r.Optimality)
	}
	for j, row := range md.rows {
		ax := md.rowDot(row, r.X)
		y := -r.Y[j] // marginal
		lowerActive, upperActive := math.Abs(ax-row.l) <= tol, math.Abs(row.u-ax) <= tol
		switch {
		case row.l == row.u:
		case y > tol && !lowerActive, y < -tol && !upperActive:
			t.Fatalf("row %d: marginal %g but a·x = %g in [%g, %g]", j, y, ax, row.l, row.u)
		}
	}
}

// Random strictly convex QPs of every shape: the result satisfies the KKT
// conditions, and agrees with the SLSQP chain's lsei (the independent
// witness of design-qp.md §2.4: min ½|Lᵀx + L⁻¹q|² s.t. E x = f, G x >= h on a
// different factorisation) wherever both are optimal.
func TestGIRandomAgainstLSEI(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 22))
	optimal, infeasible, witnessed := 0, 0, 0
	for trial := 0; trial < 600; trial++ {
		n := 1 + rng.IntN(7)
		pm := make([]float64, n*n)
		l := make([]float64, n*n)
		for i := range l {
			l[i] = rng.NormFloat64()
		}
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				for k := 0; k < n; k++ {
					pm[i*n+j] += float64(l[i*n+k] * l[j*n+k])
				}
			}
			pm[i*n+i] += 0.1
		}
		q := make([]float64, n)
		for i := range q {
			q[i] = 3 * rng.NormFloat64()
		}
		m := rng.IntN(2 * n)
		var eqA, eqB, inA, inB []float64
		var lin []LinearConstraint
		for r := 0; r < m; r++ {
			a := make([]float64, n)
			for i := range a {
				a[i] = float64(rng.IntN(5) - 2)
			}
			b := float64(rng.IntN(5) - 2)
			if rng.IntN(4) == 0 {
				eqA, eqB = append(eqA, a...), append(eqB, b)
				lin = append(lin, LinearConstraint{A: qpArr(t, a, 1, n), Lb: []float64{b}, Ub: []float64{b}})
			} else {
				inA, inB = append(inA, a...), append(inB, b) // a·x >= b
				lin = append(lin, LinearConstraint{A: qpArr(t, a, 1, n), Lb: []float64{b}})
			}
		}
		pr := QPProblem{P: qpArr(t, pm, n, n), Q: q, Cons: QPConstraints{Linear: lin}}
		s, err := NewQPSolver(pr, &QPOptions{Method: QPActiveSet})
		if err != nil {
			t.Fatal(err)
		}
		r, err := s.Solve()
		if err != nil {
			t.Fatal(err)
		}
		switch r.Status {
		case QPInfeasible:
			infeasible++
			continue
		case QPOptimal:
			optimal++
		default:
			t.Fatalf("trial %d: status %d (%s) x %v active %v nit %d; eq %v=%v ineq %v>=%v P %v q %v", trial, r.Status, r.Message, r.X, r.ActiveSet, r.Nit, eqA, eqB, inA, inB, pm, q)
		}
		kktOK(t, s.m, r, 1e-7)
		// Witness: lsei on the Cholesky transformation.
		ws := &arena{}
		cm := clone(s.m.chol)
		for i := 0; i < n; i++ { // lpCholesky leaves P's upper triangle in place
			for j := i + 1; j < n; j++ {
				cm[i*n+j] = 0
			}
		}
		a := make([]float64, n*n) // column-major Lᵀ
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				a[i+j*n] = cm[j*n+i]
			}
		}
		b := forwardSolve(cm, n, q)
		for i := range b {
			b[i] = -b[i]
		}
		me, mg := len(eqB), len(inB)
		cmaj := func(rows []float64, mr int) []float64 {
			o := make([]float64, mr*max(n, 1))
			for i := 0; i < mr; i++ {
				for j := 0; j < n; j++ {
					o[i+j*mr] = rows[i*n+j]
				}
			}
			return o
		}
		x := make([]float64, n)
		mode, _, _ := lsei(ws, n, me, mg, n, a, b, cmaj(eqA, me), clone(eqB), cmaj(inA, mg), clone(inB), x)
		if mode != 1 {
			continue // the witness declines (rank-deficient E, ...)
		}
		// lsei trusts its equality rows to be independent; when they are not it
		// may return a point off the constraints. Only a feasible witness counts.
		viol := 0.0
		for _, row := range s.m.rows {
			ax := s.m.rowDot(row, x)
			viol = math.Max(viol, math.Max(row.l-ax, ax-row.u))
		}
		if viol > 1e-7 {
			continue
		}
		witnessed++
		fw := 0.5*dot(s.m.px(x), x) + dot(q, x)
		// A feasible point with a lower objective would refute GI; a higher
		// one is lsei's (it also stops early on rank-deficient E); an equal
		// one must be the same x (the QP is strictly convex).
		if fw < r.Fun-1e-7*(1+math.Abs(r.Fun)) ||
			(fw <= r.Fun+1e-7*(1+math.Abs(r.Fun)) && maxAbsDiff(x, r.X) > 1e-5*(1+vnormInf(x))) {
			t.Fatalf("trial %d: GI %v, lsei %v; eq %v=%v ineq %v>=%v P %v q %v", trial, r.X, x, eqA, eqB, inA, inB, pm, q)
		}
	}
	if optimal < 200 || infeasible < 20 || witnessed < 150 {
		t.Errorf("optimal %d, infeasible %d, witnessed %d", optimal, infeasible, witnessed)
	}
}

func TestInfeasibleOutcome(t *testing.T) {
	if how, msg, cert := infeasibleOutcome(true, []float64{1}); how != giInfeasible || cert == nil || strings.Contains(msg, "Numerical") {
		t.Errorf("%d %q %v", how, msg, cert)
	}
	if how, msg, cert := infeasibleOutcome(false, []float64{1}); how != giInfeasible || cert != nil || !strings.Contains(msg, "Numerical") {
		t.Errorf("%d %q %v", how, msg, cert)
	}
}

func TestGIBranches(t *testing.T) {
	id1 := qpArr(t, []float64{1}, 1, 1)
	id2 := qpArr(t, []float64{1, 0, 0, 1}, 2, 2)
	row := func(v ...float64) *LinearConstraint { return &LinearConstraint{A: qpArr(t, v, 1, len(v))} }
	lc := func(a []float64, lb, ub float64) LinearConstraint {
		return LinearConstraint{A: qpArr(t, a, 1, len(a)), Lb: []float64{lb}, Ub: []float64{ub}}
	}
	_ = row
	inf := math.Inf(1)
	solve := func(p QPProblem, o *QPOptions) *QPResult {
		t.Helper()
		if o == nil {
			o = &QPOptions{Method: QPActiveSet}
		}
		r, err := Quadprog(p, o)
		if err != nil {
			t.Fatal(err)
		}
		return r
	}
	// Infeasible bounds x >= 1, x <= 0: a certificate that verifies.
	r := solve(QPProblem{P: id1, Q: []float64{0}, Cons: QPConstraints{Linear: []LinearConstraint{lc([]float64{1}, 1, inf), lc([]float64{1}, -inf, 0)}}}, nil)
	if r.Status != QPInfeasible || r.Success || r.PrimalCert == nil {
		t.Fatalf("%+v", r)
	}
	// Inconsistent and redundant equalities.
	r = solve(QPProblem{P: id2, Q: []float64{0, 0}, Cons: QPConstraints{Linear: []LinearConstraint{lc([]float64{1, 1}, 1, 1), lc([]float64{1, 1}, 2, 2)}}}, nil)
	if r.Status != QPInfeasible {
		t.Fatalf("inconsistent equalities: %+v", r)
	}
	r = solve(QPProblem{P: id2, Q: []float64{0, 0}, Cons: QPConstraints{Linear: []LinearConstraint{lc([]float64{1, 1}, 1, 1), lc([]float64{2, 2}, 2, 2), lc([]float64{1, 1}, 1, 1)}}}, nil)
	if r.Status != QPOptimal || !near(r.X[0], .5, 1e-9) || !near(r.X[1], .5, 1e-9) {
		t.Fatalf("redundant equalities: %+v", r)
	}
	// An equality violated from above is entered negated: x0 + x1 = -1 with q pushing up.
	r = solve(QPProblem{P: id2, Q: []float64{-5, -5}, Cons: QPConstraints{Linear: []LinearConstraint{lc([]float64{1, 1}, -1, -1)}}}, nil)
	if r.Status != QPOptimal || !near(r.X[0], -.5, 1e-9) || !near(r.Linear.Marginals[0], -5.5, 1e-9) {
		t.Fatalf("equality from above: %+v", r)
	}
	// Zero rows: vacuous (0 >= -1) and infeasible (0 >= 1).
	r = solve(QPProblem{P: id1, Q: []float64{0}, Cons: QPConstraints{Linear: []LinearConstraint{lc([]float64{0}, -1, inf)}}}, nil)
	if r.Status != QPOptimal {
		t.Fatalf("vacuous zero row: %+v", r)
	}
	r = solve(QPProblem{P: id1, Q: []float64{0}, Cons: QPConstraints{Linear: []LinearConstraint{lc([]float64{0}, 1, inf)}}}, nil)
	if r.Status != QPInfeasible {
		t.Fatalf("0 >= 1: %+v", r)
	}
	// A rounding-level contradiction is set aside, then fails the final check.
	r = solve(QPProblem{P: id1, Q: []float64{0}, Cons: QPConstraints{Linear: []LinearConstraint{lc([]float64{1}, 1, inf), lc([]float64{1}, -inf, 1-1e-7)}}}, nil)
	if r.Success || r.Status != QPNumericalFail {
		t.Fatalf("set aside: %+v", r)
	}
	// Iteration limit returns the last point, not an error.
	r = solve(QPProblem{P: id2, Q: []float64{-3, -3}, Cons: QPConstraints{Linear: []LinearConstraint{lc([]float64{1, 0}, -inf, 1), lc([]float64{0, 1}, -inf, 1)}}}, &QPOptions{Method: QPActiveSet, MaxIter: 1})
	if r.Status != QPIterLimit || r.Success || len(r.X) != 2 {
		t.Fatalf("iteration limit: %+v", r)
	}
	// Partial steps: a constraint enters, then leaves when a second one makes it redundant.
	p := QPProblem{P: qpArr(t, []float64{2, 0, 0, 2}, 2, 2), Q: []float64{-2, -5}, Cons: QPConstraints{Linear: []LinearConstraint{
		lc([]float64{1, 0}, -inf, 0.2), lc([]float64{0, 1}, -inf, 1), lc([]float64{1, 1}, -inf, 1), lc([]float64{-1, 1}, -inf, 0.5)}}}
	r = solve(p, nil)
	if r.Status != QPOptimal {
		t.Fatalf("%+v", r)
	}
	// Many random inequality systems exercise insertions and deletions (see TestGIRandomAgainstLSEI);
	// here the iteration count is positive and the active set is a subset of the rows.
	if r.Nit < 2 || len(r.ActiveSet) > 4 {
		t.Fatalf("nit %d active %v", r.Nit, r.ActiveSet)
	}
}
