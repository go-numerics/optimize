package optimize

import (
	"errors"
	"math"
	"testing"

	"github.com/go-ndarray/ndarray"
	"github.com/go-sparse/sparse"
)

func qpArr(t testing.TB, data []float64, shape ...int) *ndarray.Array {
	t.Helper()
	a, err := ndarray.FromData(append([]float64(nil), data...), shape...)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func qpCSC(t testing.TB, rows, cols int, indptr []int, idx []int32, data []float64) *sparse.CSC[float64] {
	t.Helper()
	a, err := sparse.NewCSC(rows, cols, indptr, idx, data)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func near(a, b, tol float64) bool { return math.Abs(a-b) <= tol*(1+math.Abs(b)) }

func wantVec(t *testing.T, name string, got, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s = %v, want %v", name, got, want)
	}
	for i := range got {
		if !near(got[i], want[i], tol) {
			t.Fatalf("%s = %v, want %v", name, got, want)
		}
	}
}

// min ½(2x₀² + 4x₁²) − 2x₀ − 4x₁: x = (1, 1), f = −3.
func TestQuadprogUnconstrained(t *testing.T) {
	p := QPProblem{P: qpArr(t, []float64{2, 0, 0, 4}, 2, 2), Q: []float64{-2, -4}}
	r, err := Quadprog(p, nil)
	if err != nil {
		t.Fatal(err)
	}
	wantVec(t, "x", r.X, []float64{1, 1}, 1e-12)
	if !near(r.Fun, -3, 1e-12) || !r.Success || r.Status != QPOptimal || r.Method != QPActiveSet {
		t.Fatalf("%+v", r)
	}
	if r.Optimality > 1e-12 || r.ConstrViolation != 0 || len(r.ActiveSet) != 0 || len(r.Y) != 0 {
		t.Fatalf("%+v", r)
	}
	// The same through every spelling of P, and an explicit method.
	for name, q := range map[string]QPProblem{
		"asymmetric": {P: qpArr(t, []float64{2, 1, -1, 4}, 2, 2), Q: p.Q}, // (P+Pᵀ)/2 = [[2 0] [0 4]]
		"sparse":     {PSparse: qpCSC(t, 2, 2, []int{0, 1, 2}, []int32{0, 1}, []float64{2, 4}), Q: p.Q},
	} {
		// QPAuto takes the active set for a dense P only; a sparse P asks for it.
		methods := []QPMethod{QPActiveSet}
		if q.P != nil {
			methods = append(methods, QPAuto)
		}
		for _, m := range methods {
			r, err := Quadprog(q, &QPOptions{Method: m})
			if err != nil {
				t.Fatalf("%s %q: %v", name, m, err)
			}
			wantVec(t, name, r.X, []float64{1, 1}, 1e-12)
		}
	}
	if _, err := Quadprog(QPProblem{PSparse: qpCSC(t, 2, 2, []int{0, 1, 2}, []int32{0, 1}, []float64{2, 4}), Q: p.Q}, nil); !errors.Is(err, ErrMethod) {
		t.Errorf("QPAuto with a sparse P: %v, want ErrMethod (ADMM)", err)
	}
	// An off-diagonal sparse entry is mirrored: P = [[2 1] [1 2]], q = −(3, 3).
	r, err = Quadprog(QPProblem{PSparse: qpCSC(t, 2, 2, []int{0, 1, 3}, []int32{0, 0, 1}, []float64{2, 1, 2}), Q: []float64{-3, -3}}, &QPOptions{Method: QPActiveSet})
	if err != nil {
		t.Fatal(err)
	}
	wantVec(t, "x", r.X, []float64{1, 1}, 1e-12)
}

// min ½‖x‖² s.t. x₀ + x₁ = 1: x = (½, ½), multiplier ½.
func TestQuadprogEqualityFamilies(t *testing.T) {
	id := qpArr(t, []float64{1, 0, 0, 1}, 2, 2)
	row := qpArr(t, []float64{1, 1}, 2)
	b1 := qpArr(t, []float64{1}, 1)
	cases := map[string]QPConstraints{
		"AEq 1-D":     {AEq: row, BEq: b1},
		"AEq 2-D":     {AEq: qpArr(t, []float64{1, 1}, 1, 2), BEq: b1},
		"Linear":      {Linear: []LinearConstraint{{A: row, Lb: []float64{1}, Ub: []float64{1}}}},
		"Linear both": {Linear: []LinearConstraint{{A: qpArr(t, []float64{1, 1}, 1, 2), Lb: []float64{1}, Ub: []float64{1}}}},
	}
	for name, c := range cases {
		r, err := Quadprog(QPProblem{P: id, Q: []float64{0, 0}, Cons: c}, nil)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		wantVec(t, name+" x", r.X, []float64{.5, .5}, 1e-12)
		if !near(r.Fun, .25, 1e-12) || len(r.ActiveSet) != 1 || r.ConstrViolation > 1e-12 || r.Optimality > 1e-12 {
			t.Fatalf("%s: %+v", name, r)
		}
		wantVec(t, name+" y", r.Y, []float64{-.5}, 1e-12)
		blk := r.EqLin
		if name == "Linear" || name == "Linear both" {
			blk = r.Linear
		}
		wantVec(t, name+" marginal", blk.Marginals, []float64{.5}, 1e-12)
		if math.Abs(blk.Residual[0]) > 1e-12 {
			t.Fatalf("%s: residual %v", name, blk.Residual)
		}
	}
	// A vacuous AUb row (+Inf) and a free Linear row are dropped.
	r, err := Quadprog(QPProblem{P: id, Q: []float64{-1, -1}, Cons: QPConstraints{
		AUb: qpArr(t, []float64{1, 0, 0, 1}, 2, 2), BUb: qpArr(t, []float64{math.Inf(1), math.Inf(1)}, 2),
		Linear: []LinearConstraint{{A: row}},
		Bounds: []Bound{Unbounded(), Unbounded()}}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	wantVec(t, "x", r.X, []float64{1, 1}, 1e-12)
	if len(r.Y) != 0 || !math.IsInf(r.IneqLin.Residual[0], 1) || !math.IsInf(r.Linear.Residual[0], 1) ||
		r.IneqLin.Marginals[1] != 0 || len(r.Lower.Residual) != 2 || !math.IsInf(r.Lower.Residual[0], 1) {
		t.Fatalf("%+v", r)
	}
}

// Variables pinned by lb == ub are equality rows: the marginal goes to Lower
// when it is >= 0 and to Upper when it is negative.
func TestQuadprogPinnedVariables(t *testing.T) {
	// min ½‖x‖² + x₀ − x₁ with x₀ = 1 pinned and x₁ = −2 pinned.
	r, err := Quadprog(QPProblem{P: qpArr(t, []float64{1, 0, 0, 1}, 2, 2), Q: []float64{1, -1},
		Cons: QPConstraints{Bounds: []Bound{{1, 1}, {-2, -2}}}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	wantVec(t, "x", r.X, []float64{1, -2}, 1e-12)
	// P·x + q = (2, −3): marginals 2 (Lower) and −3 (Upper).
	wantVec(t, "lower", r.Lower.Marginals, []float64{2, 0}, 1e-12)
	wantVec(t, "upper", r.Upper.Marginals, []float64{0, -3}, 1e-12)
	wantVec(t, "y", r.Y, []float64{-2, 3}, 1e-12)
	if len(r.ActiveSet) != 2 || r.Optimality > 1e-12 {
		t.Fatalf("%+v", r)
	}
}

func TestQuadprogNotImplemented(t *testing.T) {
	id := qpArr(t, []float64{1, 0, 0, 1}, 2, 2)
	for name, tc := range map[string]struct {
		p    QPProblem
		opts *QPOptions
	}{
		"ADMM requested":    {QPProblem{P: id, Q: []float64{1, 1}}, &QPOptions{Method: QPADMM}},
		"P = 0 (LP)":        {QPProblem{Q: []float64{1, 1}}, nil},
		"singular P":        {QPProblem{P: qpArr(t, []float64{1, 1, 1, 1}, 2, 2), Q: []float64{1, 1}}, nil},
		"sparse constraint": {QPProblem{P: id, Q: []float64{1, 1}, Cons: QPConstraints{ASparse: qpCSC(t, 1, 2, []int{0, 1, 2}, []int32{0, 0}, []float64{1, 1}), L: []float64{1}, U: []float64{1}}}, nil},
	} {
		if _, err := Quadprog(tc.p, tc.opts); !errors.Is(err, ErrMethod) {
			t.Errorf("%s: %v, want ErrMethod", name, err)
		}
	}
}

func TestQuadprogRejects(t *testing.T) {
	inf, nan := math.Inf(1), math.NaN()
	id := qpArr(t, []float64{1, 0, 0, 1}, 2, 2)
	row := qpArr(t, []float64{1, 1}, 2)
	one := qpArr(t, []float64{1}, 1)
	bad := qpCSC(t, 2, 2, []int{0, 1, 2}, []int32{0, 1}, []float64{1, 1})
	bad.Indices[1] = 5 // corrupted after construction: NewCSC would have refused
	cases := []struct {
		name string
		p    QPProblem
		o    *QPOptions
		want error
	}{
		{"no Q", QPProblem{}, nil, ErrDimension},
		{"NaN in Q", QPProblem{Q: []float64{nan}}, nil, ErrInput},
		{"Inf in Q", QPProblem{Q: []float64{inf, 1}, P: id}, nil, ErrInput},
		{"P and PSparse", QPProblem{P: id, PSparse: qpCSC(t, 2, 2, []int{0, 1, 2}, []int32{0, 1}, []float64{1, 1}), Q: []float64{0, 0}}, nil, ErrInput},
		{"P shape", QPProblem{P: qpArr(t, []float64{1, 2, 3}, 3), Q: []float64{0, 0}}, nil, ErrDimension},
		{"P not square", QPProblem{P: qpArr(t, []float64{1, 2, 3, 4, 5, 6}, 3, 2), Q: []float64{0, 0}}, nil, ErrDimension},
		{"NaN in P", QPProblem{P: qpArr(t, []float64{1, nan, 0, 1}, 2, 2), Q: []float64{0, 0}}, nil, ErrInput},
		{"PSparse corrupt", QPProblem{PSparse: bad, Q: []float64{0, 0}}, nil, ErrInput},
		{"PSparse shape", QPProblem{PSparse: qpCSC(t, 3, 3, []int{0, 0, 0, 0}, nil, nil), Q: []float64{0, 0}}, nil, ErrDimension},
		{"PSparse NaN", QPProblem{PSparse: qpCSC(t, 2, 2, []int{0, 1, 2}, []int32{0, 1}, []float64{nan, 1}), Q: []float64{0, 0}}, nil, ErrInput},
		{"PSparse lower triangle", QPProblem{PSparse: qpCSC(t, 2, 2, []int{0, 2, 3}, []int32{0, 1, 1}, []float64{1, 1, 1}), Q: []float64{0, 0}}, nil, ErrInput},
		{"AEq shape", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{AEq: qpArr(t, []float64{1, 2, 3}, 3), BEq: one}}, nil, ErrDimension},
		{"AEq NaN", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{AEq: qpArr(t, []float64{1, nan}, 2), BEq: one}}, nil, ErrInput},
		{"AEq without BEq", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{AEq: row}}, nil, ErrDimension},
		{"BEq without AEq", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{BEq: one}}, nil, ErrDimension},
		{"BEq length", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{AEq: row, BEq: qpArr(t, []float64{1, 2}, 2)}}, nil, ErrDimension},
		{"BEq 2-D", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{AEq: row, BEq: qpArr(t, []float64{1}, 1, 1)}}, nil, ErrDimension},
		{"BEq NaN", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{AEq: row, BEq: qpArr(t, []float64{nan}, 1)}}, nil, ErrBounds},
		{"BEq Inf", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{AEq: row, BEq: qpArr(t, []float64{inf}, 1)}}, nil, ErrInput},
		{"BUb -Inf", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{AUb: row, BUb: qpArr(t, []float64{-inf}, 1)}}, nil, ErrBounds},
		{"BUb NaN", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{AUb: row, BUb: qpArr(t, []float64{nan}, 1)}}, nil, ErrBounds},
		{"Linear without A", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{Linear: []LinearConstraint{{}}}}, nil, ErrInput},
		{"Linear shape", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{Linear: []LinearConstraint{{A: qpArr(t, []float64{1, 2, 3}, 3)}}}}, nil, ErrDimension},
		{"Linear no rows", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{Linear: []LinearConstraint{{A: qpArr(t, nil, 0, 2)}}}}, nil, ErrInput},
		{"Linear Lb length", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{Linear: []LinearConstraint{{A: row, Lb: []float64{1, 2, 3}}}}}, nil, ErrDimension},
		{"Linear Ub length", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{Linear: []LinearConstraint{{A: row, Ub: []float64{1, 2, 3}}}}}, nil, ErrDimension},
		{"Linear Lb > Ub", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{Linear: []LinearConstraint{{A: row, Lb: []float64{2}, Ub: []float64{1}}}}}, nil, ErrBounds},
		{"Linear Lb +Inf", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{Linear: []LinearConstraint{{A: row, Lb: []float64{inf}}}}}, nil, ErrBounds},
		{"Linear Ub -Inf", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{Linear: []LinearConstraint{{A: row, Ub: []float64{-inf}}}}}, nil, ErrBounds},
		{"Linear Ub NaN", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{Linear: []LinearConstraint{{A: row, Ub: []float64{nan}}}}}, nil, ErrBounds},
		{"L without ASparse", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{L: []float64{1}}}, nil, ErrDimension},
		{"ASparse corrupt", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{ASparse: bad}}, nil, ErrInput},
		{"ASparse columns", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{ASparse: qpCSC(t, 1, 3, []int{0, 0, 0, 0}, nil, nil)}}, nil, ErrDimension},
		{"ASparse L length", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{ASparse: qpCSC(t, 1, 2, []int{0, 0, 0}, nil, nil), L: []float64{1, 2, 3}}}, nil, ErrDimension},
		{"ASparse U length", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{ASparse: qpCSC(t, 1, 2, []int{0, 0, 0}, nil, nil), U: []float64{1, 2, 3}}}, nil, ErrDimension},
		{"ASparse L > U", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{ASparse: qpCSC(t, 1, 2, []int{0, 0, 0}, nil, nil), L: []float64{2}, U: []float64{1}}}, nil, ErrBounds},
		// 2^31-1 empty rows are a valid CSC in a few bytes: refused before any
		// allocation proportional to them.
		{"ASparse huge row count", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{ASparse: &sparse.CSC[float64]{Rows: math.MaxInt32, Cols: 2, Indptr: []int{0, 0, 0}}, U: []float64{1}}}, nil, ErrInput},
		{"Bounds length", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{Bounds: []Bound{{0, 1}}}}, nil, ErrDimension},
		{"Bounds lo > hi", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{Bounds: []Bound{{2, 1}, {0, 1}}}}, nil, ErrBounds},
		{"Bounds NaN", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{Bounds: []Bound{{nan, 1}, {0, 1}}}}, nil, ErrBounds},
		{"Bounds lo +Inf", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{Bounds: []Bound{{inf, inf}, {0, 1}}}}, nil, ErrBounds},
		{"Bounds hi -Inf", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{Bounds: []Bound{{-inf, -inf}, {0, 1}}}}, nil, ErrBounds},
		{"unknown method", QPProblem{P: id, Q: []float64{0, 0}}, &QPOptions{Method: "simplex"}, ErrMethod},
		{"active set, P singular", QPProblem{P: qpArr(t, []float64{1, 1, 1, 1}, 2, 2), Q: []float64{0, 0}}, &QPOptions{Method: QPActiveSet}, ErrInput},
		{"active set, P = 0", QPProblem{Q: []float64{0, 0}}, &QPOptions{Method: QPActiveSet}, ErrInput},
		{"active set, ASparse", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{ASparse: qpCSC(t, 1, 2, []int{0, 1, 2}, []int32{0, 0}, []float64{1, 1}), U: []float64{1}}}, &QPOptions{Method: QPActiveSet}, ErrInput},
		{"negative MaxIter", QPProblem{P: id, Q: []float64{0, 0}}, &QPOptions{MaxIter: -1}, ErrInput},
		{"negative PolishRefine", QPProblem{P: id, Q: []float64{0, 0}}, &QPOptions{PolishRefine: -1}, ErrInput},
		{"negative CheckInterval", QPProblem{P: id, Q: []float64{0, 0}}, &QPOptions{CheckInterval: -1}, ErrInput},
		{"NaN FeasTol", QPProblem{P: id, Q: []float64{0, 0}}, &QPOptions{FeasTol: nan}, ErrInput},
		{"Inf EpsAbs", QPProblem{P: id, Q: []float64{0, 0}}, &QPOptions{EpsAbs: inf}, ErrInput},
		{"negative Rho", QPProblem{P: id, Q: []float64{0, 0}}, &QPOptions{Rho: -1}, ErrInput},
		{"Alpha 2", QPProblem{P: id, Q: []float64{0, 0}}, &QPOptions{Alpha: 2}, ErrInput},
		{"X0 length", QPProblem{P: id, Q: []float64{0, 0}}, &QPOptions{X0: []float64{1}}, ErrDimension},
		{"Y0 length", QPProblem{P: id, Q: []float64{0, 0}}, &QPOptions{Y0: []float64{1}}, ErrDimension},
		{"X0 NaN", QPProblem{P: id, Q: []float64{0, 0}}, &QPOptions{X0: []float64{nan, 0}}, ErrInput},
		{"WorkingSet range", QPProblem{P: id, Q: []float64{0, 0}}, &QPOptions{WorkingSet: []int{0}}, ErrInput},
		{"WorkingSet repeated", QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{AEq: row, BEq: one}}, &QPOptions{WorkingSet: []int{0, 0}}, ErrInput},
	}
	for _, c := range cases {
		r, err := Quadprog(c.p, c.o)
		if !errors.Is(err, c.want) || r != nil {
			t.Errorf("%s: got (%v, %v), want %v", c.name, r, err, c.want)
		}
	}
	// Valid options are accepted and ignored by the direct solve.
	ok := &QPOptions{X0: []float64{0, 0}, Y0: []float64{0}, WorkingSet: []int{0}, Alpha: 1.5, Rho: 1, MaxIter: 7, FeasTol: 1e-8}
	if _, err := Quadprog(QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{AEq: row, BEq: one}}, ok); err != nil {
		t.Fatal(err)
	}
}

// The sizes derived from the input are checked against qpMaxSize (math.MaxInt32
// in production, lowered here) before anything is allocated for them, and the
// direct solve against qpDirectLimit.
func TestQuadprogSizeLimits(t *testing.T) {
	oldMax, oldDirect := qpMaxSize, qpDirectLimit
	defer func() { qpMaxSize, qpDirectLimit = oldMax, oldDirect }()
	id := qpArr(t, []float64{1, 0, 0, 1}, 2, 2)
	row := qpArr(t, []float64{1, 1}, 2)
	one := qpArr(t, []float64{1}, 1)
	sp := qpCSC(t, 1, 2, []int{0, 1, 2}, []int32{0, 0}, []float64{1, 1})
	pu := qpCSC(t, 2, 2, []int{0, 1, 2}, []int32{0, 1}, []float64{1, 1})
	for name, tc := range map[string]struct {
		limit int64
		p     QPProblem
	}{
		"n":             {1, QPProblem{Q: []float64{0, 0}}},
		"nnz(P) dense":  {4, QPProblem{P: id, Q: []float64{0, 0}}},
		"nnz(P) sparse": {3, QPProblem{PSparse: pu, Q: []float64{0, 0}}},
		"AEq":           {5, QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{AEq: row, BEq: one}}},
		"AUb":           {5, QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{AUb: row, BUb: one}}},
		"Linear":        {5, QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{Linear: []LinearConstraint{{A: row}}}}},
		"ASparse":       {5, QPProblem{P: id, Q: []float64{0, 0}, Cons: QPConstraints{ASparse: sp, U: []float64{1}}}},
		"bounds (rows)": {3, QPProblem{Q: []float64{0, 0}, Cons: QPConstraints{Bounds: []Bound{{0, 1}, {0, 1}}}}},
	} {
		qpMaxSize = tc.limit
		if _, err := Quadprog(tc.p, nil); !errors.Is(err, ErrInput) {
			t.Errorf("%s: %v, want ErrInput", name, err)
		}
	}
	qpMaxSize = oldMax
	// The direct solve: n·n for a sparse P too large to densify (QPActiveSet
	// cannot be chosen, so QPAuto goes to ADMM; asking for the active set
	// reports the P that cannot be factored), and m·n for the constraints.
	qpDirectLimit = 3
	if _, err := Quadprog(QPProblem{PSparse: pu, Q: []float64{0, 0}}, &QPOptions{Method: QPActiveSet}); !errors.Is(err, ErrInput) {
		t.Errorf("dense P above the limit: %v", err)
	}
	qpDirectLimit = oldDirect
	if _, err := symmetricPLimit(t); err == nil {
		t.Error("symmetricP accepted n·n above the limit")
	}
}

func symmetricPLimit(t *testing.T) (*qpModel, error) {
	t.Helper()
	old := qpDirectLimit
	defer func() { qpDirectLimit = old }()
	qpDirectLimit = 3
	md := &qpModel{n: 2}
	_, err := md.symmetricP()
	return md, err
}

// assemble fills the result blocks of every family and checks the recomputed
// residuals: the marginals here are made up, so the stationarity residual is
// what they say it is.
func TestQPAssembleBlocks(t *testing.T) {
	id := qpArr(t, []float64{2, 0, 0, 2}, 2, 2)
	sp := qpCSC(t, 2, 2, []int{0, 1, 3}, []int32{0, 0, 1}, []float64{1, 1, 1}) // rows (1, 1) and (0, 1)
	md, err := newQPModel(QPProblem{P: id, Q: []float64{1, 1}, Cons: QPConstraints{
		AEq: qpArr(t, []float64{1, 1}, 1, 2), BEq: qpArr(t, []float64{1}, 1),
		AUb: qpArr(t, []float64{1, 0, 0, 1}, 2, 2), BUb: qpArr(t, []float64{1, math.Inf(1)}, 2),
		Linear:  []LinearConstraint{{A: qpArr(t, []float64{0, 1}, 2), Lb: []float64{-1}, Ub: []float64{3}}},
		ASparse: sp, L: []float64{-2, math.Inf(-1)}, U: []float64{2, math.Inf(1)},
		Bounds: []Bound{{0, math.Inf(1)}, {math.Inf(-1), 4}}}})
	if err != nil {
		t.Fatal(err)
	}
	// canonical rows: AEq0, AUb0 (AUb1 vacuous), Linear0, ASparse0 (ASparse1
	// vacuous), bound 0 (lower only), bound 1 (upper only).
	if len(md.rows) != 6 || md.nEq != 1 || md.nUb != 2 || md.nLin != 1 || md.nSp != 2 {
		t.Fatalf("rows %+v", md.rows)
	}
	x := []float64{0.5, 0.5}
	r := md.assemble(x, []float64{1, -1, 2, -3, 4, -5}, QPActiveSet, 1e-9)
	if r.Success || r.Status != QPNumericalFail { // made-up marginals: stationarity fails
		t.Fatalf("%+v", r)
	}
	wantVec(t, "eq marg", r.EqLin.Marginals, []float64{1}, 0)
	wantVec(t, "ub marg", r.IneqLin.Marginals, []float64{-1, 0}, 0)
	wantVec(t, "ub res", r.IneqLin.Residual[:1], []float64{0.5}, 1e-12)
	if !math.IsInf(r.IneqLin.Residual[1], 1) {
		t.Fatalf("vacuous row residual %v", r.IneqLin.Residual)
	}
	wantVec(t, "lin marg", r.Linear.Marginals, []float64{2}, 0)
	wantVec(t, "lin res", r.Linear.Residual, []float64{1.5}, 1e-12) // min(0.5+1, 3-0.5)
	wantVec(t, "sparse marg", r.Sparse.Marginals, []float64{-3, 0}, 0)
	wantVec(t, "lower marg", r.Lower.Marginals, []float64{4, 0}, 0)
	wantVec(t, "upper marg", r.Upper.Marginals, []float64{0, -5}, 0)
	// ASparse row 0 is (1, 1): a·x = 1, so the nearer side is min(1+2, 2−1) = 1.
	if !near(r.Sparse.Residual[0], 1, 1e-12) || !math.IsInf(r.Sparse.Residual[1], 1) {
		t.Fatalf("sparse residual %v", r.Sparse.Residual)
	}
	// P·x + q = (2, 2); Σ marginal·a = (1 − 1 + 0 − 3 + 4, 1 + 0 + 2 − 3 − 5) =
	// (1, −5): the identity fails by (1, 7).
	if !near(r.Optimality, 7, 1e-12) {
		t.Fatalf("optimality %v", r.Optimality)
	}
	// A sparse P goes through px too.
	mp, err := newQPModel(QPProblem{PSparse: qpCSC(t, 2, 2, []int{0, 1, 3}, []int32{0, 0, 1}, []float64{2, 1, 2}), Q: []float64{0, 0}})
	if err != nil {
		t.Fatal(err)
	}
	wantVec(t, "P·x", mp.px([]float64{1, 1}), []float64{3, 3}, 1e-12)
	// A NaN residual fails both tests.
	md.q[0] = math.NaN()
	if r := md.assemble(x, make([]float64, 6), QPActiveSet, 1e-9); r.Success {
		t.Error("NaN q reported as optimal")
	}
}

// A real solution of a problem with all-equality families passes the tests of
// assemble; the identity P·x + q = Σ marginal·a holds to rounding.
func TestQuadprogStationarityIdentity(t *testing.T) {
	p := QPProblem{
		P: qpArr(t, []float64{4, 1, 0, 1, 3, 1, 0, 1, 2}, 3, 3), Q: []float64{1, -2, 3},
		Cons: QPConstraints{
			AEq: qpArr(t, []float64{1, 1, 1}, 1, 3), BEq: qpArr(t, []float64{1}, 1),
			Linear: []LinearConstraint{{A: qpArr(t, []float64{1, -1, 0}, 1, 3), Lb: []float64{0.5}, Ub: []float64{0.5}}},
			Bounds: []Bound{Unbounded(), Unbounded(), {0.25, 0.25}},
		}}
	r, err := Quadprog(p, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !r.Success || r.Optimality > 1e-12 || r.ConstrViolation > 1e-12 {
		t.Fatalf("%+v", r)
	}
	if !near(r.X[2], 0.25, 1e-12) || !near(r.X[0]-r.X[1], 0.5, 1e-12) || !near(r.X[0]+r.X[1]+r.X[2], 1, 1e-12) {
		t.Fatalf("x = %v", r.X)
	}
	if len(r.ActiveSet) != 3 || len(r.Y) != 3 {
		t.Fatalf("%+v", r)
	}
}
