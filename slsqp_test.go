package optimize

import (
	"errors"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/go-ndarray/ndarray"
)

// Unit tests of the dense kernels on inputs that reach their rare branches.

func TestDlapy2(t *testing.T) {
	nan := math.NaN()
	if !math.IsNaN(dlapy2(nan, 1)) || !math.IsNaN(dlapy2(1, nan)) {
		t.Error("NaN not propagated")
	}
	if got := dlapy2(3, 4); got != 5 {
		t.Errorf("dlapy2(3,4) = %v", got)
	}
	if got := dlapy2(0, -2); got != 2 {
		t.Errorf("dlapy2(0,-2) = %v", got)
	}
}

// reflect applies I - tau*[1;v][1;v]' to the vector (alpha0; x0).
func reflect(alpha0 float64, x0, v []float64, tau float64) []float64 {
	w := alpha0
	for i := range v {
		w += v[i] * x0[i]
	}
	out := []float64{alpha0 - tau*w}
	for i := range v {
		out = append(out, x0[i]-tau*w*v[i])
	}
	return out
}

func TestHouseholder(t *testing.T) {
	for _, tc := range []struct {
		alpha float64
		x     []float64
	}{
		{3, []float64{4}},
		{-3, []float64{4, 0}},
		{1e-310, []float64{1e-310, 2e-310}}, // rescaling loop
		{-1e-310, []float64{3e-310}},
		{2, []float64{0, 0}},
		{-2, []float64{0}},
	} {
		for _, positive := range []bool{false, true} {
			a, x := tc.alpha, clone(tc.x)
			var tau float64
			if positive {
				tau = dlarfgp(len(x)+1, &a, x, 1)
			} else {
				tau = dlarfg(len(x)+1, &a, x, 1)
			}
			got := reflect(tc.alpha, tc.x, x, tau)
			scale := math.Abs(tc.alpha) + nrm2(len(tc.x), tc.x, 1)
			if math.Abs(got[0]-a) > 1e-14*scale {
				t.Errorf("%v %v positive=%v: H*v[0] = %g, beta %g", tc.alpha, tc.x, positive, got[0], a)
			}
			for i := 1; i < len(got); i++ {
				if math.Abs(got[i]) > 1e-14*scale {
					t.Errorf("%v %v positive=%v: H*v[%d] = %g", tc.alpha, tc.x, positive, i, got[i])
				}
			}
			if positive && a < 0 {
				t.Errorf("dlarfgp beta %g < 0", a)
			}
		}
	}
	a := 5.0
	if dlarfg(1, &a, nil, 1) != 0 || dlarfgp(0, &a, nil, 1) != 0 {
		t.Error("trivial reflectors should have tau 0")
	}
}

func TestDlartgp(t *testing.T) {
	for _, fg := range [][2]float64{
		{3, 4}, {0, -2}, {-2, 0}, {1e300, 1e300}, {1e-300, -1e-300}, {-1e200, 3e200}, {5e-320, 5e-320},
	} {
		cs, sn, r := dlartgp(fg[0], fg[1])
		if r < 0 {
			t.Errorf("%v: r = %g", fg, r)
		}
		scale := math.Max(math.Abs(fg[0]), math.Abs(fg[1]))
		if d := cs*fg[0]/scale + sn*fg[1]/scale - r/scale; math.Abs(d) > 1e-12 {
			t.Errorf("%v: rotation residual %g", fg, d)
		}
		if d := -sn*fg[0]/scale + cs*fg[1]/scale; math.Abs(d) > 1e-12 {
			t.Errorf("%v: rotation does not annihilate g (%g)", fg, d)
		}
	}
}

// trsmRUNRef is the reference BLAS loop that trsmRUN unrolls.
func trsmRUNRef(m, n int, a []float64, lda int, b []float64, ldb int) {
	for j := 0; j < n; j++ {
		for k := 0; k < j; k++ {
			if akj := a[k+j*lda]; akj != 0 {
				for i := 0; i < m; i++ {
					b[i+j*ldb] -= float64(akj * b[i+k*ldb])
				}
			}
		}
		t := 1 / a[j+j*lda]
		for i := 0; i < m; i++ {
			b[i+j*ldb] = float64(t * b[i+j*ldb])
		}
	}
}

func trsmInput(m, n int, zeros bool) (a, b []float64) {
	rng := rand.New(rand.NewPCG(3, 4))
	a, b = make([]float64, n*n), make([]float64, m*n)
	for j := 0; j < n; j++ {
		for k := 0; k <= j; k++ {
			if !zeros || rng.IntN(3) != 0 || k == j {
				a[k+j*n] = rng.NormFloat64()
			}
		}
		a[j+j*n] += 4
	}
	for i := range b {
		b[i] = rng.NormFloat64()
	}
	return a, b
}

func TestTrsmMatchesReference(t *testing.T) {
	for _, zeros := range []bool{false, true} {
		a, b := trsmInput(13, 11, zeros)
		want := clone(b)
		trsmRUNRef(13, 11, a, 11, want, 13)
		trsmRUN(13, 11, a, 11, b, 13)
		for i := range b {
			if math.Float64bits(b[i]) != math.Float64bits(want[i]) {
				t.Fatalf("zeros=%v: b[%d] = %v, reference %v", zeros, i, b[i], want[i])
			}
		}
	}
	trsmRUN(0, 3, nil, 3, nil, 1)
}

func BenchmarkTrsm(b *testing.B) {
	a, g := trsmInput(101, 50, false)
	work := make([]float64, len(g))
	for _, impl := range []struct {
		name string
		f    func(m, n int, a []float64, lda int, b []float64, ldb int)
	}{{"unrolled", trsmRUN}, {"reference", trsmRUNRef}} {
		b.Run(impl.name, func(b *testing.B) {
			for b.Loop() {
				copy(work, g)
				impl.f(101, 50, a, 50, work, 101)
			}
		})
	}
}

func TestLstsqPivotedRankDeficient(t *testing.T) {
	// Two equal columns: rank 1.
	a := []float64{1, 2, 3, 1, 2, 3}
	b := []float64{1, 2, 3}
	if r := lstsqPivoted(3, 2, a, b, 1e-8, make([]float64, 4)); r != 1 {
		t.Fatalf("rank %d, want 1", r)
	}
	if got := b[0] + b[1]; math.Abs(got-1) > 1e-12 {
		t.Errorf("x0 + x1 = %g, want 1", got)
	}
}

func TestLsiRankDeficient(t *testing.T) {
	a := []float64{1, 0, 0, 0} // 2×2 column-major, second column zero
	mode, _ := lsi(&arena{}, 2, 1, 2, a, []float64{1, 1}, []float64{1, 1}, []float64{0}, make([]float64, 2))
	if mode != 5 {
		t.Errorf("mode %d, want 5", mode)
	}
}

func TestLdpIncompatible(t *testing.T) {
	// x >= 1 and -x >= 0: incompatible.
	ws := &arena{}
	mode, _ := ldp(ws, 2, 1, []float64{1, -1}, []float64{1, 0}, make([]float64, 1))
	if mode != 4 {
		t.Errorf("mode %d, want 4", mode)
	}
	// The NNLS iteration limit surfaces as mode 3.
	ldpMaxIter = 0
	mode, _ = ldp(ws, 1, 1, []float64{1}, []float64{1}, make([]float64, 1))
	ldpMaxIter = 3
	if mode != 3 {
		t.Errorf("mode %d, want 3", mode)
	}
	// x >= 1 alone: x = 1 with multiplier 1.
	x := make([]float64, 1)
	mode, mults := ldp(ws, 1, 1, []float64{1}, []float64{1}, x)
	if mode != 1 || math.Abs(x[0]-1) > 1e-15 || math.Abs(mults[0]-1) > 1e-15 {
		t.Errorf("mode %d x %v mults %v", mode, x, mults)
	}
}

// TestNNLSRandom checks the KKT conditions of nnls on random problems, which
// also drives its inner loop (coefficients leaving the passive set).
func TestNNLSRandom(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for trial := range 3000 {
		m, n := 1+rng.IntN(12), 1+rng.IntN(12)
		a := make([]float64, m*n)
		for i := range a {
			a[i] = rng.NormFloat64()
		}
		b := make([]float64, m)
		for i := range b {
			b[i] = rng.NormFloat64()
		}
		a0, b0 := clone(a), clone(b)
		x := make([]float64, n)
		_, info := nnls(&arena{}, m, n, a, b, x, 3*n)
		if info != 1 {
			continue
		}
		// Gradient w = A'(b - A x) must be <= 0 where x = 0 and ~0 where x > 0.
		r := clone(b0)
		for j := 0; j < n; j++ {
			for i := 0; i < m; i++ {
				r[i] -= a0[i+j*m] * x[j]
			}
		}
		for j := 0; j < n; j++ {
			w := 0.0
			for i := 0; i < m; i++ {
				w += a0[i+j*m] * r[i]
			}
			if x[j] < 0 || (x[j] == 0 && w > 1e-9) || (x[j] > 0 && math.Abs(w) > 1e-9) {
				t.Fatalf("trial %d: x[%d] = %g, w = %g", trial, j, x[j], w)
			}
		}
	}
	// Two passive coefficients reach zero at the same step (x1 = x2 = -4 in
	// the unconstrained fit once column 3 enters): both leave the set.
	a := []float64{1, 0, 0, 0, 1, 0, 0.1, 0.1, 0.1}
	x := make([]float64, 3)
	if _, info := nnls(&arena{}, 3, 3, a, []float64{1, 1, 5}, x, 9); info != 1 || x[0] != 0 || x[1] != 0 || math.Abs(x[2]-(7.0/0.03)/10) > 1e-9 {
		t.Errorf("info %d x %v", info, x)
	}
	// One inner iteration allowed: info 3.
	a = []float64{1, 0, 1, 1, 0, 1}
	if _, info := nnls(&arena{}, 2, 3, a, []float64{1, -1}, make([]float64, 3), 1); info != 3 {
		t.Errorf("info %d, want 3", info)
	}
}

func TestLDLUpdateNegative(t *testing.T) {
	// a = 1 (1×1); subtracting more than a forces the t >= 0 safeguard.
	a := []float64{1}
	ldlUpdate(1, a, []float64{2}, -1, make([]float64, 1))
	if !(a[0] > 0) {
		t.Errorf("factor %g is not positive", a[0])
	}
	a = []float64{4, 0.5, 2}
	ldlUpdate(2, a, []float64{10, 1}, 1, nil) // alpha > 4 branch
	if !(a[0] > 4) {
		t.Errorf("factor %v", a)
	}
}

// Driver tests.

func quad(x []float64) float64 {
	s := 0.0
	for i, v := range x {
		s += float64(float64(i+1) * (v - 1) * (v - 1))
	}
	return s
}

func quadGrad(x, g []float64) {
	for i, v := range x {
		g[i] = float64(2 * float64(i+1) * (v - 1))
	}
}

func TestSLSQPErrors(t *testing.T) {
	p := Problem{Func: quad}
	a2, _ := ndarray.FromData([]float64{1, 2}, 2)
	a3, _ := ndarray.FromData([]float64{1, 2, 3}, 3)
	a0, _ := ndarray.FromData([]float64{}, 0, 2)
	a22, _ := ndarray.FromData([]float64{1, 2, 3, 4}, 2, 2)
	a2x3, _ := ndarray.FromData([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	a3d, _ := ndarray.FromData([]float64{1, 2}, 1, 1, 2)
	f := func(x, c []float64) { c[0] = x[0] }
	for name, cons := range map[string][]ConstraintSpec{
		"nil spec":         {nil},
		"no func":          {Constraint{M: 1}},
		"no rows":          {Constraint{Func: f}},
		"bad kind":         {Constraint{Kind: 7, M: 1, Func: f}},
		"nl no func":       {NonlinearConstraint{M: 1}},
		"nl no rows":       {NonlinearConstraint{Func: f}},
		"nl lb length":     {NonlinearConstraint{M: 1, Func: f, Lb: []float64{0, 1}}},
		"nl ub length":     {NonlinearConstraint{M: 1, Func: f, Ub: []float64{0, 1}}},
		"nl lb > ub":       {NonlinearConstraint{M: 1, Func: f, Lb: []float64{2}, Ub: []float64{1}}},
		"nl NaN":           {NonlinearConstraint{M: 1, Func: f, Lb: []float64{math.NaN()}}},
		"lin no A":         {LinearConstraint{}},
		"lin 1-D mismatch": {LinearConstraint{A: a3}},
		"lin 2-D mismatch": {LinearConstraint{A: a2x3}},
		"lin no rows":      {LinearConstraint{A: a0}},
		"lin 3-D":          {LinearConstraint{A: a3d}},
	} {
		_, err := Minimize(p, []float64{0, 0}, SLSQP, &Options{Constraints: cons})
		if err == nil {
			t.Errorf("%s: no error", name)
		}
	}
	_ = a22
	_ = a2
	_, err := Minimize(p, []float64{0, 0}, BFGS, &Options{Constraints: []ConstraintSpec{Constraint{M: 1, Func: f}}})
	if !errors.Is(err, ErrMethod) {
		t.Errorf("BFGS with constraints: %v", err)
	}
}

func TestSLSQPConstraintForms(t *testing.T) {
	// min sum i(x_i-1)^2 s.t. x0 + x1 <= 1 (1-D LinearConstraint, broadcast
	// bound), x1 == 0.25 (NonlinearConstraint lb == ub), x0 >= -5 (lower
	// bound only), an ignored component unbounded on both sides.
	a, _ := ndarray.FromData([]float64{1, 1}, 2)
	cons := []ConstraintSpec{
		LinearConstraint{A: a, Ub: []float64{1}},
		NonlinearConstraint{M: 3,
			Func: func(x, c []float64) { c[0], c[1], c[2] = x[1], x[0], x[0]*x[1] },
			Lb:   []float64{0.25, -5, math.Inf(-1)}, Ub: []float64{0.25, math.Inf(1), math.Inf(1)}},
	}
	r, err := Minimize(Problem{Func: quad, Grad: quadGrad}, []float64{0, 0}, SLSQP, &Options{Constraints: cons})
	if err != nil || !r.Success {
		t.Fatalf("%v %+v", err, r)
	}
	if math.Abs(r.X[0]-0.75) > 1e-6 || math.Abs(r.X[1]-0.25) > 1e-6 {
		t.Errorf("x = %v, want [0.75 0.25]", r.X)
	}
	if len(r.Multipliers) != 3 {
		t.Errorf("multipliers %v", r.Multipliers)
	}
}

func TestSLSQPCallbackAndFD(t *testing.T) {
	calls := 0
	o := &Options{
		Bounds:   []Bound{{0, 0.5}, Unbounded()},
		FDMethod: FD3Point,
		Constraints: []ConstraintSpec{Constraint{Kind: Ineq, M: 1,
			Func: func(x, c []float64) { c[0] = 1 - x[0] - x[1] }}},
		Callback: func(x []float64, f float64) bool { calls++; return calls == 2 },
	}
	r, err := Minimize(Problem{Func: quad}, []float64{0, 0}, SLSQP, o)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 2 || r.Status != 99 || r.Success {
		t.Errorf("calls %d status %d", calls, r.Status)
	}
	o.Callback = nil
	o.FDMethod, o.FiniteDiffRelStep = FD2Point, 1e-7
	r, err = Minimize(Problem{Func: quad}, []float64{0, 0}, SLSQP, o)
	// (x0-1)² + 2(x1-1)² on x0 + x1 <= 1: x = (1/3, 2/3), inside x0 <= 0.5.
	if err != nil || !r.Success || math.Abs(r.X[0]-1.0/3) > 1e-5 {
		t.Errorf("%v %+v", err, r)
	}
}

func TestSLSQPFixedVariables(t *testing.T) {
	cons := []ConstraintSpec{Constraint{Kind: Ineq, M: 1,
		Func: func(x, c []float64) { c[0] = 3 - x[0] - x[2] },
		Jac:  func(x, j []float64) { j[0], j[1], j[2] = -1, 0, -1 }}}
	bounds := []Bound{{-5, 5}, {2, 2}, {-5, 5}}
	fg := func(x, g []float64) float64 { quadGrad(x, g); return quad(x) }
	stops := 0
	for name, p := range map[string]Problem{
		"Func":       {Func: quad},
		"Func+Grad":  {Func: quad, Grad: quadGrad},
		"FuncGrad":   {FuncGrad: fg},
		"constraint": {Func: quad, Grad: quadGrad},
	} {
		o := &Options{Bounds: bounds, Constraints: cons,
			Callback: func(x []float64, _ float64) bool {
				if len(x) != 3 || x[1] != 2 {
					t.Errorf("callback x = %v", x)
				}
				return false
			}}
		if name != "Func" {
			// Force the removal path through a finite-difference constraint.
			o.Constraints = []ConstraintSpec{cons[0], Constraint{Kind: Ineq, M: 1,
				Func: func(x, c []float64) { c[0] = 10 + x[0] }}}
		}
		r, err := Minimize(p, []float64{0, 2, 0}, SLSQP, o)
		if err != nil || !r.Success {
			t.Fatalf("%s: %v %+v", name, err, r)
		}
		if r.X[1] != 2 || !math.IsNaN(r.Jac[1]) {
			t.Errorf("%s: x %v jac %v", name, r.X, r.Jac)
		}
		// Optimum of x0-1)^2 + 3(x2-1)^2 on x0 + x2 <= 3 is unconstrained: (1, 1).
		if math.Abs(r.X[0]-1) > 1e-5 || math.Abs(r.X[2]-1) > 1e-5 {
			t.Errorf("%s: x %v", name, r.X)
		}
		stops++
	}
	// All fixed, without constraints and with a violated equality.
	r, err := Minimize(Problem{Func: quad}, []float64{0}, SLSQP, &Options{Bounds: []Bound{{2, 2}}})
	if err != nil || !r.Success || r.Fun != 1 || r.Nfev != 1 {
		t.Errorf("%v %+v", err, r)
	}
	r, _ = Minimize(Problem{Func: quad}, []float64{0}, SLSQP, &Options{Bounds: []Bound{{2, 2}},
		Constraints: []ConstraintSpec{Constraint{Kind: Eq, M: 1, Func: func(x, c []float64) { c[0] = x[0] - 1 }}}})
	if r.Success || !strings.Contains(r.Message, "Maximum violation: 1.0") {
		t.Errorf("%+v", r)
	}
}

// TestSLSQPRelaxedConvergence drives the state machine directly into its
// gradient-update step with a zero step (s'Bs = 0, a singular update), on
// the sixth BFGS reset: SLSQP then applies its relaxed test (tol = 10*acc).
func TestSLSQPRelaxedConvergence(t *testing.T) {
	for _, tc := range []struct {
		d    float64
		mode int
	}{{-5e-6, 0}, {-5e-5, 8}} {
		st := newSLSQPState(1, 1, 1, 1e-6, 10)
		st.mode, st.reset, st.f0 = -1, 5, 3
		st.bfgs[0] = 1
		nan := math.NaN()
		st.body(3, []float64{1}, []float64{1}, 1, []float64{tc.d}, []float64{0},
			make([]float64, 5), []float64{nan, nan}, []float64{nan, nan})
		if st.mode != tc.mode {
			t.Errorf("d = %g: mode %d, want %d", tc.d, st.mode, tc.mode)
		}
	}
}

// TestSLSQPPathological feeds NaN, Inf and degenerate data: no panic, a
// bounded number of evaluations, and a non-success status where it applies.
func TestSLSQPPathological(t *testing.T) {
	nan := math.NaN()
	for name, tc := range map[string]struct {
		f    func(x []float64) float64
		cons []ConstraintSpec
		x0   []float64
	}{
		"NaN objective": {func(x []float64) float64 { return nan }, nil, []float64{1, 2}},
		"Inf objective": {func(x []float64) float64 { return math.Inf(1) }, nil, []float64{1, 2}},
		"NaN constraint": {quad, []ConstraintSpec{Constraint{Kind: Ineq, M: 1,
			Func: func(x, c []float64) { c[0] = nan }}}, []float64{1, 2}},
		"NaN start": {quad, nil, []float64{nan, 0}},
		"zero Jac eq": {quad, []ConstraintSpec{Constraint{Kind: Eq, M: 1,
			Func: func(x, c []float64) { c[0] = 1 }, Jac: func(x, j []float64) { j[0], j[1] = 0, 0 }}}, []float64{0, 0}},
		"cliff": {func(x []float64) float64 {
			if x[0] > 0.5 {
				return -1e300
			}
			return quad(x)
		}, nil, []float64{0, 0}},
	} {
		evals := 0
		f := tc.f
		r, err := Minimize(Problem{Func: func(x []float64) float64 { evals++; return f(x) }},
			tc.x0, SLSQP, &Options{Constraints: tc.cons})
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if evals > 100*(len(tc.x0)+12)*11 {
			t.Errorf("%s: %d evaluations", name, evals)
		}
		if name != "cliff" && r.Success {
			t.Errorf("%s: success %+v", name, r)
		}
	}
}
