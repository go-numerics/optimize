package optimize

import (
	"errors"
	"fmt"
	"math"
	deepeq "reflect"
	"testing"
)

func dgBox(n int, lo, hi float64) []Bound {
	b := make([]Bound, n)
	for i := range b {
		b[i] = Bound{lo, hi}
	}
	return b
}

func TestDirectDocstring(t *testing.T) {
	// SciPy's documented example (written with products instead of **).
	bnd := dgBox(2, -4, 4)
	r, err := Direct(dgStyblinskiTangDoc, bnd, nil)
	if err != nil {
		t.Fatal(err)
	}
	if r.Nfev != 2011 || math.Abs(r.Fun-(-78.3323279095383)) > 1e-12 || r.Status != 1 || r.Success {
		t.Fatalf("got %+v", r)
	}
	r, _ = Direct(dgStyblinskiTangDoc, bnd, &DirectOptions{LenTol: 1e-3})
	if r.Nfev != 207 || math.Abs(r.Fun-(-78.33230330754142)) > 1e-12 || r.Status != 5 || !r.Success {
		t.Fatalf("got %+v", r)
	}
}

func TestDirectValidation(t *testing.T) {
	f := dgShifted(2)
	nan, inf := math.NaN(), math.Inf(1)
	big := math.MaxFloat64
	cases := []struct {
		name string
		f    func([]float64) float64
		b    []Bound
		o    *DirectOptions
		want error
	}{
		{"nil objective", nil, dgBox(2, 0, 1), nil, ErrInput},
		{"no dimension", f, nil, nil, ErrInput},
		{"too many dimensions", f, make([]Bound, 1<<20+1), nil, ErrInput},
		{"inverted", f, []Bound{{1, 0}, {0, 1}}, nil, ErrBounds},
		{"degenerate", f, []Bound{{1, 1}, {0, 1}}, nil, ErrBounds},
		{"NaN bound", f, []Bound{{nan, 1}, {0, 1}}, nil, ErrBounds},
		{"infinite lower", f, []Bound{{-inf, 1}, {0, 1}}, nil, ErrBounds},
		{"infinite upper", f, []Bound{{0, inf}, {0, 1}}, nil, ErrBounds},
		{"range overflows", f, []Bound{{-big, big}, {0, 1}}, nil, ErrBounds},
		{"NaN eps", f, dgBox(2, 0, 1), &DirectOptions{Eps: nan}, ErrInput},
		{"infinite eps", f, dgBox(2, 0, 1), &DirectOptions{Eps: inf}, ErrInput},
		{"VolTol > 1", f, dgBox(2, 0, 1), &DirectOptions{VolTol: 2}, ErrInput},
		{"LenTol > 1", f, dgBox(2, 0, 1), &DirectOptions{LenTol: 2}, ErrInput},
		{"FMinRtol > 1", f, dgBox(2, 0, 1), &DirectOptions{FMinRtol: 2}, ErrInput},
		{"NaN tolerance", f, dgBox(2, 0, 1), &DirectOptions{LenTol: nan}, ErrInput},
		{"negative MaxFev", f, dgBox(2, 0, 1), &DirectOptions{MaxFev: -1}, ErrInput},
		{"negative MaxIter", f, dgBox(2, 0, 1), &DirectOptions{MaxIter: -1}, ErrInput},
		{"huge MaxFev", f, dgBox(2, 0, 1), &DirectOptions{MaxFev: math.MaxInt32}, ErrInput},
		{"huge MaxIter", f, dgBox(2, 0, 1), &DirectOptions{MaxIter: math.MaxInt32}, ErrInput},
	}
	for _, c := range cases {
		calls := 0
		g := c.f
		if g != nil {
			g = func(x []float64) float64 { calls++; return c.f(x) }
		}
		r, err := Direct(g, c.b, c.o)
		if !errors.Is(err, c.want) || r != nil || calls != 0 {
			t.Errorf("%s: got %v, %v after %d calls; want %v", c.name, r, err, calls, c.want)
		}
	}
}

func TestDirectMessages(t *testing.T) {
	for _, c := range []struct {
		code int
		want string
	}{
		{1, "Number of function evaluations done is larger than maxfun=2000"},
		{2, "Number of iterations is larger than maxiter=1000"},
		{3, "The best function value found is within a relative error=0.0001 of the (known) global optimum f_min"},
		{4, "The volume of the hyperrectangle containing the lowest function value found is below vol_tol=1e-16"},
		{5, "The side length measure of the hyperrectangle containing the lowest function value found is below len_tol=1e-06"},
		{-1, "u[i] < l[i] for some i"},
		{-2, "maxfun is too large"},
		{-3, "Initialization failed"},
		{-4, "There was an error in the creation of the sample points"},
		{-5, "An error occurred while the function was sampled"},
		{-6, "Maximum number of levels has been reached."},
		{-100, "Out of memory"},
		{-101, "Invalid arguments"},
		{-102, "Forced stop"},
	} {
		if got := directMessage(c.code, 2000, 1000, 1e-4, 1e-16, 1e-6); got != c.want {
			t.Errorf("code %d: %q, want %q", c.code, got, c.want)
		}
	}
}

func TestDirectSizes(t *testing.T) {
	for _, c := range []struct {
		n                                  int
		maxf, maxt                         int64
		maxfunc, maxfAdj, maxdeep, maxtAdj int64
	}{
		{2, 2000, 1000, 4000, 2000, 2000, 1000},
		{2, 1001, 1000, 2501, 1000, 2000, 1000}, // odd budgets round down
		{2, 0, 0, 101000, 100000, 20200, 20200}, // the C defaults
		{1, 0, 5, 101000, 100000, 1005, 5},
		// the 1 GiB memory cap: 2^30 / (12 n + 20)
		{10, 1 << 30, 1000, (1 << 30) / 140, 2 * ((1<<30)/140 - 1000) / 3, 2000, 1000},
		{1, 100, 1<<31 - 1, 1150, 100, ((1 << 30) - 24) / 20, ((1<<30)-24)/20 - 1000},
	} {
		a, b, cc, d := dirSizes(c.n, c.maxf, c.maxt)
		if a != c.maxfunc || b != c.maxfAdj || cc != c.maxdeep || d != c.maxtAdj {
			t.Errorf("dirSizes(%d, %d, %d) = %d %d %d %d, want %d %d %d %d",
				c.n, c.maxf, c.maxt, a, b, cc, d, c.maxfunc, c.maxfAdj, c.maxdeep, c.maxtAdj)
		}
	}
}

func TestDirectOddBudgetAndOverrun(t *testing.T) {
	// The budget is checked once per iteration, so Nfev can exceed MaxFev.
	r, err := Direct(dgShifted(5), dgBox(5, -5, 5), &DirectOptions{MaxFev: 30})
	if err != nil {
		t.Fatal(err)
	}
	if r.Status != 1 || r.Nfev <= 30 {
		t.Fatalf("%+v", r)
	}
}

func TestDirectResultIsFresh(t *testing.T) {
	var kept []float64
	o := &DirectOptions{MaxIter: 4, Callback: func(xl, xb []float64, fb float64) bool {
		kept = xb
		xl[0], xb[0] = math.NaN(), math.NaN() // callbacks get copies
		return false
	}}
	r, err := Direct(dgShifted(2), dgBox(2, -5, 5), o)
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(r.X[0]) || math.IsNaN(r.Fun) {
		t.Fatalf("callback copies leaked into the result: %+v", r)
	}
	kept[1] = 99
	if r.X[1] == 99 {
		t.Fatal("result aliases the callback's slice")
	}
}

func TestDirectCallbackStop(t *testing.T) {
	calls := 0
	o := &DirectOptions{Callback: func(xl, xb []float64, fb float64) bool {
		calls++
		if math.Abs(fb-dgSh2(xb)) > 1e-12 {
			t.Errorf("callback fBest %v is not f(xBest) %v", fb, dgSh2(xb))
		}
		return calls == 3
	}}
	r, err := Direct(dgShifted(2), dgBox(2, -5, 5), o)
	if err != nil {
		t.Fatal(err)
	}
	if r.Status != -102 || r.Success || r.Message != "Forced stop" || r.Nit != 4 || calls != 3 {
		t.Fatalf("%+v after %d callbacks", r, calls)
	}
}

func TestDirectWorkersAreDeterministic(t *testing.T) {
	for _, name := range []string{"shifted_sphere_5", "rosenbrock_5", "styblinski_tang_4", "hartmann_6", "nan_region"} {
		f := dgProblems[name]
		n := 5
		lo, hi := -2.0, 2.0
		switch name {
		case "styblinski_tang_4":
			n = 4
		case "hartmann_6":
			n, lo, hi = 6, 0, 1
		case "nan_region":
			n = 2
		}
		for _, orig := range []bool{false, true} {
			var first *DirectResult
			for _, w := range []int{0, 1, 2, 7} {
				r, err := Direct(f, dgBox(n, lo, hi), &DirectOptions{Original: orig, Workers: w, MaxFev: 3000})
				if err != nil {
					t.Fatal(err)
				}
				if first == nil {
					first = r
				} else if !deepeq.DeepEqual(first, r) && !(math.IsNaN(r.Fun) && math.IsNaN(first.Fun)) {
					t.Fatalf("%s original=%v: Workers=%d differs from serial:\n%+v\n%+v", name, orig, w, r, first)
				}
			}
		}
	}
}

func TestDirectWorkersPanicIsReRaised(t *testing.T) {
	boom := func(x []float64) float64 {
		if x[0] > 0.8 {
			panic("boom")
		}
		return x[0]
	}
	defer func() {
		if r := recover(); r != "boom" {
			t.Fatalf("recovered %v", r)
		}
	}()
	Direct(boom, dgBox(3, 0, 1), &DirectOptions{Workers: 4}) //nolint:errcheck
	t.Fatal("no panic")
}

func TestDirectLevelLimits(t *testing.T) {
	lin := func(x []float64) float64 { return x[0] }
	// The deepest level is MaxIter+1000. Jones' original DIRECT numbers its
	// levels k*n+p, so with n = 30 it gets there in a few dozen iterations;
	// DIRECT-L in one dimension instead fills the list S (5000 levels)
	// first, after ~5000 iterations of a monotone function.
	for _, c := range []struct {
		name    string
		n       int
		orig    bool
		maxiter int
	}{{"maxdeep", 30, true, 100}, {"S overflow", 1, false, 300}} {
		if c.name == "S overflow" {
			old := dirMaxDiv
			dirMaxDiv = 25
			defer func() { dirMaxDiv = old }()
		}
		r, err := Direct(lin, dgBox(c.n, 0, 1), &DirectOptions{MaxIter: c.maxiter, Original: c.orig, MaxFev: 1 << 24, VolTol: -1, LenTol: -1})
		if err != nil {
			t.Fatal(err)
		}
		if r.Status != -6 || r.Success || r.Message != "Maximum number of levels has been reached." ||
			math.IsNaN(r.X[0]) || r.X[0] < 0 || r.X[0] > 1 {
			t.Errorf("%s: %+v", c.name, r)
		}
		t.Logf("%s: nit %d nfev %d", c.name, r.Nit, r.Nfev)
	}
}

func TestDirectDeepLevelsAndTies(t *testing.T) {
	// Past level ~650 1/3^k is exactly 0 and the tables are not read any
	// more; the original DIRECT's level tables are computed on demand.
	d := &dirState{n: 2, jones: 0}
	d.initTables(5000)
	if d.lev(2*800) != 0 || d.lev(2*3000) != 0 || d.lev(3) == 0 || tableAt(d.thirds, 99999) != 0 {
		t.Fatal("level tables")
	}
	d.anchor = make([]int32, 2)
	d.setAnc(50, 7)
	d.setAnc(51, 8)
	if d.anc(50) != 7 || d.anc(51) != 8 || d.anc(1000) != 0 {
		t.Fatal("anchor growth")
	}
	// Ties: a constant function makes every rectangle equal; DIRECT-L takes
	// the first of each level and the search just runs to its limit.
	zero := func([]float64) float64 { return 0 }
	r, err := Direct(zero, dgBox(2, 0, 1), &DirectOptions{MaxIter: 12})
	if err != nil || r.Status != 2 || r.Fun != 0 {
		t.Fatalf("%+v, %v", r, err)
	}
	// The deepest level of the original DIRECT with many dimensions.
	// Jones' levels are k*n+p, so with many dimensions the divided centre
	// already sits at level n; maxiter=3 gives MAXDEEP = 1003, so the
	// second iteration selects the centre rectangle (level 1100) and stops
	// with "maximum number of levels".
	sq := func(x []float64) (float64, int) { return float64(x[0] * x[0]), 0 }
	out := directCore(sq, []float64{-1}, []float64{3}, dirParams{eps: 1e-4, maxf: 100000, maxt: 500,
		fglobal: math.Inf(-1), volper: -1, sigmaper: -1, deepCap: 40}, nil)
	if out.ierror != -6 || out.numiter < 30 {
		t.Fatalf("status %d nit %d nfev %d", out.ierror, out.numiter, out.numfunc)
	}
}

func TestDirectDoubleInsertOverflow(t *testing.T) {
	// Every hyperrectangle of a level has the same value of a constant
	// function; Jones' original DIRECT divides all of them, and the list S
	// overflows. SciPy's C code would write out of bounds; Go reports -6.
	zero := func([]float64) float64 { return 0 }
	old := dirMaxDiv
	dirMaxDiv = 30
	defer func() { dirMaxDiv = old }()
	r, err := Direct(zero, dgBox(2, 0, 1), &DirectOptions{Original: true, MaxFev: 1 << 22, MaxIter: 5000, VolTol: -1, LenTol: -1})
	if err != nil {
		t.Fatal(err)
	}
	if r.Status != -6 || r.Fun != 0 {
		t.Fatalf("%+v", r)
	}
	for _, x := range r.X {
		if math.IsNaN(x) || x < 0 || x > 1 {
			t.Fatalf("x outside the box: %v", r.X)
		}
	}
}

func TestDirectSamplePointFailures(t *testing.T) {
	// The sample points of the first division need 2n+1 slots.
	r, err := Direct(dgShifted(1000), dgBox(1000, 0, 1), &DirectOptions{MaxFev: 1})
	if err != nil {
		t.Fatal(err)
	}
	if r.Status != -4 || r.Message != "There was an error in the creation of the sample points" {
		t.Fatalf("init: %+v", r)
	}
	// A later division can run out of slots too: one iteration may take
	// more slots than the budget left.
	r, err = Direct(dgShifted(300), dgBox(300, 0, 1), &DirectOptions{MaxFev: 10})
	if err != nil {
		t.Fatal(err)
	}
	if r.Status != -4 {
		t.Fatalf("main loop: %+v", r)
	}
	if len(r.X) != 300 || math.IsNaN(r.X[0]) {
		t.Fatalf("x is the last evaluated point: %v", r.X[:3])
	}
}

func TestDirectFMinStop(t *testing.T) {
	fm := 0.0
	r, err := Direct(dgShifted(2), dgBox(2, -5, 5), &DirectOptions{FMin: &fm, FMinRtol: 1e-3})
	if err != nil {
		t.Fatal(err)
	}
	if r.Status != 3 || !r.Success || r.Fun > 1e-3 {
		t.Fatalf("%+v", r)
	}
	// A negative FMinRtol disables the test; a known minimum of -Inf is "unknown".
	r2, _ := Direct(dgShifted(2), dgBox(2, -5, 5), &DirectOptions{FMin: &fm, FMinRtol: -1, MaxIter: 20})
	if r2.Status != 2 {
		t.Fatalf("%+v", r2)
	}
	ninf := math.Inf(-1)
	r3, _ := Direct(dgShifted(2), dgBox(2, -5, 5), &DirectOptions{FMin: &ninf, MaxIter: 20})
	if r3.Status != 2 {
		t.Fatalf("%+v", r3)
	}
}

func ExampleDirect() {
	r, _ := Direct(dgStyblinskiTangDoc, dgBox(2, -4, 4), &DirectOptions{LenTol: 1e-3})
	fmt.Printf("%.4f %.4f %d %d %v\n", r.X[0], r.X[1], r.Nfev, r.Nit, r.Success)
	// Output: -2.9044 -2.9044 207 19 true
}
