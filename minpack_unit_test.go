package optimize

import (
	"errors"
	"math"
	"testing"
)

func TestMpEnorm(t *testing.T) {
	for _, c := range []struct {
		x    []float64
		want float64
	}{
		{nil, 0},
		{[]float64{0, 0}, 0},
		{[]float64{3, 4}, 5},
		{[]float64{3e30, 4e30}, 5e30},        // giant, increasing
		{[]float64{4e30, 3e30}, 5e30},        // giant, decreasing
		{[]float64{3e-30, 4e-30}, 5e-30},     // dwarf, increasing
		{[]float64{4e-30, 0, 3e-30}, 5e-30},  // dwarf, decreasing, a zero
		{[]float64{1e30, 1, 1e-30}, 1e30},    // all three ranges
		{[]float64{1, 1e-30}, 1},             // s2 >= x3max
		{[]float64{1e-19, 3e-20}, 1.044e-19}, // s2 < x3max
		{[]float64{1e300, 1e300}, 1e300 * math.Sqrt2},
	} {
		got := mpEnorm(c.x)
		if math.Abs(got-c.want) > 1e-3*c.want {
			t.Errorf("mpEnorm(%v) = %v, want %v", c.x, got, c.want)
		}
	}
}

// qrsolv with a zero diagonal entry in D and a singular R returns the
// least-squares solution with the singular component set to zero.
func TestQrsolvSingular(t *testing.T) {
	r := []float64{2, 0, 0, 0} // column-major [[2, 0], [0, 0]]
	x := make([]float64, 2)
	sdiag := make([]float64, 2)
	wa := make([]float64, 2)
	qrsolv(2, r, 2, []int{0, 1}, []float64{1, 0}, []float64{1, 5}, x, sdiag, wa)
	if math.Abs(x[0]-0.4) > 1e-15 || x[1] != 0 || sdiag[1] != 0 {
		t.Errorf("x %v sdiag %v, want [0.4 0] with sdiag[1] = 0", x, sdiag)
	}
}

// lmpar falls back to paru = dwarf/min(delta, 0.1) when the scaled gradient
// underflows to zero although the Gauss–Newton step does not.
func TestLmparGradientUnderflow(t *testing.T) {
	r := []float64{1e-200}
	x := make([]float64, 1)
	w := make([]float64, 1)
	par := lmpar(1, r, 1, []int{0}, []float64{1}, []float64{1e-200}, 0.5, 0, x, w, make([]float64, 1), make([]float64, 1))
	if !(par > 0) || math.IsInf(par, 0) || math.IsNaN(x[0]) {
		t.Errorf("par %v x %v", par, x)
	}
}

// r1updt computes Q with (S + u·vᵀ)·Q = S' lower trapezoidal, and r1mpyq
// applies that Q; check the identity, including zero and degenerate
// rotations, and the singularity flag.
func TestR1updt(t *testing.T) {
	for _, c := range []struct {
		m, n    int
		s, u, v []float64 // s packed by columns, lower trapezoidal
		sing    bool
	}{
		{3, 3, []float64{2, 1, 0.5, 3, -1, 4}, []float64{0.3, -0.2, 0.7}, []float64{1, -2, 0.5}, false},
		{4, 3, []float64{2, 1, 0.5, 1, 3, -1, 2, 4, 1}, []float64{0.3, -0.2, 0.7, 1}, []float64{0, 2, 0.5}, false},
		{2, 2, []float64{1, 2, 3}, []float64{0, 0}, []float64{1, 0}, false}, // v[n-1] = 0: cos = 0
		{2, 2, []float64{0, 0, 1}, []float64{0, 0}, []float64{0, 0}, true},  // zero diagonal kept
		{2, 2, []float64{1, 0, 1}, []float64{-1, 0}, []float64{1, 0}, true}, // S + uvᵀ singular
	} {
		m, n := c.m, c.n
		// The full m×n matrix A = S + u·vᵀ, column-major.
		a := make([]float64, m*n)
		l := 0
		for j := 0; j < n; j++ {
			for i := j; i < m; i++ {
				a[i+m*j] = c.s[l]
				l++
			}
		}
		for j := 0; j < n; j++ {
			for i := 0; i < m; i++ {
				a[i+m*j] += c.u[i] * c.v[j]
			}
		}
		s := clone(c.s)
		v := clone(c.v)
		w := make([]float64, m)
		sing := r1updt(m, n, s, c.u, v, w)
		r1mpyq(m, n, a, m, v, w)
		l = 0
		for j := 0; j < n; j++ {
			for i := 0; i < m; i++ {
				want := 0.0
				if i >= j {
					want = s[l]
					l++
				}
				if math.Abs(a[i+m*j]-want) > 1e-12 {
					t.Errorf("m=%d n=%d: (S+uvᵀ)Q[%d,%d] = %v, S'[%d,%d] = %v", m, n, i, j, a[i+m*j], i, j, want)
				}
			}
		}
		if sing != c.sing {
			t.Errorf("m=%d n=%d s=%v: sing %v, want %v", m, n, c.s, sing, c.sing)
		}
	}
}

func TestMinpackErrors(t *testing.T) {
	sq := func(x, f []float64) { f[0] = x[0] - 1; f[1] = x[1] + 1 }
	sys := SystemProblem{Func: sq}
	ls := LSProblem{Func: sq, M: 2}
	x0 := []float64{0, 0}
	for name, err := range map[string]error{
		"root nil":         mpErr(Root(SystemProblem{}, x0, nil)),
		"root empty":       mpErr(Root(sys, nil, nil)),
		"root maxfev":      mpErr(Root(sys, x0, &RootOptions{Maxfev: -1})),
		"root factor":      mpErr(Root(sys, x0, &RootOptions{Factor: -1})),
		"root diag len":    mpErr(Root(sys, x0, &RootOptions{Diag: []float64{1}})),
		"root diag sign":   mpErr(Fsolve(sys, x0, &RootOptions{Diag: []float64{1, 0}})),
		"leastsq nil":      mpErr(Leastsq(LSProblem{M: 2}, x0, nil)),
		"leastsq n > m":    mpErr(Leastsq(LSProblem{Func: sq, M: 1}, x0, nil)),
		"leastsq diag":     mpErr(Leastsq(ls, x0, &LeastsqOptions{Diag: []float64{-1, 1}})),
		"lm tolerance":     mpErr(LeastSquares(ls, x0, LM, &LSOptions{Ftol: 1e-20})),
		"lm max_nfev":      mpErr(LeastSquares(ls, x0, LM, &LSOptions{MaxNfev: -3})),
		"lm disabled gtol": mpErr(LeastSquares(ls, x0, LM, &LSOptions{Gtol: -1})),
	} {
		if err == nil {
			t.Errorf("%s: no error", name)
		} else if !errors.Is(err, ErrInput) && !errors.Is(err, ErrDimension) {
			t.Errorf("%s: %v", name, err)
		}
	}
}

func mpErr[T any](_ T, err error) error { return err }

func TestMinpackAccessors(t *testing.T) {
	p := mpBard()
	r, err := Leastsq(LSProblem{Func: p.f, M: p.m}, p.x0, nil)
	if err != nil || !r.Success() {
		t.Fatal(err, r.Mesg)
	}
	if a := r.XArray(); a.Size() != 3 {
		t.Error("XArray")
	}
	if a := r.CovXArray(); a == nil || a.Shape()[0] != 3 || a.Shape()[1] != 3 {
		t.Error("CovXArray")
	}
	if a := r.FjacArray(); a.Shape()[0] != 3 || a.Shape()[1] != 15 {
		t.Error("FjacArray")
	}
	// A run stopped by Maxfev has no covariance.
	r, _ = Leastsq(LSProblem{Func: p.f, M: p.m}, p.x0, &LeastsqOptions{Maxfev: 3})
	if r.Success() || r.CovX != nil || r.CovXArray() != nil {
		t.Errorf("ier %d cov %v", r.Ier, r.CovX)
	}

	q := mpRosenbrock()
	s, err := Root(SystemProblem{Func: q.f, Jac: q.j}, q.x0, nil)
	if err != nil || !s.Success {
		t.Fatal(err, s.Message)
	}
	if s.XArray().Size() != 2 || s.FjacArray().Shape()[1] != 2 {
		t.Error("RootResult accessors")
	}
	// x_scale='jac' with an explicit XScale still uses MINPACK's scaling.
	ls, err := LeastSquares(LSProblem{Func: p.f, M: p.m}, p.x0, LM, &LSOptions{XScale: []float64{1, 2, 3}, XScaleJac: true})
	if err != nil || !ls.Success || ls.Njev != 0 {
		t.Fatal(err, ls)
	}
	ref, _ := LeastSquares(LSProblem{Func: p.f, M: p.m}, p.x0, LM, nil)
	if relDiff(ls.X, ref.X) != 0 || ls.Nfev != ref.Nfev {
		t.Errorf("XScaleJac %v (%d), default %v (%d)", ls.X, ls.Nfev, ref.X, ref.Nfev)
	}
}

// A singular R (here from an exactly rank-deficient Jacobian at the
// solution) gives no covariance, as SciPy's trtri failure does.
func TestLeastsqSingularCov(t *testing.T) {
	f := func(x, out []float64) { out[0] = x[0] + x[1] - 2; out[1] = 2*x[0] + 2*x[1] - 4 }
	j := func(x, out []float64) { copy(out, []float64{1, 1, 2, 2}) }
	r, err := Leastsq(LSProblem{Func: f, Jac: j, M: 2}, []float64{0, 0}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !r.Success() || r.CovX != nil {
		t.Errorf("ier %d %q cov %v", r.Ier, r.Mesg, r.CovX)
	}
}
