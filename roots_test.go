package optimize

import (
	_ "embed"
	"encoding/json"
	"errors"
	"math"
	"testing"
)

//go:embed testdata/roots.json
var rootsJSON []byte

type rootFn struct {
	f, fp, fpp func(float64) float64
	br         [2]float64
	x0         float64
}

// The functions of testdata/gen/roots.py.
var rootFns = map[string]rootFn{
	"cubic": {func(x float64) float64 { return float64(x*x*x) - float64(2*x) - 5 }, func(x float64) float64 { return float64(3*x*x) - 2 },
		func(x float64) float64 { return float64(6 * x) }, [2]float64{2, 3}, 2},
	"cosx": {func(x float64) float64 { return math.Cos(x) - x }, func(x float64) float64 { return -math.Sin(x) - 1 },
		func(x float64) float64 { return -math.Cos(x) }, [2]float64{0, 1}, 0.5},
	"triple": {func(x float64) float64 { return math.Pow(x-1, 3) }, func(x float64) float64 { return float64(3 * (x - 1) * (x - 1)) },
		func(x float64) float64 { return float64(6 * (x - 1)) }, [2]float64{0, 2.5}, 2},
	"ninth": {func(x float64) float64 { return math.Pow(x-1, 9) }, func(x float64) float64 { return float64(9 * math.Pow(x-1, 8)) },
		func(x float64) float64 { return float64(72 * math.Pow(x-1, 7)) }, [2]float64{-1, 4}, 3},
	"tiny": {func(x float64) float64 { return float64(1e-12 * (x - 1)) }, func(float64) float64 { return 1e-12 },
		func(float64) float64 { return 0 }, [2]float64{0, 3}, 0},
	"steep": {func(x float64) float64 { return math.Exp(20*x) - 1e8 }, func(x float64) float64 { return float64(20 * math.Exp(20*x)) },
		func(x float64) float64 { return float64(400 * math.Exp(20*x)) }, [2]float64{0, 2}, 1.5},
	"flat": {f: func(x float64) float64 {
		if x == 0 {
			return -1e-3
		}
		return math.Exp(-1/(x*x)) - 1e-3
	}, br: [2]float64{0.01, 2}},
	"sinhalf": {func(x float64) float64 { return math.Sin(x) - x/2 }, func(x float64) float64 { return math.Cos(x) - 0.5 },
		func(x float64) float64 { return -math.Sin(x) }, [2]float64{math.Pi / 2, math.Pi}, 2},
	"apsx": {func(x float64) float64 { return float64(x*math.Exp(-x)) - 0.1 }, func(x float64) float64 { return float64(math.Exp(-x) * (1 - x)) },
		func(x float64) float64 { return float64(math.Exp(-x) * (x - 2)) }, [2]float64{0, 1}, 0},
	"kink": {f: func(x float64) float64 {
		if x < 0.3 {
			return x - 0.3
		}
		return float64(1000 * (x - 0.3))
	}, br: [2]float64{-1, 1}},
	"pole": {f: func(x float64) float64 {
		if x == 0.5 {
			return math.Inf(1)
		}
		return 1 / (x - 0.5)
	}, br: [2]float64{0, 1}},
}

type rootCase struct {
	Fn         string  `json:"fn"`
	Method     string  `json:"method"`
	Root       float64 `json:"root"`
	Iterations int     `json:"iterations"`
	Calls      int     `json:"calls"`
	Converged  bool    `json:"converged"`
	Flag       string  `json:"flag"`
	MaxIter    int     `json:"maxiter"`
}

// TestRootsAgainstSciPy replays SciPy's root_scalar on every function and
// method. The solvers are statement-for-statement ports, so on functions
// built from + − × ÷ alone (exactArith) the root, the iteration count and
// the call count must agree exactly. Functions calling cos, exp or pow differ
// from the platform's C library in the last ulp, which can move an iterate
// across a decision; there the root must still meet the convergence
// tolerance and the counts are only reported. Overall at least 90% of the
// cases must take SciPy's exact path.
func TestRootsAgainstSciPy(t *testing.T) {
	var cs []rootCase
	if err := json.Unmarshal(rootsJSON, &cs); err != nil {
		t.Fatal(err)
	}
	exactArith := map[string]bool{"cubic": true, "tiny": true, "kink": true, "pole": true}
	same := 0
	defer func() {
		t.Logf("%d of %d cases on SciPy's exact path", same, len(cs))
		if same*10 < 9*len(cs) {
			t.Errorf("only %d of %d cases on SciPy's exact path", same, len(cs))
		}
	}()
	for _, c := range cs {
		fn := rootFns[c.Fn]
		t.Run(c.Fn+"/"+c.Method, func(t *testing.T) {
			var r *RootResult
			var err error
			switch c.Method {
			case "newton-secant-default":
				r, err = Newton(fn.f, fn.x0, nil)
			case "auto-x0":
				r, err = RootScalar(fn.f, "", &RootScalarOptions{X0: fn.x0})
			default:
				o := &RootScalarOptions{MaxIter: c.MaxIter}
				switch c.Method {
				case "newton", "halley", "secant":
					o.X0, o.X0Set = fn.x0, true
					o.Fprime = fn.fp
					if c.Method == "halley" {
						o.Fprime2 = fn.fpp
					}
					if c.Method == "secant" {
						o.Fprime, o.X1 = nil, fn.x0+0.1
					}
				default:
					o.Bracket = fn.br[:]
				}
				r, err = RootScalar(fn.f, c.Method, o)
			}
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("COUNTS %-28s iter %3d/%-3d calls %3d/%-3d root %.17g/%.17g",
				c.Fn+"/"+c.Method, r.Iterations, c.Iterations, r.FunctionCalls, c.Calls, r.Root, c.Root)
			if r.Converged != c.Converged || r.Flag != c.Flag {
				t.Errorf("converged %v %q, SciPy %v %q", r.Converged, r.Flag, c.Converged, c.Flag)
			}
			onPath := r.Iterations == c.Iterations && r.FunctionCalls == c.Calls &&
				math.Abs(r.Root-c.Root) <= 4*epsmch*math.Abs(c.Root)
			if onPath {
				same++
			} else if exactArith[c.Fn] {
				t.Errorf("iterations/calls/root %d/%d/%.17g, SciPy %d/%d/%.17g",
					r.Iterations, r.FunctionCalls, r.Root, c.Iterations, c.Calls, c.Root)
			}
			// Bracketing tolerance 4(xtol + rtol|x|); Newton's 1.48e-8 step
			// test can leave the root further off on multiple roots, where
			// the comparison is with SciPy's own distance to the root.
			tol := float64(4 * (2e-12 + float64(4*epsmch*math.Abs(c.Root))))
			if !c.Converged || c.Method == "newton" || c.Method == "halley" || c.Method == "secant" {
				tol = math.Max(tol, float64(2*math.Abs(c.Root-math.Round(c.Root*1e6)/1e6))+1e-8)
			}
			if math.Abs(r.Root-c.Root) > tol {
				t.Errorf("root %.17g, SciPy %.17g", r.Root, c.Root)
			}
		})
	}
}

func TestRootErrors(t *testing.T) {
	sq := func(x float64) float64 { return float64(x*x) + 1 }
	for _, s := range []func(func(float64) float64, float64, float64, *BracketOptions) (*RootResult, error){Brentq, Brenth, Ridder, Bisect} {
		if _, err := s(sq, -1, 1, nil); !errors.Is(err, ErrBracket) {
			t.Errorf("no sign change: %v", err)
		}
		if _, err := s(sq, -1, 1, &BracketOptions{Xtol: -1}); !errors.Is(err, ErrInput) {
			t.Errorf("xtol: %v", err)
		}
		if _, err := s(sq, -1, 1, &BracketOptions{Rtol: 1e-20}); !errors.Is(err, ErrInput) {
			t.Errorf("rtol: %v", err)
		}
		if _, err := s(sq, -1, 1, &BracketOptions{MaxIter: -1}); !errors.Is(err, ErrInput) {
			t.Errorf("maxiter: %v", err)
		}
		// Exact roots at the ends.
		lin := func(x float64) float64 { return x - 1 }
		if r, _ := s(lin, 1, 2, nil); r.Root != 1 || !r.Converged {
			t.Errorf("root at a: %+v", r)
		}
		if r, _ := s(lin, 0, 1, nil); r.Root != 1 || !r.Converged {
			t.Errorf("root at b: %+v", r)
		}
		// A NaN stops the search with an error describing where.
		nan := func(x float64) float64 {
			if x > 0.1 && x < 0.9 {
				return math.NaN()
			}
			return x - 0.5
		}
		r, err := s(nan, 0, 1, nil)
		if err == nil || r == nil || !(r.Root > 0.1 && r.Root < 0.9) {
			t.Errorf("NaN: %v %+v", err, r)
		}
		// Through RootScalar the NaN is a result, as in SciPy.
		r, err = RootScalar(nan, r.Method, &RootScalarOptions{Bracket: []float64{0, 1}})
		if err != nil || r.Converged || r.Iterations != -1 {
			t.Errorf("RootScalar NaN: %v %+v", err, r)
		}
	}
	// Ridder's midpoint landing exactly on the root.
	if r, _ := Ridder(func(x float64) float64 { return x }, -1, 1, nil); r.Root != 0 || !r.Converged {
		t.Errorf("ridder midpoint: %+v", r)
	}
	// Bisection running out of iterations.
	if r, _ := Bisect(func(x float64) float64 { return x - 0.3 }, 0, 1, &BracketOptions{MaxIter: 3}); r.Converged || r.Flag != FlagConvErr {
		t.Errorf("bisect maxiter: %+v", r)
	}
	if r, _ := Ridder(func(x float64) float64 { return float64(x*x*x) - 0.3 }, 0, 1, &BracketOptions{MaxIter: 1}); r.Converged {
		t.Errorf("ridder maxiter: %+v", r)
	}
	// Newton argument checks and failure modes.
	if _, err := Newton(sq, 1, &NewtonOptions{Tol: -1}); !errors.Is(err, ErrInput) {
		t.Error(err)
	}
	if _, err := Newton(sq, 1, &NewtonOptions{MaxIter: -1}); !errors.Is(err, ErrInput) {
		t.Error(err)
	}
	if _, err := Newton(sq, 1, &NewtonOptions{X1: 1}); !errors.Is(err, ErrInput) {
		t.Error(err)
	}
	if r, _ := Newton(sq, 0, &NewtonOptions{Fprime: func(x float64) float64 { return float64(2 * x) }}); r.Converged {
		t.Errorf("zero derivative: %+v", r)
	}
	if r, _ := Newton(func(x float64) float64 { return x }, 0, &NewtonOptions{Fprime: func(float64) float64 { return 1 }}); !r.Converged || r.Root != 0 {
		t.Errorf("f(x0)=0: %+v", r)
	}
	if r, _ := Newton(sq, 1, &NewtonOptions{Fprime: func(x float64) float64 { return float64(2 * x) }, MaxIter: 5}); r.Converged {
		t.Errorf("newton maxiter: %+v", r)
	}
	if r, _ := Newton(sq, 1, &NewtonOptions{MaxIter: 5}); r.Converged {
		t.Errorf("secant maxiter: %+v", r)
	}
	if r, _ := Newton(func(float64) float64 { return 1 }, 1, nil); r.Converged {
		t.Errorf("secant flat: %+v", r)
	}
	if r, _ := Newton(func(x float64) float64 { return x + 1 }, -5, nil); !r.Converged || math.Abs(r.Root+1) > 1e-9 {
		t.Errorf("secant negative start: %+v", r)
	}
	if r, _ := Newton(func(x float64) float64 { return x - 2 }, 1, &NewtonOptions{X1: 0, X1Set: true}); !r.Converged {
		t.Errorf("explicit x1=0: %+v", r)
	}
	// RootScalar selection and argument errors.
	cube := func(x float64) float64 { return float64(x*x*x) - 2 }
	d := func(x float64) float64 { return float64(3 * x * x) }
	d2 := func(x float64) float64 { return float64(6 * x) }
	for _, o := range []*RootScalarOptions{
		{Bracket: []float64{0, 2}}, {X0: 1, Fprime: d}, {X0: 1, Fprime: d, Fprime2: d2}, {X0: 1, X1: 2},
	} {
		if r, err := RootScalar(cube, "", o); err != nil || !r.Converged || math.Abs(r.Root-math.Cbrt(2)) > 1e-9 {
			t.Errorf("auto %+v: %v %+v", o, err, r)
		}
	}
	bad := []struct {
		m string
		o *RootScalarOptions
	}{
		{"", nil}, {"nope", &RootScalarOptions{X0: 1}}, {"brentq", &RootScalarOptions{}},
		{"newton", &RootScalarOptions{}}, {"halley", &RootScalarOptions{X0: 1}},
	}
	for _, b := range bad {
		if _, err := RootScalar(cube, b.m, b.o); err == nil {
			t.Errorf("%q %+v: no error", b.m, b.o)
		}
	}
	for _, m := range []string{"brenth", "ridder", "bisect"} {
		if r, err := RootScalar(cube, m, &RootScalarOptions{Bracket: []float64{0, 2}}); err != nil || !r.Converged {
			t.Errorf("%s: %v", m, err)
		}
	}
}
