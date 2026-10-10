package optimize

import (
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strconv"
	"testing"
)

//go:embed testdata/scalar.json
var scalarJSON []byte

// The functions of testdata/gen/scalar.py, written without fused
// multiply-adds so that arithmetic matches NumPy's.
var scalarFns = map[string]func(float64) float64{
	"quad":    func(x float64) float64 { return float64((x-2)*(x-2)) + 1 },
	"quartic": func(x float64) float64 { return float64(math.Pow(x, 4)-float64(3*math.Pow(x, 3))) + 2 },
	"cosx":    math.Cos,
	"abs":     func(x float64) float64 { return math.Abs(x - 0.7) },
	"bessel_like": func(x float64) float64 {
		return float64(math.Exp(-x) * math.Sin(3*x))
	},
	"linear": func(x float64) float64 { return -x },
}

// jfloat reads a JSON number or a string such as "inf" (JSON has no
// infinities, so the generators write them as strings).
type jfloat float64

func (f *jfloat) UnmarshalJSON(b []byte) error {
	var s string
	if json.Unmarshal(b, &s) == nil {
		v, err := strconv.ParseFloat(s, 64)
		*f = jfloat(v)
		return err
	}
	var v float64
	err := json.Unmarshal(b, &v)
	*f = jfloat(v)
	return err
}

type scalarCase struct {
	Fn      string    `json:"fn"`
	Method  string    `json:"method"`
	X       jfloat    `json:"x"`
	Fun     jfloat    `json:"fun"`
	Nit     int       `json:"nit"`
	Nfev    int       `json:"nfev"`
	Success bool      `json:"success"`
	Message string    `json:"message"`
	Bracket []float64 `json:"bracket"`
	Bounds  []float64 `json:"bounds"`
	MaxIter int       `json:"maxiter"`
	Tol     float64   `json:"tol"`
}

type bracketCase struct {
	Fn   string    `json:"fn"`
	X    []float64 `json:"x"`
	F    []float64 `json:"f"`
	Nfev int       `json:"nfev"`
	OK   bool      `json:"ok"`
	Msg  string    `json:"msg"`
}

// TestMinimizeScalarAgainstSciPy replays minimize_scalar. Brent, golden and
// bounded are statement-for-statement ports: on functions of plain
// arithmetic the iterates must be identical (same x, nit, nfev and message);
// on libm functions (cos, exp, sin) the last ulp may differ, and the minimiser
// must agree to the method's tolerance.
func TestMinimizeScalarAgainstSciPy(t *testing.T) {
	var d struct {
		Cases   []scalarCase  `json:"cases"`
		Bracket []bracketCase `json:"bracket"`
	}
	if err := json.Unmarshal(scalarJSON, &d); err != nil {
		t.Fatal(err)
	}
	arith := map[string]bool{"quad": true, "abs": true, "linear": true}
	for _, c := range d.Cases {
		name := fmt.Sprintf("%s/%s/%v%v/%d", c.Fn, c.Method, c.Bracket, c.Bounds, c.MaxIter)
		t.Run(name, func(t *testing.T) {
			o := &ScalarOptions{Bracket: c.Bracket, MaxIter: c.MaxIter, Tol: c.Tol}
			if c.Bounds != nil {
				o.Bounds = &Bound{c.Bounds[0], c.Bounds[1]}
			}
			r, err := MinimizeScalar(scalarFns[c.Fn], ScalarMethod(c.Method), o)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("COUNTS %-40s nit %3d/%-3d nfev %3d/%-3d x %.17g/%.17g", name, r.Nit, c.Nit, r.Nfev, c.Nfev, r.X, c.X)
			if r.Success != c.Success || r.Message != c.Message {
				t.Errorf("success %v %q, SciPy %v %q", r.Success, r.Message, c.Success, c.Message)
			}
			if arith[c.Fn] {
				if r.X != float64(c.X) || r.Nit != c.Nit || r.Nfev != c.Nfev {
					t.Errorf("x/nit/nfev %.17g/%d/%d, SciPy %.17g/%d/%d", r.X, r.Nit, r.Nfev, c.X, c.Nit, c.Nfev)
				}
				return
			}
			if math.Abs(r.X-float64(c.X)) > 1e-4*math.Max(1, math.Abs(float64(c.X))) {
				t.Errorf("x %.17g, SciPy %.17g", r.X, c.X)
			}
		})
	}
	for _, c := range d.Bracket {
		x, fx, n, err := Bracket(scalarFns[c.Fn], 0, 0)
		if !c.OK {
			var be *BracketError
			if !errors.As(err, &be) || be.Msg != c.Msg {
				t.Errorf("%s: bracket error %v, SciPy %q", c.Fn, err, c.Msg)
			}
			continue
		}
		if err != nil || n != c.Nfev {
			t.Errorf("%s: %v nfev %d/%d", c.Fn, err, n, c.Nfev)
		}
		for i := range 3 {
			if math.Abs(x[i]-c.X[i]) > 1e-15*math.Max(1, math.Abs(c.X[i])) || math.Abs(fx[i]-c.F[i]) > 1e-14*math.Max(1, math.Abs(c.F[i])) {
				t.Errorf("%s: bracket %v %v, SciPy %v %v", c.Fn, x, fx, c.X, c.F)
			}
		}
	}
}

func TestMinimizeScalarErrors(t *testing.T) {
	q := scalarFns["quad"]
	for _, tc := range []struct {
		m ScalarMethod
		o *ScalarOptions
	}{
		{Brent, &ScalarOptions{Bounds: &Bound{0, 1}}},
		{Golden, &ScalarOptions{Xtol: -1}},
		{Bounded, nil},
		{Bounded, &ScalarOptions{Bounds: &Bound{0, math.Inf(1)}}},
		{Bounded, &ScalarOptions{Bounds: &Bound{2, 1}}},
		{"nope", nil},
		{Brent, &ScalarOptions{Bracket: []float64{0}}},
		{Brent, &ScalarOptions{Bracket: []float64{0, 3, 1}}},
		{Golden, &ScalarOptions{Bracket: []float64{0, 3, 1}}},
		{Brent, &ScalarOptions{Bracket: []float64{0, 0.5, 1}}},
	} {
		if _, err := MinimizeScalar(q, tc.m, tc.o); err == nil {
			t.Errorf("%s %+v: no error", tc.m, tc.o)
		}
	}
	// Defaults: no method selects Brent, Bounds selects Bounded.
	if r, err := MinimizeScalar(q, "", nil); err != nil || math.Abs(r.X-2) > 1e-7 {
		t.Errorf("default: %v %+v", err, r)
	}
	if r, err := Fminbound(q, -1, 1, 0, 0); err != nil || math.Abs(r.X-1) > 1e-4 {
		t.Errorf("Fminbound: %v %+v", err, r)
	}
	if r, err := MinimizeScalar(q, "", &ScalarOptions{Bounds: &Bound{3, 4}, Tol: 1e-8}); err != nil || math.Abs(r.X-3) > 1e-6 {
		t.Errorf("bounded tol: %v %+v", err, r)
	}
	// NaN objective.
	nan := func(float64) float64 { return math.NaN() }
	if r, _ := MinimizeScalar(nan, Bounded, &ScalarOptions{Bounds: &Bound{0, 1}}); r.Success || r.Status != 2 {
		t.Errorf("bounded NaN: %+v", r)
	}
	if r, _ := MinimizeScalar(nan, Brent, nil); r.Success || !math.IsNaN(r.X) {
		t.Errorf("brent NaN: %+v", r)
	}
	if r, _ := MinimizeScalar(func(x float64) float64 {
		if x > 1.5 {
			return math.NaN()
		}
		return float64((x - 1) * (x - 1))
	}, Golden, &ScalarOptions{Bracket: []float64{0, 1, 1.4}}); !r.Success {
		t.Errorf("golden: %+v", r)
	}
	// Bracket iteration limit and its error text.
	_, _, _, err := bracket(func(x float64) float64 { return -x }, 0, 1, 1.1, 2)
	var be *BracketError
	if !errors.As(err, &be) || be.Error() == "" {
		t.Errorf("bracket limit: %v", err)
	}
	// A bracket that never closes on a NaN function keeps NaN as the answer.
	be = &BracketError{Xa: 0, Xb: math.NaN(), Xc: 1, Fa: 0, Fb: 0, Fc: 0}
	if r := be.result(); !math.IsNaN(r.X) {
		t.Errorf("NaN bracket result: %+v", r)
	}
}

func TestPyRepr(t *testing.T) {
	for v, want := range map[float64]string{
		1.48e-8: "1.48e-08", 0.001: "0.001", 1e-5: "1e-05", 1e16: "1e+16", 123456.0: "123456.0",
		math.NaN(): "nan", math.Inf(1): "inf", math.Inf(-1): "-inf", 0: "0.0", -1.5: "-1.5",
		1.4901161193847656e-08: "1.4901161193847656e-08", 1e15: "1000000000000000.0", 9.999999999999999e-05: "9.999999999999999e-05",
	} {
		if got := pyRepr(v); got != want {
			t.Errorf("pyRepr(%v) = %q, want %q", v, got, want)
		}
	}
	if got := pyRepr(math.Copysign(0, -1)); got != "-0.0" {
		t.Errorf("-0: %q", got)
	}
}
