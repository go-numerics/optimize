package optimize

import (
	_ "embed"
	"encoding/json"
	"errors"
	"math"
	"testing"

	"github.com/go-ndarray/ndarray"
)

//go:embed testdata/curvefit.json
var curvefitJSON []byte

func cfGauss(x float64, a, mu, s float64) float64 {
	d := x - mu
	return a * math.Exp(-float64(d*d)/float64(2*float64(s*s)))
}

func cfExp(x, p []float64) float64 { return float64(p[0]*math.Exp(-float64(p[1]*x[0]))) + p[2] }

func cfExpGrad(x, p, g []float64) {
	e := math.Exp(-p[1] * x[0])
	g[0], g[1], g[2] = e, -p[0]*x[0]*e, 1
}

// The models of testdata/gen/curvefit.py.
var cfModels = map[string]Model{
	"expdecay": {F: cfExp},
	"gauss": {F: func(x, p []float64) float64 { return cfGauss(x[0], p[0], p[1], p[2]) }},
	"gauss2": {F: func(x, p []float64) float64 {
		return cfGauss(x[0], p[0], p[1], p[2]) + cfGauss(x[0], p[3], p[4], p[5])
	}},
	"plane": {F: func(x, p []float64) float64 { return float64(p[0]*x[0]) + float64(p[1]*x[1]) + p[2] }},
}

type cfCase struct {
	Name          string      `json:"name"`
	Model         string      `json:"model"`
	X             any         `json:"x"`
	XShape        []int       `json:"xshape"`
	Y             []float64   `json:"y"`
	P0            []float64   `json:"p0"`
	Popt          []float64   `json:"popt"`
	Pcov          [][]float64 `json:"pcov"`
	Nfev          int         `json:"nfev"`
	Ier           int         `json:"ier"`
	Msg           string      `json:"msg"`
	Sigma         any         `json:"sigma"`
	AbsoluteSigma bool        `json:"absolute_sigma"`
	Method        string      `json:"method"`
	Bounds        [][]float64 `json:"bounds"`
}

func flat(v any) []float64 {
	switch t := v.(type) {
	case float64:
		return []float64{t}
	case []float64:
		return t
	case []any:
		var out []float64
		for _, e := range t {
			out = append(out, flat(e)...)
		}
		return out
	}
	return nil
}

func cfArray(v any) *ndarray.Array {
	d := flat(v)
	if inner, ok := v.([]any); ok && len(inner) > 0 {
		if _, two := inner[0].([]any); two {
			a, _ := ndarray.FromData(d, len(inner), len(d)/len(inner))
			return a
		}
	}
	a, _ := ndarray.FromData(d, len(d))
	return a
}

// TestCurveFitAgainstSciPy fits every fixture and compares popt and pcov
// with curve_fit's. LM runs the same MINPACK iterations as SciPy; TRF the
// same trust-region path; differences come only from exp() in the last ulp.
func TestCurveFitAgainstSciPy(t *testing.T) {
	var cs []cfCase
	if err := json.Unmarshal(curvefitJSON, &cs); err != nil {
		t.Fatal(err)
	}
	for _, c := range cs {
		t.Run(c.Name, func(t *testing.T) {
			o := &CurveFitOptions{AbsoluteSigma: c.AbsoluteSigma, Method: LSMethod(c.Method)}
			if c.Sigma != nil {
				o.Sigma = cfArray(c.Sigma)
			}
			for _, b := range c.Bounds {
				o.Bounds = append(o.Bounds, Bound{b[0], b[1]})
			}
			r, err := CurveFitModel(cfModels[c.Model], cfArray(c.X), cfArray(c.Y), c.P0, o)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("COUNTS %-14s nfev %3d/%-3d status %d/%d", c.Name, r.Nfev, c.Nfev, r.Status, c.Ier)
			n := len(c.Popt)
			for i := range n {
				if math.Abs(r.Popt[i]-c.Popt[i]) > 1e-6*math.Max(1, math.Abs(c.Popt[i])) {
					t.Errorf("popt[%d] = %.12g, SciPy %.12g", i, r.Popt[i], c.Popt[i])
				}
				for j := range n {
					want := c.Pcov[i][j]
					got := r.Pcov[i*n+j]
					if want == 1e300 {
						if !math.IsInf(got, 1) || r.CovOK {
							t.Errorf("pcov[%d][%d] = %g, SciPy inf", i, j, got)
						}
						continue
					}
					scale := math.Sqrt(math.Abs(c.Pcov[i][i] * c.Pcov[j][j]))
					if math.Abs(got-want) > 1e-5*scale {
						t.Errorf("pcov[%d][%d] = %.10g, SciPy %.10g", i, j, got, want)
					}
				}
			}
		})
	}
}

func TestCurveFitErrors(t *testing.T) {
	x, _ := ndarray.FromData([]float64{0, 1, 2, 3}, 4)
	y, _ := ndarray.FromData([]float64{1, 3, 5, 7}, 4)
	lin := func(x float64, p []float64) float64 { return p[0]*x + p[1] }
	r, err := CurveFit(lin, x, y, []float64{0, 0}, nil)
	if err != nil || math.Abs(r.Popt[0]-2) > 1e-9 || math.Abs(r.Popt[1]-1) > 1e-9 {
		t.Fatalf("line: %v %+v", err, r)
	}
	if r.PoptArray().Size() != 2 || r.PcovArray().Size() != 4 || len(r.Perr()) != 2 {
		t.Error("accessors")
	}
	// Bounds alone give p0.
	if r, err := CurveFit(lin, x, y, nil, &CurveFitOptions{Bounds: []Bound{{0, 4}, {-1, math.Inf(1)}}}); err != nil || math.Abs(r.Popt[0]-2) > 1e-6 {
		t.Errorf("bounded default p0: %v %+v", err, r)
	}
	if got := initializeFeasible([]float64{math.Inf(-1), 0, math.Inf(-1), 1}, []float64{math.Inf(1), math.Inf(1), 3, 3}); got[0] != 1 || got[1] != 1 || got[2] != 2 || got[3] != 2 {
		t.Errorf("initializeFeasible %v", got)
	}
	bad, _ := ndarray.FromData([]float64{1, math.NaN(), 5, 7}, 4)
	empty, _ := ndarray.FromData([]float64{}, 0)
	x3, _ := ndarray.Zeros(2, 2, 1)
	short, _ := ndarray.FromData([]float64{1, 2}, 2)
	sig3, _ := ndarray.FromData([]float64{1, 2, 3}, 3)
	notPD, _ := ndarray.FromData([]float64{1, 2, 0, 0, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, 4, 4)
	for i, tc := range []struct {
		x, y *ndarray.Array
		p0   []float64
		o    *CurveFitOptions
	}{
		{nil, y, []float64{0, 0}, nil},
		{x, empty, []float64{0, 0}, nil},
		{x3, y, []float64{0, 0}, nil},
		{x, short, []float64{0, 0}, nil},
		{x, bad, []float64{0, 0}, nil},
		{x, y, nil, nil},
		{x, y, []float64{0, 0}, &CurveFitOptions{Bounds: []Bound{{1, 0}, {0, 1}}}},
		{x, y, []float64{0, 0}, &CurveFitOptions{Bounds: []Bound{{0, 3}, {0, 3}}, Method: LM}},
		{x, y, []float64{0, 0}, &CurveFitOptions{Sigma: sig3}},
		{x, y, []float64{0, 0}, &CurveFitOptions{Sigma: notPD}},
		{x, y, []float64{0, 0, 0, 0, 0}, nil},
		{x, y, []float64{0, 0}, &CurveFitOptions{Method: TRF, Loss: "bogus"}},
	} {
		if _, err := CurveFit(lin, tc.x, tc.y, tc.p0, tc.o); err == nil {
			t.Errorf("case %d: no error", i)
		}
	}
	// Non-convergence through both solvers.
	hard := func(x float64, p []float64) float64 { return math.Exp(p[0]*x) + p[1] }
	for _, m := range []LSMethod{LM, TRF} {
		if _, err := CurveFit(hard, x, y, []float64{5, 0}, &CurveFitOptions{Method: m, MaxFev: 2}); !errors.Is(err, ErrNotConverged) {
			t.Errorf("%s: %v", m, err)
		}
	}
	// As many parameters as data points: covariance undetermined.
	two, _ := ndarray.FromData([]float64{0, 1}, 2)
	if r, err := CurveFit(lin, two, short, []float64{0, 0}, &CurveFitOptions{Method: TRF}); err != nil || r.CovOK || !math.IsInf(r.Pcov[0], 1) {
		t.Errorf("m == n: %v %+v", err, r)
	}
	// An analytic gradient gives the same fit as finite differences, with
	// either solver and with a covariance weighting.
	xe, _ := ndarray.Linspace(0, 4, 30)
	ye := make([]float64, 30)
	for i := range ye {
		ye[i] = cfExp([]float64{xe.At(i)}, []float64{2, 1.5, 0.3}) + 0.01*math.Sin(float64(7*i))
	}
	yea, _ := ndarray.FromData(ye, 30)
	cov, _ := ndarray.Eye(30, 30, 0)
	for _, o := range []*CurveFitOptions{{}, {Method: TRF}, {Sigma: cov}} {
		a, err1 := CurveFitModel(Model{F: cfExp}, xe, yea, []float64{1, 1, 0}, o)
		b, err2 := CurveFitModel(Model{F: cfExp, Grad: cfExpGrad}, xe, yea, []float64{1, 1, 0}, o)
		if err1 != nil || err2 != nil {
			t.Fatal(err1, err2)
		}
		for i := range a.Popt {
			if math.Abs(a.Popt[i]-b.Popt[i]) > 1e-6 || math.Abs(a.Pcov[i*3+i]-b.Pcov[i*3+i]) > 1e-6*a.Pcov[i*3+i] {
				t.Errorf("%+v: analytic %v %v, numeric %v %v", o, b.Popt, b.Pcov, a.Popt, a.Pcov)
			}
		}
	}
	// Scalar sigma and absolute sigma.
	s1, _ := ndarray.FromData([]float64{0.5}, 1)
	if r, err := CurveFit(lin, x, y, []float64{0, 0}, &CurveFitOptions{Sigma: s1, AbsoluteSigma: true, Method: TRF}); err != nil || !r.CovOK {
		t.Errorf("scalar sigma: %v %+v", err, r)
	}
}
