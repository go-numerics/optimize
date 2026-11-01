package optimize

import (
	"math"
	"testing"
)

// styblinskiTang is the SciPy docstring example of scipy.optimize.direct,
// written with products only (no pow), see docs/design-global.md 7.2.
func styblinskiTang(x []float64) float64 {
	a, b := x[0], x[1]
	return 0.5 * (float64(a*a*a*a) - float64(16*a*a) + float64(5*a) + float64(b*b*b*b) - float64(16*b*b) + float64(5*b))
}

func TestDirectDocstring(t *testing.T) {
	bnd := []Bound{{-4, 4}, {-4, 4}}
	r, err := Direct(styblinskiTang, bnd, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%+v", r)
	if r.Nfev != 2011 || math.Abs(r.Fun-(-78.3323279095383)) > 1e-12 {
		t.Fatalf("got nfev=%d fun=%v", r.Nfev, r.Fun)
	}
	r, _ = Direct(styblinskiTang, bnd, &DirectOptions{LenTol: 1e-3})
	t.Logf("%+v", r)
	if r.Nfev != 207 || math.Abs(r.Fun-(-78.33230330754142)) > 1e-12 {
		t.Fatalf("got nfev=%d fun=%v", r.Nfev, r.Fun)
	}
}
