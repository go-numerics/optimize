package gofuncs

import (
	"encoding/json"
	"math"
	"os"
	"strconv"
	"testing"
)

type hexf float64

func (h *hexf) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	u, err := strconv.ParseUint(s[2:], 16, 64)
	if err != nil {
		return err
	}
	*h = hexf(math.Float64frombits(u))
	return nil
}

func hexes(h []hexf) []float64 {
	out := make([]float64, len(h))
	for i, v := range h {
		out[i] = float64(v)
	}
	return out
}

type funcFixture struct {
	Scipy   string            `json:"scipy"`
	Sources map[string]string `json:"sources"`
	Funcs   []struct {
		Name   string `json:"name"`
		N      int    `json:"n"`
		Lower  []hexf `json:"lower"`
		Upper  []hexf `json:"upper"`
		FGlob  hexf   `json:"fglob"`
		Points []struct {
			X []hexf `json:"x"`
			F hexf   `json:"f"`
		} `json:"points"`
	} `json:"funcs"`
}

// TestAgainstSciPy checks every function against SciPy 1.18.1's own class
// (testdata/global/funcs.json, tools/globaloracle gen funcs): dimension,
// bounds and fglob exactly, the value at three points to 1e-12 relative.
func TestAgainstSciPy(t *testing.T) {
	b, err := os.ReadFile("../../testdata/global/funcs.json")
	if err != nil {
		t.Fatal(err)
	}
	var fx funcFixture
	if err := json.Unmarshal(b, &fx); err != nil {
		t.Fatal(err)
	}
	if fx.Scipy != "1.18.1" || len(fx.Sources) == 0 {
		t.Fatalf("fixture: SciPy %q, %d sources", fx.Scipy, len(fx.Sources))
	}
	all := All()
	if len(all) != len(fx.Funcs) {
		t.Fatalf("%d functions, fixture %d", len(all), len(fx.Funcs))
	}
	worst := 0.0
	for i, c := range fx.Funcs {
		f := all[i]
		if f.Name != c.Name || f.Dim() != c.N {
			t.Fatalf("function %d: %s/%d, SciPy %s/%d", i, f.Name, f.Dim(), c.Name, c.N)
		}
		for j := range f.Lo {
			if f.Lo[j] != float64(c.Lower[j]) || f.Hi[j] != float64(c.Upper[j]) {
				t.Errorf("%s: bounds [%v, %v], SciPy [%v, %v]", f.Name, f.Lo, f.Hi, hexes(c.Lower), hexes(c.Upper))
				break
			}
		}
		if f.FGlob != float64(c.FGlob) {
			t.Errorf("%s: fglob %v, SciPy %v", f.Name, f.FGlob, float64(c.FGlob))
		}
		for k, p := range c.Points {
			x := hexes(p.X)
			got, want := f.F(x), float64(p.F)
			d := math.Abs(got-want) / math.Max(1, math.Abs(want))
			worst = math.Max(worst, d)
			if d > 1e-12 {
				t.Errorf("%s point %d: %v, SciPy %v (rel %.2g)", f.Name, k, got, want, d)
			}
		}
		// The first listed optimum is a success (SciPy's own constants are
		// rounded for a few functions, hence the 1e-5 of the rule).
		if got := f.F(f.XGlob); !(math.Abs(got-f.FGlob) < 1e-4) {
			t.Errorf("%s: f(xglob) = %v, fglob %v", f.Name, got, f.FGlob)
		}
	}
	t.Logf("%d functions agree with SciPy; worst relative difference %.2g", len(all), worst)
}

func TestSuccessAndLookup(t *testing.T) {
	f, ok := ByName("Rastrigin/5")
	if !ok || f.Dim() != 5 {
		t.Fatalf("ByName: %v %v", f, ok)
	}
	if !f.Success(make([]float64, 5)) {
		t.Error("the origin is a success for Rastrigin")
	}
	if f.Success([]float64{1, 1, 1, 1, 1}) {
		t.Error("(1,...,1) is not a success")
	}
	if f.Success([]float64{0}) {
		t.Error("a point of the wrong length is not a success")
	}
	if _, ok := ByName("nope/2"); ok {
		t.Error("unknown function found")
	}
	names := map[string]bool{}
	for _, g := range All() {
		if names[g.Name] {
			t.Errorf("duplicate %s", g.Name)
		}
		names[g.Name] = true
	}
}
