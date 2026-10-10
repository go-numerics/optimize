package optimize

import (
	_ "embed"
	"encoding/json"
	"math"
	"testing"
)

//go:embed testdata/brute.json
var bruteJSON []byte

func bruteDocs(z []float64) float64 {
	x, y := z[0], z[1]
	a, b, c, d, e, f, g, h, i, j, k, l, scale := 2.0, 3.0, 7.0, 8.0, 9.0, 10.0, 44.0, -1.0, 2.0, 26.0, 1.0, -2.0, 0.5
	f1 := float64(a*x*x) + float64(b*x*y) + float64(c*y*y) + float64(d*x) + float64(e*y) + f
	f2 := -g * math.Exp(-(float64((x-h)*(x-h))+float64((y-i)*(y-i)))/scale)
	f3 := -j * math.Exp(-(float64((x-k)*(x-k))+float64((y-l)*(y-l)))/scale)
	return f1 + f2 + f3
}

func TestBruteAgainstSciPy(t *testing.T) {
	var cs []struct {
		Name   string      `json:"name"`
		Ranges [][]float64 `json:"ranges"`
		Ns     int         `json:"ns"`
		Finish bool        `json:"finish"`
		X      []float64   `json:"x"`
		Fun    float64     `json:"fun"`
		Jmin   float64     `json:"jmin"`
		Jsize  int         `json:"jsize"`
	}
	if err := json.Unmarshal(bruteJSON, &cs); err != nil {
		t.Fatal(err)
	}
	fns := map[string]func([]float64) float64{"docs": bruteDocs, "docs_nofinish": bruteDocs,
		"himmelblau": himmelblau, "rosen3": rosen}
	for _, c := range cs {
		t.Run(c.Name, func(t *testing.T) {
			var rs []Range
			for _, r := range c.Ranges {
				rs = append(rs, Range{r[0], r[1], r[2]})
			}
			r, err := Brute(fns[c.Name], rs, &BruteOptions{Ns: c.Ns, NoFinish: !c.Finish, Workers: 3})
			if err != nil {
				t.Fatal(err)
			}
			if len(r.Jout) != c.Jsize {
				t.Errorf("grid size %d, SciPy %d", len(r.Jout), c.Jsize)
			}
			for i := range c.X {
				if math.Abs(r.X[i]-c.X[i]) > 1e-9*math.Max(1, math.Abs(c.X[i])) {
					t.Errorf("x %v, SciPy %v", r.X, c.X)
					break
				}
			}
			if math.Abs(r.Fun-c.Fun) > 1e-9*math.Max(1, math.Abs(c.Fun)) || !r.Success {
				t.Errorf("fun %.15g, SciPy %.15g", r.Fun, c.Fun)
			}
		})
	}
}

func TestBruteErrors(t *testing.T) {
	f := func(x []float64) float64 { return x[0] * x[0] }
	for i, tc := range []struct {
		r []Range
		o *BruteOptions
	}{
		{nil, nil},
		{make([]Range, 41), nil},
		{[]Range{{0, 1, 0}}, &BruteOptions{Ns: -1}},
		{[]Range{{1, 0, 0.5}}, nil},
		{[]Range{{-1, 1, 0}}, &BruteOptions{Finish: "nope"}},
	} {
		if _, err := Brute(f, tc.r, tc.o); err == nil {
			t.Errorf("case %d: no error", i)
		}
	}
	// One point per axis, and a NaN objective (numpy's argmin picks it).
	if r, err := Brute(f, []Range{{2, 3, 0}}, &BruteOptions{Ns: 1, NoFinish: true}); err != nil || r.X[0] != 2 {
		t.Errorf("Ns=1: %v %+v", err, r)
	}
	nan := func(x []float64) float64 {
		if x[0] > 0.5 {
			return math.NaN()
		}
		return x[0]
	}
	if r, _ := Brute(nan, []Range{{0, 1, 0}}, &BruteOptions{Ns: 5, NoFinish: true}); r.X[0] != 0.75 {
		t.Errorf("NaN argmin: %+v", r)
	}
}
