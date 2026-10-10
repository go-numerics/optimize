package optimize

import "math"

// Standard test functions (Moré, Garbow & Hillstrom 1981; Rosenbrock 1960),
// identical to the definitions in testdata/gen/minimize.py.

type testFunc struct {
	f    func(x []float64) float64
	grad func(x, g []float64)
}

func rosen(x []float64) float64 {
	s := 0.0
	for i := 0; i+1 < len(x); i++ {
		// float64() conversions forbid fused multiply-adds, so that the
		// rounding matches NumPy's and finite differences agree bit for bit.
		a := x[i+1] - float64(x[i]*x[i])
		b := 1 - x[i]
		s += float64(100*float64(a*a)) + float64(b*b)
	}
	return s
}

func rosenDer(x, g []float64) {
	n := len(x)
	for i := range g {
		g[i] = 0
	}
	for i := 0; i+1 < n; i++ {
		a := x[i+1] - x[i]*x[i]
		g[i] += -400*x[i]*a - 2*(1-x[i])
		g[i+1] += 200 * a
	}
}

func rosenHess(x, h []float64) {
	n := len(x)
	for i := range h {
		h[i] = 0
	}
	for i := 0; i+1 < n; i++ {
		h[i*n+i] += 1200*x[i]*x[i] - 400*x[i+1] + 2
		h[i*n+i+1] += -400 * x[i]
		h[(i+1)*n+i] += -400 * x[i]
		h[(i+1)*n+i+1] += 200
	}
}

func rosenHessP(x, p, hp []float64) {
	n := len(x)
	h := make([]float64, n*n)
	rosenHess(x, h)
	for i := range n {
		hp[i] = dot(h[i*n:i*n+n], p)
	}
}

func beale(x []float64) float64 {
	a, b := x[0], x[1]
	t1 := 1.5 - a + a*b
	t2 := 2.25 - a + a*b*b
	t3 := 2.625 - a + a*b*b*b
	return t1*t1 + t2*t2 + t3*t3
}

func bealeDer(x, g []float64) {
	a, b := x[0], x[1]
	t1 := 1.5 - a + a*b
	t2 := 2.25 - a + a*b*b
	t3 := 2.625 - a + a*b*b*b
	g[0] = 2*t1*(b-1) + 2*t2*(b*b-1) + 2*t3*(b*b*b-1)
	g[1] = 2*t1*a + 2*t2*2*a*b + 2*t3*3*a*b*b
}

func himmelblau(x []float64) float64 {
	a, b := x[0], x[1]
	u := a*a + b - 11
	v := a + b*b - 7
	return u*u + v*v
}

func himmelblauDer(x, g []float64) {
	a, b := x[0], x[1]
	u := a*a + b - 11
	v := a + b*b - 7
	g[0] = 4*a*u + 2*v
	g[1] = 2*u + 4*b*v
}

func powellSing(x []float64) float64 {
	a, b, c, d := x[0], x[1], x[2], x[3]
	return (a+10*b)*(a+10*b) + 5*(c-d)*(c-d) + math.Pow(b-2*c, 4) + 10*math.Pow(a-d, 4)
}

func powellSingDer(x, g []float64) {
	a, b, c, d := x[0], x[1], x[2], x[3]
	g[0] = 2*(a+10*b) + 40*math.Pow(a-d, 3)
	g[1] = 20*(a+10*b) + 4*math.Pow(b-2*c, 3)
	g[2] = 10*(c-d) - 8*math.Pow(b-2*c, 3)
	g[3] = -10*(c-d) - 40*math.Pow(a-d, 3)
}

func wood(x []float64) float64 {
	a, b, c, d := x[0], x[1], x[2], x[3]
	return 100*(b-a*a)*(b-a*a) + (1-a)*(1-a) + 90*(d-c*c)*(d-c*c) + (1-c)*(1-c) +
		10.1*((b-1)*(b-1)+(d-1)*(d-1)) + 19.8*(b-1)*(d-1)
}

func woodDer(x, g []float64) {
	a, b, c, d := x[0], x[1], x[2], x[3]
	g[0] = -400*a*(b-a*a) - 2*(1-a)
	g[1] = 200*(b-a*a) + 20.2*(b-1) + 19.8*(d-1)
	g[2] = -360*c*(d-c*c) - 2*(1-c)
	g[3] = 180*(d-c*c) + 20.2*(d-1) + 19.8*(b-1)
}

var testFuncs = map[string]testFunc{
	"rosen2":     {rosen, rosenDer},
	"rosen10":    {rosen, rosenDer},
	"rosen100":   {rosen, rosenDer},
	"beale":      {beale, bealeDer},
	"himmelblau": {himmelblau, himmelblauDer},
	"powell":     {powellSing, powellSingDer},
	"wood":       {wood, woodDer},
}

func altStart(n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		if i%2 == 0 {
			x[i] = -1.2
		} else {
			x[i] = 1
		}
	}
	return x
}

var testStarts = map[string][]float64{
	"rosen2":     {-1.2, 1},
	"rosen10":    altStart(10),
	"rosen100":   altStart(100),
	"beale":      {1, 1},
	"himmelblau": {0, 0},
	"powell":     {3, -1, 0, 1},
	"wood":       {-3, -1, -3, -1},
}
