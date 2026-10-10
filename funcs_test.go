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
		a := x[i+1] - float64(x[i]*x[i])
		g[i] += float64(-400*x[i]*a) - float64(2*(1-x[i]))
		g[i+1] += float64(200 * a)
	}
}

func rosenHess(x, h []float64) {
	n := len(x)
	for i := range h {
		h[i] = 0
	}
	for i := 0; i+1 < n; i++ {
		h[i*n+i] += float64(1200*x[i]*x[i]) - float64(400*x[i+1]) + 2
		h[i*n+i+1] += float64(-400 * x[i])
		h[(i+1)*n+i] += float64(-400 * x[i])
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
	t1 := 1.5 - a + float64(a*b)
	t2 := 2.25 - a + float64(a*b*b)
	t3 := 2.625 - a + float64(a*b*b*b)
	return float64(t1*t1) + float64(t2*t2) + float64(t3*t3)
}

func bealeDer(x, g []float64) {
	a, b := x[0], x[1]
	t1 := 1.5 - a + float64(a*b)
	t2 := 2.25 - a + float64(a*b*b)
	t3 := 2.625 - a + float64(a*b*b*b)
	g[0] = float64(2*t1*(b-1)) + float64(2*t2*(float64(b*b)-1)) + float64(2*t3*(float64(b*b*b)-1))
	g[1] = float64(2*t1*a) + float64(2*t2*2*a*b) + float64(2*t3*3*a*b*b)
}

func himmelblau(x []float64) float64 {
	a, b := x[0], x[1]
	u := float64(a*a) + b - 11
	v := a + float64(b*b) - 7
	return float64(u*u) + float64(v*v)
}

func himmelblauDer(x, g []float64) {
	a, b := x[0], x[1]
	u := float64(a*a) + b - 11
	v := a + float64(b*b) - 7
	g[0] = float64(4*a*u) + float64(2*v)
	g[1] = float64(2*u) + float64(4*b*v)
}

func powellSing(x []float64) float64 {
	a, b, c, d := x[0], x[1], x[2], x[3]
	return float64((a+float64(10*b))*(a+float64(10*b))) + float64(5*(c-d)*(c-d)) + math.Pow(b-float64(2*c), 4) + float64(10*math.Pow(a-d, 4))
}

func powellSingDer(x, g []float64) {
	a, b, c, d := x[0], x[1], x[2], x[3]
	g[0] = float64(2*(a+float64(10*b))) + float64(40*math.Pow(a-d, 3))
	g[1] = float64(20*(a+float64(10*b))) + float64(4*math.Pow(b-float64(2*c), 3))
	g[2] = float64(10*(c-d)) - float64(8*math.Pow(b-float64(2*c), 3))
	g[3] = float64(-10*(c-d)) - float64(40*math.Pow(a-d, 3))
}

func wood(x []float64) float64 {
	a, b, c, d := x[0], x[1], x[2], x[3]
	return float64(100*(b-float64(a*a))*(b-float64(a*a))) + float64((1-a)*(1-a)) + float64(90*(d-float64(c*c))*(d-float64(c*c))) + float64((1-c)*(1-c)) + float64(10.1*(float64((b-1)*(b-1))+float64((d-1)*(d-1)))) + float64(19.8*(b-1)*(d-1))
}

func woodDer(x, g []float64) {
	a, b, c, d := x[0], x[1], x[2], x[3]
	g[0] = float64(-400*a*(b-float64(a*a))) - float64(2*(1-a))
	g[1] = float64(200*(b-float64(a*a))) + float64(20.2*(b-1)) + float64(19.8*(d-1))
	g[2] = float64(-360*c*(d-float64(c*c))) - float64(2*(1-c))
	g[3] = float64(180*(d-float64(c*c))) + float64(20.2*(d-1)) + float64(19.8*(b-1))
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
