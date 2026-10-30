package optimize

import "math"

// Hock–Schittkowski test problems (W. Hock and K. Schittkowski, Test Examples
// for Nonlinear Programming Codes, LNEMS 187, Springer 1981), the Go twins of
// tools/slsqporacle/hs.py. Keep the two in step: the fixtures in
// testdata/slsqp.json were produced by SciPy from the Python definitions.

type hsCon struct {
	kind ConstraintKind
	m    int
	fun  func(x, c []float64)
	jac  func(x, j []float64)
}

type hsProblem struct {
	x0     []float64
	f      func(x []float64) float64
	g      func(x, g []float64)
	cons   []hsCon
	bounds []Bound
}

func sq(x float64) float64 { return float64(x * x) }

func pinf() float64 { return math.Inf(1) }
func ninf() float64 { return math.Inf(-1) }

func hsRosen(x []float64) float64 { return float64(100*sq(x[1]-sq(x[0]))) + sq(1-x[0]) }
func hsRosenG(x, g []float64) {
	g[0] = float64(-400*x[0]*(x[1]-sq(x[0]))) - float64(2*(1-x[0]))
	g[1] = float64(200 * (x[1] - sq(x[0])))
}

func hs35f(x []float64) float64 {
	return 9 - float64(8*x[0]) - float64(6*x[1]) - float64(4*x[2]) + float64(2*sq(x[0])) + float64(2*sq(x[1])) + sq(x[2]) + float64(2*x[0]*x[1]) + float64(2*x[0]*x[2])
}

func hs35g(x, g []float64) {
	g[0] = -8 + float64(4*x[0]) + float64(2*x[1]) + float64(2*x[2])
	g[1] = -6 + float64(4*x[1]) + float64(2*x[0])
	g[2] = -4 + float64(2*x[2]) + float64(2*x[0])
}

func hs38f(x []float64) float64 {
	return float64(100*sq(x[1]-sq(x[0]))) + sq(1-x[0]) + float64(90*sq(x[3]-sq(x[2]))) + sq(1-x[2]) + float64(10.1*(sq(x[1]-1)+sq(x[3]-1))) + float64(19.8*(x[1]-1)*(x[3]-1))
}

func hs38g(x, g []float64) {
	g[0] = float64(-400*x[0]*(x[1]-sq(x[0]))) - float64(2*(1-x[0]))
	g[1] = float64(200*(x[1]-sq(x[0]))) + float64(20.2*(x[1]-1)) + float64(19.8*(x[3]-1))
	g[2] = float64(-360*x[2]*(x[3]-sq(x[2]))) - float64(2*(1-x[2]))
	g[3] = float64(180*(x[3]-sq(x[2]))) + float64(20.2*(x[3]-1)) + float64(19.8*(x[1]-1))
}

func hs43c(x, c []float64) {
	c[0] = 8 - sq(x[0]) - sq(x[1]) - sq(x[2]) - sq(x[3]) - x[0] + x[1] - x[2] + x[3]
	c[1] = 10 - sq(x[0]) - float64(2*sq(x[1])) - sq(x[2]) - float64(2*sq(x[3])) + x[0] + x[3]
	c[2] = 5 - float64(2*sq(x[0])) - sq(x[1]) - sq(x[2]) - float64(2*x[0]) + x[1] + x[3]
}

func hs43j(x, j []float64) {
	copy(j, []float64{float64(-2*x[0]) - 1, float64(-2*x[1]) + 1, float64(-2*x[2]) - 1, float64(-2*x[3]) + 1, float64(-2*x[0]) + 1, -4 * x[1], -2 * x[2], float64(-4*x[3]) + 1, float64(-4*x[0]) - 2, float64(-2*x[1]) + 1, -2 * x[2], 1})
}

func hs71c1(x, c []float64) { c[0] = float64(x[0]*x[1]*x[2]*x[3]) - 25 }
func hs71j1(x, j []float64) {
	copy(j, []float64{x[1] * x[2] * x[3], x[0] * x[2] * x[3], x[0] * x[1] * x[3], x[0] * x[1] * x[2]})
}
func hs71c2(x, c []float64) { c[0] = sq(x[0]) + sq(x[1]) + sq(x[2]) + sq(x[3]) - 40 }
func hs71j2(x, j []float64) { copy(j, []float64{2 * x[0], 2 * x[1], 2 * x[2], 2 * x[3]}) }

func rep(b Bound, n int) []Bound {
	out := make([]Bound, n)
	for i := range out {
		out[i] = b
	}
	return out
}

var hsProblems = map[string]hsProblem{
	"HS1": {x0: []float64{-2, 1}, f: hsRosen, g: hsRosenG,
		bounds: []Bound{Unbounded(), {-1.5, pinf()}}},
	"HS2": {x0: []float64{-2, 1}, f: hsRosen, g: hsRosenG,
		bounds: []Bound{Unbounded(), {1.5, pinf()}}},
	"HS3": {x0: []float64{10, 1},
		f:      func(x []float64) float64 { return x[1] + float64(1e-5*sq(x[1]-x[0])) },
		g:      func(x, g []float64) { g[0] = float64(-2e-5 * (x[1] - x[0])); g[1] = 1 + float64(2e-5*(x[1]-x[0])) },
		bounds: []Bound{Unbounded(), {0, pinf()}}},
	"HS4": {x0: []float64{1.125, 0.125},
		f:      func(x []float64) float64 { return math.Pow(x[0]+1, 3)/3 + x[1] },
		g:      func(x, g []float64) { g[0] = sq(x[0] + 1); g[1] = 1 },
		bounds: []Bound{{1, pinf()}, {0, pinf()}}},
	"HS5": {x0: []float64{0, 0},
		f: func(x []float64) float64 {
			return math.Sin(x[0]+x[1]) + sq(x[0]-x[1]) - float64(1.5*x[0]) + float64(2.5*x[1]) + 1
		},
		g: func(x, g []float64) {
			g[0] = math.Cos(x[0]+x[1]) + float64(2*(x[0]-x[1])) - 1.5
			g[1] = math.Cos(x[0]+x[1]) - float64(2*(x[0]-x[1])) + 2.5
		},
		bounds: []Bound{{-1.5, 4}, {-3, 3}}},
	"HS6": {x0: []float64{-1.2, 1},
		f: func(x []float64) float64 { return sq(1 - x[0]) },
		g: func(x, g []float64) { g[0] = float64(-2 * (1 - x[0])); g[1] = 0 },
		cons: []hsCon{{Eq, 1,
			func(x, c []float64) { c[0] = float64(10 * (x[1] - sq(x[0]))) },
			func(x, j []float64) { j[0], j[1] = float64(-20*x[0]), 10 }}}},
	"HS7": {x0: []float64{2, 2},
		f: func(x []float64) float64 { return math.Log(1+sq(x[0])) - x[1] },
		g: func(x, g []float64) { g[0] = 2 * x[0] / (1 + sq(x[0])); g[1] = -1 },
		cons: []hsCon{{Eq, 1,
			func(x, c []float64) { c[0] = sq(1+sq(x[0])) + sq(x[1]) - 4 },
			func(x, j []float64) { j[0], j[1] = float64(4*x[0]*(1+sq(x[0]))), float64(2*x[1]) }}}},
	"HS8": {x0: []float64{2, 1},
		f: func(x []float64) float64 { return -1 },
		g: func(x, g []float64) { g[0], g[1] = 0, 0 },
		cons: []hsCon{{Eq, 2,
			func(x, c []float64) { c[0], c[1] = sq(x[0])+sq(x[1])-25, float64(x[0]*x[1])-9 },
			func(x, j []float64) { copy(j, []float64{2 * x[0], 2 * x[1], x[1], x[0]}) }}}},
	"HS9": {x0: []float64{0, 0},
		f: func(x []float64) float64 { return float64(math.Sin(math.Pi*x[0]/12) * math.Cos(math.Pi*x[1]/16)) },
		g: func(x, g []float64) {
			g[0] = float64(math.Pi / 12 * math.Cos(math.Pi*x[0]/12) * math.Cos(math.Pi*x[1]/16))
			g[1] = float64(-math.Pi / 16 * math.Sin(math.Pi*x[0]/12) * math.Sin(math.Pi*x[1]/16))
		},
		cons: []hsCon{{Eq, 1,
			func(x, c []float64) { c[0] = float64(4*x[0]) - float64(3*x[1]) },
			func(x, j []float64) { j[0], j[1] = 4, -3 }}}},
	"HS10": {x0: []float64{-10, 10},
		f: func(x []float64) float64 { return x[0] - x[1] },
		g: func(x, g []float64) { g[0], g[1] = 1, -1 },
		cons: []hsCon{{Ineq, 1,
			func(x, c []float64) { c[0] = float64(-3*sq(x[0])) + float64(2*x[0]*x[1]) - sq(x[1]) + 1 },
			func(x, j []float64) { j[0], j[1] = float64(-6*x[0])+float64(2*x[1]), float64(2*x[0])-float64(2*x[1]) }}}},
	"HS11": {x0: []float64{4.9, 0.1},
		f: func(x []float64) float64 { return sq(x[0]-5) + sq(x[1]) - 25 },
		g: func(x, g []float64) { g[0], g[1] = float64(2*(x[0]-5)), float64(2*x[1]) },
		cons: []hsCon{{Ineq, 1,
			func(x, c []float64) { c[0] = -sq(x[0]) + x[1] },
			func(x, j []float64) { j[0], j[1] = float64(-2*x[0]), 1 }}}},
	"HS12": {x0: []float64{0, 0},
		f: func(x []float64) float64 {
			return float64(0.5*sq(x[0])) + sq(x[1]) - float64(x[0]*x[1]) - float64(7*x[0]) - float64(7*x[1])
		},
		g: func(x, g []float64) { g[0], g[1] = x[0]-x[1]-7, float64(2*x[1])-x[0]-7 },
		cons: []hsCon{{Ineq, 1,
			func(x, c []float64) { c[0] = 25 - float64(4*sq(x[0])) - sq(x[1]) },
			func(x, j []float64) { j[0], j[1] = float64(-8*x[0]), float64(-2*x[1]) }}}},
	"HS13": {x0: []float64{-2, -2},
		f: func(x []float64) float64 { return sq(x[0]-2) + sq(x[1]) },
		g: func(x, g []float64) { g[0], g[1] = float64(2*(x[0]-2)), float64(2*x[1]) },
		cons: []hsCon{{Ineq, 1,
			func(x, c []float64) { c[0] = math.Pow(1-x[0], 3) - x[1] },
			func(x, j []float64) { j[0], j[1] = float64(-3*sq(1-x[0])), -1 }}},
		bounds: []Bound{{0, pinf()}, {0, pinf()}}},
	"HS14": {x0: []float64{2, 2},
		f: func(x []float64) float64 { return sq(x[0]-2) + sq(x[1]-1) },
		g: func(x, g []float64) { g[0], g[1] = float64(2*(x[0]-2)), float64(2*(x[1]-1)) },
		cons: []hsCon{
			{Eq, 1, func(x, c []float64) { c[0] = x[0] - float64(2*x[1]) + 1 },
				func(x, j []float64) { j[0], j[1] = 1, -2 }},
			{Ineq, 1, func(x, c []float64) { c[0] = -sq(x[0])/4 - sq(x[1]) + 1 },
				func(x, j []float64) { j[0], j[1] = -x[0]/2, float64(-2*x[1]) }}}},
	"HS15": {x0: []float64{-2, 1}, f: hsRosen, g: hsRosenG,
		cons: []hsCon{{Ineq, 2,
			func(x, c []float64) { c[0], c[1] = float64(x[0]*x[1])-1, x[0]+sq(x[1]) },
			func(x, j []float64) { copy(j, []float64{x[1], x[0], 1, 2 * x[1]}) }}},
		bounds: []Bound{{ninf(), 0.5}, Unbounded()}},
	"HS16": {x0: []float64{-2, 1}, f: hsRosen, g: hsRosenG,
		cons: []hsCon{{Ineq, 2,
			func(x, c []float64) { c[0], c[1] = x[0]+sq(x[1]), sq(x[0])+x[1] },
			func(x, j []float64) { copy(j, []float64{1, 2 * x[1], 2 * x[0], 1}) }}},
		bounds: []Bound{{-2, 0.5}, {ninf(), 1}}},
	"HS17": {x0: []float64{-2, 1}, f: hsRosen, g: hsRosenG,
		cons: []hsCon{{Ineq, 2,
			func(x, c []float64) { c[0], c[1] = sq(x[1])-x[0], sq(x[0])-x[1] },
			func(x, j []float64) { copy(j, []float64{-1, 2 * x[1], 2 * x[0], -1}) }}},
		bounds: []Bound{{-2, 0.5}, {ninf(), 1}}},
	"HS18": {x0: []float64{2, 2},
		f: func(x []float64) float64 { return float64(0.01*sq(x[0])) + sq(x[1]) },
		g: func(x, g []float64) { g[0], g[1] = float64(0.02*x[0]), float64(2*x[1]) },
		cons: []hsCon{{Ineq, 2,
			func(x, c []float64) { c[0], c[1] = float64(x[0]*x[1])-25, sq(x[0])+sq(x[1])-25 },
			func(x, j []float64) { copy(j, []float64{x[1], x[0], 2 * x[0], 2 * x[1]}) }}},
		bounds: []Bound{{2, 50}, {0, 50}}},
	"HS19": {x0: []float64{20.1, 5.84},
		f: func(x []float64) float64 { return math.Pow(x[0]-10, 3) + math.Pow(x[1]-20, 3) },
		g: func(x, g []float64) { g[0], g[1] = float64(3*sq(x[0]-10)), float64(3*sq(x[1]-20)) },
		cons: []hsCon{{Ineq, 2,
			func(x, c []float64) {
				c[0] = sq(x[0]-5) + sq(x[1]-5) - 100
				c[1] = -sq(x[1]-5) - sq(x[0]-6) + 82.81
			},
			func(x, j []float64) {
				copy(j, []float64{2 * (x[0] - 5), 2 * (x[1] - 5), -2 * (x[0] - 6), -2 * (x[1] - 5)})
			}}},
		bounds: []Bound{{13, 100}, {0, 100}}},
	"HS21": {x0: []float64{-1, -1},
		f: func(x []float64) float64 { return float64(0.01*sq(x[0])) + sq(x[1]) - 100 },
		g: func(x, g []float64) { g[0], g[1] = float64(0.02*x[0]), float64(2*x[1]) },
		cons: []hsCon{{Ineq, 1,
			func(x, c []float64) { c[0] = float64(10*x[0]) - x[1] - 10 },
			func(x, j []float64) { j[0], j[1] = 10, -1 }}},
		bounds: []Bound{{2, 50}, {-50, 50}}},
	"HS22": {x0: []float64{2, 2},
		f: func(x []float64) float64 { return sq(x[0]-2) + sq(x[1]-1) },
		g: func(x, g []float64) { g[0], g[1] = float64(2*(x[0]-2)), float64(2*(x[1]-1)) },
		cons: []hsCon{{Ineq, 2,
			func(x, c []float64) { c[0], c[1] = -x[0]-x[1]+2, -sq(x[0])+x[1] },
			func(x, j []float64) { copy(j, []float64{-1, -1, -2 * x[0], 1}) }}}},
	"HS23": {x0: []float64{3, 1},
		f: func(x []float64) float64 { return sq(x[0]) + sq(x[1]) },
		g: func(x, g []float64) { g[0], g[1] = float64(2*x[0]), float64(2*x[1]) },
		cons: []hsCon{{Ineq, 5,
			func(x, c []float64) {
				c[0] = x[0] + x[1] - 1
				c[1] = sq(x[0]) + sq(x[1]) - 1
				c[2] = float64(9*sq(x[0])) + sq(x[1]) - 9
				c[3] = sq(x[0]) - x[1]
				c[4] = sq(x[1]) - x[0]
			},
			func(x, j []float64) {
				copy(j, []float64{1, 1, 2 * x[0], 2 * x[1], 18 * x[0], 2 * x[1], 2 * x[0], -1, -1, 2 * x[1]})
			}}},
		bounds: []Bound{{-50, 50}, {-50, 50}}},
	"HS26": {x0: []float64{-2.6, 2, 2},
		f: func(x []float64) float64 { return sq(x[0]-x[1]) + math.Pow(x[1]-x[2], 4) },
		g: func(x, g []float64) {
			g[0] = float64(2 * (x[0] - x[1]))
			g[1] = float64(-2*(x[0]-x[1])) + float64(4*math.Pow(x[1]-x[2], 3))
			g[2] = float64(-4 * math.Pow(x[1]-x[2], 3))
		},
		cons: []hsCon{{Eq, 1,
			func(x, c []float64) { c[0] = float64((1+sq(x[1]))*x[0]) + math.Pow(x[2], 4) - 3 },
			func(x, j []float64) { copy(j, []float64{1 + sq(x[1]), 2 * x[0] * x[1], 4 * math.Pow(x[2], 3)}) }}}},
	"HS27": {x0: []float64{2, 2, 2},
		f: func(x []float64) float64 { return float64(0.01*sq(x[0]-1)) + sq(x[1]-sq(x[0])) },
		g: func(x, g []float64) {
			g[0] = float64(0.02*(x[0]-1)) - float64(4*x[0]*(x[1]-sq(x[0])))
			g[1] = float64(2 * (x[1] - sq(x[0])))
			g[2] = 0
		},
		cons: []hsCon{{Eq, 1,
			func(x, c []float64) { c[0] = x[0] + sq(x[2]) + 1 },
			func(x, j []float64) { copy(j, []float64{1, 0, 2 * x[2]}) }}}},
	"HS28": {x0: []float64{-4, 1, 1},
		f: func(x []float64) float64 { return sq(x[0]+x[1]) + sq(x[1]+x[2]) },
		g: func(x, g []float64) {
			g[0] = float64(2 * (x[0] + x[1]))
			g[1] = float64(2*(x[0]+x[1])) + float64(2*(x[1]+x[2]))
			g[2] = float64(2 * (x[1] + x[2]))
		},
		cons: []hsCon{{Eq, 1,
			func(x, c []float64) { c[0] = x[0] + float64(2*x[1]) + float64(3*x[2]) - 1 },
			func(x, j []float64) { copy(j, []float64{1, 2, 3}) }}}},
	"HS29": {x0: []float64{1, 1, 1},
		f: func(x []float64) float64 { return float64(-x[0] * x[1] * x[2]) },
		g: func(x, g []float64) { g[0], g[1], g[2] = float64(-x[1]*x[2]), float64(-x[0]*x[2]), float64(-x[0]*x[1]) },
		cons: []hsCon{{Ineq, 1,
			func(x, c []float64) { c[0] = -sq(x[0]) - float64(2*sq(x[1])) - float64(4*sq(x[2])) + 48 },
			func(x, j []float64) { copy(j, []float64{-2 * x[0], -4 * x[1], -8 * x[2]}) }}}},
	"HS30": {x0: []float64{1, 1, 1},
		f: func(x []float64) float64 { return sq(x[0]) + sq(x[1]) + sq(x[2]) },
		g: func(x, g []float64) { g[0], g[1], g[2] = float64(2*x[0]), float64(2*x[1]), float64(2*x[2]) },
		cons: []hsCon{{Ineq, 1,
			func(x, c []float64) { c[0] = sq(x[0]) + sq(x[1]) - 1 },
			func(x, j []float64) { copy(j, []float64{2 * x[0], 2 * x[1], 0}) }}},
		bounds: []Bound{{1, 10}, {-10, 10}, {-10, 10}}},
	"HS35": {x0: []float64{0.5, 0.5, 0.5}, f: hs35f, g: hs35g,
		cons: []hsCon{{Ineq, 1,
			func(x, c []float64) { c[0] = 3 - x[0] - x[1] - float64(2*x[2]) },
			func(x, j []float64) { copy(j, []float64{-1, -1, -2}) }}},
		bounds: rep(Bound{0, math.Inf(1)}, 3)},
	"HS38": {x0: []float64{-3, -1, -3, -1}, f: hs38f, g: hs38g, bounds: rep(Bound{-10, 10}, 4)},
	"HS39": {x0: []float64{2, 2, 2, 2},
		f: func(x []float64) float64 { return -x[0] },
		g: func(x, g []float64) { copy(g, []float64{-1, 0, 0, 0}) },
		cons: []hsCon{{Eq, 2,
			func(x, c []float64) {
				c[0] = x[1] - math.Pow(x[0], 3) - sq(x[2])
				c[1] = sq(x[0]) - x[1] - sq(x[3])
			},
			func(x, j []float64) {
				copy(j, []float64{-3 * sq(x[0]), 1, -2 * x[2], 0, 2 * x[0], -1, 0, -2 * x[3]})
			}}}},
	"HS40": {x0: []float64{0.8, 0.8, 0.8, 0.8},
		f: func(x []float64) float64 { return float64(-x[0] * x[1] * x[2] * x[3]) },
		g: func(x, g []float64) {
			copy(g, []float64{-x[1] * x[2] * x[3], -x[0] * x[2] * x[3], -x[0] * x[1] * x[3], -x[0] * x[1] * x[2]})
		},
		cons: []hsCon{{Eq, 3,
			func(x, c []float64) {
				c[0] = math.Pow(x[0], 3) + sq(x[1]) - 1
				c[1] = float64(sq(x[0])*x[3]) - x[2]
				c[2] = sq(x[3]) - x[1]
			},
			func(x, j []float64) {
				copy(j, []float64{3 * sq(x[0]), 2 * x[1], 0, 0, 2 * x[0] * x[3], 0, -1, sq(x[0]), 0, -1, 0, 2 * x[3]})
			}}}},
	"HS42": {x0: []float64{1, 1, 1, 1},
		f: func(x []float64) float64 {
			s := 0.0
			for i := range 4 {
				s += sq(x[i] - float64(i+1))
			}
			return s
		},
		g: func(x, g []float64) {
			for i := range 4 {
				g[i] = float64(2 * (x[i] - float64(i+1)))
			}
		},
		cons: []hsCon{{Eq, 2,
			func(x, c []float64) { c[0], c[1] = x[0]-2, sq(x[2])+sq(x[3])-2 },
			func(x, j []float64) { copy(j, []float64{1, 0, 0, 0, 0, 0, 2 * x[2], 2 * x[3]}) }}}},
	"HS43": {x0: []float64{0, 0, 0, 0},
		f: func(x []float64) float64 {
			return sq(x[0]) + sq(x[1]) + float64(2*sq(x[2])) + sq(x[3]) - float64(5*x[0]) - float64(5*x[1]) - float64(21*x[2]) + float64(7*x[3])
		},
		g: func(x, g []float64) {
			copy(g, []float64{float64(2*x[0]) - 5, float64(2*x[1]) - 5, float64(4*x[2]) - 21, float64(2*x[3]) + 7})
		},
		cons: []hsCon{{Ineq, 3, hs43c, hs43j}}},
	"HS48": {x0: []float64{3, 5, -3, 2, -2},
		f: func(x []float64) float64 { return sq(x[0]-1) + sq(x[1]-x[2]) + sq(x[3]-x[4]) },
		g: func(x, g []float64) {
			copy(g, []float64{2 * (x[0] - 1), 2 * (x[1] - x[2]), -2 * (x[1] - x[2]), 2 * (x[3] - x[4]), -2 * (x[3] - x[4])})
		},
		cons: []hsCon{{Eq, 2,
			func(x, c []float64) { c[0], c[1] = x[0]+x[1]+x[2]+x[3]+x[4]-5, x[2]-float64(2*(x[3]+x[4]))+3 },
			func(x, j []float64) { copy(j, []float64{1, 1, 1, 1, 1, 0, 0, 1, -2, -2}) }}}},
	"HS71": {x0: []float64{1, 5, 5, 1},
		f: func(x []float64) float64 { return float64(x[0]*x[3]*(x[0]+x[1]+x[2])) + x[2] },
		g: func(x, g []float64) {
			copy(g, []float64{x[3] * (float64(2*x[0]) + x[1] + x[2]), x[0] * x[3], float64(x[0]*x[3]) + 1, x[0] * (x[0] + x[1] + x[2])})
		},
		cons:   []hsCon{{Ineq, 1, hs71c1, hs71j1}, {Eq, 1, hs71c2, hs71j2}},
		bounds: rep(Bound{1, 5}, 4)},
	"HS76": {x0: []float64{0.5, 0.5, 0.5, 0.5},
		f: func(x []float64) float64 {
			return sq(x[0]) + float64(0.5*sq(x[1])) + sq(x[2]) + float64(0.5*sq(x[3])) - float64(x[0]*x[2]) + float64(x[2]*x[3]) - x[0] - float64(3*x[1]) + x[2] - x[3]
		},
		g: func(x, g []float64) {
			copy(g, []float64{float64(2*x[0]) - x[2] - 1, x[1] - 3, float64(2*x[2]) - x[0] + x[3] + 1, x[3] + x[2] - 1})
		},
		cons: []hsCon{{Ineq, 3,
			func(x, c []float64) {
				c[0] = 5 - x[0] - float64(2*x[1]) - x[2] - x[3]
				c[1] = 4 - float64(3*x[0]) - x[1] - float64(2*x[2]) + x[3]
				c[2] = x[1] + float64(4*x[2]) - 1.5
			},
			func(x, j []float64) { copy(j, []float64{-1, -2, -1, -1, -3, -1, -2, 1, 0, 1, 4, 0}) }}},
		bounds: rep(Bound{0, math.Inf(1)}, 4)},
	"Rosen-free": {x0: []float64{-1.2, 1}, f: hsRosen, g: hsRosenG},
	"Infeasible": {x0: []float64{0},
		f: func(x []float64) float64 { return sq(x[0]) },
		g: func(x, g []float64) { g[0] = float64(2 * x[0]) },
		cons: []hsCon{{Ineq, 2,
			func(x, c []float64) { c[0], c[1] = x[0]-2, 1-x[0] },
			func(x, j []float64) { j[0], j[1] = 1, -1 }}}},
	"Overdetermined": {x0: []float64{0.5},
		f: func(x []float64) float64 { return sq(x[0]) },
		g: func(x, g []float64) { g[0] = float64(2 * x[0]) },
		cons: []hsCon{{Eq, 2,
			func(x, c []float64) { c[0], c[1] = x[0]-1, x[0]+1 },
			func(x, j []float64) { j[0], j[1] = 1, 1 }}}},
	"SingularEq": {x0: []float64{0, 0},
		f: func(x []float64) float64 { return sq(x[0]) + sq(x[1]) },
		g: func(x, g []float64) { g[0], g[1] = float64(2*x[0]), float64(2*x[1]) },
		cons: []hsCon{{Eq, 2,
			func(x, c []float64) { c[0], c[1] = x[0]+x[1]-1, x[0]+x[1]-2 },
			func(x, j []float64) { copy(j, []float64{1, 1, 1, 1}) }}}},
	"SingularEq3": {x0: []float64{0, 0, 0},
		f: func(x []float64) float64 { return sq(x[0]) + sq(x[1]) + sq(x[2]) },
		g: func(x, g []float64) { g[0], g[1], g[2] = float64(2*x[0]), float64(2*x[1]), float64(2*x[2]) },
		cons: []hsCon{{Eq, 2,
			func(x, c []float64) { c[0], c[1] = x[0]+x[1]-1, x[0]+x[1]-2 },
			func(x, j []float64) { copy(j, []float64{1, 1, 0, 1, 1, 0}) }}}},
	"DuplicateEq": {x0: []float64{0, 0},
		f: func(x []float64) float64 { return sq(x[0]) + sq(x[1]) },
		g: func(x, g []float64) { g[0], g[1] = float64(2*x[0]), float64(2*x[1]) },
		cons: []hsCon{{Eq, 2,
			func(x, c []float64) { c[0], c[1] = x[0]+x[1]-1, x[0]+x[1]-1 },
			func(x, j []float64) { copy(j, []float64{1, 1, 1, 1}) }}}},
	"Stationary": {x0: []float64{1 - 8e-7},
		f: func(x []float64) float64 { return x[0] },
		g: func(x, g []float64) { g[0] = 1 },
		cons: []hsCon{{Eq, 1,
			func(x, c []float64) { c[0] = x[0] - 1 },
			func(x, j []float64) { j[0] = 1 }}}},
	"FixedAll": {x0: []float64{1, 2}, f: hsRosen, g: hsRosenG,
		cons: []hsCon{{Ineq, 1,
			func(x, c []float64) { c[0] = x[0] + x[1] },
			func(x, j []float64) { j[0], j[1] = 1, 1 }}},
		bounds: []Bound{{1, 1}, {2, 2}}},
	"FixedAllViolated": {x0: []float64{1, 2}, f: hsRosen, g: hsRosenG,
		cons: []hsCon{{Ineq, 1,
			func(x, c []float64) { c[0] = x[0] - x[1] },
			func(x, j []float64) { j[0], j[1] = 1, -1 }}},
		bounds: []Bound{{1, 1}, {2, 2}}},
	"FixedSome": {x0: []float64{0.5, 2, 0.5},
		f: func(x []float64) float64 { return sq(x[0]-1) + sq(x[1]-2) + sq(x[2]-3) },
		g: func(x, g []float64) { g[0], g[1], g[2] = float64(2*(x[0]-1)), float64(2*(x[1]-2)), float64(2*(x[2]-3)) },
		cons: []hsCon{{Ineq, 1,
			func(x, c []float64) { c[0] = 2 - x[0] - x[2] },
			func(x, j []float64) { copy(j, []float64{-1, 0, -1}) }}},
		bounds: []Bound{{0, 5}, {2, 2}, {0, 5}}},
}

// specs returns the problem's constraints as Constraint values, with or
// without their Jacobians.
func (p hsProblem) specs(jac bool) []ConstraintSpec {
	var out []ConstraintSpec
	for _, c := range p.cons {
		k := Constraint{Kind: c.kind, M: c.m, Func: c.fun}
		if jac {
			k.Jac = c.jac
		}
		out = append(out, k)
	}
	return out
}

// hs71Nonlinear is hs.py's hs71_nonlinear: 25 <= x1x2x3x4, sum x^2 == 40.
func hs71Nonlinear(jac bool) []ConstraintSpec {
	c := NonlinearConstraint{M: 2,
		Func: func(x, c []float64) {
			c[0] = float64(x[0] * x[1] * x[2] * x[3])
			c[1] = sq(x[0]) + sq(x[1]) + sq(x[2]) + sq(x[3])
		},
		Lb: []float64{25, 40}, Ub: []float64{math.Inf(1), 40}}
	if jac {
		c.Jac = func(x, j []float64) { hs71j1(x, j[:4]); hs71j2(x, j[4:]) }
	}
	return []ConstraintSpec{c}
}

// hs43Nonlinear is hs.py's hs43_nonlinear: -c(x) <= 0.
func hs43Nonlinear() []ConstraintSpec {
	return []ConstraintSpec{NonlinearConstraint{M: 3,
		Func: func(x, c []float64) {
			hs43c(x, c)
			for i := range c {
				c[i] = -c[i]
			}
		},
		Jac: func(x, j []float64) {
			hs43j(x, j)
			for i := range j {
				j[i] = -j[i]
			}
		},
		Ub: []float64{0}}}
}
