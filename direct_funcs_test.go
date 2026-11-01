package optimize

import "math"

// The objectives of tools/globaloracle/global.py, written as the same chain
// of single IEEE operations: every product that feeds a sum is wrapped in
// float64() so that no architecture fuses it (Python never does), and no
// constant expression is folded (math.Pi is copied to a variable first).
// The "arith" family uses no libm and so is bit-identical to Python's; the
// "trans" family calls cos/exp/sqrt, whose Go and glibc results may differ in
// the last bit.

func dgShifted(n int) func(x []float64) float64 {
	c := make([]float64, n)
	for i := range c {
		c[i] = 0.3 + float64(0.17*float64(i))
	}
	return func(x []float64) float64 {
		s := 0.0
		for i := 0; i < n; i++ {
			t := x[i] - c[i]
			s = s + float64(t*t)
		}
		return s
	}
}

func dgRosenbrock(n int) func(x []float64) float64 {
	return func(x []float64) float64 {
		s := 0.0
		for i := 0; i < n-1; i++ {
			t := x[i+1] - float64(x[i]*x[i])
			a := float64(100.0 * float64(t*t))
			u := 1.0 - x[i]
			b := float64(u * u)
			s = s + (a + b)
		}
		return s
	}
}

func dgGoldsteinPrice(x []float64) float64 {
	a, b := x[0], x[1]
	s := a + b
	s = s + 1.0
	s2 := float64(s * s)
	p := 19.0
	p = p - float64(14.0*a)
	p = p + float64(3.0*float64(a*a))
	p = p - float64(14.0*b)
	p = p + float64(6.0*float64(a*b))
	p = p + float64(3.0*float64(b*b))
	f1 := 1.0 + float64(s2*p)
	d := float64(2.0 * a)
	d = d - float64(3.0*b)
	d2 := float64(d * d)
	q := 18.0
	q = q - float64(32.0*a)
	q = q + float64(12.0*float64(a*a))
	q = q + float64(48.0*b)
	q = q - float64(36.0*float64(a*b))
	q = q + float64(27.0*float64(b*b))
	f2 := 30.0 + float64(d2*q)
	return float64(f1 * f2)
}

func dgSixHumpCamel(x []float64) float64 {
	a, b := x[0], x[1]
	a2 := float64(a * a)
	a4 := float64(a2 * a2)
	t := 4.0 - float64(2.1*a2)
	t = t + a4/3.0
	t1 := float64(t * a2)
	t2 := float64(a * b)
	b2 := float64(b * b)
	u := -4.0 + float64(4.0*b2)
	t3 := float64(u * b2)
	f := t1 + t2
	return f + t3
}

func dgStyblinskiTang(n int) func(x []float64) float64 {
	return func(x []float64) float64 {
		s := 0.0
		for i := 0; i < n; i++ {
			x2 := float64(x[i] * x[i])
			x4 := float64(x2 * x2)
			t := x4 - float64(16.0*x2)
			t = t + float64(5.0*x[i])
			s = s + t
		}
		return 0.5 * s
	}
}

var dgShekelA = [10][4]float64{
	{4, 4, 4, 4}, {1, 1, 1, 1}, {8, 8, 8, 8}, {6, 6, 6, 6}, {3, 7, 3, 7},
	{2, 9, 2, 9}, {5, 5, 3, 3}, {8, 1, 8, 1}, {6, 2, 6, 2}, {7, 3.6, 7, 3.6},
}
var dgShekelC = [10]float64{0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5}

func dgShekel(m int) func(x []float64) float64 {
	return func(x []float64) float64 {
		s := 0.0
		for i := 0; i < m; i++ {
			d := dgShekelC[i]
			for j := 0; j < 4; j++ {
				t := x[j] - dgShekelA[i][j]
				d = d + float64(t*t)
			}
			s = s + 1.0/d
		}
		return -s
	}
}

func dgBooth(x []float64) float64 {
	a := x[0] + float64(2.0*x[1])
	a = a - 7.0
	b := float64(2.0 * x[0])
	b = b + x[1]
	b = b - 5.0
	return float64(a*a) + float64(b*b)
}

func dgMatyas(x []float64) float64 {
	s := float64(x[0]*x[0]) + float64(x[1]*x[1])
	t := float64(0.26 * s)
	u := float64(0.48 * x[0])
	u = float64(u * x[1])
	return t - u
}

func dgBeale(x []float64) float64 {
	a, b := x[0], x[1]
	xy := float64(a * b)
	y2 := float64(b * b)
	y3 := float64(y2 * b)
	t1 := 1.5 - a
	t1 = t1 + xy
	t2 := 2.25 - a
	t2 = t2 + float64(a*y2)
	t3 := 2.625 - a
	t3 = t3 + float64(a*y3)
	f := float64(t1*t1) + float64(t2*t2)
	return f + float64(t3*t3)
}

var dgPi, dgE = math.Pi, math.E

func dgBranin(x []float64) float64 {
	a, b := x[0], x[1]
	k := float64(float64(4.0*dgPi) * dgPi)
	t := b - float64(float64(5.1/k)*float64(a*a))
	t = t + float64(5.0/dgPi*a)
	t = t - 6.0
	r := float64(t * t)
	s := float64(float64(10.0*(1.0-1.0/float64(8.0*dgPi))) * math.Cos(a))
	return (r + s) + 10.0
}

var (
	dgH3A = [4][3]float64{{3, 10, 30}, {0.1, 10, 35}, {3, 10, 30}, {0.1, 10, 35}}
	dgH3C = [4]float64{1, 1.2, 3, 3.2}
	dgH3P = [4][3]float64{{0.3689, 0.1170, 0.2673}, {0.4699, 0.4387, 0.7470},
		{0.1091, 0.8732, 0.5547}, {0.03815, 0.5743, 0.8828}}
	dgH6A = [4][6]float64{{10, 3, 17, 3.5, 1.7, 8}, {0.05, 10, 17, 0.1, 8, 14},
		{3, 3.5, 1.7, 10, 17, 8}, {17, 8, 0.05, 10, 0.1, 14}}
	dgH6P = [4][6]float64{{0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886},
		{0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991},
		{0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650},
		{0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381}}
)

func dgHartmann3(x []float64) float64 {
	s := 0.0
	for i := 0; i < 4; i++ {
		e := 0.0
		for j := 0; j < 3; j++ {
			t := x[j] - dgH3P[i][j]
			e = e + float64(dgH3A[i][j]*float64(t*t))
		}
		s = s + float64(dgH3C[i]*math.Exp(-e))
	}
	return -s
}

func dgHartmann6(x []float64) float64 {
	s := 0.0
	for i := 0; i < 4; i++ {
		e := 0.0
		for j := 0; j < 6; j++ {
			t := x[j] - dgH6P[i][j]
			e = e + float64(dgH6A[i][j]*float64(t*t))
		}
		s = s + float64(dgH3C[i]*math.Exp(-e))
	}
	return -s
}

func dgAckley2(x []float64) float64 {
	const n = 2
	s1, s2 := 0.0, 0.0
	for i := 0; i < n; i++ {
		s1 = s1 + float64(x[i]*x[i])
		s2 = s2 + math.Cos(float64(float64(2.0*dgPi)*x[i]))
	}
	a := float64(-0.2 * math.Sqrt(s1/n))
	return (float64(-20.0*math.Exp(a)) - math.Exp(s2/n)) + (20.0 + dgE)
}

func dgRastrigin2(x []float64) float64 {
	const n = 2
	s := 10.0 * n
	for i := 0; i < n; i++ {
		s = s + (float64(x[i]*x[i]) - float64(10.0*math.Cos(float64(float64(2.0*dgPi)*x[i]))))
	}
	return s
}

var dgSh2 = dgShifted(2)

func dgNaNRegion(x []float64) float64 {
	s := dgSh2(x)
	if x[0] > 0.1 {
		return math.NaN()
	}
	return s
}

func dgInfRegion(x []float64) float64 {
	s := dgSh2(x)
	if x[1] < 0.0 {
		return math.Inf(1)
	}
	return s
}

func dgNegInfRegion(x []float64) float64 {
	s := dgSh2(x)
	if x[0] > 3.0 && x[1] > 3.0 {
		return math.Inf(-1)
	}
	return s
}

func dgNaNCenter(x []float64) float64 {
	if x[0] == 0.0 && x[1] == 0.0 {
		return math.NaN()
	}
	return dgSh2(x)
}

func dgNaNAll([]float64) float64 { return math.NaN() }

// dgStyblinskiTangDoc is the docstring example of scipy.optimize.direct.
func dgStyblinskiTangDoc(x []float64) float64 {
	a, b := x[0], x[1]
	return 0.5 * (float64(a*a*a*a) - float64(16*a*a) + float64(5*a) + float64(b*b*b*b) - float64(16*b*b) + float64(5*b))
}

// dgProblems maps the problem names of global.py to their Go twins.
var dgProblems = map[string]func(x []float64) float64{
	"shifted_sphere_1": dgShifted(1), "shifted_sphere_2": dgShifted(2), "shifted_sphere_3": dgShifted(3),
	"shifted_sphere_5": dgShifted(5), "shifted_sphere_10": dgShifted(10),
	"rosenbrock_2": dgRosenbrock(2), "rosenbrock_5": dgRosenbrock(5),
	"goldstein_price": dgGoldsteinPrice, "six_hump_camel": dgSixHumpCamel,
	"styblinski_tang_2": dgStyblinskiTang(2), "styblinski_tang_4": dgStyblinskiTang(4),
	"shekel_5": dgShekel(5), "shekel_7": dgShekel(7), "shekel_10": dgShekel(10),
	"booth": dgBooth, "matyas": dgMatyas, "beale": dgBeale,
	"branin": dgBranin, "hartmann_3": dgHartmann3, "hartmann_6": dgHartmann6,
	"ackley_2": dgAckley2, "rastrigin_2": dgRastrigin2,
	"nan_region": dgNaNRegion, "inf_region": dgInfRegion, "neginf_region": dgNegInfRegion,
	"nan_center": dgNaNCenter, "nan_all": dgNaNAll,
	"docstring_pow": dgStyblinskiTangDoc,
}
