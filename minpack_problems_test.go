package optimize

import "math"

// The Moré–Garbow–Hillstrom test problems ("Testing unconstrained
// optimization software", ACM TOMS 7, 1981), written in the same operation
// order as testdata/gen/minpack.py so that both sides evaluate them
// identically up to the libm. Jacobians are m×n row-major.

type mpProblem struct {
	name string
	m, n int
	x0   []float64
	f    func(x, f []float64)
	j    func(x, j []float64)
}

func mpRosenbrock() mpProblem {
	return mpProblem{"rosenbrock", 2, 2, []float64{-1.2, 1},
		func(x, f []float64) { f[0] = 10 * (x[1] - float64(x[0]*x[0])); f[1] = 1 - x[0] },
		func(x, j []float64) { copy(j, []float64{-20 * x[0], 10, -1, 0}) }}
}

func mpPowellSingular() mpProblem {
	s5, s10 := math.Sqrt(5), math.Sqrt(10)
	return mpProblem{"powell_singular", 4, 4, []float64{3, -1, 0, 1},
		func(x, f []float64) {
			f[0] = x[0] + float64(10*x[1])
			f[1] = s5 * (x[2] - x[3])
			a := x[1] - float64(2*x[2])
			b := x[0] - x[3]
			f[2] = a * a
			f[3] = s10 * (b * b)
		},
		func(x, j []float64) {
			a := 2 * (x[1] - float64(2*x[2]))
			b := 2 * s10 * (x[0] - x[3])
			copy(j, []float64{1, 10, 0, 0, 0, 0, s5, -s5, 0, a, -2 * a, 0, b, 0, 0, -b})
		}}
}

func mpPowellBadlyScaled() mpProblem {
	return mpProblem{"powell_badly_scaled", 2, 2, []float64{0, 1},
		func(x, f []float64) {
			f[0] = float64(1e4*x[0]*x[1]) - 1
			f[1] = math.Exp(-x[0]) + math.Exp(-x[1]) - 1.0001
		},
		func(x, j []float64) {
			copy(j, []float64{1e4 * x[1], 1e4 * x[0], -math.Exp(-x[0]), -math.Exp(-x[1])})
		}}
}

func mpHelicalValley() mpProblem {
	tpi := 8 * math.Atan(1)
	return mpProblem{"helical_valley", 3, 3, []float64{-1, 0, 0},
		func(x, f []float64) {
			var theta float64
			if x[0] == 0 {
				theta = math.Copysign(0.25, x[1])
			} else {
				theta = math.Atan(x[1]/x[0]) / tpi
				if x[0] < 0 {
					theta += 0.5
				}
			}
			f[0] = 10 * (x[2] - float64(10*theta))
			f[1] = 10 * (math.Sqrt(float64(x[0]*x[0])+float64(x[1]*x[1])) - 1)
			f[2] = x[2]
		},
		func(x, j []float64) {
			r2 := float64(x[0]*x[0]) + float64(x[1]*x[1])
			r := math.Sqrt(r2)
			t := 100 / (tpi * r2)
			copy(j, []float64{t * x[1], -t * x[0], 10, 10 * x[0] / r, 10 * x[1] / r, 0, 0, 0, 1})
		}}
}

func mpTrigonometric(n int) mpProblem {
	x0 := make([]float64, n)
	for i := range x0 {
		x0[i] = 1 / float64(n)
	}
	return mpProblem{"trigonometric", n, n, x0,
		func(x, f []float64) {
			s := 0.0
			for _, v := range x {
				s += math.Cos(v)
			}
			for i := range n {
				f[i] = float64(n) - s + float64(float64(i+1)*(1-math.Cos(x[i]))) - math.Sin(x[i])
			}
		},
		func(x, j []float64) {
			for i := range n {
				for k := range n {
					j[i*n+k] = math.Sin(x[k])
				}
				j[i*n+i] = float64(float64(i+2)*math.Sin(x[i])) - math.Cos(x[i])
			}
		}}
}

func mpBroydenTridiagonal(n int) mpProblem {
	x0 := make([]float64, n)
	for i := range x0 {
		x0[i] = -1
	}
	return mpProblem{"broyden_tridiagonal", n, n, x0,
		func(x, f []float64) {
			for i := range n {
				v := float64((3-float64(2*x[i]))*x[i]) + 1
				if i > 0 {
					v -= x[i-1]
				}
				if i < n-1 {
					v -= float64(2 * x[i+1])
				}
				f[i] = v
			}
		},
		func(x, j []float64) {
			clear(j)
			for i := range n {
				j[i*n+i] = 3 - float64(4*x[i])
				if i > 0 {
					j[i*n+i-1] = -1
				}
				if i < n-1 {
					j[i*n+i+1] = -2
				}
			}
		}}
}

func mpBroydenBanded(n int) mpProblem {
	const ml, mu = 5, 1
	x0 := make([]float64, n)
	for i := range x0 {
		x0[i] = -1
	}
	return mpProblem{"broyden_banded", n, n, x0,
		func(x, f []float64) {
			for i := range n {
				v := float64(x[i]*(2+float64(5*x[i]*x[i]))) + 1
				for k := max(0, i-ml); k < min(n, i+mu+1); k++ {
					if k != i {
						v -= float64(x[k] * (1 + x[k]))
					}
				}
				f[i] = v
			}
		},
		func(x, j []float64) {
			clear(j)
			for i := range n {
				for k := max(0, i-ml); k < min(n, i+mu+1); k++ {
					if k != i {
						j[i*n+k] = -(1 + float64(2*x[k]))
					}
				}
				j[i*n+i] = 2 + float64(15*x[i]*x[i])
			}
		}}
}

func mpChebyquad(n int) mpProblem {
	m := n
	x0 := make([]float64, n)
	for k := range x0 {
		x0[k] = float64(k+1) / float64(n+1)
	}
	return mpProblem{"chebyquad", m, n, x0,
		func(x, f []float64) {
			clear(f[:m])
			for _, v := range x {
				t1, t2 := 1.0, float64(2*v)-1
				t := 2 * t2
				for i := range m {
					f[i] += t2
					th := float64(t*t2) - t1
					t1, t2 = t2, th
				}
			}
			for i := range m {
				f[i] /= float64(n)
				if (i+1)%2 == 0 {
					f[i] += 1 / float64((i+1)*(i+1)-1)
				}
			}
		},
		func(x, j []float64) {
			for k, v := range x {
				t1, t2 := 1.0, float64(2*v)-1
				t := 2 * t2
				s1, s2 := 0.0, 2.0
				for i := range m {
					j[i*n+k] = s2 / float64(n)
					th := float64(4*t2) + float64(t*s2) - s1
					s1, s2 = s2, th
					th = float64(t*t2) - t1
					t1, t2 = t2, th
				}
			}
		}}
}

func mpOnes(n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = 1
	}
	return x
}

func mpLinearFullRank() mpProblem {
	const n, m = 5, 10
	return mpProblem{"linear_full_rank", m, n, mpOnes(n),
		func(x, f []float64) {
			s := 0.0
			for _, v := range x {
				s += v
			}
			t := float64(2*s/m) + 1
			for i := range m {
				xi := 0.0
				if i < n {
					xi = x[i]
				}
				f[i] = xi - t
			}
		},
		func(x, j []float64) {
			for i := range m {
				for k := range n {
					d := 0.0
					if i == k {
						d = 1
					}
					j[i*n+k] = d - 2.0/m
				}
			}
		}}
}

func mpLinearRank1() mpProblem {
	const n, m = 5, 10
	return mpProblem{"linear_rank1", m, n, mpOnes(n),
		func(x, f []float64) {
			s := 0.0
			for k := range n {
				s += float64(float64(k+1) * x[k])
			}
			for i := range m {
				f[i] = float64(float64(i+1)*s) - 1
			}
		},
		func(x, j []float64) {
			for i := range m {
				for k := range n {
					j[i*n+k] = float64((i + 1) * (k + 1))
				}
			}
		}}
}

func mpLinearRank1Zero() mpProblem {
	const n, m = 5, 10
	return mpProblem{"linear_rank1_zero", m, n, mpOnes(n),
		func(x, f []float64) {
			s := 0.0
			for k := 1; k < n-1; k++ {
				s += float64(float64(k+1) * x[k])
			}
			for i := range m {
				f[i] = float64(float64(i)*s) - 1
			}
			f[0], f[m-1] = -1, -1
		},
		func(x, j []float64) {
			clear(j)
			for i := 1; i < m-1; i++ {
				for k := 1; k < n-1; k++ {
					j[i*n+k] = float64(i * (k + 1))
				}
			}
		}}
}

func mpFreudensteinRoth() mpProblem {
	return mpProblem{"freudenstein_roth", 2, 2, []float64{0.5, -2},
		func(x, f []float64) {
			f[0] = -13 + x[0] + float64((float64((5-x[1])*x[1])-2)*x[1])
			f[1] = -29 + x[0] + float64((float64((x[1]+1)*x[1])-14)*x[1])
		},
		func(x, j []float64) {
			copy(j, []float64{1, float64(x[1]*(10-float64(3*x[1]))) - 2, 1, float64(x[1]*(float64(3*x[1])+2)) - 14})
		}}
}

var mpBardY = []float64{0.14, 0.18, 0.22, 0.25, 0.29, 0.32, 0.35, 0.39, 0.37, 0.58, 0.73, 0.96, 1.34, 2.10, 4.39}

func mpBard() mpProblem {
	return mpProblem{"bard", 15, 3, []float64{1, 1, 1},
		func(x, f []float64) {
			for i := range 15 {
				u, v := float64(i)+1, 15-float64(i)
				w := math.Min(u, v)
				f[i] = mpBardY[i] - (x[0] + u/(float64(v*x[1])+float64(w*x[2])))
			}
		},
		func(x, j []float64) {
			for i := range 15 {
				u, v := float64(i)+1, 15-float64(i)
				w := math.Min(u, v)
				d := float64(v*x[1]) + float64(w*x[2])
				d2 := d * d
				copy(j[3*i:], []float64{-1, u * v / d2, u * w / d2})
			}
		}}
}

var (
	mpKOY = []float64{0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246}
	mpKOU = []float64{4, 2, 1, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625}
)

func mpKowalikOsborne() mpProblem {
	return mpProblem{"kowalik_osborne", 11, 4, []float64{0.25, 0.39, 0.415, 0.39},
		func(x, f []float64) {
			for i := range 11 {
				u := mpKOU[i]
				f[i] = mpKOY[i] - x[0]*(float64(u*u)+float64(u*x[1]))/(float64(u*u)+float64(u*x[2])+x[3])
			}
		},
		func(x, j []float64) {
			for i := range 11 {
				u := mpKOU[i]
				num := float64(u*u) + float64(u*x[1])
				den := float64(u*u) + float64(u*x[2]) + x[3]
				copy(j[4*i:], []float64{-num / den, -x[0] * u / den, x[0] * num * u / (den * den), x[0] * num / (den * den)})
			}
		}}
}

var mpMeyerY = []float64{34780, 28610, 23650, 19630, 16370, 13720, 11540, 9744,
	8261, 7030, 6005, 5147, 4427, 3820, 3307, 2872}

func mpMeyer() mpProblem {
	return mpProblem{"meyer", 16, 3, []float64{0.02, 4000, 250},
		func(x, f []float64) {
			for i := range 16 {
				f[i] = float64(x[0]*math.Exp(x[1]/(45.0+float64(5*(i+1))+x[2]))) - mpMeyerY[i]
			}
		},
		func(x, j []float64) {
			for i := range 16 {
				d := 45.0 + float64(5*(i+1)) + x[2]
				e := math.Exp(x[1] / d)
				copy(j[3*i:], []float64{e, x[0] * e / d, -x[0] * x[1] * e / (d * d)})
			}
		}}
}

func mpWatson(n int) mpProblem {
	name := map[int]string{6: "watson6", 9: "watson9"}[n]
	return mpProblem{name, 31, n, make([]float64, n),
		func(x, f []float64) {
			for i := range 29 {
				t := float64(i+1) / 29
				s1, dx := 0.0, 1.0
				for k := 1; k < n; k++ {
					s1 += float64(float64(k) * x[k] * dx)
					dx *= t
				}
				s2 := 0.0
				dx = 1
				for k := range n {
					s2 += float64(x[k] * dx)
					dx *= t
				}
				f[i] = s1 - float64(s2*s2) - 1
			}
			f[29] = x[0]
			f[30] = x[1] - float64(x[0]*x[0]) - 1
		},
		func(x, j []float64) {
			clear(j)
			for i := range 29 {
				t := float64(i+1) / 29
				s2, dx := 0.0, 1.0
				for k := range n {
					s2 += float64(x[k] * dx)
					dx *= t
				}
				dx = 1
				for k := range n {
					j[i*n+k] = dx * (float64(k)/t - float64(2*s2))
					dx *= t
				}
			}
			j[29*n] = 1
			j[30*n] = -2 * x[0]
			j[30*n+1] = 1
		}}
}

func mpBox3D() mpProblem {
	return mpProblem{"box3d", 10, 3, []float64{0, 10, 20},
		func(x, f []float64) {
			for i := range 10 {
				t := 0.1 * float64(i+1)
				f[i] = math.Exp(-t*x[0]) - math.Exp(-t*x[1]) - float64(x[2]*(math.Exp(-t)-math.Exp(-10*t)))
			}
		},
		func(x, j []float64) {
			for i := range 10 {
				t := 0.1 * float64(i+1)
				copy(j[3*i:], []float64{-t * math.Exp(-t*x[0]), t * math.Exp(-t*x[1]), -(math.Exp(-t) - math.Exp(-10*t))})
			}
		}}
}

func mpJennrichSampson() mpProblem {
	return mpProblem{"jennrich_sampson", 10, 2, []float64{0.3, 0.4},
		func(x, f []float64) {
			for i := range 10 {
				k := float64(i + 1)
				f[i] = 2 + float64(2*k) - (math.Exp(k*x[0]) + math.Exp(k*x[1]))
			}
		},
		func(x, j []float64) {
			for i := range 10 {
				k := float64(i + 1)
				j[2*i] = -k * math.Exp(k*x[0])
				j[2*i+1] = -k * math.Exp(k*x[1])
			}
		}}
}

func mpBrownDennis() mpProblem {
	return mpProblem{"brown_dennis", 20, 4, []float64{25, 5, -5, -1},
		func(x, f []float64) {
			for i := range 20 {
				t := float64(i+1) / 5
				a := x[0] + float64(t*x[1]) - math.Exp(t)
				b := x[2] + float64(x[3]*math.Sin(t)) - math.Cos(t)
				f[i] = float64(a*a) + float64(b*b)
			}
		},
		func(x, j []float64) {
			for i := range 20 {
				t := float64(i+1) / 5
				a := x[0] + float64(t*x[1]) - math.Exp(t)
				b := x[2] + float64(x[3]*math.Sin(t)) - math.Cos(t)
				copy(j[4*i:], []float64{2 * a, 2 * t * a, 2 * b, 2 * math.Sin(t) * b})
			}
		}}
}

var mpOsb1Y = []float64{0.844, 0.908, 0.932, 0.936, 0.925, 0.908, 0.881, 0.850, 0.818, 0.784, 0.751,
	0.718, 0.685, 0.658, 0.628, 0.603, 0.580, 0.558, 0.538, 0.522, 0.506, 0.490,
	0.478, 0.467, 0.457, 0.448, 0.438, 0.431, 0.424, 0.420, 0.414, 0.411, 0.406}

func mpOsborne1() mpProblem {
	return mpProblem{"osborne1", 33, 5, []float64{0.5, 1.5, -1, 0.01, 0.02},
		func(x, f []float64) {
			for i := range 33 {
				t := 10 * float64(i)
				f[i] = mpOsb1Y[i] - (x[0] + float64(x[1]*math.Exp(-t*x[3])) + float64(x[2]*math.Exp(-t*x[4])))
			}
		},
		func(x, j []float64) {
			for i := range 33 {
				t := 10 * float64(i)
				e4, e5 := math.Exp(-t*x[3]), math.Exp(-t*x[4])
				copy(j[5*i:], []float64{-1, -e4, -e5, t * x[1] * e4, t * x[2] * e5})
			}
		}}
}

func mpBiggsExp6() mpProblem {
	return mpProblem{"biggs_exp6", 13, 6, []float64{1, 2, 1, 1, 1, 1},
		func(x, f []float64) {
			for i := range 13 {
				t := 0.1 * float64(i+1)
				y := math.Exp(-t) - float64(5*math.Exp(-10*t)) + float64(3*math.Exp(-4*t))
				f[i] = float64(x[2]*math.Exp(-t*x[0])) - float64(x[3]*math.Exp(-t*x[1])) + float64(x[5]*math.Exp(-t*x[4])) - y
			}
		},
		func(x, j []float64) {
			for i := range 13 {
				t := 0.1 * float64(i+1)
				e1, e2, e5 := math.Exp(-t*x[0]), math.Exp(-t*x[1]), math.Exp(-t*x[4])
				copy(j[6*i:], []float64{-t * x[2] * e1, t * x[3] * e2, e1, -e2, -t * x[5] * e5, e5})
			}
		}}
}

var mpOsb2Y = []float64{1.366, 1.191, 1.112, 1.013, 0.991, 0.885, 0.831, 0.847, 0.786, 0.725, 0.746,
	0.679, 0.608, 0.655, 0.616, 0.606, 0.602, 0.626, 0.651, 0.724, 0.649, 0.649,
	0.694, 0.644, 0.624, 0.661, 0.612, 0.558, 0.533, 0.495, 0.500, 0.423, 0.395,
	0.375, 0.372, 0.391, 0.396, 0.405, 0.428, 0.429, 0.523, 0.562, 0.607, 0.653,
	0.672, 0.708, 0.633, 0.668, 0.645, 0.632, 0.591, 0.559, 0.597, 0.625, 0.739,
	0.710, 0.729, 0.720, 0.636, 0.581, 0.428, 0.292, 0.162, 0.098, 0.054}

func mpOsborne2() mpProblem {
	return mpProblem{"osborne2", 65, 11, []float64{1.3, 0.65, 0.65, 0.7, 0.6, 3, 5, 7, 2, 4.5, 5.5},
		func(x, f []float64) {
			for i := range 65 {
				t := float64(i) / 10
				a, b, c := t-x[8], t-x[9], t-x[10]
				f[i] = mpOsb2Y[i] - (float64(x[0]*math.Exp(-t*x[4])) + float64(x[1]*math.Exp(-a*a*x[5])) +
					float64(x[2]*math.Exp(-b*b*x[6])) + float64(x[3]*math.Exp(-c*c*x[7])))
			}
		},
		func(x, j []float64) {
			for i := range 65 {
				t := float64(i) / 10
				a, b, c := t-x[8], t-x[9], t-x[10]
				e0, e1 := math.Exp(-t*x[4]), math.Exp(-a*a*x[5])
				e2, e3 := math.Exp(-b*b*x[6]), math.Exp(-c*c*x[7])
				copy(j[11*i:], []float64{-e0, -e1, -e2, -e3, t * x[0] * e0, a * a * x[1] * e1, b * b * x[2] * e2,
					c * c * x[3] * e3, -2 * a * x[5] * x[1] * e1, -2 * b * x[6] * x[2] * e2, -2 * c * x[7] * x[3] * e3})
			}
		}}
}

func mpNoRoot(x0 float64) mpProblem {
	name := "no_root"
	if x0 == 0 {
		name = "no_root_at_0"
	}
	return mpProblem{name, 1, 1, []float64{x0},
		func(x, f []float64) { f[0] = float64(x[0]*x[0]) + 1 },
		func(x, j []float64) { j[0] = 2 * x[0] }}
}

// mpProblems indexes every problem by the name the fixture uses.
func mpProblems() map[string]mpProblem {
	ps := []mpProblem{mpRosenbrock(), mpPowellSingular(), mpPowellBadlyScaled(), mpHelicalValley(),
		mpTrigonometric(10), mpBroydenTridiagonal(10), mpBroydenBanded(10), mpChebyquad(5),
		mpLinearFullRank(), mpLinearRank1(), mpLinearRank1Zero(), mpFreudensteinRoth(), mpBard(),
		mpKowalikOsborne(), mpMeyer(), mpWatson(6), mpWatson(9), mpBox3D(), mpJennrichSampson(),
		mpBrownDennis(), mpOsborne1(), mpBiggsExp6(), mpOsborne2(), mpNoRoot(1), mpNoRoot(0)}
	out := make(map[string]mpProblem, len(ps))
	for _, p := range ps {
		out[p.name] = p
	}
	return out
}
