package optimize

import (
	_ "embed"
	"encoding/json"
	"errors"
	"math"
	"testing"
)

//go:embed testdata/trf.json
var trfJSON []byte

type trfFixture struct {
	Scipy string
	Data  map[string][]float64
	Cases []trfCase
}

type trfCase struct {
	Name       string
	Problem    string
	Jac        string
	X0         []float64
	Lb, Ub     []*float64
	Options    map[string]any
	X          []float64
	Cost       float64
	Fun        []float64
	Grad       []float64
	Optimality float64
	ActiveMask []int `json:"active_mask"`
	Nfev, Njev int
	Status     int
	Message    string
}

func loadTRFFixture(t *testing.T) trfFixture {
	t.Helper()
	var fx trfFixture
	if err := json.Unmarshal(trfJSON, &fx); err != nil {
		t.Fatal(err)
	}
	return fx
}

// trfProblem mirrors, operation for operation, the residuals and Jacobians
// of testdata/gen/trf.py.
func trfProblem(name string, data map[string][]float64) LSProblem {
	s5, s10 := math.Sqrt(5), math.Sqrt(10)
	switch name {
	case "rosenbrock":
		return LSProblem{M: 2,
			Func: func(x, f []float64) { f[0], f[1] = float64(10*(x[1]-float64(x[0]*x[0]))), 1-x[0] },
			Jac:  func(x, j []float64) { copy(j, []float64{-20 * x[0], 10, -1, 0}) }}
	case "helical":
		theta := func(x []float64) float64 {
			th := math.Atan(x[1]/x[0]) / (2 * math.Pi)
			if x[0] < 0 {
				th += 0.5
			}
			return th
		}
		return LSProblem{M: 3,
			Func: func(x, f []float64) {
				r := math.Sqrt(float64(x[0]*x[0]) + float64(x[1]*x[1]))
				f[0], f[1], f[2] = float64(10*(x[2]-float64(10*theta(x)))), float64(10*(r-1)), x[2]
			},
			Jac: func(x, j []float64) {
				r2 := float64(x[0]*x[0]) + float64(x[1]*x[1])
				r := math.Sqrt(r2)
				c := 100 / (2 * math.Pi * r2)
				copy(j, []float64{c * x[1], -c * x[0], 10, 10 * x[0] / r, 10 * x[1] / r, 0, 0, 0, 1})
			}}
	case "powell":
		return LSProblem{M: 4,
			Func: func(x, f []float64) {
				a, b := x[1]-float64(2*x[2]), x[0]-x[3]
				f[0], f[1], f[2], f[3] = x[0]+float64(10*x[1]), float64(s5*(x[2]-x[3])), float64(a*a), float64(s10*b*b)
			},
			Jac: func(x, j []float64) {
				a, b := x[1]-float64(2*x[2]), x[0]-x[3]
				copy(j, []float64{1, 10, 0, 0, 0, 0, s5, -s5, 0, 2 * a, -4 * a, 0, 2 * s10 * b, 0, 0, -2 * s10 * b})
			}}
	case "freudenstein":
		return LSProblem{M: 2,
			Func: func(x, f []float64) {
				f[0] = -13 + x[0] + float64((float64((5-x[1])*x[1])-2)*x[1])
				f[1] = -29 + x[0] + float64((float64((x[1]+1)*x[1])-14)*x[1])
			},
			Jac: func(x, j []float64) {
				copy(j, []float64{1, float64((10-float64(3*x[1]))*x[1]) - 2, 1, float64((float64(3*x[1])+2)*x[1]) - 14})
			}}
	case "bard":
		y := data["bard_y"]
		uvw := func(i int) (float64, float64, float64) {
			u, v := float64(i)+1, 15-float64(i)
			return u, v, math.Min(u, v)
		}
		return LSProblem{M: 15,
			Func: func(x, f []float64) {
				for i := range f {
					u, v, w := uvw(i)
					f[i] = y[i] - (x[0] + u/(float64(v*x[1])+float64(w*x[2])))
				}
			},
			Jac: func(x, j []float64) {
				for i := 0; i < 15; i++ {
					u, v, w := uvw(i)
					den := float64(v*x[1]) + float64(w*x[2])
					den2 := float64(den * den)
					copy(j[3*i:], []float64{-1, u * v / den2, u * w / den2})
				}
			}}
	case "kowalik":
		y, us := data["kowalik_y"], data["kowalik_u"]
		return LSProblem{M: 11,
			Func: func(x, f []float64) {
				for i, u := range us {
					num := float64(u*u) + float64(u*x[1])
					den := float64(u*u) + float64(u*x[2]) + x[3]
					f[i] = y[i] - x[0]*num/den
				}
			},
			Jac: func(x, j []float64) {
				for i, u := range us {
					num := float64(u*u) + float64(u*x[1])
					den := float64(u*u) + float64(u*x[2]) + x[3]
					den2 := float64(den * den)
					copy(j[4*i:], []float64{-num / den, -x[0] * u / den, x[0] * num * u / den2, x[0] * num / den2})
				}
			}}
	case "box3d":
		return LSProblem{M: 10,
			Func: func(x, f []float64) {
				for i := range f {
					t := float64(0.1 * float64(i+1))
					f[i] = math.Exp(-t*x[0]) - math.Exp(-t*x[1]) - float64(x[2]*(math.Exp(-t)-math.Exp(-10*t)))
				}
			},
			Jac: func(x, j []float64) {
				for i := 0; i < 10; i++ {
					t := float64(0.1 * float64(i+1))
					copy(j[3*i:], []float64{-t * math.Exp(-t*x[0]), t * math.Exp(-t*x[1]), -(math.Exp(-t) - math.Exp(-10*t))})
				}
			}}
	case "jennrich":
		return LSProblem{M: 10,
			Func: func(x, f []float64) {
				for i := range f {
					k := float64(i) + 1
					f[i] = 2 + float64(2*k) - (math.Exp(k*x[0]) + math.Exp(k*x[1]))
				}
			},
			Jac: func(x, j []float64) {
				for i := 0; i < 10; i++ {
					k := float64(i) + 1
					j[2*i], j[2*i+1] = float64(-k*math.Exp(k*x[0])), float64(-k*math.Exp(k*x[1]))
				}
			}}
	case "brown_dennis":
		ab := func(x []float64, i int) (float64, float64, float64) {
			t := float64(i+1) / 5
			return x[0] + float64(t*x[1]) - math.Exp(t), x[2] + float64(x[3]*math.Sin(t)) - math.Cos(t), t
		}
		return LSProblem{M: 20,
			Func: func(x, f []float64) {
				for i := range f {
					a, b, _ := ab(x, i)
					f[i] = float64(a*a) + float64(b*b)
				}
			},
			Jac: func(x, j []float64) {
				for i := 0; i < 20; i++ {
					a, b, t := ab(x, i)
					copy(j[4*i:], []float64{2 * a, 2 * a * t, 2 * b, 2 * b * math.Sin(t)})
				}
			}}
	case "osborne1":
		y := data["osborne1_y"]
		return LSProblem{M: 33,
			Func: func(x, f []float64) {
				for i := range f {
					t := float64(10 * float64(i))
					f[i] = y[i] - (x[0] + float64(x[1]*math.Exp(-t*x[3])) + float64(x[2]*math.Exp(-t*x[4])))
				}
			},
			Jac: func(x, j []float64) {
				for i := 0; i < 33; i++ {
					t := float64(10 * float64(i))
					e4, e5 := math.Exp(-t*x[3]), math.Exp(-t*x[4])
					copy(j[5*i:], []float64{-1, -e4, -e5, t * x[1] * e4, t * x[2] * e5})
				}
			}}
	case "biggs":
		return LSProblem{M: 13,
			Func: func(x, f []float64) {
				for i := range f {
					t := float64(0.1 * float64(i+1))
					y := math.Exp(-t) - float64(5*math.Exp(-10*t)) + float64(3*math.Exp(-4*t))
					f[i] = float64(x[2]*math.Exp(-t*x[0])) - float64(x[3]*math.Exp(-t*x[1])) + float64(x[5]*math.Exp(-t*x[4])) - y
				}
			},
			Jac: func(x, j []float64) {
				for i := 0; i < 13; i++ {
					t := float64(0.1 * float64(i+1))
					e1, e2, e5 := math.Exp(-t*x[0]), math.Exp(-t*x[1]), math.Exp(-t*x[4])
					copy(j[6*i:], []float64{-t * x[2] * e1, t * x[3] * e2, e1, -e2, -t * x[5] * e5, e5})
				}
			}}
	case "osborne2":
		y := data["osborne2_y"]
		return LSProblem{M: 65,
			Func: func(x, f []float64) {
				for i := range f {
					t := float64(i) / 10
					d9, d10, d11 := t-x[8], t-x[9], t-x[10]
					f[i] = y[i] - (float64(x[0]*math.Exp(-t*x[4])) + float64(x[1]*math.Exp(-d9*d9*x[5])) + float64(x[2]*math.Exp(-d10*d10*x[6])) + float64(x[3]*math.Exp(-d11*d11*x[7])))
				}
			},
			Jac: func(x, j []float64) {
				for i := 0; i < 65; i++ {
					t := float64(i) / 10
					d9, d10, d11 := t-x[8], t-x[9], t-x[10]
					e1, e2 := math.Exp(-t*x[4]), math.Exp(-d9*d9*x[5])
					e3, e4 := math.Exp(-d10*d10*x[6]), math.Exp(-d11*d11*x[7])
					copy(j[11*i:], []float64{-e1, -e2, -e3, -e4, t * x[0] * e1,
						d9 * d9 * x[1] * e2, d10 * d10 * x[2] * e3, d11 * d11 * x[3] * e4,
						-2 * x[1] * x[5] * d9 * e2, -2 * x[2] * x[6] * d10 * e3, -2 * x[3] * x[7] * d11 * e4})
				}
			}}
	case "watson":
		const n = 6
		return LSProblem{M: 31,
			Func: func(x, f []float64) {
				for i := 0; i < 29; i++ {
					t := float64(i+1) / 29
					s1, tj := 0.0, 1.0
					for j := 1; j < n; j++ {
						s1 += float64(float64(j) * x[j] * tj)
						tj *= t
					}
					s2 := 0.0
					tj = 1
					for j := 0; j < n; j++ {
						s2 += float64(x[j] * tj)
						tj *= t
					}
					f[i] = s1 - float64(s2*s2) - 1
				}
				f[29] = x[0]
				f[30] = x[1] - float64(x[0]*x[0]) - 1
			},
			Jac: func(x, jac []float64) {
				for k := range jac {
					jac[k] = 0
				}
				for i := 0; i < 29; i++ {
					t := float64(i+1) / 29
					s2, tj := 0.0, 1.0
					for j := 0; j < n; j++ {
						s2 += float64(x[j] * tj)
						tj *= t
					}
					tj = 1
					tjm := 0.0
					for j := 0; j < n; j++ {
						jac[i*n+j] = float64(float64(j)*tjm) - float64(2*s2*tj)
						tjm = tj
						tj *= t
					}
				}
				jac[29*n] = 1
				jac[30*n] = float64(-2 * x[0])
				jac[30*n+1] = 1
			}}
	case "expfit", "clean":
		ts, y := data["expfit_t"], data["expfit_y"]
		if name == "clean" {
			y = data["clean_y"]
		}
		return LSProblem{M: len(ts),
			Func: func(x, f []float64) {
				for i, t := range ts {
					f[i] = x[0] + float64(x[1]*math.Exp(x[2]*t)) - y[i]
				}
			},
			Jac: func(x, j []float64) {
				for i, t := range ts {
					e := math.Exp(x[2] * t)
					copy(j[3*i:], []float64{1, e, x[1] * t * e})
				}
			}}
	case "logfit":
		return LSProblem{M: 2,
			Func: func(x, f []float64) { f[0], f[1] = math.Log(x[0])+3, x[1]-3 },
			Jac:  func(x, j []float64) { copy(j, []float64{1 / x[0], 0, 0, 1}) }}
	case "circle":
		return LSProblem{M: 1,
			Func: func(x, f []float64) { f[0] = float64(x[0]*x[0]) + float64(x[1]*x[1]) - 1 },
			Jac:  func(x, j []float64) { j[0], j[1] = float64(2*x[0]), float64(2*x[1]) }}
	case "zerocol":
		return LSProblem{M: 2,
			Func: func(x, f []float64) { f[0], f[1] = float64(x[0]*x[0])+x[1]-3, x[1]-1 },
			Jac:  func(x, j []float64) { copy(j, []float64{2 * x[0], 1, 0, 1}) }}
	case "balanced":
		return LSProblem{M: 2,
			Func: func(x, f []float64) { f[0], f[1] = x[0]-1, 1 },
			Jac:  func(x, j []float64) { j[0], j[1] = 1, 0 }}
	}
	panic("unknown problem " + name)
}

// trfOptions translates a fixture case into LeastSquares arguments.
func trfOptions(c trfCase) *LSOptions {
	o := &LSOptions{}
	if c.Lb != nil {
		o.Bounds = make([]Bound, len(c.Lb))
		for i := range c.Lb {
			o.Bounds[i] = Unbounded()
			if c.Lb[i] != nil {
				o.Bounds[i].Lo = *c.Lb[i]
			}
			if c.Ub[i] != nil {
				o.Bounds[i].Hi = *c.Ub[i]
			}
		}
	}
	switch c.Jac {
	case "3-point":
		o.FDMethod = FD3Point
	}
	for k, v := range c.Options {
		switch k {
		case "loss":
			o.Loss = Loss(v.(string))
		case "f_scale":
			o.FScale = v.(float64)
		case "diff_step":
			o.DiffStep = v.(float64)
		case "max_nfev":
			o.MaxNfev = int(v.(float64))
		case "x_scale":
			if s, ok := v.(string); ok && s == "jac" {
				o.XScaleJac = true
				continue
			}
			for _, e := range v.([]any) {
				o.XScale = append(o.XScale, e.(float64))
			}
		case "ftol", "xtol", "gtol":
			t := -1.0 // None disables the criterion
			if v != nil {
				t = v.(float64)
			}
			switch k {
			case "ftol":
				o.Ftol = t
			case "xtol":
				o.Xtol = t
			default:
				o.Gtol = t
			}
		default:
			panic("unknown option " + k)
		}
	}
	return o
}

func TestTRFAgainstSciPy(t *testing.T) {
	fx := loadTRFFixture(t)
	same := 0
	for _, c := range fx.Cases {
		p := trfProblem(c.Problem, fx.Data)
		if c.Jac != "analytic" {
			p.Jac = nil
		}
		r, err := LeastSquares(p, c.X0, TRF, trfOptions(c))
		if err != nil {
			t.Fatalf("%s: %v", c.Name, err)
		}
		dx := 0.0
		for i := range c.X {
			dx = math.Max(dx, math.Abs(r.X[i]-c.X[i])/(1+math.Abs(c.X[i])))
		}
		dcost := math.Abs(r.Cost-c.Cost) / math.Max(c.Cost, 1e-12)
		exact := r.Nfev == c.Nfev && r.Njev == c.Njev && r.Status == c.Status && dx <= 1e-6
		if exact {
			same++
		}
		t.Logf("%-32s nfev %3d/%3d njev %3d/%3d status %d/%d  |dx| %.1e  dcost %.1e  opt %.2e/%.2e",
			c.Name, r.Nfev, c.Nfev, r.Njev, c.Njev, r.Status, c.Status, dx, dcost, r.Optimality, c.Optimality)
		switch {
		case math.Abs(r.Cost-c.Cost) > float64(1e-6*c.Cost)+1e-12:
			t.Errorf("%s: cost %.17g, SciPy %.17g", c.Name, r.Cost, c.Cost)
		}
		for i := range c.ActiveMask {
			if r.ActiveMask[i] != c.ActiveMask[i] {
				t.Errorf("%s: active mask %v, SciPy %v", c.Name, r.ActiveMask, c.ActiveMask)
				break
			}
		}
		if r.Success != (c.Status > 0) || (r.Status == c.Status && r.Message != c.Message) {
			t.Errorf("%s: status %d %q, SciPy %d %q", c.Name, r.Status, r.Message, c.Status, c.Message)
		}
	}
	t.Logf("SciPy %s: %d/%d cases with identical nfev, njev and status", fx.Scipy, same, len(fx.Cases))
	// The divergent cases start from a rank-deficient Jacobian or a
	// symmetric point, where rounding decides the path (see trf.go).
	if same < len(fx.Cases)*9/10 {
		t.Errorf("only %d/%d cases follow SciPy's path", same, len(fx.Cases))
	}
}

func TestTRFErrors(t *testing.T) {
	nan := math.NaN()
	lin := func(x, f []float64) { f[0], f[1] = x[0]-1, x[1]+x[0] }
	cases := []struct {
		name string
		p    LSProblem
		o    *LSOptions
	}{
		{"residuals", LSProblem{M: 1, Func: func(x, f []float64) { f[0] = nan }}, nil},
		{"jac0", LSProblem{M: 2, Func: lin, Jac: func(x, j []float64) { j[0] = nan }}, nil},
		{"jac-later", LSProblem{M: 2, Func: lin, Jac: func(x, j []float64) {
			copy(j, []float64{1, 0, 1, 1})
			if x[0] != 3 {
				j[3] = math.Inf(1)
			}
		}}, nil},
		{"jac-fd", LSProblem{M: 2, Func: func(x, f []float64) {
			f[0], f[1] = x[0], 0
			if x[0] != 3 {
				f[1] = float64(1e308 * (x[0] - 3) * 1e10)
			}
		}}, nil},
		{"svd", LSProblem{M: 2, Func: lin, Jac: func(x, j []float64) { copy(j, []float64{1e300, 0, 0, 1}) }},
			&LSOptions{XScale: []float64{1e10, 1}}},
	}
	for _, c := range cases {
		_, err := LeastSquares(c.p, []float64{3, 4}, TRF, c.o)
		if !errors.Is(err, ErrInput) {
			t.Errorf("%s: err = %v, want ErrInput", c.name, err)
		}
	}
}
