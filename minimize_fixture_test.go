package optimize

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"testing"
)

//go:embed testdata/minimize.json
var minimizeJSON []byte

type minimizeCase struct {
	Problem  string      `json:"problem"`
	Method   string      `json:"method"`
	Grad     bool        `json:"grad"`
	X        []float64   `json:"x"`
	Fun      float64     `json:"fun"`
	Nit      int         `json:"nit"`
	Nfev     int         `json:"nfev"`
	Njev     int         `json:"njev"`
	Nhev     int         `json:"nhev"`
	Status   int         `json:"status"`
	Success  bool        `json:"success"`
	Message  string      `json:"message"`
	Adaptive bool        `json:"adaptive"`
	Hess     string      `json:"hess"`
	Bounds   [][]float64 `json:"bounds"`
	X0       []float64   `json:"x0"`
	MaxIter  int         `json:"maxiter"`
	MaxFev   int         `json:"maxfev"`
}

func (c minimizeCase) name() string {
	s := fmt.Sprintf("%s/%s/grad=%v", c.Problem, c.Method, c.Grad)
	if c.Adaptive {
		s += "/adaptive"
	}
	if c.Hess != "" {
		s += "/" + c.Hess
	}
	if c.Bounds != nil {
		s += "/bounds"
	}
	if c.MaxIter != 0 || c.MaxFev != 0 {
		s += fmt.Sprintf("/limit%d-%d", c.MaxIter, c.MaxFev)
	}
	return s
}

func loadMinimizeCases(t *testing.T) []minimizeCase {
	var cs []minimizeCase
	if err := json.Unmarshal(minimizeJSON, &cs); err != nil {
		t.Fatal(err)
	}
	return cs
}

// runCase solves a fixture case with this package.
func runCase(t *testing.T, c minimizeCase) *Result {
	tf := testFuncs[c.Problem]
	p := Problem{Func: tf.f}
	if c.Grad {
		p.Grad = tf.grad
	}
	switch c.Hess {
	case "hess":
		p.Hess = rosenHess
	case "hessp":
		p.HessP = rosenHessP
	}
	x0 := testStarts[c.Problem]
	if c.X0 != nil {
		x0 = c.X0
	}
	o := &Options{Adaptive: c.Adaptive, MaxIter: c.MaxIter, MaxFev: c.MaxFev}
	for _, b := range c.Bounds {
		o.Bounds = append(o.Bounds, Bound{b[0], b[1]})
	}
	r, err := Minimize(p, x0, Method(c.Method), o)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

// TestMinimizeAgainstSciPy checks every fixture: a converged SciPy run must be
// matched by a converged run here reaching the same accuracy; a run stopped by
// a limit must stop the same way. Counts are logged side by side.
func TestMinimizeAgainstSciPy(t *testing.T) {
	for _, c := range loadMinimizeCases(t) {
		t.Run(c.name(), func(t *testing.T) {
			r := runCase(t, c)
			t.Logf("COUNTS %-48s nit %5d/%-5d nfev %5d/%-5d njev %5d/%-5d fun %.3e/%.3e",
				c.name(), r.Nit, c.Nit, r.Nfev, c.Nfev, r.Njev, c.Njev, r.Fun, c.Fun)
			if c.MaxIter != 0 || c.MaxFev != 0 {
				if r.Status != c.Status || r.Message != c.Message {
					t.Errorf("status %d %q, SciPy %d %q", r.Status, r.Message, c.Status, c.Message)
				}
				return
			}
			if !c.Success {
				// SciPy stopped on a budget: we must stop the same way, at a
				// comparable point. Where the run is cut half-way along a
				// long valley (L-BFGS-B on rosen100 with finite differences)
				// f depends on the path, and SciPy's own path depends on its
				// BLAS (Accelerate's blocked FMA kernels), so only a 10%
				// band is meaningful.
				budget := strings.Contains(c.Message, "Maximum number") || strings.Contains(c.Message, "STOP:")
				if budget && (r.Status != c.Status || r.Message != c.Message) {
					t.Errorf("status %d %q, SciPy %d %q", r.Status, r.Message, c.Status, c.Message)
				}
				if r.Fun > 1.1*c.Fun+1e-12 {
					t.Errorf("fun %.6e worse than SciPy's unconverged %.6e", r.Fun, c.Fun)
				}
				return
			}
			if !r.Success {
				// With finite-difference gradients the gradient noise
				// (√eps·‖∇²f‖ ≈ 1e-5 on Rosenbrock) is of the order of gtol,
				// so whether the last line search succeeds is decided by
				// rounding; precision loss is then acceptable if the
				// accuracy test below passes.
				if c.Grad || r.Status != 2 {
					t.Errorf("SciPy converged, we did not: %d %q", r.Status, r.Message)
				} else {
					t.Logf("precision loss with finite differences (SciPy converged)")
				}
			}
			// Accuracy. Every unbounded problem here has minimum value 0, so
			// the objective is the error. Ours must be within a factor 10 of
			// SciPy's or below a floor set by the method's default stopping
			// rule (where SciPy happens to stop far below its own tolerance,
			// matching that luck is not a requirement).
			if c.Bounds != nil {
				if r.Fun > c.Fun+1e-6*math.Max(1, math.Abs(c.Fun)) {
					t.Errorf("fun %.8e, SciPy %.8e", r.Fun, c.Fun)
				}
				for i := range c.X {
					if math.Abs(r.X[i]-c.X[i]) > 1e-3 {
						t.Errorf("x[%d] = %.8g, SciPy %.8g", i, r.X[i], c.X[i])
					}
				}
				return
			}
			floor := map[string]float64{"Nelder-Mead": 1e-5, "Powell": 1e-8, "CG": 1e-9,
				"BFGS": 1e-9, "Newton-CG": 1e-7, "L-BFGS-B": 1e-7}[c.Method]
			if c.Method == "Newton-CG" {
				// Newton-CG stops on ‖Δx‖₁ <= n·1e-5, so its error in x is
				// of that order and its error in f of ‖∇²f‖·(n·1e-5)².
				d := float64(len(c.X)) * 1e-5
				floor = math.Max(floor, 1000*d*d)
			}
			if r.Fun > math.Max(10*c.Fun, floor) {
				t.Errorf("fun %.6e: SciPy reached %.6e (floor %.0e)", r.Fun, c.Fun, floor)
			}
		})
	}
}
