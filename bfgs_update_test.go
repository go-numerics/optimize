package optimize

import (
	"math"
	"math/rand/v2"
	"testing"
)

func rosenN(x, g []float64) float64 {
	clear(g)
	f := 0.0
	for i := 0; i+1 < len(x); i++ {
		a, b := x[i+1]-x[i]*x[i], 1-x[i]
		f += 100*a*a + b*b
		g[i] += -400*x[i]*a - 2*b
		g[i+1] += 200 * a
	}
	return f
}

// TestBFGSRankTwoMatchesDenseUpdate: the O(n²) update is the same matrix as
// SciPy's two dense products to rounding, and stays exactly symmetric.
func TestBFGSRankTwoMatchesDenseUpdate(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	for _, n := range []int{1, 2, 3, 7, 33, 100} {
		H, s, y := make([]float64, n*n), make([]float64, n), make([]float64, n)
		for i := range n {
			for j := range i + 1 {
				v := r.NormFloat64() / float64(n)
				H[i*n+j], H[j*n+i] = v, v
			}
			H[i*n+i] += 1
			s[i], y[i] = r.NormFloat64(), r.NormFloat64()+0.5*s[i]
		}
		rho := 1 / dot(y, s)
		want := make([]float64, n*n)
		a1, a2, tmp := make([]float64, n*n), make([]float64, n*n), make([]float64, n*n)
		for i := range n {
			for j := range n {
				id := 0.0
				if i == j {
					id = 1
				}
				a1[i*n+j] = id - s[i]*y[j]*rho
				a2[i*n+j] = id - y[i]*s[j]*rho
			}
		}
		matmul(tmp, H, a2, n)
		matmul(want, a1, tmp, n)
		for i := range n {
			for j := range n {
				want[i*n+j] += rho * s[i] * s[j]
			}
		}
		got := append([]float64(nil), H...)
		updateHessInvRankTwo(got, s, y, make([]float64, n), rho, n)
		scale := 0.0
		for _, v := range want {
			scale = math.Max(scale, math.Abs(v))
		}
		for i := range n {
			for j := range n {
				if d := math.Abs(got[i*n+j] - want[i*n+j]); d > 1e-10*scale {
					t.Fatalf("n=%d [%d,%d]: %v vs dense %v", n, i, j, got[i*n+j], want[i*n+j])
				}
				if got[i*n+j] != got[j*n+i] {
					t.Fatalf("n=%d: not symmetric at [%d,%d]", n, i, j)
				}
			}
		}
	}
}

// TestBFGSDefaultUpdateAgainstExact: the default (O(n²), FMA) and SciPy's dense
// update find the same minimiser within tolerance with comparable iteration
// counts. The default is validated by tolerance, not bit for bit.
func TestBFGSDefaultUpdateAgainstExact(t *testing.T) {
	for _, n := range []int{2, 5, 10, 30, 100} {
		x0 := make([]float64, n)
		for i := range x0 {
			x0[i] = -1.2
			if i%2 == 1 {
				x0[i] = 1
			}
		}
		p := Problem{FuncGrad: rosenN}
		fast, err := Minimize(p, x0, BFGS, nil)
		if err != nil {
			t.Fatal(err)
		}
		exact, err := Minimize(p, x0, BFGS, &Options{SciPyExactUpdate: true})
		if err != nil {
			t.Fatal(err)
		}
		if !fast.Success || !exact.Success {
			t.Fatalf("n=%d: success fast %v exact %v (%q)", n, fast.Success, exact.Success, fast.Message)
		}
		for i := range fast.X {
			// the same minimiser (Rosenbrock with n >= 4 has a second, local one)
			if math.Abs(fast.X[i]-exact.X[i]) > 1e-3 {
				t.Fatalf("n=%d: x[%d] fast %v exact %v", n, i, fast.X[i], exact.X[i])
			}
		}
		if math.Abs(fast.Fun-exact.Fun) > 1e-8*math.Max(1, exact.Fun) {
			t.Fatalf("n=%d: f fast %g exact %g", n, fast.Fun, exact.Fun)
		}
		if float64(fast.Nit) > 1.25*float64(exact.Nit)+5 || float64(exact.Nit) > 1.25*float64(fast.Nit)+5 {
			t.Errorf("n=%d: iterations fast %d, exact %d", n, fast.Nit, exact.Nit)
		}
		t.Logf("n=%d nit fast %d exact %d; nfev %d %d; f %.2e %.2e", n, fast.Nit, exact.Nit, fast.Nfev, exact.Nfev, fast.Fun, exact.Fun)
		for i := range fast.HessInv {
			if i/n < i%n && fast.HessInv[i] != fast.HessInv[(i%n)*n+i/n] {
				t.Fatalf("n=%d: HessInv not symmetric", n)
			}
		}
	}
}

// TestBFGSAsymmetricHessInv0TakesDenseUpdate: the rank-two form needs a
// symmetric H, so an asymmetric HessInv0 runs SciPy's dense update (the bits
// are the exact ones).
func TestBFGSAsymmetricHessInv0TakesDenseUpdate(t *testing.T) {
	h0 := []float64{1, 0.2, 0, 1}
	p := Problem{FuncGrad: rosenN}
	x0 := []float64{-1.2, 1}
	a, err := Minimize(p, x0, BFGS, &Options{HessInv0: h0})
	if err != nil {
		t.Fatal(err)
	}
	b, err := Minimize(p, x0, BFGS, &Options{HessInv0: h0, SciPyExactUpdate: true})
	if err != nil {
		t.Fatal(err)
	}
	if a.Nit != b.Nit || a.Nfev != b.Nfev || math.Float64bits(a.Fun) != math.Float64bits(b.Fun) {
		t.Errorf("asymmetric HessInv0: default nit/nfev/f %d/%d/%g, exact %d/%d/%g", a.Nit, a.Nfev, a.Fun, b.Nit, b.Nfev, b.Fun)
	}
}
