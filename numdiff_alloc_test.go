package optimize

import "testing"

// A finite-difference gradient or Jacobian with a work buffer allocates nothing
// per call (before: the steps, the one-sided flags and the perturbed point, three
// slices per gradient, which made a finite-difference L-BFGS-B run at n = 100
// allocate 25063 times).
func TestFiniteDifferenceKeepsItsWorkVectors(t *testing.T) {
	x := altStart(100)
	g := make([]float64, len(x))
	f0 := rosen(x)
	for _, m := range []FDMethod{FD2Point, FD3Point} {
		fd := fdStep{method: m, abs: 1e-8, buf: &fdBuf{}}
		fd.gradFD(rosen, x, f0, g) // sizes the buffer
		if a := testing.AllocsPerRun(20, func() { fd.gradFD(rosen, x, f0, g) }); a != 0 {
			t.Errorf("gradFD (method %v) allocates %v times with a buffer, want 0", m, a)
		}
		// Without a buffer: the three slices.
		bare := fdStep{method: m, abs: 1e-8}
		if a := testing.AllocsPerRun(20, func() { bare.gradFD(rosen, x, f0, g) }); a != 3 {
			t.Errorf("gradFD (method %v) allocates %v times without a buffer, want 3", m, a)
		}
	}
	// The buffer changes nothing: the same bits with and without it.
	w := make([]float64, len(x))
	for _, m := range []FDMethod{FD2Point, FD3Point} {
		(fdStep{method: m, abs: 1e-8, buf: &fdBuf{}}).gradFD(rosen, x, f0, g)
		(fdStep{method: m, abs: 1e-8}).gradFD(rosen, x, f0, w)
		for i := range g {
			if g[i] != w[i] {
				t.Fatalf("method %v: gradient[%d] differs with a buffer: %v vs %v", m, i, g[i], w[i])
			}
		}
	}
}

// Nelder-Mead sorts its simplex without allocating per iteration: the total
// allocations of a solve are a constant of the problem size, not of the
// iterations (270 for 85 iterations before, 16 now).
func TestNelderMeadAllocationsDoNotGrowWithIterations(t *testing.T) {
	x0 := altStart(2)
	run := func(maxIter int) float64 {
		return testing.AllocsPerRun(10, func() {
			if _, err := Minimize(Problem{Func: rosen}, x0, NelderMead, &Options{MaxIter: maxIter, Tol: 1e-300}); err != nil {
				t.Fatal(err)
			}
		})
	}
	short, long := run(10), run(100)
	if short != long {
		t.Errorf("allocations grow with the iterations: %v for 10, %v for 100", short, long)
	}
}
