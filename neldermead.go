package optimize

import (
	"fmt"
	"math"
	"sort"
)

// maxFunReached unwinds an iteration when the evaluation budget is spent,
// mirroring SciPy's _MaxFuncCallError: the evaluation that would exceed the
// budget is not performed.
type maxFunReached struct{}

// budget counts direct objective evaluations against a limit (no cache:
// Nelder–Mead and Powell count every call, as SciPy does).
type budget struct {
	sf     *scalarFunc
	maxfun int
	ncalls int
	buf    []float64
}

func (b *budget) f(x []float64) float64 {
	if b.ncalls >= b.maxfun {
		panic(maxFunReached{})
	}
	b.ncalls++
	b.sf.nfev++
	if b.sf.p.Func != nil {
		return b.sf.p.Func(x)
	}
	if len(b.buf) != len(x) {
		b.buf = make([]float64, len(x))
	}
	return b.sf.p.FuncGrad(x, b.buf)
}

// guard runs step and reports whether it was cut short by the budget.
func guard(step func()) (cut bool) {
	defer func() {
		if r := recover(); r != nil {
			if _, ok := r.(maxFunReached); !ok {
				panic(r)
			}
			cut = true
		}
	}()
	step()
	return false
}

// iterLimits resolves maxiter/maxfev the way SciPy's Nelder–Mead and Powell
// do: both default to perN·n, and setting only one leaves the other
// unlimited.
func iterLimits(o *Options, n, perN int) (maxiter, maxfun int) {
	switch {
	case o.MaxIter == 0 && o.MaxFev == 0:
		return perN * n, perN * n
	case o.MaxIter == 0:
		return math.MaxInt, o.MaxFev
	case o.MaxFev == 0:
		return o.MaxIter, math.MaxInt
	}
	return o.MaxIter, o.MaxFev
}

func clip(x, lb, ub []float64) {
	if lb == nil {
		return
	}
	for i := range x {
		x[i] = math.Min(math.Max(x[i], lb[i]), ub[i])
	}
}

// minimizeNelderMead is the downhill simplex method of Nelder and Mead (1965)
// with the optional dimension-adaptive coefficients of Gao and Han (2012) and
// bound handling by clipping, following SciPy's _minimize_neldermead.
func minimizeNelderMead(sf *scalarFunc, x0 []float64, o *Options, lb, ub []float64) (*Result, error) {
	n := len(x0)
	rho, chi, psi, sigma := 1.0, 2.0, 0.5, 0.5
	if o.Adaptive {
		dim := float64(n)
		chi = 1 + 2/dim
		psi = 0.75 - 1/(2*dim)
		sigma = 1 - 1/dim
	}
	xatol, fatol := o.Xatol, o.Fatol
	if xatol == 0 {
		xatol = o.Tol
	}
	if fatol == 0 {
		fatol = o.Tol
	}
	if xatol == 0 {
		xatol = 1e-4
	}
	if fatol == 0 {
		fatol = 1e-4
	}
	const nonzdelt, zdelt = 0.05, 0.00025

	sim := make([][]float64, n+1)
	if o.InitialSimplex == nil {
		sim[0] = clone(x0)
		for k := range n {
			y := clone(x0)
			if y[k] != 0 {
				y[k] = float64((1 + nonzdelt) * y[k])
			} else {
				y[k] = zdelt
			}
			sim[k+1] = y
		}
	} else {
		if len(o.InitialSimplex) != n+1 {
			return nil, fmt.Errorf("%w: InitialSimplex must hold n+1 points", ErrDimension)
		}
		for i, p := range o.InitialSimplex {
			if len(p) != n {
				return nil, fmt.Errorf("%w: InitialSimplex points must have length n", ErrDimension)
			}
			sim[i] = clone(p)
		}
	}
	maxiter, maxfun := iterLimits(o, n, 200)
	if lb != nil {
		// Reflect points above the upper bound back inside, then clip.
		for _, p := range sim {
			for j := range p {
				if p[j] > ub[j] {
					p[j] = float64(2*ub[j]) - p[j]
				}
			}
			clip(p, lb, ub)
		}
	}
	bud := &budget{sf: sf, maxfun: maxfun}
	fsim := make([]float64, n+1)
	for i := range fsim {
		fsim[i] = math.Inf(1)
	}
	idx := make([]int, n+1)
	order := func() {
		for i := range idx {
			idx[i] = i
		}
		sort.SliceStable(idx, func(a, b int) bool { return fsim[idx[a]] < fsim[idx[b]] })
		ns := make([][]float64, n+1)
		nf := make([]float64, n+1)
		for i, j := range idx {
			ns[i], nf[i] = sim[j], fsim[j]
		}
		copy(sim, ns)
		copy(fsim, nf)
	}
	guard(func() {
		for k := range sim {
			fsim[k] = bud.f(sim[k])
		}
	})
	order()

	xbar := make([]float64, n)
	xr := make([]float64, n)
	xe := make([]float64, n)
	xc := make([]float64, n)
	comb := func(dst []float64, a float64, b float64) {
		// dst = a·xbar − b·sim[n]
		last := sim[n]
		for j := range dst {
			dst[j] = float64(a*xbar[j]) - float64(b*last[j])
		}
		clip(dst, lb, ub)
	}
	iterations := 1
	for bud.ncalls < maxfun && iterations < maxiter {
		conv := true
		for _, p := range sim[1:] {
			for j := range p {
				if !(math.Abs(p[j]-sim[0][j]) <= xatol) {
					conv = false
				}
			}
		}
		if conv {
			for _, f := range fsim[1:] {
				if !(math.Abs(fsim[0]-f) <= fatol) {
					conv = false
				}
			}
		}
		if conv {
			break
		}
		guard(func() {
			for j := range xbar {
				s := 0.0
				for _, p := range sim[:n] {
					s += p[j]
				}
				xbar[j] = s / float64(n)
			}
			comb(xr, 1+rho, rho)
			fxr := bud.f(xr)
			shrink := false
			switch {
			case fxr < fsim[0]:
				comb(xe, 1+float64(rho*chi), rho*chi)
				fxe := bud.f(xe)
				if fxe < fxr {
					copy(sim[n], xe)
					fsim[n] = fxe
				} else {
					copy(sim[n], xr)
					fsim[n] = fxr
				}
			case fxr < fsim[n-1]:
				copy(sim[n], xr)
				fsim[n] = fxr
			case fxr < fsim[n]:
				comb(xc, 1+float64(psi*rho), psi*rho)
				fxc := bud.f(xc)
				if fxc <= fxr {
					copy(sim[n], xc)
					fsim[n] = fxc
				} else {
					shrink = true
				}
			default:
				// Inside contraction: (1−ψ)·xbar + ψ·x_worst.
				comb(xc, 1-psi, -psi)
				fxcc := bud.f(xc)
				if fxcc < fsim[n] {
					copy(sim[n], xc)
					fsim[n] = fxcc
				} else {
					shrink = true
				}
			}
			if shrink {
				for j := 1; j <= n; j++ {
					p := sim[j]
					for k := range p {
						p[k] = sim[0][k] + float64(sigma*(p[k]-sim[0][k]))
					}
					clip(p, lb, ub)
					fsim[j] = bud.f(p)
				}
			}
			iterations++
		})
		order()
		if o.Callback != nil && o.Callback(clone(sim[0]), fsim[0]) {
			break
		}
	}
	fval := fsim[0]
	for _, f := range fsim {
		fval = math.Min(fval, f)
	}
	r := &Result{X: clone(sim[0]), Fun: fval, Nit: iterations}
	switch {
	case bud.ncalls >= maxfun:
		r.Status, r.Message = 1, msgMaxFev
	case iterations >= maxiter:
		r.Status, r.Message = 2, msgMaxIter
	default:
		r.Message = msgSuccess
	}
	r.Success = r.Status == 0
	return sf.result(r), nil
}
