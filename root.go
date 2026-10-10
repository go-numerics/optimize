package optimize

import (
	"fmt"

	"github.com/go-ndarray/ndarray"
)

// SystemProblem is a square system of nonlinear equations F(x) = 0 with n
// equations in n unknowns.
type SystemProblem struct {
	// Func writes the n values of F at x into f. Required. It must not
	// modify x.
	Func func(x, f []float64)
	// Jac writes the n×n Jacobian of F at x into jac, row-major
	// (jac[i*n+j] = ∂F_i/∂x_j). Optional: MINPACK's forward differences are
	// used when it is nil.
	Jac func(x, jac []float64)
}

// RootOptions configures Root and Fsolve. Zero values select SciPy's
// defaults.
type RootOptions struct {
	// Xtol stops the iteration when the relative error between two
	// consecutive iterates is at most Xtol (1.49012e-8; a negative value
	// means exactly 0).
	Xtol float64
	// Maxfev bounds the calls to Func: 0 selects 200·(n+1) without a
	// Jacobian (forward-difference calls included) and 100·(n+1) with one.
	Maxfev int
	// Eps sets the forward-difference step sqrt(Eps)·|x_j| when there is no
	// Jacobian (SciPy's eps for root, epsfcn for fsolve); 0 selects the
	// machine epsilon.
	Eps float64
	// Factor sets the initial step bound Factor·|Diag·x0| (100).
	Factor float64
	// Diag holds n positive scale factors for the variables; nil lets
	// MINPACK scale by the Jacobian column norms.
	Diag []float64
}

// RootResult is the outcome of Root and Fsolve, mirroring the
// OptimizeResult of scipy.optimize.root(method='hybr').
type RootResult struct {
	// X is the solution (or the last iterate).
	X []float64
	// Fun holds F(X).
	Fun []float64
	// Fjac is SciPy's fjac: the orthogonal factor Q of the QR factorisation
	// of the final approximate Jacobian, stored column-wise, so that as an
	// n×n row-major array it reads Qᵀ (row i is column i of Q).
	Fjac []float64
	// R is the upper-triangular factor packed by rows (n(n+1)/2 entries).
	R []float64
	// Qtf is Qᵀ·Fun.
	Qtf []float64
	// Nfev counts calls to Func, including those of finite differences;
	// Njev counts calls to Jac (0 without one). SciPy's root and fsolve
	// report two more calls: those their wrappers make outside MINPACK (a
	// shape check and the initial evaluation), which do not happen here.
	Nfev, Njev int
	// Status is MINPACK's info: 1 converged, 2 Maxfev reached, 3 Xtol too
	// small, 4 and 5 no progress.
	Status int
	// Success is Status == 1.
	Success bool
	// Message is SciPy's message for Status.
	Message string
}

// XArray returns X as a 1-D ndarray.
func (r *RootResult) XArray() *ndarray.Array { return vecArray(r.X) }

// FjacArray returns Fjac as an n×n ndarray.
func (r *RootResult) FjacArray() *ndarray.Array { return matArray(r.Fjac, len(r.X), len(r.X)) }

// Root solves the square system p.Func(x) = 0 from x0 with Powell's hybrid
// method of MINPACK; it is scipy.optimize.root(fun, x0, jac=p.Jac,
// method='hybr', options={'xtol', 'maxfev', 'eps', 'factor', 'diag'})
// (the 'band' and 'col_deriv' options are not supported). Without p.Jac it
// runs hybrd, otherwise hybrj. Non-convergence is reported through Status
// and Message.
func Root(p SystemProblem, x0 []float64, opts *RootOptions) (*RootResult, error) {
	n := len(x0)
	if p.Func == nil || n == 0 {
		return nil, fmt.Errorf("%w: SystemProblem needs Func and a non-empty x0", ErrInput)
	}
	var o RootOptions
	if opts != nil {
		o = *opts
	}
	c := &hybrdConfig{xtol: tolDefault(o.Xtol), maxfev: o.Maxfev, epsfcn: o.Eps, diag: o.Diag, factor: o.Factor}
	if c.maxfev == 0 {
		c.maxfev = 200 * (n + 1)
		if p.Jac != nil {
			c.maxfev = 100 * (n + 1)
		}
	}
	if c.factor == 0 {
		c.factor = 100
	}
	if err := checkMinpack(c.maxfev, c.factor, o.Diag, n); err != nil {
		return nil, err
	}
	var jac func(x, fvec, fjac []float64)
	if p.Jac != nil {
		jac = rowToCol(func(x, _, j []float64) { p.Jac(x, j) }, n, n)
	}
	x := clone(x0)
	out := hybrd(p.Func, jac, x, c)
	return &RootResult{
		X: x, Fun: out.fvec, Fjac: out.fjac, R: out.r, Qtf: out.qtf,
		Nfev: out.nfev, Njev: out.njev,
		Status: out.info, Success: out.info == 1, Message: hybrdMessage(out.info, c),
	}, nil
}

// Fsolve is scipy.optimize.fsolve(func, x0, fprime=p.Jac, full_output=True,
// xtol, maxfev, epsfcn=opts.Eps, factor, diag): the same computation as
// Root, returning the same result (SciPy's infodict holds Fun as fvec, Nfev,
// Njev, Fjac, R and Qtf; ier is Status and mesg Message).
func Fsolve(p SystemProblem, x0 []float64, opts *RootOptions) (*RootResult, error) {
	return Root(p, x0, opts)
}

// hybrdMessage is SciPy's message for a hybrd/hybrj info code (Python's
// {:f} is Go's %f).
func hybrdMessage(info int, c *hybrdConfig) string {
	switch info {
	case 1:
		return "The solution converged."
	case 2:
		return fmt.Sprintf("The number of calls to function has reached maxfev = %d.", c.maxfev)
	case 3:
		return fmt.Sprintf("xtol=%f is too small, no further improvement in the approximate\n solution is possible.", c.xtol)
	case 4:
		return "The iteration is not making good progress, as measured by the \n improvement from the last five Jacobian evaluations."
	}
	return "The iteration is not making good progress, as measured by the \n improvement from the last ten iterations."
}
