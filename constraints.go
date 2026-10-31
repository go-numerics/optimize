package optimize

import (
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
)

// ConstraintSpec is a general constraint accepted by the constrained
// minimisers (Options.Constraints): a Constraint, a NonlinearConstraint or a
// LinearConstraint, the three forms scipy.optimize.minimize accepts. The
// interface is sealed. SLSQP reduces every form to equality blocks c(x) = 0
// and inequality blocks c(x) >= 0; trust-constr keeps SciPy's general form
// lb <= f(x) <= ub.
type ConstraintSpec interface {
	standardize(n int) ([]stdConstraint, error)
	general(n int) (genConstraint, error)
}

// genConstraint is a constraint in SciPy's general form lb <= f(x) <= ub
// (NonlinearConstraint), the form trust-constr works with. jac nil means
// finite differences; linear constraints carry their m×n matrix in a.
type genConstraint struct {
	m      int
	fun    func(x, c []float64)
	jac    func(x, j []float64)
	lb, ub []float64
	linear bool
	a      []float64
}

func (c Constraint) general(n int) (genConstraint, error) {
	if _, err := c.standardize(n); err != nil {
		return genConstraint{}, err
	}
	ub := 0.0
	if c.Kind == Ineq {
		ub = math.Inf(1)
	}
	lb, _ := boundVector([]float64{0}, c.M, 0)
	ubv, _ := boundVector([]float64{ub}, c.M, 0)
	return genConstraint{m: c.M, fun: c.Func, jac: c.Jac, lb: lb, ub: ubv}, nil
}

func (c NonlinearConstraint) general(n int) (genConstraint, error) {
	if _, err := c.standardize(n); err != nil {
		return genConstraint{}, err
	}
	lb, _ := boundVector(c.Lb, c.M, math.Inf(-1))
	ub, _ := boundVector(c.Ub, c.M, math.Inf(1))
	return genConstraint{m: c.M, fun: c.Func, jac: c.Jac, lb: lb, ub: ub}, nil
}

func (c LinearConstraint) general(n int) (genConstraint, error) {
	if _, err := c.standardize(n); err != nil {
		return genConstraint{}, err
	}
	a := arrayData(c.A)
	m := len(a) / n
	lb, _ := boundVector(c.Lb, m, math.Inf(-1))
	ub, _ := boundVector(c.Ub, m, math.Inf(1))
	return genConstraint{m: m, lb: lb, ub: ub, linear: true, a: a,
		fun: func(x, out []float64) { matvec(m, n, a, x, out) }}, nil
}

// generalConstraints converts specs to the general form.
func generalConstraints(specs []ConstraintSpec, n int) ([]genConstraint, error) {
	var out []genConstraint
	for i, s := range specs {
		if s == nil {
			return nil, fmt.Errorf("%w: constraint %d is nil", ErrInput, i)
		}
		g, err := s.general(n)
		if err != nil {
			return nil, err
		}
		out = append(out, g)
	}
	return out, nil
}

// ConstraintKind tells whether a Constraint is an equality or an inequality.
type ConstraintKind int

const (
	// Eq is the equality constraint c(x) = 0 (SciPy type 'eq').
	Eq ConstraintKind = iota
	// Ineq is the inequality constraint c(x) >= 0 (SciPy type 'ineq').
	Ineq
)

// Constraint is a block of M constraint functions of one kind, SciPy's
// dictionary form {'type': 'eq'|'ineq', 'fun': ..., 'jac': ...}.
type Constraint struct {
	// Kind selects c(x) = 0 (Eq) or c(x) >= 0 (Ineq).
	Kind ConstraintKind
	// M is the number of outputs of Func.
	M int
	// Func writes the M constraint values at x into c.
	Func func(x, c []float64)
	// Jac writes the M×n Jacobian at x into jac, row-major. When nil it is
	// estimated by forward differences.
	Jac func(x, jac []float64)
}

// NonlinearConstraint is Lb <= Func(x) <= Ub, SciPy's NonlinearConstraint.
// Lb and Ub have M entries or one entry broadcast to all; nil means -Inf and
// +Inf. Components with Lb == Ub are equalities; components unbounded on
// both sides are ignored, as SciPy does.
type NonlinearConstraint struct {
	// M is the number of outputs of Func.
	M int
	// Func writes the M constraint values at x into c.
	Func func(x, c []float64)
	// Jac writes the M×n Jacobian, row-major; nil means finite differences.
	Jac func(x, jac []float64)
	// Lb and Ub bound Func(x).
	Lb, Ub []float64
}

// LinearConstraint is Lb <= A·x <= Ub, SciPy's LinearConstraint. A is 2-D
// with n columns (or 1-D, one row); Lb and Ub follow NonlinearConstraint's
// rules.
type LinearConstraint struct {
	A      *ndarray.Array
	Lb, Ub []float64
}

// stdConstraint is a standardised block: m functions, all equalities (eq) or
// all inequalities >= 0. jac nil means finite differences.
type stdConstraint struct {
	eq  bool
	m   int
	fun func(x, c []float64)
	jac func(x, jac []float64)
}

func (c Constraint) standardize(n int) ([]stdConstraint, error) {
	if c.Func == nil {
		return nil, fmt.Errorf("%w: Constraint.Func is required", ErrInput)
	}
	if c.M <= 0 {
		return nil, fmt.Errorf("%w: Constraint.M must be positive", ErrInput)
	}
	if c.Kind != Eq && c.Kind != Ineq {
		return nil, fmt.Errorf("%w: unknown constraint kind %d", ErrInput, c.Kind)
	}
	return []stdConstraint{{eq: c.Kind == Eq, m: c.M, fun: c.Func, jac: c.Jac}}, nil
}

func (c NonlinearConstraint) standardize(n int) ([]stdConstraint, error) {
	if c.Func == nil {
		return nil, fmt.Errorf("%w: NonlinearConstraint.Func is required", ErrInput)
	}
	if c.M <= 0 {
		return nil, fmt.Errorf("%w: NonlinearConstraint.M must be positive", ErrInput)
	}
	return splitByBounds(n, c.M, c.Func, c.Jac, c.Lb, c.Ub)
}

func (c LinearConstraint) standardize(n int) ([]stdConstraint, error) {
	if c.A == nil {
		return nil, fmt.Errorf("%w: LinearConstraint.A is required", ErrInput)
	}
	sh := c.A.Shape()
	var m int
	switch len(sh) {
	case 1:
		m = 1
		if sh[0] != n {
			return nil, ErrDimension
		}
	case 2:
		m = sh[0]
		if sh[1] != n {
			return nil, ErrDimension
		}
	default:
		return nil, fmt.Errorf("%w: LinearConstraint.A must be 1-D or 2-D", ErrInput)
	}
	if m == 0 {
		return nil, fmt.Errorf("%w: LinearConstraint.A has no rows", ErrInput)
	}
	a := arrayData(c.A)
	fun := func(x, out []float64) {
		for i := 0; i < m; i++ {
			out[i] = dot(a[i*n:(i+1)*n], x)
		}
	}
	jac := func(_, j []float64) { copy(j, a) }
	return splitByBounds(n, m, fun, jac, c.Lb, c.Ub)
}

// boundVector expands a bound of length 0 (fill), 1 (broadcast) or m.
func boundVector(b []float64, m int, fill float64) ([]float64, error) {
	out := make([]float64, m)
	switch len(b) {
	case 0:
		for i := range out {
			out[i] = fill
		}
	case 1:
		for i := range out {
			out[i] = b[0]
		}
	case m:
		copy(out, b)
	default:
		return nil, ErrDimension
	}
	return out, nil
}

// splitByBounds turns lb <= f(x) <= ub into SciPy's old-style blocks
// (new_constraint_to_old): one equality block f(x)[eq] - lb[eq], then one
// inequality block with f(x)[below] - lb[below] followed by
// ub[above] - f(x)[above].
func splitByBounds(n, m int, f func(x, c []float64), jac func(x, j []float64),
	lbIn, ubIn []float64) ([]stdConstraint, error) {
	lb, err := boundVector(lbIn, m, math.Inf(-1))
	if err != nil {
		return nil, err
	}
	ub, err := boundVector(ubIn, m, math.Inf(1))
	if err != nil {
		return nil, err
	}
	var ieq, below, above []int
	for i := 0; i < m; i++ {
		if math.IsNaN(lb[i]) || math.IsNaN(ub[i]) || lb[i] > ub[i] {
			return nil, fmt.Errorf("%w: constraint bounds must satisfy lb <= ub", ErrBounds)
		}
		eq := lb[i] == ub[i]
		if eq {
			ieq = append(ieq, i)
			continue
		}
		if !math.IsInf(lb[i], -1) {
			below = append(below, i)
		}
		if !math.IsInf(ub[i], 1) {
			above = append(above, i)
		}
	}
	y := make([]float64, m)
	var dy []float64
	if jac != nil {
		dy = make([]float64, m*n)
	}
	var out []stdConstraint
	if len(ieq) > 0 {
		c := stdConstraint{eq: true, m: len(ieq), fun: func(x, c []float64) {
			f(x, y)
			for k, i := range ieq {
				c[k] = y[i] - lb[i]
			}
		}}
		if jac != nil {
			c.jac = func(x, j []float64) {
				jac(x, dy)
				for k, i := range ieq {
					copy(j[k*n:(k+1)*n], dy[i*n:(i+1)*n])
				}
			}
		}
		out = append(out, c)
	}
	nb := len(below)
	if nb+len(above) > 0 {
		c := stdConstraint{m: nb + len(above), fun: func(x, c []float64) {
			f(x, y)
			for k, i := range below {
				c[k] = y[i] - lb[i]
			}
			for k, i := range above {
				c[nb+k] = -(y[i] - ub[i])
			}
		}}
		if jac != nil {
			c.jac = func(x, j []float64) {
				jac(x, dy)
				for k, i := range below {
					copy(j[k*n:(k+1)*n], dy[i*n:(i+1)*n])
				}
				for k, i := range above {
					row := j[(nb+k)*n : (nb+k+1)*n]
					for l := range row {
						row[l] = -dy[i*n+l]
					}
				}
			}
		}
		out = append(out, c)
	}
	return out, nil
}

// standardizeConstraints reduces specs to blocks for n variables.
func standardizeConstraints(specs []ConstraintSpec, n int) ([]stdConstraint, error) {
	var out []stdConstraint
	for i, s := range specs {
		if s == nil {
			return nil, fmt.Errorf("%w: constraint %d is nil", ErrInput, i)
		}
		c, err := s.standardize(n)
		if err != nil {
			return nil, err
		}
		out = append(out, c...)
	}
	return out, nil
}
