// Package optimize provides numerical optimisation and root finding in pure
// Go: unconstrained and bound-constrained minimisation (Nelder–Mead, Powell,
// CG, BFGS, Newton-CG, L-BFGS-B), scalar minimisation (Brent, golden section,
// bounded), scalar roots (Brent, Ridder, bisection, Newton, secant, Halley),
// systems of equations (MINPACK's hybrid Powell method), nonlinear least
// squares (MINPACK Levenberg–Marquardt and trust-region reflective with
// bounds and robust losses), curve fitting with parameter covariance, and
// linear programming.
//
// The API follows scipy.optimize closely enough that each function documents
// the SciPy call it corresponds to, and the test suite checks the results
// against SciPy fixtures; the implementation is independent Go.
package optimize
