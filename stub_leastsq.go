package optimize

// TEMPORARY until the MINPACK branch is merged.
type LeastsqOptions struct {
	Ftol, Xtol, Gtol float64
	MaxFev           int
}
type LeastsqResult struct {
	X, Fvec, CovX []float64
	Ier, Nfev     int
	Mesg          string
}

func Leastsq(f func(x, fvec []float64), jac func(x, J []float64), m int, x0 []float64, o *LeastsqOptions) (*LeastsqResult, error) {
	panic("not implemented")
}
