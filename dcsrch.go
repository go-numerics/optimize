package optimize

import "math"

// lsTask is the reverse-communication state of the Moré–Thuente line search.
type lsTask int

const (
	lsStart lsTask = iota // initial entry
	lsFG                  // evaluate f and f' at the returned step
	lsConv                // sufficient decrease and curvature both hold
	lsWarn                // no progress possible; the step is the best found
	lsError               // invalid input
)

// dcsrch is the line search of Moré and Thuente (ACM TOMS 20, 1994) in the
// reverse-communication form of MINPACK-2's dcsrch, the search used by
// SciPy's BFGS and CG (scalar_search_wolfe1) and by L-BFGS-B. It finds a step
// satisfying the strong Wolfe conditions
//
//	f(stp) <= f(0) + ftol·stp·f'(0),   |f'(stp)| <= gtol·|f'(0)|.
//
// Usage: set the tolerances and bounds, call iterate with task lsStart, f(0)
// and f'(0); while it returns lsFG evaluate f and f' at the returned step and
// call again with them.
type dcsrch struct {
	ftol, gtol, xtol, stpmin, stpmax float64

	msg                                   string
	brackt                                bool
	stage                                 int
	ginit, gtest, gx, gy, finit, fx, fy   float64
	stx, sty, stmin, stmax, width, width1 float64
}

func (d *dcsrch) iterate(stp, f, g float64, task lsTask) (float64, lsTask) {
	const (
		p5     = 0.5
		p66    = 0.66
		xtrapl = 1.1
		xtrapu = 4.0
	)
	if task == lsStart {
		d.msg = ""
		switch {
		case d.stpmax < d.stpmin:
			d.msg = "ERROR: STPMAX .LT. STPMIN"
		case d.stpmin < 0:
			d.msg = "ERROR: STPMIN .LT. ZERO"
		case d.xtol < 0:
			d.msg = "ERROR: XTOL .LT. ZERO"
		case d.gtol < 0:
			d.msg = "ERROR: GTOL .LT. ZERO"
		case d.ftol < 0:
			d.msg = "ERROR: FTOL .LT. ZERO"
		case g >= 0:
			d.msg = "ERROR: INITIAL G .GE. ZERO"
		case stp > d.stpmax:
			d.msg = "ERROR: STP .GT. STPMAX"
		case stp < d.stpmin:
			d.msg = "ERROR: STP .LT. STPMIN"
		}
		if d.msg != "" {
			return stp, lsError
		}
		d.brackt = false
		d.stage = 1
		d.finit, d.ginit = f, g
		d.gtest = float64(d.ftol * d.ginit)
		d.width = d.stpmax - d.stpmin
		d.width1 = d.width / p5
		d.stx, d.fx, d.gx = 0, d.finit, d.ginit
		d.sty, d.fy, d.gy = 0, d.finit, d.ginit
		d.stmin = 0
		d.stmax = stp + float64(xtrapu*stp)
		return stp, lsFG
	}

	ftest := d.finit + float64(stp*d.gtest)
	if d.stage == 1 && f <= ftest && g >= 0 {
		d.stage = 2
	}
	// Test for warnings, then for convergence (which overrides them).
	if d.brackt && (stp <= d.stmin || stp >= d.stmax) {
		d.msg = "WARNING: ROUNDING ERRORS PREVENT PROGRESS"
	}
	if d.brackt && d.stmax-d.stmin <= d.xtol*d.stmax {
		d.msg = "WARNING: XTOL TEST SATISFIED"
	}
	if stp == d.stpmax && f <= ftest && g <= d.gtest {
		d.msg = "WARNING: STP = STPMAX"
	}
	if stp == d.stpmin && (f > ftest || g >= d.gtest) {
		d.msg = "WARNING: STP = STPMIN"
	}
	if f <= ftest && math.Abs(g) <= d.gtol*(-d.ginit) {
		d.msg = "CONVERGENCE"
		return stp, lsConv
	}
	if d.msg != "" {
		return stp, lsWarn
	}

	// In the first stage use the modified function ψ(stp) = f − f(0) −
	// ftol·stp·f'(0) while the step has not achieved sufficient decrease.
	if d.stage == 1 && f <= d.fx && f > ftest {
		fm := f - float64(stp*d.gtest)
		fxm := d.fx - float64(d.stx*d.gtest)
		fym := d.fy - float64(d.sty*d.gtest)
		gm := g - d.gtest
		gxm := d.gx - d.gtest
		gym := d.gy - d.gtest
		stp = dcstep(&d.stx, &fxm, &gxm, &d.sty, &fym, &gym, stp, fm, gm, &d.brackt, d.stmin, d.stmax)
		d.fx = fxm + float64(d.stx*d.gtest)
		d.fy = fym + float64(d.sty*d.gtest)
		d.gx = gxm + d.gtest
		d.gy = gym + d.gtest
	} else {
		stp = dcstep(&d.stx, &d.fx, &d.gx, &d.sty, &d.fy, &d.gy, stp, f, g, &d.brackt, d.stmin, d.stmax)
	}

	// Force a sufficient decrease in the size of the bracketing interval.
	if d.brackt {
		if math.Abs(d.sty-d.stx) >= p66*d.width1 {
			stp = d.stx + float64(p5*(d.sty-d.stx))
		}
		d.width1 = d.width
		d.width = math.Abs(d.sty - d.stx)
		d.stmin = math.Min(d.stx, d.sty)
		d.stmax = math.Max(d.stx, d.sty)
	} else {
		d.stmin = stp + float64(xtrapl*(stp-d.stx))
		d.stmax = stp + float64(xtrapu*(stp-d.stx))
	}
	stp = math.Min(math.Max(stp, d.stpmin), d.stpmax)
	// If further progress is not possible, let stp be the best point.
	if d.brackt && (stp <= d.stmin || stp >= d.stmax || d.stmax-d.stmin <= d.xtol*d.stmax) {
		stp = d.stx
	}
	return stp, lsFG
}

// sign returns -1, 0 or +1 like numpy.sign (0 for 0).
func sign(x float64) float64 {
	switch {
	case x > 0:
		return 1
	case x < 0:
		return -1
	}
	return 0
}

// dcstep computes a safeguarded step for the line search and updates the
// interval [stx, sty] that contains a step satisfying the Wolfe conditions
// (MINPACK-2 dcstep). It returns the new trial step.
func dcstep(stx, fx, dx, sty, fy, dy *float64, stp, fp, dp float64, brackt *bool, stpmin, stpmax float64) float64 {
	sgnd := float64(sign(dp) * sign(*dx))
	var stpf float64
	switch {
	case fp > *fx:
		// Higher function value: the minimum is bracketed. Take the cubic
		// step if it is closer to stx than the quadratic step, otherwise
		// their average.
		theta := 3*(*fx-fp)/(stp-*stx) + *dx + dp
		s := math.Max(math.Abs(theta), math.Max(math.Abs(*dx), math.Abs(dp)))
		gamma := float64(s * math.Sqrt(float64((theta/s)*(theta/s))-float64((*dx/s)*(dp/s))))
		if stp < *stx {
			gamma = -gamma
		}
		p := (gamma - *dx) + theta
		q := ((gamma - *dx) + gamma) + dp
		r := p / q
		stpc := *stx + float64(r*(stp-*stx))
		stpq := *stx + float64(((*dx/((*fx-fp)/(stp-*stx)+*dx))/2)*(stp-*stx))
		if math.Abs(stpc-*stx) <= math.Abs(stpq-*stx) {
			stpf = stpc
		} else {
			stpf = stpc + (stpq-stpc)/2
		}
		*brackt = true
	case sgnd < 0:
		// Derivatives of opposite sign: the minimum is bracketed. Take the
		// step farther from stp.
		theta := 3*(*fx-fp)/(stp-*stx) + *dx + dp
		s := math.Max(math.Abs(theta), math.Max(math.Abs(*dx), math.Abs(dp)))
		gamma := float64(s * math.Sqrt(float64((theta/s)*(theta/s))-float64((*dx/s)*(dp/s))))
		if stp > *stx {
			gamma = -gamma
		}
		p := (gamma - dp) + theta
		q := ((gamma - dp) + gamma) + *dx
		r := p / q
		stpc := stp + float64(r*(*stx-stp))
		stpq := stp + float64((dp/(dp-*dx))*(*stx-stp))
		if math.Abs(stpc-stp) > math.Abs(stpq-stp) {
			stpf = stpc
		} else {
			stpf = stpq
		}
		*brackt = true
	case math.Abs(dp) < math.Abs(*dx):
		// Same sign, derivative magnitude decreases. The cubic step is used
		// only if the cubic tends to infinity in the direction of the step
		// or its minimum is beyond stp.
		theta := 3*(*fx-fp)/(stp-*stx) + *dx + dp
		s := math.Max(math.Abs(theta), math.Max(math.Abs(*dx), math.Abs(dp)))
		gamma := float64(s * math.Sqrt(math.Max(0, float64((theta/s)*(theta/s))-float64((*dx/s)*(dp/s)))))
		if stp > *stx {
			gamma = -gamma
		}
		p := (gamma - dp) + theta
		q := (gamma + (*dx - dp)) + gamma
		r := p / q
		var stpc float64
		switch {
		case r < 0 && gamma != 0:
			stpc = stp + float64(r*(*stx-stp))
		case stp > *stx:
			stpc = stpmax
		default:
			stpc = stpmin
		}
		stpq := stp + float64((dp/(dp-*dx))*(*stx-stp))
		if *brackt {
			if math.Abs(stpc-stp) < math.Abs(stpq-stp) {
				stpf = stpc
			} else {
				stpf = stpq
			}
			if stp > *stx {
				stpf = math.Min(stp+float64(0.66*(*sty-stp)), stpf)
			} else {
				stpf = math.Max(stp+float64(0.66*(*sty-stp)), stpf)
			}
		} else {
			if math.Abs(stpc-stp) > math.Abs(stpq-stp) {
				stpf = stpc
			} else {
				stpf = stpq
			}
			stpf = math.Min(math.Max(stpf, stpmin), stpmax)
		}
	default:
		// Same sign, derivative magnitude does not decrease: the minimum is
		// not bracketed unless it already was.
		switch {
		case *brackt:
			theta := 3*(fp-*fy)/(*sty-stp) + *dy + dp
			s := math.Max(math.Abs(theta), math.Max(math.Abs(*dy), math.Abs(dp)))
			gamma := float64(s * math.Sqrt(float64((theta/s)*(theta/s))-float64((*dy/s)*(dp/s))))
			if stp > *sty {
				gamma = -gamma
			}
			p := (gamma - dp) + theta
			q := ((gamma - dp) + gamma) + *dy
			r := p / q
			stpf = stp + float64(r*(*sty-stp))
		case stp > *stx:
			stpf = stpmax
		default:
			stpf = stpmin
		}
	}
	// Update the interval which contains a minimizer.
	if fp > *fx {
		*sty, *fy, *dy = stp, fp, dp
	} else {
		if sgnd < 0 {
			*sty, *fy, *dy = *stx, *fx, *dx
		}
		*stx, *fx, *dx = stp, fp, dp
	}
	return stpf
}
