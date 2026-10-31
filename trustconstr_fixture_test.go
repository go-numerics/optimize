package optimize

import (
	"math"
	"runtime"
	"testing"
)

// tcRoundingSensitive are the runs on which SciPy's own x86_64 and arm64
// wheels take different paths (slsqporacle diff of the two trust-constr
// fixtures): 35 runs, mostly with finite-difference derivatives, where the
// last bit of a rounding decides the iteration count.
var tcRoundingSensitive = map[string]bool{
	"FixedAllViolated/fd": true, "FixedAllViolated/jac": true, "HS1/fd": true, "HS1/jac": true,
	"HS10/fd": true, "HS11/fd": true, "HS12/fd": true, "HS18/fd": true, "HS19/fd": true,
	"HS2/fd": true, "HS21/fd": true, "HS27/fd": true, "HS28/fd": true, "HS29/fd": true,
	"HS30/fd": true, "HS35/fd": true, "HS38/fd": true, "HS38/jac": true, "HS38/xtol": true,
	"HS39/fd": true, "HS42/fd": true, "HS43/fd": true, "HS48/jac": true, "HS5/fd": true,
	"HS7/fd": true, "HS71-nonlinear": true, "HS71-nonlinear-fd": true, "HS71/fd": true,
	"HS71/gtol": true, "HS71/jac": true, "HS76/fd": true, "HS9/fd": true, "HS9/jac": true,
	"Infeasible/fd": true, "Infeasible/jac": true,
}

// tcPathExceptions are the runs whose evaluation counts differ from SciPy's
// although both SciPy wheels agree on them. Each was traced iterate by iterate
// (`slsqporacle trace` and `tracediff`, docs/design.md "Why four runs differ
// from SciPy"); on all four the iteration count is SciPy's and only the
// number of evaluations differs:
//
//   - HS48/fd and Rosen-free/fd: a tangential step differs by 1e-16 to 5e-16
//     (relative) — no SVD, and on Rosen-free no constraint at all, so only
//     dot products and updates of two to five entries: go-simd/floats fuses
//     each multiply-add, the OpenBLAS core that generated the fixture
//     (Nehalem, `OPENBLAS_VERBOSE=2`) does not. The finite-difference Jacobian
//     of a later iterate turns that into 4e-8 to 1e-7, and the run stops one
//     gradient earlier or later;
//   - SingularEq3/jac and /fd: the Jacobian is rank-deficient, so the
//     projections use an SVD, and go-ndarray's U and Vt differ from LAPACK's
//     dgesdd in the last bit (1.5e-16 on the first Newton point); a trial
//     point is evaluated once more at iteration 3.
//
// HS21/jac (60 iterations in SciPy, a different count here) and HS35/jac (46)
// were our defects, now fixed: projected_cg's rt_g and the tangential radius
// use norm(x)**2, the square of the rounded norm, not x·x, and its boundary
// step is x + (theta*alpha)*p, not x + theta*(alpha*p). Both take SciPy's
// exact path. On the four left the solution, status and objective are
// asserted as everywhere else, and the iteration count must equal SciPy's.
var tcPathExceptions = map[string]bool{
	"HS48/fd": true, "Rosen-free/fd": true, "SingularEq3/jac": true, "SingularEq3/fd": true,
}

// TestTrustConstrFixtures replays testdata/trustconstr.json (SciPy 1.18.1's
// trust-constr on the problems of tools/slsqporacle/hs.py, x86_64 wheel).
// Statuses, objective, solution and constraint violation are asserted;
// iteration and evaluation counts are logged and summarised.
func TestTrustConstrFixtures(t *testing.T) {
	f := loadFixture(t, "testdata/trustconstr.json")
	var total, statusOK, pathOK, funOK, errOK int
	for _, c := range f.Cases {
		total++
		r, err := replayCase(t, c, TrustConstr)
		if c.Error != "" {
			if err == nil {
				t.Errorf("%s: SciPy raised %q, Go returned %+v", c.Name, c.Error, r)
			} else {
				errOK++
				statusOK++
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: %v", c.Name, err)
			continue
		}
		p := hsProblems[c.Problem]
		cv := hsViolation(p, r.X)
		same := r.Nit == c.Nit && r.Nfev == c.Nfev && r.Njev == c.Njev
		if same {
			pathOK++
		}
		if r.Status == c.Status {
			statusOK++
		}
		fok := math.Abs(r.Fun-c.Fun) <= 1e-6*(1+math.Abs(c.Fun))
		if fok {
			funOK++
		}
		t.Logf("%-22s status %d/%d nit %d/%d nfev %d/%d njev %d/%d fun %.12g/%.12g cv %.2g/%.2g",
			c.Name, r.Status, c.Status, r.Nit, c.Nit, r.Nfev, c.Nfev, r.Njev, c.Njev, r.Fun, c.Fun, cv, c.Maxcv)
		if r.Message != trustConstrMessages[r.Status] {
			t.Errorf("%s: message %q for status %d", c.Name, r.Message, r.Status)
		}
		if r.Success != c.Success {
			t.Errorf("%s: success %v (status %d), SciPy %v (status %d)", c.Name, r.Success, r.Status, c.Success, c.Status)
		}
		// Statuses 1 and 2 (gtol, xtol) both succeed; which one fires is
		// rounding-dependent on the runs where SciPy's wheels disagree.
		if r.Status != c.Status && !(r.Success && c.Success) {
			t.Errorf("%s: status %d, SciPy %d", c.Name, r.Status, c.Status)
		}
		if c.Success {
			// trust-constr's xtol stop leaves the objective a few 1e-6
			// relative from SciPy's on some finite-difference runs.
			near := math.Abs(r.Fun-c.Fun) <= 1e-5*(1+math.Abs(c.Fun))
			if !near && (c.Fstar == nil || math.Abs(r.Fun-*c.Fstar) > 1e-4*(1+math.Abs(*c.Fstar))) {
				t.Errorf("%s: fun %.17g, SciPy %.17g", c.Name, r.Fun, c.Fun)
			}
			if cv > math.Max(1e-6, 10*c.Maxcv) {
				t.Errorf("%s: constraint violation %g, SciPy %g", c.Name, cv, c.Maxcv)
			}
		}
		// trust-constr's reductions use go-simd/floats, whose lane order
		// depends on the architecture: counts are asserted where they were
		// checked (amd64 and arm64).
		deterministic := !tcRoundingSensitive[c.Name] && !tcPathExceptions[c.Name] &&
			(runtime.GOARCH == "amd64" || runtime.GOARCH == "arm64")
		if tcPathExceptions[c.Name] && r.Nit != c.Nit {
			t.Errorf("%s: nit %d, SciPy %d", c.Name, r.Nit, c.Nit)
		}
		if deterministic && !same {
			t.Errorf("%s: nit/nfev/njev %d/%d/%d, SciPy %d/%d/%d", c.Name, r.Nit, r.Nfev, r.Njev, c.Nit, c.Nfev, c.Njev)
		}
	}
	t.Logf("SciPy %s (%s) trust-constr: %d cases, status %d (errors %d), exact nit/nfev/njev %d, fun within 1e-6 %d",
		f.Scipy, f.Machine, total, statusOK, errOK, pathOK, funOK)
}
