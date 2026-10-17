"""SciPy side of the benchmarks in bench_test.go.

Run: uv run --with scipy --with numpy python benchmarks/bench_scipy.py [reps]
Prints one JSON object: benchmark name -> median seconds per call, plus the
part of it spent in the Python objective/gradient callbacks ("<name>/callbacks").
"""
import json, math, os, re, sys, time
import numpy as np
from scipy import optimize

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
WANT = re.compile(os.environ.get("BENCH", "."))


def want(name):
    """Whether name is selected by $BENCH (a regexp, as go test -bench)."""
    return WANT.search(name) is not None


def alt(n):
    return np.array([-1.2 if i % 2 == 0 else 1.0 for i in range(n)])


def timed(fn, min_time=0.2):
    """Median over REPS batches of the mean time per call in a batch."""
    fn()  # warm-up
    t0 = time.perf_counter(); fn(); one = time.perf_counter() - t0
    k = max(1, int(min_time / max(one, 1e-9)))
    meds = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        for _ in range(k):
            fn()
        meds.append((time.perf_counter() - t0) / k)
    return float(np.median(meds))


def callbacks_cost(f, g, x0, res):
    """Time to make res.nfev f calls and res.njev g calls at x0 (the Python
    callback share of a minimize call)."""
    nf = res.nfev
    nj = res.get("njev", 0) or 0
    def run():
        for _ in range(nf):
            f(x0)
        for _ in range(nj):
            g(x0)
    return timed(run, 0.05)


out = {}
rosen, rosen_der = optimize.rosen, optimize.rosen_der
cases = {
    "Rosen2BFGS": (2, "BFGS", True), "Rosen100BFGS": (100, "BFGS", True),
    "Rosen2LBFGSB": (2, "L-BFGS-B", True), "Rosen100LBFGSB": (100, "L-BFGS-B", True),
    "Rosen2BFGSFD": (2, "BFGS", False), "Rosen100LBFGSBFD": (100, "L-BFGS-B", False),
    "Rosen2NelderMead": (2, "Nelder-Mead", False),
}
for name, (n, m, grad) in cases.items():
    if not want(name):
        continue
    x0 = alt(n)
    jac = rosen_der if grad else None
    call = lambda: optimize.minimize(rosen, x0, jac=jac, method=m)
    out[name] = timed(call)
    out[name + "/callbacks"] = callbacks_cost(rosen, rosen_der, x0, call())

m = 10000
x = 4 * np.arange(m) / (m - 1)
y = 2.5 * np.exp(-1.3 * x) + 0.5 + 0.2 * np.sin(np.arange(m) * 12.9898)
model = lambda x, a, b, c: a * np.exp(-b * x) + c
for meth, name in (("lm", "CurveFit1e4LM"), ("trf", "CurveFit1e4TRF")):
    if not want(name):
        continue
    out[name] = timed(lambda: optimize.curve_fit(model, x, y, p0=[1, 1, 1], method=meth))

f = lambda x: x * x * x - 2 * x - 5
if want("Brentq"):
    out["Brentq"] = timed(lambda: optimize.brentq(f, 2, 3), 0.05)
# The rest of the public operations (bench_test.go poses the same problems).


def case(name, fn, min_time=0.2, callbacks=None):
    """Time fn under name if selected; callbacks() estimates the time of the
    Python callbacks of one call."""
    if not want(name):
        return
    out[name] = timed(fn, min_time)
    if callbacks is not None:
        out[name + "/callbacks"] = callbacks()


def minimize_case(name, n, method, grad=False, hess=None, hessp=None):
    x0 = alt(n)
    kw = dict(method=method)
    if grad:
        kw["jac"] = rosen_der
    if hess is not None:
        kw["hess"] = hess
    if hessp is not None:
        kw["hessp"] = hessp
    call = lambda: optimize.minimize(rosen, x0, **kw)
    case(name, call, callbacks=lambda: callbacks_cost(rosen, rosen_der, x0, call()))


minimize_case("Rosen2Powell", 2, "Powell")
minimize_case("Rosen2CG", 2, "CG", grad=True)
minimize_case("Rosen100CG", 100, "CG", grad=True)
minimize_case("Rosen100BFGSFD", 100, "BFGS")
minimize_case("Rosen2NewtonCGHess", 2, "Newton-CG", grad=True, hess=optimize.rosen_hess)
minimize_case("Rosen100NewtonCGHessP", 100, "Newton-CG", grad=True, hessp=optimize.rosen_hess_prod)


def rosen_resid(x):
    return np.array([10 * (x[1] - x[0] ** 2), 1 - x[0]])


x0r = np.array([-1.2, 1.0])
case("LeastSquaresRosenTRF", lambda: optimize.least_squares(rosen_resid, x0r, method="trf"))
case("LeastSquaresRosenLM", lambda: optimize.least_squares(rosen_resid, x0r, method="lm"))
case("LeastsqRosen", lambda: optimize.leastsq(rosen_resid, x0r, full_output=True))
case("CurveFit1e4TRFBounds", lambda: optimize.curve_fit(model, x, y, p0=[1, 1, 0],
                                                        bounds=([0, 0, -1], [10, 3, 1])))


def root_fun(x):
    return [x[0] + 0.5 * (x[0] - x[1]) ** 3 - 1.0, 0.5 * (x[1] - x[0]) ** 3 + x[1]]


def broyden_tri(x):
    f = (3 - 2 * x) * x + 1
    f[1:] -= x[:-1]
    f[:-1] -= 2 * x[1:]
    return f


case("RootHybr", lambda: optimize.root(root_fun, [0, 0], method="hybr"))
case("FsolveHybr", lambda: optimize.fsolve(root_fun, [0, 0], full_output=True))
case("RootBroyden100", lambda: optimize.root(broyden_tri, -np.ones(100), method="hybr"))


def lp_dense(m, n):
    i = np.arange(m * n).reshape(m, n)
    A = 1 + np.sin(i * 12.9898) * 0.5
    b = n * (1 + 0.5 * np.sin(np.arange(m) * 78.233))
    c = -1 - np.sin(np.arange(n) * 4.1414) * 0.5
    return c, A, b


case("LinprogSmall", lambda: optimize.linprog([-1, 4], A_ub=[[-3, 1], [1, 2]], b_ub=[6, 4],
                                              bounds=[(None, None), (-3, None)]))
cl, Al, bl = lp_dense(30, 50)
case("LinprogDense30x50", lambda: optimize.linprog(cl, A_ub=Al, b_ub=bl, method="highs-ds"))
case("LinprogDense30x50IP", lambda: optimize.linprog(cl, A_ub=Al, b_ub=bl, method="highs-ipm"))

cubic = lambda x: x * x * x - 2 * x - 5
cubic1 = lambda x: 3 * x * x - 2
cubic2 = lambda x: 6 * x
case("Brenth", lambda: optimize.brenth(cubic, 2, 3), 0.05)
case("Ridder", lambda: optimize.ridder(cubic, 2, 3), 0.05)
case("Bisect", lambda: optimize.bisect(cubic, 2, 3), 0.05)
case("Newton", lambda: optimize.newton(cubic, 2, fprime=cubic1), 0.05)
case("Secant", lambda: optimize.newton(cubic, 2), 0.05)
case("Halley", lambda: optimize.newton(cubic, 2, fprime=cubic1, fprime2=cubic2), 0.05)

scalar_f = lambda x: (x - 2) * x * (x + 2) ** 2
case("MinScalarBrent", lambda: optimize.minimize_scalar(scalar_f, method="brent"), 0.05)
case("MinScalarGolden", lambda: optimize.minimize_scalar(scalar_f, method="golden"), 0.05)
case("MinScalarBounded", lambda: optimize.minimize_scalar(scalar_f, bounds=(-3, -1), method="bounded"), 0.05)
case("Fminbound", lambda: optimize.fminbound(math.cos, 3, 4), 0.05)
case("Bracket", lambda: optimize.bracket(scalar_f, 0, 1), 0.05)


def himmelblau(p):
    a = p[0] * p[0] + p[1] - 11
    c = p[0] + p[1] * p[1] - 7
    return a * a + c * c


case("Brute2D", lambda: optimize.brute(himmelblau, ((-4, 4), (-4, 4)), Ns=50, finish=None))
case("Brute2DFinish", lambda: optimize.brute(himmelblau, ((-4, 4), (-4, 4)), Ns=20))
x100 = alt(100)
case("ApproxFprime100", lambda: optimize.approx_fprime(x100, rosen))
case("CheckGrad100", lambda: optimize.check_grad(rosen, rosen_der, x100))

print(json.dumps(out))
