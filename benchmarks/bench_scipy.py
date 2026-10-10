"""SciPy side of the benchmarks in bench_test.go.

Run: uv run --with scipy --with numpy python benchmarks/bench_scipy.py [reps]
Prints one JSON object: benchmark name -> median seconds per call, plus the
part of it spent in the Python objective/gradient callbacks ("<name>/callbacks").
"""
import json, math, sys, time
import numpy as np
from scipy import optimize

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5


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
    out[name] = timed(lambda: optimize.curve_fit(model, x, y, p0=[1, 1, 1], method=meth))

f = lambda x: x * x * x - 2 * x - 5
out["Brentq"] = timed(lambda: optimize.brentq(f, 2, 3), 0.05)
print(json.dumps(out))
