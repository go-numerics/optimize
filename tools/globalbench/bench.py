# Reference side of tools/globalbench, fed to `python -I -` by the Go harness,
# which defines MIN_TIME (seconds of repetition per row) ahead of this script.
# Prints one JSON object {meta, rows: {name: {time, nfev, nit, reps}}}.
# Single thread: BLAS threads are pinned to 1; the global optimisers are
# sequential (workers=1). Row names and parameters are those of
# global_bench_test.go (BenchmarkGlobal) with "." for "/".
import json
import math
import os
import statistics
import time
from importlib.metadata import version

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
from scipy.optimize import basinhopping, differential_evolution, direct  # noqa: E402

PI2 = 2 * math.pi


def sphere(x):
    return float(np.sum(x * x))


def rosenbrock(x):
    t = x[1:] - x[:-1] * x[:-1]
    u = 1 - x[:-1]
    return float(np.sum(100.0 * (t * t) + u * u))


def rastrigin(x):
    return float(10.0 * len(x) + np.sum(x * x - 10.0 * np.cos(PI2 * x)))


# vectorised forms for differential_evolution(vectorized=True): x is (N, S)
def sphere_v(x):
    return np.sum(x * x, axis=0)


def rosenbrock_v(x):
    t = x[1:] - x[:-1] * x[:-1]
    u = 1 - x[:-1]
    return np.sum(100.0 * (t * t) + u * u, axis=0)


def rastrigin_v(x):
    return 10.0 * x.shape[0] + np.sum(x * x - 10.0 * np.cos(PI2 * x), axis=0)


def styblinski_tang(x):
    x2 = x * x
    return float(np.sum(x2 * x2 - 16 * x2 + 5 * x) / 2)


def himmelblau(v):
    x, y = v[0], v[1]
    a = x * x + y - 11.0
    b = x + y * y - 7.0
    return a * a + b * b, np.array([4.0 * x * a + 2.0 * b, 2.0 * a + 4.0 * y * b])


OBJ = {"sphere": sphere, "rosenbrock": rosenbrock, "rastrigin": rastrigin}
OBJV = {"sphere": sphere_v, "rosenbrock": rosenbrock_v, "rastrigin": rastrigin_v}


def box(n, lo, hi):
    return [(lo, hi)] * n


def de_row(f, n, **kw):
    def run():
        r = differential_evolution(f, box(n, -5, 5), rng=1, **kw)
        return r.nfev, r.nit
    return run


ROWS = {}
for fn in ("sphere", "rosenbrock", "rastrigin"):
    for n in (2, 10, 30):
        ROWS["de.%s.%d" % (fn, n)] = de_row(OBJ[fn], n)
for fn in ("sphere", "rosenbrock", "rastrigin"):
    ROWS["dev.%s.10" % fn] = de_row(OBJV[fn], 10, vectorized=True, updating="deferred")
ROWS["deoverhead.trivial.10"] = lambda: (lambda r: (r.nfev, r.nit))(
    differential_evolution(lambda x: x[0], box(10, -5, 5), rng=1, maxiter=50, tol=-1, polish=False))
ROWS["bh.himmelblau.100"] = lambda: (lambda r: (r.nfev, r.nit))(
    basinhopping(himmelblau, np.array([0.0, 0.0]), niter=100, rng=1,
                 minimizer_kwargs=dict(method="L-BFGS-B", jac=True)))


def direct_row(f, n, lo, hi):
    return lambda: (lambda r: (r.nfev, r.nit))(direct(f, box(n, lo, hi)))


ROWS["direct.styblinskitang.2"] = direct_row(styblinski_tang, 2, -4, 4)
ROWS["direct.rosenbrock.2"] = direct_row(rosenbrock, 2, -5, 5)
ROWS["direct.rosenbrock.5"] = direct_row(rosenbrock, 5, -5, 5)
ROWS["direct.sphere.10"] = direct_row(sphere, 10, -5, 5)


def measure(run):
    ts, last, t0 = [], None, time.perf_counter()
    while len(ts) < 1 or (time.perf_counter() - t0 < MIN_TIME and len(ts) < 50):  # noqa: F821
        a = time.perf_counter()
        last = run()
        ts.append(time.perf_counter() - a)
    return dict(time=statistics.median(ts), nfev=int(last[0]), nit=int(last[1]), reps=len(ts))


out = dict(meta=dict(scipy=version("scipy"), numpy=np.__version__), rows={})
for name, run in ROWS.items():
    out["rows"][name] = measure(run)
print(json.dumps(out, allow_nan=False))
