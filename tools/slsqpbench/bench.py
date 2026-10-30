# SciPy side of the SLSQP benchmarks of slsqp_bench_test.go, fed to
# `python -I -` by tools/slsqpbench (Go). Prints one JSON object:
# benchmark name -> median seconds per minimize call (single thread).
import json
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402
from scipy.optimize import LinearConstraint, minimize  # noqa: E402

A = np.array
inf = np.inf


def hs71():
    f = lambda x: x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]
    g = lambda x: A([x[3] * (2 * x[0] + x[1] + x[2]), x[0] * x[3], x[0] * x[3] + 1, x[0] * (x[0] + x[1] + x[2])])
    c1 = {"type": "ineq", "fun": lambda x: A([x[0] * x[1] * x[2] * x[3] - 25]),
          "jac": lambda x: A([[x[1] * x[2] * x[3], x[0] * x[2] * x[3], x[0] * x[1] * x[3], x[0] * x[1] * x[2]]])}
    c2 = {"type": "eq", "fun": lambda x: A([x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 - 40]),
          "jac": lambda x: A([[2 * x[0], 2 * x[1], 2 * x[2], 2 * x[3]]])}
    return f, g, [c1, c2], [(1, 5)] * 4, A([1.0, 5, 5, 1])


def hs38():
    f = lambda x: (100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2 + 90 * (x[3] - x[2] ** 2) ** 2
                   + (1 - x[2]) ** 2 + 10.1 * ((x[1] - 1) ** 2 + (x[3] - 1) ** 2) + 19.8 * (x[1] - 1) * (x[3] - 1))
    g = lambda x: A([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
                     200 * (x[1] - x[0] ** 2) + 20.2 * (x[1] - 1) + 19.8 * (x[3] - 1),
                     -360 * x[2] * (x[3] - x[2] ** 2) - 2 * (1 - x[2]),
                     180 * (x[3] - x[2] ** 2) + 20.2 * (x[3] - 1) + 19.8 * (x[1] - 1)])
    return f, g, [], [(-10, 10)] * 4, A([-3.0, -1, -3, -1])


def hs43():
    def c(x):
        return A([8 - x[0] ** 2 - x[1] ** 2 - x[2] ** 2 - x[3] ** 2 - x[0] + x[1] - x[2] + x[3],
                  10 - x[0] ** 2 - 2 * x[1] ** 2 - x[2] ** 2 - 2 * x[3] ** 2 + x[0] + x[3],
                  5 - 2 * x[0] ** 2 - x[1] ** 2 - x[2] ** 2 - 2 * x[0] + x[1] + x[3]])

    def j(x):
        return A([[-2 * x[0] - 1, -2 * x[1] + 1, -2 * x[2] - 1, -2 * x[3] + 1],
                  [-2 * x[0] + 1, -4 * x[1], -2 * x[2], -4 * x[3] + 1],
                  [-4 * x[0] - 2, -2 * x[1] + 1, -2 * x[2], 1.0]])

    f = lambda x: x[0] ** 2 + x[1] ** 2 + 2 * x[2] ** 2 + x[3] ** 2 - 5 * x[0] - 5 * x[1] - 21 * x[2] + 7 * x[3]
    g = lambda x: A([2 * x[0] - 5, 2 * x[1] - 5, 4 * x[2] - 21, 2 * x[3] + 7])
    return f, g, [{"type": "ineq", "fun": c, "jac": j}], None, np.zeros(4)


def quad50():
    n = 50
    w = np.arange(1, n + 1, dtype=float)
    f = lambda x: float(np.sum(w * (x - 1) ** 2))
    g = lambda x: 2 * w * (x - 1)
    return f, g, [LinearConstraint(np.ones((1, n)), -inf, 25)], [(-10, 10)] * n, np.zeros(n)


def timed(fn, min_time=0.2, reps=5):
    fn()
    t0 = time.perf_counter()
    fn()
    one = time.perf_counter() - t0
    k = max(1, int(min_time / max(one, 1e-9)))
    meds = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(k):
            fn()
        meds.append((time.perf_counter() - t0) / k)
    return float(np.median(meds))


def case(mk, jac):
    f, g, cons, bounds, x0 = mk()
    if not jac:
        cons = [{k: v for k, v in c.items() if k != "jac"} for c in cons]
    kw = dict(method="SLSQP", constraints=cons, bounds=bounds, jac=g if jac else None)
    return lambda: minimize(f, x0, **kw)


out = {
    "SLSQPHS71": timed(case(hs71, True)),
    "SLSQPHS71FD": timed(case(hs71, False)),
    "SLSQPHS38": timed(case(hs38, True)),
    "SLSQPHS43": timed(case(hs43, True)),
    "SLSQPQuad50": timed(case(quad50, True)),
}
print(json.dumps(out))
