"""Fixture generator: SciPy minimize on standard test functions.

Run: uv run --with scipy --with numpy python testdata/gen/minimize.py
Writes testdata/minimize.json. The Go tests define the same functions.
"""
import json, os, warnings
import numpy as np
from scipy import optimize

warnings.simplefilter("ignore")


def rosen(x):
    x = np.asarray(x)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def rosen_der(x):
    return optimize.rosen_der(np.asarray(x))


def beale(x):
    a, b = x
    return (1.5 - a + a * b) ** 2 + (2.25 - a + a * b * b) ** 2 + (2.625 - a + a * b ** 3) ** 2


def beale_der(x):
    a, b = x
    t1 = 1.5 - a + a * b
    t2 = 2.25 - a + a * b * b
    t3 = 2.625 - a + a * b ** 3
    return np.array([
        2 * t1 * (b - 1) + 2 * t2 * (b * b - 1) + 2 * t3 * (b ** 3 - 1),
        2 * t1 * a + 2 * t2 * 2 * a * b + 2 * t3 * 3 * a * b * b,
    ])


def himmelblau(x):
    a, b = x
    return (a * a + b - 11) ** 2 + (a + b * b - 7) ** 2


def himmelblau_der(x):
    a, b = x
    u = a * a + b - 11
    v = a + b * b - 7
    return np.array([4 * a * u + 2 * v, 2 * u + 4 * b * v])


def powell(x):
    a, b, c, d = x
    return (a + 10 * b) ** 2 + 5 * (c - d) ** 2 + (b - 2 * c) ** 4 + 10 * (a - d) ** 4


def powell_der(x):
    a, b, c, d = x
    return np.array([
        2 * (a + 10 * b) + 40 * (a - d) ** 3,
        20 * (a + 10 * b) + 4 * (b - 2 * c) ** 3,
        10 * (c - d) - 8 * (b - 2 * c) ** 3,
        -10 * (c - d) - 40 * (a - d) ** 3,
    ])


def wood(x):
    a, b, c, d = x
    return (100 * (b - a * a) ** 2 + (1 - a) ** 2 + 90 * (d - c * c) ** 2 + (1 - c) ** 2
            + 10.1 * ((b - 1) ** 2 + (d - 1) ** 2) + 19.8 * (b - 1) * (d - 1))


def wood_der(x):
    a, b, c, d = x
    return np.array([
        -400 * a * (b - a * a) - 2 * (1 - a),
        200 * (b - a * a) + 20.2 * (b - 1) + 19.8 * (d - 1),
        -360 * c * (d - c * c) - 2 * (1 - c),
        180 * (d - c * c) + 20.2 * (d - 1) + 19.8 * (b - 1),
    ])


def alt(n):
    return [-1.2 if i % 2 == 0 else 1.0 for i in range(n)]


problems = {
    "rosen2": (rosen, rosen_der, [-1.2, 1.0]),
    "beale": (beale, beale_der, [1.0, 1.0]),
    "himmelblau": (himmelblau, himmelblau_der, [0.0, 0.0]),
    "powell": (powell, powell_der, [3.0, -1.0, 0.0, 1.0]),
    "wood": (wood, wood_der, [-3.0, -1.0, -3.0, -1.0]),
    "rosen10": (rosen, rosen_der, alt(10)),
    "rosen100": (rosen, rosen_der, alt(100)),
}

cases = []


def record(name, method, grad, res, **extra):
    d = dict(problem=name, method=method, grad=grad,
             x=[float(v) for v in np.atleast_1d(res.x)], fun=float(res.fun),
             nit=int(res.get("nit", -1)), nfev=int(res.get("nfev", -1)),
             njev=int(res.get("njev", -1)), nhev=int(res.get("nhev", -1)),
             status=int(res.status), success=bool(res.success),
             message=str(res.message))
    d.update(extra)
    cases.append(d)


for name, (f, g, x0) in problems.items():
    n = len(x0)
    for method in ["Nelder-Mead", "Powell"]:
        if n > 10:
            continue
        record(name, method, False, optimize.minimize(f, x0, method=method))
        if method == "Nelder-Mead":
            record(name, method, False,
                   optimize.minimize(f, x0, method=method, options=dict(adaptive=True)),
                   adaptive=True)
    for method in ["CG", "BFGS", "L-BFGS-B"]:
        record(name, method, True, optimize.minimize(f, x0, jac=g, method=method))
        record(name, method, False, optimize.minimize(f, x0, method=method))
    record(name, "Newton-CG", True, optimize.minimize(f, x0, jac=g, method="Newton-CG"))
    if name.startswith("rosen"):
        record(name, "Newton-CG", True,
               optimize.minimize(f, x0, jac=g, hess=optimize.rosen_hess, method="Newton-CG"), hess="hess")
        record(name, "Newton-CG", True,
               optimize.minimize(f, x0, jac=g, hessp=optimize.rosen_hess_prod, method="Newton-CG"), hess="hessp")

# Bounded Nelder-Mead and Powell (bounds cut off the optimum).
bnd = [(-2, 0.8), (-2, 0.8)]
for method in ["Nelder-Mead", "Powell"]:
    record("rosen2", method, False,
           optimize.minimize(rosen, [-1.2, 0.5], method=method, bounds=bnd), bounds=bnd, x0=[-1.2, 0.5])

# Iteration limits.
record("rosen2", "BFGS", True, optimize.minimize(rosen, [-1.2, 1.0], jac=rosen_der, method="BFGS",
                                                  options=dict(maxiter=5)), maxiter=5)
record("rosen2", "Nelder-Mead", False, optimize.minimize(rosen, [-1.2, 1.0], method="Nelder-Mead",
                                                         options=dict(maxfev=50)), maxfev=50)
record("rosen2", "Powell", False, optimize.minimize(rosen, [-1.2, 1.0], method="Powell",
                                                    options=dict(maxiter=2)), maxiter=2)
record("rosen2", "CG", True, optimize.minimize(rosen, [-1.2, 1.0], jac=rosen_der, method="CG",
                                                options=dict(maxiter=3)), maxiter=3)

out = os.path.join(os.path.dirname(__file__), "..", "minimize.json")
with open(out, "w") as fh:
    json.dump(cases, fh, indent=1)
print(len(cases), "cases ->", out)
