"""Fixtures for the L-BFGS-B port: scipy.optimize.minimize(method='L-BFGS-B').

Run from the repository root:
    ~/.local/bin/uv run --with scipy --with numpy python testdata/gen/lbfgsb.py
It writes testdata/lbfgsb.json. The Go test defines the same objectives by
name; keep the two in sync.
"""
import json
import math
import warnings

import numpy as np
import scipy
from scipy import optimize
from scipy.optimize import minimize

INF = math.inf
warnings.simplefilter("ignore")


# --- objectives and gradients -------------------------------------------

# The standard functions are copied verbatim from testdata/gen/minimize.py
# (the Go side reuses funcs_test.go).

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




def quad_ab(n):
    a = np.zeros((n, n))
    for i in range(n):
        a[i, i] = i + 1.0
        if i + 1 < n:
            a[i, i + 1] = a[i + 1, i] = 0.5
    b = np.array([(-1.0) ** i * (1.0 + 0.1 * i) for i in range(n)])
    return a, b


QA, QB = quad_ab(50)


def quad(x):
    # 0.5 x'Ax - b'x, written as explicit sums so the Go side can match it.
    f = 0.0
    n = len(x)
    for i in range(n):
        ax = (i + 1.0) * x[i]
        if i > 0:
            ax += 0.5 * x[i - 1]
        if i + 1 < n:
            ax += 0.5 * x[i + 1]
        f += 0.5 * x[i] * ax - QB[i] * x[i]
    return f


def quad_g(x):
    n = len(x)
    g = np.zeros(n)
    for i in range(n):
        ax = (i + 1.0) * x[i]
        if i > 0:
            ax += 0.5 * x[i - 1]
        if i + 1 < n:
            ax += 0.5 * x[i + 1]
        g[i] = ax - QB[i]
    return g


def sphere(x):
    return x[0] * x[0] + x[1] * x[1]


def sphere_badg(x):
    # Inconsistent gradient (wrong sign): the search direction is uphill.
    return -2.0 * x


def tiny(x):
    return 1e-160 * x[0]


def tiny_g(x):
    # So small that the Cauchy step underflows: d = 0 and g'd = 0.
    return np.array([1e-160])


def nan_g(x):
    return np.array([math.nan, math.nan])


OBJ = {
    "sphere_nang": (sphere, nan_g),
    "tiny": (tiny, tiny_g),
    "rosen": (rosen, rosen_der),
    "beale": (beale, beale_der),
    "himmelblau": (himmelblau, himmelblau_der),
    "wood": (wood, wood_der),
    "powell": (powell, powell_der),
    "quad": (quad, quad_g),
    "sphere_badg": (sphere, sphere_badg),
}


def poly(x, prm):
    # 0.5 x'Ax - b'x + c sum x^4 with explicit loops (same order as the Go test).
    A, b, c = prm["A"], prm["b"], prm["c"]
    n = len(x)
    f = 0.0
    for i in range(n):
        ax = 0.0
        for j in range(n):
            ax += A[i * n + j] * x[j]
        f += 0.5 * x[i] * ax - b[i] * x[i]
        x2 = x[i] * x[i]
        f += c * (x2 * x2)
    return f


def poly_g(x, prm):
    A, b, c = prm["A"], prm["b"], prm["c"]
    n = len(x)
    g = np.zeros(n)
    for i in range(n):
        ax = 0.0
        for j in range(n):
            ax += A[i * n + j] * x[j]
        x3 = (x[i] * x[i]) * x[i]
        g[i] = ax - b[i] + 4 * c * x3
    return g


def alt(n):
    return [(-1.2 if i % 2 == 0 else 1.0) for i in range(n)]


def mixed_bounds(n):
    out = []
    for i in range(n):
        k = i % 5
        if k == 0:
            out.append((0.05, INF))
        elif k == 1:
            out.append((-INF, -0.05))
        elif k == 2:
            out.append((-INF, INF))
        elif k == 3:
            out.append((0.2, 0.2))
        else:
            out.append((-0.01, 0.01))
    return out


CASES = []


def case(name, obj, x0, analytic=True, bounds=None, params=None, **opts):
    CASES.append(dict(name=name, obj=obj, x0=list(map(float, x0)), analytic=analytic,
                      bounds=bounds, params=params, options=opts))


# Small polynomial problems found by a random search over bound types, memory
# sizes and maxls, each reaching a rarely taken branch of the algorithm
# (named in the case): a failed Cholesky factorisation in formt or formk, a
# skipped BFGS update, a restart after a failed line search with a
# non-empty memory, the 3.0 subspace backtrack, a Cauchy point with every
# variable at a breakpoint, a zero gradient component on a bounded variable.
def pcase(name, x0, A, b, c, lo, hi, m, maxls):
    bounds = None
    if any(v != -INF for v in lo) or any(v != INF for v in hi):
        bounds = list(zip(lo, hi))
    case(name, "poly", x0, True, bounds, dict(A=A, b=b, c=c), maxcor=m, maxls=maxls)


# Unbounded below: theta grows until formt's Cholesky factorisation fails.
pcase("poly_formt_fail", [-1, -2.25], [2, 2, 2, 2], [-1.5, 0], 0, [-INF, -0.5], [0.25, INF], 10, 20)
pcase("poly_zero_grad_bounded", [-1, -1.25], [1.625, -1.25, -1.25, 1], [1.25, -0.0], 0,
      [-1, -INF], [INF, -0.25], 2, 20)
pcase("poly_formk_fail_allfixed", [2.5], [0], [-1.25], 1, [-1], [INF], 2, 20)
pcase("poly_restart", [-3.5], [0], [1.25], 0.25, [-INF], [INF], 3, 1)
pcase("poly_subsm_upper", [-0.75, -2.5, -1.75],
      [1.0625, -1.3125, 0.4375, -1.3125, 1.625, -1.8125, 0.4375, -1.8125, 2.5625],
      [-0.5, 2.75, -4.25], 0, [-INF, -INF, -0.25], [-0.0, INF, 0.25], 2, 20)
pcase("poly_formk_no_update", [0.25, -1.25], [0.625, -0.25, -0.25, -0.875], [-0.25, 0.5], 1,
      [-INF, -INF], [INF, -0.0], 2, 20)
pcase("poly_subsm_backtrack", [1.25, 0.5, 0.5],
      [1.8125, 0.4375, 0.625, 0.4375, 3.8125, 3.25, 0.625, 3.25, 2.3125], [1.25, 1, 1.25], 0,
      [-0.75, -0.5, 0.25], [INF, INF, 0.75], 1, 1)
pcase("poly_cauchy_at_breakpoint", [1.5, 0], [5.3125, -0.375, -0.375, 0.5625], [-1.75, -1.75], 0,
      [-INF, -1.5], [0.5, -1], 1, 20)


for analytic in (True, False):
    sfx = "" if analytic else "_fd"
    case("rosen2" + sfx, "rosen", [-1.2, 1.0], analytic)
    case("rosen10" + sfx, "rosen", alt(10), analytic)
    case("rosen100" + sfx, "rosen", alt(100), analytic)
    case("beale" + sfx, "beale", [1.0, 1.0], analytic)
    case("himmelblau" + sfx, "himmelblau", [0.0, 0.0], analytic)
    case("wood" + sfx, "wood", [-3.0, -1.0, -3.0, -1.0], analytic)
    case("powell" + sfx, "powell", [3.0, -1.0, 0.0, 1.0], analytic)
    case("quad50" + sfx, "quad", [0.0] * 50, analytic)
    case("rosen2_box" + sfx, "rosen", [-1.2, 1.0], analytic, [(-1.5, 0.8)] * 2)
    case("rosen10_box" + sfx, "rosen", alt(10), analytic, [(-1.5, 0.8)] * 10)
    case("quad50_mixed" + sfx, "quad", [0.0] * 50, analytic, mixed_bounds(50))

case("rosen2_lower", "rosen", [-1.2, 1.0], True, [(-INF, 0.5), (1.2, INF)])
case("beale_box_m3", "beale", [1.0, 1.0], True, [(-4.5, 4.5), (-0.5, 0.4)], maxcor=3)
case("rosen10_maxiter", "rosen", alt(10), True, None, maxiter=5)
case("rosen2_maxfun", "rosen", [-1.2, 1.0], True, None, maxfun=10)
case("rosen2_maxfun_fd", "rosen", [-1.2, 1.0], False, None, maxfun=10)
case("rosen2_maxls1", "rosen", [-1.2, 1.0], True, None, maxls=1)
case("rosen2_tol", "rosen", [-1.2, 1.0], True, None, tol=1e-3)
case("rosen10_m3", "rosen", alt(10), True, None, maxcor=3)
case("abnormal", "sphere_badg", [1.0, 2.0], True)
case("abnormal_gd", "tiny", [1.0], True, None, gtol=1e-300)
case("x0_optimal", "rosen", [1.0, 1.0], True)
case("x0_optimal_negative_gtol", "rosen", [1.0, 1.0], True, None, gtol=-1.0)
case("nan_gradient", "sphere_nang", [1.0, 2.0], True)
case("nan_gradient_bounded", "sphere_nang", [1.0, 2.0], True, [(0.0, 3.0), (-INF, INF)])



def num(v):
    # JSON has no NaN or infinities: spell them as strings.
    v = float(v)
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "+Inf" if v > 0 else "-Inf"
    return v


def run(c):
    if c["obj"] == "poly":
        prm = c["params"]
        f, g = (lambda x: poly(x, prm)), (lambda x: poly_g(x, prm))
    else:
        f, g = OBJ[c["obj"]]
    opts = dict(c["options"])
    tol = opts.pop("tol", None)
    fs = []

    def cb(intermediate_result):
        fs.append(float(intermediate_result.fun))

    bounds = c["bounds"]
    r = minimize(f, np.array(c["x0"]), method="L-BFGS-B", jac=g if c["analytic"] else None,
                 bounds=bounds, tol=tol, options=opts, callback=cb)
    enc = lambda b: [None if math.isinf(v) else v for v in b]
    out = dict(c)
    out["bounds"] = None if bounds is None else [enc(b) for b in bounds]
    out["tol"] = tol
    out["options"] = opts
    out.update(x=[num(v) for v in r.x], fun=num(r.fun), jac=[num(v) for v in r.jac],
               nit=int(r.nit), nfev=int(r.nfev), njev=int(r.njev), status=int(r.status),
               success=bool(r.success), message=str(r.message), trace=[num(v) for v in fs])
    return out


def main():
    out = dict(scipy=scipy.__version__, numpy=np.__version__, cases=[run(c) for c in CASES])
    with open("testdata/lbfgsb.json", "w") as fh:
        json.dump(out, fh, indent=1, allow_nan=False)
        fh.write("\n")
    for c in out["cases"]:
        print(f'{c["name"]:22s} nit={c["nit"]:4d} nfev={c["nfev"]:5d} njev={c["njev"]:4d} '
              f'st={c["status"]} fun={float(c["fun"]):.3e} {c["message"]}')


if __name__ == "__main__":
    main()
