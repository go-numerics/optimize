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


OBJ = {
    "tiny": (tiny, tiny_g),
    "rosen": (rosen, rosen_der),
    "beale": (beale, beale_der),
    "himmelblau": (himmelblau, himmelblau_der),
    "wood": (wood, wood_der),
    "powell": (powell, powell_der),
    "quad": (quad, quad_g),
    "sphere_badg": (sphere, sphere_badg),
}


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


def case(name, obj, x0, analytic=True, bounds=None, **opts):
    CASES.append(dict(name=name, obj=obj, x0=list(map(float, x0)), analytic=analytic,
                      bounds=bounds, options=opts))


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


def run(c):
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
    out.update(x=[float(v) for v in r.x], fun=float(r.fun), jac=[None if math.isnan(v) else float(v) for v in r.jac],
               nit=int(r.nit), nfev=int(r.nfev), njev=int(r.njev), status=int(r.status),
               success=bool(r.success), message=str(r.message), trace=fs)
    return out


def main():
    out = dict(scipy=scipy.__version__, numpy=np.__version__, cases=[run(c) for c in CASES])
    with open("testdata/lbfgsb.json", "w") as fh:
        json.dump(out, fh, indent=1, allow_nan=False)
        fh.write("\n")
    for c in out["cases"]:
        print(f'{c["name"]:22s} nit={c["nit"]:4d} nfev={c["nfev"]:5d} njev={c["njev"]:4d} '
              f'st={c["status"]} fun={c["fun"]:.3e} {c["message"]}')


if __name__ == "__main__":
    main()
