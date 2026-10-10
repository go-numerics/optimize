"""Fixtures for the L-BFGS-B port: scipy.optimize.minimize(method='L-BFGS-B').

Run from the repository root:
    ~/.local/bin/uv run --with scipy --with numpy python testdata/gen/lbfgsb.py
It writes testdata/lbfgsb.json. The Go test defines the same objectives by
name; keep the two in sync.
"""
import json
import math

import numpy as np
import scipy
from scipy.optimize import minimize

INF = math.inf


# --- objectives and gradients -------------------------------------------

def rosen2(x):
    return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2


def rosen2_g(x):
    return np.array([-400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0]),
                     200.0 * (x[1] - x[0] ** 2)])


def extrosen(x):
    f = 0.0
    for i in range(0, len(x), 2):
        f += 100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
    return f


def extrosen_g(x):
    g = np.zeros(len(x))
    for i in range(0, len(x), 2):
        t = x[i + 1] - x[i] ** 2
        g[i] = -400.0 * x[i] * t - 2.0 * (1.0 - x[i])
        g[i + 1] = 200.0 * t
    return g


def beale(x):
    a = 1.5 - x[0] + x[0] * x[1]
    b = 2.25 - x[0] + x[0] * x[1] ** 2
    c = 2.625 - x[0] + x[0] * x[1] ** 3
    return a * a + b * b + c * c


def beale_g(x):
    a = 1.5 - x[0] + x[0] * x[1]
    b = 2.25 - x[0] + x[0] * x[1] ** 2
    c = 2.625 - x[0] + x[0] * x[1] ** 3
    return np.array([2 * a * (x[1] - 1) + 2 * b * (x[1] ** 2 - 1) + 2 * c * (x[1] ** 3 - 1),
                     2 * a * x[0] + 2 * b * 2 * x[0] * x[1] + 2 * c * 3 * x[0] * x[1] ** 2])


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def himmelblau_g(x):
    a = x[0] ** 2 + x[1] - 11
    b = x[0] + x[1] ** 2 - 7
    return np.array([4 * a * x[0] + 2 * b, 2 * a + 4 * b * x[1]])


def wood(x):
    return (100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2 + 90 * (x[3] - x[2] ** 2) ** 2
            + (1 - x[2]) ** 2 + 10.1 * ((x[1] - 1) ** 2 + (x[3] - 1) ** 2)
            + 19.8 * (x[1] - 1) * (x[3] - 1))


def wood_g(x):
    return np.array([
        -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
        200 * (x[1] - x[0] ** 2) + 20.2 * (x[1] - 1) + 19.8 * (x[3] - 1),
        -360 * x[2] * (x[3] - x[2] ** 2) - 2 * (1 - x[2]),
        180 * (x[3] - x[2] ** 2) + 20.2 * (x[3] - 1) + 19.8 * (x[1] - 1),
    ])


def powell(x):
    return ((x[0] + 10 * x[1]) ** 2 + 5 * (x[2] - x[3]) ** 2 + (x[1] - 2 * x[2]) ** 4
            + 10 * (x[0] - x[3]) ** 4)


def powell_g(x):
    a = x[0] + 10 * x[1]
    b = x[2] - x[3]
    c = x[1] - 2 * x[2]
    d = x[0] - x[3]
    return np.array([2 * a + 40 * d ** 3, 20 * a + 4 * c ** 3,
                     10 * b - 8 * c ** 3, -10 * b - 40 * d ** 3])


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
    return float(np.dot(x, x))


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
    "rosen2": (rosen2, rosen2_g),
    "extrosen": (extrosen, extrosen_g),
    "beale": (beale, beale_g),
    "himmelblau": (himmelblau, himmelblau_g),
    "wood": (wood, wood_g),
    "powell": (powell, powell_g),
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
    case("rosen2" + sfx, "rosen2", [-1.2, 1.0], analytic)
    case("extrosen10" + sfx, "extrosen", alt(10), analytic)
    case("extrosen100" + sfx, "extrosen", alt(100), analytic)
    case("beale" + sfx, "beale", [1.0, 1.0], analytic)
    case("himmelblau" + sfx, "himmelblau", [0.0, 0.0], analytic)
    case("wood" + sfx, "wood", [-3.0, -1.0, -3.0, -1.0], analytic)
    case("powell" + sfx, "powell", [3.0, -1.0, 0.0, 1.0], analytic)
    case("quad50" + sfx, "quad", [0.0] * 50, analytic)
    case("rosen2_box" + sfx, "rosen2", [-1.2, 1.0], analytic, [(-1.5, 0.8)] * 2)
    case("extrosen10_box" + sfx, "extrosen", alt(10), analytic, [(-1.5, 0.8)] * 10)
    case("quad50_mixed" + sfx, "quad", [0.0] * 50, analytic, mixed_bounds(50))

case("rosen2_lower", "rosen2", [-1.2, 1.0], True, [(-INF, 0.5), (1.2, INF)])
case("beale_box_m3", "beale", [1.0, 1.0], True, [(-4.5, 4.5), (-0.5, 0.4)], maxcor=3)
case("extrosen10_maxiter", "extrosen", alt(10), True, None, maxiter=5)
case("rosen2_maxfun", "rosen2", [-1.2, 1.0], True, None, maxfun=10)
case("rosen2_maxfun_fd", "rosen2", [-1.2, 1.0], False, None, maxfun=10)
case("rosen2_maxls1", "rosen2", [-1.2, 1.0], True, None, maxls=1)
case("rosen2_tol", "rosen2", [-1.2, 1.0], True, None, tol=1e-3)
case("extrosen10_m3", "extrosen", alt(10), True, None, maxcor=3)
case("abnormal", "sphere_badg", [1.0, 2.0], True)
case("abnormal_gd", "tiny", [1.0], True, None, gtol=1e-300)
case("x0_optimal", "rosen2", [1.0, 1.0], True)


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
