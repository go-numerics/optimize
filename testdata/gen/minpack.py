"""Fixtures for the MINPACK port: SciPy's root(method='hybr'), fsolve,
least_squares(method='lm') and leastsq on the Moré–Garbow–Hillstrom test
problems.

The residuals and Jacobians are written with scalar Python floats, in the same
operation order as the Go test file (minpack_problems_test.go), so that both
sides evaluate the functions identically up to the libm (exp, sin, cos, atan);
squares are written x * x because the platform pow is not always correctly
rounded.

Run it with an x86_64 interpreter (on Apple silicon, under Rosetta):

    uv run --python cpython-3.12-macos-x86_64-none --with scipy==1.18.1 \
        --with numpy python testdata/gen/minpack.py

The arm64 macOS wheels of SciPy compile MINPACK with fused multiply-adds
(clang's default -ffp-contract=on), which changes the last bits of the
iterates; the x86_64 build has no FMA and is the reference the Go port (which
forbids fusion) reproduces bit for bit.
"""

import json
import math
import os
import warnings

import platform

import numpy as np
import scipy
from scipy.optimize import fsolve, least_squares, leastsq, root

# ---------------------------------------------------------------------------
# Problems. Each is (name, m, n, x0, fun(x) -> list, jac(x) -> list of rows).


def rosenbrock():
    def f(x):
        return [10 * (x[1] - x[0] * x[0]), 1 - x[0]]

    def j(x):
        return [[-20 * x[0], 10.0], [-1.0, 0.0]]

    return "rosenbrock", 2, 2, [-1.2, 1.0], f, j


def powell_singular():
    s5, s10 = math.sqrt(5), math.sqrt(10)

    def f(x):
        a = x[1] - 2 * x[2]
        b = x[0] - x[3]
        return [x[0] + 10 * x[1], s5 * (x[2] - x[3]), a * a, s10 * (b * b)]

    def j(x):
        a = 2 * (x[1] - 2 * x[2])
        b = 2 * s10 * (x[0] - x[3])
        return [[1.0, 10.0, 0.0, 0.0], [0.0, 0.0, s5, -s5],
                [0.0, a, -2 * a, 0.0], [b, 0.0, 0.0, -b]]

    return "powell_singular", 4, 4, [3.0, -1.0, 0.0, 1.0], f, j


def powell_badly_scaled():
    def f(x):
        return [1e4 * x[0] * x[1] - 1, math.exp(-x[0]) + math.exp(-x[1]) - 1.0001]

    def j(x):
        return [[1e4 * x[1], 1e4 * x[0]], [-math.exp(-x[0]), -math.exp(-x[1])]]

    return "powell_badly_scaled", 2, 2, [0.0, 1.0], f, j


def helical_valley():
    tpi = 8 * math.atan(1.0)

    def f(x):
        if x[0] == 0:
            theta = math.copysign(0.25, x[1])
        else:
            theta = math.atan(x[1] / x[0]) / tpi
            if x[0] < 0:
                theta += 0.5
        return [10 * (x[2] - 10 * theta), 10 * (math.sqrt(x[0] * x[0] + x[1] * x[1]) - 1), x[2]]

    def j(x):
        r2 = x[0] * x[0] + x[1] * x[1]
        r = math.sqrt(r2)
        t = 100 / (tpi * r2)
        return [[t * x[1], -t * x[0], 10.0], [10 * x[0] / r, 10 * x[1] / r, 0.0], [0.0, 0.0, 1.0]]

    return "helical_valley", 3, 3, [-1.0, 0.0, 0.0], f, j


def trigonometric(n=10):
    def f(x):
        s = 0.0
        for v in x:
            s += math.cos(v)
        return [n - s + (i + 1) * (1 - math.cos(x[i])) - math.sin(x[i]) for i in range(n)]

    def j(x):
        rows = []
        for i in range(n):
            row = [math.sin(x[k]) for k in range(n)]
            row[i] = (i + 2) * math.sin(x[i]) - math.cos(x[i])
            rows.append(row)
        return rows

    return "trigonometric", n, n, [1.0 / n] * n, f, j


def broyden_tridiagonal(n=10):
    def f(x):
        out = []
        for i in range(n):
            v = (3 - 2 * x[i]) * x[i] + 1
            if i > 0:
                v -= x[i - 1]
            if i < n - 1:
                v -= 2 * x[i + 1]
            out.append(v)
        return out

    def j(x):
        rows = []
        for i in range(n):
            row = [0.0] * n
            row[i] = 3 - 4 * x[i]
            if i > 0:
                row[i - 1] = -1.0
            if i < n - 1:
                row[i + 1] = -2.0
            rows.append(row)
        return rows

    return "broyden_tridiagonal", n, n, [-1.0] * n, f, j


def broyden_banded(n=10):
    ml, mu = 5, 1

    def f(x):
        out = []
        for i in range(n):
            v = x[i] * (2 + 5 * x[i] * x[i]) + 1
            for k in range(max(0, i - ml), min(n, i + mu + 1)):
                if k != i:
                    v -= x[k] * (1 + x[k])
            out.append(v)
        return out

    def j(x):
        rows = []
        for i in range(n):
            row = [0.0] * n
            for k in range(max(0, i - ml), min(n, i + mu + 1)):
                if k != i:
                    row[k] = -(1 + 2 * x[k])
            row[i] = 2 + 15 * x[i] * x[i]
            rows.append(row)
        return rows

    return "broyden_banded", n, n, [-1.0] * n, f, j


def chebyquad(n=5, m=None):
    m = m or n

    def f(x):
        out = [0.0] * m
        for v in x:
            t1, t2 = 1.0, 2 * v - 1
            t = 2 * t2
            for i in range(m):
                out[i] += t2
                th = t * t2 - t1
                t1, t2 = t2, th
        for i in range(m):
            out[i] /= n
            if (i + 1) % 2 == 0:
                out[i] += 1.0 / ((i + 1) * (i + 1) - 1)
        return out

    def j(x):
        rows = [[0.0] * n for _ in range(m)]
        for k, v in enumerate(x):
            t1, t2 = 1.0, 2 * v - 1
            t = 2 * t2
            s1, s2 = 0.0, 2.0
            for i in range(m):
                rows[i][k] = s2 / n
                th = 4 * t2 + t * s2 - s1
                s1, s2 = s2, th
                th = t * t2 - t1
                t1, t2 = t2, th
        return rows

    return "chebyquad", m, n, [(k + 1) / (n + 1) for k in range(n)], f, j


def linear_full_rank(n=5, m=10):
    def f(x):
        s = 0.0
        for v in x:
            s += v
        t = 2 * s / m + 1
        return [(x[i] if i < n else 0.0) - t for i in range(m)]

    def j(x):
        return [[(1.0 if i == k else 0.0) - 2.0 / m for k in range(n)] for i in range(m)]

    return "linear_full_rank", m, n, [1.0] * n, f, j


def linear_rank1(n=5, m=10):
    def f(x):
        s = 0.0
        for k in range(n):
            s += (k + 1) * x[k]
        return [(i + 1) * s - 1 for i in range(m)]

    def j(x):
        return [[float((i + 1) * (k + 1)) for k in range(n)] for i in range(m)]

    return "linear_rank1", m, n, [1.0] * n, f, j


def linear_rank1_zero(n=5, m=10):
    def f(x):
        s = 0.0
        for k in range(1, n - 1):
            s += (k + 1) * x[k]
        out = [i * s - 1 for i in range(m)]
        out[0] = -1.0
        out[m - 1] = -1.0
        return out

    def j(x):
        rows = []
        for i in range(m):
            row = [0.0] * n
            if 0 < i < m - 1:
                for k in range(1, n - 1):
                    row[k] = float(i * (k + 1))
            rows.append(row)
        return rows

    return "linear_rank1_zero", m, n, [1.0] * n, f, j


def freudenstein_roth():
    def f(x):
        return [-13 + x[0] + ((5 - x[1]) * x[1] - 2) * x[1],
                -29 + x[0] + ((x[1] + 1) * x[1] - 14) * x[1]]

    def j(x):
        return [[1.0, x[1] * (10 - 3 * x[1]) - 2], [1.0, x[1] * (3 * x[1] + 2) - 14]]

    return "freudenstein_roth", 2, 2, [0.5, -2.0], f, j


BARD_Y = [0.14, 0.18, 0.22, 0.25, 0.29, 0.32, 0.35, 0.39, 0.37, 0.58, 0.73, 0.96, 1.34, 2.10, 4.39]


def bard():
    def f(x):
        out = []
        for i in range(15):
            u = i + 1.0
            v = 15.0 - i
            w = min(u, v)
            out.append(BARD_Y[i] - (x[0] + u / (v * x[1] + w * x[2])))
        return out

    def j(x):
        rows = []
        for i in range(15):
            u = i + 1.0
            v = 15.0 - i
            w = min(u, v)
            d = v * x[1] + w * x[2]
            d2 = d * d
            rows.append([-1.0, u * v / d2, u * w / d2])
        return rows

    return "bard", 15, 3, [1.0, 1.0, 1.0], f, j


KO_Y = [0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246]
KO_U = [4.0, 2.0, 1.0, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625]


def kowalik_osborne():
    def f(x):
        out = []
        for i in range(11):
            u = KO_U[i]
            out.append(KO_Y[i] - x[0] * (u * u + u * x[1]) / (u * u + u * x[2] + x[3]))
        return out

    def j(x):
        rows = []
        for i in range(11):
            u = KO_U[i]
            num = u * u + u * x[1]
            den = u * u + u * x[2] + x[3]
            rows.append([-num / den, -x[0] * u / den, x[0] * num * u / (den * den), x[0] * num / (den * den)])
        return rows

    return "kowalik_osborne", 11, 4, [0.25, 0.39, 0.415, 0.39], f, j


MEYER_Y = [34780.0, 28610.0, 23650.0, 19630.0, 16370.0, 13720.0, 11540.0, 9744.0,
           8261.0, 7030.0, 6005.0, 5147.0, 4427.0, 3820.0, 3307.0, 2872.0]


def meyer():
    def f(x):
        return [x[0] * math.exp(x[1] / (45.0 + 5 * (i + 1) + x[2])) - MEYER_Y[i] for i in range(16)]

    def j(x):
        rows = []
        for i in range(16):
            d = 45.0 + 5 * (i + 1) + x[2]
            e = math.exp(x[1] / d)
            rows.append([e, x[0] * e / d, -x[0] * x[1] * e / (d * d)])
        return rows

    return "meyer", 16, 3, [0.02, 4000.0, 250.0], f, j


def watson(n):
    def f(x):
        out = []
        for i in range(29):
            t = (i + 1) / 29.0
            s1 = 0.0
            dx = 1.0
            for k in range(1, n):
                s1 += k * x[k] * dx
                dx *= t
            s2 = 0.0
            dx = 1.0
            for k in range(n):
                s2 += x[k] * dx
                dx *= t
            out.append(s1 - s2 * s2 - 1)
        out.append(x[0])
        out.append(x[1] - x[0] * x[0] - 1)
        return out

    def j(x):
        rows = []
        for i in range(29):
            t = (i + 1) / 29.0
            s2 = 0.0
            dx = 1.0
            for k in range(n):
                s2 += x[k] * dx
                dx *= t
            row = []
            dx = 1.0
            for k in range(n):
                row.append(dx * (k / t - 2 * s2))
                dx *= t
            rows.append(row)
        rows.append([1.0] + [0.0] * (n - 1))
        rows.append([-2 * x[0], 1.0] + [0.0] * (n - 2))
        return rows

    return "watson%d" % n, 31, n, [0.0] * n, f, j


def box3d():
    def f(x):
        out = []
        for i in range(10):
            t = 0.1 * (i + 1)
            out.append(math.exp(-t * x[0]) - math.exp(-t * x[1]) - x[2] * (math.exp(-t) - math.exp(-10 * t)))
        return out

    def j(x):
        rows = []
        for i in range(10):
            t = 0.1 * (i + 1)
            rows.append([-t * math.exp(-t * x[0]), t * math.exp(-t * x[1]), -(math.exp(-t) - math.exp(-10 * t))])
        return rows

    return "box3d", 10, 3, [0.0, 10.0, 20.0], f, j


def jennrich_sampson():
    def f(x):
        return [2 + 2 * (i + 1.0) - (math.exp((i + 1) * x[0]) + math.exp((i + 1) * x[1])) for i in range(10)]

    def j(x):
        return [[-(i + 1) * math.exp((i + 1) * x[0]), -(i + 1) * math.exp((i + 1) * x[1])] for i in range(10)]

    return "jennrich_sampson", 10, 2, [0.3, 0.4], f, j


def brown_dennis():
    def f(x):
        out = []
        for i in range(20):
            t = (i + 1) / 5.0
            a = x[0] + t * x[1] - math.exp(t)
            b = x[2] + x[3] * math.sin(t) - math.cos(t)
            out.append(a * a + b * b)
        return out

    def j(x):
        rows = []
        for i in range(20):
            t = (i + 1) / 5.0
            a = x[0] + t * x[1] - math.exp(t)
            b = x[2] + x[3] * math.sin(t) - math.cos(t)
            rows.append([2 * a, 2 * t * a, 2 * b, 2 * math.sin(t) * b])
        return rows

    return "brown_dennis", 20, 4, [25.0, 5.0, -5.0, -1.0], f, j


OSB1_Y = [0.844, 0.908, 0.932, 0.936, 0.925, 0.908, 0.881, 0.850, 0.818, 0.784, 0.751,
          0.718, 0.685, 0.658, 0.628, 0.603, 0.580, 0.558, 0.538, 0.522, 0.506, 0.490,
          0.478, 0.467, 0.457, 0.448, 0.438, 0.431, 0.424, 0.420, 0.414, 0.411, 0.406]


def osborne1():
    def f(x):
        out = []
        for i in range(33):
            t = 10.0 * i
            out.append(OSB1_Y[i] - (x[0] + x[1] * math.exp(-t * x[3]) + x[2] * math.exp(-t * x[4])))
        return out

    def j(x):
        rows = []
        for i in range(33):
            t = 10.0 * i
            e4 = math.exp(-t * x[3])
            e5 = math.exp(-t * x[4])
            rows.append([-1.0, -e4, -e5, t * x[1] * e4, t * x[2] * e5])
        return rows

    return "osborne1", 33, 5, [0.5, 1.5, -1.0, 0.01, 0.02], f, j


def biggs_exp6():
    def f(x):
        out = []
        for i in range(13):
            t = 0.1 * (i + 1)
            y = math.exp(-t) - 5 * math.exp(-10 * t) + 3 * math.exp(-4 * t)
            out.append(x[2] * math.exp(-t * x[0]) - x[3] * math.exp(-t * x[1]) + x[5] * math.exp(-t * x[4]) - y)
        return out

    def j(x):
        rows = []
        for i in range(13):
            t = 0.1 * (i + 1)
            e1 = math.exp(-t * x[0])
            e2 = math.exp(-t * x[1])
            e5 = math.exp(-t * x[4])
            rows.append([-t * x[2] * e1, t * x[3] * e2, e1, -e2, -t * x[5] * e5, e5])
        return rows

    return "biggs_exp6", 13, 6, [1.0, 2.0, 1.0, 1.0, 1.0, 1.0], f, j


OSB2_Y = [1.366, 1.191, 1.112, 1.013, 0.991, 0.885, 0.831, 0.847, 0.786, 0.725, 0.746,
          0.679, 0.608, 0.655, 0.616, 0.606, 0.602, 0.626, 0.651, 0.724, 0.649, 0.649,
          0.694, 0.644, 0.624, 0.661, 0.612, 0.558, 0.533, 0.495, 0.500, 0.423, 0.395,
          0.375, 0.372, 0.391, 0.396, 0.405, 0.428, 0.429, 0.523, 0.562, 0.607, 0.653,
          0.672, 0.708, 0.633, 0.668, 0.645, 0.632, 0.591, 0.559, 0.597, 0.625, 0.739,
          0.710, 0.729, 0.720, 0.636, 0.581, 0.428, 0.292, 0.162, 0.098, 0.054]


def osborne2():
    def f(x):
        out = []
        for i in range(65):
            t = i / 10.0
            a = t - x[8]
            b = t - x[9]
            c = t - x[10]
            out.append(OSB2_Y[i] - (x[0] * math.exp(-t * x[4]) + x[1] * math.exp(-a * a * x[5])
                                    + x[2] * math.exp(-b * b * x[6]) + x[3] * math.exp(-c * c * x[7])))
        return out

    def j(x):
        rows = []
        for i in range(65):
            t = i / 10.0
            a = t - x[8]
            b = t - x[9]
            c = t - x[10]
            e0 = math.exp(-t * x[4])
            e1 = math.exp(-a * a * x[5])
            e2 = math.exp(-b * b * x[6])
            e3 = math.exp(-c * c * x[7])
            rows.append([-e0, -e1, -e2, -e3, t * x[0] * e0, a * a * x[1] * e1, b * b * x[2] * e2,
                         c * c * x[3] * e3, -2 * a * x[5] * x[1] * e1, -2 * b * x[6] * x[2] * e2,
                         -2 * c * x[7] * x[3] * e3])
        return rows

    return "osborne2", 65, 11, [1.3, 0.65, 0.65, 0.7, 0.6, 3.0, 5.0, 7.0, 2.0, 4.5, 5.5], f, j


def no_root():
    def f(x):
        return [x[0] * x[0] + 1]

    def j(x):
        return [[2 * x[0]]]

    return "no_root", 1, 1, [1.0], f, j


# ---------------------------------------------------------------------------


def wrap(fun, jac):
    def f(x):
        return np.array(fun([float(v) for v in x]))

    def j(x):
        return np.array(jac([float(v) for v in x]))

    return f, j


def lst(a):
    return None if a is None else np.asarray(a, dtype=float).ravel().tolist()


def root_case(prob, use_jac, **opts):
    name, m, n, x0, fun, jac = prob
    f, j = wrap(fun, jac)
    x0 = list(x0)
    res = root(f, x0, jac=j if use_jac else None, method="hybr", options=dict(opts))
    fopts = {"xtol": opts.get("xtol", 1.49012e-8), "maxfev": opts.get("maxfev", 0),
             "epsfcn": opts.get("eps"), "factor": opts.get("factor", 100), "diag": opts.get("diag")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, info, ier, mesg = fsolve(f, x0, fprime=j if use_jac else None, full_output=True, **fopts)
    return {
        "problem": name, "jac": use_jac, "options": {k: (lst(v) if k == "diag" else v) for k, v in opts.items()},
        "x": lst(res.x), "fun": lst(res.fun), "fjac": lst(res.fjac), "r": lst(res.r), "qtf": lst(res.qtf),
        "nfev": int(res.nfev), "njev": int(res.get("njev", 0)), "status": int(res.status),
        "success": bool(res.success), "message": res.message,
        "fsolve_nfev": int(info["nfev"]), "fsolve_ier": int(ier), "fsolve_mesg": mesg,
        "fsolve_fjac": lst(info["fjac"]),
    }


def lm_case(prob, use_jac, **opts):
    name, m, n, x0, fun, jac = prob
    f, j = wrap(fun, jac)
    kw = dict(opts)
    res = least_squares(f, list(x0), jac=j if use_jac else "2-point", method="lm", **kw)
    return {
        "problem": name, "jac": use_jac, "options": {k: (lst(v) if k == "x_scale" else v) for k, v in opts.items()},
        "x": lst(res.x), "cost": float(res.cost), "fun": lst(res.fun), "grad": lst(res.grad),
        "jacobian": lst(res.jac), "optimality": float(res.optimality),
        "nfev": int(res.nfev), "njev": int(res.njev) if res.njev is not None else 0,
        "status": int(res.status), "success": bool(res.success), "message": res.message,
    }


def leastsq_case(prob, use_jac, **opts):
    name, m, n, x0, fun, jac = prob
    f, j = wrap(fun, jac)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, cov, info, mesg, ier = leastsq(f, list(x0), Dfun=j if use_jac else None, full_output=True, **opts)
    return {
        "problem": name, "jac": use_jac, "options": {k: (lst(v) if k == "diag" else v) for k, v in opts.items()},
        "x": lst(x), "cov_x": lst(cov), "nfev": int(info["nfev"]), "njev": int(info.get("njev", 0)),
        "fvec": lst(info["fvec"]), "fjac": lst(info["fjac"]), "ipvt": [int(v) for v in info["ipvt"]],
        "qtf": lst(info["qtf"]), "ier": int(ier), "mesg": mesg,
    }


SYSTEMS = [rosenbrock(), powell_singular(), powell_badly_scaled(), helical_valley(), trigonometric(),
           broyden_tridiagonal(), broyden_banded(), chebyquad()]

LSQ = [linear_full_rank(), linear_rank1(), linear_rank1_zero(), rosenbrock(), helical_valley(),
       powell_singular(), freudenstein_roth(), bard(), kowalik_osborne(), meyer(), watson(6), watson(9),
       box3d(), jennrich_sampson(), brown_dennis(), osborne1(), biggs_exp6(), osborne2()]


def main():
    roots = []
    for prob in SYSTEMS:
        for use_jac in (False, True):
            roots.append(root_case(prob, use_jac))
    # Failures: no real root, and maxfev exhausted; options.
    roots.append(root_case(no_root(), False))
    roots.append(root_case(no_root(), True))
    nr = no_root()
    roots.append(root_case(("no_root_at_0",) + nr[1:3] + ([0.0],) + nr[4:], True))
    roots.append(root_case(rosenbrock(), False, maxfev=5))
    roots.append(root_case(trigonometric(), True, xtol=1e-12, factor=1.0, diag=[1.0] * 10))
    roots.append(root_case(powell_badly_scaled(), False, eps=1e-10, factor=10.0))

    lm = []
    sq = []
    for prob in LSQ:
        for use_jac in (False, True):
            lm.append(lm_case(prob, use_jac))
            sq.append(leastsq_case(prob, use_jac))
    lm.append(lm_case(bard(), True, x_scale=[1.0, 2.0, 3.0]))
    lm.append(lm_case(meyer(), False, diff_step=1e-6, max_nfev=20))
    lm.append(lm_case(osborne1(), True, ftol=1e-12, xtol=1e-12, gtol=1e-12))
    sq.append(leastsq_case(bard(), False, diag=[1.0, 2.0, 3.0], epsfcn=1e-12, factor=10.0))
    sq.append(leastsq_case(meyer(), True, maxfev=20))
    sq.append(leastsq_case(rosenbrock(), True, gtol=1e-3))
    sq.append(leastsq_case(rosenbrock(), True, ftol=0.0, xtol=0.0))
    sq.append(leastsq_case(bard(), True, ftol=0.0, xtol=0.0))
    sq.append(leastsq_case(kowalik_osborne(), False, ftol=0.0, xtol=0.0))

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "minpack.json")
    with open(out, "w") as fh:
        json.dump({"scipy": scipy.__version__, "numpy": np.__version__, "machine": platform.machine(),
                   "root": roots, "least_squares": lm, "leastsq": sq}, fh, indent=1)
        fh.write("\n")


if __name__ == "__main__":
    main()
