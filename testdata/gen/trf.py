"""Fixture generator for the trust-region reflective least-squares solver.

Run from the module root:

    ~/.local/bin/uv run --with scipy --with numpy python testdata/gen/trf.py

It writes testdata/trf.json: the problem data, then one record per case with
the options passed to scipy.optimize.least_squares(method='trf') and its
result. The residual and Jacobian formulas below are mirrored, operation for
operation, by trf_test.go so that both sides evaluate the same numbers.
"""

import json
import math

import numpy as np
import scipy
from scipy.optimize import least_squares

DATA = {
    "bard_y": [0.14, 0.18, 0.22, 0.25, 0.29, 0.32, 0.35, 0.39, 0.37, 0.58,
               0.73, 0.96, 1.34, 2.10, 4.39],
    "kowalik_y": [0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456,
                  0.0342, 0.0323, 0.0235, 0.0246],
    "kowalik_u": [4.0, 2.0, 1.0, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833,
                  0.0714, 0.0625],
    "osborne1_y": [0.844, 0.908, 0.932, 0.936, 0.925, 0.908, 0.881, 0.850,
                   0.818, 0.784, 0.751, 0.718, 0.685, 0.658, 0.628, 0.603,
                   0.580, 0.558, 0.538, 0.522, 0.506, 0.490, 0.478, 0.467,
                   0.457, 0.448, 0.438, 0.431, 0.424, 0.420, 0.414, 0.411,
                   0.406],
    "osborne2_y": [1.366, 1.191, 1.112, 1.013, 0.991, 0.885, 0.831, 0.847,
                   0.786, 0.725, 0.746, 0.679, 0.608, 0.655, 0.616, 0.606,
                   0.602, 0.626, 0.651, 0.724, 0.649, 0.649, 0.694, 0.644,
                   0.624, 0.661, 0.612, 0.558, 0.533, 0.495, 0.500, 0.423,
                   0.395, 0.375, 0.372, 0.391, 0.396, 0.405, 0.428, 0.429,
                   0.523, 0.562, 0.607, 0.653, 0.672, 0.708, 0.633, 0.668,
                   0.645, 0.632, 0.591, 0.559, 0.597, 0.625, 0.739, 0.710,
                   0.729, 0.720, 0.636, 0.581, 0.428, 0.292, 0.162, 0.098,
                   0.054],
}

# Synthetic exponential data a + b*exp(c*t) with 10% gross outliers.
_rng = np.random.RandomState(0)
_t = np.linspace(0.0, 10.0, 40)
_y = 0.5 + 2.0 * np.exp(-1.0 * _t)
_y = _y + 0.1 * _rng.standard_normal(_t.size)
_out = _rng.choice(_t.size, 4, replace=False)
_y[_out] += 4.0 * _rng.uniform(1.0, 2.0, 4)
DATA["expfit_t"] = _t.tolist()
DATA["expfit_y"] = _y.tolist()
_yc = 0.5 + 2.0 * np.exp(-1.0 * _t) + 0.05 * _rng.standard_normal(_t.size)
DATA["clean_y"] = _yc.tolist()

A = {k: np.array(v) for k, v in DATA.items()}


def rosenbrock(x):
    return np.array([10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]])


def rosenbrock_j(x):
    return np.array([[-20.0 * x[0], 10.0], [-1.0, 0.0]])


def _theta(x):
    th = math.atan(x[1] / x[0]) / (2.0 * math.pi)
    if x[0] < 0:
        th += 0.5
    return th


def helical(x):
    r = math.sqrt(x[0] * x[0] + x[1] * x[1])
    return np.array([10.0 * (x[2] - 10.0 * _theta(x)), 10.0 * (r - 1.0), x[2]])


def helical_j(x):
    r2 = x[0] * x[0] + x[1] * x[1]
    r = math.sqrt(r2)
    c = 100.0 / (2.0 * math.pi * r2)
    return np.array([[c * x[1], -c * x[0], 10.0],
                     [10.0 * x[0] / r, 10.0 * x[1] / r, 0.0],
                     [0.0, 0.0, 1.0]])


S5 = math.sqrt(5.0)
S10 = math.sqrt(10.0)


def powell(x):
    a = x[1] - 2.0 * x[2]
    b = x[0] - x[3]
    return np.array([x[0] + 10.0 * x[1], S5 * (x[2] - x[3]), a * a, S10 * b * b])


def powell_j(x):
    a = x[1] - 2.0 * x[2]
    b = x[0] - x[3]
    return np.array([[1.0, 10.0, 0.0, 0.0],
                     [0.0, 0.0, S5, -S5],
                     [0.0, 2.0 * a, -4.0 * a, 0.0],
                     [2.0 * S10 * b, 0.0, 0.0, -2.0 * S10 * b]])


def freudenstein(x):
    return np.array([-13.0 + x[0] + ((5.0 - x[1]) * x[1] - 2.0) * x[1],
                     -29.0 + x[0] + ((x[1] + 1.0) * x[1] - 14.0) * x[1]])


def freudenstein_j(x):
    return np.array([[1.0, (10.0 - 3.0 * x[1]) * x[1] - 2.0],
                     [1.0, (3.0 * x[1] + 2.0) * x[1] - 14.0]])


def bard(x):
    f = np.empty(15)
    for i in range(15):
        u = i + 1.0
        v = 15.0 - i
        w = min(u, v)
        f[i] = A["bard_y"][i] - (x[0] + u / (v * x[1] + w * x[2]))
    return f


def bard_j(x):
    J = np.empty((15, 3))
    for i in range(15):
        u = i + 1.0
        v = 15.0 - i
        w = min(u, v)
        den = v * x[1] + w * x[2]
        den2 = den * den
        J[i] = [-1.0, u * v / den2, u * w / den2]
    return J


def kowalik(x):
    f = np.empty(11)
    for i in range(11):
        u = A["kowalik_u"][i]
        num = u * u + u * x[1]
        den = u * u + u * x[2] + x[3]
        f[i] = A["kowalik_y"][i] - x[0] * num / den
    return f


def kowalik_j(x):
    J = np.empty((11, 4))
    for i in range(11):
        u = A["kowalik_u"][i]
        num = u * u + u * x[1]
        den = u * u + u * x[2] + x[3]
        den2 = den * den
        J[i] = [-num / den, -x[0] * u / den, x[0] * num * u / den2, x[0] * num / den2]
    return J


def box3d(x):
    f = np.empty(10)
    for i in range(10):
        t = 0.1 * (i + 1)
        f[i] = (math.exp(-t * x[0]) - math.exp(-t * x[1])
                - x[2] * (math.exp(-t) - math.exp(-10.0 * t)))
    return f


def box3d_j(x):
    J = np.empty((10, 3))
    for i in range(10):
        t = 0.1 * (i + 1)
        J[i] = [-t * math.exp(-t * x[0]), t * math.exp(-t * x[1]),
                -(math.exp(-t) - math.exp(-10.0 * t))]
    return J


def jennrich(x):
    f = np.empty(10)
    for i in range(10):
        k = i + 1.0
        f[i] = 2.0 + 2.0 * k - (math.exp(k * x[0]) + math.exp(k * x[1]))
    return f


def jennrich_j(x):
    J = np.empty((10, 2))
    for i in range(10):
        k = i + 1.0
        J[i] = [-k * math.exp(k * x[0]), -k * math.exp(k * x[1])]
    return J


def brown_dennis(x):
    f = np.empty(20)
    for i in range(20):
        t = (i + 1) / 5.0
        a = x[0] + t * x[1] - math.exp(t)
        b = x[2] + x[3] * math.sin(t) - math.cos(t)
        f[i] = a * a + b * b
    return f


def brown_dennis_j(x):
    J = np.empty((20, 4))
    for i in range(20):
        t = (i + 1) / 5.0
        a = x[0] + t * x[1] - math.exp(t)
        b = x[2] + x[3] * math.sin(t) - math.cos(t)
        J[i] = [2.0 * a, 2.0 * a * t, 2.0 * b, 2.0 * b * math.sin(t)]
    return J


def osborne1(x):
    f = np.empty(33)
    for i in range(33):
        t = 10.0 * i
        f[i] = A["osborne1_y"][i] - (x[0] + x[1] * math.exp(-t * x[3])
                                     + x[2] * math.exp(-t * x[4]))
    return f


def osborne1_j(x):
    J = np.empty((33, 5))
    for i in range(33):
        t = 10.0 * i
        e4 = math.exp(-t * x[3])
        e5 = math.exp(-t * x[4])
        J[i] = [-1.0, -e4, -e5, t * x[1] * e4, t * x[2] * e5]
    return J


def biggs(x):
    f = np.empty(13)
    for i in range(13):
        t = 0.1 * (i + 1)
        y = math.exp(-t) - 5.0 * math.exp(-10.0 * t) + 3.0 * math.exp(-4.0 * t)
        f[i] = (x[2] * math.exp(-t * x[0]) - x[3] * math.exp(-t * x[1])
                + x[5] * math.exp(-t * x[4]) - y)
    return f


def biggs_j(x):
    J = np.empty((13, 6))
    for i in range(13):
        t = 0.1 * (i + 1)
        e1 = math.exp(-t * x[0])
        e2 = math.exp(-t * x[1])
        e5 = math.exp(-t * x[4])
        J[i] = [-t * x[2] * e1, t * x[3] * e2, e1, -e2, -t * x[5] * e5, e5]
    return J


def osborne2(x):
    f = np.empty(65)
    for i in range(65):
        t = i / 10.0
        d9 = t - x[8]
        d10 = t - x[9]
        d11 = t - x[10]
        f[i] = A["osborne2_y"][i] - (x[0] * math.exp(-t * x[4])
                                     + x[1] * math.exp(-d9 * d9 * x[5])
                                     + x[2] * math.exp(-d10 * d10 * x[6])
                                     + x[3] * math.exp(-d11 * d11 * x[7]))
    return f


def osborne2_j(x):
    J = np.empty((65, 11))
    for i in range(65):
        t = i / 10.0
        d9 = t - x[8]
        d10 = t - x[9]
        d11 = t - x[10]
        e1 = math.exp(-t * x[4])
        e2 = math.exp(-d9 * d9 * x[5])
        e3 = math.exp(-d10 * d10 * x[6])
        e4 = math.exp(-d11 * d11 * x[7])
        J[i] = [-e1, -e2, -e3, -e4, t * x[0] * e1,
                d9 * d9 * x[1] * e2, d10 * d10 * x[2] * e3, d11 * d11 * x[3] * e4,
                -2.0 * x[1] * x[5] * d9 * e2, -2.0 * x[2] * x[6] * d10 * e3,
                -2.0 * x[3] * x[7] * d11 * e4]
    return J


def watson(x):
    n = 6
    f = np.empty(31)
    for i in range(29):
        t = (i + 1) / 29.0
        s1 = 0.0
        tj = 1.0
        for j in range(1, n):
            s1 += j * x[j] * tj
            tj *= t
        s2 = 0.0
        tj = 1.0
        for j in range(n):
            s2 += x[j] * tj
            tj *= t
        f[i] = s1 - s2 * s2 - 1.0
    f[29] = x[0]
    f[30] = x[1] - x[0] * x[0] - 1.0
    return f


def watson_j(x):
    n = 6
    J = np.zeros((31, n))
    for i in range(29):
        t = (i + 1) / 29.0
        s2 = 0.0
        tj = 1.0
        for j in range(n):
            s2 += x[j] * tj
            tj *= t
        tj = 1.0
        tjm = 0.0
        for j in range(n):
            J[i, j] = j * tjm - 2.0 * s2 * tj
            tjm = tj
            tj *= t
    J[29, 0] = 1.0
    J[30, 0] = -2.0 * x[0]
    J[30, 1] = 1.0
    return J


def expfit(x, y=None):
    t = A["expfit_t"]
    yy = A["expfit_y"] if y is None else y
    f = np.empty(t.size)
    for i in range(t.size):
        f[i] = x[0] + x[1] * math.exp(x[2] * t[i]) - yy[i]
    return f


def expfit_j(x):
    t = A["expfit_t"]
    J = np.empty((t.size, 3))
    for i in range(t.size):
        e = math.exp(x[2] * t[i])
        J[i] = [1.0, e, x[1] * t[i] * e]
    return J


def clean(x):
    return expfit(x, A["clean_y"])


def logfit(x):
    # Steps that leave x[0] > 0 make the residual infinite or NaN.
    with np.errstate(all="ignore"):
        return np.array([np.log(x[0]) + 3.0, x[1] - 3.0])


def logfit_j(x):
    return np.array([[1.0 / x[0], 0.0], [0.0, 1.0]])


def circle(x):
    # Fewer residuals than variables.
    return np.array([x[0] * x[0] + x[1] * x[1] - 1.0])


def circle_j(x):
    return np.array([[2.0 * x[0], 2.0 * x[1]]])


def zerocol(x):
    # The first Jacobian column vanishes at x0 = (0, 0).
    return np.array([x[0] * x[0] + x[1] - 3.0, x[1] - 1.0])


def zerocol_j(x):
    return np.array([[2.0 * x[0], 1.0], [0.0, 1.0]])


def balanced(x):
    # The gradient is exactly zero at x0 = 1: the first step is zero.
    return np.array([x[0] - 1.0, 1.0])


def balanced_j(x):
    return np.array([[1.0], [0.0]])


PROBLEMS = {
    "rosenbrock": (rosenbrock, rosenbrock_j, [-1.2, 1.0]),
    "helical": (helical, helical_j, [-1.0, 0.0, 0.0]),
    "powell": (powell, powell_j, [3.0, -1.0, 0.0, 1.0]),
    "freudenstein": (freudenstein, freudenstein_j, [0.5, -2.0]),
    "bard": (bard, bard_j, [1.0, 1.0, 1.0]),
    "kowalik": (kowalik, kowalik_j, [0.25, 0.39, 0.415, 0.39]),
    "box3d": (box3d, box3d_j, [0.0, 10.0, 20.0]),
    "jennrich": (jennrich, jennrich_j, [0.3, 0.4]),
    "brown_dennis": (brown_dennis, brown_dennis_j, [25.0, 5.0, -5.0, -1.0]),
    "osborne1": (osborne1, osborne1_j, [0.5, 1.5, -1.0, 0.01, 0.02]),
    "biggs": (biggs, biggs_j, [1.0, 2.0, 1.0, 1.0, 1.0, 1.0]),
    "osborne2": (osborne2, osborne2_j,
                 [1.3, 0.65, 0.65, 0.7, 0.6, 3.0, 5.0, 7.0, 2.0, 4.5, 5.5]),
    "watson": (watson, watson_j, [0.0] * 6),
    "expfit": (expfit, expfit_j, [1.0, 1.0, 0.0]),
    "clean": (clean, expfit_j, [1.0, 1.0, 0.0]),
    "logfit": (logfit, logfit_j, [5.0, 0.0]),
    "circle": (circle, circle_j, [2.0, 3.0]),
    "zerocol": (zerocol, zerocol_j, [0.0, 0.0]),
    "balanced": (balanced, balanced_j, [1.0]),
}

MGH = ["rosenbrock", "helical", "powell", "freudenstein", "bard", "kowalik",
       "box3d", "jennrich", "brown_dennis", "osborne1", "biggs", "osborne2",
       "watson"]

INF = math.inf

# Bounds chosen to cut off each unconstrained optimum (None = infinite).
BOUNDED = {
    "rosenbrock": ([None, None], [0.5, None]),
    "helical": ([-1.0, -1.0, -1.0], [0.8, 1.0, 1.0]),
    "powell": ([0.1, -20.0, -1.0, -1.0], [100.0, 20.0, 1.0, 50.0]),
    "freudenstein": ([0.0, -3.0], [20.0, 3.5]),
    "bard": ([0.1, 0.0, 0.0], [None, 1.5, None]),
    "kowalik": ([0.0, 0.0, 0.0, 0.2], [1.0, 1.0, 1.0, 1.0]),
    "box3d": ([0.0, 5.0, 0.0], [2.0, 10.0, 20.0]),
    "jennrich": ([0.26, 0.0], [10.0, 20.0]),
    "brown_dennis": ([-10.0, 0.0, -100.0, -20.0], [100.0, 15.0, 0.0, 0.2]),
    "osborne1": ([0.0, 0.0, -10.0, 0.0, 0.0], [1.0, 1.8, 10.0, 1.0, 1.0]),
    "biggs": ([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [2.0, 8.0, 1.0, 7.0, 5.0, 5.0]),
    "osborne2": ([1.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 4.0, 5.0],
                 [1.3, 2.0, 2.0, 2.0, 2.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
    "watson": ([-1.0, 0.0, -1.0, -1.0, -1.0, 0.0], [0.0, 0.9, 1.0, 1.0, 1.0, 1.0]),
}


def bounds_arrays(lb, ub):
    return (np.array([-INF if v is None else v for v in lb]),
            np.array([INF if v is None else v for v in ub]))


def run(name, prob, jac="analytic", lb=None, ub=None, x0=None, **kw):
    fun, jfun, px0 = PROBLEMS[prob]
    x0 = px0 if x0 is None else x0
    args = dict(method="trf")
    args["jac"] = jfun if jac == "analytic" else jac
    if lb is not None:
        args["bounds"] = bounds_arrays(lb, ub)
    args.update(kw)
    r = least_squares(fun, np.array(x0, dtype=float), **args)
    case = {
        "name": name,
        "problem": prob,
        "x0": list(map(float, x0)),
        "jac": jac,
        "lb": lb,
        "ub": ub,
        "options": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in kw.items()},
        "x": r.x.tolist(),
        "cost": float(r.cost),
        "fun": r.fun.tolist(),
        "grad": r.grad.tolist(),
        "optimality": float(r.optimality),
        "active_mask": [int(v) for v in r.active_mask],
        "nfev": int(r.nfev),
        "njev": int(r.njev),
        "status": int(r.status),
        "message": r.message,
    }
    return case


def main():
    cases = []
    for p in MGH:
        cases.append(run(p + "/analytic", p))
    for p in MGH:
        cases.append(run(p + "/2-point", p, jac="2-point"))
    for p in ["rosenbrock", "bard", "osborne1"]:
        cases.append(run(p + "/3-point", p, jac="3-point"))
    cases.append(run("kowalik/diff_step", "kowalik", jac="2-point", diff_step=1e-6))
    for p in MGH:
        lb, ub = BOUNDED[p]
        cases.append(run(p + "/bounded", p, lb=lb, ub=ub))
    for p in ["rosenbrock", "box3d", "osborne1", "kowalik"]:
        lb, ub = BOUNDED[p]
        cases.append(run(p + "/bounded/2-point", p, jac="2-point", lb=lb, ub=ub))
    cases.append(run("bard/bounded/3-point", "bard", jac="3-point",
                     lb=BOUNDED["bard"][0], ub=BOUNDED["bard"][1]))
    # x0 on a bound: least_squares moves it strictly inside first.
    cases.append(run("rosenbrock/x0-on-bound", "rosenbrock",
                     lb=[-1.2, None], ub=[0.5, 1.0]))
    cases.append(run("helical/x0-on-bound-fd", "helical", jac="2-point",
                     lb=[-1.0, 0.0, 0.0], ub=[0.8, 1.0, 1.0]))
    cases.append(run("clean/box", "clean", x0=[0.1, 1.0, -0.5], lb=[0.0, 0.0, -5.0],
                     ub=[0.3, 10.0, 0.0]))
    cases.append(run("clean/box/2-point", "clean", jac="2-point", x0=[0.1, 1.0, -0.5],
                     lb=[0.0, 0.0, -5.0], ub=[0.3, 10.0, 0.0]))
    for loss in ["linear", "soft_l1", "huber", "cauchy", "arctan"]:
        for fs in [0.1, 1.0]:
            cases.append(run(f"expfit/{loss}/{fs}", "expfit", loss=loss, f_scale=fs))
    for loss in ["soft_l1", "huber", "cauchy", "arctan"]:
        cases.append(run(f"expfit/{loss}/0.1/bounded", "expfit", loss=loss,
                         f_scale=0.1, x0=[0.1, 1.0, -0.5], lb=[0.0, 0.0, -5.0], ub=[0.4, 10.0, 0.0]))
    cases.append(run("expfit/huber/2-point", "expfit", jac="2-point", loss="huber",
                     f_scale=0.5))
    cases.append(run("rosenbrock/x_scale", "rosenbrock", x_scale=[1.0, 10.0]))
    cases.append(run("kowalik/x_scale", "kowalik", x_scale=[0.1, 1.0, 1.0, 1.0]))
    for p in ["bard", "osborne1", "box3d", "brown_dennis"]:
        cases.append(run(p + "/x_scale_jac", p, x_scale="jac"))
    cases.append(run("osborne1/x_scale_jac/bounded", "osborne1", x_scale="jac",
                     lb=BOUNDED["osborne1"][0], ub=BOUNDED["osborne1"][1]))
    cases.append(run("box3d/x_scale/bounded", "box3d", x_scale=[1.0, 10.0, 1.0],
                     lb=BOUNDED["box3d"][0], ub=BOUNDED["box3d"][1]))
    cases.append(run("expfit/cauchy/x_scale_jac", "expfit", x_scale="jac",
                     loss="cauchy", f_scale=0.1))
    # Evaluation budget exhausted.
    cases.append(run("osborne2/max_nfev", "osborne2", max_nfev=5))
    cases.append(run("osborne2/bounded/max_nfev", "osborne2", max_nfev=5,
                     lb=BOUNDED["osborne2"][0], ub=BOUNDED["osborne2"][1]))
    # Tolerances: a single criterion enabled.
    cases.append(run("bard/gtol-only", "bard", ftol=None, xtol=None, gtol=1e-6))
    cases.append(run("bard/xtol-only", "bard", ftol=None, gtol=None, xtol=1e-6))
    cases.append(run("bard/ftol-only", "bard", xtol=None, gtol=None, ftol=1e-6))
    cases.append(run("box3d/bounded/xtol-zero", "box3d", xtol=None,
                     lb=BOUNDED["box3d"][0], ub=BOUNDED["box3d"][1]))
    # Non-finite residuals at a trial point.
    cases.append(run("logfit", "logfit"))
    cases.append(run("logfit/bounded", "logfit", lb=[None, -10.0], ub=[None, 10.0]))
    # Fewer residuals than variables.
    cases.append(run("circle", "circle"))
    cases.append(run("circle/bounded", "circle", lb=[0.0, 0.0], ub=[5.0, 5.0]))
    # A zero Jacobian column under x_scale='jac'.
    cases.append(run("zerocol/x_scale_jac", "zerocol", x_scale="jac"))
    # A zero step: predicted and actual reductions both vanish.
    cases.append(run("balanced/gtol-off", "balanced", gtol=None))
    # Bounds tighter than make_strictly_feasible's relative step.
    cases.append(run("rosenbrock/tight", "rosenbrock", x0=[0.0, 1.0],
                     lb=[0.0, None], ub=[1e-11, None]))
    out = {"scipy": scipy.__version__, "numpy": np.__version__, "data": DATA,
           "cases": cases}
    with open("testdata/trf.json", "w") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")


if __name__ == "__main__":
    main()
