# Oracle for the global optimisers: runs SciPy on the problems below and
# prints the fixture as JSON on stdout. The Go harness prepends `METHOD = "..."`
# and feeds this file to `python -I -`.
#
# Every objective is written as a chain of single IEEE operations on Python
# floats (no libm for the "arith" family, no ** and no numpy), so that the Go
# twin in direct_funcs_test.go computes bit-identical values; the "trans"
# family calls libm (cos, exp, sqrt) and is compared with a tolerance.
# Outputs are bit patterns ("0x%016x") so NaN, Inf and signed zeros survive.
import json
import math
import platform
import struct
import sys

import numpy as np
import scipy
from scipy.optimize import Bounds, direct

NAN = float("nan")
INF = float("inf")


def hx(v):
    return "0x%016x" % struct.unpack("<Q", struct.pack("<d", float(v)))[0]


# ---------------------------------------------------------------- objectives


def shifted(n):
    c = [0.3 + 0.17 * i for i in range(n)]

    def f(x):
        s = 0.0
        for i in range(n):
            t = x[i] - c[i]
            s = s + t * t
        return s

    return f


def rosenbrock(n):
    def f(x):
        s = 0.0
        for i in range(n - 1):
            t = x[i + 1] - x[i] * x[i]
            a = 100.0 * (t * t)
            u = 1.0 - x[i]
            b = u * u
            s = s + (a + b)
        return s

    return f


def goldstein_price(x):
    a, b = x[0], x[1]
    s = a + b
    s = s + 1.0
    s2 = s * s
    p = 19.0
    p = p - 14.0 * a
    p = p + 3.0 * (a * a)
    p = p - 14.0 * b
    p = p + 6.0 * (a * b)
    p = p + 3.0 * (b * b)
    f1 = 1.0 + s2 * p
    d = 2.0 * a
    d = d - 3.0 * b
    d2 = d * d
    q = 18.0
    q = q - 32.0 * a
    q = q + 12.0 * (a * a)
    q = q + 48.0 * b
    q = q - 36.0 * (a * b)
    q = q + 27.0 * (b * b)
    f2 = 30.0 + d2 * q
    return f1 * f2


def six_hump_camel(x):
    a, b = x[0], x[1]
    a2 = a * a
    a4 = a2 * a2
    t = 4.0 - 2.1 * a2
    t = t + a4 / 3.0
    t1 = t * a2
    t2 = a * b
    b2 = b * b
    u = -4.0 + 4.0 * b2
    t3 = u * b2
    f = t1 + t2
    return f + t3


def styblinski_tang(n):
    def f(x):
        s = 0.0
        for i in range(n):
            x2 = x[i] * x[i]
            x4 = x2 * x2
            t = x4 - 16.0 * x2
            t = t + 5.0 * x[i]
            s = s + t
        return 0.5 * s

    return f


SHEKEL_A = [
    [4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7],
    [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6],
]
SHEKEL_C = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]


def shekel(m):
    def f(x):
        s = 0.0
        for i in range(m):
            d = SHEKEL_C[i]
            for j in range(4):
                t = x[j] - SHEKEL_A[i][j]
                d = d + t * t
            s = s + 1.0 / d
        return -s

    return f


def booth(x):
    a = x[0] + 2.0 * x[1]
    a = a - 7.0
    b = 2.0 * x[0]
    b = b + x[1]
    b = b - 5.0
    return a * a + b * b


def matyas(x):
    s = x[0] * x[0] + x[1] * x[1]
    t = 0.26 * s
    u = 0.48 * x[0]
    u = u * x[1]
    return t - u


def beale(x):
    a, b = x[0], x[1]
    xy = a * b
    y2 = b * b
    y3 = y2 * b
    t1 = 1.5 - a
    t1 = t1 + xy
    t2 = 2.25 - a
    t2 = t2 + a * y2
    t3 = 2.625 - a
    t3 = t3 + a * y3
    f = t1 * t1 + t2 * t2
    return f + t3 * t3


def branin(x):
    a, b = x[0], x[1]
    t = b - 5.1 / (4.0 * math.pi * math.pi) * (a * a)
    t = t + 5.0 / math.pi * a
    t = t - 6.0
    r = t * t
    s = 10.0 * (1.0 - 1.0 / (8.0 * math.pi)) * math.cos(a)
    return (r + s) + 10.0


H3_A = [[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]]
H3_C = [1, 1.2, 3, 3.2]
H3_P = [[0.3689, 0.1170, 0.2673], [0.4699, 0.4387, 0.7470],
        [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]]
H6_A = [[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]]
H6_P = [[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
        [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
        [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
        [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]]


def hartmann(A, P, n):
    def f(x):
        s = 0.0
        for i in range(4):
            e = 0.0
            for j in range(n):
                t = x[j] - P[i][j]
                e = e + A[i][j] * (t * t)
            s = s + H3_C[i] * math.exp(-e)
        return -s

    return f


def ackley(n):
    def f(x):
        s1 = 0.0
        s2 = 0.0
        for i in range(n):
            s1 = s1 + x[i] * x[i]
            s2 = s2 + math.cos(2.0 * math.pi * x[i])
        a = -0.2 * math.sqrt(s1 / n)
        return (-20.0 * math.exp(a) - math.exp(s2 / n)) + (20.0 + math.e)

    return f


def rastrigin(n):
    def f(x):
        s = 10.0 * n
        for i in range(n):
            s = s + (x[i] * x[i] - 10.0 * math.cos(2.0 * math.pi * x[i]))
        return s

    return f


def nan_region(x):
    s = shifted(2)(x)
    if x[0] > 0.1:
        return NAN
    return s


def inf_region(x):
    s = shifted(2)(x)
    if x[1] < 0.0:
        return INF
    return s


def neginf_region(x):
    s = shifted(2)(x)
    if x[0] > 3.0 and x[1] > 3.0:
        return -INF
    return s


def nan_center(x):
    if x[0] == 0.0 and x[1] == 0.0:
        return NAN
    return shifted(2)(x)


def nan_all(x):
    return NAN


# name -> (function, lower, upper, global minimum, family)
PROBLEMS = {}


def add(name, f, lo, hi, fglob, fam):
    PROBLEMS[name] = (f, lo, hi, fglob, fam)


for n in (1, 2, 3, 5, 10):
    add("shifted_sphere_%d" % n, shifted(n), [-5.0] * n, [5.0] * n, 0.0, "arith")
for n in (2, 5):
    add("rosenbrock_%d" % n, rosenbrock(n), [-2.048] * n, [2.048] * n, 0.0, "arith")
add("goldstein_price", goldstein_price, [-2.0, -2.0], [2.0, 2.0], 3.0, "arith")
add("six_hump_camel", six_hump_camel, [-3.0, -2.0], [3.0, 2.0], -1.0316284534898774, "arith")
for n in (2, 4):
    add("styblinski_tang_%d" % n, styblinski_tang(n), [-5.0] * n, [5.0] * n,
        -39.16616570377142 * n, "arith")
for m, fg in ((5, -10.15319967905823), (7, -10.40294056605128), (10, -10.53628372109013)):
    add("shekel_%d" % m, shekel(m), [0.0] * 4, [10.0] * 4, fg, "arith")
add("booth", booth, [-10.0, -10.0], [12.0, 12.0], 0.0, "arith")
add("matyas", matyas, [-10.0, -10.0], [12.0, 12.0], 0.0, "arith")
add("beale", beale, [-4.5, -4.5], [4.5, 4.5], 0.0, "arith")
add("branin", branin, [-5.0, 0.0], [10.0, 15.0], 0.397887357729738, "trans")
add("hartmann_3", hartmann(H3_A, H3_P, 3), [0.0] * 3, [1.0] * 3, -3.86278214782076, "trans")
add("hartmann_6", hartmann(H6_A, H6_P, 6), [0.0] * 6, [1.0] * 6, -3.32236801141551, "trans")
add("ackley_2", ackley(2), [-32.768] * 2, [30.0] * 2, 0.0, "trans")
add("rastrigin_2", rastrigin(2), [-5.12] * 2, [5.12] * 2, 0.0, "trans")
add("nan_region", nan_region, [-5.0] * 2, [5.0] * 2, 0.0, "arith")
add("inf_region", inf_region, [-5.0] * 2, [5.0] * 2, 0.0, "arith")
add("neginf_region", neginf_region, [-5.0] * 2, [5.0] * 2, 0.0, "arith")
add("nan_center", nan_center, [-5.0] * 2, [5.0] * 2, 0.0, "arith")
add("nan_all", nan_all, [-5.0] * 2, [5.0] * 2, 0.0, "arith")

# ------------------------------------------------------------------ variants

# tweaks to the SciPy keyword arguments; "FG" stands for the problem's f_min.
TWEAKS = [
    ("default", {}),
    ("len1e-3", {"len_tol": 1e-3}),
    ("len1e-2_vol0", {"len_tol": 1e-2, "vol_tol": 0.0}),
    ("eps1e-2", {"eps": 1e-2}),
    ("eps1e-6", {"eps": 1e-6}),
    ("epsjones", {"eps": -1e-4}),
    ("vol1e-4", {"vol_tol": 1e-4, "len_tol": 0.0}),
    ("nolen_fev100", {"vol_tol": 0.0, "len_tol": 0.0, "maxfun": 100}),
    ("fev1001", {"maxfun": 1001}),
    ("fev57", {"maxfun": 57}),
    ("fev1", {"maxfun": 1}),
    ("iter5", {"maxiter": 5}),
    ("iter2", {"maxiter": 2}),
    ("iter30_nolen", {"maxiter": 30, "vol_tol": 0.0, "len_tol": 0.0}),
    ("fmin", {"f_min": "FG"}),
    ("fmin_rtol1e-2", {"f_min": "FG", "f_min_rtol": 1e-2}),
    ("fmin_rtol1e-8", {"f_min": "FG", "f_min_rtol": 1e-8}),
]
# the special cases only run a few tweaks
SPECIAL = ("nan_region", "inf_region", "neginf_region", "nan_center", "nan_all")
SPECIAL_TWEAKS = ("default", "len1e-3", "iter5", "fev1001", "eps1e-2", "epsjones")
BIG = ("shifted_sphere_10", "hartmann_6")
BIG_TWEAKS = ("default", "len1e-3", "fev1001", "iter5", "fmin", "epsjones", "iter30_nolen")


def run_direct():
    cases = []
    for pname, (f, lo, hi, fglob, fam) in PROBLEMS.items():
        n = len(lo)
        for locally in (True, False):
            for tname, tw in TWEAKS:
                if pname in SPECIAL and tname not in SPECIAL_TWEAKS:
                    continue
                if pname in BIG and tname not in BIG_TWEAKS:
                    continue
                if pname == "nan_all" and tname == "default":
                    tw = {"maxiter": 8}
                kw = dict(tw)
                for k, v in kw.items():
                    if v == "FG":
                        kw[k] = fglob
                opts = dict(eps=1e-4, maxfun=1000 * n, maxiter=1000, f_min=-INF,
                            f_min_rtol=1e-4, vol_tol=1e-16, len_tol=1e-6)
                opts.update(kw)
                trace = []
                total = [0]

                def cb(x, trace=trace, total=total):
                    total[0] += 1
                    if len(trace) < 25:
                        trace.append([hx(v) for v in x])

                err = None
                res = None
                try:
                    res = direct(lambda x, f=f: f(x.tolist()), Bounds(lo, hi), eps=opts["eps"], maxfun=opts["maxfun"],
                                 maxiter=opts["maxiter"], locally_biased=locally,
                                 f_min=opts["f_min"], f_min_rtol=opts["f_min_rtol"],
                                 vol_tol=opts["vol_tol"], len_tol=opts["len_tol"], callback=cb)
                except Exception as e:  # recorded, never raised
                    err = "%s: %s" % (type(e).__name__, e)
                name = "%s/%s/%s" % (pname, "L" if locally else "orig", tname)
                case = dict(
                    name=name, problem=pname, family=fam, dim=n, lower=lo, upper=hi,
                    locally_biased=locally,
                    options=dict(
                        eps=opts["eps"], maxfun=opts["maxfun"], maxiter=opts["maxiter"],
                        f_min=None if opts["f_min"] == -INF else opts["f_min"],
                        f_min_rtol=opts["f_min_rtol"], vol_tol=opts["vol_tol"],
                        len_tol=opts["len_tol"]),
                    error=err,
                )
                if res is not None:
                    case.update(
                        x=[hx(v) for v in res.x], fun=hx(res.fun), nfev=int(res.nfev),
                        nit=int(res.nit), status=int(res.status), success=bool(res.success),
                        message=str(res.message), trace=trace, ncallbacks=total[0])
                cases.append(case)
    return cases


# SciPy's own documentation example (written with ** 4: pow, "trans" family).
def docstring_example():
    def styblinski_tang_pow(pos):
        x, y = pos
        return 0.5 * (x**4 - 16 * x**2 + 5 * x + y**4 - 16 * y**2 + 5 * y)

    out = []
    for tname, kw in (("default", {}), ("len1e-3", {"len_tol": 1e-3})):
        r = direct(styblinski_tang_pow, Bounds([-4.0, -4.0], [4.0, 4.0]), **kw)
        opts = dict(eps=1e-4, maxfun=2000, maxiter=1000, f_min=None, f_min_rtol=1e-4,
                    vol_tol=1e-16, len_tol=kw.get("len_tol", 1e-6))
        out.append(dict(
            name="docstring_pow/L/%s" % tname, problem="docstring_pow", family="trans", dim=2,
            lower=[-4.0, -4.0], upper=[4.0, 4.0], locally_biased=True, options=opts, error=None,
            x=[hx(v) for v in r.x], fun=hx(r.fun), nfev=int(r.nfev), nit=int(r.nit),
            status=int(r.status), success=bool(r.success), message=str(r.message),
            trace=[], ncallbacks=0))
    return out


# ------------------------------------------------- differential evolution

DE_PROBLEMS = ("shifted_sphere_2", "shifted_sphere_5", "rosenbrock_2", "goldstein_price",
               "six_hump_camel", "styblinski_tang_2", "shekel_5", "booth", "matyas", "beale")
DE_STRATEGIES = ("best1bin", "best1exp", "rand1bin", "rand1exp", "randtobest1bin",
                 "randtobest1exp", "currenttobest1bin", "currenttobest1exp", "best2bin",
                 "best2exp", "rand2bin", "rand2exp")
DE_DEFAULTS = dict(strategy="best1bin", maxiter=1000, popsize=15, tol=0.01, atol=0.0,
                   mutation=[0.5, 1.0], recombination=0.7, init="latinhypercube", init_pop=None,
                   x0=None, updating="immediate", polish=False, maxfun=None)


def de_one(name, pname, kind, seed, **over):
    from scipy.optimize._differentialevolution import DifferentialEvolutionSolver

    f, lo, hi, fglob, fam = PROBLEMS[pname]
    o = dict(DE_DEFAULTS)
    o.update(over)
    rng = np.random.default_rng(seed) if kind == "rng" else np.random.RandomState(seed)
    mut = tuple(o["mutation"]) if len(o["mutation"]) > 1 else o["mutation"][0]
    init = o["init_pop"] if o["init_pop"] is not None else o["init"]
    case = dict(name=name, problem=pname, family=fam, dim=len(lo), lower=lo, upper=hi,
                kind=kind, seed=seed, options=o, error=None)
    try:
        solver = DifferentialEvolutionSolver(
            lambda x: f(x.tolist()), list(zip(lo, hi)), strategy=o["strategy"],
            maxiter=o["maxiter"], popsize=o["popsize"], tol=o["tol"], atol=o["atol"],
            mutation=mut, recombination=o["recombination"], rng=rng, polish=o["polish"],
            init=init, updating=o["updating"], x0=o["x0"],
            maxfun=np.inf if o["maxfun"] is None else o["maxfun"])
        res = solver.solve()
    except Exception as e:  # recorded, never raised
        case["error"] = "%s: %s" % (type(e).__name__, e)
        return case
    case.update(
        x=[hx(v) for v in res.x], fun=hx(res.fun), nfev=int(res.nfev), nit=int(res.nit),
        success=bool(res.success), message=str(res.message),
        population=[[hx(v) for v in row] for row in res.population],
        population_energies=[hx(v) for v in res.population_energies])
    return case


def run_de():
    cases = []
    for kind, seed in (("rng", 7), ("seed", 7)):
        for pname in ("rosenbrock_2", "styblinski_tang_2"):
            for st in DE_STRATEGIES:
                for upd in ("immediate", "deferred"):
                    cases.append(de_one("%s/%s/%s/%s" % (pname, st, upd, kind), pname, kind, seed,
                                        strategy=st, updating=upd))
        for pname in DE_PROBLEMS:
            for pol in (False, True):
                cases.append(de_one("%s/default/polish%d/%s" % (pname, pol, kind), pname, kind, 3,
                                    polish=pol))
        for pname in ("shifted_sphere_2", "six_hump_camel", "matyas"):
            lo, hi = PROBLEMS[pname][1], PROBLEMS[pname][2]
            rows = [[lo[j] + (hi[j] - lo[j]) * [0.1, 0.9, 0.5, 0.3, 0.7, 0.2, 0.8, 0.6][(i * (j + 1)) % 8]
                     for j in range(len(lo))] for i in range(8)]
            x0 = [lo[j] + (hi[j] - lo[j]) * 0.3 for j in range(len(lo))]
            variants = [
                ("random", dict(init="random")),
                ("array", dict(init_pop=rows)),
                ("x0", dict(x0=x0)),
                ("mut0.6", dict(mutation=[0.6])),
                ("dither0.2-0.9", dict(mutation=[0.9, 0.2])),
                ("atol", dict(atol=1e-3, tol=0.0)),
                ("tol0.1", dict(tol=0.1)),
                ("maxfun100", dict(maxfun=100)),
                ("maxfun333_def", dict(maxfun=333, updating="deferred")),
                ("maxfun30_def", dict(maxfun=30, updating="deferred")),
                ("maxiter0", dict(maxiter=0)),
                ("maxiter1", dict(maxiter=1)),
                ("maxiter3_def", dict(maxiter=3, updating="deferred")),
                ("cr0", dict(recombination=0.0)),
                ("cr1", dict(recombination=1.0)),
                ("pop1", dict(popsize=1)),
                ("pop1_rand1exp", dict(popsize=1, strategy="rand1exp")),
                ("pop1_best2bin_def", dict(popsize=1, strategy="best2bin", updating="deferred")),
                ("pop3", dict(popsize=3)),
                ("pop30_def", dict(popsize=30, updating="deferred")),
                ("polish_x0", dict(x0=x0, polish=True)),
            ]
            for vname, kw in variants:
                cases.append(de_one("%s/%s/%s" % (pname, vname, kind), pname, kind, 11, **kw))
        cases.append(de_one("nan_region/maxiter25/%s" % kind, "nan_region", kind, 5, maxiter=25))
        cases.append(de_one("nan_all/maxiter5/%s" % kind, "nan_all", kind, 5, maxiter=5))
    return cases


def run_reductions():
    # np.mean and np.std of arrays whose plain and pairwise sums differ: the
    # witness of the summation order of DE's termination test.
    rs = np.random.RandomState(123)
    out = []
    for n in list(range(1, 40)) + [63, 64, 65, 127, 128, 129, 130, 255, 256, 300, 1000]:
        v = rs.standard_normal(n) * 10.0 ** rs.randint(-8, 8, n)
        out.append(dict(name="n%d" % n, values=[hx(x) for x in v], mean=hx(np.mean(v)),
                        std=hx(np.std(v))))
    return out


if METHOD == "direct":  # noqa: F821  (prepended by the Go harness)
    cases = run_direct() + docstring_example()
    de_cases = []
    reductions = []
elif METHOD == "de":  # noqa: F821
    cases = []
    de_cases = run_de()
    reductions = run_reductions()
else:
    raise SystemExit("unsupported method %r" % METHOD)  # noqa: F821

json.dump(dict(method=METHOD, scipy=scipy.__version__, numpy=np.__version__,  # noqa: F821
               python=platform.python_version(), machine=platform.machine(),
               system=platform.system(), cases=cases, de=de_cases, reductions=reductions), sys.stdout, allow_nan=False)
