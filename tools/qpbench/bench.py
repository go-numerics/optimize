# Reference side of tools/qpbench, fed to `python -I -` by the Go harness, which
# defines QPBENCH_DIR (the instance directory), MIN_TIME (seconds of repetition per
# measurement) and EPS (eps_abs = eps_rel) ahead of this script. Prints one JSON
# object. Single thread: BLAS threads are pinned to 1 and QDLDL is sequential.
#
# OSQP times come from the solver itself (info.setup_time and info.solve_time,
# measured in C: Python call overhead is not in them, as in osqp_benchmarks'
# solvers/osqp.py, which reports the same fields). quadprog (a GPL package, run
# and never read) has no timer: the call is timed and the overhead of an
# empty-sized call is measured and subtracted.
import glob
import json
import math
import os
import statistics
import time
from importlib.metadata import version

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import osqp  # noqa: E402
import quadprog  # noqa: E402

BIG = 1e20
OSQP_DENSE_MAX_N = 500  # a dense KKT of order 3000 is a different experiment


def num(v):
    v = float(v)
    return v if math.isfinite(v) else None


def csc(d):
    return sp.csc_matrix((d["data"], d["indices"], d["indptr"]), shape=(d["rows"], d["cols"]))


def load():
    out = []
    for path in sorted(glob.glob(os.path.join(QPBENCH_DIR, "*.json"))):
        with open(path) as f:
            out.append(json.load(f))
    return out


def reps_until(fn, min_time, lo=3, hi=400):
    """Calls fn() repeatedly until min_time has elapsed (between lo and hi calls)
    and returns the list of its results."""
    res, t0 = [], time.perf_counter()
    while len(res) < lo or (time.perf_counter() - t0 < min_time and len(res) < hi):
        res.append(fn())
    return res


def run_osqp(inst):
    P, A = csc(inst["p"]), csc(inst["a"])
    q = np.array(inst["q"])
    l, u = np.array(inst["l"]), np.array(inst["u"])  # +-1e30 is OSQP's infinity

    def once():
        s = osqp.OSQP()
        s.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, eps_abs=EPS, eps_rel=EPS, polishing=False)
        r = s.solve()
        i = r.info
        return dict(setup=float(i.setup_time), solve=float(i.solve_time), iter=int(i.iter), status=str(i.status),
                    obj=num(i.obj_val), rho_updates=int(i.rho_updates))

    runs = reps_until(once, MIN_TIME)
    last = runs[-1]
    return dict(setup=statistics.median(r["setup"] for r in runs), solve=statistics.median(r["solve"] for r in runs),
                iter=last["iter"], status=last["status"], obj=last["obj"], rho_updates=last["rho_updates"],
                reps=len(runs))


def quadprog_args(inst):
    n = inst["n"]
    P = csc(inst["p"]).toarray()
    P = np.triu(P) + np.triu(P, 1).T
    A = csc(inst["a"]).toarray()
    eq, ineq = [], []
    for a, lo, hi in zip(A, inst["l"], inst["u"]):
        if lo == hi:
            eq.append((a, lo))
        else:
            if lo > -BIG:
                ineq.append((a, lo))
            if hi < BIG:
                ineq.append((-a, -hi))
    cols = eq + ineq
    C = np.array([a for a, _ in cols]).T
    b = np.array([v for _, v in cols], float)
    return P, -np.array(inst["q"]), C, b, len(eq)


def run_quadprog(inst, overhead):
    P, a, C, b, meq = quadprog_args(inst)

    def once():
        t0 = time.perf_counter()
        x, f, *_ = quadprog.solve_qp(P, a, C, b, meq)
        return time.perf_counter() - t0, float(f)

    runs = reps_until(once, MIN_TIME)
    t = statistics.median(r[0] for r in runs)
    return dict(total=max(t - overhead, 1e-9), raw=t, obj=runs[-1][1], reps=len(runs))


def call_overhead():
    G, a, C, b = np.eye(1), np.zeros(1), np.zeros((1, 1)), np.zeros(1)
    ts = []
    for _ in range(2000):
        t0 = time.perf_counter()
        quadprog.solve_qp(G, a, C, b, 0)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


overhead = call_overhead()
out = dict(meta=dict(osqp=version("osqp"), quadprog=version("quadprog"), numpy=np.__version__,
                     scipy=version("scipy"), eps=EPS, quadprog_call_overhead=overhead), osqp={}, quadprog={})
for inst in load():
    if inst["class"] != "dense" or inst["n"] <= OSQP_DENSE_MAX_N:
        out["osqp"][inst["name"]] = run_osqp(inst)
    if inst["class"] == "dense":
        out["quadprog"][inst["name"]] = run_quadprog(inst, overhead)
print(json.dumps(out, allow_nan=False))
