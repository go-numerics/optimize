# Planted and hand-made quadratic programs run through OSQP, Clarabel and
# quadprog. This script is not run directly: tools/qporacle (Go) feeds it to
# `uv run python -I -` and writes its JSON output to testdata/qp/<set>.json.
#
# Every problem is min 1/2 x'Px + q'x subject to l <= Ax <= u (the form of
# OSQP, docs/design-qp.md section 2.2); +-inf is written +-1e30 in the output.
# quadprog (a GPL package) is only run, never read.
import ctypes
import importlib.util
import json
import math
import os
import platform
import sys
from importlib.metadata import version

import numpy as np
import scipy
import scipy.sparse as sp

import osqp

inf = math.inf

# The solvers' native code may print banners straight to file descriptor 1:
# keep the real standard output for the JSON alone and send the rest to stderr.
JSON_OUT = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)

# The first thing the script does: the C core of the pinned wheel must be
# OSQP v1.0.0 (docs/design-qp.md 6.1). The Go side checks it again.
WANT_CORE = "1.0.0"


def core_version():
    """The version the statically linked OSQP C core reports: osqp 1.1.3 has
    no Python call for it, but its extension module exports osqp_version()
    (the wheel's ext_builtin.so holds the symbol and the string "1.0.0")."""
    spec = importlib.util.find_spec("osqp.ext_builtin")
    if spec is None or spec.origin is None:
        sys.exit("cannot find the osqp extension module: the C core version is unknown")
    try:
        fn = ctypes.CDLL(spec.origin).osqp_version
    except (OSError, AttributeError) as e:
        sys.exit(f"the osqp extension does not export osqp_version(): {e}")
    fn.restype = ctypes.c_char_p
    return fn().decode()


CORE = core_version()
if CORE != WANT_CORE:
    sys.exit(f"osqp {version('osqp')} bundles the OSQP C core {CORE}, not {WANT_CORE}")

import clarabel
import quadprog

CASES = []


def case(name, note, P, q, A=None, l=None, u=None):
    n = len(q)
    P = np.array(P, float).reshape(n, n) if P is not None else np.zeros((n, n))
    A = np.array(A, float).reshape(-1, n) if A is not None else np.zeros((0, n))
    m = A.shape[0]
    l = np.array(l if l is not None else [-inf] * m, float)
    u = np.array(u if u is not None else [inf] * m, float)
    CASES.append(dict(name=name, note=note, P=P, q=np.array(q, float), A=A, l=l, u=u))


def big(v):
    return [max(-1e30, min(1e30, float(t))) for t in v]


def vec(v):
    a = np.asarray(v, float)
    return a.tolist() if np.all(np.isfinite(a)) else None


def num(v):
    v = float(v)
    return v if math.isfinite(v) else None


# --- hand problems with known optima -----------------------------------------
case("unconstrained", "x = -P^-1 q = (-1/7, -3/7)", [[4, 1], [1, 2]], [1, 1])
case("one-bound", "min 1/2|x|^2 - x0 - x1, x0 <= 0.5: x = (0.5, 1)", np.eye(2), [-1, -1],
     [[1, 0]], [-inf], [0.5])
case("one-equality", "min 1/2|x|^2, x0 + x1 = 1: x = (0.5, 0.5)", np.eye(2), [0, 0], [[1, 1]], [1], [1])
case("vertex", "min 1/2|x - (2, 2)|^2, x0 <= 1, x1 <= 1: x = (1, 1), two active", np.eye(2), [-2, -2],
     [[1, 0], [0, 1]], [-inf, -inf], [1, 1])
case("degenerate-vertex", "as vertex plus x0 + x1 <= 2, active at the same point (n + 1 active)", np.eye(2),
     [-2, -2], [[1, 0], [0, 1], [1, 1]], [-inf] * 3, [1, 1, 2])
case("lp", "P = 0: min x0 + x1, x >= 0, x0 + x1 >= 1 (degenerate: a whole edge)", None, [1, 1],
     [[1, 0], [0, 1], [1, 1]], [0, 0, 1], [inf] * 3)
case("two-sided", "min 1/2|x|^2 - 3 x0, 0 <= x0 <= 2, -1 <= x1 <= 1: x = (2, 0)", np.eye(2), [-3, 0],
     [[1, 0], [0, 1]], [0, -1], [2, 1])
case("duplicated-rows", "min 1/2|x|^2 - x0, x0 <= 0.5 stated three times: linearly dependent active set",
     np.eye(2), [-1, 0], [[1, 0]] * 3, [-inf] * 3, [0.5] * 3)

# --- infeasible, with a feasible twin ----------------------------------------
case("infeasible-bounds", "x >= 1 and x <= 0", [[1]], [0], [[1], [1]], [1, -inf], [inf, 0])
case("feasible-twin", "x >= 1 and x <= 2: x = 1", [[1]], [0], [[1], [1]], [1, -inf], [inf, 2])
case("infeasible-equalities", "x0 + x1 = 1 and x0 + x1 = 2", np.eye(2), [0, 0], [[1, 1], [1, 1]], [1, 2], [1, 2])
case("consistent-equalities", "x0 + x1 = 1 stated twice: x = (0.5, 0.5)", np.eye(2), [0, 0],
     [[1, 1], [1, 1]], [1, 1], [1, 1])

# --- unbounded and non-convex ------------------------------------------------
case("unbounded", "free x1 with q1 < 0 and no curvature on it", [[1, 0], [0, 0]], [0, -1], [[1, 0]], [-1], [1])
case("non-convex", "P has a negative eigenvalue", [[1, 0], [0, -1]], [0, 0], [[1, 0], [0, 1]], [-1, -1], [1, 1])

# --- a seeded dense problem --------------------------------------------------
_r = np.random.RandomState(20261031)
# Rounded to 3 and 6 decimals so that the data are the same on every platform
# (a BLAS product differs in the last bit between the arm64 and x86_64 wheels).
_L = np.round(_r.randn(5, 5), 3)
case("random-5", "P = LL' + I, 3 inequalities, 1 equality, bounds, seed 20261031",
     np.round(_L @ _L.T + np.eye(5), 6), np.round(_r.randn(5), 3),
     np.round(np.vstack([_r.randn(4, 5), np.eye(5)]), 3), [-inf] * 3 + [0.0] + [-1.0] * 5,
     [1.0] * 3 + [0.0] + [1.0] * 5)


# --- solvers -----------------------------------------------------------------
def run_osqp(c, **settings):
    n = len(c["q"])
    try:
        s = osqp.OSQP()
        s.setup(P=sp.triu(sp.csc_matrix(c["P"])).tocsc(), q=c["q"], A=sp.csc_matrix(c["A"]),
                l=c["l"], u=c["u"], verbose=False, **settings)
        r = s.solve()
    except Exception as e:  # e.g. a non-convex P is refused at setup
        return dict(error=f"{type(e).__name__}: {e}")
    i = r.info
    return dict(status=str(i.status), x=vec(r.x) if r.x is not None else None,
                y=vec(r.y) if r.y is not None else None, obj=num(i.obj_val), iter=int(i.iter),
                rho_updates=int(i.rho_updates), prim_res=num(i.prim_res), dual_res=num(i.dual_res))


def run_clarabel(c):
    n, m = len(c["q"]), c["A"].shape[0]
    rows, rhs, zero, nonneg = [], [], [], []
    for a, lo, hi in zip(c["A"], c["l"], c["u"]):
        if lo == hi:
            zero.append((a, lo))
        else:
            if math.isfinite(lo):
                nonneg.append((-a, -lo))
            if math.isfinite(hi):
                nonneg.append((a, hi))
    blocks = zero + nonneg
    A = sp.csc_matrix(np.array([a for a, _ in blocks]).reshape(-1, n))
    b = np.array([v for _, v in blocks], float)
    cones = [clarabel.ZeroConeT(len(zero))] if zero else []
    cones += [clarabel.NonnegativeConeT(len(nonneg))] if nonneg else []
    st = clarabel.DefaultSettings()
    st.verbose = False
    st.tol_gap_abs = st.tol_gap_rel = st.tol_feas = 1e-10
    try:
        s = clarabel.DefaultSolver(sp.triu(sp.csc_matrix(c["P"])).tocsc(), c["q"], A, b, cones, st)
        r = s.solve()
    except Exception as e:
        return dict(error=f"{type(e).__name__}: {e}")
    return dict(status=str(r.status), x=vec(r.x), y=vec(r.z), obj=num(r.obj_val), iter=int(r.iterations))


def run_quadprog(c):
    n = len(c["q"])
    try:
        np.linalg.cholesky(c["P"])
    except np.linalg.LinAlgError:
        return None  # quadprog needs P positive definite
    eq, ineq = [], []
    for a, lo, hi in zip(c["A"], c["l"], c["u"]):
        if lo == hi:
            eq.append((a, lo))
        else:
            if math.isfinite(lo):
                ineq.append((a, lo))
            if math.isfinite(hi):
                ineq.append((-a, -hi))
    cols = eq + ineq
    if not cols:  # quadprog cannot take zero constraints: x0 >= -1e12 never binds
        cols = [(np.eye(n)[0], -1e12)]
    C = np.array([a for a, _ in cols]).T
    b = np.array([v for _, v in cols], float)
    try:
        x, f, _, it, _, _ = quadprog.solve_qp(c["P"], -c["q"], C, b, len(eq))
    except Exception as e:
        return dict(error=f"{type(e).__name__}: {e}")
    return dict(status="solved", x=vec(x), obj=num(f), iter=int(it[0]))


out = []
for c in CASES:
    res = dict(osqp=run_osqp(c),
               osqp_tight=run_osqp(c, eps_abs=1e-9, eps_rel=1e-9, polishing=True, max_iter=100000),
               clarabel=run_clarabel(c))
    qpres = run_quadprog(c)
    if qpres is not None:
        res["quadprog"] = qpres
    out.append(dict(name=c["name"], note=c["note"], n=len(c["q"]), p=c["P"].tolist(), q=c["q"].tolist(),
                    a=c["A"].tolist(), l=big(c["l"]), u=big(c["u"]), results=res))

json.dump(dict(osqp=version("osqp"), osqp_core=CORE, clarabel=version("clarabel"),
               quadprog=version("quadprog"), numpy=np.__version__,
               scipy=scipy.__version__, python=platform.python_version(), machine=platform.machine(),
               system=platform.system(), cases=out), JSON_OUT, allow_nan=False)
JSON_OUT.flush()
