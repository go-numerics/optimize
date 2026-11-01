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

# The set to generate is chosen by a variable that tools/qporacle defines ahead of
# this script (gen --set); run by hand, the planted set is the default.
try:
    QP_SET
except NameError:
    QP_SET = "planted"

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

# --- a larger generated set: feasible, strictly convex, mixed constraints ------
# Feasible by construction (a point x0 satisfies every row up to the rounding of
# the data), data rounded to 3 and 6 decimals so they are the same on every
# platform.
_g = np.random.RandomState(31102026)
for _i in range(24):
    _n = [5, 8, 12, 20][_i % 4]
    _L = np.round(_g.randn(_n, _n) / math.sqrt(_n), 3)
    _P = np.round(_L @ _L.T + 0.5 * np.eye(_n), 6)
    _q = np.round(_g.randn(_n), 3)
    _x0 = np.round(_g.randn(_n), 3)
    _Ai = np.round(_g.randn(_n, _n), 3)
    _Ae = np.round(_g.randn(max(1, _n // 4), _n), 3)
    _A = np.vstack([_Ae, _Ai, np.eye(_n)])
    _be = np.round(_Ae @ _x0, 3)
    _l = np.concatenate([_be, [-inf] * _n, np.full(_n, -3.0)])
    _u = np.concatenate([_be, np.round(_Ai @ _x0 + np.abs(_g.randn(_n)) * (_i % 3) * 0.2, 3), np.full(_n, 3.0)])
    case(f"gen-{_i:02d}-n{_n}", f"generated, n = {_n}: {len(_Ae)} equalities, {_n} inequalities, bounds [-3, 3]",
         _P, _q, _A, _l, _u)

# --- the generated sparse set: --set sparse ----------------------------------
# Problems in the shape of the classes of the OSQP paper's section 8 (written
# from their description, not from osqp_benchmarks): random QP, equality QP, Lasso
# and portfolio, three sizes each. Sparse structure, data rounded to 3 decimals.
if QP_SET == "sparse":
    CASES.clear()
    _g = np.random.RandomState(20261101)

    def _sprand(m, n, density):
        M = sp.random(m, n, density=density, random_state=_g, data_rvs=_g.randn, format="csc")
        return np.round(M.toarray(), 3)

    for _n in (20, 30, 40):
        _M = _sprand(_n, _n, 0.15)
        _A = _sprand(2 * _n, _n, 0.15)
        case(f"sp-random-n{_n}", f"random QP: P = MM' + 0.01 I (M 15% dense), {2 * _n} rows, n = {_n}",
             np.round(_M @ _M.T + 0.01 * np.eye(_n), 6), np.round(_g.randn(_n), 3), _A,
             np.round(-_g.uniform(0, 1, 2 * _n), 3), np.round(_g.uniform(0, 1, 2 * _n), 3))
    for _n in (20, 30, 40):
        _M = _sprand(_n, _n, 0.15)
        _A = _sprand(_n // 2, _n, 0.3)
        _b = np.round(_A @ np.round(_g.randn(_n), 3), 3)
        case(f"sp-equality-n{_n}", f"equality QP: {_n // 2} equalities (feasible by construction), n = {_n}",
             np.round(_M @ _M.T + 0.01 * np.eye(_n), 6), np.round(_g.randn(_n), 3), _A, _b, _b)
    for _k in (8, 12, 16):
        _m = 2 * _k
        _Ad = _sprand(_m, _k, 0.25)
        _b = np.round(_g.randn(_m), 3)
        # variables (x, y, t): min 1/2 |y|^2 + lam 1't, y = Ax - b, -t <= x <= t
        _n = 2 * _k + _m
        _P = np.zeros((_n, _n))
        _P[_k:_k + _m, _k:_k + _m] = np.eye(_m)
        _q = np.concatenate([np.zeros(_k + _m), np.full(_k, 0.5)])
        _A = np.zeros((_m + 2 * _k, _n))
        _A[:_m, :_k] = _Ad
        _A[:_m, _k:_k + _m] = -np.eye(_m)
        _A[_m:_m + _k, :_k] = np.eye(_k)
        _A[_m:_m + _k, _k + _m:] = -np.eye(_k)
        _A[_m + _k:, :_k] = np.eye(_k)
        _A[_m + _k:, _k + _m:] = np.eye(_k)
        case(f"sp-lasso-k{_k}", f"Lasso, {_k} features, {_m} samples, P singular (zero on x and t)", _P, _q, _A,
             np.concatenate([_b, np.full(_k, -inf), np.zeros(_k)]),
             np.concatenate([_b, np.zeros(_k), np.full(_k, inf)]))
    for _n in (12, 20, 30):
        _k = _n // 4
        _F = _sprand(_n, _k, 0.5)
        _N = _n + _k
        _P = np.zeros((_N, _N))
        _P[:_n, :_n] = np.diag(np.round(_g.uniform(0, np.sqrt(_k), _n), 3))
        _P[_n:, _n:] = np.eye(_k)
        _P = 2 * _P
        _q = np.concatenate([np.round(-_g.randn(_n), 3), np.zeros(_k)])
        _A = np.zeros((_k + 1 + _n, _N))
        _A[:_k, :_n] = _F.T
        _A[:_k, _n:] = -np.eye(_k)
        _A[_k, :_n] = 1.0
        _A[_k + 1:, :_n] = np.eye(_n)
        case(f"sp-portfolio-n{_n}", f"portfolio, {_n} assets, {_k} factors, P = 2 blkdiag(D, I)", _P, _q, _A,
             np.concatenate([np.zeros(_k), [1.0], np.zeros(_n)]),
             np.concatenate([np.zeros(_k), [1.0], np.full(_n, inf)]))

# --- the Maros-Meszaros set: mm convert -------------------------------------
# tools/qporacle defines MM_DIR and MM_NAMES ahead of the script. The MAT files
# (qpsolvers/maros_meszaros_qpbenchmark) hold P, q, A = [C; I], l, u with +-1e20 as
# infinity. Fixtures store dense matrices, so a problem with n*(n+m) above
# MM_DENSE_CAP entries is skipped (named on stderr).
if QP_SET == "mm":
    import scipy.io as spio
    CASES.clear()
    for _nm in MM_NAMES.split(","):
        _d = spio.loadmat(f"{MM_DIR}/{_nm}.mat")
        _P = _d["P"].astype(float).toarray()
        if not np.allclose(_P, _P.T):  # an upper triangle only: mirror it
            _P = _P + np.triu(_P, 1).T
        _q = _d["q"].T.flatten().astype(float)
        _A = _d["A"].astype(float).toarray()
        _l = _d["l"].T.flatten().astype(float)
        _u = _d["u"].T.flatten().astype(float)
        _n, _m = len(_q), _A.shape[0]
        _l[_l > 9e19] = inf
        _u[_u > 9e19] = inf
        _l[_l < -9e19] = -inf
        _u[_u < -9e19] = -inf
        try:
            np.linalg.cholesky(_P)
            _pd = True
        except np.linalg.LinAlgError:
            _pd = False
        case(_nm, f"Maros-Meszaros {_nm}: n={_n} m={_m} nnz(P)={int(np.count_nonzero(_P))} "
                  f"nnz(A)={int(np.count_nonzero(_A))} positive definite={_pd}", _P, _q, _A, _l, _u)

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


# A problem whose dense P and A would hold more than this many entries is written
# in compressed sparse column form.
DENSE_CAP = 30000


def csc_json(M):
    S = sp.csc_matrix(M)
    S.eliminate_zeros()
    S.sort_indices()
    return dict(indptr=[int(v) for v in S.indptr], indices=[int(v) for v in S.indices],
                data=[float(v) for v in S.data])


out = []
for c in CASES:
    res = dict(osqp=run_osqp(c),
               osqp_tight=run_osqp(c, eps_abs=1e-9, eps_rel=1e-9, polishing=True, max_iter=100000),
               clarabel=run_clarabel(c))
    qpres = run_quadprog(c)
    if qpres is not None:
        res["quadprog"] = qpres
    if QP_SET == "mm":
        # The same settings as the Go solver's Maros-Meszaros test (400000
        # iterations): says whether a problem OSQP fails at 100000 is also failed by
        # OSQP at the budget we give ours.
        res["osqp_long"] = run_osqp(c, eps_abs=1e-9, eps_rel=1e-9, polishing=True, max_iter=400000)
    n, m = len(c["q"]), c["A"].shape[0]
    if n * (n + m) > DENSE_CAP:
        # Over the dense cap: the sparse form of the fixture (m, p_csc, a_csc).
        mats = dict(p=None, a=None, m=m, p_csc=csc_json(c["P"]), a_csc=csc_json(c["A"]))
    else:
        mats = dict(p=c["P"].tolist(), a=c["A"].tolist())
    out.append(dict(name=c["name"], note=c["note"], n=n, q=c["q"].tolist(), l=big(c["l"]), u=big(c["u"]),
                    results=res, **mats))

json.dump(dict(osqp=version("osqp"), osqp_core=CORE, clarabel=version("clarabel"),
               quadprog=version("quadprog"), numpy=np.__version__,
               scipy=scipy.__version__, python=platform.python_version(), machine=platform.machine(),
               system=platform.system(), cases=out), JSON_OUT, allow_nan=False)
JSON_OUT.flush()
