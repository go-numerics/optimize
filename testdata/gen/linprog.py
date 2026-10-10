"""Fixtures for Linprog: SciPy's linprog with the HiGHS solvers as oracle.

For every problem it records the HiGHS result (x, fun, status, nit,
marginals) of method='highs', the iteration counts of 'highs-ds' and
'highs-ipm', the legacy 'interior-point' iteration count with
presolve=False (its standard form matches ours), and, for optimal problems,
the range of every primal variable and every marginal over the optimal
face, so tests can compare non-unique solutions too.

Run: uv run --with scipy --with numpy python testdata/gen/linprog.py
"""
import json
import os
import warnings

import numpy as np
from scipy.optimize import linprog

INF = 1e300  # JSON stand-in for infinity


def enc(v):
    v = np.asarray(v, dtype=float)
    return [INF if x == np.inf else (-INF if x == -np.inf else float(x)) for x in v.ravel()]


def bounds_arr(bounds, n):
    if bounds is None:
        return np.array([[0, np.inf]] * n, dtype=float)
    out = np.array([[-np.inf if lo is None else lo, np.inf if hi is None else hi]
                    for lo, hi in bounds], dtype=float)
    return out


def face_ranges(c, Aub, bub, Aeq, beq, B, fun):
    """Range of each x_j over {feasible, c.x <= fun + eps}."""
    n = len(c)
    eps = 1e-7 * (1 + abs(fun))
    A = np.vstack([Aub, c[None, :]]) if Aub is not None else c[None, :]
    b = np.concatenate([bub, [fun + eps]]) if bub is not None else np.array([fun + eps])
    lo, hi = np.empty(n), np.empty(n)
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1
        for sgn, out in ((1, lo), (-1, hi)):
            r = linprog(sgn * e, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=B, method="highs")
            out[j] = sgn * r.fun if r.status == 0 else sgn * -np.inf
    return lo, hi


def dual_ranges(c, Aub, bub, Aeq, beq, B, fun):
    """Range of each marginal (ineqlin, eqlin, lower, upper) over the dual
    optimal face, with HiGHS's sign convention."""
    n = len(c)
    mub = 0 if Aub is None else Aub.shape[0]
    meq = 0 if Aeq is None else Aeq.shape[0]
    lbf = np.isfinite(B[:, 0])
    ubf = np.isfinite(B[:, 1])
    nv = mub + meq + 2 * n
    M = np.zeros((n, nv))
    if mub:
        M[:, :mub] = Aub.T
    if meq:
        M[:, mub:mub + meq] = Aeq.T
    M[:, mub + meq:mub + meq + n] = np.eye(n)
    M[:, mub + meq + n:] = np.eye(n)
    obj = np.zeros(nv)
    if mub:
        obj[:mub] = bub
    if meq:
        obj[mub:mub + meq] = beq
    obj[mub + meq:mub + meq + n] = np.where(lbf, B[:, 0], 0)
    obj[mub + meq + n:] = np.where(ubf, B[:, 1], 0)
    vb = ([(None, 0)] * mub + [(None, None)] * meq +
          [(0, None) if f else (0, 0) for f in lbf] +
          [(None, 0) if f else (0, 0) for f in ubf])
    eps = 1e-7 * (1 + abs(fun))
    lo, hi = np.empty(nv), np.empty(nv)
    for k in range(nv):
        e = np.zeros(nv)
        e[k] = 1
        for sgn, out in ((1, lo), (-1, hi)):
            r = linprog(sgn * e, A_ub=-obj[None, :], b_ub=[-(fun - eps)], A_eq=M, b_eq=c,
                        bounds=vb, method="highs")
            out[k] = sgn * r.fun if r.status == 0 else sgn * -np.inf
    return lo, hi


def solve(name, c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    c = np.asarray(c, float)
    n = len(c)
    Aub = None if A_ub is None else np.asarray(A_ub, float)
    bub = None if b_ub is None else np.asarray(b_ub, float)
    Aeq = None if A_eq is None else np.asarray(A_eq, float)
    beq = None if b_eq is None else np.asarray(b_eq, float)
    B = bounds_arr(bounds, n)
    kw = dict(A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=[tuple(r) for r in B])
    r = linprog(c, method="highs", **kw)
    rec = {
        "name": name, "c": enc(c),
        "a_ub": None if Aub is None else enc(Aub), "m_ub": 0 if Aub is None else Aub.shape[0],
        "b_ub": None if bub is None else enc(bub),
        "a_eq": None if Aeq is None else enc(Aeq), "m_eq": 0 if Aeq is None else Aeq.shape[0],
        "b_eq": None if beq is None else enc(beq),
        "bounds": None if bounds is None else enc(B),
        "status": int(r.status), "message": r.message, "nit": int(r.nit),
    }
    nits = {}
    for meth in ("highs-ds", "highs-ipm"):
        nits[meth] = int(linprog(c, method=meth, **kw).nit)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for meth, opts in (("interior-point", {"presolve": False}),
                           ("revised simplex", {})):
            try:
                rl = linprog(c, method=meth, options=opts, **kw)
                nits[meth] = int(rl.nit)
                if meth == "interior-point":
                    rec["ip_status"] = int(rl.status)
                    rec["ip_x"] = enc(rl.x)
            except Exception as e:  # legacy solvers may refuse
                nits[meth] = -1
    rec["nits"] = nits
    if r.status == 0:
        rec["x"] = enc(r.x)
        rec["fun"] = float(r.fun)
        rec["ineqlin"] = enc(r.ineqlin.marginals)
        rec["eqlin"] = enc(r.eqlin.marginals)
        rec["lower"] = enc(r.lower.marginals)
        rec["upper"] = enc(r.upper.marginals)
        lo, hi = face_ranges(c, Aub, bub, Aeq, beq, B, r.fun)
        rec["x_lo"], rec["x_hi"] = enc(lo), enc(hi)
        dlo, dhi = dual_ranges(c, Aub, bub, Aeq, beq, B, r.fun)
        rec["d_lo"], rec["d_hi"] = enc(dlo), enc(dhi)
    return rec


def klee_minty(n):
    c = -np.array([2.0 ** (n - 1 - j) for j in range(n)])
    A = np.zeros((n, n))
    b = np.array([5.0 ** (i + 1) for i in range(n)])
    for i in range(n):
        for j in range(i):
            A[i, j] = 2.0 ** (i - j + 1)
        A[i, i] = 1
    return c, A, b


def random_lp(seed, mub, meq, n):
    rng = np.random.default_rng(seed)
    kinds = rng.integers(0, 4, n)  # 0: [0,inf) 1: free 2: box 3: (-inf, u]
    B = []
    for k in kinds:
        if k == 0:
            B.append((0.0, None))
        elif k == 1:
            B.append((None, None))
        elif k == 2:
            lo = float(np.round(rng.uniform(-2, 0), 3))
            B.append((lo, float(np.round(lo + rng.uniform(0.5, 3), 3))))
        else:
            B.append((None, float(np.round(rng.uniform(0, 2), 3))))
    Bn = bounds_arr(B, n)
    xf = np.empty(n)
    for j in range(n):
        lo, hi = Bn[j]
        lo = lo if np.isfinite(lo) else (hi - 3 if np.isfinite(hi) else -1.5)
        hi = hi if np.isfinite(hi) else lo + 3
        xf[j] = rng.uniform(lo, hi)
    Aub = np.round(rng.normal(size=(mub, n)), 3)
    bub = np.round(Aub @ xf + rng.uniform(0, 1, mub), 3) + 0.001
    Aeq = np.round(rng.normal(size=(meq, n)), 3) if meq else None
    xf3 = xf
    beq = Aeq @ xf3 if meq else None
    # Dual-feasible cost: c = -A_ub^T u + A_eq^T v + w_l - w_u keeps it bounded.
    u = rng.uniform(0, 1, mub) * (rng.uniform(size=mub) < 0.6)
    c = -Aub.T @ u
    if meq:
        c += Aeq.T @ rng.normal(size=meq)
    for j in range(n):
        lo, hi = Bn[j]
        if np.isfinite(lo) and np.isfinite(hi):
            c[j] += rng.normal()
        elif np.isfinite(lo):
            c[j] += rng.uniform(0, 1)
        elif np.isfinite(hi):
            c[j] -= rng.uniform(0, 1)
    return np.round(c, 4), Aub, bub, Aeq, beq, B


def main():
    P = []
    P.append(solve("docs_example", [-1, 4], [[-3, 1], [1, 2]], [6, 4],
                   bounds=[(None, None), (-3, None)]))
    P.append(solve("tutorial", [-29.0, -45.0, 0.0, 0.0],
                   [[1.0, -1.0, -3.0, 0.0], [-2.0, 3.0, 7.0, -3.0]], [5.0, -10.0],
                   [[2.0, 8.0, 1.0, 0.0], [4.0, 4.0, 0.0, 1.0]], [60.0, 60.0],
                   bounds=[(0, None), (0, 5.0), (None, 0.5), (-3.0, None)]))
    P.append(solve("tutorial_fixed", [-29.0, -45.0, 0.0, 0.0],
                   [[1.0, -1.0, -3.0, 0.0], [-2.0, 3.0, 7.0, -3.0]], [5.0, -10.0],
                   [[2.0, 8.0, 1.0, 0.0], [4.0, 4.0, 0.0, 1.0]], [60.0, 60.0],
                   bounds=[(0, None), (0, 6.0), (None, 0.5), (-3.0, None)]))
    for n in range(3, 9):
        c, A, b = klee_minty(n)
        P.append(solve(f"klee_minty_{n}", c, A, b))
    P.append(solve("beale_cycling", [-0.75, 150, -0.02, 6],
                   [[0.25, -60, -0.04, 9], [0.5, -90, -0.02, 3], [0, 0, 1, 0]], [0, 0, 1]))
    # Kuhn's cycling example (Bland 1977).
    P.append(solve("kuhn_cycling", [-2, -3, 1, 12],
                   [[-2, -9, 1, 9], [1 / 3, 1, -1 / 3, -2], [2, 3, -1, -12]], [0, 0, 2]))
    P.append(solve("degenerate_vertex", [-1, -1],
                   [[1, 1], [1, 0], [0, 1], [1, 2], [2, 1], [3, 3]], [1, 1, 1, 1.5, 1.5, 3]))
    # L1 regression with free slope and intercept.
    t = np.arange(8.0)
    yv = np.array([1.0, 2.9, 5.2, 7.1, 8.8, 11.3, 13.0, 15.2])
    m = len(t)
    c = np.concatenate([[0, 0], np.ones(m)])
    A = np.zeros((2 * m, m + 2))
    bb = np.zeros(2 * m)
    for i in range(m):
        A[2 * i, :2] = [-t[i], -1]
        A[2 * i, 2 + i] = -1
        bb[2 * i] = -yv[i]
        A[2 * i + 1, :2] = [t[i], 1]
        A[2 * i + 1, 2 + i] = -1
        bb[2 * i + 1] = yv[i]
    P.append(solve("l1_regression_free", c, A, bb,
                   bounds=[(None, None), (None, None)] + [(0, None)] * m))
    P.append(solve("free_vars_eq", [1, 2, -1], A_eq=[[1, 1, 1], [1, -1, 2]], b_eq=[2, 1],
                   bounds=[(None, None), (None, None), (-1, 3)]))
    P.append(solve("equality_only", [2, 3, 1, 4], A_eq=[[1, 1, 1, 1], [1, 2, 0, -1]],
                   b_eq=[10, 4]))
    # Transportation problem: supplies = demands, so one equality is redundant.
    sup, dem = [20, 30, 25], [10, 25, 15, 25]
    cost = np.array([[4, 6, 9, 5], [3, 8, 4, 7], [6, 2, 5, 8]], float)
    Aeq = []
    for i in range(3):
        r = np.zeros(12)
        r[4 * i:4 * i + 4] = 1
        Aeq.append(r)
    for j in range(4):
        r = np.zeros(12)
        r[j::4] = 1
        Aeq.append(r)
    P.append(solve("transport_redundant", cost.ravel(), A_eq=np.array(Aeq), b_eq=sup + dem))
    P.append(solve("duplicate_eq_rows", [1, 1, 2], A_eq=[[1, 2, 1], [2, 4, 2], [1, 0, 1]],
                   b_eq=[4, 8, 1]))
    P.append(solve("knapsack_upper", -np.array([10, 13, 7, 8, 5.0]),
                   [[3, 4, 2, 3, 1.5]], [7], bounds=[(0, 1)] * 5))
    P.append(solve("mixed_bounds", [1, -2, 3, -1, 0.5],
                   [[1, 1, 1, 1, 1], [-1, 2, 0, 1, -1], [0, 1, -1, 0, 2]], [10, 4, 3],
                   [[1, 0, 1, 0, 1]], [2],
                   bounds=[(-1, 2), (None, 3), (0, None), (-2, 5), (None, None)]))
    P.append(solve("upper_only", [1, 1], [[-1, -1]], [-1], bounds=[(None, 2), (None, 3)]))
    P.append(solve("presolve_fixed_zero", [1, -1, 0, 2],
                   [[1, 1, 0, 1], [0, 0, 0, 0], [1, -1, 0, 0]], [4, 1, 1],
                   bounds=[(0, None), (0, 3), (-5, 5), (2, 2)]))
    P.append(solve("infeasible_ub", [1, 1], [[1, 1], [-1, -1]], [1, -3]))
    P.append(solve("infeasible_eq_bounds", [1, 1], A_eq=[[1, 1]], b_eq=[5],
                   bounds=[(0, 1), (0, 1)]))
    P.append(solve("infeasible_eq_conflict", [1, 1, 1], A_eq=[[1, 1, 0], [0, 1, 1], [1, 2, 1]],
                   b_eq=[1, 1, 3]))
    P.append(solve("unbounded", [-1, -1], [[1, -1]], [1]))
    P.append(solve("unbounded_free", [1, 0], [[1, 1]], [1], bounds=[(None, None), (0, None)]))
    P.append(solve("unbounded_zero_column", [1, -1], [[1, 0]], [3]))
    P.append(solve("no_constraints", [1, -1, 0], bounds=[(0, 2), (-1, 4), (None, None)]))
    P.append(solve("bounds_infeasible", [1, 1], [[1, 1]], [3], bounds=[(0, 1), (2, 1)]))
    sizes = [(5, 0, 4), (8, 2, 6), (12, 4, 10), (20, 5, 15), (25, 10, 20),
             (35, 10, 30), (40, 15, 35), (45, 15, 40), (50, 10, 40), (60, 0, 40)]
    for k, (mub, meq, n) in enumerate(sizes):
        c, Aub, bub, Aeq, beq, B = random_lp(1000 + k, mub, meq, n)
        P.append(solve(f"random_{mub}x{meq}x{n}", c, Aub, bub, Aeq, beq, B))
    out = os.path.join(os.path.dirname(__file__), "..", "linprog.json")
    with open(out, "w") as f:
        json.dump(P, f, indent=0)
    for p in P:
        print(f"{p['name']:24s} status={p['status']} fun={p.get('fun')} nits={p['nits']} nit={p['nit']}")


main()
