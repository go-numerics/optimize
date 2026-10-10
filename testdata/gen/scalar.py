"""Fixture generator: SciPy minimize_scalar (brent, golden, bounded) and bracket.

Run: uv run --with scipy --with numpy python testdata/gen/scalar.py
Writes testdata/scalar.json. scalar_test.go defines the same functions.
"""
import json, math, os, warnings
from scipy import optimize

warnings.simplefilter("ignore")

F = {
    "quad": lambda x: (x - 2.0) * (x - 2.0) + 1.0,
    "quartic": lambda x: x**4 - 3.0 * x**3 + 2.0,
    "cosx": lambda x: math.cos(x),
    "abs": lambda x: abs(x - 0.7),
    "bessel_like": lambda x: math.exp(-x) * math.sin(3 * x),
    "linear": lambda x: -x,
}
cases = []


def rec(name, method, r, **kw):
    d = dict(fn=name, method=method, x=float(r.x), fun=float(r.fun), nit=int(r.get("nit", -1)),
             nfev=int(r.nfev), success=bool(r.success), message=str(r.message))
    d.update(kw)
    cases.append(d)


for name, f in F.items():
    for m in ["brent", "golden"]:
        rec(name, m, optimize.minimize_scalar(f, method=m))
        rec(name, m, optimize.minimize_scalar(f, method=m, bracket=(0, 1.5)), bracket=[0, 1.5])
    rec(name, "bounded", optimize.minimize_scalar(f, method="bounded", bounds=(-1, 3)), bounds=[-1, 3])
rec("quad", "brent", optimize.minimize_scalar(F["quad"], method="brent", bracket=(0, 1, 5)), bracket=[0, 1, 5])
rec("quad", "golden", optimize.minimize_scalar(F["quad"], method="golden", bracket=(5, 1, 0)), bracket=[5, 1, 0])
rec("quad", "brent", optimize.minimize_scalar(F["quad"], method="brent", options=dict(maxiter=3)), maxiter=3)
rec("quad", "golden", optimize.minimize_scalar(F["quad"], method="golden", options=dict(maxiter=3)), maxiter=3)
rec("quad", "bounded", optimize.minimize_scalar(F["quad"], method="bounded", bounds=(-1, 3),
                                                options=dict(maxiter=4)), bounds=[-1, 3], maxiter=4)
rec("quad", "brent", optimize.minimize_scalar(F["quad"], method="brent", tol=1e-3), tol=1e-3)

br = []
for name, f in F.items():
    try:
        xa, xb, xc, fa, fb, fc, n = optimize.bracket(f)
        br.append(dict(fn=name, x=[xa, xb, xc], f=[fa, fb, fc], nfev=n, ok=True))
    except Exception as e:  # BracketError
        br.append(dict(fn=name, ok=False, msg=str(e)))

def finite(v):
    """JSON has no infinities: write non-finite floats as strings."""
    if isinstance(v, float) and not math.isfinite(v):
        return repr(v)
    if isinstance(v, dict):
        return {k: finite(x) for k, x in v.items()}
    if isinstance(v, list):
        return [finite(x) for x in v]
    return v


out = os.path.join(os.path.dirname(__file__), "..", "scalar.json")
with open(out, "w") as fh:
    json.dump(finite(dict(cases=cases, bracket=br)), fh, indent=1)
print(len(cases), "cases ->", out)
