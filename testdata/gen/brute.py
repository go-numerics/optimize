"""Fixture generator: SciPy brute on the documentation example and test functions.

Run: uv run --with scipy --with numpy python testdata/gen/brute.py
Writes testdata/brute.json.
"""
import json, os, warnings
import numpy as np
from scipy import optimize

warnings.simplefilter("ignore")
params = (2, 3, 7, 8, 9, 10, 44, -1, 2, 26, 1, -2, 0.5)


def docs(z):
    x, y = z
    a, b, c, d, e, f, g, h, i, j, k, l, scale = params
    f1 = a * x**2 + b * x * y + c * y**2 + d * x + e * y + f
    f2 = -g * np.exp(-((x - h)**2 + (y - i)**2) / scale)
    f3 = -j * np.exp(-((x - k)**2 + (y - l)**2) / scale)
    return f1 + f2 + f3


def himmelblau(z):
    a, b = z
    return (a * a + b - 11)**2 + (a + b * b - 7)**2


def rosen(z):
    return optimize.rosen(np.asarray(z))


cases = []


def rec(name, f, ranges, **kw):
    x, fun, grid, jout = optimize.brute(f, ranges, full_output=True, **kw)
    ranges_out = []
    for r in ranges:
        if isinstance(r, slice):
            ranges_out.append([r.start, r.stop, r.step])
        else:
            ranges_out.append([r[0], r[1], 0])
    cases.append(dict(name=name, ranges=ranges_out, ns=kw.get("Ns", 20), finish=kw.get("finish", 1) is not None,
                      x=np.atleast_1d(x).tolist(), fun=float(fun), jmin=float(np.min(jout)),
                      jsize=int(np.size(jout))))


rec("docs", docs, (slice(-4, 4, 0.25), slice(-4, 4, 0.25)))
rec("docs_nofinish", docs, (slice(-4, 4, 0.25), slice(-4, 4, 0.25)), finish=None)
rec("himmelblau", himmelblau, ((-5, 5), (-5, 5)), Ns=15)
rec("rosen3", rosen, ((-2, 2), (-2, 2), (-2, 2)), Ns=9)

out = os.path.join(os.path.dirname(__file__), "..", "brute.json")
with open(out, "w") as fh:
    json.dump(cases, fh, indent=1)
for c in cases:
    print(c["name"], c["x"], c["fun"], c["jsize"])
