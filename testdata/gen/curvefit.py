"""Fixture generator: SciPy curve_fit on noisy exponentials, Gaussians and more.

Run: uv run --with scipy --with numpy python testdata/gen/curvefit.py
Writes testdata/curvefit.json (data included, so Go needs no RNG parity).
"""
import json, os, warnings
import numpy as np
from scipy import optimize

warnings.simplefilter("ignore")
rng = np.random.default_rng(20261010)


def expdecay(x, a, b, c):
    return a * np.exp(-b * x) + c


def gauss(x, a, mu, s):
    return a * np.exp(-((x - mu) ** 2) / (2 * s * s))


def gauss2(x, a1, m1, s1, a2, m2, s2):
    return gauss(x, a1, m1, s1) + gauss(x, a2, m2, s2)


def plane(X, a, b, c):
    return a * X[0] + b * X[1] + c


MODELS = {"expdecay": expdecay, "gauss": gauss, "gauss2": gauss2, "plane": plane}
cases = []


def add(name, model, x, y, p0, **kw):
    extra = {}
    for k in ("sigma",):
        if k in kw and kw[k] is not None:
            extra[k] = np.asarray(kw[k]).tolist()
    for k in ("absolute_sigma", "method"):
        if k in kw:
            extra[k] = kw[k]
    if "bounds" in kw:
        lo, hi = kw["bounds"]
        extra["bounds"] = [[float(a), float(b)] for a, b in zip(lo, hi)]
    popt, pcov, info, msg, ier = optimize.curve_fit(MODELS[model], x, y, p0=p0, full_output=True, **kw)
    cases.append(dict(name=name, model=model, x=np.asarray(x).tolist(), xshape=list(np.shape(x)), y=y.tolist(),
                      p0=list(p0), popt=popt.tolist(), pcov=np.where(np.isinf(pcov), 1e300, pcov).tolist(),
                      nfev=int(info["nfev"]), ier=int(ier), msg=str(msg), **extra))


x = np.linspace(0, 4, 50)
y = expdecay(x, 2.5, 1.3, 0.5) + 0.2 * rng.normal(size=x.size)
add("exp", "expdecay", x, y, [1, 1, 1])
add("exp_trf", "expdecay", x, y, [1, 1, 1], method="trf")
add("exp_bounds", "expdecay", x, y, [1, 1, 0.1], bounds=([0, 0, -0.5], [3, 1, 0.5]))
sig = 0.1 + 0.2 * rng.random(x.size)
add("exp_sigma", "expdecay", x, y, [1, 1, 1], sigma=sig)
add("exp_sigma_abs", "expdecay", x, y, [1, 1, 1], sigma=sig, absolute_sigma=True)
C = np.diag(sig ** 2) + 0.001 * np.exp(-np.abs(np.subtract.outer(x, x)))
add("exp_cov", "expdecay", x, y, [1, 1, 1], sigma=C)

x = np.linspace(-5, 5, 200)
y = gauss(x, 3.0, 0.7, 1.1) + 0.15 * rng.normal(size=x.size)
add("gauss", "gauss", x, y, [1, 0, 1])
add("gauss_trf", "gauss", x, y, [1, 0, 1], method="trf")
y2 = gauss2(x, 2, -1.5, 0.6, 1.2, 2.0, 0.9) + 0.05 * rng.normal(size=x.size)
add("gauss2", "gauss2", x, y2, [1, -1, 1, 1, 1.5, 1])

X = rng.random((2, 30))
yp = plane(X, 1.5, -2.0, 0.3) + 0.01 * rng.normal(size=30)
add("plane", "plane", X, yp, [0, 0, 0])

# A degenerate model: a parameter that does not influence the fit.
x = np.linspace(0, 1, 20)
y = 2 * x + 1 + 0.01 * rng.normal(size=20)
add("exact3", "expdecay", np.linspace(0, 1, 3), np.array([1.0, 0.5, 0.2]), [1, 1, 0])

out = os.path.join(os.path.dirname(__file__), "..", "curvefit.json")
with open(out, "w") as fh:
    json.dump(cases, fh)
print(len(cases), "cases ->", out)
for c in cases:
    print(c["name"], c["popt"], c["nfev"], c["ier"])
