"""Fixture generator: SciPy scalar root finders on hard scalar functions.

Run: uv run --with scipy --with numpy python testdata/gen/roots.py
Writes testdata/roots.json. roots_test.go defines the same functions.
"""
import json, math, os, warnings
from scipy import optimize

warnings.simplefilter("ignore")

F = {
    # name: (f, f', f'', bracket, x0)
    "cubic": (lambda x: x**3 - 2*x - 5, lambda x: 3*x*x - 2, lambda x: 6*x, (2, 3), 2.0),
    "cosx": (lambda x: math.cos(x) - x, lambda x: -math.sin(x) - 1, lambda x: -math.cos(x), (0, 1), 0.5),
    "triple": (lambda x: (x - 1)**3, lambda x: 3*(x - 1)**2, lambda x: 6*(x - 1), (0, 2.5), 2.0),
    "ninth": (lambda x: (x - 1)**9, lambda x: 9*(x - 1)**8, lambda x: 72*(x - 1)**7, (-1, 4), 3.0),
    "tiny": (lambda x: 1e-12*(x - 1), lambda x: 1e-12, lambda x: 0.0, (0, 3), 0.0),
    "steep": (lambda x: math.exp(20*x) - 1e8, lambda x: 20*math.exp(20*x), lambda x: 400*math.exp(20*x), (0, 2), 1.5),
    "flat": (lambda x: math.exp(-1/(x*x)) - 1e-3 if x != 0 else -1e-3, None, None, (0.01, 2), None),
    "sinhalf": (lambda x: math.sin(x) - x/2, lambda x: math.cos(x) - 0.5, lambda x: -math.sin(x), (math.pi/2, math.pi), 2.0),
    "apsx": (lambda x: x*math.exp(-x) - 0.1, lambda x: math.exp(-x)*(1 - x), lambda x: math.exp(-x)*(x - 2), (0, 1), 0.0),
    "kink": (lambda x: (x - 0.3) if x < 0.3 else 1000*(x - 0.3), None, None, (-1, 1), None),
    "pole": (lambda x: 1/(x - 0.5) if x != 0.5 else math.inf, None, None, (0, 1), None),
}

cases = []
for name, (f, fp, fpp, br, x0) in F.items():
    for m in ["brentq", "brenth", "ridder", "bisect"]:
        r = optimize.root_scalar(f, bracket=br, method=m)
        cases.append(dict(fn=name, method=m, root=r.root, iterations=r.iterations,
                          calls=r.function_calls, converged=bool(r.converged), flag=r.flag))
    if x0 is None:
        continue
    for m in ["newton", "halley", "secant"]:
        kw = dict(x0=x0, method=m)
        if m in ("newton", "halley"):
            kw["fprime"] = fp
        if m == "halley":
            kw["fprime2"] = fpp
        if m == "secant":
            kw["x1"] = x0 + 0.1
        r = optimize.root_scalar(f, **kw)
        cases.append(dict(fn=name, method=m, root=r.root, iterations=r.iterations,
                          calls=r.function_calls, converged=bool(r.converged), flag=r.flag))

# Limits and defaults.
r = optimize.root_scalar(F["cubic"][0], bracket=(2, 3), method="bisect", maxiter=10)
cases.append(dict(fn="cubic", method="bisect", maxiter=10, root=r.root, iterations=r.iterations,
                  calls=r.function_calls, converged=bool(r.converged), flag=r.flag))
r = optimize.newton(F["cosx"][0], 0.5, full_output=True, disp=False)[1]
cases.append(dict(fn="cosx", method="newton-secant-default", root=r.root, iterations=r.iterations,
                  calls=r.function_calls, converged=bool(r.converged), flag=r.flag))
r = optimize.root_scalar(F["cosx"][0], x0=0.5)
cases.append(dict(fn="cosx", method="auto-x0", root=r.root, iterations=r.iterations,
                  calls=r.function_calls, converged=bool(r.converged), flag=r.flag))

out = os.path.join(os.path.dirname(__file__), "..", "roots.json")
with open(out, "w") as fh:
    json.dump(cases, fh, indent=1)
print(len(cases), "cases ->", out)
