# Oracle for the benchmark functions and the statistical layer of the global
# optimisers (design-global.md 7.3). The Go harness prepends `METHOD = "..."`
# and feeds this file to `python -I -`.
#
# The benchmark functions are SciPy's own: the Jamil & Yang collection of
# benchmarks/benchmarks/go_benchmark_functions at the tag below, downloaded
# (BSD-3, SciPy) into a temporary package, hashed (the hashes go into the
# fixture) and imported as-is, so the Go twins in internal/gofuncs are
# checked against SciPy's code, not against a transcription of it.
#   METHOD "funcs": the bounds, fglob and the value at three points of each.
#   METHOD "stats": 100 seeded runs of differential_evolution and
#                   basinhopping (L-BFGS-B) and one run of direct, with
#                   SciPy's success rule (Benchmark.success).
import contextlib
import hashlib
import json
import os
import platform
import struct
import sys
import tempfile
import urllib.request

import numpy as np
import scipy
from scipy.optimize import basinhopping, differential_evolution, direct

TAG = "v1.18.1"
BASE = "https://raw.githubusercontent.com/scipy/scipy/%s/benchmarks/benchmarks/go_benchmark_functions/" % TAG
FILES = ["go_benchmark"] + ["go_funcs_" + c for c in "ABCDEGHRSW"]


def hx(v):
    return "0x%016x" % struct.unpack("<Q", struct.pack("<d", float(v)))[0]


def load():
    root = tempfile.mkdtemp()
    pkg = os.path.join(root, "bmk")
    sub = os.path.join(pkg, "go_benchmark_functions")
    os.makedirs(sub)
    with open(os.path.join(pkg, "__init__.py"), "w") as f:
        f.write("")
    with open(os.path.join(sub, "__init__.py"), "w") as f:
        f.write("")
    with open(os.path.join(pkg, "common.py"), "w") as f:
        f.write("import contextlib\n\n@contextlib.contextmanager\ndef safe_import():\n"
                "    try:\n        yield\n    except ImportError:\n        pass\n")
    hashes = {}
    for name in FILES:
        with urllib.request.urlopen(BASE + name + ".py", timeout=60) as r:
            data = r.read()
        hashes[name + ".py"] = hashlib.sha256(data).hexdigest()
        with open(os.path.join(sub, name + ".py"), "wb") as f:
            f.write(data)
    sys.path.insert(0, root)
    import importlib

    mods = {}
    for name in FILES[1:]:
        mods[name] = importlib.import_module("bmk.go_benchmark_functions." + name)
    return mods, hashes


MODS, HASHES = load()


def make(cls, n):
    for m in MODS.values():
        if hasattr(m, cls):
            c = getattr(m, cls)
            return c() if n == 2 else c(dimensions=n)
    raise SystemExit("no class " + cls)


# (class, dimension) in the order of internal/gofuncs.
FUNCS = [("Ackley01", 2), ("Ackley01", 5), ("Alpine01", 2), ("Bohachevsky1", 2), ("Branin01", 2),
         ("Bukin06", 2), ("CrossInTray", 2), ("Deb01", 2), ("DropWave", 2), ("EggHolder", 2),
         ("Griewank", 2), ("HimmelBlau", 2), ("Rastrigin", 2), ("Rastrigin", 5), ("Rosenbrock", 2),
         ("Salomon", 2), ("Schwefel26", 2), ("SixHumpCamel", 2), ("Sphere", 2),
         ("StyblinskiTang", 2), ("Wavy", 2)]
# Transcendental objectives of the statistical layer.
DE_SET = ["Ackley01/2", "Ackley01/5", "Alpine01/2", "Bohachevsky1/2", "Branin01/2", "Bukin06/2",
          "CrossInTray/2", "Deb01/2", "DropWave/2", "EggHolder/2", "Griewank/2", "Rastrigin/2",
          "Rastrigin/5", "Salomon/2", "Schwefel26/2", "Wavy/2"]
BH_SET = ["Ackley01/2", "Alpine01/2", "Bohachevsky1/2", "Branin01/2", "DropWave/2", "Griewank/2",
          "Rastrigin/2", "Salomon/2"]
SEEDS = 100
BH_NITER = 30


def fname(cls, n):
    return "%s/%d" % (cls, n)


def run_funcs():
    out = []
    for cls, n in FUNCS:
        f = make(cls, n)
        lo = [float(b[0]) for b in f.bounds]
        hi = [float(b[1]) for b in f.bounds]
        pts = [list(map(float, f.global_optimum[0]))]
        pts.append([lo[i] + (0.3 + 0.05 * i) * (hi[i] - lo[i]) for i in range(n)])
        pts.append([lo[i] + (0.8 - 0.07 * i) * (hi[i] - lo[i]) for i in range(n)])
        out.append(dict(name=fname(cls, n), cls=cls, n=n, lower=[hx(v) for v in lo],
                        upper=[hx(v) for v in hi], fglob=hx(f.fglob),
                        points=[dict(x=[hx(v) for v in p], f=hx(f.fun(np.array(p)))) for p in pts]))
    return out


def success(f, x):
    try:
        return bool(f.success(np.asarray(x, dtype=float)))
    except ValueError:  # "Found a lower global minimum": SciPy's own rule raises
        return False


def run_stats():
    byname = {fname(c, n): (c, n) for c, n in FUNCS}
    out = []
    for name in DE_SET:
        f = make(*byname[name])
        runs = []
        for s in range(SEEDS):
            r = differential_evolution(f.fun, f.bounds, popsize=20, rng=s)
            runs.append(dict(seed=s, success=success(f, r.x), nfev=int(r.nfev), nit=int(r.nit),
                             fun=hx(r.fun), x=[hx(v) for v in r.x]))
        out.append(dict(name=name, method="de", runs=runs))
    for name in BH_SET:
        f = make(*byname[name])
        lo = np.array([b[0] for b in f.bounds], dtype=float)
        hi = np.array([b[1] for b in f.bounds], dtype=float)
        runs = []
        for s in range(SEEDS):
            x0 = lo + (hi - lo) * np.random.default_rng(10000 + s).random(len(lo))
            r = basinhopping(f.fun, x0, niter=BH_NITER, rng=s,
                             minimizer_kwargs=dict(method="L-BFGS-B", bounds=f.bounds))
            runs.append(dict(seed=s, success=success(f, r.x), nfev=int(r.nfev), nit=int(r.nit),
                             fun=hx(r.fun), x=[hx(v) for v in r.x], x0=[hx(v) for v in x0]))
        out.append(dict(name=name, method="bh", runs=runs))
    for cls, n in FUNCS:
        f = make(cls, n)
        r = direct(f.fun, f.bounds)
        out.append(dict(name=fname(cls, n), method="direct", runs=[dict(
            seed=0, success=success(f, r.x), nfev=int(r.nfev), nit=int(r.nit), fun=hx(r.fun),
            x=[hx(v) for v in r.x], status=int(r.status))]))
    return out


funcs, stats = [], []
if METHOD == "funcs":  # noqa: F821  (prepended by the Go harness)
    funcs = run_funcs()
elif METHOD == "stats":  # noqa: F821
    stats = run_stats()
else:
    raise SystemExit("unsupported method %r" % METHOD)  # noqa: F821

json.dump(dict(method=METHOD, scipy=scipy.__version__, numpy=np.__version__,  # noqa: F821
               python=platform.python_version(), machine=platform.machine(),
               system=platform.system(), cases=[], sources=HASHES, funcs=funcs, stats=stats),
          sys.stdout, allow_nan=False)
