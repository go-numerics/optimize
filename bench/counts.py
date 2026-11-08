# The SciPy side of TestCounts (perf_test.go): reads [{Name, Setup, Call, N}] on
# stdin, runs every call once on the setup's arguments and prints
# {name: {"Nfev": int|null, "Nit": int|null}} (or {"Error": ...}). nfev and nit
# are those SciPy reports for the same call the timing rows make.
import json
import sys
import warnings

warnings.simplefilter("ignore")
import numpy as np  # noqa: E402


def count(r):
    """(nfev, nit) of whatever the SciPy function returned."""
    if isinstance(r, tuple):
        if len(r) == 5 and isinstance(r[2], dict) and "nfev" in r[2]:  # leastsq, fsolve, curve_fit
            return r[2]["nfev"], None
        if len(r) == 2 and hasattr(r[1], "function_calls"):  # brentq, newton, ... with full_output
            return r[1].function_calls, r[1].iterations
        if len(r) == 4 and isinstance(r[1], dict):  # fsolve: x, infodict, ier, mesg
            return r[1]["nfev"], None
        if len(r) == 4:  # fminbound: xopt, fval, ierr, numfunc
            return r[3], None
        if len(r) == 7:  # bracket: ..., funcalls
            return r[6], None
        if len(r) == 3 and isinstance(r[2], dict):  # curve_fit, trf
            return r[2].get("nfev"), None
    if hasattr(r, "function_calls"):  # root_scalar
        return r.function_calls, r.iterations
    g = r.get if hasattr(r, "get") else (lambda k: getattr(r, k, None))
    return g("nfev"), g("nit")


out = {}
for q in json.load(sys.stdin):
    try:
        env = {"np": np, "n": q["N"]}
        args = eval(q["Setup"], env)
        r = eval(q["Call"], env)(*args)
        nfev, nit = count(r)
        out[q["Name"]] = {"Nfev": None if nfev is None else int(nfev), "Nit": None if nit is None else int(nit)}
    except Exception as e:  # noqa: BLE001
        out[q["Name"]] = {"Error": "%s: %s" % (type(e).__name__, e)}
json.dump(out, sys.stdout)
