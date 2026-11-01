import json, sys
import numpy as np
from scipy.optimize import minimize
def ros(v):
    x, y = v[0], v[1]
    t = y - x * x
    u = 1.0 - x
    f = 100.0 * (t * t) + u * u
    return f, np.array([-400.0 * x * t - 2.0 * u, 200.0 * t])
out = {}
for m in ("BFGS", "L-BFGS-B"):
    ev = []
    def fg(v):
        f, g = ros(v)
        ev.append([float(v[0]).hex(), float(v[1]).hex(), float(f).hex(), float(g[0]).hex(), float(g[1]).hex()])
        return f, g
    r = minimize(fg, np.array([-1.2, 1.0]), method=m, jac=True)
    out[m] = dict(ev=ev, x=[float(a).hex() for a in r.x], fun=float(r.fun).hex(), nit=int(r.nit), nfev=int(r.nfev))
print(json.dumps(out))
