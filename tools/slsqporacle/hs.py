# Hock–Schittkowski problems and SLSQP cases run through SciPy.
# This script is not run directly: tools/slsqporacle (Go) feeds it to
# `uv run python -I -` and writes its JSON output to testdata/slsqp.json.
# Problem statements: W. Hock and K. Schittkowski, Test Examples for Nonlinear
# Programming Codes, Lecture Notes in Economics and Mathematical Systems 187,
# Springer 1981, doi:10.1007/978-3-642-48320-2. Every function here has a twin
# in slsqp_problems_test.go; keep them in step.
import json
import math
import platform
import sys

import numpy as np
import scipy
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize

inf = np.inf
P = {}


def prob(name, x0, f, g, cons=(), bounds=None, fstar=None):
    P[name] = dict(x0=x0, f=f, g=g, cons=list(cons), bounds=bounds, fstar=fstar)


def eq(fun, jac):
    return ("eq", fun, jac)


def ineq(fun, jac):
    return ("ineq", fun, jac)


A = np.array


def rosen(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def rosen_g(x):
    return A([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)])


prob("HS1", [-2, 1], rosen, rosen_g, bounds=[(-inf, inf), (-1.5, inf)], fstar=0)
prob("HS2", [-2, 1], rosen, rosen_g, bounds=[(-inf, inf), (1.5, inf)], fstar=0.0504261879)
prob("HS3", [10, 1], lambda x: x[1] + 1e-5 * (x[1] - x[0]) ** 2,
     lambda x: A([-2e-5 * (x[1] - x[0]), 1 + 2e-5 * (x[1] - x[0])]),
     bounds=[(-inf, inf), (0, inf)], fstar=0)
prob("HS4", [1.125, 0.125], lambda x: (x[0] + 1) ** 3 / 3 + x[1],
     lambda x: A([(x[0] + 1) ** 2, 1.0]), bounds=[(1, inf), (0, inf)], fstar=8 / 3)
prob("HS5", [0, 0],
     lambda x: math.sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1,
     lambda x: A([math.cos(x[0] + x[1]) + 2 * (x[0] - x[1]) - 1.5,
                  math.cos(x[0] + x[1]) - 2 * (x[0] - x[1]) + 2.5]),
     bounds=[(-1.5, 4), (-3, 3)], fstar=-math.sqrt(3) / 2 - math.pi / 3)
prob("HS6", [-1.2, 1], lambda x: (1 - x[0]) ** 2, lambda x: A([-2 * (1 - x[0]), 0.0]),
     [eq(lambda x: A([10 * (x[1] - x[0] ** 2)]), lambda x: A([[-20 * x[0], 10.0]]))], fstar=0)
prob("HS7", [2, 2], lambda x: math.log(1 + x[0] ** 2) - x[1],
     lambda x: A([2 * x[0] / (1 + x[0] ** 2), -1.0]),
     [eq(lambda x: A([(1 + x[0] ** 2) ** 2 + x[1] ** 2 - 4]),
         lambda x: A([[4 * x[0] * (1 + x[0] ** 2), 2 * x[1]]]))], fstar=-math.sqrt(3))
prob("HS8", [2, 1], lambda x: -1.0, lambda x: A([0.0, 0.0]),
     [eq(lambda x: A([x[0] ** 2 + x[1] ** 2 - 25, x[0] * x[1] - 9]),
         lambda x: A([[2 * x[0], 2 * x[1]], [x[1], x[0]]]))], fstar=-1)
prob("HS9", [0, 0], lambda x: math.sin(math.pi * x[0] / 12) * math.cos(math.pi * x[1] / 16),
     lambda x: A([math.pi / 12 * math.cos(math.pi * x[0] / 12) * math.cos(math.pi * x[1] / 16),
                  -math.pi / 16 * math.sin(math.pi * x[0] / 12) * math.sin(math.pi * x[1] / 16)]),
     [eq(lambda x: A([4 * x[0] - 3 * x[1]]), lambda x: A([[4.0, -3.0]]))], fstar=-0.5)
prob("HS10", [-10, 10], lambda x: x[0] - x[1], lambda x: A([1.0, -1.0]),
     [ineq(lambda x: A([-3 * x[0] ** 2 + 2 * x[0] * x[1] - x[1] ** 2 + 1]),
           lambda x: A([[-6 * x[0] + 2 * x[1], 2 * x[0] - 2 * x[1]]]))], fstar=-1)
prob("HS11", [4.9, 0.1], lambda x: (x[0] - 5) ** 2 + x[1] ** 2 - 25,
     lambda x: A([2 * (x[0] - 5), 2 * x[1]]),
     [ineq(lambda x: A([-x[0] ** 2 + x[1]]), lambda x: A([[-2 * x[0], 1.0]]))], fstar=-8.498464223)
prob("HS12", [0, 0], lambda x: 0.5 * x[0] ** 2 + x[1] ** 2 - x[0] * x[1] - 7 * x[0] - 7 * x[1],
     lambda x: A([x[0] - x[1] - 7, 2 * x[1] - x[0] - 7]),
     [ineq(lambda x: A([25 - 4 * x[0] ** 2 - x[1] ** 2]), lambda x: A([[-8 * x[0], -2 * x[1]]]))],
     fstar=-30)
prob("HS13", [-2, -2], lambda x: (x[0] - 2) ** 2 + x[1] ** 2, lambda x: A([2 * (x[0] - 2), 2 * x[1]]),
     [ineq(lambda x: A([(1 - x[0]) ** 3 - x[1]]), lambda x: A([[-3 * (1 - x[0]) ** 2, -1.0]]))],
     bounds=[(0, inf), (0, inf)], fstar=1)
prob("HS14", [2, 2], lambda x: (x[0] - 2) ** 2 + (x[1] - 1) ** 2,
     lambda x: A([2 * (x[0] - 2), 2 * (x[1] - 1)]),
     [eq(lambda x: A([x[0] - 2 * x[1] + 1]), lambda x: A([[1.0, -2.0]])),
      ineq(lambda x: A([-x[0] ** 2 / 4 - x[1] ** 2 + 1]), lambda x: A([[-x[0] / 2, -2 * x[1]]]))],
     fstar=9 - 23 * math.sqrt(7) / 8)
prob("HS15", [-2, 1], rosen, rosen_g,
     [ineq(lambda x: A([x[0] * x[1] - 1, x[0] + x[1] ** 2]),
           lambda x: A([[x[1], x[0]], [1.0, 2 * x[1]]]))],
     bounds=[(-inf, 0.5), (-inf, inf)], fstar=306.5)
prob("HS16", [-2, 1], rosen, rosen_g,
     [ineq(lambda x: A([x[0] + x[1] ** 2, x[0] ** 2 + x[1]]),
           lambda x: A([[1.0, 2 * x[1]], [2 * x[0], 1.0]]))],
     bounds=[(-2, 0.5), (-inf, 1)], fstar=0.25)
prob("HS17", [-2, 1], rosen, rosen_g,
     [ineq(lambda x: A([x[1] ** 2 - x[0], x[0] ** 2 - x[1]]),
           lambda x: A([[-1.0, 2 * x[1]], [2 * x[0], -1.0]]))],
     bounds=[(-2, 0.5), (-inf, 1)], fstar=1)
prob("HS18", [2, 2], lambda x: 0.01 * x[0] ** 2 + x[1] ** 2, lambda x: A([0.02 * x[0], 2 * x[1]]),
     [ineq(lambda x: A([x[0] * x[1] - 25, x[0] ** 2 + x[1] ** 2 - 25]),
           lambda x: A([[x[1], x[0]], [2 * x[0], 2 * x[1]]]))],
     bounds=[(2, 50), (0, 50)], fstar=5)
prob("HS19", [20.1, 5.84], lambda x: (x[0] - 10) ** 3 + (x[1] - 20) ** 3,
     lambda x: A([3 * (x[0] - 10) ** 2, 3 * (x[1] - 20) ** 2]),
     [ineq(lambda x: A([(x[0] - 5) ** 2 + (x[1] - 5) ** 2 - 100,
                        -(x[1] - 5) ** 2 - (x[0] - 6) ** 2 + 82.81]),
           lambda x: A([[2 * (x[0] - 5), 2 * (x[1] - 5)], [-2 * (x[0] - 6), -2 * (x[1] - 5)]]))],
     bounds=[(13, 100), (0, 100)], fstar=-6961.81381)
prob("HS21", [-1, -1], lambda x: 0.01 * x[0] ** 2 + x[1] ** 2 - 100, lambda x: A([0.02 * x[0], 2 * x[1]]),
     [ineq(lambda x: A([10 * x[0] - x[1] - 10]), lambda x: A([[10.0, -1.0]]))],
     bounds=[(2, 50), (-50, 50)], fstar=-99.96)
prob("HS22", [2, 2], lambda x: (x[0] - 2) ** 2 + (x[1] - 1) ** 2,
     lambda x: A([2 * (x[0] - 2), 2 * (x[1] - 1)]),
     [ineq(lambda x: A([-x[0] - x[1] + 2, -x[0] ** 2 + x[1]]),
           lambda x: A([[-1.0, -1.0], [-2 * x[0], 1.0]]))], fstar=1)
prob("HS23", [3, 1], lambda x: x[0] ** 2 + x[1] ** 2, lambda x: A([2 * x[0], 2 * x[1]]),
     [ineq(lambda x: A([x[0] + x[1] - 1, x[0] ** 2 + x[1] ** 2 - 1, 9 * x[0] ** 2 + x[1] ** 2 - 9,
                        x[0] ** 2 - x[1], x[1] ** 2 - x[0]]),
           lambda x: A([[1.0, 1.0], [2 * x[0], 2 * x[1]], [18 * x[0], 2 * x[1]],
                        [2 * x[0], -1.0], [-1.0, 2 * x[1]]]))],
     bounds=[(-50, 50), (-50, 50)], fstar=2)
prob("HS26", [-2.6, 2, 2], lambda x: (x[0] - x[1]) ** 2 + (x[1] - x[2]) ** 4,
     lambda x: A([2 * (x[0] - x[1]), -2 * (x[0] - x[1]) + 4 * (x[1] - x[2]) ** 3, -4 * (x[1] - x[2]) ** 3]),
     [eq(lambda x: A([(1 + x[1] ** 2) * x[0] + x[2] ** 4 - 3]),
         lambda x: A([[1 + x[1] ** 2, 2 * x[0] * x[1], 4 * x[2] ** 3]]))], fstar=0)
prob("HS27", [2, 2, 2], lambda x: 0.01 * (x[0] - 1) ** 2 + (x[1] - x[0] ** 2) ** 2,
     lambda x: A([0.02 * (x[0] - 1) - 4 * x[0] * (x[1] - x[0] ** 2), 2 * (x[1] - x[0] ** 2), 0.0]),
     [eq(lambda x: A([x[0] + x[2] ** 2 + 1]), lambda x: A([[1.0, 0.0, 2 * x[2]]]))], fstar=0.04)
prob("HS28", [-4, 1, 1], lambda x: (x[0] + x[1]) ** 2 + (x[1] + x[2]) ** 2,
     lambda x: A([2 * (x[0] + x[1]), 2 * (x[0] + x[1]) + 2 * (x[1] + x[2]), 2 * (x[1] + x[2])]),
     [eq(lambda x: A([x[0] + 2 * x[1] + 3 * x[2] - 1]), lambda x: A([[1.0, 2.0, 3.0]]))], fstar=0)
prob("HS29", [1, 1, 1], lambda x: -x[0] * x[1] * x[2],
     lambda x: A([-x[1] * x[2], -x[0] * x[2], -x[0] * x[1]]),
     [ineq(lambda x: A([-x[0] ** 2 - 2 * x[1] ** 2 - 4 * x[2] ** 2 + 48]),
           lambda x: A([[-2 * x[0], -4 * x[1], -8 * x[2]]]))], fstar=-16 * math.sqrt(2))
prob("HS30", [1, 1, 1], lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2,
     lambda x: A([2 * x[0], 2 * x[1], 2 * x[2]]),
     [ineq(lambda x: A([x[0] ** 2 + x[1] ** 2 - 1]), lambda x: A([[2 * x[0], 2 * x[1], 0.0]]))],
     bounds=[(1, 10), (-10, 10), (-10, 10)], fstar=1)


def hs35(x):
    return (9 - 8 * x[0] - 6 * x[1] - 4 * x[2] + 2 * x[0] ** 2 + 2 * x[1] ** 2 + x[2] ** 2
            + 2 * x[0] * x[1] + 2 * x[0] * x[2])


def hs35_g(x):
    return A([-8 + 4 * x[0] + 2 * x[1] + 2 * x[2], -6 + 4 * x[1] + 2 * x[0], -4 + 2 * x[2] + 2 * x[0]])


prob("HS35", [0.5, 0.5, 0.5], hs35, hs35_g,
     [ineq(lambda x: A([3 - x[0] - x[1] - 2 * x[2]]), lambda x: A([[-1.0, -1.0, -2.0]]))],
     bounds=[(0, inf)] * 3, fstar=1 / 9)


def hs38(x):
    return (100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2 + 90 * (x[3] - x[2] ** 2) ** 2
            + (1 - x[2]) ** 2 + 10.1 * ((x[1] - 1) ** 2 + (x[3] - 1) ** 2) + 19.8 * (x[1] - 1) * (x[3] - 1))


def hs38_g(x):
    return A([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
              200 * (x[1] - x[0] ** 2) + 20.2 * (x[1] - 1) + 19.8 * (x[3] - 1),
              -360 * x[2] * (x[3] - x[2] ** 2) - 2 * (1 - x[2]),
              180 * (x[3] - x[2] ** 2) + 20.2 * (x[3] - 1) + 19.8 * (x[1] - 1)])


prob("HS38", [-3, -1, -3, -1], hs38, hs38_g, bounds=[(-10, 10)] * 4, fstar=0)
prob("HS39", [2, 2, 2, 2], lambda x: -x[0], lambda x: A([-1.0, 0.0, 0.0, 0.0]),
     [eq(lambda x: A([x[1] - x[0] ** 3 - x[2] ** 2, x[0] ** 2 - x[1] - x[3] ** 2]),
         lambda x: A([[-3 * x[0] ** 2, 1.0, -2 * x[2], 0.0], [2 * x[0], -1.0, 0.0, -2 * x[3]]]))],
     fstar=-1)
prob("HS40", [0.8, 0.8, 0.8, 0.8], lambda x: -x[0] * x[1] * x[2] * x[3],
     lambda x: A([-x[1] * x[2] * x[3], -x[0] * x[2] * x[3], -x[0] * x[1] * x[3], -x[0] * x[1] * x[2]]),
     [eq(lambda x: A([x[0] ** 3 + x[1] ** 2 - 1, x[0] ** 2 * x[3] - x[2], x[3] ** 2 - x[1]]),
         lambda x: A([[3 * x[0] ** 2, 2 * x[1], 0.0, 0.0], [2 * x[0] * x[3], 0.0, -1.0, x[0] ** 2],
                      [0.0, -1.0, 0.0, 2 * x[3]]]))], fstar=-0.25)
prob("HS42", [1, 1, 1, 1], lambda x: sum((x[i] - (i + 1)) ** 2 for i in range(4)),
     lambda x: A([2 * (x[i] - (i + 1)) for i in range(4)]),
     [eq(lambda x: A([x[0] - 2, x[2] ** 2 + x[3] ** 2 - 2]),
         lambda x: A([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2 * x[2], 2 * x[3]]]))],
     fstar=28 - 10 * math.sqrt(2))


def hs43_c(x):
    return A([8 - x[0] ** 2 - x[1] ** 2 - x[2] ** 2 - x[3] ** 2 - x[0] + x[1] - x[2] + x[3],
              10 - x[0] ** 2 - 2 * x[1] ** 2 - x[2] ** 2 - 2 * x[3] ** 2 + x[0] + x[3],
              5 - 2 * x[0] ** 2 - x[1] ** 2 - x[2] ** 2 - 2 * x[0] + x[1] + x[3]])


def hs43_j(x):
    return A([[-2 * x[0] - 1, -2 * x[1] + 1, -2 * x[2] - 1, -2 * x[3] + 1],
              [-2 * x[0] + 1, -4 * x[1], -2 * x[2], -4 * x[3] + 1],
              [-4 * x[0] - 2, -2 * x[1] + 1, -2 * x[2], 1.0]])


prob("HS43", [0, 0, 0, 0],
     lambda x: x[0] ** 2 + x[1] ** 2 + 2 * x[2] ** 2 + x[3] ** 2 - 5 * x[0] - 5 * x[1] - 21 * x[2] + 7 * x[3],
     lambda x: A([2 * x[0] - 5, 2 * x[1] - 5, 4 * x[2] - 21, 2 * x[3] + 7]),
     [ineq(hs43_c, hs43_j)], fstar=-44)
prob("HS48", [3, 5, -3, 2, -2], lambda x: (x[0] - 1) ** 2 + (x[1] - x[2]) ** 2 + (x[3] - x[4]) ** 2,
     lambda x: A([2 * (x[0] - 1), 2 * (x[1] - x[2]), -2 * (x[1] - x[2]), 2 * (x[3] - x[4]), -2 * (x[3] - x[4])]),
     [eq(lambda x: A([x[0] + x[1] + x[2] + x[3] + x[4] - 5, x[2] - 2 * (x[3] + x[4]) + 3]),
         lambda x: A([[1.0] * 5, [0.0, 0.0, 1.0, -2.0, -2.0]]))], fstar=0)
prob("HS71", [1, 5, 5, 1], lambda x: x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2],
     lambda x: A([x[3] * (2 * x[0] + x[1] + x[2]), x[0] * x[3], x[0] * x[3] + 1, x[0] * (x[0] + x[1] + x[2])]),
     [ineq(lambda x: A([x[0] * x[1] * x[2] * x[3] - 25]),
           lambda x: A([[x[1] * x[2] * x[3], x[0] * x[2] * x[3], x[0] * x[1] * x[3], x[0] * x[1] * x[2]]])),
      eq(lambda x: A([x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 - 40]),
         lambda x: A([[2 * x[0], 2 * x[1], 2 * x[2], 2 * x[3]]]))],
     bounds=[(1, 5)] * 4, fstar=17.0140173)
prob("HS76", [0.5, 0.5, 0.5, 0.5],
     lambda x: (x[0] ** 2 + 0.5 * x[1] ** 2 + x[2] ** 2 + 0.5 * x[3] ** 2 - x[0] * x[2] + x[2] * x[3]
                - x[0] - 3 * x[1] + x[2] - x[3]),
     lambda x: A([2 * x[0] - x[2] - 1, x[1] - 3, 2 * x[2] - x[0] + x[3] + 1, x[3] + x[2] - 1]),
     [ineq(lambda x: A([5 - x[0] - 2 * x[1] - x[2] - x[3], 4 - 3 * x[0] - x[1] - 2 * x[2] + x[3],
                        x[1] + 4 * x[2] - 1.5]),
           lambda x: A([[-1.0, -2.0, -1.0, -1.0], [-3.0, -1.0, -2.0, 1.0], [0.0, 1.0, 4.0, 0.0]]))],
     bounds=[(0, inf)] * 4, fstar=-4.681818181)

# Stress problems that are not from Hock–Schittkowski.
prob("Rosen-free", [-1.2, 1], rosen, rosen_g, fstar=0)
prob("Infeasible", [0.0], lambda x: x[0] ** 2, lambda x: A([2 * x[0]]),
     [ineq(lambda x: A([x[0] - 2, 1 - x[0]]), lambda x: A([[1.0], [-1.0]]))])
prob("Overdetermined", [0.5], lambda x: x[0] ** 2, lambda x: A([2 * x[0]]),
     [eq(lambda x: A([x[0] - 1, x[0] + 1]), lambda x: A([[1.0], [1.0]]))])
prob("SingularEq", [0, 0], lambda x: x[0] ** 2 + x[1] ** 2, lambda x: A([2 * x[0], 2 * x[1]]),
     [eq(lambda x: A([x[0] + x[1] - 1, x[0] + x[1] - 2]), lambda x: A([[1.0, 1.0], [1.0, 1.0]]))])
prob("SingularEq3", [0, 0, 0], lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2,
     lambda x: A([2 * x[0], 2 * x[1], 2 * x[2]]),
     [eq(lambda x: A([x[0] + x[1] - 1, x[0] + x[1] - 2]), lambda x: A([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]))])
prob("DuplicateEq", [0, 0], lambda x: x[0] ** 2 + x[1] ** 2, lambda x: A([2 * x[0], 2 * x[1]]),
     [eq(lambda x: A([x[0] + x[1] - 1, x[0] + x[1] - 1]), lambda x: A([[1.0, 1.0], [1.0, 1.0]]))])
# A start 8e-7 from the equality x = 1 with f = x: the QP step is an ascent
# direction balanced by the L1 penalty, so SLSQP resets its BFGS matrix five
# times and accepts the point by the relaxed test (tol = 10*acc).
prob("Stationary", [1 - 8e-7], lambda x: x[0], lambda x: A([1.0]),
     [eq(lambda x: A([x[0] - 1]), lambda x: A([[1.0]]))], fstar=1)
prob("FixedAll", [1, 2], rosen, rosen_g,
     [ineq(lambda x: A([x[0] + x[1]]), lambda x: A([[1.0, 1.0]]))], bounds=[(1, 1), (2, 2)])
prob("FixedAllViolated", [1, 2], rosen, rosen_g,
     [ineq(lambda x: A([x[0] - x[1]]), lambda x: A([[1.0, -1.0]]))], bounds=[(1, 1), (2, 2)])
prob("FixedSome", [0.5, 2, 0.5], lambda x: (x[0] - 1) ** 2 + (x[1] - 2) ** 2 + (x[2] - 3) ** 2,
     lambda x: A([2 * (x[0] - 1), 2 * (x[1] - 2), 2 * (x[2] - 3)]),
     [ineq(lambda x: A([2 - x[0] - x[2]]), lambda x: A([[-1.0, 0.0, -1.0]]))],
     bounds=[(0, 5), (2, 2), (0, 5)], fstar=1.5)


def old_cons(p, jac):
    out = []
    for kind, fun, j in p["cons"]:
        c = {"type": kind, "fun": fun}
        if jac:
            c["jac"] = j
        out.append(c)
    return out


def maxcv(p, x):
    v = 0.0
    for kind, fun, _ in p["cons"]:
        c = np.atleast_1d(fun(x))
        if kind == "eq":
            v = max(v, float(np.max(np.abs(c))))
        else:
            v = max(v, float(np.max(np.maximum(-c, 0))))
    if p["bounds"] is not None:
        for xi, (lo, hi) in zip(x, p["bounds"]):
            v = max(v, lo - xi, xi - hi)
    return v


# New-style constraint variants (problem, constraints).
def hs35_linear():
    return [LinearConstraint([[1.0, 1.0, 2.0]], -inf, 3)]


def hs71_nonlinear(jac):
    def f(x):
        return A([x[0] * x[1] * x[2] * x[3], x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2])

    def j(x):
        return A([[x[1] * x[2] * x[3], x[0] * x[2] * x[3], x[0] * x[1] * x[3], x[0] * x[1] * x[2]],
                  [2 * x[0], 2 * x[1], 2 * x[2], 2 * x[3]]])

    return [NonlinearConstraint(f, [25, 40], [inf, 40], jac=j if jac else "2-point")]


def hs43_nonlinear(jac):
    # c(x) >= 0 written as -inf <= -c(x) <= 0 to exercise upper bounds.
    return [NonlinearConstraint(lambda x: -hs43_c(x), -inf, 0, jac=(lambda x: -hs43_j(x)) if jac else "2-point")]


VARIANTS = {
    "HS35-linear": ("HS35", hs35_linear),
    "HS71-nonlinear": ("HS71", lambda: hs71_nonlinear(True)),
    "HS71-nonlinear-fd": ("HS71", lambda: hs71_nonlinear(False)),
    "HS43-nonlinear": ("HS43", lambda: hs43_nonlinear(True)),
}


def run(case, pname, jac, cons, opts, tol=None):
    p = P[pname]
    kw = dict(method="SLSQP", constraints=cons, options=opts)
    if p["bounds"] is not None:
        kw["bounds"] = p["bounds"]
    if tol is not None:
        kw["tol"] = tol
    if jac:
        kw["jac"] = p["g"]
    r = minimize(p["f"], np.array(p["x0"], float), **kw)
    mult = getattr(r, "multipliers", None)
    return dict(
        name=case, problem=pname, jac=jac, options=opts, tol=tol,
        x=[float(v) for v in r.x], fun=float(r.fun),
        nit=int(r.get("nit", 0)), nfev=int(r.nfev), njev=int(r.get("njev", 0)),
        status=int(r.get("status", 0)), success=bool(r.success), message=str(r.message),
        multipliers=None if mult is None else [float(v) for v in mult],
        maxcv=maxcv(p, r.x), fstar=p["fstar"],
    )


cases = []
for name in P:
    for jac in (True, False):
        cases.append(run(f"{name}/{'jac' if jac else 'fd'}", name, jac, old_cons(P[name], jac), {}))
for vname, (pname, mk) in VARIANTS.items():
    cases.append(run(vname, pname, not vname.endswith("-fd"), mk(), {}))
cases.append(run("HS7/maxiter3", "HS7", True, old_cons(P["HS7"], True), {"maxiter": 3}))
cases.append(run("HS28/tol", "HS28", True, old_cons(P["HS28"], True), {}, tol=1e-10))
cases.append(run("HS71/ftol", "HS71", True, old_cons(P["HS71"], True), {"ftol": 1e-10}))
cases.append(run("HS38/eps", "HS38", False, old_cons(P["HS38"], False), {"eps": 1e-6}))

json.dump(dict(scipy=scipy.__version__, numpy=np.__version__, python=platform.python_version(),
               machine=platform.machine(), system=platform.system(), cases=cases), sys.stdout)
