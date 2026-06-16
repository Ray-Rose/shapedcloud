"""Solve canonical SOCP test problems via CVXPY and print results.

Used as the P8 oracle for diff-testing the Rust IPM. The same problems are
encoded in `tools/jl-oracle/solve_canonical.jl` and in
`crates/scvx-solver/tests/oracle_diff.rs` — when they disagree, one of the
three solvers is wrong.
"""

import sys
import cvxpy as cp
import numpy as np


def solve_toy_socp():
    """P1: min x_0  s.t.  x_1 + x_2 = 1,  (x_0, x_1, x_2) in SOC^3.

    Closed form: x_2 = x_3 = 0.5,  x_0 = sqrt(0.5) ~ 0.7071.
    """
    x = cp.Variable(3)
    cons = [cp.norm(x[1:]) <= x[0], x[1] + x[2] == 1]
    prob = cp.Problem(cp.Minimize(x[0]), cons)
    prob.solve(solver=cp.CLARABEL, verbose=False)
    return {
        "name":   "toy_1cone",
        "status": prob.status,
        "cost":   float(prob.value),
        "x":      x.value.tolist(),
    }


def solve_two_cone_mixed():
    """P2: 2-cone mixed.

    min x_3
    s.t. x_1 = 1, x_2 = 1
         (x_3, x_1, x_2) in SOC^3
         x_3 - 2 >= 0

    Cone 2 binds (x_3 - 2 = 0); cone 1 strictly interior. Solution: x = (1, 1, 2).
    """
    x = cp.Variable(3)
    cons = [
        x[0] == 1,
        x[1] == 1,
        cp.norm(cp.hstack([x[0], x[1]])) <= x[2],  # (x_3, x_1, x_2) in SOC^3
        x[2] >= 2,
    ]
    prob = cp.Problem(cp.Minimize(x[2]), cons)
    prob.solve(solver=cp.CLARABEL, verbose=False)
    return {
        "name":   "two_cone_mixed",
        "status": prob.status,
        "cost":   float(prob.value),
        "x":      x.value.tolist(),
    }


def solve_socp_4d_mixed():
    """P3: 4-D primal, 1 equality, 1 SOC of dim 4.

    min x_0
    s.t. x_0 + x_1 + x_2 + x_3 = 4
         (x_0, x_1, x_2, x_3) in SOC^4

    By symmetry, x_1 = x_2 = x_3 at optimum. The cone constraint becomes
    x_0 >= sqrt(3)*x_1, and the equality is x_0 + 3*x_1 = 4. At minimum,
    the cone binds: x_0 = sqrt(3)*x_1. So x_1*(sqrt(3) + 3) = 4 gives
    x_1 = 4/(3 + sqrt(3)) ~ 0.8453, x_0 = sqrt(3)*x_1 ~ 1.4641.
    Cost = x_0 ~ 1.4641 = 2*(sqrt(3) - 1).
    """
    x = cp.Variable(4)
    cons = [
        cp.norm(x[1:]) <= x[0],
        cp.sum(x) == 4,
    ]
    prob = cp.Problem(cp.Minimize(x[0]), cons)
    prob.solve(solver=cp.CLARABEL, verbose=False)
    return {
        "name":   "socp_4d",
        "status": prob.status,
        "cost":   float(prob.value),
        "x":      x.value.tolist(),
    }


def main():
    results = []
    for solver in (solve_toy_socp, solve_two_cone_mixed, solve_socp_4d_mixed):
        r = solver()
        results.append(r)
        print(f"--- {r['name']} ---")
        print(f"  solver = CLARABEL (via CVXPY {cp.__version__})")
        print(f"  status = {r['status']}")
        print(f"  cost   = {r['cost']:.12f}")
        print(f"  x      = [{', '.join(f'{v:.12f}' for v in r['x'])}]")
        print()
    return results


if __name__ == "__main__":
    main()
