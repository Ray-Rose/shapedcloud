"""Solve a dumped *assembled* SCvx subproblem via CVXPY/Clarabel.

The Rust test `oracle_scvx_subproblem.rs` (run `dump_oracle_fixtures --ignored`)
writes the standard-form matrices `(c, A, b, G, h, cones)` of a real
preconditioned SCvx subproblem to `tools/oracle-data/scvx_*.txt`. This script
reconstructs the GENERIC SOCP

    min cᵀx  s.t.  A x = b,  (h − G x)[cone] ∈ SOC^dim  for each cone

— the exact same problem the Rust IPM consumes, no physics re-encoding — and
solves it with Clarabel. Print the optimal cost (the baked oracle quantity) and
a few solution diagnostics.

Usage:
    python solve_scvx_subproblem.py [path-or-name ...]
    # default: solves tools/oracle-data/scvx_fixedtf.txt and scvx_freetf.txt
"""

import os
import sys
import numpy as np
import cvxpy as cp


def _here():
    return os.path.dirname(os.path.abspath(__file__))


def _data_dir():
    return os.path.normpath(os.path.join(_here(), "..", "oracle-data"))


def parse_dump(path):
    """Parse the line-based dump into a dict of numpy arrays."""
    with open(path, "r") as f:
        raw = [ln.strip() for ln in f]
    lines = [ln for ln in raw if ln and not ln.startswith("#")]

    i = 0

    def expect_scalar(key):
        nonlocal i
        parts = lines[i].split()
        assert parts[0] == key, f"expected {key}, got {lines[i]!r}"
        i += 1
        return int(parts[1])

    NP = expect_scalar("NP")
    NE = expect_scalar("NE")
    NCT = expect_scalar("NCT")
    NCONES = expect_scalar("NCONES")

    def read_vec(key, n):
        nonlocal i
        assert lines[i] == key, f"expected section {key!r}, got {lines[i]!r}"
        i += 1
        vals = [float(lines[i + k]) for k in range(n)]
        i += n
        return np.array(vals, dtype=float)

    c = read_vec("c", NP)
    b = read_vec("b", NE)
    h = read_vec("h", NCT)

    def read_mat(key):
        nonlocal i
        parts = lines[i].split()
        assert parts[0] == key, f"expected matrix {key}, got {lines[i]!r}"
        r, col = int(parts[1]), int(parts[2])
        i += 1
        vals = [float(lines[i + k]) for k in range(r * col)]
        i += r * col
        return np.array(vals, dtype=float).reshape(r, col)

    A = read_mat("A")
    G = read_mat("G")

    parts = lines[i].split()
    assert parts[0] == "cones"
    i += 1
    cones = []
    for _ in range(NCONES):
        o, d = lines[i].split()
        cones.append((int(o), int(d)))
        i += 1

    assert A.shape == (NE, NP) and G.shape == (NCT, NP)
    return dict(NP=NP, NE=NE, NCT=NCT, NCONES=NCONES, c=c, A=A, b=b, h=h, G=G, cones=cones)


def solve(prob_data):
    NP = prob_data["NP"]
    A, b, G, h, c = (prob_data[k] for k in ("A", "b", "G", "h", "c"))
    x = cp.Variable(NP)
    s = h - G @ x
    cons = [A @ x == b]
    for (off, dim) in prob_data["cones"]:
        if dim == 1:
            cons.append(s[off] >= 0)
        else:
            cons.append(cp.SOC(s[off], s[off + 1:off + dim]))
    problem = cp.Problem(cp.Minimize(c @ x), cons)
    problem.solve(solver=cp.CLARABEL, verbose=False)
    return problem, x


def main():
    names = sys.argv[1:] or ["fixedtf", "freetf"]
    for name in names:
        path = name if os.path.isfile(name) else os.path.join(_data_dir(), f"scvx_{name}.txt")
        if not os.path.isfile(path):
            print(f"--- {name} ---\n  MISSING dump: {path}\n")
            continue
        d = parse_dump(path)
        problem, x = solve(d)
        print(f"--- {os.path.basename(path)} ---")
        print(f"  dims   : NP={d['NP']} NE={d['NE']} NCT={d['NCT']} NCONES={d['NCONES']}")
        print(f"  solver = CLARABEL (via CVXPY {cp.__version__})")
        print(f"  status = {problem.status}")
        if problem.value is not None and np.isfinite(problem.value):
            print(f"  cost   = {problem.value:.12e}")
            xv = np.asarray(x.value, dtype=float)
            print(f"  |x|inf = {np.max(np.abs(xv)):.6e}")
            print(f"  x[0..6]= [{', '.join(f'{v:.8e}' for v in xv[:7])}]")
        print()


if __name__ == "__main__":
    main()
