"""Verify the simplex-constrained QP solver.

Checks, on random instances and on real ones taken from the algorithm:

* ``1'z = 1`` and ``z >= 0``;
* the KKT system of ``min 1/2 z'Hz - c'z  s.t. 1'z=1, z>=0``
  (``Hz - c`` constant on the support and ``>=`` that constant off it);
* the three kernels agree with each other;
* a warm start gives the same answer as a cold start.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import efimvc  # noqa: E402,F401

import numpy as np  # noqa: E402

from efimvc.core import efimvc as E  # noqa: E402
from efimvc.data.loader import load_proteinfold  # noqa: E402
from efimvc.qp import _inv_pd, solve_simplex_qp  # noqa: E402


def kkt_violation(H, c, z):
    g = H @ z - c
    free = z > 1e-9
    if not free.any():
        return np.inf
    mu = g[free].mean()
    v = float(np.abs(g[free] - mu).max())
    act = ~free
    if act.any():
        v = max(v, float(max(0.0, (mu - g[act]).max())))
    return v


def check(H, C, label, tol=1e-9):
    Z = solve_simplex_qp(H, C)
    assert np.allclose(Z.sum(0), 1.0, atol=1e-9), f"{label}: columns do not sum to 1"
    assert Z.min() > -1e-12, f"{label}: negative entry {Z.min()}"
    worst = max(kkt_violation(H, C[:, j], Z[:, j]) for j in range(C.shape[1]))
    print(f"  {label:28s} m={H.shape[0]:3d} n={C.shape[1]:4d}  max KKT violation {worst:.2e}")
    return worst


def main() -> int:
    rng = np.random.default_rng(0)
    worst = 0.0
    print("random instances")
    for m, d in [(6, 3), (20, 10), (30, 5), (81, 27)]:
        A = rng.standard_normal((d, m))
        H = A.T @ A + 0.1 * np.eye(m)
        C = A.T @ rng.standard_normal((d, 60))
        worst = max(worst, check(H, C, f"random m={m}"))

    print("real client QPs (ProteinFold, missing rate 0.5)")
    views, ind, _ = load_proteinfold(0.5)
    for m, lam in [(27, 0.001), (54, 0.1), (81, 1.0)]:
        _, A = E.client_pretrain(views[0], m, lam, ind[:, 0])
        H = A.T @ A + lam * np.eye(m)
        C = views[0][:, np.flatnonzero(ind[:, 0] > 0)].T @ A
        worst = max(worst, check(H, C.T, f"client m={m} lam={lam}"))

        Z1 = solve_simplex_qp(H, C.T, kernel="fast")
        Zs = solve_simplex_qp(H, C.T, kernel="schur")
        Zn = solve_simplex_qp(H, C.T, kernel="numpy")
        d1 = np.abs(Z1 - Zs).max()
        d2 = np.abs(Z1 - Zn).max()
        w = Z1 <= 1e-12
        Z2 = solve_simplex_qp(H, C.T, warm=w)
        d3 = np.abs(Z1 - Z2).max()
        print(f"     kernels: fast-vs-schur {d1:.1e}, fast-vs-numpy {d2:.1e}, warm-vs-cold {d3:.1e}")
        worst = max(worst, d1, d2, d3)

    ok = worst < 1e-9
    print(f"\nmax deviation {worst:.3e} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
