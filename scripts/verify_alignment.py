"""Regression test for the anchor-alignment (Procrustes) update.

``server_train`` updates ``P^(v)`` by maximising ``Tr(P' M)`` over orthogonal
``P``.  The optimum is ``P = U V'`` where ``M = U S V'`` and its value equals
the sum of the singular values of ``M``.

An earlier port wrote ``Um @ Vm.T`` with *numpy*'s ``svd`` -- ``Vm`` there is
already ``V'``, so that produced ``U V`` instead of ``U V'``.  The result is
still an orthogonal matrix (so an orthogonality check passes!) but it is the
*wrong* rotation: it maximises nothing, the alignment never happens, and the
whole federated term collapses.  This test pins the value down.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import efimvc  # noqa: E402,F401

import numpy as np  # noqa: E402

from efimvc.core.efimvc import server_train  # noqa: E402


def procrustes_ok(M: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Return (value, optimum, P) for the maximisation over orthogonal P."""
    U, s, Vh = np.linalg.svd(M, full_matrices=False)
    P = U @ Vh
    return float(np.trace(P.T @ M)), float(s.sum()), P


def main() -> int:
    rng = np.random.default_rng(0)
    worst_gap = 0.0
    worst_orth = 0.0
    for m in (5, 27, 81):
        for _ in range(5):
            M = rng.standard_normal((m, m))
            val, opt, P = procrustes_ok(M)
            worst_gap = max(worst_gap, abs(val - opt))
            # the wrong variant U V
            U, _, Vh = np.linalg.svd(M, full_matrices=False)
            wrong = float(np.trace((U @ Vh.T).T @ M))
            assert wrong <= opt + 1e-9
            worst_orth = max(worst_orth, float(np.abs(P @ P.T - np.eye(m)).max()))
    print(f"  procrustes value gap to optimum : {worst_gap:.3e}")
    print(f"  procrustes orthogonality        : {worst_orth:.3e}")

    # ...and the same check through the real server_train P step
    m, n, V = 12, 200, 3
    ind = (rng.random((n, V)) > 0.3).astype(np.int64)
    ind[:, 0] = 1
    Z = []
    for iv in range(V):
        Zv = rng.random((m, n))
        Zv[:, ind[:, iv] == 0] = 0
        Z.append(Zv / np.maximum(Zv.sum(axis=0), 1e-12))
    S = [np.abs(rng.standard_normal((m, m))) for _ in range(V)]
    S = [(s + s.T) / 2 for s in S]
    Zall = np.zeros((m, n))
    for Zv in Z:
        Zall += Zv
    Zall = Zall / np.maximum(ind.sum(axis=1), 1)[None, :]
    P = [np.eye(m)] * V
    before = sum(float(np.sum((Zv * ind[:, iv][None, :] - Zall) ** 2)) for iv, Zv in enumerate(Z))
    Zall, P, _ = server_train(P, Zall, Z, S, 0.001, ind, max_iter=1)
    after = sum(float(np.sum((P[iv] @ Zall * ind[:, iv][None, :] - Z[iv]) ** 2)) for iv in range(V))
    print(f"  server_train reduces the alignment term: {before:.3f} -> {after:.3f}")
    ok = worst_gap < 1e-9 and worst_orth < 1e-12 and after < before
    print("\nalignment verification:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
