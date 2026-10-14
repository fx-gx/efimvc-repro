"""Clustering metrics, mirroring the official MATLAB ``Clustering8Measure``.

``Clustering8Measure`` returns::

    [ACC nmi Purity Fscore Precision Recall AR Entropy]

with

* ``ACC``      -- Hungarian-matched accuracy (``bestMap`` + ``hungarian``)
* ``nmi``      -- ``2*MI / (H_clust + H_class)``  (arithmetic-mean NMI)
* ``Purity``   -- sum over clusters of the majority-class count / n
* ``Fscore``   -- *pairwise* F1 (``compute_f``: harmonic mean of pairwise
  precision and recall), **not** the Fowlkes-Mallows geometric mean.

Everything here is checked against the MATLAB definitions in
``scripts/verify_metrics.py``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _as_labels(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).ravel()
    _, inv = np.unique(x, return_inverse=True)
    return inv.astype(np.int64)


def acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Hungarian-matched accuracy (best permutation of predicted labels)."""
    y_true = _as_labels(y_true)
    y_pred = _as_labels(y_pred)
    n = y_true.size
    d = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((d, d), dtype=np.float64)
    for i in range(y_true.size):
        w[y_pred[i], y_true[i]] += 1.0
    row, col = linear_sum_assignment(-w)
    return float(w[row, col].sum() / n)


def _contingency(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = _as_labels(y_true)
    y_pred = _as_labels(y_pred)
    return np.array(
        [
            [np.sum((y_true == i) & (y_pred == j)) for i in range(y_true.max() + 1)]
            for j in range(y_pred.max() + 1)
        ],
        dtype=np.float64,
    )


def nmi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """``2*MI / (H_clust + H_class)`` -- identical to the MATLAB ``compute_nmi``."""
    A = _contingency(y_true, y_pred)
    n = A.sum()
    B = A.sum(axis=1)  # cluster sizes
    D = A.sum(axis=0)  # class sizes
    mi = 0.0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] != 0:
                mi += A[i, j] / n * np.log2(n * A[i, j] / (B[i] * D[j]))
    with np.errstate(divide="ignore", invalid="ignore"):
        class_ent = float(np.nansum(np.where(D > 0, D / n * np.log2(n / np.where(D > 0, D, 1)), 0.0)))
        clust_ent = float(np.nansum(np.where(B > 0, B / n * np.log2(n / np.where(B > 0, B, 1)), 0.0)))
    denom = clust_ent + class_ent
    return float(2 * mi / denom) if denom > 0 else 0.0


def purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _as_labels(y_true)
    y_pred = _as_labels(y_pred)
    total = 0
    for c in np.unique(y_pred):
        vals, counts = np.unique(y_true[y_pred == c], return_counts=True)
        total += int(counts.max()) if counts.size else 0
    return float(total / y_true.size)


def pairwise_f(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """``compute_f``: pairwise precision / recall / F1."""
    y_true = _as_labels(y_true)
    y_pred = _as_labels(y_pred)
    n = y_true.size
    num_t = 0.0
    num_h = 0.0
    num_i = 0.0
    for k in range(n):
        tn = y_true[k + 1 :] == y_true[k]
        hn = y_pred[k + 1 :] == y_pred[k]
        num_t += tn.sum()
        num_h += hn.sum()
        num_i += (tn & hn).sum()
    p = num_i / num_h if num_h > 0 else 1.0
    r = num_i / num_t if num_t > 0 else 1.0
    f = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return float(f), float(p), float(r)


def rand_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _as_labels(y_true)
    y_pred = _as_labels(y_pred)
    n = y_true.size
    a = b = c = d = 0.0
    for k in range(n):
        tn = y_true[k + 1 :] == y_true[k]
        hn = y_pred[k + 1 :] == y_pred[k]
        a += (tn & hn).sum()
        b += (tn & ~hn).sum()
        c += (~tn & hn).sum()
        d += (~tn & ~hn).sum()
    denom = a + b + c + d
    return float((a + d) / denom) if denom > 0 else 0.0


def clustering_8measure(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return the same eight numbers as MATLAB's ``Clustering8Measure``."""
    f, p, r = pairwise_f(y_true, y_pred)
    out = {
        "ACC": acc(y_true, y_pred),
        "NMI": nmi(y_true, y_pred),
        "Purity": purity(y_true, y_pred),
        "Fscore": f,
        "Precision": p,
        "Recall": r,
        "AR": rand_index(y_true, y_pred),
    }
    # entropy of the predicted partition (last slot of the MATLAB vector)
    yp = _as_labels(y_pred)
    _, counts = np.unique(yp, return_counts=True)
    pr = counts / counts.sum()
    out["Entropy"] = float(-(pr * np.log2(pr)).sum())
    return out


PAPER_METRICS = ("ACC", "NMI", "Purity", "Fscore")
