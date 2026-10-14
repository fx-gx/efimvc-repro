"""EFIMVC -- Efficient Federated Incomplete Multi-View Clustering.

Faithful Python port of the official MATLAB code (``official_code/``) for
Liu et al., *Efficient Federated Incomplete Multi-View Clustering*, ICML 2025.

Equation numbering in the comments follows the paper; the MATLAB routine it
mirrors is named next to each function.  See ``FORMULAS.md`` for the full
line-by-line mapping.

Notation
--------
V   number of views / clients
n   number of samples
m   number of anchors (called ``lambda1`` in the MATLAB code)
lam client regularisation   (``lambda2``)
beta server regularisation  (``lambda3``)
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from ..qp import solve_simplex_qp


# --------------------------------------------------------------------------
# small helpers mirroring the MATLAB ``functions/`` folder
# --------------------------------------------------------------------------
def normalize_data(X: np.ndarray) -> np.ndarray:
    """``NormalizeData.m``: per-sample (column) L2 normalisation."""
    norms = np.linalg.norm(X, axis=0)
    norms = np.maximum(norms, 1e-12)
    return X / norms



def _relative_drop(prev: float, cur: float) -> float:
    """``|prev - cur| / prev``, returning NaN when ``prev == 0`` (as MATLAB's 0/0)."""
    if prev == 0.0 or not np.isfinite(prev):
        return float("nan")
    return abs((prev - cur) / prev)

def eucl_dist2(X1: np.ndarray, X2: np.ndarray | None = None) -> np.ndarray:
    """``eucl_dist2.m`` with ``bSqrt=1`` -- Euclidean (not squared) distances."""
    if X2 is None:
        X2 = X1
    aa = (X1 * X1).sum(axis=1)
    bb = (X2 * X2).sum(axis=1)
    D = aa[:, None] + bb[None, :] - 2.0 * (X1 @ X2.T)
    D = np.sqrt(np.abs(D))
    return np.abs(D)


def litekmeans(X: np.ndarray, k: int, max_iter: int = 100, replicates: int = 10, seed: int = 12):
    """Light wrapper around k-means (MATLAB ``litekmeans.m``)."""
    km = KMeans(n_clusters=k, n_init=replicates, max_iter=max_iter, random_state=seed)
    labels = km.fit_predict(X)
    return labels, km.cluster_centers_


# --------------------------------------------------------------------------
# client side
# --------------------------------------------------------------------------
def client_pretrain(X: np.ndarray, m: int, lam: float, ind: np.ndarray, seed: int = 12):
    """MATLAB ``client_pretrain.m``.

    ``X``  : (d, n) normalised features, missing columns already zeroed.
    ``ind``: (n,) 0/1 availability for this view.

    Returns the initial anchor graph ``Z`` (m, n) (Eq. (10) solved once with
    the k-means anchors).
    """
    n = X.shape[1]
    present = np.flatnonzero(ind > 0)
    _, A = litekmeans(X[:, present].T, m, max_iter=100, replicates=10, seed=seed)
    A = A.T  # (d, m)

    Z = np.zeros((m, n))
    if present.size == 0:
        return Z, A
    H = A.T @ A + lam * np.eye(m)
    C1 = X[:, present].T @ A  # (n_present, m)
    Z[:, present] = solve_simplex_qp(H, C1.T)
    return Z, A


def client_train(
    X: np.ndarray,
    Z: np.ndarray,
    m: int,
    lam: float,
    ind: np.ndarray,
    Zall: np.ndarray,
    P: np.ndarray,
    max_iter: int = 30,
    tol: float = 1e-3,
    qp_tol: float = 1e-11,
):
    """MATLAB ``client_train.m`` -- solves Eq. (2) by alternating A / Z.

    Returns ``(Z, S, A, obj)`` where ``S`` is the anchor similarity matrix of
    Eq. (3).
    """
    present = np.flatnonzero(ind > 0)
    PZall = P @ Zall
    warm = None  # active-set warm start, carried between inner iterations

    obj = np.inf
    A = None
    for it in range(1, max_iter + 1):
        # ---- update A (Eq. (7)) -----------------------------------------
        C = Z @ Z.T
        D = X @ Z.T
        if np.linalg.cond(C) > 1e12:
            A = D @ np.linalg.pinv(C)
        else:
            A = D @ np.linalg.inv(C)

        # ---- update Z (Eq. (10)) ----------------------------------------
        if present.size:
            H = A.T @ A + lam * np.eye(m)
            C1 = X[:, present].T @ A + lam * PZall[:, present].T  # (n_present, m)
            Zp = solve_simplex_qp(H, C1.T, tol=qp_tol, warm=warm)
            Z[:, present] = Zp
            warm = Zp <= 1e-12

        prev = obj
        obj = float(np.sum((X - A @ Z) ** 2))
        obj += lam * float(np.sum((PZall * ind[None, :] - Z) ** 2))
        # MATLAB evaluates ``abs((obj_prev - obj)/obj_prev) < tol`` first; with
        # obj_prev == 0 that is 0/0 = NaN (not an error) and the ``obj < 1e-10``
        # arm then stops the loop.  Mirror that instead of dividing by zero.
        if it > 1 and (obj < 1e-10 or _relative_drop(prev, obj) < tol):
            break

    # ---- anchor similarity (Eq. (3)) ------------------------------------
    D = eucl_dist2(A.T, A.T)
    sigma = 0.3 * ((m - 1) * 0.5 - 1) + 0.8
    S = np.exp(-D / (2.0 * sigma * sigma))
    return Z, S, A, obj


# --------------------------------------------------------------------------
# server side
# --------------------------------------------------------------------------
def server_pretrain(Z: list[np.ndarray], ind: np.ndarray) -> np.ndarray:
    """MATLAB ``server_pretrain.m``: availability-weighted average of Z^(v)."""
    m = Z[0].shape[0]
    n = Z[0].shape[1]
    Zall = np.zeros((m, n))
    for Zv in Z:
        Zall += Zv
    Zall = Zall / np.maximum(ind.sum(axis=1), 1)[None, :]
    return Zall


def server_train(
    P: list[np.ndarray],
    Zall: np.ndarray,
    Z: list[np.ndarray],
    S: list[np.ndarray],
    beta: float,
    ind: np.ndarray,
    max_iter: int = 30,
    tol: float = 1e-3,
    qp_tol: float = 1e-11,
    update_P: bool = True,
):
    """MATLAB ``server_train.m`` -- solves Eq. (5) / (12) / (14).

    ``update_P=False`` is the ablation ``P^(v) = I``: the alignment matrices
    stay at the identity and only the global anchor graph ``Z`` is re-optimised.

    Returns ``(Zall, P, obj)``.
    """
    V = len(Z)
    m = Z[0].shape[0]
    n = Z[0].shape[1]

    L = []
    Lmx = []
    for iv in range(V):
        Sv = (S[iv] + S[iv].T) / 2.0
        D = np.diag(Sv.sum(axis=1))
        Lv = D - Sv
        L.append(Lv)
        lmax = float(np.linalg.eigvalsh(Lv)[-1])
        Lmx.append(lmax * np.eye(m) - Lv)

    obj = np.inf
    for it in range(1, max_iter + 1):
        # ---- update P^(v) (Eq. (12), solved as in Appendix A.1) ---------
        ZZ = Zall @ Zall.T
        for iv in range(V) if update_P else []:
            Pv = P[iv]
            fobj1 = 0.0
            for rep in range(1, 101):
                LPZZ = Lmx[iv] @ Pv @ ZZ
                ZZall = Z[iv] @ Zall.T
                M = 2.0 * beta * LPZZ + ZZall
                Um, _, Vm = np.linalg.svd(M, full_matrices=False)
                # MATLAB: [Um,~,Vm] = svd(M,'econ'); Pv = Um*Vm';
                # numpy returns Vh == V', so the Procrustes solution U V' is
                # Um @ Vm -- NOT Um @ Vm.T.
                Pv = Um @ Vm
                fobj2 = fobj1
                fobj1 = beta * float(np.trace(Pv.T @ LPZZ)) + float(np.trace(Pv.T @ ZZall))
                if rep > 3 and abs((fobj1 - fobj2) / fobj1) < tol:
                    break
            P[iv] = Pv

        # ---- update Z (Eq. (13) / (14)) ---------------------------------
        Lp = np.zeros((m, m))
        Zp = np.zeros((n, m))
        for iv in range(V):
            Lp += P[iv].T @ L[iv] @ P[iv]
            Zp += Z[iv].T @ P[iv]

        # H depends only on how many views sample j is present in  -> group
        counts = ind.sum(axis=1).astype(np.int64)
        warm_Z = Zall <= 1e-12
        for k in np.unique(counts):
            cols = np.flatnonzero(counts == k)
            if cols.size == 0:
                continue
            H = k * np.eye(m) + 2.0 * beta * Lp
            Zc = solve_simplex_qp(H, Zp[cols].T, tol=qp_tol, warm=warm_Z[:, cols])
            Zall[:, cols] = Zc
            warm_Z[:, cols] = Zc <= 1e-12

        # ---- objective (Eq. (23)) ---------------------------------------
        part1 = 0.0
        part2 = 0.0
        for iv in range(V):
            part1 += float(np.sum((P[iv] @ Zall * ind[:, iv][None, :] - Z[iv]) ** 2))
            part2 += float(np.trace(Zall.T @ P[iv].T @ L[iv] @ P[iv] @ Zall))
        prev = obj
        obj = part1 + 2.0 * beta * part2
        if it > 1 and (obj < 1e-10 or _relative_drop(prev, obj) < tol):
            break

    return Zall, P, obj


# --------------------------------------------------------------------------
# main algorithm
# --------------------------------------------------------------------------
class EFIMVC:
    """Algorithm 1 of the paper.

    Parameters
    ----------
    n_clusters : k
    n_anchors  : m  (``lambda1``; the paper/offical grid uses ``[k, 2k, 3k]``)
    alpha      : client regularisation ``lambda2``
    beta       : server regularisation ``lambda3``
    """

    def __init__(
        self,
        n_clusters: int,
        n_anchors: int,
        alpha: float = 0.1,
        beta: float = 0.1,
        max_iter: int = 5,
        inner_max_iter: int = 30,
        tol: float = 1e-3,
        seed: int = 12,
        rank_feature_alignment: bool = True,
        anchor_alignment: bool = True,
        verbose: bool = False,
    ):
        self.k = int(n_clusters)
        self.m = int(n_anchors)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_iter = int(max_iter)
        self.inner_max_iter = int(inner_max_iter)
        self.tol = float(tol)
        self.seed = int(seed)
        self.rank_feature_alignment = rank_feature_alignment
        self.anchor_alignment = anchor_alignment
        self.verbose = verbose
        self.history_ = None

    # -- rescaled switches used by the ablation study ---------------------
    def _alpha(self) -> float:
        # ablation "lambda = 0": drop the feature-alignment regularisation
        return self.alpha if self.rank_feature_alignment else 0.0

    def fit(self, views: list[np.ndarray], ind: np.ndarray, init_Z=None, init_Zall=None):
        """``views``: list of (d_v, n) arrays; ``ind``: (n, V) 0/1 matrix.

        ``init_Z`` / ``init_Zall`` warm start the anchor graphs.  ``run.m``
        relies on this implicitly: inside one ``(m, alpha)`` cell it loops over
        ``beta`` **without re-running ``client_pretrain``**, so ``Z`` (and
        ``Zall``) leak from one beta to the next while ``P`` is reset to ``I``.
        """
        V = len(views)
        n = views[0].shape[1]
        m = self.m
        lam = self._alpha()

        # ---- initialization ------------------------------------------------
        if init_Z is None:
            Z = []
            for iv in range(V):
                Zv, _ = client_pretrain(views[iv], m, lam, ind[:, iv], seed=self.seed)
                Z.append(Zv)
            Zall = server_pretrain(Z, ind)
        else:
            Z = [np.array(zv, copy=True) for zv in init_Z]
            Zall = np.array(init_Zall, copy=True)

        P = [np.eye(m) for _ in range(V)]  # ablation "P = I" keeps these frozen

        obj_hist = []
        obj_client_hist = []
        obj_server_hist = []
        for it in range(1, self.max_iter + 1):
            # ---- clients (Algorithm 1) ------------------------------------
            S = []
            obj_client = 0.0
            for iv in range(V):
                Z[iv], Sv, _, oc = client_train(
                    views[iv], Z[iv], m, lam, ind[:, iv], Zall, P[iv],
                    max_iter=self.inner_max_iter, tol=self.tol,
                )
                S.append(Sv)
                obj_client += oc

            # ---- server ---------------------------------------------------
            Zall, P, obj_server = server_train(
                P, Zall, Z, S, self.beta, ind,
                max_iter=self.inner_max_iter, tol=self.tol,
                update_P=self.anchor_alignment,
            )

            obj = obj_client + obj_server
            obj_hist.append(obj)
            obj_client_hist.append(obj_client)
            obj_server_hist.append(obj_server)
            if self.verbose:
                print(f"  [EFIMVC] iter {it}: obj={obj:.6e}")
            if it > 1 and obj < 1e-10:
                break
            if it > 1 and _relative_drop(obj_hist[-2], obj) < self.tol:
                break

        self.history_ = np.asarray(obj_hist)
        self.history_client_ = np.asarray(obj_client_hist)
        self.history_server_ = np.asarray(obj_server_hist)
        self.Z_ = Z
        self.Zall_ = Zall
        self.P_ = P
        return self

    # ------------------------------------------------------------------
    def embedding(self) -> np.ndarray:
        """Left singular vectors of ``Z`` (Algorithm 1, final step)."""
        U, _, _ = np.linalg.svd(self.Zall_.T, full_matrices=False)
        U = U[:, : self.k]
        norms = np.linalg.norm(U, axis=1, keepdims=True)
        return U / np.maximum(norms, 1e-12)

    def predict(self, n_reps: int = 20, seed: int | None = None):
        """k-means read-out, repeated ``n_reps`` times (as ``run.m`` does)."""
        F = self.embedding()
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        labels = []
        for rep in range(n_reps):
            km = KMeans(
                n_clusters=self.k,
                n_init=20,
                max_iter=1000,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            labels.append(km.fit_predict(F))
        return np.asarray(labels)
