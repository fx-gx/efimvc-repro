"""Simplex-constrained quadratic program solver.

The whole EFIMVC algorithm reduces, at every update step, to a large batch of
identical-structure QPs

    min_z  1/2 z' H z - c' z     s.t.  1'z = 1,  z >= 0                 (*)

with the *same* PD matrix ``H`` for every column ``j`` (only ``c = c_j``
changes).  That is exactly what MATLAB's ``quadprog`` is called with in the
official code::

    quadprog(H, -c, [], [], ones(1,m), 1, zeros(m,1), ones(m,1))

Because ``1'z = 1, z >= 0`` already implies ``z <= 1``, the upper bound is
redundant.  We solve (*) exactly with a primal active-set method:

  * ``H`` is factorised once (``M = H^-1``).
  * For an active set ``A`` (variables pinned to 0) the reduced solution is
    obtained from ``H_FF^-1 = M_FF - M_FA (M_AA)^-1 M_AF`` (Schur complement
    of the already-inverted matrix) -- cheap because ``|A|`` is small.
  * A variable leaving the free set is chosen by the most negative component;
    a variable entering it is chosen by the most violated dual value.

The inner loop is JIT-compiled with numba and parallelised over columns.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - numba is optional
    from numba import njit, prange

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):  # type: ignore
        def deco(f):
            return f

        return deco if a and callable(a[0]) else deco

    prange = range



#: relative ridge added when ``H`` turns out to be (numerically) singular.
#: The ``alpha = 0`` ablation makes ``H = A'A`` with ``rank(A) <= d < m``, so the
#: QP has a whole affine set of minimisers; the ridge only picks one of them and
#: keeps the active-set iterations numerically stable.  Override with the
#: ``EFIMVC_QP_JITTER`` environment variable.
import os as _os

_JITTER_REL = float(_os.environ.get("EFIMVC_QP_JITTER", "1e-10"))

def _inv_pd(H: np.ndarray) -> np.ndarray:
    """Inverse of a symmetric positive-definite matrix (via Cholesky).

    ``H`` is only *positive semi*-definite when the ablation ``alpha = 0`` is
    used (``H = A'A`` with ``m > d``).  A tiny relative jitter is then added --
    the perturbed problem has the same minimiser to machine precision while
    keeping the Schur-complement machinery well defined.
    """
    from scipy.linalg import cho_factor, cho_solve

    try:
        c, low = cho_factor(H, lower=True, check_finite=False)
        Hs = H
    except np.linalg.LinAlgError:
        m = H.shape[0]
        eps = _JITTER_REL * max(float(np.trace(H)) / m, 1.0)
        Hs = H + eps * np.eye(m)
        c, low = cho_factor(Hs, lower=True, check_finite=False)
    return cho_solve((c, low), np.eye(H.shape[0]), check_finite=False)


# --------------------------------------------------------------------------
# Reference (pure numpy, one column at a time) -- used by the test suite.
# --------------------------------------------------------------------------
def solve_one_numpy(M: np.ndarray, H: np.ndarray, c: np.ndarray, tol: float = 1e-11) -> np.ndarray:
    """Solve (*) for a single ``c``.  ``M`` must be ``inv(H)``."""
    m = c.shape[0]
    active = np.zeros(m, dtype=bool)
    z = np.zeros(m)

    for _ in range(4 * m + 16):
        free = np.flatnonzero(~active)
        act = np.flatnonzero(active)
        if free.size == 0:  # pragma: no cover - cannot happen for a feasible QP
            return z

        Mff = M[np.ix_(free, free)]
        cF = c[free]
        oneF = np.ones(free.size)
        if act.size:
            Mfa = M[np.ix_(free, act)]
            Maf = M[np.ix_(act, free)]
            Maa = M[np.ix_(act, act)]
            # H_FF^-1 v = M_FF v - M_FA (M_AA)^-1 (M_AF v)
            Maainv_Maf = np.linalg.solve(Maa, Maf)
            inv_cF = Mff @ cF - Mfa @ (Maainv_Maf @ cF)
            inv_1F = Mff @ oneF - Mfa @ (Maainv_Maf @ oneF)
        else:
            inv_cF = Mff @ cF
            inv_1F = Mff @ oneF

        mu = (1.0 - inv_cF.sum()) / inv_1F.sum()
        zF = inv_cF + mu * inv_1F

        k = int(np.argmin(zF))
        if zF[k] < -tol:
            active[free[k]] = True
            continue

        z[:] = 0.0
        z[free] = np.maximum(zF, 0.0)

        if act.size:
            g = H[np.ix_(act, free)] @ z[free] - c[act] - mu
            j = int(np.argmin(g))
            if g[j] < -tol:
                active[act[j]] = False
                continue
        return z

    return z  # pragma: no cover


# --------------------------------------------------------------------------
# numba kernels
# --------------------------------------------------------------------------
@njit(cache=True, fastmath=True, nogil=True, error_model="numpy")
def _solve_one_jit_fast(H, Minv, c, z, warm, Fidx, MFF, MFFtmp, tmp, tol):
    """Primal active set, maintaining ``H_FF^{-1}`` incrementally.

    The simplex solutions here are *sparse* (a handful of positive entries), so
    the free set ``F`` is small while the active set ``A`` is nearly the whole
    index set.  Forming ``H_FF^{-1}`` from ``Minv`` via a Schur complement would
    therefore need the inverse of a huge ``A``-block; instead ``H_FF^{-1}`` is
    started at ``Minv`` and *downdated* by one index at a time -- O(nF^2) per
    step.  Growing the set back (the rare dual-feasibility repair) rebuilds the
    small ``H_FF`` directly.
    """
    m = c.shape[0]
    free = np.ones(m, dtype=np.bool_)
    nF = 0
    for i in range(m):
        if warm[i]:
            free[i] = False
        else:
            free[i] = True
            Fidx[nF] = i
            nF += 1

    if nF == 0:  # degenerate warm start -> fall back to everything free
        for i in range(m):
            free[i] = True
            Fidx[i] = i
        nF = m

    if nF == m:
        for i in range(m):
            for j in range(m):
                MFF[i, j] = Minv[i, j]
    else:
        Hsub = np.empty((nF, nF))
        for i in range(nF):
            for j in range(nF):
                Hsub[i, j] = H[Fidx[i], Fidx[j]]
        inv = np.linalg.inv(Hsub)
        for i in range(nF):
            for j in range(nF):
                MFF[i, j] = inv[i, j]

    for _ in range(4 * m + 16):
        # ---- solution of the equality-constrained problem on F ----------
        su = 0.0
        sv = 0.0
        for i in range(nF):
            row = MFF[i]
            s = 0.0
            s1 = 0.0
            for j in range(nF):
                s += row[j] * c[Fidx[j]]
                s1 += row[j]
            tmp[i] = s
            tmp[m + i] = s1
            su += s
            sv += s1
        mu = (1.0 - su) / sv

        kmin = 0
        best = 1e300
        for i in range(nF):
            v = tmp[i] + mu * tmp[m + i]
            tmp[2 * m + i] = v
            if v < best:
                best = v
                kmin = i

        if best < -tol:
            # ---- remove Fidx[kmin] from the free set ---------------------
            kk = Fidx[kmin]
            pivot = MFF[kmin, kmin]
            nN = nF - 1
            for i in range(nN):
                ii = i if i < kmin else i + 1
                for j in range(nN):
                    jj = j if j < kmin else j + 1
                    MFFtmp[i, j] = MFF[ii, jj] - MFF[ii, kmin] * MFF[kmin, jj] / pivot
            for i in range(nN):
                for j in range(nN):
                    MFF[i, j] = MFFtmp[i, j]
            for i in range(kmin, nN):
                Fidx[i] = Fidx[i + 1]
            nF = nN
            free[kk] = False
            continue

        # ---- candidate solution -----------------------------------------
        for i in range(m):
            z[i] = 0.0
        for i in range(nF):
            v = tmp[2 * m + i]
            z[Fidx[i]] = v if v > 0.0 else 0.0

        # ---- dual feasibility on the active set --------------------------
        jmin = -1
        gmin = 0.0
        for a in range(m):
            if free[a]:
                continue
            s = -c[a] - mu
            for j in range(nF):
                s += H[a, Fidx[j]] * z[Fidx[j]]
            if s < gmin:
                gmin = s
                jmin = a
        if jmin < 0:
            return

        # ---- release jmin back into the free set -------------------------
        free[jmin] = True
        Fidx[nF] = jmin
        nF += 1
        Hsub = np.empty((nF, nF))
        for i in range(nF):
            for j in range(nF):
                Hsub[i, j] = H[Fidx[i], Fidx[j]]
        inv = np.linalg.inv(Hsub)
        for i in range(nF):
            for j in range(nF):
                MFF[i, j] = inv[i, j]


@njit(cache=True, fastmath=True, nogil=True, error_model="numpy")
def _solve_batch_jit_fast(H, Minv, C, Z, WARM, tol):
    """Serial over columns on purpose.

    A ``prange`` version of this loop produced wrong active sets on this
    machine (numba 0.61 / Windows); parallel speed-up is obtained by running
    many fits as separate processes instead (see ``run_protocol.py``).

    ``WARM[j, i] == True`` means "variable ``i`` of column ``j`` is believed to
    be zero" -- used to warm start from the previous inner iteration.
    """
    m, n = C.shape
    Fidx = np.empty(m, dtype=np.int64)
    MFF = np.empty((m, m))
    MFFtmp = np.empty((m, m))
    tmp = np.empty(3 * m)
    warm = np.empty(m, dtype=np.bool_)
    for j in range(n):
        for i in range(m):
            warm[i] = WARM[i, j]
        _solve_one_jit_fast(H, Minv, C[:, j], Z[:, j], warm, Fidx, MFF, MFFtmp, tmp, tol)


@njit(cache=True, fastmath=True, nogil=True, error_model="numpy")
def _solve_one_jit(M, H, c, z, A_list, F_list, tmp_c, tmp_1, tmp_g, tol):
    m = c.shape[0]
    active = np.zeros(m, dtype=np.bool_)
    for i in range(m):
        z[i] = 0.0

    for _ in range(4 * m + 16):
        nF = 0
        nA = 0
        for i in range(m):
            if active[i]:
                A_list[nA] = i
                nA += 1
            else:
                F_list[nF] = i
                nF += 1
        if nF == 0:
            return

        # (M_AA)^-1 @ M_AF   ->  nA x nF, stored in tmp_g buffer (own memory)
        # then inv_cF = M_FF cF - M_FA (that @ cF),  inv_1F likewise.
        for i in range(nF):
            tmp_c[i] = c[F_list[i]]
            tmp_1[i] = 1.0

        # Maainv_Maf @ cF  and  @ 1F  -> length nA
        if nA > 0:
            Maa = np.empty((nA, nA))
            Maf = np.empty((nA, nF))
            for i in range(nA):
                for j in range(nA):
                    Maa[i, j] = M[A_list[i], A_list[j]]
                for j in range(nF):
                    Maf[i, j] = M[A_list[i], F_list[j]]
            t_c = np.linalg.solve(Maa, Maf @ tmp_c[:nF])
            t_1 = np.linalg.solve(Maa, Maf @ tmp_1[:nF])
            cbar = t_c
            onebar = t_1
        else:
            cbar = np.zeros(0)
            onebar = np.zeros(0)

        inv_cF = np.zeros(nF)
        inv_1F = np.zeros(nF)
        for i in range(nF):
            fi = F_list[i]
            s = 0.0
            s1 = 0.0
            for j in range(nF):
                s += M[fi, F_list[j]] * tmp_c[j]
                s1 += M[fi, F_list[j]] * tmp_1[j]
            for j in range(nA):
                s -= M[fi, A_list[j]] * cbar[j]
                s1 -= M[fi, A_list[j]] * onebar[j]
            inv_cF[i] = s
            inv_1F[i] = s1

        sum_c = 0.0
        sum_1 = 0.0
        for i in range(nF):
            sum_c += inv_cF[i]
            sum_1 += inv_1F[i]
        mu = (1.0 - sum_c) / sum_1

        kmin = 0
        for i in range(nF):
            zF = inv_cF[i] + mu * inv_1F[i]
            tmp_g[i] = zF
            if zF < tmp_g[kmin]:
                kmin = i
        if tmp_g[kmin] < -tol:
            active[F_list[kmin]] = True
            continue

        z[:] = 0.0
        for i in range(nF):
            v = inv_cF[i] + mu * inv_1F[i]
            z[F_list[i]] = v if v > 0.0 else 0.0

        if nA > 0:
            jmin = -1
            gmin = 0.0
            for i in range(nA):
                ai = A_list[i]
                s = -c[ai] - mu
                for j in range(nF):
                    s += H[ai, F_list[j]] * z[F_list[j]]
                if s < gmin:
                    gmin = s
                    jmin = i
            if jmin >= 0:
                active[A_list[jmin]] = False
                continue
        return


@njit(cache=True, fastmath=True, nogil=True, error_model="numpy")
def _solve_batch_jit(M, H, C, Z, tol):
    m, n = C.shape
    for j in prange(n):
        A_list = np.empty(m, dtype=np.int64)
        F_list = np.empty(m, dtype=np.int64)
        tmp_c = np.empty(m)
        tmp_1 = np.empty(m)
        tmp_g = np.empty(m)
        z = Z[:, j]
        _solve_one_jit(M, H, C[:, j], z, A_list, F_list, tmp_c, tmp_1, tmp_g, tol)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def solve_simplex_qp(
    H: np.ndarray,
    C: np.ndarray,
    tol: float = 1e-11,
    numba: bool | None = None,
    kernel: str = "fast",
    warm: np.ndarray | None = None,
) -> np.ndarray:
    """Solve (*) for every column of ``C``.

    Parameters
    ----------
    H : (m, m) symmetric positive definite.
    C : (m, n) linear term; column ``j`` gives ``c_j``.
    kernel : ``"fast"`` (incremental submatrix inverse), ``"schur"`` (the
        earlier, slower implementation, kept for cross-checking) or
        ``"numpy"``.
    warm : optional ``(m, n)`` boolean array, ``True`` where the variable is
        *expected to be zero*.  Only an initial guess: the active-set method
        still repairs it via its dual-feasibility step, so the returned
        solution is unchanged.

    Returns
    -------
    Z : (m, n) with ``1'Z[:, j] = 1`` and ``Z[:, j] >= 0``.
    """
    H = np.ascontiguousarray(H, dtype=np.float64)
    C = np.ascontiguousarray(C, dtype=np.float64)
    m, n = C.shape
    M = _inv_pd(H)
    # ``H`` is only PSD for the ``alpha = 0`` ablation (``H = A'A`` with
    # ``m > d``); use the same tiny regularisation the inverse used, so the
    # active-set kernel never sees a singular system.
    if not np.all(np.isfinite(M)):
        H = H + _JITTER_REL * max(float(np.trace(H)) / max(m, 1), 1.0) * np.eye(m)
        H = np.ascontiguousarray(H)
        M = _inv_pd(H)
    else:
        from scipy.linalg import cho_factor

        try:
            cho_factor(H, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            H = H + _JITTER_REL * max(float(np.trace(H)) / max(m, 1), 1.0) * np.eye(m)
            H = np.ascontiguousarray(H)
            M = _inv_pd(H)
    Z = np.empty((m, n), dtype=np.float64)

    use_numba = HAVE_NUMBA if numba is None else (numba and HAVE_NUMBA)
    if n == 0:
        return Z
    if use_numba and kernel == "fast":
        W = np.zeros((m, n), dtype=np.bool_) if warm is None else np.ascontiguousarray(warm, dtype=np.bool_)
        _solve_batch_jit_fast(H, M, C, Z, W, tol)
        return Z
    if use_numba and kernel == "schur":
        _solve_batch_jit(M, H, C, Z, tol)
        return Z

    for j in range(n):
        Z[:, j] = solve_one_numpy(M, H, C[:, j], tol)
    return Z


# --------------------------------------------------------------------------
# MATLAB-compatible single-column wrapper (quadprog(H, f) form)
# --------------------------------------------------------------------------
def quadprog_simplex(H: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Mirror of ``quadprog(H, f, [], [], ones(1,m), 1, zeros(m,1), ones(m,1))``."""
    return solve_simplex_qp(H, (-f).reshape(-1, 1))[:, 0]
