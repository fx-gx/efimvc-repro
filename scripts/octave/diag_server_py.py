"""Run the Python ``server_train`` on the inputs dumped by ``diag_server.m``.

Prints part1 / part2 / obj per outer iteration so it can be diffed against the
official MATLAB numbers.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import efimvc  # noqa: E402,F401

import numpy as np  # noqa: E402
import scipy.io as sio  # noqa: E402

from efimvc.core.efimvc import server_train  # noqa: E402

d = sio.loadmat(ROOT / "results" / "_srv_input.mat")
V = int(d["V"][0, 0])
m = int(d["m"][0, 0])
n = int(d["n"][0, 0])
beta = float(d["beta"][0, 0])

Z = [np.asarray(d["Z"][iv, 0], dtype=float) for iv in range(V)]
S = [np.asarray(d["S"][iv, 0], dtype=float) for iv in range(V)]
ind = np.asarray(d["ind"], dtype=np.int64)
Zall = np.asarray(d["Zall0"], dtype=float)
P = [np.asarray(d["P0"][iv, 0], dtype=float) for iv in range(V)]

L = []
for iv in range(V):
    Sv = (S[iv] + S[iv].T) / 2.0
    L.append(np.diag(Sv.sum(axis=1)) - Sv)

for it in range(1, 7):
    Zall, P, _ = server_train(P, Zall, Z, S, beta, ind, max_iter=1, tol=1e-3)
    part1 = 0.0
    part2 = 0.0
    for iv in range(V):
        part1 += float(np.sum((P[iv] @ Zall * ind[:, iv][None, :] - Z[iv]) ** 2))
        part2 += float(np.trace(Zall.T @ P[iv].T @ L[iv] @ P[iv] @ Zall))
    print(f"SRV iter={it} part1={part1:.6f} part2={part2:.6f} obj={part1 + 2 * beta * part2:.6f}")
