"""EFIMVC reproduction package.

Thread pinning is done here on purpose: EFIMVC is a sequence of very small
dense linear-algebra calls (``m x m`` SVDs / solves with ``m <= 300``).  Left
to its own devices OpenBLAS/MKL spawns one thread per core for each of those
calls, and the resulting contention made ``server_train`` ~200x slower on this
machine.  Scripts that shell out to many worker processes should keep this at
1 and parallelise across runs instead.
"""

from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

__all__ = ["qp", "metrics", "core", "data"]
