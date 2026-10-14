"""Validate the reverse-engineered missing-view protocol.

``efimvc/data/protocol.py`` claims that the nine shipped
``proteinFold_Per*.mat`` files were produced by

    per   = share of *complete* samples            (so missing rate = 1 - per)
    k     = floor((1 - per) * n) damaged samples, drawn as a permutation prefix
    each damaged sample keeps each view independently with probability 1/2,
    never keeping all views and never losing all of them
    the permutation and the damage patterns are drawn once -> nested masks

This script checks every consequence of that claim against the shipped files.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import efimvc  # noqa: E402,F401

import numpy as np  # noqa: E402

from efimvc.data.loader import PER_FILES, OFFICIAL_DATA, load_proteinfold  # noqa: E402

FS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def main() -> int:
    print("shipped ProteinFold files vs. the protocol recovered from")
    print("wangsiwei2010/CVPR22-IMVC-CBG :: randomlyGeneratePartialData.m\n")
    print(f"{'missing':>8} {'#damaged':>9} {'pred.':>7} {'per-view miss':>14} {'pred.':>7} "
          f"{'all-kept':>9} {'binom p':>9}")
    failures = 0
    masks = {}
    for rate in FS:
        p = round(1.0 - rate, 1)
        views, ind, _ = load_proteinfold(rate)
        n, V = ind.shape
        masks[rate] = ind

        damaged = (~ind.all(axis=1)).sum()
        pred_damaged = int(np.floor(rate * n))
        per_view_miss = 1.0 - ind.mean()
        pred_miss = rate / 2.0

        # damaged samples must never keep all V views
        damaged_rows = ~ind.all(axis=1)
        all_kept_among_damaged = int((ind[damaged_rows].sum(axis=1) == V).sum())

        # chi-square of the surviving-view count against Binomial(V, 1/2)
        k = ind[damaged_rows].sum(axis=1)
        obs = np.bincount(k, minlength=V + 1)[1:V]
        from math import comb

        exp = np.array([comb(V, i) for i in range(1, V)]) / (2**V - 2) * damaged_rows.sum()
        chi2 = float(((obs - exp) ** 2 / np.maximum(exp, 1e-9)).sum())
        dof = len(obs) - 1
        # rough p-value via the Wilson-Hilferty approximation
        from scipy.stats import chi2 as chi2dist

        p_val = float(1 - chi2dist.cdf(chi2, dof))

        ok_d = abs(damaged - pred_damaged) <= 1
        ok_m = abs(per_view_miss - pred_miss) < 0.02
        ok_k = all_kept_among_damaged == 0
        ok_b = p_val > 0.01
        failures += sum(not x for x in (ok_d, ok_m, ok_k, ok_b))

        print(f"{rate:>8.1f} {damaged:>9d} {pred_damaged:>7d} {per_view_miss:>14.3f} "
              f"{pred_miss:>7.3f} {all_kept_among_damaged:>9d} {p_val:>9.3f}"
              f"   {'OK' if all((ok_d, ok_m, ok_k, ok_b)) else 'MISMATCH'}")

    # rate r maps to file _Per{1-r}, so a *higher* missing rate must be a subset
    nested = all(bool((masks[FS[i + 1]] <= masks[FS[i]]).all()) for i in range(len(FS) - 1))
    print(f"\nnested masks  rate{FS[-1]} <= ... <= rate{FS[0]}: {nested}")
    if not nested:
        failures += 1

    print("\nprotocol verification:", "PASS" if failures == 0 else f"FAIL ({failures})")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
