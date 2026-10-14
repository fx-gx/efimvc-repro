"""Check the metric implementations against independent references.

* ACC       -- brute-force over all permutations for tiny label vectors
* NMI       -- sklearn ``normalized_mutual_info_score(average_method='arithmetic')``
* Purity    -- direct re-implementation from the definition
* Fscore    -- pairwise F1 recomputed with an O(n^2) boolean loop
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import efimvc  # noqa: E402,F401

import numpy as np  # noqa: E402
from sklearn.metrics import normalized_mutual_info_score  # noqa: E402

from efimvc.metrics import acc, nmi, purity, pairwise_f, rand_index  # noqa: E402


def brute_force_acc(a, b):
    classes = sorted(set(b.tolist()))
    best = 0.0
    for perm in itertools.permutations(range(len(classes) + 1)):
        mapping = {c: perm[i] for i, c in enumerate(classes)}
        pred = np.array([mapping[v] for v in b])
        best = max(best, float((pred == a).mean()))
    return best


def brute_force_purity(a, b):
    tot = 0
    for c in np.unique(b):
        _, cnt = np.unique(a[b == c], return_counts=True)
        tot += cnt.max()
    return tot / a.size


def brute_force_pairwise_f(a, b):
    n = a.size
    tp = fp = fn = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_a = a[i] == a[j]
            same_b = b[i] == b[j]
            if same_a and same_b:
                tp += 1
            elif same_b and not same_a:
                fp += 1
            elif same_a and not same_b:
                fn += 1
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def brute_force_rand(a, b):
    n = a.size
    agree = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if (a[i] == a[j]) == (b[i] == b[j]):
                agree += 1
    return agree / total if total else 1.0


def main() -> int:
    rng = np.random.default_rng(0)
    worst = {"acc": 0.0, "nmi": 0.0, "purity": 0.0, "f": 0.0, "ari": 0.0}
    for trial in range(60):
        n = rng.integers(20, 60)
        k = int(rng.integers(2, 5))
        y_true = rng.integers(0, k, n)
        y_pred = rng.integers(0, k, n)
        worst["acc"] = max(worst["acc"], abs(acc(y_true, y_pred) - brute_force_acc(y_true, y_pred)))
        worst["nmi"] = max(
            worst["nmi"], abs(nmi(y_true, y_pred) - normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic"))
        )
        worst["purity"] = max(worst["purity"], abs(purity(y_true, y_pred) - brute_force_purity(y_true, y_pred)))
        worst["f"] = max(worst["f"], abs(pairwise_f(y_true, y_pred)[0] - brute_force_pairwise_f(y_true, y_pred)))
        worst["ari"] = max(worst["ari"], abs(rand_index(y_true, y_pred) - brute_force_rand(y_true, y_pred)))

    ok = True
    for name, v in worst.items():
        good = v < 1e-9
        ok &= bool(good)
        print(f"  {name:7s} max |mine - reference| = {v:.3e}   {'OK' if good else 'FAIL'}")
    print("metrics verification:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
