"""Centralised upper bound: plain k-means on the pooled views.

EFIMVC writes every intermediate quantity as an anchor graph, so it is useful
to know what the *features themselves* support.  Two references are computed
per dataset / missing rate:

* ``concat0``   -- concatenate the normalised views, missing entries = 0
  (exactly the matrix the federated algorithm sees);
* ``concatmean``-- same but missing entries filled with the view's mean over
  its present samples.

Both are centralised (no privacy, no federation) and use the same k-means
read-out as the model, so they are an optimistic reference point.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import efimvc  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

from efimvc.data.loader import DATASETS, load_dataset  # noqa: E402
from efimvc.data.protocol import MISSING_RATES  # noqa: E402
from efimvc.metrics import PAPER_METRICS, clustering_8measure  # noqa: E402

RESULTS = ROOT / "results"


def fill_mean(views, ind):
    out = []
    for iv, X in enumerate(views):
        Y = X.copy()
        missing = ind[:, iv] == 0
        if missing.any() and (~missing).any():
            Y[:, missing] = X[:, ~missing].mean(axis=1, keepdims=True)
        out.append(Y)
    return out


def score(F, y, k, reps=20, seed=0):
    rng = np.random.default_rng(seed)
    res = [clustering_8measure(y, KMeans(k, n_init=20, max_iter=1000, random_state=int(rng.integers(2**31))).fit_predict(F))
           for _ in range(reps)]
    return {m: 100 * float(np.mean([r[m] for r in res])) for m in PAPER_METRICS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="proteinFold")
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    rows = []
    for rate in MISSING_RATES:
        views, ind, y = load_dataset(args.dataset, rate)
        k = len(np.unique(y))
        rec = {"dataset": args.dataset, "rate": rate, "k": k}
        rec["concat0"] = score(np.vstack(views).T, y, k, args.reps)
        rec["concatmean"] = score(np.vstack(fill_mean(views, ind)).T, y, k, args.reps)
        rows.append(rec)
        print(f"rate={rate}: concat0 ACC={rec['concat0']['ACC']:.2f} | "
              f"concatmean ACC={rec['concatmean']['ACC']:.2f}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"upper_bound_{args.dataset}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")
    for key in ("concat0", "concatmean"):
        mean = {m: float(np.mean([r[key][m] for r in rows])) for m in PAPER_METRICS}
        print(f"mean over 9 rates [{key}]: " + "  ".join(f"{m}={v:.2f}" for m, v in mean.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
