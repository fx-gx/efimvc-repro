"""Ablation study -- Table 3 of the paper.

Four variants, exactly the three removals the paper describes plus the full
model:

===========  =========================================================
``full``     everything on
``lambda=0`` drop the feature-alignment regularisation (``alpha = 0``)
``beta=0``   drop the anchor-similarity / structure term (``beta = 0``)
``P=I``      drop the anchor-alignment matrices (``P^(v)`` frozen at ``I``)
===========  =========================================================

Same hyper-parameter grid and the same aggregation as ``run_protocol.py``
(per missing rate: best over the grid, then mean over the nine rates), so the
``full`` row reproduces the Table-2 "Proposed" numbers.

    python scripts/run_ablation.py --dataset proteinFold --jobs 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import efimvc  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from efimvc.core.efimvc import EFIMVC  # noqa: E402
from efimvc.data.loader import DATASETS, load_dataset  # noqa: E402
from efimvc.data.protocol import MISSING_RATES  # noqa: E402
from efimvc.metrics import PAPER_METRICS, clustering_8measure  # noqa: E402

RESULTS = ROOT / "results"

VARIANTS = ("full", "lambda=0", "beta=0", "P=I")


def grid_for(k):
    return [k, 2 * k, 3 * k], [0.001, 0.1, 1.0, 100.0], [0.001, 0.1, 1.0, 100.0]


def run_one(dataset, rate, m, alpha, betas, variant, reps, seed):
    views, ind, y = load_dataset(dataset, rate, seed=seed)
    k = len(np.unique(y))
    common = {}
    grid_alpha, grid_beta = alpha, None
    if variant == "lambda=0":
        alpha = 0.0          # the ablation sets lambda = 0 regardless of the grid
    if variant == "P=I":
        common["anchor_alignment"] = False
    out = []
    for beta in betas:
        grid_beta = beta
        if variant == "beta=0":
            beta = 0.0
        t0 = time.time()
        model = EFIMVC(n_clusters=k, n_anchors=m, alpha=alpha, beta=beta, seed=seed, **common).fit(views, ind)
        labels = model.predict(n_reps=reps)
        res = [clustering_8measure(y, lab) for lab in labels]
        rec = {
            "dataset": dataset, "variant": variant, "rate": round(float(rate), 2),
            "m": int(m), "alpha": float(alpha), "beta": float(beta),
            # for lambda=0 / beta=0 the effective value is pinned to 0, so also
            # keep the grid cell that produced this run (otherwise every alpha
            # collapses onto the same key and the run is duplicated 4x)
            "alpha_grid": float(grid_alpha), "beta_grid": float(grid_beta),
            "obj_final": float(model.history_[-1]), "seconds": round(time.time() - t0, 2),
        }
        for met in PAPER_METRICS:
            vals = np.array([r[met] for r in res], dtype=float)
            rec[met] = float(vals.mean())
        out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="proteinFold", choices=sorted(DATASETS))
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--rates", default="all")
    ap.add_argument("--select-from", default=None, metavar="METRIC",
                    help="instead of the full grid, evaluate every variant at the "
                         "hyper-parameters that were best for the full model in "
                         "results/protocol_<dataset>.jsonl (per missing rate, by METRIC)")
    args = ap.parse_args()

    variants = [v for v in args.variants.split(",") if v]
    rates = MISSING_RATES if args.rates == "all" else tuple(float(x) for x in args.rates.split(","))
    RESULTS.mkdir(exist_ok=True)
    rec_path = RESULTS / f"ablation_{args.dataset}.jsonl"

    done = set()
    if rec_path.exists():
        for line in rec_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["variant"], r["rate"], r["m"], r["alpha"], r["beta"]))

    jobs = []
    if args.select_from:
        proto = RESULTS / f"protocol_{args.dataset}.jsonl"
        rows = [json.loads(l) for l in proto.read_text(encoding="utf-8").splitlines() if l.strip()]
        ms, alphas, betas = [], [], []
        for rate in rates:
            sub = [r for r in rows if r["rate"] == rate]
            if not sub:
                continue
            best = max(sub, key=lambda r: r[args.select_from])
            m, alpha, beta = best["m"], best["alpha"], best["beta"]
            print(f"  rate {rate}: selected m={m} alpha={alpha} beta={beta} "
                  f"({args.select_from}={100 * best[args.select_from]:.2f})")
            for variant in variants:
                key = (variant, round(float(rate), 2), m, alpha, beta)
                if key not in done:
                    jobs.append((rate, m, alpha, [beta], variant))
    else:
        ms, alphas, betas = grid_for(DATASETS[args.dataset][2])
        for variant in variants:
            for rate in rates:
                for m in ms:
                    for alpha in alphas:
                        todo = [b for b in betas if (variant, round(float(rate), 2), m, alpha, b) not in done]
                        if todo:
                            jobs.append((rate, m, alpha, todo, variant))

    print(f"[{args.dataset}] ablation: {len(jobs)} jobs ({len(done)} cells already done)", flush=True)
    if not jobs:
        return 0

    with rec_path.open("a", encoding="utf-8") as fh:
        gen = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
            delayed(run_one)(args.dataset, rate, m, alpha, tuple(todo), variant, args.reps, args.seed)
            for (rate, m, alpha, todo, variant) in jobs
        )
        for i, recs in enumerate(gen, 1):
            for rec in recs:
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 5 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} jobs", flush=True)

    rows = [json.loads(l) for l in rec_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"\n[{args.dataset}] ablation (mean over rates of the per-rate grid best, %)")
    header = "variant   " + "".join(f"{m:>10s}" for m in PAPER_METRICS)
    print(header)
    for variant in variants:
        sub = [r for r in rows if r["variant"] == variant]
        if not sub:
            continue
        vals = []
        for met in PAPER_METRICS:
            per_rate = []
            for rate in sorted({r["rate"] for r in sub}):
                per_rate.append(max(r[met] for r in sub if r["rate"] == rate))
            vals.append(100 * float(np.mean(per_rate)))
        print(f"{variant:10s}" + "".join(f"{v:10.2f}" for v in vals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
