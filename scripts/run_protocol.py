"""Main comparison protocol: Table 2 of the paper.

For every dataset, every missing rate and every point of the hyper-parameter
grid

    anchors m   in  {k, 2k, 3k}
    alpha       in  {0.001, 0.1, 1, 100}     (client regularisation, lambda2)
    beta        in  {0.001, 0.1, 1, 100}     (server regularisation, lambda3)

run EFIMVC exactly as ``official_code/run.m`` does, i.e. read out the left
singular vectors of ``Z`` with k-means repeated ``--reps`` times (each with 20
internal restarts) and report the **mean** over the repetitions.

Aggregation mirrors ``run.m``: per missing rate keep the best value of each
metric over the grid, then average those nine numbers.

Resumable: every finished (rate, m, alpha, beta) is appended to a JSONL file
and skipped on the next invocation.

    python scripts/run_protocol.py --dataset proteinFold --jobs 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import efimvc  # noqa: E402  (pins BLAS threads before numpy loads)

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from efimvc.core.efimvc import EFIMVC  # noqa: E402
from efimvc.data.loader import DATASETS, load_dataset  # noqa: E402
from efimvc.metrics import PAPER_METRICS, clustering_8measure  # noqa: E402
from efimvc.data.protocol import MISSING_RATES  # noqa: E402

RESULTS = ROOT / "results"


DEFAULT_ALPHAS = (0.001, 0.1, 1.0, 100.0)
DEFAULT_BETAS = (0.001, 0.1, 1.0, 100.0)


def grid_for(k: int, anchors=None, alphas=None, betas=None):
    return (
        list(anchors) if anchors else [k, 2 * k, 3 * k],
        list(alphas) if alphas else list(DEFAULT_ALPHAS),
        list(betas) if betas else list(DEFAULT_BETAS),
    )


def run_one(dataset: str, rate: float, m: int, alpha: float, betas, reps: int, seed: int, carry: bool = True):
    """Fit once for (rate, m, alpha) and evaluate every beta.

    ``carry=True`` reproduces ``run.m``: ``client_pretrain`` is run once per
    ``(m, alpha)`` and ``Z``/``Zall`` are handed from one ``beta`` to the next
    (``P`` is reset to ``I`` each time).  ``carry=False`` restarts from scratch
    for every ``beta``.
    """
    views, ind, y = load_dataset(dataset, rate, seed=seed)
    k = len(np.unique(y))
    out = []
    warm_Z = warm_Zall = None
    for beta in betas:
        t0 = time.time()
        model = EFIMVC(n_clusters=k, n_anchors=m, alpha=alpha, beta=beta, seed=seed).fit(
            views, ind, init_Z=warm_Z, init_Zall=warm_Zall
        )
        if carry:
            warm_Z = [np.array(z) for z in model.Z_]
            warm_Zall = np.array(model.Zall_)
        labels = model.predict(n_reps=reps)
        res = [clustering_8measure(y, lab) for lab in labels]
        rec = {
            "dataset": dataset,
            "rate": round(float(rate), 2),
            "m": int(m),
            "k": int(k),
            "alpha": float(alpha),
            "beta": float(beta),
            "n_iters": int(len(model.history_)),
            "obj_final": float(model.history_[-1]),
            "obj_hist": [float(v) for v in model.history_],
            "obj_client_hist": [float(v) for v in model.history_client_],
            "obj_server_hist": [float(v) for v in model.history_server_],
            "seconds": round(time.time() - t0, 2),
            "carry": bool(carry),
        }
        for met in PAPER_METRICS:
            vals = np.array([r[met] for r in res], dtype=float)
            rec[f"{met}"] = float(vals.mean())
            rec[f"{met}_std"] = float(vals.std())
        out.append(rec)
    return out


def load_done(path: Path) -> set:
    done = set()
    if path.exists():
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((r["dataset"], r["rate"], r["m"], r["alpha"], r["beta"]))
    return done


def evaluate(records_path: Path, dataset: str, jobs: int, reps: int, seed: int, rates, grid=None, carry=True):
    k = DATASETS[dataset][2]
    ms, alphas, betas = grid_for(k, *(grid or (None, None, None)))
    done = load_done(records_path)

    jobs_list = []
    for rate in rates:
        for m in ms:
            for alpha in alphas:
                todo = [b for b in betas if (dataset, round(float(rate), 2), m, alpha, b) not in done]
                if todo:
                    jobs_list.append((rate, m, alpha, todo))

    print(f"[{dataset}] {len(jobs_list)} jobs to run ({len(done)} cells already done)", flush=True)
    if not jobs_list:
        return

    with records_path.open("a", encoding="utf-8") as fh:
        gen = Parallel(n_jobs=jobs, return_as="generator_unordered")(
            delayed(run_one)(dataset, rate, m, alpha, tuple(todo), reps, seed, carry)
            for (rate, m, alpha, todo) in jobs_list
        )
        for i, recs in enumerate(gen, 1):
            for rec in recs:
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 5 == 0 or i == len(jobs_list):
                print(f"  [{dataset}] {i}/{len(jobs_list)} jobs done", flush=True)


def summarise(records_path: Path, dataset: str):
    """Replicate ``run.m``: per rate best-over-grid per metric, then mean over rates."""
    rows = [json.loads(l) for l in records_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["dataset"] == dataset]
    if not rows:
        return None
    rates = sorted({r["rate"] for r in rows})
    per_rate = {}
    for rate in rates:
        sub = [r for r in rows if r["rate"] == rate]
        per_rate[rate] = {met: max(r[met] for r in sub) for met in PAPER_METRICS}
    table = {met: float(np.mean([per_rate[r][met] for r in rates])) * 100 for met in PAPER_METRICS}
    return {"table2": table, "per_rate_best": per_rate, "n_cells": len(rows)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="proteinFold", choices=sorted(DATASETS))
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--reps", type=int, default=20, help="k-means read-out repetitions")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rates", default="all", help="'all' or comma separated")
    ap.add_argument("--alphas", default=None, help="override the alpha grid, e.g. 0.001,1")
    ap.add_argument("--betas", default=None, help="override the beta grid")
    ap.add_argument("--anchors", default=None, help="override the anchors grid, e.g. 27,54")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-carry", action="store_true",
                    help="do not hand Z between beta values (run.m does)")
    args = ap.parse_args()

    rates = MISSING_RATES if args.rates == "all" else tuple(float(x) for x in args.rates.split(","))
    grid = (
        [float(x) for x in args.anchors.split(",")] if args.anchors else None,
        [float(x) for x in args.alphas.split(",")] if args.alphas else None,
        [float(x) for x in args.betas.split(",")] if args.betas else None,
    )
    RESULTS.mkdir(exist_ok=True)
    rec_path = RESULTS / f"protocol_{args.dataset}{args.tag}.jsonl"

    evaluate(rec_path, args.dataset, args.jobs, args.reps, args.seed, rates, grid, carry=not args.no_carry)
    summary = summarise(rec_path, args.dataset)
    if summary:
        print(f"\n[{args.dataset}] Table-2 style (mean over 9 rates of the per-rate grid best):")
        for met, v in summary["table2"].items():
            print(f"   {met:8s} {v:6.2f}")
        (RESULTS / f"protocol_{args.dataset}{args.tag}_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
