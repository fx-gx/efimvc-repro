"""Hyper-parameter sensitivity -- Figure 4 / Figure 8 of the paper.

The official grid already contains every (alpha, beta) combination, so this
script only *re-aggregates* ``results/protocol_<dataset>.jsonl`` (produced by
``run_protocol.py``); it runs no new experiments.

Aggregation for a fixed (alpha, beta):

    value(alpha, beta) = mean over the nine missing rates of
                         max over m in {k, 2k, 3k} of the metric

which is the same per-rate-best-then-average rule used for Table 2, just
without maximising over (alpha, beta).

Writes ``results/sensitivity_<dataset>.csv`` and a heat-map per metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import efimvc  # noqa: E402,F401

import numpy as np  # noqa: E402

from efimvc.data.protocol import MISSING_RATES  # noqa: E402
from efimvc.metrics import PAPER_METRICS  # noqa: E402

RESULTS = ROOT / "results"
ALPHAS = [0.001, 0.1, 1.0, 100.0]
BETAS = [0.001, 0.1, 1.0, 100.0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="proteinFold")
    ap.add_argument("--metric", default="ACC", choices=list(PAPER_METRICS))
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    path = RESULTS / f"protocol_{args.dataset}.jsonl"
    if not path.exists():
        print(f"missing {path}; run scripts/run_protocol.py first", file=sys.stderr)
        return 2
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["dataset"] == args.dataset]

    table = np.full((len(ALPHAS), len(BETAS)), np.nan)
    for i, a in enumerate(ALPHAS):
        for j, b in enumerate(BETAS):
            per_rate = []
            for rate in MISSING_RATES:
                sub = [r for r in rows if abs(r["alpha"] - a) < 1e-12 and abs(r["beta"] - b) < 1e-12 and r["rate"] == rate]
                if sub:
                    per_rate.append(max(r[args.metric] for r in sub))
            if per_rate:
                table[i, j] = 100 * float(np.mean(per_rate))

    out = RESULTS / f"sensitivity_{args.dataset}.csv"
    RESULTS.mkdir(exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["alpha\\beta"] + [str(b) for b in BETAS])
        for i, a in enumerate(ALPHAS):
            w.writerow([str(a)] + [f"{v:.4f}" if np.isfinite(v) else "" for v in table[i]])
    print(f"[{args.dataset}] {args.metric} (%) -- rows alpha, cols beta")
    print("        " + "".join(f"{b:>9}" for b in BETAS))
    for i, a in enumerate(ALPHAS):
        print(f"{a:>7}" + "".join(f"{v:9.2f}" if np.isfinite(v) else f"{'--':>9}" for v in table[i]))
    best = np.unravel_index(np.nanargmax(table), table.shape)
    print(f"\nbest: alpha={ALPHAS[best[0]]} beta={BETAS[best[1]]} -> {table[best]:.2f}%")
    print(f"wrote {out}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        im = ax.imshow(table, cmap="viridis")
        ax.set_xticks(range(len(BETAS)), [str(b) for b in BETAS])
        ax.set_yticks(range(len(ALPHAS)), [str(a) for a in ALPHAS])
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"$\alpha$")
        ax.set_title(f"{args.dataset} -- {args.metric} (%)")
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                if np.isfinite(table[i, j]):
                    ax.text(j, i, f"{table[i, j]:.1f}", ha="center", va="center", color="w", fontsize=8)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(RESULTS / "figures" / f"sensitivity_{args.dataset}_{args.metric}.png", dpi=150)
        print(f"wrote results/figures/sensitivity_{args.dataset}_{args.metric}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
