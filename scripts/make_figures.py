"""Figures 2, 3 and 4 of the paper, rebuilt from ``results/*.jsonl``.

* ``fig2_missing_rate`` -- metric vs missing rate, one panel per dataset
  (paper Fig. 2 / 5 / 6 / 7).
* ``fig3_convergence``  -- client / server / overall objective vs iteration
  (paper Fig. 3).  Needs ``obj_client_hist`` / ``obj_server_hist``, which
  ``run_protocol.py`` records.
* ``fig4_sensitivity``  -- alpha x beta heat maps (paper Fig. 4 / 8).

    python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
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
FIGS = RESULTS / "figures"
ALPHAS = [0.001, 0.1, 1.0, 100.0]
BETAS = [0.001, 0.1, 1.0, 100.0]


def read_all() -> list[dict]:
    rows = []
    for p in sorted(RESULTS.glob("protocol_*.jsonl")):
        if "_pilot" in p.stem or "_bigm" in p.stem:
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def per_rate_best(rows, dataset, metric, rate):
    sub = [r for r in rows if r["dataset"] == dataset and r["rate"] == rate]
    return max((r[metric] for r in sub), default=None)


def fig_missing_rate(plt, rows, datasets, metric):
    n = len(datasets)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False)
    for i, ds in enumerate(datasets):
        ax = axes[i // ncol][i % ncol]
        xs, ys = [], []
        for rate in MISSING_RATES:
            v = per_rate_best(rows, ds, metric, rate)
            if v is not None:
                xs.append(rate)
                ys.append(100 * v)
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], "o-", lw=2, ms=5)
        ax.set_title(ds)
        ax.set_xlabel("missing rate")
        ax.set_ylabel(f"{metric} (%)")
        ax.grid(alpha=0.3)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"EFIMVC -- {metric} vs missing rate (best over the paper's grid)")
    fig.tight_layout()
    out = FIGS / f"fig2_missing_rate_{metric}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def fig_convergence(plt, rows, datasets):
    have = [r for r in rows if "obj_client_hist" in r]
    if not have:
        print("no objective histories recorded -- skipping convergence figure")
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 3.4), squeeze=False)
    for i, ds in enumerate(datasets):
        sub = [r for r in have if r["dataset"] == ds]
        if not sub:
            continue
        # pick the configuration with the lowest final total objective
        best = min(sub, key=lambda r: r["obj_final"])
        it = np.arange(1, len(best["obj_client_hist"]) + 1)
        ax = axes[0][i]
        ax.plot(it, best["obj_client_hist"], "o-", label="client")
        ax.plot(it, best["obj_server_hist"], "s-", label="server")
        ax.plot(it, best["obj_hist"], "^-", label="all")
        ax.set_title(ds)
        ax.set_xlabel("iteration")
        ax.set_ylabel("objective value")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGS / "fig3_convergence.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def fig_sensitivity(plt, rows, datasets, metric):
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.4 * len(datasets), 3.8), squeeze=False)
    for i, ds in enumerate(datasets):
        table = np.full((len(ALPHAS), len(BETAS)), np.nan)
        for a_i, a in enumerate(ALPHAS):
            for b_i, b in enumerate(BETAS):
                vals = []
                for rate in MISSING_RATES:
                    sub = [
                        r for r in rows
                        if r["dataset"] == ds and r["rate"] == rate
                        and abs(r["alpha"] - a) < 1e-12 and abs(r["beta"] - b) < 1e-12
                    ]
                    if sub:
                        vals.append(max(r[metric] for r in sub))
                if vals:
                    table[a_i, b_i] = 100 * float(np.mean(vals))
        ax = axes[0][i]
        im = ax.imshow(table, cmap="viridis")
        ax.set_xticks(range(len(BETAS)), [str(b) for b in BETAS])
        ax.set_yticks(range(len(ALPHAS)), [str(a) for a in ALPHAS])
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"$\alpha$")
        ax.set_title(ds)
        for r_i in range(table.shape[0]):
            for c_i in range(table.shape[1]):
                if np.isfinite(table[r_i, c_i]):
                    ax.text(c_i, r_i, f"{table[r_i, c_i]:.0f}", ha="center", va="center", color="w", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"EFIMVC -- sensitivity of {metric} (%) to alpha / beta")
    fig.tight_layout()
    out = FIGS / f"fig4_sensitivity_{metric}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="ACC")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(parents=True, exist_ok=True)
    rows = read_all()
    if not rows:
        print("no protocol results found; run scripts/run_protocol.py first")
        return 2
    datasets = sorted({r["dataset"] for r in rows})
    print("datasets:", datasets, f"({len(rows)} records)")

    for metric in PAPER_METRICS if args.metric == "all" else [args.metric]:
        fig_missing_rate(plt, rows, datasets, metric)
    fig_sensitivity(plt, rows, datasets, "ACC")
    fig_convergence(plt, rows, datasets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
