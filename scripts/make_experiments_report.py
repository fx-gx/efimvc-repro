"""Summarise every ``results/*.jsonl`` into ``results/EXPERIMENTS.md``.

The aggregation is the one ``official_code/run.m`` uses: for a given dataset
and missing rate keep the best value of each metric over the hyper-parameter
grid, then average those nine numbers.  The paper's Table 2 / Table 3 columns
are shown next to the reproduced ones.
"""

from __future__ import annotations

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

#: paper Table 2, "Proposed" row
PAPER_TABLE2 = {
    "proteinFold": {"ACC": 31.71, "NMI": 40.33, "Purity": 37.95, "Fscore": 18.71},
    "WebKB": {"ACC": 90.64, "NMI": 50.62, "Purity": 90.64, "Fscore": 86.80},
    "100Leaves": {"ACC": 72.88, "NMI": 84.76, "Purity": 74.79, "Fscore": 61.84},
    "CCV": {"ACC": 20.04, "NMI": 15.66, "Purity": 23.45, "Fscore": 11.79},
    "ORL_mtv": None,
}

#: paper Table 3, per dataset, per variant
PAPER_TABLE3 = {
    "proteinFold": {
        "ACC": {"lambda=0": 30.85, "beta=0": 30.80, "P=I": 20.18, "full": 31.71},
        "NMI": {"lambda=0": 40.17, "beta=0": 39.87, "P=I": 26.96, "full": 40.33},
        "Purity": {"lambda=0": 37.83, "beta=0": 37.79, "P=I": 25.21, "full": 37.95},
        "Fscore": {"lambda=0": 17.79, "beta=0": 17.81, "P=I": 9.01, "full": 18.71},
    },
    "WebKB": {
        "ACC": {"lambda=0": 90.36, "beta=0": 78.68, "P=I": 73.61, "full": 90.64},
        "NMI": {"lambda=0": 49.78, "beta=0": 29.43, "P=I": 22.66, "full": 50.62},
        "Purity": {"lambda=0": 90.36, "beta=0": 82.34, "P=I": 81.60, "full": 90.64},
        "Fscore": {"lambda=0": 86.42, "beta=0": 73.21, "P=I": 69.83, "full": 86.80},
    },
    "100Leaves": {
        "ACC": {"lambda=0": 68.12, "beta=0": 64.09, "P=I": 54.46, "full": 72.88},
        "NMI": {"lambda=0": 82.10, "beta=0": 79.91, "P=I": 74.86, "full": 84.76},
        "Purity": {"lambda=0": 70.44, "beta=0": 66.66, "P=I": 56.83, "full": 74.79},
        "Fscore": {"lambda=0": 55.60, "beta=0": 50.67, "P=I": 39.62, "full": 61.84},
    },
    "CCV": {
        "ACC": {"lambda=0": 14.85, "beta=0": 17.96, "P=I": 14.82, "full": 20.04},
        "NMI": {"lambda=0": 10.77, "beta=0": 12.90, "P=I": 10.96, "full": 15.66},
        "Purity": {"lambda=0": 18.54, "beta=0": 21.48, "P=I": 18.70, "full": 23.45},
        "Fscore": {"lambda=0": 9.06, "beta=0": 10.34, "P=I": 9.05, "full": 11.79},
    },
}


def read(path: Path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def dedupe_protocol(rows):
    seen, out = set(), []
    for r in rows:
        key = (r["dataset"], r["rate"], r["m"], r["alpha"], r["beta"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def table2_agg(rows, dataset):
    sub = dedupe_protocol([r for r in rows if r["dataset"] == dataset])
    if not sub:
        return None
    out = {}
    for met in PAPER_METRICS:
        per_rate = []
        for rate in sorted({r["rate"] for r in sub}):
            vals = [r[met] for r in sub if r["rate"] == rate]
            if vals:
                per_rate.append(max(vals))
        if per_rate:
            out[met] = 100 * float(np.mean(per_rate))
    return out


def dedupe(rows):
    """Drop duplicate cells.

    ``run_ablation`` pins ``alpha = 0`` for the ``lambda=0`` variant and
    ``beta = 0`` for ``beta=0``, so the *effective* hyper-parameters no longer
    identify the grid cell and older records appear several times.  The
    duplicates are identical computations, so ``max`` is unaffected -- but the
    counts (and any average) must not see them twice.
    """
    seen = set()
    out = []
    for r in rows:
        key = (
            r.get("dataset"), r.get("variant"), r.get("rate"), r.get("m"),
            r.get("alpha_grid", r.get("alpha")), r.get("beta_grid", r.get("beta")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def ablation_agg(rows, dataset):
    sub = dedupe([r for r in rows if r["dataset"] == dataset])
    if not sub:
        return {}
    variants = sorted({r["variant"] for r in sub})
    out = {}
    for met in PAPER_METRICS:
        out[met] = {}
        for variant in variants:
            rows_v = [r for r in sub if r["variant"] == variant]
            per_rate = []
            for rate in sorted({r["rate"] for r in rows_v}):
                vals = [r[met] for r in rows_v if r["rate"] == rate]
                if vals:
                    per_rate.append(max(vals))
            if per_rate:
                out[met][variant] = 100 * float(np.mean(per_rate))
    return out


def fmt(v):
    return f"{v:6.2f}" if v is not None else "   -- "


def main() -> int:
    protocol = []
    for p in sorted(RESULTS.glob("protocol_*.jsonl")):
        if "_pilot" in p.stem or "_bigm" in p.stem:
            continue
        protocol += read(p)
    ablation = []
    for p in sorted(RESULTS.glob("ablation_*.jsonl")):
        ablation += read(p)

    datasets = sorted({r["dataset"] for r in protocol})
    lines = [
        "# EFIMVC 复现实验结果",
        "",
        "由 `scripts/make_experiments_report.py` 自动生成。",
        "聚合方式完全照搬 `official_code/run.m`：**每个缺失率下，各指标在超参网格上取最优，",
        "再对 9 个缺失率取平均**（因此不同指标可能对应不同的超参）。",
        "",
        "记录文件：`results/protocol_*.jsonl`、`results/ablation_*.jsonl`。",
        "",
        "## 表 2 复现——聚类性能（%，均值 over 9 missing rates）",
        "",
    ]

    header = "| 数据集 | 指标 | 复现 | 论文 Proposed | 差值 |"
    lines += [header, "|---|---|---|---|---|"]
    for ds in datasets:
        got = table2_agg(protocol, ds)
        if not got:
            continue
        ref = PAPER_TABLE2.get(ds)
        for met in PAPER_METRICS:
            r = ref.get(met) if ref else None
            d = f"{got[met] - r:+.2f}" if r is not None else "--"
            lines.append(f"| {ds} | {met} | **{got[met]:.2f}** | {fmt(r)} | {d} |")

    if ablation:
        lines += ["", "## 表 3 复现——消融（%）", ""]
        lines += ["| 数据集 | 指标 | 变体 | 复现 | 论文 | 差值 |", "|---|---|---|---|---|---|"]
        for ds in sorted({r["dataset"] for r in ablation}):
            got = ablation_agg(ablation, ds)
            ref = PAPER_TABLE3.get(ds, {})
            for met in PAPER_METRICS:
                for variant in ("full", "lambda=0", "beta=0", "P=I"):
                    if met not in got or variant not in got[met]:
                        continue
                    g = got[met][variant]
                    r = ref.get(met, {}).get(variant)
                    d = f"{g - r:+.2f}" if r is not None else "--"
                    lines.append(f"| {ds} | {met} | {variant} | **{g:.2f}** | {fmt(r)} | {d} |")

    best = []
    for ds in datasets:
        sub = dedupe_protocol([r for r in protocol if r["dataset"] == ds])
        if sub:
            b = max(sub, key=lambda r: r["ACC"])
            best.append(
                f"- **{ds}**: ACC 最优时 m={b['m']} (k={b['k']}), alpha={b['alpha']}, "
                f"beta={b['beta']}, rate={b['rate']} -> {100*b['ACC']:.2f}%"
            )
    if best:
        lines += ["", "## 复现中各数据集的最优配置", ""] + best

    lines += [
        "",
        "## 网格",
        "",
        "默认网格（与 `official_code/run.m` 一致）：",
        "`m ∈ {k, 2k, 3k}`，`alpha ∈ {0.001, 0.1, 1, 100}`，`beta ∈ {0.001, 0.1, 1, 100}`，",
        "缺失率 `{0.1, ..., 0.9}`（共 9 档），读出方式为 20 次 k-means 取均值。",
        "",
        "**使用了缩减网格的数据集**（计算量原因，会低估其「网格最优」，比较时需注意）：",
        "",
        "- `ORL_mtv`：`alpha, beta ∈ {0.001, 1}`（视图维度最高 6750，KMeans 初始化很慢）",
        "- `100Leaves`：`m ∈ {k, 2k}`，`alpha, beta ∈ {0.001, 1}`",
        "",
    ]

    out = RESULTS / "EXPERIMENTS.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}  ({len(dedupe_protocol(protocol))} distinct protocol cells, {len(dedupe(ablation))} distinct ablation cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
