# EFIMVC 复现实验结果

由 `scripts/make_experiments_report.py` 自动生成。
聚合方式完全照搬 `official_code/run.m`：**每个缺失率下，各指标在超参网格上取最优，
再对 9 个缺失率取平均**（因此不同指标可能对应不同的超参）。

记录文件：`results/protocol_*.jsonl`、`results/ablation_*.jsonl`。

## 表 2 复现——聚类性能（%，均值 over 9 missing rates）

| 数据集 | 指标 | 复现 | 论文 Proposed | 差值 |
|---|---|---|---|---|
| 100Leaves | ACC | **71.11** |  72.88 | -1.77 |
| 100Leaves | NMI | **82.94** |  84.76 | -1.82 |
| 100Leaves | Purity | **72.63** |  74.79 | -2.16 |
| 100Leaves | Fscore | **57.78** |  61.84 | -4.06 |
| ORL_mtv | ACC | **81.40** |    --  | -- |
| ORL_mtv | NMI | **90.33** |    --  | -- |
| ORL_mtv | Purity | **83.80** |    --  | -- |
| ORL_mtv | Fscore | **74.31** |    --  | -- |
| WebKB | ACC | **92.25** |  90.64 | +1.61 |
| WebKB | NMI | **54.87** |  50.62 | +4.25 |
| WebKB | Purity | **92.25** |  90.64 | +1.61 |
| WebKB | Fscore | **89.19** |  86.80 | +2.39 |
| proteinFold | ACC | **31.79** |  31.71 | +0.08 |
| proteinFold | NMI | **40.71** |  40.33 | +0.38 |
| proteinFold | Purity | **38.25** |  37.95 | +0.30 |
| proteinFold | Fscore | **18.56** |  18.71 | -0.15 |

## 表 3 复现——消融（%）

| 数据集 | 指标 | 变体 | 复现 | 论文 | 差值 |
|---|---|---|---|---|---|
| WebKB | ACC | full | **92.23** |  90.64 | +1.59 |
| WebKB | ACC | lambda=0 | **87.36** |  90.36 | -3.00 |
| WebKB | ACC | beta=0 | **92.26** |  78.68 | +13.58 |
| WebKB | ACC | P=I | **85.20** |  73.61 | +11.59 |
| WebKB | NMI | full | **54.87** |  50.62 | +4.25 |
| WebKB | NMI | lambda=0 | **42.99** |  49.78 | -6.79 |
| WebKB | NMI | beta=0 | **55.01** |  29.43 | +25.58 |
| WebKB | NMI | P=I | **38.35** |  22.66 | +15.69 |
| WebKB | Purity | full | **92.23** |  90.64 | +1.59 |
| WebKB | Purity | lambda=0 | **87.77** |  90.36 | -2.59 |
| WebKB | Purity | beta=0 | **92.26** |  82.34 | +9.92 |
| WebKB | Purity | P=I | **86.17** |  81.60 | +4.57 |
| WebKB | Fscore | full | **89.17** |  86.80 | +2.37 |
| WebKB | Fscore | lambda=0 | **82.77** |  86.42 | -3.65 |
| WebKB | Fscore | beta=0 | **89.22** |  73.21 | +16.01 |
| WebKB | Fscore | P=I | **80.50** |  69.83 | +10.67 |
| proteinFold | ACC | full | **31.61** |  31.71 | -0.10 |
| proteinFold | ACC | lambda=0 | **31.30** |  30.85 | +0.45 |
| proteinFold | ACC | beta=0 | **30.93** |  30.80 | +0.13 |
| proteinFold | ACC | P=I | **20.86** |  20.18 | +0.68 |
| proteinFold | NMI | full | **40.22** |  40.33 | -0.11 |
| proteinFold | NMI | lambda=0 | **40.48** |  40.17 | +0.31 |
| proteinFold | NMI | beta=0 | **40.16** |  39.87 | +0.29 |
| proteinFold | NMI | P=I | **27.97** |  26.96 | +1.01 |
| proteinFold | Purity | full | **37.86** |  37.95 | -0.09 |
| proteinFold | Purity | lambda=0 | **38.24** |  37.83 | +0.41 |
| proteinFold | Purity | beta=0 | **37.82** |  37.79 | +0.03 |
| proteinFold | Purity | P=I | **26.14** |  25.21 | +0.93 |
| proteinFold | Fscore | full | **18.31** |  18.71 | -0.40 |
| proteinFold | Fscore | lambda=0 | **18.11** |  17.79 | +0.32 |
| proteinFold | Fscore | beta=0 | **17.83** |  17.81 | +0.02 |
| proteinFold | Fscore | P=I | **9.59** |   9.01 | +0.58 |

## 复现中各数据集的最优配置

- **100Leaves**: ACC 最优时 m=200 (k=100), alpha=0.001, beta=0.001, rate=0.1 -> 86.26%
- **ORL_mtv**: ACC 最优时 m=120 (k=40), alpha=1.0, beta=0.001, rate=0.3 -> 87.60%
- **WebKB**: ACC 最优时 m=4 (k=2), alpha=0.1, beta=0.001, rate=0.1 -> 96.57%
- **proteinFold**: ACC 最优时 m=81 (k=27), alpha=0.001, beta=1.0, rate=0.2 -> 33.38%

## 网格

默认网格（与 `official_code/run.m` 一致）：
`m ∈ {k, 2k, 3k}`，`alpha ∈ {0.001, 0.1, 1, 100}`，`beta ∈ {0.001, 0.1, 1, 100}`，
缺失率 `{0.1, ..., 0.9}`（共 9 档），读出方式为 20 次 k-means 取均值。

**使用了缩减网格的数据集**（计算量原因，会低估其「网格最优」，比较时需注意）：

- `ORL_mtv`：`alpha, beta ∈ {0.001, 1}`（视图维度最高 6750，KMeans 初始化很慢）
- `100Leaves`：`m ∈ {k, 2k}`，`alpha, beta ∈ {0.001, 1}`
