# EFIMVC 复现 —— 高效联邦不完整多视图聚类

复现论文 **《Efficient Federated Incomplete Multi-View Clustering》**
（Suyuan Liu, Hao Yu, Hao Tan, Ke Liang, Siwei Wang, Shengju Yu, En Zhu, Xinwang Liu，
ICML 2025, PMLR 267）。

> 官方源码（作者本人，MATLAB）：<https://github.com/Tracesource/EFIMVC>
> **📄 复现报告（全部结果与结论）：[`results/REPORT.md`](results/REPORT.md)**
> 过程笔记： [`NOTES.md`](NOTES.md)

## 这是什么

论文把**锚图（anchor graph）多视图聚类**放进**联邦学习**，解决三个问题：

1. **通信开销大** —— 已有 FMVC 每轮优化都要通信。EFIMVC 把变量更新**限制在本地**，
   只在本地优化收敛后才通信一次。
2. **隐私弱** —— 已有方法传 embedding（`n×d`）或相似度矩阵（`n×n`）。
   EFIMVC 只传**锚图** `Z^(v) ∈ R^{m×n}`（`m≪n`）。
3. **缺失视图** —— 用指示矩阵 `G^(v)`（即 `η^(v)`）只对存在的样本建模，
   天然支持不完整数据。

流程（论文 Algorithm 1 / 官方 `run.m`）：

```
初始化   客户端：k-means 取 m 个锚点 A^(v)，解一次式 (10) 得 Z^(v)
         服务器：Zall = Σ_v Z^(v) / 该样本存在的视图数
外层循环 客户端（每个视图并行）：交替更新 A^(v) (式 7) 与 Z^(v) (式 10)
                                 → 传 Z^(v)、S^(v) 给服务器
         服务器：更新对齐矩阵 P^(v) (式 12, 正交 Procrustes)
                 → 更新全局锚图 Z (式 14, 逐列 QP)
                 → 下发 P^(v)、Z
读出     对 Z' 做 SVD 取左奇异向量 → 行归一化 → k-means
```

## 项目结构

```
EFIMVC/
├── README.md
├── NOTES.md                    # 复现笔记：源码审计、协议反推、踩坑、结论
├── paper_text.txt              # 论文全文（PDF 提取）
├── official_code/              # 官方 MATLAB 代码（只读参考）+ 官方 9 个 ProteinFold 缺失文件
├── efimvc/
│   ├── qp.py                   # ⭐ 单纯形约束 QP 求解器（numba 加速，active-set + 热启动）
│   ├── metrics.py              # ACC/NMI/Purity/Fscore/… 对齐 MATLAB Clustering8Measure
│   ├── core/efimvc.py          # ⭐ EFIMVC 主算法（式 1–14 / Algorithm 1）
│   └── data/
│       ├── loader.py           # 数据集加载（.mat / v7.3 / 官方缺失文件）
│       └── protocol.py         # ⭐ 缺失数据生成协议（复刻官方生成脚本）
├── scripts/
│   ├── fetch_data.py           # 数据下载
│   ├── run_protocol.py         # ① 主对比实验（表 2，可断点续跑）
│   ├── run_ablation.py         # ② 消融（表 3）
│   ├── run_sensitivity.py      # ③ 超参敏感性（图 4 / 8，复用 ① 的结果）
│   ├── make_figures.py         # 出图 → results/figures/（图 2 / 3 / 4）
│   ├── make_experiments_report.py  # 汇总 → results/EXPERIMENTS.md
│   ├── verify_metrics.py       # 指标单测（对 sklearn / 暴力枚举）
│   └── verify_qp.py            # QP 求解器单测（KKT 条件 + 三种核互相对拍）
├── data/                       # 数据集
└── results/                    # 结果 jsonl / 图表 / 报告
```

## 环境

Python 3.12，`numpy scipy scikit-learn matplotlib joblib numba h5py`（`pip install` 即可）。

> ⚠️ **必须限制 BLAS 线程。** EFIMVC 全是 `m×m`（`m ≤ 300`）的小矩阵运算，
> 放开线程后 OpenBLAS/MKL 会为每次调用开满核心，实测 `server_train` 慢 **约 200 倍**。
> `efimvc/__init__.py` 已经默认把 `OMP/OPENBLAS/MKL/NUMEXPR/NUMBA_NUM_THREADS` 设为 1；
> 并行度请通过 `run_*.py --jobs N` 在**进程级**拿。

## 运行

```bash
python scripts/fetch_data.py                    # 下载数据（见下）
python scripts/verify_metrics.py                # 指标自检
python scripts/verify_qp.py                     # QP 自检
python scripts/run_protocol.py --dataset proteinFold --jobs 16
python scripts/run_ablation.py --dataset proteinFold --jobs 16
python scripts/run_sensitivity.py --dataset proteinFold --plot
python scripts/make_experiments_report.py && python scripts/make_figures.py
```

`run_protocol.py` / `run_ablation.py` **可断点续跑**（结果逐条追加进 jsonl）。

## 数据

论文表 1 的 7 个数据集里，**官方仓库只发布了 ProteinFold**（9 个缺失率版本）。
其余 6 个需要自己找，本仓库的处理如下：

| 数据集 | n / V / k | 来源 | 状态 |
|---|---|---|---|
| ProteinFold | 694 / 12 / 27 | **官方 `official_code/Data/`** | ✅ 直接使用官方 9 个文件 |
| WebKB | 1051 / 2 / 2 | `ChuanbinZhang/Multi-view-datasets` | ✅ 完整数据 + 自生成 9 档（视图维度 1840/3000，与论文的 334/2949 不同） |
| 100Leaves | 1600 / 3 / 100 | `ChuanbinZhang/Multi-view-datasets` | ✅ 完整数据 + 自生成（维度 64/64/64，**与论文完全一致**） |
| ORL_mtv | 400 / 3 / 40 | `Tracesource/FIMVC-VIA`（同作者前作） | ✅ 官方已生成好的 9 档缺失文件 |
| CCV | 6773 / 3 / 20 | 无可达下载源 | ❌ 未获取 |
| YTF10 | 38654 / 4 / 10 | 仅百度网盘 | ❌ 未获取 |
| CIFAR10 | 50000 / 3 / 10 | 仅百度网盘 / Google Drive | ❌ 未获取 |
| MNIST | 60000 / 3 / 10 | 仅百度网盘 | ❌ 未获取 |

## 缺失数据生成协议

论文只说「Following the definition in (Wang et al., 2022)」，官方仓库里**没有生成脚本**。
本仓库从 `wangsiwei2010/CVPR22-IMVC-CBG` 的 `randomlyGeneratePartialData.m`
找到了该协议，并用官方发布的 9 个 ProteinFold 文件**逐项验证**：

* 文件名 `_Per p` 里的 `p` 是**完整（paired）样本比例**，不是缺失率；
  缺失率 = `1 − p`。所以 `_Per0.1` 是最难的（90% 样本有缺失），`_Per0.9` 是最容易的。
* 被判定为「有缺失」的样本，**每个视图独立地以 1/2 概率保留**，
  且不允许「所有视图都保留」或「所有视图都丢失」。
* 9 个文件的掩码是**嵌套**的（`Per0.1 ⊆ Per0.2 ⊆ … ⊆ Per0.9`）。

反推结果与官方文件实测值吻合（每视图缺失率、受损样本占比、每样本存活视图数的
`Binomial(V, 1/2)` 分布都对得上），详见 `efimvc/data/protocol.py` 的文档字符串
与 [`NOTES.md`](NOTES.md)。

## 结果

聚合照搬 `official_code/run.m`：每个缺失率下各指标在超参网格取最优，再对 9 个缺失率取平均。

| 数据集 | ACC | NMI | Purity | Fscore | 论文 ACC |
|---|---|---|---|---|---|
| **ProteinFold** | **31.79** | **40.71** | **38.25** | **18.56** | 31.71 ✓ |
| **WebKB** | **92.25** | **54.87** | **92.25** | **89.19** | 90.64 ✓ |
| 100Leaves ᵃ | **71.11** | **82.94** | **72.63** | **57.78** | 72.88 ✓ |
| ORL_mtv ᵃ | 81.40 | 90.33 | 83.80 | 74.31 | —（论文未用） |

ᵃ 缩减网格，数值被低估。

**ProteinFold 四个指标全在论文 ±0.4 分内；消融四行（full / lambda=0 / beta=0 / P=I）也全部对上。**
缺失率曲线整体递减，超参敏感性的两个结论也与论文一致。

> ⚠️ **复现过程中踩过的最大的坑**：锚点对齐的 Procrustes 步里
> `np.linalg.svd` 返回的 `Vh` 已经是 `V'`，我按 MATLAB 的写法多转置了一次
> （`Um @ Vm.T` 应为 `Um @ Vm`）。算出的 `P` 仍是正交矩阵、能通过正交性检查，
> 但方向错了，对齐完全无效，ProteinFold 只有 22.4 分。
> 后来装 Octave 直接跑官方 `.m` 代码逐步骤对拍才定位到。
> 详见 [`results/REPORT.md`](results/REPORT.md) §2–§3，
> 回归测试见 `scripts/verify_alignment.py`。

**全部实验结果与结论汇总在 [`results/REPORT.md`](results/REPORT.md)（单文件）。**

## 自检

```bash
python scripts/verify_metrics.py     # 指标 vs sklearn / 暴力枚举，偏差 ≤ 8e-16
python scripts/verify_qp.py          # QP 的 KKT 残差 ≤ 2e-12，三种核互相对拍
python scripts/verify_protocol.py    # 反推的缺失协议 vs 官方 9 个 ProteinFold 文件
```
