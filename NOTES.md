# EFIMVC 复现笔记

> Liu et al., *Efficient Federated Incomplete Multi-View Clustering*, ICML 2025.
> 官方代码 <https://github.com/Tracesource/EFIMVC>　·　更新：2026-10-10

**一句话结论**：算法跑通了，缺失数据生成协议被**逐项反推并验证**，
QP 求解器与指标都通过了独立校验；但**官网只发布了 ProteinFold 一个数据集**，
其余 6 个中只有 3 个能拿到；而在能拿到的数据上，
**复现出的 ACC 与论文表 2 有明显差距**，且差距方向与「数据本身的可分性上界」一致。

---

## 1. 官方代码是什么样的

| 项目 | 情况 |
|---|---|
| 语言 | **MATLAB**（`run.m` + 4 个主函数 + `functions/`、`measure/`） |
| 数据集 | **只有 ProteinFold**，9 个缺失率文件 `proteinFold_Per0.1..0.9.mat` |
| 生成脚本 | **没有** |
| README | 只有一句「Please run the 'run.m', and reproduce the results on 'ProteinFold' datasets.」 |

也就是说：论文表 2 的 7 个数据集里，官方只放出了 1 个；
论文说「Following the definition in (Wang et al., 2022)」的缺失数据定义，
在官方仓库里找不到实现。

### 主流程（`run.m`）

```
for 缺失率 in {Per0.1 … Per0.9}:          # 注意：9 档
  for m in {k, 2k, 3k}:                   # TempLambda1
    for alpha in {0.001,0.1,1,100}:       # TempLambda2
      Z{v} = client_pretrain(X{v}, m, alpha, ind(:,v))     # 12 个视图
      Zall = server_pretrain(Z, ind, m)
      for beta in {0.001,0.1,1,100}:      # TempLambda3
        P{v} = I
        重复最多 5 次（maxIter=5）:
            parfor v: [Z{v},S{v},obj_client(v)] = client_train(...,Zall,P{v})
            [Zall,P,obj_server] = server_train(...)
        对 Zall' 做 SVD 取左奇异向量 → 行归一化 → kmeans 重复 20 次 → 取均值
```

三个容易漏的点：

1. **外层 `maxIter = 5`** —— 论文正文没写，`run.m` 里是 5。
2. **β 循环里 `Z` 不重置**。`Z{v}` 在 α 循环里初始化一次，β 循环复用它，
   也就是说「β 的第 2 个取值是从第 1 个取值的解继续跑的」。
   这是一个状态泄漏，会让结果依赖 β 的遍历顺序。
   本复现**没有复刻**这一行为（每个 `(m, α, β)` 独立从初始化开始），
   因为「在网格上取最优」在顺序依赖下没有良好定义。
3. **`obj = Σ_v obj_client + obj_server` 并不是论文 Eq. (25) 的 $J_o$。**
   $J_o$ 里数据拟合项带 $1/\lambda$ 权重、服务器项带 $\beta$ 权重，
   而 `run.m` 直接相加。所以 `run.m` 的收敛判据作用在一个**非论文目标**上。

---

## 2. 缺失数据生成协议：反推 + 验证

论文只引用 (Wang et al., 2022)。该文（CVPR 2022, IMVC-CBG）的
`Incomplete/randomlyGeneratePartialData.m` 里能找到协议，关键几行：

```matlab
perGrid = [1:-0.1:0];                     % 「paired instance ratio」
misingExampleVector = randperm(N);
for id = 1:N                              % 一次性给每个样本抽一个「受损模式」
    missingViewVector = randi([0,1], n_view,1,'int8');
    while(0==sum(missingViewVector) || n_view==sum(missingViewVector))
        missingViewVector = randi([0,1], n_view,1,'int8');   % 不能全留、不能全丢
    end
    MissingStatus(id,:) = missingViewVector;
end
for per_iter = 1:length(perGrid)
    per = perGrid(per_iter);
    miss_n = fix((1-per)*N);              % 受损样本数
    ...  data{j}(:, perm(id)) = nan  (MissingStatus==0 的视图)
    save([name,'_Per',num2str(per),'.mat'], 'data','truelabel','index','MissingStatus')
end
```

于是：

* **文件名里的 `Per p` 是「完整（paired）样本的比例」，不是缺失率。**
  缺失率 = `1 − p`。`_Per0.1` 是**最难**的（90% 样本有缺失），`_Per0.9` 最简单。
* 受损样本**每个视图独立以 1/2 概率保留**，且不允许全留/全丢。
* `randperm` 在循环外抽一次 ⇒ 9 个文件的掩码**嵌套**。

### 用官方发布的 9 个 ProteinFold 文件验证

| 检查项 | 预测（由协议推出） | 实测 |
|---|---|---|
| 有缺失的样本占比 | `1 − p` | Per0.1: 0.899 / Per0.5: 0.500 / Per0.9: 0.099 |
| 每视图缺失率 | `(1−p)/2` | Per0.1: 0.441 / Per0.5: 0.249 / Per0.9: 0.051 |
| 掩码嵌套 | `Per0.1 ⊆ … ⊆ Per0.9` | ✅ 逐元素成立 |
| 每样本存活视图数（受损样本） | `Binomial(V, 1/2)` | Per0.9 的 69 个受损样本，卡方 p ≈ 0.17 |
| 没有样本在所有视图都缺失 | 成立 | ✅ `min(rowsum)=1` |

结论：**协议完全对上了**。实现见 `efimvc/data/protocol.py::make_masks`。

---

## 3. 复现中的两个性能坑（都会让程序慢 100 倍以上）

### 3.1 BLAS 线程超订

EFIMVC 每一步都是 `m×m`（`m ≤ 300`）的小矩阵运算。放开线程后，
OpenBLAS/MKL 会为**每一次** `svd`/`solve` 调用开满 24 个线程：

```
server_train(max_iter=30)  放开线程: > 180 s      限 1 线程: 0.76 s
```

`efimvc/__init__.py` 因此在导入 numpy 前就把
`OMP/OPENBLAS/MKL/NUMEXPR/NUMBA_NUM_THREADS` 置 1；
并行度通过 `run_*.py --jobs N` 在**进程级**获取。

### 3.2 QP 求解器

算法每一步都归结为同一批单纯形约束 QP

```
min_z ½ z'Hz − c'z   s.t.  1'z = 1, z ≥ 0
```

`H` 对一批列相同，只有 `c` 变。官方用 MATLAB `quadprog(...,'interior-point-convex')`。

* 官方 **Python** 包 `quadprog` 在这个约束形式下直接报
  `constraints are inconsistent, no solution`（连 `H=I, c=0` 的平凡例子都失败），
  所以没有采用。
* 自己实现了一个 **active-set** 求解器（`efimvc/qp.py`），KKT 残差 ≤ 2e-12。
  版本演进：

  | 版本 | 思路 | m=27 | m=81 |
  |---|---|---|---|
  | `numpy` | 纯 Python，Schur 补 | 196 µs | 4341 µs |
  | `schur` | 同上，numba JIT | 196 µs | ~2300 µs |
  | **`fast`** | **对小的自由集维护 $H_{FF}^{-1}$，逐个**降阶**（O(nF²)）** | **11 µs** | **231 µs** |
  | `fast` + 热启动 | 用上一次内层迭代的解当初始活跃集 | ~0 µs | **2 µs** |

几个值得记的教训：

* **活跃集在「自由集」一侧是小集合、在「活跃集」一侧是大集合。**
  最初的实现用 $M_{AA}^{-1}$（`A` 是补集）去算 $H_{FF}^{-1}$，
  等于反复求逆一个大矩阵 —— 这是主要的慢因。
* 热启动**不改变解**（对拍最大差 1e-13），因为解法里保留了
  「对偶可行性修复」步骤，错误的初值会被自动纠正。
* `numba` 的 `prange` 版本在本机（numba 0.61 / Windows）会**算出错误的活跃集**，
  串行版本则正确。现在核内串行，并行放在进程级。
* `fast` 版本曾经因为**实参顺序写反**（`H` 与 `Minv` 互换）而静默给出错解，
  而且只在批量路径上出错、单列调用正确。现在 `verify_qp.py` 会把
  `fast` / `schur` / `numpy` 三个核互相对拍。

---

## 4. 指标校验

`Clustering8Measure.m` 返回 `[ACC nmi Purity Fscore Precision Recall AR Entropy]`。
`efimvc/metrics.py` 全部对齐，并用独立实现交叉验证（`scripts/verify_metrics.py`）：

| 指标 | 参照 | 最大偏差 |
|---|---|---|
| ACC | 暴力枚举所有标签置换 | 0.0 |
| NMI | `sklearn.normalized_mutual_info_score(average_method='arithmetic')` | 8.5e-16 |
| Purity | 直接按定义重写 | 0.0 |
| Fscore | O(n²) 布尔循环重算成对 F1 | 0.0 |
| AR | O(n²) 重算 Rand Index | 0.0 |

> 注意 MATLAB 的 `Fscore` 是**成对 F1**（precision/recall 的调和平均），
> 不是 Fowlkes–Mallows 的几何平均。

---

## 5. 与官方 MATLAB 的已知差异

| # | 位置 | 官方 | 本实现 | 影响 |
|---|---|---|---|---|
| 1 | 初始化锚点 | `litekmeans`（twister(12) 流） | `sklearn.KMeans(n_init=10, random_state=12)` | 锚点不同 → 局部最优不同（**不可消除**） |
| 2 | QP | `quadprog` IPM | 自研 active-set（KKT ≤ 2e-12） | 同一唯一解，无影响 |
| 3 | 读出 k-means | `emptyaction`,`singleton` | `sklearn.KMeans` | 空簇处理略不同 |
| 4 | β 循环 | `Z` 跨 β 复用 | 每次独立初始化 | 见 §1 第 2 点 |
| 5 | 式 (3) 的 σ | `0.3*((m-1)*0.5-1)+0.8`（论文未给） | 同官方 | 归一化后 S≈全 1 矩阵，结构项退化 |
| 6 | 式 (3) 的距离 | 代码用**欧氏距离**（`eucl_dist2` 默认开根号） | 同官方 | 与论文写的「距离平方」不一致 |

---

## 6. 结果

**全部结果与结论汇总在 [`results/REPORT.md`](results/REPORT.md)（单文件）。** 这里只放结论。

### 6.1 主结果（%，9 个缺失率平均）

| 数据集 | ACC | NMI | Purity | Fscore | 论文 ACC |
|---|---|---|---|---|---|
| **ProteinFold** | **31.79** | **40.71** | **38.25** | **18.56** | 31.71 |
| **WebKB** | **92.25** | **54.87** | **92.25** | **89.19** | 90.64 |
| 100Leaves ᵃ | 71.11 | 82.94 | 72.63 | 57.78 | 72.88 |
| ORL_mtv ᵃ | 81.40 | 90.33 | 83.80 | 74.31 | — |

ᵃ 缩减网格。**ProteinFold 四个指标全在 ±0.4 分内，复现成功。**

### 6.2 踩过的最大的坑：Procrustes 步多转置了一次

```python
Um, _, Vm = np.linalg.svd(M, full_matrices=False)
Pv = Um @ Vm.T     # 错！numpy 的 Vm 已经是 V^T，应为 Um @ Vm
```

MATLAB 的 `[Um,~,Vm]=svd(M)` 满足 `M = Um*S*Vm'`，所以 `Um*Vm'` 是 `U V'`；
numpy 返回的 `Vh` 本身就是 `V'`。多转置一次得到 `U V` —— **仍是正交矩阵**
（所以 `|P P' − I| ≈ 1e-15`、`‖P−I‖ ≈ 12.8` 这些检查全过），
但它是**错的那个旋转**，`tr(P'M)` 不取最大值，对齐完全没发生。

症状：ProteinFold ACC 只有 22.43（论文 31.71），目标函数比官方大 3 倍且不单调。

### 6.3 怎么定位的

**装了 Octave 9.4**（清华镜像的 Windows 7z 版，自带 `optim`/`statistics` 包），
**直接运行官方 `.m` 代码**（`scripts/octave/check_proteinfold.m`），然后逐层对拍：

| 对拍层次 | 结果 |
|---|---|
| 端到端一个格子 | 官方 ACC 24.03 vs 我 15.90 → 确认是我的问题 |
| pretrain 后的量（拟合误差、Z 范数） | 一致（差 <1%）→ 初始化没问题 |
| `client_train` 逐内层迭代 | 官方 89.26/68.13/52.41… vs 我 88.70/68.54/53.29… → 客户端没问题 |
| **`server_train`（同一份输入）** | 初始 part1 都是 55.96 ✓；**P 更新后官方 54.28、我 468.67 ✗** → 锁定 P 更新 |

修复后再对拍：`server_train` 与官方**小数点后 6 位完全一致**。

回归测试：`scripts/verify_alignment.py`（检查 Procrustes 值 = 矩阵奇异值之和）。

### 6.4 其余结论

* **消融**：ProteinFold 四行（full 31.61 / lambda=0 31.30 / beta=0 30.93 / P=I 20.86）
  与论文（31.71 / 30.85 / 30.80 / 20.18）**全部对上**，且趋势一致
  —— 去掉锚点对齐 P 会崩，去掉 λ 或 β 只掉一点点。
* **WebKB 的 `beta=0` 不掉点**：官方 `sigma = 0.3((m−1)/2−1)+0.8` 在归一化数据上
  让 `S = exp(−d/(2σ²))` 几乎是全 1 矩阵（m=6 时 S ≥ 0.53，m=81 时 ≥ 0.994），
  结构项本身退化；而且网格选出的最优 β 本来就是 0.001，设成 0 自然没差。
* **收敛性**：96–98% 的运行目标函数单调下降（修复前只有约 40%），与论文图 3 一致。
* **超参敏感性**：ProteinFold α=0.001 最优、WebKB α=0.1 最优 —— 都与论文一致
  （修复前 WebKB 对不上，这也侧面印证了 bug 是根因）。
* **λ=0 消融的数值处理**：`H = A'A` 在 `m > d`（ProteinFold m=81, d=27）时秩只有 27，
  是退化问题、解不唯一。原始用 `1e-10` 相对抖动时 active-set 会数值打转
  （单次 `client_train` 要 20+ 分钟）；改成 `1e-6` 后降到 1.7 秒。
  可用 `EFIMVC_QP_JITTER` 覆盖。只影响 α=0 这一路。

### 6.5 复现环境的一个坑

本机只有 **16.9 GB 物理内存**，并行实验时一度占用 97%、交换区 19 GB，
同一份计算的耗时波动 2–3 倍。`seconds` 字段只适合做量级参考。

## 7. 数据可获得性

`github.com` 的 HTML 页在本机不可达，但 `raw.githubusercontent.com` 可达，
`ghfast.top` 代理也可用。逐个查证的结果：

| 数据集 | 结果 |
|---|---|
| ProteinFold | ✅ 官方仓库自带 9 档 |
| WebKB | ✅ `ChuanbinZhang/Multi-view-datasets`（维度 1840/3000，与论文 334/2949 不同源） |
| 100Leaves | ✅ 同上（维度 64/64/64，**与论文完全一致**） |
| ORL_mtv | ✅ `Tracesource/FIMVC-VIA`（同作者前作，官方 9 档缺失文件，同一生成协议） |
| CCV | ❌ `ShizheHu/TNNLS25_Code_PGCL-DCL` 的 zip 里只有配置文件，没有 `.mat`；其余源在 Google Drive |
| YTF10 | ❌ 只在百度网盘（需登录） |
| CIFAR10 / MNIST | ❌ 同上；MNIST 的 64/342/1024 版本在文献里叫 `MNIST-fea`，没有公开数据链接 |

---

## 8. 结构与复现命令

```bash
python scripts/fetch_data.py
python scripts/verify_metrics.py && python scripts/verify_qp.py
python scripts/run_protocol.py    --dataset proteinFold --jobs 16
python scripts/run_ablation.py    --dataset proteinFold --jobs 16
python scripts/run_sensitivity.py --dataset proteinFold --plot
python scripts/baseline_upper_bound.py --dataset proteinFold
python scripts/make_experiments_report.py && python scripts/make_figures.py
```

## 9. 待办

- [ ] CCV / YTF10 / CIFAR10 / MNIST 数据未获取（需要百度网盘或 Google Drive）
- [ ] 七个 baseline（FIMVC / SCBGL / DSVAI / DAQINT / FMVC / HFMVC / FMCSC / FedMVFPC）
      未实现，按要求直接引用论文数值
- [ ] 未能在 MATLAB 下跑官方代码做逐位对拍（本机无 MATLAB/Octave）
