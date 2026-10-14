# 已作废：此文件的分析基于有 bug 的版本

这份文件写于复现失败阶段，当时的结论是「ProteinFold 复现不出来，可能是数据/论文的问题」。

**那个结论是错的。** 根因是本实现 `server_train` 里锚点对齐的 Procrustes 步
多转置了一次（`np.linalg.svd` 返回的已经是 `V'`）：

```python
Pv = Um @ Vm.T     # 错，应为 Um @ Vm
```

修掉之后 **ProteinFold 复现成功（ACC 31.79 vs 论文 31.71）**。

**请阅读 [`REPORT.md`](REPORT.md)** —— 那里有完整的结果、bug 定位过程和结论。

保留本文件仅为记录当时的（错误）推理，避免重复走弯路。
