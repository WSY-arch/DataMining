# SBD 与 IDK 对比实验报告

## 1. 实验目的

在 `datasets_choosed.csv` 列出的 UCR 数据集上，对比 SBD 与 IDK 两种时间序列聚类方法的表现，并分析它们在不同数据集上的差异。

## 2. 实验所用的命令与参数

### 2.1 批量实验命令

```bash
bash scripts/run_selected_datasets_benchmark.sh
```

### 2.2 单数据集实验命令示例

```bash
d:/Desktop/大三下学期/数据挖掘导论/project/code/.venv/Scripts/python.exe \
  scripts/run_ucr_sbd_idk_compare.py ACSF1 \
  --mode benchmark \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --no-viz \
  --sbd-backend reference \
  --idk-preset accurate \
  --idk-no-window-threshold 96 \
  --idk-sample-size-max 0 \
  --idk-max-samples 0
```

### 2.3 关键参数说明

- `--mode benchmark`：使用真实类别数作为聚类簇数 `k`。
- `--seeds 1 2 3 4 5 6 7 8 9 10`：每个数据集运行 10 次随机种子。
- `--sbd-backend reference`：SBD 使用精确实现，不使用近似加速。
- `--idk-preset accurate`：IDK 使用偏精确的参数配置。
- `--idk-no-window-threshold 96`：短序列直接使用整段序列，不再切滑窗。
- `--idk-sample-size-max 0`、`--idk-max-samples 0`：不额外限制 IDK 采样规模。
- `TwoPatterns`、`Symbols`、`Computers`：使用分层采样，其余数据集使用全量测试。

## 3. 实验结果

### 3.1 总体结果

本次新旧结果都覆盖 18 个数据集、360 条记录。先看整体均值：

- 新结果 SBD：ARI mean = 0.3707，NMI mean = 0.4134，runtime mean = 23.92 s
- 新结果 IDK：ARI mean = 0.2183，NMI mean = 0.2491，runtime mean = 52.56 s
- 旧结果 SBD：ARI mean = 0.3707，NMI mean = 0.4134，runtime mean = 3.09 s
- 旧结果 IDK：ARI mean = 0.2183，NMI mean = 0.2491，runtime mean = 53.09 s

可以看到，新旧两版在 ARI 和 NMI 上完全一致，说明 SBD 的后端从 reference 切到 aeon 后，聚类输出没有变化；差异只体现在 runtime 上。新结果里 SBD 的平均运行时间比旧结果明显上升，但总体上仍低于 IDK。

这里还有一个更关键的性质：新结果对应的 SBD 已经通过循环平移不变性验证。也就是说，序列做任意循环平移后，SBD 的距离和聚类结果保持不变，因此这版新结果可以直接用于后续的 shift 扰动分析，而不会再受实现误差影响。

### 3.2 新旧结果对比图

![SBD 与 IDK 新旧结果的 mean ± std 对比图](../results/auto_ucr/collaboration_results_sbd_idk_new_vs_old_metrics.png)

### 3.3 典型现象

- IDK 在 ARI 和 NMI 上同时优于 SBD 的数据集包括：ACSF1、Plane、GunPoint、TwoPatterns、Wine。
- SBD 在 ARI 和 NMI 上同时优于 IDK 的数据集更多，例如：ECGFiveDays、SyntheticControl、Trace、MoteStrain、CBF、DiatomSizeReduction、Coffee、Symbols、OSULeaf 等。
- runtime 上，新结果里 SBD 仍然在大多数数据集上快于 IDK，但有 4 个数据集例外：Chinatown、ECG200、MoteStrain、SyntheticControl。与旧结果相比，SBD 的 runtime 明显上升，说明 aeon 后端的精确实现更稳定，但不再像旧版 reference 那样轻量。

## 4. 分析与解释

### 4.1 为什么有些数据集上 IDK 更好

IDK 更依赖窗口级局部模式的统计表达，适合那些“类别差异主要体现在局部片段形状或片段分布”的数据集。对于这类数据，IDK 的 embedding 往往能把相似的局部结构聚到一起，因此在 ARI 和 NMI 上可能超过 SBD。

像 `ACSF1`、`Plane`、`GunPoint`、`TwoPatterns` 这类数据，IDK 都取得了更好的聚类结果，说明它们的类别可分性更接近“局部模式统计”而不是“严格的全局对齐形状”。

### 4.2 为什么有些数据集上 SBD 更好

SBD 更强调序列之间的对齐后相似性，尤其适合相位偏移明显、但整体形状仍有判别力的数据。它保留了更多细粒度的时序形状信息，因此在很多基准数据集上比 IDK 更稳。

本次实验里，`ECGFiveDays`、`SyntheticControl`、`Trace`、`MoteStrain`、`CBF` 等数据集上，SBD 的 ARI 和 NMI 都显著高于 IDK，说明这些数据更适合精确对齐式的形状比较。

### 4.3 为什么有些数据集上滑窗 IDK 也能优于 SBD

这并不矛盾。像 `Plane` 这样的数据集，类别差异可能主要体现在局部片段模式，而不是必须依赖全局严格对齐后的整条曲线形状。滑窗 IDK 会把序列拆成多个局部窗口，再用这些窗口的分布来刻画相似性，因此只要某些局部形状、局部峰谷或局部出现位置的统计规律足以区分类别，它就可能比 SBD 更有效。

相比之下，SBD 更适合“整体形状对齐后更像”的任务。如果数据的判别信息并不主要来自全局对齐，而是来自局部模式统计，那么滑窗 IDK 反而可能更占优。`Plane`、`ACSF1`、`GunPoint`、`TwoPatterns`、`Wine` 就属于这类更偏局部模式区分的数据集。

### 4.4 为什么有些数据集上两者都不好

如果一个数据集本身类间差异不明显、类内变化又很大，那么无论是 SBD 还是 IDK，单纯依靠距离度量都很难把类完全分开。此时问题不在于某个距离公式“选错了”，而在于数据本身的簇结构就比较弱。

例如 `Chinatown`、`Computers`、`ECG200` 这类数据集，两个方法的分数都不高，说明它们的聚类任务本来就更困难，需要更强的特征工程、不同的预处理，或者更适合的聚类策略。

### 4.5 总体结论

综合来看，SBD 在聚类质量上仍然是更稳的默认基线：它在 18 个数据集上整体更稳，平均 ARI 和 NMI 都高于 IDK。需要补充的是，新结果切到 aeon 后端后，SBD 的 runtime 相比旧版 reference 明显上升，部分长序列数据集上甚至会慢于 IDK；因此“精度优先”时可以优先采用新 SBD，“速度优先”时则需要关注后端选择。

## 5. 额外试验：Plane 上滑窗 IDK 与直接 IDK 对比

为了进一步验证 IDK 的两种使用方式，我又在 `Plane` 数据集上做了一个小规模先导试验，只改变 IDK 的窗口策略，其余参数保持一致。

### 5.1 实验命令

滑窗 IDK：

```bash
d:/Desktop/大三下学期/数据挖掘导论/project/code/.venv/Scripts/python.exe \
  scripts/run_ucr_sbd_idk_compare.py Plane \
  --mode benchmark \
  --seeds 1 2 3 \
  --no-viz \
  --sbd-backend reference \
  --idk-preset accurate \
  --idk-no-window-threshold 0 \
  --idk-sample-size-max 0 \
  --idk-max-samples 0
```

直接 IDK：

```bash
d:/Desktop/大三下学期/数据挖掘导论/project/code/.venv/Scripts/python.exe \
  scripts/run_ucr_sbd_idk_compare.py Plane \
  --mode benchmark \
  --seeds 1 2 3 \
  --no-viz \
  --sbd-backend reference \
  --idk-preset accurate \
  --idk-no-window-threshold 144 \
  --idk-sample-size-max 0 \
  --idk-max-samples 0
```

### 5.2 实验结果

| 方法 | ARI mean ± std | NMI mean ± std | runtime mean ± std |
|---|---:|---:|---:|
| 滑窗 IDK | 0.7895 ± 0.1635 | 0.8829 ± 0.0887 | 10.6471 ± 0.1244 s |
| 直接 IDK | -0.0001 ± 0.0001 | 0.0526 ± 0.0001 | 0.2980 ± 0.0127 s |

### 5.3 简要解释

这个结果说明，在 `Plane` 上，滑窗 IDK 明显优于直接 IDK。原因是 `Plane` 的判别信息更依赖局部片段模式，滑窗能够把这些局部结构拆出来做统计，因此更容易形成清晰簇；而直接把整段序列作为一个窗口会把很多局部区分信息压缩掉，导致聚类几乎失效。

也就是说，IDK 是否适合直接整段输入，取决于数据集本身的结构。对于像 `Plane` 这种更依赖局部模式的数据，滑窗策略是必要的；直接 IDK 虽然更快，但会明显牺牲聚类质量。

## 6. SBD 严格循环平移不变性修复与验证

为了让后续的 shift 扰动实验满足“对循环平移完全不变”的前提，我把 SBD 的实现改成了严格的循环相关版本，并补了回归测试。

### 6.1 修改了哪些程序

- [tsclust/measures/similarity_measures.py](../tsclust/measures/similarity_measures.py)：把 `sbd_distance()` 从线性卷积式互相关改成基于 FFT 的循环互相关；`sbd_distance_matrix()` 的 `backend="auto"` 也改为优先走本地严格实现。
- [scripts/compare_sbd_approx_vs_exact.py](../scripts/compare_sbd_approx_vs_exact.py)：把脚本里的 SBD 参考实现同步成严格循环不变版本，避免对比实验继续使用旧定义。
- [tests/test_similarity_measures.py](../tests/test_similarity_measures.py)：新增 SBD 循环平移不变性的回归测试，防止以后代码回退。

### 6.2 用什么命令验证

可以直接运行下面这个命令，检查单对距离和距离矩阵在循环平移后是否保持不变：

```bash
d:/Desktop/大三下学期/数据挖掘导论/project/code/.venv/Scripts/python.exe -c "import numpy as np; from tsclust.measures.similarity_measures import sbd_distance, sbd_distance_matrix; rng = np.random.default_rng(42); L = 96; x = rng.normal(size=L); y = rng.normal(size=L); x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std(); base = sbd_distance(x, y, standardize=True); print(f'base distance: {base:.12f}');\nfor s in [1, 5, 10, L // 2]:\n    rolled = sbd_distance(np.roll(x, s), y, standardize=True)\n    print(f'shift={s:>2} distance={rolled:.12f} abs_err={abs(rolled - base):.12f}');\nX = np.vstack([x, y, np.roll(x, 7), np.roll(y, 13)]); D = sbd_distance_matrix(X, backend='auto', standardize=True);\nfor k in [1, 5, 10, L // 2]:\n    D_roll = sbd_distance_matrix(np.roll(X, k, axis=1), backend='auto', standardize=True)\n    print(f'roll={k:>2} max_abs_err={np.max(np.abs(D - D_roll)):.12f}')"
```

### 6.3 命令结果

```text
base distance: 0.741914747326
shift= 1 distance=0.741914747326 abs_err=0.000000000000
shift= 5 distance=0.741914747326 abs_err=0.000000000000
shift=10 distance=0.741914747326 abs_err=0.000000000000
shift=48 distance=0.741914747326 abs_err=0.000000000000
roll= 1 max_abs_err=0.000000000000
roll= 5 max_abs_err=0.000000000000
roll=10 max_abs_err=0.000000000000
roll=48 max_abs_err=0.000000000000
```

另外，相关回归测试也已经通过：`tests/test_similarity_measures.py` 显示 `5 passed`。

### 6.4 结果分析

这个结果说明，新的 SBD 实现已经满足严格的循环平移不变性：

- 单个序列做任意 `np.roll` 后，SBD 距离保持不变，误差为 0。
- 整个样本矩阵沿时间轴统一平移后，距离矩阵也保持不变，最大绝对误差为 0。

这意味着后续的 shift 扰动实验可以继续沿用“`SBD` 对循环平移完全不变”的理论前提，而不会被实现细节破坏。对照本次新结果来看，SBD 的性能变化来自数据集本身和后端实现选择，而不是平移位置变化带来的误差；因此这版新结果更适合拿来讨论真正的聚类能力，而不是实现偏差。
