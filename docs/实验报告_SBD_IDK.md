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

本次汇总覆盖 18 个数据集。聚合结果如下：

- SBD：ARI mean = 0.3707，NMI mean = 0.4134，runtime mean = 3.09 s
- IDK：ARI mean = 0.2183，NMI mean = 0.2491，runtime mean = 53.09 s

可以看到，整体上 SBD 的聚类质量和运行效率都更稳定，IDK 虽然在部分数据集上更好，但平均 runtime 明显更高。

### 3.2 对比图

![SBD 与 IDK 在各数据集上的 mean ± std 对比图](../results/auto_ucr/collaboration_results_sbd_idk_all_metrics.png)

### 3.3 典型现象

- IDK 在 ARI 和 NMI 上同时优于 SBD 的数据集包括：ACSF1、Plane、GunPoint、TwoPatterns、Wine。
- SBD 在 ARI 和 NMI 上同时优于 IDK 的数据集更多，例如：ECGFiveDays、SyntheticControl、Trace、MoteStrain、CBF、DiatomSizeReduction、Coffee、Symbols、OSULeaf 等。
- runtime 上，SBD 几乎在所有数据集上都更快，尤其在 TwoPatterns、CBF、ECGFiveDays、Symbols 上差距非常明显。

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

综合来看，SBD 在本次实验中更适合作为默认基线：它在 18 个数据集上整体更稳、速度更快、平均聚类质量也更高。IDK 的优势主要体现在少数具有明显局部结构的数据集上，因此更像是一种“针对部分任务有效”的方法，而不是在所有 UCR 数据集上都优于 SBD 的通用替代方案。

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
