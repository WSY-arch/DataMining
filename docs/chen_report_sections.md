# Chen 报告草稿部分

## Introduction draft

时间序列聚类在很大程度上取决于如何定义序列之间的相似性。不同相似性度量编码了不同的 alignment、temporal variation 和 signal structure 假设。Euclidean distance 这类 lock-step measure 会逐时间点直接比较；DTW 和 MSM 这类 elastic measures 允许局部 temporal warping；SBD 这类 sliding measure 强调 global phase shift 下的 shape similarity；IDK 这类 distributional kernel 则采用不同视角，通过 distributions 比较时间序列，而不是显式进行 point-to-point alignment。

已有 benchmark studies 比较了许多 time-series distances 和 clustering methods，但它们通常关注 archive datasets 上的平均表现。这留下了一个机制层面的问题：noise、temporal misalignment 和 sequence length 如何改变不同 similarity paradigms 的相对表现？这个问题尤其重要，因为一种 measure 可能在干净、对齐的数据上表现很好，但在 shift 或 noise 下失效；而更具 invariance 的 measure 又可能丢失某些类别所需的 temporal information。

本项目通过统一的 time-series clustering framework 比较 ED、DTW、MSM、SBD 和 IDK，从而回应这一 gap。我们结合 selected UCR datasets 上的 real-data benchmark 和 controlled perturbation experiments。我们的目标不是寻找单一 universal winner，而是解释每种 similarity paradigm 在什么条件下更合适。

## Related Work structure

### Benchmark studies of time-series distances

讨论 Paparrizos et al. 和 d'Hondt et al. 等 broad distance benchmarks，将它们作为 cross-paradigm comparison 的基础。强调它们的 taxonomy、normalization findings 和 statistical rigor，但也指出它们并不是以机制导向 clustering 为主。

### Time-series clustering benchmarks

讨论 ED/DTW/SBD clustering comparisons 和 comprehensive clustering benchmarks。将它们定位为本项目的直接 baseline，同时指出这些工作缺少 IDK，也缺少受控 noise/shift/length sweeps。

### Elastic distances for clustering

使用 Holder and Bagnall 说明为什么除了 DTW 之外还要包含 MSM。强调 clustering performance 取决于 distance geometry 与 clustering objective 的匹配；这也支持本项目采用 k-medoids 的设计。

### Distributional and IDK-based approaches

介绍时间序列的 distributional treatments 和 IDK-related clustering work。核心 framing 是：IDK 代表一种 distributional paradigm，而不仅仅是 benchmark 中额外加入的一个 distance。

## Methodology draft

我们使用统一的 k-medoids pipeline 评估单变量 whole time-series clustering。每个数据集表示为 shape 为 `(n_samples, series_length)` 的 `X`；聚类数设置为 ground-truth 类别数，仅用于公平评价。Labels 不参与 clustering。所有序列在计算相似性前都会进行 z-normalization。

Real-data benchmark 使用 selected UCR datasets，覆盖短到长的序列、binary 到 multi-class 的设置，以及多个 domains。对于每个 dataset 和 similarity measure，我们计算 pairwise distance matrix，使用固定 random seed 运行 k-medoids，并报告 ARI、NMI 和 runtime。结果按 dataset 汇总，并通过 average ranks 进行比较；global comparison 使用 Friedman test。

Perturbation study 使用 CBF、Trace 和 ECG200 等代表性数据集。我们施加三种受控 transformation：additive Gaussian noise、random global temporal shift，以及 truncate with resample 来模拟 reduced temporal resolution。对于每个 perturbation level，我们重新运行相同的 clustering pipeline，并绘制 degradation curves。这些曲线会根据每种 similarity paradigm 的 invariance assumptions 进行解释。

**Distance derivation for IDK.** Unlike traditional distance measures (ED, DTW, MSM, SBD) that directly produce dissimilarity values, IDK [Ting et al. 2022] is a positive semi-definite kernel yielding similarity scores in [0, 1]. To integrate IDK into our unified k-medoids framework, we convert the kernel into a metric distance using the kernel-induced distance:

```
d_IDK(x, y) = sqrt(K(x,x) + K(y,y) - 2·K(x,y))
```

Because IDK feature maps are L2-normalized (i.e., K(x,x) = 1), this simplifies to `sqrt(2 - 2·K(x,y))`. This formulation is mathematically equivalent to the Euclidean distance implicitly computed by CTDS [Gong et al. 2024] when applying KMeans on L2-normalized IDK feature mean maps, ensuring consistency with prior work while adapting to our medoid-based clustering pipeline.

## Method mechanism table

| Measure | Mechanism                                        | Expected perturbation behavior                            |
| ------- | ------------------------------------------------ | --------------------------------------------------------- |
| ED      | Pointwise lock-step alignment                    | 对 aligned data 快且强；在 shift 下退化。                 |
| DTW     | Local elastic warping                            | 对 local timing changes 更鲁棒；可能 overfit noise。      |
| MSM     | Edit plus warp operations                        | 通常比 DTW 更稳定；对参数敏感。                           |
| SBD     | Normalized cross-correlation with sliding shift  | 在 global phase shift 下表现强；在 local warping 下较弱。 |
| IDK     | Distributional kernel over subsequence structure | 可能随序列变长而改善；可能丢失 temporal order。           |
