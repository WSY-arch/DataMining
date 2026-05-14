# Novelty Memo：机制导向的时间序列聚类相似性比较

## 一句话定位

本项目研究不同时间序列相似性范式在什么条件下有效，而不仅仅是谁的平均表现最好。我们会在受控噪声、时间错位和长度扰动下，比较 lock-step、elastic、sliding 和 distributional 相似性。

## 文献 gap

现有时间序列距离 benchmark 覆盖范围很广，但通常偏向分类任务或整体性能比较。大型研究比较了许多距离和 normalization 方法，但很少在 clustering 场景中把 noise、temporal shift 和 sequence length 作为受控实验因素单独分析。

现有 clustering benchmark 覆盖了 ED、DTW 和 SBD 等重要 baseline；elastic-distance 研究也表明，当 clustering objective 合适时，MSM/TWE 可以优于普通 DTW。虽然 IDK 已被应用于时间序列聚类（Gong et al. PAKDD 2024），但它尚未在统一的 k-medoids 框架下与 ED/DTW/MSM/SBD 进行系统性的跨范式比较，其在受控 noise/shift/length 扰动下的响应特性也未被刻画。

鲁棒性和不变性相关论文为 shift、warp 和 noise 提供了有用的扰动设计，但它们通常是 model-selection framework 或特定领域研究，而不是跨范式 clustering benchmark。Sequence length 作为独立机制尤其缺乏研究；多数论文只是把它当作计算复杂度问题，或让它在数据集之间自然变化。

## 拟贡献

我们贡献一个公平的、机制导向的代表性相似性度量比较：

- ED 作为 lock-step baseline。
- DTW 和 MSM 作为 elastic alignment measures。
- SBD 作为 sliding/shift-invariant measure。
- IDK 作为 distributional-kernel measure。

核心 novelty 来自以下组合：

- 通用的单变量 whole-series clustering。
- 统一的 k-medoids 接口和共享结果 schema。
- 真实 UCR benchmark 数据集加受控扰动实验。
- 对 noise、shift 和 length 变化下 degradation curves 的机制解释。

## 预期机制假设

| Measure | Paradigm       | Expected strength                                                              | Expected weakness                                                       |
| ------- | -------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| ED      | Lock-step      | 当序列对齐良好、shape 差异直接可见时表现强。                                   | 对 temporal shift 和 local warping 敏感。                               |
| DTW     | Elastic        | 通过 time-axis warping 处理局部 timing differences。                           | 可能 over-warp noise，且计算代价较高。                                  |
| MSM     | Elastic/edit   | edit operations 对 warping 起到 regularization 作用，因此通常比纯 DTW 更稳定。 | 需要 cost parameter，并且仍可能对强噪声敏感。                           |
| SBD     | Sliding        | 通过 cross-correlation 对 global phase shift 鲁棒。                            | 不太适合 local speed changes 或非 shape 的 distributional differences。 |
| IDK     | Distributional | 当顺序/对齐不那么重要时可能更鲁棒；长序列能更好估计 distributions。            | 可能丢失某些类别所需的 temporal-order information。                     |

## Claim discipline

项目不应宣称 IDK 普遍更好。更强且更可辩护的 claim 是：

> 不同相似性度量编码了不同 invariances；最佳选择取决于 data-generating mechanism，尤其是 noise、misalignment 和 effective sequence length。
