# UCR 数据使用说明

本项目默认通过 `aeon` 加载 UCR Time Series Archive 数据集。首次运行时，数据会缓存到：

```text
datasets/aeon/
```

通常不需要手动下载 UCR 压缩包。只有在 aeon 不可用，或需要读取本地 `TRAIN/TEST` 文件时，才使用 `--data-source files`。

## 最终 18 个数据集

|    # | Dataset             | Length |    k |    n | 建议                                        |
| ---: | ------------------- | -----: | ---: | ---: | ------------------------------------------- |
|    1 | Chinatown           |     24 |    2 |  365 | 全量；**首选 smoke test**                   |
|    2 | SyntheticControl    |     60 |    6 |  600 | 全量；备用 smoke test                       |
|    3 | MoteStrain          |     84 |    2 | 1272 | 全量（length 短，1272² 不慢）               |
|    4 | ECG200              |     96 |    2 |  200 | 全量                                        |
|    5 | CBF                 |    128 |    3 |  930 | 全量；**Part 2 perturbation 主力**          |
|    6 | TwoPatterns         |    128 |    4 | 5000 | **必须 stratified sampling 至每类 200-300** |
|    7 | ECGFiveDays         |    136 |    2 |  884 | 全量                                        |
|    8 | Plane               |    144 |    7 |  210 | 全量                                        |
|    9 | GunPoint            |    150 |    2 |  200 | 全量                                        |
|   10 | Wine                |    234 |    2 |  111 | 全量                                        |
|   11 | ArrowHead           |    251 |    3 |  211 | 全量                                        |
|   12 | Trace               |    275 |    4 |  200 | 全量；**Part 2 perturbation 主力**          |
|   13 | Coffee              |    286 |    2 |   56 | 全量                                        |
|   14 | DiatomSizeReduction |    345 |    4 |  322 | 全量                                        |
|   15 | Symbols             |    398 |    6 | 1020 | **采样至每类 120**，或用 LB_Keogh 加速      |
|   16 | OSULeaf             |    427 |    6 |  442 | 全量可行（过夜跑）                          |
|   17 | Computers           |    720 |    2 |  500 | **采样至每类 150**，或只用 train 集         |
|   18 | ACSF1               |   1460 |   10 |  200 | 全量；记录 runtime（IDK 注意内存）          |

## 默认 aeon 模式

运行 ECG200 smoke test：

```bash
python scripts/chen_part1_benchmark.py --datasets ECG200 --metrics ed dtw msm --samples-per-class 5 --seeds 1 --metric-backend reference --output scratch/results/ecg200_smoke.csv
```

运行最终清单中的所有数据集：

```bash
python scripts/chen_part1_benchmark.py --samples-per-class 0 --seeds 1 2 3 4 5 6 7 8 9 10 --metric-backend aeon
```

## 本地文件模式

如果需要读取本地 UCR `TRAIN/TEST` 文件，可以使用：

```bash
python scripts/chen_part1_benchmark.py --data-source files --data-root datasets --datasets ECG200
```

本地文件模式的目标是兼容手动下载的数据，不是当前推荐主流程。

## 注意事项

- `k` 固定为 ground-truth 类别数，用于保证 benchmark 可比性。
- `y` 只用于计算 ARI/NMI，不参与聚类训练。
- 小规模采样只用于 smoke test；正式结论应优先来自全量实验。
- 如果必须采样，应使用 stratified sampling（每类等量抽取），train+test 合并后再采样，固定 random seed（如 42），并在报告中写明原因、样本数和种子。

## 运行优先级

**第一批（验证 pipeline，约 1 小时全部跑完）**：Chinatown, SyntheticControl, ECG200, CBF, GunPoint, Wine, ArrowHead, Coffee——全部 length < 300 且规模小。

**第二批（扩展）**：MoteStrain, ECGFiveDays, Plane, Trace, DiatomSizeReduction, OSULeaf。

**第三批（重量级，并行/过夜跑）**：TwoPatterns(采样), Symbols(采样), Computers(采样), ACSF1。

## Part 2 perturbation 推荐数据集

CBF（必选，shape 极清晰）、Trace（必选，合成可控）、ECG200（推荐，真实数据代表）——全量、规模小、跑得快，适合大量扰动 × 多 seed 的密集实验。
