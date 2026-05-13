# tsclust：Isolation Kernel + K-Medoids 单变量时间序列聚类

本项目用于课程实验：在同一聚类算法（K-Medoids）下，对比不同相似度度量（IDK / ED / DTW / MSM 等）对聚类效果的影响。

## 1. 项目结构

```text
code/
├── tsclust/                        # 项目核心包
│   ├── measures/                   # 相似性 / 距离度量
│   │   ├── isolation_kernel.py     # IDK 相似度实现
│   │   └── similarity_measures.py  # ED / DTW / MSM
│   ├── clustering/                 # 聚类算法 + 统一入口
│   │   ├── k_medoids.py            # K-Medoids(PAM)
│   │   └── clustering.py           # cluster_time_series() dispatch
│   └── visualization/              # 可视化工具
│       └── visualization.py
├── tests/
│   ├── test_clustering.py          # 合成数据测试
│   ├── test_similarity_measures.py # DTW/MSM 基本性质
│   └── test_ucr_clustering.py      # UCR 数据测试 + 指标对比
├── scripts/                        # 实验脚本（chen_*、run_*）
├── datasets/                       # aeon / 本地 UCR 数据缓存，不上传 Git
├── results/
└── requirements.txt
```

## 2. 环境与安装

### 2.1. 环境配置

**Python 版本要求：Python 3.11（至少 3.10，因为代码使用 `X | Y` PEP 604 联合类型语法）。**

建议使用项目本地虚拟环境 .venv，并与协作者统一到同一 Python 版本，避免随机数流/数值精度跨版本漂移。

```bash
cd <repo-root>

# 创建虚拟环境（首次）
python -m venv .venv

# Windows Git Bash
source .venv/Scripts/activate

# 安装依赖
pip install -r requirements.txt
```

如果你已在 .venv 中，可以直接执行下面所有命令。

### 2.2. UCR 数据入口

Chen 侧实验脚本默认使用 `aeon` 加载 UCR Time Series Archive 数据集，不需要手动下载完整 UCR 压缩包。首次运行时，数据会自动缓存到：

```text
datasets/aeon/
```

默认数据入口示例：

```bash
python scripts/chen_part1_benchmark.py --datasets ECG200 --metrics ed dtw msm --samples-per-class 5 --seeds 1 --metric-backend reference --output scratch/results/ecg200_smoke.csv
```

正式实验建议显式指定 aeon backend：

```bash
python scripts/chen_part1_benchmark.py --samples-per-class 0 --seeds 1 2 3 4 5 6 7 8 9 10 --metric-backend aeon
```

只有在需要读取本地 UCR `TRAIN/TEST` 文件时，才使用文件模式：

```bash
python scripts/chen_part1_benchmark.py --data-source files --data-root datasets --datasets ECG200
```

### 2.3. Chen 侧协作入口

Week 3 之后，Chen/Wang 协作以统一数据接口、统一结果 schema 和统一 benchmark pipeline 为核心。相关文档：

- `docs/CHEN_QUICKSTART.md`
- `docs/协作方法.md`
- `docs/数据集选择.md`
- `docs/待完善.md`
- `docs/chen_report_sections.md`

## 3. 测试脚本用法

### 3.1 合成数据基础测试

```bash
python tests/test_clustering.py
```

### 3.2 UCR 数据单次测试（test_ucr_clustering.py）

```bash
python tests/test_ucr_clustering.py --train <TRAIN文件> --test <TEST文件> [可选参数]
```

示例：

```bash
python tests/test_ucr_clustering.py \
  --train data/Univariate_arff/ACSF1/ACSF1_TRAIN.txt \
  --test data/Univariate_arff/ACSF1/ACSF1_TEST.txt \
  --k 10 --no-viz
```

### 3.3 IDK vs Euclidean 对比测试

```bash
python tests/test_ucr_clustering.py \
  --train data/Univariate_arff/ACSF1/ACSF1_TRAIN.txt \
  --test data/Univariate_arff/ACSF1/ACSF1_TEST.txt \
  --k 10 --no-viz --compare-metrics
```

`scripts/run_ucr_unsupervised_compare.py`提供了简便的对比方式，可以自动选择较为合适的参数。

```bash
python scripts/run_ucr_unsupervised_compare.py \
ACSF1 --k-min 2 --k-max 6 [no-viz]
```


说明：
- compare 模式会连续跑两次：idk 与 euclidean。
- 末尾会输出汇总表：NMI、ARI、Runtime 以及差值 ΔNMI/ΔARI。

### 3.4 大序列数据的稳定参数（避免 IDK 内存爆炸）

对于 ACSF1 这类长度较大的序列，推荐显式设置窗口参数：

```bash
python tests/test_ucr_clustering.py \
  --train data/Univariate_arff/ACSF1/ACSF1_TRAIN.txt \
  --test data/Univariate_arff/ACSF1/ACSF1_TEST.txt \
  --k 10 --no-viz --compare-metrics \
  --window-size 200 --window-step 50 \
  --n-trees 100 --sample-size 128
```

## 4. test_ucr_clustering.py 参数说明

| 参数                | 说明                                     | 默认值 |
| ------------------- | ---------------------------------------- | ------ |
| --train             | 训练文件路径（必填）                     | 无     |
| --test              | 测试文件路径（可选）                     | None   |
| --k                 | 聚类数；不填则使用真实类别数（监督模式） | None   |
| --no-normalize      | 关闭 z-score 标准化                      | False  |
| --no-viz            | 跳过可视化（加速）                       | False  |
| --similarity-metric | 单次测试所用度量：idk 或 euclidean       | idk    |
| --compare-metrics   | 开启 idk 与 euclidean 对比模式           | False  |
| --window-size       | IDK 滑窗长度                             | None   |
| --window-step       | IDK 滑窗步长                             | None   |
| --n-trees           | IDK 树数量                               | 200    |
| --sample-size       | IDK 每棵树采样数                         | 256    |
| --n-samples         | 随机抽样样本数（用于快速实验）           | None   |

## 5. 输出与结果

- 单次测试输出：
  - NMI
  - ARI
  - 混淆矩阵
  - 每个簇的样本数和簇内平均距离
- 对比模式额外输出：
  - idk / euclidean 的并列表
  - ΔNMI、ΔARI、ΔTime
- 可视化输出目录：
  - results/<数据集名>_idk_viz
  - results/<数据集名>_euclidean_viz

## 6. Python API 最小示例

```python
import numpy as np
from tsclust.clustering import cluster_time_series

X = np.random.randn(50, 100)

result = cluster_time_series(
    X,
    k=3,
    similarity_metric="idk",  # 或 "euclidean"
    n_trees=200,
    sample_size=256,
    normalize=True,
    random_state=42,
)

print(result.medoids)
print(result.labels)
```

## 7. 常见问题

1) 运行时报 NumPy / sklearn / scipy 版本冲突
- 现象：提示 compiled using NumPy 1.x cannot run in NumPy 2.x
- 处理：在 .venv 中重新安装兼容版本（建议使用 requirements.txt）

2) IDK 内存占用过高
- 处理：增大 window-size、增大 window-step，或减小 n-trees / sample-size
- 推荐起点：window-size=200, window-step=50, n-trees=100, sample-size=128

## 8. 备注

本 README 仅保留当前代码已实现且可直接运行的流程，已删除历史版本中不再使用或与现有脚本不一致的说明。

## 9. 扩展：如何添加其它距离/相似度度量（例如 DTW、MSM）

当你想比较更多时间序列距离度量（例如 DTW、MSM、LB_Keogh、Shape-Based Distance 等），建议按下面步骤将新度量集成到项目中：

1) 安装所需依赖（示例）

```bash
# 推荐的库：
# - dtaidistance: 高效的 DTW 实现
# - tslearn: 提供 DTW、MSM、KShape 等时间序列方法
pip install dtaidistance tslearn
```

2) 在 `tsclust/clustering/clustering.py` 中添加 dispatch 分支

示例（伪代码）：

```python
from dtaidistance import dtw
from tslearn.metrics import cdist_dtw

if similarity_metric == "dtw":
  # 计算成距离矩阵（示例：使用 tslearn 的 cdist_dtw）
  dist = cdist_dtw(X)
  sim = 1.0 / (1.0 + dist)
elif similarity_metric == "msm":
  # tslearn 中可能没有直接 MSM 实现，或使用自定义实现
  dist = custom_msm_distance_matrix(X, **similarity_params)
  sim = 1.0 / (1.0 + dist)
```

3) 如果度量需要预处理（例如 DTW 常处理不同长度的序列或需要归一化），请在 `cluster_time_series()` 中的 `normalize` 或在度量分支里显式处理。

4) 写入包装函数与测试用例

- 在 `tests/test_ucr_clustering.py` 的 CLI 中，`--similarity-metric` 已支持传入字符串。添加新度量后，可以直接通过命令行调用：

```bash
python tests/test_ucr_clustering.py --train <TRAIN> --test <TEST> --similarity-metric dtw --k 3
```

- 为新度量写一个单元测试（例如 `tests/test_metrics.py`），验证计算的距离矩阵满足对称性、对角为 0 等属性。

5) 运行基准并记录结果

- 使用已有的 `scripts/run_ucr_unsupervised_compare.py` 做批量对比（脚本会自动调用 `test_ucr_clustering.py`），例如：

```bash
python scripts/run_ucr_unsupervised_compare.py BeetleFly --k-min 2 --k-max 6
```

6) 性能与加速建议

- DTW 等度量计算成本高，建议：
  - 在 sweep 前使用 `--n-samples` 做抽样测试
  - 使用向量化或 C/Numba 实现的库（如 `dtaidistance` 的 C 绑定）
  - 并行计算距离矩阵（注意内存）

7) 参数和可复现性

- 将所有重要参数（度量名、window_size、window_step、n_trees、sample_size、随机种子）写入输出 `summary.json`，便于复现实验。

8) 示例：集成 DTW（详细示例）

在 `tsclust/clustering/clustering.py` 的 `cluster_time_series()` 中添加：

```python
elif similarity_metric == "dtw":
  # 使用 tslearn 计算序列间 DTW 距离（需要先 pip install tslearn）
  from tslearn.metrics import cdist_dtw

  # X 的 shape 为 (n_samples, series_length)
  dist = cdist_dtw(X)
  sim = 1.0 / (1.0 + dist)
```

注意：对于变长序列，先用 `np.nan` 填充到相同长度或使用 tslearn 的工具将序列包装为 `TimeSeries`。一般来说直接用等长数据集即可。

9) 将新度量记录到 README

在添加新度量后，更新本 README 的第 3 节或第 5 节，给出使用示例（如上）。

