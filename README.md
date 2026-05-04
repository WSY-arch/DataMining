# Isolation Kernel + K-Medoids 单变量时间序列聚类

本项目用于课程实验：在同一聚类算法（K-Medoids）下，对比不同相似度度量（IDK 与 Euclidean）对聚类效果的影响。

## 1. 项目结构

```text
code/
├── isolation_kernel/
│   ├── isolation_kernel.py      # IDK 相似度实现
│   ├── k_medoids.py             # K-Medoids(PAM)
│   ├── clustering.py            # 统一聚类入口（支持 idk/euclidean）
│   └── visualization.py         # 可视化工具
├── tests/
│   ├── test_clustering.py       # 合成数据测试
│   └── test_ucr_clustering.py   # UCR 数据测试 + 指标对比
├── data/
├── results/
└── requirements.txt
```

## 2. 环境与安装

### 2.1. 环境配置

建议使用项目本地虚拟环境 .venv。

```bash
cd code

# 创建虚拟环境（首次）
python -m venv .venv

# Windows Git Bash
source .venv/Scripts/activate

# 安装依赖
pip install -r requirements.txt
```

如果你已在 .venv 中，可以直接执行下面所有命令。

### 2.2. 下载数据集

在 http://timeseriesclassification.com/dataset.php 下载Univariate2018_arff.zip，将解压缩得到的文件存放在`data/Univariate_arff`中，形成如下结构：

```text
data
|
|- Univariate_arff
    |
    |-testset1
    |     |- testset1_TRAIN.txt
    |     |- testset1_TEST.txt
    |     |- ...
    |-testset2
    |     |- ...
    |     |- ...
    |- ...

```


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

| 参数 | 说明 | 默认值 |
|---|---|---|
| --train | 训练文件路径（必填） | 无 |
| --test | 测试文件路径（可选） | None |
| --k | 聚类数；不填则使用真实类别数（监督模式） | None |
| --no-normalize | 关闭 z-score 标准化 | False |
| --no-viz | 跳过可视化（加速） | False |
| --similarity-metric | 单次测试所用度量：idk 或 euclidean | idk |
| --compare-metrics | 开启 idk 与 euclidean 对比模式 | False |
| --window-size | IDK 滑窗长度 | None |
| --window-step | IDK 滑窗步长 | None |
| --n-trees | IDK 树数量 | 200 |
| --sample-size | IDK 每棵树采样数 | 256 |
| --n-samples | 随机抽样样本数（用于快速实验） | None |

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
from isolation_kernel.clustering import cluster_time_series

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
