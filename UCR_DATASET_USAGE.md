# 使用 UCR 时间序列数据测试聚类

## 快速开始

### 1. 下载 UCR 数据

从 [UCR Time Series 官方网站](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) 下载你想要的数据集。

**推荐数据集：**
- **GunPoint** - 易分类，2个类，150个训练样本 + 150个测试样本
- **ECG200** - 医疗数据，2个类，100个训练样本 + 100个测试样本  
- **Trace** - 4个类，100个训练样本 + 100个测试样本
- **FaceAll** - 14个类，560个训练样本 + 1690个测试样本

### 2. 解压数据

下载的文件通常为 ZIP 格式，包含 `*_TRAIN.tsv` 和 `*_TEST.tsv` 两个文件。

解压到项目的 `code/data/` 目录：

```bash
cd project/code/data
# 解压你下载的数据集
unzip GunPoint.zip
```

### 3. 运行聚类测试

使用本地 TSV 文件运行聚类：

```bash
cd code

# 基本用法（只用训练数据）
python tests/test_ucr_clustering.py \
  --train data/GunPoint_TRAIN.tsv

# 完整用法（训练 + 测试数据合并）
python tests/test_ucr_clustering.py \
  --train data/GunPoint_TRAIN.tsv \
  --test data/GunPoint_TEST.tsv

# 使用子集（100个样本）
python tests/test_ucr_clustering.py \
  --train data/GunPoint_TRAIN.tsv \
  --test data/GunPoint_TEST.tsv \
  --n-samples 100

# 禁用 Z-score 归一化
python tests/test_ucr_clustering.py \
  --train data/GunPoint_TRAIN.tsv \
  --test data/GunPoint_TEST.tsv \
  --no-normalize
```

## 输出结果

脚本会生成以下内容：

1. **聚类质量评估指标**：
   - NMI (Normalized Mutual Information) - 值域 [0, 1]，越高越好
   - ARI (Adjusted Rand Index) - 值域 [-1, 1]，越高越好
   - 混淆矩阵 - 显示真实标签 vs 预测标签的对应关系

2. **可视化图表**（保存到 `results/{dataset_name}_viz/`）：
   - `clustering_results.png` - 各聚类的时间序列可视化
   - `distance_matrix.png` - 距离矩阵热力图
   - `medoids_comparison.png` - 各聚类中心的代表序列
   - `cluster_statistics.png` - 聚类统计信息

## 数据格式说明

UCR 数据使用 TSV (Tab-Separated Values) 格式：

```
label value1 value2 value3 ... valueN
0     0.1    0.2    0.3    ... 0.5
1     0.3    0.4    0.5    ... 0.7
```

其中：
- 第一列为类别标签（整数）
- 后续列为时间序列的各个时间点的值

## 故障排除

### "Train file not found"

检查文件路径是否正确，确保在 `code` 目录下运行脚本。

### "混淆矩阵不完美"

如果 NMI 或 ARI 不是 1.0，可能需要调整聚类参数。编辑 `tests/test_ucr_clustering.py` 中的：

```python
result = cluster_time_series(
    X,
    k=n_clusters,
    n_trees=200,           # 增加树数量
    sample_size=256,       # 增加样本大小
    normalize=normalize,
    random_state=42,
)
```

### 内存不足

如果数据集太大，使用 `--n-samples` 限制样本数量：

```bash
python tests/test_ucr_clustering.py \
  --train data/FaceAll_TRAIN.tsv \
  --test data/FaceAll_TEST.tsv \
  --n-samples 500
```

## 下一步：实现不同相似度度量

一旦验证了基本聚类的效果，可以开始探究不同相似度度量的影响：

1. **修改** `isolation_kernel/isolation_kernel.py` 中的 `similarity_matrix()` 方法
2. **实现**不同的相似度计算方法（Euclidean、DTW、Correlation 等）
3. **对比**不同相似度度量的聚类质量
4. **生成**对比报告和可视化

详见项目README中的"实现其他相似度度量"章节。
