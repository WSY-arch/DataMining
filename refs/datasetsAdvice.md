
## 选择原则

基于你们的研究设计——主实验（Part 1 真实数据 benchmark）+ 扰动实验（Part 2 噪声/时移/长度），数据集需要满足：

**多样性维度**：长度跨度（短 < 100 → 长 > 1000）、类别数跨度（k=2 → k=8+）、域覆盖（ECG/motion/shape/spectroscopy/synthetic/sensor），这样才能在 Friedman + Nemenyi 检验下发现度量间的显著差异。

**实验可行性**：UCR 全部 equal-length（满足你们 scope 中"Equal-length series for fair comparison"的要求）；规模适中（DTW 是 O(n²)，过长序列 + 大样本量会让 pairwise distance matrix 计算不现实）。

**度量区分度**：选含有时移、warping、shape patterns 的数据，能让不同范式的度量呈现差异化表现，避免所有度量在某些数据上"打平"导致结论失效。

**避坑**：避免极端不平衡数据集（如 Wafer 90% 单类）、过多类别数据集（如 Adiac k=37、Phoneme k=39 聚类难度过高）、过长数据集（如 HandOutlines length=2709 算 DTW 太慢）。

## 推荐数据集（18 个）

| #   | Dataset             | Length | k   | Train+Test | Domain       | 选择理由                                                                  |
| --- | ------------------- | ------ | --- | ---------- | ------------ | ------------------------------------------------------------------------- |
| 1   | SyntheticControl    | 60     | 6   | 600        | Synthetic    | 聚类经典 benchmark，6 类受控模式（normal/cyclic/trend/shift），shape 清晰 |
| 2   | CBF                 | 128    | 3   | 930        | Synthetic    | Cylinder-Bell-Funnel 三类，shape 区分度极强，DTW 文献必引                 |
| 3   | TwoPatterns         | 128    | 4   | 5000       | Synthetic    | 四类合成模式，含时间扭曲，DTW 优势场景                                    |
| 4   | ItalyPowerDemand    | 24     | 2   | 1096       | Sensor       | **极短序列**，length 实验下界，能源域                                     |
| 5   | MoteStrain          | 84     | 2   | 1272       | Sensor       | 短序列 IoT 传感器数据                                                     |
| 6   | ECG200              | 96     | 2   | 200        | Medical      | 经典 ECG 二分类，文献广泛对比                                             |
| 7   | ECGFiveDays         | 136    | 2   | 884        | Medical      | 跨日 ECG，含相位差异，对 SBD 友好                                         |
| 8   | GunPoint            | 150    | 2   | 200        | Motion       | 经典动作捕捉，含轻微 warping                                              |
| 9   | Plane               | 144    | 7   | 210        | Sensor       | **k=7 多类**，雷达回波，shape 各异                                        |
| 10  | Trace               | 275    | 4   | 200        | Synthetic    | 工业过程合成，明显 shape 模式                                             |
| 11  | ArrowHead           | 251    | 3   | 211        | Shape        | 考古箭头轮廓，shape based                                                 |
| 12  | Coffee              | 286    | 2   | 56         | Spectroscopy | **光谱域**，与时序模式不同的数据特性                                      |
| 13  | DiatomSizeReduction | 345    | 4   | 322        | Image        | 硅藻图像轮廓，4 类形状                                                    |
| 14  | FaceFour            | 350    | 4   | 112        | Image        | 人脸轮廓投影，少样本但 shape 清晰                                         |
| 15  | Symbols             | 398    | 6   | 1020       | Shape        | 手绘符号，6 类，shape clustering 经典                                     |
| 16  | OSULeaf             | 427    | 6   | 442        | Shape        | 6 种树叶轮廓                                                              |
| 17  | Beef                | 470    | 5   | 60         | Spectroscopy | 牛肉光谱五分类，长序列+多类                                               |
| 18  | Mallat              | 1024   | 8   | 2400       | Synthetic    | **长序列**（length=1024），length 实验上界                                |

## 各维度覆盖情况

**长度分布**：极短 24（1）→ 短 60-150（5）→ 中 200-400（5）→ 较长 400-500（3）→ 长 1024（1）。完整覆盖了你们要做的 length 扰动实验需要的跨度。

**类别数分布**：k=2（5 个）→ k=3-4（4 个）→ k=5-7（7 个）→ k=8（1 个）。多数据集偏向 k=2-7 是合理的，因为 IDK 等度量在过多类别下区分度也会下降，不利于公平对比。

**域分布**：合成 5 个（baseline，shape 清晰），ECG/医疗 2 个，motion 1 个，sensor 3 个，shape/image 5 个，spectroscopy 2 个。多样性足够发表级研究。

## Part 2 扰动实验推荐选 3 个

从上述 18 个中挑选这三个做受控扰动实验最合适：

**CBF**（length 128, k=3）——shape 区分极清晰，扰动后退化曲线最易解读，是文献中扰动实验的标准选择。

**Trace**（length 275, k=4）——合成数据但 shape 多样，中等长度，适合做完整的 noise/shift/length 三种扰动。

**ECG200**（length 96, k=2）——真实数据代表，验证扰动实验结论在真实场景的迁移性。

## 下载地址

**主站点**（一站式下载）：[http://www.timeseriesclassification.com/dataset.php](http://www.timeseriesclassification.com/dataset.php)

**整包下载**（推荐，UCR 2018 archive 全 128 个数据集打包）：[http://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_ts.zip](http://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_ts.zip)

**单数据集页面**（含描述与下载，用 `Dataset=<Name>` 替换数据集名）：
`http://www.timeseriesclassification.com/description.php?Dataset=ECG200`

**通过 Python 库直接加载**（最方便，推荐）：

```python
from aeon.datasets import load_classification
X_train, y_train = load_classification("ECG200", split="train")
X_test, y_test = load_classification("ECG200", split="test")
# 聚类时一般合并 train+test 一起用
```

aeon 会自动下载到本地缓存，无需手动管理文件。tslearn 也有类似 `UCR_UEA_datasets()` 接口。

## 几点提醒

如果计算资源有限，可以**先跑前 12 个**（去掉 Mallat、Beef、OSULeaf、Symbols、ECGFiveDays、Coffee 这些较长或样本数中等的），跑通 pipeline 之后再扩展。

Pairwise distance matrix 在样本数 N 较大时会成为瓶颈（O(N²) 次距离计算）。TwoPatterns 有 5000 样本，MoteStrain 1272 样本，可以**对样本做随机子采样**到每类 50-100 个，这是聚类 benchmark 中的标准做法（Paparrizos 2025 也对部分大数据集做了子采样）。

数据预处理：所有度量比较前**都用 z-normalization 统一处理**，这是 UCR 标准做法，也是 Paparrizos 2020 文章证明过的——不做归一化的对比是 unfair 的。

