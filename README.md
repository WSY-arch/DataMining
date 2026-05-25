# Time-Series Clustering Similarity Benchmark

本项目用于比较不同时间序列相似性度量在无监督聚类任务中的表现，并分析它们在噪声、时间错位和长度变化下的机制差异。

核心思想是：不同 similarity measures 编码了不同的 invariance。Euclidean Distance 强调逐点对齐，DTW/MSM 允许弹性时间轴匹配，SBD 强调全局 shift invariance，IDK 则从 distributional kernel 的角度比较时间序列。本项目关注的不是寻找一个永远最优的方法，而是解释不同方法在什么数据条件下更合适。

## 1. Project Scope

本项目聚焦于：

1. 单变量 whole time-series clustering。
2. UCR Time Series Archive 中的代表性数据集。
3. 统一聚类框架下的相似性度量比较。
4. 噪声、temporal shift 和 sequence length 的受控扰动实验。
5. ARI、NMI、runtime、平均排名和统计检验。

本项目不覆盖：

1. deep learning-based clustering。
2. multivariate time series。
3. supervised classification。
4. feature-based 或 model-based representations。

## 2. Compared Measures

| Paradigm       | Measure | Role                             |
| -------------- | ------- | -------------------------------- |
| Lock-step      | ED      | Baseline point-to-point distance |
| Elastic        | DTW     | Time-axis warping distance       |
| Elastic        | MSM     | Edit-based elastic distance      |
| Sliding        | SBD     | Shift-invariant shape distance   |
| Distributional | IDK     | Distributional kernel similarity |

所有方法最终需要进入统一的 clustering/evaluation pipeline。对于 distance measures，直接产生 pairwise distance matrix；对于 kernel/similarity measures，需要明确 similarity-to-distance 转换或使用等价的统一方案。

## 3. Repository Structure

```text
.
├── tsclust/                    # Core package
│   ├── measures/               # ED / DTW / MSM / IDK-related measures
│   ├── clustering/             # k-medoids and clustering dispatch
│   └── visualization/          # Plotting helpers
├── scripts/                    # Benchmark, perturbation, and analysis scripts
├── tests/                      # Unit and smoke tests
├── docs/                       # Project notes, collaboration protocol, report drafts
├── refs/                       # Proposal and reference materials
├── scratch/                    # Local smoke-test helpers; not formal results
├── datasets/                   # Local data cache, ignored by Git
├── results/                    # Local experiment outputs, ignored by Git
└── requirements.txt
```

## 4. Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On Git Bash or Linux/macOS:

```bash
source .venv/Scripts/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The benchmark scripts use `aeon` as the default UCR data interface. Downloaded datasets are cached locally under:

```text
datasets/aeon/
```

The `datasets/` and `results/` directories are ignored by Git because they may contain large local files.

## 5. Quick Smoke Test

Run a small ECG200 smoke test with ED, DTW, and MSM:

```bash
python scripts/chen_part1_benchmark.py --datasets ECG200 --metrics ed dtw msm --samples-per-class 5 --seeds 1 --metric-backend reference --output scratch/results/ecg200_smoke.csv
```

This checks that the data loader, distance computation, k-medoids pipeline, and result writer are connected. The output is only for smoke testing and should not be interpreted as a formal benchmark result.

Run the core similarity-measure tests:

```bash
python -m pytest tests/test_similarity_measures.py -q
```

## 6. Main Benchmark Workflow

Part 1 runs the real-data UCR benchmark (10 seeds for `mean ± std` reporting):

```bash
python scripts/chen_part1_benchmark.py --samples-per-class 0 --seeds 1 2 3 4 5 6 7 8 9 10 --metric-backend aeon
```

A single-seed run can be used as a faster sanity check:

```bash
python scripts/chen_part1_benchmark.py --samples-per-class 0 --seeds 1 --metric-backend aeon
```

Part 2 runs controlled perturbation experiments:

```bash
python scripts/chen_part2_perturbations.py --samples-per-class 50 --metric-backend aeon
```

Once per-measure results are concatenated into a single CSV, aggregate seed-level scores and compute rankings:

```bash
python scripts/chen_analyze_results.py --input <combined_results.csv> --score-field ari
python scripts/chen_analyze_results.py --input <combined_results.csv> --score-field nmi
```

## 7. Shared Result Schema

Every benchmark run should write one row per dataset, measure, seed, and perturbation level:

```text
dataset,measure,paradigm,ari,nmi,runtime,seed,perturbation_type,perturbation_level,n_samples,series_length,k
```

This schema is used to keep ED, DTW, MSM, SBD, and IDK comparable under the same evaluation framework.

## 8. Reproducibility Notes

1. Use the same selected UCR datasets for all compared measures.
2. Use the same preprocessing and normalization settings.
3. Set `k` to the ground-truth number of classes for benchmark comparability.
4. Run multiple seeds for k-medoids and report mean/std when possible.
5. Use the same backend setting when comparing runtime.
6. Treat small sampled runs as smoke tests, not formal evidence.

## 9. Documentation

Additional project notes are maintained under `docs/`, including dataset selection rationale and a list of known implementation limitations. Internal working notes and draft report sections also live there but are not part of the public interface.

## 10. Current Status

The current implementation provides a shared k-medoids pipeline, aeon-based UCR loading, ED/DTW/MSM benchmark runs, controlled perturbation experiments, and result aggregation utilities. SBD and IDK integrations are in progress and will be released alongside the corresponding result tables.

## 11. Acknowledgments

This project uses the [UCR Time Series Classification Archive](https://www.timeseriesclassification.com/) for benchmark data, accessed via the [`aeon`](https://www.aeon-toolkit.org/) toolkit. Please cite the UCR Archive (Dau et al., 2018) and aeon (Middlehurst et al., 2024) when using this codebase.

## 12. License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for details.
