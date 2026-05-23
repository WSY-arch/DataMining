# Chen 快速开始

Chen 侧实现为现有的 k-medoids 聚类流程增加了 ED、DTW 和 MSM 支持，同时补充了 proposal 计划所需的实验脚本和写作材料。

## 运行 Part 1

```bash
python scripts/chen_part1_benchmark.py --samples-per-class 50
```

该命令会写出：

```text
results/chen/part1_results.csv
```

快速 smoke test 可以使用 `--samples-per-class 20`；如果要使用全部样本，则使用 `--samples-per-class 0`。

默认情况下，脚本使用 `--data-source aeon`，因此 UCR 数据集会通过 aeon 加载，并缓存到：

```text
datasets/aeon/
```

只有在需要读取本地 TRAIN/TEST 文件时，才使用本地文件模式：

```bash
python scripts/chen_part1_benchmark.py --data-source files --data-root datasets --datasets ECG200
```

正式重复实验示例：

```bash
python scripts/chen_part1_benchmark.py --samples-per-class 0 --seeds 1 2 3 4 5 6 7 8 9 10 --metric-backend aeon
```

`--metric-backend aeon` 需要先安装 aeon。`--metric-backend reference` 只建议用于学习、调试或小规模 smoke test。

## 运行 Part 2

```bash
python scripts/chen_part2_perturbations.py --samples-per-class 50
```

该命令会写出：

```text
results/chen/part2_perturbation_results.csv
results/chen/perturbation_curves/
```

## 分析合并后的 Chen/Wang 结果

当 Wang 按相同 schema 追加 SBD/IDK 结果行后，将合并文件保存为：

```text
results/chen/combined_part1_results.csv
```

然后运行：

```bash
python scripts/chen_analyze_results.py --input results/chen/combined_part1_results.csv --score-field ari
python scripts/chen_analyze_results.py --input results/chen/combined_part1_results.csv --score-field nmi
```

分析脚本会先聚合重复 seed，并报告 mean/std，然后再计算排名。

## 写作材料

- `docs/chen_novelty_memo.md`
- `docs/数据集选择.md`
- `docs/协作方法.md`
- `docs/待完善.md`
- `docs/chen_report_sections.md`
