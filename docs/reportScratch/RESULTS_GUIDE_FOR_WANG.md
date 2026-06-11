# Results & Discussion 写作指南（给 Wang）

## 0. 前置说明

- 论文 Intro / Related Work / Methodology 已写完 → `docs/reportScratch/sections_intro_rw_method.tex`
- 编译入口 → `docs/reportScratch/main_compile_test.tex`（XeLaTeX）
- 请保持相同写作风格：正式学术英语，使用 `\cite{}`，表格用 booktabs

---

## 1. 数据文件位置

| 文件 | 路径 | 行数 | 说明 |
|------|------|------|------|
| Part 1 合并 | `results/merged/part1_all_5measures.csv` | 900 | 18 datasets × 5 measures × 10 seeds |
| Part 2 合并 (padding) | `results/merged/part2_all_5measures.csv` | 2250 | 3 datasets × 5 measures × 15 cells × 10 seeds |
| Circular ablation SBD/IDK | `results/merged/part2_circular_ablation_sbd_idk.csv` | 900 | 3 datasets × 2 measures × 15 cells × 10 seeds |
| Circular ablation ED/DTW/MSM | `results/merged/part2_circular_ablation_ed_dtw_msm.csv` | 1350 | 3 datasets × 3 measures × 15 cells × 10 seeds |
| Friedman test 结果 | `results/merged/friedman_cd/` | — | 6 CD图 + 汇总CSV |
| Degradation curves | `results/merged/part2_degradation_curves_meanstd/` | — | 18 张 mean±std 图 |
| IDK windowing 对比 | `results/merged/idk_windowing_compare/` | — | 3 张图 + CSV |

---

## 2. Results 章节结构建议

### 4.1 Part 1: Cross-Dataset Benchmark (对应 Part 1 CSV)

**要呈现的内容：**
- Table: 18 datasets 的 Mean ARI（5 measures × 18 rows）
- Friedman test on Part 1: average rank + p-value
- 关键发现：DTW rank=1.67 (best), MSM=1.92, ED=2.42; p=0.07 接近显著

**数据提取方式：**
```python
import pandas as pd
df = pd.read_csv('results/merged/part1_all_5measures.csv')
pivot = df.groupby(['dataset','measure'])['ari'].mean().unstack()
```

### 4.2 Part 2: Degradation Curves (对应 Part 2 合并 CSV)

**要呈现的内容：**
- Figure: 引用 `results/merged/part2_degradation_curves_meanstd/` 中的图
- 3 组 Friedman test（noise/shift/length），数据见 `friedman_cd/friedman_ari_combined_summary.csv`

**Friedman 结果速查（ARI）：**

| Perturbation | χ² | p-value | Best measure | Rank |
|---|---|---|---|---|
| Noise | 27.41 | 1.6e-5 | **SBD** | 1.60 |
| Shift | 20.91 | 3.3e-4 | **SBD** | 2.13 |
| Length | 15.63 | 0.0036 | **MSM** | 2.00 |

**CD = 1.575** (Nemenyi α=0.05)

**关键论点：**
- SBD 在 noise + shift 下最鲁棒（cross-correlation 的内积结构抗噪，多 lag 搜索容忍位移）
- MSM 在 length 下最鲁棒（split/merge 操作对 FFT resample 引起的频谱变化有独特适应性）
- IDK 整体排名靠后（distributional kernel 在扰动实验中表现不如 shape-based 方法）
- ED 始终最差（lock-step 无法应对任何扰动）

### 4.3 Circular Shift Ablation (对应 circular ablation CSVs)

**核心论证：**

**使用修复后的 true circular CC SBD（Wang 已实现）：**
- SBD 在 circular shift 下 **零退化**（误差 = 0.000000000000）
- 这是 metric 定义的结构性质，不是经验观察

**使用标准库 aeon 的 linear CC SBD：**
- SBD 在 circular shift 下退化 53-86%（因为 zero-padded linear CC ≠ true circular）
- 仍然比 ED(-87%) / DTW(-84%) / MSM(-83%) 好

**IDK 的意外发现：**
- IDK 在 circular shift 下近乎零退化（-3.5% 平均）
- isolation kernel 不依赖时间对齐位置，天然位移不变

**写作角度：**
- Theory vs Practice gap: 理论上 SBD 完美不变，实际标准库实现有 gap
- IDK 的 distributional similarity 在 misalignment 场景下有独特优势

**数据提取方式：**
```python
sbd_idk = pd.read_csv('results/merged/part2_circular_ablation_sbd_idk.csv')
ed_dtw_msm = pd.read_csv('results/merged/part2_circular_ablation_ed_dtw_msm.csv')
all_circ = pd.concat([sbd_idk, ed_dtw_msm])
# 只看 shift
shift = all_circ[all_circ['perturbation_type']=='shift']
pivot = shift.groupby(['perturbation_level','measure'])['ari'].mean().unstack()
```

### 4.4 IDK Windowing Supplementary (对应 idk_windowing_compare/)

**关键数据（已在你的实验报告里）：**
- Direct IDK ARI≈0（完全无效）
- Windowed IDK: Plane 上 ARI = 0.80, ECG200 = 0.15, Trace = 0.29
- 结论：IDK 需要 windowing 策略才能有效，最优窗口因数据集而异

---

## 3. Discussion 章节要点

1. **No single best measure**：DTW 在 clean data 上领先（Part 1），SBD 在扰动下领先（Part 2）——选择取决于数据特性
2. **Mechanism-performance alignment**：每个 measure 的退化模式可从其数学定义推导
   - ED: lock-step → shift 致命
   - DTW: elastic warping → noise 过拟合
   - MSM: edit operations → length 鲁棒
   - SBD: cross-correlation → shift 鲁棒
   - IDK: distributional → shift 意外鲁棒，但整体不如 shape-based
3. **Theory-practice gap for SBD**：true circular CC 实现 vs 标准库 linear CC 实现
4. **IDK 的局限性**：需要 windowing 才有效；在扰动实验中整体排名靠后
5. **Practical recommendations**：
   - 数据对齐良好 → DTW
   - 有噪声/可能错位 → SBD
   - 有截断/采样率变化 → MSM
   - 需要 permutation invariance → IDK (with windowing)

---

## 4. 论文中需要的图表清单

| 图/表 | 内容 | 来源 |
|--------|------|------|
| Table 3 | Part 1 Mean ARI (18 datasets × 5 measures) | `part1_all_5measures.csv` |
| Table 4 | Friedman rank summary (3 perturbation types) | `friedman_cd/friedman_ari_combined_summary.csv` |
| Fig 1-3 | Degradation curves (CBF/Trace/ECG200) | `part2_degradation_curves_meanstd/` |
| Fig 4-6 | CD diagrams (noise/shift/length) | `friedman_cd/cd_*.png` |
| Fig 7 | Circular shift ablation (5 measures) | 从 circular CSVs 生成 |
| Table 5 | IDK windowing comparison | `idk_windowing_compare/idk_windowing_summary.csv` |

---

## 5. SBD 循环不变性验证命令（用于论文引用）

```bash
python -c "from tsclust.measures.similarity_measures import sbd_distance, sbd_distance_matrix; ..."
```

详见 `docs/实验报告_SBD_IDK.md` §6。
