# Week 3 Chen 侧执行计划

主线：把 ED/DTW/MSM/SBD/IDK 放进同一个公平实验系统，产出可合并结果。

---

## DONE（已完成）

- [x] README 面向公众、QUICKSTART 链接修复、协作方法精简。
- [x] `docs/数据集选择.md` 18 个数据集与 SELECTED_DATASETS 一致。
- [x] `requirements.txt` 含 `aeon>=1.1`；.gitignore 排除 datasets/results/。
- [x] pytest 通过；18 数据集 aeon 加载全部 OK。
- [x] push feature/chen-week3；通知 Wang 新包结构和接入方式。
- [x] LICENSE + Acknowledgments。

---

## P0：跑 Part 1 主实验

目标：18 datasets × ED/DTW/MSM × 10 seeds → `results/chen/part1_results.csv`

- [ ] 确认 `--metric-backend aeon` 在 ECG200 上能正常工作。
- [ ] 全量运行 Part 1（如 10 seed 太慢，先 3-5 seed 标注中间版本）。
- [ ] 检查输出：每个 dataset×measure 行数 = seed 数；perturbation_type=none, level=0。
- [ ] 运行 `chen_analyze_results.py` 生成 Chen-only average rank 预览。
- [ ] 记录异常（ARI 为负 / runtime 过长 / 加载失败）。

## P1：跑 Part 2 扰动实验

目标：3-5 代表数据集 × ED/DTW/MSM × noise/shift/length → degradation curves

代表数据集候选：Chinatown（短）、ECG200/GunPoint（经典二分类）、CBF/Trace（shape 清晰）、ACSF1（长）。

- [ ] 确定 perturbation levels（noise: 0.05/0.10/0.20/0.30; shift: 2/5/10/20; length: 0.7/0.8/0.9/1.0）。
- [ ] 先跑 1 个数据集的完整 perturbation smoke test，确认 CSV 输出正确。
- [ ] 正式跑 ED/DTW/MSM 扰动实验。
- [ ] 生成初版 degradation curves。
- [ ] 把 Part 2 扰动函数/数据发给 Wang，让他用同样数据跑 SBD/IDK。

## P2：和 Wang 对齐（⏳ blocked on Wang）

- [ ] 确认 SBD 输出 distance matrix 还是 similarity matrix。
- [x] 确认 IDK 输出 kernel/similarity → 已改用 kernel-induced distance `sqrt(2-2·sim)`（与 CTDS 一致）。
- [ ] 确认 Wang 的 CSV schema〘18 数据集、seed、k 全部一致。

## P3：合并五方法结果（⏳ blocked on P2）

- [ ] 合并 Chen + Wang 的 Part 1 CSV。
- [ ] 运行分析脚本 → per-dataset table、ARI/NMI average rank、Friedman test。
- [ ] Nemenyi/CD diagram 如果来不及，放 Week 4 前置。

---

## Buffer：补跑 + 整理

- [ ] 补跑 P0/P1 中失败或缺失的组合。
- [ ] 检查 combined CSV 无重复行/缺字段/seed 不一致。
- [ ] 写一张机制假设表（每个 measure 适合什么条件），标出支持/不支持/不确定的结果。

---

## 本周完成标准

- [ ] ED/DTW/MSM Part 1 结果就绪。
- [ ] Part 2 至少 3 个数据集有初版 degradation curves。
- [ ] 五方法结果可合并（或 blocked 原因已记录 + Wang 侧接口已确认）。
- [ ] 所有正式结论可追溯到 CSV/图。
