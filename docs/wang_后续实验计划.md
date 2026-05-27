# Wang — 后续扰动实验计划

目的
- 补充 Part2：比较 `IDK` 在有滑窗（默认）与无滑窗（embedding 使用整条序列）两种设置下的鲁棒性。
- 与已有 `SBD` 一致，使用相同的噪声/位移/长度扰动协议，保证可比性（相同的 per-cell .npz 输入）。

数据集（本次补充使用 Part2 同三集）
- `CBF`：序列长度 128
- `Trace`：序列长度 275
- `ECG200`：序列长度 96

实验设计（继承 docs/methodology/perturbation_design.md）
- 扰动类型：noise / shift / length
- 噪声等级：0.0, 0.1, 0.2, 0.4, 0.8
- 位移百分比：0, 5, 10, 20, 30（记录绝对移位样本数）
- 长度保留率：1.0, 0.9, 0.75, 0.5, 0.25
- 随机重复：seeds 1..10
- 每类采样：`--samples-per-class` 与主实验保持一致（默认 50）

IDK 两个设置
- 有滑窗（默认）：`IsolationKernel` 使用内部窗口抽取（默认 `window_size=min(10,L)`，`window_step=1`），适合捕捉局部模式。
- 无滑窗（强制 embedding 整条序列）：通过设置 `no_window_threshold >= series_length` 强制无窗。例如，传递 `no_window_threshold=series_length`。

执行方式（建议）
- 先确保 Part2 脚本生成 per-cell .npz（同一个 `--cells-dir`），以保证 SBD/IDK 在同一输入上比较。

示例：有滑窗（默认）
```bash
.venv/Scripts/python.exe scripts/chen_part2_perturbations.py \
  --data-source aeon \
  --datasets CBF Trace ECG200 \
  --metrics idk \
  --metric-backend aeon \
  --shift-mode padding \
  --noise-levels 0.0 0.1 0.2 0.4 0.8 \
  --shift-pct 0 5 10 20 30 \
  --length-fractions 1.0 0.9 0.75 0.5 0.25 \
  --samples-per-class 50 \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --cells-dir results/_perturbed_cells \
  --output results/chen/part2_idk_with_window.csv \
  --plot-dir results/chen/perturbation_curves_with_window
```

示例：无滑窗（强制整条序列）
- 方法 A（推荐，利用 `similarity_params` 传入 `no_window_threshold`）：
```bash
.venv/Scripts/python.exe scripts/chen_part2_perturbations.py \
  --data-source aeon \
  --datasets CBF Trace ECG200 \
  --metrics idk \
  --metric-backend aeon \
  --shift-mode padding \
  --noise-levels 0.0 0.1 0.2 0.4 0.8 \
  --shift-pct 0 5 10 20 30 \
  --length-fractions 1.0 0.9 0.75 0.5 0.25 \
  --samples-per-class 50 \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --cells-dir results/_perturbed_cells \
  --output results/chen/part2_idk_no_window.csv \
  --plot-dir results/chen/perturbation_curves_no_window \
  --continue-on-error
```

说明：当前 `chen_part2_perturbations.py` 已实现 `similarity_params` 传递路径；要强制 `no_window_threshold`，可以临时在脚本中把 `similarity_params` 设置为 `{"backend": args.metric_backend, "no_window_threshold": series_length}`，或手动在两次运行之间替换脚本中的默认 `similarity_params`。

输出与后续分析
- 输出 CSV：`results/chen/part2_idk_with_window.csv` 与 `results/chen/part2_idk_no_window.csv`。
- 绘图目录：`results/chen/perturbation_curves_with_window` / `..._no_window`。
- 汇总分析：计算每个 dataset/measure/perturbation_level 的 mean±std（ARI, NMI, runtime），并绘制对比曲线（SBD 已有基线，可把曲线并入同一图做对比）。
- 报告更新：把关键图和表格插入 `docs/实验报告_SBD_IDK.md`，并说明 `no_window_threshold` 的设置与实验可比性。

重复性与记录
- 确保 `--cells-dir` 使用同一路径，并且不要在两个实验之间清空它，这样两个设置会在相同 perturbed cells 上运行。
- 在 CSV 的 `measure_params` 字段记录 `no_window_threshold` 值以便追溯。

时间与资源估计
- 运行时间取决于数据集与 `idk` 参数（窗口/样本数）。估计三数据集、10 seeds、全部扰动组合会耗数小时到十数小时（视 CPU 核数与 `aeon` 加速情况）。建议先在 1-2 seeds 上做 smoke-run 验证。 

结论
- 该补充分两次运行（有窗/无窗），保持其它参数相同并保存 per-cell 输入以保证严格可比。完成后汇总并把图表写入报告。
