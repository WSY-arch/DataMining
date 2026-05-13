# Chen Quickstart

The Chen-side implementation adds ED, DTW, and MSM support to the existing k-medoids pipeline, plus scripts and writing artifacts for the proposal plan.

## Run Part 1

```bash
python scripts/chen_part1_benchmark.py --samples-per-class 50
```

This writes:

```text
results/chen/part1_results.csv
```

Use `--samples-per-class 20` for a fast smoke test, or `--samples-per-class 0` to use all samples.

By default, scripts use `--data-source aeon`, so UCR datasets are loaded through aeon and cached under:

```text
datasets/aeon/
```

Use local TRAIN/TEST files only when needed:

```bash
python scripts/chen_part1_benchmark.py --data-source files --data-root datasets --datasets ECG200
```

Formal repeated run example:

```bash
python scripts/chen_part1_benchmark.py --samples-per-class 0 --seeds 1 2 3 4 5 6 7 8 9 10 --metric-backend aeon
```

`--metric-backend aeon` requires aeon to be installed. Use `--metric-backend reference` only for learning, debugging, or small smoke tests.

## Run Part 2

```bash
python scripts/chen_part2_perturbations.py --samples-per-class 50
```

This writes:

```text
results/chen/part2_perturbation_results.csv
results/chen/perturbation_curves/
```

## Analyze combined Chen/Wang results

After Wang appends SBD/IDK rows with the same schema, save the combined file as:

```text
results/chen/combined_part1_results.csv
```

Then run:

```bash
python scripts/chen_analyze_results.py --input results/chen/combined_part1_results.csv --score-field ari
python scripts/chen_analyze_results.py --input results/chen/combined_part1_results.csv --score-field nmi
```

The analysis script first aggregates repeated seeds and reports mean/std before computing ranks.

## Writing artifacts

- `docs/chen_novelty_memo.md`
- `docs/chen_dataset_selection.md`
- `docs/chen_collaboration_protocol.md`
- `docs/chen_report_sections.md`
