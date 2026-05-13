# Chen-Wang Collaboration Protocol

## Shared data interface

Every method receives:

```text
X: float array with shape (n_samples, series_length)
y: integer labels used only for evaluation
dataset_meta: dataset name, length, class count, domain
```

All methods should z-normalize each time series before similarity computation unless a specific robustness check says otherwise.

## Shared result CSV schema

Every run must write one row per dataset, measure, seed, and perturbation level:

```text
dataset,measure,paradigm,ari,nmi,runtime,seed,perturbation_type,perturbation_level,n_samples,series_length,k
```

Required values:

- `measure`: `ed`, `dtw`, `msm`, `sbd`, or `idk`.
- `paradigm`: `lock-step`, `elastic`, `sliding`, or `distributional`.
- `perturbation_type`: `none`, `noise`, `shift`, or `length`.
- `perturbation_level`: `0` for main benchmark, numeric level for perturbations.
- `k`: number of ground-truth classes, used as the clustering cluster count for fair comparison.

## Responsibility split

Chen owns:

- Literature positioning and novelty argument.
- Dataset selection table and metadata.
- ED, DTW, MSM implementation and Part 1 runs.
- Initial Part 2 noise/shift/length perturbation runs.
- Introduction, Related Work, and Methodology draft.

Wang owns:

- SBD and IDK implementation or integration.
- SBD/IDK result rows using the same CSV schema.
- Results, Discussion, and Conclusion first draft.
- IDK-specific explanation and limitations.

## Daily sync checklist

Use a 15-minute check-in with only these questions:

1. What new result or artifact was produced today?
2. What is blocking tomorrow's run or writing?
3. Which file or result does the other person depend on next?

## Merge rule

Do not merge result tables by hand. Append CSV rows with the shared schema, then run:

```bash
python scripts/chen_analyze_results.py --input results/chen/combined_part1_results.csv --score-field ari
python scripts/chen_analyze_results.py --input results/chen/combined_part1_results.csv --score-field nmi
```

For formal runs, use multiple seeds:

```bash
python scripts/chen_part1_benchmark.py --samples-per-class 0 --seeds 1 2 3 4 5 6 7 8 9 10 --metric-backend aeon
```

The analyzer aggregates seed-level rows into `mean ± std` before computing average ranks.
