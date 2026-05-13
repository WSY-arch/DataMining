# Implementation Maturity Notes

This file separates learning scaffolding from formal experiment components.

## Ready for formal use

- Unified pipeline: each measure produces a pairwise distance matrix, then k-medoids clusters it.
- Shared CSV schema for Chen/Wang results.
- Fixed `k = number of ground-truth classes` for benchmark comparability.
- Multi-seed repeated runs via `--seeds`.
- Seed aggregation with mean/std in `scripts/chen_analyze_results.py`.

## Learning/reference implementation only

- The built-in DTW and MSM implementations in `tsclust/measures/similarity_measures.py` are readable reference implementations. They are useful for understanding and smoke tests, but they are not the preferred backend for final benchmark claims.
- The `--metric-backend reference` option should be used for debugging small datasets only.

## Preferred formal backend

- Use `--metric-backend aeon` for formal DTW/MSM runs after installing aeon.
- Use `--data-source aeon` to let aeon load UCR datasets and cache them under `datasets/aeon/`.
- If aeon is unavailable, `--metric-backend tslearn` can be used for DTW only; MSM still needs aeon or the reference implementation.
- If `--metric-backend auto` is used, the code tries installed libraries and falls back to reference implementations. This is convenient but less explicit, so formal runs should force the intended backend.

## Still incomplete / future improvements

- SBD is not implemented in Chen's code. Wang should add it using the shared result schema.
- IDK integration exists in the original project, but final experiments need the IDK dependency installed and the similarity-to-distance conversion documented.
- Nemenyi post-hoc and CD diagram are not fully implemented yet. The current analyzer provides average ranks and Friedman test summary.
- The length perturbation currently truncates and pads. This is simple and explainable, but a resampling-based length experiment could be added as a robustness check.
- Runtime comparisons should be interpreted carefully if different backends are mixed. Formal tables should use one explicit backend setting.
