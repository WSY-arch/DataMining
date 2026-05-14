"""Run Chen's Part 1 benchmark for ED, DTW, and MSM on selected UCR datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.chen_experiment_utils import (  # noqa: E402
    DEFAULT_RESULTS_ROOT,
    SELECTED_DATASETS,
    balanced_subsample,
    DEFAULT_AEON_DATA_ROOT,
    load_dataset,
    resolve_data_root,
    run_single_measure,
    write_result_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-source",
        choices=["aeon", "files"],
        default="aeon",
        help="Load UCR datasets through aeon auto-download/cache or local TRAIN/TEST files.",
    )
    parser.add_argument("--data-root", type=str, default=None,
                        help="Dataset root or aeon cache root.")
    parser.add_argument("--output", type=str,
                        default=str(DEFAULT_RESULTS_ROOT / "part1_results.csv"))
    parser.add_argument("--datasets", nargs="*",
                        default=[meta.name for meta in SELECTED_DATASETS])
    parser.add_argument("--metrics", nargs="*",
                        default=["ed", "dtw", "msm"], help="Metrics to run.")
    parser.add_argument("--samples-per-class", type=int, default=50,
                        help="Balanced cap per class; use 0 for all samples.")
    parser.add_argument("--seeds", nargs="*", type=int,
                        default=[42], help="Random seeds to repeat for each dataset/metric.")
    parser.add_argument(
        "--metric-backend",
        choices=["auto", "reference", "aeon", "tslearn"],
        default="auto",
        help="Backend for DTW/MSM. Use 'aeon' for formal runs after installing aeon.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.data_source == "aeon":
        data_root = Path(
            args.data_root) if args.data_root else DEFAULT_AEON_DATA_ROOT
    else:
        data_root = resolve_data_root(args.data_root)
    samples_per_class = None if args.samples_per_class == 0 else args.samples_per_class
    rows: list[dict[str, object]] = []

    print(f"[INFO] data_source={args.data_source}")
    print(f"[INFO] data_root={data_root}")
    print(f"[INFO] output={args.output}")
    print(f"[INFO] metrics={', '.join(args.metrics)}")

    for dataset_name in args.datasets:
        try:
            X, y = load_dataset(dataset_name, args.data_source, data_root)
            # fixed subsample; all seeds share same subset
            subsample_seed = args.seeds[0]
            X, y = balanced_subsample(X, y, samples_per_class, subsample_seed)
            print(f"[DATASET] {dataset_name}: X={X.shape}, k={len(set(y))}")
            for seed in args.seeds:
                for metric in args.metrics:
                    row = run_single_measure(
                        X,
                        y,
                        dataset_name,
                        metric,
                        clustering_seed=seed,
                        subsample_seed=subsample_seed,
                        similarity_params={"backend": args.metric_backend},
                    )
                    rows.append(row)
                    print(
                        f"  seed={seed:<4} {metric:<4} ARI={row['ari']:.4f} NMI={row['nmi']:.4f} "
                        f"runtime={row['runtime']:.2f}s"
                    )
        except Exception as exc:
            print(f"[ERROR] {dataset_name}: {exc}")
            if not args.continue_on_error:
                return 1

    write_result_rows(rows, Path(args.output))
    print(f"[OK] wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
