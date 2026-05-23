"""Run Chen's Part 1 benchmark for ED, DTW, and MSM on selected UCR datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.chen_experiment_utils import (  # noqa: E402
    DEFAULT_RESULTS_ROOT,
    SELECTED_DATASETS,
    append_result_row,
    balanced_subsample,
    DEFAULT_AEON_DATA_ROOT,
    load_dataset,
    load_done_keys,
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
    parser.add_argument("--resume", action="store_true",
                        help="Skip (dataset, measure, seed) triples already present in --output CSV.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.data_source == "aeon":
        data_root = Path(
            args.data_root) if args.data_root else DEFAULT_AEON_DATA_ROOT
    else:
        data_root = resolve_data_root(args.data_root)
    samples_per_class = None if args.samples_per_class == 0 else args.samples_per_class
    output_path = Path(args.output)

    print(f"[INFO] data_source={args.data_source}")
    print(f"[INFO] data_root={data_root}")
    print(f"[INFO] output={args.output}")
    print(f"[INFO] metrics={', '.join(args.metrics)}")

    done_keys: set = set()
    if args.resume:
        done_keys = load_done_keys(output_path, perturbation_type="none")
        print(
            f"[RESUME] {len(done_keys)} (dataset, measure, level, seed) tuples already done; will skip.")
    elif output_path.exists():
        print(
            f"[WARN] {output_path} exists and --resume not set; file will be APPENDED to (header preserved).")

    n_written = 0
    # unified across all measures (Chen+Wang collab schema)
    SUBSAMPLE_SEED = 42
    for dataset_name in args.datasets:
        try:
            X, y = load_dataset(dataset_name, args.data_source, data_root)
            n_original = int(X.shape[0])
            X, y = balanced_subsample(X, y, samples_per_class, SUBSAMPLE_SEED)
            print(
                f"[DATASET] {dataset_name}: X={X.shape}, k={len(set(y))}, n_original={n_original}")
            for seed in args.seeds:
                for metric in args.metrics:
                    key = (dataset_name, metric, "0", str(seed))
                    if key in done_keys:
                        print(
                            f"  [SKIP] seed={seed:<4} {metric:<4} already done")
                        continue
                    row = run_single_measure(
                        X,
                        y,
                        dataset_name,
                        metric,
                        clustering_seed=seed,
                        subsample_seed=SUBSAMPLE_SEED,
                        similarity_params={"backend": args.metric_backend},
                        n_original=n_original,
                    )
                    append_result_row(row, output_path)
                    n_written += 1
                    print(
                        f"  seed={seed:<4} {metric:<4} ARI={row['ari']:.4f} NMI={row['nmi']:.4f} "
                        f"runtime={row['runtime']:.2f}s"
                    )
        except Exception as exc:
            print(f"[ERROR] {dataset_name}: {exc}")
            if not args.continue_on_error:
                return 1

    print(f"[OK] wrote {n_written} new rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
