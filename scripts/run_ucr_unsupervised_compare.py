"""Run unsupervised comparison experiments on a named UCR dataset.

Usage example:
  python scripts/run_ucr_unsupervised_compare.py BeetleFly --k-min 2 --k-max 10
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Make the project package importable when running from the code root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_ucr_clustering import test_ucr_dataset  # noqa: E402
from tests.test_ucr_clustering import load_ucr_from_file  # noqa: E402


DEFAULT_DATA_ROOT = Path(__file__).parent.parent / "data" / "Univariate_arff"
DEFAULT_RESULTS_ROOT = Path(__file__).parent.parent / "results" / "auto_ucr"


def resolve_ucr_files(dataset_name: str, data_root: Path) -> Tuple[Path, Path]:
    dataset_dir = data_root / dataset_name
    train_file = dataset_dir / f"{dataset_name}_TRAIN.txt"
    test_file = dataset_dir / f"{dataset_name}_TEST.txt"

    if not train_file.exists():
        raise FileNotFoundError(f"Train file not found: {train_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    return train_file, test_file


def score_result(result: Dict[str, float], mode: str) -> float:
    if mode == "nmi":
        return float(result["nmi"])
    if mode == "ari":
        return float(result["ari"])
    return float(result["nmi"] + result["ari"])


def derive_window_params(series_length: int, window_size: int | None, window_step: int | None) -> tuple[int, int]:
    if window_size is None:
        # Keep the window large enough to preserve local shape, but not so small
        # that the number of extracted windows explodes on long series.
        window_size = max(10, min(series_length // 6, 60))
        window_size = min(window_size, series_length)

    if window_step is None:
        window_step = max(1, window_size // 2)

    return window_size, window_step


def derive_k_values(
    y_true: np.ndarray,
    k_min: int,
    k_max: int,
    auto_k: bool,
    lower_band: int,
    upper_band: int,
) -> tuple[List[int], int, int]:
    unique_classes = int(len(np.unique(y_true)))

    if auto_k:
        derived_min = max(2, unique_classes - lower_band)
        derived_max = max(derived_min, unique_classes + upper_band)
        k_min = max(k_min, derived_min)
        k_max = min(k_max, derived_max)

    if k_min > k_max:
        k_min = max(2, min(k_min, unique_classes))
        k_max = max(k_min, unique_classes)

    return list(range(k_min, k_max + 1)), k_min, k_max


def run_sweep(
    train_file: Path,
    test_file: Path,
    k_values: List[int],
    metrics: List[str],
    n_samples: int | None,
    normalize: bool,
    window_size: int | None,
    window_step: int | None,
    n_trees: int,
    sample_size: int,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []

    for k in k_values:
        for metric in metrics:
            buffer = StringIO()
            with redirect_stdout(buffer):
                ok, details = test_ucr_dataset(
                    train_file=str(train_file),
                    test_file=str(test_file),
                    n_samples=n_samples,
                    normalize=normalize,
                    k=k,
                    generate_viz=False,
                    similarity_metric=metric,
                    window_size=window_size,
                    window_step=window_step,
                    n_trees=n_trees,
                    sample_size=sample_size,
                    return_details=True,
                )
            if not ok:
                raise RuntimeError(f"Experiment failed for k={k}, metric={metric}")

            details["k"] = int(k)
            details["score"] = float(details["nmi"] + details["ari"])
            rows.append(details)

            print(
                f"[SWEEP] k={k:<2} metric={metric:<10} "
                f"NMI={details['nmi']:.4f} ARI={details['ari']:.4f} "
                f"score={details['score']:.4f} runtime={details['runtime_sec']:.2f}s"
            )

    return rows


def rerun_best_with_visualization(
    train_file: Path,
    test_file: Path,
    dataset_output_dir: Path,
    best_row: Dict[str, float],
    n_samples: int | None,
    normalize: bool,
    window_size: int,
    window_step: int,
    n_trees: int,
    sample_size: int,
) -> Tuple[Path, Dict[str, float]]:
    viz_dir = dataset_output_dir / f"best_{best_row['metric']}_run" / f"k_{best_row['k']}"
    buffer = StringIO()
    with redirect_stdout(buffer):
        ok, details = test_ucr_dataset(
            train_file=str(train_file),
            test_file=str(test_file),
            n_samples=n_samples,
            normalize=normalize,
            k=int(best_row["k"]),
            generate_viz=True,
            viz_dir=str(viz_dir),
            similarity_metric=str(best_row["metric"]),
            window_size=window_size,
            window_step=window_step,
            n_trees=n_trees,
            sample_size=sample_size,
            return_details=True,
        )
    if not ok:
        raise RuntimeError(f"Visualization rerun failed for metric={best_row['metric']}, k={best_row['k']}")

    details = details | {"k": int(best_row["k"]), "metric": best_row["metric"], "score": float(best_row["score"])}
    return viz_dir, details


def write_csv(rows: List[Dict[str, float]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "metric",
        "k",
        "nmi",
        "ari",
        "score",
        "runtime_sec",
        "n_samples",
        "series_length",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_json(payload: Dict[str, object], json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automatically compare IDK vs Euclidean on a named UCR dataset and save the best run.",
    )
    parser.add_argument("dataset", type=str, help="UCR dataset name, e.g. BeetleFly or WINE")
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT), help="Root directory of UCR data")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_RESULTS_ROOT), help="Root directory for saved results")
    parser.add_argument("--k-min", type=int, default=2, help="Minimum k to try")
    parser.add_argument("--k-max", type=int, default=10, help="Maximum k to try (inclusive)")
    parser.add_argument("--no-auto-k", action="store_true", help="Disable automatic k-range narrowing around the dataset class count")
    parser.add_argument("--k-band-low", type=int, default=1, help="How many clusters below the class count to include when auto k is enabled")
    parser.add_argument("--k-band-high", type=int, default=2, help="How many clusters above the class count to include when auto k is enabled")
    parser.add_argument("--best-by", type=str, choices=["composite", "nmi", "ari"], default="composite", help="How to rank candidate runs")
    parser.add_argument("--n-samples", type=int, default=None, help="Optional sample limit")
    parser.add_argument("--no-normalize", action="store_true", help="Disable z-score normalization")
    parser.add_argument("--window-size", type=int, default=None, help="Sliding window size for IDK")
    parser.add_argument("--window-step", type=int, default=None, help="Sliding window step for IDK")
    parser.add_argument("--n-trees", type=int, default=200, help="Number of IDK trees")
    parser.add_argument("--sample-size", type=int, default=256, help="IDK tree sample size")
    parser.add_argument("--no-viz", action="store_true", help="Skip saving visualizations for the best run")
    args = parser.parse_args()

    dataset_name = args.dataset.strip()
    data_root = Path(args.data_root)
    output_root = Path(args.output_dir)

    train_file, test_file = resolve_ucr_files(dataset_name, data_root)
    preview_X, y_preview = load_ucr_from_file(str(train_file), str(test_file))
    effective_window_size, effective_window_step = derive_window_params(
        series_length=int(preview_X.shape[1]),
        window_size=args.window_size,
        window_step=args.window_step,
    )
    k_values, effective_k_min, effective_k_max = derive_k_values(
        y_true=y_preview,
        k_min=args.k_min,
        k_max=args.k_max,
        auto_k=not args.no_auto_k,
        lower_band=args.k_band_low,
        upper_band=args.k_band_high,
    )
    metrics = ["idk", "euclidean"]

    print("=" * 78)
    print(f"Dataset: {dataset_name}")
    print(f"Train:   {train_file}")
    print(f"Test:    {test_file}")
    print(f"k range: {effective_k_min}..{effective_k_max}")
    print(f"Metrics: {', '.join(metrics)}")
    print(f"Window:  size={effective_window_size}, step={effective_window_step}")
    print("=" * 78)

    rows = run_sweep(
        train_file=train_file,
        test_file=test_file,
        k_values=k_values,
        metrics=metrics,
        n_samples=args.n_samples,
        normalize=not args.no_normalize,
        window_size=effective_window_size,
        window_step=effective_window_step,
        n_trees=args.n_trees,
        sample_size=args.sample_size,
    )

    best_overall = max(rows, key=lambda row: score_result(row, args.best_by))
    best_by_metric: Dict[str, Dict[str, float]] = {}
    for metric in metrics:
        metric_rows = [row for row in rows if row["metric"] == metric]
        best_by_metric[metric] = max(metric_rows, key=lambda row: score_result(row, args.best_by))

    dataset_output_dir = output_root / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = dataset_output_dir / "comparison_table.csv"
    json_path = dataset_output_dir / "summary.json"
    write_csv(rows, csv_path)

    best_viz_dirs: Dict[str, Path] = {}
    if not args.no_viz:
        for metric in metrics:
            best_row = best_by_metric[metric]
            print()
            print(f"[BEST] Re-running best {metric} configuration with visualizations: k={best_row['k']}, metric={best_row['metric']}")
            viz_dir, details = rerun_best_with_visualization(
                train_file=train_file,
                test_file=test_file,
                dataset_output_dir=dataset_output_dir,
                best_row=best_row,
                n_samples=args.n_samples,
                normalize=not args.no_normalize,
                window_size=effective_window_size,
                window_step=effective_window_step,
                n_trees=args.n_trees,
                sample_size=args.sample_size,
            )
            best_viz_dirs[metric] = viz_dir
            best_by_metric[metric] = details

        best_overall = max(best_by_metric.values(), key=lambda row: score_result(row, args.best_by))

    summary = {
        "dataset": dataset_name,
        "train_file": str(train_file),
        "test_file": str(test_file),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "auto_k": not args.no_auto_k,
        "k_range": {"min": args.k_min, "max": args.k_max},
        "effective_k_range": {"min": effective_k_min, "max": effective_k_max},
        "ranking_mode": args.best_by,
        "comparison_table": str(csv_path),
        "best_visualization_dirs": {metric: str(path) for metric, path in best_viz_dirs.items()},
        "best_overall": {
            "metric": best_overall["metric"],
            "k": int(best_overall["k"]),
            "nmi": float(best_overall["nmi"]),
            "ari": float(best_overall["ari"]),
            "score": score_result(best_overall, args.best_by),
            "runtime_sec": float(best_overall["runtime_sec"]),
        },
        "best_by_metric": {
            metric: {
                "k": int(best_by_metric[metric]["k"]),
                "nmi": float(best_by_metric[metric]["nmi"]),
                "ari": float(best_by_metric[metric]["ari"]),
                "score": score_result(best_by_metric[metric], args.best_by),
                "runtime_sec": float(best_by_metric[metric]["runtime_sec"]),
            }
            for metric in metrics
        },
    }
    write_json(summary, json_path)

    print()
    print("=" * 78)
    print("Best overall run")
    print("=" * 78)
    print(
        f"metric={summary['best_overall']['metric']} k={summary['best_overall']['k']} "
        f"NMI={summary['best_overall']['nmi']:.4f} ARI={summary['best_overall']['ari']:.4f} "
        f"score={summary['best_overall']['score']:.4f}"
    )
    print(f"Comparison table saved to: {csv_path}")
    print(f"Summary saved to: {json_path}")
    for metric, path in best_viz_dirs.items():
        print(f"Best {metric} visualizations saved to: {path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
