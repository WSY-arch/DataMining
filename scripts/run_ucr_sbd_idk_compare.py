"""Run collaboration-ready SBD vs IDK experiments on a named UCR dataset.

Examples
--------
Benchmark mode (collaboration default, k = ground-truth class count):
  python scripts/run_ucr_sbd_idk_compare.py GunPoint --mode benchmark --seeds 1 2 3

Explore mode (search k and keep best per metric/seed):
  python scripts/run_ucr_sbd_idk_compare.py GunPoint --mode explore --seeds 1 2 3

Outputs under results/auto_ucr/<dataset>/:
- comparison_table_sbd_idk.csv          (detailed sweep rows)
- collaboration_results_sbd_idk.csv     (shared schema rows)
- summary_sbd_idk.json
- best_sbd_run/k_<k>/ and best_idk_run/k_<k>/ visualizations
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np

# Force non-interactive plotting so figures are saved but never shown.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
# Additionally, silence the specific loky warning about falling back to logical cores.
warnings.filterwarnings(
    "ignore",
    message=r"Returning the number of logical cores instead. You can silence this warning by setting LOKY_MAX_CPU_COUNT",
)
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

plt.show = lambda *args, **kwargs: None

# Make the project package importable when running from the code root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_ucr_clustering import load_ucr_from_file, test_ucr_dataset  # noqa: E402


DEFAULT_DATA_ROOT = Path(__file__).parent.parent / "data" / "Univariate_arff"
DEFAULT_RESULTS_ROOT = Path(__file__).parent.parent / "results" / "auto_ucr"

MEASURE_PARADIGMS = {
    "sbd": "sliding",
    "idk": "distributional",
}

RESULT_FIELDS = [
    "dataset",
    "measure",
    "paradigm",
    "measure_params",
    "clustering_params",
    "ari",
    "nmi",
    "runtime",
    "seed",
    "perturbation_type",
    "perturbation_level",
    "n_samples",
    "series_length",
    "k",
]


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


def dump_params(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


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
        k_min = max(2, unique_classes - lower_band)
        k_max = max(k_min, unique_classes + upper_band)

    if k_min > k_max:
        k_min = max(2, min(k_min, unique_classes))
        k_max = max(k_min, unique_classes)

    return list(range(k_min, k_max + 1)), k_min, k_max


def derive_window_params(
    series_length: int,
    window_size: int | None,
    window_step: int | None,
    n_series: int | None = None,
    max_total_windows: int = 50000,
) -> tuple[int, int]:
    """Choose conservative sliding-window params based on series length.

    Recommendation: window_size in [0.05*T, 0.2*T], window_step in [w/4, w/2].
    We keep the window size conservative, but use a smaller stride (about w/4)
    to preserve more local alignment information for higher quality embeddings.
    """
    T = int(series_length)

    if window_size is None:
        # prefer 10% of series length, but at least 1 and at most 20% of T
        w = max(1, int(round(T * 0.10)))
        w = max(w, 10) if T >= 10 else max(1, w)
        max_w = max(1, int(round(T * 0.20)))
        w = min(w, max_w, T)
    else:
        w = int(window_size)
        w = max(1, min(w, T))

    # Default step: use a smaller stride (w/4) for better coverage / quality.
    if window_step is None:
        step = max(1, int(max(1, w // 4)))
    else:
        step = int(window_step)
        step = max(1, min(step, w))

    # If we know how many series will be processed, limit the total number of
    # windows across the dataset. If the estimated total exceeds
    # `max_total_windows`, increase `step` to reduce overlap accordingly.
    if n_series is not None and max_total_windows is not None and max_total_windows > 0:
        # approximate windows per series as ceil(T / step)
        import math

        est_windows_per_series = max(1, math.ceil(series_length / step))
        total_windows = int(n_series) * int(est_windows_per_series)
        if total_windows > int(max_total_windows):
            # required step to bring total_windows <= max_total_windows
            required_step = math.ceil((n_series * series_length) / max_total_windows)
            step = max(step, required_step)
            step = max(1, min(step, w))

    return w, step


def derive_idk_params(
    series_length: int,
    n_samples: int,
    preset: str,
    window_size: int | None,
    window_step: int | None,
    n_trees: int | None,
    sample_size: int | None,
    no_window_threshold: int = 0,
) -> tuple[int, int, int, int, str]:
    """Return effective IDK params using preset, overridden by explicit args."""
    preset = preset.lower().strip()

    if preset not in {"auto", "fast", "balanced", "accurate"}:
        raise ValueError("idk preset must be one of: auto, fast, balanced, accurate")

    effective_preset = preset
    if preset == "auto":
        if n_samples >= 1000 or series_length >= 500:
            effective_preset = "balanced"
        elif n_samples <= 120 and series_length <= 300:
            effective_preset = "accurate"
        else:
            effective_preset = "balanced"

    if effective_preset == "fast":
        base_window_size = max(12, min(series_length // 8, 64))
        base_window_step = max(1, base_window_size // 4)
        base_n_trees = 120
        base_sample_size = 256
    elif effective_preset == "accurate":
        base_window_size = max(16, min(series_length // 5, 80))
        base_window_step = max(1, base_window_size // 4)
        base_n_trees = 220
        base_sample_size = 384
    else:  # balanced
        base_window_size = max(12, min(series_length // 6, 72))
        base_window_step = max(1, base_window_size // 4)
        base_n_trees = 160
        base_sample_size = 256

    effective_window_size = int(window_size) if window_size is not None else int(base_window_size)
    effective_window_size = max(1, min(effective_window_size, series_length))

    effective_window_step = int(window_step) if window_step is not None else int(base_window_step)
    effective_window_step = max(1, effective_window_step)

    # For short series, skip sliding windows entirely and use the full series
    # as a single window.
    if int(no_window_threshold) > 0 and series_length <= int(no_window_threshold):
        effective_window_size = int(series_length)
        effective_window_step = int(series_length)

    effective_n_trees = int(n_trees) if n_trees is not None else int(base_n_trees)
    effective_n_trees = max(10, effective_n_trees)

    # If the user explicitly provided a sample_size, use it. Otherwise prefer
    # to use the actual number of series available for this dataset (`n_samples`).
    # Fall back to the preset base if `n_samples` is not provided.
    if sample_size is not None:
        effective_sample_size = int(sample_size)
    else:
        effective_sample_size = int(n_samples) if n_samples is not None else int(base_sample_size)
    effective_sample_size = max(8, effective_sample_size)

    return (
        effective_window_size,
        effective_window_step,
        effective_n_trees,
        effective_sample_size,
        effective_preset,
    )


def run_one(
    train_file: Path,
    test_file: Path,
    metric: str,
    seed: int,
    k: int,
    n_samples: int | None,
    normalize: bool,
    idk_window_size: int,
    idk_window_step: int,
    idk_n_trees: int,
    idk_sample_size: int,
    idk_no_window_threshold: int,
    samples_per_class: int | None,
    sbd_backend: str,
    sbd_n_jobs: int,
    sbd_candidate_k: int,
    sbd_coarse_method: str,
    sbd_paa_segments: int,
) -> Dict[str, float]:
    similarity_params = (
        {
            "backend": sbd_backend,
            "n_jobs": sbd_n_jobs,
            "candidate_k": sbd_candidate_k,
            "coarse_method": sbd_coarse_method,
            "paa_segments": sbd_paa_segments,
        }
        if metric == "sbd"
        else None
    )
    window_size = idk_window_size if metric == "idk" else None
    window_step = idk_window_step if metric == "idk" else None
    n_trees = idk_n_trees if metric == "idk" else 50
    sample_size = idk_sample_size if metric == "idk" else 32
    if metric == "sbd":
        measure_params = {
            "backend": sbd_backend,
            "n_jobs": sbd_n_jobs,
            "standardize": not normalize,
            "candidate_k": sbd_candidate_k,
            "coarse_method": sbd_coarse_method,
            "paa_segments": sbd_paa_segments,
        }
    else:
        measure_params = {
            "psi": idk_sample_size,
            "t": idk_n_trees,
            "window_size": idk_window_size,
            "window_step": idk_window_step,
            "no_window_threshold": idk_no_window_threshold,
        }

    clustering_params = {
        "k": int(k),
        "normalize": bool(normalize),
        "random_state": int(seed),
        "n_samples": None if n_samples is None else int(n_samples),
        "samples_per_class": None if samples_per_class is None else int(samples_per_class),
        "no_window_threshold": int(idk_no_window_threshold),
    }

    buffer = StringIO()
    with redirect_stdout(buffer):
        ok, details = test_ucr_dataset(
            train_file=str(train_file),
            test_file=str(test_file),
            n_samples=n_samples,
            normalize=normalize,
            k=int(k),
            generate_viz=False,
            similarity_metric=metric,
            window_size=window_size,
            window_step=window_step,
            no_window_threshold=idk_no_window_threshold,
            samples_per_class=samples_per_class,
            n_trees=n_trees,
            sample_size=sample_size,
            random_state=int(seed),
            similarity_params=similarity_params,
            return_details=True,
        )
    if not ok:
        raise RuntimeError(f"Experiment failed for metric={metric}, seed={seed}, k={k}")

    details = dict(details)
    details["k"] = int(k)
    details["seed"] = int(seed)
    details["score"] = float(details["nmi"] + details["ari"])
    details["measure_params"] = dump_params(measure_params)
    details["clustering_params"] = dump_params(clustering_params)
    return details


def rerun_best_with_visualization(
    train_file: Path,
    test_file: Path,
    dataset_output_dir: Path,
    best_row: Dict[str, float],
    n_samples: int | None,
    normalize: bool,
    idk_window_size: int,
    idk_window_step: int,
    idk_n_trees: int,
    idk_sample_size: int,
    idk_no_window_threshold: int,
    samples_per_class: int | None,
    sbd_backend: str,
    sbd_n_jobs: int,
    sbd_candidate_k: int,
    sbd_coarse_method: str,
    sbd_paa_segments: int,
) -> Tuple[Path, Dict[str, float]]:
    metric = str(best_row["metric"])
    seed = int(best_row["seed"])
    k = int(best_row["k"])

    viz_dir = dataset_output_dir / f"best_{metric}_run" / f"k_{k}"

    similarity_params = (
        {
            "backend": sbd_backend,
            "n_jobs": sbd_n_jobs,
            "candidate_k": sbd_candidate_k,
            "coarse_method": sbd_coarse_method,
            "paa_segments": sbd_paa_segments,
        }
        if metric == "sbd"
        else None
    )
    window_size = idk_window_size if metric == "idk" else None
    window_step = idk_window_step if metric == "idk" else None
    n_trees = idk_n_trees if metric == "idk" else 50
    sample_size = idk_sample_size if metric == "idk" else 32

    buffer = StringIO()
    with redirect_stdout(buffer):
        ok, details = test_ucr_dataset(
            train_file=str(train_file),
            test_file=str(test_file),
            n_samples=n_samples,
            normalize=normalize,
            k=k,
            generate_viz=True,
            viz_dir=str(viz_dir),
            similarity_metric=metric,
            window_size=window_size,
            window_step=window_step,
            no_window_threshold=idk_no_window_threshold,
            samples_per_class=samples_per_class,
            n_trees=n_trees,
            sample_size=sample_size,
            random_state=seed,
            similarity_params=similarity_params,
            return_details=True,
        )

    if not ok:
        raise RuntimeError(f"Visualization rerun failed for metric={metric}, seed={seed}, k={k}")

    details = dict(details)
    details.update({"metric": metric, "seed": seed, "k": k, "score": float(best_row["score"])})
    if "measure_params" not in details:
        if metric == "sbd":
            details["measure_params"] = dump_params({
                "backend": sbd_backend,
                "n_jobs": sbd_n_jobs,
                "standardize": not normalize,
                "candidate_k": sbd_candidate_k,
                "coarse_method": sbd_coarse_method,
                "paa_segments": sbd_paa_segments,
            })
        else:
            details["measure_params"] = dump_params({
                "psi": idk_sample_size,
                "t": idk_n_trees,
                "window_size": idk_window_size,
                "window_step": idk_window_step,
                "no_window_threshold": idk_no_window_threshold,
            })
    if "clustering_params" not in details:
        details["clustering_params"] = dump_params({
            "k": k,
            "normalize": bool(normalize),
            "random_state": seed,
            "n_samples": None if n_samples is None else int(n_samples),
            "samples_per_class": None if samples_per_class is None else int(samples_per_class),
            "no_window_threshold": int(idk_no_window_threshold),
        })
    return viz_dir, details


def write_detailed_csv(rows: List[Dict[str, float]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "metric",
        "seed",
        "k",
        "measure_params",
        "clustering_params",
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


def to_collaboration_row(dataset: str, row: Dict[str, float]) -> Dict[str, object]:
    metric = str(row["metric"]).lower().strip()
    return {
        "dataset": dataset,
        "measure": metric,
        "paradigm": MEASURE_PARADIGMS[metric],
        "measure_params": row.get("measure_params", ""),
        "clustering_params": row.get("clustering_params", ""),
        "ari": float(row["ari"]),
        "nmi": float(row["nmi"]),
        "runtime": float(row["runtime_sec"]),
        "seed": int(row["seed"]),
        "perturbation_type": "none",
        "perturbation_level": 0,
        "n_samples": int(row["n_samples"]),
        "series_length": int(row["series_length"]),
        "k": int(row["k"]),
    }


def write_collaboration_csv(rows: List[Dict[str, object]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in RESULT_FIELDS})


def write_json(payload: Dict[str, object], json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def summarize_metric(rows: List[Dict[str, float]]) -> Dict[str, float]:
    nmi = np.array([float(r["nmi"]) for r in rows], dtype=float)
    ari = np.array([float(r["ari"]) for r in rows], dtype=float)
    runtime = np.array([float(r["runtime_sec"]) for r in rows], dtype=float)
    return {
        "n_runs": int(len(rows)),
        "nmi_mean": float(np.mean(nmi)),
        "nmi_std": float(np.std(nmi)),
        "ari_mean": float(np.mean(ari)),
        "ari_std": float(np.std(ari)),
        "runtime_mean": float(np.mean(runtime)),
        "runtime_std": float(np.std(runtime)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collaboration-ready SBD vs IDK benchmark/explore runner for UCR datasets.",
    )
    parser.add_argument("dataset", type=str, help="UCR dataset name, e.g. GunPoint")
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT), help="Root directory of UCR data")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_RESULTS_ROOT), help="Root directory for saved results")

    parser.add_argument("--mode", type=str, choices=["benchmark", "explore"], default="benchmark", help="benchmark: k=ground truth; explore: sweep k and keep best per seed/metric")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1], help="Random seeds, e.g. --seeds 1 2 3 4 5")

    parser.add_argument("--k-min", type=int, default=2, help="Minimum k for explore mode when --no-auto-k is set")
    parser.add_argument("--k-max", type=int, default=10, help="Maximum k for explore mode when --no-auto-k is set")
    parser.add_argument("--no-auto-k", action="store_true", help="In explore mode, disable n±band range and use --k-min/--k-max")
    parser.add_argument("--k-band-low", type=int, default=2, help="In explore mode auto-k, low band from n_classes")
    parser.add_argument("--k-band-high", type=int, default=2, help="In explore mode auto-k, high band from n_classes")
    parser.add_argument("--best-by", type=str, choices=["composite", "nmi", "ari"], default="composite", help="Ranking rule when selecting best k in explore mode")

    parser.add_argument("--n-samples", type=int, default=None, help="Optional sample limit")
    parser.add_argument("--samples-per-class", type=int, default=None, help="Optional per-class sampling cap for balanced subsampling")
    parser.add_argument("--no-normalize", action="store_true", help="Disable z-score normalization")
    parser.add_argument("--no-viz", action="store_true", help="Skip best-run visualization rerun")

    parser.add_argument("--idk-preset", type=str, default="auto", choices=["auto", "fast", "balanced", "accurate"], help="Preset for IDK hyperparameters")
    parser.add_argument("--window-size", type=int, default=None, help="IDK sliding window size (override preset)")
    parser.add_argument("--window-step", type=int, default=None, help="IDK sliding window step (override preset)")
    parser.add_argument("--n-trees", type=int, default=None, help="IDK number of trees (override preset)")
    parser.add_argument("--sample-size", type=int, default=None, help="IDK tree sample size (override preset)")
    parser.add_argument("--idk-no-window-threshold", type=int, default=0, help="If series length is at most this value, disable IDK sliding windows by using the full series as one window (0 to disable)")

    parser.add_argument("--sbd-backend", type=str, default="auto", choices=["auto", "approx", "candidate", "pruned", "aeon", "reference"], help="SBD backend")
    parser.add_argument("--sbd-n-jobs", type=int, default=-1, help="Parallel jobs for aeon SBD backend (-1 uses all cores)")
    parser.add_argument("--sbd-candidate-k", type=int, default=20, help="Approximate SBD: candidates per series from the coarse search")
    parser.add_argument("--sbd-coarse-method", type=str, default="paa", choices=["paa"], help="Approximate SBD coarse representation")
    parser.add_argument("--sbd-paa-segments", type=int, default=32, help="Approximate SBD: number of PAA segments")
    parser.add_argument("--idk-max-samples", type=int, default=0, help="When using IDK, subsample to at most this many series to avoid OOM (0 to disable)")
    parser.add_argument("--idk-sample-size-max", type=int, default=1000, help="Maximum `sample_size` used to build IDK trees (0 to disable)")

    args = parser.parse_args()

    dataset_name = args.dataset.strip()
    data_root = Path(args.data_root)
    output_root = Path(args.output_dir)
    seeds = [int(seed) for seed in args.seeds]

    train_file, test_file = resolve_ucr_files(dataset_name, data_root)
    preview_X, y_preview = load_ucr_from_file(str(train_file), str(test_file))

    if args.samples_per_class is not None and args.samples_per_class > 0:
        _, class_counts = np.unique(y_preview, return_counts=True)
        effective_n_samples = int(np.sum(np.minimum(class_counts, int(args.samples_per_class))))
    elif args.n_samples is not None and args.n_samples > 0:
        effective_n_samples = min(int(args.n_samples), int(preview_X.shape[0]))
    else:
        effective_n_samples = int(preview_X.shape[0])

    # Derive sensible sliding-window defaults using fractional rules recommended
    # by the user: window_size ∈ [0.05T, 0.2T], default ~0.1T; step ∈ [w/4, w/2].
    # Also cap the total number of windows across the dataset to avoid OOM by
    # increasing step when needed. CLI overrides (--window-size/--window-step)
    # remain available.
    effective_window_size, effective_window_step = derive_window_params(
        series_length=int(preview_X.shape[1]),
        window_size=args.window_size,
        window_step=args.window_step,
        n_series=effective_n_samples,
        max_total_windows=250000,
    )

    idk_window_size, idk_window_step, idk_n_trees, idk_sample_size, effective_idk_preset = derive_idk_params(
        series_length=int(preview_X.shape[1]),
        n_samples=effective_n_samples,
        preset=args.idk_preset,
        window_size=effective_window_size,
        window_step=effective_window_step,
        n_trees=args.n_trees,
        sample_size=args.sample_size,
        no_window_threshold=args.idk_no_window_threshold,
    )

    use_direct_raw_series = bool(
        args.idk_no_window_threshold and int(preview_X.shape[1]) <= int(args.idk_no_window_threshold)
    )

    # Option B: enforce an upper bound on IDK's sample_size to limit memory.
    if args.idk_sample_size_max and args.idk_sample_size_max > 0:
        if idk_sample_size > args.idk_sample_size_max:
            print(f"[INFO] IDK: capping sample_size {idk_sample_size} -> {args.idk_sample_size_max} to limit memory")
            idk_sample_size = int(args.idk_sample_size_max)

    n_classes = int(len(np.unique(y_preview)))
    if args.mode == "benchmark":
        k_values = [n_classes]
        effective_k_min = n_classes
        effective_k_max = n_classes
    else:
        k_values, effective_k_min, effective_k_max = derive_k_values(
            y_true=y_preview,
            k_min=args.k_min,
            k_max=args.k_max,
            auto_k=not args.no_auto_k,
            lower_band=args.k_band_low,
            upper_band=args.k_band_high,
        )

    metrics = ["sbd", "idk"]

    print("=" * 88)
    print(f"Dataset: {dataset_name}")
    print(f"Mode:    {args.mode}")
    print(f"Train:   {train_file}")
    print(f"Test:    {test_file}")
    print(f"Seeds:   {seeds}")
    print(f"k range: {effective_k_min}..{effective_k_max}")
    print(f"Metrics: {', '.join(metrics)}")
    print(f"IDK preset: {args.idk_preset} -> {effective_idk_preset}")
    if use_direct_raw_series:
        print(
            f"IDK mode: direct raw-series path (length={int(preview_X.shape[1])} <= threshold={int(args.idk_no_window_threshold)})"
        )
        print(f"IDK params: n_trees={idk_n_trees}, sample_size={idk_sample_size}")
    else:
        print(
            f"IDK mode: sliding-window path (window_size={idk_window_size}, window_step={idk_window_step})"
        )
        print(
            f"IDK params: window_size={idk_window_size}, window_step={idk_window_step}, "
            f"n_trees={idk_n_trees}, sample_size={idk_sample_size}"
        )
    if args.samples_per_class is not None and args.samples_per_class > 0:
        print(f"Sampling: balanced {args.samples_per_class} per class (effective_n_samples={effective_n_samples})")
    elif args.n_samples is not None and args.n_samples > 0:
        print(f"Sampling: uniform n_samples={effective_n_samples}")
    else:
        print("Sampling: full dataset")
    print(f"SBD backend: {args.sbd_backend}")
    print("=" * 88)

    detailed_rows: List[Dict[str, float]] = []
    selected_rows: List[Dict[str, float]] = []

    for seed in seeds:
        for metric in metrics:
            # Decide per-metric sample budget: overall effective_n_samples may be reduced
            # for IDK if --idk-max-samples is set and smaller than the dataset.
            n_samples_for_metric = effective_n_samples
            if metric == "idk" and args.idk_max_samples and args.idk_max_samples > 0:
                if effective_n_samples > args.idk_max_samples:
                    n_samples_for_metric = int(args.idk_max_samples)
                    print(f"[INFO] IDK: subsampling from {effective_n_samples} -> {n_samples_for_metric} samples to limit memory")
            if args.mode == "benchmark":
                row = run_one(
                    train_file=train_file,
                    test_file=test_file,
                    metric=metric,
                    seed=seed,
                    k=n_classes,
                    n_samples=n_samples_for_metric,
                    normalize=not args.no_normalize,
                    idk_window_size=idk_window_size,
                    idk_window_step=idk_window_step,
                    idk_n_trees=idk_n_trees,
                    idk_sample_size=idk_sample_size,
                    idk_no_window_threshold=args.idk_no_window_threshold,
                    samples_per_class=args.samples_per_class,
                    sbd_backend=args.sbd_backend,
                    sbd_n_jobs=args.sbd_n_jobs,
                    sbd_candidate_k=args.sbd_candidate_k,
                    sbd_coarse_method=args.sbd_coarse_method,
                    sbd_paa_segments=args.sbd_paa_segments,
                )
                detailed_rows.append(row)
                selected_rows.append(row)
                print(
                    f"[RUN] seed={seed:<3} metric={metric:<4} k={n_classes:<2} "
                    f"NMI={row['nmi']:.4f} ARI={row['ari']:.4f} "
                    f"score={row['score']:.4f} runtime={row['runtime_sec']:.2f}s"
                )
            else:
                candidates: List[Dict[str, float]] = []
                for k in k_values:
                    row = run_one(
                        train_file=train_file,
                        test_file=test_file,
                        metric=metric,
                        seed=seed,
                        k=int(k),
                        n_samples=args.n_samples,
                        normalize=not args.no_normalize,
                        idk_window_size=idk_window_size,
                        idk_window_step=idk_window_step,
                        idk_n_trees=idk_n_trees,
                        idk_sample_size=idk_sample_size,
                        idk_no_window_threshold=args.idk_no_window_threshold,
                        samples_per_class=args.samples_per_class,
                        sbd_backend=args.sbd_backend,
                        sbd_n_jobs=args.sbd_n_jobs,
                        sbd_candidate_k=args.sbd_candidate_k,
                        sbd_coarse_method=args.sbd_coarse_method,
                        sbd_paa_segments=args.sbd_paa_segments,
                    )
                    detailed_rows.append(row)
                    candidates.append(row)
                    print(
                        f"[SWEEP] seed={seed:<3} metric={metric:<4} k={k:<2} "
                        f"NMI={row['nmi']:.4f} ARI={row['ari']:.4f} "
                        f"score={row['score']:.4f} runtime={row['runtime_sec']:.2f}s"
                    )

                best_row = max(candidates, key=lambda r: score_result(r, args.best_by))
                selected_rows.append(best_row)
                print(
                    f"[BEST ] seed={seed:<3} metric={metric:<4} k={best_row['k']:<2} "
                    f"NMI={best_row['nmi']:.4f} ARI={best_row['ari']:.4f}"
                )

    dataset_output_dir = output_root / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    detailed_csv_path = dataset_output_dir / "comparison_table_sbd_idk.csv"
    collab_csv_path = dataset_output_dir / "collaboration_results_sbd_idk.csv"
    json_path = dataset_output_dir / "summary_sbd_idk.json"

    write_detailed_csv(detailed_rows, detailed_csv_path)

    collaboration_rows = [to_collaboration_row(dataset_name, row) for row in selected_rows]
    write_collaboration_csv(collaboration_rows, collab_csv_path)

    best_by_metric: Dict[str, Dict[str, float]] = {}
    metric_stats: Dict[str, Dict[str, float]] = {}
    for metric in metrics:
        metric_rows = [row for row in selected_rows if row["metric"] == metric]
        best_by_metric[metric] = max(metric_rows, key=lambda row: score_result(row, args.best_by))
        metric_stats[metric] = summarize_metric(metric_rows)

    best_overall = max(selected_rows, key=lambda row: score_result(row, args.best_by))

    best_viz_dirs: Dict[str, Path] = {}
    if not args.no_viz:
        for metric in metrics:
            best_row = best_by_metric[metric]
            print()
            print(
                f"[VIZ ] Re-running best {metric} with visualizations: "
                f"seed={best_row['seed']}, k={best_row['k']}"
            )
            viz_dir, details = rerun_best_with_visualization(
                train_file=train_file,
                test_file=test_file,
                dataset_output_dir=dataset_output_dir,
                best_row=best_row,
                n_samples=args.n_samples,
                normalize=not args.no_normalize,
                idk_window_size=idk_window_size,
                idk_window_step=idk_window_step,
                idk_n_trees=idk_n_trees,
                idk_sample_size=idk_sample_size,
                idk_no_window_threshold=args.idk_no_window_threshold,
                samples_per_class=args.samples_per_class,
                sbd_backend=args.sbd_backend,
                sbd_n_jobs=args.sbd_n_jobs,
                sbd_candidate_k=args.sbd_candidate_k,
                sbd_coarse_method=args.sbd_coarse_method,
                sbd_paa_segments=args.sbd_paa_segments,
            )
            best_viz_dirs[metric] = viz_dir
            best_by_metric[metric] = details

        best_overall = max(best_by_metric.values(), key=lambda row: score_result(row, args.best_by))

    summary = {
        "dataset": dataset_name,
        "mode": args.mode,
        "seeds": seeds,
        "train_file": str(train_file),
        "test_file": str(test_file),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "n_classes": n_classes,
        "auto_k": (args.mode == "explore" and not args.no_auto_k),
        "k_range": {"min": args.k_min, "max": args.k_max},
        "effective_k_range": {"min": effective_k_min, "max": effective_k_max},
        "ranking_mode": args.best_by,
        "detailed_table": str(detailed_csv_path),
        "collaboration_table": str(collab_csv_path),
        "best_visualization_dirs": {metric: str(path) for metric, path in best_viz_dirs.items()},
        "idk": {
            "requested_preset": args.idk_preset,
            "effective_preset": effective_idk_preset,
            "window_size": idk_window_size,
            "window_step": idk_window_step,
            "n_trees": idk_n_trees,
            "sample_size": idk_sample_size,
        },
        "sbd": {
            "backend": args.sbd_backend,
            "n_jobs": args.sbd_n_jobs,
            "candidate_k": args.sbd_candidate_k,
            "coarse_method": args.sbd_coarse_method,
            "paa_segments": args.sbd_paa_segments,
        },
        "metric_stats": metric_stats,
        "best_overall": {
            "metric": best_overall["metric"],
            "seed": int(best_overall["seed"]),
            "k": int(best_overall["k"]),
            "nmi": float(best_overall["nmi"]),
            "ari": float(best_overall["ari"]),
            "score": score_result(best_overall, args.best_by),
            "runtime_sec": float(best_overall["runtime_sec"]),
        },
        "best_by_metric": {
            metric: {
                "seed": int(best_by_metric[metric]["seed"]),
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
    print("=" * 88)
    print("Best overall run")
    print("=" * 88)
    print(
        f"metric={summary['best_overall']['metric']} seed={summary['best_overall']['seed']} "
        f"k={summary['best_overall']['k']} NMI={summary['best_overall']['nmi']:.4f} "
        f"ARI={summary['best_overall']['ari']:.4f} score={summary['best_overall']['score']:.4f}"
    )
    print(f"Detailed table saved to: {detailed_csv_path}")
    print(f"Collaboration table saved to: {collab_csv_path}")
    print(f"Summary saved to: {json_path}")
    for metric, path in best_viz_dirs.items():
        print(f"Best {metric} visualizations saved to: {path}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
