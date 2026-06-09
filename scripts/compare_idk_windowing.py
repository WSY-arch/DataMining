"""Compare sliding-window IDK against direct IDK on multiple UCR datasets.

This script runs IDK only, for a small set of datasets, with multiple window
sizes and a direct/raw-series baseline. It writes a detailed CSV, a summary CSV,
and one comparison plot per dataset.

Default experiment design:
- Datasets: ECG200, Plane, Trace
- Seeds: 1, 2, 3
- Sliding window fractions: 0.05, 0.10, 0.20
- Direct IDK: no_window_threshold = series_length
- Metric backend: aeon (for consistency with the current SBD/IDK setup)

The resulting plots show mean ± std over seeds, with the mean connected as a
line so the effect of window size is easy to read.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.chen_experiment_utils import (
    DEFAULT_AEON_DATA_ROOT,
    balanced_subsample,
    load_dataset,
    run_single_measure,
)


@dataclass(frozen=True)
class Setting:
    label: str
    window_size: int
    window_step: int
    no_window_threshold: int
    kind: str
    window_fraction: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["ECG200", "Plane", "Trace"],
        help="Datasets to compare (default: ECG200 Plane Trace).",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=[1, 2, 3],
        help="Random seeds to repeat (default: 1 2 3).",
    )
    parser.add_argument(
        "--window-fractions",
        nargs="*",
        type=float,
        default=[0.05, 0.10, 0.20],
        help="Sliding window sizes as fractions of series length (default: 0.05 0.10 0.20).",
    )
    parser.add_argument(
        "--idk-preset",
        choices=["auto", "fast", "balanced", "accurate"],
        default="accurate",
        help="IDK preset used for n_trees/sample_size defaults.",
    )
    parser.add_argument(
        "--metric-backend",
        choices=["auto", "reference", "aeon", "tslearn"],
        default="aeon",
        help="Backend hint passed into IDK params for consistency.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=0,
        help="Balanced samples per class cap; 0 means use the full dataset.",
    )
    parser.add_argument(
        "--data-source",
        choices=["aeon", "files"],
        default="aeon",
        help="Load datasets through aeon cache/download or local files.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Optional dataset root override.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/merged/idk_windowing"),
        help="Directory for CSV and plots.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    return parser.parse_args()


def derive_idk_defaults(series_length: int, n_samples: int, preset: str) -> tuple[int, int]:
    preset = preset.lower().strip()
    if preset == "fast":
        n_trees = 120
        sample_size = min(n_samples, 256)
    elif preset == "balanced":
        n_trees = 160
        sample_size = min(n_samples, 256)
    elif preset == "accurate":
        n_trees = 220
        sample_size = n_samples
    else:  # auto
        if n_samples >= 1000 or series_length >= 500:
            n_trees = 160
            sample_size = min(n_samples, 256)
        elif n_samples <= 120 and series_length <= 300:
            n_trees = 220
            sample_size = n_samples
        else:
            n_trees = 160
            sample_size = min(n_samples, 256)
    return n_trees, max(8, int(sample_size))


def build_settings(series_length: int, window_fractions: list[float]) -> list[Setting]:
    settings: list[Setting] = [
        Setting(
            label="direct",
            window_size=series_length,
            window_step=series_length,
            no_window_threshold=series_length,
            kind="direct",
            window_fraction=None,
        )
    ]
    for fraction in window_fractions:
        window_size = max(1, int(round(series_length * fraction)))
        window_step = max(1, int(round(window_size / 4)))
        settings.append(
            Setting(
                label=f"w={window_size} ({fraction:.2f}L)",
                window_size=window_size,
                window_step=window_step,
                no_window_threshold=0,
                kind="sliding",
                window_fraction=float(fraction),
            )
        )
    return settings


def run_one_setting(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    seed: int,
    setting: Setting,
    n_trees: int,
    sample_size: int,
    backend: str,
) -> dict[str, object]:
    row = run_single_measure(
        X,
        y,
        dataset_name=dataset_name,
        measure="idk",
        clustering_seed=seed,
        subsample_seed=42,
        perturbation_type=setting.kind,
        perturbation_level=setting.label,
        similarity_params={
            "backend": backend,
            "window_size": setting.window_size,
            "window_step": setting.window_step,
            "no_window_threshold": setting.no_window_threshold,
            "n_trees": n_trees,
            "sample_size": sample_size,
        },
        n_original=int(X.shape[0]),
    )
    row["seed"] = int(seed)
    row["window_label"] = setting.label
    row["window_size"] = setting.window_size
    row["window_step"] = setting.window_step
    row["window_fraction"] = "" if setting.window_fraction is None else setting.window_fraction
    row["mode"] = setting.kind
    return row


def summarize_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["dataset", "mode", "window_label", "window_size", "window_step"], dropna=False)
        .agg(
            ari_mean=("ari", "mean"),
            ari_std=("ari", "std"),
            nmi_mean=("nmi", "mean"),
            nmi_std=("nmi", "std"),
            runtime_mean=("runtime", "mean"),
            runtime_std=("runtime", "std"),
            seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    grouped["ari_std"] = grouped["ari_std"].fillna(0.0)
    grouped["nmi_std"] = grouped["nmi_std"].fillna(0.0)
    grouped["runtime_std"] = grouped["runtime_std"].fillna(0.0)
    return grouped


def plot_dataset(summary: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    sub = summary[summary["dataset"] == dataset].copy()
    if sub.empty:
        return

    # Separate direct and sliding window data
    direct_df = sub[sub["mode"] == "direct"]
    sliding_df = sub[sub["mode"] == "sliding"]
    
    # Sort sliding window data by window size ascending
    sliding_df = sliding_df.sort_values("window_size").reset_index(drop=True)
    
    # Prepare x-axis for sliding window settings only
    x_labels = sliding_df["window_label"].tolist()
    x = np.arange(len(x_labels))

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    metrics = [("ari", "ARI"), ("nmi", "NMI")]
    colors = {"sliding": "#1f77b4"}

    for ax, (metric, label) in zip(axes, metrics, strict=True):
        # Plot sliding window results with error bars
        if not sliding_df.empty:
            means = sliding_df[f"{metric}_mean"].to_numpy(dtype=float)
            stds = sliding_df[f"{metric}_std"].to_numpy(dtype=float)
            ax.errorbar(
                x,
                means,
                yerr=stds,
                marker="o",
                linewidth=2,
                capsize=3,
                label="sliding",
                color=colors["sliding"],
            )
            ax.plot(x, means, linewidth=2, color=colors["sliding"])
        
        # Plot direct IDK as a horizontal dashed line across the entire plot
        if not direct_df.empty:
            direct_mean = float(direct_df[f"{metric}_mean"].iloc[0])
            direct_std = float(direct_df[f"{metric}_std"].iloc[0])
            # Horizontal line for mean
            ax.axhline(
                y=direct_mean,
                color="#7f7f7f",
                linestyle="--",
                linewidth=2,
                label="direct",
            )
            # Optional: Add shaded region for std
            ax.fill_between(
                x,
                direct_mean - direct_std,
                direct_mean + direct_std,
                color="#7f7f7f",
                alpha=0.1,
            )
        
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(title="Mode")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(x_labels, rotation=20, ha="right")
    axes[-1].set_xlabel("Sliding window size (fraction of series length)")
    fig.suptitle(f"{dataset}: direct IDK vs sliding-window IDK", fontsize=15)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{dataset}_idk_windowing_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root) if args.data_root else DEFAULT_AEON_DATA_ROOT
    samples_per_class = None if args.samples_per_class <= 0 else int(args.samples_per_class)

    detailed_rows: list[dict[str, object]] = []
    for dataset_name in args.datasets:
        X, y = load_dataset(dataset_name, data_source=args.data_source, data_root=data_root)
        if samples_per_class is not None:
            X, y = balanced_subsample(X, y, samples_per_class=samples_per_class, seed=42)
        series_length = int(X.shape[1])
        n_samples = int(X.shape[0])
        n_trees, sample_size = derive_idk_defaults(series_length, n_samples, args.idk_preset)
        settings = build_settings(series_length, args.window_fractions)

        print(f"[DATASET] {dataset_name}: X={X.shape}, n_trees={n_trees}, sample_size={sample_size}")
        for seed in args.seeds:
            for setting in settings:
                row = run_one_setting(
                    X=X,
                    y=y,
                    dataset_name=dataset_name,
                    seed=int(seed),
                    setting=setting,
                    n_trees=n_trees,
                    sample_size=sample_size,
                    backend=args.metric_backend,
                )
                detailed_rows.append(row)
                print(
                    f"  seed={seed:<2} mode={setting.kind:<7} {setting.label:<16} "
                    f"ARI={row['ari']:.4f} NMI={row['nmi']:.4f} runtime={row['runtime']:.2f}s"
                )

    detailed_df = pd.DataFrame(detailed_rows)
    detailed_csv = output_dir / "idk_windowing_detailed.csv"
    detailed_df.to_csv(detailed_csv, index=False)

    summary_df = summarize_rows(detailed_rows)
    summary_csv = output_dir / "idk_windowing_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    if not args.no_plots:
        for dataset_name in args.datasets:
            plot_dataset(summary_df, dataset_name, output_dir)

    print(f"[SAVED] {detailed_csv}")
    print(f"[SAVED] {summary_csv}")
    if not args.no_plots:
        print(f"[SAVED] plots under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
