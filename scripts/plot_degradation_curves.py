"""Plot degradation curves (mean ± std) from a perturbation results CSV.

Reads a CSV with fields including: dataset, measure, perturbation_type, perturbation_level, ari, nmi
Produces one PNG per (dataset, perturbation_type, metric_name) showing mean±std
curves for each `measure` across perturbation levels.

Usage:
    .venv/Scripts/python.exe scripts/plot_degradation_curves.py \
        --input path/to/part2_perturbation_results.csv \
        --output-dir results/chen/perturbation_curves_meanstd
"""
from __future__ import annotations

import argparse
from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_mean_std(df: pd.DataFrame, output_dir: Path, metrics: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Ensure perturbation_level numeric
    if "perturbation_level" not in df.columns:
        raise ValueError("Input CSV missing 'perturbation_level' column")
    df = df.copy()
    # some rows may have empty strings; coerce errors to NaN then drop
    df["perturbation_level"] = pd.to_numeric(df["perturbation_level"], errors="coerce")
    df = df.dropna(subset=["perturbation_level"]).reset_index(drop=True)

    groups = sorted({(r["dataset"], r["perturbation_type"]) for _, r in df.iterrows()})
    measures = sorted(df["measure"].unique())

    for dataset, pert_type in groups:
        sub = df[(df["dataset"] == dataset) & (df["perturbation_type"] == pert_type)]
        if sub.empty:
            continue
        for metric_name in metrics:
            fig, ax = plt.subplots(figsize=(7, 4))
            plotted = 0
            for measure in measures:
                msub = sub[sub["measure"] == measure]
                if msub.empty or metric_name not in msub.columns:
                    continue
                agg = msub.groupby("perturbation_level")[metric_name].agg(["mean", "std"]).reset_index()
                agg = agg.sort_values("perturbation_level")
                if agg.empty:
                    continue
                x = agg["perturbation_level"].to_numpy(dtype=float)
                mean = agg["mean"].to_numpy(dtype=float)
                std = agg["std"].to_numpy(dtype=float)
                # If any NaNs in std (single sample), replace with 0
                std = np.nan_to_num(std, nan=0.0)
                ax.plot(x, mean, marker="o", label=measure)
                ax.fill_between(x, mean - std, mean + std, alpha=0.2)
                plotted += 1
            if plotted == 0:
                plt.close(fig)
                continue
            ax.set_title(f"{dataset} {pert_type} degradation ({metric_name.upper()})")
            ax.set_xlabel("Perturbation level")
            ax.set_ylabel(metric_name.upper())
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            out_path = output_dir / f"{dataset}_{pert_type}_{metric_name}.png"
            fig.savefig(out_path, dpi=160)
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Path to perturbation results CSV")
    p.add_argument("--output-dir", required=True, help="Directory to save PNGs")
    p.add_argument("--metrics", nargs="*", default=["ari", "nmi"], help="Metrics to plot (default: ari nmi)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return 2
    df = pd.read_csv(input_path)
    plot_mean_std(df, Path(args.output_dir), args.metrics)
    print(f"[OK] Plots saved under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
