"""Plot IDK windowing comparison results directly from CSV.

This script reads the summary CSV file and generates updated plots where:
- Sliding window settings are sorted by window size (ascending)
- Direct IDK is shown as a horizontal dashed gray line across the plot
- Direct is removed from x-axis for clarity
"""
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
            # Add shaded region for std
            ax.fill_between(
                x,
                max(-0.05, direct_mean - direct_std),
                min(1.05, direct_mean + direct_std),
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
    print(f"[PLOT] {dataset}_idk_windowing_comparison.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the idk_windowing_summary.csv file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/merged/idk_windowing_compare"),
        help="Output directory for plots",
    )
    args = parser.parse_args()

    # Read the summary CSV
    df = pd.read_csv(args.input)
    print(f"[READ] {len(df)} rows from {args.input}")

    # Generate plots for each dataset
    datasets = df["dataset"].unique()
    for dataset in datasets:
        plot_dataset(df, dataset, args.output_dir)

    print(f"[DONE] Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
