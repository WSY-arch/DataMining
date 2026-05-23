from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = [
    ("ari", "ARI"),
    ("nmi", "NMI"),
    ("runtime", "Runtime (s)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and visualize SBD vs IDK benchmark results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/auto_ucr/collaboration_results_sbd_idk_all.csv"),
        help="Path to the detailed CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/auto_ucr"),
        help="Directory for generated summary CSV and figure.",
    )
    parser.add_argument(
        "--figure-name",
        default="collaboration_results_sbd_idk_all_metrics.png",
        help="File name for the saved figure.",
    )
    parser.add_argument(
        "--summary-name",
        default="collaboration_results_sbd_idk_all_summary.csv",
        help="File name for the aggregated CSV summary.",
    )
    return parser.parse_args()


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["dataset", "measure"])[["ari", "nmi", "runtime"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "dataset",
        "measure",
        "ari_mean",
        "ari_std",
        "nmi_mean",
        "nmi_std",
        "runtime_mean",
        "runtime_std",
    ]
    return summary


def build_order(summary: pd.DataFrame) -> list[str]:
    pivot = summary.pivot(index="dataset", columns="measure", values="ari_mean")
    if {"sbd", "idk"}.issubset(pivot.columns):
        order = (
            ((pivot["sbd"] + pivot["idk"]) / 2.0)
            .sort_values(ascending=False)
            .index.tolist()
        )
    else:
        order = sorted(summary["dataset"].unique())
    return order


def plot_metric(
    ax: plt.Axes,
    summary: pd.DataFrame,
    metric: str,
    title: str,
    order: list[str],
) -> None:
    measure_styles = {
        "sbd": {"color": "#1f77b4", "label": "SBD"},
        "idk": {"color": "#ff7f0e", "label": "IDK"},
    }
    metric_mean = f"{metric}_mean"
    metric_std = f"{metric}_std"
    y = range(len(order))
    bar_height = 0.34
    offsets = {"sbd": -bar_height / 2.0, "idk": bar_height / 2.0}

    for measure, style in measure_styles.items():
        values = (
            summary[summary["measure"] == measure]
            .set_index("dataset")
            .reindex(order)
        )
        ax.barh(
            [pos + offsets[measure] for pos in y],
            values[metric_mean],
            height=bar_height,
            xerr=values[metric_std],
            color=style["color"],
            alpha=0.88,
            label=style["label"],
            capsize=3,
            error_kw={"elinewidth": 0.8},
        )

    ax.set_title(title)
    ax.set_yticks(list(y))
    ax.set_yticklabels(order)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.legend(loc="lower right", frameon=True)


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    summary = aggregate_results(df)
    summary = summary.sort_values(["dataset", "measure"]).reset_index(drop=True)

    summary_path = output_dir / args.summary_name
    summary.to_csv(summary_path, index=False)

    order = build_order(summary)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(21, max(8, 0.45 * len(order) + 2)),
        constrained_layout=True,
    )

    for ax, (metric, label) in zip(axes, METRICS, strict=True):
        plot_metric(ax, summary, metric, label, order)
        if metric in {"ari", "nmi"}:
            spread = summary[f"{metric}_std"].fillna(0)
            left = min(0.0, float(summary[f"{metric}_mean"].min() - spread.max() - 0.02))
            ax.set_xlim(left=left, right=1.05)
        else:
            ax.set_xscale("log")
            ax.set_xlim(left=max(1e-3, float(summary[f"{metric}_mean"].min() * 0.7)))

    axes[0].set_ylabel("Dataset")
    fig.suptitle("SBD vs IDK on selected UCR datasets: mean ± std", fontsize=16)

    figure_path = output_dir / args.figure_name
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    overall = df.groupby("measure")[ ["ari", "nmi", "runtime"] ].agg(["mean", "std"]).round(6)
    print(f"Saved summary CSV: {summary_path}")
    print(f"Saved figure: {figure_path}")
    print("\nOverall statistics:")
    print(overall.to_string())
    print("\nDataset order used in the figure:")
    print(json.dumps(order, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())