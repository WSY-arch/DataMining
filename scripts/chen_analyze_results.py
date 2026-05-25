"""Summarize combined Part 1 results with per-dataset scores and rank statistics."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from statistics import mean, pstdev
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def aggregate_seed_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str],
                  list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["measure"],
            row.get("paradigm", ""),
            row.get("perturbation_type", "none") or "none",
            row.get("perturbation_level", "0") or "0",
        )
        grouped[key].append(row)

    aggregated: list[dict[str, object]] = []
    for (dataset, measure, paradigm, perturbation_type, perturbation_level), seed_rows in sorted(grouped.items()):
        ari_values = [float(row["ari"]) for row in seed_rows]
        nmi_values = [float(row["nmi"]) for row in seed_rows]
        runtime_values = [float(row["runtime"]) for row in seed_rows]
        first = seed_rows[0]
        # v2 schema uses n_sampled; legacy CSVs used n_samples. Read both.
        n_sampled = first.get("n_sampled") or first.get("n_samples", "")
        n_original = first.get("n_original", "")
        aggregated.append(
            {
                "dataset": dataset,
                "measure": measure,
                "paradigm": paradigm,
                "perturbation_type": perturbation_type,
                "perturbation_level": perturbation_level,
                "ari_mean": mean(ari_values),
                "ari_std": pstdev(ari_values) if len(ari_values) > 1 else 0.0,
                "nmi_mean": mean(nmi_values),
                "nmi_std": pstdev(nmi_values) if len(nmi_values) > 1 else 0.0,
                "runtime_mean": mean(runtime_values),
                "runtime_std": pstdev(runtime_values) if len(runtime_values) > 1 else 0.0,
                "seeds": len(seed_rows),
                "n_original": n_original,
                "n_sampled": n_sampled,
                "series_length": first.get("series_length", ""),
                "k": first.get("k", ""),
            }
        )
    return aggregated


def average_ranks(rows: list[dict[str, object]], score_field: str) -> list[dict[str, object]]:
    score_key = f"{score_field}_mean"
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("perturbation_type", "none") in {"", "none"}:
            grouped[str(row["dataset"])].append(row)

    rank_sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    per_dataset_rows: list[dict[str, object]] = []

    for dataset, dataset_rows in sorted(grouped.items()):
        scored = sorted(
            dataset_rows,
            key=lambda row: float(row[score_key]) if row.get(
                score_key) not in {"", None} else -math.inf,
            reverse=True,
        )
        for rank, row in enumerate(scored, start=1):
            measure = row["measure"]
            rank_sums[measure] += rank
            counts[measure] += 1
            per_dataset_rows.append(
                {
                    "dataset": dataset,
                    "measure": measure,
                    "paradigm": row.get("paradigm", ""),
                    score_key: float(row[score_key]),
                    "rank": rank,
                }
            )

    return [
        {
            "measure": measure,
            "datasets": counts[measure],
            "average_rank": rank_sums[measure] / counts[measure],
        }
        for measure in sorted(rank_sums)
    ], per_dataset_rows


def friedman_summary(rows: list[dict[str, object]], score_field: str) -> dict[str, object]:
    score_key = f"{score_field}_mean"
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    measures = sorted({row["measure"] for row in rows if row.get(
        "perturbation_type", "none") in {"", "none"}})
    for row in rows:
        if row.get("perturbation_type", "none") in {"", "none"}:
            grouped[str(row["dataset"])][str(
                row["measure"])] = float(row[score_key])

    complete_datasets = [
        dataset for dataset, scores in grouped.items() if all(measure in scores for measure in measures)
    ]
    if len(complete_datasets) < 2 or len(measures) < 3:
        return {
            "status": "skipped",
            "reason": "Need at least two complete datasets and three measures for Friedman test.",
            "complete_datasets": len(complete_datasets),
            "measures": ",".join(measures),
        }

    try:
        from scipy.stats import friedmanchisquare
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": f"scipy is unavailable: {exc}",
            "complete_datasets": len(complete_datasets),
            "measures": ",".join(measures),
        }

    samples = [[grouped[dataset][measure]
                for dataset in complete_datasets] for measure in measures]
    statistic, p_value = friedmanchisquare(*samples)
    return {
        "status": "ok",
        "score_field": score_field,
        "complete_datasets": len(complete_datasets),
        "measures": ",".join(measures),
        "friedman_statistic": statistic,
        "friedman_p_value": p_value,
        "note": "If p < 0.05, run Nemenyi post-hoc via scikit-posthocs or report average-rank/CD diagram.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="Combined result CSV using the shared schema.")
    parser.add_argument("--output-dir", default="results/chen/analysis")
    parser.add_argument("--score-field", choices=["ari", "nmi"], default="ari")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(Path(args.input))
    aggregated_rows = aggregate_seed_rows(rows)
    output_dir = Path(args.output_dir)

    rank_rows, per_dataset_rows = average_ranks(
        aggregated_rows, args.score_field)
    summary = friedman_summary(aggregated_rows, args.score_field)

    aggregate_fields = [
        "dataset",
        "measure",
        "paradigm",
        "perturbation_type",
        "perturbation_level",
        "ari_mean",
        "ari_std",
        "nmi_mean",
        "nmi_std",
        "runtime_mean",
        "runtime_std",
        "seeds",
        "n_original",
        "n_sampled",
        "series_length",
        "k",
    ]
    write_csv(output_dir / "seed_aggregated_results.csv",
              aggregated_rows, aggregate_fields)
    write_csv(output_dir / f"per_dataset_{args.score_field}_ranks.csv", per_dataset_rows, [
              "dataset", "measure", "paradigm", f"{args.score_field}_mean", "rank"])
    write_csv(output_dir / f"average_{args.score_field}_ranks.csv",
              rank_rows, ["measure", "datasets", "average_rank"])
    write_csv(
        output_dir / f"friedman_{args.score_field}.csv", [summary], list(summary.keys()))

    print(f"[OK] wrote analysis files under {output_dir}")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
