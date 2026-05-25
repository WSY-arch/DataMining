"""Migrate old 17-field Part 1 CSVs (A/B/C batches) to v2 16-field schema.

Mapping rules
-------------
- drop:    runtime_distance_matrix, runtime_clustering, n_classes_predicted
- rename:  n_samples -> n_sampled
- add:     n_original = n_sampled  (A-C ran on the full dataset, no subsample)
- add:     clustering_params = default JSON
- remap:   subsample_seed: 1 -> 0  (sentinel = "no subsample path was taken")

Usage
-----
    python -m scripts.migrate_old_csv_to_v2

Idempotent: a CSV that already has v2 columns is left untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "chen"

OLD_BATCHES = ["A", "B", "C"]

V2_COLUMNS = [
    "dataset",
    "measure",
    "paradigm",
    "ari",
    "nmi",
    "runtime",
    "subsample_seed",
    "clustering_seed",
    "perturbation_type",
    "perturbation_level",
    "n_original",
    "n_sampled",
    "series_length",
    "k",
    "measure_params",
    "clustering_params",
]

DEFAULT_CLUSTERING_PARAMS = json.dumps(
    {"init": "random", "max_iter": 300, "method": "alternate"}, separators=(",", ":")
)

DROP_COLS = ["runtime_distance_matrix", "runtime_clustering", "n_classes_predicted"]


def migrate_one(path: Path) -> tuple[bool, str]:
    df = pd.read_csv(path)

    # Already v2? bail out.
    if "n_original" in df.columns and "clustering_params" in df.columns:
        return False, "already v2"

    # 1) drop unused runtime breakdown + predicted-class diagnostic
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # 2) rename n_samples -> n_sampled
    if "n_samples" in df.columns:
        df = df.rename(columns={"n_samples": "n_sampled"})

    # 3) add n_original = n_sampled (A-C never went through subsample path)
    df["n_original"] = df["n_sampled"]

    # 4) fill clustering_params with the default training config
    df["clustering_params"] = DEFAULT_CLUSTERING_PARAMS

    # 5) remap subsample_seed: 1 (dead value) -> 0 (sentinel)
    if "subsample_seed" in df.columns:
        df["subsample_seed"] = df["subsample_seed"].replace({1: 0})

    # 6) reorder to canonical v2 column order
    missing = [c for c in V2_COLUMNS if c not in df.columns]
    if missing:
        return False, f"missing columns after migration: {missing}"
    df = df[V2_COLUMNS]

    df.to_csv(path, index=False)
    return True, f"{len(df)} rows migrated"


def main() -> None:
    print(f"results dir: {RESULTS_DIR}")
    for tag in OLD_BATCHES:
        path = RESULTS_DIR / f"part1_batch{tag}.csv"
        if not path.exists():
            print(f"  [skip] {path.name}: not found")
            continue
        changed, msg = migrate_one(path)
        flag = "[ok]" if changed else "[--]"
        print(f"  {flag} {path.name}: {msg}")


if __name__ == "__main__":
    main()
