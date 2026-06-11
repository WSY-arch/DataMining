"""Merge Wang's CSV results into v2 schema and combine with Chen's.

Usage:
    python scripts/merge_wang_results.py

Input files (from Wang's wang_week_3 branch):
    - Wang Part 1: part1_results_sbd_idk_all.csv  (old 14-col schema)
    - Wang Part 2: part2_perturbation_results_sbd_idk.csv  (v2 16-col schema)

Output:
    - results/merged/part1_all_5measures.csv   (Chen ED/DTW/MSM + Wang SBD/IDK)
    - results/merged/part2_all_5measures.csv   (Chen ED/DTW/MSM + Wang SBD/IDK)

The script handles:
    1. Wang Part 1 schema migration (14 cols → v2 16 cols)
    2. Field renaming: seed → clustering_seed, n_samples → n_sampled
    3. Inferring subsample_seed and n_original from Chen's existing results
    4. Column reordering to v2 standard
    5. Direct concatenation of Part 2 (already v2)
    6. UTF-16 encoding detection for Windows-generated CSVs
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# v2 schema field order (authoritative)
V2_FIELDS = [
    "dataset", "measure", "paradigm", "ari", "nmi", "runtime",
    "subsample_seed", "clustering_seed", "perturbation_type",
    "perturbation_level", "n_original", "n_sampled", "series_length",
    "k", "measure_params", "clustering_params",
]


def detect_encoding(path: Path) -> str:
    """Detect UTF-16 BOM vs UTF-8."""
    with open(path, "rb") as f:
        bom = f.read(2)
    if bom in (b"\xff\xfe", b"\xfe\xff"):
        return "utf-16"
    return "utf-8"


def read_csv(path: Path) -> list[dict[str, str]]:
    enc = detect_encoding(path)
    with open(path, encoding=enc, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=V2_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [OK] wrote {len(rows)} rows → {path}")


def build_dataset_meta(chen_part1_path: Path) -> dict[str, dict]:
    """Build (dataset → {n_original, subsample_seed}) lookup from Chen's results."""
    rows = read_csv(chen_part1_path)
    meta: dict[str, dict] = {}
    for r in rows:
        ds = r["dataset"]
        if ds not in meta:
            meta[ds] = {
                "n_original": r["n_original"],
                "subsample_seed": r["subsample_seed"],
            }
    return meta


def migrate_wang_part1(
    wang_rows: list[dict[str, str]],
    dataset_meta: dict[str, dict],
) -> list[dict[str, str]]:
    """Migrate Wang Part 1 from 14-col old schema to v2 16-col schema."""
    migrated = []
    for r in wang_rows:
        ds = r["dataset"]
        meta = dataset_meta.get(ds, {})

        # Infer subsample_seed and n_original
        n_sampled = r["n_samples"]
        n_original = meta.get("n_original", n_sampled)
        subsample_seed = meta.get("subsample_seed", "0")

        # Clean up clustering_params (Wang stores extra fields; normalize to standard)
        clustering_params = json.dumps(
            {"init": "random", "max_iter": 300, "method": "alternate"},
            separators=(",", ": "),
        )

        new_row = {
            "dataset": ds,
            "measure": r["measure"],
            "paradigm": r["paradigm"],
            "ari": r["ari"],
            "nmi": r["nmi"],
            "runtime": r["runtime"],
            "subsample_seed": subsample_seed,
            "clustering_seed": r["seed"],
            "perturbation_type": r["perturbation_type"],
            "perturbation_level": r["perturbation_level"],
            "n_original": n_original,
            "n_sampled": n_sampled,
            "series_length": r["series_length"],
            "k": r["k"],
            "measure_params": r["measure_params"],
            "clustering_params": clustering_params,
        }
        migrated.append(new_row)
    return migrated


def main() -> int:
    results_root = PROJECT_ROOT / "results"
    chen_part1_path = results_root / "chen" / "part1_results_all18.csv"
    chen_part2_path = results_root / "chen" / "part2_perturbation_results.csv"

    # Wang input files (checkout from wang_week_3 or place in project root)
    wang_part1_path = PROJECT_ROOT / "_tmp_wang_part1.csv"
    wang_part2_path = PROJECT_ROOT / "_tmp_wang_part2.csv"

    # If tmp files don't exist, try extracting from git
    if not wang_part1_path.exists():
        import subprocess
        print("  Extracting Wang files from origin/wang_week_3...")
        for fname, out in [
            ("part1_results_sbd_idk_all.csv", wang_part1_path),
            ("part2_perturbation_results_sbd_idk.csv", wang_part2_path),
        ]:
            subprocess.run(
                ["git", "show", f"origin/wang_week_3:{fname}"],
                stdout=open(out, "wb"),
                cwd=PROJECT_ROOT,
                check=True,
            )

    if not wang_part1_path.exists() or not wang_part2_path.exists():
        print("[ERROR] Wang CSV files not found. Run: git fetch origin wang_week_3")
        return 1

    # --- Part 1: migrate + merge ---
    print("=== Part 1 Merge ===")
    dataset_meta = build_dataset_meta(chen_part1_path)
    print(
        f"  Chen Part 1: {chen_part1_path.name} (meta for {len(dataset_meta)} datasets)")

    wang_p1_raw = read_csv(wang_part1_path)
    print(f"  Wang Part 1: {len(wang_p1_raw)} rows (old schema)")
    wang_p1_v2 = migrate_wang_part1(wang_p1_raw, dataset_meta)
    print(f"  Migrated to v2: {len(wang_p1_v2)} rows")

    chen_p1 = read_csv(chen_part1_path)
    print(f"  Chen Part 1: {len(chen_p1)} rows")

    merged_p1 = chen_p1 + wang_p1_v2
    merged_p1.sort(key=lambda r: (
        r["dataset"], r["measure"], int(r["clustering_seed"])))
    write_csv(merged_p1, results_root / "merged" / "part1_all_5measures.csv")

    # --- Part 2: direct concat ---
    print("\n=== Part 2 Merge ===")
    wang_p2 = read_csv(wang_part2_path)
    print(f"  Wang Part 2: {len(wang_p2)} rows (v2 schema)")

    chen_p2 = read_csv(chen_part2_path)
    print(f"  Chen Part 2: {len(chen_p2)} rows")

    merged_p2 = chen_p2 + wang_p2
    merged_p2.sort(key=lambda r: (
        r["dataset"], r["perturbation_type"],
        r["perturbation_level"], r["measure"],
        int(r["clustering_seed"]),
    ))
    write_csv(merged_p2, results_root / "merged" / "part2_all_5measures.csv")

    # --- Summary ---
    print("\n=== Summary ===")
    p1_measures = sorted(set(r["measure"] for r in merged_p1))
    p2_measures = sorted(set(r["measure"] for r in merged_p2))
    print(f"  Part 1: {len(merged_p1)} rows, measures={p1_measures}")
    print(f"  Part 2: {len(merged_p2)} rows, measures={p2_measures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
