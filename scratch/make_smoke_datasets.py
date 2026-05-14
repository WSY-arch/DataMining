"""
Generate minimal synthetic UCR-format datasets for smoke testing.
Each dataset gets a TRAIN and TEST file under datasets/<name>/.
Run once before running benchmark smoke tests:
    python scratch/make_smoke_datasets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = PROJECT_ROOT / "datasets"

# Dataset specs: (name, series_length, n_classes, n_per_class_train, n_per_class_test)
SMOKE_DATASETS = [
    ("SyntheticControl", 60, 6, 8, 4),
    ("CBF", 128, 3, 8, 4),
    ("Chinatown", 24, 2, 10, 5),
    ("ECG200", 96, 2, 10, 5),
    ("GunPoint", 150, 2, 10, 5),
]


def _make_series(length: int, class_id: int, rng: np.random.Generator) -> np.ndarray:
    """Synthetic series: sine base + class offset + noise."""
    t = np.linspace(0, 2 * np.pi, length)
    base = np.sin(t + class_id * (np.pi / 3))
    return base + rng.normal(0, 0.1, length)


def write_ucr_file(path: Path, series: list[np.ndarray], labels: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for label, s in zip(labels, series):
            row = str(label) + " " + " ".join(f"{v:.6f}" for v in s)
            fh.write(row + "\n")


def main() -> None:
    rng = np.random.default_rng(0)
    for name, length, n_classes, n_train, n_test in SMOKE_DATASETS:
        for split, n_per_class in [("TRAIN", n_train), ("TEST", n_test)]:
            series, labels = [], []
            for cid in range(n_classes):
                for _ in range(n_per_class):
                    series.append(_make_series(length, cid, rng))
                    labels.append(cid + 1)           # UCR labels are 1-indexed strings
            path = DATASETS_ROOT / name / f"{name}_{split}.txt"
            write_ucr_file(path, series, labels)
            print(f"  wrote {path}  ({len(labels)} samples)")
    print(f"\n[OK] Synthetic UCR datasets written to {DATASETS_ROOT}")


if __name__ == "__main__":
    main()
