from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from tsclust.clustering import cluster_time_series


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "chen"
DEFAULT_AEON_DATA_ROOT = PROJECT_ROOT / "datasets" / "aeon"


@dataclass(frozen=True)
class DatasetMeta:
    name: str
    length: int
    n_classes: int
    n_samples: int
    domain: str
    rationale: str


SELECTED_DATASETS: tuple[DatasetMeta, ...] = (
    DatasetMeta("SyntheticControl", 60, 6, 600, "Synthetic", "Classic controlled shapes with trend and shift-like classes."),
    DatasetMeta("CBF", 128, 3, 930, "Synthetic", "Cylinder-Bell-Funnel motifs; strong shape separation."),
    DatasetMeta("TwoPatterns", 128, 4, 5000, "Synthetic", "Four synthetic patterns with timing variation."),
    DatasetMeta("ItalyPowerDemand", 24, 2, 1096, "Sensor", "Very short energy-demand series for lower length bound."),
    DatasetMeta("MoteStrain", 84, 2, 1272, "Sensor", "Short IoT sensor series with real measurement variation."),
    DatasetMeta("ECG200", 96, 2, 200, "Medical", "Canonical ECG binary dataset."),
    DatasetMeta("ECGFiveDays", 136, 2, 884, "Medical", "ECG across days with likely phase variation."),
    DatasetMeta("GunPoint", 150, 2, 200, "Motion", "Classic motion-capture benchmark with mild warping."),
    DatasetMeta("Plane", 144, 7, 210, "Sensor", "Multi-class radar-return shapes."),
    DatasetMeta("Trace", 275, 4, 200, "Synthetic", "Process-control shapes; good perturbation candidate."),
    DatasetMeta("ArrowHead", 251, 3, 211, "Shape", "Object outline series with shape-based classes."),
    DatasetMeta("Coffee", 286, 2, 56, "Spectroscopy", "Small spectroscopy dataset; non-temporal shape signal."),
    DatasetMeta("DiatomSizeReduction", 345, 4, 322, "Image", "Image-outline classes with clear morphology."),
    DatasetMeta("FaceFour", 350, 4, 112, "Image", "Small face-outline projection dataset."),
    DatasetMeta("Symbols", 398, 6, 1020, "Shape", "Handwritten symbol outlines with six classes."),
    DatasetMeta("OSULeaf", 427, 6, 442, "Shape", "Leaf outlines; longer multi-class shape benchmark."),
    DatasetMeta("Beef", 470, 5, 60, "Spectroscopy", "Small long spectroscopy dataset."),
    DatasetMeta("Mallat", 1024, 8, 2400, "Synthetic", "Long synthetic benchmark for upper length bound."),
)


MEASURE_PARADIGMS = {
    "ed": "lock-step",
    "euclidean": "lock-step",
    "dtw": "elastic",
    "msm": "elastic",
    "sbd": "sliding",
    "idk": "distributional",
}


RESULT_FIELDS = [
    "dataset",
    "measure",
    "paradigm",
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


def candidate_data_roots() -> list[Path]:
    return [
        PROJECT_ROOT / "datasets" / "Univariate_arff",
        PROJECT_ROOT / "data" / "Univariate_arff",
        PROJECT_ROOT / "datasets",
        PROJECT_ROOT / "data",
    ]


def resolve_data_root(data_root: str | None) -> Path:
    if data_root:
        return Path(data_root)
    for root in candidate_data_roots():
        if root.exists():
            return root
    return candidate_data_roots()[0]


def resolve_ucr_files(dataset_name: str, data_root: Path) -> tuple[Path, Path]:
    dataset_dir = data_root / dataset_name
    candidates = [
        (dataset_dir / f"{dataset_name}_TRAIN.txt", dataset_dir / f"{dataset_name}_TEST.txt"),
        (dataset_dir / f"{dataset_name}_TRAIN.tsv", dataset_dir / f"{dataset_name}_TEST.tsv"),
        (data_root / f"{dataset_name}_TRAIN.txt", data_root / f"{dataset_name}_TEST.txt"),
        (data_root / f"{dataset_name}_TRAIN.tsv", data_root / f"{dataset_name}_TEST.tsv"),
    ]
    for train_file, test_file in candidates:
        if train_file.exists() and test_file.exists():
            return train_file, test_file
    raise FileNotFoundError(f"Could not find TRAIN/TEST files for {dataset_name} under {data_root}")


def load_ucr_file(path: Path) -> tuple[list[np.ndarray], list[str]]:
    series: list[np.ndarray] = []
    labels: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            parts = raw_line.strip().replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                labels.append(parts[0])
                series.append(np.array([float(value) for value in parts[1:]], dtype=float))
            except ValueError:
                continue
    return series, labels


def load_ucr_dataset(train_file: Path, test_file: Path) -> tuple[np.ndarray, np.ndarray]:
    train_series, train_labels = load_ucr_file(train_file)
    test_series, test_labels = load_ucr_file(test_file)
    all_series = train_series + test_series
    all_labels = train_labels + test_labels
    if not all_series:
        raise ValueError(f"No valid time series found in {train_file} / {test_file}")

    max_len = max(len(row) for row in all_series)
    X = np.zeros((len(all_series), max_len), dtype=float)
    for idx, row in enumerate(all_series):
        X[idx, : len(row)] = row

    _, y = np.unique(np.array(all_labels), return_inverse=True)
    return X, y


def _coerce_aeon_X_to_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        return X
    if X.ndim == 3 and X.shape[1] == 1:
        return X[:, 0, :]
    if X.ndim == 3:
        raise ValueError(
            f"Expected univariate UCR data, but aeon returned shape {X.shape}. "
            "This project scope is univariate whole-series clustering."
        )
    raise ValueError(f"Expected aeon data with 2 or 3 dimensions, got shape {X.shape}")


def load_aeon_dataset(dataset_name: str, extract_path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    try:
        from aeon.datasets import load_classification
        from aeon.datasets.tsc_datasets import tsc_zenodo
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("aeon is not installed. Run `python -m pip install aeon`.") from exc

    cache_root = Path(extract_path) if extract_path is not None else DEFAULT_AEON_DATA_ROOT
    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        X, y_raw = load_classification(dataset_name, split=None, extract_path=str(cache_root))
    except Exception as first_error:
        if dataset_name not in tsc_zenodo:
            raise
        dataset_dir = cache_root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        record_id = tsc_zenodo[dataset_name]
        for split in ("TRAIN", "TEST"):
            target = dataset_dir / f"{dataset_name}_{split}.ts"
            if not target.exists():
                url = f"https://zenodo.org/records/{record_id}/files/{dataset_name}_{split}.ts"
                try:
                    urlretrieve(url, target)
                except Exception as download_error:
                    raise RuntimeError(
                        f"aeon could not auto-download {dataset_name}, and Zenodo fallback failed. "
                        f"First error: {first_error}; fallback error: {download_error}"
                    ) from download_error
        X, y_raw = load_classification(dataset_name, split=None, extract_path=str(cache_root))
    X_2d = _coerce_aeon_X_to_2d(X)
    _, y = np.unique(np.asarray(y_raw), return_inverse=True)
    return X_2d, y


def load_dataset(
    dataset_name: str,
    data_source: str,
    data_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    data_source = data_source.lower().strip()
    if data_source == "aeon":
        return load_aeon_dataset(dataset_name, extract_path=data_root)
    if data_source == "files":
        train_file, test_file = resolve_ucr_files(dataset_name, data_root)
        return load_ucr_dataset(train_file, test_file)
    raise ValueError("data_source must be either 'aeon' or 'files'")


def balanced_subsample(
    X: np.ndarray,
    y: np.ndarray,
    samples_per_class: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if samples_per_class is None:
        return X, y
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in np.unique(y):
        label_idx = np.where(y == label)[0]
        n_take = min(samples_per_class, len(label_idx))
        selected.extend(rng.choice(label_idx, size=n_take, replace=False).tolist())
    selected_array = np.array(selected, dtype=int)
    rng.shuffle(selected_array)
    return X[selected_array], y[selected_array]


def run_single_measure(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    measure: str,
    seed: int,
    perturbation_type: str = "none",
    perturbation_level: str = "0",
    similarity_params: dict | None = None,
) -> dict[str, object]:
    k = int(len(np.unique(y)))
    start = time.perf_counter()
    effective_params = dict(similarity_params or {})
    result = cluster_time_series(
        X,
        k=k,
        normalize=True,
        random_state=seed,
        similarity_metric=measure,
        similarity_params=effective_params,
    )
    runtime = time.perf_counter() - start
    canonical_measure = "ed" if measure == "euclidean" else measure
    return {
        "dataset": dataset_name,
        "measure": canonical_measure,
        "paradigm": MEASURE_PARADIGMS.get(canonical_measure, "unknown"),
        "ari": adjusted_rand_score(y, result.labels),
        "nmi": normalized_mutual_info_score(y, result.labels),
        "runtime": runtime,
        "seed": seed,
        "perturbation_type": perturbation_type,
        "perturbation_level": perturbation_level,
        "n_samples": int(X.shape[0]),
        "series_length": int(X.shape[1]),
        "k": k,
    }


def write_result_rows(rows: Iterable[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})
