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
    DatasetMeta("Chinatown", 24, 2, 363, "Traffic",
                "Very short traffic series; useful lower length bound."),
    DatasetMeta("SyntheticControl", 60, 6, 600, "Synthetic",
                "Classic controlled shapes with trend and shift-like classes."),
    DatasetMeta("MoteStrain", 84, 2, 1272, "Sensor",
                "Short IoT sensor series with real measurement variation."),
    DatasetMeta("ECG200", 96, 2, 200, "Medical",
                "Canonical ECG binary dataset."),
    DatasetMeta("CBF", 128, 3, 930, "Synthetic",
                "Cylinder-Bell-Funnel motifs; strong shape separation."),
    DatasetMeta("TwoPatterns", 128, 4, 5000, "Synthetic",
                "Four synthetic patterns with timing variation."),
    DatasetMeta("ECGFiveDays", 136, 2, 884, "Medical",
                "ECG across days with likely phase variation."),
    DatasetMeta("Plane", 144, 7, 210, "Sensor",
                "Multi-class radar-return shapes."),
    DatasetMeta("GunPoint", 150, 2, 200, "Motion",
                "Classic motion-capture benchmark with mild warping."),
    DatasetMeta("Wine", 234, 2, 111, "Spectroscopy",
                "Small spectroscopy dataset with medium-length series."),
    DatasetMeta("ArrowHead", 251, 3, 211, "Shape",
                "Object outline series with shape-based classes."),
    DatasetMeta("Trace", 275, 4, 200, "Synthetic",
                "Process-control shapes; good perturbation candidate."),
    DatasetMeta("Coffee", 286, 2, 56, "Spectroscopy",
                "Small spectroscopy dataset; non-temporal shape signal."),
    DatasetMeta("DiatomSizeReduction", 345, 4, 322, "Image",
                "Image-outline classes with clear morphology."),
    DatasetMeta("Symbols", 398, 6, 1020, "Shape",
                "Handwritten symbol outlines with six classes."),
    DatasetMeta("OSULeaf", 427, 6, 442, "Shape",
                "Leaf outlines; longer multi-class shape benchmark."),
    DatasetMeta("Computers", 720, 2, 500, "Device",
                "Longer binary device-use series."),
    DatasetMeta("ACSF1", 1460, 10, 200, "Device",
                "Very long multi-class appliance-control series."),
)


MEASURE_PARADIGMS = {
    "ed": "lock-step",
    "euclidean": "lock-step",
    "dtw": "elastic",
    "msm": "elastic",
    "sbd": "sliding",
    "idk": "distributional",
}


# Unified collaboration schema (16 fields, agreed with Wang 2026-05).
# - subsample_seed is fixed to 42 across all measures and seeds.
# - measure_params / clustering_params are JSON-encoded hyperparameter dumps.
RESULT_FIELDS = [
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
        (dataset_dir / f"{dataset_name}_TRAIN.txt",
         dataset_dir / f"{dataset_name}_TEST.txt"),
        (dataset_dir / f"{dataset_name}_TRAIN.tsv",
         dataset_dir / f"{dataset_name}_TEST.tsv"),
        (data_root / f"{dataset_name}_TRAIN.txt",
         data_root / f"{dataset_name}_TEST.txt"),
        (data_root / f"{dataset_name}_TRAIN.tsv",
         data_root / f"{dataset_name}_TEST.tsv"),
    ]
    for train_file, test_file in candidates:
        if train_file.exists() and test_file.exists():
            return train_file, test_file
    raise FileNotFoundError(
        f"Could not find TRAIN/TEST files for {dataset_name} under {data_root}")


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
                series.append(np.array([float(value)
                              for value in parts[1:]], dtype=float))
            except ValueError:
                continue
    return series, labels


def load_ucr_dataset(train_file: Path, test_file: Path) -> tuple[np.ndarray, np.ndarray]:
    train_series, train_labels = load_ucr_file(train_file)
    test_series, test_labels = load_ucr_file(test_file)
    all_series = train_series + test_series
    all_labels = train_labels + test_labels
    if not all_series:
        raise ValueError(
            f"No valid time series found in {train_file} / {test_file}")

    max_len = max(len(row) for row in all_series)
    X = np.zeros((len(all_series), max_len), dtype=float)
    for idx, row in enumerate(all_series):
        X[idx, : len(row)] = row

    _, y = np.unique(np.array(all_labels), return_inverse=True)
    return X, y


def _to_2d_univariate_array(X: object) -> np.ndarray:
    """Convert aeon UCR output to (n_samples, series_length)."""
    try:
        X_array = np.asarray(X, dtype=float)
    except (TypeError, ValueError):
        rows = [np.ravel(np.asarray(row, dtype=float))
                for row in X]  # type: ignore[arg-type]
        lengths = {len(row) for row in rows}
        if len(lengths) != 1:
            raise ValueError(
                "Expected equal-length univariate time series from aeon.")
        return np.vstack(rows)

    if X_array.ndim == 2:
        return X_array
    if X_array.ndim == 3:
        if X_array.shape[1] == 1:
            return X_array[:, 0, :]
        if X_array.shape[2] == 1:
            return X_array[:, :, 0]
        raise ValueError(
            "Expected univariate UCR data, but aeon returned multiple channels.")
    raise ValueError(
        f"Expected aeon data with 2 or 3 dimensions, got shape {X_array.shape}.")


def load_aeon_dataset(
    dataset_name: str,
    data_root: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a UCR classification dataset through aeon and return X, y.

    aeon stores univariate datasets as (n_samples, 1, length). Chen's
    benchmark pipeline uses the simpler 2D shape (n_samples, length), so this
    helper is the single conversion point shared by Part 1 and Part 2.
    """
    try:
        from aeon.datasets import load_classification, load_from_ts_file
        from aeon.datasets.tsc_datasets import tsc_zenodo
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "aeon is not installed. Install aeon or run with --data-source files."
        ) from exc

    root = Path(data_root) if data_root is not None else DEFAULT_AEON_DATA_ROOT
    root.mkdir(parents=True, exist_ok=True)
    try:
        X_raw, y_raw = load_classification(
            dataset_name,
            split=None,
            extract_path=str(root),
            load_equal_length=True,
            load_no_missing=True,
        )
    except Exception:
        if dataset_name not in tsc_zenodo:
            raise
        dataset_dir = root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        record_id = tsc_zenodo[dataset_name]
        for split_name in ("TRAIN", "TEST"):
            target = dataset_dir / f"{dataset_name}_{split_name}.ts"
            if not target.exists():
                url = (
                    f"https://zenodo.org/records/{record_id}/files/"
                    f"{dataset_name}_{split_name}.ts"
                )
                _download_ts_file(url, target)
        X_train, y_train = load_from_ts_file(
            str(dataset_dir / f"{dataset_name}_TRAIN.ts"))
        X_test, y_test = load_from_ts_file(
            str(dataset_dir / f"{dataset_name}_TEST.ts"))
        X_raw = np.concatenate([X_train, X_test])
        y_raw = np.concatenate([y_train, y_test])
    X = _to_2d_univariate_array(X_raw)
    _, y = np.unique(np.asarray(y_raw), return_inverse=True)
    return X, y


def _download_ts_file(url: str, target: Path, max_attempts: int = 3) -> None:
    """Download a UCR .ts file from Zenodo with integrity check + retry.

    Saves first to a .tmp sibling file, verifies the download is non-empty
    and ends with a newline byte (catches truncated transfers), then
    atomically renames into place. Retries up to `max_attempts` times.
    """
    tmp = target.parent / (target.name + ".tmp")
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        if tmp.exists():
            tmp.unlink()
        try:
            urlretrieve(url, tmp)
            size = tmp.stat().st_size
            if size == 0:
                raise OSError(f"Downloaded file is empty: {url}")
            with open(tmp, "rb") as fh:
                fh.seek(-1, 2)
                last_byte = fh.read(1)
            if last_byte not in (b"\n", b"\r"):
                raise OSError(
                    f"Downloaded file looks truncated "
                    f"(size={size}, last_byte={last_byte!r}): {url}"
                )
            tmp.replace(target)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if tmp.exists():
                tmp.unlink()
            if attempt == max_attempts:
                raise OSError(
                    f"Failed to download {url} after {max_attempts} attempts: {exc!r}"
                ) from exc
    # Unreachable, but keeps type checkers happy.
    if last_error is not None:
        raise last_error


def load_dataset(
    dataset_name: str,
    data_source: str = "aeon",
    data_root: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a dataset from either aeon cache/download or local UCR files."""
    normalized_source = data_source.lower().strip()
    root = Path(data_root) if data_root is not None else None
    if normalized_source == "aeon":
        return load_aeon_dataset(dataset_name, root)
    if normalized_source == "files":
        file_root = resolve_data_root(str(root) if root is not None else None)
        train_file, test_file = resolve_ucr_files(dataset_name, file_root)
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
        selected.extend(rng.choice(
            label_idx, size=n_take, replace=False).tolist())
    selected_array = np.array(selected, dtype=int)
    rng.shuffle(selected_array)
    return X[selected_array], y[selected_array]


def run_single_measure(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    measure: str,
    clustering_seed: int,
    subsample_seed: int = 42,
    perturbation_type: str = "none",
    perturbation_level: str = "0",
    similarity_params: dict | None = None,
    n_original: int | None = None,
) -> dict[str, object]:
    import json
    from tsclust.clustering.clustering import _zscore_normalize
    from tsclust.clustering.k_medoids import k_medoids
    from tsclust.measures.similarity_measures import (
        dtw_distance_matrix,
        euclidean_distance_matrix,
        msm_distance_matrix,
    )

    k = int(len(np.unique(y)))
    effective_params = dict(similarity_params or {})
    canonical_measure = "ed" if measure == "euclidean" else measure

    # --- z-normalization ---
    X_norm = _zscore_normalize(X)

    # --- Distance matrix computation (timed) ---
    params_record: dict[str, object] = {}
    t0 = time.perf_counter()
    if canonical_measure in {"euclidean", "ed"}:
        dist = euclidean_distance_matrix(X_norm)
    elif canonical_measure == "dtw":
        backend = effective_params.pop("backend", "auto")
        # Default: Sakoe-Chiba band = 10% of series length (Javed et al. 2020)
        default_window = max(1, int(round(X_norm.shape[1] * 0.1)))
        dtw_window = effective_params.pop("window", default_window)
        dist = dtw_distance_matrix(X_norm, window=dtw_window, backend=backend)
        params_record["window"] = dtw_window
        params_record["backend"] = backend
    elif canonical_measure == "msm":
        backend = effective_params.pop("backend", "auto")
        # Default: c=1.0 (aeon default; no hyperparameter tuning)
        msm_c = float(effective_params.pop("c", 1.0))
        dist = msm_distance_matrix(X_norm, c=msm_c, backend=backend)
        params_record["c"] = msm_c
        params_record["backend"] = backend
    elif canonical_measure == "sbd":
        from tsclust.measures.similarity_measures import sbd_distance_matrix
        backend = effective_params.pop("backend", "aeon")
        n_jobs = int(effective_params.pop("n_jobs", -1))
        # standardize=False because we already z-normalized above
        dist = sbd_distance_matrix(
            X_norm, backend=backend, n_jobs=n_jobs, standardize=False,
        )
        params_record["backend"] = backend
        params_record["n_jobs"] = n_jobs
    elif canonical_measure == "idk":
        from tsclust.measures.isolation_kernel import IsolationKernel
        # Pop keys that IsolationKernel does not accept
        effective_params.pop("backend", None)
        kernel = IsolationKernel(
            random_state=clustering_seed,
            **effective_params,
        ).fit(X_norm)
        sim = kernel.similarity_matrix(X_norm)
        # Kernel-induced distance: sqrt(K(x,x)+K(y,y)-2K(x,y))
        # For L2-normalized feature maps (as in CTDS / Gong et al.),
        # K(x,x)=1, so this equals sqrt(2-2·sim).  Equivalent to
        # Euclidean distance in normalised feature space — matches
        # the distance implicitly used by CTDS's KMeans pipeline.
        dist = np.sqrt(np.clip(2.0 - 2.0 * sim, 0.0, None))
        params_record.update(effective_params)
    else:
        raise NotImplementedError(f"Unsupported measure: {measure!r}")
    runtime_dist = time.perf_counter() - t0

    # --- Clustering (timed) ---
    t1 = time.perf_counter()
    medoids, labels = k_medoids(dist, k=k, random_state=clustering_seed)
    runtime_clust = time.perf_counter() - t1

    runtime_total = runtime_dist + runtime_clust

    clustering_params = {
        "init": "random",
        "max_iter": 300,
        "method": "alternate",
    }

    n_sampled = int(X.shape[0])
    return {
        "dataset": dataset_name,
        "measure": canonical_measure,
        "paradigm": MEASURE_PARADIGMS.get(canonical_measure, "unknown"),
        "ari": adjusted_rand_score(y, labels),
        "nmi": normalized_mutual_info_score(y, labels),
        "runtime": runtime_total,
        "subsample_seed": subsample_seed,
        "clustering_seed": clustering_seed,
        "perturbation_type": perturbation_type,
        "perturbation_level": perturbation_level,
        "n_original": int(n_original) if n_original is not None else n_sampled,
        "n_sampled": n_sampled,
        "series_length": int(X.shape[1]),
        "k": k,
        "measure_params": json.dumps(params_record) if params_record else "",
        "clustering_params": json.dumps(clustering_params),
    }


def write_result_rows(rows: Iterable[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "")
                            for field in RESULT_FIELDS})


def append_result_row(row: dict[str, object], output_path: Path) -> None:
    """Append a single row to CSV; write header if file is new.

    Used for incremental writing so a crash mid-batch does not lose progress.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    mode = "w" if write_header else "a"
    with output_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "")
                        for field in RESULT_FIELDS})


def load_done_keys(
    output_path: Path,
    perturbation_type: str = "none",
) -> set[tuple[str, str, str, str]]:
    """Load (dataset, measure, perturbation_level, clustering_seed) keys already in CSV.

    Returns empty set if file does not exist.
    """
    if not output_path.exists():
        return set()
    done: set[tuple[str, str, str, str]] = set()
    with output_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("perturbation_type", "none") != perturbation_type:
                continue
            done.add((
                str(row.get("dataset", "")),
                str(row.get("measure", "")),
                str(row.get("perturbation_level", "0")),
                str(row.get("clustering_seed", "")),
            ))
    return done
