from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .k_medoids import k_medoids
from ..measures.similarity_measures import (
    distance_to_similarity,
    dtw_distance_matrix,
    euclidean_distance_matrix,
    msm_distance_matrix,
    sbd_distance_matrix,
)


@dataclass
class ClusteringResult:
    medoids: np.ndarray
    labels: np.ndarray
    similarity_matrix: np.ndarray
    distance_matrix: np.ndarray


def _zscore_normalize(X: np.ndarray) -> np.ndarray:
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std


def cluster_time_series(
    X: np.ndarray,
    k: int,
    n_trees: int = 200,
    sample_size: int = 256,
    normalize: bool = True,
    random_state: Optional[int] = None,
    window_size: Optional[int] = None,
    window_step: Optional[int] = None,
    similarity_metric: str = "idk",
    similarity_params: Optional[dict] = None,
) -> ClusteringResult:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(
            "X must be a 2D array with shape (n_samples, series_length)")

    similarity_metric = similarity_metric.lower().strip()
    similarity_params = dict(similarity_params or {})

    if normalize:
        X = _zscore_normalize(X)

    if similarity_metric == "idk":
        from ..measures.isolation_kernel import IsolationKernel

        # Extract IDK-specific parameters (ignore others like 'backend')
        ik_method = similarity_params.pop("method", "anne")
        
        kernel = IsolationKernel(
            n_trees=n_trees,
            sample_size=sample_size,
            random_state=random_state,
            method=ik_method,
            window_size=window_size,
            window_step=window_step,
        ).fit(X)
        sim = kernel.similarity_matrix(X)
        dist = 1.0 - sim
    elif similarity_metric in {"euclidean", "euclid", "ed"}:
        dist = euclidean_distance_matrix(X)
        sim = distance_to_similarity(dist)
    elif similarity_metric == "dtw":
        dtw_window = similarity_params.pop("window", None)
        backend = similarity_params.pop("backend", "auto")
        dist = dtw_distance_matrix(X, window=dtw_window, backend=backend)
        sim = distance_to_similarity(dist)
    elif similarity_metric == "msm":
        msm_c = float(similarity_params.pop("c", 0.1))
        backend = similarity_params.pop("backend", "auto")
        dist = msm_distance_matrix(X, c=msm_c, backend=backend)
        sim = distance_to_similarity(dist)
    elif similarity_metric == "sbd":
        backend = similarity_params.pop("backend", "auto")
        n_jobs = int(similarity_params.pop("n_jobs", -1))
        standardize = bool(similarity_params.pop("standardize", not normalize))
        candidate_k = int(similarity_params.pop("candidate_k", 20))
        coarse_method = str(similarity_params.pop("coarse_method", "paa"))
        paa_segments = int(similarity_params.pop("paa_segments", 32))
        dist = sbd_distance_matrix(
            X,
            backend=backend,
            n_jobs=n_jobs,
            standardize=standardize,
            candidate_k=candidate_k,
            coarse_method=coarse_method,
            paa_segments=paa_segments,
        )
        sim = distance_to_similarity(dist)
    else:
        raise NotImplementedError(
            f"similarity_metric={similarity_metric!r} is reserved for future comparison experiments. Supported values: 'idk', 'euclidean', 'dtw', 'msm', 'sbd'."
        )

    medoids, labels = k_medoids(dist, k=k, random_state=random_state)

    return ClusteringResult(
        medoids=medoids,
        labels=labels,
        similarity_matrix=sim,
        distance_matrix=dist,
    )
