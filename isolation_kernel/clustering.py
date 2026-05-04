from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics import pairwise_distances

from .isolation_kernel import IsolationKernel
from .k_medoids import k_medoids


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
        raise ValueError("X must be a 2D array with shape (n_samples, series_length)")

    similarity_metric = similarity_metric.lower().strip()
    similarity_params = dict(similarity_params or {})

    if normalize:
        X = _zscore_normalize(X)

    if similarity_metric == "idk":
        kernel = IsolationKernel(
            n_trees=n_trees,
            sample_size=sample_size,
            random_state=random_state,
            window_size=window_size,
            window_step=window_step,
            **similarity_params,
        ).fit(X)
        sim = kernel.similarity_matrix(X)
        dist = 1.0 - sim
    elif similarity_metric in {"euclidean", "euclid"}:
        dist = pairwise_distances(X, metric="euclidean")
        sim = 1.0 / (1.0 + dist)
    else:
        raise NotImplementedError(
            f"similarity_metric={similarity_metric!r} is reserved for future comparison experiments. Supported values: 'idk', 'euclidean'."
        )

    medoids, labels = k_medoids(dist, k=k, random_state=random_state)

    return ClusteringResult(
        medoids=medoids,
        labels=labels,
        similarity_matrix=sim,
        distance_matrix=dist,
    )
