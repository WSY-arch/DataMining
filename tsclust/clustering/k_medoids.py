from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _initialize_medoids(n_samples: int, k: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(n_samples, size=k, replace=False)


def _handle_empty_cluster(
    distance_matrix: np.ndarray,
    medoids: np.ndarray,
    rng: np.random.Generator,
) -> int:
    n_samples = distance_matrix.shape[0]
    non_medoids = np.setdiff1d(np.arange(n_samples), medoids)
    if non_medoids.size == 0:
        return int(rng.integers(0, n_samples))
    min_dist = np.min(distance_matrix[:, medoids], axis=1)
    candidate = non_medoids[np.argmax(min_dist[non_medoids])]
    return int(candidate)


def k_medoids(
    distance_matrix: np.ndarray,
    k: int,
    max_iter: int = 300,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n_samples = distance_matrix.shape[0]
    if k <= 0 or k > n_samples:
        raise ValueError("k must be in the range [1, n_samples]")

    rng = np.random.default_rng(random_state)
    medoids = _initialize_medoids(n_samples, k, rng)
    labels = np.argmin(distance_matrix[:, medoids], axis=1)

    for _ in range(max_iter):
        new_medoids = medoids.copy()
        for cluster_id in range(k):
            cluster_idx = np.where(labels == cluster_id)[0]
            if cluster_idx.size == 0:
                new_medoids[cluster_id] = _handle_empty_cluster(
                    distance_matrix, new_medoids, rng
                )
                continue
            cluster_dist = distance_matrix[np.ix_(cluster_idx, cluster_idx)]
            total_dist = np.sum(cluster_dist, axis=1)
            best = cluster_idx[np.argmin(total_dist)]
            new_medoids[cluster_id] = best

        if np.array_equal(new_medoids, medoids):
            break
        medoids = new_medoids
        labels = np.argmin(distance_matrix[:, medoids], axis=1)

    return medoids, labels
