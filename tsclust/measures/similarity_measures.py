from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


def euclidean_distance_matrix(X: np.ndarray) -> np.ndarray:
    """Return the pairwise Euclidean distance matrix."""
    return pairwise_distances(np.asarray(X, dtype=float), metric="euclidean")


def _call_aeon_pairwise_distance(X: np.ndarray, method: str) -> np.ndarray:
    try:
        from aeon import distances as aeon_distances
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "aeon is not installed. Install it or use backend='reference'.") from exc

    direct_name = f"{method}_pairwise_distance"
    if hasattr(aeon_distances, direct_name):
        return np.asarray(getattr(aeon_distances, direct_name)(X), dtype=float)

    if hasattr(aeon_distances, "pairwise_distance"):
        return np.asarray(aeon_distances.pairwise_distance(X, method=method), dtype=float)

    raise AttributeError(
        f"Installed aeon version does not expose a pairwise {method!r} distance API.")


def _call_aeon_sbd_pairwise_distance(
    X: np.ndarray,
    standardize: bool = True,
    n_jobs: int = 1,
) -> np.ndarray:
    try:
        from aeon import distances as aeon_distances
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "aeon is not installed. Install it or use backend='reference'.") from exc

    if hasattr(aeon_distances, "sbd_pairwise_distance"):
        return np.asarray(
            aeon_distances.sbd_pairwise_distance(
                X,
                standardize=standardize,
                n_jobs=int(n_jobs),
            ),
            dtype=float,
        )

    raise AttributeError(
        "Installed aeon version does not expose an SBD pairwise distance API.")


def _paa_transform(X: np.ndarray, n_segments: int) -> np.ndarray:
    """Reduce each series to a low-dimensional PAA feature vector."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must have shape (n_samples, series_length)")

    n_samples, series_length = X.shape
    n_segments = max(1, min(int(n_segments), series_length))
    if n_segments == series_length:
        return X.copy()

    edges = np.linspace(0, series_length, n_segments + 1, dtype=int)
    features = np.empty((n_samples, n_segments), dtype=float)
    for idx in range(n_segments):
        start = int(edges[idx])
        end = int(edges[idx + 1])
        if end <= start:
            end = min(series_length, start + 1)
        features[:, idx] = np.mean(X[:, start:end], axis=1)
    return features


def _candidate_pairs_from_features(
    features: np.ndarray,
    candidate_k: int,
    n_jobs: int,
) -> list[tuple[int, int]]:
    """Return unique pair candidates from a coarse nearest-neighbor search."""
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must have shape (n_samples, n_features)")

    n_samples = features.shape[0]
    if n_samples <= 1:
        return []

    candidate_k = max(1, min(int(candidate_k), n_samples - 1))
    effective_n_jobs = 1 if int(n_jobs) == -1 else int(n_jobs)
    nn = NearestNeighbors(n_neighbors=candidate_k + 1, metric="euclidean", n_jobs=effective_n_jobs)
    nn.fit(features)

    # To avoid allocating a huge (n_samples, n_samples) distance matrix inside
    # sklearn when both query and training sets are large, query in batches.
    # Choose a chunk size that keeps an intermediate distance buffer modest
    # (approx `max_mem_bytes`). Use 200MB default.
    max_mem_bytes = 200 * 1024 * 1024
    bytes_per_distance = 8  # float64
    # For a chunk of size c, the temporary distances array is shape (c, n_samples)
    # so memory ~= c * n_samples * bytes_per_distance. Solve for c.
    est_c = max(1, int(max_mem_bytes / (max(1, n_samples) * bytes_per_distance)))
    chunk_size = min(n_samples, max(1, est_c))

    pairs: set[tuple[int, int]] = set()
    for start in range(0, n_samples, chunk_size):
        end = min(n_samples, start + chunk_size)
        # kneighbors will compute distances between the small query chunk and
        # the full training set; by keeping chunk small we bound memory.
        _, indices = nn.kneighbors(features[start:end], return_distance=True)
        for local_i, neighs in enumerate(indices):
            i = start + int(local_i)
            for j in neighs[1:]:
                a = int(i)
                b = int(j)
                if a == b:
                    continue
                if a > b:
                    a, b = b, a
                pairs.add((a, b))

    return sorted(pairs)


def _call_tslearn_dtw_distance_matrix(X: np.ndarray, window: int | None = None) -> np.ndarray:
    try:
        from tslearn.metrics import cdist_dtw
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tslearn is not installed. Install it or use backend='reference'.") from exc

    if window is None:
        return np.asarray(cdist_dtw(X), dtype=float)
    return np.asarray(cdist_dtw(X, global_constraint="sakoe_chiba", sakoe_chiba_radius=int(window)), dtype=float)


def dtw_distance(x: np.ndarray, y: np.ndarray, window: int | None = None) -> float:
    """Compute Dynamic Time Warping distance for two one-dimensional series."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        raise ValueError("DTW inputs must be non-empty")

    if window is None:
        window = max(n, m)
    window = max(int(window), abs(n - m))

    previous = np.full(m + 1, np.inf)
    current = np.full(m + 1, np.inf)
    previous[0] = 0.0

    for i in range(1, n + 1):
        current.fill(np.inf)
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        for j in range(j_start, j_end + 1):
            cost = (x[i - 1] - y[j - 1]) ** 2
            current[j] = cost + \
                min(previous[j], current[j - 1], previous[j - 1])
        previous, current = current, previous

    return float(np.sqrt(previous[m]))


def _msm_cost(new_value: float, x_value: float, y_value: float, c: float) -> float:
    if (x_value <= new_value <= y_value) or (y_value <= new_value <= x_value):
        return c
    return c + min(abs(new_value - x_value), abs(new_value - y_value))


def msm_distance(x: np.ndarray, y: np.ndarray, c: float = 0.1) -> float:
    """Compute Move-Split-Merge distance for two one-dimensional series."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        raise ValueError("MSM inputs must be non-empty")
    if c <= 0:
        raise ValueError("MSM cost parameter c must be positive")

    dp = np.empty((n, m), dtype=float)
    dp[0, 0] = abs(x[0] - y[0])

    for i in range(1, n):
        dp[i, 0] = dp[i - 1, 0] + _msm_cost(x[i], x[i - 1], y[0], c)
    for j in range(1, m):
        dp[0, j] = dp[0, j - 1] + _msm_cost(y[j], x[0], y[j - 1], c)

    for i in range(1, n):
        for j in range(1, m):
            move = dp[i - 1, j - 1] + abs(x[i] - y[j])
            split = dp[i - 1, j] + _msm_cost(x[i], x[i - 1], y[j], c)
            merge = dp[i, j - 1] + _msm_cost(y[j], x[i], y[j - 1], c)
            dp[i, j] = min(move, split, merge)

    return float(dp[n - 1, m - 1])


def dtw_distance_matrix(
    X: np.ndarray,
    window: int | None = None,
    backend: str = "auto",
) -> np.ndarray:
    """Return a DTW pairwise distance matrix using a library backend when available."""
    backend = backend.lower().strip()
    if backend in {"auto", "aeon"}:
        try:
            return _call_aeon_pairwise_distance(X, "dtw")
        except Exception:
            if backend == "aeon":
                raise
    if backend in {"auto", "tslearn"}:
        try:
            return _call_tslearn_dtw_distance_matrix(X, window=window)
        except Exception:
            if backend == "tslearn":
                raise
    if backend in {"auto", "reference"}:
        return pairwise_time_series_distance_matrix(X, dtw_distance, window=window)
    raise ValueError("backend must be one of: auto, reference, aeon, tslearn")


def msm_distance_matrix(
    X: np.ndarray,
    c: float = 0.1,
    backend: str = "auto",
) -> np.ndarray:
    """Return an MSM pairwise distance matrix using aeon when available."""
    backend = backend.lower().strip()
    if backend in {"auto", "aeon"}:
        try:
            return _call_aeon_pairwise_distance(X, "msm")
        except Exception:
            if backend == "aeon":
                raise
    if backend in {"auto", "reference"}:
        return pairwise_time_series_distance_matrix(X, msm_distance, c=c)
    raise ValueError("backend must be one of: auto, reference, aeon")


def pairwise_time_series_distance_matrix(
    X: np.ndarray,
    distance_fn: Callable[..., float],
    **distance_params: object,
) -> np.ndarray:
    """Compute a symmetric pairwise distance matrix for a 2D time-series array."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must have shape (n_samples, series_length)")

    n_samples = X.shape[0]
    dist = np.zeros((n_samples, n_samples), dtype=float)
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            value = distance_fn(X[i], X[j], **distance_params)
            dist[i, j] = value
            dist[j, i] = value
    return dist


def distance_to_similarity(distance_matrix: np.ndarray) -> np.ndarray:
    """Convert a non-negative distance matrix to a bounded similarity matrix."""
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    return 1.0 / (1.0 + distance_matrix)


def sbd_distance(x: np.ndarray, y: np.ndarray, standardize: bool = True) -> float:
    """
    Compute Shape-Based Distance between two z-normalized sequences.
    
    SBD is based on cross-correlation of the sequences and measures shape similarity.
    Lower values indicate more similar shapes.
    
    Reference:
        Paparrizos, J., & Gravano, L. (2015).
        k-Shape: Efficient and Accurate Clustering of Time Series.
        ACM SIGMOD Record, 45(1), 69-76.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    if len(x) == 0 or len(y) == 0:
        raise ValueError("SBD inputs must be non-empty")
    
    # Ensure sequences are 1D
    x = x.ravel()
    y = y.ravel()
    
    if standardize:
        # z-normalize each sequence independently
        x_mean, x_std = np.mean(x), np.std(x)
        if x_std == 0:
            x_norm = x - x_mean
        else:
            x_norm = (x - x_mean) / x_std

        y_mean, y_std = np.mean(y), np.std(y)
        if y_std == 0:
            y_norm = y - y_mean
        else:
            y_norm = (y - y_mean) / y_std
    else:
        x_norm = x
        y_norm = y
    
    # Compute cross-correlation via convolution
    # np.convolve computes the discrete convolution of two sequences
    ncc = np.convolve(x_norm, y_norm[::-1], mode='full')

    # Normalize by length only when raw inputs are provided.
    if standardize:
        x_std = float(np.std(x))
        y_std = float(np.std(y))
        ncc = ncc / (len(x) * x_std * y_std) if (x_std > 0 and y_std > 0) else ncc / len(x)
    else:
        ncc = ncc / len(x)
    
    # Maximum normalized cross-correlation
    max_ncc = np.max(ncc)
    
    # SBD distance: 1 - max_ncc (range [0, 2])
    # Clamped to [0, 2]
    sbd = 1.0 - max_ncc
    return float(np.clip(sbd, 0.0, 2.0))


def sbd_approx_distance_matrix(
    X: np.ndarray,
    candidate_k: int = 20,
    coarse_method: str = "paa",
    paa_segments: int = 32,
    n_jobs: int = 1,
    standardize: bool = True,
    fallback_distance: float = 2.0,
) -> np.ndarray:
    """Approximate SBD distance matrix via coarse candidate pruning.

    Steps:
    1. Build cheap coarse features (currently PAA).
    2. Use nearest-neighbor search in coarse space to get a candidate set.
    3. Compute exact SBD only for candidate pairs.
    4. Fill all non-candidate pairs with a large fallback distance.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must have shape (n_samples, series_length)")

    coarse_method = coarse_method.lower().strip()
    if coarse_method != "paa":
        raise ValueError("Currently supported coarse_method values: 'paa'")

    n_samples = X.shape[0]
    dist = np.full((n_samples, n_samples), float(fallback_distance), dtype=float)
    np.fill_diagonal(dist, 0.0)

    coarse_features = _paa_transform(X, paa_segments)
    candidate_pairs = _candidate_pairs_from_features(coarse_features, candidate_k=candidate_k, n_jobs=n_jobs)

    for i, j in candidate_pairs:
        value = sbd_distance(X[i], X[j], standardize=standardize)
        dist[i, j] = value
        dist[j, i] = value

    return dist


def sbd_distance_matrix(
    X: np.ndarray,
    backend: str = "auto",
    n_jobs: int = -1,
    standardize: bool = True,
    candidate_k: int = 20,
    coarse_method: str = "paa",
    paa_segments: int = 32,
) -> np.ndarray:
    """
    Return an SBD pairwise distance matrix.
    
    Parameters
    ----------
    X : np.ndarray
        Time series array with shape (n_samples, series_length).
    backend : str
        Backend to use. Currently only 'reference' is fully supported.
    
    Returns
    -------
    np.ndarray
        Symmetric distance matrix of shape (n_samples, n_samples).
    """
    backend = backend.lower().strip()

    if backend in {"approx", "candidate", "pruned"}:
        return sbd_approx_distance_matrix(
            X,
            candidate_k=candidate_k,
            coarse_method=coarse_method,
            paa_segments=paa_segments,
            n_jobs=n_jobs,
            standardize=standardize,
        )
    
    if backend in {"auto", "aeon"}:
        try:
            return _call_aeon_sbd_pairwise_distance(X, standardize=standardize, n_jobs=n_jobs)
        except Exception:
            if backend == "aeon":
                raise

    if backend in {"auto", "reference"}:
        return pairwise_time_series_distance_matrix(X, sbd_distance, standardize=standardize)
    else:
        raise ValueError("backend must be one of: auto, approx, candidate, pruned, aeon, reference")
