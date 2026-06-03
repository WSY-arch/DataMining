from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import pairwise_distances


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


def sbd_distance_matrix(
    X: np.ndarray,
    backend: str = "aeon",
    n_jobs: int = -1,
    standardize: bool = True,
) -> np.ndarray:
    """Return an SBD pairwise distance matrix.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, series_length)
    backend : 'aeon' (exact, recommended) or 'reference' (slow O(n^2) Python loop)
    standardize : whether to z-normalize within the SBD computation
    """
    X = np.asarray(X, dtype=float)
    backend = backend.lower().strip()
    if backend in {"auto", "aeon"}:
        from aeon.distances import sbd_pairwise_distance
        return np.asarray(
            sbd_pairwise_distance(X, standardize=standardize),
            dtype=float,
        )
    if backend == "reference":
        # Slow pure-Python fallback using circular cross-correlation
        def _sbd_pair(x, y):
            if standardize:
                xm, xs = x.mean(), x.std()
                ym, ys = y.mean(), y.std()
                x = (x - xm) / xs if xs > 0 else x - xm
                y = (y - ym) / ys if ys > 0 else y - ym
            corr = np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(y))).real
            denom = float(np.linalg.norm(x) * np.linalg.norm(y))
            ncc_max = float(np.max(corr / denom)) if denom > 0 else 0.0
            return float(np.clip(1.0 - ncc_max, 0.0, 2.0))
        return pairwise_time_series_distance_matrix(X, _sbd_pair)
    raise ValueError(f"sbd_distance_matrix: unsupported backend {backend!r}")


def distance_to_similarity(distance_matrix: np.ndarray) -> np.ndarray:
    """Convert a non-negative distance matrix to a bounded similarity matrix."""
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    return 1.0 / (1.0 + distance_matrix)
