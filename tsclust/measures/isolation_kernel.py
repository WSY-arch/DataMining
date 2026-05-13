from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ikpykit.kernel import IsoDisKernel


class IsolationKernel:
    def __init__(
        self,
        n_trees: int = 200,
        sample_size: int = 256,
        random_state: Optional[int] = None,
        method: str = "anne",
        window_size: Optional[int] = None,
        window_step: Optional[int] = None,
    ) -> None:
        self.n_trees = int(n_trees)
        self.sample_size = int(sample_size)
        self.random_state = random_state
        self.method = method
        self.window_size = window_size
        self.window_step = window_step

    def _check_dependency(self) -> None:
        return None

    def _validate_series_matrix(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array with shape (n_samples, series_length)")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        return X

    def _series_matrix(self, series: np.ndarray) -> np.ndarray:
        return np.asarray(series, dtype=float).reshape(-1, 1)

    # _series_windows: Extracts overlapping windows from a time series.
    # If the series is shorter than the window size,
    # it returns the entire series as a single window.
    # series: 1D array of shape (series_length,)
    # returns: 2D array of shape (n_windows, window_size)
    def _series_windows(self, series: np.ndarray) -> np.ndarray:
        series = np.asarray(series, dtype=float)
        window_size = self.window_size_
        window_step = self.window_step_
        if series.shape[0] <= window_size:
            return series.reshape(1, -1)
        # Generate windows using a sliding window approach
        return np.vstack(
            [
                series[i: i + window_size]
                for i in range(0, series.shape[0] - window_size + 1, window_step)
            ]
        )

    # _window_batches: Processes a batch of time series and extracts windows for each series.
    # It returns a combined matrix of all windows and an array of counts indicating how many windows
    # were extracted from each series.
    # X: 2D array of shape (n_samples, series_length)
    # returns: (window_matrix, counts)
    #   - window_matrix: 2D array of shape (total_windows, window_size)
    #   - counts: 1D array of shape (n_samples,) indicating the number of windows for each series
    def _window_batches(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        window_blocks: list[np.ndarray] = []
        counts = np.empty(X.shape[0], dtype=int)
        for i, series in enumerate(X):
            windows = self._series_windows(series)
            window_blocks.append(windows)
            counts[i] = windows.shape[0]
        return np.vstack(window_blocks), counts

    def _series_embedding(self, series: np.ndarray) -> np.ndarray:
        embedded = self._kernel.iso_kernel_.transform(
            self._series_windows(series))
        return np.asarray(embedded.mean(axis=0)).ravel()

    # fit: Fits the Isolation Kernel model to the input time series data.
    # It processes the input data to extract windows and then fits the underlying IsoDisKernel model
    # using the extracted windows. The method also determines the appropriate window size and step
    # if they are not provided.
    # X: 2D array of shape (n_samples, series_length)
    # returns: self (the fitted model)
    def fit(self, X: np.ndarray) -> "IsolationKernel":
        self._check_dependency()
        X = self._validate_series_matrix(X)

        # fit window size and step
        # window_size_: The actual window size to be used for extracting windows from the time series.
        # window_step_: The actual step size to be used for sliding the window across the time series.
        if self.window_size is None:
            self.window_size_ = min(10, X.shape[1])
        else:
            self.window_size_ = max(1, min(int(self.window_size), X.shape[1]))

        if self.window_step is None:
            self.window_step_ = 1
        else:
            self.window_step_ = max(1, int(self.window_step))

        window_matrix, _ = self._window_batches(X)

        # Fit the IsoDisKernel model using the extracted windows
        # .kernel: The fitted IsoDisKernel model that will be used for
        # transforming the time series data into embeddings.
        self._kernel: Any = IsoDisKernel(
            method=self.method,
            n_estimators=self.n_trees,
            max_samples=self.sample_size,  # type: ignore[arg-type]
            random_state=self.random_state,
        ).fit(window_matrix)  # type: ignore[arg-type]
        self.series_length_ = X.shape[1]
        return self

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "_kernel"):
            raise RuntimeError(
                "IsolationKernel is not fitted. Call fit() first.")

    # transform: Transforms the input time series data into a new feature space
    # using the fitted Isolation Kernel model.
    # X: 2D array of shape (n_samples, series_length)
    # returns: array of embedding for each sample. shape (n_samples, n_features) where n_features
    # is the dimensionality of the embedding space
    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = self._validate_series_matrix(X)
        window_matrix, counts = self._window_batches(X)
        window_embeddings = self._kernel.iso_kernel_.transform(window_matrix)

        embeddings = []
        start = 0
        for count in counts:
            series_windows = window_embeddings[start: start + count]
            embeddings.append(np.asarray(series_windows.mean(axis=0)).ravel())
            start += count
        return np.vstack(embeddings)

    def similarity_matrix(self, X: np.ndarray) -> np.ndarray:
        embeddings = self.transform(X)
        # calculate norms for each embedding vector
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # if any norms are zero, set them to 1 to avoid division by zero
        norms[norms == 0] = 1.0
        # normalize embeddings to unit length
        normalized_embeddings = embeddings / norms
        # calculate similarity matrix
        sim = normalized_embeddings @ normalized_embeddings.T
        # clip values to [0, 1] range
        return np.clip(sim, 0.0, 1.0)

    def distance_matrix(self, X: np.ndarray) -> np.ndarray:
        return 1.0 - self.similarity_matrix(X)
