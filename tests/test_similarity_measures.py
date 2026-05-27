from tsclust.measures.similarity_measures import (
    dtw_distance,
    msm_distance,
    sbd_distance,
    sbd_distance_matrix,
    pairwise_time_series_distance_matrix,
)
from tsclust.clustering import cluster_time_series
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_dtw_distance_basic_properties():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    z = np.array([3.0, 2.0, 1.0, 0.0])

    assert dtw_distance(x, y) == 0.0
    assert dtw_distance(x, z) > 0.0
    assert np.isclose(dtw_distance(x, z), dtw_distance(z, x))


def test_msm_distance_basic_properties():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    z = np.array([3.0, 2.0, 1.0, 0.0])

    assert msm_distance(x, y) == 0.0
    assert msm_distance(x, z) > 0.0
    assert np.isclose(msm_distance(x, z), msm_distance(z, x))


def test_pairwise_matrix_and_clustering_dispatch():
    X = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.1, 0.9, 1.0],
            [3.0, 3.0, 2.0, 2.0],
            [3.0, 2.9, 2.1, 2.0],
        ]
    )

    dist = pairwise_time_series_distance_matrix(X, dtw_distance)
    assert dist.shape == (4, 4)
    assert np.allclose(dist, dist.T)
    assert np.allclose(np.diag(dist), 0.0)

    for metric in ["ed", "dtw", "msm"]:
        result = cluster_time_series(
            X,
            k=2,
            similarity_metric=metric,
            similarity_params={"backend": "reference"},
            normalize=False,
            random_state=42,
        )
        assert result.distance_matrix.shape == (4, 4)
        assert result.labels.shape == (4,)


def test_sbd_is_circular_shift_invariant():
    rng = np.random.default_rng(42)
    x = rng.normal(size=96)
    y = rng.normal(size=96)

    base = sbd_distance(x, y, standardize=True)
    for shift in [1, 5, 10, 48]:
        rolled = sbd_distance(np.roll(x, shift), y, standardize=True)
        assert np.isclose(rolled, base, atol=1e-10)


def test_sbd_distance_matrix_is_circular_shift_invariant():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(5, 96))
    base = sbd_distance_matrix(X, backend="reference", standardize=True)

    for shift in [1, 5, 10, 48]:
        rolled = sbd_distance_matrix(np.roll(X, shift, axis=1), backend="reference", standardize=True)
        assert np.allclose(base, rolled, atol=1e-10)
