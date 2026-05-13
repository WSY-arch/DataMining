import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tsclust.clustering import cluster_time_series
from tsclust.measures.similarity_measures import (
    dtw_distance,
    msm_distance,
    pairwise_time_series_distance_matrix,
)


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
        result = cluster_time_series(X, k=2, similarity_metric=metric, normalize=False, random_state=42)
        assert result.distance_matrix.shape == (4, 4)
        assert result.labels.shape == (4,)
