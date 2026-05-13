"""
Complete test suite for Isolation Kernel + K-Medoids clustering
All tests in English for better cross-platform compatibility
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tsclust.measures.isolation_kernel import IsolationKernel
from tsclust.clustering import cluster_time_series, k_medoids
from tsclust.visualization import (
    plot_clustering_results,
    plot_distance_matrix,
    plot_medoids_comparison,
    plot_cluster_statistics,
)


def test_isolation_kernel_basic():
    """Test IsolationKernel basic functionality"""
    print("=" * 60)
    print("TEST 1: IsolationKernel Basic Functionality")
    print("=" * 60)
    
    # Generate simple data
    X = np.random.randn(20, 10)
    
    # Initialize and fit
    kernel = IsolationKernel(n_trees=10, sample_size=8, random_state=42)
    kernel.fit(X)
    print("[PASS] fit() successful")
    
    # Transform
    embeddings = kernel.transform(X)
    assert embeddings.shape[0] == 20, f"Expected 20 rows, got {embeddings.shape[0]}"
    assert embeddings.ndim == 2, f"Expected 2D embeddings, got {embeddings.ndim}D"
    print(f"[PASS] transform() successful - embeddings shape: {embeddings.shape}")
    
    # Similarity matrix
    sim = kernel.similarity_matrix(X)
    assert sim.shape == (20, 20), f"Expected shape (20, 20), got {sim.shape}"
    assert np.allclose(sim, sim.T), "Similarity matrix should be symmetric"
    assert np.all(np.diag(sim) > 0), "Diagonal similarity should be positive"
    print("[PASS] similarity_matrix() successful")
    print(f"       - Shape: {sim.shape}")
    print(f"       - Symmetric: YES")
    print(f"       - Diagonal values: {np.diag(sim)[:5]}...")
    
    # Distance matrix
    dist = kernel.distance_matrix(X)
    assert dist.shape == (20, 20), f"Expected shape (20, 20), got {dist.shape}"
    assert np.allclose(dist, dist.T), "Distance matrix should be symmetric"
    assert np.allclose(np.diag(dist), 1.0 - np.diag(sim)), "Distance should be 1 - similarity"
    print("[PASS] distance_matrix() successful")
    print(f"       - Diagonal values: {np.diag(dist)[:5]}...")
    print()


def test_k_medoids_basic():
    """Test k-medoids basic functionality"""
    print("=" * 60)
    print("TEST 2: K-Medoids Basic Functionality")
    print("=" * 60)
    
    # Simple distance matrix
    dist_matrix = np.array([
        [0.0, 0.1, 0.9, 0.95],
        [0.1, 0.0, 0.85, 0.9],
        [0.9, 0.85, 0.0, 0.05],
        [0.95, 0.9, 0.05, 0.0],
    ], dtype=float)
    
    medoids, labels = k_medoids(dist_matrix, k=2, random_state=42)
    assert len(medoids) == 2, f"Expected 2 medoids, got {len(medoids)}"
    assert len(labels) == 4, f"Expected 4 labels, got {len(labels)}"
    assert np.min(labels) >= 0 and np.max(labels) < 2, "Labels should be in range [0, k-1]"
    print("[PASS] k_medoids() successful")
    print(f"       - Medoid indices: {medoids}")
    print(f"       - Sample labels: {labels}")
    print(f"       - Expected: samples 0,1 in one cluster, samples 2,3 in another")
    print()


def test_cluster_time_series_demo():
    """Test clustering on synthetic time series data"""
    print("=" * 60)
    print("TEST 3: Time Series Clustering (Synthetic Data)")
    print("=" * 60)
    
    # Generate synthetic time series
    rng = np.random.default_rng(42)
    t = np.linspace(0, 2 * np.pi, 50)
    
    # 3 clusters with different patterns
    X = []
    # Cluster 1: sin(t)
    for _ in range(20):
        X.append(np.sin(t) + 0.05 * rng.standard_normal(len(t)))
    # Cluster 2: cos(t)
    for _ in range(20):
        X.append(np.cos(t) + 0.05 * rng.standard_normal(len(t)))
    # Cluster 3: sin(2t)
    for _ in range(20):
        X.append(np.sin(2 * t) + 0.05 * rng.standard_normal(len(t)))
    
    X = np.vstack(X)
    # Create ground truth labels BEFORE shuffling
    true_labels = np.array([0]*20 + [1]*20 + [2]*20)  # 20x sin, 20x cos, 20x sin2
    print(f"Generated data shape: {X.shape} (60 series, length 50)")
    print(f"Ground truth: 20x class 0 (sin), 20x class 1 (cos), 20x class 2 (sin2)")
    
    # Shuffle samples AND labels together to maintain correspondence
    shuffle_idx = rng.permutation(len(X))
    X = X[shuffle_idx]
    true_labels = true_labels[shuffle_idx]  # IMPORTANT: shuffle labels too!
    print("[INFO] Data shuffled randomly (labels maintained)")
    
    # Clustering
    result = cluster_time_series(
        X,
        k=3,
        n_trees=200,      # Increased from 50 to 200
        sample_size=256,   # Increased from 8 to 256
        normalize=True,
        random_state=42,
    )
    
    print("[PASS] cluster_time_series() successful")
    print(f"       - Medoid indices: {result.medoids}")
    print(f"       - Predicted labels: {result.labels}")
    print(f"       - Distance matrix shape: {result.distance_matrix.shape}")
    
    # Verify clustering quality
    n_clusters = len(np.unique(result.labels))
    print(f"       - Actual clusters: {n_clusters} (expected 3)")
    
    # EVALUATE: Compare predicted vs ground truth labels
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, confusion_matrix
    
    nmi = normalized_mutual_info_score(true_labels, result.labels)
    ari = adjusted_rand_score(true_labels, result.labels)
    conf_matrix = confusion_matrix(true_labels, result.labels)
    
    print(f"\n[CLUSTERING QUALITY - GROUND TRUTH COMPARISON]")
    print(f"       - Normalized Mutual Information (NMI): {nmi:.4f} (0-1, higher is better)")
    print(f"       - Adjusted Rand Index (ARI): {ari:.4f} (-1 to 1, higher is better)")
    print(f"       - Confusion Matrix:")
    print(f"         (rows=true_labels, cols=predicted_labels)")
    for row in conf_matrix:
        print(f"         {row}")
    
    # Compute intra-cluster distances
    print(f"\n[CLUSTER STATISTICS]")
    for cluster_id in range(3):
        cluster_mask = result.labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        if len(cluster_indices) > 0:
            if len(cluster_indices) > 1:
                cluster_dist = result.distance_matrix[np.ix_(cluster_indices, cluster_indices)]
                mask = ~np.eye(len(cluster_indices), dtype=bool)
                avg_intra_dist = np.mean(cluster_dist[mask])
            else:
                avg_intra_dist = 0.0
            print(f"       - Cluster {cluster_id}: {len(cluster_indices)} samples, avg intra-dist: {avg_intra_dist:.4f}")
    
    # Generate visualizations
    print("\n[INFO] Generating visualizations...")
    output_dir = Path(__file__).parent.parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("       [1/4] Plotting clustering results...")
        plot_clustering_results(
            X, result.labels, result.medoids,
            figsize=(12, 8),
            save_path=str(output_dir / "clustering_results.png"),
        )
        
        print("       [2/4] Plotting distance matrix...")
        plot_distance_matrix(
            result.distance_matrix, labels=result.labels,
            figsize=(10, 9),
            save_path=str(output_dir / "distance_matrix.png"),
        )
        
        print("       [3/4] Plotting medoids comparison...")
        plot_medoids_comparison(
            X, result.labels, result.medoids,
            figsize=(14, 4),
            save_path=str(output_dir / "medoids_comparison.png"),
        )
        
        print("       [4/4] Plotting cluster statistics...")
        plot_cluster_statistics(
            result.distance_matrix, result.labels,
            figsize=(12, 4),
            save_path=str(output_dir / "cluster_statistics.png"),
        )
        
        print(f"[PASS] All visualizations saved to: {output_dir}")
    except Exception as e:
        print(f"[WARNING] Visualization failed: {e}")
        print("         (Make sure matplotlib is installed)")
    
    print()


def test_different_series_lengths():
    """Test clustering with different series lengths"""
    print("=" * 60)
    print("TEST 4: Different Series Lengths")
    print("=" * 60)
    
    rng = np.random.default_rng(123)
    
    # Test different lengths
    for length in [30, 100, 200]:
        X = rng.standard_normal((10, length))
        result = cluster_time_series(X, k=2, n_trees=30, normalize=True, random_state=123)
        n_actual_clusters = len(np.unique(result.labels))
        print(f"[PASS] Length {length:3d}: clustered into {n_actual_clusters} cluster(s)")
    
    print()


def run_all_tests():
    """Run all tests"""
    print("\n" + "#" * 60)
    print("# Isolation Kernel + K-Medoids Clustering Test Suite")
    print("#" * 60 + "\n")
    
    try:
        test_isolation_kernel_basic()
        test_k_medoids_basic()
        test_cluster_time_series_demo()
        test_different_series_lengths()
        
        print("#" * 60)
        print("# ALL TESTS PASSED!")
        print("#" * 60 + "\n")
        return True
    except Exception as e:
        print("\n" + "#" * 60)
        print(f"# TEST FAILED: {e}")
        print("#" * 60 + "\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
