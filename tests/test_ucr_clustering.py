"""
Test clustering with UCR Time Series datasets.
Compares ground truth labels with clustering predictions.
Supports loading from locally downloaded UCR data files in TSV format.
"""

from tsclust.visualization import (
    plot_clustering_results,
    plot_distance_matrix,
    plot_medoids_comparison,
    plot_cluster_statistics,
)
from tsclust.clustering import cluster_time_series
import numpy as np
import sys
import time
from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, confusion_matrix

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_ucr_from_file(train_file, test_file=None):
    """
    Load UCR time series data from locally downloaded data files.

    Format: label followed by time series values (space-separated or tab-separated)
    Values may be in scientific notation (e.g., 1.7309910e+00)

    Args:
        train_file: Path to train data file (.txt or .tsv)
        test_file: Path to test data file (optional, for combining)

    Returns:
        Tuple of (X, y) where X is (n_samples, series_length)
    """
    def load_data_file(file_path):
        """Load UCR format data file (space or tab separated)"""
        data = []
        labels = []

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Split by both tabs and multiple spaces
                values = line.split()
                if not values:
                    continue

                try:
                    label = int(float(values[0]))
                    series = np.array([float(v) for v in values[1:]])

                    if len(series) > 0:  # Only keep non-empty series
                        labels.append(label)
                        data.append(series)
                except (ValueError, IndexError):
                    continue  # Skip malformed lines

        return data, labels

    # Load train data
    X_train, y_train = load_data_file(train_file)
    print(f"[OK] Loaded train data from {Path(train_file).name}")
    print(f"     Train: {len(X_train)} samples")

    # Load test data if provided
    if test_file:
        X_test, y_test = load_data_file(test_file)
        X_data = X_train + X_test
        y_data = y_train + y_test
        print(f"[OK] Loaded test data from {Path(test_file).name}")
        print(f"     Test: {len(X_test)} samples")
        print(f"     Combined: {len(X_data)} samples")
    else:
        X_data = X_train
        y_data = y_train

    # Handle variable-length series by padding
    max_len = max(len(s) for s in X_data)
    X = np.zeros((len(X_data), max_len))
    for i, series in enumerate(X_data):
        X[i, :len(series)] = series

    y = np.array(y_data)

    return X, y


def test_ucr_dataset(
    train_file,
    test_file=None,
    n_samples=None,
    normalize=True,
    k=None,
    generate_viz=True,
    output_root=None,
    viz_dir=None,
    similarity_metric="idk",
    window_size=None,
    window_step=None,
    n_trees=200,
    sample_size=256,
    random_state=42,
    similarity_params=None,
    return_details=False,
):
    """
    Test clustering on UCR time series data from local files.

    Args:
        train_file: Path to train data file
        test_file: Path to test data file (optional)
        n_samples: Number of samples to use (None = all)
        normalize: Whether to apply z-score normalization
        k: Number of clusters. If None, uses ground truth number (supervised).
           Specify a number for true unsupervised clustering.
        generate_viz: Whether to generate visualization plots (default: True)
        output_root: Root directory for saved visualizations. Defaults to project/results.
        viz_dir: Exact visualization output directory. Overrides output_root if provided.
        similarity_metric: Similarity metric name. Defaults to "idk".
        window_size: Sliding window size for IDK-based time series representation.
        window_step: Step size between successive sliding windows.
        n_trees: Number of isolation trees used by the IDK backend.
        sample_size: Number of samples per tree used by the IDK backend.
        similarity_params: Extra keyword arguments reserved for future metrics.
        return_details: If True, returns (success, details_dict).
    """
    dataset_name = Path(train_file).stem.replace('_TRAIN', '')

    print("=" * 70)
    print(f"TEST: UCR Dataset - {dataset_name}")
    print("=" * 70)

    # Load dataset
    print(f"\n[STEP 1] Loading {dataset_name} dataset from local files...")
    try:
        X, y_true = load_ucr_from_file(train_file, test_file)
    except Exception as e:
        print(f"[FAIL] Could not load dataset: {e}")
        return False

    # Use subset if requested
    if n_samples and n_samples < len(X):
        rng = np.random.default_rng(42)
        indices = rng.choice(len(X), n_samples, replace=False)
        X = X[indices]
        y_true = y_true[indices]
        print(f"[INFO] Using subset: {n_samples} samples")

    print(f"       X shape: {X.shape} (samples × time points)")
    print(f"       y shape: {y_true.shape}")
    print(f"       Classes: {np.unique(y_true)}")
    classes, counts = np.unique(y_true, return_counts=True)
    print(
        f"       Class distribution: {dict(zip(classes.tolist(), counts.tolist()))}")

    # Determine number of clusters
    if k is None:
        # Use ground truth (supervised - for validation only)
        k = len(np.unique(y_true))
        print(f"\n[INFO] SUPERVISED MODE: k determined from ground truth")
    else:
        print(f"\n[INFO] UNSUPERVISED MODE: k specified by user")

    print(f"\n[STEP 2] Clustering with k={k}...")

    # Perform clustering
    start_time = time.perf_counter()
    try:
        result = cluster_time_series(
            X,
            k=k,
            n_trees=n_trees,
            sample_size=sample_size,
            normalize=normalize,
            random_state=random_state,
            window_size=window_size,
            window_step=window_step,
            similarity_metric=similarity_metric,
            similarity_params=similarity_params,
        )
        print(f"[OK] Clustering completed")
        print(f"     Predicted labels: {np.unique(result.labels)}")
        print(f"     Similarity metric: {similarity_metric}")
        if window_size is not None:
            print(f"     Window size: {window_size}")
        if window_step is not None:
            print(f"     Window step: {window_step}")
    except Exception as e:
        print(f"[FAIL] Clustering failed: {e}")
        import traceback
        traceback.print_exc()
        if return_details:
            return False, {}
        return False
    elapsed = time.perf_counter() - start_time
    print(f"     Runtime: {elapsed:.2f}s")

    # Evaluate clustering quality
    print(f"\n[STEP 3] Evaluating clustering quality...")

    nmi = normalized_mutual_info_score(y_true, result.labels)
    ari = adjusted_rand_score(y_true, result.labels)
    conf_matrix = confusion_matrix(y_true, result.labels)

    print(f"       Normalized Mutual Information (NMI): {nmi:.4f}")
    print(f"       Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"       Confusion Matrix:")
    for row in conf_matrix:
        print(f"         {row}")

    # Compute cluster statistics
    print(f"\n[STEP 4] Cluster Statistics:")
    for cluster_id in range(k):
        cluster_mask = result.labels == cluster_id
        cluster_count = np.sum(cluster_mask)
        if cluster_count > 0:
            if cluster_count > 1:
                cluster_dist = result.distance_matrix[np.ix_(
                    np.where(cluster_mask)[0],
                    np.where(cluster_mask)[0]
                )]
                mask = ~np.eye(cluster_count, dtype=bool)
                avg_intra_dist = np.mean(cluster_dist[mask])
            else:
                avg_intra_dist = 0.0

            print(f"       Cluster {cluster_id}: {cluster_count} samples, "
                  f"avg intra-dist: {avg_intra_dist:.4f}")

    # Generate visualizations
    if generate_viz:
        print(f"\n[STEP 5] Generating visualizations...")
        metric_tag = similarity_metric.lower().strip()
        if viz_dir is not None:
            output_dir = Path(viz_dir)
        else:
            results_root = Path(output_root) if output_root is not None else Path(
                __file__).parent.parent / "results"
            output_dir = results_root / dataset_name / \
                f"{dataset_name}_{metric_tag}_viz"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create metric display name
        metric_display = similarity_metric.upper() if similarity_metric.lower(
        ) == "idk" else similarity_metric.capitalize()

        try:
            print(f"       [1/4] Plotting clustering results...")
            plot_clustering_results(
                X, result.labels, result.medoids,
                title=f"Time Series Clustering Results - {metric_display} Similarity",
                figsize=(14, 10),
                save_path=str(output_dir / "clustering_results.png"),
            )

            print(f"       [2/4] Plotting distance matrix...")
            plot_distance_matrix(
                result.distance_matrix, labels=result.labels,
                title=f"Distance Matrix Heatmap - {metric_display} Similarity",
                figsize=(12, 11),
                save_path=str(output_dir / "distance_matrix.png"),
            )

            print(f"       [3/4] Plotting medoids comparison...")
            plot_medoids_comparison(
                X, result.labels, result.medoids,
                title=f"Medoids Comparison - {metric_display} Similarity",
                figsize=(16, 5),
                save_path=str(output_dir / "medoids_comparison.png"),
            )

            print(f"       [4/4] Plotting cluster statistics...")
            plot_cluster_statistics(
                result.distance_matrix, result.labels,
                title=f"Clustering Statistics - {metric_display} Similarity",
                figsize=(14, 5),
                save_path=str(output_dir / "cluster_statistics.png"),
            )

            print(f"[OK] All visualizations saved to: {output_dir}")
        except Exception as e:
            print(f"[WARNING] Visualization failed: {e}")
    else:
        print(f"\n[INFO] Skipping visualization generation (use --no-viz)")

    # Overall result
    print("\n" + "=" * 70)
    if nmi > 0.9 and ari > 0.9:
        print("[PASS] Excellent clustering quality (NMI > 0.9, ARI > 0.9)")
    elif nmi > 0.5 and ari > 0.5:
        print("[PASS] Good clustering quality (NMI > 0.5, ARI > 0.5)")
    else:
        print("[PASS] Clustering completed (may need parameter tuning)")
    print("=" * 70)

    details = {
        "metric": similarity_metric,
        "k": k,
        "nmi": float(nmi),
        "ari": float(ari),
        "runtime_sec": float(elapsed),
        "n_samples": int(X.shape[0]),
        "series_length": int(X.shape[1]),
    }

    if return_details:
        return True, details
    return True


def compare_idk_vs_euclidean(
    train_file,
    test_file=None,
    n_samples=None,
    normalize=True,
    k=None,
    generate_viz=False,
    output_root=None,
    viz_dir=None,
    window_size=None,
    window_step=None,
    n_trees=200,
    sample_size=256,
):
    """Run side-by-side comparison between IDK and Euclidean similarity metrics."""
    print("\n" + "#" * 70)
    print("# COMPARISON: IDK vs EUCLIDEAN")
    print("#" * 70)

    metric_runs = [
        {"name": "idk", "n_trees": n_trees, "sample_size": sample_size},
        {"name": "euclidean", "n_trees": n_trees, "sample_size": sample_size},
    ]
    results = []

    for run in metric_runs:
        ok, details = test_ucr_dataset(
            train_file=train_file,
            test_file=test_file,
            n_samples=n_samples,
            normalize=normalize,
            k=k,
            generate_viz=generate_viz,
            output_root=output_root,
            viz_dir=viz_dir,
            similarity_metric=run["name"],
            window_size=window_size,
            window_step=window_step,
            n_trees=run["n_trees"],
            sample_size=run["sample_size"],
            similarity_params=None,
            return_details=True,
        )
        if not ok:
            print(f"[FAIL] Comparison aborted at metric={run['name']}")
            return False
        results.append(details)

    idk_res = next(r for r in results if r["metric"] == "idk")
    eu_res = next(r for r in results if r["metric"] == "euclidean")

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<12} {'NMI':>10} {'ARI':>10} {'Runtime(s)':>12}")
    print("-" * 70)
    for row in results:
        print(
            f"{row['metric']:<12} {row['nmi']:>10.4f} {row['ari']:>10.4f} {row['runtime_sec']:>12.2f}"
        )
    print("-" * 70)
    print(f"ΔNMI (IDK - Euclidean): {idk_res['nmi'] - eu_res['nmi']:+.4f}")
    print(f"ΔARI (IDK - Euclidean): {idk_res['ari'] - eu_res['ari']:+.4f}")
    print(
        f"ΔTime(IDK - Euclidean): {idk_res['runtime_sec'] - eu_res['runtime_sec']:+.2f}s")
    print("=" * 70)

    if idk_res["nmi"] > eu_res["nmi"] and idk_res["ari"] > eu_res["ari"]:
        print("[RESULT] IDK performs better on both NMI and ARI.")
    elif idk_res["nmi"] < eu_res["nmi"] and idk_res["ari"] < eu_res["ari"]:
        print("[RESULT] Euclidean performs better on both NMI and ARI.")
    else:
        print("[RESULT] Mixed outcome: one metric is better in NMI, the other in ARI.")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test clustering on locally downloaded UCR datasets",
        epilog="""
Examples:
  python test_ucr_clustering.py --train ./data/GunPoint_TRAIN.tsv --test ./data/GunPoint_TEST.tsv
  python test_ucr_clustering.py --train ./data/ECG200_TRAIN.tsv --test ./data/ECG200_TEST.tsv --n-samples 100
        """
    )
    parser.add_argument("--train", type=str, required=True,
                        help="Path to train TSV file (required)")
    parser.add_argument("--test", type=str, default=None,
                        help="Path to test TSV file (optional)")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of samples to use (default: all)")
    parser.add_argument("--k", type=int, default=None,
                        help="Number of clusters. If not specified, uses ground truth k (supervised).")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Disable z-score normalization")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization generation (faster)")
    parser.add_argument("--similarity-metric", type=str, default="idk",
                        help="Similarity metric to use (default: idk; reserved for future comparison experiments)")
    parser.add_argument("--window-size", type=int, default=None,
                        help="Sliding window size for IDK-based time series representation")
    parser.add_argument("--window-step", type=int, default=None,
                        help="Stride between successive sliding windows (default: 1)")
    parser.add_argument("--n-trees", type=int, default=200,
                        help="Number of trees used by the IDK backend")
    parser.add_argument("--sample-size", type=int, default=256,
                        help="Sample size used by the IDK backend")
    parser.add_argument("--compare-metrics", action="store_true",
                        help="Run side-by-side comparison between idk and euclidean")

    args = parser.parse_args()

    # Verify train file exists
    if not Path(args.train).exists():
        print(f"[ERROR] Train file not found: {args.train}")
        sys.exit(1)

    # Verify test file exists if provided
    if args.test and not Path(args.test).exists():
        print(f"[ERROR] Test file not found: {args.test}")
        sys.exit(1)

    if args.compare_metrics:
        success = compare_idk_vs_euclidean(
            train_file=args.train,
            test_file=args.test,
            n_samples=args.n_samples,
            normalize=not args.no_normalize,
            k=args.k,
            generate_viz=not args.no_viz,
            window_size=args.window_size,
            window_step=args.window_step,
            n_trees=args.n_trees,
            sample_size=args.sample_size,
        )
    else:
        success = test_ucr_dataset(
            train_file=args.train,
            test_file=args.test,
            n_samples=args.n_samples,
            normalize=not args.no_normalize,
            k=args.k,
            generate_viz=not args.no_viz,
            similarity_metric=args.similarity_metric,
            window_size=args.window_size,
            window_step=args.window_step,
            n_trees=args.n_trees,
            sample_size=args.sample_size,
        )
    sys.exit(0 if success else 1)
