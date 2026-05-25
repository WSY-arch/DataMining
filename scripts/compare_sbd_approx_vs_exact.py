"""Compare Wang's approximate SBD vs exact SBD.

Goals
-----
1. Verify whether exact SBD really blows up memory on a moderately large UCR
   dataset (TwoPatterns, n=5000, L=128). The full N*N distance matrix is
   5000*5000*8 ~= 200 MB, which is large but not catastrophic; the real
   bottleneck is N*(N-1)/2 ~= 12.5M pairwise SBD calls, each costing O(L log L).
2. On small datasets (GunPoint, Coffee), compare the approximate SBD distance
   matrix against the exact SBD distance matrix:
   - Frobenius norm of the difference (raw and relative).
   - Mean per-pair absolute error on candidate pairs only.
   - K-medoids ARI/NMI gap across multiple seeds.

If ARI gap < 0.02, Wang's approximation is acceptable. Otherwise the main
experiment must use exact SBD.

The Wang approximate SBD code is COPIED below (verbatim from
origin/wang_week_2:tsclust/measures/similarity_measures.py) so this script
is reproducible without depending on Wang's branch being checked out.
"""

from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

# Allow running from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.chen_experiment_utils import (  # noqa: E402
    balanced_subsample,
    load_dataset,
)
from tsclust.clustering.k_medoids import k_medoids  # noqa: E402


# --------------------------------------------------------------------------
# Wang's approximate SBD implementation (verbatim copy with minimal renames).
# Source: origin/wang_week_2:tsclust/measures/similarity_measures.py
# --------------------------------------------------------------------------


def _paa_transform(X: np.ndarray, n_segments: int) -> np.ndarray:
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
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must have shape (n_samples, n_features)")
    n_samples = features.shape[0]
    if n_samples <= 1:
        return []
    candidate_k = max(1, min(int(candidate_k), n_samples - 1))
    effective_n_jobs = 1 if int(n_jobs) == -1 else int(n_jobs)
    nn = NearestNeighbors(
        n_neighbors=candidate_k + 1, metric="euclidean",
        n_jobs=effective_n_jobs,
    )
    nn.fit(features)
    max_mem_bytes = 200 * 1024 * 1024
    bytes_per_distance = 8
    est_c = max(1, int(max_mem_bytes / (max(1, n_samples) * bytes_per_distance)))
    chunk_size = min(n_samples, max(1, est_c))
    pairs: set[tuple[int, int]] = set()
    for start in range(0, n_samples, chunk_size):
        end = min(n_samples, start + chunk_size)
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


def sbd_distance_pair(x: np.ndarray, y: np.ndarray, standardize: bool = True) -> float:
    """SBD between two 1-D series. Verbatim from Wang (renamed)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if len(x) == 0 or len(y) == 0:
        raise ValueError("SBD inputs must be non-empty")
    if standardize:
        x_mean, x_std = np.mean(x), np.std(x)
        x_norm = x - x_mean if x_std == 0 else (x - x_mean) / x_std
        y_mean, y_std = np.mean(y), np.std(y)
        y_norm = y - y_mean if y_std == 0 else (y - y_mean) / y_std
    else:
        x_norm = x
        y_norm = y
    ncc = np.convolve(x_norm, y_norm[::-1], mode="full")
    if standardize:
        x_std_o = float(np.std(x))
        y_std_o = float(np.std(y))
        if x_std_o > 0 and y_std_o > 0:
            ncc = ncc / (len(x) * x_std_o * y_std_o)
        else:
            ncc = ncc / len(x)
    else:
        ncc = ncc / len(x)
    return float(np.clip(1.0 - np.max(ncc), 0.0, 2.0))


def wang_sbd_approx_distance_matrix(
    X: np.ndarray,
    candidate_k: int = 20,
    paa_segments: int = 32,
    n_jobs: int = 1,
    standardize: bool = True,
    fallback_distance: float = 2.0,
) -> tuple[np.ndarray, int]:
    """Wang's approximate SBD. Returns (matrix, n_candidate_pairs)."""
    X = np.asarray(X, dtype=float)
    n_samples = X.shape[0]
    dist = np.full((n_samples, n_samples), float(
        fallback_distance), dtype=float)
    np.fill_diagonal(dist, 0.0)
    coarse_features = _paa_transform(X, paa_segments)
    candidate_pairs = _candidate_pairs_from_features(
        coarse_features, candidate_k=candidate_k, n_jobs=n_jobs,
    )
    for i, j in candidate_pairs:
        value = sbd_distance_pair(X[i], X[j], standardize=standardize)
        dist[i, j] = value
        dist[j, i] = value
    return dist, len(candidate_pairs)


# --------------------------------------------------------------------------
# Exact SBD via aeon (preferred) with reference fallback.
# --------------------------------------------------------------------------


def exact_sbd_distance_matrix(X: np.ndarray, standardize: bool = True) -> np.ndarray:
    """Exact SBD pairwise distance matrix.

    Try aeon first (fast C impl), fall back to a Python double loop.
    """
    try:
        from aeon import distances as aeon_distances
        if hasattr(aeon_distances, "sbd_pairwise_distance"):
            return np.asarray(
                aeon_distances.sbd_pairwise_distance(
                    X, standardize=standardize),
                dtype=float,
            )
    except ModuleNotFoundError:
        pass
    n = X.shape[0]
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            v = sbd_distance_pair(X[i], X[j], standardize=standardize)
            dist[i, j] = v
            dist[j, i] = v
    return dist


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def zscore_normalize(X: np.ndarray) -> np.ndarray:
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std


def evaluate_clustering(
    dist: np.ndarray, y_true: np.ndarray, k: int, seeds: list[int],
) -> tuple[float, float]:
    aris: list[float] = []
    nmis: list[float] = []
    for seed in seeds:
        _, labels = k_medoids(dist, k=k, random_state=seed)
        aris.append(float(adjusted_rand_score(y_true, labels)))
        nmis.append(float(normalized_mutual_info_score(y_true, labels)))
    return float(np.mean(aris)), float(np.mean(nmis))


def compare_on_dataset(
    dataset_name: str,
    seeds: list[int],
    candidate_k: int,
    paa_segments: int,
    samples_per_class: int | None,
) -> dict[str, Any]:
    print(f"\n{'=' * 78}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'=' * 78}")
    X, y = load_dataset(dataset_name, data_source="aeon")
    if samples_per_class is not None and samples_per_class > 0:
        X, y = balanced_subsample(
            X, y, samples_per_class, subsample_seed=seeds[0])
    print(f"  shape={X.shape}  classes={len(np.unique(y))}")

    # Match cluster_time_series: z-normalise upstream, then standardize=False.
    X_norm = zscore_normalize(X)
    k = int(len(np.unique(y)))

    # Exact SBD
    t0 = time.perf_counter()
    D_exact = exact_sbd_distance_matrix(X_norm, standardize=False)
    t_exact = time.perf_counter() - t0
    print(f"  [exact ] runtime={t_exact:.2f}s   "
          f"min={D_exact.min():.4f} max={D_exact.max():.4f}")

    # Approximate SBD
    t0 = time.perf_counter()
    D_approx, n_cand = wang_sbd_approx_distance_matrix(
        X_norm,
        candidate_k=candidate_k,
        paa_segments=paa_segments,
        n_jobs=1,
        standardize=False,
        fallback_distance=2.0,
    )
    t_approx = time.perf_counter() - t0
    n_total_pairs = X.shape[0] * (X.shape[0] - 1) // 2
    coverage = n_cand / max(1, n_total_pairs)
    print(f"  [approx] runtime={t_approx:.2f}s   "
          f"candidate_pairs={n_cand}/{n_total_pairs} "
          f"({coverage:.1%})")

    # Distance matrix difference
    diff = D_approx - D_exact
    fro_diff = float(np.linalg.norm(diff, ord="fro"))
    fro_exact = float(np.linalg.norm(D_exact, ord="fro"))
    rel_fro = fro_diff / max(fro_exact, 1e-12)

    # Error restricted to candidate pairs (where approx actually computes SBD)
    mask = (D_approx < 2.0 - 1e-9) & ~np.eye(D_approx.shape[0], dtype=bool)
    if mask.any():
        cand_max_err = float(np.max(np.abs(diff[mask])))
        cand_mean_err = float(np.mean(np.abs(diff[mask])))
    else:
        cand_max_err = 0.0
        cand_mean_err = 0.0

    print(f"  [diff  ] frobenius={fro_diff:.4f}  rel={rel_fro:.4%}  "
          f"cand_pair_max={cand_max_err:.2e}  cand_pair_mean={cand_mean_err:.2e}")

    # Clustering
    ari_exact, nmi_exact = evaluate_clustering(D_exact, y, k, seeds)
    ari_approx, nmi_approx = evaluate_clustering(D_approx, y, k, seeds)
    print(f"  [exact ] mean ARI={ari_exact:.4f}  NMI={nmi_exact:.4f}")
    print(f"  [approx] mean ARI={ari_approx:.4f}  NMI={nmi_approx:.4f}")
    print(f"  [delta ] ARI gap={ari_approx - ari_exact:+.4f}  "
          f"NMI gap={nmi_approx - nmi_exact:+.4f}")

    return {
        "dataset": dataset_name,
        "n_samples": int(X.shape[0]),
        "series_length": int(X.shape[1]),
        "k": k,
        "exact_runtime": t_exact,
        "approx_runtime": t_approx,
        "candidate_pairs": n_cand,
        "total_pairs": n_total_pairs,
        "coverage": coverage,
        "fro_diff": fro_diff,
        "rel_fro_diff": rel_fro,
        "cand_pair_max_err": cand_max_err,
        "cand_pair_mean_err": cand_mean_err,
        "ari_exact": ari_exact,
        "ari_approx": ari_approx,
        "ari_gap": ari_approx - ari_exact,
        "nmi_exact": nmi_exact,
        "nmi_approx": nmi_approx,
    }


def memory_stress_test(
    dataset_name: str, samples_per_class: int | None, mode: str,
) -> None:
    print(f"\n{'=' * 78}")
    print(f"  Memory stress test: {dataset_name} (mode={mode})")
    print(f"{'=' * 78}")
    X, y = load_dataset(dataset_name, data_source="aeon")
    if samples_per_class is not None and samples_per_class > 0:
        X, y = balanced_subsample(X, y, samples_per_class, subsample_seed=1)
    print(f"  shape={X.shape}  predicted matrix size="
          f"{X.shape[0] ** 2 * 8 / 1024 / 1024:.1f} MB")
    X_norm = zscore_normalize(X)

    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        if mode == "exact":
            _ = exact_sbd_distance_matrix(X_norm, standardize=False)
        else:
            _, _ = wang_sbd_approx_distance_matrix(
                X_norm, candidate_k=20, paa_segments=32,
                n_jobs=1, standardize=False,
            )
        elapsed = time.perf_counter() - t0
        current, peak = tracemalloc.get_traced_memory()
        print(f"  [{mode}] runtime={elapsed:.2f}s  "
              f"peak_python_mem={peak / 1024 / 1024:.1f} MB")
    except MemoryError as exc:
        print(f"  [{mode}] MemoryError: {exc}")
    finally:
        tracemalloc.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["GunPoint", "Coffee"],
        help="Small datasets to compare exact vs approx SBD on.",
    )
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[1, 2, 3, 4, 5])
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--paa-segments", type=int, default=32)
    parser.add_argument("--samples-per-class", type=int, default=0,
                        help="0 = use full dataset (default).")
    parser.add_argument(
        "--memory-stress",
        nargs="*",
        default=[],
        help="Datasets to stress-test for memory (e.g. TwoPatterns).",
    )
    parser.add_argument(
        "--memory-stress-mode",
        choices=["exact", "approx", "both"],
        default="both",
    )
    args = parser.parse_args()

    samples_per_class = args.samples_per_class if args.samples_per_class > 0 else None

    results = []
    for ds in args.datasets:
        try:
            row = compare_on_dataset(
                ds,
                seeds=args.seeds,
                candidate_k=args.candidate_k,
                paa_segments=args.paa_segments,
                samples_per_class=samples_per_class,
            )
            results.append(row)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {ds}: {exc!r}")

    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(f"{'dataset':<22}{'rel_fro':>10}{'ARI_exact':>11}"
          f"{'ARI_approx':>12}{'ARI_gap':>10}{'verdict':>14}")
    for r in results:
        verdict = "ACCEPTABLE" if abs(r["ari_gap"]) < 0.02 else "REJECT"
        print(f"{r['dataset']:<22}{r['rel_fro_diff']:>10.4%}"
              f"{r['ari_exact']:>11.4f}{r['ari_approx']:>12.4f}"
              f"{r['ari_gap']:>+10.4f}{verdict:>14}")

    for ds in args.memory_stress:
        modes = ["exact", "approx"] if args.memory_stress_mode == "both" else [
            args.memory_stress_mode]
        for mode in modes:
            try:
                memory_stress_test(ds, samples_per_class, mode)
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] memory_stress {ds}/{mode}: {exc!r}")


if __name__ == "__main__":
    main()
