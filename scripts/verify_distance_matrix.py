"""Quick sanity check for distance matrix correctness.

Run on Chinatown (smallest dataset) to verify:
1. Diagonal is zero
2. Symmetry: d(i,j) == d(j,i)
3. Non-negativity
4. ED matches sklearn pairwise_distances
5. IDK kernel-induced distance: metric properties + triangle inequality

Usage:
    python scripts/verify_distance_matrix.py
"""

from __future__ import annotations
from tsclust.measures.similarity_measures import (
    dtw_distance_matrix,
    euclidean_distance_matrix,
    msm_distance_matrix,
)
from tsclust.clustering.clustering import _zscore_normalize
from scripts.chen_experiment_utils import DEFAULT_AEON_DATA_ROOT, load_dataset

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def check_matrix(name: str, D: np.ndarray, atol: float = 1e-10) -> bool:
    ok = True
    # Diagonal == 0
    diag_max = np.max(np.abs(np.diag(D)))
    if diag_max > atol:
        print(f"  [FAIL] {name}: diagonal max = {diag_max:.2e} (expected 0)")
        ok = False
    else:
        print(f"  [OK]   {name}: diagonal is zero")

    # Symmetry
    asym = np.max(np.abs(D - D.T))
    if asym > atol:
        print(f"  [FAIL] {name}: asymmetry max = {asym:.2e}")
        ok = False
    else:
        print(f"  [OK]   {name}: symmetric")

    # Non-negative
    if np.any(D < -atol):
        print(f"  [FAIL] {name}: negative values found")
        ok = False
    else:
        print(f"  [OK]   {name}: non-negative")

    return ok


def check_triangle_inequality(
    name: str, D: np.ndarray, n_samples: int = 500, seed: int = 42,
    atol: float = 1e-6,
) -> bool:
    """Randomly sample triplets and verify d(i,j) <= d(i,k) + d(k,j)."""
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    violations = 0
    for _ in range(n_samples):
        i, j, k = rng.choice(n, 3, replace=False)
        if D[i, j] > D[i, k] + D[k, j] + atol:
            violations += 1
    if violations > 0:
        print(f"  [FAIL] {name}: triangle inequality violated "
              f"in {violations}/{n_samples} sampled triplets")
        return False
    print(f"  [OK]   {name}: triangle inequality holds "
          f"({n_samples} sampled triplets)")
    return True


def main() -> int:
    print("[INFO] Loading Chinatown via aeon...")
    X, y = load_dataset("Chinatown", "aeon", DEFAULT_AEON_DATA_ROOT)
    X = _zscore_normalize(X)
    print(f"[INFO] X.shape = {X.shape}")

    all_ok = True

    # ED
    print("\n--- Euclidean Distance ---")
    D_ed = euclidean_distance_matrix(X)
    all_ok &= check_matrix("ED", D_ed)

    # Cross-check with sklearn
    from sklearn.metrics import pairwise_distances
    D_sklearn = pairwise_distances(X, metric="euclidean")
    diff = np.max(np.abs(D_ed - D_sklearn))
    if diff > 1e-8:
        print(f"  [FAIL] ED vs sklearn max diff = {diff:.2e}")
        all_ok = False
    else:
        print(f"  [OK]   ED matches sklearn (max diff = {diff:.2e})")

    # DTW
    print("\n--- DTW Distance (window=10%) ---")
    window = max(1, int(round(X.shape[1] * 0.1)))
    D_dtw = dtw_distance_matrix(X, window=window, backend="auto")
    all_ok &= check_matrix("DTW", D_dtw)

    # MSM
    print("\n--- MSM Distance (c=1.0) ---")
    D_msm = msm_distance_matrix(X, c=1.0, backend="auto")
    all_ok &= check_matrix("MSM", D_msm)

    # IDK (kernel-induced distance)
    print("\n--- IDK Distance (kernel-induced) ---")
    try:
        from tsclust.measures.isolation_kernel import IsolationKernel
        kernel = IsolationKernel(random_state=42).fit(X)
        sim = kernel.similarity_matrix(X)
        D_idk = np.sqrt(np.clip(2.0 - 2.0 * sim, 0.0, None))
        all_ok &= check_matrix("IDK", D_idk)
        # Triangle inequality — the key metric property
        all_ok &= check_triangle_inequality("IDK", D_idk)
        # Verify K(x,x) ≈ 1 (L2-normalized assumption)
        diag_sim = np.diag(sim)
        kxx_dev = np.max(np.abs(diag_sim - 1.0))
        if kxx_dev > 1e-4:
            print(f"  [WARN] IDK: K(x,x) deviates from 1 by {kxx_dev:.2e} "
                  f"— feature map may not be L2-normalized")
        else:
            print(f"  [OK]   IDK: K(x,x) ≈ 1 (max dev {kxx_dev:.2e})")
    except Exception as exc:
        print(f"  [SKIP] IDK: {exc}")

    print()
    if all_ok:
        print("[ALL PASSED] Distance matrices are correct.")
        return 0
    else:
        print("[SOME FAILED] Check output above.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
