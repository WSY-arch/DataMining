"""Shared perturbation primitives for Part 2 robustness experiments.

These three functions are byte-exact contracts between Chen and Wang.
Any divergence in implementation will make the resulting CSVs incomparable.

Reference: refs/Undermind - Mechanism-oriented comparisons of time-series
clustering similarity measures across noise misalignment and length.pdf
Design notes: docs/methodology/perturbation_design.md

Locked design choices (Chen+Wang 2026-05 collab):
    1. noise scale  : per-series std (level is RELATIVE sigma multiplier).
                      On a z-normalized pipeline std=1 so sigma == level
                      numerically, but the relative formulation must be kept
                      in writing for cross-pipeline reproducibility.
    2. RNG          : np.random.default_rng (PCG64), NOT legacy np.random.seed.
    3. shift sample : per-series independent integer in [-max_shift, +max_shift].
                      The realized shifts are returned alongside the data so
                      downstream analysis can correlate ARI with actual shift
                      magnitude (storing only the seed is fragile).
    4. shift mode   : default 'padding' (edge-fill); 'circular' (np.roll) is
                      reported as ablation only. Rationale: SBD is by
                      construction invariant to circular shifts of its inputs
                      (1 - max_w NCC_w over all circular alignments), so a
                      circular default would trivialize SBD's shift-robustness
                      curve into a tautology. CBF/Trace/ECG200 are also not
                      genuinely periodic.
    5. length op    : two-sided symmetric truncate + scipy.signal.resample
                      (FFT-based) back to L. Equal removal from head/tail
                      preserves the temporal centre of mass; zero-padding is
                      forbidden because on z-normalized data it injects an
                      artificial step edge that confounds the ablation.

The accompanying fixture in tests/test_perturbations.py pins the byte-exact
output of each function to a sha256 hash. Both collaborators MUST keep the
test green.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "add_gaussian_noise",
    "random_global_shift",
    "truncate_and_resample",
]


def add_gaussian_noise(X: np.ndarray, level: float, seed: int) -> np.ndarray:
    """Add Gaussian noise scaled by per-series std.

    For each series x_i we draw eps_i ~ N(0, (level * std(x_i))^2 * I)
    and return X + eps. Setting level=0 returns a copy unchanged.

    On a z-normalized pipeline std(x_i) == 1 so sigma numerically equals
    `level`, but the relative formulation must still be reported in the
    paper so non-z-normalized re-implementations recover the intended
    behaviour.

    Parameters
    ----------
    X : (n, L) float array
    level : float
        Relative sigma. e.g. level=0.2 means noise std is 20% of signal std.
    seed : int
        Forwarded to np.random.default_rng (PCG64). Same seed -> same output.
    """
    if level == 0:
        return X.copy()
    rng = np.random.default_rng(seed)
    scale = np.std(X, axis=1, keepdims=True)
    scale[scale == 0.0] = 1.0
    return X + rng.normal(0.0, level * scale, size=X.shape)


def random_global_shift(
    X: np.ndarray,
    max_shift: int,
    seed: int,
    mode: str = "padding",
) -> tuple[np.ndarray, np.ndarray]:
    """Apply per-series random temporal shift sampled in [-max_shift, +max_shift].

    Each series gets an INDEPENDENT integer shift drawn from
    rng.integers(-max_shift, max_shift + 1) (inclusive both ends).
    Setting max_shift=0 returns a copy unchanged together with an
    all-zero shift vector.

    Returns
    -------
    shifted : (n, L) float array
        The perturbed series.
    shift_amounts : (n,) int array
        The realized integer shift applied to each series. Persist this
        alongside the perturbed data so downstream analysis can stratify
        ARI by actual shift magnitude. Recovering shift_amounts from a
        seed alone requires re-simulating the entire RNG draw order,
        which is brittle if the code path changes.

    Modes
    -----
    padding (default)
        Shift the contents and replicate the boundary value (`row[0]` on
        the head when shift > 0, `row[-1]` on the tail when shift < 0).
        This is the default because:

        - CBF, Trace, ECG200 are not periodic; circular wrap-around grafts
          semantically incompatible regions onto each other (e.g. for
          ECG200 it would attach the post-T recovery to the front of
          the P-wave).
        - SBD is defined as `1 - max_w NCC_w` over circular alignments
          and is therefore exactly invariant under circular shifts of
          either input. A circular-default protocol would yield zero
          degradation for SBD irrespective of the shift magnitude — that
          is a property of the metric definition, not an empirical
          finding, and would render the shift-robustness comparison
          across measures meaningless.

    circular
        np.roll wrap-around. Reserved as an ablation that explicitly
        showcases SBD's intrinsic translation invariance; do not use as
        the main protocol.

    zero-fill is forbidden (a step discontinuity at the boundary on
    z-normalized data would conflate length/edge artifacts with the
    shift perturbation).
    """
    n = X.shape[0]
    if max_shift == 0:
        return X.copy(), np.zeros(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    shifted = np.empty_like(X)
    shift_amounts = np.empty(n, dtype=np.int64)
    for idx, row in enumerate(X):
        shift = int(rng.integers(-max_shift, max_shift + 1))
        shift_amounts[idx] = shift
        if mode == "circular":
            shifted[idx] = np.roll(row, shift)
        else:  # padding (edge-fill)
            if shift > 0:
                shifted[idx, shift:] = row[:-shift]
                shifted[idx, :shift] = row[0]
            elif shift < 0:
                shifted[idx, :shift] = row[-shift:]
                shifted[idx, shift:] = row[-1]
            else:
                shifted[idx] = row
    return shifted, shift_amounts


def truncate_and_resample(X: np.ndarray, keep_fraction: float) -> np.ndarray:
    """Two-sided symmetric truncate + FFT-resample back to original length.

    Pipeline
    --------
    1. keep = max(2, round(L * keep_fraction))
    2. drop = L - keep; head_drop = drop // 2 (floor),
       tail_drop = drop - head_drop (ceil when drop is odd).
    3. truncated = X[:, head_drop : head_drop + keep]
    4. resample back to length L via scipy.signal.resample (FFT-based).

    Setting keep_fraction >= 1.0 returns a copy unchanged.

    Why two-sided
    -------------
    Equal removal from both ends preserves the signal's temporal centre
    of mass. For CBF (centred bell/cylinder/funnel transients) and
    ECG200 (QRS complex centred in the heartbeat segment), one-sided
    truncation would systematically destroy more of the discriminative
    region than the baseline. For Trace (4-class transient onsets and
    offsets), two-sided is also defensible since both ends carry
    information.

    Why FFT resample, not linear interpolation
    ------------------------------------------
    scipy.signal.resample is FFT-based, exact for band-limited signals
    and near-optimal for smooth signals. Linear interpolation would
    low-pass-filter the signal more aggressively, conflating "shorter
    series" with "smoother series".

    Why max(2, ...)
    ---------------
    Length 1 series cannot support any pairwise metric (ED degenerates,
    DTW has no path, SBD has no cross-correlation).

    Why no zero-padding
    -------------------
    On z-normalized data zero-pad introduces a step discontinuity at
    the boundary equal to the original endpoint value, conflating the
    length perturbation with an artificial edge artifact and breaking
    the ablation.
    """
    if keep_fraction >= 1.0:
        return X.copy()
    from scipy.signal import resample as scipy_resample
    length = X.shape[1]
    keep = max(2, int(round(length * keep_fraction)))
    drop = length - keep
    head_drop = drop // 2
    truncated = X[:, head_drop : head_drop + keep]
    return scipy_resample(truncated, length, axis=1)
