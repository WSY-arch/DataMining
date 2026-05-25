"""Byte-exact fixture tests for tsclust.perturbations.

These tests are the contract between Chen and Wang. Both sides must produce
the exact same numpy arrays (down to the last bit) when given the same
synthetic input. Any drift breaks CSV comparability across collaborators.

If a test fails after a deliberate algorithm change, regenerate the golden
hashes by running:

    python -c "from tests.test_perturbations import _print_golden; _print_golden()"

and update GOLDEN below; commit the change with a message that explains why.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tsclust.perturbations import (
    add_gaussian_noise,
    random_global_shift,
    truncate_and_resample,
)


# Deterministic synthetic input: 8 series x length 64, sine + small Gaussian
# noise. Built with default_rng(0) so it is reproducible across machines.
def _make_input() -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 4.0 * np.pi, 64)
    X = np.stack(
        [np.sin(t + i * 0.3) + 0.1 * rng.standard_normal(64) for i in range(8)]
    ).astype(np.float64)
    return X


def _sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


# Golden hashes captured on numpy 1.26 / scipy 1.12 with default_rng(PCG64).
# DO NOT edit unless the underlying algorithm intentionally changes.
# Last regenerated: 2026-05-21 after Choice 4 (default mode flip) and
# Choice 5 (two-sided symmetric truncation) were locked.
INPUT_HASH = "ab8faed67d030fc0b1f73c293c8080ddc45181ed280e48f8c46daac8d07d9ed6"

GOLDEN: dict[str, str] = {
    # noise: (level, seed) -> hash
    "noise_l0.2_s42": "9eb2e18489952aab22d8e28478fcc3ed2d739ca30f6a554cdfa1961ae7da4794",
    "noise_l0.4_s7":  "efed1e593bf461466a7f9c56950fd3ea023f2461d33dbf32a8f8b99ca3dd418e",
    # shift: (max_shift, seed, mode) -> hash
    "shift_5_s42_circular":  "189d907e5d1c6b08e713b5f6c74225ed183bc222837bc8b4ea1ad5d6ca5fa98b",
    "shift_10_s7_circular":  "b620b42b6c8113835ce8f89d99c0ebfb0560048e20bf35fc2c63dfc228df6a6d",
    "shift_5_s42_padding":   "022bce2cfece0b1cdba7cc7e61b9539e64a81ef6c71423bed0d53d64b0727709",
    # length: keep_fraction -> hash (two-sided symmetric truncate + FFT resample)
    "len_0.75": "a60179692b747229f70b6e0b3f8627baf4ffea198296d4a96af7c4e29c677729",
    "len_0.5":  "57adac6a46f1c2896f6708c32b1f109e7d48f5424dfcf806c1e5aca04bde7966",
    "len_0.25": "8dde241a95d31afb04436cf9884b7d4f5fc3c39b0c6efba3f689337d140162ba",
}

# Realized shift_amounts for the (max_shift=5, seed=42) cell. Pinned so the
# downstream "ARI vs actual shift magnitude" analysis is reproducible.
GOLDEN_SHIFT_AMOUNTS_5_42 = [-5, 3, 2, -1, -1, 4, -5, 2]


def test_input_fixture_is_stable():
    """Synthetic input itself must be byte-stable across machines."""
    assert _sha(_make_input()) == INPUT_HASH


def test_noise_level_zero_is_identity():
    X = _make_input()
    out = add_gaussian_noise(X, 0.0, seed=999)
    assert _sha(out) == INPUT_HASH
    # also must not mutate the input
    assert out is not X


@pytest.mark.parametrize(
    "level,seed,key",
    [
        (0.2, 42, "noise_l0.2_s42"),
        (0.4, 7,  "noise_l0.4_s7"),
    ],
)
def test_noise_byte_exact(level: float, seed: int, key: str):
    X = _make_input()
    out = add_gaussian_noise(X, level, seed)
    assert _sha(out) == GOLDEN[key], f"add_gaussian_noise drift on {key}"


def test_shift_zero_is_identity():
    X = _make_input()
    out, amounts = random_global_shift(X, 0, seed=999)
    assert _sha(out) == INPUT_HASH
    assert out is not X
    # max_shift=0 must yield an all-zero shift vector of the right shape.
    assert amounts.shape == (X.shape[0],)
    assert np.all(amounts == 0)


@pytest.mark.parametrize(
    "max_shift,seed,mode,key",
    [
        (5,  42, "circular", "shift_5_s42_circular"),
        (10, 7,  "circular", "shift_10_s7_circular"),
        (5,  42, "padding",  "shift_5_s42_padding"),
    ],
)
def test_shift_byte_exact(max_shift: int, seed: int, mode: str, key: str):
    X = _make_input()
    out, amounts = random_global_shift(X, max_shift, seed, mode=mode)
    assert _sha(out) == GOLDEN[key], f"random_global_shift drift on {key}"
    # Shape and dtype contract for the realized shifts.
    assert amounts.shape == (X.shape[0],)
    assert amounts.dtype == np.int64
    # Every realized shift must lie within the requested bound.
    assert np.all(np.abs(amounts) <= max_shift)


def test_shift_default_mode_is_padding():
    """Choice 4: omitting `mode=` must behave as padding (not circular)."""
    X = _make_input()
    default_out, _ = random_global_shift(X, 5, seed=42)
    padding_out, _ = random_global_shift(X, 5, seed=42, mode="padding")
    assert _sha(default_out) == _sha(padding_out)
    # And it must NOT be circular.
    circular_out, _ = random_global_shift(X, 5, seed=42, mode="circular")
    assert _sha(default_out) != _sha(circular_out)


def test_shift_amounts_pinned():
    """The realized shifts for (max_shift=5, seed=42) are part of the contract."""
    X = _make_input()
    _, amounts = random_global_shift(X, 5, seed=42, mode="padding")
    assert amounts.tolist() == GOLDEN_SHIFT_AMOUNTS_5_42


def test_length_one_is_identity():
    X = _make_input()
    out = truncate_and_resample(X, 1.0)
    assert _sha(out) == INPUT_HASH
    assert out is not X


@pytest.mark.parametrize(
    "frac,key",
    [
        (0.75, "len_0.75"),
        (0.5,  "len_0.5"),
        (0.25, "len_0.25"),
    ],
)
def test_length_byte_exact(frac: float, key: str):
    X = _make_input()
    out = truncate_and_resample(X, frac)
    assert out.shape == X.shape, "FFT-resample must return original length"
    assert _sha(out) == GOLDEN[key], f"truncate_and_resample drift on {key}"


def _print_golden() -> None:
    """Convenience helper to regenerate golden hashes after a deliberate change."""
    X = _make_input()
    print(f"INPUT_HASH = \"{_sha(X)}\"")
    print("GOLDEN = {")
    for level, seed in [(0.2, 42), (0.4, 7)]:
        h = _sha(add_gaussian_noise(X, level, seed))
        print(f"    \"noise_l{level}_s{seed}\": \"{h}\",")
    for ms, seed, mode in [(5, 42, "circular"), (10, 7, "circular"), (5, 42, "padding")]:
        out, _ = random_global_shift(X, ms, seed, mode=mode)
        print(f"    \"shift_{ms}_s{seed}_{mode}\": \"{_sha(out)}\",")
    for frac in [0.75, 0.5, 0.25]:
        h = _sha(truncate_and_resample(X, frac))
        print(f"    \"len_{frac}\": \"{h}\",")
    print("}")
    _, amounts = random_global_shift(X, 5, 42, mode="padding")
    print(f"GOLDEN_SHIFT_AMOUNTS_5_42 = {amounts.tolist()}")
