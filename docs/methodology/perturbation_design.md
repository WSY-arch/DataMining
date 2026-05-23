# Perturbation Experiment Design — Methodology Notes

**Project:** Mechanism-oriented comparison of time-series clustering similarity measures
**Scope:** Part 2 controlled perturbation experiments (noise / misalignment / length)
**Target datasets:** CBF (length 128, k=3), Trace (length 275, k=4), ECG200 (length 96, k=2)
**Last updated:** 2026-05-21

This document records the five core design choices for the perturbation pipeline, the rationale behind each, the literature/benchmark precedent, and the exact English phrasing intended for direct use in the paper's Methodology section.

---

## Pipeline assumptions (read first)

All raw series are z-normalized **before** any perturbation is applied. This is the UCR-standard pretreatment (Paparrizos & Liu, SIGMOD 2020) and is required for cross-dataset fairness. All perturbations are therefore applied on signals with per-series mean 0 and std 1.

All randomness is centralized through a single top-level `numpy.random.Generator` seeded with the experiment seed; per-task RNGs are derived via `SeedSequence.spawn(n)` to guarantee independence across parallel workers and reproducibility across re-runs.

---

## Choice 1 — Noise magnitude: relative σ, not absolute σ

**Decision.** Additive Gaussian noise with σ defined as `level × per-series std`. Because the pipeline z-normalizes before perturbation, per-series std equals 1, so numerically `σ = level`. The relative formulation must still be stated explicitly in the paper so that anyone reproducing the work without z-normalization recovers the intended behavior.

**Rationale.** UCR datasets span four orders of magnitude in raw amplitude (Coffee spectroscopy values ~0.01 vs. accelerometer signals in the hundreds). A single absolute σ value would be background noise in one dataset and would saturate the signal in another, conflating "noise robustness" with "amplitude scale."

**Precedent.**
- Paparrizos & Gravano, *k-Shape* (SIGMOD 2015): noise robustness defined as a fraction of signal std.
- Schäfer, *BOSS* (DAMI 2015): SNR-based noise injection.
- Bagnall et al., *The great time series classification bake off* (DAMI 2017): all noise robustness experiments use signal-relative SNR.

**Methodology text (English).**
> *Gaussian noise was injected as $x'(t) = x(t) + \varepsilon(t)$ with $\varepsilon(t) \sim \mathcal{N}(0, \sigma^2)$, where $\sigma = \ell \cdot \mathrm{std}(x)$ and $\ell$ is the noise level swept over $\{0.0, 0.1, 0.2, 0.4, 0.8\}$. The endpoints serve specific purposes: $\ell=0.0$ is the unperturbed baseline reused as the within-experiment control; $\ell=0.8$ (SNR $\approx 2$ dB) is a saturation probe that exposes whether each metric retains residual discriminative power once noise approaches the signal's own variance. Because all series were z-normalized prior to perturbation, $\sigma$ numerically equals $\ell$; the relative formulation is retained for cross-pipeline reproducibility.*

---

## Choice 2 — Random number generation: PCG64 via `np.random.default_rng`

**Decision.** All stochastic operations use `numpy.random.Generator` instances obtained from `np.random.default_rng(seed)` (PCG64 backend). Legacy `np.random.seed` / `np.random.normal` (MT19937 global state) is forbidden anywhere in the perturbation, sampling, or k-medoids initialization code.

**Rationale.** PCG64 has a longer period, better statistical properties, and — critically — **bit-exact cross-platform reproducibility** that MT19937 in legacy mode does not always guarantee. Using the modern Generator API also eliminates the global-state class of bugs where one library call silently advances the RNG and shifts every downstream draw.

**Implementation constraint.** A single top-level `SeedSequence(experiment_seed)` is created once. Child seeds for each (dataset, perturbation, replicate) cell are spawned via `ss.spawn(n)`. Worker processes receive a spawned `SeedSequence`, not a raw integer. This is what guarantees that re-running the experiment on a different machine yields the same distance matrices to floating-point precision.

**Precedent.** numpy 1.17+ official recommendation (NEP 19); adopted as the default in `tslearn` ≥ 0.6, `aeon` ≥ 0.5, and `scikit-learn` ≥ 1.4 random-state APIs.

**Methodology text (English).**
> *All randomness is generated through `numpy.random.Generator` (PCG64) seeded from a single `SeedSequence`; per-cell sub-streams are obtained via `SeedSequence.spawn`, ensuring independence and bit-exact reproducibility across runs.*

---

## Choice 3 — Misalignment: per-series independent shift

**Decision.** For each series $x_i$ in the dataset, draw an independent integer shift $s_i \sim \mathrm{Uniform}\{-S, \ldots, +S\}$, where $S$ is the maximum-shift parameter. The realized shift vector $(s_1, \ldots, s_N)$ is persisted to disk per (dataset, seed, max_shift) cell for downstream fine-grained analysis.

**Rationale.** A globally uniform shift applied to every series in the dataset has a critical pathology: under z-normalized data, it is mathematically equivalent to a constant time offset that **leaves every pairwise Euclidean distance unchanged**. The ED baseline would then show perfect robustness to "misalignment," collapsing the entire research question. Misalignment in real datasets is, by definition, *between-instance* variation in temporal alignment, which only per-series independent shifts can model.

**Precedent.**
- Paparrizos & Gravano, *k-Shape* (SIGMOD 2015): alignment perturbations applied per-series.
- Cuturi & Blondel, *Soft-DTW* (ICML 2017): independent shifts per instance in the alignment robustness study.
- Forestier et al., *Generating synthetic time series* (DAMI 2017): per-instance temporal jitter as the standard misalignment protocol.

**Storage convention.** Each perturbed dataset cell saves `shift_amounts.npy` alongside the perturbed series array. Storing only the seed is insufficient — recovering the exact shift values from a seed requires re-simulating the entire RNG draw order, which is brittle if the code path changes.

**Methodology text (English).**
> *Misalignment was introduced by shifting each series independently by an integer amount $s_i \sim \mathrm{Uniform}\{-S, +S\}$. Max-shift $S$ was swept over values corresponding to $\{0\%, 5\%, 10\%, 20\%, 30\%\}$ of each dataset's series length, with $0\%$ serving as the within-experiment unperturbed control and $30\%$ chosen as the upper bound: beyond this fraction (e.g.\ $40\%$ on ECG200, $\approx 38$ samples on $L=96$) the QRS complex would be displaced past the analysis window, so the experiment would no longer measure misalignment but rather signal truncation, functionally overlapping with the length perturbation of Choice~5 and confounding the ablation. Per-instance shifts are required because a globally uniform shift leaves all pairwise Euclidean distances invariant and therefore fails to constitute a meaningful perturbation.*

---

## Choice 4 — Shift boundary handling: padding default, circular as ablation

**Decision.** **Default mode is `padding=edge`** (replicate the first/last value to fill the vacated tail/head). `circular` (`np.roll`) is retained as an explicit ablation, not as the default. `zero-fill` is excluded entirely.

> **This reverses the initial proposal.** The original draft had `circular` as default with `padding` as ablation. The reasoning for the reversal is below; the change is load-bearing for the validity of the SBD comparison.

**Rationale — three problems with circular as default.**

*Problem 1: circular shift introduces artifacts incompatible with the data semantics.* CBF is bounded but not periodic — its three classes (Cylinder, Bell, Funnel) are transient shapes, not cycles. ECG200 is a *single* heartbeat segmentation: the head contains the pre-P baseline and the tail contains the post-T recovery. A `circular` wrap-around grafts the recovery segment onto the front of the P-wave, producing a waveform that does not exist in any real ECG and would not survive clinical inspection. Trace consists of nuclear-plant transient events with semantically distinct head and tail. For these datasets, circular wrap-around is not a benign boundary choice; it manufactures synthetic out-of-distribution artifacts.

*Problem 2: circular shift trivializes SBD by construction.* SBD is defined as $1 - \max_w \mathrm{NCC}_w(x, y)$ over all circular alignments $w$ of cross-correlation. By definition, SBD is **invariant to circular shifts of either input**: $\mathrm{SBD}(\mathrm{roll}(x, s), y) = \mathrm{SBD}(x, y)$ exactly. Using circular as the default shift mode therefore produces a degenerate experiment in which SBD's distance matrix is bit-identical before and after perturbation. SBD would appear "perfectly robust to misalignment," but this is a mathematical identity of the metric definition, not an empirical finding. It tells the reader nothing about SBD's behavior on real-world misalignment, which is precisely what the experiment is supposed to measure.

*Problem 3: padding gives every metric a fair degradation curve.* Under `edge` padding, ED, DTW, MSM, SBD, and IDK all face a genuine boundary-induced perturbation and exhibit a measurable ARI/NMI degradation as $S$ grows. The comparison across metrics becomes meaningful.

**Why `circular` is still retained as ablation.** Running the same misalignment sweep under `circular` mode is itself a publishable finding: it produces a clean empirical demonstration of SBD's translation invariance and provides a natural section in the Discussion contrasting *metric-intrinsic invariance* (SBD under circular) with *metric-empirical robustness* (SBD under padding). This is a strict gain over the original design — the ablation now carries scientific content rather than serving as a fallback.

**Why `zero-fill` is excluded.** On z-normalized signals, zero-fill introduces a step discontinuity at the boundary equal to the original endpoint value. This conflates a length-1-style amplitude artifact with the shift perturbation and confounds the ablation.

**Precedent.**
- Cuturi & Blondel, *Soft-DTW* (ICML 2017): edge padding for shift experiments.
- Forestier et al. 2017: edge padding default; circular reported only when the underlying signal is genuinely cyclic.
- Paparrizos & Gravano, *k-Shape* (SIGMOD 2015) §5.4: explicit discussion of SBD's circular-shift invariance and its implication that empirical shift robustness must be measured under non-circular boundary conditions.

**Methodology text (English).**
> *After shifting, vacated positions were filled by edge replication (the first or last observed value, depending on shift direction). Edge padding was chosen as the default because (i) the three target datasets — CBF, Trace, ECG200 — are not periodic and circular wrap-around would graft semantically incompatible regions onto each other, and (ii) SBD is by construction invariant under circular shifts of its inputs, so a circular-default protocol would trivially yield zero degradation for SBD irrespective of the shift magnitude. A circular ablation is reported separately in §X.Y to characterize this intrinsic invariance.*

**For Wang (SBD/IDK branch) — contract test required.**

Because Choice 4 hinges on SBD's intrinsic translation invariance, the SBD implementation must ship with a unit test that pins this property. The test belongs in the SBD PR (Chen's main has no SBD code) and must be green before merge. Two assertions are required:

```python
# 1) Per-series invariance (atol = 1e-10 on z-normalized inputs)
for s in [1, 5, 10, L // 2]:
    assert np.allclose(sbd(np.roll(x, s), y), sbd(x, y), atol=1e-10)

# 2) Pairwise distance matrix invariance under uniform circular shift
D       = pairwise_sbd(X)
D_roll  = pairwise_sbd(np.roll(X, k, axis=1))
assert np.allclose(D, D_roll, atol=1e-10)
```

This is also the analytical basis for reporting `--shift-mode=circular` as an ablation that **demonstrates** SBD's intrinsic invariance, rather than as a robustness ranking.

---

## Choice 5 — Length: truncate-only, then FFT-based resample to original length

**Decision.** To produce a length-perturbed copy at fraction $f \in (0, 1]$ of the original length $L$:

1. **Truncate** to `keep = max(2, round(L × f))` samples, removing equal amounts from both ends of the series (floor on head, ceil on tail when odd).
2. **Resample** back to length $L$ via `scipy.signal.resample` (FFT-based).

Zero-padding is forbidden. Single-end truncation is forbidden.

**Rationale.**

*Why truncate-only, not zero-pad.* On z-normalized signals, zero-padding introduces a step discontinuity at the boundary, which is a *second* perturbation (an artificial step edge) injected on top of the length change. This breaks the ablation — degradation could be attributed to either factor, with no way to disentangle them. Truncate-only models the realistic scenario of "the recording was shorter / cut off / sampled at lower rate," which is the actual phenomenon length perturbation is meant to probe.

*Why two-sided truncation.* Equal removal from both ends preserves the signal's temporal center of mass. For CBF and ECG200, the diagnostic waveform (the bell shape; the QRS complex) is approximately centered; one-sided truncation would systematically destroy more of the discriminative region than the baseline. For Trace, two-sided is also defensible since transient onset and offset both carry information.

*Why FFT resample, not linear interpolation.* `scipy.signal.resample` uses an FFT-based polyphase implementation that is exact for band-limited signals and near-optimal for smooth signals like CBF and ECG200. Linear interpolation would low-pass-filter the signal more aggressively, conflating "shorter series" with "smoother series." Mild Gibbs ringing at the boundaries of Trace's transient events is acceptable under the truncate-only constraint (no zero-padding to amplify it).

*Why `max(2, ·)` floor.* Length 1 series cannot support any pairwise metric (ED degenerates, DTW has no path, SBD has no cross-correlation). The floor of 2 is the minimum at which every metric remains defined.

**Precedent.**
- Project internal standard: *"Length perturbation implementation: truncate-only + scipy.signal.resample; zero-padding forbidden."*
- Paparrizos & Bogireddy, *Time-series clustering benchmark* (VLDB 2025): variable-length handling via resampling, not padding.
- `scipy.signal.resample` documentation: FFT-based, recommended for smooth periodic or quasi-periodic signals.

**Methodology text (English).**
> *Length perturbation was performed by truncating each series symmetrically from both ends to $\lceil L \cdot f \rceil$ samples (with a floor of 2), then resampling back to the original length $L$ via FFT-based polyphase interpolation (`scipy.signal.resample`). Truncation fractions $f$ were swept over $\{1.0, 0.9, 0.75, 0.5, 0.25\}$, where $f=1.0$ is the unperturbed control. The point $f=0.9$ is retained (unlike the analogous low-amplitude noise level $\ell=0.05$, which was dropped as redundant with the baseline) because mild truncation already triggers FFT-resample spectral recomposition: SBD, being NCC-based, reacts measurably while DTW remains comparatively robust, so this point carries discriminative information for the cross-metric comparison. The point $f=0.25$ on ECG200 (kept window $\approx 24$ samples, comparable to the QRS-complex span) is read as `metric behavior at the recoverability boundary' rather than typical short-signal performance, and is flagged as such in the Limitations section. Zero-padding was deliberately excluded: on z-normalized signals it would introduce a boundary step discontinuity, conflating length change with edge-artifact injection.*

---

## Summary table for the paper

| Aspect | Decision | Key rationale |
|---|---|---|
| Noise magnitude | Relative σ = `level × per-series std` | Cross-dataset amplitude varies ~10⁴; absolute σ conflates noise with scale |
| RNG | `default_rng` (PCG64) + `SeedSequence.spawn` | Bit-exact cross-platform reproducibility; eliminates global-state bugs |
| Shift application | Per-series independent uniform integer | Global uniform shift leaves pairwise ED invariant — degenerate for ED baseline |
| Shift boundary | `edge` padding default; `circular` ablation; `zero` excluded | Circular default would trivialize SBD (intrinsic invariance); ablation reports this as a finding |
| Length | Two-sided truncate + FFT resample to original $L$; floor 2 | Truncate-only avoids the step-edge confound; FFT resample preserves spectral content |

---

## Open items to confirm before locking the protocol

> **All items finalized 2026-05-21.** This section is retained as a record of decisions; no further consensus is pending.

1. **Number of seeds.** ✅ **Finalized: 10** (`clustering_seed = 1, 2, ..., 10`). The Friedman test power gain over 5 seeds is modest but real, and per-cell runtime on the locked sweep grid (5 noise × 5 shift × 5 length levels × 3 datasets) keeps total wall-clock well within a single overnight run on the team's hardware.

2. **Per-cell metadata schema.** ✅ **Finalized.** Each `.npz` cell under `results/_perturbed_cells/{dataset}/` stores the following fields. This is the authoritative schema; Wang's SBD/IDK reader code must consume these names verbatim.

   **Top-level npz arrays:**

   | Field | Type | Always present | Description |
   |---|---|---|---|
   | `X` | `(n, L) float64` | yes | Perturbed series (z-normalized prior to perturbation). |
   | `y` | `(n,) int` | yes | Class labels, row-aligned with `X`. |
   | `params` | `object` (json string) | yes | Serialized parameter dict, see below. |
   | `shift_amounts` | `(n,) int64` | shift cells only | Realized integer shift applied to each series. |

   **`params` JSON fields:**

   | Field | Type | Notes |
   |---|---|---|
   | `perturbation_type` | `"noise"` / `"shift"` / `"length"` | |
   | `perturbation_level` | float | Cross-dataset comparable level: noise→ℓ, shift→pct, length→f. |
   | `seed` | int | RNG seed for noise/shift; `null` for length (deterministic). |
   | `shift_mode` | `"padding"` / `"circular"` / `null` | shift cells only. |
   | `shift_pct` | float / null | Shift as percentage of L (shift cells only). |
   | `shift_abs` | int / null | Shift as absolute samples (shift cells only). |
   | `dataset` | str | Dataset name (self-describing even if file is moved). |
   | `series_length` | int | L; sanity-check against `X.shape[1]`. |
   | `n_sampled` | int | n; sanity-check against `X.shape[0]`. |
   | `n_classes` | int | k; lets the reader call k-medoids without recomputing `len(set(y))`. |
   | `pipeline_version` | str | `"2026-05-21-spec"` (git-traceability tag). |

   **Filename convention:**

   ```text
   shift_p{pct}_a{abs}_s{seed}_{mode}.npz   # e.g. shift_p10_a13_s42_padding.npz
   noise_l{level}_s{seed}.npz               # e.g. noise_l0.2_s42.npz
   length_f{frac}.npz                       # e.g. length_f0.75.npz   (no seed: deterministic)
   ```

3. **Code–doc alignment (Choice 3 action item).** ✅ **Done 2026-05-21** — `scripts/chen_part2_perturbations.py` now uses `--shift-pct` and persists both `shift_pct` and `shift_abs` in the per-cell metadata.

---

## Implementation sync log

| Date | Item | Status |
|---|---|---|
| 2026-05-21 | All five Choices reflected in `tsclust/perturbations.py`; `random_global_shift` default flipped to `padding`; `truncate_and_resample` switched to two-sided symmetric truncation; `random_global_shift` now returns `(shifted, shift_amounts)`; `tests/test_perturbations.py` regenerated GOLDEN, 14/14 PASS. | ✅ in sync |
| 2026-05-21 | `scripts/chen_part2_perturbations.py` `--shift-mode` default flipped to `padding`; per-cell `.npz` dump added (X, y, shift_amounts, params) under `results/_perturbed_cells/{dataset}/` for Wang's SBD/IDK reads. | ✅ in sync |
| 2026-05-21 | `docs/工作记录.md` §16.2 ·16.4 updated to reflect the flip and the Wang invariance-test contract. | ✅ in sync |
| 2026-05-21 | Sweep grids finalized in Methodology text: noise $\{0.0, 0.1, 0.2, 0.4, 0.8\}$, shift $\{0, 5, 10, 20, 30\}\%$ of $L$, length $\{1.0, 0.9, 0.75, 0.5, 0.25\}$. Open items 1–3 collapsed; rationale for dropping noise 0.05 and shift 40% recorded inline. | ✅ doc-only |
| 2026-05-21 | `scripts/chen_part2_perturbations.py` CLI migrated from absolute `--shift-levels` to relative `--shift-pct`; per-dataset `abs_shift = round(pct * L / 100)`; per-cell `.npz` metadata now stores both `shift_pct` and `shift_abs`; `--length-fractions` default updated to `[1.0, 0.9, 0.75, 0.5, 0.25]`. | ✅ in sync |
| 2026-05-21 | Open items 1–2 finalized: `clustering_seed = 1..10` (Open#1); per-cell `params` schema locked with full field table (Open#2). `chen_part2_perturbations.py --seeds` default updated to `1..10`; `_dump_cell` now persists `dataset`, `series_length`, `n_sampled`, `n_classes` so cells are self-describing. | ✅ in sync |
