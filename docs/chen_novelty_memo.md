# Novelty Memo: Mechanism-Oriented Time-Series Clustering Similarity Comparison

## One-sentence position

This project studies when different time-series similarity paradigms work, not only which one wins on average, by comparing lock-step, elastic, sliding, and distributional similarities under controlled noise, temporal misalignment, and length perturbations.

## Literature gap

Existing time-series distance benchmarks are broad but usually classification-oriented or performance-oriented. Large studies compare many distances and normalizations, but they rarely isolate noise, temporal shift, and sequence length as controlled experimental factors in clustering.

Existing clustering benchmarks cover important baselines such as ED, DTW, and SBD, and elastic-distance studies show that MSM/TWE can outperform plain DTW when the clustering objective is appropriate. However, these works do not give distributional kernels such as IDK equal footing in a unified clustering comparison.

Robustness and invariance papers provide useful perturbation designs for shift, warp, and noise, but they are usually model-selection frameworks or domain-specific studies rather than a cross-paradigm clustering benchmark. Sequence length is especially under-studied as an independent mechanism; most papers treat it as a computational issue or let it vary naturally across datasets.

## Proposed contribution

We contribute a fair, mechanism-oriented comparison of representative similarity measures:

- ED as the lock-step baseline.
- DTW and MSM as elastic alignment measures.
- SBD as the sliding/shift-invariant measure.
- IDK as the distributional-kernel measure.

The key novelty is the combination of:

- General-purpose univariate whole-series clustering.
- A unified k-medoids interface and shared result schema.
- Real UCR benchmark datasets plus controlled perturbation experiments.
- Mechanistic interpretation of degradation curves under noise, shift, and length changes.

## Expected mechanism hypotheses

| Measure | Paradigm | Expected strength | Expected weakness |
|---|---|---|---|
| ED | Lock-step | Strong when series are well aligned and shape differences are direct. | Sensitive to temporal shift and local warping. |
| DTW | Elastic | Handles local timing differences by warping the time axis. | Can over-warp noise and is computationally expensive. |
| MSM | Elastic/edit | More stable than pure DTW because edit operations regularize warping. | Requires a cost parameter and may still be sensitive to strong noise. |
| SBD | Sliding | Robust to global phase shift via cross-correlation. | Less suited to local speed changes or non-shape distributional differences. |
| IDK | Distributional | Can be robust when order/alignment is less important and longer sequences estimate distributions better. | May lose temporal-order information that matters for some classes. |

## Claim discipline

The project should not claim that IDK is universally better. A stronger and more defensible claim is:

> Different similarity measures encode different invariances; the best choice depends on the data-generating mechanism, especially noise, misalignment, and effective sequence length.
