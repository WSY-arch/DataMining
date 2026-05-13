# Chen Report Draft Sections

## Introduction draft

Time-series clustering depends critically on how similarity between series is defined. Different similarity measures encode different assumptions about alignment, temporal variation, and signal structure. A lock-step measure such as Euclidean distance compares time points directly, while elastic measures such as DTW and MSM allow local temporal warping. Sliding measures such as SBD emphasize shape similarity under global phase shift. Distributional kernels such as IDK take a different view by comparing time series through distributions rather than explicit point-to-point alignment.

Prior benchmark studies have compared many time-series distances and clustering methods, but they usually focus on average performance across archive datasets. This leaves a mechanism-level question open: how do noise, temporal misalignment, and sequence length change the relative behavior of different similarity paradigms? This question is especially important because a measure that performs well on clean, aligned series may fail under shift or noise, while a more invariant measure may discard temporal information needed for some classes.

This project addresses the gap by comparing ED, DTW, MSM, SBD, and IDK under a unified time-series clustering framework. We combine a real-data benchmark on selected UCR datasets with controlled perturbation experiments. Rather than seeking a single universal winner, we aim to explain when each similarity paradigm is appropriate.

## Related Work structure

### Benchmark studies of time-series distances

Discuss broad distance benchmarks such as Paparrizos et al. and d'Hondt et al. as the foundation for cross-paradigm comparison. Emphasize their taxonomy, normalization findings, and statistical rigor, but note that they are not primarily mechanism-oriented clustering studies.

### Time-series clustering benchmarks

Discuss ED/DTW/SBD clustering comparisons and comprehensive clustering benchmarks. Position them as direct baselines for the project, while noting the absence of IDK and controlled noise/shift/length sweeps.

### Elastic distances for clustering

Use Holder and Bagnall to justify including MSM in addition to DTW. Highlight the finding that clustering performance depends on matching the distance geometry to the clustering objective; this supports the project's k-medoids design.

### Distributional and IDK-based approaches

Introduce distributional treatments of time series and IDK-related clustering work. The key framing is that IDK represents a distributional paradigm, not merely an extra distance plugged into the benchmark.

## Methodology draft

We evaluate univariate whole time-series clustering with a unified k-medoids pipeline. Each dataset is represented as `X` with shape `(n_samples, series_length)`, and the number of clusters is set to the ground-truth number of classes only for evaluation fairness. Labels are not used during clustering. All series are z-normalized before similarity computation.

The real-data benchmark uses selected UCR datasets spanning short to long sequences, binary to multi-class settings, and multiple domains. For each dataset and similarity measure, we compute a pairwise distance matrix, run k-medoids with a fixed random seed, and report ARI, NMI, and runtime. Results are summarized per dataset and by average ranks, with Friedman testing used for global comparison.

The perturbation study uses representative datasets such as CBF, Trace, and ECG200. We apply three controlled transformations: additive Gaussian noise, random global temporal shift, and truncation with padding to simulate reduced effective length. For each perturbation level, we rerun the same clustering pipeline and plot degradation curves. These curves are interpreted through the invariance assumptions of each similarity paradigm.

## Method mechanism table

| Measure | Mechanism | Expected perturbation behavior |
|---|---|---|
| ED | Pointwise lock-step alignment | Fast and strong on aligned data; degrades under shift. |
| DTW | Local elastic warping | More robust to local timing changes; may overfit noise. |
| MSM | Edit plus warp operations | Often more stable than DTW; parameter-sensitive. |
| SBD | Normalized cross-correlation with sliding shift | Strong under global phase shift; weaker under local warping. |
| IDK | Distributional kernel over subsequence structure | May improve with longer sequences; can lose temporal order. |
