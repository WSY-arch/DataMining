# Chen Dataset Selection Table

Use these 18 UCR datasets for Part 1 if compute allows. For quick smoke tests, run the first 6-8 datasets with `--samples-per-class 20`; for final experiments, use `--samples-per-class 50` or all samples if runtime is acceptable.

| # | Dataset | Length | Classes | Total samples | Domain | Rationale |
|---:|---|---:|---:|---:|---|---|
| 1 | SyntheticControl | 60 | 6 | 600 | Synthetic | Controlled patterns with trends and shift-like classes. |
| 2 | CBF | 128 | 3 | 930 | Synthetic | Clear Cylinder-Bell-Funnel shapes; useful for perturbation curves. |
| 3 | TwoPatterns | 128 | 4 | 5000 | Synthetic | Four synthetic motifs with timing variation; sample if needed. |
| 4 | ItalyPowerDemand | 24 | 2 | 1096 | Sensor | Very short series; lower bound for length sensitivity. |
| 5 | MoteStrain | 84 | 2 | 1272 | Sensor | Short real IoT sensor dataset. |
| 6 | ECG200 | 96 | 2 | 200 | Medical | Canonical ECG benchmark and realistic perturbation candidate. |
| 7 | ECGFiveDays | 136 | 2 | 884 | Medical | ECG across days; useful for phase variation. |
| 8 | GunPoint | 150 | 2 | 200 | Motion | Classic motion dataset with mild warping. |
| 9 | Plane | 144 | 7 | 210 | Sensor | Multi-class radar-return shapes. |
| 10 | Trace | 275 | 4 | 200 | Synthetic | Process-control shapes; strong Part 2 candidate. |
| 11 | ArrowHead | 251 | 3 | 211 | Shape | Object outline shape differences. |
| 12 | Coffee | 286 | 2 | 56 | Spectroscopy | Small non-temporal spectral shape dataset. |
| 13 | DiatomSizeReduction | 345 | 4 | 322 | Image | Image-outline morphology. |
| 14 | FaceFour | 350 | 4 | 112 | Image | Small face-outline projection dataset. |
| 15 | Symbols | 398 | 6 | 1020 | Shape | Handwritten-symbol outlines. |
| 16 | OSULeaf | 427 | 6 | 442 | Shape | Longer leaf-outline benchmark. |
| 17 | Beef | 470 | 5 | 60 | Spectroscopy | Small long spectroscopy dataset. |
| 18 | Mallat | 1024 | 8 | 2400 | Synthetic | Long-series upper bound; sample for DTW/MSM. |

## Part 2 representative subset

Use `CBF`, `Trace`, and `ECG200` first:

- `CBF`: clean synthetic shape structure; easiest to interpret.
- `Trace`: medium length and multi-class shape variation.
- `ECG200`: real medical data; tests whether synthetic perturbation findings transfer.

## Avoided datasets

Avoid extremely imbalanced datasets such as `Wafer`, very high-class-count datasets such as `Adiac` or `Phoneme`, and very long datasets such as `HandOutlines` for the main run unless compute is abundant.
