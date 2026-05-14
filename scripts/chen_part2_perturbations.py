"""Run Chen's controlled noise, shift, and length perturbation experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.chen_experiment_utils import (  # noqa: E402
    DEFAULT_RESULTS_ROOT,
    balanced_subsample,
    DEFAULT_AEON_DATA_ROOT,
    load_dataset,
    resolve_data_root,
    run_single_measure,
    write_result_rows,
)


def add_gaussian_noise(X: np.ndarray, level: float, seed: int) -> np.ndarray:
    if level == 0:
        return X.copy()
    rng = np.random.default_rng(seed)
    scale = np.std(X, axis=1, keepdims=True)
    scale[scale == 0.0] = 1.0
    return X + rng.normal(0.0, level * scale, size=X.shape)


def random_global_shift(X: np.ndarray, max_shift: int, seed: int, mode: str = "circular") -> np.ndarray:
    """Apply random temporal shift to each series.

    Modes:
        circular: np.roll (wrap-around). Suitable for periodic signals (ECG, CBF).
        padding: shift and fill exposed region with edge value. More realistic for
                 non-periodic signals (Trace, Plane) where wrap-around is artificial.
    """
    if max_shift == 0:
        return X.copy()
    rng = np.random.default_rng(seed)
    shifted = np.empty_like(X)
    for idx, row in enumerate(X):
        shift = int(rng.integers(-max_shift, max_shift + 1))
        if mode == "circular":
            shifted[idx] = np.roll(row, shift)
        else:  # padding
            if shift > 0:
                shifted[idx, shift:] = row[:-shift]
                shifted[idx, :shift] = row[0]
            elif shift < 0:
                shifted[idx, :shift] = row[-shift:]
                shifted[idx, shift:] = row[-1]
            else:
                shifted[idx] = row
    return shifted


def truncate_and_resample(X: np.ndarray, keep_fraction: float) -> np.ndarray:
    """Truncate series to keep_fraction of length, then resample back to original length.

    Uses linear interpolation (scipy.signal.resample) to restore original length,
    simulating lower time-resolution rather than introducing zero-padding artifacts.
    """
    if keep_fraction >= 1.0:
        return X.copy()
    from scipy.signal import resample as scipy_resample
    length = X.shape[1]
    keep = max(2, int(round(length * keep_fraction)))
    truncated = X[:, :keep]
    return scipy_resample(truncated, length, axis=1)


def maybe_plot_curves(rows: list[dict[str, object]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib is unavailable; skipping degradation curve plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({(row["dataset"], row["perturbation_type"]) for row in rows})
    for dataset, perturbation_type in keys:
        subset = [row for row in rows if row["dataset"] ==
                  dataset and row["perturbation_type"] == perturbation_type]
        measures = sorted({row["measure"] for row in subset})
        for metric_name in ["ari", "nmi"]:
            fig, ax = plt.subplots(figsize=(7, 4))
            for measure in measures:
                series = [row for row in subset if row["measure"] == measure]
                series.sort(key=lambda row: float(row["perturbation_level"]))
                x_values = [float(row["perturbation_level"]) for row in series]
                y_values = [float(row[metric_name]) for row in series]
                ax.plot(x_values, y_values, marker="o", label=measure)
            ax.set_title(
                f"{dataset} {perturbation_type} degradation ({metric_name.upper()})")
            ax.set_xlabel("Perturbation level")
            ax.set_ylabel(metric_name.upper())
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(
                output_dir / f"{dataset}_{perturbation_type}_{metric_name}.png", dpi=160)
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-source",
        choices=["aeon", "files"],
        default="aeon",
        help="Load UCR datasets through aeon auto-download/cache or local TRAIN/TEST files.",
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_RESULTS_ROOT / "part2_perturbation_results.csv"))
    parser.add_argument("--plot-dir", type=str,
                        default=str(DEFAULT_RESULTS_ROOT / "perturbation_curves"))
    parser.add_argument("--datasets", nargs="*",
                        default=["CBF", "Trace", "ECG200"])
    parser.add_argument("--metrics", nargs="*", default=["ed", "dtw", "msm"])
    parser.add_argument("--noise-levels", nargs="*",
                        type=float, default=[0.0, 0.1, 0.2, 0.4, 0.8])
    parser.add_argument("--shift-levels", nargs="*",
                        type=int, default=[0, 5, 10, 20, 40])
    parser.add_argument("--length-fractions", nargs="*",
                        type=float, default=[1.0, 0.75, 0.5, 0.25])
    parser.add_argument("--samples-per-class", type=int, default=50)
    parser.add_argument("--seeds", nargs="*", type=int,
                        default=[42], help="Random seeds to repeat for each dataset/metric/level.")
    parser.add_argument(
        "--metric-backend",
        choices=["auto", "reference", "aeon", "tslearn"],
        default="auto",
        help="Backend for DTW/MSM. Use 'aeon' for formal runs after installing aeon.",
    )
    parser.add_argument("--shift-mode", choices=["circular", "padding"], default="circular",
                        help="circular: wrap-around (periodic signals); padding: edge-fill (non-periodic).")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.data_source == "aeon":
        data_root = Path(
            args.data_root) if args.data_root else DEFAULT_AEON_DATA_ROOT
    else:
        data_root = resolve_data_root(args.data_root)
    rows: list[dict[str, object]] = []
    samples_per_class = None if args.samples_per_class == 0 else args.samples_per_class
    base_seed = args.seeds[0]

    for dataset_name in args.datasets:
        try:
            X, y = load_dataset(dataset_name, args.data_source, data_root)
            X, y = balanced_subsample(X, y, samples_per_class, base_seed)
            print(f"[DATASET] {dataset_name}: X={X.shape}, k={len(set(y))}")

            perturbations: list[tuple[str, float, np.ndarray]] = []
            perturbations.extend(("noise", level, add_gaussian_noise(
                X, level, base_seed)) for level in args.noise_levels)
            perturbations.extend(("shift", float(level), random_global_shift(
                X, level, base_seed, mode=args.shift_mode)) for level in args.shift_levels)
            perturbations.extend(("length", level, truncate_and_resample(
                X, level)) for level in args.length_fractions)

            for perturbation_type, perturbation_level, X_perturbed in perturbations:
                for seed in args.seeds:
                    for metric in args.metrics:
                        row = run_single_measure(
                            X_perturbed,
                            y,
                            dataset_name,
                            metric,
                            clustering_seed=seed,
                            subsample_seed=base_seed,
                            perturbation_type=perturbation_type,
                            perturbation_level=str(perturbation_level),
                            similarity_params={"backend": args.metric_backend},
                        )
                        rows.append(row)
                        print(
                            f"  seed={seed:<4} {perturbation_type:<6}={perturbation_level:<4} {metric:<4} "
                            f"ARI={row['ari']:.4f} NMI={row['nmi']:.4f}"
                        )
        except Exception as exc:
            print(f"[ERROR] {dataset_name}: {exc}")
            if not args.continue_on_error:
                return 1

    output_path = Path(args.output)
    write_result_rows(rows, output_path)
    print(f"[OK] wrote {len(rows)} rows to {output_path}")
    if not args.no_plots:
        maybe_plot_curves(rows, Path(args.plot_dir))
        print(f"[OK] plots saved under {args.plot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
