"""Run Chen's controlled noise, shift, and length perturbation experiments."""

from __future__ import annotations

import argparse
import json
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
from tsclust.perturbations import (  # noqa: E402
    add_gaussian_noise,
    random_global_shift,
    truncate_and_resample,
)


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


def _dump_cell(
    cells_root: Path,
    dataset: str,
    perturbation_type: str,
    perturbation_level: float | int,
    seed: int,
    mode: str,
    X_perturbed: np.ndarray,
    y: np.ndarray,
    shift_amounts: np.ndarray | None,
    shift_pct: float | None = None,
    shift_abs: int | None = None,
) -> None:
    """Persist a perturbed cell so Wang (SBD/IDK) can read it without rerunning
    the perturbation pipeline. See docs/methodology/perturbation_design.md."""
    out_dir = cells_root / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    if perturbation_type == "shift":
        # Filename uses both percentage (cross-dataset comparable) and absolute
        # samples (per-dataset reproducible) per Choice 3 metadata schema.
        fname = f"shift_p{int(shift_pct)}_a{int(shift_abs)}_s{seed}_{mode}.npz"
    elif perturbation_type == "noise":
        fname = f"noise_l{perturbation_level}_s{seed}.npz"
    elif perturbation_type == "length":
        fname = f"length_f{perturbation_level}.npz"
    else:
        fname = f"{perturbation_type}_{perturbation_level}_s{seed}.npz"
    y_arr = np.asarray(y)
    params = {
        "perturbation_type": perturbation_type,
        "perturbation_level": perturbation_level,
        "seed": seed if perturbation_type != "length" else None,
        "shift_mode": mode if perturbation_type == "shift" else None,
        "shift_pct": shift_pct if perturbation_type == "shift" else None,
        "shift_abs": shift_abs if perturbation_type == "shift" else None,
        "dataset": dataset,
        "series_length": int(X_perturbed.shape[1]),
        "n_sampled": int(X_perturbed.shape[0]),
        "n_classes": int(len(np.unique(y_arr))),
        "pipeline_version": "2026-05-21-spec",
    }
    payload = {
        "X": X_perturbed.astype(np.float64, copy=False),
        "y": y_arr,
        "params": np.array(json.dumps(params), dtype=object),
    }
    if shift_amounts is not None:
        payload["shift_amounts"] = shift_amounts.astype(np.int64, copy=False)
    np.savez_compressed(out_dir / fname, **payload)


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
    parser.add_argument(
        "--cells-dir",
        type=str,
        default=str(DEFAULT_RESULTS_ROOT.parent / "_perturbed_cells"),
        help="Output dir for per-cell .npz dumps (X_perturbed + y + shift_amounts + params).",
    )
    parser.add_argument(
        "--no-cells",
        action="store_true",
        help="Skip dumping perturbed cells to disk (for quick smoke runs).",
    )
    parser.add_argument("--datasets", nargs="*",
                        default=["CBF", "Trace", "ECG200"])
    parser.add_argument("--metrics", nargs="*", default=["ed", "dtw", "msm"])
    parser.add_argument("--noise-levels", nargs="*",
                        type=float, default=[0.0, 0.1, 0.2, 0.4, 0.8])
    parser.add_argument("--shift-pct", nargs="*",
                        type=float, default=[0, 5, 10, 20, 30],
                        help="Shift sweep as PERCENTAGE of L (per-dataset abs_shift = round(pct * L / 100)). "
                             "30%% is the locked upper bound; beyond it the experiment functionally overlaps "
                             "with the length perturbation (Choice 5). See docs/methodology/perturbation_design.md.")
    parser.add_argument("--length-fractions", nargs="*",
                        type=float, default=[1.0, 0.9, 0.75, 0.5, 0.25])
    parser.add_argument("--samples-per-class", type=int, default=50)
    parser.add_argument("--seeds", nargs="*", type=int,
                        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        help="clustering_seed values to repeat for each dataset/metric/level. "
                             "Locked at 10 seeds per docs/methodology/perturbation_design.md Open#1.")
    parser.add_argument(
        "--metric-backend",
        choices=["auto", "reference", "aeon", "tslearn"],
        default="auto",
        help="Backend for DTW/MSM. Use 'aeon' for formal runs after installing aeon.",
    )
    parser.add_argument("--shift-mode", choices=["circular", "padding"], default="padding",
                        help="padding (default): edge-fill, the main protocol. "
                             "circular: np.roll wrap-around, reserved as ablation "
                             "that exposes SBD's intrinsic translation invariance.")
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
    # unified across all measures (Chen+Wang collab schema)
    SUBSAMPLE_SEED = 42

    for dataset_name in args.datasets:
        try:
            X, y = load_dataset(dataset_name, args.data_source, data_root)
            n_original = int(X.shape[0])
            X, y = balanced_subsample(X, y, samples_per_class, SUBSAMPLE_SEED)
            print(
                f"[DATASET] {dataset_name}: X={X.shape}, k={len(set(y))}, n_original={n_original}")

            perturbations: list[tuple[str, float, np.ndarray, np.ndarray | None, float | None, int | None]] = []
            for level in args.noise_levels:
                perturbations.append((
                    "noise", float(level), add_gaussian_noise(X, level, base_seed), None, None, None,
                ))
            series_length = int(X.shape[1])
            for pct in args.shift_pct:
                abs_shift = int(round(float(pct) * series_length / 100))
                shifted, shift_amounts = random_global_shift(
                    X, abs_shift, base_seed, mode=args.shift_mode
                )
                # perturbation_level recorded as percentage so cross-dataset
                # CSV rows are directly comparable; abs_shift is preserved in
                # the per-cell metadata for reproducibility.
                perturbations.append((
                    "shift", float(pct), shifted, shift_amounts, float(pct), abs_shift,
                ))
            for level in args.length_fractions:
                perturbations.append((
                    "length", float(level), truncate_and_resample(X, level), None, None, None,
                ))

            cells_root = Path(args.cells_dir) if not args.no_cells else None

            for perturbation_type, perturbation_level, X_perturbed, shift_amounts, shift_pct, shift_abs in perturbations:
                if cells_root is not None:
                    _dump_cell(
                        cells_root, dataset_name, perturbation_type,
                        perturbation_level, base_seed, args.shift_mode,
                        X_perturbed, y, shift_amounts,
                        shift_pct=shift_pct, shift_abs=shift_abs,
                    )
                for seed in args.seeds:
                    for metric in args.metrics:
                        row = run_single_measure(
                            X_perturbed,
                            y,
                            dataset_name,
                            metric,
                            clustering_seed=seed,
                            subsample_seed=SUBSAMPLE_SEED,
                            perturbation_type=perturbation_type,
                            perturbation_level=str(perturbation_level),
                            similarity_params={"backend": args.metric_backend},
                            n_original=n_original,
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
