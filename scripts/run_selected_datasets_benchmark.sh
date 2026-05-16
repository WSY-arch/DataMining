#!/usr/bin/env bash
set -u

# Batch runner for collaboration benchmark mode.
# It calls scripts/run_ucr_sbd_idk_compare.py for the datasets listed in docs/数据集选择.md.
#
# Usage examples:
#   bash scripts/run_selected_datasets_benchmark.sh --seeds 1 2 3 --no-viz
#   bash scripts/run_selected_datasets_benchmark.sh --quick --seeds 1 --n-samples 20 --no-viz
#
# Notes:
# - The script forces --mode benchmark.
# - All extra args are passed through to run_ucr_sbd_idk_compare.py.

DATASETS=(
  SyntheticControl
  CBF
  ItalyPowerDemand
  MoteStrain
  ECG200
  ECGFiveDays
  GunPoint
  Plane
  TwoPatterns
  Trace
  ArrowHead
  Coffee
  DiatomSizeReduction
  FaceFour
  Symbols
  OSULeaf
  Beef
  Mallat
)

QUICK=0
STOP_ON_ERROR=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      QUICK=1
      shift
      ;;
    --stop-on-error)
      STOP_ON_ERROR=1
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$QUICK" -eq 1 ]]; then
  DATASETS=(
    SyntheticControl
    CBF
    TwoPatterns
    ItalyPowerDemand
    MoteStrain
    ECG200
    ECGFiveDays
    GunPoint
  )
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY_SCRIPT="$PROJECT_ROOT/scripts/run_ucr_sbd_idk_compare.py"
RESULTS_ROOT="$PROJECT_ROOT/results/auto_ucr"
MERGED_CSV="$RESULTS_ROOT/collaboration_results_sbd_idk_all.csv"

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "[ERROR] Missing script: $PY_SCRIPT"
  exit 1
fi

echo "============================================================"
echo "Batch benchmark runner (SBD vs IDK)"
echo "Project root: $PROJECT_ROOT"
echo "Datasets: ${#DATASETS[@]}"
echo "Quick mode: $QUICK"
echo "Stop on error: $STOP_ON_ERROR"
echo "Extra args: ${EXTRA_ARGS[*]:-(none)}"
echo "============================================================"

FAILED=()
SUCCEEDED=()

for dataset in "${DATASETS[@]}"; do
  echo
echo "[RUN] Dataset: $dataset"
  echo "------------------------------------------------------------"

  (
    cd "$PROJECT_ROOT" || exit 2
    python "$PY_SCRIPT" "$dataset" --mode benchmark "${EXTRA_ARGS[@]}"
  )
  status=$?

  if [[ $status -ne 0 ]]; then
    echo "[FAIL] $dataset (exit=$status)"
    FAILED+=("$dataset")
    if [[ "$STOP_ON_ERROR" -eq 1 ]]; then
      break
    fi
  else
    echo "[OK] $dataset"
    SUCCEEDED+=("$dataset")
  fi
done

# Merge per-dataset collaboration CSVs into one file.
echo
echo "[MERGE] Building aggregated CSV..."
mkdir -p "$RESULTS_ROOT"

HEADER_WRITTEN=0
MERGED_ROWS=0
MISSING_CSV=()

: > "$MERGED_CSV"

for dataset in "${DATASETS[@]}"; do
  DATASET_CSV="$RESULTS_ROOT/$dataset/collaboration_results_sbd_idk.csv"
  if [[ ! -f "$DATASET_CSV" ]]; then
    MISSING_CSV+=("$dataset")
    continue
  fi

  if [[ "$HEADER_WRITTEN" -eq 0 ]]; then
    head -n 1 "$DATASET_CSV" > "$MERGED_CSV"
    HEADER_WRITTEN=1
  fi

  # Append data rows only (skip header).
  tail -n +2 "$DATASET_CSV" >> "$MERGED_CSV"
  DATA_ROWS=$(awk 'END{print NR-1}' "$DATASET_CSV")
  if [[ "$DATA_ROWS" -gt 0 ]]; then
    MERGED_ROWS=$((MERGED_ROWS + DATA_ROWS))
  fi
done

if [[ "$HEADER_WRITTEN" -eq 0 ]]; then
  echo "[WARNING] No collaboration CSV found. Aggregated CSV not created."
  rm -f "$MERGED_CSV"
else
  echo "[OK] Aggregated CSV saved to: $MERGED_CSV"
  echo "[OK] Aggregated rows: $MERGED_ROWS"
  if [[ ${#MISSING_CSV[@]} -gt 0 ]]; then
    echo "[WARNING] Missing per-dataset CSV for: ${MISSING_CSV[*]}"
  fi
fi

echo
echo "============================================================"
echo "Batch finished"
echo "Succeeded: ${#SUCCEEDED[@]}"
echo "Failed: ${#FAILED[@]}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "Failed datasets: ${FAILED[*]}"
fi
echo "============================================================"

if [[ ${#FAILED[@]} -gt 0 ]]; then
  exit 1
fi

exit 0
