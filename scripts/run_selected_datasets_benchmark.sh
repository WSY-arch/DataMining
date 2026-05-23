#!/usr/bin/env bash
set -u

# Batch runner for collaboration benchmark mode.
# It calls scripts/run_ucr_sbd_idk_compare.py for the datasets listed in datasets_choosed.csv.
#
# Usage examples:
#   bash scripts/run_selected_datasets_benchmark.sh --seeds 1 2 3 --no-viz
#   bash scripts/run_selected_datasets_benchmark.sh --quick --seeds 1 --n-samples 20 --no-viz
#
# Notes:
# - The script forces --mode benchmark.
# - All extra args are passed through to run_ucr_sbd_idk_compare.py.

DATASET_CSV=""
DATASETS=()
declare -A DATASET_LENGTHS=()

read_dataset_csv() {
  local csv_path="$1"
  while IFS=, read -r dataset length k || [[ -n "${dataset:-}" ]]; do
    dataset="${dataset%$'\r'}"
    length="${length%$'\r'}"
    k="${k%$'\r'}"
    if [[ "$dataset" == "Dataset" || -z "$dataset" ]]; then
      continue
    fi
    DATASETS+=("$dataset")
    DATASET_LENGTHS["$dataset"]="$length"
  done < "$csv_path"
}

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASET_CSV="$PROJECT_ROOT/datasets_choosed.csv"

if [[ -f "$DATASET_CSV" ]]; then
  read_dataset_csv "$DATASET_CSV"
else
  DATASETS=(
    Chinatown,
    SyntheticControl,
    MoteStrain,
    ECG200,
    CBF,
    TwoPatterns,
    ECGFiveDays,
    Plane,
    GunPoint,
    Wine,
    ArrowHead,
    Trace,
    Coffee,
    DiatomSizeReduction,
    Symbols,
    OSULeaf,
    Computers,
    ACSF1
  )
fi

if [[ "$QUICK" -eq 1 ]]; then
  DATASETS=("${DATASETS[@]:0:8}")
fi
PY_SCRIPT="$PROJECT_ROOT/scripts/run_ucr_sbd_idk_compare.py"
PYTHON_EXE="$PROJECT_ROOT/.venv/Scripts/python.exe"
RESULTS_ROOT="$PROJECT_ROOT/results/auto_ucr"
MERGED_CSV="$RESULTS_ROOT/collaboration_results_sbd_idk_all.csv"

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "[ERROR] Missing script: $PY_SCRIPT"
  exit 1
fi

if [[ -f "$PYTHON_EXE" ]]; then
  PYTHON_CMD=("$PYTHON_EXE")
else
  PYTHON_CMD=(python)
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
  length="${DATASET_LENGTHS[$dataset]:-0}"
  samples_per_class=0
  case "$dataset" in
    TwoPatterns)
      samples_per_class=300
      ;;
    Symbols)
      samples_per_class=120
      ;;
    Computers)
      samples_per_class=150
      ;;
  esac

  idk_threshold=0
  if [[ "$length" -gt 0 && "$length" -le 96 ]]; then
    idk_threshold=96
  fi

  echo
echo "[RUN] Dataset: $dataset"
  echo "------------------------------------------------------------"

  (
    cd "$PROJECT_ROOT" || exit 2
    run_args=(
      "$PY_SCRIPT"
      "$dataset"
      --mode benchmark
      --seeds 1 2 3 4 5 6 7 8 9 10
      --no-viz
      --sbd-backend reference
      --idk-preset accurate
      --idk-no-window-threshold "$idk_threshold"
      --idk-sample-size-max 0
      --idk-max-samples 0
    )
    if [[ "$samples_per_class" -gt 0 ]]; then
      run_args+=(--samples-per-class "$samples_per_class")
    fi
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
      run_args+=("${EXTRA_ARGS[@]}")
    fi
    "${PYTHON_CMD[@]}" "${run_args[@]}"
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
