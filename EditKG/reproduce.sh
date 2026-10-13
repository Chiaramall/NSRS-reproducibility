#!/usr/bin/env bash
# Usage: bash EditKG/reproduce.sh [all|search|final|report|plan|validation|historical-test] [DATASET ...]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
GPU_ID="${GPU_ID:-0}"
run() {
  printf '%q ' "$@"
  printf '\n'
  if [[ "${DRY_RUN:-0}" != 1 ]]; then "$@"; fi
}
stage="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
case "$stage" in all|search|final|report|plan|validation|historical-test) ;; *) echo 'Expected all, search, final, report, plan, validation or historical-test' >&2; exit 2;; esac
if [[ $# -eq 0 ]]; then set -- last-fm amazon-book yelp2018; fi
for dataset in "$@"; do
  case "$dataset" in last-fm|amazon-book|yelp2018) ;; *) echo "Unknown dataset: $dataset" >&2; exit 2;; esac
done
cd "$REPO_ROOT/EditKG"
export GPU_ID
case "$stage" in
  all|search|final|report|plan)
    read -r -a final_seeds <<< "${FINAL_SEEDS:-2020 2021 2022 2023 2024}"
    run "$PYTHON" scripts/benchmark.py --stage "$stage" --datasets "$@" \
      --trials "${TRIALS:-30}" --epochs "${EPOCHS:-300}" --gpu "$GPU_ID" \
      --output "${OUTPUT_DIR:-$REPO_ROOT/EditKG/results/benchmark}" \
      --final-seeds "${final_seeds[@]}"
    exit 0
    ;;
esac
if [[ "$stage" == validation ]]; then
  export EDITKG_DATA_ROOT="$PWD/data/editkg-validation"
else
  export EDITKG_DATA_ROOT="$PWD/data/editkg-filtered"
fi
# Honor an explicitly selected interpreter; otherwise use the activated environment.
export EDITKG_PYTHON="${EDITKG_PYTHON:-$PYTHON}"
for dataset in "$@"; do
  run bash scripts/run_baseline.sh editkg "$dataset"
  while IFS=$'\t' read -r model source_basis arguments; do
    [[ -z "$model" || "$model" == \#* ]] && continue
    # The versioned TSV contains whitespace-separated CLI arguments, no shell code.
    read -r -a extra <<< "$arguments"
    run bash scripts/run_baseline.sh "$model" "$dataset" "${extra[@]}"
  done < configs/paper_hyperparameters.tsv
done
