#!/usr/bin/env bash
# Usage: bash NCR/reproduce.sh [all|tune|evaluate|variants] [DATASET ...]
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
case "$stage" in all|tune|evaluate|variants) ;; *) echo 'Expected all, tune, evaluate or variants' >&2; exit 2;; esac
if [[ $# -eq 0 ]]; then set -- ml100k01-1-5 5MoviesTV01-1-5 Electronics01-1-5; fi
for dataset in "$@"; do
  case "$dataset" in ml100k01-1-5|5MoviesTV01-1-5|Electronics01-1-5) ;; *) echo "Unknown dataset: $dataset" >&2; exit 2;; esac
done
cd "$REPO_ROOT/NCR"
for dataset in "$@"; do
  if [[ "$stage" == all || "$stage" == tune ]]; then
    run "$PYTHON" scripts/tune.py --models all --dataset "$dataset" \
      --max_evals "${MAX_EVALS:-30}" --gpu "$GPU_ID" --output_dir "tuning/$dataset"
  fi
  if [[ "$stage" == all || "$stage" == evaluate ]]; then
    run "$PYTHON" scripts/evaluate.py --models all --dataset "$dataset" \
      --best_dir "tuning/$dataset" --output_dir "tuning/${dataset}_seeds" \
      --seeds "${SEEDS:-2019,2020,2021,2022,2023}" --gpu "$GPU_ID"
  fi
  if [[ "$stage" == all || "$stage" == variants ]]; then
    run "$PYTHON" scripts/run_variants.py --dataset "$dataset" \
      --seeds "${SEEDS:-2019,2020,2021,2022,2023}" --gpu "$GPU_ID"
  fi
done
