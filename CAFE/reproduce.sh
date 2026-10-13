#!/usr/bin/env bash
# Usage: bash CAFE/reproduce.sh [all|prepare|cafe|baselines|legacy-grid] [DATASET ...]
# First extract/link the public archive as described in CAFE/baselines/DATASETS.md.
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
case "$stage" in all|prepare|cafe|baselines|legacy-grid) ;; *) echo 'Expected all, prepare, cafe, baselines or legacy-grid' >&2; exit 2;; esac
if [[ $# -eq 0 ]]; then set -- beauty cell clothing cd; fi
for dataset in "$@"; do
  case "$dataset" in beauty|cell|clothing|cd) ;; *) echo "Unknown dataset: $dataset" >&2; exit 2;; esac
done
# CAFE's original runner uses logical GPU 0.
export CUDA_VISIBLE_DEVICES="$GPU_ID"
cd "$REPO_ROOT/CAFE"
for dataset in "$@"; do
  if [[ "$stage" == all || "$stage" == prepare ]]; then
    if [[ "$dataset" == beauty ]]; then
      run "$PYTHON" scripts/prepare_paper_baselines.py
    else
      run "$PYTHON" scripts/prepare_all_amazon_datasets.py --datasets "$dataset"
      run "$PYTHON" scripts/run_paper_pgpr.py --dataset "$dataset" --seeds 123 --gpu 0
      run "$PYTHON" scripts/install_cafe_kg_embedding.py --dataset "$dataset"
    fi
  fi
  if [[ "$stage" == all || "$stage" == cafe ]]; then
    # Fixed paper parameters, five seeds, training + inference + test evaluation.
    run "$PYTHON" scripts/cafe_paper_reproducibility.py baseline --dataset "$dataset" --python "$PYTHON"
  fi
  if [[ "$stage" == legacy-grid ]]; then
    # Historical test-scored sensitivity analyses, not validation selection.
    run "$PYTHON" scripts/reproducibility.py all --dataset "$dataset" --python "$PYTHON"
    run "$PYTHON" scripts/cafe_paper_reproducibility.py all --dataset "$dataset" --python "$PYTHON"
  fi
  if [[ "$stage" == all || "$stage" == baselines ]]; then
    # Each tuner includes validation selection, refitting and final evaluation.
    run "$PYTHON" scripts/tune_paper_baselines.py --datasets "$dataset" --gpu 0
    run "$PYTHON" scripts/tune_paper_heteroembed.py --datasets "$dataset" --gpu 0
    run "$PYTHON" scripts/tune_knn_hybrids.py --datasets "$dataset"
    run "$PYTHON" scripts/run_paper_pgpr.py --dataset "$dataset" --gpu 0
  fi
done
