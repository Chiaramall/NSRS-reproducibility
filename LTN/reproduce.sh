#!/usr/bin/env bash
# Usage: bash LTN/reproduce.sh [all|mindreader|movielens]
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
if [[ $# -gt 1 ]]; then echo 'Expected one stage' >&2; exit 2; fi
case "$stage" in all|mindreader|movielens) ;; *) echo 'Expected all, mindreader or movielens' >&2; exit 2;; esac
cd "$REPO_ROOT/LTN"
if [[ "$stage" == all || "$stage" == mindreader ]]; then
  run "$PYTHON" main.py
  run "$PYTHON" mindreader_large_embeddings.py --stage all
  run "$PYTHON" mindreader_baselines.py --stage all
  run "$PYTHON" mindreader_classic_baselines.py --stage all
fi
if [[ "$stage" == all || "$stage" == movielens ]]; then
  run "$PYTHON" -m movielens_paper_benchmark.download --datasets ml-100k ml-1m --root data/movielens
  run "$PYTHON" -m movielens_paper_benchmark --config movielens_paper_benchmark/configs/full.json --stage all --device "${DEVICE:-auto}"
fi
