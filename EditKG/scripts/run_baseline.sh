#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?usage: $0 MODEL [DATASET] [extra arguments]}"
shift
DATASET="${1:-last-fm}"
if [[ $# -gt 0 ]]; then shift; fi
case "$DATASET" in last-fm|amazon-book|yelp2018) ;; *) echo "dataset sconosciuto: $DATASET" >&2; exit 2;; esac
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPU_ID="${GPU_ID:-0}"
DATA_ROOT="${EDITKG_DATA_ROOT:-$ROOT/data/editkg-filtered}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
if [[ -x "$ROOT/.envs/editkg/bin/python" ]]; then DEFAULT_PYTHON="$ROOT/.envs/editkg/bin/python"; else DEFAULT_PYTHON="$(command -v python)"; fi
PYTHON="${EDITKG_PYTHON:-$DEFAULT_PYTHON}"
LEGACY_PYTHON="${KGAT_PYTHON:-$ROOT/.envs/kgat-legacy/bin/python}"
if [[ "${NSRS_PHASE:-legacy}" == legacy ]]; then
  mkdir -p "$ROOT/results/runs/baselines/$DATASET"
fi

case "${MODEL,,}" in
  gcmc|cke|ktup|kgcn)
    if [[ "${NSRS_PHASE:-legacy}" == legacy ]]; then cd "$ROOT"; fi
    EDITKG_DATA_ROOT="$DATA_ROOT" "$PYTHON" "$ROOT/scripts/prepare_baseline_data.py" "$DATASET"
    PYTHONPATH="$ROOT/baselines/external/recbole${PYTHONPATH:+:$PYTHONPATH}" \
      "$PYTHON" "$ROOT/baselines/external/recbole/run_recbole.py" \
      --model "${MODEL^^}" --dataset "$DATASET" \
      --config_files "$ROOT/configs/recbole_lastfm.yaml" "$@"
    ;;
  kgat)
    if [[ "${NSRS_PHASE:-legacy}" == legacy ]]; then cd "$ROOT/baselines/external/kgat/Model"; fi
    # TensorFlow 1.15/CUDA 10 is incompatible with Ampere; default KGAT to CPU.
    env OMP_NUM_THREADS="${KGAT_CPU_THREADS:-8}" MKL_NUM_THREADS="${KGAT_CPU_THREADS:-8}" \
    "$LEGACY_PYTHON" "$ROOT/baselines/external/kgat/Model/Main.py" --model_type kgat --alg_type bi --dataset "$DATASET" \
      --data_path "$DATA_ROOT/" --regs '[1e-5,1e-5]' --layer_size '[128,128]' \
      --embed_size 128 --lr 0.0005 --epoch 300 --verbose 1 --pretrain 0 \
      --batch_size 4096 --node_dropout '[0.1]' --mess_dropout '[0.1,0.1]' \
      --use_att True --use_kge True --gpu_id "${KGAT_GPU_ID:--1}" --Ks '[20]' "$@"
    ;;
  kgin)
    if [[ "${NSRS_PHASE:-legacy}" == legacy ]]; then cd "$ROOT/baselines/external/kgin"; fi
    "$PYTHON" "$ROOT/baselines/external/kgin/main.py" --dataset "$DATASET" --data_path "$DATA_ROOT/" --dim 128 \
      --lr 0.0005 --epoch 300 --batch_size 4096 --node_dropout True \
      --node_dropout_rate 0.5 --mess_dropout True --mess_dropout_rate 0.1 \
      --gpu_id 0 --context_hops 2 --Ks '[20]' "$@"
    ;;
  kgcl)
    if [[ "${NSRS_PHASE:-legacy}" == legacy ]]; then cd "$ROOT/baselines/external/kgrec"; fi
    "$PYTHON" "$ROOT/baselines/external/kgrec/run_kgcl.py" --dataset "$DATASET" --data_path "$DATA_ROOT/" \
      --dim 128 --lr 0.0005 --epoch 300 --batch_size 4096 --gpu_id 0 --Ks '[20]' "$@"
    ;;
  kgrec)
    if [[ "${NSRS_PHASE:-legacy}" == legacy ]]; then cd "$ROOT/baselines/external/kgrec"; fi
    "$PYTHON" "$ROOT/baselines/external/kgrec/run_kgrec.py" --dataset "$DATASET" --data_path "$DATA_ROOT/" \
      --dim 128 --lr 0.0005 --epoch 300 --batch_size 4096 --gpu_id 0 --Ks '[20]' "$@"
    ;;
  krdn)
    if [[ "${NSRS_PHASE:-legacy}" == legacy ]]; then cd "$ROOT/baselines/external/krdn"; fi
    mkdir -p result
    "$PYTHON" "$ROOT/baselines/external/krdn/main.py" --dataset "$DATASET" --data_path "$DATA_ROOT/" \
      --dim 128 --epoch 300 --batch_size 4096 --lr 0.0005 --context_hops 2 \
      --num_neg_sample 400 --margin 0.7 --max_iter 2 --gpu_id 0 --Ks '[20]' "$@"
    ;;
  editkg)
    if [[ "${NSRS_PHASE:-legacy}" == legacy ]]; then cd "$ROOT"; fi
    EDITKG_DATA_ROOT="$DATA_ROOT" EDITKG_EVAL_STEP="${EDITKG_EVAL_STEP:-10}" \
      EDITKG_STOPPING_STEP="${EDITKG_STOPPING_STEP:-10}" \
      "$PYTHON" "$ROOT/main.py" --dataset "$DATASET" --data_path "$DATA_ROOT/" \
      --dim 100 --lr 0.0005 --epoch 300 --batch_size 4096 --gpu_id 0 --Ks '[20]' "$@"
    ;;
  bpr|lightgcn|sgl)
    if [[ "${NSRS_PHASE:-legacy}" == legacy ]]; then cd "$ROOT"; fi
    "$PYTHON" "$ROOT/baselines/simple.py" --model "${MODEL,,}" --dataset "$DATASET" \
      --data_path "$DATA_ROOT" --dim 128 --lr 0.0005 \
      --batch_size 4096 --epochs 300 --gpu_id 0 "$@"
    ;;
  *) echo "modello sconosciuto: $MODEL" >&2; exit 2 ;;
esac
