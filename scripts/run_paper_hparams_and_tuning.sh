#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPU_ID="${GPU_ID:-0}"
RUN_ROOT="$ROOT/results/runs/paper_hparams_tuning"
PAPER_DIR="$RUN_ROOT/paper"
TUNE_DIR="$RUN_ROOT/tuning"
STATUS="$RUN_ROOT/status.tsv"
REGISTRY="$ROOT/configs/paper_hyperparameters.tsv"
mkdir -p "$PAPER_DIR" "$TUNE_DIR" "$ROOT/results/runs/paper_hparams"
LOCK="$RUN_ROOT/pipeline.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "EditKG paper/tuning pipeline already running" >&2
    exit 0
fi
stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
record() { printf '%s\t%s\t%s\n' "$(stamp)" "$1" "$2" | tee -a "$STATUS"; }
paper_args() { awk -F '\t' -v wanted="$1" '$1 == wanted { print $3; exit }' "$REGISTRY"; }
run_logged() {
    local name="$1" log="$2"; shift 2
    if awk -F '\t' -v wanted="$name" \
      '$2 == "DONE" && $3 == wanted { found=1 } END { exit !found }' "$STATUS"; then
        record SKIP "$name already completed"
        return 0
    fi
    record START "$name"
    if "$@" > "$log" 2>&1; then record DONE "$name"; else record FAILED "$name"; fi
}

if [[ "${TUNING_ONLY:-0}" != "1" ]]; then
record PIPELINE "paper configurations started"
for dataset in last-fm amazon-book yelp2018; do
  for model in bpr gcmc lightgcn sgl cke ktup kgcn kgat kgin kgcl kgrec krdn; do
    read -r -a extra <<< "$(paper_args "$model")"
    run_logged "paper_${dataset}_${model}" "$PAPER_DIR/${dataset}_${model}.log" \
      env GPU_ID="$GPU_ID" EDITKG_DATA_ROOT="$ROOT/data/editkg-filtered" \
      "$ROOT/scripts/run_baseline.sh" "$model" "$dataset" "${extra[@]}"
  done
done

# Retry KGAT jobs that previously failed on Ampere. The launcher now uses the
# legacy TensorFlow 1.15 implementation on CPU and fixes sparse relation sizing.
record PIPELINE "KGAT compatibility recovery started"
for dataset in last-fm amazon-book yelp2018; do
  read -r -a extra <<< "$(paper_args kgat)"
  run_logged "paper_${dataset}_kgat" "$PAPER_DIR/${dataset}_kgat.log" \
    env GPU_ID="$GPU_ID" EDITKG_DATA_ROOT="$ROOT/data/editkg-filtered" \
    "$ROOT/scripts/run_baseline.sh" kgat "$dataset" "${extra[@]}"
done
record PIPELINE "KGAT compatibility recovery completed"
fi

# Compact model-specific screening: paper, common, and two targeted variants.
# Six candidates x seven models x three datasets = 126 runs of at most 25 epochs.
CANDIDATES="$ROOT/configs/tuning_candidates.tsv"
record PIPELINE "compact model-specific validation screening started"
for dataset in last-fm amazon-book yelp2018; do
  while IFS=$'\t' read -r model tag candidate_args; do
    [[ "$model" == "model" || -z "$model" ]] && continue
    read -r -a extra <<< "$candidate_args"
    name="tune_${dataset}_${model}_${tag}"
    if [[ "$model" =~ ^(bpr|lightgcn|sgl)$ ]]; then
      run_logged "$name" "$TUNE_DIR/${name}.log" \
        env GPU_ID="$GPU_ID" "$ROOT/scripts/run_baseline.sh" "$model" "$dataset" \
        "${extra[@]}" --epochs 25 --patience 4 --validation_only \
        --output "$TUNE_DIR/$name"
    else
      run_logged "$name" "$TUNE_DIR/${name}.log" \
        env GPU_ID="$GPU_ID" PYTHONPATH="$ROOT/baselines/external/recbole" \
        "$ROOT/.envs/editkg/bin/python" "$ROOT/scripts/run_recbole_validation.py" \
        --model "${model^^}" --dataset "$dataset" \
        --config_files "$ROOT/configs/recbole_lastfm.yaml" \
        "${extra[@]}" --epochs=25 --eval_step=5 --stopping_step=4
    fi
  done < "$CANDIDATES"
done
record PIPELINE "validation tuning completed; running selected final tests"
"$ROOT/.envs/editkg/bin/python" "$ROOT/scripts/select_and_run_tuned_finals.py" --gpu-id "$GPU_ID"
record PIPELINE "paper configurations, validation tuning, and selected final tests completed"
