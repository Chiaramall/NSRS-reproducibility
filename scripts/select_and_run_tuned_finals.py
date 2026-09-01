#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

PATTERN = re.compile(r"tune_(last-fm|amazon-book|yelp2018)_([a-z0-9]+)_([a-z0-9]+)")


def validation_result(path):
    for line in reversed(path.read_text(errors="replace").splitlines()):
        if "VALIDATION_RESULT=" in line:
            return json.loads(line.split("VALIDATION_RESULT=", 1)[1])
    return None


def load_candidates(path):
    with path.open(newline="") as stream:
        return {(row["model"], row["tag"]): row["args"]
                for row in csv.DictReader(stream, delimiter="\t")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/runs/paper_hparams_tuning")
    parser.add_argument("--gpu-id", default="0")
    args = parser.parse_args()
    project = Path(__file__).resolve().parent.parent
    run_root = project / args.root
    tuning_dir = run_root / "tuning"
    final_dir = run_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    registry = load_candidates(project / "configs/tuning_candidates.tsv")
    best = {}
    for log in tuning_dir.glob("tune_*.log"):
        match = PATTERN.fullmatch(log.stem)
        result = validation_result(log)
        if not match or not result:
            continue
        dataset, model, tag = match.groups()
        candidate_args = registry.get((model, tag))
        if candidate_args is None:
            continue
        recall = float(result.get("recall@20", result.get("valid_recall", -1)))
        ndcg = float(result.get("ndcg@20", result.get("valid_ndcg", -1)))
        key = (dataset, model)
        candidate = {"dataset": dataset, "model": model, "tag": tag,
                     "args": candidate_args, "valid_recall": recall,
                     "valid_ndcg": ndcg, "source_log": str(log)}
        if key not in best or (recall, ndcg) > (best[key]["valid_recall"], best[key]["valid_ndcg"]):
            best[key] = candidate
    selected = sorted(best.values(), key=lambda x: (x["dataset"], x["model"]))
    (run_root / "selected_configs.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n"
    )
    for choice in selected:
        dataset, model = choice["dataset"], choice["model"]
        command = [str(project / "scripts" / "run_baseline.sh"), model, dataset]
        command += shlex.split(choice["args"])
        if model in {"bpr", "lightgcn", "sgl"}:
            command += ["--epochs", "150", "--patience", "10",
                        "--output", str(final_dir)]
        else:
            command += ["--epochs=150", "--eval_step=10", "--stopping_step=10"]
        log_path = final_dir / f"{dataset}_{model}_{choice['tag']}.log"
        with log_path.open("w") as stream:
            completed = subprocess.run(command, cwd=project, stdout=stream,
                                       stderr=subprocess.STDOUT, check=False,
                                       env={**os.environ, "GPU_ID": args.gpu_id})
        choice["final_log"] = str(log_path)
        choice["final_returncode"] = completed.returncode
    (run_root / "selected_configs.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
