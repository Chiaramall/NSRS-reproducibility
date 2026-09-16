#!/usr/bin/env python3
"""Run the complete, sequential and resumable CAFE paper pipeline."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "paper_pipeline"
LOGS = OUTPUT / "logs"
STATUS = OUTPUT / "status.json"
LOCK = OUTPUT / "pipeline.lock"
CAFE_PYTHON = ROOT / ".venv" / "bin" / "python"
BASELINE_PYTHON = ROOT / ".venv-baselines" / "bin" / "python"
PGPR_ROOT = ROOT / "baselines" / "models" / "pgpr"
DATASETS = {
    "beauty": {
        "pgpr": "beauty",
        "pgpr_dir": "Amazon_Beauty",
        "embedding": ROOT / "data" / "Beauty" / "kg_embedding.ckpt",
    },
    "cell": {
        "pgpr": "cell",
        "pgpr_dir": "Amazon_Cellphones",
        "embedding": ROOT / "data" / "Cellphones_Accessories" / "embedding_des_epoch_30.ckpt",
    },
    "clothing": {
        "pgpr": "cloth",
        "pgpr_dir": "Amazon_Clothing",
        "embedding": ROOT / "data" / "Clothing" / "embedding_des_epoch_29.ckpt",
    },
    "cd": {
        "pgpr": "cd",
        "pgpr_dir": "Amazon_CDs",
        "embedding": ROOT / "data" / "CDs_Vinyl" / "embedding_des_epoch_30.ckpt",
    },
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_status() -> dict[str, object]:
    if STATUS.exists():
        return json.loads(STATUS.read_text(encoding="utf-8"))
    return {"created_at": now(), "stages": {}}


def update_status(
    state: dict[str, object], stage: str, status: str, detail: str = ""
) -> None:
    stages = state.setdefault("stages", {})
    assert isinstance(stages, dict)
    stages[stage] = {"status": status, "updated_at": now(), "detail": detail}
    state["updated_at"] = now()
    atomic_json(STATUS, state)


def process_identity(pid: int) -> tuple[int, str] | None:
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
        start_time = int(stat.rsplit(")", 1)[1].split()[19])
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
        return start_time, raw.replace(b"\0", b" ").decode(errors="replace")
    except (FileNotFoundError, PermissionError, IndexError, ValueError):
        return None


def discover_existing_cafe_jobs() -> dict[int, tuple[int, str]]:
    matches: dict[int, tuple[int, str]] = {}
    own_pid = os.getpid()
    patterns = ("scripts/reproducibility.py", "train_neural_symbol.py")
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        identity = process_identity(int(entry.name))
        if identity and any(pattern in identity[1] for pattern in patterns):
            matches[int(entry.name)] = identity
    return matches


def wait_for_existing_jobs(state: dict[str, object]) -> None:
    jobs = discover_existing_cafe_jobs()
    if not jobs:
        update_status(state, "wait_existing_jobs", "complete", "No existing CAFE jobs")
        return
    detail = "; ".join(f"PID {pid}: {value[1]}" for pid, value in sorted(jobs.items()))
    update_status(state, "wait_existing_jobs", "waiting", detail)
    print(">>> Existing CAFE work detected; it will not be interrupted:", flush=True)
    for pid, (_start, command) in sorted(jobs.items()):
        print(f"    PID {pid}: {command}", flush=True)
    while jobs:
        remaining: dict[int, tuple[int, str]] = {}
        for pid, original in jobs.items():
            current = process_identity(pid)
            if current is not None and current[0] == original[0]:
                remaining[pid] = original
        jobs = remaining
        if jobs:
            print(f">>> Waiting for existing PIDs: {sorted(jobs)}", flush=True)
            time.sleep(60)
    update_status(state, "wait_existing_jobs", "complete", "Existing jobs finished")


def run_command(
    command: list[str], log_path: Path, *, cwd: Path = ROOT, retries: int = 2
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(command)
    for attempt in range(1, retries + 2):
        print(f"\n>>> {printable} (attempt {attempt}/{retries + 1})", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{now()}] >>> {printable} (attempt {attempt})\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4"),
                    "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "4"),
                },
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            return_code = process.wait()
        if return_code == 0:
            return
        if attempt <= retries:
            print(f">>> Command failed with {return_code}; retrying in 60 seconds.", flush=True)
            time.sleep(60)
    raise subprocess.CalledProcessError(return_code, command)


def run_stage(
    state: dict[str, object], stage: str, command: list[str], log_path: Path,
    *, cwd: Path = ROOT
) -> None:
    update_status(state, stage, "running", " ".join(command))
    try:
        run_command(command, log_path, cwd=cwd)
    except Exception as error:
        update_status(state, stage, "failed", repr(error))
        raise
    update_status(state, stage, "complete")


def ensure_transe(state: dict[str, object], dataset: str) -> None:
    spec = DATASETS[dataset]
    destination = spec["embedding"]
    assert isinstance(destination, Path)
    stage = f"{dataset}.cafe_transe"
    if destination.exists() and destination.stat().st_size > 0:
        update_status(state, stage, "complete", f"Existing {destination.relative_to(ROOT)}")
        print(f">>> Found {destination.relative_to(ROOT)}; skipping CAFE TransE.", flush=True)
        return

    run_stage(
        state,
        f"{dataset}.pgpr_preprocess",
        [
            str(CAFE_PYTHON), "-u", "scripts/run_paper_pgpr_official.py",
            "--dataset", dataset, "--python", str(CAFE_PYTHON), "--preprocess-only",
        ],
        LOGS / f"{dataset}_pgpr_preprocess.log",
    )
    name = "cafe_pretrain_transe_seed_123"
    checkpoint = (
        PGPR_ROOT / "tmp" / str(spec["pgpr_dir"]) / name /
        "transe_model_sd_epoch_30.ckpt"
    )
    if not checkpoint.exists():
        run_stage(
            state,
            stage,
            [
                str(CAFE_PYTHON), "-u", "train_transe_model.py",
                "--dataset", str(spec["pgpr"]), "--name", name,
                "--seed", "123", "--epochs", "30", "--gpu", "0",
            ],
            LOGS / f"{dataset}_cafe_transe.log",
            cwd=PGPR_ROOT,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(checkpoint, temporary)
    temporary.replace(destination)
    provenance = {
        "dataset": dataset,
        "created_at": now(),
        "source": str(checkpoint.relative_to(ROOT)),
        "destination": str(destination.relative_to(ROOT)),
        "seed": 123,
        "epochs": 30,
        "note": "Clothing keeps the filename expected by released CAFE utils.py; the copied checkpoint is epoch 30.",
    }
    atomic_json(destination.with_suffix(destination.suffix + ".provenance.json"), provenance)
    update_status(state, stage, "complete", str(destination.relative_to(ROOT)))


def validate_environment() -> None:
    missing = [path for path in (CAFE_PYTHON, BASELINE_PYTHON, PGPR_ROOT) if not path.exists()]
    if missing:
        raise RuntimeError("Missing required environment paths: " + ", ".join(map(str, missing)))


def collect_results(datasets: list[str], include_baselines: bool) -> None:
    manifest = {"created_at": now(), "datasets": {}}
    for dataset in datasets:
        files = {
            "cafe_five_seed": ROOT / "results" / "reproducibility_paper" / dataset / "baseline_summary.json",
            "cafe_grid": ROOT / "results" / "reproducibility_paper" / dataset / "grid_summary.json",
        }
        if include_baselines:
            files.update({
                "recbole_baselines": ROOT / "results" / "paper_baselines" / dataset / "recbole" / "summary.json",
                "pgpr_baseline": ROOT / "results" / "paper_baselines" / dataset / "pgpr" / "summary.json",
            })
        dataset_results = {}
        for label, path in files.items():
            dataset_results[label] = {
                "path": str(path.relative_to(ROOT)),
                "data": json.loads(path.read_text(encoding="utf-8")) if path.exists() else None,
            }
        manifest["datasets"][dataset] = dataset_results
    atomic_json(OUTPUT / "final_manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS)
    )
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Another full paper pipeline is already running") from error
    lock_handle.write(str(os.getpid()))
    lock_handle.flush()

    validate_environment()
    state = load_status()
    state["pipeline_pid"] = os.getpid()
    state["datasets"] = args.datasets
    state["protocol"] = {
        "CAFE": "paper: Adam, lr=1e-4, batch=128, epochs=20",
        "seeds": [123, 124, 125, 126, 127],
        "grid_lr": [5e-5, 1e-4, 2e-4],
        "grid_rank_weight": [5.0, 10.0, 20.0],
    }
    atomic_json(STATUS, state)
    wait_for_existing_jobs(state)

    non_beauty = [dataset for dataset in args.datasets if dataset != "beauty"]
    if non_beauty:
        run_stage(
            state,
            "datasets.prepare",
            [
                str(BASELINE_PYTHON), "-u", "scripts/prepare_all_amazon_datasets.py",
                "--datasets", *non_beauty,
            ],
            LOGS / "datasets_prepare.log",
        )

    for dataset in args.datasets:
        ensure_transe(state, dataset)
        run_stage(
            state,
            f"{dataset}.cafe_preprocess",
            [str(CAFE_PYTHON), "-u", "preprocess.py", "--dataset", dataset, "--seed", "123"],
            LOGS / f"{dataset}_cafe_preprocess.log",
        )
        run_stage(
            state,
            f"{dataset}.cafe_reproducibility_and_grid",
            [
                str(CAFE_PYTHON), "-u", "scripts/cafe_paper_reproducibility.py",
                "all", "--dataset", dataset, "--python", str(CAFE_PYTHON),
            ],
            LOGS / f"{dataset}_cafe_reproducibility_and_grid.log",
        )

        if args.skip_baselines:
            continue
        run_stage(
            state,
            f"{dataset}.recbole_baselines",
            [
                str(BASELINE_PYTHON), "-u", "scripts/run_paper_recbole.py",
                "--dataset", dataset, "--gpu", "0",
            ],
            LOGS / f"{dataset}_recbole_baselines.log",
        )
        run_stage(
            state,
            f"{dataset}.pgpr_baseline",
            [
                str(BASELINE_PYTHON), "-u", "scripts/run_paper_pgpr_official.py",
                "--dataset", dataset, "--python", str(BASELINE_PYTHON), "--gpu", "0",
            ],
            LOGS / f"{dataset}_pgpr_baseline.log",
        )

    collect_results(args.datasets, not args.skip_baselines)
    state["completed_at"] = now()
    state["status"] = "complete"
    atomic_json(STATUS, state)
    print(">>> Full paper pipeline complete.", flush=True)


if __name__ == "__main__":
    main()
