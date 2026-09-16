"""Command-line entry point for the isolated MovieLens benchmark."""
from __future__ import annotations

import argparse
import copy
import fcntl
import os
import importlib.metadata
import json
import platform
import signal
import socket
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

# Required by deterministic CUDA matrix multiplications (CUDA >= 10.2). It is
# set before importing torch, and is also shown explicitly in the run commands.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from .config import (
    SUPPORTED_DATASETS,
    SUPPORTED_MODELS,
    load_config,
    validate_config,
)
from .download import URLS
from .evaluation import BenchmarkRunner, PROTOCOL_VERSION
from .report import write_reports, write_search_audit
from .search import atomic_json_dump


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "full.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def resolve_device(requested: str) -> str:
    normalized = requested.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(normalized)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"invalid device {requested!r}") from error
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("only CPU and CUDA devices are supported")
    if device.type == "cpu":
        if device.index is not None:
            raise ValueError("CPU device indices are not supported")
        return "cpu"
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false; use --device cpu"
        )
    index = torch.cuda.current_device() if device.index is None else device.index
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device index {index} is outside the available range "
            f"[0, {torch.cuda.device_count() - 1}]"
        )
    return f"cuda:{index}"


def _selected(
    values: Optional[Sequence[Any]],
    configured: Sequence[Any],
    label: str,
) -> list:
    if values is None:
        return list(configured)
    missing = [value for value in values if value not in configured]
    if missing:
        raise ValueError(
            f"{label} {missing} are not enabled by the selected configuration"
        )
    if len(set(values)) != len(values):
        raise ValueError(f"{label} selection contains duplicates")
    return list(values)


def resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = copy.deepcopy(load_config(args.config))
    if args.data_root is not None:
        config["data_root"] = args.data_root
    if args.output_root is not None:
        config["output_root"] = args.output_root
    if args.search_budget is not None:
        config["search"]["budget"] = args.search_budget
    if args.validation_seeds is not None:
        config["search"]["validation_seeds"] = args.validation_seeds
    if args.final_seeds is not None:
        config["final_seeds"] = args.final_seeds
    if args.max_epochs is not None:
        config["training"]["max_epochs"] = args.max_epochs
    if args.checkpoint_policy is not None:
        config["checkpoint_policy"] = args.checkpoint_policy

    config["datasets"] = _selected(
        args.datasets, config["datasets"], "datasets"
    )
    config["models"] = _selected(args.models, config["models"], "models")
    requested_proportions = (
        None
        if args.proportions is None
        else [float(value) for value in args.proportions]
    )
    config["proportions"] = _selected(
        requested_proportions, config["proportions"], "proportions"
    )
    validate_config(config)
    return config


def _manifest(
    config: Mapping[str, Any], device: str, command: Sequence[str]
) -> Dict[str, Any]:
    cuda = None
    if torch.device(device).type == "cuda":
        index = torch.device(device).index
        cuda = {
            "device_count": torch.cuda.device_count(),
            "selected_index": index,
            "selected_name": torch.cuda.get_device_name(index),
            "cuda_runtime": torch.version.cuda,
        }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "generated_at_utc": _utc_now(),
        "command": list(command),
        "resolved_config": dict(config),
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "LTNtorch": _version("LTNtorch"),
            "scipy": _version("scipy"),
            "device": device,
            "cuda": cuda,
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        },
        "datasets": {
            name: {"official_url": URLS[name]} for name in config["datasets"]
        },
        "protocol": {
            "feedback": (
                f"positive if rating >= {config['positive_threshold']}; "
                f"negative if rating <= {config['negative_threshold']}; "
                "intermediate ratings are neutral"
            ),
            "split": (
                "strict per-user temporal leave-two-positive-out: penultimate "
                "positive for validation, final positive for test, with at least "
                f"{config.get('min_train_positives', 1)} positive in training; "
                "events at or after validation are excluded from training; "
                f"warm-start targets required={config.get('require_warm_start_targets', False)}"
            ),
            "sparsification": (
                "deterministic nested per-user prefixes for 100/80/60/40/20 "
                "percent, retaining one positive anchor; validation/test and "
                "candidate masks remain fixed"
            ),
            "training_signals": (
                "MF-MSE and LTN use observed positive and explicit-negative "
                "training ratings; BPR-MF, EASE and LightGCN use positive implicit "
                "training interactions; rating 3 is neutral"
            ),
            "negative_sampling": (
                "unobserved negatives are sampled using sparse-training positives "
                "only; removed, validation and test labels are never consulted"
            ),
            "genre_information": (
                "MovieLens item genres are static catalog metadata; user genre "
                "preferences use non-neutral sparse-training ratings only; the "
                "catalog-wide rule is a transductive side-information adaptation, "
                "not direct MindReader genre feedback"
            ),
            "ltn_comparability": (
                "MF-MSE and LTN share the same seeded MF scorer; the weighted "
                "genre/data aggregation is p-mean error with p=2 and recovers the "
                "original two-formula SatAgg when genre_rule_weight=1"
            ),
            "model_selection": (
                "validation-only deterministic random search; test metrics are "
                "never computed during hyperparameter selection"
            ),
            "ranking": (
                "paper-compatible sampled-100 ranking: one held-out positive "
                "plus exactly 100 fixed unrated negatives per user and stage; "
                "candidate sets are identical across models, search/final seeds, "
                "and training proportions"
            ),
            "final_statistics": (
                "paired final seeds across models; arithmetic mean and population "
                "standard deviation (ddof=0) across final runs"
            ),
        },
    }


def write_manifest(
    config: Mapping[str, Any], device: str, command: Sequence[str]
) -> Dict[str, Any]:
    """Append an attempted-command manifest without replacing canonical files."""
    output = Path(config["output_root"])
    manifest = _manifest(config, device, command)
    stamp = str(manifest["generated_at_utc"]).replace(":", "-")
    atomic_json_dump(
        manifest,
        output
        / "manifests"
        / f"{stamp}-pid-{os.getpid()}-{os.urandom(4).hex()}.json",
    )
    return manifest


def write_canonical_manifest(
    config: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    """Publish canonical config/protocol only after a successful command."""
    output = Path(config["output_root"])
    atomic_json_dump(dict(config), output / "resolved_config.json")
    atomic_json_dump(dict(manifest), output / "protocol.json")


def _interrupt_on_sigterm(signum: int, frame: Any) -> None:
    """Turn a normal nohup stop into a cleanup-aware interruption."""
    del signum, frame
    raise KeyboardInterrupt


@contextmanager
def exclusive_output_lock(output_root: Path):
    """Prevent concurrent writers from corrupting one benchmark output tree."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / ".benchmark.lock"
    pid_path = output_root / "benchmark.pid"
    handle = path.open("a+", encoding="utf-8")
    locked = False
    previous_sigterm = None
    sigterm_handler_installed = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except BlockingIOError as error:
            handle.seek(0)
            owner = handle.read().strip() or "owner metadata unavailable"
            raise RuntimeError(
                f"another benchmark process is writing to {output_root}: {owner}"
            ) from error
        handle.seek(0)
        handle.truncate()
        json.dump(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "started_at_utc": _utc_now(),
            },
            handle,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        if threading.current_thread() is threading.main_thread():
            previous_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, _interrupt_on_sigterm)
            sigterm_handler_installed = True
        pid_path.write_text(f"{os.getpid()}\n", encoding="ascii")
        yield
    finally:
        if locked:
            try:
                if pid_path.read_text(encoding="ascii").strip() == str(
                    os.getpid()
                ):
                    pid_path.unlink()
            except OSError:
                pass
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        if sigterm_handler_installed:
            signal.signal(signal.SIGTERM, previous_sigterm)
        handle.close()


def _percent(completed: int, planned: int) -> str:
    return "100.0%" if planned == 0 else f"{100.0 * completed / planned:.1f}%"


def print_status(status: Mapping[str, Any], details: bool = False) -> None:
    search = status["search"]
    final = status["final"]
    print(
        "Search: "
        f"{search['completed_fits']}/{search['planned_fits']} fit complete "
        f"({_percent(search['completed_fits'], search['planned_fits'])}); "
        f"{search['remaining_fits']} mancanti"
    )
    print(
        "Final runs: "
        f"{final['completed_runs']}/{final['planned_runs']} complete "
        f"({_percent(final['completed_runs'], final['planned_runs'])}); "
        f"{final['remaining_runs']} mancanti"
    )
    if details:
        print()
        print(
            "dataset\tproportion\tmodel\tsearch state\tsearch fits\tfinal runs"
        )
        for row in status["combinations"]:
            search_state = (
                "missing"
                if not row["search_state_present"]
                else "compatible"
                if row["search_compatible"]
                else "stale"
            )
            if not row["dataset_verified"] and search_state == "compatible":
                search_state = "compatible*"
            print(
                f"{row['dataset']}\t{row['proportion']:.1f}\t{row['model']}\t"
                f"{search_state}\t"
                f"{row['search_completed_fits']}/{row['search_planned_fits']}\t"
                f"{row['final_completed_runs']}/{row['final_planned_runs']}"
            )


def _write_final_reports(
    runner: BenchmarkRunner,
    datasets: Sequence[str],
    models: Sequence[str],
    *,
    allow_partial: bool,
) -> None:
    summary = runner.summarize(
        datasets, models, require_complete=not allow_partial
    )
    expected_combinations = (
        len(datasets) * len(models) * len(runner.config["proportions"])
    )
    complete_combinations = sum(
        1
        for dataset in summary.values()
        for proportion in dataset.values()
        for record in proportion.values()
        if record.get("complete") is True
    )
    final_results_partial = complete_combinations != expected_combinations

    reports_dir = runner.output_root / "reports"
    paths = write_reports(summary, reports_dir)
    warning_path = reports_dir / "PARTIAL_RESULTS_WARNING.txt"
    audit_paths = (
        reports_dir / "search_configurations.csv",
        reports_dir / "embedding_validation_summary.csv",
    )
    audit_complete = True
    try:
        search_records = runner.search_records(datasets, models)
    except RuntimeError:
        if not allow_partial:
            raise
        audit_complete = False
        for stale_path in audit_paths:
            stale_path.unlink(missing_ok=True)
        print(
            "AVVISO: audit della search omesso perché incompleto; eventuali "
            "CSV precedenti sono stati rimossi.",
            flush=True,
        )
    else:
        paths.update(write_search_audit(search_records, reports_dir))

    partial_output = final_results_partial or not audit_complete
    if partial_output:
        warning_text = (
            "WARNING: PARTIAL RESULTS. One or more selected final runs are "
            "missing or incompatible, or the search audit is incomplete. Do "
            "not present these tables as the final 30-run benchmark.\n"
        )
        warning_path.write_text(warning_text, encoding="utf-8")
        markdown_path = paths["markdown"]
        markdown_path.write_text(
            "# WARNING: PARTIAL RESULTS\n\n"
            + warning_text
            + "\n"
            + markdown_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        latex_path = paths["latex"]
        latex_path.write_text(
            "% WARNING: PARTIAL RESULTS - NOT THE FINAL 30-RUN BENCHMARK.\n"
            + latex_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        paths["partial_warning"] = warning_path
    else:
        warning_path.unlink(missing_ok=True)

    print("Report generati:", flush=True)
    for label, path in sorted(paths.items()):
        print(f"  {label}: {path}", flush=True)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--stage",
        choices=("plan", "status", "search", "final", "report", "all"),
        default="all",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(SUPPORTED_DATASETS)
    )
    parser.add_argument(
        "--models", nargs="+", choices=sorted(SUPPORTED_MODELS)
    )
    parser.add_argument("--proportions", nargs="+", type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--search-budget", type=int)
    parser.add_argument("--validation-seeds", type=int)
    parser.add_argument("--final-seeds", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument(
        "--checkpoint-policy", choices=("none", "first", "all")
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="restart selected completed work instead of resuming it",
    )
    parser.add_argument("--details", action="store_true")
    parser.add_argument(
        "--allow-partial-report",
        action="store_true",
        help="explicitly allow report tables with missing final runs",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = resolve_config(args)
    device = resolve_device(args.device)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    runner = BenchmarkRunner(config, device=device, verbose=not args.quiet)
    datasets = config["datasets"]
    models = config["models"]

    if args.stage in {"plan", "status"}:
        print(f"Device risolto: {device}")
        print(f"Output: {runner.output_root}")
        print_status(
            runner.status(
                datasets,
                models,
                verify_datasets=args.stage == "status",
            ),
            details=args.details,
        )
        return 0

    with exclusive_output_lock(runner.output_root):
        manifest = write_manifest(
            config, device, [sys.executable, *sys.argv]
        )
        print(f"Device: {device}", flush=True)
        print(f"Output: {runner.output_root}", flush=True)
        if args.force:
            print(
                "ATTENZIONE: --force riavvia e sovrascrive il lavoro selezionato.",
                flush=True,
            )

        try:
            if args.stage in {"search", "all"}:
                runner.search(datasets, models, force=args.force)
            if args.stage in {"final", "all"}:
                runner.final(datasets, models, force=args.force)
            if args.stage in {"final", "report", "all"}:
                _write_final_reports(
                    runner,
                    datasets,
                    models,
                    allow_partial=args.allow_partial_report,
                )
        except KeyboardInterrupt:
            print(
                "\nInterrotto in sicurezza. Rilancia lo stesso comando senza --force "
                "per riprendere dall'ultimo risultato atomico.",
                file=sys.stderr,
                flush=True,
            )
            return 130

        final_status = runner.status(
            datasets, models, verify_datasets=True
        )
        if args.stage != "report":
            write_canonical_manifest(config, manifest)
        print_status(final_status, details=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
