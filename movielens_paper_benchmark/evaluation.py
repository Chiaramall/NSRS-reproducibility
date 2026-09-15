"""Leakage-safe orchestration for the isolated MovieLens benchmark."""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import struct
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .data import Feedback, Interaction, MovieLensData, load_movielens
from .metrics import aggregate_seed_results, evaluate_sampled_ranking
from .models.ease import EASE
from .models.lightgcn import LightGCN
from .models.ltn import LTNRecommender
from .models.mf import MatrixFactorization
from .search import atomic_json_dump, sample_configurations
from .splits import (
    DataSplit,
    derive_user_genre_preferences,
    sparsify_training,
    temporal_leave_two_out,
)


SEARCH_FORMAT_VERSION = 5
PROTOCOL_VERSION = 4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")

def _json_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=1)
def _code_digest() -> str:
    """Hash executable benchmark sources so changed code cannot reuse old runs."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if "tests" in relative.parts or "__pycache__" in relative.parts:
            continue
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _distribution_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


@lru_cache(maxsize=None)
def _runtime_signature(device: str) -> Dict[str, Any]:
    target = torch.device(device)
    cuda_name = None
    if target.type == "cuda" and torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(target)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "ltn": _distribution_version("LTNtorch"),
        "device": str(target),
        "cuda_name": cuda_name,
        "cuda_runtime": torch.version.cuda,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def _dataset_digest(data: MovieLensData) -> str:
    """Hash all model-relevant ratings, ids, thresholds and genre metadata."""
    digest = hashlib.sha256()
    header = {
        "variant": data.variant,
        "raw_user_ids": data.raw_user_ids,
        "raw_item_ids": data.raw_item_ids,
        "genre_names": data.genre_names,
        "positive_threshold": data.positive_threshold,
        "negative_threshold": data.negative_threshold,
    }
    digest.update(
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for genres in data.item_genres:
        digest.update(struct.pack("<q", len(genres)))
        for genre in genres:
            digest.update(struct.pack("<q", int(genre)))
    for event in data.interactions:
        digest.update(
            struct.pack(
                "<qqdqb",
                int(event.user_idx),
                int(event.item_idx),
                float(event.rating),
                int(event.timestamp),
                int(event.feedback),
            )
        )
    return digest.hexdigest()


def training_positive_items(
    interactions: Iterable[Interaction], n_users: int
) -> Dict[int, set]:
    """Return positive exclusions known from the supplied training data only."""
    result = {user: set() for user in range(int(n_users))}
    for event in interactions:
        if event.feedback is Feedback.POSITIVE:
            result[event.user_idx].add(event.item_idx)
    return result


def _proportion_label(proportion: float) -> str:
    return f"{float(proportion):.1f}"


def _event_key(event: Interaction) -> Tuple[int, int, int, float]:
    return (event.user_idx, event.timestamp, event.item_idx, event.rating)


def explicit_interactions(split: DataSplit) -> np.ndarray:
    """Return observed positive/negative ratings as binary pointwise targets."""
    rows = [
        (
            event.user_idx,
            event.item_idx,
            1.0 if event.feedback is Feedback.POSITIVE else 0.0,
        )
        for event in split.train
        if event.feedback is not Feedback.NEUTRAL
    ]
    return np.asarray(rows, dtype=np.float32).reshape(-1, 3)


def positive_interactions(split: DataSplit) -> np.ndarray:
    """Return positive implicit interactions for ranking-based recommenders."""
    rows = [
        (event.user_idx, event.item_idx)
        for event in split.train
        if event.feedback is Feedback.POSITIVE
    ]
    return np.asarray(rows, dtype=np.int64).reshape(-1, 2)


def seen_items(interactions: Iterable[Interaction]) -> Dict[int, set]:
    result: Dict[int, set] = {}
    for event in interactions:
        result.setdefault(int(event.user_idx), set()).add(int(event.item_idx))
    return result


def target_items(interactions: Iterable[Interaction]) -> Dict[int, int]:
    result: Dict[int, int] = {}
    for event in interactions:
        user = int(event.user_idx)
        if user in result:
            raise ValueError(f"more than one evaluation target for user {user}")
        result[user] = int(event.item_idx)
    return result


def evaluation_seen_items(
    data: MovieLensData,
    base_split: DataSplit,
    stage: str,
) -> Dict[int, set]:
    """Build the chronological candidate mask independently of sparsification.

    Validation masks the complete pre-validation history. Test masks every event
    strictly before the test event, including the validation item and any
    intervening event. Thus known items are not false negatives, and candidate
    sets remain identical for every training proportion.
    """
    if stage == "validation":
        return seen_items(base_split.train)
    if stage != "test":
        raise ValueError("stage must be validation or test")

    targets = {event.user_idx: event for event in base_split.test}
    result: Dict[int, set] = {user: set() for user in base_split.eligible_users}
    for event in data.interactions:
        target = targets.get(event.user_idx)
        if target is not None and _event_key(event) < _event_key(target):
            result[event.user_idx].add(event.item_idx)
    return result


def build_sampled_candidates(
    data: MovieLensData,
    base_split: DataSplit,
    stage: str,
    negative_candidates: int,
    candidate_seed: int,
) -> Dict[int, np.ndarray]:
    """Create fixed target-plus-negative sets without false negatives.

    Negatives are sampled without replacement from items that the user never
    rated anywhere in MovieLens. This uses no rating label as a training signal;
    it only prevents known interactions from being treated as negatives.
    """
    if stage not in {"validation", "test"}:
        raise ValueError("stage must be validation or test")
    if negative_candidates <= 0:
        raise ValueError("negative_candidates must be positive")
    events = base_split.validation if stage == "validation" else base_split.test
    targets = target_items(events)
    rated = seen_items(data.interactions)
    catalog = np.arange(data.n_items, dtype=np.int64)
    result: Dict[int, np.ndarray] = {}
    for raw_user in base_split.eligible_users:
        user = int(raw_user)
        target = int(targets[user])
        excluded = set(int(item) for item in rated.get(user, ()))
        excluded.add(target)
        available = np.asarray(
            [item for item in catalog if int(item) not in excluded], dtype=np.int64
        )
        if available.size < negative_candidates:
            raise ValueError(
                f"user {user} has only {available.size} unrated candidates; "
                f"{negative_candidates} are required"
            )
        rng = np.random.default_rng(
            _stable_seed("sampled-evaluation", candidate_seed, data.variant, stage, user)
        )
        negatives = rng.choice(
            available, size=negative_candidates, replace=False
        ).astype(np.int64, copy=False)
        result[user] = np.concatenate((negatives, np.asarray([target], dtype=np.int64)))
    return result


def _call_score_users(model: Any, users: np.ndarray, batch_size: int) -> Any:
    parameters = inspect.signature(model.score_users).parameters
    if "batch_size" in parameters:
        return model.score_users(users, batch_size=batch_size)
    return model.score_users(users)


def evaluate_model(
    model: Any,
    data: MovieLensData,
    base_split: DataSplit,
    stage: str,
    ranking: Mapping[str, Any],
    candidate_items: Optional[Mapping[int, Sequence[int]]] = None,
) -> Dict[str, float]:
    if stage not in {"validation", "test"}:
        raise ValueError("stage must be validation or test")
    events = base_split.validation if stage == "validation" else base_split.test
    candidates = candidate_items
    if candidates is None:
        candidates = build_sampled_candidates(
            data,
            base_split,
            stage,
            int(ranking["negative_candidates"]),
            int(ranking["candidate_seed"]),
        )
    user_batch_size = int(ranking["user_batch_size"])
    result = evaluate_sampled_ranking(
        lambda users: _call_score_users(model, users, user_batch_size),
        base_split.eligible_users,
        target_items(events),
        candidates,
        data.n_items,
        ks=ranking["ks"],
        batch_size=user_batch_size,
    )
    return result.means


def _genre_inputs(
    data: MovieLensData,
    split: DataSplit,
    min_support: int,
    seed: int,
    max_items_per_user: int,
) -> np.ndarray:
    """Create LTN genre constraints from training ratings and static metadata."""
    facts = derive_user_genre_preferences(
        data,
        split.train,
        min_support=int(min_support),
        liked_threshold=data.positive_threshold,
        disliked_threshold=data.negative_threshold,
    )
    preferences: Dict[int, Dict[str, set]] = {}
    for fact in facts:
        bucket = preferences.setdefault(fact.user_idx, {"liked": set(), "disliked": set()})
        key = "liked" if fact.feedback is Feedback.POSITIVE else "disliked"
        bucket[key].add(fact.genre_idx)
    item_genres = {item: genres for item, genres in enumerate(data.item_genres)}
    positive_items: Dict[int, set] = {}
    for event in split.train:
        if event.feedback is Feedback.POSITIVE:
            positive_items.setdefault(event.user_idx, set()).add(event.item_idx)
    return LTNRecommender.build_genre_examples(
        preferences,
        item_genres,
        max_items_per_user=int(max_items_per_user),
        seed=int(seed),
        excluded_positive_items=positive_items,
    )


def create_model(
    name: str,
    parameters: Mapping[str, Any],
    data: MovieLensData,
    seed: int,
    device: str,
    training: Mapping[str, Any],
) -> Any:
    params = dict(parameters)
    if name in {"mf-mse", "bpr-mf"}:
        return MatrixFactorization(
            data.n_users,
            data.n_items,
            embedding_dim=int(params["embedding_dim"]),
            objective="mse" if name == "mf-mse" else "bpr",
            biased=bool(params["biased"]),
            device=device,
            seed=int(seed),
        )
    if name == "ease":
        return EASE(
            regularization=float(params["regularization"]),
            n_users=data.n_users,
            n_items=data.n_items,
        )
    if name == "lightgcn":
        return LightGCN(
            data.n_users,
            data.n_items,
            embedding_dim=int(params["embedding_dim"]),
            num_layers=int(params["num_layers"]),
            learning_rate=float(params["learning_rate"]),
            l2_reg=float(params["l2"]),
            batch_size=int(params["batch_size"]),
            epochs=int(training["max_epochs"]),
            seed=int(seed),
            device=device,
        )
    if name in {"ltn", "ltn-genres"}:
        return LTNRecommender(
            data.n_users,
            data.n_items,
            embedding_dim=int(params["embedding_dim"]),
            biased=bool(params["biased"]),
            learning_rate=float(params["learning_rate"]),
            l2=float(params["l2"]),
            alpha=float(params["alpha"]),
            p=int(params.get("p", 2)),
            genre_rule_weight=float(params.get("genre_rule_weight", 0.0)),
            seed=int(seed),
            device=device,
        )
    raise ValueError(f"unsupported model: {name}")


def _history_length(model: Any) -> int:
    history = getattr(model, "history_", getattr(model, "history", None))
    if isinstance(history, Mapping):
        lengths = [len(value) for value in history.values() if isinstance(value, list)]
        return max(lengths, default=0)
    return len(history) if isinstance(history, list) else 0


def fit_model(
    name: str,
    model: Any,
    parameters: Mapping[str, Any],
    data: MovieLensData,
    sparse_split: DataSplit,
    base_split: DataSplit,
    seed: int,
    training: Mapping[str, Any],
    ranking: Mapping[str, Any],
    objective_name: str,
    validation_candidates: Mapping[int, Sequence[int]],
) -> Dict[str, Any]:
    """Fit using validation for early stopping; never access test events."""

    def objective(fitted: Any) -> float:
        metrics = evaluate_model(
            fitted,
            data,
            base_split,
            "validation",
            ranking,
            validation_candidates,
        )
        if objective_name not in metrics:
            raise KeyError(
                f"objective {objective_name!r} unavailable; computed {sorted(metrics)}"
            )
        return float(metrics[objective_name])
    # Only sparse-training positives are known to the sampler. Validation/test
    # targets and events removed by sparsification must not influence gradients.
    negative_exclusions = training_positive_items(
        sparse_split.train, data.n_users
    )

    genre_examples: Optional[np.ndarray] = None
    if name == "mf-mse":
        model.fit(
            explicit_interactions(sparse_split),
            epochs=int(training["max_epochs"]),
            batch_size=int(parameters["batch_size"]),
            learning_rate=float(parameters["learning_rate"]),
            weight_decay=float(parameters["l2"]),
            forbidden_items_by_user=negative_exclusions,
            num_negatives=int(training.get("negative_samples", 0)),
            seed=int(seed),
            validation_callback=objective,
            patience=int(training["patience"]),
        )
    elif name == "bpr-mf":
        model.fit(
            positive_interactions(sparse_split),
            epochs=int(training["max_epochs"]),
            batch_size=int(parameters["batch_size"]),
            learning_rate=float(parameters["learning_rate"]),
            forbidden_items_by_user=negative_exclusions,
            weight_decay=float(parameters["l2"]),
            seed=int(seed),
            validation_callback=objective,
            patience=int(training["patience"]),
        )
    elif name == "ease":
        model.fit(positive_interactions(sparse_split))
    elif name == "lightgcn":
        model.fit(
            positive_interactions(sparse_split),
            forbidden_items_by_user=negative_exclusions,
            validation_callback=objective,
            patience=int(training["patience"]),
        )
    elif name in {"ltn", "ltn-genres"}:
        if name == "ltn-genres":
            genre_examples = _genre_inputs(
                data,
                sparse_split,
                int(parameters["min_genre_support"]),
                seed,
                int(training.get("genre_samples_per_user", 20)),
            )
        model.fit(
            explicit_interactions(sparse_split),
            batch_size=int(parameters["batch_size"]),
            epochs=int(training["max_epochs"]),
            patience=int(training["patience"]),
            seed=int(seed),
            validation_callback=objective,
            genre_examples=genre_examples,
        )
    else:
        raise ValueError(f"unsupported model: {name}")

    counts = {
        "positive": sum(x.feedback is Feedback.POSITIVE for x in sparse_split.train),
        "negative": sum(x.feedback is Feedback.NEGATIVE for x in sparse_split.train),
        "neutral": sum(x.feedback is Feedback.NEUTRAL for x in sparse_split.train),
    }
    positive_catalog = {
        event.item_idx
        for event in sparse_split.train
        if event.feedback is Feedback.POSITIVE
    }
    target_support = {
        "positive_catalog_items": len(positive_catalog),
        "validation_targets_with_positive_support": sum(
            event.item_idx in positive_catalog
            for event in base_split.validation
        ),
        "test_targets_with_positive_support": sum(
            event.item_idx in positive_catalog for event in base_split.test
        ),
        "evaluation_users": len(base_split.eligible_users),
    }
    best_epoch = getattr(model, "best_epoch_", None)
    return {
        "n_training_events": len(sparse_split.train),
        "feedback_counts": counts,
        "target_support": target_support,
        "n_genre_constraints": 0 if genre_examples is None else int(len(genre_examples)),
        "epochs_trained": _history_length(model),
        "best_epoch": None if best_epoch is None else int(best_epoch) + 1,
    }


def _memory_snapshot(device: str) -> Dict[str, float]:
    """Return only per-run resettable measurements.

    CPU ru_maxrss is intentionally omitted because it is a cumulative lifetime
    process high-water mark and cannot support fair per-model comparisons.
    """
    target = torch.device(device)
    if target.type == "cuda" and torch.cuda.is_available():
        cuda_index = (
            torch.cuda.current_device() if target.index is None else target.index
        )
        value = float(torch.cuda.max_memory_allocated(cuda_index)) / (1024.0 ** 2)
        return {"cuda_peak_allocated_mib": value}
    return {}


def train_model(
    name: str,
    parameters: Mapping[str, Any],
    data: MovieLensData,
    base_split: DataSplit,
    sparse_split: DataSplit,
    seed: int,
    config: Mapping[str, Any],
    device: str,
    validation_candidates: Optional[Mapping[int, Sequence[int]]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    target = torch.device(device)
    model = create_model(name, parameters, data, seed, device, config["training"])
    # Constructing a CUDA model initializes the allocator. Resetting before the
    # first allocation raises "Invalid device argument" in some PyTorch builds.
    if (
        name != "ease"
        and target.type == "cuda"
        and torch.cuda.is_available()
    ):
        cuda_index = (
            torch.cuda.current_device() if target.index is None else target.index
        )
        torch.cuda.reset_peak_memory_stats(cuda_index)
    started = time.perf_counter()
    if validation_candidates is None:
        validation_candidates = build_sampled_candidates(
            data,
            base_split,
            "validation",
            int(config["ranking"]["negative_candidates"]),
            int(config["ranking"]["candidate_seed"]),
        )
    diagnostics = fit_model(
        name,
        model,
        parameters,
        data,
        sparse_split,
        base_split,
        seed,
        config["training"],
        config["ranking"],
        str(config["search"]["objective"]),
        validation_candidates,
    )
    return model, {
        "training_seconds": time.perf_counter() - started,
        "memory": {} if name == "ease" else _memory_snapshot(device),
        "diagnostics": diagnostics,
    }


def train_and_evaluate(
    name: str,
    parameters: Mapping[str, Any],
    data: MovieLensData,
    base_split: DataSplit,
    sparse_split: DataSplit,
    seed: int,
    config: Mapping[str, Any],
    device: str,
    validation_candidates: Optional[Mapping[int, Sequence[int]]] = None,
    test_candidates: Optional[Mapping[int, Sequence[int]]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Train with validation, then evaluate the test target exactly once."""
    model, payload = train_model(
        name,
        parameters,
        data,
        base_split,
        sparse_split,
        seed,
        config,
        device,
        validation_candidates,
    )
    started = time.perf_counter()
    payload["metrics"] = evaluate_model(
        model,
        data,
        base_split,
        "test",
        config["ranking"],
        test_candidates,
    )
    payload["inference_seconds"] = time.perf_counter() - started
    return model, payload


def save_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_checkpoint"):
        model.save_checkpoint(path)
    elif hasattr(model, "save"):
        model.save(path)
    else:
        torch.save(model.state_dict(), path)


def _configuration_digest(configurations: Sequence[Mapping[str, Any]]) -> str:
    return _json_digest(list(configurations))


def _search_protocol_digest(
    config: Mapping[str, Any],
    device: str,
    dataset_digest: str,
    dataset: str,
    proportion: float,
    model: str,
    configurations: Sequence[Mapping[str, Any]],
    runtime_signature: Optional[Mapping[str, Any]] = None,
) -> str:
    """Fingerprint every input capable of changing validation results."""
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "code_digest": _code_digest(),
        "runtime": dict(
            _runtime_signature(str(device))
            if runtime_signature is None
            else runtime_signature
        ),
        "dataset": dataset,
        "dataset_digest": dataset_digest,
        "proportion": float(proportion),
        "model": model,
        "feedback_thresholds": {
            "positive": config["positive_threshold"],
            "negative": config["negative_threshold"],
        },
        "min_train_positives": config.get("min_train_positives", 1),
        "require_warm_start_targets": config.get(
            "require_warm_start_targets", False
        ),
        "search": config["search"],
        "training": config["training"],
        "ranking": config["ranking"],
        "space": config["spaces"][model],
        "configurations": list(configurations),
    }
    return _json_digest(payload)


def _final_run_fingerprint(
    search_state: Mapping[str, Any],
    seed_index: int,
    seed: int,
    checkpoint_required: bool,
) -> str:
    return _json_digest(
        {
            "protocol_digest": search_state["protocol_digest"],
            "best_parameters": search_state["best_parameters"],
            "seed_index": int(seed_index),
            "seed": int(seed),
            "checkpoint_required": bool(checkpoint_required),
        }
    )


def _refresh_search_state(state: Dict[str, Any]) -> None:
    required_scores = int(state["validation_seeds"])
    complete = []
    completed_fits = 0
    for record in state["records"]:
        scores = record.get("validation_scores", [])
        completed_fits += len(scores)
        if len(scores) == required_scores:
            record["mean_validation_score"] = float(np.mean(scores))
            complete.append(record)
        else:
            record.pop("mean_validation_score", None)

    best = None
    for record in complete:
        if best is None or record["mean_validation_score"] > best["mean_validation_score"]:
            best = record
    state["completed"] = len(complete)
    state["completed_fits"] = completed_fits
    state["planned_fits"] = int(state["planned"]) * required_scores
    state["best_score"] = None if best is None else best["mean_validation_score"]
    state["best_parameters"] = None if best is None else best["parameters"]
    state["updated_at"] = _utc_now()


def _release_model(model: Any, device: str) -> None:
    del model
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


class BenchmarkRunner:
    """Run deterministic search/final stages with atomic, resumable outputs."""

    def __init__(
        self,
        config: Mapping[str, Any],
        device: str = "cpu",
        verbose: bool = True,
    ) -> None:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        self.config = dict(config)
        self.device = str(device)
        self.verbose = bool(verbose)
        self.output_root = Path(config["output_root"])
        self.output_root.mkdir(parents=True, exist_ok=True)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[{_utc_now()}] {message}", flush=True)

    def load_dataset(self, name: str) -> Tuple[MovieLensData, DataSplit]:
        data = load_movielens(
            self.config["data_root"],
            name,
            positive_threshold=float(self.config["positive_threshold"]),
            negative_threshold=float(self.config["negative_threshold"]),
        )
        min_train = int(self.config.get("min_train_positives", 1))
        return data, temporal_leave_two_out(
            data,
            min_train_positives=min_train,
            require_warm_start_targets=bool(
                self.config.get("require_warm_start_targets", False)

            ),
        )
    def _search_path(self, dataset: str, proportion: float, model: str) -> Path:
        return (
            self.output_root
            / "search"
            / dataset
            / _proportion_label(proportion)
            / f"{model}.json"
        )

    def _configurations(
        self, dataset: str, proportion: float, model: str
    ) -> Tuple[int, List[Dict[str, Any]]]:
        search = self.config["search"]
        sample_seed = _stable_seed(search["seed"], dataset, proportion, model)
        configurations = sample_configurations(
            self.config["spaces"][model],
            int(search["budget"]),
            sample_seed,
        )
        return sample_seed, configurations

    def _new_search_state(
        self,
        dataset: str,
        proportion: float,
        dataset_digest: str,
        model: str,
        sample_seed: int,
        configurations: Sequence[Mapping[str, Any]],
        runtime_signature: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        runtime = dict(
            _runtime_signature(self.device)
            if runtime_signature is None
            else runtime_signature
        )
        state: Dict[str, Any] = {
            "format_version": SEARCH_FORMAT_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "dataset": dataset,
            "proportion": float(proportion),
            "model": model,
            "objective": self.config["search"]["objective"],
            "search_seed": int(sample_seed),
            "validation_seeds": int(self.config["search"]["validation_seeds"]),
            "configuration_digest": _configuration_digest(configurations),
            "dataset_digest": dataset_digest,
            "code_digest": _code_digest(),
            "runtime": runtime,
            "protocol_digest": _search_protocol_digest(
                self.config, self.device, dataset_digest, dataset,
                float(proportion), model, configurations,
                runtime_signature=runtime,
            ),
            "planned": len(configurations),
            "records": [],
            "test_metrics_computed": False,
            "created_at": _utc_now(),
        }
        _refresh_search_state(state)
        return state

    def _load_or_create_search_state(
        self,
        path: Path,
        dataset: str,
        proportion: float,
        dataset_digest: str,
        model: str,
        sample_seed: int,
        configurations: Sequence[Mapping[str, Any]],
        force: bool,
    ) -> Dict[str, Any]:
        expected = self._new_search_state(
            dataset, proportion, dataset_digest, model, sample_seed, configurations
        )
        if force or not path.exists():
            return expected
        with path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        keys = (
            "format_version",
            "protocol_version",
            "dataset",
            "proportion",
            "model",
            "objective",
            "dataset_digest",
            "code_digest",
            "runtime",
            "protocol_digest",
            "search_seed",
            "validation_seeds",
            "configuration_digest",
            "planned",
        )
        mismatches = [key for key in keys if state.get(key) != expected.get(key)]
        if mismatches:
            raise ValueError(
                f"incompatible partial search state {path}; changed fields: "
                f"{mismatches}. Use --force only to restart it."
            )
        if state.get("test_metrics_computed") is not False:
            raise ValueError(f"invalid search audit flag in {path}")
        for index, record in enumerate(state.get("records", [])):
            if index >= len(configurations):
                raise ValueError(f"too many records in {path}")
            if record.get("index") != index or record.get("parameters") != configurations[index]:
                raise ValueError(f"configuration mismatch in {path}")
            scores = record.get("validation_scores", [])
            seed_values = record.get("validation_seed_values", [])
            if len(scores) > int(expected["validation_seeds"]):
                raise ValueError(f"too many seed scores in {path}")
            if len(seed_values) != len(scores):
                raise ValueError(f"seed/score mismatch in {path}")
            expected_seed_values = [
                _stable_seed("validation", validation_index)
                for validation_index in range(len(scores))
            ]
            if seed_values != expected_seed_values:
                raise ValueError(f"validation seed mismatch in {path}")
        _refresh_search_state(state)
        return state

    def search(
        self,
        datasets: Sequence[str],
        models: Sequence[str],
        force: bool = False,
    ) -> None:
        n_validation_seeds = int(self.config["search"]["validation_seeds"])
        for dataset_name in datasets:
            data, base_split = self.load_dataset(dataset_name)
            validation_candidates = build_sampled_candidates(
                data,
                base_split,
                "validation",
                int(self.config["ranking"]["negative_candidates"]),
                int(self.config["ranking"]["candidate_seed"]),
            )
            dataset_digest = _dataset_digest(data)
            self._log(
                f"loaded {dataset_name}: {data.n_users} users, {data.n_items} items, "
                f"{len(base_split.eligible_users)} evaluation users"
            )
            for proportion in self.config["proportions"]:
                for model_name in models:
                    path = self._search_path(dataset_name, float(proportion), model_name)
                    sample_seed, configurations = self._configurations(
                        dataset_name, float(proportion), model_name
                    )
                    state = self._load_or_create_search_state(
                        path,
                        dataset_name,
                        float(proportion),
                        dataset_digest,
                        model_name,
                        sample_seed,
                        configurations,
                        force,
                    )
                    if int(state["completed"]) == int(state["planned"]):
                        self._log(
                            f"search skip complete: {dataset_name} p={proportion} "
                            f"{model_name}"
                        )
                        continue
                    atomic_json_dump(state, path)

                    for index, parameters in enumerate(configurations):
                        if index < len(state["records"]):
                            record = state["records"][index]
                        else:
                            record = {
                                "index": index,
                                "parameters": parameters,
                                "validation_scores": [],
                                "validation_seed_values": [],
                                "training_seconds": [],
                                "diagnostics": [],
                            }
                            state["records"].append(record)
                            _refresh_search_state(state)
                            atomic_json_dump(state, path)

                        for validation_index in range(
                            len(record["validation_scores"]), n_validation_seeds
                        ):
                            seed = _stable_seed("validation", validation_index)
                            self._log(
                                f"search {dataset_name} p={proportion} {model_name} "
                                f"config {index + 1}/{len(configurations)} "
                                f"validation seed {validation_index + 1}/{n_validation_seeds}"
                            )
                            sparse = sparsify_training(
                                base_split, float(proportion), seed=seed
                            )
                            model, training_payload = train_model(
                                model_name,
                                parameters,
                                data,
                                base_split,
                                sparse,
                                seed,
                                self.config,
                                self.device,
                                validation_candidates,
                            )
                            validation = evaluate_model(
                                model,
                                data,
                                base_split,
                                "validation",
                                self.config["ranking"],
                                validation_candidates,
                            )
                            score = float(
                                validation[self.config["search"]["objective"]]
                            )
                            if not math.isfinite(score):
                                raise ValueError("validation objective is non-finite")
                            record["validation_scores"].append(score)
                            record["validation_seed_values"].append(int(seed))
                            record["training_seconds"].append(
                                training_payload["training_seconds"]
                            )
                            record["diagnostics"].append(
                                training_payload["diagnostics"]
                            )
                            _refresh_search_state(state)
                            atomic_json_dump(state, path)
                            _release_model(model, self.device)

                    self._log(
                        f"search complete: {dataset_name} p={proportion} {model_name}; "
                        f"best {state['objective']}={state['best_score']:.6f}"
                    )

    def _checkpoint_policy(self, seed_index: int) -> bool:
        policy = str(self.config.get("checkpoint_policy", "none"))
        return policy == "all" or (policy == "first" and seed_index == 0)

    @staticmethod
    def _run_complete(path: Path, expected_fingerprint: Optional[str] = None) -> bool:
        if not path.exists():
            return False
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            return False
        if payload.get("status") != "complete":
            return False
        if expected_fingerprint is not None:
            if payload.get("run_fingerprint") != expected_fingerprint:
                return False
        checkpoint = payload.get("checkpoint")
        return checkpoint is None or Path(checkpoint).exists()

    def final(
        self,
        datasets: Sequence[str],
        models: Sequence[str],
        force: bool = False,
    ) -> None:
        n_seeds = int(self.config["final_seeds"])
        for dataset_name in datasets:
            data, base_split = self.load_dataset(dataset_name)
            validation_candidates = build_sampled_candidates(
                data,
                base_split,
                "validation",
                int(self.config["ranking"]["negative_candidates"]),
                int(self.config["ranking"]["candidate_seed"]),
            )
            test_candidates = build_sampled_candidates(
                data,
                base_split,
                "test",
                int(self.config["ranking"]["negative_candidates"]),
                int(self.config["ranking"]["candidate_seed"]),
            )
            dataset_digest = _dataset_digest(data)
            for proportion in self.config["proportions"]:
                for model_name in models:
                    search_path = self._search_path(
                        dataset_name, float(proportion), model_name
                    )
                    if not search_path.exists():
                        raise FileNotFoundError(
                            f"search result is missing: {search_path}"
                        )
                    with search_path.open("r", encoding="utf-8") as handle:
                        search_state = json.load(handle)
                    sample_seed, configurations = self._configurations(
                        dataset_name, float(proportion), model_name
                    )
                    expected_search = self._new_search_state(
                        dataset_name,
                        float(proportion),
                        dataset_digest,
                        model_name,
                        sample_seed,
                        configurations,
                    )
                    if search_state.get("protocol_digest") != expected_search["protocol_digest"]:
                        raise RuntimeError(
                            f"search protocol is stale for {dataset_name}, "
                            f"p={proportion}, {model_name}; rerun search with --force"
                        )
                    if (
                        search_state.get("completed") != search_state.get("planned")
                        or search_state.get("best_parameters") is None
                    ):
                        raise RuntimeError(
                            f"search incomplete for {dataset_name}, "
                            f"p={proportion}, {model_name}"
                        )
                    parameters = search_state["best_parameters"]

                    for seed_index in range(n_seeds):
                        run_path = (
                            self.output_root
                            / "runs"
                            / dataset_name
                            / _proportion_label(float(proportion))
                            / model_name
                            / f"seed-{seed_index}.json"
                        )
                        seed = _stable_seed("final", seed_index)
                        checkpoint_required = self._checkpoint_policy(seed_index)
                        run_fingerprint = _final_run_fingerprint(
                            search_state,
                            seed_index,
                            seed,
                            checkpoint_required,
                        )
                        if not force and self._run_complete(run_path, run_fingerprint):
                            continue
                        self._log(
                            f"final {dataset_name} p={proportion} {model_name} "
                            f"seed {seed_index + 1}/{n_seeds}"
                        )
                        sparse = sparsify_training(
                            base_split, float(proportion), seed=seed
                        )
                        model, result = train_and_evaluate(
                            model_name,
                            parameters,
                            data,
                            base_split,
                            sparse,
                            seed,
                            self.config,
                            self.device,
                            validation_candidates,
                            test_candidates,
                        )
                        checkpoint: Optional[Path] = None
                        if checkpoint_required:
                            checkpoint = run_path.with_suffix(".pth")
                            save_model(model, checkpoint)
                        atomic_json_dump(
                            {
                                "status": "complete",
                                "protocol_version": PROTOCOL_VERSION,
                                "run_fingerprint": run_fingerprint,
                                "dataset": dataset_name,
                                "proportion": float(proportion),
                                "model": model_name,
                                "seed_index": seed_index,
                                "seed": seed,
                                "parameters": parameters,
                                "checkpoint_required": checkpoint_required,
                                "checkpoint": (
                                    None if checkpoint is None else str(checkpoint)
                                ),
                                "split": {
                                    "eligible_users": len(base_split.eligible_users),
                                    "dropped_users": len(base_split.dropped_users),
                                    "excluded_future_events": base_split.excluded_future_count,
                                },
                                **result,
                            },
                            run_path,
                        )
                        metrics_text = ", ".join(
                            f"{key}={value:.6f}"
                            for key, value in result["metrics"].items()
                        )
                        self._log(
                            f"final complete: {dataset_name} p={proportion} "
                            f"{model_name} seed={seed_index}; {metrics_text}"
                        )
                        _release_model(model, self.device)

    def status(
        self,
        datasets: Sequence[str],
        models: Sequence[str],
        *,
        verify_datasets: bool = True,
    ) -> Dict[str, Any]:
        dataset_digests: Dict[str, str] = {}
        if verify_datasets:
            for dataset in datasets:
                data, _ = self.load_dataset(dataset)
                dataset_digests[dataset] = _dataset_digest(data)

        search_planned = 0
        search_completed = 0
        final_planned = (
            len(datasets)
            * len(models)
            * len(self.config["proportions"])
            * int(self.config["final_seeds"])
        )
        final_completed = 0
        combinations: List[Dict[str, Any]] = []

        for dataset in datasets:
            for proportion in self.config["proportions"]:
                for model in models:
                    sample_seed, configurations = self._configurations(
                        dataset, float(proportion), model
                    )
                    planned_fits = (
                        len(configurations)
                        * int(self.config["search"]["validation_seeds"])
                    )
                    search_path = self._search_path(
                        dataset, float(proportion), model
                    )
                    completed_fits = 0
                    state = None
                    search_state_present = search_path.exists()
                    search_compatible = False
                    if search_state_present:
                        try:
                            with search_path.open("r", encoding="utf-8") as handle:
                                state = json.load(handle)
                            stored_runtime = state.get("runtime")
                            if not isinstance(stored_runtime, Mapping):
                                raise TypeError("search runtime must be a mapping")
                            digest = (
                                dataset_digests[dataset]
                                if verify_datasets
                                else str(state.get("dataset_digest", ""))
                            )
                            expected = self._new_search_state(
                                dataset,
                                float(proportion),
                                digest,
                                model,
                                sample_seed,
                                configurations,
                                runtime_signature=stored_runtime,
                            )
                            search_compatible = (
                                state.get("protocol_digest")
                                == expected["protocol_digest"]
                            )
                            if search_compatible:
                                completed_fits = int(
                                    state.get("completed_fits", 0)
                                )
                        except (OSError, ValueError, TypeError):
                            completed_fits = 0
                    completed_runs = 0
                    run_dir = (
                        self.output_root
                        / "runs"
                        / dataset
                        / _proportion_label(float(proportion))
                        / model
                    )
                    for seed_index in range(int(self.config["final_seeds"])):
                        expected_run = None
                        if search_compatible and state is not None:
                            if state.get("completed") == state.get("planned"):
                                if state.get("best_parameters") is not None:
                                    seed = _stable_seed("final", seed_index)
                                    expected_run = _final_run_fingerprint(
                                        state,
                                        seed_index,
                                        seed,
                                        self._checkpoint_policy(seed_index),
                                    )
                        if expected_run is not None and self._run_complete(
                            run_dir / f"seed-{seed_index}.json", expected_run
                        ):
                            completed_runs += 1
                    search_planned += planned_fits
                    search_completed += min(completed_fits, planned_fits)
                    final_completed += completed_runs
                    combinations.append(
                        {
                            "dataset": dataset,
                            "proportion": float(proportion),
                            "model": model,
                            "dataset_verified": bool(verify_datasets),
                            "search_state_present": search_state_present,
                            "search_compatible": search_compatible,
                            "search_completed_fits": completed_fits,
                            "search_planned_fits": planned_fits,
                            "final_completed_runs": completed_runs,
                            "final_planned_runs": int(self.config["final_seeds"]),
                        }
                    )

        return {
            "search": {
                "completed_fits": search_completed,
                "planned_fits": search_planned,
                "remaining_fits": search_planned - search_completed,
            },
            "final": {
                "completed_runs": final_completed,
                "planned_runs": final_planned,
                "remaining_runs": final_planned - final_completed,
            },
            "combinations": combinations,
        }

    def search_records(
        self, datasets: Sequence[str], models: Sequence[str]
    ) -> List[Dict[str, Any]]:
        """Return a validated, flat audit trail of every validation config."""
        rows: List[Dict[str, Any]] = []
        expected_scores = int(self.config["search"]["validation_seeds"])
        for dataset_name in datasets:
            data, _ = self.load_dataset(dataset_name)
            dataset_digest = _dataset_digest(data)
            for proportion in self.config["proportions"]:
                for model_name in models:
                    path = self._search_path(
                        dataset_name, float(proportion), model_name
                    )
                    if not path.exists():
                        raise RuntimeError(f"search result is missing: {path}")
                    try:
                        with path.open("r", encoding="utf-8") as handle:
                            state = json.load(handle)
                    except (OSError, ValueError) as error:
                        raise RuntimeError(
                            f"cannot read search result: {path}"
                        ) from error
                    runtime = state.get("runtime")
                    if not isinstance(runtime, Mapping):
                        raise RuntimeError(f"invalid runtime in {path}")
                    sample_seed, configurations = self._configurations(
                        dataset_name, float(proportion), model_name
                    )
                    expected = self._new_search_state(
                        dataset_name,
                        float(proportion),
                        dataset_digest,
                        model_name,
                        sample_seed,
                        configurations,
                        runtime_signature=runtime,
                    )
                    identity_keys = (
                        "format_version",
                        "protocol_version",
                        "dataset",
                        "proportion",
                        "model",
                        "objective",
                        "search_seed",
                        "validation_seeds",
                        "configuration_digest",
                        "dataset_digest",
                        "code_digest",
                        "protocol_digest",
                        "planned",
                    )
                    mismatches = [
                        key
                        for key in identity_keys
                        if state.get(key) != expected.get(key)
                    ]
                    if mismatches:
                        raise RuntimeError(
                            f"incompatible search result {path}; changed fields: "
                            f"{mismatches}"
                        )
                    if state.get("test_metrics_computed") is not False:
                        raise RuntimeError(f"invalid search audit flag in {path}")
                    if (
                        state.get("completed") != state.get("planned")
                        or len(state.get("records", [])) != len(configurations)
                    ):
                        raise RuntimeError(f"incomplete search result: {path}")
                    for index, record in enumerate(state["records"]):
                        scores = record.get("validation_scores", [])
                        seed_values = record.get("validation_seed_values", [])
                        expected_seed_values = [
                            _stable_seed("validation", validation_index)
                            for validation_index in range(expected_scores)
                        ]
                        if (
                            record.get("index") != index
                            or record.get("parameters") != configurations[index]
                            or len(scores) != expected_scores
                            or seed_values != expected_seed_values
                        ):
                            raise RuntimeError(
                                f"invalid search record {index} in {path}"
                            )
                        rows.append(
                            {
                                "dataset": dataset_name,
                                "proportion": float(proportion),
                                "model": model_name,
                                "objective": str(expected["objective"]),
                                "config_index": index,
                                "parameters": dict(record["parameters"]),
                                "validation_scores": [
                                    float(score) for score in scores
                                ],
                                "validation_seed_values": [
                                    int(seed) for seed in seed_values
                                ],
                                "mean_validation_score": float(np.mean(scores)),
                            }
                        )
        return rows

    def summarize(
        self,
        datasets: Sequence[str],
        models: Sequence[str],
        *,
        require_complete: bool = True,
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        expected_runs = int(self.config["final_seeds"])
        for dataset_name in datasets:
            data, _ = self.load_dataset(dataset_name)
            dataset_digest = _dataset_digest(data)
            summary[dataset_name] = {}
            for proportion in self.config["proportions"]:
                label = _proportion_label(float(proportion))
                summary[dataset_name][label] = {}
                for model_name in models:
                    search_path = self._search_path(
                        dataset_name, float(proportion), model_name
                    )
                    if not search_path.exists():
                        continue
                    try:
                        with search_path.open("r", encoding="utf-8") as handle:
                            search_state = json.load(handle)
                    except (OSError, ValueError):
                        continue
                    sample_seed, configurations = self._configurations(
                        dataset_name, float(proportion), model_name
                    )
                    stored_runtime = search_state.get("runtime")
                    if not isinstance(stored_runtime, Mapping):
                        continue
                    expected_search = self._new_search_state(
                        dataset_name,
                        float(proportion),
                        dataset_digest,
                        model_name,
                        sample_seed,
                        configurations,
                        runtime_signature=stored_runtime,
                    )
                    if (
                        search_state.get("protocol_digest")
                        != expected_search["protocol_digest"]
                    ):
                        continue
                    if (
                        search_state.get("completed") != search_state.get("planned")
                        or search_state.get("best_parameters") is None
                    ):
                        continue

                    directory = (
                        self.output_root
                        / "runs"
                        / dataset_name
                        / label
                        / model_name
                    )
                    runs = []
                    for seed_index in range(expected_runs):
                        path = directory / f"seed-{seed_index}.json"
                        seed = _stable_seed("final", seed_index)
                        fingerprint = _final_run_fingerprint(
                            search_state,
                            seed_index,
                            seed,
                            self._checkpoint_policy(seed_index),
                        )
                        if not self._run_complete(path, fingerprint):
                            continue
                        with path.open("r", encoding="utf-8") as handle:
                            runs.append(json.load(handle))
                    if not runs:
                        continue
                    first_parameters = runs[0]["parameters"]
                    if any(run["parameters"] != first_parameters for run in runs):
                        raise ValueError(
                            f"inconsistent final parameters in {directory}"
                        )
                    memory_keys = sorted(
                        {
                            key
                            for run in runs
                            for key in (
                                run.get("memory", {})
                                if isinstance(run.get("memory"), Mapping)
                                else {}
                            ).keys()
                        }
                    )
                    support_rows = []
                    for run in runs:
                        diagnostics = run.get("diagnostics")
                        support = (
                            diagnostics.get("target_support")
                            if isinstance(diagnostics, Mapping)
                            else None
                        )
                        if not isinstance(support, Mapping):
                            raise ValueError(
                                f"missing target-support diagnostics in {directory}"
                            )
                        evaluation_users = int(support["evaluation_users"])
                        if evaluation_users <= 0:
                            raise ValueError(
                                f"invalid target-support denominator in {directory}"
                            )
                        support_rows.append(
                            {
                                "positive_catalog_items": float(
                                    support["positive_catalog_items"]
                                ),
                                "validation_target_positive_support_rate": (
                                    float(
                                        support[
                                            "validation_targets_with_positive_support"
                                        ]
                                    )
                                    / evaluation_users
                                ),
                                "test_target_positive_support_rate": (
                                    float(
                                        support[
                                            "test_targets_with_positive_support"
                                        ]
                                    )
                                    / evaluation_users
                                ),
                            }
                        )
                    target_support_summary = aggregate_seed_results(
                        support_rows
                    )

                    memory_summary = {}
                    for key in memory_keys:
                        values = [
                            {"value": run["memory"][key]}
                            for run in runs
                            if isinstance(run.get("memory"), Mapping)
                            and key in run["memory"]
                        ]
                        memory_summary[key] = aggregate_seed_results(values)["value"]

                    summary[dataset_name][label][model_name] = {
                        "metrics": aggregate_seed_results(
                            [run["metrics"] for run in runs]
                        ),
                        "training_seconds": aggregate_seed_results(
                            [{"value": run["training_seconds"]} for run in runs]
                        )["value"],
                        "inference_seconds": aggregate_seed_results(
                            [{"value": run["inference_seconds"]} for run in runs]
                        )["value"],
                        "memory": memory_summary,
                        "target_support": target_support_summary,
                        "parameters": first_parameters,
                        "n_runs": len(runs),
                        "n_expected_runs": expected_runs,
                        "complete": len(runs) == expected_runs,
                    }
        expected_combinations = (
            len(datasets) * len(models) * len(self.config["proportions"])
        )
        complete_combinations = sum(
            1
            for dataset in summary.values()
            for proportion in dataset.values()
            for record in proportion.values()
            if record.get("complete") is True
        )
        if require_complete and complete_combinations != expected_combinations:
            raise RuntimeError(
                "cannot generate a final report from incomplete or incompatible "
                f"results: {complete_combinations}/{expected_combinations} "
                "dataset/proportion/model combinations are complete. Finish the "
                "search/final stages, or pass --allow-partial-report explicitly."
            )
        atomic_json_dump(summary, self.output_root / "summary.json")
        return summary
