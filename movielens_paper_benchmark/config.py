"""Configuration loading and strict validation for MovieLens benchmarks."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Set


SUPPORTED_DATASETS = {"ml-100k", "ml-1m"}
SUPPORTED_MODELS = {
    "mf-mse",
    "bpr-mf",
    "ease",
    "lightgcn",
    "ltn",
    "ltn-genres",
}
REQUIRED_SPACE_KEYS = {
    "mf-mse": {"embedding_dim", "learning_rate", "l2", "biased", "batch_size"},
    "bpr-mf": {"embedding_dim", "learning_rate", "l2", "biased", "batch_size"},
    "ease": {"regularization"},
    "lightgcn": {
        "embedding_dim",
        "num_layers",
        "learning_rate",
        "l2",
        "batch_size",
    },
    "ltn": {
        "embedding_dim",
        "learning_rate",
        "l2",
        "biased",
        "batch_size",
        "alpha",
    },
    "ltn-genres": {
        "embedding_dim",
        "learning_rate",
        "l2",
        "biased",
        "batch_size",
        "alpha",
        "p",
        "genre_rule_weight",
        "min_genre_support",
    },
}

_INTEGER_SPACE_KEYS = {
    "embedding_dim",
    "batch_size",
    "num_layers",
    "p",
    "min_genre_support",
}
_POSITIVE_NUMBER_SPACE_KEYS = {
    "learning_rate",
    "regularization",
    "alpha",
}
_NONNEGATIVE_NUMBER_SPACE_KEYS = {"l2", "genre_rule_weight"}


def load_config(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _require_list(value: Any, label: str) -> list:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a JSON list")
    return value


def _require_int(value: Any, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < minimum:
        comparison = "positive" if minimum == 1 else f">= {minimum}"
        raise ValueError(f"{label} must be {comparison}")
    return value


def _require_number(
    value: Any, label: str, *, minimum: float, strict: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    invalid = result <= minimum if strict else result < minimum
    if invalid:
        operator = ">" if strict else ">="
        raise ValueError(f"{label} must be {operator} {minimum}")
    return result


def _unique_supported(
    values: Sequence[Any], supported: Set[str], label: str
) -> None:
    _require_list(values, label)
    if not values:
        raise ValueError(f"{label} cannot be empty")
    if any(not isinstance(value, str) for value in values):
        raise TypeError(f"{label} must contain strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")
    unknown = set(values).difference(supported)
    if unknown:
        raise ValueError(f"unsupported {label}: {sorted(unknown)}")


def _validate_space(model: str, value: Any) -> None:
    space = _require_mapping(value, f"spaces.{model}")
    expected = REQUIRED_SPACE_KEYS[model]
    actual = set(space)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        raise ValueError(
            f"search space for {model} has missing keys {missing} "
            f"and unsupported keys {extra}"
        )

    for key, raw_values in space.items():
        values = _require_list(raw_values, f"{model}.{key}")
        if not values:
            raise ValueError(f"{model}.{key} must be non-empty")
        canonical = [
            json.dumps(candidate, sort_keys=True, separators=(",", ":"))
            for candidate in values
        ]
        if len(set(canonical)) != len(canonical):
            raise ValueError(f"{model}.{key} candidates must be unique")

        for index, candidate in enumerate(values):
            label = f"{model}.{key}[{index}]"
            if key == "biased":
                if not isinstance(candidate, bool):
                    raise TypeError(f"{label} must be boolean")
            elif key in _INTEGER_SPACE_KEYS:
                _require_int(candidate, label, 1)
            elif key in _POSITIVE_NUMBER_SPACE_KEYS:
                _require_number(candidate, label, minimum=0.0, strict=True)
            elif key in _NONNEGATIVE_NUMBER_SPACE_KEYS:
                _require_number(candidate, label, minimum=0.0)
            else:  # Defensive: exact key validation above should make this unreachable.
                raise ValueError(f"unsupported hyperparameter {model}.{key}")

        normalized = (
            values
            if key == "biased" or key in _INTEGER_SPACE_KEYS
            else [float(candidate) for candidate in values]
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                f"{model}.{key} candidates must be numerically unique"
            )


def _require_exact_keys(
    value: Mapping[str, Any], expected: Set[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        raise ValueError(
            f"{label} has missing keys {missing} and unsupported keys {extra}"
        )


def validate_config(config: Mapping[str, Any]) -> None:
    _require_mapping(config, "configuration")
    required = {
        "data_root",
        "output_root",
        "datasets",
        "models",
        "proportions",
        "positive_threshold",
        "negative_threshold",
        "final_seeds",
        "search",
        "training",
        "ranking",
        "spaces",
    }
    optional = {
        "min_train_positives",
        "require_warm_start_targets",
        "checkpoint_policy",
    }
    missing = required.difference(config)
    extra = set(config).difference(required | optional)
    if missing or extra:
        raise ValueError(
            f"configuration has missing fields {sorted(missing)} "
            f"and unsupported fields {sorted(extra)}"
        )

    for key in ("data_root", "output_root"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise TypeError(f"{key} must be a non-empty string")

    _unique_supported(config["datasets"], SUPPORTED_DATASETS, "datasets")
    _unique_supported(config["models"], SUPPORTED_MODELS, "models")

    raw_proportions = _require_list(config["proportions"], "proportions")
    if not raw_proportions:
        raise ValueError("proportions cannot be empty")
    proportions = [
        _require_number(value, f"proportions[{index}]", minimum=0.0, strict=True)
        for index, value in enumerate(raw_proportions)
    ]
    if any(value > 1.0 for value in proportions):
        raise ValueError("proportions must be in (0, 1]")
    if len(set(proportions)) != len(proportions):
        raise ValueError("proportions must be unique")

    positive = _require_number(
        config["positive_threshold"], "positive_threshold", minimum=1.0
    )
    negative = _require_number(
        config["negative_threshold"], "negative_threshold", minimum=1.0
    )
    if positive > 5.0 or negative > 5.0:
        raise ValueError("MovieLens thresholds must be in [1, 5]")
    if negative >= positive:
        raise ValueError("negative_threshold must be below positive_threshold")

    if not isinstance(config.get("require_warm_start_targets", False), bool):
        raise TypeError("require_warm_start_targets must be boolean")
    _require_int(config.get("min_train_positives", 1), "min_train_positives", 1)
    _require_int(config["final_seeds"], "final_seeds", 1)
    checkpoint_policy = config.get("checkpoint_policy", "none")
    if checkpoint_policy not in {"none", "first", "all"}:
        raise ValueError("checkpoint_policy must be none, first, or all")

    search = _require_mapping(config["search"], "search")
    _require_exact_keys(
        search,
        {"strategy", "budget", "validation_seeds", "seed", "objective"},
        "search",
    )
    if search["strategy"] != "random":
        raise ValueError("only deterministic random search is supported")
    _require_int(search["budget"], "search.budget", 1)
    _require_int(search["validation_seeds"], "search.validation_seeds", 1)
    _require_int(search["seed"], "search.seed", 0)

    training = _require_mapping(config["training"], "training")
    _require_exact_keys(
        training,
        {
            "max_epochs",
            "patience",
            "negative_samples",
            "genre_samples_per_user",
        },
        "training",
    )
    _require_int(training["max_epochs"], "training.max_epochs", 1)
    _require_int(training["patience"], "training.patience", 1)
    _require_int(training["negative_samples"], "training.negative_samples", 0)
    _require_int(
        training["genre_samples_per_user"],
        "training.genre_samples_per_user",
        1,
    )

    ranking = _require_mapping(config["ranking"], "ranking")
    _require_exact_keys(
        ranking,
        {"protocol", "negative_candidates", "candidate_seed", "ks", "user_batch_size"},
        "ranking",
    )
    if ranking["protocol"] != "sampled-100":
        raise ValueError("ranking.protocol must be sampled-100")
    negatives = _require_int(
        ranking["negative_candidates"], "ranking.negative_candidates", 1
    )
    if negatives != 100:
        raise ValueError("ranking.negative_candidates must equal 100")
    _require_int(ranking["candidate_seed"], "ranking.candidate_seed", 0)
    raw_ks = _require_list(ranking["ks"], "ranking.ks")
    if not raw_ks:
        raise ValueError("ranking.ks cannot be empty")
    ks = [
        _require_int(cutoff, f"ranking.ks[{index}]", 1)
        for index, cutoff in enumerate(raw_ks)
    ]
    if len(set(ks)) != len(ks):
        raise ValueError("ranking.ks must be unique")
    if any(cutoff > negatives + 1 for cutoff in ks):
        raise ValueError("ranking.ks cannot exceed the sampled candidate-set size")
    _require_int(ranking["user_batch_size"], "ranking.user_batch_size", 1)
    valid_metrics = {
        f"{metric}@{cutoff}"
        for cutoff in ks
        for metric in ("hit", "ndcg")
    }
    if search["objective"] not in valid_metrics:
        raise ValueError(
            f"search.objective must be one of {sorted(valid_metrics)}"
        )

    spaces = _require_mapping(config["spaces"], "spaces")
    missing_spaces = set(config["models"]).difference(spaces)
    if missing_spaces:
        raise ValueError(f"missing search spaces: {sorted(missing_spaces)}")
    for model in config["models"]:
        _validate_space(model, spaces[model])
