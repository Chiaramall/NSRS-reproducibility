"""Deterministic hyperparameter enumeration and stratified random search."""
from __future__ import annotations

import itertools
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


Config = Dict[str, Any]


def cartesian_configurations(space: Mapping[str, Sequence[Any]]) -> List[Config]:
    """Expand a JSON-style search space into a deterministic Cartesian product."""
    if not space:
        return [{}]
    keys = tuple(space.keys())
    values = []
    for key in keys:
        candidates = list(space[key])
        if not candidates:
            raise ValueError(f"search dimension {key!r} has no candidates")
        values.append(candidates)
    return [
        dict(zip(keys, combination))
        for combination in itertools.product(*values)
    ]


def sample_configurations(
    space: Mapping[str, Sequence[Any]],
    budget: Optional[int],
    seed: int,
    coverage_keys: Sequence[str] = ("embedding_dim",),
) -> List[Config]:
    """Sample without replacement, stratified by the first coverage dimension.

    When an embedding dimension is present, configurations are selected in
    round-robin strata. Therefore a budget of 36 gives exactly 6 configurations
    for each of 6 MF/LTN dimensions and 12 for each of 3 LightGCN dimensions.
    This is substantially stronger than merely forcing one occurrence of every
    dimension, while all remaining hyperparameters remain randomized.
    """
    all_configs = cartesian_configurations(space)
    if budget is None or budget >= len(all_configs):
        return all_configs
    if budget <= 0:
        raise ValueError("search budget must be positive")

    rng = random.Random(int(seed))
    stratification_key = next(
        (key for key in coverage_keys if key in space), None
    )
    if stratification_key is None:
        shuffled = list(all_configs)
        rng.shuffle(shuffled)
        return [dict(config) for config in shuffled[:budget]]

    strata: Dict[str, List[Config]] = {}
    ordered_values = list(space[stratification_key])
    for value in ordered_values:
        key = json.dumps(value, sort_keys=True, separators=(",", ":"))
        group = [
            dict(config)
            for config in all_configs
            if config[stratification_key] == value
        ]
        rng.shuffle(group)
        strata[key] = group

    # Randomize which stratum receives a possible remainder while preserving
    # deterministic selection for a given search seed.
    value_order = list(ordered_values)
    rng.shuffle(value_order)
    selected: List[Config] = []
    offsets = {json.dumps(value, sort_keys=True, separators=(",", ":")): 0 for value in value_order}
    while len(selected) < budget:
        made_progress = False
        for value in value_order:
            key = json.dumps(value, sort_keys=True, separators=(",", ":"))
            offset = offsets[key]
            if offset >= len(strata[key]):
                continue
            selected.append(dict(strata[key][offset]))
            offsets[key] = offset + 1
            made_progress = True
            if len(selected) == budget:
                break
        if not made_progress:
            break
    if len(selected) != budget:
        raise RuntimeError("could not satisfy the requested search budget")
    return selected


def select_best(
    configurations: Iterable[Config],
    objective: Callable[[Config], float],
) -> Tuple[Config, float, List[Dict[str, Any]]]:
    """Evaluate configurations and return the maximum objective and audit trail."""
    best_config: Optional[Config] = None
    best_score = float("-inf")
    history: List[Dict[str, Any]] = []
    for index, config in enumerate(configurations):
        score = float(objective(config))
        history.append({"index": index, "config": dict(config), "score": score})
        if score > best_score:
            best_score = score
            best_config = dict(config)
    if best_config is None:
        raise ValueError("no configurations were evaluated")
    return best_config, best_score, history


def atomic_json_dump(payload: Any, path: Path) -> None:
    """Write JSON through a sibling temporary file to avoid partial reports."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
