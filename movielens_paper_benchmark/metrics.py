"""Ranking metrics for full-catalog and fixed sampled-candidate evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Set

import numpy as np
import torch


@dataclass(frozen=True)
class RankingResult:
    """Per-user and aggregate ranking metrics."""

    means: Dict[str, float]
    per_user: Dict[str, np.ndarray]
    user_ids: np.ndarray


def _as_numpy(scores: object) -> np.ndarray:
    if isinstance(scores, torch.Tensor):
        return scores.detach().cpu().numpy()
    return np.asarray(scores)


def evaluate_full_ranking(
    score_users: Callable[[np.ndarray], object],
    user_ids: Sequence[int],
    targets: Mapping[int, int],
    seen_items: Mapping[int, Iterable[int]],
    n_items: int,
    ks: Sequence[int] = (10,),
    batch_size: int = 256,
    extra_exclusions: Optional[Mapping[int, Iterable[int]]] = None,
) -> RankingResult:
    """Evaluate one held-out relevant item per user against the full item set.

    ``score_users`` must return a ``[batch, n_items]`` score matrix. Training
    items and optional additional items (for example validation targets during
    test evaluation) are masked. The target is always restored in case it is
    also present in an exclusion collection.
    """
    users = np.asarray(user_ids, dtype=np.int64)
    if users.ndim != 1:
        raise ValueError("user_ids must be one-dimensional")
    if n_items <= 0 or batch_size <= 0:
        raise ValueError("n_items and batch_size must be positive")
    clean_ks = tuple(sorted({int(k) for k in ks}))
    if not clean_ks or clean_ks[0] <= 0:
        raise ValueError("ks must contain positive integers")
    missing = [int(u) for u in users if int(u) not in targets]
    if missing:
        raise ValueError(f"missing targets for users: {missing[:5]}")

    values: Dict[str, list] = {}
    for k in clean_ks:
        values[f"hit@{k}"] = []
        values[f"ndcg@{k}"] = []

    for start in range(0, len(users), batch_size):
        batch_users = users[start : start + batch_size]
        scores = _as_numpy(score_users(batch_users)).astype(np.float64, copy=True)
        expected = (len(batch_users), n_items)
        if scores.shape != expected:
            raise ValueError(f"score_users returned {scores.shape}, expected {expected}")

        target_array = np.asarray([targets[int(u)] for u in batch_users], dtype=np.int64)
        if np.any(target_array < 0) or np.any(target_array >= n_items):
            raise ValueError("a target item is outside the item vocabulary")

        for row, user in enumerate(batch_users):
            excluded: Set[int] = set(int(i) for i in seen_items.get(int(user), ()))
            if extra_exclusions is not None:
                excluded.update(int(i) for i in extra_exclusions.get(int(user), ()))
            excluded.discard(int(target_array[row]))
            if excluded:
                valid = np.fromiter((i for i in excluded if 0 <= i < n_items), dtype=np.int64)
                scores[row, valid] = -np.inf

        # Stable sorting gives deterministic tie handling by mapped item id.
        ranking = np.argsort(-scores, axis=1, kind="stable")
        positions = np.empty(len(batch_users), dtype=np.int64)
        for row, target in enumerate(target_array):
            positions[row] = int(np.flatnonzero(ranking[row] == target)[0])

        for k in clean_ks:
            hit = positions < min(k, n_items)
            ndcg = np.where(hit, 1.0 / np.log2(positions + 2.0), 0.0)
            values[f"hit@{k}"].extend(hit.astype(np.float64).tolist())
            values[f"ndcg@{k}"].extend(ndcg.tolist())

    per_user = {name: np.asarray(metric_values, dtype=np.float64) for name, metric_values in values.items()}
    means = {name: float(metric_values.mean()) for name, metric_values in per_user.items()}
    return RankingResult(means=means, per_user=per_user, user_ids=users)


def evaluate_sampled_ranking(
    score_users: Callable[[np.ndarray], object],
    user_ids: Sequence[int],
    targets: Mapping[int, int],
    candidate_items: Mapping[int, Sequence[int]],
    n_items: int,
    ks: Sequence[int] = (10,),
    batch_size: int = 256,
) -> RankingResult:
    """Evaluate one target per user among fixed sampled candidates.

    Candidate collections must contain the target exactly once. Scores are
    ordered descending; stable sorting preserves candidate order for score ties.
    """
    users = np.asarray(user_ids, dtype=np.int64)
    if users.ndim != 1:
        raise ValueError("user_ids must be one-dimensional")
    if n_items <= 0 or batch_size <= 0:
        raise ValueError("n_items and batch_size must be positive")
    clean_ks = tuple(sorted({int(k) for k in ks}))
    if not clean_ks or clean_ks[0] <= 0:
        raise ValueError("ks must contain positive integers")
    missing = [int(u) for u in users if int(u) not in targets or int(u) not in candidate_items]
    if missing:
        raise ValueError(f"missing targets or candidates for users: {missing[:5]}")

    values: Dict[str, list] = {}
    for k in clean_ks:
        values[f"hit@{k}"] = []
        values[f"ndcg@{k}"] = []

    for start in range(0, len(users), batch_size):
        batch_users = users[start : start + batch_size]
        scores = _as_numpy(score_users(batch_users)).astype(np.float64, copy=False)
        expected = (len(batch_users), n_items)
        if scores.shape != expected:
            raise ValueError(f"score_users returned {scores.shape}, expected {expected}")

        for row, raw_user in enumerate(batch_users):
            user = int(raw_user)
            target = int(targets[user])
            candidates = np.asarray(candidate_items[user], dtype=np.int64)
            if candidates.ndim != 1 or candidates.size == 0:
                raise ValueError(f"candidates for user {user} must be non-empty and one-dimensional")
            if np.unique(candidates).size != candidates.size:
                raise ValueError(f"candidates for user {user} must be unique")
            if np.any(candidates < 0) or np.any(candidates >= n_items):
                raise ValueError(f"a candidate for user {user} is outside the item vocabulary")
            if int(np.count_nonzero(candidates == target)) != 1:
                raise ValueError(f"target for user {user} must occur exactly once in candidates")
            candidate_scores = scores[row, candidates]
            order = np.argsort(-candidate_scores, kind="stable")
            position = int(np.flatnonzero(candidates[order] == target)[0])
            for k in clean_ks:
                hit = position < min(k, candidates.size)
                values[f"hit@{k}"].append(float(hit))
                values[f"ndcg@{k}"].append(
                    1.0 / np.log2(position + 2.0) if hit else 0.0
                )

    per_user = {name: np.asarray(metric_values, dtype=np.float64) for name, metric_values in values.items()}
    means = {name: float(metric_values.mean()) for name, metric_values in per_user.items()}
    return RankingResult(means=means, per_user=per_user, user_ids=users)


def aggregate_seed_results(results: Sequence[Mapping[str, float]]) -> Dict[str, Dict[str, float]]:
    """Return mean and population standard deviation across random seeds."""
    if not results:
        raise ValueError("at least one seed result is required")
    names = tuple(results[0].keys())
    if any(tuple(result.keys()) != names for result in results):
        raise ValueError("all seed results must contain metrics in the same order")
    summary: Dict[str, Dict[str, float]] = {}
    for name in names:
        values = np.asarray([float(result[name]) for result in results], dtype=np.float64)
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "n": int(values.size),
        }
    return summary
