"""Leakage-safe temporal splitting and deterministic training sparsification."""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

from .data import Feedback, Interaction, MovieLensData


def _interaction_key(interaction: Interaction) -> Tuple[int, int, int, float]:
    return (
        interaction.user_idx,
        interaction.timestamp,
        interaction.item_idx,
        interaction.rating,
    )


@dataclass(frozen=True)
class DataSplit:
    """Common split consumed by every model in the benchmark.

    ``excluded_future_count`` counts non-test events at or after the validation
    boundary.  They are intentionally discarded rather than leaked into training.
    """

    train: Tuple[Interaction, ...]
    validation: Tuple[Interaction, ...]
    test: Tuple[Interaction, ...]
    eligible_users: Tuple[int, ...]
    dropped_users: Tuple[int, ...]
    excluded_future_count: int = 0
    train_proportion: float = 1.0
    sparsity_seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not 0.0 < self.train_proportion <= 1.0:
            raise ValueError("train_proportion must be in (0, 1]")
        if len(self.validation) != len(self.eligible_users):
            raise ValueError("validation must contain exactly one item per eligible user")
        if len(self.test) != len(self.eligible_users):
            raise ValueError("test must contain exactly one item per eligible user")
        if tuple(sorted(set(self.eligible_users))) != self.eligible_users:
            raise ValueError("eligible_users must be sorted and unique")
        if set(self.eligible_users) & set(self.dropped_users):
            raise ValueError("eligible_users and dropped_users must be disjoint")

    @property
    def n_users(self) -> int:
        return len(self.eligible_users)


@dataclass(frozen=True, order=True)
class UserGenrePreference:
    """A training-only, thresholded aggregate preference for one user/genre."""

    user_idx: int
    genre_idx: int
    mean_rating: float
    support: int
    feedback: Feedback

    def __post_init__(self) -> None:
        if self.feedback is Feedback.NEUTRAL:
            raise ValueError("only liked or disliked genre preferences are materialized")
        if self.support < 1:
            raise ValueError("support must be positive")


def temporal_leave_two_out(
    data: MovieLensData,
    *,
    min_train_positives: int = 1,
    require_warm_start_targets: bool = False,
) -> DataSplit:
    """Create a strict temporal leave-two-positive-out split per eligible user.

    The penultimate positive event is validation and the last positive event is
    test.  Only events strictly before the validation event in deterministic
    temporal order enter training.  Thus explicit negatives or neutral ratings
    recorded between the two holdouts cannot leak future information. When
    ``require_warm_start_targets`` is true, users are iteratively removed until
    every validation and test item occurs as a positive interaction in the
    remaining users' global training set.
    """

    if min_train_positives < 1:
        raise ValueError("min_train_positives must be at least 1")
    histories: DefaultDict[int, List[Interaction]] = defaultdict(list)
    for interaction in data.interactions:
        histories[interaction.user_idx].append(interaction)

    train: List[Interaction] = []
    validation: List[Interaction] = []
    test: List[Interaction] = []
    eligible: List[int] = []
    dropped: List[int] = []
    excluded_future_by_user: Dict[int, int] = {}

    for user_idx in range(data.n_users):
        history = sorted(histories.get(user_idx, ()), key=_interaction_key)
        positive_positions = [
            idx for idx, event in enumerate(history) if event.feedback is Feedback.POSITIVE
        ]
        if len(positive_positions) < min_train_positives + 2:
            dropped.append(user_idx)
            continue
        validation_position = positive_positions[-2]
        test_position = positive_positions[-1]
        user_train = history[:validation_position]
        if sum(x.feedback is Feedback.POSITIVE for x in user_train) < min_train_positives:
            dropped.append(user_idx)
            continue
        eligible.append(user_idx)
        train.extend(user_train)
        validation.append(history[validation_position])
        test.append(history[test_position])
        excluded_future_by_user[user_idx] = test_position - validation_position - 1

    if require_warm_start_targets:
        active_users = set(eligible)
        validation_targets = {event.user_idx: event.item_idx for event in validation}
        test_targets = {event.user_idx: event.item_idx for event in test}
        while active_users:
            training_items = {
                event.item_idx
                for event in train
                if event.user_idx in active_users
                and event.feedback is Feedback.POSITIVE
            }
            cold_users = {
                user_idx
                for user_idx in active_users
                if validation_targets[user_idx] not in training_items
                or test_targets[user_idx] not in training_items
            }
            if not cold_users:
                break
            active_users.difference_update(cold_users)
            dropped.extend(cold_users)
        eligible = sorted(active_users)
        train = [event for event in train if event.user_idx in active_users]
        validation = [
            event for event in validation if event.user_idx in active_users
        ]
        test = [event for event in test if event.user_idx in active_users]

    excluded_future_count = sum(
        excluded_future_by_user[user_idx] for user_idx in eligible
    )

    return DataSplit(
        train=tuple(sorted(train, key=_interaction_key)),
        validation=tuple(sorted(validation, key=_interaction_key)),
        test=tuple(sorted(test, key=_interaction_key)),
        eligible_users=tuple(eligible),
        dropped_users=tuple(sorted(set(dropped))),
        excluded_future_count=excluded_future_count,
    )


def _user_seed(seed: int, user_idx: int) -> int:
    digest = hashlib.sha256(f"{seed}:{user_idx}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def sparsify_training(
    split: DataSplit,
    proportion: float,
    *,
    seed: int,
) -> DataSplit:
    """Sample each user's training history, always retaining a positive event.

    Sampling is independent per user and stable across process counts and input
    ordering.  The target size is ``ceil(n_user_events * proportion)``.
    Validation and test are returned unchanged.
    """

    if not 0.0 < proportion <= 1.0:
        raise ValueError("proportion must be in (0, 1]")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if proportion == 1.0:
        return replace(split, train_proportion=1.0, sparsity_seed=seed)

    by_user: DefaultDict[int, List[Interaction]] = defaultdict(list)
    for interaction in split.train:
        by_user[interaction.user_idx].append(interaction)
    sampled: List[Interaction] = []
    for user_idx in split.eligible_users:
        history = sorted(by_user[user_idx], key=_interaction_key)
        positives = [event for event in history if event.feedback is Feedback.POSITIVE]
        if not positives:
            raise ValueError(f"eligible user {user_idx} has no positive training event")
        target_size = max(1, min(len(history), math.ceil(len(history) * proportion)))
        rng = random.Random(_user_seed(seed, user_idx))
        anchor = rng.choice(positives)
        remaining = [event for event in history if event is not anchor]
        # Every proportion uses a prefix of one deterministic per-user order.
        rng.shuffle(remaining)
        chosen = [anchor, *remaining[: target_size - 1]]
        sampled.extend(chosen)

    return replace(
        split,
        train=tuple(sorted(sampled, key=_interaction_key)),
        train_proportion=proportion,
        sparsity_seed=seed,
    )


def derive_user_genre_preferences(
    data: MovieLensData,
    training: Sequence[Interaction],
    *,
    min_support: int = 3,
    liked_threshold: float = 4.0,
    disliked_threshold: float = 2.0,
) -> Tuple[UserGenrePreference, ...]:
    """Derive genre preferences using *only* the supplied training interactions.

    Each non-neutral rating contributes to every genre assigned to its movie.
    Aggregates in
    the open interval ``(disliked_threshold, liked_threshold)`` are left unknown
    and therefore do not generate a logical fact.
    """

    if min_support < 1:
        raise ValueError("min_support must be at least 1")
    if disliked_threshold >= liked_threshold:
        raise ValueError("disliked_threshold must be lower than liked_threshold")

    aggregates: DefaultDict[Tuple[int, int], List[float]] = defaultdict(list)
    for interaction in training:
        if interaction.feedback is Feedback.NEUTRAL:
            continue
        if interaction.user_idx >= data.n_users or interaction.item_idx >= data.n_items:
            raise ValueError("training interaction index is outside the supplied dataset")
        for genre_idx in data.item_genres[interaction.item_idx]:
            aggregates[(interaction.user_idx, genre_idx)].append(interaction.rating)

    preferences: List[UserGenrePreference] = []
    for (user_idx, genre_idx), ratings in sorted(aggregates.items()):
        if len(ratings) < min_support:
            continue
        mean_rating = sum(ratings) / len(ratings)
        if mean_rating >= liked_threshold:
            feedback = Feedback.POSITIVE
        elif mean_rating <= disliked_threshold:
            feedback = Feedback.NEGATIVE
        else:
            continue
        preferences.append(
            UserGenrePreference(
                user_idx=user_idx,
                genre_idx=genre_idx,
                mean_rating=mean_rating,
                support=len(ratings),
                feedback=feedback,
            )
        )
    return tuple(preferences)
