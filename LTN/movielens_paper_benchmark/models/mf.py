"""Matrix-factorization baselines with pointwise MSE and pairwise BPR."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .base import ArrayLike, BaseRecommender, interaction_arrays


class MFScorer(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int, biased: bool):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.biased = biased
        if biased:
            self.user_bias = nn.Embedding(n_users, 1)
            self.item_bias = nn.Embedding(n_items, 1)
        else:
            self.user_bias = None
            self.item_bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
        if self.biased:
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        scores = (self.user_embedding(users) * self.item_embedding(items)).sum(-1)
        if self.biased:
            scores = (
                scores
                + self.user_bias(users).squeeze(-1)
                + self.item_bias(items).squeeze(-1)
            )
        return scores

    def score_users(self, users: torch.Tensor) -> torch.Tensor:
        scores = self.user_embedding(users) @ self.item_embedding.weight.T
        if self.biased:
            scores = scores + self.user_bias(users) + self.item_bias.weight.T
        return scores


def _positive_sets(users: np.ndarray, items: np.ndarray, n_users: int) -> Sequence[set]:
    seen = [set() for _ in range(n_users)]
    for user, item in zip(users.tolist(), items.tolist()):
        seen[user].add(item)
    return seen


def _sampling_exclusions(
    positive_users: np.ndarray,
    positive_items: np.ndarray,
    n_users: int,
    n_items: int,
    forbidden_items_by_user: Optional[
        Union[Mapping[int, Iterable[int]], Sequence[Iterable[int]]]
    ],
) -> Sequence[set]:
    """Combine train positives with items forbidden only for negative sampling.

    Forbidden items are deliberately not returned as positive examples and never
    enter the optimization objective. Callers must derive them from training data
    only; validation and test labels must never be supplied here.
    """

    exclusions = _positive_sets(positive_users, positive_items, n_users)
    if forbidden_items_by_user is None:
        return exclusions

    if isinstance(forbidden_items_by_user, Mapping):
        entries = forbidden_items_by_user.items()
    else:
        if isinstance(forbidden_items_by_user, (str, bytes)):
            raise TypeError("forbidden_items_by_user must not be a string")
        try:
            length = len(forbidden_items_by_user)
        except TypeError as error:
            raise TypeError(
                "forbidden_items_by_user must be a mapping or a sized sequence"
            ) from error
        if length != n_users:
            raise ValueError(
                "a forbidden item sequence must contain exactly one entry per user"
            )
        entries = enumerate(forbidden_items_by_user)

    for raw_user, raw_items in entries:
        if isinstance(raw_user, (bool, np.bool_)) or not isinstance(
            raw_user, (int, np.integer)
        ):
            raise TypeError("forbidden user ids must be integers")
        user = int(raw_user)
        if user < 0 or user >= n_users:
            raise IndexError("forbidden user id out of range")
        if isinstance(raw_items, (str, bytes)):
            raise TypeError("forbidden item collections must not be strings")
        try:
            iterator = iter(raw_items)
        except TypeError as error:
            raise TypeError("forbidden items for each user must be iterable") from error
        for raw_item in iterator:
            if isinstance(raw_item, (bool, np.bool_)) or not isinstance(
                raw_item, (int, np.integer)
            ):
                raise TypeError("forbidden item ids must be integers")
            item = int(raw_item)
            if item < 0 or item >= n_items:
                raise IndexError("forbidden item id out of range")
            exclusions[user].add(item)
    return exclusions


def sample_unobserved_items(
    users: np.ndarray,
    seen_items: Sequence[set],
    n_items: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample one unseen item per user, omitting users with no candidate item."""

    kept_users = []
    negatives = []
    for user in np.asarray(users, dtype=np.int64):
        observed = seen_items[int(user)]
        if len(observed) >= n_items:
            continue
        candidate = int(rng.integers(n_items))
        while candidate in observed:
            candidate = int(rng.integers(n_items))
        kept_users.append(int(user))
        negatives.append(candidate)
    return np.asarray(kept_users, dtype=np.int64), np.asarray(negatives, dtype=np.int64)


class MatrixFactorization(BaseRecommender):
    """MF trained with either pointwise MSE or pairwise BPR.

    Parameters are deliberately explicit so the same class can represent the
    paper's pointwise MF baseline and the stronger BPR-MF baseline.
    """

    VALID_OBJECTIVES = {"mse", "bpr"}

    def __init__(
        self,
        n_users: int,
        n_items: int,
        *,
        embedding_dim: int = 64,
        objective: str = "mse",
        biased: bool = True,
        device: Union[str, torch.device] = "cpu",
        seed: int = 0,
    ) -> None:
        if objective not in self.VALID_OBJECTIVES:
            raise ValueError(f"objective must be one of {sorted(self.VALID_OBJECTIVES)}")
        if n_users <= 0 or n_items <= 0 or embedding_dim <= 0:
            raise ValueError("n_users, n_items, and embedding_dim must be positive")
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.embedding_dim = int(embedding_dim)
        self.objective = objective
        self.biased = bool(biased)
        self.device = torch.device(device)
        self.seed = int(seed)
        self._set_seed(self.seed)
        self.model = MFScorer(
            self.n_users, self.n_items, self.embedding_dim, self.biased
        ).to(self.device)
        self.history_: Dict[str, list] = {"loss": [], "validation_score": []}
        self.best_epoch_: Optional[int] = None
        self.best_score_: Optional[float] = None

    @staticmethod
    def _set_seed(seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # This requests deterministic kernels without failing on unsupported ops.
        torch.use_deterministic_algorithms(True)

    def fit(
        self,
        interactions: Any,
        *,
        epochs: int = 100,
        batch_size: int = 1024,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        num_negatives: int = 0,
        forbidden_items_by_user: Optional[
            Union[Mapping[int, Iterable[int]], Sequence[Iterable[int]]]
        ] = None,
        seed: Optional[int] = None,
        shuffle: bool = True,
        verbose: bool = False,
        validation_callback: Optional[Callable[["MatrixFactorization"], float]] = None,
        patience: Optional[int] = None,
        min_delta: float = 0.0,
    ) -> "MatrixFactorization":
        users, items, targets, _, _ = interaction_arrays(
            interactions, n_users=self.n_users, n_items=self.n_items
        )
        if not len(users):
            raise ValueError("cannot fit MF on an empty interaction set")
        if epochs <= 0 or batch_size <= 0 or num_negatives < 0:
            raise ValueError("epochs/batch_size must be positive and num_negatives non-negative")
        if patience is not None and patience <= 0:
            raise ValueError("patience must be positive")
        if patience is not None and validation_callback is None:
            raise ValueError("patience requires validation_callback")

        run_seed = self.seed if seed is None else int(seed)
        self._set_seed(run_seed)
        rng = np.random.default_rng(run_seed)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        positive_mask = targets > 0
        positive_users, positive_items = users[positive_mask], items[positive_mask]
        seen = _sampling_exclusions(
            positive_users,
            positive_items,
            self.n_users,
            self.n_items,
            forbidden_items_by_user,
        )
        self.history_ = {"loss": [], "validation_score": []}
        self.best_epoch_ = None
        self.best_score_ = None
        best_state = None
        stale_epochs = 0
        self.model.train()

        for epoch in range(epochs):
            if self.objective == "bpr":
                epoch_loss = self._fit_bpr_epoch(
                    positive_users,
                    positive_items,
                    seen,
                    rng,
                    optimizer,
                    batch_size,
                    shuffle,
                )
            else:
                epoch_users, epoch_items, epoch_targets = users, items, targets
                if num_negatives:
                    neg_users_parts, neg_items_parts = [], []
                    for _ in range(num_negatives):
                        neg_users, neg_items = sample_unobserved_items(
                            positive_users, seen, self.n_items, rng
                        )
                        neg_users_parts.append(neg_users)
                        neg_items_parts.append(neg_items)
                    if neg_users_parts:
                        neg_users = np.concatenate(neg_users_parts)
                        neg_items = np.concatenate(neg_items_parts)
                        epoch_users = np.concatenate((users, neg_users))
                        epoch_items = np.concatenate((items, neg_items))
                        epoch_targets = np.concatenate(
                            (targets, np.zeros(len(neg_users), dtype=np.float32))
                        )
                epoch_loss = self._fit_mse_epoch(
                    epoch_users,
                    epoch_items,
                    epoch_targets,
                    rng,
                    optimizer,
                    batch_size,
                    shuffle,
                )
            self.history_["loss"].append(epoch_loss)
            if validation_callback is not None:
                score = float(validation_callback(self))
                if not np.isfinite(score):
                    raise ValueError("validation_callback returned a non-finite score")
                self.history_["validation_score"].append(score)
                if self.best_score_ is None or score > self.best_score_ + min_delta:
                    self.best_score_ = score
                    self.best_epoch_ = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }
                    stale_epochs = 0
                else:
                    stale_epochs += 1
            if verbose:
                message = f"epoch={epoch + 1} loss={epoch_loss:.6f}"
                if validation_callback is not None:
                    message += f" validation={score:.6f}"
                print(message)
            if patience is not None and stale_epochs >= patience:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        return self

    def _fit_mse_epoch(
        self,
        users: np.ndarray,
        items: np.ndarray,
        targets: np.ndarray,
        rng: np.random.Generator,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        shuffle: bool,
    ) -> float:
        order = rng.permutation(len(users)) if shuffle else np.arange(len(users))
        loss_sum = 0.0
        count = 0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            user_tensor = torch.as_tensor(users[indices], device=self.device)
            item_tensor = torch.as_tensor(items[indices], device=self.device)
            target_tensor = torch.as_tensor(
                targets[indices], dtype=torch.float32, device=self.device
            )
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(self.model(user_tensor, item_tensor), target_tensor)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            count += len(indices)
        return loss_sum / max(count, 1)

    def _fit_bpr_epoch(
        self,
        users: np.ndarray,
        positive_items: np.ndarray,
        seen: Sequence[set],
        rng: np.random.Generator,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        shuffle: bool,
    ) -> float:
        eligible = np.asarray(
            [i for i, user in enumerate(users) if len(seen[int(user)]) < self.n_items],
            dtype=np.int64,
        )
        if not len(eligible):
            raise ValueError("BPR requires at least one unobserved item for a training user")
        if shuffle:
            eligible = rng.permutation(eligible)
        epoch_users = users[eligible]
        epoch_positive = positive_items[eligible]
        sampled_users, negatives = sample_unobserved_items(
            epoch_users, seen, self.n_items, rng
        )
        # ``eligible`` guarantees no user is dropped by the sampler.
        assert len(sampled_users) == len(epoch_users)
        loss_sum = 0.0
        count = 0
        for start in range(0, len(epoch_users), batch_size):
            end = start + batch_size
            user_tensor = torch.as_tensor(epoch_users[start:end], device=self.device)
            pos_tensor = torch.as_tensor(epoch_positive[start:end], device=self.device)
            neg_tensor = torch.as_tensor(negatives[start:end], device=self.device)
            optimizer.zero_grad(set_to_none=True)
            positive_scores = self.model(user_tensor, pos_tensor)
            negative_scores = self.model(user_tensor, neg_tensor)
            loss = F.softplus(-(positive_scores - negative_scores)).mean()
            loss.backward()
            optimizer.step()
            size = len(user_tensor)
            loss_sum += float(loss.detach()) * size
            count += size
        return loss_sum / count

    @torch.inference_mode()
    def score_users(
        self,
        user_ids: ArrayLike,
        item_ids: Optional[ArrayLike] = None,
        *,
        batch_size: int = 1024,
    ) -> np.ndarray:
        users = np.asarray(user_ids, dtype=np.int64).reshape(-1)
        if users.size and (users.min() < 0 or users.max() >= self.n_users):
            raise IndexError("user id out of range")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        items = None if item_ids is None else np.asarray(item_ids, dtype=np.int64)
        if items is not None and items.ndim not in (1, 2):
            raise ValueError("item_ids must be one- or two-dimensional")
        if items is not None and items.size and (items.min() < 0 or items.max() >= self.n_items):
            raise IndexError("item id out of range")
        if items is not None and items.ndim == 2 and items.shape[0] != len(users):
            raise ValueError("2D item_ids must have one row per user")

        self.model.eval()
        outputs = []
        for start in range(0, len(users), batch_size):
            user_batch = torch.as_tensor(users[start : start + batch_size], device=self.device)
            user_vectors = self.model.user_embedding(user_batch)
            if items is None or items.ndim == 1:
                selected = np.arange(self.n_items) if items is None else items
                item_tensor = torch.as_tensor(selected, device=self.device)
                scores = user_vectors @ self.model.item_embedding(item_tensor).T
                if self.biased:
                    scores = (
                        scores
                        + self.model.user_bias(user_batch)
                        + self.model.item_bias(item_tensor).T
                    )
            else:
                item_tensor = torch.as_tensor(items[start : start + batch_size], device=self.device)
                item_vectors = self.model.item_embedding(item_tensor)
                scores = (user_vectors[:, None, :] * item_vectors).sum(-1)
                if self.biased:
                    scores = (
                        scores
                        + self.model.user_bias(user_batch)
                        + self.model.item_bias(item_tensor).squeeze(-1)
                    )
            outputs.append(scores.detach().cpu().numpy().astype(np.float32, copy=False))
        width = self.n_items if items is None else (items.shape[-1] if items.ndim == 2 else len(items))
        return np.concatenate(outputs, axis=0) if outputs else np.empty((0, width), np.float32)

    def predict(self, user_ids: ArrayLike, item_ids: ArrayLike) -> np.ndarray:
        users = np.asarray(user_ids, dtype=np.int64).reshape(-1)
        items = np.asarray(item_ids, dtype=np.int64).reshape(-1)
        if len(users) != len(items):
            raise ValueError("user_ids and item_ids must have the same length")
        paired_items = items[:, None]
        return self.score_users(users, paired_items).reshape(-1)

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "format_version": 1,
            "class": self.__class__.__name__,
            "config": {
                "n_users": self.n_users,
                "n_items": self.n_items,
                "embedding_dim": self.embedding_dim,
                "objective": self.objective,
                "biased": self.biased,
                "seed": self.seed,
            },
            "model_state": {
                key: value.detach().cpu() for key, value in self.model.state_dict().items()
            },
            "history": self.history_,
            "best_epoch": self.best_epoch_,
            "best_score": self.best_score_,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        config = dict(state["config"])
        expected = {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "embedding_dim": self.embedding_dim,
            "objective": self.objective,
            "biased": self.biased,
            "seed": self.seed,
        }
        if config != expected:
            raise ValueError(f"checkpoint config {config} does not match model config {expected}")
        self.model.load_state_dict(state["model_state"])
        self.model.to(self.device)
        self.history_ = dict(state.get("history", {"loss": [], "validation_score": []}))
        self.best_epoch_ = state.get("best_epoch")
        self.best_score_ = state.get("best_score")

    @classmethod
    def load(
        cls, path: Union[str, Path], *, device: Union[str, torch.device] = "cpu"
    ) -> "MatrixFactorization":
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        config = dict(state["config"])
        model = cls(**config, device=device)
        model.load_state_dict(state)
        return model


MF = MatrixFactorization

