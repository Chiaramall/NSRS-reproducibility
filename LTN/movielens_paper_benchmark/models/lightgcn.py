"""A dependency-free LightGCN recommender implemented with PyTorch.

The implementation deliberately keeps data handling small and explicit.  It
expects positive user-item pairs, constructs the symmetrically-normalized
bipartite adjacency matrix, and optimizes the standard BPR objective.  It does
not depend on PyTorch Geometric or on the original MindReader pipeline.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .base import BaseRecommender


PairInput = Union[Tensor, np.ndarray, Sequence[Sequence[int]], Any]
ValidationCallback = Callable[["LightGCN"], float]


def _as_pair_tensor(interactions: PairInput) -> Tensor:
    """Convert dense, torch-sparse, or scipy-like interactions to ``[N, 2]``.

    A scipy sparse matrix is interpreted as a user-item matrix.  A torch sparse
    tensor is interpreted in the same way.  Dense inputs must contain user-item
    index pairs.
    """

    if isinstance(interactions, Tensor):
        if interactions.layout != torch.strided:
            sparse = interactions.coalesce()
            indices = sparse.indices()[:, sparse.values() > 0].t()
            return indices.detach().to(device="cpu", dtype=torch.long)
        dense = interactions.detach().to(device="cpu")
        if dense.ndim == 2 and dense.shape[1] == 3:
            dense = dense[dense[:, 2] > 0, :2]
        pairs = dense.to(dtype=torch.long)
    elif hasattr(interactions, "tocoo"):
        coo = interactions.tocoo()
        positive = np.asarray(coo.data) > 0
        pairs = torch.as_tensor(
            np.column_stack((coo.row[positive], coo.col[positive])), dtype=torch.long
        )
    elif isinstance(interactions, (tuple, list)) and len(interactions) in (2, 3):
        users = np.asarray(interactions[0]).reshape(-1)
        items = np.asarray(interactions[1]).reshape(-1)
        if len(users) != len(items):
            raise ValueError("interaction user and item arrays must have equal length")
        positive = (
            np.ones(len(users), dtype=bool)
            if len(interactions) == 2
            else np.asarray(interactions[2]).reshape(-1) > 0
        )
        if len(positive) != len(users):
            raise ValueError("interaction targets must match users and items")
        pairs = torch.as_tensor(
            np.column_stack((users[positive], items[positive])), dtype=torch.long
        )
    else:
        dense = np.asarray(interactions)
        if dense.ndim == 2 and dense.shape[1] == 3:
            dense = dense[dense[:, 2] > 0, :2]
        pairs = torch.as_tensor(dense, dtype=torch.long)

    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("interactions must have shape [n_interactions, 2]")
    return pairs


class LightGCN(nn.Module, BaseRecommender):
    """LightGCN trained with Bayesian Personalized Ranking (BPR).

    Parameters are intentionally compatible with the terminology used by the
    rest of this repository.  ``embedding_dim`` is typically selected from
    ``(64, 128, 256)``, but any positive dimension is supported for smoke tests
    and ablations.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        learning_rate: float = 1e-3,
        l2_reg: float = 1e-4,
        batch_size: int = 1024,
        epochs: int = 100,
        seed: int = 0,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        if n_users <= 0 or n_items <= 1:
            raise ValueError("n_users must be positive and n_items must exceed one")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if num_layers < 0:
            raise ValueError("num_layers cannot be negative")
        if learning_rate <= 0 or l2_reg < 0:
            raise ValueError("learning_rate must be positive and l2_reg non-negative")
        if batch_size <= 0 or epochs < 0:
            raise ValueError("batch_size must be positive and epochs non-negative")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.embedding_dim = int(embedding_dim)
        self.num_layers = int(num_layers)
        self.learning_rate = float(learning_rate)
        self.l2_reg = float(l2_reg)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.requested_device = str(torch.device(device))

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        self._reset_parameters()

        # The graph can be reconstructed from ``_train_interactions`` and is
        # therefore excluded from state_dict.  As a registered buffer it still
        # follows ordinary ``model.to(device)`` calls.
        self.register_buffer("_normalized_adj", None, persistent=False)
        self._train_interactions: Optional[Tensor] = None
        self._positive_items: Optional[list[set[int]]] = None
        self._negative_exclusions: Optional[list[set[int]]] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._cached_propagated: Optional[tuple[Tensor, Tensor]] = None
        self.history_: dict[str, list[float]] = {"loss": []}
        self.best_epoch_: Optional[int] = None
        self.best_score_: Optional[float] = None
        self.to(torch.device(device))

    @property
    def device(self) -> torch.device:
        """The actual device holding the learnable parameters."""

        return self.user_embedding.weight.device

    def _invalidate_cache(self) -> None:
        self._cached_propagated = None

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> "LightGCN":
        result = super()._apply(fn, recurse=recurse)
        self._invalidate_cache()
        return result

    def train(self, mode: bool = True) -> "LightGCN":
        super().train(mode)
        if mode:
            self._invalidate_cache()
        return self

    def load_state_dict(self, state_dict: Mapping[str, Tensor], strict: bool = True):
        result = super().load_state_dict(state_dict, strict=strict)
        self._invalidate_cache()
        return result

    def _reset_parameters(self) -> None:
        # A private CPU generator avoids changing the application's global RNG.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        scale = 0.1
        with torch.no_grad():
            self.user_embedding.weight.copy_(
                torch.randn(
                    self.user_embedding.weight.shape,
                    generator=generator,
                    dtype=self.user_embedding.weight.dtype,
                )
                * scale
            )
            self.item_embedding.weight.copy_(
                torch.randn(
                    self.item_embedding.weight.shape,
                    generator=generator,
                    dtype=self.item_embedding.weight.dtype,
                )
                * scale
            )

    def _validate_pairs(self, interactions: PairInput) -> Tensor:
        pairs = _as_pair_tensor(interactions)
        if pairs.shape[0] == 0:
            raise ValueError("at least one positive interaction is required")
        if torch.any(pairs[:, 0] < 0) or torch.any(pairs[:, 0] >= self.n_users):
            raise IndexError("user index outside [0, n_users)")
        if torch.any(pairs[:, 1] < 0) or torch.any(pairs[:, 1] >= self.n_items):
            raise IndexError("item index outside [0, n_items)")
        # Duplicate edges would otherwise change node degrees and graph weights.
        return torch.unique(pairs, dim=0, sorted=True)

    def build_graph(self, interactions: PairInput) -> Tensor:
        """Build and store ``D^-1/2 A D^-1/2`` for the bipartite graph."""

        self._invalidate_cache()
        pairs = self._validate_pairs(interactions)
        users = pairs[:, 0]
        items = pairs[:, 1] + self.n_users
        src = torch.cat((users, items))
        dst = torch.cat((items, users))
        indices = torch.stack((src, dst))

        n_nodes = self.n_users + self.n_items
        degree = torch.bincount(src, minlength=n_nodes).to(torch.float32)
        inverse_sqrt_degree = torch.zeros_like(degree)
        nonzero = degree > 0
        inverse_sqrt_degree[nonzero] = degree[nonzero].pow(-0.5)
        values = inverse_sqrt_degree[src] * inverse_sqrt_degree[dst]

        adjacency = torch.sparse_coo_tensor(
            indices,
            values,
            size=(n_nodes, n_nodes),
            dtype=self.user_embedding.weight.dtype,
            device="cpu",
        ).coalesce()
        self._normalized_adj = adjacency.to(self.device)
        self._train_interactions = pairs.clone()
        self._positive_items = [set() for _ in range(self.n_users)]
        for user, item in pairs.tolist():
            self._positive_items[user].add(item)
        self._negative_exclusions = [set(items) for items in self._positive_items]
        return self._normalized_adj

    def _set_negative_exclusions(
        self,
        forbidden_items_by_user: Optional[Mapping[int, Iterable[int]]],
    ) -> None:
        if self._positive_items is None:
            raise RuntimeError("build the training graph before setting exclusions")
        exclusions = [set(items) for items in self._positive_items]
        if forbidden_items_by_user is not None:
            if not isinstance(forbidden_items_by_user, Mapping):
                raise TypeError("forbidden_items_by_user must be a mapping")
            for raw_user, raw_items in forbidden_items_by_user.items():
                if isinstance(raw_user, bool) or not isinstance(raw_user, (int, np.integer)):
                    raise TypeError("forbidden user ids must be integers")
                user = int(raw_user)
                if user < 0 or user >= self.n_users:
                    raise IndexError("forbidden user index outside [0, n_users)")
                try:
                    items = list(raw_items)
                except TypeError as error:
                    raise TypeError("forbidden items must be iterable") from error
                for raw_item in items:
                    if isinstance(raw_item, bool) or not isinstance(
                        raw_item, (int, np.integer)
                    ):
                        raise TypeError("forbidden item ids must be integers")
                    item = int(raw_item)
                    if item < 0 or item >= self.n_items:
                        raise IndexError("forbidden item index outside [0, n_items)")
                    exclusions[user].add(item)
        self._negative_exclusions = exclusions

    def _propagate_once(self) -> tuple[Tensor, Tensor]:
        if self._normalized_adj is None:
            raise RuntimeError("the graph is not built; call fit() or build_graph() first")
        ego = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        layers = [ego]
        current = ego
        for _ in range(self.num_layers):
            current = torch.sparse.mm(self._normalized_adj, current)
            layers.append(current)
        combined = torch.stack(layers, dim=0).mean(dim=0)
        return combined[: self.n_users], combined[self.n_users :]

    def propagated_embeddings(self) -> tuple[Tensor, Tensor]:
        """Return user and item embeddings averaged over all graph layers."""

        if not self.training and self._cached_propagated is not None:
            return self._cached_propagated
        propagated = self._propagate_once()
        if not self.training:
            self._cached_propagated = tuple(value.detach() for value in propagated)
            return self._cached_propagated
        return propagated

    def forward(self, users: Tensor, items: Tensor) -> Tensor:
        """Return unnormalized dot-product scores for matched user-item pairs."""

        user_vectors, item_vectors = self.propagated_embeddings()
        users = users.to(device=self.device, dtype=torch.long)
        items = items.to(device=self.device, dtype=torch.long)
        return (user_vectors[users] * item_vectors[items]).sum(dim=-1)

    def bpr_loss(
        self, users: Tensor, positive_items: Tensor, negative_items: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute total, ranking, and unweighted embedding regularization loss."""

        user_vectors, item_vectors = self.propagated_embeddings()
        return self._bpr_loss_from_embeddings(
            users, positive_items, negative_items, user_vectors, item_vectors
        )

    def _bpr_loss_from_embeddings(
        self,
        users: Tensor,
        positive_items: Tensor,
        negative_items: Tensor,
        user_vectors: Tensor,
        item_vectors: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute BPR from one fixed propagation of the current parameters."""

        users = users.to(device=self.device, dtype=torch.long)
        positive_items = positive_items.to(device=self.device, dtype=torch.long)
        negative_items = negative_items.to(device=self.device, dtype=torch.long)

        positive_scores = (user_vectors[users] * item_vectors[positive_items]).sum(-1)
        negative_scores = (user_vectors[users] * item_vectors[negative_items]).sum(-1)
        ranking_loss = F.softplus(negative_scores - positive_scores).mean()

        # LightGCN regularizes the layer-0 (ego) embeddings, not the propagated
        # representation.  Division by batch size keeps its scale predictable.
        ego_users = self.user_embedding(users)
        ego_positive = self.item_embedding(positive_items)
        ego_negative = self.item_embedding(negative_items)
        reg_loss = 0.5 * (
            ego_users.square().sum()
            + ego_positive.square().sum()
            + ego_negative.square().sum()
        ) / max(1, users.numel())
        total_loss = ranking_loss + self.l2_reg * reg_loss
        return total_loss, ranking_loss, reg_loss

    def _sample_negatives(self, users: Tensor, generator: torch.Generator) -> Tensor:
        if self._negative_exclusions is None:
            raise RuntimeError("negative-sampling exclusions are unavailable")
        sampled: list[int] = []
        for user in users.tolist():
            excluded = self._negative_exclusions[user]
            if len(excluded) >= self.n_items:
                raise ValueError(f"user {user} has no permitted negative item")
            candidate = int(
                torch.randint(self.n_items, (1,), generator=generator).item()
            )
            while candidate in excluded:
                candidate = int(
                    torch.randint(self.n_items, (1,), generator=generator).item()
                )
            sampled.append(candidate)
        return torch.tensor(sampled, dtype=torch.long)

    def fit(
        self,
        train_interactions: PairInput,
        *,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        validation_callback: Optional[ValidationCallback] = None,
        patience: Optional[int] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        verbose: bool = False,
        seed: Optional[int] = None,
        shuffle: bool = True,
        forbidden_items_by_user: Optional[Mapping[int, Iterable[int]]] = None,
        **_: Any,
    ) -> "LightGCN":
        """Fit the model and return ``self``.

        Randomness is fully driven by a private CPU generator seeded with
        ``self.seed``.  Recreating a model with the same seed and fitting it on
        the same CPU inputs produces the same shuffles and negative samples.
        ``validation_callback`` must return a scalar for which larger is better.
        """

        pairs = self._validate_pairs(train_interactions)
        self.build_graph(pairs)
        self._set_negative_exclusions(forbidden_items_by_user)
        effective_epochs = self.epochs if epochs is None else int(epochs)
        effective_batch = self.batch_size if batch_size is None else int(batch_size)
        if effective_epochs < 0 or effective_batch <= 0:
            raise ValueError("epochs must be non-negative and batch_size positive")
        if patience is not None and patience < 0:
            raise ValueError("patience cannot be negative")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed if seed is None else int(seed))
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self._optimizer = optimizer
        self.history_ = {"loss": []}
        if validation_callback is not None:
            self.history_["validation_score"] = []
        best_score = -float("inf")
        best_state: Optional[dict[str, Tensor]] = None
        stale_epochs = 0
        self.best_epoch_ = None
        self.best_score_ = None

        self.train()
        for epoch in range(effective_epochs):
            permutation = (
                torch.randperm(pairs.shape[0], generator=generator)
                if shuffle
                else torch.arange(pairs.shape[0])
            )
            epoch_loss = 0.0
            example_count = int(pairs.shape[0])
            ordered_pairs = pairs[permutation]
            epoch_users = ordered_pairs[:, 0]
            epoch_positive = ordered_pairs[:, 1]
            epoch_negative = self._sample_negatives(epoch_users, generator)
            optimizer.zero_grad(set_to_none=True)
            user_vectors, item_vectors = self.propagated_embeddings()
            for start in range(0, pairs.shape[0], effective_batch):
                end = min(start + effective_batch, example_count)
                users = epoch_users[start:end]
                positive_items = epoch_positive[start:end]
                negative_items = epoch_negative[start:end]
                loss, _, _ = self._bpr_loss_from_embeddings(
                    users,
                    positive_items,
                    negative_items,
                    user_vectors,
                    item_vectors,
                )
                size = int(users.numel())
                epoch_loss += float(loss.detach().cpu()) * size
                # This is the gradient of the mean epoch objective evaluated at
                # one fixed set of propagated embeddings. Retaining only the
                # propagation graph makes minibatch accumulation substantially
                # cheaper than propagating the full graph for every minibatch.
                (loss * (size / example_count)).backward(
                    retain_graph=end < example_count
                )
            optimizer.step()
            self._invalidate_cache()

            record = {
                "epoch": float(epoch + 1),
                "loss": epoch_loss / max(1, example_count),
            }
            self.history_["loss"].append(record["loss"])
            if validation_callback is not None:
                self.eval()
                validation_score = float(validation_callback(self))
                self.train()
                record["validation_score"] = validation_score
                self.history_["validation_score"].append(validation_score)
                if validation_score > best_score:
                    best_score = validation_score
                    self.best_score_ = validation_score
                    self.best_epoch_ = epoch
                    stale_epochs = 0
                    best_state = copy.deepcopy(self.state_dict())
                    if checkpoint_path is not None:
                        self.save_checkpoint(checkpoint_path)
                else:
                    stale_epochs += 1
            elif checkpoint_path is not None and epoch == effective_epochs - 1:
                self.save_checkpoint(checkpoint_path)

            if verbose:
                validation_text = (
                    f", validation={record['validation_score']:.6f}"
                    if "validation_score" in record
                    else ""
                )
                print(
                    f"LightGCN epoch {epoch + 1}/{effective_epochs}: "
                    f"loss={record['loss']:.6f}{validation_text}"
                )
            if (
                validation_callback is not None
                and patience is not None
                and stale_epochs >= patience
            ):
                break

        if best_state is not None:
            self.load_state_dict(best_state)
        self.eval()
        return self

    def score_users(
        self,
        user_ids: Union[Tensor, np.ndarray, Sequence[int]],
        item_ids: Optional[Union[Tensor, np.ndarray, Sequence[int]]] = None,
        *,
        exclude_interactions: Optional[PairInput] = None,
        return_numpy: bool = True,
        batch_size: int = 1024,
    ) -> Union[Tensor, np.ndarray]:
        """Score items for one or more users.

        With ``item_ids=None`` this returns a full ``[n_users, n_items]`` score
        matrix.  A one-dimensional ``item_ids`` array denotes a shared candidate
        set.  A two-dimensional array denotes one candidate row per user.
        Interactions passed via ``exclude_interactions`` are masked with ``-inf``.
        Returned tensors live on CPU so evaluators cannot accidentally retain a
        large CUDA computation graph.
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        users = torch.as_tensor(user_ids, dtype=torch.long, device=self.device)
        if users.ndim == 0:
            users = users.unsqueeze(0)
        if users.ndim != 1:
            raise ValueError("user_ids must be one-dimensional")
        if torch.any(users < 0) or torch.any(users >= self.n_users):
            raise IndexError("user index outside [0, n_users)")

        self.eval()
        with torch.no_grad():
            user_vectors, item_vectors = self.propagated_embeddings()
            selected_users = user_vectors[users]
            if item_ids is None:
                scores = selected_users @ item_vectors.t()
            else:
                items = torch.as_tensor(item_ids, dtype=torch.long, device=self.device)
                if torch.any(items < 0) or torch.any(items >= self.n_items):
                    raise IndexError("item index outside [0, n_items)")
                if items.ndim == 1:
                    scores = selected_users @ item_vectors[items].t()
                elif items.ndim == 2 and items.shape[0] == users.shape[0]:
                    scores = (selected_users[:, None, :] * item_vectors[items]).sum(-1)
                else:
                    raise ValueError(
                        "item_ids must be one-dimensional or have one row per user"
                    )

            scores = scores.detach().cpu()
            if exclude_interactions is not None:
                if item_ids is not None:
                    raise ValueError(
                        "exclude_interactions is supported only for full-item scoring"
                    )
                exclusions = _as_pair_tensor(exclude_interactions)
                row_by_user = {int(user): row for row, user in enumerate(users.cpu())}
                for user, item in exclusions.tolist():
                    row = row_by_user.get(user)
                    if row is not None and 0 <= item < self.n_items:
                        scores[row, item] = -float("inf")

        return scores.numpy() if return_numpy else scores

    def get_config(self) -> dict[str, Any]:
        return {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "embedding_dim": self.embedding_dim,
            "num_layers": self.num_layers,
            "learning_rate": self.learning_rate,
            "l2_reg": self.l2_reg,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "seed": self.seed,
            "device": self.requested_device,
        }

    def save_checkpoint(
        self,
        path: Union[str, Path],
        *,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Save configuration, weights, graph edges, and optimizer state."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "model_class": self.__class__.__name__,
            "checkpoint_purpose": "inference_only",
            "config": self.get_config(),
            "model_state_dict": self.state_dict(),
            "train_interactions": self._train_interactions,
            "extra": dict(extra or {}),
            "best_epoch": self.best_epoch_,
            "best_score": self.best_score_,
        }
        torch.save(payload, target)

    def save(self, path: Union[str, Path]) -> None:
        """Save a complete checkpoint through the common recommender API."""

        self.save_checkpoint(path)

    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        *,
        device: Union[str, torch.device] = "cpu",
    ) -> "LightGCN":
        """Load a model saved by :meth:`save_checkpoint`."""

        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:  # PyTorch < 2.6 has no ``weights_only`` argument.
            checkpoint = torch.load(path, map_location=device)
        config = dict(checkpoint["config"])
        config["device"] = str(torch.device(device))
        model = cls(**config)
        interactions = checkpoint.get("train_interactions")
        if interactions is not None:
            model.build_graph(interactions)
        model.load_state_dict(checkpoint["model_state_dict"])
        model._optimizer = None
        model.best_epoch_ = checkpoint.get("best_epoch")
        model.best_score_ = checkpoint.get("best_score")
        model._invalidate_cache()
        model.eval()
        return model

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        *,
        device: Union[str, torch.device] = "cpu",
    ) -> "LightGCN":
        """Load a checkpoint through the common recommender API."""

        return cls.load_checkpoint(path, device=device)
