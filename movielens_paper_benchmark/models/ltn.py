"""Logic Tensor Network recommenders for MovieLens benchmarks."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from .mf import MFScorer

try:
    import ltn
except ImportError as error:  # pragma: no cover - environment check
    ltn = None
    _LTN_IMPORT_ERROR = error
else:
    _LTN_IMPORT_ERROR = None


class LTNRecommender:
    """MF optimized through fuzzy satisfaction, with an optional genre rule."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        biased: bool = True,
        learning_rate: float = 1e-3,
        l2: float = 1e-4,
        alpha: float = 0.05,
        p: int = 2,
        genre_rule_weight: float = 0.0,
        seed: int = 0,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        if ltn is None:
            raise ImportError(
                "LTNRecommender requires the LTNtorch package"
            ) from _LTN_IMPORT_ERROR
        if n_users <= 0 or n_items <= 1 or embedding_dim <= 0:
            raise ValueError("n_users/embedding_dim must be positive and n_items > 1")
        if learning_rate <= 0 or l2 < 0 or alpha <= 0:
            raise ValueError("learning_rate/alpha must be positive and l2 non-negative")
        if p <= 0 or genre_rule_weight < 0:
            raise ValueError("p must be positive and genre_rule_weight non-negative")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.embedding_dim = int(embedding_dim)
        self.biased = bool(biased)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.alpha = float(alpha)
        self.p = int(p)
        self.genre_rule_weight = float(genre_rule_weight)
        self.seed = int(seed)
        self.device = torch.device(device)

        # Embeddings are initialized on CPU inside an isolated RNG context, then
        # moved. Recreating a model with the same seed gives identical weights
        # without changing the caller's global RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            module = MFScorer(
                self.n_users, self.n_items, self.embedding_dim, self.biased
            )
        self.model = module.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2
        )

        # LTNtorch variables and module parameters must always share this device.
        ltn.device = self.device
        self.likes = ltn.Function(self.model)
        self.similarity = ltn.Predicate(
            func=lambda prediction, target: torch.exp(
                -self.alpha * torch.square(prediction - target)
            )
        )
        self.forall = ltn.Quantifier(
            ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f"
        )
        self.sat_agg = ltn.fuzzy_ops.SatAgg()
        self.history: list = []
        self.best_epoch_: Optional[int] = None
        self.best_score_: Optional[float] = None

    def _satisfaction(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        targets: torch.Tensor,
        p: Optional[int] = None,
    ):
        u = ltn.Variable("users", users.long(), add_batch_dim=False)
        i = ltn.Variable("items", items.long(), add_batch_dim=False)
        r = ltn.Variable("targets", targets.float(), add_batch_dim=False)
        kwargs = {} if p is None else {"p": int(p)}
        return self.forall(
            ltn.diag(u, i, r), self.similarity(self.likes(u, i), r), **kwargs
        )

    def _knowledge_base_loss(
        self,
        data_satisfaction: torch.Tensor,
        genre_satisfaction: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate closed formulas with stable weighted p-mean error."""
        if self.genre_rule_weight == 1.0:
            # Literal original-repository SatAgg path, including pi_1 stability.
            return 1.0 - self.sat_agg(
                data_satisfaction, genre_satisfaction
            )
        data_error = 1.0 - ltn.fuzzy_ops.pi_1(data_satisfaction)
        genre_error = 1.0 - ltn.fuzzy_ops.pi_1(genre_satisfaction)
        return torch.sqrt(
            (
                data_error.square()
                + self.genre_rule_weight * genre_error.square()
            )
            / (1.0 + self.genre_rule_weight)
        )

    @staticmethod
    def build_genre_examples(
        user_genre_preferences: Mapping[int, Mapping[str, Iterable[int]]],
        item_genres: Mapping[int, Iterable[int]],
        max_items_per_user: int,
        seed: int,
        excluded_positive_items: Optional[
            Mapping[int, Iterable[int]]
        ] = None,
    ) -> np.ndarray:
        """Sample unambiguous negative constraints from training-only genres.

        A candidate is skipped when it is a positive training item or when one
        of its genres is also liked by the user. This prevents contradictory
        supervision without consulting validation or test labels.
        """
        if max_items_per_user <= 0:
            return np.empty((0, 3), dtype=np.float32)

        normalized_item_genres = {
            int(item): {int(genre) for genre in genres}
            for item, genres in item_genres.items()
        }
        genre_items: Dict[int, list] = {}
        for item, genres in normalized_item_genres.items():
            for genre in genres:
                genre_items.setdefault(genre, []).append(item)

        rng = np.random.default_rng(int(seed))
        examples = []
        for user in sorted(user_genre_preferences):
            preferences = user_genre_preferences[user]
            disliked = {int(value) for value in preferences.get("disliked", ())}
            liked = {int(value) for value in preferences.get("liked", ())}
            excluded = set()
            if excluded_positive_items is not None:
                excluded = {
                    int(value)
                    for value in excluded_positive_items.get(user, ())
                }
            candidates = sorted(
                {
                    item
                    for genre in disliked
                    for item in genre_items.get(genre, ())
                    if item not in excluded
                    and not normalized_item_genres[item].intersection(liked)
                }
            )
            if len(candidates) > max_items_per_user:
                candidates = sorted(
                    rng.choice(
                        candidates, size=max_items_per_user, replace=False
                    ).tolist()
                )
            examples.extend((int(user), int(item), 0.0) for item in candidates)
        return np.asarray(examples, dtype=np.float32).reshape(-1, 3)

    def _validate_examples(self, data: np.ndarray, label: str) -> np.ndarray:
        array = np.asarray(data)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(f"{label} must have shape [n, 3]")
        if array.size:
            users = array[:, 0]
            items = array[:, 1]
            targets = array[:, 2]
            if np.any(users < 0) or np.any(users >= self.n_users):
                raise IndexError(f"{label} contains an invalid user index")
            if np.any(items < 0) or np.any(items >= self.n_items):
                raise IndexError(f"{label} contains an invalid item index")
            if not np.all(np.isin(targets, (0.0, 1.0))):
                raise ValueError(f"{label} targets must be binary")
        return array.astype(np.float32, copy=False)

    def fit(
        self,
        interactions: np.ndarray,
        batch_size: int = 512,
        epochs: int = 100,
        patience: int = 10,
        seed: Optional[int] = None,
        validation_callback: Optional[Callable[["LTNRecommender"], float]] = None,
        genre_examples: Optional[np.ndarray] = None,
    ) -> "LTNRecommender":
        data = self._validate_examples(interactions, "interactions")
        if len(data) == 0:
            raise ValueError("interactions cannot be empty")
        if batch_size <= 0 or epochs <= 0 or patience <= 0:
            raise ValueError("batch_size/epochs/patience must be positive")

        run_seed = self.seed if seed is None else int(seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)
        torch.use_deterministic_algorithms(True)
        np_rng = np.random.default_rng(run_seed)
        genre_data = (
            None
            if genre_examples is None
            else self._validate_examples(genre_examples, "genre_examples")
        )

        best_state = copy.deepcopy(self.model.state_dict())
        self.best_score_ = None
        self.best_epoch_ = None
        stale = 0
        self.history = []

        for epoch in range(int(epochs)):
            self.model.train()
            order = np_rng.permutation(len(data))
            genre_order = None
            if genre_data is not None and len(genre_data):
                genre_order = np_rng.permutation(len(genre_data))
            losses = []

            for batch_index, start in enumerate(
                range(0, len(order), int(batch_size))
            ):
                indices = order[start : start + int(batch_size)]
                batch = data[indices]
                users = torch.as_tensor(
                    batch[:, 0], dtype=torch.long, device=self.device
                )
                items = torch.as_tensor(
                    batch[:, 1], dtype=torch.long, device=self.device
                )
                targets = torch.as_tensor(
                    batch[:, 2], dtype=torch.float32, device=self.device
                )
                self.optimizer.zero_grad(set_to_none=True)
                data_satisfaction = self._satisfaction(
                    users, items, targets
                ).value
                loss = 1.0 - data_satisfaction

                if genre_order is not None and self.genre_rule_weight > 0.0:
                    genre_start = (batch_index * int(batch_size)) % len(genre_order)
                    genre_indices = genre_order[
                        genre_start : genre_start + int(batch_size)
                    ]
                    if len(genre_indices) < int(batch_size):
                        missing = int(batch_size) - len(genre_indices)
                        genre_indices = np.concatenate(
                            (genre_indices, genre_order[:missing])
                        )
                    genre_batch = genre_data[genre_indices]
                    genre_satisfaction = self._satisfaction(
                        torch.as_tensor(
                            genre_batch[:, 0], dtype=torch.long, device=self.device
                        ),
                        torch.as_tensor(
                            genre_batch[:, 1], dtype=torch.long, device=self.device
                        ),
                        torch.as_tensor(
                            genre_batch[:, 2],
                            dtype=torch.float32,
                            device=self.device,
                        ),
                        p=self.p,
                    ).value
                    loss = self._knowledge_base_loss(
                        data_satisfaction, genre_satisfaction
                    )

                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.detach().cpu()))

            validation_score = (
                None
                if validation_callback is None
                else float(validation_callback(self))
            )
            if validation_score is not None and not np.isfinite(validation_score):
                raise ValueError("validation_callback returned a non-finite score")
            self.history.append(
                {
                    "epoch": epoch + 1,
                    "loss": float(np.mean(losses)),
                    "validation_score": validation_score,
                }
            )
            if validation_score is None:
                best_state = copy.deepcopy(self.model.state_dict())
                continue
            if self.best_score_ is None or validation_score > self.best_score_:
                self.best_score_ = validation_score
                self.best_epoch_ = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= int(patience):
                    break

        self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def score_users(
        self,
        user_ids: Sequence[int],
        *,
        batch_size: int = 1024,
    ) -> np.ndarray:
        users = np.asarray(user_ids, dtype=np.int64).reshape(-1)
        if users.size and (users.min() < 0 or users.max() >= self.n_users):
            raise IndexError("user id out of range")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        outputs = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(users), int(batch_size)):
                batch = torch.as_tensor(
                    users[start : start + int(batch_size)],
                    dtype=torch.long,
                    device=self.device,
                )
                outputs.append(
                    self.model.score_users(batch)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
        return (
            np.concatenate(outputs, axis=0)
            if outputs
            else np.empty((0, self.n_items), dtype=np.float32)
        )

    def state_dict(self) -> Dict[str, object]:
        return {
            "format_version": 1,
            "class": self.__class__.__name__,
            "checkpoint_purpose": "inference_only",
            "model": {
                key: value.detach().cpu()
                for key, value in self.model.state_dict().items()
            },
            "config": {
                "n_users": self.n_users,
                "n_items": self.n_items,
                "embedding_dim": self.embedding_dim,
                "biased": self.biased,
                "learning_rate": self.learning_rate,
                "l2": self.l2,
                "alpha": self.alpha,
                "p": self.p,
                "genre_rule_weight": self.genre_rule_weight,
                "seed": self.seed,
            },
            "history": self.history,
            "best_epoch": self.best_epoch_,
            "best_score": self.best_score_,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        config = dict(state["config"])
        expected = {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "embedding_dim": self.embedding_dim,
            "biased": self.biased,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "alpha": self.alpha,
            "p": self.p,
            "genre_rule_weight": self.genre_rule_weight,
            "seed": self.seed,
        }
        if config != expected:
            raise ValueError(
                f"checkpoint config {config} does not match model config {expected}"
            )
        self.model.load_state_dict(state["model"])
        self.model.to(self.device)
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        self.history = list(state.get("history", []))
        self.best_epoch_ = state.get("best_epoch")
        self.best_score_ = state.get("best_score")

    def save(self, path: Union[str, Path]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), target)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        *,
        device: Union[str, torch.device] = "cpu",
    ) -> "LTNRecommender":
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        model = cls(**dict(state["config"]), device=device)
        model.load_state_dict(state)
        return model
