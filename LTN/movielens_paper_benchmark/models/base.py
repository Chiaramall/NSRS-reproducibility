"""Common interfaces and small data utilities for benchmark recommenders.

The evaluator intentionally consumes NumPy arrays.  Individual recommenders are
free to train with PyTorch or SciPy, but ``score_users`` always returns a dense
``(n_users, n_items)`` NumPy array (or the requested item subset).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, Sequence[int]]


def require_scipy():
    """Import SciPy lazily and fail with an actionable error message."""

    try:
        import scipy.sparse as sp
    except ImportError as exc:  # pragma: no cover - exercised without SciPy
        raise ImportError(
            "SciPy is required for sparse interaction matrices and EASE. "
            "Install it with `python -m pip install scipy`."
        ) from exc
    return sp


def interaction_arrays(
    interactions: Any,
    *,
    n_users: Optional[int] = None,
    n_items: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Normalize a sparse matrix, tuple, or ``N x (2|3)`` interaction array.

    Two-column input is interpreted as positive implicit feedback and receives
    targets equal to one. Sparse matrices retain their non-zero values.
    """

    if hasattr(interactions, "tocsr") and hasattr(interactions, "nonzero"):
        matrix = interactions.tocsr()
        users, items = matrix.nonzero()
        values = np.asarray(matrix[users, items]).reshape(-1)
        inferred_users, inferred_items = matrix.shape
    elif isinstance(interactions, (tuple, list)) and len(interactions) in (2, 3):
        users = np.asarray(interactions[0])
        items = np.asarray(interactions[1])
        values = (
            np.asarray(interactions[2], dtype=np.float32)
            if len(interactions) == 3
            else np.ones(len(users), dtype=np.float32)
        )
        inferred_users = int(users.max()) + 1 if users.size else 0
        inferred_items = int(items.max()) + 1 if items.size else 0
    else:
        array = np.asarray(interactions)
        if array.ndim != 2 or array.shape[1] not in (2, 3):
            raise ValueError(
                "interactions must be a sparse matrix, (users, items[, targets]), "
                "or an N x 2/3 array"
            )
        users, items = array[:, 0], array[:, 1]
        values = (
            array[:, 2].astype(np.float32, copy=False)
            if array.shape[1] == 3
            else np.ones(array.shape[0], dtype=np.float32)
        )
        inferred_users = int(users.max()) + 1 if users.size else 0
        inferred_items = int(items.max()) + 1 if items.size else 0

    users = np.asarray(users, dtype=np.int64).reshape(-1)
    items = np.asarray(items, dtype=np.int64).reshape(-1)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if not (len(users) == len(items) == len(values)):
        raise ValueError("users, items, and targets must have the same length")
    if users.size and (users.min() < 0 or items.min() < 0):
        raise ValueError("user and item indices must be non-negative")

    final_users = inferred_users if n_users is None else int(n_users)
    final_items = inferred_items if n_items is None else int(n_items)
    if final_users < inferred_users or final_items < inferred_items:
        raise ValueError("n_users/n_items are smaller than indices in interactions")
    return users, items, values, final_users, final_items


class BaseRecommender(ABC):
    """Minimal API shared by trainable benchmark recommenders."""

    n_users: int
    n_items: int

    @abstractmethod
    def fit(self, interactions: Any, **kwargs: Any) -> "BaseRecommender":
        """Fit the recommender and return ``self``."""

    @abstractmethod
    def score_users(
        self,
        user_ids: ArrayLike,
        item_ids: Optional[ArrayLike] = None,
        *,
        batch_size: int = 1024,
    ) -> np.ndarray:
        """Score users against every item or a supplied item subset."""

    @abstractmethod
    def state_dict(self) -> Mapping[str, Any]:
        """Return all state required to reproduce scores."""

    @abstractmethod
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore state produced by :meth:`state_dict`."""

    def save(self, path: Union[str, Path]) -> None:
        """Save a portable checkpoint, creating its parent directory."""

        import torch

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(self.state_dict()), destination)

