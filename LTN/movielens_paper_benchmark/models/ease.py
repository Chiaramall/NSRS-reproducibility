"""Embarrassingly Shallow Autoencoder (EASE) for implicit feedback."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np
import torch

from .base import ArrayLike, BaseRecommender, interaction_arrays, require_scipy


class EASE(BaseRecommender):
    """Closed-form EASE recommender with dense NumPy and optional CSR input.

    ``regularization`` is added to the diagonal of ``X.T @ X``. Training stores
    the binary interaction matrix because user scores are ``X @ B``. Scoring is
    batched so all users' scores need not be materialized simultaneously.
    """

    def __init__(
        self,
        *,
        regularization: float = 200.0,
        n_users: Optional[int] = None,
        n_items: Optional[int] = None,
        dtype: Union[str, np.dtype] = np.float64,
    ) -> None:
        if regularization <= 0:
            raise ValueError("regularization must be positive")
        self.regularization = float(regularization)
        self.n_users = 0 if n_users is None else int(n_users)
        self.n_items = 0 if n_items is None else int(n_items)
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("EASE dtype must be float32 or float64")
        self.coef_: Optional[np.ndarray] = None
        self.interactions_ = None

    def fit(self, interactions: Any, **_: Any) -> "EASE":
        """Fit EASE on a binary matrix, CSR matrix, or coordinate interactions."""

        dense = np.asarray(interactions) if isinstance(interactions, np.ndarray) else None
        is_dense_matrix = (
            dense is not None
            and dense.ndim == 2
            and (
                dense.shape[1] not in (2, 3)
                or (self.n_users == dense.shape[0] and self.n_items == dense.shape[1])
            )
        )
        if is_dense_matrix:
            matrix = (dense > 0).astype(self.dtype, copy=False)
            if not matrix.any():
                raise ValueError("EASE requires positive implicit interactions")
            inferred_users, inferred_items = matrix.shape
            if self.n_users not in (0, inferred_users) or self.n_items not in (0, inferred_items):
                raise ValueError("dense interaction matrix shape does not match n_users/n_items")
            self.n_users, self.n_items = inferred_users, inferred_items
        elif hasattr(interactions, "tocsr"):
            matrix = interactions.tocsr().astype(self.dtype)
            if matrix.nnz == 0:
                raise ValueError("EASE requires positive implicit interactions")
            matrix.data[:] = matrix.data > 0
            matrix.eliminate_zeros()
            inferred_users, inferred_items = matrix.shape
            if self.n_users not in (0, inferred_users) or self.n_items not in (0, inferred_items):
                raise ValueError("sparse interaction matrix shape does not match n_users/n_items")
            self.n_users, self.n_items = inferred_users, inferred_items
        else:
            users, items, values, n_users, n_items = interaction_arrays(
                interactions,
                n_users=self.n_users or None,
                n_items=self.n_items or None,
            )
            positive = values > 0
            if not positive.any():
                raise ValueError("EASE requires positive implicit interactions")
            self.n_users, self.n_items = n_users, n_items
            matrix = np.zeros((self.n_users, self.n_items), dtype=self.dtype)
            matrix[users[positive], items[positive]] = 1.0

        product = matrix.T @ matrix
        gram = product.toarray() if hasattr(product, "toarray") else np.asarray(product)
        diagonal = np.diag_indices_from(gram)
        gram[diagonal] += self.regularization
        precision = np.linalg.inv(gram)
        precision_diagonal = np.diag(precision).copy()
        if np.any(precision_diagonal == 0):
            raise np.linalg.LinAlgError("zero diagonal encountered in EASE precision matrix")
        coefficients = -precision / precision_diagonal[np.newaxis, :]
        coefficients[diagonal] = 0.0
        self.coef_ = np.asarray(coefficients, dtype=self.dtype)
        self.interactions_ = matrix
        return self

    def _check_fitted(self) -> None:
        if self.coef_ is None or self.interactions_ is None:
            raise RuntimeError("EASE must be fitted before scoring")

    def score_users(
        self,
        user_ids: ArrayLike,
        item_ids: Optional[ArrayLike] = None,
        *,
        batch_size: int = 1024,
    ) -> np.ndarray:
        self._check_fitted()
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

        outputs = []
        for start in range(0, len(users), batch_size):
            end = start + batch_size
            scores = np.asarray(self.interactions_[users[start:end]] @ self.coef_)
            if items is not None:
                scores = scores[:, items] if items.ndim == 1 else np.take_along_axis(
                    scores, items[start:end], axis=1
                )
            outputs.append(scores.astype(np.float32, copy=False))
        width = self.n_items if items is None else (items.shape[-1] if items.ndim == 2 else len(items))
        return np.concatenate(outputs, axis=0) if outputs else np.empty((0, width), np.float32)

    def state_dict(self) -> Mapping[str, Any]:
        self._check_fitted()
        state = {
            "format_version": 1,
            "class": self.__class__.__name__,
            "config": {
                "regularization": self.regularization,
                "n_users": self.n_users,
                "n_items": self.n_items,
                "dtype": self.dtype.str,
            },
            "coef": self.coef_,
        }
        if hasattr(self.interactions_, "tocsr"):
            matrix = self.interactions_.tocsr()
            state["interactions_format"] = "csr"
            state["interactions"] = {
                "data": matrix.data,
                "indices": matrix.indices,
                "indptr": matrix.indptr,
                "shape": matrix.shape,
            }
        else:
            state["interactions_format"] = "dense"
            state["interactions"] = np.asarray(self.interactions_, dtype=self.dtype)
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        config = state["config"]
        if int(config["n_users"]) != self.n_users or int(config["n_items"]) != self.n_items:
            raise ValueError("checkpoint dimensions do not match EASE instance")
        if float(config["regularization"]) != self.regularization:
            raise ValueError("checkpoint regularization does not match EASE instance")
        self.coef_ = np.asarray(state["coef"], dtype=self.dtype)
        if state["interactions_format"] == "csr":
            sp = require_scipy()
            csr = state["interactions"]
            self.interactions_ = sp.csr_matrix(
                (csr["data"], csr["indices"], csr["indptr"]), shape=tuple(csr["shape"])
            )
        else:
            self.interactions_ = np.asarray(state["interactions"], dtype=self.dtype)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EASE":
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        model = cls(**dict(state["config"]))
        model.load_state_dict(state)
        return model
