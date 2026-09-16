"""Recommendation models used by the MovieLens benchmark."""

import os

# Must be present before the first deterministic CUDA matrix multiplication.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .base import BaseRecommender
from .ease import EASE
from .lightgcn import LightGCN
from .ltn import LTNRecommender
from .mf import MF, MatrixFactorization, sample_unobserved_items

__all__ = [
    "BaseRecommender",
    "EASE",
    "LightGCN",
    "LTNRecommender",
    "MF",
    "MatrixFactorization",
    "sample_unobserved_items",
]
