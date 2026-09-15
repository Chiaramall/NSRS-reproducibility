"""Candidate layout shared by the MindReader baselines."""
import numpy as np


def candidate_arrays(rows):
    values = np.asarray(rows, dtype=np.int64)
    if values.ndim != 3 or values.shape[1:] != (101, 2):
        raise ValueError(f"expected [users,101,2] candidates, got {values.shape}")
    users = values[:, -1, 0]
    if np.any(values[:, :, 0] != users[:, None]):
        raise ValueError("a candidate row contains more than one user")
    return users, values[:, :, 1]
