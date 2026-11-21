"""Small-vector helpers that give exactly numpy's results without its per-call overhead.

The planners and the MPC call ``np.cross`` and ``np.linalg.norm`` hundreds of thousands of times per episode on
3-vectors, where numpy's axis handling dominates the cost. These helpers perform the same floating-point operations in
the same order (``np.cross``: a1 b2 - a2 b1, a2 b0 - a0 b2, a0 b1 - a1 b0; ``np.linalg.norm`` of a 1-D float vector:
sqrt(x . x)), so the results are bitwise equal; any other shape or dtype falls back to numpy itself.
"""
from __future__ import annotations

import math

import numpy as np

_F64 = np.dtype(np.float64)


def cross3(a, b) -> np.ndarray:
    """``np.cross(a, b)``; fast path for two float64 vectors of shape (3,)."""
    if type(a) is np.ndarray and type(b) is np.ndarray and a.shape == (3,) and b.shape == (3,) \
            and a.dtype == _F64 and b.dtype == _F64:
        a0, a1, a2 = a
        b0, b1, b2 = b
        return np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])
    return np.cross(a, b)


def norm3(x) -> float:
    """``float(np.linalg.norm(x))``; fast path for a float64 vector of shape (3,)."""
    if type(x) is np.ndarray and x.shape == (3,) and x.dtype == _F64:
        return math.sqrt(x.dot(x))
    return float(np.linalg.norm(x))


__all__ = ["cross3", "norm3"]
