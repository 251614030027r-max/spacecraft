"""Learned convex-quadratic terminal cost for the constrained MPC.

The nominal MPC ends its horizon with a *fixed diagonal* quadratic penalty on
the deviation from the terminal reference, ``terminal_weight * ||S^-1 (x - r)||^2``
(see :class:`controllers.mpc.controller.MPCController`). On a short horizon that
penalty is a poor stand-in for the cost-to-go: it only says "be near the
reference at the last stage", it carries no information about the sustained
co-rotation effort the mission still owes past the horizon.

This module supplies the alternative the SAC-MPC coupling line rests on: a
single *convex quadratic* terminal value ``V(x) = (x-c)^T H (x-c) + g^T (x-c)``
with ``H`` positive semi-definite. It is convex by construction, so it enters
the QP as ``||L (x-c)||^2 + g^T (x-c)`` (with ``L = chol(H)``) without turning
the lower layer into a non-convex program -- this is exactly the mitigation for
the AC4MPC "critic trap": no neural network in the solver, no manifold
Jacobian, and a bad fit can only make the QP *suboptimal*, never unsafe, because
the truth constraints are separate hard/soft constraints the terminal value
never touches.

The first version is fit offline by supervised regression against the
closed-loop cost-to-go of a reference policy (``experiments/fit_terminal_value.py``),
following the convex-terminal-cost surrogate line (Bemporad, CDC 2021;
arXiv 2508.05804). It is defined on the 12-dimensional MPC relative state, never
on the 24-d learned observation, so it needs no observation reconstruction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

# Floor for the eigenvalues of the fitted Hessian. Keeps H strictly PSD (so the
# Cholesky factor exists) even when the least-squares fit produces a near-zero
# or slightly negative curvature direction.
_EIGENVALUE_FLOOR = 1.0e-9


@dataclass(frozen=True)
class ConvexQuadraticTerminalValue:
    """Convex quadratic ``V(x) = (x-c)^T H (x-c) + g^T (x-c) + constant``.

    ``weight_matrix`` (``H``) is symmetric positive semi-definite; ``linear``
    (``g``) and ``center`` (``c``) are 12-vectors. ``constant`` is retained for
    faithful value reporting but is irrelevant to the QP minimiser and is
    dropped from the terminal objective.
    """

    center: FloatArray
    weight_matrix: FloatArray
    linear: FloatArray
    constant: float = 0.0

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        weight = np.asarray(self.weight_matrix, dtype=np.float64)
        linear = np.asarray(self.linear, dtype=np.float64)
        if center.shape != (12,) or not np.all(np.isfinite(center)):
            raise ValueError("terminal-value center must be finite and shape=(12,)")
        if linear.shape != (12,) or not np.all(np.isfinite(linear)):
            raise ValueError("terminal-value linear term must be finite and shape=(12,)")
        if weight.shape != (12, 12) or not np.all(np.isfinite(weight)):
            raise ValueError("terminal-value weight must be finite and shape=(12, 12)")
        # Symmetrise defensively; a round-tripped or hand-built matrix may carry
        # tiny asymmetry that would otherwise fail the PSD check below.
        weight = 0.5 * (weight + weight.T)
        eigenvalues = np.linalg.eigvalsh(weight)
        if float(np.min(eigenvalues)) < -1.0e-8:
            raise ValueError(
                "terminal-value weight must be positive semi-definite "
                f"(min eigenvalue {float(np.min(eigenvalues)):.3e})"
            )
        object.__setattr__(self, "center", center.copy())
        object.__setattr__(self, "weight_matrix", weight.copy())
        object.__setattr__(self, "linear", linear.copy())
        object.__setattr__(self, "constant", float(self.constant))

    def cholesky_factor(self) -> FloatArray:
        """Return ``L`` with ``L^T L = H`` so that ``||L z||^2 = z^T H z``.

        numpy's ``cholesky`` returns a lower factor ``Lc`` with ``H = Lc Lc^T``;
        the QP wants ``||L z||^2 = z^T (L^T L) z = z^T H z``, so ``L = Lc^T``.
        A small diagonal jitter guarantees the factorisation exists for a matrix
        that is PSD but only marginally so.
        """

        weight = self.weight_matrix + _EIGENVALUE_FLOOR * np.eye(12)
        return np.linalg.cholesky(weight).T

    def value(self, state: ArrayLike) -> float:
        """Evaluate ``V`` at a 12-d relative state (for diagnostics and tests)."""

        z = np.asarray(state, dtype=np.float64) - self.center
        if z.shape != (12,):
            raise ValueError("state must be shape=(12,)")
        return (
            float(z @ self.weight_matrix @ z)
            + float(self.linear @ z)
            + self.constant
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "center": self.center.tolist(),
            "weight_matrix": self.weight_matrix.tolist(),
            "linear": self.linear.tolist(),
            "constant": self.constant,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ConvexQuadraticTerminalValue":
        return cls(
            center=np.asarray(payload["center"], dtype=np.float64),
            weight_matrix=np.asarray(payload["weight_matrix"], dtype=np.float64),
            linear=np.asarray(payload["linear"], dtype=np.float64),
            constant=float(payload.get("constant", 0.0)),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ConvexQuadraticTerminalValue":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _quadratic_features(deviations: FloatArray) -> tuple[FloatArray, list[tuple[int, int]]]:
    """Design matrix ``[1, z_i, z_i z_j (i<=j)]`` for a batch of deviations.

    Returns the ``(N, 1 + 12 + 78)`` feature matrix and the list of ``(i, j)``
    index pairs (``i <= j``) in the order their columns appear, so the fitted
    coefficients can be folded back into a symmetric ``12x12`` matrix.
    """

    n = deviations.shape[0]
    pairs = [(i, j) for i in range(12) for j in range(i, 12)]
    columns = [np.ones((n, 1)), deviations]
    quadratic = np.stack(
        [deviations[:, i] * deviations[:, j] for (i, j) in pairs], axis=1
    )
    columns.append(quadratic)
    return np.concatenate(columns, axis=1), pairs


def fit_convex_quadratic(
    states: ArrayLike,
    costs_to_go: ArrayLike,
    *,
    center: ArrayLike,
    ridge: float = 1.0e-6,
) -> ConvexQuadraticTerminalValue:
    """Fit the best *convex* quadratic cost-to-go under a PSD constraint.

    ``V(x) = constant + g^T z + z^T M z`` with ``z = x - center`` is fit by
    least squares over quadratic features, but the quadratic block ``M`` is
    constrained ``M >= 0`` *inside* the optimisation rather than clipped after
    an unconstrained fit. Post-hoc clipping is wrong: it changes ``M`` while
    leaving the jointly-fit ``g`` and ``constant`` in place, so the surrogate no
    longer matches the data at all (a smoke fit that way scored R^2 far below
    zero). Solving the convex program returns the genuinely best convex fit, so
    the reported residual is meaningful and the lower QP stays convex by
    construction -- the AC4MPC "critic trap" mitigation done cleanly.

    The program is assembled in normal-equation form (a fixed 91x91 quadratic in
    the coefficients) so its size is independent of the number of samples.
    """

    import cvxpy as cp

    x = np.asarray(states, dtype=np.float64)
    y = np.asarray(costs_to_go, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 12:
        raise ValueError("states must have shape (N, 12)")
    if y.shape != (x.shape[0],):
        raise ValueError("costs_to_go must have shape (N,)")
    if c.shape != (12,):
        raise ValueError("center must have shape (12,)")

    z = x - c
    features, pairs = _quadratic_features(z)
    n_features = features.shape[1]
    # Normal equations: minimise theta^T A theta - 2 b^T theta (+ const), which
    # equals ||features theta - y||^2 up to a theta-independent term.
    gram = features.T @ features + ridge * np.eye(n_features)
    rhs = features.T @ y

    constant = cp.Variable()
    linear = cp.Variable(12)
    weight = cp.Variable((12, 12), PSD=True)
    # The design column for z_i z_j carries coefficient (M_ij + M_ji) = 2 M_ij
    # off the diagonal and M_ii on it, so map the PSD variable to that ordering.
    quadratic_terms = [
        weight[i, i] if i == j else 2.0 * weight[i, j] for (i, j) in pairs
    ]
    theta = cp.hstack([constant, linear, cp.hstack(quadratic_terms)])
    objective = cp.quad_form(theta, cp.psd_wrap(gram)) - 2.0 * rhs @ theta
    problem = cp.Problem(cp.Minimize(objective))
    problem.solve(solver=cp.CLARABEL)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"terminal-value fit did not converge: {problem.status}")

    fitted_weight = np.asarray(weight.value, dtype=np.float64)
    fitted_weight = 0.5 * (fitted_weight + fitted_weight.T)
    # Lift any tiny negative eigenvalue left by solver tolerance to the floor so
    # the ConvexQuadraticTerminalValue PSD check and Cholesky both succeed.
    eigenvalues, vectors = np.linalg.eigh(fitted_weight)
    clipped = np.clip(eigenvalues, _EIGENVALUE_FLOOR, None)
    fitted_weight = (vectors * clipped) @ vectors.T
    fitted_weight = 0.5 * (fitted_weight + fitted_weight.T)

    return ConvexQuadraticTerminalValue(
        center=c,
        weight_matrix=fitted_weight,
        linear=np.asarray(linear.value, dtype=np.float64),
        constant=float(constant.value),
    )


__all__ = [
    "ConvexQuadraticTerminalValue",
    "fit_convex_quadratic",
]
