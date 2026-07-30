"""Streaming algebra for large nonlinear least-squares datasets.

Residuals and their Jacobian rows can be evaluated a shard or row group at a
time. This module accumulates the sufficient statistics needed by a
Gauss--Newton step without retaining the raw observations or full Jacobian.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from .model import FloatArray


@dataclass(slots=True)
class GaussNewtonAccumulator:
    """Accumulate normal-equation blocks from whitened residual chunks."""

    variable_dim: int
    hessian: FloatArray = field(init=False)
    gradient: FloatArray = field(init=False)
    objective: float = field(init=False, default=0.0)
    residual_count: int = field(init=False, default=0)
    block_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if not isinstance(self.variable_dim, int) or self.variable_dim <= 0:
            raise ValueError("variable_dim must be a positive integer")
        self.hessian = np.zeros((self.variable_dim, self.variable_dim), dtype=float)
        self.gradient = np.zeros(self.variable_dim, dtype=float)

    def add_block(self, residuals: ArrayLike, jacobian: ArrayLike) -> None:
        residual_array = np.asarray(residuals, dtype=float)
        jacobian_array = np.asarray(jacobian, dtype=float)
        if residual_array.ndim != 1:
            raise ValueError("residuals must be one-dimensional")
        expected_shape = (residual_array.size, self.variable_dim)
        if jacobian_array.shape != expected_shape:
            raise ValueError(
                f"jacobian must have shape {expected_shape}; got {jacobian_array.shape}"
            )
        if not np.all(np.isfinite(residual_array)) or not np.all(
            np.isfinite(jacobian_array)
        ):
            raise ValueError("residuals and jacobian must be finite")

        self.hessian += jacobian_array.T @ jacobian_array
        self.gradient += jacobian_array.T @ residual_array
        self.objective += 0.5 * float(residual_array @ residual_array)
        self.residual_count += residual_array.size
        self.block_count += 1

    def merge(self, other: "GaussNewtonAccumulator") -> None:
        if other.variable_dim != self.variable_dim:
            raise ValueError("Cannot merge accumulators with different dimensions")
        self.hessian += other.hessian
        self.gradient += other.gradient
        self.objective += other.objective
        self.residual_count += other.residual_count
        self.block_count += other.block_count
