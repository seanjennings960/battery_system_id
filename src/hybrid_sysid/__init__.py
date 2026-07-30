"""Hybrid, multiscale nonlinear ODE system-identification tools."""

from .model import HybridODEModel, Segment
from .multiple_shooting import MultipleShootingProblem, ProblemEvaluation
from .streaming import GaussNewtonAccumulator
from .solver import (
    SolverIteration,
    SolverOptions,
    SolverResult,
    finite_difference_jacobian,
    solve_constrained_gauss_newton,
)

__all__ = [
    "GaussNewtonAccumulator",
    "HybridODEModel",
    "MultipleShootingProblem",
    "ProblemEvaluation",
    "Segment",
    "SolverIteration",
    "SolverOptions",
    "SolverResult",
    "finite_difference_jacobian",
    "solve_constrained_gauss_newton",
]
