"""A compact equality-constrained generalized Gauss-Newton/SQP solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import lstsq

from .model import FloatArray
from .multiple_shooting import MultipleShootingProblem, ProblemEvaluation


@dataclass(frozen=True, slots=True)
class SolverOptions:
    max_iterations: int = 25
    finite_difference_step: float = 1e-6
    initial_damping: float = 1e-4
    minimum_damping: float = 1e-10
    maximum_damping: float = 1e8
    constraint_penalty: float = 1e4
    step_tolerance: float = 1e-8
    constraint_tolerance: float = 1e-7
    gradient_tolerance: float = 1e-7
    max_damping_trials: int = 8
    max_line_search_steps: int = 14
    minimum_step_fraction: float = 2.0**-14


@dataclass(frozen=True, slots=True)
class SolverIteration:
    iteration: int
    objective: float
    constraint_norm: float
    merit: float
    damping: float
    accepted_step_fraction: float
    step_norm: float
    parameter_coordinates: FloatArray


@dataclass(frozen=True, slots=True)
class SolverResult:
    decision: FloatArray
    evaluation: ProblemEvaluation
    success: bool
    message: str
    history: tuple[SolverIteration, ...]


def finite_difference_jacobian(
    function: Callable[[FloatArray], FloatArray],
    point: ArrayLike,
    *,
    relative_step: float = 1e-6,
) -> FloatArray:
    """Central-difference Jacobian with a coordinate-relative perturbation."""

    x = np.asarray(point, dtype=float)
    value = np.asarray(function(x), dtype=float)
    if value.ndim != 1:
        raise ValueError("finite_difference_jacobian expects a vector-valued function")
    jacobian = np.empty((value.size, x.size), dtype=float)
    if value.size == 0:
        return jacobian

    for column in range(x.size):
        step = relative_step * max(1.0, abs(x[column]))
        plus = x.copy()
        minus = x.copy()
        plus[column] += step
        minus[column] -= step
        jacobian[:, column] = (
            np.asarray(function(plus), dtype=float)
            - np.asarray(function(minus), dtype=float)
        ) / (2.0 * step)
    return jacobian


def _merit(evaluation: ProblemEvaluation, constraint_penalty: float) -> float:
    return evaluation.objective + 0.5 * constraint_penalty * float(
        evaluation.constraints @ evaluation.constraints
    )


def _solve_kkt_step(
    residuals: FloatArray,
    constraints: FloatArray,
    residual_jacobian: FloatArray,
    constraint_jacobian: FloatArray,
    damping: float,
) -> tuple[FloatArray, FloatArray]:
    variable_count = residual_jacobian.shape[1]
    constraint_count = constraint_jacobian.shape[0]
    hessian = residual_jacobian.T @ residual_jacobian
    hessian.flat[:: variable_count + 1] += damping
    gradient = residual_jacobian.T @ residuals

    if constraint_count == 0:
        step, *_ = lstsq(hessian, -gradient, lapack_driver="gelsy")
        return np.asarray(step, dtype=float), np.empty(0, dtype=float)

    kkt = np.block(
        [
            [hessian, constraint_jacobian.T],
            [constraint_jacobian, np.zeros((constraint_count, constraint_count))],
        ]
    )
    rhs = -np.concatenate([gradient, constraints])
    solution, *_ = lstsq(kkt, rhs, lapack_driver="gelsy")
    return (
        np.asarray(solution[:variable_count], dtype=float),
        np.asarray(solution[variable_count:], dtype=float),
    )


def solve_constrained_gauss_newton(
    problem: MultipleShootingProblem,
    initial_decision: ArrayLike,
    *,
    options: SolverOptions | None = None,
) -> SolverResult:
    """Solve the direct multiple-shooting least-squares problem."""

    config = options or SolverOptions()
    decision = np.asarray(initial_decision, dtype=float).copy()
    if decision.shape != (problem.decision_size,):
        raise ValueError(f"initial_decision must have shape {(problem.decision_size,)}")
    damping = config.initial_damping
    history: list[SolverIteration] = []
    evaluation = problem.evaluate(decision)

    p = problem.model.parameter_dim
    history.append(
        SolverIteration(
            iteration=0,
            objective=evaluation.objective,
            constraint_norm=float(np.linalg.norm(evaluation.constraints, ord=np.inf))
            if evaluation.constraints.size
            else 0.0,
            merit=_merit(evaluation, config.constraint_penalty),
            damping=damping,
            accepted_step_fraction=0.0,
            step_norm=0.0,
            parameter_coordinates=decision[:p].copy(),
        )
    )

    success = False
    message = "Maximum iterations reached"

    for iteration in range(1, config.max_iterations + 1):
        base_residuals = evaluation.residuals
        base_constraints = evaluation.constraints
        residual_jacobian = finite_difference_jacobian(
            lambda z: problem.evaluate(z).residuals,
            decision,
            relative_step=config.finite_difference_step,
        )
        constraint_jacobian = finite_difference_jacobian(
            lambda z: problem.evaluate(z).constraints,
            decision,
            relative_step=config.finite_difference_step,
        )

        accepted = False
        accepted_fraction = 0.0
        accepted_step = np.zeros_like(decision)
        trial_evaluation = evaluation
        trial_decision = decision
        current_merit = _merit(evaluation, config.constraint_penalty)
        last_multiplier = np.empty(0, dtype=float)

        for _damping_trial in range(config.max_damping_trials):
            step, multiplier = _solve_kkt_step(
                base_residuals,
                base_constraints,
                residual_jacobian,
                constraint_jacobian,
                damping,
            )
            last_multiplier = multiplier
            if not np.all(np.isfinite(step)):
                damping = min(config.maximum_damping, damping * 10.0)
                continue

            fraction = 1.0
            for _line_search in range(config.max_line_search_steps):
                candidate = decision + fraction * step
                try:
                    candidate_evaluation = problem.evaluate(candidate)
                except (RuntimeError, ValueError, FloatingPointError):
                    fraction *= 0.5
                    if fraction < config.minimum_step_fraction:
                        break
                    continue
                candidate_merit = _merit(candidate_evaluation, config.constraint_penalty)
                if candidate_merit < current_merit - 1e-14 * max(1.0, current_merit):
                    accepted = True
                    accepted_fraction = fraction
                    accepted_step = fraction * step
                    trial_decision = candidate
                    trial_evaluation = candidate_evaluation
                    break
                fraction *= 0.5
                if fraction < config.minimum_step_fraction:
                    break

            if accepted:
                break
            damping = min(config.maximum_damping, damping * 10.0)

        if not accepted:
            constraint_norm = (
                float(np.linalg.norm(evaluation.constraints, ord=np.inf))
                if evaluation.constraints.size
                else 0.0
            )
            gradient = residual_jacobian.T @ base_residuals
            if constraint_jacobian.shape[0] and last_multiplier.size:
                gradient = gradient + constraint_jacobian.T @ last_multiplier
            if (
                np.linalg.norm(gradient, ord=np.inf) < config.gradient_tolerance
                and constraint_norm < config.constraint_tolerance
            ):
                success = True
                message = "First-order conditions satisfied"
            else:
                message = "No merit-decreasing damped Gauss-Newton step found"
            break

        decision = trial_decision
        evaluation = trial_evaluation
        step_norm = float(np.linalg.norm(accepted_step, ord=np.inf))
        constraint_norm = (
            float(np.linalg.norm(evaluation.constraints, ord=np.inf))
            if evaluation.constraints.size
            else 0.0
        )
        history.append(
            SolverIteration(
                iteration=iteration,
                objective=evaluation.objective,
                constraint_norm=constraint_norm,
                merit=_merit(evaluation, config.constraint_penalty),
                damping=damping,
                accepted_step_fraction=accepted_fraction,
                step_norm=step_norm,
                parameter_coordinates=decision[:p].copy(),
            )
        )

        if accepted_fraction == 1.0:
            damping = max(config.minimum_damping, damping * 0.3)

        scaled_step_tolerance = config.step_tolerance * (
            1.0 + float(np.linalg.norm(decision, ord=np.inf))
        )
        if step_norm < scaled_step_tolerance and constraint_norm < config.constraint_tolerance:
            success = True
            message = "Step and continuity tolerances satisfied"
            break

    return SolverResult(
        decision=decision,
        evaluation=evaluation,
        success=success,
        message=message,
        history=tuple(history),
    )
