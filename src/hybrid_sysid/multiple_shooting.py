"""Direct multiple-shooting transcription for nonlinear hybrid ODE models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike

from .model import FloatArray, HybridODEModel, Segment


@dataclass(frozen=True, slots=True)
class ProblemEvaluation:
    """Residual and constraint vectors at one decision point."""

    residuals: FloatArray
    constraints: FloatArray
    endpoint_states: FloatArray

    @property
    def objective(self) -> float:
        return 0.5 * float(self.residuals @ self.residuals)


@dataclass(slots=True)
class MultipleShootingProblem:
    """All-at-once least-squares transcription of a hybrid ODE ID problem.

    Decision ordering is ``[eta, s_0, ..., s_(K-1)]`` where ``eta`` are
    unconstrained parameter coordinates and each ``s_k`` is the independently
    optimized initial state for segment ``k``.
    """

    model: HybridODEModel
    segments: Sequence[Segment]
    parameter_prior: FloatArray | None = None
    parameter_prior_sigma: FloatArray | None = None
    initial_state_prior: FloatArray | None = None
    initial_state_prior_sigma: FloatArray | None = None
    continuity_scale: FloatArray | None = None

    def __post_init__(self) -> None:
        self.segments = tuple(self.segments)
        if not self.segments:
            raise ValueError("At least one segment is required")
        for left, right in zip(self.segments[:-1], self.segments[1:], strict=True):
            if not np.isclose(left.t1, right.t0, rtol=0.0, atol=1e-12):
                raise ValueError("Segments must form a contiguous time partition")

        self.parameter_prior = self._optional_vector(
            self.parameter_prior, self.model.parameter_dim, "parameter_prior"
        )
        self.parameter_prior_sigma = self._optional_positive_vector(
            self.parameter_prior_sigma,
            self.model.parameter_dim,
            "parameter_prior_sigma",
        )
        self.initial_state_prior = self._optional_vector(
            self.initial_state_prior, self.model.state_dim, "initial_state_prior"
        )
        self.initial_state_prior_sigma = self._optional_positive_vector(
            self.initial_state_prior_sigma,
            self.model.state_dim,
            "initial_state_prior_sigma",
        )
        if (self.parameter_prior is None) != (self.parameter_prior_sigma is None):
            raise ValueError("parameter prior and sigma must be provided together")
        if (self.initial_state_prior is None) != (self.initial_state_prior_sigma is None):
            raise ValueError("initial-state prior and sigma must be provided together")

        if self.continuity_scale is None:
            self.continuity_scale = np.ones(self.model.state_dim, dtype=float)
        else:
            self.continuity_scale = self._optional_positive_vector(
                self.continuity_scale,
                self.model.state_dim,
                "continuity_scale",
            )
            assert self.continuity_scale is not None

    @staticmethod
    def _optional_vector(
        value: ArrayLike | None, length: int, name: str
    ) -> FloatArray | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=float)
        if array.shape != (length,):
            raise ValueError(f"{name} must have shape {(length,)}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        return array

    @staticmethod
    def _optional_positive_vector(
        value: ArrayLike | None, length: int, name: str
    ) -> FloatArray | None:
        array = MultipleShootingProblem._optional_vector(value, length, name)
        if array is not None and np.any(array <= 0.0):
            raise ValueError(f"{name} must be positive")
        return array

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    @property
    def decision_size(self) -> int:
        return self.model.parameter_dim + self.segment_count * self.model.state_dim

    @property
    def constraint_size(self) -> int:
        return (self.segment_count - 1) * self.model.state_dim

    def pack(self, eta: ArrayLike, shooting_states: ArrayLike) -> FloatArray:
        eta_array = np.asarray(eta, dtype=float)
        states = np.asarray(shooting_states, dtype=float)
        expected_states = (self.segment_count, self.model.state_dim)
        if eta_array.shape != (self.model.parameter_dim,):
            raise ValueError(
                f"eta must have shape {(self.model.parameter_dim,)}; got {eta_array.shape}"
            )
        if states.shape != expected_states:
            raise ValueError(
                f"shooting_states must have shape {expected_states}; got {states.shape}"
            )
        if not np.all(np.isfinite(eta_array)) or not np.all(np.isfinite(states)):
            raise ValueError("Decision variables must be finite")
        return np.concatenate([eta_array, states.ravel()])

    def unpack(self, decision: ArrayLike) -> tuple[FloatArray, FloatArray]:
        decision_array = np.asarray(decision, dtype=float)
        if decision_array.shape != (self.decision_size,):
            raise ValueError(
                f"decision must have shape {(self.decision_size,)}; got {decision_array.shape}"
            )
        p = self.model.parameter_dim
        eta = decision_array[:p].copy()
        states = decision_array[p:].reshape(self.segment_count, self.model.state_dim).copy()
        return eta, states

    def evaluate(self, decision: ArrayLike) -> ProblemEvaluation:
        eta, shooting_states = self.unpack(decision)
        theta = self.model.physical_parameters(eta)
        residual_blocks: list[FloatArray] = []
        endpoint_states = np.empty((self.segment_count, self.model.state_dim), dtype=float)

        for index, (segment, state0) in enumerate(
            zip(self.segments, shooting_states, strict=True)
        ):
            all_times = np.concatenate(
                [segment.observation_times, np.array([segment.t1], dtype=float)]
            )
            states = self.model.flow(segment, state0, theta, all_times)
            endpoint_states[index] = states[-1]
            if segment.observation_times.size:
                observed_states = states[:-1]
                predictions = np.vstack(
                    [
                        self.model.output(segment, time, state, theta)
                        for time, state in zip(
                            segment.observation_times, observed_states, strict=True
                        )
                    ]
                )
                if predictions.shape != segment.observations.shape:
                    raise ValueError(
                        f"Observation map returned shape {predictions.shape} for "
                        f"segment {index}; data have shape {segment.observations.shape}"
                    )
                residual_blocks.append(
                    ((predictions - segment.observations) / segment.observation_sigma).ravel()
                )

        if self.parameter_prior is not None:
            assert self.parameter_prior_sigma is not None
            residual_blocks.append((eta - self.parameter_prior) / self.parameter_prior_sigma)
        if self.initial_state_prior is not None:
            assert self.initial_state_prior_sigma is not None
            residual_blocks.append(
                (shooting_states[0] - self.initial_state_prior)
                / self.initial_state_prior_sigma
            )

        residuals = np.concatenate(residual_blocks) if residual_blocks else np.empty(0)

        continuity_blocks: list[FloatArray] = []
        assert self.continuity_scale is not None
        for index in range(self.segment_count - 1):
            from_segment = self.segments[index]
            to_segment = self.segments[index + 1]
            mapped = np.asarray(
                self.model.reset(
                    from_segment.t1,
                    endpoint_states[index],
                    theta,
                    from_segment,
                    to_segment,
                ),
                dtype=float,
            )
            if mapped.shape != (self.model.state_dim,):
                raise ValueError("reset map returned the wrong state shape")
            continuity_blocks.append(
                (shooting_states[index + 1] - mapped) / self.continuity_scale
            )

        constraints = (
            np.concatenate(continuity_blocks)
            if continuity_blocks
            else np.empty(0, dtype=float)
        )
        return ProblemEvaluation(
            residuals=residuals,
            constraints=constraints,
            endpoint_states=endpoint_states,
        )
