"""Model and data contracts for mode-aware ODE system identification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

FloatArray = NDArray[np.float64]
Rhs = Callable[[float, FloatArray, FloatArray, "Segment"], ArrayLike]
ObservationMap = Callable[[float, FloatArray, FloatArray, "Segment"], ArrayLike]
ParameterTransform = Callable[[FloatArray], ArrayLike]
ResetMap = Callable[[float, FloatArray, FloatArray, "Segment", "Segment"], ArrayLike]
CustomFlow = Callable[["Segment", FloatArray, FloatArray, FloatArray], ArrayLike]


def _identity_parameters(eta: FloatArray) -> FloatArray:
    return eta.copy()


def _identity_reset(
    _time: float,
    state: FloatArray,
    _theta: FloatArray,
    _from_segment: "Segment",
    _to_segment: "Segment",
) -> FloatArray:
    return state.copy()


@dataclass(frozen=True, slots=True)
class Segment:
    """One known dynamical mode on a contiguous time interval.

    Measurement times may be empty. This is how unobserved rest periods are
    represented: the state still propagates through the interval, but the
    interval contributes no measurement residuals.
    """

    t0: float
    t1: float
    mode: str
    observation_times: FloatArray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    observations: FloatArray = field(
        default_factory=lambda: np.empty((0, 0), dtype=float)
    )
    observation_sigma: FloatArray = field(
        default_factory=lambda: np.empty((0, 0), dtype=float)
    )
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        times = np.asarray(self.observation_times, dtype=float)
        values = np.asarray(self.observations, dtype=float)
        sigma = np.asarray(self.observation_sigma, dtype=float)

        if not np.isfinite(self.t0) or not np.isfinite(self.t1) or self.t1 <= self.t0:
            raise ValueError("Segment requires finite times with t1 > t0")
        if times.ndim != 1:
            raise ValueError("observation_times must be one-dimensional")
        if values.ndim != 2 or sigma.ndim != 2:
            raise ValueError("observations and observation_sigma must be two-dimensional")
        if times.shape[0] != values.shape[0] or times.shape[0] != sigma.shape[0]:
            raise ValueError(
                "observation_times, observations, and observation_sigma must have "
                "the same leading dimension"
            )
        if values.shape != sigma.shape:
            raise ValueError("observations and observation_sigma must have equal shapes")
        if times.size and (np.any(times < self.t0) or np.any(times > self.t1)):
            raise ValueError("All observation times must lie inside the segment")
        if times.size > 1 and np.any(np.diff(times) <= 0.0):
            raise ValueError("observation_times must be strictly increasing")
        if sigma.size and (not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0)):
            raise ValueError("observation_sigma must contain finite positive values")
        if values.size and not np.all(np.isfinite(values):
            raise ValueError("observations must contain finite values")

        object.__setattr__(self, "observation_times", times)
        object.__setattr__(self, "observations", values)
        object.__setattr__(self, "observation_sigma", sigma)

    @classmethod
    def empty(
        cls,
        *,
        t0: float,
        t1: float,
        mode: str,
        output_dim: int = 0,
        context: Mapping[str, Any] | None = None,
    ) -> "Segment":
        """Construct a measurement-free segment, such as a storage/rest gap."""

        return cls(
            t0=t0,
            t1=t1,
            mode=mode,
            observation_times=np.empty(0, dtype=float),
            observations=np.empty((0, output_dim), dtype=float),
            observation_sigma=np.empty((0, output_dim), dtype=float),
            context={} if context is None else context,
        )


@dataclass(slots=True)
class HybridODEModel:
    """A mode-aware ODE with a pluggable propagator and reset map."""

    state_dim: int
    parameter_dim: int
    rhs: Rhs
    observe: ObservationMap
    parameter_transform: ParameterTransform = _identity_parameters
    reset: ResetMap = _identity_reset
    custom_flow: CustomFlow | None = None
    rtol: float = 1e-9
    atol: float = 1e-11
    method: str = "DOP853"

    def physical_parameters(self, eta: ArrayLike) -> FloatArray:
        eta_array = np.asarray(eta, dtype=float)
        if eta_array.shape != (self.parameter_dim,):
            raise ValueError(
                f"Expected {self.parameter_dim} unconstrained parameters; "
                f"got shape {eta_array.shape}"
            )
        theta = np.asarray(self.parameter_transform(eta_array), dtype=float)
        if theta.shape != (self.parameter_dim,):
            raise ValueError("parameter_transform returned the wrong shape")
        if not np.all(np.isfinite(theta)):
            raise ValueError("parameter_transform returned non-finite values")
        return theta

    def flow(
        self,
        segment: Segment,
        initial_state: ArrayLike,
        theta: ArrayLike,
        evaluation_times: ArrayLike,
    ) -> FloatArray:
        """Propagate a state through one segment at requested absolute times."""

        state0 = np.asarray(initial_state, dtype=float)
        parameters = np.asarray(theta, dtype=float)
        times = np.asarray(evaluation_times, dtype=float)
        if state0.shape != (self.state_dim,):
            raise ValueError(f"initial_state must have shape {(self.state_dim,)}")
        if parameters.shape != (self.parameter_dim,):
            raise ValueError(f"theta must have shape {(self.parameter_dim,)}")
        if times.ndim != 1:
            raise ValueError("evaluation_times must be one-dimensional")
        if times.size == 0:
            return np.empty((0, self.state_dim), dtype=float)
        if np.any(times < segment.t0) or np.any(times > segment.t1):
            raise ValueError("evaluation_times must lie inside the segment")
        if times.size > 1 and np.any(np.diff(times) < 0.0):
            raise ValueError("evaluation_times must be nondecreasing")

        if self.custom_flow is not None:
            result = np.asarray(
                self.custom_flow(segment, state0, parameters, times), dtype=float
            )
            expected_shape = (times.size, self.state_dim)
            if result.shape != expected_shape:
                raise ValueError(
                    f"custom_flow returned shape {result.shape}; expected {expected_shape}"
                )
            return result

        unique_times, inverse = np.unique(times, return_inverse=True)
        requested_t0 = unique_times[0] == segment.t0
        solve_times = unique_times[1:] if requested_t0 else unique_times

        if solve_times.size:
            solution = solve_ivp(
                fun=lambda t, x: np.asarray(
                    self.rhs(t, x, parameters, segment), dtype=float
                ),
                t_span=(segment.t0, segment.t1),
                y0=state0,
                t_eval=solve_times,
                rtol=self.rtol,
                atol=self.atol,
                method=self.method,
            )
            if not solution.success:
                raise RuntimeError(f"ODE integration failed: {solution.message}")
            propagated = solution.y.T
        else:
            propagated = np.empty((0, self.state_dim), dtype=float)

        unique_states = np.vstack([state0, propagated]) if requested_t0 else propagated
        return unique_states[inverse]

    def output(
        self,
        segment: Segment,
        time: float,
        state: ArrayLike,
        theta: ArrayLike,
    ) -> FloatArray:
        value = np.asarray(
            self.observe(
                float(time),
                np.asarray(state, dtype=float),
                np.asarray(theta, dtype=float),
                segment,
            ),
            dtype=float,
        )
        if value.ndim != 1:
            raise ValueError("observe must return a one-dimensional vector")
        return value
