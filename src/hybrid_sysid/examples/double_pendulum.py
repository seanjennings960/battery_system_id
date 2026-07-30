"""Hybrid double-pendulum example with fast damping and slow wear."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model import FloatArray, HybridODEModel, Segment
from ..multiple_shooting import MultipleShootingProblem
from ..solver import SolverOptions, SolverResult, solve_constrained_gauss_newton


@dataclass(frozen=True, slots=True)
class SyntheticDoublePendulumExperiment:
    model: HybridODEModel
    segments: tuple[Segment, ...]
    problem: MultipleShootingProblem
    true_parameters: FloatArray
    true_parameter_coordinates: FloatArray
    true_shooting_states: FloatArray
    initial_decision: FloatArray


def _input_torque(time: float, segment: Segment) -> FloatArray:
    local_time = time - segment.t0
    phase = float(segment.context.get("phase", 0.0))
    amplitude = float(segment.context.get("amplitude", 1.0))
    return amplitude * np.array(
        [
            0.75 * np.sin(2.2 * local_time + phase)
            + 0.22 * np.sin(5.4 * local_time + 0.4),
            0.48 * np.cos(1.6 * local_time + 0.7 * phase)
            - 0.16 * np.sin(4.1 * local_time),
        ]
    )


def make_double_pendulum_model() -> HybridODEModel:
    gravity = 9.81
    mass_1, mass_2 = 1.0, 0.8
    length_1, length_2 = 0.85, 0.72
    wear_gain = 8.0

    def rhs(time: float, state: FloatArray, theta: FloatArray, segment: Segment) -> FloatArray:
        q1, q2, w1, w2, wear = state
        damping_1, damping_2, wear_rate = theta
        delta = q1 - q2
        mass_matrix = np.array(
            [
                [
                    (mass_1 + mass_2) * length_1**2,
                    mass_2 * length_1 * length_2 * np.cos(delta),
                ],
                [
                    mass_2 * length_1 * length_2 * np.cos(delta),
                    mass_2 * length_2**2,
                ],
            ]
        )
        coriolis = np.array(
            [
                mass_2 * length_1 * length_2 * np.sin(delta) * w2**2,
                -mass_2 * length_1 * length_2 * np.sin(delta) * w1**2,
            ]
        )
        gravity_vector = np.array(
            [
                (mass_1 + mass_2) * gravity * length_1 * np.sin(q1),
                mass_2 * gravity * length_2 * np.sin(q2),
            ]
        )
        degradation_multiplier = 1.0 + wear_gain * max(wear, 0.0)
        damping = degradation_multiplier * np.array(
            [damping_1 * w1, damping_2 * w2]
        )

        if segment.mode == "excited":
            torque = _input_torque(time, segment)
            wear_drive = 0.05 + 0.22 * (w1**2 + w2**2) + 0.015 * float(torque @ torque)
        elif segment.mode == "rest":
            torque = np.zeros(2)
            damping = damping + np.array([0.75 * w1, 0.65 * w2])
            wear_drive = 0.35 + 0.01 * (w1**2 + w2**2)
        else:
            raise ValueError(f"Unknown double-pendulum mode: {segment.mode}")

        acceleration = np.linalg.solve(
            mass_matrix, torque - coriolis - gravity_vector - damping
        )
        return np.array(
            [w1, w2, acceleration[0], acceleration[1], wear_rate * wear_drive]
        )

    def observe(
        _time: float,
        state: FloatArray,
        _theta: FloatArray,
        _segment: Segment,
    ) -> FloatArray:
        return state[[0, 1, 4]]

    return HybridODEModel(
        state_dim=5,
        parameter_dim=3,
        rhs=rhs,
        observe=observe,
        parameter_transform=np.exp,
        rtol=3e-8,
        atol=3e-10,
        method="DOP853",
    )


def _make_skeleton_segments() -> tuple[Segment, ...]:
    return (
        Segment.empty(
            t0=0.0,
            t1=2.0,
            mode="excited",
            output_dim=3,
            context={"phase": 0.0, "amplitude": 1.0},
        ),
        Segment.empty(t0=2.0, t1=8.0, mode="rest", output_dim=3),
        Segment.empty(
            t0=8.0,
            t1=10.0,
            mode="excited",
            output_dim=3,
            context={"phase": 0.9, "amplitude": 0.85},
        ),
    )


def generate_synthetic_double_pendulum(
    *, seed: int = 0, noise_scale: float = 1.0
) -> SyntheticDoublePendulumExperiment:
    if noise_scale < 0.0:
        raise ValueError("noise_scale must be non-negative")
    rng = np.random.default_rng(seed)
    model = make_double_pendulum_model()
    skeleton = _make_skeleton_segments()
    true_parameters = np.array([0.12, 0.085, 0.015])
    true_eta = np.log(true_parameters)
    initial_state = np.array([0.52, -0.34, 0.03, -0.02, 0.018])

    true_starts: list[FloatArray] = []
    observed_segments: list[Segment] = []
    state = initial_state.copy()

    for segment in skeleton:
        true_starts.append(state.copy())
        if segment.mode == "excited":
            times = np.linspace(segment.t0, segment.t1, 33)
            states = model.flow(segment, state, true_parameters, times)
            clean = np.vstack(
                [
                    model.output(segment, time, sample, true_parameters)
                    for time, sample in zip(times, states, strict=True)
                ]
            )
            sigma = np.empty_like(clean)
            sigma[:, :2] = 0.004
            sigma[:, 2] = 0.35
            sigma[[0, -1], 2] = 0.0025
            noisy = clean + noise_scale * sigma * rng.normal(size=clean.shape)
            observed_segments.append(
                Segment(
                    t0=segment.t0,
                    t1=segment.t1,
                    mode=segment.mode,
                    observation_times=times,
                    observations=noisy,
                    observation_sigma=sigma,
                    context=segment.context,
                )
            )
        else:
            observed_segments.append(segment)
        state = model.flow(
            segment, state, true_parameters, np.array([segment.t1])
        )[-1]

    true_shooting_states = np.vstack(true_starts)
    segments = tuple(observed_segments)
    problem = MultipleShootingProblem(
        model=model,
        segments=segments,
        parameter_prior=np.log(np.array([0.17, 0.055, 0.009])),
        parameter_prior_sigma=np.array([1.6, 1.6, 1.8]),
        initial_state_prior=initial_state,
        initial_state_prior_sigma=np.array([0.004, 0.004, 0.04, 0.04, 0.0025]),
        continuity_scale=np.array([0.15, 0.15, 1.0, 1.0, 0.03]),
    )
    parameter_guess = np.log(np.array([0.18, 0.052, 0.0085]))
    perturbation = np.array(
        [
            [0.018, -0.014, 0.045, -0.035, 0.0015],
            [-0.012, 0.015, -0.05, 0.04, -0.0020],
            [0.014, -0.012, 0.035, -0.03, 0.0025],
        ]
    )
    initial_decision = problem.pack(parameter_guess, true_shooting_states + perturbation)
    return SyntheticDoublePendulumExperiment(
        model=model,
        segments=segments,
        problem=problem,
        true_parameters=true_parameters,
        true_parameter_coordinates=true_eta,
        true_shooting_states=true_shooting_states,
        initial_decision=initial_decision,
    )


def fit_synthetic_double_pendulum(
    experiment: SyntheticDoublePendulumExperiment,
) -> SolverResult:
    return solve_constrained_gauss_newton(
        experiment.problem,
        experiment.initial_decision,
        options=SolverOptions(
            max_iterations=24,
            finite_difference_step=2e-6,
            initial_damping=3e-4,
            constraint_penalty=2e5,
            constraint_tolerance=1e-6,
            step_tolerance=3e-8,
            gradient_tolerance=2e-6,
        ),
    )
