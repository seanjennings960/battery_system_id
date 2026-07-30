from __future__ import annotations

import numpy as np

from hybrid_sysid.battery_template import default_battery_parameters, make_battery_template_model
from hybrid_sysid.examples.double_pendulum import (
    fit_synthetic_double_pendulum,
    generate_synthetic_double_pendulum,
)
from hybrid_sysid.model import HybridODEModel, Segment
from hybrid_sysid.multiple_shooting import MultipleShootingProblem
from hybrid_sysid.solver import SolverOptions, solve_constrained_gauss_newton
from hybrid_sysid.streaming import GaussNewtonAccumulator


def test_unobserved_rest_segment_still_enforces_state_continuity() -> None:
    model = HybridODEModel(
        state_dim=1,
        parameter_dim=1,
        rhs=lambda _t, x, theta, _segment: -theta[0] * x,
        observe=lambda _t, x, _theta, _segment: x,
        parameter_transform=np.exp,
    )
    segments = [
        Segment.empty(t0=0.0, t1=1.0, mode="active"),
        Segment.empty(t0=1.0, t1=3.0, mode="rest"),
    ]
    problem = MultipleShootingProblem(model=model, segments=segments)
    theta = 0.5
    states = np.array([[1.0], [np.exp(-theta)]])
    evaluation = problem.evaluate(problem.pack(np.log([theta]), states))
    np.testing.assert_allclose(evaluation.constraints, 0.0, atol=1e-8)
    assert evaluation.residuals.size == 0


def test_constrained_solver_recovers_decay_rate_across_gap() -> None:
    true_theta = 0.65
    true_x0 = 1.8
    model = HybridODEModel(
        state_dim=1,
        parameter_dim=1,
        rhs=lambda _t, x, theta, _segment: -theta[0] * x,
        observe=lambda _t, x, _theta, _segment: x,
        parameter_transform=np.exp,
    )

    def observed(t0: float, t1: float, x0: float) -> Segment:
        times = np.linspace(t0, t1, 5)
        values = x0 * np.exp(-true_theta * (times - t0))
        return Segment(
            t0=t0,
            t1=t1,
            mode="active",
            observation_times=times,
            observations=values[:, None],
            observation_sigma=np.full((times.size, 1), 0.01),
        )

    x1 = true_x0 * np.exp(-true_theta)
    x4 = true_x0 * np.exp(-4.0 * true_theta)
    problem = MultipleShootingProblem(
        model=model,
        segments=[
            observed(0.0, 1.0, true_x0),
            Segment.empty(t0=1.0, t1=4.0, mode="rest", output_dim=1),
            observed(4.0, 5.0, x4),
        ],
        initial_state_prior=np.array([true_x0]),
        initial_state_prior_sigma=np.array([1e-3]),
    )
    initial = problem.pack(np.log([0.25]), np.array([[1.75], [0.95 * x1], [1.2 * x4]]))
    result = solve_constrained_gauss_newton(
        problem,
        initial,
        options=SolverOptions(max_iterations=30, constraint_penalty=1e5),
    )
    estimate = model.physical_parameters(problem.unpack(result.decision)[0])[0]
    assert result.success, result.message
    assert abs(estimate - true_theta) < 3e-4


def test_streamed_normal_equations_match_batch() -> None:
    rng = np.random.default_rng(4)
    blocks = [(rng.normal(size=7), rng.normal(size=(7, 5))), (rng.normal(size=11), rng.normal(size=(11, 5)))]
    accumulator = GaussNewtonAccumulator(variable_dim=5)
    for residuals, jacobian in blocks:
        accumulator.add_block(residuals, jacobian)
    residuals = np.concatenate([block[0] for block in blocks])
    jacobian = np.vstack([block[1] for block in blocks])
    np.testing.assert_allclose(accumulator.hessian, jacobian.T @ jacobian)
    np.testing.assert_allclose(accumulator.gradient, jacobian.T @ residuals)


def test_battery_rest_relaxes_fast_states_but_ages_slow_states() -> None:
    model = make_battery_template_model()
    theta = default_battery_parameters()
    state0 = np.array([0.62, 0.08, -0.04, 0.02, 0.10])
    rest = Segment.empty(
        t0=0.0,
        t1=3600.0,
        mode="rest",
        output_dim=1,
        context={"temperature_c": 25.0, "current_a": 0.0},
    )
    final = model.flow(rest, state0, theta, np.array([rest.t1]))[-1]
    assert np.isclose(final[0], state0[0])
    assert abs(final[1]) < abs(state0[1]) * 1e-4
    assert abs(final[2]) < abs(state0[2]) * 1e-4
    assert final[3] > state0[3]
    assert final[4] > state0[4]


def test_double_pendulum_joint_fit_recovers_fast_and_slow_parameters() -> None:
    experiment = generate_synthetic_double_pendulum(seed=12, noise_scale=0.35)
    result = fit_synthetic_double_pendulum(experiment)
    estimated = experiment.model.physical_parameters(
        experiment.problem.unpack(result.decision)[0]
    )
    relative_error = np.abs((estimated - experiment.true_parameters) / experiment.true_parameters)
    assert result.success, result.message
    assert relative_error[0] < 0.08
    assert relative_error[1] < 0.10
    assert relative_error[2] < 0.12
