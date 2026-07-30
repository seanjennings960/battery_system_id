"""Battery-scale template for hybrid, multirate system identification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp

from .model import FloatArray, HybridODEModel, Segment

BATTERY_STATE_NAMES = (
    "soc",
    "fast_polarization_v",
    "slow_polarization_v",
    "capacity_loss_fraction",
    "resistance_growth_fraction",
)
BATTERY_PARAMETER_NAMES = (
    "ohmic_resistance_ohm",
    "fast_resistance_ohm",
    "fast_capacitance_f",
    "slow_resistance_ohm",
    "slow_capacitance_f",
    "nominal_capacity_ah",
    "calendar_capacity_rate_per_s",
    "cycle_capacity_rate_per_s",
    "calendar_resistance_rate_per_s",
    "cycle_resistance_rate_per_s",
)


@dataclass(frozen=True, slots=True)
class BatteryDataShard:
    path: Path
    t0: float
    t1: float
    mode: Literal["active", "rest", "checkup"]
    cell_id: str
    protocol_id: str
    format: Literal["parquet", "zarr", "hdf5"] = "parquet"


@dataclass(frozen=True, slots=True)
class BatteryScalePlan:
    chunk_rows: int = 250_000
    derivative_backend: Literal["tangent", "adjoint"] = "adjoint"
    linear_solver: Literal["sparse_kkt", "condensed_schur"] = "condensed_schur"
    parallelize_over_segments: bool = True
    keep_all_samples_in_objective: bool = True


def default_battery_parameters() -> FloatArray:
    return np.array(
        [0.018, 0.012, 450.0, 0.025, 4_000.0, 5.0, 2e-9, 1.5e-8, 4e-9, 2.5e-8],
        dtype=float,
    )


def _temperature_factor(temperature_c: float) -> float:
    temperature_k = temperature_c + 273.15
    exponent = 3_500.0 * (1.0 / 298.15 - 1.0 / temperature_k)
    return float(np.exp(np.clip(exponent, -8.0, 8.0)))


def _current(segment: Segment, time: float) -> float:
    value = segment.context.get("current_a", 0.0)
    return float(value(time)) if callable(value) else float(value)


def _open_circuit_voltage(soc: float) -> float:
    clipped = float(np.clip(soc, 0.0, 1.0))
    return 3.0 + 1.05 * clipped + 0.08 * np.tanh(8.0 * (clipped - 0.5))


def make_battery_template_model() -> HybridODEModel:
    """Create a 2-RC ECM coupled to slow capacity and resistance degradation."""

    def rhs(time: float, state: FloatArray, theta: FloatArray, segment: Segment) -> FloatArray:
        (
            _r0,
            r_fast,
            c_fast,
            r_slow,
            c_slow,
            capacity_ah,
            k_cap_calendar,
            k_cap_cycle,
            k_res_calendar,
            k_res_cycle,
        ) = theta
        soc, v_fast, v_slow, capacity_loss, _resistance_growth = state
        current_a = _current(segment, time)
        temperature_factor = _temperature_factor(
            float(segment.context.get("temperature_c", 25.0))
        )
        remaining_capacity_ah = capacity_ah * max(0.15, 1.0 - capacity_loss)
        c_rate = abs(current_a) / max(capacity_ah, 1e-12)
        return np.array(
            [
                -current_a / (3_600.0 * remaining_capacity_ah),
                -v_fast / (r_fast * c_fast) + current_a / c_fast,
                -v_slow / (r_slow * c_slow) + current_a / c_slow,
                temperature_factor * (k_cap_calendar + k_cap_cycle * c_rate**1.25),
                temperature_factor * (k_res_calendar + k_res_cycle * c_rate**1.15),
            ]
        )

    def observe(time: float, state: FloatArray, theta: FloatArray, segment: Segment) -> FloatArray:
        r0 = theta[0]
        soc, v_fast, v_slow, _capacity_loss, resistance_growth = state
        current_a = _current(segment, time)
        voltage = (
            _open_circuit_voltage(soc)
            - current_a * r0 * (1.0 + resistance_growth)
            - v_fast
            - v_slow
        )
        kind = str(segment.context.get("observation_kind", "voltage"))
        if kind == "voltage":
            return np.array([voltage])
        if kind == "checkup":
            return np.array([voltage, 1.0 - state[3], r0 * (1.0 + resistance_growth)])
        raise ValueError(f"Unknown observation_kind: {kind}")

    def custom_flow(
        segment: Segment,
        initial_state: FloatArray,
        theta: FloatArray,
        times: FloatArray,
    ) -> FloatArray:
        current_value = segment.context.get("current_a", 0.0)
        zero_current = not callable(current_value) and np.isclose(float(current_value), 0.0)
        if segment.mode == "rest" and zero_current:
            (
                _r0,
                r_fast,
                c_fast,
                r_slow,
                c_slow,
                _capacity_ah,
                k_cap_calendar,
                _k_cap_cycle,
                k_res_calendar,
                _k_res_cycle,
            ) = theta
            factor = _temperature_factor(float(segment.context.get("temperature_c", 25.0)))
            elapsed = times - segment.t0
            result = np.empty((times.size, 5))
            result[:, 0] = initial_state[0]
            result[:, 1] = initial_state[1] * np.exp(-elapsed / (r_fast * c_fast))
            result[:, 2] = initial_state[2] * np.exp(-elapsed / (r_slow * c_slow))
            result[:, 3] = initial_state[3] + factor * k_cap_calendar * elapsed
            result[:, 4] = initial_state[4] + factor * k_res_calendar * elapsed
            return result

        unique_times, inverse = np.unique(times, return_inverse=True)
        includes_t0 = unique_times[0] == segment.t0
        solve_times = unique_times[1:] if includes_t0 else unique_times
        propagated = np.empty((0, 5))
        if solve_times.size:
            solution = solve_ivp(
                lambda time, state: rhs(time, state, theta, segment),
                (segment.t0, segment.t1),
                initial_state,
                t_eval=solve_times,
                method="BDF",
                rtol=2e-8,
                atol=2e-10,
            )
            if not solution.success:
                raise RuntimeError(f"Battery integration failed: {solution.message}")
            propagated = solution.y.T
        states = np.vstack([initial_state, propagated]) if includes_t0 else propagated
        return states[inverse]

    return HybridODEModel(
        state_dim=5,
        parameter_dim=len(BATTERY_PARAMETER_NAMES),
        rhs=rhs,
        observe=observe,
        parameter_transform=np.exp,
        custom_flow=custom_flow,
    )
