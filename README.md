# Battery System Identification

A reference implementation of hybrid, multiscale ODE-constrained parameter estimation using direct multiple shooting and equality-constrained generalized Gauss–Newton steps.

## Problem formulation

For known modes `m_k`, each segment follows

```text
x_dot = f_m(t, x, u, theta)
y     = h_m(t, x, u, theta) + noise
```

A separate shooting state is introduced at every experimental or rest segment. Measurement-free rest intervals contribute no observation residuals, but their state propagation remains coupled to subsequent experiments through continuity constraints.

The resulting nonlinear least-squares problem is

```text
minimize    1/2 ||r(eta, s)||^2
subject to  c(eta, s) = 0
```

where `eta` are unconstrained parameter coordinates and `s` contains segment initial states. The reference solver linearizes residuals and continuity constraints, solves the generalized Gauss–Newton KKT system, and globalizes the step with damping and a merit-function line search.

## Repository contents

- `src/hybrid_sysid/model.py`: hybrid model and segment contracts.
- `src/hybrid_sysid/multiple_shooting.py`: all-at-once multiple-shooting transcription.
- `src/hybrid_sysid/solver.py`: constrained generalized Gauss–Newton/SQP solver.
- `src/hybrid_sysid/streaming.py`: chunk-wise normal-equation accumulation for large datasets.
- `src/hybrid_sysid/examples/double_pendulum.py`: synthetic fast-damping/slow-wear example with an unobserved rest gap.
- `src/hybrid_sysid/battery_template.py`: 2-RC battery model coupled to capacity and resistance degradation, including exact rest propagation.
- `tests/`: integrated behavioral tests.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

## Scaling path for the battery dataset

The dense finite-difference solver is intentionally transparent rather than production-scale. The intended next steps are sparse tangent or adjoint sensitivities, segment-parallel propagation, sparse KKT or condensed Schur solves, and streaming accumulation of

```text
H += J_b.T @ J_b
g += J_b.T @ r_b
F += 0.5 * r_b.T @ r_b
```

for each on-disk row group. This retains every data point in the objective without retaining the full decoded dataset or Jacobian in memory.

## Review workflow

Generated changes are developed on `agent/*` branches and opened as draft pull requests. Review comments can then be read as unresolved threads, implemented in follow-up commits, tested, and pushed back to the same PR for another review round.
