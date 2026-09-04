from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from femx.backends.jax.port_eigen_adjoint import (  # noqa: E402
    SimplePortEigenpairPolicy,
    inspect_simple_port_eigenpair,
    solve_simple_port_eigenpair,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]

_STIFFNESS = np.asarray(
    (
        (-4.0, 0.7, -0.2),
        (0.3, -2.2, 0.4),
        (0.0, -0.1, -0.7),
    ),
    dtype=np.float64,
)
_MASS = np.asarray(
    (
        (1.3, 0.1, 0.0),
        (0.1, 1.1, 0.05),
        (0.0, 0.05, 0.9),
    ),
    dtype=np.float64,
)
_STIFFNESS_DIRECTION = np.asarray(
    (
        (0.17, -0.08, 0.03),
        (0.04, -0.11, 0.09),
        (-0.02, 0.06, 0.05),
    ),
    dtype=np.float64,
)
_MASS_DIRECTION = np.asarray(
    (
        (0.03, -0.01, 0.0),
        (-0.01, -0.02, 0.005),
        (0.0, 0.005, 0.01),
    ),
    dtype=np.float64,
)
_SCALE = 2.0


def _pair(parameter: jax.Array):
    return solve_simple_port_eigenpair(
        jnp.asarray(_STIFFNESS) + parameter * jnp.asarray(_STIFFNESS_DIRECTION),
        jnp.asarray(_MASS) + parameter * jnp.asarray(_MASS_DIRECTION),
        jnp.asarray(_SCALE, dtype=jnp.float64),
        selected_mode_index=0,
        phase_anchor_edge_dof=0,
    )


def _objective(parameter: jax.Array) -> jax.Array:
    pair = _pair(parameter)
    weights = jnp.asarray((0.2, -0.35, 0.17), dtype=jnp.float64)
    return (
        0.1 * pair.eigenvalue_per_m2
        + 0.4 * pair.propagation_constant_per_m
        + weights @ pair.edge_coefficients
    )


def test_simple_nonsymmetric_port_pair_is_real_normalized_and_jittable() -> None:
    inspection = inspect_simple_port_eigenpair(
        jnp.asarray(_STIFFNESS),
        jnp.asarray(_MASS),
        jnp.asarray(_SCALE, dtype=jnp.float64),
        selected_mode_index=0,
        phase_anchor_edge_dof=0,
    )
    pair = jax.jit(_pair)(jnp.asarray(0.0, dtype=jnp.float64))

    assert bool(inspection.diagnostics.is_valid)
    assert float(inspection.diagnostics.relative_eigenvalue_gap) > 0.1
    assert float(inspection.diagnostics.relative_residual) < 1.0e-14
    assert float(inspection.diagnostics.mass_symmetry_error) < 1.0e-15
    assert float(inspection.diagnostics.phase_anchor_relative_magnitude) == pytest.approx(1.0)
    np.testing.assert_allclose(
        np.asarray(pair.edge_coefficients) @ _MASS @ np.asarray(pair.edge_coefficients),
        1.0,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        _STIFFNESS @ np.asarray(pair.edge_coefficients),
        float(pair.eigenvalue_per_m2) * _MASS @ np.asarray(pair.edge_coefficients),
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    assert float(pair.edge_coefficients[0]) > 0.0
    assert float(pair.propagation_constant_per_m) == pytest.approx(
        math.sqrt(-float(pair.eigenvalue_per_m2)),
        rel=2.0e-15,
    )


def test_bordered_reverse_mode_matches_independent_sensitivity_and_central_difference() -> None:
    parameter = jnp.asarray(0.0, dtype=jnp.float64)
    value, reverse_gradient = jax.jit(jax.value_and_grad(_objective))(parameter)
    pair = _pair(parameter)
    eigenvalue = float(pair.eigenvalue_per_m2)
    propagation_constant = float(pair.propagation_constant_per_m)
    edge = np.asarray(pair.edge_coefficients)

    jacobian = np.block(
        [
            [
                _STIFFNESS - eigenvalue * _MASS,
                -(_MASS @ edge)[:, None],
            ],
            [(edge @ _MASS)[None, :], np.zeros((1, 1))],
        ]
    )
    forward_right_hand_side = np.concatenate(
        (
            -(_STIFFNESS_DIRECTION - eigenvalue * _MASS_DIRECTION) @ edge,
            np.asarray((-0.5 * edge @ _MASS_DIRECTION @ edge,)),
        )
    )
    sensitivity = np.linalg.solve(jacobian, forward_right_hand_side)
    edge_sensitivity = sensitivity[:-1]
    eigenvalue_sensitivity = sensitivity[-1]
    beta_sensitivity = -eigenvalue_sensitivity / (2.0 * propagation_constant)
    independent_gradient = (
        0.1 * eigenvalue_sensitivity
        + 0.4 * beta_sensitivity
        + np.asarray((0.2, -0.35, 0.17)) @ edge_sensitivity
    )

    step = 2.0e-5
    central_difference = (
        float(_objective(jnp.asarray(step, dtype=jnp.float64)))
        - float(_objective(jnp.asarray(-step, dtype=jnp.float64)))
    ) / (2.0 * step)
    assert math.isfinite(float(value))
    assert float(reverse_gradient) == pytest.approx(independent_gradient, rel=2.0e-11, abs=2.0e-12)
    assert float(reverse_gradient) == pytest.approx(central_difference, rel=3.0e-9, abs=3.0e-10)


def test_degenerate_or_unstable_individual_mode_fails_closed() -> None:
    repeated = jnp.diag(jnp.asarray((-4.0, -4.0, -1.0), dtype=jnp.float64))
    mass = jnp.eye(3, dtype=jnp.float64)
    inspection = inspect_simple_port_eigenpair(
        repeated,
        mass,
        jnp.asarray(2.0, dtype=jnp.float64),
        selected_mode_index=0,
        phase_anchor_edge_dof=0,
    )
    pair = solve_simple_port_eigenpair(
        repeated,
        mass,
        jnp.asarray(2.0, dtype=jnp.float64),
        selected_mode_index=0,
        phase_anchor_edge_dof=0,
    )

    assert not bool(inspection.diagnostics.is_valid)
    assert float(inspection.diagnostics.relative_eigenvalue_gap) == 0.0
    assert math.isnan(float(pair.eigenvalue_per_m2))
    assert np.isnan(np.asarray(pair.edge_coefficients)).all()

    def objective(shift: jax.Array) -> jax.Array:
        shifted = repeated.at[0, 0].add(shift)
        return solve_simple_port_eigenpair(
            shifted,
            mass,
            jnp.asarray(2.0, dtype=jnp.float64),
            selected_mode_index=0,
            phase_anchor_edge_dof=0,
        ).propagation_constant_per_m

    assert math.isnan(float(jax.grad(objective)(jnp.asarray(0.0, dtype=jnp.float64))))


def test_complex_mode_nonsymmetric_mass_and_weak_anchor_are_rejected() -> None:
    rotation = jnp.asarray(((0.0, -1.0), (1.0, 0.0)), dtype=jnp.float64)
    identity = jnp.eye(2, dtype=jnp.float64)
    complex_mode = inspect_simple_port_eigenpair(
        rotation,
        identity,
        jnp.asarray(1.0, dtype=jnp.float64),
        selected_mode_index=0,
        phase_anchor_edge_dof=0,
    )
    assert not bool(complex_mode.diagnostics.is_valid)
    assert float(complex_mode.diagnostics.relative_eigenvalue_imaginary_part) > 0.0

    nonsymmetric_mass = inspect_simple_port_eigenpair(
        jnp.asarray(_STIFFNESS),
        jnp.asarray(_MASS).at[0, 1].add(0.2),
        jnp.asarray(_SCALE, dtype=jnp.float64),
        selected_mode_index=0,
        phase_anchor_edge_dof=0,
    )
    assert not bool(nonsymmetric_mass.diagnostics.is_valid)
    assert float(nonsymmetric_mass.diagnostics.mass_symmetry_error) > 1.0e-3

    weak_anchor = inspect_simple_port_eigenpair(
        jnp.asarray(_STIFFNESS),
        jnp.asarray(_MASS),
        jnp.asarray(_SCALE, dtype=jnp.float64),
        selected_mode_index=0,
        phase_anchor_edge_dof=2,
        policy=SimplePortEigenpairPolicy(minimum_phase_anchor_relative_magnitude=0.1),
    )
    assert not bool(weak_anchor.diagnostics.is_valid)
    assert float(weak_anchor.diagnostics.phase_anchor_relative_magnitude) < 0.1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"minimum_relative_eigenvalue_gap": 0.0}, "minimum eigenvalue gap"),
        ({"maximum_relative_residual": math.inf}, "maximum residual"),
        ({"minimum_phase_anchor_relative_magnitude": 1.1}, "cannot exceed one"),
    ],
)
def test_simple_mode_policy_rejects_invalid_thresholds(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SimplePortEigenpairPolicy(**kwargs)


def test_simple_mode_layout_and_precision_are_explicit() -> None:
    with pytest.raises(ValueError, match="square"):
        solve_simple_port_eigenpair(
            jnp.ones((2, 3), dtype=jnp.float64),
            jnp.eye(2, dtype=jnp.float64),
            jnp.asarray(1.0, dtype=jnp.float64),
            selected_mode_index=0,
            phase_anchor_edge_dof=0,
        )
    with pytest.raises(ValueError, match="edge mass"):
        solve_simple_port_eigenpair(
            jnp.eye(2, dtype=jnp.float64),
            jnp.eye(3, dtype=jnp.float64),
            jnp.asarray(1.0, dtype=jnp.float64),
            selected_mode_index=0,
            phase_anchor_edge_dof=0,
        )
    with pytest.raises(TypeError, match="float64 stiffness"):
        solve_simple_port_eigenpair(
            jnp.eye(2, dtype=jnp.float32),
            jnp.eye(2, dtype=jnp.float64),
            jnp.asarray(1.0, dtype=jnp.float64),
            selected_mode_index=0,
            phase_anchor_edge_dof=0,
        )
    with pytest.raises(TypeError, match="float64 edge mass"):
        solve_simple_port_eigenpair(
            jnp.eye(2, dtype=jnp.float64),
            jnp.eye(2, dtype=jnp.float32),
            jnp.asarray(1.0, dtype=jnp.float64),
            selected_mode_index=0,
            phase_anchor_edge_dof=0,
        )
    with pytest.raises(TypeError, match="selected_mode_index"):
        solve_simple_port_eigenpair(
            jnp.eye(2, dtype=jnp.float64),
            jnp.eye(2, dtype=jnp.float64),
            jnp.asarray(1.0, dtype=jnp.float64),
            selected_mode_index=True,  # type: ignore[arg-type]
            phase_anchor_edge_dof=0,
        )
    with pytest.raises(TypeError, match="phase_anchor_edge_dof"):
        solve_simple_port_eigenpair(
            jnp.eye(2, dtype=jnp.float64),
            jnp.eye(2, dtype=jnp.float64),
            jnp.asarray(1.0, dtype=jnp.float64),
            selected_mode_index=0,
            phase_anchor_edge_dof=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="propagation scale"):
        solve_simple_port_eigenpair(
            jnp.eye(2, dtype=jnp.float64),
            jnp.eye(2, dtype=jnp.float64),
            jnp.ones((1,), dtype=jnp.float64),
            selected_mode_index=0,
            phase_anchor_edge_dof=0,
        )
    with pytest.raises(ValueError, match="selected_mode_index"):
        solve_simple_port_eigenpair(
            jnp.eye(2, dtype=jnp.float64),
            jnp.eye(2, dtype=jnp.float64),
            jnp.asarray(1.0, dtype=jnp.float64),
            selected_mode_index=2,
            phase_anchor_edge_dof=0,
        )
    with pytest.raises(ValueError, match="phase_anchor_edge_dof"):
        solve_simple_port_eigenpair(
            jnp.eye(2, dtype=jnp.float64),
            jnp.eye(2, dtype=jnp.float64),
            jnp.asarray(1.0, dtype=jnp.float64),
            selected_mode_index=0,
            phase_anchor_edge_dof=2,
        )


def test_simple_mode_inspection_rejects_ambiguous_layout_and_precision() -> None:
    common = {
        "selected_mode_index": 0,
        "phase_anchor_edge_dof": 0,
    }
    with pytest.raises(ValueError, match="propagation scale"):
        inspect_simple_port_eigenpair(
            jnp.eye(2, dtype=jnp.float64),
            jnp.eye(2, dtype=jnp.float64),
            jnp.ones((1,), dtype=jnp.float64),
            **common,
        )
    with pytest.raises(TypeError, match="float64 propagation scale"):
        inspect_simple_port_eigenpair(
            jnp.eye(2, dtype=jnp.float64),
            jnp.eye(2, dtype=jnp.float64),
            jnp.asarray(1.0, dtype=jnp.float32),
            **common,
        )
    with pytest.raises(TypeError, match="float64 stiffness"):
        inspect_simple_port_eigenpair(
            jnp.eye(2, dtype=jnp.float32),
            jnp.eye(2, dtype=jnp.float64),
            jnp.asarray(1.0, dtype=jnp.float64),
            **common,
        )
    with pytest.raises(TypeError, match="float64 edge mass"):
        inspect_simple_port_eigenpair(
            jnp.eye(2, dtype=jnp.float64),
            jnp.eye(2, dtype=jnp.float32),
            jnp.asarray(1.0, dtype=jnp.float64),
            **common,
        )
    with pytest.raises(TypeError, match="float64 propagation scale"):
        solve_simple_port_eigenpair(
            jnp.eye(2, dtype=jnp.float64),
            jnp.eye(2, dtype=jnp.float64),
            jnp.asarray(1.0, dtype=jnp.float32),
            **common,
        )
