from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from femx.backends.jax.port_cluster_adjoint import (  # noqa: E402
    PortClusterContour,
    inspect_invariant_port_cluster,
    solve_invariant_port_cluster,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]

_SCALE = 2.0
_TRANSFORM = np.asarray(
    (
        (1.0, 0.2, 0.1, 0.0),
        (0.0, 1.0, 0.3, 0.1),
        (0.1, 0.0, 1.0, 0.2),
        (0.0, 0.1, 0.0, 1.0),
    ),
    dtype=np.float64,
)
_TRANSFORM_DIRECTION = np.asarray(
    (
        (0.0, 0.0, 0.12, 0.0),
        (0.0, 0.0, 0.0, -0.08),
        (0.05, 0.0, 0.0, 0.0),
        (0.0, -0.04, 0.0, 0.0),
    ),
    dtype=np.float64,
)
_MASS = np.asarray(
    (
        (1.4, 0.1, 0.0, 0.02),
        (0.1, 1.2, 0.04, 0.0),
        (0.0, 0.04, 1.1, 0.03),
        (0.02, 0.0, 0.03, 0.9),
    ),
    dtype=np.float64,
)
_MASS_DIRECTION = np.asarray(
    (
        (0.03, -0.01, 0.0, 0.0),
        (-0.01, -0.02, 0.005, 0.0),
        (0.0, 0.005, 0.01, -0.004),
        (0.0, 0.0, -0.004, 0.015),
    ),
    dtype=np.float64,
)
_RIGHT_PROBE = _TRANSFORM[:, :2]
_LEFT_PROBE = _MASS @ _RIGHT_PROBE
_CONTOUR = PortClusterContour(center=-1.0, radius=0.2, expected_cluster_size=2)


def _pencil(parameter: jax.Array) -> tuple[jax.Array, jax.Array]:
    transform = jnp.asarray(_TRANSFORM) @ (
        jnp.eye(4, dtype=jnp.float64) + parameter * jnp.asarray(_TRANSFORM_DIRECTION)
    )
    eigenvalues = jnp.asarray(
        (-1.0 + 0.25 * parameter, -1.0 - 0.05 * parameter, -0.25, -0.05),
        dtype=jnp.float64,
    )
    operator = transform @ jnp.diag(eigenvalues) @ jnp.linalg.inv(transform)
    mass = jnp.asarray(_MASS) + parameter * jnp.asarray(_MASS_DIRECTION)
    stiffness = (_SCALE**2) * mass @ operator
    return stiffness, mass


def _cluster(parameter: jax.Array, *, right_probe: jax.Array | None = None, left_probe=None):
    stiffness, mass = _pencil(parameter)
    return solve_invariant_port_cluster(
        stiffness,
        mass,
        jnp.asarray(_SCALE, dtype=jnp.float64),
        jnp.asarray(_RIGHT_PROBE) if right_probe is None else right_probe,
        jnp.asarray(_LEFT_PROBE) if left_probe is None else left_probe,
        contour=_CONTOUR,
    )


def _exact_numpy(parameter: float) -> tuple[float, np.ndarray]:
    transform = _TRANSFORM @ (np.eye(4) + parameter * _TRANSFORM_DIRECTION)
    mass = _MASS + parameter * _MASS_DIRECTION
    right_basis = transform[:, :2]
    gram = right_basis.T @ mass @ right_basis
    projector = right_basis @ np.linalg.solve(gram, right_basis.T @ mass)
    eigenvalues = np.asarray((-1.0 + 0.25 * parameter, -1.0 - 0.05 * parameter))
    propagation_sum = _SCALE * float(np.sum(np.sqrt(-eigenvalues)))
    return propagation_sum, projector


def test_repeated_nonsymmetric_cluster_is_basis_invariant_and_jittable() -> None:
    stiffness, mass = _pencil(jnp.asarray(0.0, dtype=jnp.float64))
    inspection = inspect_invariant_port_cluster(
        stiffness,
        mass,
        jnp.asarray(_SCALE, dtype=jnp.float64),
        jnp.asarray(_RIGHT_PROBE),
        jnp.asarray(_LEFT_PROBE),
        contour=_CONTOUR,
    )
    cluster = jax.jit(_cluster)(jnp.asarray(0.0, dtype=jnp.float64))
    exact_beta_sum, exact_projector = _exact_numpy(0.0)

    assert bool(inspection.diagnostics.is_valid)
    assert int(inspection.diagnostics.observed_cluster_size) == 2
    assert float(inspection.diagnostics.relative_contour_clearance) > 0.9
    assert float(inspection.diagnostics.relative_quadrature_error) < 1.0e-9
    assert float(inspection.diagnostics.maximum_relative_shifted_residual) < 1.0e-14
    assert float(inspection.diagnostics.mass_symmetry_error) < 1.0e-15
    assert float(inspection.diagnostics.probe_moment_singular_value_ratio) > 0.1
    assert float(inspection.diagnostics.projector_idempotency_error) < 1.0e-13
    assert float(inspection.diagnostics.projector_mass_self_adjoint_error) < 1.0e-13
    assert float(cluster.dimensionless_eigenvalue_sum) == pytest.approx(-2.0, abs=2.0e-13)
    assert float(cluster.eigenvalue_sum_per_m2) == pytest.approx(-8.0, abs=8.0e-13)
    assert float(cluster.propagation_constant_sum_per_m) == pytest.approx(
        exact_beta_sum, abs=3.0e-13
    )
    assert float(cluster.mean_propagation_constant_per_m) == pytest.approx(
        exact_beta_sum / 2.0, abs=2.0e-13
    )
    np.testing.assert_allclose(
        np.asarray(cluster.reduced_edge_mass_projector),
        exact_projector,
        rtol=2.0e-13,
        atol=2.0e-13,
    )

    right_mixing = jnp.asarray(((1.0, -0.4), (0.3, 1.2)), dtype=jnp.float64)
    left_mixing = jnp.asarray(((0.8, 0.2), (-0.1, 1.1)), dtype=jnp.float64)
    mixed = _cluster(
        jnp.asarray(0.0, dtype=jnp.float64),
        right_probe=jnp.asarray(_RIGHT_PROBE) @ right_mixing,
        left_probe=jnp.asarray(_LEFT_PROBE) @ left_mixing,
    )
    np.testing.assert_allclose(
        np.asarray(mixed.reduced_edge_mass_projector),
        exact_projector,
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    assert float(mixed.propagation_constant_sum_per_m) == pytest.approx(exact_beta_sum, abs=4.0e-13)


def test_cluster_reverse_mode_matches_independent_exact_model_and_finite_difference() -> None:
    weights = jnp.asarray(
        (
            (0.1, -0.2, 0.03, 0.04),
            (0.05, 0.17, -0.08, 0.02),
            (-0.03, 0.01, 0.11, -0.06),
            (0.04, -0.09, 0.02, 0.13),
        ),
        dtype=jnp.float64,
    )

    def objective(parameter: jax.Array) -> jax.Array:
        cluster = _cluster(parameter)
        return 0.3 * cluster.propagation_constant_sum_per_m + jnp.sum(
            weights * cluster.reduced_edge_mass_projector
        )

    def exact_objective(parameter: float) -> float:
        propagation_sum, projector = _exact_numpy(parameter)
        return 0.3 * propagation_sum + float(np.sum(np.asarray(weights) * projector))

    parameter = jnp.asarray(0.0, dtype=jnp.float64)
    value, reverse_gradient = jax.jit(jax.value_and_grad(objective))(parameter)
    step = 2.0e-5
    contour_difference = (
        float(objective(jnp.asarray(step))) - float(objective(jnp.asarray(-step)))
    ) / (2.0 * step)
    exact_difference = (exact_objective(step) - exact_objective(-step)) / (2.0 * step)

    assert float(value) == pytest.approx(exact_objective(0.0), rel=2.0e-13, abs=2.0e-13)
    assert float(reverse_gradient) == pytest.approx(exact_difference, rel=3.0e-9, abs=3.0e-10)
    assert float(reverse_gradient) == pytest.approx(contour_difference, rel=3.0e-9, abs=3.0e-10)


def test_cluster_fails_closed_when_contour_membership_or_metric_is_invalid() -> None:
    stiffness, mass = _pencil(jnp.asarray(0.0, dtype=jnp.float64))
    wrong_membership = solve_invariant_port_cluster(
        stiffness,
        mass,
        jnp.asarray(_SCALE, dtype=jnp.float64),
        jnp.asarray(_RIGHT_PROBE),
        jnp.asarray(_LEFT_PROBE),
        contour=PortClusterContour(center=-0.7, radius=0.6, expected_cluster_size=2),
    )
    assert math.isnan(float(wrong_membership.propagation_constant_sum_per_m))
    assert np.isnan(np.asarray(wrong_membership.reduced_edge_mass_projector)).all()

    nonsymmetric_mass = mass.at[0, 1].add(0.2)
    invalid_metric = inspect_invariant_port_cluster(
        stiffness,
        nonsymmetric_mass,
        jnp.asarray(_SCALE, dtype=jnp.float64),
        jnp.asarray(_RIGHT_PROBE),
        jnp.asarray(_LEFT_PROBE),
        contour=_CONTOUR,
    )
    assert not bool(invalid_metric.diagnostics.is_valid)
    assert float(invalid_metric.diagnostics.mass_symmetry_error) > 1.0e-3

    def invalid_objective(parameter: jax.Array) -> jax.Array:
        shifted_stiffness, shifted_mass = _pencil(parameter)
        return solve_invariant_port_cluster(
            shifted_stiffness,
            shifted_mass,
            jnp.asarray(_SCALE, dtype=jnp.float64),
            jnp.asarray(_RIGHT_PROBE),
            jnp.asarray(_LEFT_PROBE),
            contour=PortClusterContour(center=-0.7, radius=0.6, expected_cluster_size=2),
        ).propagation_constant_sum_per_m

    assert math.isnan(float(jax.grad(invalid_objective)(jnp.asarray(0.0))))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"center": math.nan}, "center"),
        ({"radius": 0.0}, "radius"),
        ({"center": -0.1, "radius": 0.2}, "negative real"),
        ({"expected_cluster_size": 0}, "expected size"),
        ({"expected_cluster_size": True}, "expected size"),
        ({"quadrature_point_count": 7}, "quadrature count"),
        ({"quadrature_point_count": 9}, "quadrature count"),
        ({"maximum_relative_quadrature_error": math.inf}, "quadrature error"),
        ({"minimum_probe_moment_singular_value_ratio": 0.0}, "singular-value ratio"),
    ],
)
def test_cluster_contour_rejects_invalid_policy(kwargs: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "center": -1.0,
        "radius": 0.2,
        "expected_cluster_size": 2,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        PortClusterContour(**values)  # type: ignore[arg-type]


def test_cluster_layout_and_precision_contracts_are_explicit() -> None:
    stiffness, mass = _pencil(jnp.asarray(0.0, dtype=jnp.float64))
    default_scale = jnp.asarray(_SCALE, dtype=jnp.float64)
    default_right = jnp.asarray(_RIGHT_PROBE)
    default_left = jnp.asarray(_LEFT_PROBE)

    def solve_with(
        candidate_stiffness=stiffness,
        candidate_mass=mass,
        scale=default_scale,
        right=default_right,
        left=default_left,
    ):
        return solve_invariant_port_cluster(
            candidate_stiffness,
            candidate_mass,
            scale,
            right,
            left,
            contour=_CONTOUR,
        )

    with pytest.raises(ValueError, match="square"):
        solve_with(candidate_stiffness=stiffness[:, :3])
    with pytest.raises(ValueError, match="edge mass"):
        solve_with(candidate_mass=mass[:3, :3])
    with pytest.raises(ValueError, match="right cluster probe"):
        solve_with(right=jnp.ones((4, 1), dtype=jnp.float64))
    with pytest.raises(ValueError, match="left cluster probe"):
        solve_with(left=jnp.ones((3, 2), dtype=jnp.float64))
    with pytest.raises(ValueError, match="scale must be a scalar"):
        solve_with(scale=jnp.asarray((_SCALE,), dtype=jnp.float64))
    with pytest.raises(TypeError, match="float64 propagation scale"):
        solve_with(scale=jnp.asarray(_SCALE, dtype=jnp.float32))
    with pytest.raises(TypeError, match="float64 stiffness"):
        solve_with(candidate_stiffness=stiffness.astype(jnp.float32))
    with pytest.raises(TypeError, match="float64 edge mass"):
        solve_with(candidate_mass=mass.astype(jnp.float32))
    with pytest.raises(TypeError, match="right cluster probe"):
        solve_with(right=jnp.asarray(_RIGHT_PROBE, dtype=jnp.float32))
    with pytest.raises(TypeError, match="left cluster probe"):
        solve_with(left=jnp.asarray(_LEFT_PROBE, dtype=jnp.float32))

    complex_result = solve_with(
        right=jnp.asarray(_RIGHT_PROBE, dtype=jnp.complex128),
        left=jnp.asarray(_LEFT_PROBE, dtype=jnp.complex128),
    )
    assert np.isfinite(np.asarray(complex_result.reduced_edge_mass_projector)).all()
