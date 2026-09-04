from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from femx.backends.jax.port_operator import (  # noqa: E402
    lossless_port_coefficients,
    triangle_port_local_pencil,
)
from femx.physics.port_eigenmode import (  # noqa: E402
    VACUUM_PERMEABILITY_H_PER_M,
    VACUUM_PERMITTIVITY_F_PER_M,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def _canonical_signs(cells: np.ndarray) -> np.ndarray:
    local_edges = cells[:, ((0, 1), (1, 2), (2, 0))]
    return np.where(local_edges[:, :, 0] < local_edges[:, :, 1], 1, -1).astype(np.int8)


def _independent_elmer_local_pencil(
    coordinates: np.ndarray,
    cell: np.ndarray,
    signs: np.ndarray,
    epsilon: float,
    nu: float,
    omega: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent quadrature transcription of locked EMPort.F90 lines 697-784."""

    vertices = coordinates[cell]
    jacobian = np.column_stack((vertices[1] - vertices[0], vertices[2] - vertices[0]))
    determinant = float(np.linalg.det(jacobian))
    reference_gradients = np.asarray(((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0)))
    gradients = reference_gradients @ np.linalg.inv(jacobian)
    quadrature = np.asarray(
        ((1.0 / 6.0, 1.0 / 6.0), (2.0 / 3.0, 1.0 / 6.0), (1.0 / 6.0, 2.0 / 3.0))
    )
    barycentric = np.column_stack(
        (1.0 - quadrature[:, 0] - quadrature[:, 1], quadrature[:, 0], quadrature[:, 1])
    )
    weights = np.full(3, abs(determinant) / 6.0)
    local_edges = ((0, 1), (1, 2), (2, 0))
    edge_basis = np.stack(
        tuple(
            signs[edge]
            * (
                barycentric[:, start, None] * gradients[end]
                - barycentric[:, end, None] * gradients[start]
            )
            for edge, (start, end) in enumerate(local_edges)
        ),
        axis=1,
    )
    edge_curl = 2.0 * signs / determinant

    stiffness = np.zeros((6, 6))
    mass = np.zeros((6, 6))
    for point, weight in enumerate(weights):
        basis = barycentric[point]
        vector_basis = edge_basis[point]
        stiffness[:3, :3] -= weight * epsilon * np.outer(basis, basis)
        coupling = gradients @ vector_basis.T
        stiffness[:3, 3:] += weight * epsilon * coupling
        stiffness[3:, :3] += weight * nu * coupling.T
        stiffness[3:, 3:] += weight * (
            nu * np.outer(edge_curl, edge_curl)
            - omega**2 * epsilon * (vector_basis @ vector_basis.T)
        )
        mass[3:, 3:] += weight * nu * (vector_basis @ vector_basis.T)
    return stiffness, mass


def test_lossless_coefficients_match_locked_elmer_material_scaling() -> None:
    epsilon, nu = jax.jit(lossless_port_coefficients)(
        jnp.asarray((2.0, 12.0)),
        jnp.asarray((1.0, 2.5)),
    )

    np.testing.assert_allclose(
        epsilon,
        VACUUM_PERMITTIVITY_F_PER_M * np.asarray((2.0, 12.0)),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        nu,
        1.0 / (VACUUM_PERMEABILITY_H_PER_M * np.asarray((1.0, 2.5))),
        rtol=0.0,
        atol=0.0,
    )


def test_local_pencil_matches_independent_locked_elmer_quadrature() -> None:
    coordinates = np.asarray(((0.2, -0.1), (1.7, 0.3), (-0.4, 1.2)))
    cells = np.asarray(((2, 0, 1),), dtype=np.int32)
    signs = _canonical_signs(cells)
    epsilon = np.asarray((1.7,))
    nu = np.asarray((0.8,))
    omega = np.asarray(0.6)

    observed = jax.jit(triangle_port_local_pencil)(
        jnp.asarray(coordinates),
        jnp.asarray(cells),
        jnp.asarray(signs),
        jnp.asarray(epsilon),
        jnp.asarray(nu),
        jnp.asarray(omega),
    )
    expected_stiffness, expected_mass = _independent_elmer_local_pencil(
        coordinates,
        cells[0],
        signs[0],
        epsilon[0],
        nu[0],
        float(omega),
    )

    np.testing.assert_allclose(
        observed.stiffness[0], expected_stiffness, rtol=3.0e-15, atol=3.0e-15
    )
    np.testing.assert_allclose(observed.mass[0], expected_mass, rtol=3.0e-15, atol=3.0e-15)
    np.testing.assert_allclose(observed.mass[0, :3, :], 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(observed.mass[0, :, :3], 0.0, rtol=0.0, atol=0.0)
    assert np.all(np.linalg.eigvalsh(np.asarray(observed.mass[0, 3:, 3:])) > 0.0)


def test_local_pencil_reverse_geometry_derivative_matches_central_difference() -> None:
    cells = jnp.asarray(((0, 1, 2),), dtype=jnp.int32)
    signs = jnp.asarray(((1, 1, -1),), dtype=jnp.int8)
    coordinates = jnp.asarray(((0.0, 0.0), (1.2, 0.1), (0.2, 0.9)))
    stiffness_weights = jnp.arange(36, dtype=jnp.float64).reshape(6, 6) / 37.0
    mass_weights = jnp.flip(stiffness_weights, axis=0)

    def objective(points: jax.Array) -> jax.Array:
        pencil = triangle_port_local_pencil(
            points,
            cells,
            signs,
            jnp.asarray((1.3,)),
            jnp.asarray((0.7,)),
            jnp.asarray(0.4),
        )
        return jnp.sum(pencil.stiffness[0] * stiffness_weights) + 0.2 * jnp.sum(
            pencil.mass[0] * mass_weights
        )

    gradient = jax.jit(jax.grad(objective))(coordinates)
    direction = jnp.asarray(((0.3, -0.2), (-0.1, 0.4), (0.2, -0.3)))
    step = 1.0e-6
    central = (
        objective(coordinates + step * direction) - objective(coordinates - step * direction)
    ) / (2.0 * step)
    reverse = jnp.vdot(gradient, direction)

    np.testing.assert_allclose(reverse, central, rtol=2.0e-9, atol=2.0e-9)
    np.testing.assert_allclose(jnp.sum(gradient, axis=0), 0.0, rtol=0.0, atol=3.0e-14)
