import itertools
from collections.abc import Callable
from typing import Any

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from femx.backends.jax.elements.tetrahedron_h1 import (  # noqa: E402
    tetrahedron_p1_cell_nodal_load_vectors,
    tetrahedron_p1_diffusion_cell_matrices,
    tetrahedron_p1_field_gradient,
    tetrahedron_p1_geometry,
    tetrahedron_p1_local_operators,
    tetrahedron_p1_nodal_diffusion_cell_matrices,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def _reference_tetrahedron() -> tuple[Any, Any]:
    coordinates = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=jnp.float64,
    )
    cells = jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32)
    return coordinates, cells


def test_reference_tet4_basis_and_local_operators_match_closed_forms() -> None:
    coordinates, cells = _reference_tetrahedron()

    operators = jax.jit(tetrahedron_p1_local_operators)(coordinates, cells)

    # Elmer 4f2d7e4 PElementBase.F90:dTetraNodalLBasisAll uses these gradients.
    reference_gradients = np.asarray(
        ((-1.0, -1.0, -1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    expected_stiffness = reference_gradients @ reference_gradients.T / 6.0
    expected_mass = (np.ones((4, 4)) + np.eye(4)) / 120.0

    np.testing.assert_allclose(operators.geometry.jacobians, np.eye(3)[None, :, :])
    np.testing.assert_allclose(operators.geometry.determinants, (1.0,))
    np.testing.assert_allclose(operators.geometry.volumes, (1.0 / 6.0,))
    np.testing.assert_allclose(
        operators.geometry.basis_gradients[0], reference_gradients, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        operators.unit_stiffness[0], expected_stiffness, rtol=0.0, atol=2.0e-16
    )
    np.testing.assert_allclose(operators.consistent_mass[0], expected_mass, rtol=0.0, atol=2.0e-16)
    np.testing.assert_allclose(operators.unit_stiffness[0].sum(axis=1), 0.0, atol=1.0e-16)
    eigenvalues = np.linalg.eigvalsh(np.asarray(operators.unit_stiffness[0]))
    assert eigenvalues[0] >= -1.0e-16
    assert np.count_nonzero(eigenvalues > 1.0e-14) == 3


def test_isotropic_and_anisotropic_diffusion_match_independent_weak_form() -> None:
    coordinates, cells = _reference_tetrahedron()
    gradients = np.asarray(((-1.0, -1.0, -1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    tensor = np.asarray(((3.0, 0.2, 0.1), (0.2, 2.0, -0.1), (0.1, -0.1, 4.0)))

    isotropic = tetrahedron_p1_diffusion_cell_matrices(
        coordinates, cells, jnp.asarray((2.5,), dtype=jnp.float64)
    )
    anisotropic = jax.jit(tetrahedron_p1_diffusion_cell_matrices)(
        coordinates, cells, jnp.asarray(tensor[None, :, :])
    )

    np.testing.assert_allclose(isotropic[0], 2.5 * gradients @ gradients.T / 6.0)
    np.testing.assert_allclose(anisotropic[0], gradients @ tensor @ gradients.T / 6.0)
    np.testing.assert_allclose(anisotropic[0], anisotropic[0].T, atol=1.0e-16)


def test_nodal_material_interpolation_reduces_to_exact_cell_mean() -> None:
    coordinates, cells = _reference_tetrahedron()
    scalar_nodes = jnp.asarray(((1.0, 2.0, 3.0, 6.0),), dtype=jnp.float64)
    tensor_nodes = jnp.stack(
        tuple(jnp.eye(3, dtype=jnp.float64) * value for value in (1.0, 2.0, 4.0, 9.0)),
        axis=0,
    )[None, ...]

    scalar = tetrahedron_p1_nodal_diffusion_cell_matrices(coordinates, cells, scalar_nodes)
    tensor = tetrahedron_p1_nodal_diffusion_cell_matrices(coordinates, cells, tensor_nodes)
    scalar_mean = jnp.mean(scalar_nodes, axis=1)
    tensor_mean = jnp.mean(tensor_nodes, axis=1)

    np.testing.assert_allclose(
        scalar,
        tetrahedron_p1_diffusion_cell_matrices(coordinates, cells, scalar_mean),
    )
    np.testing.assert_allclose(
        tensor,
        tetrahedron_p1_diffusion_cell_matrices(coordinates, cells, tensor_mean),
    )


def test_consistent_p1_source_preserves_exact_tetrahedron_integral() -> None:
    coordinates, cells = _reference_tetrahedron()
    source = jnp.asarray(((1.0, 2.0, 4.0, 8.0),), dtype=jnp.float64)

    local_load = jax.jit(tetrahedron_p1_cell_nodal_load_vectors)(coordinates, cells, source)

    expected = (np.sum(source) + np.asarray(source[0])) / 120.0
    np.testing.assert_allclose(local_load[0], expected, rtol=0.0, atol=2.0e-16)
    np.testing.assert_allclose(
        jnp.sum(local_load),
        jnp.sum(source) / 24.0,
        rtol=0.0,
        atol=2.0e-16,
    )


def test_affine_field_gradient_is_exact_on_a_skew_tetrahedron() -> None:
    coordinates = jnp.asarray(
        (
            (0.2, -0.4, 0.1),
            (1.7, -0.1, 0.4),
            (0.4, 1.3, -0.2),
            (-0.3, 0.2, 1.8),
        ),
        dtype=jnp.float64,
    )
    cells = jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32)
    expected_gradient = jnp.asarray((2.0, -3.0, 0.75), dtype=jnp.float64)
    values = 4.5 + coordinates @ expected_gradient

    gradient = jax.jit(tetrahedron_p1_field_gradient)(coordinates, cells, values)

    np.testing.assert_allclose(gradient[0], expected_gradient, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(
        tetrahedron_p1_geometry(coordinates, cells).basis_gradients.sum(axis=1),
        0.0,
        atol=2.0e-15,
    )


@pytest.mark.parametrize("permutation", tuple(itertools.permutations(range(4))))
def test_global_tet4_stiffness_is_invariant_to_all_node_permutations(
    permutation: tuple[int, int, int, int],
) -> None:
    coordinates = jnp.asarray(
        ((0.1, 0.2, -0.1), (1.3, 0.0, 0.4), (0.2, 1.1, 0.3), (-0.2, 0.4, 1.5)),
        dtype=jnp.float64,
    )
    canonical_cells = jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32)
    permuted_cells = jnp.asarray((permutation,), dtype=jnp.int32)

    canonical = np.asarray(
        tetrahedron_p1_diffusion_cell_matrices(
            coordinates, canonical_cells, jnp.asarray((2.0,), dtype=jnp.float64)
        )[0]
    )
    permuted = np.asarray(
        tetrahedron_p1_diffusion_cell_matrices(
            coordinates, permuted_cells, jnp.asarray((2.0,), dtype=jnp.float64)
        )[0]
    )
    scattered = np.zeros((4, 4))
    scattered[np.ix_(permutation, permutation)] = permuted
    determinant = float(tetrahedron_p1_geometry(coordinates, permuted_cells).determinants[0])

    np.testing.assert_allclose(scattered, canonical, rtol=0.0, atol=5.0e-15)
    assert abs(determinant) > 0.0


def test_geometry_and_material_reverse_mode_match_directional_central_difference() -> None:
    coordinates = jnp.asarray(
        ((0.1, 0.2, -0.1), (1.3, 0.0, 0.4), (0.2, 1.1, 0.3), (-0.2, 0.4, 1.5)),
        dtype=jnp.float64,
    )
    cells = jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32)
    diffusion = jnp.asarray((2.3,), dtype=jnp.float64)
    direction = jnp.asarray(
        ((0.2, -0.1, 0.3), (-0.3, 0.4, 0.1), (0.5, -0.2, -0.4), (-0.1, 0.2, 0.6)),
        dtype=jnp.float64,
    )
    weights = jnp.asarray(
        (
            (0.2, -0.1, 0.4, 0.7),
            (0.3, 0.5, -0.2, 0.1),
            (-0.4, 0.2, 0.6, -0.3),
            (0.1, -0.5, 0.3, 0.8),
        ),
        dtype=jnp.float64,
    )

    def objective(points: Any, coefficient: Any) -> Any:
        stiffness = tetrahedron_p1_diffusion_cell_matrices(points, cells, coefficient)
        mass = tetrahedron_p1_local_operators(points, cells).consistent_mass
        return jnp.sum(weights * stiffness[0]) + 0.17 * jnp.sum(weights * mass[0])

    coordinate_gradient, material_gradient = jax.jit(jax.grad(objective, argnums=(0, 1)))(
        coordinates, diffusion
    )
    step = 1.0e-6
    finite_difference = (
        objective(coordinates + step * direction, diffusion + step * 0.4)
        - objective(coordinates - step * direction, diffusion - step * 0.4)
    ) / (2.0 * step)
    reverse_directional = jnp.vdot(coordinate_gradient, direction) + 0.4 * material_gradient[0]

    np.testing.assert_allclose(reverse_directional, finite_difference, rtol=2.0e-9, atol=2.0e-10)
    np.testing.assert_allclose(coordinate_gradient.sum(axis=0), 0.0, atol=2.0e-15)


@pytest.mark.parametrize(
    ("coordinates", "cells", "error", "message"),
    [
        (
            jnp.zeros((4,), dtype=jnp.float64),
            jnp.zeros((1, 4), dtype=jnp.int32),
            ValueError,
            "coordinates",
        ),
        (
            jnp.zeros((4, 2), dtype=jnp.float64),
            jnp.zeros((1, 4), dtype=jnp.int32),
            ValueError,
            "coordinates",
        ),
        (
            jnp.zeros((4, 3), dtype=jnp.int32),
            jnp.zeros((1, 4), dtype=jnp.int32),
            TypeError,
            "floating",
        ),
        (
            jnp.zeros((4, 3), dtype=jnp.float64),
            jnp.zeros((4,), dtype=jnp.int32),
            ValueError,
            "cells",
        ),
        (
            jnp.zeros((4, 3), dtype=jnp.float64),
            jnp.zeros((1, 3), dtype=jnp.int32),
            ValueError,
            "cells",
        ),
        (
            jnp.zeros((4, 3), dtype=jnp.float64),
            jnp.zeros((0, 4), dtype=jnp.int32),
            ValueError,
            "at least one",
        ),
        (
            jnp.zeros((4, 3), dtype=jnp.float64),
            jnp.zeros((1, 4), dtype=jnp.float64),
            TypeError,
            "integer",
        ),
    ],
)
def test_tet4_geometry_rejects_ambiguous_array_contracts(
    coordinates: Any,
    cells: Any,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        tetrahedron_p1_geometry(coordinates, cells)


@pytest.mark.parametrize(
    ("factory", "error", "message"),
    [
        (
            lambda c, e: tetrahedron_p1_diffusion_cell_matrices(c, e, jnp.ones((1, 1))),
            ValueError,
            "cell diffusion",
        ),
        (
            lambda c, e: tetrahedron_p1_diffusion_cell_matrices(
                c, e, jnp.ones((1,), dtype=jnp.int32)
            ),
            TypeError,
            "floating",
        ),
        (
            lambda c, e: tetrahedron_p1_diffusion_cell_matrices(
                c, e, jnp.ones((1,), dtype=jnp.float32)
            ),
            TypeError,
            "same dtype",
        ),
        (
            lambda c, e: tetrahedron_p1_nodal_diffusion_cell_matrices(c, e, jnp.ones((1, 3))),
            ValueError,
            "nodal diffusion",
        ),
        (
            lambda c, e: tetrahedron_p1_nodal_diffusion_cell_matrices(
                c, e, jnp.ones((1, 4), dtype=jnp.int32)
            ),
            TypeError,
            "floating",
        ),
        (
            lambda c, e: tetrahedron_p1_nodal_diffusion_cell_matrices(
                c, e, jnp.ones((1, 4), dtype=jnp.float32)
            ),
            TypeError,
            "same dtype",
        ),
        (
            lambda c, e: tetrahedron_p1_cell_nodal_load_vectors(c, e, jnp.ones((1, 3))),
            ValueError,
            "nodal source",
        ),
        (
            lambda c, e: tetrahedron_p1_cell_nodal_load_vectors(
                c, e, jnp.ones((1, 4), dtype=jnp.int32)
            ),
            TypeError,
            "floating",
        ),
        (
            lambda c, e: tetrahedron_p1_cell_nodal_load_vectors(
                c, e, jnp.ones((1, 4), dtype=jnp.float32)
            ),
            TypeError,
            "same dtype",
        ),
        (
            lambda c, e: tetrahedron_p1_field_gradient(c, e, jnp.ones((4, 1))),
            ValueError,
            "nodal field",
        ),
        (
            lambda c, e: tetrahedron_p1_field_gradient(c, e, jnp.ones((4,), dtype=jnp.int32)),
            TypeError,
            "floating",
        ),
        (
            lambda c, e: tetrahedron_p1_field_gradient(c, e, jnp.ones((4,), dtype=jnp.float32)),
            TypeError,
            "same dtype",
        ),
    ],
)
def test_tet4_coefficients_reject_shape_and_dtype_drift(
    factory: Callable[[Any, Any], Any],
    error: type[Exception],
    message: str,
) -> None:
    coordinates, cells = _reference_tetrahedron()
    with pytest.raises(error, match=message):
        factory(coordinates, cells)


def test_degenerate_tetrahedron_remains_explicitly_nonfinite() -> None:
    coordinates = jnp.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)),
        dtype=jnp.float64,
    )
    cells = jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32)

    geometry = tetrahedron_p1_geometry(coordinates, cells)

    np.testing.assert_allclose(geometry.determinants, 0.0, atol=0.0)
    np.testing.assert_allclose(geometry.volumes, 0.0, atol=0.0)
    assert not np.isfinite(np.asarray(geometry.basis_gradients)).all()
