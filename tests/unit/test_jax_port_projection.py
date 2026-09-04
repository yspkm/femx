from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from femx.backends.jax.elements.triangle_nedelec import (  # noqa: E402
    evaluate_triangle_nedelec1,
)
from femx.backends.jax.port_projection import (  # noqa: E402
    expand_reduced_port_coefficients,
    project_port_electric_field_to_nodes,
    project_port_electromagnetic_fields_to_nodes,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def test_reduced_port_expansion_places_each_family_once_and_is_jittable() -> None:
    free_dofs = np.asarray((1, 2, 4, 6, 8), dtype=np.int64)
    scalar = jnp.asarray(((1.0 + 1.0j, 2.0), (3.0, 4.0 - 1.0j)))
    edge = jnp.asarray(((5.0 + 0.0j, 6.0), (7.0, 8.0), (9.0, 10.0)))
    expand = jax.jit(
        lambda scalar_values, edge_values: expand_reduced_port_coefficients(
            scalar_values,
            edge_values,
            free_dofs,
            node_count=4,
            edge_dof_count=5,
        )
    )

    expanded = expand(scalar, edge)

    np.testing.assert_array_equal(
        expanded.scalar_coefficients,
        ((0.0, 0.0), (1.0 + 1.0j, 2.0), (3.0, 4.0 - 1.0j), (0.0, 0.0)),
    )
    np.testing.assert_array_equal(
        expanded.edge_coefficients,
        ((5.0, 6.0), (0.0, 0.0), (7.0, 8.0), (0.0, 0.0), (9.0, 10.0)),
    )


@pytest.mark.parametrize(
    ("scalar", "edge", "free_dofs", "node_count", "edge_count", "message"),
    (
        (jnp.ones(2), jnp.ones((2, 1)), (0, 1, 2, 3), 2, 2, "rank-two"),
        (jnp.ones((2, 1)), jnp.ones((2, 2)), (0, 1, 2, 3), 2, 2, "mode count"),
        (
            jnp.ones((2, 1), dtype=jnp.float64),
            jnp.ones((2, 1), dtype=jnp.complex128),
            (0, 1, 2, 3),
            2,
            2,
            "same dtype",
        ),
        (jnp.ones((1, 1)), jnp.ones((1, 1)), (0, 1), True, 2, "node_count"),
        (jnp.ones((1, 1)), jnp.ones((1, 1)), (0, 1), 1, 0, "edge_dof_count"),
        (jnp.ones((1, 1)), jnp.ones((1, 1)), (0.0, 1.0), 1, 2, "integer"),
        (jnp.ones((1, 1)), jnp.ones((1, 1)), ((0, 1),), 1, 2, "rank-one"),
        (jnp.ones((1, 1)), jnp.ones((1, 1)), (0, 1, 2), 1, 2, "size"),
        (jnp.ones((1, 1)), jnp.ones((1, 1)), (-1, 1), 1, 2, "out-of-range"),
        (jnp.ones((1, 1)), jnp.ones((1, 1)), (0, 0), 1, 2, "strictly increasing"),
        (jnp.ones((1, 1)), jnp.ones((1, 1)), (0, 1), 2, 2, "nodal-first"),
    ),
)
def test_reduced_port_expansion_rejects_ambiguous_layouts(
    scalar: jax.Array,
    edge: jax.Array,
    free_dofs: object,
    node_count: int,
    edge_count: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        expand_reduced_port_coefficients(
            scalar,
            edge,
            free_dofs,
            node_count=node_count,
            edge_dof_count=edge_count,
        )


def test_port_projection_exactly_recovers_one_affine_nedelec_basis() -> None:
    coordinates = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    cells = jnp.asarray(((0, 1, 2),), dtype=jnp.int64)
    cell_edge_dofs = jnp.asarray(((0, 2, 1),), dtype=jnp.int64)
    cell_edge_signs = jnp.asarray(((1, 1, -1),), dtype=jnp.int8)
    scalar = jnp.zeros((3, 1), dtype=jnp.complex128)
    edge = jnp.asarray(((1.0 + 0.0j,), (0.0j,), (0.0j,)))
    free_dofs = np.arange(6, dtype=np.int64)

    project = jax.jit(
        lambda scalar_values, edge_values: project_port_electric_field_to_nodes(
            coordinates,
            cells,
            cell_edge_dofs,
            cell_edge_signs,
            scalar_values,
            edge_values,
            jnp.asarray((2.0 + 0.0j,)),
            free_dofs,
            edge_dof_count=3,
        )
    )
    projected = project(scalar, edge)
    nodal_basis = evaluate_triangle_nedelec1(
        coordinates,
        cells,
        cell_edge_signs,
        coordinates,
    ).basis[0, :, 0, :]

    np.testing.assert_allclose(projected.values[:, 0, :2], nodal_basis, atol=2.0e-15)
    np.testing.assert_array_equal(projected.values[:, 0, 2], 0.0)
    np.testing.assert_allclose(
        projected.nodal_mass,
        (np.ones((3, 3)) + np.eye(3)) / 24.0,
        atol=2.0e-15,
    )


@pytest.mark.parametrize(
    ("coordinates", "cells", "edge_dofs", "signs", "beta", "message"),
    (
        (
            jnp.ones(3),
            jnp.ones((1, 3), int),
            jnp.ones((1, 3), int),
            jnp.ones((1, 3), int),
            jnp.ones(1),
            "coordinates",
        ),
        (
            jnp.ones((3, 2)),
            jnp.ones((1, 4), int),
            jnp.ones((1, 4), int),
            jnp.ones((1, 4), int),
            jnp.ones(1),
            "cells",
        ),
        (
            jnp.ones((3, 2)),
            jnp.ones((1, 3), int),
            jnp.ones((1, 2), int),
            jnp.ones((1, 3), int),
            jnp.ones(1),
            "edge topology",
        ),
        (
            jnp.ones((3, 2)),
            jnp.ones((1, 3), int),
            jnp.ones((1, 3), int),
            jnp.ones((1, 3), int),
            jnp.ones((1, 1)),
            "rank one",
        ),
        (
            jnp.ones((3, 2)),
            jnp.ones((1, 3), int),
            jnp.ones((1, 3), int),
            jnp.ones((1, 3), int),
            jnp.ones(2),
            "mode count",
        ),
    ),
)
def test_port_projection_rejects_incompatible_shapes(
    coordinates: jax.Array,
    cells: jax.Array,
    edge_dofs: jax.Array,
    signs: jax.Array,
    beta: jax.Array,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        project_port_electric_field_to_nodes(
            coordinates,
            cells,
            edge_dofs,
            signs,
            jnp.zeros((3, 1), dtype=jnp.complex128),
            jnp.zeros((3, 1), dtype=jnp.complex128),
            beta,
            np.arange(6),
            edge_dof_count=3,
        )


def test_physical_port_fields_match_maxwell_curl_and_native_power_under_jit() -> None:
    coordinates = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    cells = jnp.asarray(((0, 1, 2),), dtype=jnp.int64)
    cell_edge_dofs = jnp.asarray(((0, 2, 1),), dtype=jnp.int64)
    cell_edge_signs = jnp.asarray(((1, 1, -1),), dtype=jnp.int8)
    scalar = jnp.asarray(((0.0j,), (1.0 + 0.0j,), (0.0j,)))
    edge = jnp.asarray(((1.0 + 0.0j,), (0.0j,), (0.0j,)))
    free_dofs = np.arange(6, dtype=np.int64)

    reconstruct = jax.jit(
        lambda scalar_values, edge_values: project_port_electromagnetic_fields_to_nodes(
            coordinates,
            cells,
            cell_edge_dofs,
            cell_edge_signs,
            scalar_values,
            edge_values,
            jnp.asarray((2.0 + 0.0j,)),
            jnp.asarray((3.0,)),
            jnp.asarray(5.0),
            free_dofs,
            edge_dof_count=3,
        )
    )
    fields = reconstruct(scalar, edge)
    nodal_basis = np.asarray(
        evaluate_triangle_nedelec1(
            coordinates,
            cells,
            cell_edge_signs,
            coordinates,
        ).basis[0, :, 0, :]
    )
    expected_electric = np.column_stack((nodal_basis, np.asarray((0.0j, -0.5j, 0.0j))))
    expected_magnetic = np.column_stack(
        (
            -1.2 * nodal_basis[:, 1],
            1.2 * nodal_basis[:, 0] + 0.3,
            np.full(3, -1.2j),
        )
    )

    np.testing.assert_allclose(fields.electric_values[:, 0, :], expected_electric, atol=2e-15)
    np.testing.assert_allclose(fields.magnetic_values[:, 0, :], expected_magnetic, atol=2e-15)
    np.testing.assert_allclose(fields.raw_forward_power_w, (0.25,), atol=2e-15)
    np.testing.assert_allclose(
        fields.nodal_mass,
        (np.ones((3, 3)) + np.eye(3)) / 24.0,
        atol=2e-15,
    )


@pytest.mark.parametrize(("beta", "expected_power"), ((2.0, 0.2), (-2.0, -0.2)))
def test_native_port_power_preserves_propagation_sign(
    beta: float,
    expected_power: float,
) -> None:
    fields = project_port_electromagnetic_fields_to_nodes(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        jnp.asarray(((0, 1, 2),), dtype=jnp.int64),
        jnp.asarray(((0, 2, 1),), dtype=jnp.int64),
        jnp.asarray(((1, 1, -1),), dtype=jnp.int8),
        jnp.zeros((3, 1), dtype=jnp.complex128),
        jnp.asarray(((1.0 + 0.0j,), (0.0j,), (0.0j,))),
        jnp.asarray((beta + 0.0j,)),
        jnp.asarray((3.0,)),
        jnp.asarray(5.0),
        np.arange(6, dtype=np.int64),
        edge_dof_count=3,
    )

    np.testing.assert_allclose(fields.raw_forward_power_w, (expected_power,), atol=2e-15)


@pytest.mark.parametrize(
    ("beta", "reluctivity", "omega", "message"),
    (
        (jnp.ones((1, 1)), jnp.ones(1), jnp.asarray(1.0), "rank one"),
        (jnp.ones(1), jnp.ones((1, 1)), jnp.asarray(1.0), "one value per triangle"),
        (jnp.ones(1), jnp.ones(2), jnp.asarray(1.0), "one value per triangle"),
        (jnp.ones(1), jnp.ones(1), jnp.ones(1), "must be a scalar"),
        (jnp.ones(2), jnp.ones(1), jnp.asarray(1.0), "mode count"),
    ),
)
def test_physical_port_projection_rejects_incompatible_physics_shapes(
    beta: jax.Array,
    reluctivity: jax.Array,
    omega: jax.Array,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        project_port_electromagnetic_fields_to_nodes(
            jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
            jnp.asarray(((0, 1, 2),), dtype=jnp.int64),
            jnp.asarray(((0, 2, 1),), dtype=jnp.int64),
            jnp.asarray(((1, 1, -1),), dtype=jnp.int8),
            jnp.zeros((3, 1), dtype=jnp.complex128),
            jnp.zeros((3, 1), dtype=jnp.complex128),
            beta,
            reluctivity,
            omega,
            np.arange(6, dtype=np.int64),
            edge_dof_count=3,
        )
