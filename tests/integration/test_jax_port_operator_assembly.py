from __future__ import annotations

import itertools

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from tests.support import structured_unit_square_mesh  # noqa: E402

from femx.backends._hcurl import (  # noqa: E402
    canonical_mixed_port_dof_partition,
    canonical_triangle_edge_map,
)
from femx.backends.jax.port_operator import (  # noqa: E402
    assemble_lossless_port_pencil,
    lossless_port_coefficients,
    reduce_port_pencil,
    triangle_port_local_pencil,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_jax]


def _edge_map(cells: np.ndarray):
    local_edges = cells[:, ((0, 1), (1, 2), (2, 0))]
    signs = np.where(local_edges[:, :, 0] < local_edges[:, :, 1], 1, -1).astype(np.int8)
    return canonical_triangle_edge_map(cells, signs)


def _assemble(
    coordinates: np.ndarray,
    cells: np.ndarray,
    epsilon: np.ndarray,
    permeability: np.ndarray,
):
    edge_map = _edge_map(cells)
    pencil = assemble_lossless_port_pencil(
        jnp.asarray(coordinates),
        jnp.asarray(cells),
        jnp.asarray(edge_map.cell_edge_dofs),
        jnp.asarray(edge_map.cell_edge_signs),
        jnp.asarray(epsilon),
        jnp.asarray(permeability),
        jnp.asarray(1.9e14),
        edge_dof_count=edge_map.dof_count,
    )
    return edge_map, pencil


def _assert_relative_matrix_close(observed: jax.Array, expected: jax.Array) -> None:
    observed_array = np.asarray(observed)
    expected_array = np.asarray(expected)
    relative_error = np.linalg.norm(observed_array - expected_array) / np.linalg.norm(
        expected_array
    )
    assert relative_error < 2.0e-15


def test_global_mixed_pencil_is_invariant_to_cell_node_permutations_and_order() -> None:
    coordinates = np.asarray(((0.0, 0.0), (1.1, 0.1), (1.0, 1.2), (-0.2, 0.9)))
    base_cells = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    epsilon = np.asarray((2.0, 5.0))
    permeability = np.asarray((1.0, 1.3))
    _, reference = _assemble(coordinates, base_cells, epsilon, permeability)

    permutations = tuple(itertools.permutations(range(3)))
    for first_permutation, second_permutation in itertools.product(permutations, repeat=2):
        permuted = np.stack(
            (
                base_cells[0, list(first_permutation)],
                base_cells[1, list(second_permutation)],
            )
        )
        _, observed = _assemble(coordinates, permuted, epsilon, permeability)
        _assert_relative_matrix_close(observed.stiffness, reference.stiffness)
        _assert_relative_matrix_close(observed.mass, reference.mass)

    _, reversed_cells = _assemble(
        coordinates,
        base_cells[::-1],
        epsilon[::-1],
        permeability[::-1],
    )
    _assert_relative_matrix_close(reversed_cells.stiffness, reference.stiffness)
    _assert_relative_matrix_close(reversed_cells.mass, reference.mass)


def test_global_scatter_matches_an_independent_cell_loop() -> None:
    coordinates = np.asarray(((0.0, 0.0), (1.0, 0.1), (1.1, 1.0), (-0.1, 0.8)))
    cells = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    relative_permittivity = np.asarray((2.3, 4.1))
    relative_permeability = np.asarray((1.0, 1.2))
    frequency = 1.6e14
    edge_map = _edge_map(cells)
    epsilon, nu = lossless_port_coefficients(
        jnp.asarray(relative_permittivity),
        jnp.asarray(relative_permeability),
    )
    local = triangle_port_local_pencil(
        jnp.asarray(coordinates),
        jnp.asarray(cells),
        jnp.asarray(edge_map.cell_edge_signs),
        epsilon,
        nu,
        jnp.asarray(2.0 * np.pi * frequency),
    )
    direct = assemble_lossless_port_pencil(
        jnp.asarray(coordinates),
        jnp.asarray(cells),
        jnp.asarray(edge_map.cell_edge_dofs),
        jnp.asarray(edge_map.cell_edge_signs),
        jnp.asarray(relative_permittivity),
        jnp.asarray(relative_permeability),
        jnp.asarray(frequency),
        edge_dof_count=edge_map.dof_count,
    )
    total_dofs = coordinates.shape[0] + edge_map.dof_count
    expected_stiffness = np.zeros((total_dofs, total_dofs))
    expected_mass = np.zeros_like(expected_stiffness)
    for cell, edge_dofs, cell_stiffness, cell_mass in zip(
        cells,
        edge_map.cell_edge_dofs,
        np.asarray(local.stiffness),
        np.asarray(local.mass),
        strict=True,
    ):
        local_dofs = np.concatenate((cell, coordinates.shape[0] + edge_dofs))
        expected_stiffness[np.ix_(local_dofs, local_dofs)] += cell_stiffness
        expected_mass[np.ix_(local_dofs, local_dofs)] += cell_mass

    np.testing.assert_allclose(direct.stiffness, expected_stiffness, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(direct.mass, expected_mass, rtol=0.0, atol=0.0)


def test_pec_reduction_extracts_only_free_principal_pencil_under_jit() -> None:
    mesh = structured_unit_square_mesh(2)
    coordinates = np.asarray(mesh.geometry.coordinates)
    cells = np.asarray(mesh.topology.connectivity)
    facets = np.asarray(mesh.boundary_facets.connectivity)
    edge_map = _edge_map(cells)
    partition = canonical_mixed_port_dof_partition(
        facets,
        edge_map,
        node_count=coordinates.shape[0],
    )
    epsilon = jnp.linspace(2.0, 4.0, cells.shape[0])
    permeability = jnp.ones(cells.shape[0])
    assemble = jax.jit(assemble_lossless_port_pencil, static_argnames=("edge_dof_count",))
    pencil = assemble(
        jnp.asarray(coordinates),
        jnp.asarray(cells),
        jnp.asarray(edge_map.cell_edge_dofs),
        jnp.asarray(edge_map.cell_edge_signs),
        epsilon,
        permeability,
        jnp.asarray(1.5e14),
        edge_dof_count=edge_map.dof_count,
    )
    reduced = jax.jit(reduce_port_pencil)(
        pencil.stiffness,
        pencil.mass,
        jnp.asarray(partition.free_dofs),
    )

    expected_indices = np.ix_(partition.free_dofs, partition.free_dofs)
    np.testing.assert_allclose(reduced.stiffness, np.asarray(pencil.stiffness)[expected_indices])
    np.testing.assert_allclose(reduced.mass, np.asarray(pencil.mass)[expected_indices])
    np.testing.assert_array_equal(reduced.full_dofs, partition.free_dofs)
    assert set(partition.constrained_dofs).isdisjoint(set(np.asarray(reduced.full_dofs)))

    scalar_free = np.flatnonzero(partition.free_dofs < coordinates.shape[0])
    edge_free = np.flatnonzero(partition.free_dofs >= coordinates.shape[0])
    assert scalar_free.size > 0
    assert edge_free.size > 0
    np.testing.assert_allclose(reduced.mass[scalar_free, :], 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(reduced.mass[:, scalar_free], 0.0, rtol=0.0, atol=0.0)
    edge_mass = np.asarray(reduced.mass)[np.ix_(edge_free, edge_free)]
    assert np.all(np.linalg.eigvalsh(edge_mass) > 0.0)


def test_global_pencil_reverse_material_derivative_matches_central_difference() -> None:
    coordinates = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    cells_np = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    edge_map = _edge_map(cells_np)
    cells = jnp.asarray(cells_np)
    edge_dofs = jnp.asarray(edge_map.cell_edge_dofs)
    signs = jnp.asarray(edge_map.cell_edge_signs)
    weights = jnp.arange((4 + edge_map.dof_count) ** 2, dtype=jnp.float64).reshape(
        4 + edge_map.dof_count,
        4 + edge_map.dof_count,
    )
    weights = weights / jnp.max(weights)

    def objective(relative_permittivity: jax.Array) -> jax.Array:
        pencil = assemble_lossless_port_pencil(
            coordinates,
            cells,
            edge_dofs,
            signs,
            relative_permittivity,
            jnp.asarray((1.0, 1.0)),
            jnp.asarray(1.7e14),
            edge_dof_count=edge_map.dof_count,
        )
        return jnp.sum(pencil.stiffness * weights)

    parameters = jnp.asarray((2.2, 3.4))
    direction = jnp.asarray((0.3, -0.2))
    gradient = jax.jit(jax.grad(objective))(parameters)
    step = 1.0e-5
    central = (
        objective(parameters + step * direction) - objective(parameters - step * direction)
    ) / (2.0 * step)
    reverse = jnp.vdot(gradient, direction)

    np.testing.assert_allclose(reverse, central, rtol=2.0e-10, atol=5.0e-7)
