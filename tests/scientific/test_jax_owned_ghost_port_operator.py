from __future__ import annotations

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
from femx.backends.jax.port_matrix_free import (  # noqa: E402
    build_lossless_matrix_free_port_pencil,
    matrix_free_port_matvec,
    prepare_port_matrix_free_topology,
)
from femx.backends.jax.port_owned_ghost import (  # noqa: E402
    owned_ghost_port_generalized_residual,
    owned_ghost_port_matvec,
    prepare_owned_ghost_port_topology,
)

pytestmark = [pytest.mark.scientific, pytest.mark.requires_jax]


def _silicon_port_case(intervals: int = 4):
    mesh = structured_unit_square_mesh(intervals)
    coordinates = np.asarray(mesh.geometry.coordinates) * np.asarray((2.0e-6, 1.0e-6))
    cells = np.asarray(mesh.topology.connectivity)
    facets = np.asarray(mesh.boundary_facets.connectivity)
    local_edges = cells[:, ((0, 1), (1, 2), (2, 0))]
    signs = np.where(local_edges[:, :, 0] < local_edges[:, :, 1], 1, -1).astype(np.int8)
    edge_map = canonical_triangle_edge_map(cells, signs)
    pec = canonical_mixed_port_dof_partition(
        facets,
        edge_map,
        node_count=coordinates.shape[0],
    )
    serial_topology = prepare_port_matrix_free_topology(
        cells,
        edge_map.cell_edge_dofs,
        pec.free_dofs,
        node_count=coordinates.shape[0],
        edge_dof_count=edge_map.dof_count,
    )
    centroids = np.mean(coordinates[cells], axis=1)
    silicon = (np.abs(centroids[:, 0] - 1.0e-6) <= 0.55e-6) & (
        np.abs(centroids[:, 1] - 0.5e-6) <= 0.24e-6
    )
    relative_permittivity = np.where(silicon, 3.48**2, 1.444**2)
    pencil = build_lossless_matrix_free_port_pencil(
        jnp.asarray(coordinates),
        jnp.asarray(cells),
        jnp.asarray(edge_map.cell_edge_signs),
        jnp.asarray(serial_topology.cell_reduced_dofs),
        jnp.asarray(relative_permittivity),
        jnp.ones(cells.shape[0]),
        jnp.asarray(193.414e12),
        free_dof_count=serial_topology.free_dof_count,
    )
    return coordinates, cells, serial_topology, pencil


def _slab_cell_owners(coordinates: np.ndarray, cells: np.ndarray, count: int) -> np.ndarray:
    normalized_x = np.mean(coordinates[cells, 0], axis=1) / np.max(coordinates[:, 0])
    return np.minimum((count * normalized_x).astype(np.int64), count - 1)


def _highest_incident_owner(
    cell_map: np.ndarray,
    cell_owners: np.ndarray,
    free_dof_count: int,
) -> np.ndarray:
    return np.asarray(
        [
            np.max(cell_owners[np.any(cell_map == global_dof, axis=1)])
            for global_dof in range(free_dof_count)
        ],
        dtype=np.int64,
    )


def _relative_difference(observed: jax.Array, expected: jax.Array) -> float:
    return float(jnp.linalg.norm(observed - expected) / jnp.linalg.norm(expected))


def test_physical_port_action_is_invariant_across_one_to_four_owned_ghost_partitions() -> None:
    coordinates, cells, serial_topology, pencil = _silicon_port_case()
    rng = np.random.default_rng(20260901)
    vector = jnp.asarray(
        rng.normal(size=serial_topology.free_dof_count)
        + 1j * rng.normal(size=serial_topology.free_dof_count)
    )
    shift = jnp.asarray(-2.1e14)
    matrices = (
        pencil.stiffness,
        pencil.mass,
        pencil.stiffness - shift * pencil.mass,
    )
    serial_actions = tuple(
        matrix_free_port_matvec(matrix, pencil.cell_reduced_dofs, vector) for matrix in matrices
    )
    maximum_relative_difference = 0.0

    for partition_count in (1, 2, 3, 4):
        cell_owners = _slab_cell_owners(coordinates, cells, partition_count)
        topology = prepare_owned_ghost_port_topology(
            serial_topology.cell_reduced_dofs,
            cell_owners,
            free_dof_count=serial_topology.free_dof_count,
            partition_count=partition_count,
        )
        for matrix, expected in zip(matrices, serial_actions, strict=True):
            observed = jax.jit(
                lambda local_matrix, x, bound_topology=topology: owned_ghost_port_matvec(
                    local_matrix,
                    bound_topology,
                    x,
                )
            )(matrix, vector)
            maximum_relative_difference = max(
                maximum_relative_difference,
                _relative_difference(observed, expected),
            )

    assert maximum_relative_difference < 2.0e-15


def test_physical_port_residual_survives_owner_policy_and_cell_order_changes() -> None:
    coordinates, cells, serial_topology, pencil = _silicon_port_case()
    rng = np.random.default_rng(32)
    vector = jnp.asarray(
        rng.normal(size=serial_topology.free_dof_count)
        + 1j * rng.normal(size=serial_topology.free_dof_count)
    )
    eigenvalue = jnp.asarray(-1.8e14 + 2.0e5j)
    cell_owners = _slab_cell_owners(coordinates, cells, 3)
    highest_owners = _highest_incident_owner(
        serial_topology.cell_reduced_dofs,
        cell_owners,
        serial_topology.free_dof_count,
    )
    topology = prepare_owned_ghost_port_topology(
        serial_topology.cell_reduced_dofs,
        cell_owners,
        free_dof_count=serial_topology.free_dof_count,
        partition_count=3,
        dof_owners=highest_owners,
    )
    baseline = owned_ghost_port_generalized_residual(
        pencil.stiffness,
        pencil.mass,
        topology,
        vector,
        eigenvalue,
    )

    permutation = np.random.default_rng(11).permutation(cells.shape[0])
    permuted_topology = prepare_owned_ghost_port_topology(
        serial_topology.cell_reduced_dofs[permutation],
        cell_owners[permutation],
        free_dof_count=serial_topology.free_dof_count,
        partition_count=3,
        dof_owners=highest_owners,
    )
    permuted = owned_ghost_port_generalized_residual(
        pencil.stiffness[jnp.asarray(permutation)],
        pencil.mass[jnp.asarray(permutation)],
        permuted_topology,
        vector,
        eigenvalue,
    )
    serial_residual = matrix_free_port_matvec(
        pencil.stiffness,
        pencil.cell_reduced_dofs,
        vector,
    ) - eigenvalue * matrix_free_port_matvec(
        pencil.mass,
        pencil.cell_reduced_dofs,
        vector,
    )
    serial_bound = matrix_free_port_matvec(
        jnp.abs(pencil.stiffness) + jnp.abs(eigenvalue) * jnp.abs(pencil.mass),
        pencil.cell_reduced_dofs,
        jnp.abs(vector),
    )

    for observed in (baseline, permuted):
        assert _relative_difference(observed.residual, serial_residual) < 2.0e-15
        assert _relative_difference(observed.row_magnitude_bound, serial_bound) < 2.0e-15
    np.testing.assert_allclose(
        permuted.relative_scaled_residual,
        baseline.relative_scaled_residual,
        rtol=2.0e-15,
        atol=0.0,
    )
