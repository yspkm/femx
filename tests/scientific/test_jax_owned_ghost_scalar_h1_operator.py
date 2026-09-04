from __future__ import annotations

import pytest
from tests.support import structured_unit_square_mesh

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from femx.backends.jax.operators import (  # noqa: E402
    assemble_scalar_h1_system,
    triangle_p1_diffusion_cell_matrices,
)
from femx.backends.jax.scalar_owned_ghost import (  # noqa: E402
    matrix_free_scalar_h1_matvec,
    owned_ghost_scalar_h1_matvec,
    prepare_scalar_h1_owned_ghost_topology,
)

pytestmark = [pytest.mark.scientific, pytest.mark.requires_jax]


def _free_nodes(coordinates: np.ndarray) -> np.ndarray:
    width = float(np.max(coordinates[:, 0]))
    constrained = np.isclose(coordinates[:, 0], 0.0) | np.isclose(coordinates[:, 0], width)
    return np.flatnonzero(~constrained).astype(np.int64)


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
    denominator = max(float(jnp.linalg.norm(expected)), np.finfo(np.float64).tiny)
    return float(jnp.linalg.norm(observed - expected)) / denominator


def test_heat_and_current_actions_match_dense_authority_across_partitions_and_energy() -> None:
    mesh = structured_unit_square_mesh(8)
    coordinates = np.asarray(mesh.geometry.coordinates) * np.asarray((2.0e-6, 0.8e-6))
    cells = np.asarray(mesh.topology.connectivity)
    facets = np.asarray(mesh.boundary_facets.connectivity)
    free_nodes = _free_nodes(coordinates)
    centroid_x = np.mean(coordinates[cells, 0], axis=1)
    coefficient_cases = {
        "heat": np.where(centroid_x < 1.0e-6, 148.0, 1.38),
        "current": np.where(centroid_x < 1.0e-6, 2.0e5, 5.0e4),
    }
    vector = jnp.asarray(np.random.default_rng(20260902).normal(size=free_nodes.shape[0]))

    for diffusion in coefficient_cases.values():
        dense = assemble_scalar_h1_system(
            jnp.asarray(coordinates),
            jnp.asarray(cells),
            jnp.asarray(diffusion),
            jnp.zeros(cells.shape[0]),
            jnp.asarray(facets),
            jnp.zeros(facets.shape[0]),
        ).stiffness
        dense_action = dense[jnp.ix_(jnp.asarray(free_nodes), jnp.asarray(free_nodes))] @ vector
        cell_stiffness = triangle_p1_diffusion_cell_matrices(
            jnp.asarray(coordinates),
            jnp.asarray(cells),
            jnp.asarray(diffusion),
        )
        maximum_partition_error = 0.0
        for partition_count in (1, 2, 3, 4):
            topology = prepare_scalar_h1_owned_ghost_topology(
                cells,
                _slab_cell_owners(coordinates, cells, partition_count),
                node_count=coordinates.shape[0],
                free_nodes=free_nodes,
                partition_count=partition_count,
            )
            serial = matrix_free_scalar_h1_matvec(cell_stiffness, topology, vector)
            partitioned = jax.jit(
                lambda matrix, x, bound=topology: owned_ghost_scalar_h1_matvec(
                    matrix,
                    bound,
                    x,
                )
            )(cell_stiffness, vector)
            maximum_partition_error = max(
                maximum_partition_error,
                _relative_difference(serial, dense_action),
                _relative_difference(partitioned, dense_action),
            )

        full_vector = jnp.zeros(coordinates.shape[0]).at[jnp.asarray(free_nodes)].set(vector)
        local_values = full_vector[jnp.asarray(cells)]
        cell_energy = jnp.einsum("ci,cij,cj->", local_values, cell_stiffness, local_values)
        global_energy = jnp.vdot(vector, dense_action).real

        assert maximum_partition_error < 2.0e-15
        np.testing.assert_allclose(global_energy, cell_energy, rtol=2.0e-15, atol=0.0)
        assert float(global_energy) > 0.0


def test_scalar_action_survives_owner_policy_and_cell_order_changes() -> None:
    mesh = structured_unit_square_mesh(6)
    coordinates = np.asarray(mesh.geometry.coordinates)
    cells = np.asarray(mesh.topology.connectivity)
    free_nodes = _free_nodes(coordinates)
    diffusion = 1.0 + 3.0 * np.mean(coordinates[cells, 0], axis=1)
    cell_owners = _slab_cell_owners(coordinates, cells, 3)
    baseline_topology = prepare_scalar_h1_owned_ghost_topology(
        cells,
        cell_owners,
        node_count=coordinates.shape[0],
        free_nodes=free_nodes,
        partition_count=3,
    )
    highest_owners = _highest_incident_owner(
        baseline_topology.cell_reduced_dofs,
        cell_owners,
        baseline_topology.free_dof_count,
    )
    topology = prepare_scalar_h1_owned_ghost_topology(
        cells,
        cell_owners,
        node_count=coordinates.shape[0],
        free_nodes=free_nodes,
        partition_count=3,
        dof_owners=highest_owners,
    )
    vector = jnp.asarray(np.random.default_rng(40).normal(size=free_nodes.shape[0]))
    stiffness = triangle_p1_diffusion_cell_matrices(
        jnp.asarray(coordinates),
        jnp.asarray(cells),
        jnp.asarray(diffusion),
    )
    baseline = owned_ghost_scalar_h1_matvec(stiffness, topology, vector)

    permutation = np.random.default_rng(41).permutation(cells.shape[0])
    permuted_topology = prepare_scalar_h1_owned_ghost_topology(
        cells[permutation],
        cell_owners[permutation],
        node_count=coordinates.shape[0],
        free_nodes=free_nodes,
        partition_count=3,
        dof_owners=highest_owners,
    )
    permuted = owned_ghost_scalar_h1_matvec(
        stiffness[jnp.asarray(permutation)],
        permuted_topology,
        vector,
    )

    np.testing.assert_allclose(permuted, baseline, rtol=2.0e-15, atol=2.0e-15)


def test_partitioned_scalar_action_vjp_matches_serial_and_central_difference() -> None:
    mesh = structured_unit_square_mesh(4)
    coordinates = jnp.asarray(mesh.geometry.coordinates)
    cells = np.asarray(mesh.topology.connectivity)
    free_nodes = _free_nodes(np.asarray(coordinates))
    cell_owners = _slab_cell_owners(np.asarray(coordinates), cells, 3)
    topology = prepare_scalar_h1_owned_ghost_topology(
        cells,
        cell_owners,
        node_count=coordinates.shape[0],
        free_nodes=free_nodes,
        partition_count=3,
    )
    rng = np.random.default_rng(20260903)
    diffusion = jnp.asarray(1.5 + rng.random(cells.shape[0]))
    vector = jnp.asarray(rng.normal(size=free_nodes.shape[0]))
    weights = jnp.asarray(rng.normal(size=free_nodes.shape[0]))

    def partitioned_objective(
        candidate_coordinates: jax.Array,
        candidate_diffusion: jax.Array,
        candidate_vector: jax.Array,
    ) -> jax.Array:
        stiffness = triangle_p1_diffusion_cell_matrices(
            candidate_coordinates,
            jnp.asarray(cells),
            candidate_diffusion,
        )
        return jnp.vdot(
            weights,
            owned_ghost_scalar_h1_matvec(stiffness, topology, candidate_vector),
        ).real

    def serial_objective(
        candidate_coordinates: jax.Array,
        candidate_diffusion: jax.Array,
        candidate_vector: jax.Array,
    ) -> jax.Array:
        stiffness = triangle_p1_diffusion_cell_matrices(
            candidate_coordinates,
            jnp.asarray(cells),
            candidate_diffusion,
        )
        return jnp.vdot(
            weights,
            matrix_free_scalar_h1_matvec(stiffness, topology, candidate_vector),
        ).real

    observed_value, observed_gradients = jax.jit(
        jax.value_and_grad(partitioned_objective, argnums=(0, 1, 2))
    )(coordinates, diffusion, vector)
    expected_value, expected_gradients = jax.jit(
        jax.value_and_grad(serial_objective, argnums=(0, 1, 2))
    )(coordinates, diffusion, vector)
    np.testing.assert_allclose(observed_value, expected_value, rtol=2.0e-15, atol=2.0e-15)
    for observed, expected in zip(observed_gradients, expected_gradients, strict=True):
        np.testing.assert_allclose(observed, expected, rtol=3.0e-14, atol=3.0e-14)

    coordinate_direction = jnp.asarray(rng.normal(size=coordinates.shape))
    coordinate_direction = coordinate_direction / jnp.linalg.norm(coordinate_direction)
    diffusion_direction = jnp.asarray(rng.normal(size=diffusion.shape))
    diffusion_direction = diffusion_direction / jnp.linalg.norm(diffusion_direction)
    vector_direction = jnp.asarray(rng.normal(size=vector.shape))
    vector_direction = vector_direction / jnp.linalg.norm(vector_direction)
    predicted = (
        jnp.vdot(observed_gradients[0], coordinate_direction)
        + jnp.vdot(observed_gradients[1], diffusion_direction)
        + jnp.vdot(observed_gradients[2], vector_direction)
    ).real
    step = 2.0e-6
    plus = partitioned_objective(
        coordinates + step * coordinate_direction,
        diffusion + step * diffusion_direction,
        vector + step * vector_direction,
    )
    minus = partitioned_objective(
        coordinates - step * coordinate_direction,
        diffusion - step * diffusion_direction,
        vector - step * vector_direction,
    )
    finite_difference = (plus - minus) / (2.0 * step)
    relative_error = jnp.abs(predicted - finite_difference) / jnp.maximum(
        jnp.abs(finite_difference),
        jnp.finfo(jnp.float64).tiny,
    )

    assert float(relative_error) < 2.0e-8
