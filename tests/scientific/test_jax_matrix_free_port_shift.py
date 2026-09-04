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
    MatrixFreePortSolvePolicy,
    apply_matrix_free_port_shift_invert,
    build_lossless_matrix_free_port_pencil,
    estimate_port_operator_storage,
    matrix_free_port_matvec,
    prepare_port_matrix_free_topology,
)
from femx.backends.jax.port_operator import (  # noqa: E402
    assemble_lossless_port_pencil,
    reduce_port_pencil,
)
from femx.physics.port_eigenmode import (  # noqa: E402
    VACUUM_PERMEABILITY_H_PER_M,
    VACUUM_PERMITTIVITY_F_PER_M,
)

pytestmark = [pytest.mark.scientific, pytest.mark.requires_jax]


def _port_mesh(intervals: int):
    mesh = structured_unit_square_mesh(intervals)
    coordinates = np.asarray(mesh.geometry.coordinates) * np.asarray((2.0e-6, 1.0e-6))
    cells = np.asarray(mesh.topology.connectivity)
    facets = np.asarray(mesh.boundary_facets.connectivity)
    local_edges = cells[:, ((0, 1), (1, 2), (2, 0))]
    signs = np.where(local_edges[:, :, 0] < local_edges[:, :, 1], 1, -1).astype(np.int8)
    edge_map = canonical_triangle_edge_map(cells, signs)
    partition = canonical_mixed_port_dof_partition(
        facets,
        edge_map,
        node_count=coordinates.shape[0],
    )
    topology = prepare_port_matrix_free_topology(
        cells,
        edge_map.cell_edge_dofs,
        partition.free_dofs,
        node_count=coordinates.shape[0],
        edge_dof_count=edge_map.dof_count,
    )
    return coordinates, cells, edge_map, partition, topology


def _automatic_elmer_shift(frequency_hz: float, maximum_relative_permittivity: float) -> float:
    maximum_permittivity = VACUUM_PERMITTIVITY_F_PER_M * maximum_relative_permittivity
    beta_limit = (
        2.0 * np.pi * frequency_hz * np.sqrt(maximum_permittivity * VACUUM_PERMEABILITY_H_PER_M)
    )
    return -(beta_limit * beta_limit)


def test_full_mixed_matrix_free_shift_invert_matches_dense_pencil_on_three_meshes() -> None:
    frequency_hz = 100.0e12
    policy = MatrixFreePortSolvePolicy(
        relative_tolerance=1.0e-11,
        restart=120,
        maximum_restart_cycles=100,
        maximum_relative_residual=5.0e-10,
    )
    storage_ratios: list[float] = []

    for intervals in (2, 4, 8):
        coordinates, cells, edge_map, partition, topology = _port_mesh(intervals)
        relative_permittivity = np.ones(cells.shape[0])
        relative_permeability = np.ones(cells.shape[0])
        pencil = build_lossless_matrix_free_port_pencil(
            jnp.asarray(coordinates),
            jnp.asarray(cells),
            jnp.asarray(edge_map.cell_edge_signs),
            jnp.asarray(topology.cell_reduced_dofs),
            jnp.asarray(relative_permittivity),
            jnp.asarray(relative_permeability),
            jnp.asarray(frequency_hz),
            free_dof_count=topology.free_dof_count,
        )
        dense = assemble_lossless_port_pencil(
            jnp.asarray(coordinates),
            jnp.asarray(cells),
            jnp.asarray(edge_map.cell_edge_dofs),
            jnp.asarray(edge_map.cell_edge_signs),
            jnp.asarray(relative_permittivity),
            jnp.asarray(relative_permeability),
            jnp.asarray(frequency_hz),
            edge_dof_count=edge_map.dof_count,
        )
        reduced = reduce_port_pencil(
            dense.stiffness,
            dense.mass,
            jnp.asarray(partition.free_dofs),
        )
        free_dof_count = topology.free_dof_count
        rng = np.random.default_rng(1000 + intervals)
        vector = jnp.asarray(rng.normal(size=free_dof_count))

        matrix_free_stiffness = matrix_free_port_matvec(
            pencil.stiffness,
            pencil.cell_reduced_dofs,
            vector,
        )
        matrix_free_mass = matrix_free_port_matvec(
            pencil.mass,
            pencil.cell_reduced_dofs,
            vector,
        )
        dense_stiffness_action = np.asarray(reduced.stiffness) @ np.asarray(vector)
        dense_mass_action = np.asarray(reduced.mass) @ np.asarray(vector)
        scalar_rows = partition.free_dofs < coordinates.shape[0]
        edge_rows = ~scalar_rows
        for rows in (scalar_rows, edge_rows):
            relative_action_error = np.linalg.norm(
                np.asarray(matrix_free_stiffness)[rows] - dense_stiffness_action[rows]
            ) / np.linalg.norm(dense_stiffness_action[rows])
            assert relative_action_error < 2.0e-14
        np.testing.assert_array_equal(np.asarray(matrix_free_mass)[scalar_rows], 0.0)
        relative_mass_error = np.linalg.norm(
            np.asarray(matrix_free_mass)[edge_rows] - dense_mass_action[edge_rows]
        ) / np.linalg.norm(dense_mass_action[edge_rows])
        assert relative_mass_error < 2.0e-14

        shift = _automatic_elmer_shift(frequency_hz, 1.0)
        result = apply_matrix_free_port_shift_invert(
            pencil,
            vector,
            jnp.asarray(shift),
            policy=policy,
        )
        assert bool(result.diagnostics.is_valid)
        assert float(result.diagnostics.equilibrated_relative_residual) < 5.0e-10

        dense_shifted = np.asarray(reduced.stiffness - shift * reduced.mass)
        dense_right_hand_side = np.asarray(reduced.mass) @ np.asarray(vector)
        left = np.asarray(result.equilibration.left_scale)
        right = np.asarray(result.equilibration.right_scale)
        dense_equilibrated = left[:, None] * dense_shifted * right[None, :]
        expected_equilibrated = np.linalg.solve(
            dense_equilibrated,
            left * dense_right_hand_side,
        )
        relative_solution_error = np.linalg.norm(
            np.asarray(result.equilibrated_solution) - expected_equilibrated
        ) / np.linalg.norm(expected_equilibrated)
        assert relative_solution_error < 5.0e-9

        storage = estimate_port_operator_storage(
            cell_count=cells.shape[0],
            free_dof_count=free_dof_count,
        )
        storage_ratios.append(storage.dense_to_matrix_free_ratio)

    assert storage_ratios[0] < storage_ratios[1] < storage_ratios[2]
    assert storage_ratios[2] > 9.0


def test_physical_matrix_free_shift_invert_adjoint_matches_independent_dense_difference() -> None:
    coordinates, cells, edge_map, partition, topology = _port_mesh(2)
    frequency_hz = 193.414e12
    baseline_maximum_relative_permittivity = 3.0
    shift = _automatic_elmer_shift(frequency_hz, baseline_maximum_relative_permittivity)
    free_dof_count = topology.free_dof_count
    edge_mask = partition.free_dofs >= coordinates.shape[0]
    vector = np.random.default_rng(2).normal(size=free_dof_count) * edge_mask
    weights = np.random.default_rng(3).normal(size=free_dof_count) * edge_mask
    cell_direction = np.linspace(0.2, 1.0, cells.shape[0])
    beta_scale_squared = -shift
    policy = MatrixFreePortSolvePolicy(
        relative_tolerance=1.0e-13,
        restart=free_dof_count,
        maximum_restart_cycles=4,
        maximum_relative_residual=1.0e-11,
    )

    coordinates_jax = jnp.asarray(coordinates)
    cells_jax = jnp.asarray(cells)
    signs_jax = jnp.asarray(edge_map.cell_edge_signs)
    local_map_jax = jnp.asarray(topology.cell_reduced_dofs)
    permeability_jax = jnp.ones(cells.shape[0])
    vector_jax = jnp.asarray(vector)
    weights_jax = jnp.asarray(weights)
    direction_jax = jnp.asarray(cell_direction)

    def objective(parameter: jax.Array) -> jax.Array:
        relative_permittivity = 2.1 + parameter * direction_jax
        pencil = build_lossless_matrix_free_port_pencil(
            coordinates_jax,
            cells_jax,
            signs_jax,
            local_map_jax,
            relative_permittivity,
            permeability_jax,
            jnp.asarray(frequency_hz),
            free_dof_count=free_dof_count,
        )
        result = apply_matrix_free_port_shift_invert(
            pencil,
            vector_jax,
            jnp.asarray(shift),
            policy=policy,
        )
        return beta_scale_squared * jnp.real(jnp.vdot(weights_jax, result.solution))

    parameter = jnp.asarray(0.4)
    baseline_value = objective(parameter)
    reverse_derivative = jax.jit(jax.grad(objective))(parameter)
    step = 1.0e-5
    matrix_free_difference = (objective(parameter + step) - objective(parameter - step)) / (
        2.0 * step
    )

    def dense_objective(parameter_value: float) -> float:
        relative_permittivity = 2.1 + parameter_value * cell_direction
        dense = assemble_lossless_port_pencil(
            coordinates_jax,
            cells_jax,
            jnp.asarray(edge_map.cell_edge_dofs),
            signs_jax,
            jnp.asarray(relative_permittivity),
            permeability_jax,
            jnp.asarray(frequency_hz),
            edge_dof_count=edge_map.dof_count,
        )
        reduced = reduce_port_pencil(
            dense.stiffness,
            dense.mass,
            jnp.asarray(partition.free_dofs),
        )
        shifted = np.asarray(reduced.stiffness - shift * reduced.mass)
        right_hand_side = np.asarray(reduced.mass) @ vector
        row_scale = 1.0 / np.sum(np.abs(shifted), axis=1)
        column_scale = 1.0 / np.sum(np.abs(shifted), axis=0)
        equilibrated = row_scale[:, None] * shifted * column_scale[None, :]
        largest = np.max(np.abs(equilibrated))
        row_scale = row_scale / largest
        equilibrated = row_scale[:, None] * shifted * column_scale[None, :]
        solution = column_scale * np.linalg.solve(
            equilibrated,
            row_scale * right_hand_side,
        )
        return float(beta_scale_squared * np.vdot(weights, solution).real)

    dense_baseline = dense_objective(float(parameter))
    dense_difference = (
        dense_objective(float(parameter) + step) - dense_objective(float(parameter) - step)
    ) / (2.0 * step)

    np.testing.assert_allclose(baseline_value, dense_baseline, rtol=2.0e-13, atol=2.0e-13)
    np.testing.assert_allclose(
        reverse_derivative,
        matrix_free_difference,
        rtol=3.0e-9,
        atol=3.0e-9,
    )
    np.testing.assert_allclose(
        reverse_derivative,
        dense_difference,
        rtol=3.0e-9,
        atol=3.0e-9,
    )
