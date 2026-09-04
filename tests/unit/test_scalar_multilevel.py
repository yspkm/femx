from __future__ import annotations

from dataclasses import replace

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh  # noqa: E402
from tests.support import structured_unit_square_mesh  # noqa: E402

from femx.backends.jax.operators import (  # noqa: E402
    triangle_p1_diffusion_cell_matrices,
)
from femx.backends.jax.scalar_cg import (  # noqa: E402
    ScalarH1CGPolicy,
    assert_scalar_h1_cg_converged,
)
from femx.backends.jax.scalar_collective import (  # noqa: E402
    pack_collective_scalar_h1_cell_matrix,
    pack_collective_scalar_h1_owned_mask,
    prepare_collective_scalar_h1_layout,
)
from femx.backends.jax.scalar_multilevel import (  # noqa: E402
    SCALAR_H1_MULTILEVEL_HIERARCHY_SCHEMA,
    SCALAR_H1_NESTED_PROLONGATION_SCHEMA,
    ScalarH1MultilevelPolicy,
    build_packed_scalar_h1_multilevel_runtime,
    build_validation_collective_scalar_h1_multilevel_pcg,
    pack_scalar_h1_multilevel_transfer,
    pack_scalar_h1_multilevel_transfer_host,
    prepare_scalar_h1_multilevel_hierarchy,
    prepare_scalar_h1_nested_prolongation,
)
from femx.backends.jax.scalar_owned_ghost import (  # noqa: E402
    prepare_scalar_h1_owned_ghost_topology,
)
from femx.core.errors import ContractError  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def _mesh_arrays(intervals: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = structured_unit_square_mesh(intervals)
    coordinates = np.asarray(mesh.geometry.coordinates)
    cells = np.asarray(mesh.topology.connectivity)
    boundary = (
        np.isclose(coordinates[:, 0], 0.0)
        | np.isclose(coordinates[:, 0], 1.0)
        | np.isclose(coordinates[:, 1], 0.0)
        | np.isclose(coordinates[:, 1], 1.0)
    )
    return coordinates, cells, np.flatnonzero(~boundary).astype(np.int64)


def _layout(intervals: int):
    coordinates, cells, free_nodes = _mesh_arrays(intervals)
    topology = prepare_scalar_h1_owned_ghost_topology(
        cells,
        np.zeros(cells.shape[0], dtype=np.int64),
        node_count=coordinates.shape[0],
        free_nodes=free_nodes,
        partition_count=1,
    )
    return prepare_collective_scalar_h1_layout(topology)


def _device_mesh() -> Mesh:
    return Mesh(np.asarray((jax.devices("cpu")[0],), dtype=object), ("partition",))


def _prolongation(fine: int, coarse: int):
    fine_coordinates, _, fine_free = _mesh_arrays(fine)
    coarse_coordinates, coarse_cells, coarse_free = _mesh_arrays(coarse)
    return prepare_scalar_h1_nested_prolongation(
        fine_coordinates,
        fine_free,
        coarse_coordinates,
        coarse_cells,
        coarse_free,
    )


def _hierarchy(intervals: int = 8):
    layout = _layout(intervals)
    return layout, prepare_scalar_h1_multilevel_hierarchy(
        layout,
        (_prolongation(intervals, intervals // 2), _prolongation(intervals // 2, 2)),
        maximum_replicated_dofs=16,
    )


def _cell_operator(intervals: int, contrast: float = 1.0) -> jax.Array:
    coordinates, cells, _ = _mesh_arrays(intervals)
    centroids = np.mean(coordinates[cells], axis=1)
    coefficients = np.where(centroids[:, 0] < 0.5, 1.0, contrast)
    return triangle_p1_diffusion_cell_matrices(
        jnp.asarray(coordinates),
        jnp.asarray(cells),
        jnp.asarray(coefficients),
    )


def _cell_rhs(intervals: int) -> jax.Array:
    coordinates, cells, _ = _mesh_arrays(intervals)
    vertices = coordinates[cells]
    edges_1 = vertices[:, 1] - vertices[:, 0]
    edges_2 = vertices[:, 2] - vertices[:, 0]
    areas = 0.5 * np.abs(edges_1[:, 0] * edges_2[:, 1] - edges_1[:, 1] * edges_2[:, 0])
    return jnp.asarray(np.repeat((areas / 3.0)[:, None], 3, axis=1))


def _dense_free_matrix(intervals: int, cell_stiffness: jax.Array) -> np.ndarray:
    _, cells, free_nodes = _mesh_arrays(intervals)
    full_to_reduced = np.full((intervals + 1) ** 2, free_nodes.size, dtype=np.int64)
    full_to_reduced[free_nodes] = np.arange(free_nodes.size)
    result = np.zeros((free_nodes.size, free_nodes.size), dtype=np.float64)
    for cell, local in zip(cells, np.asarray(cell_stiffness), strict=True):
        reduced = full_to_reduced[cell]
        for first in range(3):
            for second in range(3):
                if reduced[first] < free_nodes.size and reduced[second] < free_nodes.size:
                    result[reduced[first], reduced[second]] += local[first, second]
    return result


def test_nested_p1_prolongation_is_canonical_sparse_and_chain_bound() -> None:
    level = _prolongation(4, 2)
    assert level.schema_version == SCALAR_H1_NESTED_PROLONGATION_SCHEMA
    assert level.dense().shape == (9, 1)
    assert level.ambiguity_count > 0
    assert len(level.digest()) == 64
    assert not level.column_indices.flags.writeable
    assert not level.weights.flags.writeable
    center = np.flatnonzero(
        np.all(np.isclose(_mesh_arrays(4)[0][_mesh_arrays(4)[2]], (0.5, 0.5)), axis=1)
    )
    np.testing.assert_array_equal(level.dense()[center], 1.0)

    layout, hierarchy = _hierarchy()
    assert hierarchy.schema_version == SCALAR_H1_MULTILEVEL_HIERARCHY_SCHEMA
    assert hierarchy.level_dof_counts == (49, 9, 1)
    assert hierarchy.layout_sha256 == layout.digest()
    assert len(hierarchy.digest()) == 64


def test_multilevel_galerkin_setup_and_inverse_match_dense_authority() -> None:
    intervals = 8
    layout, hierarchy = _hierarchy(intervals)
    cell_stiffness = _cell_operator(intervals, contrast=100.0)
    transfer = pack_scalar_h1_multilevel_transfer(
        layout,
        hierarchy,
        value_dtype=np.float64,
    )
    runtime = build_packed_scalar_h1_multilevel_runtime(
        layout,
        _device_mesh(),
        hierarchy,
        ScalarH1MultilevelPolicy(maximum_coarse_condition_number=1.0e8),
        transfer,
    )
    state = runtime.setup(
        pack_collective_scalar_h1_cell_matrix(layout, cell_stiffness),
        jnp.asarray(layout.transport.cell_local_dofs),
        pack_collective_scalar_h1_owned_mask(layout),
    )
    assert bool(state.valid)
    assert len(state.coarse_matrices) == 2
    fine = _dense_free_matrix(intervals, cell_stiffness)
    first = hierarchy.prolongations[0].dense()
    second = hierarchy.prolongations[1].dense()
    np.testing.assert_allclose(state.coarse_matrices[0], first.T @ fine @ first, rtol=2e-14)
    np.testing.assert_allclose(
        state.coarse_matrices[1],
        second.T @ first.T @ fine @ first @ second,
        rtol=2e-14,
    )

    columns = []
    for column in range(layout.topology.free_dof_count):
        residual = np.zeros((1, layout.owned_dof_capacity))
        residual[0, column] = 1.0
        columns.append(np.asarray(runtime.apply(state, jnp.asarray(residual)))[0])
    inverse = np.column_stack(columns)
    np.testing.assert_allclose(inverse, inverse.T, rtol=1e-13, atol=1e-13)
    assert np.min(np.linalg.eigvalsh(inverse)) > 0.0


def test_multilevel_transfer_can_cross_an_explicit_distributed_input_boundary() -> None:
    layout, hierarchy = _hierarchy()
    host = pack_scalar_h1_multilevel_transfer_host(
        layout,
        hierarchy,
        value_dtype=np.float64,
    )
    assert host.owner_columns.dtype == np.int32
    assert host.owner_weights.dtype == np.float64
    assert not host.owner_columns.flags.writeable
    assert not host.owner_weights.flags.writeable
    transfer = pack_scalar_h1_multilevel_transfer(
        layout,
        hierarchy,
        value_dtype=np.float64,
    )
    runtime = build_packed_scalar_h1_multilevel_runtime(
        layout,
        _device_mesh(),
        hierarchy,
        ScalarH1MultilevelPolicy(maximum_coarse_condition_number=1.0e8),
    )
    stiffness = pack_collective_scalar_h1_cell_matrix(layout, _cell_operator(8, contrast=10.0))
    mapping = jnp.asarray(layout.transport.cell_local_dofs)
    mask = pack_collective_scalar_h1_owned_mask(layout)
    state = jax.jit(runtime.setup)(stiffness, mapping, mask, transfer)
    residual = jnp.ones(mask.shape, dtype=stiffness.dtype)
    correction = jax.jit(runtime.apply)(state, residual, transfer)
    assert bool(state.valid)
    assert bool(jnp.all(jnp.isfinite(correction)))
    with pytest.raises(ContractError, match="requires one explicit packed transfer"):
        runtime.setup(stiffness, mapping, mask)

    bound = build_packed_scalar_h1_multilevel_runtime(
        layout,
        _device_mesh(),
        hierarchy,
        ScalarH1MultilevelPolicy(maximum_coarse_condition_number=1.0e8),
        transfer,
    )
    with pytest.raises(ContractError, match="cannot receive another"):
        bound.setup(stiffness, mapping, mask, transfer)


def test_multilevel_pcg_matches_dense_and_keeps_residual_defined_gradient() -> None:
    intervals = 8
    layout, hierarchy = _hierarchy(intervals)
    solve = jax.jit(
        build_validation_collective_scalar_h1_multilevel_pcg(
            layout,
            _device_mesh(),
            hierarchy,
            ScalarH1MultilevelPolicy(maximum_coarse_condition_number=1.0e8),
            ScalarH1CGPolicy(1.0e-11, 1.0e-14, 200),
            value_dtype=np.float64,
        )
    )
    rhs = _cell_rhs(intervals)
    base_stiffness = _cell_operator(intervals, contrast=10.0)
    result = solve(base_stiffness, rhs)
    assert_scalar_h1_cg_converged(result)
    dense = _dense_free_matrix(intervals, base_stiffness)
    np.testing.assert_allclose(
        result.solution,
        np.linalg.solve(dense, np.asarray(result.right_hand_side)),
        rtol=2.0e-10,
        atol=2.0e-12,
    )

    weights = jnp.linspace(0.2, 1.0, layout.topology.free_dof_count)

    def objective(scale: jax.Array) -> jax.Array:
        return jnp.vdot(weights, solve(scale * base_stiffness, rhs).solution).real

    value, derivative = jax.value_and_grad(objective)(jnp.asarray(1.1))
    step = 2.0e-5
    finite_difference = (objective(1.1 + step) - objective(1.1 - step)) / (2.0 * step)
    assert np.isfinite(float(value))
    np.testing.assert_allclose(derivative, finite_difference, rtol=3.0e-7, atol=3.0e-9)


def test_multilevel_records_fail_closed_on_identity_and_policy_drift() -> None:
    layout, hierarchy = _hierarchy()
    level = hierarchy.prolongations[0]
    with pytest.raises(ContractError, match="schema"):
        replace(level, schema_version="femx.invalid/v2")
    with pytest.raises(ContractError, match="exact intermediate space"):
        prepare_scalar_h1_multilevel_hierarchy(
            layout,
            (
                level,
                replace(_prolongation(4, 2), fine_source_sha256="0" * 64),
            ),
            maximum_replicated_dofs=16,
        )
    with pytest.raises(ContractError, match="DOF limit"):
        replace(hierarchy, maximum_replicated_dofs=8)
    with pytest.raises(ContractError, match="positive"):
        ScalarH1MultilevelPolicy(diagonal_weight=0.0)
