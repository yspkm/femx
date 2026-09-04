from __future__ import annotations

from dataclasses import replace

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

import femx.backends.jax.collective as collective_module  # noqa: E402
from femx.backends.jax.collective import (  # noqa: E402
    build_packed_collective_cell_gather,
    build_packed_collective_row_assembly,
    pack_collective_cell_vector,
    prepare_collective_layout,
)
from femx.backends.jax.elements.tetrahedron_h1 import (  # noqa: E402
    tetrahedron_p1_cell_nodal_load_vectors,
    tetrahedron_p1_diffusion_cell_matrices,
)
from femx.backends.jax.scalar_cg import (  # noqa: E402
    ScalarH1CGPolicy,
    ScalarH1JacobiPolicy,
    _build_packed_scalar_h1_dot,
    build_packed_collective_scalar_h1_cg,
    build_packed_scalar_h1_jacobi_preconditioner_factory,
)
from femx.backends.jax.scalar_collective import (  # noqa: E402
    SCALAR_H1_TET4_COLLECTIVE_LAYOUT_SCHEMA,
    ScalarH1CollectiveLayout,
    build_packed_collective_scalar_h1_cell_gather,
    build_packed_collective_scalar_h1_rhs_assembly,
    pack_collective_scalar_h1_cell_matrix,
    pack_collective_scalar_h1_cell_vector,
    pack_collective_scalar_h1_owned_mask,
    pack_collective_scalar_h1_owned_vector,
    prepare_collective_scalar_h1_layout,
    prepare_scalar_h1_boundary_facet_map,
    reconstruct_scalar_h1_state,
    scalar_h1_reduced_cell_rhs,
    triangle_p1_scalar_cell_load_vectors,
    triangle_p1_scalar_cell_nodal_load_vectors,
)
from femx.backends.jax.scalar_owned_ghost import (  # noqa: E402
    prepare_scalar_h1_owned_ghost_topology,
)
from femx.core.errors import ContractError  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


COORDINATES = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)))
CELLS = np.asarray(((0, 1, 3), (0, 3, 2)), dtype=np.int64)
FACETS = np.asarray(((0, 1), (1, 3), (3, 2), (2, 0)), dtype=np.int64)
FREE_NODES = np.asarray((1, 2, 3), dtype=np.int64)


def _topology(*, partition_count: int = 1):
    owners = np.asarray((0, 0) if partition_count == 1 else (0, 1), dtype=np.int64)
    return prepare_scalar_h1_owned_ghost_topology(
        CELLS,
        owners,
        node_count=4,
        free_nodes=FREE_NODES,
        partition_count=partition_count,
    )


def _layout(*, partition_count: int = 1):
    return prepare_collective_scalar_h1_layout(_topology(partition_count=partition_count))


def _mesh() -> Mesh:
    return Mesh(np.asarray((jax.devices("cpu")[0],), dtype=object), ("partition",))


def _boundary_map():
    return prepare_scalar_h1_boundary_facet_map(CELLS, FACETS, node_count=4)


def test_boundary_map_record_rejects_noncanonical_mesh_and_derived_identity() -> None:
    boundary_map = _boundary_map()
    with pytest.raises(ContractError, match="node count must be an integer"):
        replace(boundary_map, node_count=True)
    with pytest.raises(ContractError, match="node count must be positive"):
        replace(boundary_map, node_count=0)
    with pytest.raises(ContractError, match="at least one triangle"):
        replace(boundary_map, cells=np.empty((0, 3), dtype=np.int64))
    with pytest.raises(ContractError, match="cells contain an out-of-range node"):
        replace(boundary_map, cells=np.asarray(((0, 1, 4), (0, 3, 2))))
    with pytest.raises(ContractError, match="facets contain an out-of-range node"):
        replace(boundary_map, boundary_facets=np.asarray(((0, 4),)))
    with pytest.raises(ContractError, match="cannot repeat a node"):
        replace(boundary_map, boundary_facets=np.asarray(((0, 0),)))
    with pytest.raises(ContractError, match="facet cells disagree"):
        replace(boundary_map, facet_cells=np.asarray((1, 0, 1, 0)))
    wrong_local = boundary_map.facet_local_nodes.copy()
    wrong_local[0] = wrong_local[0, ::-1]
    with pytest.raises(ContractError, match="local nodes disagree"):
        replace(boundary_map, facet_local_nodes=wrong_local)


def test_scalar_collective_layout_rejects_wrong_record_types_and_transport_identity() -> None:
    layout = _layout()
    with pytest.raises(ContractError, match="scalar owned/ghost topology"):
        replace(layout, topology=object())
    with pytest.raises(ContractError, match="collective transport"):
        replace(layout, transport=object())
    with pytest.raises(ContractError, match="exact scalar topology"):
        replace(layout, transport=_layout().transport)
    wrong_schema = prepare_collective_layout(
        layout.topology.owned_ghost,
        schema_version="femx.test.scalar_collective/wrong",
    )
    with pytest.raises(ContractError, match="schema disagrees"):
        ScalarH1CollectiveLayout(layout.topology, wrong_schema)

    with pytest.raises(ContractError, match="OwnedGhostTopology"):
        replace(layout.transport, topology=object())
    with pytest.raises(ContractError, match="expected schema version"):
        replace(layout.transport, expected_schema_version="")
    with pytest.raises(ContractError, match="OwnedGhostTopology"):
        prepare_collective_layout(object(), schema_version="femx.test/v1")  # type: ignore[arg-type]


def test_tet4_collective_layout_and_reduced_rhs_keep_four_local_rows() -> None:
    coordinates = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
        ),
        dtype=jnp.float64,
    )
    cells = np.asarray(((0, 1, 2, 3), (1, 2, 3, 4)), dtype=np.int64)
    topology = prepare_scalar_h1_owned_ghost_topology(
        cells,
        np.asarray((0, 0), dtype=np.int64),
        node_count=5,
        free_nodes=np.asarray((1, 2, 3), dtype=np.int64),
        partition_count=1,
    )
    layout = prepare_collective_scalar_h1_layout(topology)
    stiffness = tetrahedron_p1_diffusion_cell_matrices(
        coordinates,
        jnp.asarray(cells),
        jnp.asarray((2.0, 3.0), dtype=jnp.float64),
    )
    load = tetrahedron_p1_cell_nodal_load_vectors(
        coordinates,
        jnp.asarray(cells),
        jnp.ones((2, 4), dtype=jnp.float64),
    )

    rhs = scalar_h1_reduced_cell_rhs(
        stiffness,
        load,
        topology,
        jnp.asarray((0.0, 1.0), dtype=jnp.float64),
    )

    assert topology.cell_dof_count == layout.cell_dof_count == 4
    assert layout.schema_version == SCALAR_H1_TET4_COLLECTIVE_LAYOUT_SCHEMA
    assert layout.transport.schema_version == SCALAR_H1_TET4_COLLECTIVE_LAYOUT_SCHEMA
    assert pack_collective_scalar_h1_cell_matrix(layout, stiffness).shape == (1, 2, 4, 4)
    assert pack_collective_scalar_h1_cell_vector(layout, rhs).shape == (1, 2, 4)
    assert rhs.shape == (2, 4)
    assert bool(jnp.all(jnp.isfinite(rhs)))


def test_scalar_cell_load_rejects_shape_and_dtype_drift() -> None:
    boundary_map = _boundary_map()
    arguments = {
        "coordinates": jnp.asarray(COORDINATES),
        "cells": jnp.asarray(CELLS),
        "cell_source": jnp.ones(2),
        "boundary_facets": jnp.asarray(FACETS),
        "facet_load": jnp.ones(4),
        "boundary_map": boundary_map,
    }

    for name, value, message in (
        ("coordinates", jnp.ones((3, 2)), "coordinates"),
        ("cells", jnp.ones((1, 3), dtype=jnp.int32), "cells"),
        ("boundary_facets", jnp.ones((1, 2), dtype=jnp.int32), "facets"),
        ("cell_source", jnp.ones(1), "cell source"),
        ("facet_load", jnp.ones(3), "facet load"),
    ):
        changed = dict(arguments)
        changed[name] = value
        with pytest.raises(ValueError, match=message):
            triangle_p1_scalar_cell_load_vectors(**changed)

    for name, value, message in (
        ("coordinates", jnp.asarray(COORDINATES, dtype=jnp.complex128), "coordinates"),
        ("cell_source", jnp.ones(2, dtype=jnp.complex128), "cell source"),
        ("facet_load", jnp.ones(4, dtype=jnp.complex128), "facet load"),
        ("cells", jnp.asarray(CELLS, dtype=jnp.float64), "cells"),
        ("boundary_facets", jnp.asarray(FACETS, dtype=jnp.float64), "boundary facets"),
    ):
        changed = dict(arguments)
        changed[name] = value
        with pytest.raises(TypeError, match=message):
            triangle_p1_scalar_cell_load_vectors(**changed)


def test_scalar_cell_nodal_load_matches_consistent_triangle_mass() -> None:
    source = jnp.asarray(((1.0, 2.0, 4.0), (3.0, 5.0, 7.0)))
    local = triangle_p1_scalar_cell_nodal_load_vectors(
        jnp.asarray(COORDINATES),
        jnp.asarray(CELLS),
        source,
    )
    reference_mass = (np.ones((3, 3)) + np.eye(3)) / 12.0
    expected = np.stack((0.5 * reference_mass @ source[0], 0.5 * reference_mass @ source[1]))
    np.testing.assert_allclose(local, expected)

    for coordinates, cells, values, error, message in (
        (jnp.ones((4, 3)), jnp.asarray(CELLS), source, ValueError, "coordinates"),
        (jnp.asarray(COORDINATES), jnp.ones((2, 2), dtype=jnp.int32), source, ValueError, "cells"),
        (
            jnp.asarray(COORDINATES),
            jnp.asarray(CELLS),
            jnp.ones((2, 2)),
            ValueError,
            "nodal source",
        ),
        (
            jnp.asarray(COORDINATES, dtype=jnp.complex128),
            jnp.asarray(CELLS),
            source,
            TypeError,
            "coordinates",
        ),
        (
            jnp.asarray(COORDINATES),
            jnp.asarray(CELLS),
            source.astype(jnp.complex128),
            TypeError,
            "nodal source",
        ),
        (
            jnp.asarray(COORDINATES),
            jnp.asarray(CELLS, dtype=jnp.float64),
            source,
            TypeError,
            "cells",
        ),
    ):
        with pytest.raises(error, match=message):
            triangle_p1_scalar_cell_nodal_load_vectors(coordinates, cells, values)


def test_reduced_rhs_and_state_reconstruction_reject_shape_and_dtype_drift() -> None:
    topology = _topology()
    stiffness = jnp.broadcast_to(jnp.eye(3), (2, 3, 3))
    load = jnp.ones((2, 3))
    boundary = jnp.asarray((2.0,))

    with pytest.raises(ValueError, match="cell stiffness"):
        scalar_h1_reduced_cell_rhs(stiffness[:1], load, topology, boundary)
    with pytest.raises(ValueError, match="cell load"):
        scalar_h1_reduced_cell_rhs(stiffness, load[:1], topology, boundary)
    with pytest.raises(ValueError, match="Dirichlet values"):
        scalar_h1_reduced_cell_rhs(stiffness, load, topology, jnp.ones(2))
    with pytest.raises(TypeError, match="cell stiffness"):
        scalar_h1_reduced_cell_rhs(stiffness.astype(jnp.complex128), load, topology, boundary)
    with pytest.raises(TypeError, match="cell load"):
        scalar_h1_reduced_cell_rhs(stiffness, load.astype(jnp.complex128), topology, boundary)
    with pytest.raises(TypeError, match="Dirichlet values"):
        scalar_h1_reduced_cell_rhs(stiffness, load, topology, boundary.astype(jnp.complex128))

    free = jnp.asarray((0.2, 0.3, 0.4))
    with pytest.raises(ValueError, match="free state"):
        reconstruct_scalar_h1_state(topology, free[:2], boundary)
    with pytest.raises(ValueError, match="Dirichlet values"):
        reconstruct_scalar_h1_state(topology, free, jnp.ones(2))
    with pytest.raises(TypeError, match="free state"):
        reconstruct_scalar_h1_state(topology, free.astype(jnp.complex128), boundary)
    with pytest.raises(TypeError, match="Dirichlet values"):
        reconstruct_scalar_h1_state(topology, free, boundary.astype(jnp.complex128))


def test_scalar_pack_and_row_assembly_contracts_fail_closed() -> None:
    layout = _layout()
    cell_vector = jnp.ones((2, 3))
    packed_vector = pack_collective_scalar_h1_cell_vector(layout, cell_vector)
    mapping = jnp.asarray(layout.transport.cell_local_dofs)
    row_assembly = build_packed_collective_row_assembly(layout.transport, _mesh())

    np.testing.assert_array_equal(
        pack_collective_scalar_h1_owned_vector(layout, jnp.asarray((1.0, 2.0, 3.0))),
        ((1.0, 2.0, 3.0),),
    )
    with pytest.raises(ValueError, match="global cells"):
        pack_collective_cell_vector(layout.transport, jnp.ones((1, 3)))
    with pytest.raises(TypeError, match="floating or complex"):
        pack_collective_cell_vector(layout.transport, jnp.ones((2, 3), dtype=jnp.int32))
    with pytest.raises(TypeError, match="owned vector"):
        pack_collective_scalar_h1_owned_vector(layout, jnp.ones(3, dtype=jnp.complex128))
    with pytest.raises(TypeError, match="cell matrix"):
        pack_collective_scalar_h1_cell_matrix(layout, jnp.ones((2, 3, 3), dtype=jnp.complex128))

    with pytest.raises(ValueError, match="packed cell vector"):
        row_assembly(packed_vector[:, :, :2], mapping)
    with pytest.raises(ValueError, match="packed cell map"):
        row_assembly(packed_vector, mapping[:, :, :2])
    with pytest.raises(TypeError, match="cell vector"):
        row_assembly(packed_vector.astype(jnp.int32), mapping)
    with pytest.raises(TypeError, match="cell map"):
        row_assembly(packed_vector, mapping.astype(jnp.float64))
    with pytest.raises(ValueError, match="equal rank-two shapes"):
        collective_module._local_row_assembly(jnp.ones((1, 3)), jnp.ones((1, 2)), 3)


def test_generic_halo_kernel_uses_pairwise_value_and_row_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _layout(partition_count=2)
    transport = layout.transport
    matrix = pack_collective_scalar_h1_cell_matrix(
        layout,
        jnp.broadcast_to(jnp.eye(3), (2, 3, 3)),
    )
    vector = pack_collective_scalar_h1_owned_vector(layout, jnp.asarray((1.0, 2.0, 3.0)))
    mapping = jnp.asarray(transport.cell_local_dofs)
    routes: list[tuple[tuple[int, int], ...]] = []

    def zero_receive(
        payload: jax.Array,
        axis_name: str,
        permutations: tuple[tuple[int, int], ...],
    ) -> jax.Array:
        assert axis_name == "partition"
        routes.append(permutations)
        return jnp.zeros_like(payload)

    monkeypatch.setattr(collective_module.lax, "ppermute", zero_receive)
    local = collective_module.collective_local_matvec(
        transport,
        matrix[0],
        mapping[0],
        vector[0],
        jnp.asarray(0),
        "partition",
    )
    assert local.shape == (transport.owned_dof_capacity,)
    assert bool(jnp.all(jnp.isfinite(local)))
    assert len(routes) == 2 * len(transport.halo_links)


def test_packed_cell_gather_reconstructs_free_values_and_zeroes_constraints() -> None:
    layout = _layout()
    mapping = jnp.asarray(layout.transport.cell_local_dofs)
    vector = pack_collective_scalar_h1_owned_vector(layout, jnp.asarray((1.0, 2.0, 3.0)))
    generic = build_packed_collective_cell_gather(layout.transport, _mesh())
    scalar = build_packed_collective_scalar_h1_cell_gather(layout, _mesh())
    expected = np.asarray(((0.0, 1.0, 3.0), (0.0, 3.0, 2.0)))[None, ...]
    np.testing.assert_array_equal(generic(mapping, vector), expected)
    np.testing.assert_array_equal(scalar(mapping, vector), expected)

    with pytest.raises(ValueError, match="cell map"):
        generic(mapping[:, :, :2], vector)
    with pytest.raises(ValueError, match="owner vector"):
        generic(mapping, vector[:, :2])
    with pytest.raises(TypeError, match="cell map"):
        generic(mapping.astype(jnp.float64), vector)
    with pytest.raises(TypeError, match="owner vector"):
        generic(mapping, vector.astype(jnp.int32))
    with pytest.raises(TypeError, match="owner vector"):
        scalar(mapping, vector.astype(jnp.complex128))


def test_packed_scalar_cg_rejects_policy_mask_shape_and_dtype_drift() -> None:
    layout = _layout()
    mesh = _mesh()
    policy = ScalarH1CGPolicy(1.0e-10, 0.0, 10)
    with pytest.raises(ContractError, match="ScalarH1CGPolicy"):
        build_packed_collective_scalar_h1_cg(layout, mesh, object())  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="preconditioner factory"):
        build_packed_collective_scalar_h1_cg(
            layout,
            mesh,
            policy,
            preconditioner_factory=object(),  # type: ignore[arg-type]
        )
    dot = _build_packed_scalar_h1_dot(layout, mesh, axis_name="partition")
    vector = jnp.ones((layout.partition_count, layout.owned_dof_capacity))
    with pytest.raises(ContractError, match="active-owner mask"):
        dot(vector, vector, jnp.ones((1, 2), dtype=jnp.bool_))
    with pytest.raises(ContractError, match="active-owner mask"):
        dot(vector, vector, jnp.ones(vector.shape))

    solver = build_packed_collective_scalar_h1_cg(layout, mesh, policy)
    stiffness = pack_collective_scalar_h1_cell_matrix(
        layout,
        jnp.broadcast_to(jnp.eye(3), (2, 3, 3)),
    )
    mapping = jnp.asarray(layout.transport.cell_local_dofs)
    mask = pack_collective_scalar_h1_owned_mask(layout)
    assemble = build_packed_collective_scalar_h1_rhs_assembly(layout, mesh)
    rhs = assemble(pack_collective_scalar_h1_cell_vector(layout, jnp.ones((2, 3))), mapping)

    with pytest.raises(ValueError, match="cell stiffness"):
        solver(stiffness[:, :, :, :2], mapping, mask, rhs)
    with pytest.raises(ValueError, match="cell map"):
        solver(stiffness, mapping[:, :, :2], mask, rhs)
    with pytest.raises(ValueError, match="owner mask"):
        solver(stiffness, mapping, mask[:, :2], rhs)
    with pytest.raises(ValueError, match="right-hand side"):
        solver(stiffness, mapping, mask, rhs[:, :2])
    with pytest.raises(TypeError, match="cell stiffness"):
        solver(stiffness.astype(jnp.complex128), mapping, mask, rhs)
    with pytest.raises(TypeError, match="cell map"):
        solver(stiffness, mapping.astype(jnp.float64), mask, rhs)
    with pytest.raises(TypeError, match="owner mask"):
        solver(stiffness, mapping, mask.astype(jnp.int32), rhs)
    with pytest.raises(TypeError, match="right-hand side"):
        solver(stiffness, mapping, mask, rhs.astype(jnp.complex128))
    with pytest.raises(ContractError, match="cannot receive strategy arguments"):
        solver(stiffness, mapping, mask, rhs, jnp.asarray(1.0))


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        ((True, 0.0, 1), "real scalar"),
        ((1.0e-8, "zero", 1), "real scalar"),
        ((1.0e-8, float("inf"), 1), "finite"),
        ((1.0e-8, 0.0, True), "positive integer"),
        ((1.0e-8, 0.0, 1.5), "positive integer"),
    ),
)
def test_scalar_cg_policy_rejects_noncanonical_python_types(
    arguments: tuple[object, object, object],
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        ScalarH1CGPolicy(*arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", (True, "small", float("inf"), 0.0, -1.0))
def test_scalar_cg_policy_rejects_invalid_backward_error_tolerance(value: object) -> None:
    with pytest.raises(ContractError, match="backward-error tolerance"):
        ScalarH1CGPolicy(1.0e-8, 0.0, 10, backward_error_tolerance=value)  # type: ignore[arg-type]


def test_positive_diagonal_jacobi_preconditions_the_exact_packed_operator() -> None:
    layout = _layout()
    mesh = _mesh()
    mapping = jnp.asarray(layout.transport.cell_local_dofs)
    mask = pack_collective_scalar_h1_owned_mask(layout)
    stiffness = pack_collective_scalar_h1_cell_matrix(
        layout,
        jnp.asarray((np.diag((2.0, 3.0, 5.0)), np.diag((7.0, 11.0, 13.0)))),
    )
    assemble = build_packed_collective_scalar_h1_rhs_assembly(layout, mesh)
    rhs = assemble(
        pack_collective_scalar_h1_cell_vector(layout, jnp.asarray(((0.0, 1.0, 2.0),) * 2)),
        mapping,
    )
    factory = build_packed_scalar_h1_jacobi_preconditioner_factory(
        layout,
        mesh,
        ScalarH1JacobiPolicy(),
    )
    solve = build_packed_collective_scalar_h1_cg(
        layout,
        mesh,
        ScalarH1CGPolicy(1.0e-12, 1.0e-14, 20),
        preconditioner_factory=factory,
    )
    result = solve(stiffness, mapping, mask, rhs)
    assert bool(result.converged)
    assert not bool(result.breakdown)
    assert int(result.iterations) == 1
    np.testing.assert_allclose(result.relative_residual, 0.0, atol=1.0e-15)
    np.testing.assert_allclose(result.backward_error, 0.0, atol=1.0e-15)

    invalid_stiffness = pack_collective_scalar_h1_cell_matrix(
        layout,
        -jnp.broadcast_to(jnp.eye(3), (2, 3, 3)),
    )
    invalid = solve(invalid_stiffness, mapping, mask, rhs)
    assert bool(invalid.breakdown)
    assert not bool(invalid.converged)
    assert bool(jnp.all(jnp.isnan(invalid.solution)))


@pytest.mark.parametrize("value", (True, "small", float("inf"), 0.0, 1.0))
def test_jacobi_policy_and_factory_reject_invalid_contracts(value: object) -> None:
    with pytest.raises(ContractError, match="minimum relative diagonal"):
        ScalarH1JacobiPolicy(value)  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="ScalarH1JacobiPolicy"):
        build_packed_scalar_h1_jacobi_preconditioner_factory(
            _layout(),
            _mesh(),
            object(),  # type: ignore[arg-type]
        )
