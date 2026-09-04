from __future__ import annotations

from dataclasses import replace

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from femx.backends.jax.elements.tetrahedron_h1 import (  # noqa: E402
    tetrahedron_p1_diffusion_cell_matrices,
)
from femx.backends.jax.operators import triangle_p1_diffusion_cell_matrices  # noqa: E402
from femx.backends.jax.owned_ghost import (  # noqa: E402
    OwnedGhostPartition,
    OwnedGhostTopology,
    element_matrix_matvec,
    local_owned_cell_matvec,
    owned_ghost_matvec,
    prepare_owned_ghost_topology,
)
from femx.backends.jax.scalar_owned_ghost import (  # noqa: E402
    ScalarH1OwnedGhostTopology,
    matrix_free_scalar_h1_matvec,
    owned_ghost_scalar_h1_matvec,
    prepare_scalar_h1_owned_ghost_topology,
)
from femx.core.errors import ContractError  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


CELLS = np.asarray(((0, 1, 3), (0, 3, 2)), dtype=np.int64)
CELL_OWNERS = np.asarray((0, 1), dtype=np.int64)
FREE_NODES = np.asarray((1, 3), dtype=np.int64)


def _topology(*, partition_count: int = 2) -> ScalarH1OwnedGhostTopology:
    return prepare_scalar_h1_owned_ghost_topology(
        CELLS,
        CELL_OWNERS,
        node_count=4,
        free_nodes=FREE_NODES,
        partition_count=partition_count,
        dof_owners=np.asarray((0, 1), dtype=np.int64),
    )


def _cell_stiffness() -> jax.Array:
    coordinates = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)))
    return triangle_p1_diffusion_cell_matrices(
        coordinates,
        jnp.asarray(CELLS),
        jnp.asarray((2.0, 3.0)),
    )


def test_scalar_topology_preserves_free_node_identity_and_halo_direction() -> None:
    topology = _topology()

    assert topology.node_count == 4
    assert topology.free_dof_count == 2
    assert topology.cell_count == 2
    np.testing.assert_array_equal(topology.free_nodes, (1, 3))
    np.testing.assert_array_equal(topology.full_to_reduced, (2, 0, 2, 1))
    np.testing.assert_array_equal(topology.cell_reduced_dofs, ((2, 0, 1), (2, 1, 2)))
    assert topology.owned_ghost.cell_dof_count == 3
    assert len(topology.owned_ghost.halo_links) == 1
    link = topology.owned_ghost.halo_links[0]
    assert (link.owner_partition, link.ghost_partition) == (1, 0)
    np.testing.assert_array_equal(link.global_dofs, (1,))
    for values in (
        topology.free_nodes,
        topology.full_to_reduced,
        topology.cell_reduced_dofs,
    ):
        assert not values.flags.writeable


def test_scalar_serial_and_owned_ghost_actions_match_with_an_empty_partition() -> None:
    topology = _topology(partition_count=3)
    stiffness = _cell_stiffness()
    vector = jnp.asarray((0.25, -0.75), dtype=jnp.float64)

    serial = jax.jit(lambda matrix, x: matrix_free_scalar_h1_matvec(matrix, topology, x))(
        stiffness,
        vector,
    )
    partitioned = jax.jit(lambda matrix, x: owned_ghost_scalar_h1_matvec(matrix, topology, x))(
        stiffness, vector
    )

    np.testing.assert_allclose(partitioned, serial, rtol=0.0, atol=0.0)
    assert topology.owned_ghost.partitions[2].owned_cells.size == 0
    assert topology.owned_ghost.partitions[2].local_dof_count == 0


def test_tet4_scalar_topology_and_actions_preserve_four_node_cells() -> None:
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
        np.asarray((0, 1), dtype=np.int64),
        node_count=5,
        free_nodes=np.asarray((1, 2, 3), dtype=np.int64),
        partition_count=2,
        dof_owners=np.asarray((0, 1, 0), dtype=np.int64),
    )
    stiffness = tetrahedron_p1_diffusion_cell_matrices(
        coordinates,
        jnp.asarray(cells),
        jnp.asarray((2.0, 3.0), dtype=jnp.float64),
    )
    vector = jnp.asarray((0.25, -0.5, 0.75), dtype=jnp.float64)

    serial = jax.jit(lambda matrix, x: matrix_free_scalar_h1_matvec(matrix, topology, x))(
        stiffness,
        vector,
    )
    partitioned = jax.jit(lambda matrix, x: owned_ghost_scalar_h1_matvec(matrix, topology, x))(
        stiffness, vector
    )

    assert topology.cell_dof_count == 4
    assert topology.owned_ghost.cell_dof_count == 4
    np.testing.assert_array_equal(topology.cell_reduced_dofs, ((3, 0, 1, 2), (0, 1, 2, 3)))
    np.testing.assert_allclose(partitioned, serial, rtol=0.0, atol=0.0)


def test_generic_element_action_rejects_malformed_inputs_and_marks_bad_map() -> None:
    matrix = jnp.eye(2, dtype=jnp.float64)[None, :, :]
    mapping = jnp.asarray(((0, 1),), dtype=jnp.int32)
    vector = jnp.asarray((2.0, 3.0))

    np.testing.assert_array_equal(element_matrix_matvec(matrix, mapping, vector), vector)
    assert np.isnan(element_matrix_matvec(matrix, mapping.at[0, 0].set(3), vector)).all()
    with pytest.raises(ValueError, match="element-local matrix"):
        element_matrix_matvec(jnp.ones((1, 2)), mapping, vector)
    with pytest.raises(ValueError, match="cell map"):
        element_matrix_matvec(matrix, jnp.ones((1, 1), dtype=jnp.int32), vector)
    with pytest.raises(ValueError, match="nonempty rank-one"):
        element_matrix_matvec(matrix, mapping, jnp.ones((1, 2)))
    with pytest.raises(TypeError, match="integer dtype"):
        element_matrix_matvec(matrix, mapping.astype(jnp.float64), vector)


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"node_count": True}, "node count"),
        ({"node_count": 0}, "node count"),
        ({"cells": np.ones((2, 3), dtype=float)}, "scalar H1 cells"),
        ({"cells": np.ones(6, dtype=np.int64)}, "scalar H1 cells"),
        ({"cells": np.ones((2, 2), dtype=np.int64)}, "3 columns"),
        ({"cells": np.ones((2, 5), dtype=np.int64)}, "3 columns"),
        ({"cells": np.empty((0, 3), dtype=np.int64)}, "at least one cell"),
        ({"cells": np.asarray(((0, 1, 4), (0, 3, 2)))}, "out-of-range node"),
        ({"cells": np.asarray(((0, 1, 1), (0, 3, 2)))}, "repeat a node"),
        ({"free_nodes": np.asarray((1.0, 3.0))}, "free nodes"),
        ({"free_nodes": np.empty(0, dtype=np.int64)}, "at least one free node"),
        ({"free_nodes": np.asarray((1, 4))}, "out-of-range node"),
        ({"free_nodes": np.asarray((3, 1))}, "strictly increasing"),
        ({"free_nodes": np.asarray((1, 1))}, "strictly increasing"),
    ),
)
def test_scalar_topology_builder_rejects_invalid_mesh_identity(
    updates: dict[str, object],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "cells": CELLS,
        "cell_owners": CELL_OWNERS,
        "node_count": 4,
        "free_nodes": FREE_NODES,
        "partition_count": 2,
    }
    arguments.update(updates)
    with pytest.raises(ContractError, match=message):
        prepare_scalar_h1_owned_ghost_topology(**arguments)  # type: ignore[arg-type]


def test_scalar_topology_record_revalidates_maps_and_element_width() -> None:
    topology = _topology()

    with pytest.raises(ContractError, match="node count"):
        replace(topology, node_count=False)
    with pytest.raises(ContractError, match="at least one cell"):
        replace(topology, cells=np.empty((0, 3), dtype=np.int64))
    with pytest.raises(ContractError, match="out-of-range node"):
        replace(topology, cells=np.asarray(((0, 1, 4), (0, 3, 2))))
    with pytest.raises(ContractError, match="repeat a node"):
        replace(topology, cells=np.asarray(((0, 1, 1), (0, 3, 2))))
    with pytest.raises(ContractError, match="at least one free node"):
        replace(topology, free_nodes=np.empty(0, dtype=np.int64))
    with pytest.raises(ContractError, match="out-of-range node"):
        replace(topology, free_nodes=np.asarray((1, 4)))
    with pytest.raises(ContractError, match="node count"):
        replace(topology, full_to_reduced=np.asarray((0, 1)))
    with pytest.raises(ContractError, match="OwnedGhostTopology"):
        replace(topology, owned_ghost=object())

    two_column = prepare_owned_ghost_topology(
        np.asarray(((0, 1),), dtype=np.int64),
        np.asarray((0,), dtype=np.int64),
        global_dof_count=2,
        partition_count=1,
    )
    with pytest.raises(ContractError, match="3 local DOFs"):
        replace(topology, owned_ghost=two_column)
    with pytest.raises(ContractError, match="global reduced DOFs"):
        replace(topology, free_nodes=np.asarray((0, 1, 3)))
    one_cell = prepare_owned_ghost_topology(
        np.asarray(((2, 0, 1),), dtype=np.int64),
        np.asarray((0,), dtype=np.int64),
        global_dof_count=2,
        partition_count=1,
    )
    with pytest.raises(ContractError, match="cell count"):
        replace(topology, owned_ghost=one_cell)
    wrong_map = topology.full_to_reduced.copy()
    wrong_map[0] = 0
    with pytest.raises(ContractError, match="free-node identity"):
        replace(topology, full_to_reduced=wrong_map)
    wrong_cell_map = prepare_owned_ghost_topology(
        np.asarray(((2, 0, 1), (2, 0, 2)), dtype=np.int64),
        CELL_OWNERS,
        global_dof_count=2,
        partition_count=2,
    )
    with pytest.raises(ContractError, match="reduced map"):
        replace(topology, owned_ghost=wrong_cell_map)


def test_generic_records_reject_zero_width_and_partition_width_drift() -> None:
    with pytest.raises(ContractError, match="at least one column"):
        OwnedGhostPartition(
            0,
            np.asarray((), dtype=np.int64),
            np.asarray((0,), dtype=np.int64),
            np.asarray((), dtype=np.int64),
            np.asarray((0,), dtype=np.int64),
            np.empty((0, 0), dtype=np.int64),
        )
    topology = _topology().owned_ghost
    wrong_partition = replace(
        topology.partitions[0],
        cell_local_dofs=np.zeros((1, 2), dtype=np.int64),
    )
    with pytest.raises(ContractError, match="cell width"):
        replace(topology, partitions=(wrong_partition, topology.partitions[1]))
    with pytest.raises(ContractError, match="at least one column"):
        OwnedGhostTopology(
            partition_count=1,
            global_dof_count=1,
            cell_reduced_dofs=np.empty((1, 0), dtype=np.int64),
            cell_owners=np.asarray((0,), dtype=np.int64),
            dof_owners=np.asarray((0,), dtype=np.int64),
            partitions=(),
            halo_links=(),
        )
    with pytest.raises(ContractError, match="at least one column"):
        prepare_owned_ghost_topology(
            np.empty((1, 0), dtype=np.int64),
            np.asarray((0,), dtype=np.int64),
            global_dof_count=1,
            partition_count=1,
        )


def test_scalar_runtime_rejects_wrong_shape_and_nonreal_dtype() -> None:
    topology = _topology()
    stiffness = _cell_stiffness()
    vector = jnp.asarray((0.25, -0.75), dtype=jnp.float64)

    with pytest.raises(ValueError, match="cell stiffness"):
        matrix_free_scalar_h1_matvec(stiffness[:1], topology, vector)
    with pytest.raises(TypeError, match="real floating"):
        matrix_free_scalar_h1_matvec(stiffness.astype(jnp.complex128), topology, vector)
    with pytest.raises(ValueError, match="global free DOFs"):
        owned_ghost_scalar_h1_matvec(stiffness, topology, vector[:1])
    with pytest.raises(TypeError, match="real floating"):
        owned_ghost_scalar_h1_matvec(stiffness, topology, vector.astype(jnp.complex128))

    generic = topology.owned_ghost
    local_vector = jnp.ones(generic.partitions[0].local_dof_count)
    with pytest.raises(ValueError, match="local cell matrix"):
        local_owned_cell_matvec(stiffness, generic.partitions[0], local_vector)
    with pytest.raises(ValueError, match="global topology"):
        owned_ghost_matvec(stiffness[:1], generic, vector)
