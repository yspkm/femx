from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from femx.backends.jax.owned_ghost import (  # noqa: E402
    HaloLink,
    prepare_owned_ghost_topology,
)
from femx.backends.jax.port_matrix_free import matrix_free_port_matvec  # noqa: E402
from femx.backends.jax.port_owned_ghost import (  # noqa: E402
    PortHaloLink,
    PortOwnedGhostPartition,
    PortOwnedGhostTopology,
    assemble_port_owned_vector,
    exchange_port_halo_values,
    local_owned_cell_port_matvec,
    owned_ghost_port_generalized_residual,
    owned_ghost_port_matvec,
    partition_owned_port_vector,
    prepare_owned_ghost_port_topology,
    reduce_port_ghost_contributions,
)
from femx.core.errors import ContractError  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


CELL_MAP = np.asarray(
    (
        (0, 1, 2, 3, 4, 5),
        (1, 2, 6, 4, 5, 3),
    ),
    dtype=np.int64,
)
CELL_OWNERS = np.asarray((0, 1), dtype=np.int64)
DOF_OWNERS = np.asarray((0, 0, 0, 0, 1, 1, 1), dtype=np.int64)


def _two_partition_topology(*, partition_count: int = 2):
    return prepare_owned_ghost_port_topology(
        CELL_MAP,
        CELL_OWNERS,
        free_dof_count=7,
        partition_count=partition_count,
        dof_owners=DOF_OWNERS,
    )


def _cell_matrices() -> jax.Array:
    values = np.arange(72, dtype=np.float64).reshape(2, 6, 6) / 19.0
    values[:, np.arange(6), np.arange(6)] += 4.0
    return jnp.asarray(values)


def test_topology_records_canonical_owned_ghost_and_bidirectional_links() -> None:
    topology = _two_partition_topology()
    first, second = topology.partitions

    np.testing.assert_array_equal(first.owned_cells, (0,))
    np.testing.assert_array_equal(first.owned_dofs, (0, 1, 2, 3))
    np.testing.assert_array_equal(first.ghost_dofs, (4, 5))
    np.testing.assert_array_equal(first.local_dofs, (0, 1, 2, 3, 4, 5))
    np.testing.assert_array_equal(first.cell_local_dofs, ((0, 1, 2, 3, 4, 5),))
    assert first.owned_dof_count == 4
    assert first.ghost_dof_count == 2
    assert first.local_dof_count == 6
    assert first.constrained_sentinel == 6

    np.testing.assert_array_equal(second.owned_cells, (1,))
    np.testing.assert_array_equal(second.owned_dofs, (4, 5, 6))
    np.testing.assert_array_equal(second.ghost_dofs, (1, 2, 3))
    np.testing.assert_array_equal(second.local_dofs, (4, 5, 6, 1, 2, 3))
    np.testing.assert_array_equal(second.cell_local_dofs, ((3, 4, 2, 0, 1, 5),))

    assert topology.cell_count == 2
    assert topology.constrained_sentinel == 7
    assert len(topology.halo_links) == 2
    forward, reverse = topology.halo_links
    assert (forward.owner_partition, forward.ghost_partition) == (0, 1)
    np.testing.assert_array_equal(forward.global_dofs, (1, 2, 3))
    np.testing.assert_array_equal(forward.owner_local_indices, (1, 2, 3))
    np.testing.assert_array_equal(forward.ghost_local_indices, (3, 4, 5))
    assert (reverse.owner_partition, reverse.ghost_partition) == (1, 0)
    np.testing.assert_array_equal(reverse.global_dofs, (4, 5))
    np.testing.assert_array_equal(reverse.owner_local_indices, (0, 1))
    np.testing.assert_array_equal(reverse.ghost_local_indices, (4, 5))

    for array in (
        topology.cell_reduced_dofs,
        topology.cell_owners,
        topology.dof_owners,
        *(partition.local_dofs for partition in topology.partitions),
        *(link.global_dofs for link in topology.halo_links),
    ):
        assert not array.flags.writeable


def test_default_dof_owner_is_the_lowest_incident_cell_partition() -> None:
    topology = prepare_owned_ghost_port_topology(
        CELL_MAP,
        CELL_OWNERS,
        free_dof_count=7,
        partition_count=2,
    )

    np.testing.assert_array_equal(topology.dof_owners, (0, 0, 0, 0, 0, 0, 1))
    assert len(topology.halo_links) == 1
    assert (topology.halo_links[0].owner_partition, topology.halo_links[0].ghost_partition) == (
        0,
        1,
    )


def test_forward_halo_and_reverse_contribution_exchange_are_explicit() -> None:
    topology = _two_partition_topology()
    vector = jnp.arange(7, dtype=jnp.float64) + 10.0
    seeded = partition_owned_port_vector(topology, vector)

    np.testing.assert_array_equal(seeded[0], (10.0, 11.0, 12.0, 13.0, 0.0, 0.0))
    np.testing.assert_array_equal(seeded[1], (14.0, 15.0, 16.0, 0.0, 0.0, 0.0))
    exchanged = exchange_port_halo_values(topology, seeded)
    np.testing.assert_array_equal(
        exchanged[0], vector[jnp.asarray(topology.partitions[0].local_dofs)]
    )
    np.testing.assert_array_equal(
        exchanged[1], vector[jnp.asarray(topology.partitions[1].local_dofs)]
    )

    local_contributions = (
        jnp.asarray((1.0, 2.0, 3.0, 4.0, 50.0, 60.0)),
        jnp.asarray((5.0, 6.0, 7.0, 20.0, 30.0, 40.0)),
    )
    owned = reduce_port_ghost_contributions(topology, local_contributions)
    np.testing.assert_array_equal(owned[0], (1.0, 22.0, 33.0, 44.0))
    np.testing.assert_array_equal(owned[1], (55.0, 66.0, 7.0))
    np.testing.assert_array_equal(
        assemble_port_owned_vector(topology, owned),
        (1.0, 22.0, 33.0, 44.0, 55.0, 66.0, 7.0),
    )


def test_partitioned_action_matches_serial_for_complex_values_and_empty_partition() -> None:
    topology = _two_partition_topology(partition_count=3)
    matrix = _cell_matrices().astype(jnp.complex128) * (1.0 + 0.2j)
    vector = jnp.asarray((1.0 + 0.1j, -0.2j, 0.4, 0.7j, -0.5, 0.8 - 0.1j, 0.3))

    observed = jax.jit(lambda local_matrix, x: owned_ghost_port_matvec(local_matrix, topology, x))(
        matrix,
        vector,
    )
    expected = matrix_free_port_matvec(matrix, jnp.asarray(CELL_MAP), vector)

    np.testing.assert_allclose(observed, expected, rtol=2.0e-15, atol=2.0e-15)
    assert observed.dtype == jnp.complex128
    empty = topology.partitions[2]
    assert empty.owned_cells.size == 0
    assert empty.local_dof_count == 0
    empty_action = local_owned_cell_port_matvec(
        matrix[jnp.asarray(empty.owned_cells)],
        empty,
        jnp.zeros((0,), dtype=vector.dtype),
    )
    assert empty_action.shape == (0,)


def test_partitioned_generalized_residual_matches_serial_componentwise_authority() -> None:
    topology = _two_partition_topology()
    stiffness = _cell_matrices()
    mass = jnp.flip(_cell_matrices(), axis=(1, 2)) / 7.0
    vector = jnp.asarray((0.2, -0.1, 0.4, 0.8, -0.3, 0.5, 0.9))
    eigenvalue = jnp.asarray(-1.7)

    observed = jax.jit(
        lambda local_stiffness, local_mass, x, value: owned_ghost_port_generalized_residual(
            local_stiffness,
            local_mass,
            topology,
            x,
            value,
        )
    )(stiffness, mass, vector, eigenvalue)
    expected_residual = matrix_free_port_matvec(
        stiffness,
        jnp.asarray(CELL_MAP),
        vector,
    ) - eigenvalue * matrix_free_port_matvec(mass, jnp.asarray(CELL_MAP), vector)
    expected_bound = matrix_free_port_matvec(
        jnp.abs(stiffness) + jnp.abs(eigenvalue) * jnp.abs(mass),
        jnp.asarray(CELL_MAP),
        jnp.abs(vector),
    )

    np.testing.assert_allclose(observed.residual, expected_residual, rtol=2.0e-15, atol=2.0e-15)
    np.testing.assert_allclose(
        observed.row_magnitude_bound,
        expected_bound,
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        observed.relative_scaled_residual,
        np.linalg.norm(expected_residual) / np.linalg.norm(expected_bound),
        rtol=2.0e-15,
    )


def test_partitioned_action_reverse_mode_matches_serial_scatter_transpose() -> None:
    topology = _two_partition_topology()
    matrix = _cell_matrices()
    vector = jnp.asarray((0.2, -0.1, 0.4, 0.8, -0.3, 0.5, 0.9))
    weights = jnp.asarray((-0.4, 0.2, 0.5, -0.1, 0.6, 0.7, -0.3))

    def partitioned_objective(local_matrix: jax.Array, x: jax.Array) -> jax.Array:
        return jnp.vdot(weights, owned_ghost_port_matvec(local_matrix, topology, x)).real

    def serial_objective(local_matrix: jax.Array, x: jax.Array) -> jax.Array:
        return jnp.vdot(
            weights,
            matrix_free_port_matvec(local_matrix, jnp.asarray(CELL_MAP), x),
        ).real

    observed_value, observed_gradients = jax.jit(
        jax.value_and_grad(partitioned_objective, argnums=(0, 1))
    )(matrix, vector)
    expected_value, expected_gradients = jax.jit(
        jax.value_and_grad(serial_objective, argnums=(0, 1))
    )(matrix, vector)

    np.testing.assert_allclose(observed_value, expected_value, rtol=2.0e-15, atol=2.0e-15)
    for observed, expected in zip(observed_gradients, expected_gradients, strict=True):
        np.testing.assert_allclose(observed, expected, rtol=2.0e-15, atol=2.0e-15)


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"free_dof_count": True}, "free DOF count"),
        ({"free_dof_count": 0}, "free DOF count"),
        ({"partition_count": True}, "partition count"),
        ({"partition_count": 0}, "partition count"),
        ({"cell_reduced_dofs": np.ones((2, 6), dtype=float)}, "cell reduced DOFs"),
        ({"cell_reduced_dofs": np.ones(12, dtype=np.int64)}, "cell reduced DOFs"),
        ({"cell_reduced_dofs": np.ones((2, 5), dtype=np.int64)}, "6 columns"),
        ({"cell_reduced_dofs": np.empty((0, 6), dtype=np.int64)}, "at least one cell"),
        (
            {
                "cell_reduced_dofs": np.asarray(
                    ((8, 1, 2, 3, 4, 5), (1, 2, 6, 4, 5, 3)),
                    dtype=np.int64,
                )
            },
            "out-of-range reduced DOF",
        ),
        ({"cell_owners": np.asarray((0.0, 1.0))}, "cell owners"),
        ({"cell_owners": np.asarray((0,), dtype=np.int64)}, "match the cell count"),
        ({"cell_owners": np.asarray((0, 2), dtype=np.int64)}, "outside the partition"),
        ({"dof_owners": np.arange(7, dtype=float)}, "DOF owners"),
        ({"dof_owners": np.asarray((0, 0), dtype=np.int64)}, "global DOF count"),
        ({"dof_owners": np.asarray((0, 0, 0, 0, 1, 1, 2))}, "outside the partition"),
        ({"dof_owners": np.asarray((1, 0, 0, 0, 1, 1, 1))}, "incident cell"),
    ),
)
def test_topology_builder_rejects_invalid_contracts(
    updates: dict[str, object],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "cell_reduced_dofs": CELL_MAP,
        "cell_owners": CELL_OWNERS,
        "free_dof_count": 7,
        "partition_count": 2,
        "dof_owners": DOF_OWNERS,
    }
    arguments.update(updates)
    with pytest.raises(ContractError, match=message):
        prepare_owned_ghost_port_topology(**arguments)  # type: ignore[arg-type]


def test_topology_builder_rejects_missing_and_repeated_active_dofs() -> None:
    missing = CELL_MAP.copy()
    missing[1, 2] = 5
    with pytest.raises(ContractError, match="every global free DOF"):
        prepare_owned_ghost_port_topology(
            missing,
            CELL_OWNERS,
            free_dof_count=7,
            partition_count=2,
        )

    repeated = CELL_MAP.copy()
    repeated[0, 5] = 0
    with pytest.raises(ContractError, match="repeats an active DOF"):
        prepare_owned_ghost_port_topology(
            repeated,
            CELL_OWNERS,
            free_dof_count=7,
            partition_count=2,
        )


@pytest.mark.parametrize(
    ("factory", "message"),
    (
        (
            lambda: PortOwnedGhostPartition(
                -1,
                np.asarray((0,)),
                np.asarray((0,)),
                np.asarray((), dtype=np.int64),
                np.asarray((0,)),
                np.asarray(((0, 1, 1, 1, 1, 1),)),
            ),
            "partition index",
        ),
        (
            lambda: PortOwnedGhostPartition(
                0,
                np.asarray((0,)),
                np.asarray((0,)),
                np.asarray((1,)),
                np.asarray((1, 0)),
                np.asarray(((0, 1, 2, 2, 2, 2),)),
            ),
            "owned entries before ghosts",
        ),
        (
            lambda: PortHaloLink(
                0,
                0,
                np.asarray((1,)),
                np.asarray((0,)),
                np.asarray((1,)),
            ),
            "must differ",
        ),
        (
            lambda: PortHaloLink(
                0,
                1,
                np.asarray((), dtype=np.int64),
                np.asarray((), dtype=np.int64),
                np.asarray((), dtype=np.int64),
            ),
            "at least one shared",
        ),
        (
            lambda: PortHaloLink(
                0,
                1,
                np.asarray((1, 2)),
                np.asarray((0,)),
                np.asarray((1, 2)),
            ),
            "equal lengths",
        ),
    ),
)
def test_partition_and_link_records_reject_ambiguous_state(
    factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        factory()


def test_partition_and_link_records_reject_malformed_arrays() -> None:
    with pytest.raises(ContractError, match="strictly increasing"):
        PortOwnedGhostPartition(
            0,
            np.asarray((0, 0)),
            np.asarray((0,)),
            np.asarray((), dtype=np.int64),
            np.asarray((0,)),
            np.zeros((2, 6), dtype=np.int64),
        )


def test_port_records_reject_generic_or_wrong_width_topologies() -> None:
    with pytest.raises(ContractError, match="6 columns"):
        PortOwnedGhostPartition(
            0,
            np.asarray((0,)),
            np.asarray((0, 1)),
            np.asarray((), dtype=np.int64),
            np.asarray((0, 1)),
            np.asarray(((0, 1),)),
        )

    two_column = prepare_owned_ghost_topology(
        np.asarray(((0, 1),), dtype=np.int64),
        np.asarray((0,), dtype=np.int64),
        global_dof_count=2,
        partition_count=1,
    )
    with pytest.raises(ContractError, match="6 columns"):
        PortOwnedGhostTopology(
            partition_count=two_column.partition_count,
            global_dof_count=two_column.global_dof_count,
            cell_reduced_dofs=two_column.cell_reduced_dofs,
            cell_owners=two_column.cell_owners,
            dof_owners=two_column.dof_owners,
            partitions=two_column.partitions,
            halo_links=two_column.halo_links,
        )

    generic = prepare_owned_ghost_topology(
        CELL_MAP,
        CELL_OWNERS,
        global_dof_count=7,
        partition_count=2,
        dof_owners=DOF_OWNERS,
    )
    with pytest.raises(ContractError, match="PortOwnedGhostPartition"):
        PortOwnedGhostTopology(
            partition_count=generic.partition_count,
            global_dof_count=generic.global_dof_count,
            cell_reduced_dofs=generic.cell_reduced_dofs,
            cell_owners=generic.cell_owners,
            dof_owners=generic.dof_owners,
            partitions=generic.partitions,
            halo_links=generic.halo_links,
        )

    valid = _two_partition_topology()
    first_link = valid.halo_links[0]
    generic_link = HaloLink(
        owner_partition=first_link.owner_partition,
        ghost_partition=first_link.ghost_partition,
        global_dofs=first_link.global_dofs,
        owner_local_indices=first_link.owner_local_indices,
        ghost_local_indices=first_link.ghost_local_indices,
    )
    with pytest.raises(ContractError, match="PortHaloLink"):
        replace(valid, halo_links=(generic_link, valid.halo_links[1]))
    with pytest.raises(ContractError, match="cannot be negative"):
        PortOwnedGhostPartition(
            0,
            np.asarray((0,)),
            np.asarray((-1,)),
            np.asarray((), dtype=np.int64),
            np.asarray((-1,)),
            np.zeros((1, 6), dtype=np.int64),
        )
    with pytest.raises(ContractError, match="local dofs must be unique"):
        PortOwnedGhostPartition(
            0,
            np.asarray((0,)),
            np.asarray((0,)),
            np.asarray((1,)),
            np.asarray((0, 0)),
            np.zeros((1, 6), dtype=np.int64),
        )
    with pytest.raises(ContractError, match="rows must match"):
        PortOwnedGhostPartition(
            0,
            np.asarray((0, 1)),
            np.asarray((0,)),
            np.asarray((), dtype=np.int64),
            np.asarray((0,)),
            np.zeros((1, 6), dtype=np.int64),
        )
    with pytest.raises(ContractError, match="out-of-range index"):
        PortOwnedGhostPartition(
            0,
            np.asarray((0,)),
            np.asarray((0,)),
            np.asarray((), dtype=np.int64),
            np.asarray((0,)),
            np.full((1, 6), 2, dtype=np.int64),
        )
    with pytest.raises(ContractError, match="nonnegative integer"):
        PortHaloLink(
            -1,
            1,
            np.asarray((0,)),
            np.asarray((0,)),
            np.asarray((0,)),
        )
    with pytest.raises(ContractError, match="cannot be negative"):
        PortHaloLink(
            0,
            1,
            np.asarray((-1,)),
            np.asarray((0,)),
            np.asarray((0,)),
        )


def test_topology_builder_rejects_a_ragged_array() -> None:
    with pytest.raises(ContractError, match="regular integer array"):
        prepare_owned_ghost_port_topology(
            [[0], [1, 2]],
            np.asarray((0, 1)),
            free_dof_count=3,
            partition_count=2,
        )


def test_topology_record_rejects_missing_links_and_noncanonical_partition_order() -> None:
    topology = _two_partition_topology()
    with pytest.raises(ContractError, match="cover every ghost"):
        replace(topology, halo_links=())
    with pytest.raises(ContractError, match="canonical index order"):
        replace(topology, partitions=tuple(reversed(topology.partitions)))
    invalid_map = topology.partitions[0].cell_local_dofs.copy()
    invalid_map[0, :2] = invalid_map[0, 1::-1]
    invalid_partition = replace(topology.partitions[0], cell_local_dofs=invalid_map)
    with pytest.raises(ContractError, match="global DOF identity"):
        replace(topology, partitions=(invalid_partition, topology.partitions[1]))


def test_topology_record_revalidates_global_identity_and_partition_contracts() -> None:
    topology = _two_partition_topology()

    with pytest.raises(ContractError, match="at least one cell"):
        replace(
            topology,
            cell_reduced_dofs=np.empty((0, 6), dtype=np.int64),
            cell_owners=np.empty(0, dtype=np.int64),
        )
    with pytest.raises(ContractError, match="cell owners must match"):
        replace(topology, cell_owners=np.asarray((0,), dtype=np.int64))
    with pytest.raises(ContractError, match="DOF owners must match"):
        replace(topology, dof_owners=np.asarray((0,), dtype=np.int64))

    invalid_map = topology.cell_reduced_dofs.copy()
    invalid_map[0, 0] = 8
    with pytest.raises(ContractError, match="out-of-range reduced DOF"):
        replace(topology, cell_reduced_dofs=invalid_map)
    with pytest.raises(ContractError, match="cell owner lies outside"):
        replace(topology, cell_owners=np.asarray((0, 2), dtype=np.int64))
    with pytest.raises(ContractError, match="DOF owner lies outside"):
        replace(topology, dof_owners=np.asarray((0, 0, 0, 0, 1, 1, 2), dtype=np.int64))

    missing_map = topology.cell_reduced_dofs.copy()
    missing_map[1, 2] = 5
    with pytest.raises(ContractError, match="every global free DOF"):
        replace(topology, cell_reduced_dofs=missing_map)
    repeated_map = topology.cell_reduced_dofs.copy()
    repeated_map[0, 5] = 0
    with pytest.raises(ContractError, match="repeats an active DOF"):
        replace(topology, cell_reduced_dofs=repeated_map)
    with pytest.raises(ContractError, match="incident cell"):
        replace(topology, dof_owners=np.asarray((1, 0, 0, 0, 1, 1, 1), dtype=np.int64))
    with pytest.raises(ContractError, match="every partition exactly once"):
        replace(topology, partitions=topology.partitions[:1])

    wrong_cells = replace(topology.partitions[0], owned_cells=np.asarray((1,), dtype=np.int64))
    with pytest.raises(ContractError, match="cells disagree"):
        replace(topology, partitions=(wrong_cells, topology.partitions[1]))
    wrong_owned = replace(
        topology.partitions[0],
        owned_dofs=np.asarray((0, 1, 2), dtype=np.int64),
        local_dofs=np.asarray((0, 1, 2, 4, 5), dtype=np.int64),
    )
    with pytest.raises(ContractError, match="owner DOFs disagree"):
        replace(topology, partitions=(wrong_owned, topology.partitions[1]))
    wrong_ghosts = replace(
        topology.partitions[0],
        ghost_dofs=np.asarray((4,), dtype=np.int64),
        local_dofs=np.asarray((0, 1, 2, 3, 4), dtype=np.int64),
    )
    with pytest.raises(ContractError, match="ghosts disagree"):
        replace(topology, partitions=(wrong_ghosts, topology.partitions[1]))


def test_topology_record_revalidates_every_halo_link_index() -> None:
    topology = _two_partition_topology()
    first, second = topology.halo_links

    with pytest.raises(ContractError, match="canonical pair order"):
        replace(topology, halo_links=(second, first))
    wrong_dofs = replace(first, global_dofs=np.asarray((0, 2, 3), dtype=np.int64))
    with pytest.raises(ContractError, match="link DOFs disagree"):
        replace(topology, halo_links=(wrong_dofs, second))
    owner_outside = replace(
        first,
        owner_local_indices=np.asarray((1, 2, 4), dtype=np.int64),
    )
    with pytest.raises(ContractError, match="owner index lies outside"):
        replace(topology, halo_links=(owner_outside, second))
    ghost_outside = replace(
        first,
        ghost_local_indices=np.asarray((0, 4, 5), dtype=np.int64),
    )
    with pytest.raises(ContractError, match="ghost index lies outside"):
        replace(topology, halo_links=(ghost_outside, second))
    wrong_owner_indices = replace(
        first,
        owner_local_indices=np.asarray((0, 2, 3), dtype=np.int64),
    )
    with pytest.raises(ContractError, match="owner indices identify the wrong"):
        replace(topology, halo_links=(wrong_owner_indices, second))

    three_way = prepare_owned_ghost_port_topology(
        np.asarray(
            (
                (0, 3, 6, 7, 8, 9),
                (1, 4, 6, 7, 10, 11),
                (2, 5, 6, 7, 8, 10),
            ),
            dtype=np.int64,
        ),
        np.asarray((0, 1, 2), dtype=np.int64),
        free_dof_count=12,
        partition_count=3,
        dof_owners=np.asarray((0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 2, 1), dtype=np.int64),
    )
    link_index = next(
        index
        for index, link in enumerate(three_way.halo_links)
        if (link.owner_partition, link.ghost_partition) == (0, 2)
    )
    wrong_ghost_indices = replace(
        three_way.halo_links[link_index],
        ghost_local_indices=np.asarray((5,), dtype=np.int64),
    )
    links = list(three_way.halo_links)
    links[link_index] = wrong_ghost_indices
    with pytest.raises(ContractError, match="ghost indices identify the wrong"):
        replace(three_way, halo_links=tuple(links))


def test_runtime_rejects_mismatched_vectors_matrices_and_eigenvalue() -> None:
    topology = _two_partition_topology()
    vector = jnp.arange(7, dtype=jnp.float64)
    matrix = _cell_matrices()
    local = exchange_port_halo_values(topology, partition_owned_port_vector(topology, vector))

    with pytest.raises(ValueError, match="global free DOFs"):
        partition_owned_port_vector(topology, jnp.ones(6))
    with pytest.raises(TypeError, match="floating or complex"):
        partition_owned_port_vector(topology, jnp.arange(7, dtype=jnp.int32))
    with pytest.raises(ValueError, match="every partition"):
        exchange_port_halo_values(topology, local[:1])
    with pytest.raises(ValueError, match="local vector"):
        exchange_port_halo_values(topology, (local[0][:-1], local[1]))
    with pytest.raises(TypeError, match="common dtype"):
        exchange_port_halo_values(topology, (local[0], local[1].astype(jnp.float32)))
    with pytest.raises(ValueError, match="local cell matrix"):
        local_owned_cell_port_matvec(matrix, topology.partitions[0], local[0])
    with pytest.raises(ValueError, match="local vector"):
        local_owned_cell_port_matvec(matrix[:1], topology.partitions[0], local[0][:-1])
    with pytest.raises(ValueError, match="global cells"):
        owned_ghost_port_matvec(matrix[:1], topology, vector)
    with pytest.raises(ValueError, match="scalar array"):
        owned_ghost_port_generalized_residual(
            matrix,
            matrix,
            topology,
            vector,
            jnp.ones(1),
        )
