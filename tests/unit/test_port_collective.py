from __future__ import annotations

import math
from dataclasses import replace

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from femx.backends.jax.port_collective import (  # noqa: E402
    PORT_COLLECTIVE_LAYOUT_SCHEMA,
    PORT_COLLECTIVE_MESH_REPORT_SCHEMA,
    PortCollectiveDeviceAssignment,
    PortCollectiveHaloLink,
    PortCollectiveMeshReport,
    PortCollectiveStorageReport,
    assert_finite_collective_port_result,
    build_packed_collective_port_matvec,
    build_validation_collective_port_matvec,
    collective_port_relative_difference,
    describe_collective_port_mesh,
    pack_collective_port_cell_matrix,
    pack_collective_port_owned_vector,
    prepare_collective_port_layout,
    unpack_collective_port_owned_vector,
)
from femx.backends.jax.port_matrix_free import matrix_free_port_matvec  # noqa: E402
from femx.backends.jax.port_owned_ghost import (  # noqa: E402
    prepare_owned_ghost_port_topology,
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


def _topology(*, partition_count: int = 2):
    return prepare_owned_ghost_port_topology(
        CELL_MAP,
        CELL_OWNERS,
        free_dof_count=7,
        partition_count=partition_count,
        dof_owners=DOF_OWNERS,
    )


def _single_partition_layout():
    topology = prepare_owned_ghost_port_topology(
        np.arange(6, dtype=np.int64).reshape(1, 6),
        np.asarray((0,), dtype=np.int64),
        free_dof_count=6,
        partition_count=1,
    )
    return prepare_collective_port_layout(topology)


def _cpu_mesh(*, axis_name: str = "partition") -> Mesh:
    return Mesh(np.asarray((jax.devices("cpu")[0],), dtype=object), (axis_name,))


def _cell_matrices() -> jax.Array:
    values = np.arange(72, dtype=np.float64).reshape(2, 6, 6) / 19.0
    values[:, np.arange(6), np.arange(6)] += 4.0
    return jnp.asarray(values)


def test_layout_preserves_canonical_identity_and_reports_transport_padding() -> None:
    topology = _topology()
    layout = prepare_collective_port_layout(topology)

    assert layout.schema_version == PORT_COLLECTIVE_LAYOUT_SCHEMA
    assert layout.partition_count == 2
    assert layout.cell_capacity == 1
    assert layout.owned_dof_capacity == 4
    assert layout.ghost_dof_capacity == 3
    assert layout.local_dof_capacity == 7
    assert layout.constrained_transport_sentinel == 7
    np.testing.assert_array_equal(layout.cell_ids, ((0,), (1,)))
    np.testing.assert_array_equal(layout.owned_dof_ids, ((0, 1, 2, 3), (4, 5, 6, 7)))
    np.testing.assert_array_equal(layout.ghost_dof_ids, ((4, 5, 7), (1, 2, 3)))
    np.testing.assert_array_equal(
        layout.cell_local_dofs,
        (((0, 1, 2, 3, 4, 5),), ((4, 5, 2, 0, 1, 6),)),
    )
    assert len(layout.halo_links) == 2
    first, second = layout.halo_links
    assert (first.owner_partition, first.ghost_partition) == (0, 1)
    np.testing.assert_array_equal(first.global_dofs, (1, 2, 3))
    np.testing.assert_array_equal(first.owner_slots, (1, 2, 3))
    np.testing.assert_array_equal(first.ghost_slots, (4, 5, 6))
    assert (second.owner_partition, second.ghost_partition) == (1, 0)
    np.testing.assert_array_equal(second.owner_slots, (0, 1))
    np.testing.assert_array_equal(second.ghost_slots, (4, 5))

    report = layout.storage_report
    assert report.actual_cell_slots == report.allocated_cell_slots == 2
    assert report.actual_owned_dof_slots == 7
    assert report.allocated_owned_dof_slots == 8
    assert report.actual_ghost_dof_slots == report.halo_value_count == 5
    assert report.allocated_ghost_dof_slots == 6
    assert report.halo_link_count == 2
    assert report.cell_padding_fraction == 0.0
    assert report.owned_dof_padding_fraction == pytest.approx(1.0 / 8.0)
    assert report.ghost_dof_padding_fraction == pytest.approx(1.0 / 6.0)
    assert layout.digest() == "e4bf997c5d265947882dc46478b1a8953e63b0f601f2f27fc5eeb05c284cd8b4"
    assert prepare_collective_port_layout(topology).digest() == layout.digest()

    for array in (
        layout.cell_ids,
        layout.owned_dof_ids,
        layout.ghost_dof_ids,
        layout.cell_local_dofs,
        *(link.owner_slots for link in layout.halo_links),
    ):
        assert not array.flags.writeable


def test_empty_partition_uses_only_explicit_inactive_transport_slots() -> None:
    layout = prepare_collective_port_layout(_topology(partition_count=3))
    matrix = _cell_matrices().astype(jnp.complex128)
    vector = jnp.arange(7, dtype=jnp.float64) + 1.0j * jnp.arange(7, dtype=jnp.float64)

    packed_cells = jax.jit(lambda values: pack_collective_port_cell_matrix(layout, values))(matrix)
    packed_owned = jax.jit(lambda values: pack_collective_port_owned_vector(layout, values))(vector)
    unpacked = jax.jit(lambda values: unpack_collective_port_owned_vector(layout, values))(
        packed_owned
    )

    assert packed_cells.shape == (3, 1, 6, 6)
    assert packed_owned.shape == (3, 4)
    np.testing.assert_array_equal(packed_cells[2], np.zeros((1, 6, 6)))
    np.testing.assert_array_equal(packed_owned[2], np.zeros(4, dtype=np.complex128))
    np.testing.assert_array_equal(layout.cell_ids[2], (2,))
    np.testing.assert_array_equal(layout.owned_dof_ids[2], (7, 7, 7, 7))
    np.testing.assert_array_equal(layout.ghost_dof_ids[2], (7, 7, 7))
    np.testing.assert_array_equal(layout.cell_local_dofs[2], np.full((1, 6), 7))
    np.testing.assert_array_equal(unpacked, vector)


def test_single_device_packed_and_validation_operators_match_serial_jit_and_vjp() -> None:
    layout = _single_partition_layout()
    mesh = _cpu_mesh()
    packed_operator = build_packed_collective_port_matvec(layout, mesh)
    validation_operator = build_validation_collective_port_matvec(layout, mesh)
    matrix = jnp.asarray(np.arange(36, dtype=np.float64).reshape(1, 6, 6) / 11.0)
    vector = jnp.asarray((0.2, -0.1, 0.4, 0.8, -0.3, 0.5))
    mapping = jnp.asarray(layout.cell_local_dofs)
    packed_cells = pack_collective_port_cell_matrix(layout, matrix)
    packed_owned = pack_collective_port_owned_vector(layout, vector)

    packed_result = jax.jit(packed_operator)(packed_cells, mapping, packed_owned)
    observed = jax.jit(validation_operator)(matrix, vector)
    expected = matrix_free_port_matvec(
        matrix,
        jnp.asarray(layout.topology.cell_reduced_dofs),
        vector,
    )
    np.testing.assert_allclose(
        unpack_collective_port_owned_vector(layout, packed_result),
        expected,
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(observed, expected, rtol=2.0e-15, atol=2.0e-15)

    weights = jnp.asarray((-0.2, 0.7, 0.3, -0.4, 0.5, 0.9))

    def collective_objective(local_matrix: jax.Array, values: jax.Array) -> jax.Array:
        return jnp.vdot(weights, validation_operator(local_matrix, values)).real

    def serial_objective(local_matrix: jax.Array, values: jax.Array) -> jax.Array:
        result = matrix_free_port_matvec(
            local_matrix,
            jnp.asarray(layout.topology.cell_reduced_dofs),
            values,
        )
        return jnp.vdot(weights, result).real

    observed_value, observed_gradients = jax.jit(
        jax.value_and_grad(collective_objective, argnums=(0, 1))
    )(matrix, vector)
    expected_value, expected_gradients = jax.jit(
        jax.value_and_grad(serial_objective, argnums=(0, 1))
    )(matrix, vector)
    np.testing.assert_allclose(observed_value, expected_value, rtol=2.0e-15, atol=2.0e-15)
    for observed_gradient, expected_gradient in zip(
        observed_gradients,
        expected_gradients,
        strict=True,
    ):
        np.testing.assert_allclose(
            observed_gradient,
            expected_gradient,
            rtol=2.0e-15,
            atol=2.0e-15,
        )

    assert_finite_collective_port_result(observed)
    assert collective_port_relative_difference(observed, expected) < 2.0e-15


def test_layout_records_fail_closed_on_noncanonical_transport_data() -> None:
    layout = prepare_collective_port_layout(_topology())
    with pytest.raises(ContractError, match="schema"):
        replace(layout, schema_version="femx.jax.port_collective/v2")
    with pytest.raises(ContractError, match="cell ids disagrees"):
        replace(layout, cell_ids=np.asarray(((1,), (0,)), dtype=np.int64))
    with pytest.raises(ContractError, match=r"owned dof ids.*rank-2"):
        replace(layout, owned_dof_ids=np.arange(8, dtype=np.int64))
    with pytest.raises(ContractError, match=r"ghost dof ids.*integer"):
        replace(layout, ghost_dof_ids=np.ones((2, 3), dtype=np.float64))
    with pytest.raises(ContractError, match=r"cell local dofs.*shape"):
        replace(layout, cell_local_dofs=np.zeros((2, 2, 6), dtype=np.int64))
    with pytest.raises(ContractError, match="halo links disagree"):
        replace(layout, halo_links=layout.halo_links[:1])
    changed = replace(
        layout.halo_links[0],
        owner_slots=np.asarray((0, 2, 3), dtype=np.int64),
    )
    with pytest.raises(ContractError, match="halo links disagree"):
        replace(layout, halo_links=(changed, layout.halo_links[1]))
    with pytest.raises(ContractError, match="PortOwnedGhostTopology"):
        prepare_collective_port_layout(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("factory", "message"),
    (
        (
            lambda: PortCollectiveHaloLink(
                -1,
                1,
                np.asarray((0,)),
                np.asarray((0,)),
                np.asarray((1,)),
            ),
            "owner partition",
        ),
        (
            lambda: PortCollectiveHaloLink(
                0,
                0,
                np.asarray((0,)),
                np.asarray((0,)),
                np.asarray((1,)),
            ),
            "must differ",
        ),
        (
            lambda: PortCollectiveHaloLink(
                0,
                1,
                np.asarray((), dtype=np.int64),
                np.asarray((), dtype=np.int64),
                np.asarray((), dtype=np.int64),
            ),
            "at least one DOF",
        ),
        (
            lambda: PortCollectiveHaloLink(
                0,
                1,
                np.asarray((0, 1)),
                np.asarray((0,)),
                np.asarray((2, 3)),
            ),
            "equal lengths",
        ),
        (
            lambda: PortCollectiveHaloLink(
                0,
                1,
                np.asarray((1, 0)),
                np.asarray((0, 1)),
                np.asarray((2, 3)),
            ),
            "strictly increasing",
        ),
        (
            lambda: PortCollectiveHaloLink(
                0,
                1,
                np.asarray((0,), dtype=float),
                np.asarray((0,)),
                np.asarray((1,)),
            ),
            "integer array",
        ),
        (
            lambda: PortCollectiveHaloLink(
                0,
                1,
                np.asarray((-1,)),
                np.asarray((0,)),
                np.asarray((1,)),
            ),
            "cannot be negative",
        ),
    ),
)
def test_collective_halo_link_rejects_ambiguous_state(
    factory: object,
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        factory()  # type: ignore[operator]


def test_storage_report_rejects_impossible_counts_and_handles_no_ghosts() -> None:
    report = PortCollectiveStorageReport(1, 1, 2, 2, 0, 0, 0, 0)
    assert report.ghost_dof_padding_fraction == 0.0

    with pytest.raises(ContractError, match="nonnegative"):
        PortCollectiveStorageReport(-1, 1, 1, 1, 0, 0, 0, 0)
    with pytest.raises(ContractError, match="cell slots"):
        PortCollectiveStorageReport(2, 1, 1, 1, 0, 0, 0, 0)
    with pytest.raises(ContractError, match="owned DOF slots"):
        PortCollectiveStorageReport(1, 1, 2, 1, 0, 0, 0, 0)
    with pytest.raises(ContractError, match="ghost DOF slots"):
        PortCollectiveStorageReport(1, 1, 1, 1, 2, 1, 1, 2)
    with pytest.raises(ContractError, match="exactly one halo"):
        PortCollectiveStorageReport(1, 1, 1, 1, 1, 1, 1, 0)


def test_pack_unpack_and_packed_operator_reject_mismatched_arrays() -> None:
    layout = _single_partition_layout()
    matrix = jnp.ones((1, 6, 6))
    vector = jnp.ones(6)
    packed_operator = build_packed_collective_port_matvec(layout, _cpu_mesh())
    packed_cells = pack_collective_port_cell_matrix(layout, matrix)
    packed_owned = pack_collective_port_owned_vector(layout, vector)
    packed_map = jnp.asarray(layout.cell_local_dofs)

    with pytest.raises(ValueError, match="global cells"):
        pack_collective_port_cell_matrix(layout, jnp.ones((2, 6, 6)))
    with pytest.raises(TypeError, match="floating or complex"):
        pack_collective_port_cell_matrix(layout, jnp.ones((1, 6, 6), dtype=jnp.int32))
    with pytest.raises(ValueError, match="global free DOFs"):
        pack_collective_port_owned_vector(layout, jnp.ones(5))
    with pytest.raises(TypeError, match="floating or complex"):
        pack_collective_port_owned_vector(layout, jnp.ones(6, dtype=jnp.int32))
    with pytest.raises(ValueError, match="packed owner vector"):
        unpack_collective_port_owned_vector(layout, jnp.ones((1, 5)))
    with pytest.raises(TypeError, match="floating or complex"):
        unpack_collective_port_owned_vector(layout, jnp.ones((1, 6), dtype=jnp.int32))

    with pytest.raises(ValueError, match="packed cell matrix"):
        packed_operator(packed_cells[:, :, :, :5], packed_map, packed_owned)
    with pytest.raises(ValueError, match="packed cell map"):
        packed_operator(packed_cells, packed_map[:, :, :5], packed_owned)
    with pytest.raises(ValueError, match="packed owner vector"):
        packed_operator(packed_cells, packed_map, packed_owned[:, :5])
    with pytest.raises(TypeError, match=r"cell matrix.*floating"):
        packed_operator(packed_cells.astype(jnp.int32), packed_map, packed_owned)
    with pytest.raises(TypeError, match=r"cell map.*integer"):
        packed_operator(packed_cells, packed_map.astype(jnp.float64), packed_owned)
    with pytest.raises(TypeError, match=r"owner vector.*floating"):
        packed_operator(packed_cells, packed_map, packed_owned.astype(jnp.int32))


def test_mesh_contract_and_synchronized_result_checks_fail_closed() -> None:
    one = _single_partition_layout()
    two = prepare_collective_port_layout(_topology())
    mesh = _cpu_mesh()

    with pytest.raises(ContractError, match="explicit JAX Mesh"):
        build_packed_collective_port_matvec(one, object())  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="nonempty string"):
        build_packed_collective_port_matvec(one, mesh, axis_name="")
    with pytest.raises(ContractError, match="one-dimensional axis"):
        build_packed_collective_port_matvec(one, mesh, axis_name="wrong")
    two_axis_mesh = Mesh(
        np.asarray((jax.devices("cpu")[0],), dtype=object).reshape(1, 1),
        ("partition", "replica"),
    )
    with pytest.raises(ContractError, match="one-dimensional axis"):
        build_packed_collective_port_matvec(one, two_axis_mesh)
    with pytest.raises(ContractError, match="one device per FEM partition"):
        build_packed_collective_port_matvec(two, mesh)

    with pytest.raises(FloatingPointError, match="non-finite"):
        assert_finite_collective_port_result(jnp.asarray((1.0, jnp.nan)))
    assert collective_port_relative_difference(jnp.zeros(2), jnp.zeros(2)) == 0.0
    assert math.isinf(collective_port_relative_difference(jnp.ones(2), jnp.zeros(2)))


def test_mesh_report_records_partition_process_and_addressability_without_worker_aliases() -> None:
    layout = _single_partition_layout()
    report = describe_collective_port_mesh(layout, _cpu_mesh())

    assert report.schema_version == PORT_COLLECTIVE_MESH_REPORT_SCHEMA
    assert report.axis_name == "partition"
    assert report.partition_count == report.global_device_count == 1
    assert report.addressable_device_count == report.process_count == 1
    assert not report.is_multi_process
    assert report.layout_sha256 == layout.digest()
    assert len(report.assignments) == 1
    assignment = report.assignments[0]
    assert assignment.partition_index == assignment.process_index == 0
    assert assignment.device_id == 0
    assert assignment.platform == "cpu"
    assert assignment.device_kind
    assert assignment.addressable
    assert report.canonical_data() == {
        "schema_version": PORT_COLLECTIVE_MESH_REPORT_SCHEMA,
        "axis_name": "partition",
        "partition_count": 1,
        "global_device_count": 1,
        "addressable_device_count": 1,
        "process_count": 1,
        "is_multi_process": False,
        "layout_sha256": layout.digest(),
        "assignments": [assignment.canonical_data()],
    }


def test_mesh_report_records_reject_inconsistent_or_ambiguous_identity() -> None:
    assignment = PortCollectiveDeviceAssignment(0, 0, 0, "cpu", "cpu", True)
    report = PortCollectiveMeshReport(
        axis_name="partition",
        partition_count=1,
        global_device_count=1,
        addressable_device_count=1,
        process_count=1,
        layout_sha256="a" * 64,
        assignments=(assignment,),
    )
    with pytest.raises(ContractError, match="schema"):
        replace(report, schema_version="wrong")
    with pytest.raises(ContractError, match="axis name"):
        replace(report, axis_name="")
    with pytest.raises(ContractError, match="must be positive"):
        replace(report, process_count=0)
    with pytest.raises(ContractError, match="one global device"):
        replace(report, global_device_count=2)
    with pytest.raises(ContractError, match="cannot exceed"):
        replace(report, addressable_device_count=2)
    with pytest.raises(ContractError, match="SHA-256"):
        replace(report, layout_sha256="A" * 64)
    with pytest.raises(ContractError, match="every partition"):
        replace(report, assignments=())
    with pytest.raises(ContractError, match="partition order"):
        replace(
            report,
            partition_count=2,
            global_device_count=2,
            assignments=(assignment, replace(assignment, partition_index=2, device_id=1)),
        )
    with pytest.raises(ContractError, match="unique devices"):
        replace(
            report,
            partition_count=2,
            global_device_count=2,
            addressable_device_count=2,
            assignments=(assignment, replace(assignment, partition_index=1)),
        )
    with pytest.raises(ContractError, match="addressable assignments"):
        replace(report, assignments=(replace(assignment, addressable=False),))
    with pytest.raises(ContractError, match="process assignments"):
        replace(report, process_count=2)

    for updates, message in (
        ({"partition_index": -1}, "nonnegative"),
        ({"process_index": True}, "nonnegative"),
        ({"platform": ""}, "platform"),
        ({"device_kind": " "}, "device kind"),
        ({"addressable": 1}, "boolean"),
    ):
        with pytest.raises(ContractError, match=message):
            replace(assignment, **updates)
