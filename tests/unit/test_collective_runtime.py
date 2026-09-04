from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from femx.backends.jax.collective_runtime import (  # noqa: E402
    COLLECTIVE_ARRAY_REPORT_SCHEMA,
    COLLECTIVE_MEMORY_REPORT_SCHEMA,
    COLLECTIVE_MESH_REPORT_SCHEMA,
    COLLECTIVE_REPLICATED_ARRAY_REPORT_SCHEMA,
    COLLECTIVE_TIMING_REPORT_SCHEMA,
    CollectiveAddressableShard,
    CollectiveArrayReport,
    CollectiveCompilerMemoryReport,
    CollectiveDeviceAssignment,
    CollectiveMeshReport,
    CollectiveReplicatedArrayReport,
    CollectiveTimingReport,
    collective_named_sharding,
    describe_collective_array,
    describe_collective_mesh,
    describe_replicated_array,
    make_collective_array_from_process_local_data,
    make_replicated_array_from_process_local_data,
)
from femx.backends.jax.scalar_collective import (  # noqa: E402
    prepare_collective_scalar_h1_layout,
)
from femx.backends.jax.scalar_owned_ghost import (  # noqa: E402
    prepare_scalar_h1_owned_ghost_topology,
)
from femx.core.errors import ContractError  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def _mesh() -> Mesh:
    return Mesh(np.asarray((jax.devices()[0],), dtype=object), ("partition",))


def _assignment(**changes: object) -> CollectiveDeviceAssignment:
    values: dict[str, object] = {
        "partition_index": 0,
        "process_index": 0,
        "device_id": 0,
        "platform": "cpu",
        "device_kind": "cpu",
        "addressable": True,
    }
    values.update(changes)
    return CollectiveDeviceAssignment(**values)  # type: ignore[arg-type]


def _mesh_report(**changes: object) -> CollectiveMeshReport:
    values: dict[str, object] = {
        "axis_name": "partition",
        "partition_count": 1,
        "global_device_count": 1,
        "addressable_device_count": 1,
        "process_count": 1,
        "layout_sha256": "a" * 64,
        "assignments": (_assignment(),),
    }
    values.update(changes)
    return CollectiveMeshReport(**values)  # type: ignore[arg-type]


def test_generic_runtime_schema_ids_are_stable() -> None:
    assert COLLECTIVE_ARRAY_REPORT_SCHEMA == "femx.jax.collective.array_report/v1"
    assert (
        COLLECTIVE_REPLICATED_ARRAY_REPORT_SCHEMA
        == "femx.jax.collective.replicated_array_report/v1"
    )
    assert COLLECTIVE_MESH_REPORT_SCHEMA == "femx.jax.collective.mesh_report/v1"
    assert COLLECTIVE_MEMORY_REPORT_SCHEMA == "femx.jax.collective.memory_report/v1"
    assert COLLECTIVE_TIMING_REPORT_SCHEMA == "femx.jax.collective.timing_report/v1"


def test_generic_loader_supports_boolean_owner_masks() -> None:
    host = np.asarray(((True, False, True),), dtype=np.bool_)
    array, report = make_collective_array_from_process_local_data("mask", host, _mesh())
    np.testing.assert_array_equal(np.asarray(jax.device_get(array)), host)
    assert report.schema_version == COLLECTIVE_ARRAY_REPORT_SCHEMA
    assert report.dtype == "bool"
    assert describe_collective_array("mask", array, _mesh()) == report


def test_generic_loader_uses_the_per_shard_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[tuple[int, ...]] = []
    original = jax.make_array_from_callback

    def recording_callback(shape, sharding, callback):  # type: ignore[no-untyped-def]
        def record(index):  # type: ignore[no-untyped-def]
            value = np.asarray(callback(index))
            observed.append(value.shape)
            return value

        return original(shape, sharding, record)

    monkeypatch.setattr(jax, "make_array_from_callback", recording_callback)
    host = np.asarray(((1.0, 2.0),), dtype=np.float32)
    array, _report = make_collective_array_from_process_local_data("values", host, _mesh())

    np.testing.assert_array_equal(np.asarray(jax.device_get(array)), host)
    assert observed == [(1, 2)]


def test_replicated_loader_records_explicit_intent_and_byte_accounting() -> None:
    host = np.arange(6, dtype=np.float32).reshape(2, 3)
    array, report = make_replicated_array_from_process_local_data(
        "coarse-weights",
        host,
        _mesh(),
        replication_intent="bounded multilevel coarse interpolation",
    )
    np.testing.assert_array_equal(np.asarray(jax.device_get(array)), host)
    assert report.schema_version == COLLECTIVE_REPLICATED_ARRAY_REPORT_SCHEMA
    assert report.logical_bytes_per_replica == host.nbytes
    assert report.addressable_logical_bytes == host.nbytes
    assert report.global_replica_logical_bytes == host.nbytes
    assert report.canonical_data()["partition_spec"] == []
    assert (
        describe_replicated_array(
            "coarse-weights",
            array,
            _mesh(),
            replication_intent="bounded multilevel coarse interpolation",
        )
        == report
    )


def test_replicated_loader_fails_closed_on_invalid_input_and_report() -> None:
    with pytest.raises(ContractError, match="nonempty and nonscalar"):
        make_replicated_array_from_process_local_data(
            "scalar",
            np.asarray(1.0),
            _mesh(),
            replication_intent="bounded test",
        )
    with pytest.raises(ContractError, match="replication intent"):
        make_replicated_array_from_process_local_data(
            "values",
            np.ones((1,), dtype=np.float32),
            _mesh(),
            replication_intent="",
        )
    array, _ = make_replicated_array_from_process_local_data(
        "values",
        np.ones((1,), dtype=np.float32),
        _mesh(),
        replication_intent="bounded test",
    )
    with pytest.raises(ContractError, match="replication intent"):
        describe_replicated_array(
            "values",
            array,
            _mesh(),
            replication_intent="",
        )
    with pytest.raises(ContractError, match="byte accounting"):
        CollectiveReplicatedArrayReport(
            name="values",
            global_shape=(2,),
            dtype="float32",
            global_device_count=1,
            addressable_device_count=1,
            process_index=0,
            process_count=1,
            logical_bytes_per_replica=8,
            addressable_logical_bytes=4,
            global_replica_logical_bytes=8,
            replication_intent="bounded test",
        )


def test_generic_loader_rejects_unsupported_data_and_wrong_sharding() -> None:
    with pytest.raises(ContractError, match="boolean, integer, floating, or complex"):
        make_collective_array_from_process_local_data(
            "text",
            np.asarray((("x",),)),
            _mesh(),
        )
    with pytest.raises(ContractError, match="leading axis"):
        make_collective_array_from_process_local_data("scalar", np.asarray(1.0), _mesh())
    with pytest.raises(ContractError, match="leading axis"):
        describe_collective_array("wrong-leading-extent", jnp.ones((2, 2)), _mesh())


def test_collective_named_sharding_validates_mesh_axis_and_rank() -> None:
    mesh = _mesh()
    assert collective_named_sharding(mesh, 2).spec == jax.sharding.PartitionSpec(
        "partition",
        None,
    )
    with pytest.raises(ContractError, match="nonempty explicit JAX Mesh"):
        collective_named_sharding(object(), 1)  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="axis name"):
        collective_named_sharding(mesh, 1, axis_name="")
    with pytest.raises(ContractError, match="requested axis"):
        collective_named_sharding(mesh, 1, axis_name="other")
    for rank in (0, True):
        with pytest.raises(ContractError, match="rank"):
            collective_named_sharding(mesh, rank)  # type: ignore[arg-type]


def test_device_assignment_serializes_and_rejects_invalid_fields() -> None:
    assert _assignment().canonical_data()["addressable"] is True
    with pytest.raises(ContractError, match="nonnegative integer"):
        _assignment(process_index=True)
    with pytest.raises(ContractError, match="platform"):
        _assignment(platform="")
    with pytest.raises(ContractError, match="addressable flag"):
        _assignment(addressable=1)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema_version": "v2"}, "schema"),
        ({"axis_name": ""}, "axis name"),
        ({"partition_count": 0}, "must be positive"),
        ({"global_device_count": 2}, "one global device"),
        ({"addressable_device_count": 2}, "cannot exceed"),
        ({"layout_sha256": "bad"}, "SHA-256"),
        ({"assignments": ()}, "every partition"),
        ({"assignments": (_assignment(partition_index=1),)}, "partition order"),
        (
            {
                "partition_count": 2,
                "global_device_count": 2,
                "assignments": (_assignment(), _assignment(partition_index=1)),
            },
            "unique devices",
        ),
        ({"addressable_device_count": 0}, "must be positive"),
        ({"process_count": 2}, "process assignments"),
    ],
)
def test_mesh_report_rejects_inconsistent_metadata(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        _mesh_report(**changes)


def test_mesh_report_and_live_description_preserve_process_device_identity() -> None:
    topology = prepare_scalar_h1_owned_ghost_topology(
        np.asarray(((0, 1, 2),), dtype=np.int64),
        np.asarray((0,), dtype=np.int64),
        node_count=3,
        free_nodes=np.asarray((0, 1), dtype=np.int64),
        partition_count=1,
    )
    layout = prepare_collective_scalar_h1_layout(topology)
    report = describe_collective_mesh(layout.transport, _mesh())
    assert report.is_multi_process is False
    assert report.layout_sha256 == layout.transport.digest()
    scalar_report = describe_collective_mesh(
        layout.transport,
        _mesh(),
        layout_sha256=layout.digest(),
    )
    assert scalar_report.layout_sha256 == layout.digest()
    assert report.canonical_data()["assignments"] == [report.assignments[0].canonical_data()]


def test_generic_reports_validate_and_serialize() -> None:
    shard = CollectiveAddressableShard(0, 0, 0, "cpu", (1, 2), 8)
    array = CollectiveArrayReport(
        name="values",
        global_shape=(1, 2),
        dtype="float32",
        partition_axis_name="partition",
        partition_count=1,
        global_device_count=1,
        process_index=0,
        process_count=1,
        global_logical_bytes=8,
        addressable_logical_bytes=8,
        addressable_shards=(shard,),
    )
    assert array.canonical_data()["schema_version"] == COLLECTIVE_ARRAY_REPORT_SCHEMA
    memory = CollectiveCompilerMemoryReport(1, 10, 2, 1, 3, 100)
    assert memory.compiler_peak_bytes == 14
    assert memory.risk == "safe"
    timing = CollectiveTimingReport(0, 1, 2, (3, 1, 2))
    assert timing.canonical_data()["execution_median_seconds"] == 2.0
    with pytest.raises(ContractError, match="schema"):
        replace(array, schema_version="v2")
    with pytest.raises(ContractError, match="schema"):
        replace(memory, schema_version="v2")
    with pytest.raises(ContractError, match="schema"):
        replace(timing, schema_version="v2")
