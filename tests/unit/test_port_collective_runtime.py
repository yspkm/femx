from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from femx.backends.jax.port_collective_runtime import (  # noqa: E402
    PortCollectiveAddressableShard,
    PortCollectiveArrayReport,
    PortCollectiveCompilerMemoryReport,
    PortCollectiveTimingReport,
    _leading_partition,
    collective_port_named_sharding,
    describe_collective_port_array,
    make_collective_port_array_from_process_local_data,
)
from femx.core.errors import ContractError  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def _one_device_mesh() -> Mesh:
    return Mesh(np.asarray((jax.devices()[0],), dtype=object), ("partition",))


def _shard(**changes: object) -> PortCollectiveAddressableShard:
    values: dict[str, object] = {
        "partition_index": 0,
        "process_index": 0,
        "device_id": 0,
        "device_kind": "cpu",
        "local_shape": (1, 2),
        "logical_bytes": 16,
    }
    values.update(changes)
    return PortCollectiveAddressableShard(**values)  # type: ignore[arg-type]


def _array_report(**changes: object) -> PortCollectiveArrayReport:
    values: dict[str, object] = {
        "name": "owner-vector",
        "global_shape": (1, 2),
        "dtype": "float64",
        "partition_axis_name": "partition",
        "partition_count": 1,
        "global_device_count": 1,
        "process_index": 0,
        "process_count": 1,
        "global_logical_bytes": 16,
        "addressable_logical_bytes": 16,
        "addressable_shards": (_shard(),),
    }
    values.update(changes)
    return PortCollectiveArrayReport(**values)  # type: ignore[arg-type]


def test_process_local_loader_records_exact_named_sharding() -> None:
    mesh = _one_device_mesh()
    host = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    array, report = make_collective_port_array_from_process_local_data(
        "cells",
        host,
        mesh,
    )

    assert np.array_equal(np.asarray(jax.device_get(array)), host)
    assert report.name == "cells"
    assert report.global_shape == (1, 2, 3)
    assert report.global_device_count == report.partition_count == 1
    assert report.process_index == 0
    assert report.process_count == 1
    assert report.global_logical_bytes == report.addressable_logical_bytes == host.nbytes
    assert report.addressable_shards[0].partition_index == 0
    assert report.addressable_shards[0].local_shape == (1, 2, 3)
    assert report.canonical_data()["replication_intent"] == (
        "none; one leading FEM partition per device"
    )
    assert report.canonical_data()["addressable_shards"] == [
        report.addressable_shards[0].canonical_data()
    ]


def test_collective_named_sharding_and_loader_reject_invalid_contracts() -> None:
    mesh = _one_device_mesh()
    with pytest.raises(ContractError, match="nonempty explicit JAX Mesh"):
        collective_port_named_sharding(object(), 1)  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="axis name"):
        collective_port_named_sharding(mesh, 1, axis_name="")
    with pytest.raises(ContractError, match="requested axis"):
        collective_port_named_sharding(mesh, 1, axis_name="wrong")
    with pytest.raises(ContractError, match="rank"):
        collective_port_named_sharding(mesh, 0)
    with pytest.raises(ContractError, match="rank"):
        collective_port_named_sharding(mesh, True)
    with pytest.raises(ContractError, match="leading axis"):
        make_collective_port_array_from_process_local_data("bad", np.asarray(1.0), mesh)
    with pytest.raises(ContractError, match="leading axis"):
        make_collective_port_array_from_process_local_data(
            "bad",
            np.ones((2, 1)),
            mesh,
        )
    with pytest.raises(ContractError, match="integer, floating, or complex"):
        make_collective_port_array_from_process_local_data(
            "bad",
            np.asarray([["text"]]),
            mesh,
        )


def test_describe_rejects_an_array_with_the_wrong_leading_extent() -> None:
    with pytest.raises(ContractError, match="leading axis"):
        describe_collective_port_array("replicated", jnp.ones((2, 2)), _one_device_mesh())


@pytest.mark.parametrize(
    ("index", "shape", "message"),
    [
        ((slice(None),), (1, 2), "rectangular slices"),
        ((0, slice(None)), (1, 2), "rectangular slices"),
        ((slice(0, 0), slice(None)), (1, 2), "one leading partition"),
        ((slice(None), slice(0, 1)), (1, 2), "only its leading axis"),
    ],
)
def test_leading_partition_rejects_noncanonical_shards(
    index: tuple[object, ...],
    shape: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        _leading_partition(index, shape)  # type: ignore[arg-type]


def test_addressable_shard_contract_and_serialization() -> None:
    shard = _shard()
    assert shard.canonical_data() == {
        "partition_index": 0,
        "process_index": 0,
        "device_id": 0,
        "device_kind": "cpu",
        "local_shape": [1, 2],
        "logical_bytes": 16,
    }
    with pytest.raises(ContractError, match="nonnegative integer"):
        _shard(partition_index=True)
    with pytest.raises(ContractError, match="device kind"):
        _shard(device_kind="")
    with pytest.raises(ContractError, match="nonempty tuple"):
        _shard(local_shape=())
    with pytest.raises(ContractError, match="dimensions must be positive"):
        _shard(local_shape=(1, 0))
    with pytest.raises(ContractError, match="exactly one leading partition"):
        _shard(local_shape=(2, 2))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema_version": "v2"}, "schema"),
        ({"name": ""}, "name"),
        ({"global_shape": []}, "nonempty tuple"),
        ({"process_count": 0}, "counts must be positive"),
        ({"process_index": 1}, "below process count"),
        ({"partition_count": 2}, "leading axis"),
        ({"global_device_count": 2}, "one partition"),
        ({"addressable_shards": ()}, "requires addressable"),
        ({"addressable_shards": (_shard(), _shard())}, "repeats"),
        ({"addressable_shards": (_shard(partition_index=1),)}, "out-of-range"),
        ({"addressable_shards": (_shard(process_index=1),)}, "process identity"),
        ({"addressable_logical_bytes": 15}, "byte count"),
        ({"global_logical_bytes": 15}, "exceed global"),
    ],
)
def test_array_report_rejects_inconsistent_metadata(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        _array_report(**changes)


@pytest.mark.parametrize(
    ("peak", "capacity", "expected_risk"),
    [
        (69, None, "not_assessed"),
        (69, 100, "safe"),
        (70, 100, "elevated"),
        (85, 100, "high"),
        (95, 100, "extreme"),
    ],
)
def test_compiler_memory_report_preserves_scope_and_risk_bands(
    peak: int,
    capacity: int | None,
    expected_risk: str,
) -> None:
    report = PortCollectiveCompilerMemoryReport(
        generated_code_bytes=5,
        argument_bytes=peak,
        output_bytes=0,
        alias_bytes=0,
        temporary_bytes=0,
        hbm_capacity_bytes_per_device=capacity,
    )
    assert report.compiler_peak_bytes == peak
    assert report.risk == expected_risk
    assert report.hbm_fraction == (None if capacity is None else peak / capacity)
    assert report.canonical_data()["claim_scope"] == "compiler estimate; not live HBM usage"


def test_compiler_memory_report_validates_schema_counts_and_alias_floor() -> None:
    report = PortCollectiveCompilerMemoryReport(
        generated_code_bytes=0,
        argument_bytes=2,
        output_bytes=3,
        alias_bytes=10,
        temporary_bytes=4,
    )
    assert report.compiler_peak_bytes == 0
    with pytest.raises(ContractError, match="schema"):
        replace(report, schema_version="v2")
    with pytest.raises(ContractError, match="nonnegative integer"):
        replace(report, argument_bytes=True)
    with pytest.raises(ContractError, match="positive"):
        replace(report, hbm_capacity_bytes_per_device=0)


def test_timing_report_serializes_phase_separated_synchronized_samples() -> None:
    report = PortCollectiveTimingReport(
        lowering_seconds=0,
        compilation_seconds=1,
        warmup_seconds=2,
        execution_seconds=(3, 1, 2),
    )
    data = report.canonical_data()
    assert data["execution_min_seconds"] == 1.0
    assert data["execution_median_seconds"] == 2.0
    assert data["execution_max_seconds"] == 3.0
    assert data["synchronization"] == "every timed result blocked until ready"
    with pytest.raises(ContractError, match="schema"):
        replace(report, schema_version="v2")
    with pytest.raises(ContractError, match="finite nonnegative"):
        replace(report, lowering_seconds=float("nan"))
    with pytest.raises(ContractError, match="finite nonnegative"):
        replace(report, lowering_seconds=True)
    with pytest.raises(ContractError, match="requires execution samples"):
        replace(report, execution_seconds=())
