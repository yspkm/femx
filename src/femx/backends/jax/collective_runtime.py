"""Runtime-only sharding and evidence helpers for fixed-capacity FEM collectives.

This module never discovers devices or initializes a distributed runtime.  Callers must construct
an explicit global :class:`jax.sharding.Mesh` after ``jax.distributed.initialize()`` and pass it in.
The helpers create buffers only for process-addressable leading-axis partitions and retain enough
information to audit global versus addressable storage.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import ClassVar

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from femx.core.errors import ContractError

from .collective import CollectiveLayout, validate_collective_mesh

COLLECTIVE_ARRAY_REPORT_SCHEMA = "femx.jax.collective.array_report/v1"
COLLECTIVE_REPLICATED_ARRAY_REPORT_SCHEMA = "femx.jax.collective.replicated_array_report/v1"
COLLECTIVE_MESH_REPORT_SCHEMA = "femx.jax.collective.mesh_report/v1"
COLLECTIVE_MEMORY_REPORT_SCHEMA = "femx.jax.collective.memory_report/v1"
COLLECTIVE_TIMING_REPORT_SCHEMA = "femx.jax.collective.timing_report/v1"


def _require_nonnegative_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{label} must be a nonnegative integer")
    return value


def _require_nonnegative_finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be a finite nonnegative number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ContractError(f"{label} must be a finite nonnegative number")
    return converted


def _canonical_shape(value: object, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise ContractError(f"{label} must be a nonempty tuple")
    result = tuple(_require_nonnegative_integer(item, label=f"{label} dimension") for item in value)
    if any(item == 0 for item in result):
        raise ContractError(f"{label} dimensions must be positive")
    return result


@dataclass(frozen=True, slots=True)
class CollectiveAddressableShard:
    """One process-addressable leading-axis partition of a global JAX array."""

    partition_index: int
    process_index: int
    device_id: int
    device_kind: str
    local_shape: tuple[int, ...]
    logical_bytes: int

    def __post_init__(self) -> None:
        for name in ("partition_index", "process_index", "device_id", "logical_bytes"):
            _require_nonnegative_integer(getattr(self, name), label=name.replace("_", " "))
        if not isinstance(self.device_kind, str) or not self.device_kind.strip():
            raise ContractError("collective shard device kind must be nonempty")
        shape = _canonical_shape(self.local_shape, label="collective shard local shape")
        if shape[0] != 1:
            raise ContractError("collective shard must contain exactly one leading partition")
        object.__setattr__(self, "local_shape", shape)

    def canonical_data(self) -> dict[str, object]:
        """Return stable JSON-compatible shard metadata."""

        return {
            "partition_index": self.partition_index,
            "process_index": self.process_index,
            "device_id": self.device_id,
            "device_kind": self.device_kind,
            "local_shape": list(self.local_shape),
            "logical_bytes": self.logical_bytes,
        }


@dataclass(frozen=True, slots=True)
class CollectiveDeviceAssignment:
    """One explicit FEM-partition to JAX-device assignment."""

    partition_index: int
    process_index: int
    device_id: int
    platform: str
    device_kind: str
    addressable: bool

    def __post_init__(self) -> None:
        for name in ("partition_index", "process_index", "device_id"):
            _require_nonnegative_integer(getattr(self, name), label=name.replace("_", " "))
        for name in ("platform", "device_kind"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f"collective device {name.replace('_', ' ')} must be nonempty")
        if not isinstance(self.addressable, bool):
            raise ContractError("collective device addressable flag must be boolean")

    def canonical_data(self) -> dict[str, object]:
        """Return stable JSON-compatible device identity."""

        return {
            "partition_index": self.partition_index,
            "process_index": self.process_index,
            "device_id": self.device_id,
            "platform": self.platform,
            "device_kind": self.device_kind,
            "addressable": self.addressable,
        }


@dataclass(frozen=True, slots=True)
class CollectiveMeshReport:
    """Observed mapping for one already-constructed solver-neutral collective Mesh."""

    axis_name: str
    partition_count: int
    global_device_count: int
    addressable_device_count: int
    process_count: int
    layout_sha256: str
    assignments: tuple[CollectiveDeviceAssignment, ...]
    schema_version: str = COLLECTIVE_MESH_REPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != COLLECTIVE_MESH_REPORT_SCHEMA:
            raise ContractError(
                f"collective Mesh report schema must be {COLLECTIVE_MESH_REPORT_SCHEMA!r}"
            )
        if not isinstance(self.axis_name, str) or not self.axis_name:
            raise ContractError("collective Mesh report axis name must be nonempty")
        for name in (
            "partition_count",
            "global_device_count",
            "addressable_device_count",
            "process_count",
        ):
            value = _require_nonnegative_integer(getattr(self, name), label=name.replace("_", " "))
            if value == 0:
                raise ContractError(
                    f"collective Mesh report {name.replace('_', ' ')} must be positive"
                )
        if self.global_device_count != self.partition_count:
            raise ContractError("collective Mesh report requires one global device per partition")
        if self.addressable_device_count > self.global_device_count:
            raise ContractError("addressable device count cannot exceed global device count")
        if (
            not isinstance(self.layout_sha256, str)
            or len(self.layout_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.layout_sha256)
        ):
            raise ContractError("collective Mesh report layout SHA-256 must be canonical")
        assignments = tuple(self.assignments)
        if len(assignments) != self.partition_count:
            raise ContractError("collective Mesh report must contain every partition assignment")
        if tuple(record.partition_index for record in assignments) != tuple(
            range(self.partition_count)
        ):
            raise ContractError("collective Mesh assignments must follow partition order")
        device_keys = tuple((record.process_index, record.device_id) for record in assignments)
        if len(set(device_keys)) != len(device_keys):
            raise ContractError("collective Mesh assignments must identify unique devices")
        if sum(record.addressable for record in assignments) != self.addressable_device_count:
            raise ContractError("collective Mesh addressable assignments disagree with their count")
        if len({record.process_index for record in assignments}) != self.process_count:
            raise ContractError("collective Mesh process assignments disagree with their count")
        object.__setattr__(self, "assignments", assignments)

    @property
    def is_multi_process(self) -> bool:
        """Return whether the Mesh spans more than one JAX process."""

        return self.process_count > 1

    def canonical_data(self) -> dict[str, object]:
        """Return stable JSON-compatible Mesh evidence."""

        return {
            "schema_version": self.schema_version,
            "axis_name": self.axis_name,
            "partition_count": self.partition_count,
            "global_device_count": self.global_device_count,
            "addressable_device_count": self.addressable_device_count,
            "process_count": self.process_count,
            "is_multi_process": self.is_multi_process,
            "layout_sha256": self.layout_sha256,
            "assignments": [record.canonical_data() for record in self.assignments],
        }


@dataclass(frozen=True, slots=True)
class CollectiveArrayReport:
    """Observed global/addressable layout for one explicitly partitioned JAX array."""

    name: str
    global_shape: tuple[int, ...]
    dtype: str
    partition_axis_name: str
    partition_count: int
    global_device_count: int
    process_index: int
    process_count: int
    global_logical_bytes: int
    addressable_logical_bytes: int
    addressable_shards: tuple[CollectiveAddressableShard, ...]
    schema_version: str = COLLECTIVE_ARRAY_REPORT_SCHEMA
    EXPECTED_SCHEMA_VERSION: ClassVar[str] = COLLECTIVE_ARRAY_REPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != self.EXPECTED_SCHEMA_VERSION:
            raise ContractError(
                f"collective array report schema must be {self.EXPECTED_SCHEMA_VERSION!r}"
            )
        for name in ("name", "dtype", "partition_axis_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f"collective array report {name.replace('_', ' ')} is required")
        shape = _canonical_shape(self.global_shape, label="collective array global shape")
        for name in (
            "partition_count",
            "global_device_count",
            "process_index",
            "process_count",
            "global_logical_bytes",
            "addressable_logical_bytes",
        ):
            _require_nonnegative_integer(getattr(self, name), label=name.replace("_", " "))
        if self.partition_count == 0 or self.global_device_count == 0 or self.process_count == 0:
            raise ContractError("collective array report counts must be positive")
        if self.process_index >= self.process_count:
            raise ContractError("collective array process index must be below process count")
        if shape[0] != self.partition_count:
            raise ContractError("collective array leading axis must equal its partition count")
        if self.partition_count != self.global_device_count:
            raise ContractError("collective array requires exactly one partition per global device")
        shards = tuple(self.addressable_shards)
        if not shards:
            raise ContractError("collective array report requires addressable shards")
        partitions = tuple(shard.partition_index for shard in shards)
        if len(partitions) != len(set(partitions)):
            raise ContractError("collective array report repeats an addressable partition")
        if any(partition >= self.partition_count for partition in partitions):
            raise ContractError("collective array report has an out-of-range partition")
        if any(shard.process_index != self.process_index for shard in shards):
            raise ContractError("collective array shard process identity disagrees with its report")
        if sum(shard.logical_bytes for shard in shards) != self.addressable_logical_bytes:
            raise ContractError("collective array addressable byte count disagrees with its shards")
        if self.addressable_logical_bytes > self.global_logical_bytes:
            raise ContractError("collective array addressable bytes exceed global logical bytes")
        object.__setattr__(self, "global_shape", shape)
        object.__setattr__(self, "addressable_shards", shards)

    def canonical_data(self) -> dict[str, object]:
        """Return stable JSON-compatible global and process-local sharding evidence."""

        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "global_shape": list(self.global_shape),
            "dtype": self.dtype,
            "partition_axis_name": self.partition_axis_name,
            "partition_count": self.partition_count,
            "global_device_count": self.global_device_count,
            "process_index": self.process_index,
            "process_count": self.process_count,
            "global_logical_bytes": self.global_logical_bytes,
            "addressable_logical_bytes": self.addressable_logical_bytes,
            "replication_intent": "none; one leading FEM partition per device",
            "addressable_shards": [shard.canonical_data() for shard in self.addressable_shards],
        }


@dataclass(frozen=True, slots=True)
class CollectiveReplicatedArrayReport:
    """Observed bounded replication for one explicit global JAX array."""

    name: str
    global_shape: tuple[int, ...]
    dtype: str
    global_device_count: int
    addressable_device_count: int
    process_index: int
    process_count: int
    logical_bytes_per_replica: int
    addressable_logical_bytes: int
    global_replica_logical_bytes: int
    replication_intent: str
    schema_version: str = COLLECTIVE_REPLICATED_ARRAY_REPORT_SCHEMA
    EXPECTED_SCHEMA_VERSION: ClassVar[str] = COLLECTIVE_REPLICATED_ARRAY_REPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != self.EXPECTED_SCHEMA_VERSION:
            raise ContractError(
                "collective replicated-array report schema must be "
                f"{self.EXPECTED_SCHEMA_VERSION!r}"
            )
        for name in ("name", "dtype", "replication_intent"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ContractError(
                    f"collective replicated-array report {name.replace('_', ' ')} is required"
                )
        shape = _canonical_shape(
            self.global_shape,
            label="collective replicated-array global shape",
        )
        for name in (
            "global_device_count",
            "addressable_device_count",
            "process_index",
            "process_count",
            "logical_bytes_per_replica",
            "addressable_logical_bytes",
            "global_replica_logical_bytes",
        ):
            _require_nonnegative_integer(getattr(self, name), label=name.replace("_", " "))
        if (
            self.global_device_count == 0
            or self.addressable_device_count == 0
            or self.process_count == 0
            or self.logical_bytes_per_replica == 0
        ):
            raise ContractError(
                "collective replicated-array report counts and bytes must be positive"
            )
        if self.addressable_device_count > self.global_device_count:
            raise ContractError("replicated addressable device count exceeds global device count")
        if self.process_index >= self.process_count:
            raise ContractError("replicated-array process index must be below process count")
        if self.addressable_logical_bytes != (
            self.addressable_device_count * self.logical_bytes_per_replica
        ):
            raise ContractError("replicated addressable byte accounting is inconsistent")
        if self.global_replica_logical_bytes != (
            self.global_device_count * self.logical_bytes_per_replica
        ):
            raise ContractError("replicated global byte accounting is inconsistent")
        object.__setattr__(self, "global_shape", shape)

    def canonical_data(self) -> dict[str, object]:
        """Return stable JSON-compatible replication evidence."""

        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "global_shape": list(self.global_shape),
            "dtype": self.dtype,
            "partition_spec": [],
            "global_device_count": self.global_device_count,
            "addressable_device_count": self.addressable_device_count,
            "process_index": self.process_index,
            "process_count": self.process_count,
            "logical_bytes_per_replica": self.logical_bytes_per_replica,
            "addressable_logical_bytes": self.addressable_logical_bytes,
            "global_replica_logical_bytes": self.global_replica_logical_bytes,
            "replication_intent": self.replication_intent,
        }


@dataclass(frozen=True, slots=True)
class CollectiveCompilerMemoryReport:
    """Compiler-reported executable memory; it is not a live HBM measurement."""

    generated_code_bytes: int
    argument_bytes: int
    output_bytes: int
    alias_bytes: int
    temporary_bytes: int
    hbm_capacity_bytes_per_device: int | None = None
    schema_version: str = COLLECTIVE_MEMORY_REPORT_SCHEMA
    EXPECTED_SCHEMA_VERSION: ClassVar[str] = COLLECTIVE_MEMORY_REPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != self.EXPECTED_SCHEMA_VERSION:
            raise ContractError(
                f"collective memory report schema must be {self.EXPECTED_SCHEMA_VERSION!r}"
            )
        for name in (
            "generated_code_bytes",
            "argument_bytes",
            "output_bytes",
            "alias_bytes",
            "temporary_bytes",
        ):
            _require_nonnegative_integer(getattr(self, name), label=name.replace("_", " "))
        if self.hbm_capacity_bytes_per_device is not None:
            capacity = _require_nonnegative_integer(
                self.hbm_capacity_bytes_per_device,
                label="HBM capacity bytes per device",
            )
            if capacity == 0:
                raise ContractError("HBM capacity bytes per device must be positive")

    @property
    def compiler_peak_bytes(self) -> int:
        """Return the conservative live-byte estimate exposed by JAX memory analysis."""

        return max(
            0,
            self.argument_bytes + self.output_bytes + self.temporary_bytes - self.alias_bytes,
        )

    @property
    def hbm_fraction(self) -> float | None:
        """Return compiler-estimate/capacity when an observed capacity was supplied."""

        if self.hbm_capacity_bytes_per_device is None:
            return None
        return self.compiler_peak_bytes / self.hbm_capacity_bytes_per_device

    @property
    def risk(self) -> str:
        """Classify the configured HBM fraction using the Phoxla safety bands."""

        fraction = self.hbm_fraction
        if fraction is None:
            return "not_assessed"
        if fraction < 0.70:
            return "safe"
        if fraction < 0.85:
            return "elevated"
        if fraction < 0.95:
            return "high"
        return "extreme"

    def canonical_data(self) -> dict[str, object]:
        """Return stable JSON-compatible compiler memory evidence."""

        return {
            "schema_version": self.schema_version,
            "generated_code_bytes": self.generated_code_bytes,
            "argument_bytes": self.argument_bytes,
            "output_bytes": self.output_bytes,
            "alias_bytes": self.alias_bytes,
            "temporary_bytes": self.temporary_bytes,
            "compiler_peak_bytes": self.compiler_peak_bytes,
            "hbm_capacity_bytes_per_device": self.hbm_capacity_bytes_per_device,
            "hbm_fraction": self.hbm_fraction,
            "risk": self.risk,
            "claim_scope": "compiler estimate; not live HBM usage",
        }


@dataclass(frozen=True, slots=True)
class CollectiveTimingReport:
    """Synchronized lowering, compilation, warmup, and execution timings."""

    lowering_seconds: float
    compilation_seconds: float
    warmup_seconds: float
    execution_seconds: tuple[float, ...]
    schema_version: str = COLLECTIVE_TIMING_REPORT_SCHEMA
    EXPECTED_SCHEMA_VERSION: ClassVar[str] = COLLECTIVE_TIMING_REPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != self.EXPECTED_SCHEMA_VERSION:
            raise ContractError(
                f"collective timing report schema must be {self.EXPECTED_SCHEMA_VERSION!r}"
            )
        for name in ("lowering_seconds", "compilation_seconds", "warmup_seconds"):
            object.__setattr__(
                self,
                name,
                _require_nonnegative_finite(getattr(self, name), label=name.replace("_", " ")),
            )
        samples = tuple(
            _require_nonnegative_finite(value, label="execution time")
            for value in self.execution_seconds
        )
        if not samples:
            raise ContractError("collective timing report requires execution samples")
        object.__setattr__(self, "execution_seconds", samples)

    def canonical_data(self) -> dict[str, object]:
        """Return stable JSON-compatible phase-separated timings."""

        return {
            "schema_version": self.schema_version,
            "lowering_seconds": self.lowering_seconds,
            "compilation_seconds": self.compilation_seconds,
            "warmup_seconds": self.warmup_seconds,
            "execution_seconds": list(self.execution_seconds),
            "execution_min_seconds": min(self.execution_seconds),
            "execution_median_seconds": statistics.median(self.execution_seconds),
            "execution_max_seconds": max(self.execution_seconds),
            "synchronization": "every timed result blocked until ready",
        }


def collective_named_sharding(
    mesh: Mesh,
    rank: int,
    *,
    axis_name: str = "partition",
) -> NamedSharding:
    """Return a leading-axis one-partition-per-device sharding contract."""

    if not isinstance(mesh, Mesh) or mesh.empty:
        raise ContractError("collective runtime requires a nonempty explicit JAX Mesh")
    if not isinstance(axis_name, str) or not axis_name:
        raise ContractError("collective runtime axis name must be nonempty")
    if tuple(mesh.axis_names) != (axis_name,):
        raise ContractError("collective runtime Mesh must have exactly the requested axis")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ContractError("collective runtime array rank must be positive")
    return NamedSharding(
        mesh,
        P(axis_name, *(None for _ in range(rank - 1))),  # type: ignore[no-untyped-call]
    )


def describe_collective_mesh(
    layout: CollectiveLayout,
    mesh: Mesh,
    *,
    axis_name: str = "partition",
    layout_sha256: str | None = None,
) -> CollectiveMeshReport:
    """Record partition/device/process identity without conflating those concepts."""

    validate_collective_mesh(layout, mesh, axis_name)
    addressable = set(mesh.local_devices)
    assignments = tuple(
        CollectiveDeviceAssignment(
            partition_index=partition_index,
            process_index=int(device.process_index),
            device_id=int(device.id),
            platform=str(device.platform),
            device_kind=str(device.device_kind),
            addressable=device in addressable,
        )
        for partition_index, device in enumerate(mesh.devices.reshape(-1))
    )
    return CollectiveMeshReport(
        axis_name=axis_name,
        partition_count=layout.partition_count,
        global_device_count=mesh.size,
        addressable_device_count=sum(record.addressable for record in assignments),
        process_count=len({record.process_index for record in assignments}),
        layout_sha256=layout.digest() if layout_sha256 is None else layout_sha256,
        assignments=assignments,
    )


def _leading_partition(index: tuple[slice, ...], shape: tuple[int, ...]) -> int:
    if len(index) != len(shape) or not all(isinstance(item, slice) for item in index):
        raise ContractError("collective array sharding must use rectangular slices")
    leading = index[0]
    start, stop, step = leading.indices(shape[0])
    if step != 1 or stop - start != 1:
        raise ContractError("collective array shard must contain one leading partition")
    for axis, item in enumerate(index[1:], start=1):
        axis_start, axis_stop, axis_step = item.indices(shape[axis])
        if axis_start != 0 or axis_stop != shape[axis] or axis_step != 1:
            raise ContractError("collective array may partition only its leading axis")
    return start


def describe_collective_array(
    name: str,
    array: jax.Array,
    mesh: Mesh,
    *,
    axis_name: str = "partition",
) -> CollectiveArrayReport:
    """Describe one already-created global array from the current process view."""

    expected = collective_named_sharding(mesh, array.ndim, axis_name=axis_name)
    if not array.sharding.is_equivalent_to(expected, array.ndim):
        raise ContractError("collective runtime array does not use the requested NamedSharding")
    shape = tuple(int(item) for item in array.shape)
    if shape[0] != mesh.size:
        raise ContractError("collective runtime array leading axis must match the Mesh size")
    itemsize = int(array.dtype.itemsize)
    process_index = jax.process_index()
    shards = []
    for shard in array.addressable_shards:
        raw_index = shard.index
        if not isinstance(raw_index, tuple):
            raise ContractError("collective runtime shard index must be a tuple")
        index = tuple(raw_index)
        partition = _leading_partition(index, shape)
        local_shape = tuple(int(item) for item in shard.data.shape)
        shards.append(
            CollectiveAddressableShard(
                partition_index=partition,
                process_index=int(shard.device.process_index),
                device_id=int(shard.device.id),
                device_kind=str(shard.device.device_kind),
                local_shape=local_shape,
                logical_bytes=math.prod(local_shape) * itemsize,
            )
        )
    ordered = tuple(sorted(shards, key=lambda item: item.partition_index))
    return CollectiveArrayReport(
        name=name,
        global_shape=shape,
        dtype=str(array.dtype),
        partition_axis_name=axis_name,
        partition_count=shape[0],
        global_device_count=mesh.size,
        process_index=process_index,
        process_count=jax.process_count(),
        global_logical_bytes=math.prod(shape) * itemsize,
        addressable_logical_bytes=sum(shard.logical_bytes for shard in ordered),
        addressable_shards=ordered,
    )


def make_collective_array_from_process_local_data(
    name: str,
    global_host_array: np.ndarray,
    mesh: Mesh,
    *,
    axis_name: str = "partition",
) -> tuple[jax.Array, CollectiveArrayReport]:
    """Create a global array while materializing only this process's addressable partitions.

    The full host value is acceptable for this bounded validation runner.  Only the slice needed by
    each current-process device is transferred to the accelerator.  The per-shard callback supports
    topology-aware Mesh orders whose process-addressable partitions are not contiguous.
    """

    raw = np.asarray(global_host_array)
    if raw.ndim == 0 or raw.shape[0] != mesh.size:
        raise ContractError("collective host array leading axis must match the Mesh size")
    if raw.dtype.kind not in "biufc":
        raise ContractError(
            "collective host array must use a boolean, integer, floating, or complex dtype"
        )
    sharding = collective_named_sharding(mesh, raw.ndim, axis_name=axis_name)
    mapping = sharding.addressable_devices_indices_map(raw.shape)
    if not mapping:
        raise ContractError("collective sharding has no process-addressable partitions")
    shape = tuple(int(item) for item in raw.shape)
    partitions = []
    for index in mapping.values():
        if index is None:
            raise ContractError("collective sharding produced a non-addressable index")
        partitions.append(_leading_partition(tuple(index), shape))
    partitions.sort()
    if len(partitions) != len(set(partitions)):
        raise ContractError("collective sharding repeats an addressable partition")

    def addressable_data(index: tuple[slice, ...] | None) -> np.ndarray:
        if index is None:  # pragma: no cover - partitioned NamedSharding invariant
            raise ContractError("collective sharding requested a replicated callback value")
        canonical_index = tuple(index)
        _leading_partition(canonical_index, shape)
        return np.ascontiguousarray(raw[canonical_index])

    array = jax.make_array_from_callback(
        raw.shape,
        sharding,
        addressable_data,
    )
    if str(array.dtype) != str(raw.dtype):
        raise ContractError(
            f"collective runtime changed dtype from {raw.dtype} to {array.dtype}; "
            "enable the requested JAX scalar type explicitly"
        )
    report = describe_collective_array(name, array, mesh, axis_name=axis_name)
    if tuple(shard.partition_index for shard in report.addressable_shards) != tuple(partitions):
        raise ContractError("collective runtime created unexpected addressable partitions")
    return array, report


def describe_replicated_array(
    name: str,
    array: jax.Array,
    mesh: Mesh,
    *,
    replication_intent: str,
) -> CollectiveReplicatedArrayReport:
    """Describe an explicitly and fully replicated bounded global array."""

    if not isinstance(replication_intent, str) or not replication_intent.strip():
        raise ContractError("collective replicated array requires an explicit replication intent")
    expected = NamedSharding(mesh, P())  # type: ignore[no-untyped-call]
    if not array.sharding.is_equivalent_to(expected, array.ndim):
        raise ContractError("collective runtime array is not fully replicated")
    shape = tuple(int(item) for item in array.shape)
    _canonical_shape(shape, label="collective replicated-array global shape")
    itemsize = int(array.dtype.itemsize)
    logical_bytes = math.prod(shape) * itemsize
    addressable_shards = tuple(array.addressable_shards)
    if len(addressable_shards) != len(mesh.local_devices):
        raise ContractError("collective replicated array does not cover every addressable device")
    if any(tuple(int(item) for item in shard.data.shape) != shape for shard in addressable_shards):
        raise ContractError("collective replicated array has a partial addressable replica")
    return CollectiveReplicatedArrayReport(
        name=name,
        global_shape=shape,
        dtype=str(array.dtype),
        global_device_count=mesh.size,
        addressable_device_count=len(addressable_shards),
        process_index=jax.process_index(),
        process_count=jax.process_count(),
        logical_bytes_per_replica=logical_bytes,
        addressable_logical_bytes=logical_bytes * len(addressable_shards),
        global_replica_logical_bytes=logical_bytes * mesh.size,
        replication_intent=replication_intent,
    )


def make_replicated_array_from_process_local_data(
    name: str,
    host_array: np.ndarray,
    mesh: Mesh,
    *,
    replication_intent: str,
) -> tuple[jax.Array, CollectiveReplicatedArrayReport]:
    """Create one bounded global array replicated on every Mesh device."""

    raw = np.asarray(host_array)
    if raw.ndim == 0 or any(item == 0 for item in raw.shape):
        raise ContractError("collective replicated host array must be nonempty and nonscalar")
    if raw.dtype.kind not in "biufc":
        raise ContractError(
            "collective replicated host array must use a boolean, integer, floating, or complex dtype"
        )
    sharding = NamedSharding(mesh, P())  # type: ignore[no-untyped-call]
    array = jax.make_array_from_process_local_data(  # type: ignore[no-untyped-call]
        sharding,
        np.ascontiguousarray(raw),
        global_shape=raw.shape,
    )
    if str(array.dtype) != str(raw.dtype):
        raise ContractError(
            f"collective runtime changed dtype from {raw.dtype} to {array.dtype}; "
            "enable the requested JAX scalar type explicitly"
        )
    report = describe_replicated_array(
        name,
        array,
        mesh,
        replication_intent=replication_intent,
    )
    return array, report
