"""Shard-preserving P1 temperature transfer to an FDTDX material parameter.

The canonical :class:`TriangleP1SamplingPlan` remains a host-owned numerical operator.  This
module lowers that operator into explicit source- and destination-partition buffers.  A JAX
runtime first samples each target point on the partition that owns its source triangle, then uses
one ``all_to_all`` to route the scalar samples to the FDTDX x shard that owns the target voxel.
No canonical FEM vector or FDTD material field is gathered to one process.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple, cast

from femx.core.arrays import ArrayLike, shape_of
from femx.core.errors import ContractError
from femx.interop.fdtdx.thermo_optic import (
    FDTDXDeviceParameterContract,
    ThermoOpticLaw,
    ThermoOpticParameterState,
    TriangleP1SamplingPlan,
)

DISTRIBUTED_THERMO_OPTIC_SCHEMA = "femx.fdtdx.distributed_thermo_optic/v1"


def _require_sha256(value: str, *, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ContractError(f"{label} must be a canonical lowercase SHA-256")


def _require_positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{label} must be a positive integer")
    return value


def _readonly(values: Any, *, dtype: Any) -> Any:
    import numpy as np

    result = np.array(values, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _hash_array(hasher: Any, label: str, values: Any, *, dtype: Any) -> None:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    hasher.update(label.encode("ascii"))
    hasher.update(np.asarray(array.shape, dtype="<i8").tobytes())
    hasher.update(array.tobytes())


def _operator_digest(
    *,
    source_cell_ids: object,
    send_source_cell_slots: object,
    send_barycentric_weights: object,
    send_active: object,
    receive_target_local_indices: object,
    receive_active: object,
    source_mesh_sha256: str,
    source_layout_sha256: str,
    sampling_operator_sha256: str,
    target_coordinate_sha256: str,
    plane_axes: tuple[int, int],
    target_shape: tuple[int, int, int],
    target_shard_shape: tuple[int, int, int],
    partition_count: int,
    source_cell_count: int,
    source_cell_capacity: int,
    transfer_capacity: int,
    mesh_axis_name: str,
) -> str:
    metadata = {
        "schema_version": DISTRIBUTED_THERMO_OPTIC_SCHEMA,
        "source_mesh_sha256": source_mesh_sha256,
        "source_layout_sha256": source_layout_sha256,
        "sampling_operator_sha256": sampling_operator_sha256,
        "target_coordinate_sha256": target_coordinate_sha256,
        "plane_axes": plane_axes,
        "target_shape": target_shape,
        "target_shard_shape": target_shard_shape,
        "partition_count": partition_count,
        "source_cell_count": source_cell_count,
        "source_cell_capacity": source_cell_capacity,
        "transfer_capacity": transfer_capacity,
        "mesh_axis_name": mesh_axis_name,
    }
    hasher = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for label, values, dtype in (
        ("source_cell_ids", source_cell_ids, "<i8"),
        ("send_source_cell_slots", send_source_cell_slots, "<i8"),
        ("send_barycentric_weights", send_barycentric_weights, "<f8"),
        ("send_active", send_active, "?"),
        ("receive_target_local_indices", receive_target_local_indices, "<i8"),
        ("receive_active", receive_active, "?"),
    ):
        _hash_array(hasher, label, values, dtype=dtype)
    return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class DistributedTriangleP1SamplingPlan:
    """Host-owned routing plan from FEM cell shards to FDTDX x shards."""

    source_cell_ids: ArrayLike
    send_source_cell_slots: ArrayLike
    send_barycentric_weights: ArrayLike
    send_active: ArrayLike
    receive_target_local_indices: ArrayLike
    receive_active: ArrayLike
    source_mesh_sha256: str
    source_layout_sha256: str
    sampling_operator_sha256: str
    target_coordinate_sha256: str
    operator_sha256: str
    plane_axes: tuple[int, int]
    target_shape: tuple[int, int, int]
    target_shard_shape: tuple[int, int, int]
    partition_count: int
    source_cell_count: int
    source_cell_capacity: int
    transfer_capacity: int
    maximum_partition_error: float
    minimum_barycentric_weight: float
    mesh_axis_name: str = "shard"
    target_sharding_axis: int = 0
    schema_version: str = DISTRIBUTED_THERMO_OPTIC_SCHEMA

    def __post_init__(self) -> None:
        import numpy as np

        if self.schema_version != DISTRIBUTED_THERMO_OPTIC_SCHEMA:
            raise ContractError(
                f"distributed thermo-optic schema must be {DISTRIBUTED_THERMO_OPTIC_SCHEMA!r}"
            )
        partition_count = _require_positive_integer(
            self.partition_count,
            label="distributed thermo-optic partition count",
        )
        source_cell_count = _require_positive_integer(
            self.source_cell_count,
            label="distributed thermo-optic source cell count",
        )
        source_cell_capacity = _require_positive_integer(
            self.source_cell_capacity,
            label="distributed thermo-optic source cell capacity",
        )
        transfer_capacity = _require_positive_integer(
            self.transfer_capacity,
            label="distributed thermo-optic transfer capacity",
        )
        if not self.mesh_axis_name or self.mesh_axis_name.strip() != self.mesh_axis_name:
            raise ContractError(
                "distributed thermo-optic mesh axis name must be non-empty and trimmed"
            )
        if self.target_sharding_axis != 0:
            raise ContractError("distributed thermo-optic v1 must shard the FDTDX x axis")
        if len(self.target_shape) != 3 or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in self.target_shape
        ):
            raise ContractError(
                "distributed thermo-optic target shape must contain three positive integers"
            )
        if self.target_shape[0] % partition_count != 0:
            raise ContractError(
                "distributed thermo-optic target x extent must divide over partitions"
            )
        expected_target_shard_shape = (
            self.target_shape[0] // partition_count,
            self.target_shape[1],
            self.target_shape[2],
        )
        if self.target_shard_shape != expected_target_shard_shape:
            raise ContractError("distributed thermo-optic target shard shape is inconsistent")
        if (
            len(self.plane_axes) != 2
            or len(set(self.plane_axes)) != 2
            or any(axis not in (0, 1, 2) for axis in self.plane_axes)
        ):
            raise ContractError("distributed thermo-optic plane axes must be distinct x/y/z axes")
        for numeric_value, label in (
            (self.maximum_partition_error, "maximum partition error"),
            (self.minimum_barycentric_weight, "minimum barycentric weight"),
        ):
            if not math.isfinite(numeric_value):
                raise ContractError(f"distributed thermo-optic {label} must be finite")
        if self.maximum_partition_error < 0.0:
            raise ContractError("distributed thermo-optic partition error cannot be negative")
        for digest, label in (
            (self.source_mesh_sha256, "source mesh digest"),
            (self.source_layout_sha256, "source layout digest"),
            (self.sampling_operator_sha256, "sampling operator digest"),
            (self.target_coordinate_sha256, "target coordinate digest"),
            (self.operator_sha256, "operator digest"),
        ):
            _require_sha256(digest, label=f"distributed thermo-optic {label}")

        pair_shape = (partition_count, partition_count, transfer_capacity)
        source_cell_ids = np.asarray(self.source_cell_ids)
        source_slots = np.asarray(self.send_source_cell_slots)
        weights = np.asarray(self.send_barycentric_weights)
        send_active = np.asarray(self.send_active)
        receive_indices = np.asarray(self.receive_target_local_indices)
        receive_active = np.asarray(self.receive_active)
        if source_cell_ids.shape != (partition_count, source_cell_capacity):
            raise ContractError("distributed thermo-optic source-cell table shape is inconsistent")
        if source_slots.shape != pair_shape or receive_indices.shape != pair_shape:
            raise ContractError("distributed thermo-optic routing-index shape is inconsistent")
        if weights.shape != (*pair_shape, 3):
            raise ContractError("distributed thermo-optic barycentric-weight shape is inconsistent")
        if send_active.shape != pair_shape or receive_active.shape != pair_shape:
            raise ContractError("distributed thermo-optic routing-mask shape is inconsistent")
        if source_cell_ids.dtype.kind not in "iu" or source_slots.dtype.kind not in "iu":
            raise ContractError("distributed thermo-optic source indices must be integers")
        if receive_indices.dtype.kind not in "iu":
            raise ContractError("distributed thermo-optic destination indices must be integers")
        if weights.dtype.kind != "f" or not np.all(np.isfinite(weights)):
            raise ContractError("distributed thermo-optic weights must be finite real values")
        if send_active.dtype.kind != "b" or receive_active.dtype.kind != "b":
            raise ContractError("distributed thermo-optic routing masks must be boolean")
        if not np.array_equal(receive_active, np.transpose(send_active, (1, 0, 2))):
            raise ContractError("distributed thermo-optic send and receive masks disagree")

        active_cell_ids = source_cell_ids[source_cell_ids < source_cell_count]
        if np.any(source_cell_ids < 0) or np.any(source_cell_ids > source_cell_count):
            raise ContractError("distributed thermo-optic source-cell table contains an invalid id")
        if not np.array_equal(np.sort(active_cell_ids), np.arange(source_cell_count)):
            raise ContractError(
                "distributed thermo-optic source-cell table must cover every cell once"
            )
        if np.any(source_slots[send_active] >= source_cell_capacity) or np.any(
            source_slots[send_active] < 0
        ):
            raise ContractError("distributed thermo-optic active source slot is invalid")
        if np.any(source_slots[~send_active] != source_cell_capacity):
            raise ContractError(
                "distributed thermo-optic inactive source slots must use the sentinel"
            )
        if np.any(weights[~send_active] != 0.0):
            raise ContractError("distributed thermo-optic inactive weights must be exact zero")

        active_weight_sums = np.sum(weights, axis=-1)[send_active]
        if active_weight_sums.size != math.prod(self.target_shape):
            raise ContractError("distributed thermo-optic routes must cover every target cell once")
        actual_partition_error = float(np.max(np.abs(active_weight_sums - 1.0)))
        if not math.isclose(
            actual_partition_error,
            self.maximum_partition_error,
            rel_tol=0.0,
            abs_tol=16.0 * np.finfo(np.float64).eps,
        ):
            raise ContractError("distributed thermo-optic weights violate partition of unity")
        actual_minimum_weight = float(np.min(weights[send_active]))
        if not math.isclose(
            actual_minimum_weight,
            self.minimum_barycentric_weight,
            rel_tol=0.0,
            abs_tol=16.0 * np.finfo(np.float64).eps,
        ):
            raise ContractError("distributed thermo-optic minimum weight metadata is inconsistent")

        local_target_count = math.prod(self.target_shard_shape)
        if np.any(receive_indices[receive_active] < 0) or np.any(
            receive_indices[receive_active] >= local_target_count
        ):
            raise ContractError("distributed thermo-optic active destination index is invalid")
        if np.any(receive_indices[~receive_active] != local_target_count):
            raise ContractError(
                "distributed thermo-optic inactive destinations must use the sentinel"
            )
        for destination in range(partition_count):
            indices = receive_indices[destination][receive_active[destination]]
            if not np.array_equal(np.sort(indices), np.arange(local_target_count)):
                raise ContractError(
                    "distributed thermo-optic destination shard must cover every local cell once"
                )

        expected_operator_sha256 = _operator_digest(
            source_cell_ids=source_cell_ids,
            send_source_cell_slots=source_slots,
            send_barycentric_weights=weights,
            send_active=send_active,
            receive_target_local_indices=receive_indices,
            receive_active=receive_active,
            source_mesh_sha256=self.source_mesh_sha256,
            source_layout_sha256=self.source_layout_sha256,
            sampling_operator_sha256=self.sampling_operator_sha256,
            target_coordinate_sha256=self.target_coordinate_sha256,
            plane_axes=self.plane_axes,
            target_shape=self.target_shape,
            target_shard_shape=self.target_shard_shape,
            partition_count=partition_count,
            source_cell_count=source_cell_count,
            source_cell_capacity=source_cell_capacity,
            transfer_capacity=transfer_capacity,
            mesh_axis_name=self.mesh_axis_name,
        )
        if self.operator_sha256 != expected_operator_sha256:
            raise ContractError("distributed thermo-optic operator digest is inconsistent")

        object.__setattr__(self, "source_cell_ids", _readonly(source_cell_ids, dtype=np.int64))
        object.__setattr__(
            self,
            "send_source_cell_slots",
            _readonly(source_slots, dtype=np.int64),
        )
        object.__setattr__(
            self,
            "send_barycentric_weights",
            _readonly(weights, dtype=np.float64),
        )
        object.__setattr__(self, "send_active", _readonly(send_active, dtype=np.bool_))
        object.__setattr__(
            self,
            "receive_target_local_indices",
            _readonly(receive_indices, dtype=np.int64),
        )
        object.__setattr__(
            self,
            "receive_active",
            _readonly(receive_active, dtype=np.bool_),
        )

    def canonical_data(self) -> Mapping[str, object]:
        """Return deterministic transfer metadata without runtime device identity."""

        return {
            "schema_version": self.schema_version,
            "source_mesh_sha256": self.source_mesh_sha256,
            "source_layout_sha256": self.source_layout_sha256,
            "sampling_operator_sha256": self.sampling_operator_sha256,
            "target_coordinate_sha256": self.target_coordinate_sha256,
            "operator_sha256": self.operator_sha256,
            "plane_axes": list(self.plane_axes),
            "target_shape_xyz": list(self.target_shape),
            "target_shard_shape_xyz": list(self.target_shard_shape),
            "target_sharding_axis": self.target_sharding_axis,
            "mesh_axis_name": self.mesh_axis_name,
            "partition_count": self.partition_count,
            "source_cell_count": self.source_cell_count,
            "source_cell_capacity": self.source_cell_capacity,
            "transfer_capacity": self.transfer_capacity,
            "actual_transfer_count": math.prod(self.target_shape),
            "allocated_transfer_slots": (
                self.partition_count * self.partition_count * self.transfer_capacity
            ),
            "maximum_partition_error": self.maximum_partition_error,
            "minimum_barycentric_weight": self.minimum_barycentric_weight,
            "routing_collective": "all_to_all",
            "global_gather": False,
        }


def prepare_distributed_triangle_p1_sampling_plan(
    sampling_plan: TriangleP1SamplingPlan,
    source_cell_ids: object,
    *,
    source_layout_sha256: str,
    mesh_axis_name: str = "shard",
) -> DistributedTriangleP1SamplingPlan:
    """Lower a canonical P1 sampler into explicit source/destination routing buffers."""

    import numpy as np

    if not isinstance(sampling_plan, TriangleP1SamplingPlan):
        raise ContractError("distributed thermo-optic lowering requires a P1 sampling plan")
    _require_sha256(
        source_layout_sha256,
        label="distributed thermo-optic source layout digest",
    )
    ids = np.asarray(source_cell_ids)
    if ids.ndim != 2 or ids.shape[0] == 0 or ids.shape[1] == 0 or ids.dtype.kind not in "iu":
        raise ContractError(
            "distributed thermo-optic source-cell table must be a rank-two integer array"
        )
    ids = np.asarray(ids, dtype=np.int64)
    partition_count, source_cell_capacity = ids.shape
    source_cell_count = int(sampling_plan.source_cells.shape[0])
    if np.any(ids < 0) or np.any(ids > source_cell_count):
        raise ContractError("distributed thermo-optic source-cell table contains an invalid id")
    cell_to_partition = np.full((source_cell_count,), -1, dtype=np.int64)
    cell_to_slot = np.full((source_cell_count,), -1, dtype=np.int64)
    for partition in range(partition_count):
        for slot, cell_id in enumerate(ids[partition]):
            if cell_id == source_cell_count:
                continue
            if cell_to_partition[cell_id] >= 0:
                raise ContractError("distributed thermo-optic source cell has multiple owners")
            cell_to_partition[cell_id] = partition
            cell_to_slot[cell_id] = slot
    if np.any(cell_to_partition < 0):
        raise ContractError("distributed thermo-optic source-cell table omits a source cell")

    target_shape = sampling_plan.target_shape
    if target_shape[0] % partition_count != 0:
        raise ContractError("distributed thermo-optic target x extent must divide over partitions")
    target_shard_shape = (
        target_shape[0] // partition_count,
        target_shape[1],
        target_shape[2],
    )
    groups: list[list[list[tuple[int, int, np.ndarray]]]] = [
        [[] for _ in range(partition_count)] for _ in range(partition_count)
    ]
    target_cells = np.asarray(sampling_plan.target_cell_indices, dtype=np.int64)
    weights = np.asarray(sampling_plan.barycentric_weights, dtype=np.float64)
    for target_index in np.ndindex(target_shape):
        cell_id = int(target_cells[target_index])
        source_partition = int(cell_to_partition[cell_id])
        destination_partition = target_index[0] // target_shard_shape[0]
        local_index = (
            (target_index[0] % target_shard_shape[0]) * target_shape[1] * target_shape[2]
            + target_index[1] * target_shape[2]
            + target_index[2]
        )
        groups[source_partition][destination_partition].append(
            (int(cell_to_slot[cell_id]), int(local_index), weights[target_index])
        )
    transfer_capacity = max(len(group) for source in groups for group in source)
    pair_shape = (partition_count, partition_count, transfer_capacity)
    source_slots = np.full(pair_shape, source_cell_capacity, dtype=np.int64)
    send_weights = np.zeros((*pair_shape, 3), dtype=np.float64)
    send_active = np.zeros(pair_shape, dtype=np.bool_)
    receive_indices = np.full(
        pair_shape,
        math.prod(target_shard_shape),
        dtype=np.int64,
    )
    receive_active = np.zeros(pair_shape, dtype=np.bool_)
    for source_partition, destination_groups in enumerate(groups):
        for destination_partition, group in enumerate(destination_groups):
            for slot, (source_slot, target_local_index, barycentric) in enumerate(group):
                source_slots[source_partition, destination_partition, slot] = source_slot
                send_weights[source_partition, destination_partition, slot] = barycentric
                send_active[source_partition, destination_partition, slot] = True
                receive_indices[destination_partition, source_partition, slot] = target_local_index
                receive_active[destination_partition, source_partition, slot] = True

    operator_sha256 = _operator_digest(
        source_cell_ids=ids,
        send_source_cell_slots=source_slots,
        send_barycentric_weights=send_weights,
        send_active=send_active,
        receive_target_local_indices=receive_indices,
        receive_active=receive_active,
        source_mesh_sha256=sampling_plan.source_mesh_sha256,
        source_layout_sha256=source_layout_sha256,
        sampling_operator_sha256=sampling_plan.operator_sha256,
        target_coordinate_sha256=sampling_plan.target_coordinate_sha256,
        plane_axes=sampling_plan.plane_axes,
        target_shape=target_shape,
        target_shard_shape=target_shard_shape,
        partition_count=partition_count,
        source_cell_count=source_cell_count,
        source_cell_capacity=source_cell_capacity,
        transfer_capacity=transfer_capacity,
        mesh_axis_name=mesh_axis_name,
    )
    return DistributedTriangleP1SamplingPlan(
        source_cell_ids=ids,
        send_source_cell_slots=source_slots,
        send_barycentric_weights=send_weights,
        send_active=send_active,
        receive_target_local_indices=receive_indices,
        receive_active=receive_active,
        source_mesh_sha256=sampling_plan.source_mesh_sha256,
        source_layout_sha256=source_layout_sha256,
        sampling_operator_sha256=sampling_plan.operator_sha256,
        target_coordinate_sha256=sampling_plan.target_coordinate_sha256,
        operator_sha256=operator_sha256,
        plane_axes=sampling_plan.plane_axes,
        target_shape=target_shape,
        target_shard_shape=target_shard_shape,
        partition_count=partition_count,
        source_cell_count=source_cell_count,
        source_cell_capacity=source_cell_capacity,
        transfer_capacity=transfer_capacity,
        maximum_partition_error=sampling_plan.maximum_partition_error,
        minimum_barycentric_weight=sampling_plan.minimum_barycentric_weight,
        mesh_axis_name=mesh_axis_name,
    )


class HostPackedDistributedThermoOpticInputs(NamedTuple):
    """Host arrays before a caller chooses devices and concrete shardings."""

    send_source_cell_slots: ArrayLike
    send_barycentric_weights: ArrayLike
    send_active: ArrayLike
    receive_target_local_indices: ArrayLike
    receive_active: ArrayLike


class PackedDistributedThermoOpticInputs(NamedTuple):
    """Explicit JAX inputs with a source/destination partition leading axis."""

    send_source_cell_slots: ArrayLike
    send_barycentric_weights: ArrayLike
    send_active: ArrayLike
    receive_target_local_indices: ArrayLike
    receive_active: ArrayLike


def pack_distributed_thermo_optic_inputs_host(
    plan: DistributedTriangleP1SamplingPlan,
    *,
    value_dtype: Any,
) -> HostPackedDistributedThermoOpticInputs:
    """Cast the immutable routing plan while retaining all arrays as explicit inputs."""

    import numpy as np

    if not isinstance(plan, DistributedTriangleP1SamplingPlan):
        raise ContractError("distributed thermo-optic packing requires a prepared plan")
    dtype = np.dtype(value_dtype)
    if dtype.kind != "f" or dtype.itemsize not in (4, 8):
        raise ContractError("distributed thermo-optic values require float32 or float64")
    if max(plan.source_cell_capacity, math.prod(plan.target_shard_shape)) > np.iinfo(np.int32).max:
        raise ContractError("distributed thermo-optic transport exceeds int32 addressability")
    return HostPackedDistributedThermoOpticInputs(
        send_source_cell_slots=_readonly(plan.send_source_cell_slots, dtype=np.int32),
        send_barycentric_weights=_readonly(plan.send_barycentric_weights, dtype=dtype),
        send_active=_readonly(plan.send_active, dtype=np.bool_),
        receive_target_local_indices=_readonly(
            plan.receive_target_local_indices,
            dtype=np.int32,
        ),
        receive_active=_readonly(plan.receive_active, dtype=np.bool_),
    )


def pack_distributed_thermo_optic_inputs(
    plan: DistributedTriangleP1SamplingPlan,
    *,
    value_dtype: Any,
) -> PackedDistributedThermoOpticInputs:
    """Create ordinary JAX arrays without selecting devices or initializing a backend."""

    import jax.numpy as jnp

    host = pack_distributed_thermo_optic_inputs_host(plan, value_dtype=value_dtype)
    return PackedDistributedThermoOpticInputs(*(jnp.asarray(value) for value in host))


@dataclass(frozen=True, slots=True)
class DistributedThermoOpticRuntime:
    """One explicit differentiable transfer bound to a caller-supplied JAX Mesh."""

    state: Callable[..., ThermoOpticParameterState]


def build_distributed_thermo_optic_runtime(
    plan: DistributedTriangleP1SamplingPlan,
    mesh: object,
    law: ThermoOpticLaw,
    contract: FDTDXDeviceParameterContract,
) -> DistributedThermoOpticRuntime:
    """Build the all-to-all sampler without capturing any large numerical input."""

    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    if not isinstance(plan, DistributedTriangleP1SamplingPlan):
        raise ContractError("distributed thermo-optic runtime requires a prepared plan")
    if not isinstance(mesh, Mesh):
        raise ContractError("distributed thermo-optic runtime requires an explicit JAX Mesh")
    if mesh.empty or tuple(mesh.axis_names) != (plan.mesh_axis_name,):
        raise ContractError(
            "distributed thermo-optic Mesh must use the declared one-dimensional axis"
        )
    if int(mesh.shape[plan.mesh_axis_name]) != plan.partition_count:
        raise ContractError("distributed thermo-optic Mesh must contain one device per partition")
    if not isinstance(law, ThermoOpticLaw):
        raise ContractError("distributed thermo-optic runtime requires a physical law")
    if not isinstance(contract, FDTDXDeviceParameterContract):
        raise ContractError("distributed thermo-optic runtime requires an FDTDX device contract")
    if contract.target_shape != plan.target_shape or contract.plane_axes != plan.plane_axes:
        raise ContractError("distributed thermo-optic device geometry differs from the plan")
    if contract.thermo_optic_law_sha256 != law.sha256:
        raise ContractError("distributed thermo-optic physical law differs from the contract")
    if contract.target_coordinate_sha256 != plan.target_coordinate_sha256:
        raise ContractError("distributed thermo-optic target coordinates differ from the contract")
    if contract.transfer_operator_sha256 != plan.sampling_operator_sha256:
        raise ContractError("distributed thermo-optic sampling operator differs from the contract")

    pair_shape = (plan.partition_count, plan.partition_count, plan.transfer_capacity)
    cell_shape = (plan.partition_count, plan.source_cell_capacity, 3)
    source_slots_shape = pair_shape
    weights_shape = (*pair_shape, 3)
    local_target_count = math.prod(plan.target_shard_shape)
    axis_name = plan.mesh_axis_name
    source_spec = P(axis_name, None, None)  # type: ignore[no-untyped-call]
    pair_spec = P(axis_name, None, None)  # type: ignore[no-untyped-call]
    weights_spec = P(axis_name, None, None, None)  # type: ignore[no-untyped-call]
    target_spec = P(axis_name, None, None)  # type: ignore[no-untyped-call]
    replicated = P()  # type: ignore[no-untyped-call]

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(source_spec, pair_spec, weights_spec, pair_spec, pair_spec, pair_spec),
        out_specs=(
            target_spec,
            target_spec,
            target_spec,
            target_spec,
            target_spec,
            replicated,
        ),
        check_vma=True,
    )
    def mapped(
        cell_nodal_temperature: Any,
        send_source_cell_slots: Any,
        send_barycentric_weights: Any,
        send_active: Any,
        receive_target_local_indices: Any,
        receive_active: Any,
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        local_temperature = cell_nodal_temperature[0]
        extended = jnp.concatenate(
            (local_temperature, jnp.zeros((1, 3), dtype=local_temperature.dtype)),
            axis=0,
        )
        source_values = extended[send_source_cell_slots[0]]
        outgoing = jnp.sum(send_barycentric_weights[0] * source_values, axis=-1)
        outgoing = jnp.where(send_active[0], outgoing, 0.0)
        received = lax.all_to_all(  # type: ignore[no-untyped-call]
            outgoing,
            axis_name,
            split_axis=0,
            concat_axis=0,
            tiled=False,
        )
        received = jnp.where(receive_active[0], received, 0.0)
        flat_temperature = (
            jnp.zeros((local_target_count + 1,), dtype=local_temperature.dtype)
            .at[receive_target_local_indices[0].reshape(-1)]
            .add(received.reshape(-1))
        )[:local_target_count]
        temperature = flat_temperature.reshape(plan.target_shard_shape)
        reference_index = jnp.asarray(law.reference_refractive_index, dtype=temperature.dtype)
        thermo_coefficient = jnp.asarray(
            law.thermo_optic_coefficient_per_k,
            dtype=temperature.dtype,
        )
        reference_temperature = jnp.asarray(
            law.reference_temperature_k,
            dtype=temperature.dtype,
        )
        lower = jnp.asarray(contract.lower_relative_permittivity, dtype=temperature.dtype)
        upper = jnp.asarray(contract.upper_relative_permittivity, dtype=temperature.dtype)
        refractive_index = reference_index + thermo_coefficient * (
            temperature - reference_temperature
        )
        relative_permittivity = refractive_index**2
        parameter = (relative_permittivity - lower) / (upper - lower)
        valid = (
            jnp.isfinite(temperature)
            & jnp.isfinite(refractive_index)
            & jnp.isfinite(relative_permittivity)
            & (refractive_index > 0.0)
            & (parameter >= 0.0)
            & (parameter <= 1.0)
        )
        parameter = jnp.where(valid, parameter, jnp.nan)
        all_valid_count = lax.psum(  # type: ignore[no-untyped-call]
            jnp.asarray(jnp.all(valid), dtype=jnp.int32),
            axis_name,
        )
        all_valid = all_valid_count == plan.partition_count
        return (
            temperature,
            refractive_index,
            relative_permittivity,
            parameter,
            valid,
            all_valid,
        )

    def state(
        inputs: PackedDistributedThermoOpticInputs,
        cell_nodal_temperature: ArrayLike,
    ) -> ThermoOpticParameterState:
        if not isinstance(inputs, PackedDistributedThermoOpticInputs):
            raise ContractError("distributed thermo-optic inputs must use the packed contract")
        expected_shapes = (
            (inputs.send_source_cell_slots, source_slots_shape),
            (inputs.send_barycentric_weights, weights_shape),
            (inputs.send_active, pair_shape),
            (inputs.receive_target_local_indices, pair_shape),
            (inputs.receive_active, pair_shape),
        )
        if any(shape_of(value) != expected for value, expected in expected_shapes):
            raise ValueError("distributed thermo-optic inputs disagree with the plan")
        if shape_of(cell_nodal_temperature) != cell_shape:
            raise ValueError("distributed thermo-optic cell temperature disagrees with the plan")
        if getattr(inputs.send_source_cell_slots.dtype, "kind", None) not in (None, "i", "u"):
            raise TypeError("distributed thermo-optic source slots must use an integer dtype")
        if getattr(inputs.receive_target_local_indices.dtype, "kind", None) not in (
            None,
            "i",
            "u",
        ):
            raise TypeError("distributed thermo-optic target indices must use an integer dtype")
        if inputs.send_active.dtype != jnp.bool_ or inputs.receive_active.dtype != jnp.bool_:
            raise TypeError("distributed thermo-optic activity masks must be boolean")
        temperature_array = cast(Any, cell_nodal_temperature)
        if not jnp.issubdtype(temperature_array.dtype, jnp.floating):
            raise TypeError("distributed thermo-optic temperatures must use a real floating dtype")
        if inputs.send_barycentric_weights.dtype != temperature_array.dtype:
            raise TypeError(
                "distributed thermo-optic weights and temperatures must share one dtype"
            )
        if str(temperature_array.dtype) != contract.parameter_dtype:
            raise TypeError("distributed thermo-optic temperature dtype differs from the contract")
        values = mapped(
            temperature_array,
            inputs.send_source_cell_slots,
            inputs.send_barycentric_weights,
            inputs.send_active,
            inputs.receive_target_local_indices,
            inputs.receive_active,
        )
        return ThermoOpticParameterState(*cast(tuple[Any, Any, Any, Any, Any, Any], values))

    return DistributedThermoOpticRuntime(state=state)


__all__ = [
    "DISTRIBUTED_THERMO_OPTIC_SCHEMA",
    "DistributedThermoOpticRuntime",
    "DistributedTriangleP1SamplingPlan",
    "HostPackedDistributedThermoOpticInputs",
    "PackedDistributedThermoOpticInputs",
    "build_distributed_thermo_optic_runtime",
    "pack_distributed_thermo_optic_inputs",
    "pack_distributed_thermo_optic_inputs_host",
    "prepare_distributed_triangle_p1_sampling_plan",
]
