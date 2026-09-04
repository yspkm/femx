r"""Fixed-capacity JAX collective lowering for the owned/ghost port operator.

The element-width-independent transport and pairwise collective algebra live in
:mod:`femx.backends.jax.collective`.  This module preserves the port-specific schema, public API,
mesh report, and historical layout digest while requiring six local mixed DOFs.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from femx.core.errors import ContractError

from .collective import (
    CollectiveHaloLink,
    CollectiveLayout,
    CollectiveStorageReport,
    build_packed_collective_matvec,
    build_validation_collective_matvec,
    pack_collective_cell_matrix,
    pack_collective_owned_vector,
    prepare_collective_layout,
    unpack_collective_owned_vector,
    validate_collective_mesh,
)
from .port_owned_ghost import PortOwnedGhostTopology

PORT_COLLECTIVE_LAYOUT_SCHEMA = "femx.jax.port_collective/v1"
PORT_COLLECTIVE_MESH_REPORT_SCHEMA = "femx.jax.port_collective.mesh_report/v1"

PortCollectiveHaloLink = CollectiveHaloLink
PortCollectiveStorageReport = CollectiveStorageReport
PortCollectiveLayout = CollectiveLayout


def _require_nonnegative_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{label} must be a nonnegative integer")
    return value


@dataclass(frozen=True, slots=True)
class PortCollectiveDeviceAssignment:
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
        """Return JSON-safe device identity without conflating it with a worker index."""

        return {
            "partition_index": self.partition_index,
            "process_index": self.process_index,
            "device_id": self.device_id,
            "platform": self.platform,
            "device_kind": self.device_kind,
            "addressable": self.addressable,
        }


@dataclass(frozen=True, slots=True)
class PortCollectiveMeshReport:
    """Serializable observed mapping for one already-constructed explicit JAX Mesh."""

    axis_name: str
    partition_count: int
    global_device_count: int
    addressable_device_count: int
    process_count: int
    layout_sha256: str
    assignments: tuple[PortCollectiveDeviceAssignment, ...]
    schema_version: str = PORT_COLLECTIVE_MESH_REPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PORT_COLLECTIVE_MESH_REPORT_SCHEMA:
            raise ContractError(
                f"collective Mesh report schema must be {PORT_COLLECTIVE_MESH_REPORT_SCHEMA!r}"
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
        """Return whether the explicit global Mesh spans more than one JAX process."""

        return self.process_count > 1

    def canonical_data(self) -> dict[str, object]:
        """Return a JSON-safe observation for a run manifest or evidence artifact."""

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


def prepare_collective_port_layout(topology: PortOwnedGhostTopology) -> PortCollectiveLayout:
    """Lower an immutable six-DOF port topology without initializing JAX devices."""

    if not isinstance(topology, PortOwnedGhostTopology):
        raise ContractError("collective lowering requires a PortOwnedGhostTopology")
    return prepare_collective_layout(
        topology,
        schema_version=PORT_COLLECTIVE_LAYOUT_SCHEMA,
    )


def pack_collective_port_cell_matrix(
    layout: PortCollectiveLayout,
    cell_matrix: jax.Array,
) -> jax.Array:
    """Pack canonical six-by-six port blocks and zero inactive cells."""

    return pack_collective_cell_matrix(layout, cell_matrix)


def pack_collective_port_owned_vector(
    layout: PortCollectiveLayout,
    vector: jax.Array,
) -> jax.Array:
    """Pack owner-authoritative port values and zero inactive owner slots."""

    return pack_collective_owned_vector(layout, vector)


def unpack_collective_port_owned_vector(
    layout: PortCollectiveLayout,
    packed_owned_vector: jax.Array,
) -> jax.Array:
    """Reconstruct canonical port order and discard inactive owner slots."""

    return unpack_collective_owned_vector(layout, packed_owned_vector)


PortPackedCollectiveMatvec = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]
PortCanonicalCollectiveMatvec = Callable[[jax.Array, jax.Array], jax.Array]


def _validate_collective_mesh(layout: PortCollectiveLayout, mesh: Mesh, axis_name: str) -> None:
    validate_collective_mesh(layout, mesh, axis_name)


def describe_collective_port_mesh(
    layout: PortCollectiveLayout,
    mesh: Mesh,
    *,
    axis_name: str = "partition",
) -> PortCollectiveMeshReport:
    """Record an explicit Mesh without equating partitions, processes, hosts, or workers."""

    _validate_collective_mesh(layout, mesh, axis_name)
    addressable = set(mesh.local_devices)
    assignments = tuple(
        PortCollectiveDeviceAssignment(
            partition_index=partition_index,
            process_index=int(device.process_index),
            device_id=int(device.id),
            platform=str(device.platform),
            device_kind=str(device.device_kind),
            addressable=device in addressable,
        )
        for partition_index, device in enumerate(mesh.devices.reshape(-1))
    )
    return PortCollectiveMeshReport(
        axis_name=axis_name,
        partition_count=layout.partition_count,
        global_device_count=mesh.size,
        addressable_device_count=sum(record.addressable for record in assignments),
        process_count=len({record.process_index for record in assignments}),
        layout_sha256=layout.digest(),
        assignments=assignments,
    )


def build_packed_collective_port_matvec(
    layout: PortCollectiveLayout,
    mesh: Mesh,
    *,
    axis_name: str = "partition",
) -> PortPackedCollectiveMatvec:
    """Build the packed port SPMD operator without selecting or discovering devices."""

    return build_packed_collective_matvec(layout, mesh, axis_name=axis_name)


def build_validation_collective_port_matvec(
    layout: PortCollectiveLayout,
    mesh: Mesh,
    *,
    axis_name: str = "partition",
) -> PortCanonicalCollectiveMatvec:
    """Build the canonical small-problem port parity and gradient wrapper."""

    return build_validation_collective_matvec(layout, mesh, axis_name=axis_name)


def assert_finite_collective_port_result(result: jax.Array) -> None:
    """Fail on invalid sentinels or non-finite arithmetic after explicit synchronization."""

    finite = bool(np.asarray(jax.device_get(jnp.all(jnp.isfinite(result)))))
    if not finite:
        raise FloatingPointError("collective port result contains a non-finite value")


def collective_port_relative_difference(observed: jax.Array, expected: jax.Array) -> float:
    """Return a synchronized normwise comparison for retained runtime evidence."""

    numerator = float(np.asarray(jax.device_get(jnp.linalg.norm(observed - expected))))
    denominator = float(np.asarray(jax.device_get(jnp.linalg.norm(expected))))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator
