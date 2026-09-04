"""Generic fixed-capacity JAX collective transport for owned/ghost FEM operators.

The canonical FEM topology remains the unpadded :mod:`owned_ghost` representation.  This module
adds only static device transport slots.  It never discovers devices, initializes a distributed
runtime, or changes scalar precision.  Callers must provide an explicit one-dimensional JAX
``Mesh`` with exactly one FEM partition per device.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from femx.core.errors import ContractError

from .owned_ghost import OwnedGhostTopology, element_matrix_matvec


def _readonly_int64_array(
    values: object,
    *,
    label: str,
    rank: int,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{label} must be a regular integer array") from error
    if raw.dtype.kind not in "iu" or raw.ndim != rank:
        raise ContractError(f"{label} must be a rank-{rank} integer array")
    if shape is not None and raw.shape != shape:
        raise ContractError(f"{label} must have shape {shape}")
    result = np.ascontiguousarray(raw, dtype=np.int64)
    result.setflags(write=False)
    return result


def _require_nonnegative_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{label} must be a nonnegative integer")
    return value


def _is_float_or_complex(dtype: jnp.dtype) -> bool:
    return bool(jnp.issubdtype(dtype, jnp.floating) or jnp.issubdtype(dtype, jnp.complexfloating))


@dataclass(frozen=True, slots=True)
class CollectiveHaloLink:
    """One pairwise owner/receiver payload in fixed-capacity transport slots."""

    owner_partition: int
    ghost_partition: int
    global_dofs: np.ndarray
    owner_slots: np.ndarray
    ghost_slots: np.ndarray

    def __post_init__(self) -> None:
        owner = _require_nonnegative_integer(
            self.owner_partition,
            label="collective halo owner partition",
        )
        ghost = _require_nonnegative_integer(
            self.ghost_partition,
            label="collective halo ghost partition",
        )
        if owner == ghost:
            raise ContractError("collective halo owner and ghost partitions must differ")
        arrays: dict[str, np.ndarray] = {}
        for name in ("global_dofs", "owner_slots", "ghost_slots"):
            array = _readonly_int64_array(
                getattr(self, name),
                label=f"collective halo {name.replace('_', ' ')}",
                rank=1,
            )
            if array.size == 0:
                raise ContractError("collective halo links must contain at least one DOF")
            if np.any(array < 0):
                raise ContractError(f"collective halo {name.replace('_', ' ')} cannot be negative")
            if array.size > 1 and np.any(np.diff(array) <= 0):
                raise ContractError(
                    f"collective halo {name.replace('_', ' ')} must be strictly increasing"
                )
            arrays[name] = array
        if len({array.shape[0] for array in arrays.values()}) != 1:
            raise ContractError("collective halo arrays must have equal lengths")
        for name, array in arrays.items():
            object.__setattr__(self, name, array)


@dataclass(frozen=True, slots=True)
class CollectiveStorageReport:
    """Actual and allocated transport slots; never measured device memory."""

    actual_cell_slots: int
    allocated_cell_slots: int
    actual_owned_dof_slots: int
    allocated_owned_dof_slots: int
    actual_ghost_dof_slots: int
    allocated_ghost_dof_slots: int
    halo_link_count: int
    halo_value_count: int

    def __post_init__(self) -> None:
        for name in (
            "actual_cell_slots",
            "allocated_cell_slots",
            "actual_owned_dof_slots",
            "allocated_owned_dof_slots",
            "actual_ghost_dof_slots",
            "allocated_ghost_dof_slots",
            "halo_link_count",
            "halo_value_count",
        ):
            _require_nonnegative_integer(getattr(self, name), label=name.replace("_", " "))
        if self.actual_cell_slots > self.allocated_cell_slots:
            raise ContractError("actual cell slots cannot exceed allocated transport slots")
        if self.actual_owned_dof_slots > self.allocated_owned_dof_slots:
            raise ContractError("actual owned DOF slots cannot exceed allocated transport slots")
        if self.actual_ghost_dof_slots > self.allocated_ghost_dof_slots:
            raise ContractError("actual ghost DOF slots cannot exceed allocated transport slots")
        if self.halo_value_count != self.actual_ghost_dof_slots:
            raise ContractError("every actual ghost slot must have exactly one halo value")

    @staticmethod
    def _padding_fraction(actual: int, allocated: int) -> float:
        if allocated == 0:
            return 0.0
        return (allocated - actual) / allocated

    @property
    def cell_padding_fraction(self) -> float:
        return self._padding_fraction(self.actual_cell_slots, self.allocated_cell_slots)

    @property
    def owned_dof_padding_fraction(self) -> float:
        return self._padding_fraction(
            self.actual_owned_dof_slots,
            self.allocated_owned_dof_slots,
        )

    @property
    def ghost_dof_padding_fraction(self) -> float:
        return self._padding_fraction(
            self.actual_ghost_dof_slots,
            self.allocated_ghost_dof_slots,
        )


@dataclass(frozen=True, slots=True)
class CollectiveLayout:
    """Static transport lowering of one canonical unpadded owned/ghost topology."""

    topology: OwnedGhostTopology
    cell_ids: np.ndarray
    owned_dof_ids: np.ndarray
    ghost_dof_ids: np.ndarray
    cell_local_dofs: np.ndarray
    halo_links: tuple[CollectiveHaloLink, ...]
    schema_version: str
    expected_schema_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.topology, OwnedGhostTopology):
            raise ContractError("collective layout requires an OwnedGhostTopology")
        if (
            not isinstance(self.expected_schema_version, str)
            or not self.expected_schema_version.strip()
        ):
            raise ContractError("collective layout expected schema version must be nonempty")
        if self.schema_version != self.expected_schema_version:
            raise ContractError(
                f"collective layout schema must be {self.expected_schema_version!r}"
            )
        expected = _build_collective_transport(self.topology)
        arrays: dict[str, np.ndarray] = {}
        for name, expected_array in zip(
            ("cell_ids", "owned_dof_ids", "ghost_dof_ids", "cell_local_dofs"),
            expected[:4],
            strict=True,
        ):
            array = _readonly_int64_array(
                getattr(self, name),
                label=f"collective {name.replace('_', ' ')}",
                rank=expected_array.ndim,
                shape=expected_array.shape,
            )
            if not np.array_equal(array, expected_array):
                raise ContractError(f"collective {name.replace('_', ' ')} disagrees with topology")
            arrays[name] = array
        links = tuple(self.halo_links)
        expected_links = expected[4]
        if len(links) != len(expected_links) or any(
            observed.owner_partition != reference.owner_partition
            or observed.ghost_partition != reference.ghost_partition
            or not np.array_equal(observed.global_dofs, reference.global_dofs)
            or not np.array_equal(observed.owner_slots, reference.owner_slots)
            or not np.array_equal(observed.ghost_slots, reference.ghost_slots)
            for observed, reference in zip(links, expected_links, strict=True)
        ):
            raise ContractError("collective halo links disagree with canonical topology")
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "halo_links", links)

    @property
    def partition_count(self) -> int:
        return self.topology.partition_count

    @property
    def cell_capacity(self) -> int:
        return int(self.cell_ids.shape[1])

    @property
    def owned_dof_capacity(self) -> int:
        return int(self.owned_dof_ids.shape[1])

    @property
    def ghost_dof_capacity(self) -> int:
        return int(self.ghost_dof_ids.shape[1])

    @property
    def local_dof_capacity(self) -> int:
        return self.owned_dof_capacity + self.ghost_dof_capacity

    @property
    def constrained_transport_sentinel(self) -> int:
        return self.local_dof_capacity

    @property
    def storage_report(self) -> CollectiveStorageReport:
        actual_ghosts = sum(partition.ghost_dof_count for partition in self.topology.partitions)
        return CollectiveStorageReport(
            actual_cell_slots=self.topology.cell_count,
            allocated_cell_slots=self.partition_count * self.cell_capacity,
            actual_owned_dof_slots=self.topology.global_dof_count,
            allocated_owned_dof_slots=self.partition_count * self.owned_dof_capacity,
            actual_ghost_dof_slots=actual_ghosts,
            allocated_ghost_dof_slots=self.partition_count * self.ghost_dof_capacity,
            halo_link_count=len(self.halo_links),
            halo_value_count=sum(link.global_dofs.shape[0] for link in self.halo_links),
        )

    def digest(self) -> str:
        metadata = {
            "schema_version": self.schema_version,
            "partition_count": self.partition_count,
            "global_cell_count": self.topology.cell_count,
            "global_dof_count": self.topology.global_dof_count,
            "cell_capacity": self.cell_capacity,
            "owned_dof_capacity": self.owned_dof_capacity,
            "ghost_dof_capacity": self.ghost_dof_capacity,
        }
        hasher = hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        for name, array in (
            ("cell_ids", self.cell_ids),
            ("owned_dof_ids", self.owned_dof_ids),
            ("ghost_dof_ids", self.ghost_dof_ids),
            ("cell_local_dofs", self.cell_local_dofs),
        ):
            canonical = np.asarray(array, dtype="<i8", order="C")
            hasher.update(name.encode("utf-8"))
            hasher.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
            hasher.update(canonical.tobytes())
        for link in self.halo_links:
            hasher.update(
                np.asarray((link.owner_partition, link.ghost_partition), dtype="<i8").tobytes()
            )
            for array in (link.global_dofs, link.owner_slots, link.ghost_slots):
                canonical = np.asarray(array, dtype="<i8", order="C")
                hasher.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
                hasher.update(canonical.tobytes())
        return hasher.hexdigest()


def _build_collective_transport(
    topology: OwnedGhostTopology,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[CollectiveHaloLink, ...]]:
    cell_capacity = max(partition.owned_cells.shape[0] for partition in topology.partitions)
    owned_capacity = max(partition.owned_dof_count for partition in topology.partitions)
    ghost_capacity = max(partition.ghost_dof_count for partition in topology.partitions)
    local_capacity = owned_capacity + ghost_capacity
    width = topology.cell_dof_count

    cell_ids = np.full(
        (topology.partition_count, cell_capacity),
        topology.cell_count,
        dtype=np.int64,
    )
    owned_dof_ids = np.full(
        (topology.partition_count, owned_capacity),
        topology.global_dof_count,
        dtype=np.int64,
    )
    ghost_dof_ids = np.full(
        (topology.partition_count, ghost_capacity),
        topology.global_dof_count,
        dtype=np.int64,
    )
    cell_local_dofs = np.full(
        (topology.partition_count, cell_capacity, width),
        local_capacity,
        dtype=np.int64,
    )
    for partition in topology.partitions:
        index = partition.partition_index
        cell_count = partition.owned_cells.shape[0]
        owned_count = partition.owned_dof_count
        ghost_count = partition.ghost_dof_count
        cell_ids[index, :cell_count] = partition.owned_cells
        owned_dof_ids[index, :owned_count] = partition.owned_dofs
        ghost_dof_ids[index, :ghost_count] = partition.ghost_dofs
        local_map = partition.cell_local_dofs.copy()
        constrained = local_map == partition.constrained_sentinel
        ghosts = (local_map >= owned_count) & (local_map < partition.local_dof_count)
        local_map[ghosts] += owned_capacity - owned_count
        local_map[constrained] = local_capacity
        cell_local_dofs[index, :cell_count] = local_map

    links = tuple(
        CollectiveHaloLink(
            owner_partition=link.owner_partition,
            ghost_partition=link.ghost_partition,
            global_dofs=link.global_dofs,
            owner_slots=link.owner_local_indices,
            ghost_slots=(
                owned_capacity
                + link.ghost_local_indices
                - topology.partitions[link.ghost_partition].owned_dof_count
            ),
        )
        for link in topology.halo_links
    )
    for array in (cell_ids, owned_dof_ids, ghost_dof_ids, cell_local_dofs):
        array.setflags(write=False)
    return cell_ids, owned_dof_ids, ghost_dof_ids, cell_local_dofs, links


def prepare_collective_layout(
    topology: OwnedGhostTopology,
    *,
    schema_version: str,
) -> CollectiveLayout:
    """Lower a canonical topology without discovering or initializing devices."""

    if not isinstance(topology, OwnedGhostTopology):
        raise ContractError("collective lowering requires an OwnedGhostTopology")
    cell_ids, owned_ids, ghost_ids, cell_map, links = _build_collective_transport(topology)
    return CollectiveLayout(
        topology=topology,
        cell_ids=cell_ids,
        owned_dof_ids=owned_ids,
        ghost_dof_ids=ghost_ids,
        cell_local_dofs=cell_map,
        halo_links=links,
        schema_version=schema_version,
        expected_schema_version=schema_version,
    )


def pack_collective_cell_matrix(layout: CollectiveLayout, cell_matrix: jax.Array) -> jax.Array:
    """Pack canonical element matrices and zero every inactive transport cell."""

    width = layout.topology.cell_dof_count
    expected = (layout.topology.cell_count, width, width)
    if cell_matrix.ndim != 3 or cell_matrix.shape != expected:
        raise ValueError(f"collective cell matrix must be shaped (global cells, {width}, {width})")
    if not _is_float_or_complex(cell_matrix.dtype):
        raise TypeError("collective cell matrix must use a floating or complex dtype")
    extended = jnp.concatenate(
        (cell_matrix, jnp.zeros((1, width, width), dtype=cell_matrix.dtype)),
        axis=0,
    )
    return extended[jnp.asarray(layout.cell_ids)]


def pack_collective_cell_vector(layout: CollectiveLayout, cell_vector: jax.Array) -> jax.Array:
    """Pack canonical element-row contributions and zero inactive transport cells."""

    width = layout.topology.cell_dof_count
    expected = (layout.topology.cell_count, width)
    if cell_vector.ndim != 2 or cell_vector.shape != expected:
        raise ValueError(f"collective cell vector must be shaped (global cells, {width})")
    if not _is_float_or_complex(cell_vector.dtype):
        raise TypeError("collective cell vector must use a floating or complex dtype")
    extended = jnp.concatenate(
        (cell_vector, jnp.zeros((1, width), dtype=cell_vector.dtype)),
        axis=0,
    )
    return extended[jnp.asarray(layout.cell_ids)]


def pack_collective_owned_vector(layout: CollectiveLayout, vector: jax.Array) -> jax.Array:
    """Pack canonical owner values and zero every inactive owner slot."""

    if vector.ndim != 1 or vector.shape != (layout.topology.global_dof_count,):
        raise ValueError("collective global vector must match the global free DOFs")
    if not _is_float_or_complex(vector.dtype):
        raise TypeError("collective global vector must use a floating or complex dtype")
    extended = jnp.concatenate((vector, jnp.zeros((1,), dtype=vector.dtype)))
    return extended[jnp.asarray(layout.owned_dof_ids)]


def pack_collective_owned_mask(layout: CollectiveLayout) -> jax.Array:
    """Return the exact active-owner mask; inactive storage must remain algebraically zero."""

    return jnp.asarray(layout.owned_dof_ids < layout.topology.global_dof_count)


def unpack_collective_owned_vector(
    layout: CollectiveLayout,
    packed_owned_vector: jax.Array,
) -> jax.Array:
    """Reconstruct canonical global order and discard inactive owner slots."""

    expected = (layout.partition_count, layout.owned_dof_capacity)
    if packed_owned_vector.ndim != 2 or packed_owned_vector.shape != expected:
        raise ValueError("collective packed owner vector does not match the transport layout")
    if not _is_float_or_complex(packed_owned_vector.dtype):
        raise TypeError("collective packed owner vector must use a floating or complex dtype")
    assembled = (
        jnp.zeros((layout.topology.global_dof_count + 1,), dtype=packed_owned_vector.dtype)
        .at[jnp.asarray(layout.owned_dof_ids).reshape(-1)]
        .add(packed_owned_vector.reshape(-1))
    )
    return assembled[: layout.topology.global_dof_count]


def validate_collective_mesh(layout: CollectiveLayout, mesh: Mesh, axis_name: str) -> None:
    """Require an explicit one-device-per-partition one-dimensional Mesh."""

    if not isinstance(mesh, Mesh):
        raise ContractError("collective runtime requires an explicit JAX Mesh")
    if not isinstance(axis_name, str) or not axis_name:
        raise ContractError("collective mesh axis name must be a nonempty string")
    if mesh.empty or tuple(mesh.axis_names) != (axis_name,):
        raise ContractError("collective Mesh must have exactly the requested one-dimensional axis")
    if int(mesh.shape[axis_name]) != layout.partition_count:
        raise ContractError("collective Mesh must contain exactly one device per FEM partition")


def _local_row_assembly(
    local_cell_vector: jax.Array,
    local_map: jax.Array,
    local_dof_capacity: int,
) -> jax.Array:
    if local_cell_vector.ndim != 2 or local_cell_vector.shape != local_map.shape:
        raise ValueError("collective local cell vector and map must have equal rank-two shapes")
    sentinel = local_dof_capacity
    valid = jnp.all((local_map >= 0) & (local_map <= sentinel))
    safe_map = jnp.clip(local_map, 0, sentinel)
    assembled = (
        jnp.zeros((sentinel + 1,), dtype=local_cell_vector.dtype)
        .at[safe_map.reshape(-1)]
        .add(local_cell_vector.reshape(-1))
    )
    return jnp.where(valid, assembled[:sentinel], jnp.asarray(jnp.nan, local_cell_vector.dtype))


def _exchange_values_and_apply(
    layout: CollectiveLayout,
    local_cells: jax.Array,
    local_map: jax.Array,
    local_owned: jax.Array,
    partition_index: jax.Array,
    axis_name: str,
) -> jax.Array:
    local_vector = _exchange_owned_values(
        layout,
        local_owned,
        partition_index,
        axis_name,
    )
    local_contribution = element_matrix_matvec(local_cells, local_map, local_vector)
    return _reduce_contributions(layout, local_contribution, partition_index, axis_name)


def _exchange_owned_values(
    layout: CollectiveLayout,
    local_owned: jax.Array,
    partition_index: jax.Array,
    axis_name: str,
) -> jax.Array:
    """Populate fixed-capacity local owner/ghost storage through pairwise routes."""

    local_vector = jnp.concatenate(
        (local_owned, jnp.zeros((layout.ghost_dof_capacity,), dtype=local_owned.dtype))
    )
    for link in layout.halo_links:
        payload = local_owned[jnp.asarray(link.owner_slots)]
        received = lax.ppermute(  # type: ignore[no-untyped-call]
            payload,
            axis_name,
            ((link.owner_partition, link.ghost_partition),),
        )
        receiver_vector = local_vector.at[jnp.asarray(link.ghost_slots)].set(received)
        local_vector = jnp.where(
            partition_index == link.ghost_partition,
            receiver_vector,
            local_vector,
        )
    return local_vector


def collective_local_cell_gather(
    layout: CollectiveLayout,
    local_map: jax.Array,
    local_owned: jax.Array,
    partition_index: jax.Array,
    axis_name: str,
) -> jax.Array:
    """Gather one SPMD-local owner/ghost state to element-local DOF values."""

    local_vector = _exchange_owned_values(
        layout,
        local_owned,
        partition_index,
        axis_name,
    )
    sentinel = layout.constrained_transport_sentinel
    extended = jnp.concatenate((local_vector, jnp.zeros((1,), dtype=local_vector.dtype)))
    valid = jnp.all((local_map >= 0) & (local_map <= sentinel))
    safe_map = jnp.clip(local_map, 0, sentinel)
    gathered = extended[safe_map]
    return jnp.where(valid, gathered, jnp.asarray(jnp.nan, gathered.dtype))


def collective_local_matvec(
    layout: CollectiveLayout,
    local_cells: jax.Array,
    local_map: jax.Array,
    local_owned: jax.Array,
    partition_index: jax.Array,
    axis_name: str,
) -> jax.Array:
    """Apply one SPMD-local shard, including pairwise value and contribution exchange."""

    return _exchange_values_and_apply(
        layout,
        local_cells,
        local_map,
        local_owned,
        partition_index,
        axis_name,
    )


def _reduce_contributions(
    layout: CollectiveLayout,
    local_contribution: jax.Array,
    partition_index: jax.Array,
    axis_name: str,
) -> jax.Array:
    owned = local_contribution[: layout.owned_dof_capacity]
    for link in layout.halo_links:
        payload = local_contribution[jnp.asarray(link.ghost_slots)]
        received = lax.ppermute(  # type: ignore[no-untyped-call]
            payload,
            axis_name,
            ((link.ghost_partition, link.owner_partition),),
        )
        owner_contribution = owned.at[jnp.asarray(link.owner_slots)].add(received)
        owned = jnp.where(
            partition_index == link.owner_partition,
            owner_contribution,
            owned,
        )
    return owned


PackedCollectiveMatvec = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]
PackedCollectiveRowAssembly = Callable[[jax.Array, jax.Array], jax.Array]
PackedCollectiveCellGather = Callable[[jax.Array, jax.Array], jax.Array]
CanonicalCollectiveMatvec = Callable[[jax.Array, jax.Array], jax.Array]


def build_packed_collective_matvec(
    layout: CollectiveLayout,
    mesh: Mesh,
    *,
    axis_name: str = "partition",
) -> PackedCollectiveMatvec:
    """Build a packed pairwise-halo operator on an explicit global JAX Mesh."""

    validate_collective_mesh(layout, mesh, axis_name)
    width = layout.topology.cell_dof_count
    cell_shape = (layout.partition_count, layout.cell_capacity, width, width)
    map_shape = (layout.partition_count, layout.cell_capacity, width)
    owned_shape = (layout.partition_count, layout.owned_dof_capacity)
    cell_spec = P(axis_name, None, None, None)  # type: ignore[no-untyped-call]
    map_spec = P(axis_name, None, None)  # type: ignore[no-untyped-call]
    owned_spec = P(axis_name, None)  # type: ignore[no-untyped-call]

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(cell_spec, map_spec, owned_spec),
        out_specs=owned_spec,
        check_vma=True,
    )
    def mapped(
        packed_cell_matrix: jax.Array,
        packed_cell_local_dofs: jax.Array,
        packed_owned_vector: jax.Array,
    ) -> jax.Array:
        partition_index = lax.axis_index(axis_name)
        result = _exchange_values_and_apply(
            layout,
            packed_cell_matrix[0],
            packed_cell_local_dofs[0],
            packed_owned_vector[0],
            partition_index,
            axis_name,
        )
        return result[None, :]

    def apply(
        packed_cell_matrix: jax.Array,
        packed_cell_local_dofs: jax.Array,
        packed_owned_vector: jax.Array,
    ) -> jax.Array:
        if packed_cell_matrix.ndim != 4 or packed_cell_matrix.shape != cell_shape:
            raise ValueError("collective packed cell matrix does not match the transport layout")
        if packed_cell_local_dofs.ndim != 3 or packed_cell_local_dofs.shape != map_shape:
            raise ValueError("collective packed cell map does not match the transport layout")
        if packed_owned_vector.ndim != 2 or packed_owned_vector.shape != owned_shape:
            raise ValueError("collective packed owner vector does not match the transport layout")
        if not _is_float_or_complex(packed_cell_matrix.dtype):
            raise TypeError("collective packed cell matrix must use a floating or complex dtype")
        if not jnp.issubdtype(packed_cell_local_dofs.dtype, jnp.integer):
            raise TypeError("collective packed cell map must use an integer dtype")
        if not _is_float_or_complex(packed_owned_vector.dtype):
            raise TypeError("collective packed owner vector must use a floating or complex dtype")
        return cast(
            jax.Array,
            mapped(packed_cell_matrix, packed_cell_local_dofs, packed_owned_vector),
        )

    return apply


def build_packed_collective_row_assembly(
    layout: CollectiveLayout,
    mesh: Mesh,
    *,
    axis_name: str = "partition",
) -> PackedCollectiveRowAssembly:
    """Assemble cell-local row values and return owner-authoritative packed coefficients."""

    validate_collective_mesh(layout, mesh, axis_name)
    width = layout.topology.cell_dof_count
    cell_shape = (layout.partition_count, layout.cell_capacity, width)
    map_shape = (layout.partition_count, layout.cell_capacity, width)
    cell_spec = P(axis_name, None, None)  # type: ignore[no-untyped-call]
    map_spec = P(axis_name, None, None)  # type: ignore[no-untyped-call]
    owned_spec = P(axis_name, None)  # type: ignore[no-untyped-call]

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(cell_spec, map_spec),
        out_specs=owned_spec,
        check_vma=True,
    )
    def mapped(packed_cell_vector: jax.Array, packed_cell_local_dofs: jax.Array) -> jax.Array:
        partition_index = lax.axis_index(axis_name)
        local = _local_row_assembly(
            packed_cell_vector[0],
            packed_cell_local_dofs[0],
            layout.local_dof_capacity,
        )
        owned = _reduce_contributions(layout, local, partition_index, axis_name)
        return owned[None, :]

    def apply(packed_cell_vector: jax.Array, packed_cell_local_dofs: jax.Array) -> jax.Array:
        if packed_cell_vector.ndim != 3 or packed_cell_vector.shape != cell_shape:
            raise ValueError("collective packed cell vector does not match the transport layout")
        if packed_cell_local_dofs.ndim != 3 or packed_cell_local_dofs.shape != map_shape:
            raise ValueError("collective packed cell map does not match the transport layout")
        if not _is_float_or_complex(packed_cell_vector.dtype):
            raise TypeError("collective packed cell vector must use a floating or complex dtype")
        if not jnp.issubdtype(packed_cell_local_dofs.dtype, jnp.integer):
            raise TypeError("collective packed cell map must use an integer dtype")
        return cast(jax.Array, mapped(packed_cell_vector, packed_cell_local_dofs))

    return apply


def build_packed_collective_cell_gather(
    layout: CollectiveLayout,
    mesh: Mesh,
    *,
    axis_name: str = "partition",
) -> PackedCollectiveCellGather:
    """Build a pairwise-halo gather from owner storage to element-local values."""

    validate_collective_mesh(layout, mesh, axis_name)
    width = layout.topology.cell_dof_count
    map_shape = (layout.partition_count, layout.cell_capacity, width)
    owned_shape = (layout.partition_count, layout.owned_dof_capacity)
    map_spec = P(axis_name, None, None)  # type: ignore[no-untyped-call]
    owned_spec = P(axis_name, None)  # type: ignore[no-untyped-call]
    cell_spec = P(axis_name, None, None)  # type: ignore[no-untyped-call]

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(map_spec, owned_spec),
        out_specs=cell_spec,
        check_vma=True,
    )
    def mapped(
        packed_cell_local_dofs: jax.Array,
        packed_owned_vector: jax.Array,
    ) -> jax.Array:
        partition_index = lax.axis_index(axis_name)
        gathered = collective_local_cell_gather(
            layout,
            packed_cell_local_dofs[0],
            packed_owned_vector[0],
            partition_index,
            axis_name,
        )
        return gathered[None, :, :]

    def apply(
        packed_cell_local_dofs: jax.Array,
        packed_owned_vector: jax.Array,
    ) -> jax.Array:
        if packed_cell_local_dofs.ndim != 3 or packed_cell_local_dofs.shape != map_shape:
            raise ValueError("collective packed cell map does not match the transport layout")
        if packed_owned_vector.ndim != 2 or packed_owned_vector.shape != owned_shape:
            raise ValueError("collective packed owner vector does not match the transport layout")
        if not jnp.issubdtype(packed_cell_local_dofs.dtype, jnp.integer):
            raise TypeError("collective packed cell map must use an integer dtype")
        if not _is_float_or_complex(packed_owned_vector.dtype):
            raise TypeError("collective packed owner vector must use a floating or complex dtype")
        return cast(jax.Array, mapped(packed_cell_local_dofs, packed_owned_vector))

    return apply


def build_validation_collective_matvec(
    layout: CollectiveLayout,
    mesh: Mesh,
    *,
    axis_name: str = "partition",
) -> CanonicalCollectiveMatvec:
    """Materialize canonical inputs only for small parity and gradient witnesses."""

    packed = build_packed_collective_matvec(layout, mesh, axis_name=axis_name)
    packed_map = jnp.asarray(layout.cell_local_dofs)

    def apply(cell_matrix: jax.Array, vector: jax.Array) -> jax.Array:
        result = packed(
            pack_collective_cell_matrix(layout, cell_matrix),
            packed_map,
            pack_collective_owned_vector(layout, vector),
        )
        return unpack_collective_owned_vector(layout, result)

    return apply
