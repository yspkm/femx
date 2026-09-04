"""Backend-internal owned/ghost algebra for element-local matrix actions.

The canonical topology is host prepared and keeps one global DOF identity, one owner for every
cell and DOF, owned-before-ghost local vectors, and explicit owner/receiver links.  This module is
an in-process algebraic substrate: it does not initialize JAX distribution, discover devices, or
advertise accelerator execution.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from femx.core.errors import ContractError


def _canonical_int64_array(
    values: object,
    *,
    label: str,
    rank: int,
    columns: int | None = None,
) -> np.ndarray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{label} must be a regular integer array") from error
    if raw.dtype.kind not in "iu" or raw.ndim != rank:
        raise ContractError(f"{label} must be a rank-{rank} integer array")
    if columns is not None and raw.shape[1] != columns:
        raise ContractError(f"{label} must have {columns} columns")
    result = np.ascontiguousarray(raw, dtype=np.int64)
    result.setflags(write=False)
    return result


def _require_static_positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{label} must be a positive integer")
    return value


def _require_sorted_unique(values: np.ndarray, *, label: str) -> None:
    if values.size > 1 and np.any(np.diff(values) <= 0):
        raise ContractError(f"{label} must be unique and strictly increasing")


def _referenced_active_dofs(cell_map: np.ndarray, global_dof_count: int) -> np.ndarray:
    """Return a bounded-memory mask of reduced DOFs referenced by at least one cell."""

    referenced = np.zeros(global_dof_count, dtype=np.bool_)
    for local_index in range(cell_map.shape[1]):
        local_dofs = cell_map[:, local_index]
        active = local_dofs < global_dof_count
        referenced[local_dofs[active]] = True
    return referenced


def _require_unique_active_cell_dofs(cell_map: np.ndarray, global_dof_count: int) -> None:
    """Reject repeated active DOFs without a Python loop over every cell."""

    for first in range(cell_map.shape[1]):
        first_dofs = cell_map[:, first]
        for second in range(first + 1, cell_map.shape[1]):
            if np.any((first_dofs == cell_map[:, second]) & (first_dofs < global_dof_count)):
                raise ContractError("owned/ghost cell map repeats an active DOF within one cell")


def _minimum_incident_cell_owners(
    cell_map: np.ndarray,
    cell_owners: np.ndarray,
    *,
    global_dof_count: int,
    partition_count: int,
) -> np.ndarray:
    """Choose the lowest incident cell owner in element-width linear passes."""

    owners = np.full(global_dof_count, partition_count, dtype=np.int64)
    for local_index in range(cell_map.shape[1]):
        local_dofs = cell_map[:, local_index]
        active = local_dofs < global_dof_count
        np.minimum.at(owners, local_dofs[active], cell_owners[active])
    return owners


def _dof_owners_are_incident(
    cell_map: np.ndarray,
    cell_owners: np.ndarray,
    dof_owners: np.ndarray,
    *,
    global_dof_count: int,
) -> bool:
    """Return whether every declared DOF owner owns at least one incident cell."""

    owner_seen = np.zeros(global_dof_count, dtype=np.bool_)
    for local_index in range(cell_map.shape[1]):
        local_dofs = cell_map[:, local_index]
        active = local_dofs < global_dof_count
        active_dofs = local_dofs[active]
        matches = cell_owners[active] == dof_owners[active_dofs]
        np.logical_or.at(owner_seen, active_dofs, matches)
    return bool(np.all(owner_seen))


@dataclass(frozen=True, slots=True)
class OwnedGhostPartition:
    """One canonical cell and local-DOF partition."""

    partition_index: int
    owned_cells: np.ndarray
    owned_dofs: np.ndarray
    ghost_dofs: np.ndarray
    local_dofs: np.ndarray
    cell_local_dofs: np.ndarray

    def __post_init__(self) -> None:
        if (
            isinstance(self.partition_index, bool)
            or not isinstance(self.partition_index, int)
            or self.partition_index < 0
        ):
            raise ContractError("owned/ghost partition index must be a nonnegative integer")
        for field_name in ("owned_cells", "owned_dofs", "ghost_dofs", "local_dofs"):
            canonical = _canonical_int64_array(
                getattr(self, field_name),
                label=f"owned/ghost {field_name.replace('_', ' ')}",
                rank=1,
            )
            if np.any(canonical < 0):
                raise ContractError(
                    f"owned/ghost {field_name.replace('_', ' ')} cannot be negative"
                )
            label = f"owned/ghost {field_name.replace('_', ' ')}"
            if field_name == "local_dofs":
                if np.unique(canonical).shape[0] != canonical.shape[0]:
                    raise ContractError(f"{label} must be unique")
            else:
                _require_sorted_unique(canonical, label=label)
            object.__setattr__(self, field_name, canonical)

        local_map = _canonical_int64_array(
            self.cell_local_dofs,
            label="owned/ghost cell-local DOFs",
            rank=2,
        )
        if local_map.shape[1] == 0:
            raise ContractError("owned/ghost cell-local DOFs require at least one column")
        if local_map.shape[0] != self.owned_cells.shape[0]:
            raise ContractError("owned/ghost cell-local rows must match the owned cells")
        if np.any(local_map < 0) or np.any(local_map > self.local_dof_count):
            raise ContractError("owned/ghost cell-local DOFs contain an out-of-range index")
        expected_local = np.concatenate((self.owned_dofs, self.ghost_dofs))
        if not np.array_equal(self.local_dofs, expected_local):
            raise ContractError("owned/ghost local DOFs must order owned entries before ghosts")
        object.__setattr__(self, "cell_local_dofs", local_map)

    @property
    def owned_dof_count(self) -> int:
        """Return the number of owner-authoritative local coefficients."""

        return int(self.owned_dofs.shape[0])

    @property
    def ghost_dof_count(self) -> int:
        """Return the number of halo-populated local coefficients."""

        return int(self.ghost_dofs.shape[0])

    @property
    def local_dof_count(self) -> int:
        """Return the complete owned-plus-ghost local vector length."""

        return int(self.local_dofs.shape[0])

    @property
    def cell_dof_count(self) -> int:
        """Return the static number of local coefficients per element."""

        return int(self.cell_local_dofs.shape[1])

    @property
    def constrained_sentinel(self) -> int:
        """Return the local map value reserved for constrained coefficients."""

        return self.local_dof_count


@dataclass(frozen=True, slots=True)
class HaloLink:
    """Canonical owner-to-ghost value link and reverse contribution link."""

    owner_partition: int
    ghost_partition: int
    global_dofs: np.ndarray
    owner_local_indices: np.ndarray
    ghost_local_indices: np.ndarray

    def __post_init__(self) -> None:
        for label, value in (
            ("owner", self.owner_partition),
            ("ghost", self.ghost_partition),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractError(f"halo {label} partition must be a nonnegative integer")
        if self.owner_partition == self.ghost_partition:
            raise ContractError("halo owner and ghost partitions must differ")
        arrays: dict[str, np.ndarray] = {}
        for field_name in ("global_dofs", "owner_local_indices", "ghost_local_indices"):
            canonical = _canonical_int64_array(
                getattr(self, field_name),
                label=f"halo {field_name.replace('_', ' ')}",
                rank=1,
            )
            if canonical.size == 0:
                raise ContractError("halo links must contain at least one shared DOF")
            if np.any(canonical < 0):
                raise ContractError(f"halo {field_name.replace('_', ' ')} cannot be negative")
            _require_sorted_unique(canonical, label=f"halo {field_name.replace('_', ' ')}")
            arrays[field_name] = canonical
        if len({array.shape[0] for array in arrays.values()}) != 1:
            raise ContractError("halo global and local index arrays must have equal lengths")
        for field_name, canonical in arrays.items():
            object.__setattr__(self, field_name, canonical)


@dataclass(frozen=True, slots=True)
class OwnedGhostTopology:
    """Globally identified DOFs plus deterministic partition communication metadata."""

    partition_count: int
    global_dof_count: int
    cell_reduced_dofs: np.ndarray
    cell_owners: np.ndarray
    dof_owners: np.ndarray
    partitions: tuple[OwnedGhostPartition, ...]
    halo_links: tuple[HaloLink, ...]

    def __post_init__(self) -> None:
        partition_count = _require_static_positive_integer(
            self.partition_count,
            label="owned/ghost partition count",
        )
        global_dof_count = _require_static_positive_integer(
            self.global_dof_count,
            label="owned/ghost global DOF count",
        )
        cell_map = _canonical_int64_array(
            self.cell_reduced_dofs,
            label="owned/ghost cell reduced DOFs",
            rank=2,
        )
        if cell_map.shape[0] == 0:
            raise ContractError("owned/ghost topology requires at least one cell")
        if cell_map.shape[1] == 0:
            raise ContractError("owned/ghost cell reduced DOFs require at least one column")
        cell_owners = _canonical_int64_array(
            self.cell_owners,
            label="owned/ghost cell owners",
            rank=1,
        )
        dof_owners = _canonical_int64_array(
            self.dof_owners,
            label="owned/ghost DOF owners",
            rank=1,
        )
        if cell_owners.shape != (cell_map.shape[0],):
            raise ContractError("owned/ghost cell owners must match the cell count")
        if dof_owners.shape != (global_dof_count,):
            raise ContractError("owned/ghost DOF owners must match the global DOF count")
        if np.any(cell_map < 0) or np.any(cell_map > global_dof_count):
            raise ContractError("owned/ghost cell map contains an out-of-range reduced DOF")
        if np.any(cell_owners < 0) or np.any(cell_owners >= partition_count):
            raise ContractError("owned/ghost cell owner lies outside the partition range")
        if np.any(dof_owners < 0) or np.any(dof_owners >= partition_count):
            raise ContractError("owned/ghost DOF owner lies outside the partition range")
        if not np.all(_referenced_active_dofs(cell_map, global_dof_count)):
            raise ContractError("owned/ghost cell map must reference every global free DOF")
        _require_unique_active_cell_dofs(cell_map, global_dof_count)
        if not _dof_owners_are_incident(
            cell_map,
            cell_owners,
            dof_owners,
            global_dof_count=global_dof_count,
        ):
            raise ContractError("owned/ghost DOF owner must own an incident cell")

        partitions = tuple(self.partitions)
        links = tuple(self.halo_links)
        if len(partitions) != partition_count:
            raise ContractError("owned/ghost topology must contain every partition exactly once")
        for partition_index, partition in enumerate(partitions):
            if partition.partition_index != partition_index:
                raise ContractError("owned/ghost partitions must follow canonical index order")
            if partition.cell_dof_count != cell_map.shape[1]:
                raise ContractError("owned/ghost partition cell width disagrees with topology")
            expected_cells = np.flatnonzero(cell_owners == partition_index).astype(np.int64)
            if not np.array_equal(partition.owned_cells, expected_cells):
                raise ContractError("owned/ghost partition cells disagree with global ownership")
            present = np.unique(cell_map[expected_cells]) if expected_cells.size else np.empty(0)
            present = np.asarray(present[present < global_dof_count], dtype=np.int64)
            expected_owned = present[dof_owners[present] == partition_index]
            expected_ghosts = present[dof_owners[present] != partition_index]
            if not np.array_equal(partition.owned_dofs, expected_owned):
                raise ContractError(
                    "owned/ghost partition owner DOFs disagree with global ownership"
                )
            if not np.array_equal(partition.ghost_dofs, expected_ghosts):
                raise ContractError("owned/ghost partition ghosts disagree with global ownership")
            global_to_local = np.full(
                global_dof_count + 1,
                partition.local_dof_count,
                dtype=np.int64,
            )
            global_to_local[partition.local_dofs] = np.arange(
                partition.local_dof_count,
                dtype=np.int64,
            )
            expected_local_map = global_to_local[cell_map[expected_cells]]
            if not np.array_equal(partition.cell_local_dofs, expected_local_map):
                raise ContractError("owned/ghost cell-local map disagrees with global DOF identity")

        expected_links: dict[tuple[int, int], np.ndarray] = {}
        for partition in partitions:
            for owner in np.unique(dof_owners[partition.ghost_dofs]):
                shared = partition.ghost_dofs[dof_owners[partition.ghost_dofs] == owner]
                expected_links[(int(owner), partition.partition_index)] = shared
        if len(links) != len(expected_links):
            raise ContractError("owned/ghost halo links must cover every ghost exactly once")
        for link, key in zip(links, sorted(expected_links), strict=True):
            if (link.owner_partition, link.ghost_partition) != key:
                raise ContractError("owned/ghost halo links must follow canonical pair order")
            if not np.array_equal(link.global_dofs, expected_links[key]):
                raise ContractError("owned/ghost halo link DOFs disagree with partition ghosts")
            owner = partitions[link.owner_partition]
            ghost = partitions[link.ghost_partition]
            if np.any(link.owner_local_indices >= owner.owned_dof_count):
                raise ContractError("owned/ghost halo owner index lies outside owner storage")
            if np.any(link.ghost_local_indices < ghost.owned_dof_count) or np.any(
                link.ghost_local_indices >= ghost.local_dof_count
            ):
                raise ContractError("owned/ghost halo ghost index lies outside ghost storage")
            if not np.array_equal(owner.local_dofs[link.owner_local_indices], link.global_dofs):
                raise ContractError("owned/ghost halo owner indices identify the wrong DOFs")
            if not np.array_equal(ghost.local_dofs[link.ghost_local_indices], link.global_dofs):
                raise ContractError("owned/ghost halo ghost indices identify the wrong DOFs")

        object.__setattr__(self, "cell_reduced_dofs", cell_map)
        object.__setattr__(self, "cell_owners", cell_owners)
        object.__setattr__(self, "dof_owners", dof_owners)
        object.__setattr__(self, "partitions", partitions)
        object.__setattr__(self, "halo_links", links)

    @property
    def cell_count(self) -> int:
        """Return the number of globally identified cells."""

        return int(self.cell_reduced_dofs.shape[0])

    @property
    def cell_dof_count(self) -> int:
        """Return the static number of local coefficients per element."""

        return int(self.cell_reduced_dofs.shape[1])

    @property
    def constrained_sentinel(self) -> int:
        """Return the global map value reserved for constrained coefficients."""

        return self.global_dof_count


PartitionVectors = tuple[jax.Array, ...]


def prepare_owned_ghost_topology(
    cell_reduced_dofs: object,
    cell_owners: object,
    *,
    global_dof_count: int,
    partition_count: int,
    dof_owners: object | None = None,
) -> OwnedGhostTopology:
    """Create a deterministic owned/ghost plan without probing devices or processes."""

    dof_count = _require_static_positive_integer(
        global_dof_count,
        label="owned/ghost global DOF count",
    )
    process_count = _require_static_positive_integer(
        partition_count,
        label="owned/ghost partition count",
    )
    cell_map = _canonical_int64_array(
        cell_reduced_dofs,
        label="owned/ghost cell reduced DOFs",
        rank=2,
    )
    if cell_map.shape[0] == 0:
        raise ContractError("owned/ghost topology requires at least one cell")
    if cell_map.shape[1] == 0:
        raise ContractError("owned/ghost cell reduced DOFs require at least one column")
    owners_by_cell = _canonical_int64_array(
        cell_owners,
        label="owned/ghost cell owners",
        rank=1,
    )
    if owners_by_cell.shape != (cell_map.shape[0],):
        raise ContractError("owned/ghost cell owners must match the cell count")
    if np.any(cell_map < 0) or np.any(cell_map > dof_count):
        raise ContractError("owned/ghost cell map contains an out-of-range reduced DOF")
    if np.any(owners_by_cell < 0) or np.any(owners_by_cell >= process_count):
        raise ContractError("owned/ghost cell owner lies outside the partition range")
    if dof_owners is None:
        owners_by_dof = _minimum_incident_cell_owners(
            cell_map,
            owners_by_cell,
            global_dof_count=dof_count,
            partition_count=process_count,
        )
        if np.any(owners_by_dof == process_count):
            raise ContractError("owned/ghost cell map must reference every global free DOF")
    else:
        owners_by_dof = _canonical_int64_array(
            dof_owners,
            label="owned/ghost DOF owners",
            rank=1,
        )
        if owners_by_dof.shape != (dof_count,):
            raise ContractError("owned/ghost DOF owners must match the global DOF count")
        if np.any(owners_by_dof < 0) or np.any(owners_by_dof >= process_count):
            raise ContractError("owned/ghost DOF owner lies outside the partition range")
        if not _dof_owners_are_incident(
            cell_map,
            owners_by_cell,
            owners_by_dof,
            global_dof_count=dof_count,
        ):
            raise ContractError("owned/ghost DOF owner must own an incident cell")

    partitions: list[OwnedGhostPartition] = []
    for partition_index in range(process_count):
        owned_cells = np.flatnonzero(owners_by_cell == partition_index).astype(np.int64)
        present = np.unique(cell_map[owned_cells]) if owned_cells.size else np.empty(0)
        present = np.asarray(present[present < dof_count], dtype=np.int64)
        owned_dofs = present[owners_by_dof[present] == partition_index]
        ghost_dofs = present[owners_by_dof[present] != partition_index]
        local_dofs = np.concatenate((owned_dofs, ghost_dofs))
        global_to_local = np.full(dof_count + 1, local_dofs.shape[0], dtype=np.int64)
        global_to_local[local_dofs] = np.arange(local_dofs.shape[0], dtype=np.int64)
        partitions.append(
            OwnedGhostPartition(
                partition_index=partition_index,
                owned_cells=owned_cells,
                owned_dofs=owned_dofs,
                ghost_dofs=ghost_dofs,
                local_dofs=local_dofs,
                cell_local_dofs=global_to_local[cell_map[owned_cells]],
            )
        )

    links: list[HaloLink] = []
    for owner_partition in range(process_count):
        owner = partitions[owner_partition]
        for ghost_partition in range(process_count):
            if owner_partition == ghost_partition:
                continue
            ghost = partitions[ghost_partition]
            shared = ghost.ghost_dofs[owners_by_dof[ghost.ghost_dofs] == owner_partition]
            if shared.size == 0:
                continue
            owner_indices = np.searchsorted(owner.owned_dofs, shared)
            ghost_indices = ghost.owned_dof_count + np.searchsorted(ghost.ghost_dofs, shared)
            links.append(
                HaloLink(
                    owner_partition=owner_partition,
                    ghost_partition=ghost_partition,
                    global_dofs=shared,
                    owner_local_indices=owner_indices,
                    ghost_local_indices=ghost_indices,
                )
            )
    return OwnedGhostTopology(
        partition_count=process_count,
        global_dof_count=dof_count,
        cell_reduced_dofs=cell_map,
        cell_owners=owners_by_cell,
        dof_owners=owners_by_dof,
        partitions=tuple(partitions),
        halo_links=tuple(links),
    )


def _validate_global_vector(topology: OwnedGhostTopology, vector: jax.Array) -> None:
    if vector.ndim != 1 or vector.shape[0] != topology.global_dof_count:
        raise ValueError("owned/ghost global vector must match the global free DOFs")
    if not (
        jnp.issubdtype(vector.dtype, jnp.floating)
        or jnp.issubdtype(vector.dtype, jnp.complexfloating)
    ):
        raise TypeError("owned/ghost global vector must use a floating or complex dtype")


def _validate_partition_vectors(
    topology: OwnedGhostTopology,
    vectors: PartitionVectors,
    *,
    owned_only: bool,
) -> None:
    if len(vectors) != topology.partition_count:
        raise ValueError("owned/ghost vectors must contain every partition exactly once")
    expected_dtype = vectors[0].dtype
    for partition, vector in zip(topology.partitions, vectors, strict=True):
        expected_count = partition.owned_dof_count if owned_only else partition.local_dof_count
        if vector.ndim != 1 or vector.shape[0] != expected_count:
            storage = "owned" if owned_only else "local"
            raise ValueError(f"owned/ghost {storage} vector does not match its partition")
        if vector.dtype != expected_dtype:
            raise TypeError("owned/ghost partition vectors must use one common dtype")


def partition_owned_vector(
    topology: OwnedGhostTopology,
    vector: jax.Array,
) -> PartitionVectors:
    """Scatter authoritative global values and initialize every ghost slot to zero."""

    _validate_global_vector(topology, vector)
    local_vectors: list[jax.Array] = []
    for partition in topology.partitions:
        owned = vector[jnp.asarray(partition.owned_dofs)]
        ghosts = jnp.zeros((partition.ghost_dof_count,), dtype=vector.dtype)
        local_vectors.append(jnp.concatenate((owned, ghosts)))
    return tuple(local_vectors)


def exchange_halo_values(
    topology: OwnedGhostTopology,
    local_vectors: PartitionVectors,
) -> PartitionVectors:
    """Copy owner-authoritative coefficients into every required ghost slot."""

    _validate_partition_vectors(topology, local_vectors, owned_only=False)
    exchanged = list(local_vectors)
    for link in topology.halo_links:
        owner_values = local_vectors[link.owner_partition][jnp.asarray(link.owner_local_indices)]
        receiver = exchanged[link.ghost_partition]
        exchanged[link.ghost_partition] = receiver.at[jnp.asarray(link.ghost_local_indices)].set(
            owner_values
        )
    return tuple(exchanged)


def element_matrix_matvec(
    cell_matrix: jax.Array,
    cell_reduced_dofs: jax.Array,
    vector: jax.Array,
) -> jax.Array:
    """Apply arbitrary-width element matrices with one constrained sentinel."""

    if cell_matrix.ndim != 3 or cell_matrix.shape[1] != cell_matrix.shape[2]:
        raise ValueError("element-local matrix must be shaped (cells, local DOFs, local DOFs)")
    if cell_reduced_dofs.shape != cell_matrix.shape[:2]:
        raise ValueError("element-local cell map must match the matrix cell and row dimensions")
    if vector.ndim != 1 or vector.shape[0] == 0:
        raise ValueError("element-local vector must be a nonempty rank-one array")
    if not jnp.issubdtype(cell_reduced_dofs.dtype, jnp.integer):
        raise TypeError("element-local cell map must use an integer dtype")
    free_dof_count = vector.shape[0]
    mapping_valid = jnp.all((cell_reduced_dofs >= 0) & (cell_reduced_dofs <= free_dof_count))
    safe_mapping = jnp.clip(cell_reduced_dofs, 0, free_dof_count)
    dtype = jnp.result_type(cell_matrix.dtype, vector.dtype)
    extended_vector = jnp.concatenate((vector.astype(dtype), jnp.zeros((1,), dtype=dtype)))
    local_input = extended_vector[safe_mapping]
    local_output = jnp.einsum("cij,cj->ci", cell_matrix.astype(dtype), local_input)
    assembled = (
        jnp.zeros((free_dof_count + 1,), dtype=dtype)
        .at[safe_mapping.reshape(-1)]
        .add(local_output.reshape(-1))
    )
    return jnp.where(
        mapping_valid,
        assembled[:free_dof_count],
        jnp.asarray(jnp.nan, dtype=dtype),
    )


def local_owned_cell_matvec(
    local_cell_matrix: jax.Array,
    partition: OwnedGhostPartition,
    local_vector: jax.Array,
) -> jax.Array:
    """Apply exactly one partition's owned cells to its owned-plus-ghost vector."""

    expected_shape = (
        partition.owned_cells.shape[0],
        partition.cell_dof_count,
        partition.cell_dof_count,
    )
    if local_cell_matrix.ndim != 3 or local_cell_matrix.shape != expected_shape:
        raise ValueError("owned/ghost local cell matrix does not match the partition cells")
    if local_vector.ndim != 1 or local_vector.shape[0] != partition.local_dof_count:
        raise ValueError("owned/ghost local vector does not match the partition DOFs")
    if partition.local_dof_count == 0:
        dtype = jnp.result_type(local_cell_matrix.dtype, local_vector.dtype)
        return jnp.zeros((0,), dtype=dtype)
    return element_matrix_matvec(
        local_cell_matrix,
        jnp.asarray(partition.cell_local_dofs),
        local_vector,
    )


def reduce_ghost_contributions(
    topology: OwnedGhostTopology,
    local_contributions: PartitionVectors,
) -> PartitionVectors:
    """Add non-owned row contributions into the unique owner coefficient."""

    _validate_partition_vectors(topology, local_contributions, owned_only=False)
    owned = [
        contribution[: partition.owned_dof_count]
        for partition, contribution in zip(
            topology.partitions,
            local_contributions,
            strict=True,
        )
    ]
    for link in topology.halo_links:
        contribution = local_contributions[link.ghost_partition][
            jnp.asarray(link.ghost_local_indices)
        ]
        owner_values = owned[link.owner_partition]
        owned[link.owner_partition] = owner_values.at[jnp.asarray(link.owner_local_indices)].add(
            contribution
        )
    return tuple(owned)


def assemble_owned_vector(
    topology: OwnedGhostTopology,
    owned_vectors: PartitionVectors,
) -> jax.Array:
    """Reconstruct canonical global order from disjoint owner vectors."""

    _validate_partition_vectors(topology, owned_vectors, owned_only=True)
    result = jnp.zeros((topology.global_dof_count,), dtype=owned_vectors[0].dtype)
    for partition, vector in zip(topology.partitions, owned_vectors, strict=True):
        result = result.at[jnp.asarray(partition.owned_dofs)].set(vector)
    return result


def owned_ghost_matvec(
    cell_matrix: jax.Array,
    topology: OwnedGhostTopology,
    vector: jax.Array,
) -> jax.Array:
    """Apply one global element operator through explicit halo and owner reduction."""

    expected_shape = (topology.cell_count, topology.cell_dof_count, topology.cell_dof_count)
    if cell_matrix.ndim != 3 or cell_matrix.shape != expected_shape:
        raise ValueError("owned/ghost cell matrix does not match the global topology")
    _validate_global_vector(topology, vector)
    local_vectors = exchange_halo_values(
        topology,
        partition_owned_vector(topology, vector),
    )
    local_contributions = tuple(
        local_owned_cell_matvec(
            cell_matrix[jnp.asarray(partition.owned_cells)],
            partition,
            local_vector,
        )
        for partition, local_vector in zip(topology.partitions, local_vectors, strict=True)
    )
    owned_contributions = reduce_ghost_contributions(topology, local_contributions)
    return assemble_owned_vector(topology, owned_contributions)
