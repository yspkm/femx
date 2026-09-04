r"""Owned/ghost specialization for the six-DOF mixed port operator.

The generic ownership, halo, and contribution-reduction algebra lives in
:mod:`femx.backends.jax.owned_ghost`.  This module preserves the reviewed port API and rejects any
element width other than the three scalar plus three edge coefficients of the mixed triangle.
It remains an in-process algebraic reference and does not claim distributed execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from femx.core.errors import ContractError

from .owned_ghost import (
    HaloLink,
    OwnedGhostPartition,
    OwnedGhostTopology,
    PartitionVectors,
    _canonical_int64_array,
    _require_static_positive_integer,
    assemble_owned_vector,
    exchange_halo_values,
    local_owned_cell_matvec,
    owned_ghost_matvec,
    partition_owned_vector,
    prepare_owned_ghost_topology,
    reduce_ghost_contributions,
)


@dataclass(frozen=True, slots=True)
class PortOwnedGhostPartition(OwnedGhostPartition):
    """One six-DOF mixed-port cell and local-vector partition."""

    def __post_init__(self) -> None:
        OwnedGhostPartition.__post_init__(self)
        if self.cell_dof_count != 6:
            raise ContractError("owned/ghost cell-local DOFs must have 6 columns")


@dataclass(frozen=True, slots=True)
class PortHaloLink(HaloLink):
    """Canonical port owner-to-ghost and reverse-contribution link."""

    def __post_init__(self) -> None:
        HaloLink.__post_init__(self)


@dataclass(frozen=True, slots=True)
class PortOwnedGhostTopology(OwnedGhostTopology):
    """Six-DOF mixed-port topology with canonical global reduced identities."""

    def __post_init__(self) -> None:
        OwnedGhostTopology.__post_init__(self)
        if self.cell_dof_count != 6:
            raise ContractError("owned/ghost cell reduced DOFs must have 6 columns")
        if any(not isinstance(partition, PortOwnedGhostPartition) for partition in self.partitions):
            raise ContractError("port topology requires PortOwnedGhostPartition records")
        if any(not isinstance(link, PortHaloLink) for link in self.halo_links):
            raise ContractError("port topology requires PortHaloLink records")


class PortOwnedGhostResidual(NamedTuple):
    """Partitioned generalized residual and its componentwise magnitude denominator."""

    residual: jax.Array
    row_magnitude_bound: jax.Array
    relative_scaled_residual: jax.Array


PortPartitionVectors = PartitionVectors


def prepare_owned_ghost_port_topology(
    cell_reduced_dofs: object,
    cell_owners: object,
    *,
    free_dof_count: int,
    partition_count: int,
    dof_owners: object | None = None,
) -> PortOwnedGhostTopology:
    """Create a deterministic six-DOF port plan without probing devices or processes."""

    free_count = _require_static_positive_integer(
        free_dof_count,
        label="owned/ghost free DOF count",
    )
    process_count = _require_static_positive_integer(
        partition_count,
        label="owned/ghost partition count",
    )
    cell_map = _canonical_int64_array(
        cell_reduced_dofs,
        label="owned/ghost cell reduced DOFs",
        rank=2,
        columns=6,
    )
    generic = prepare_owned_ghost_topology(
        cell_map,
        cell_owners,
        global_dof_count=free_count,
        partition_count=process_count,
        dof_owners=dof_owners,
    )
    partitions = tuple(
        PortOwnedGhostPartition(
            partition_index=partition.partition_index,
            owned_cells=partition.owned_cells,
            owned_dofs=partition.owned_dofs,
            ghost_dofs=partition.ghost_dofs,
            local_dofs=partition.local_dofs,
            cell_local_dofs=partition.cell_local_dofs,
        )
        for partition in generic.partitions
    )
    links = tuple(
        PortHaloLink(
            owner_partition=link.owner_partition,
            ghost_partition=link.ghost_partition,
            global_dofs=link.global_dofs,
            owner_local_indices=link.owner_local_indices,
            ghost_local_indices=link.ghost_local_indices,
        )
        for link in generic.halo_links
    )
    return PortOwnedGhostTopology(
        partition_count=generic.partition_count,
        global_dof_count=generic.global_dof_count,
        cell_reduced_dofs=generic.cell_reduced_dofs,
        cell_owners=generic.cell_owners,
        dof_owners=generic.dof_owners,
        partitions=partitions,
        halo_links=links,
    )


def partition_owned_port_vector(
    topology: PortOwnedGhostTopology,
    vector: jax.Array,
) -> PortPartitionVectors:
    """Scatter authoritative port values and initialize ghost slots to zero."""

    return partition_owned_vector(topology, vector)


def exchange_port_halo_values(
    topology: PortOwnedGhostTopology,
    local_vectors: PortPartitionVectors,
) -> PortPartitionVectors:
    """Copy owner-authoritative port coefficients into required ghost slots."""

    return exchange_halo_values(topology, local_vectors)


def local_owned_cell_port_matvec(
    local_cell_matrix: jax.Array,
    partition: PortOwnedGhostPartition,
    local_vector: jax.Array,
) -> jax.Array:
    """Apply one partition's six-by-six owned port cells."""

    expected_shape = (partition.owned_cells.shape[0], 6, 6)
    if local_cell_matrix.ndim != 3 or local_cell_matrix.shape != expected_shape:
        raise ValueError("owned/ghost local cell matrix does not match the partition cells")
    return local_owned_cell_matvec(local_cell_matrix, partition, local_vector)


def reduce_port_ghost_contributions(
    topology: PortOwnedGhostTopology,
    local_contributions: PortPartitionVectors,
) -> PortPartitionVectors:
    """Add non-owned port-row contributions into the unique owner coefficient."""

    return reduce_ghost_contributions(topology, local_contributions)


def assemble_port_owned_vector(
    topology: PortOwnedGhostTopology,
    owned_vectors: PortPartitionVectors,
) -> jax.Array:
    """Reconstruct canonical port order from disjoint owner vectors."""

    return assemble_owned_vector(topology, owned_vectors)


def owned_ghost_port_matvec(
    cell_matrix: jax.Array,
    topology: PortOwnedGhostTopology,
    vector: jax.Array,
) -> jax.Array:
    """Apply one global port operator through explicit halo and owner reduction."""

    if cell_matrix.ndim != 3 or cell_matrix.shape != (topology.cell_count, 6, 6):
        raise ValueError("owned/ghost cell matrix must be shaped (global cells, 6, 6)")
    return owned_ghost_matvec(cell_matrix, topology, vector)


def owned_ghost_port_generalized_residual(
    stiffness: jax.Array,
    mass: jax.Array,
    topology: PortOwnedGhostTopology,
    vector: jax.Array,
    eigenvalue: jax.Array,
) -> PortOwnedGhostResidual:
    r"""Evaluate ``A x - lambda B x`` and its absolute cell-contribution denominator."""

    if eigenvalue.ndim != 0:
        raise ValueError("owned/ghost generalized eigenvalue must be a scalar array")
    stiffness_action = owned_ghost_port_matvec(stiffness, topology, vector)
    mass_action = owned_ghost_port_matvec(mass, topology, vector)
    residual = stiffness_action - eigenvalue * mass_action
    magnitude_matrix = jnp.abs(stiffness) + jnp.abs(eigenvalue) * jnp.abs(mass)
    row_magnitude_bound = owned_ghost_port_matvec(magnitude_matrix, topology, jnp.abs(vector))
    numerator = jnp.linalg.norm(residual)
    denominator = jnp.linalg.norm(row_magnitude_bound)
    relative = jnp.where(
        denominator > 0.0,
        numerator / denominator,
        jnp.where(numerator == 0.0, 0.0, jnp.inf),
    )
    return PortOwnedGhostResidual(
        residual=residual,
        row_magnitude_bound=row_magnitude_bound,
        relative_scaled_residual=relative,
    )
