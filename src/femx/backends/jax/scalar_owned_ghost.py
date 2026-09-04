"""Owned/ghost matrix-free action for real simplex H1/P1 diffusion operators.

Heat and steady-current conduction share this algebra after their adapters bind conductivity and
units.  The topology is a portable in-process reference.  It does not run a collective, solve a
linear system, or change either public backend's serial capability declaration.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from femx.core.errors import ContractError

from .owned_ghost import (
    OwnedGhostTopology,
    _canonical_int64_array,
    _require_sorted_unique,
    _require_static_positive_integer,
    element_matrix_matvec,
    owned_ghost_matvec,
    prepare_owned_ghost_topology,
)


def _require_unique_simplex_nodes(cells: np.ndarray) -> None:
    for first in range(cells.shape[1]):
        for second in range(first + 1, cells.shape[1]):
            if np.any(cells[:, first] == cells[:, second]):
                raise ContractError("scalar H1 cells cannot repeat a node")


@dataclass(frozen=True, slots=True)
class ScalarH1OwnedGhostTopology:
    """Canonical free-node identity plus a triangle or Tet4 owned/ghost topology."""

    node_count: int
    cells: np.ndarray
    free_nodes: np.ndarray
    full_to_reduced: np.ndarray
    owned_ghost: OwnedGhostTopology

    def __post_init__(self) -> None:
        node_count = _require_static_positive_integer(
            self.node_count,
            label="scalar H1 node count",
        )
        cells = _canonical_int64_array(
            self.cells,
            label="scalar H1 cells",
            rank=2,
        )
        if cells.shape[0] == 0:
            raise ContractError("scalar H1 topology requires at least one cell")
        if cells.shape[1] not in (3, 4):
            raise ContractError(
                "scalar H1 cells must have 3 columns for triangles or 4 columns for tetrahedra"
            )
        if np.any(cells < 0) or np.any(cells >= node_count):
            raise ContractError("scalar H1 cells contain an out-of-range node")
        _require_unique_simplex_nodes(cells)
        free_nodes = _canonical_int64_array(
            self.free_nodes,
            label="scalar H1 free nodes",
            rank=1,
        )
        if free_nodes.size == 0:
            raise ContractError("scalar H1 topology requires at least one free node")
        if np.any(free_nodes < 0) or np.any(free_nodes >= node_count):
            raise ContractError("scalar H1 free nodes contain an out-of-range node")
        _require_sorted_unique(free_nodes, label="scalar H1 free nodes")
        full_to_reduced = _canonical_int64_array(
            self.full_to_reduced,
            label="scalar H1 full-to-reduced map",
            rank=1,
        )
        if full_to_reduced.shape != (node_count,):
            raise ContractError("scalar H1 full-to-reduced map must match the node count")
        if not isinstance(self.owned_ghost, OwnedGhostTopology):
            raise ContractError("scalar H1 topology requires an OwnedGhostTopology")
        if self.owned_ghost.cell_dof_count != cells.shape[1]:
            raise ContractError(
                f"scalar H1 owned/ghost cells must have {cells.shape[1]} local DOFs "
                "to match the simplex cells"
            )
        if self.owned_ghost.cell_count != cells.shape[0]:
            raise ContractError("scalar H1 cells must match the owned/ghost cell count")
        if self.owned_ghost.global_dof_count != free_nodes.shape[0]:
            raise ContractError("scalar H1 free nodes must match the global reduced DOFs")
        expected_map = np.full(node_count, free_nodes.shape[0], dtype=np.int64)
        expected_map[free_nodes] = np.arange(free_nodes.shape[0], dtype=np.int64)
        if not np.array_equal(full_to_reduced, expected_map):
            raise ContractError("scalar H1 full-to-reduced map disagrees with free-node identity")
        if not np.array_equal(self.owned_ghost.cell_reduced_dofs, expected_map[cells]):
            raise ContractError("scalar H1 cells disagree with the owned/ghost reduced map")
        object.__setattr__(self, "node_count", node_count)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "free_nodes", free_nodes)
        object.__setattr__(self, "full_to_reduced", full_to_reduced)

    @property
    def free_dof_count(self) -> int:
        """Return the number of globally identified unconstrained nodal DOFs."""

        return self.owned_ghost.global_dof_count

    @property
    def cell_count(self) -> int:
        """Return the number of globally identified simplex cells."""

        return self.owned_ghost.cell_count

    @property
    def cell_reduced_dofs(self) -> np.ndarray:
        """Return each cell's reduced free-node map and constrained sentinel."""

        return self.owned_ghost.cell_reduced_dofs

    @property
    def cell_dof_count(self) -> int:
        """Return three for triangles or four for tetrahedra."""

        return self.owned_ghost.cell_dof_count

    @property
    def constrained_nodes(self) -> np.ndarray:
        """Return strong-boundary nodes in canonical full-node order."""

        return np.flatnonzero(self.full_to_reduced == self.free_dof_count).astype(np.int64)


def prepare_scalar_h1_owned_ghost_topology(
    cells: object,
    cell_owners: object,
    *,
    node_count: int,
    free_nodes: object,
    partition_count: int,
    dof_owners: object | None = None,
) -> ScalarH1OwnedGhostTopology:
    """Prepare one canonical triangle or Tet4 H1 topology without device discovery."""

    nodes = _require_static_positive_integer(node_count, label="scalar H1 node count")
    canonical_cells = _canonical_int64_array(
        cells,
        label="scalar H1 cells",
        rank=2,
    )
    if canonical_cells.shape[0] == 0:
        raise ContractError("scalar H1 topology requires at least one cell")
    if canonical_cells.shape[1] not in (3, 4):
        raise ContractError(
            "scalar H1 cells must have 3 columns for triangles or 4 columns for tetrahedra"
        )
    if np.any(canonical_cells < 0) or np.any(canonical_cells >= nodes):
        raise ContractError("scalar H1 cells contain an out-of-range node")
    _require_unique_simplex_nodes(canonical_cells)
    canonical_free_nodes = _canonical_int64_array(
        free_nodes,
        label="scalar H1 free nodes",
        rank=1,
    )
    if canonical_free_nodes.size == 0:
        raise ContractError("scalar H1 topology requires at least one free node")
    if np.any(canonical_free_nodes < 0) or np.any(canonical_free_nodes >= nodes):
        raise ContractError("scalar H1 free nodes contain an out-of-range node")
    _require_sorted_unique(canonical_free_nodes, label="scalar H1 free nodes")

    full_to_reduced = np.full(nodes, canonical_free_nodes.shape[0], dtype=np.int64)
    full_to_reduced[canonical_free_nodes] = np.arange(
        canonical_free_nodes.shape[0],
        dtype=np.int64,
    )
    owned_ghost = prepare_owned_ghost_topology(
        full_to_reduced[canonical_cells],
        cell_owners,
        global_dof_count=int(canonical_free_nodes.shape[0]),
        partition_count=partition_count,
        dof_owners=dof_owners,
    )
    return ScalarH1OwnedGhostTopology(
        node_count=nodes,
        cells=canonical_cells,
        free_nodes=canonical_free_nodes,
        full_to_reduced=full_to_reduced,
        owned_ghost=owned_ghost,
    )


def _validate_real_scalar_action(
    cell_stiffness: jax.Array,
    topology: ScalarH1OwnedGhostTopology,
    vector: jax.Array,
) -> None:
    width = topology.cell_dof_count
    if cell_stiffness.ndim != 3 or cell_stiffness.shape != (
        topology.cell_count,
        width,
        width,
    ):
        raise ValueError(
            f"scalar H1 cell stiffness must be shaped (global cells, {width}, {width})"
        )
    if not jnp.issubdtype(cell_stiffness.dtype, jnp.floating):
        raise TypeError("scalar H1 cell stiffness must use a real floating dtype")
    if vector.ndim != 1 or vector.shape != (topology.free_dof_count,):
        raise ValueError("scalar H1 vector must match the global free DOFs")
    if not jnp.issubdtype(vector.dtype, jnp.floating):
        raise TypeError("scalar H1 vector must use a real floating dtype")


def matrix_free_scalar_h1_matvec(
    cell_stiffness: jax.Array,
    topology: ScalarH1OwnedGhostTopology,
    vector: jax.Array,
) -> jax.Array:
    """Apply the serial free-node principal operator from simplex cell matrices."""

    _validate_real_scalar_action(cell_stiffness, topology, vector)
    return element_matrix_matvec(
        cell_stiffness,
        jnp.asarray(topology.cell_reduced_dofs),
        vector,
    )


def owned_ghost_scalar_h1_matvec(
    cell_stiffness: jax.Array,
    topology: ScalarH1OwnedGhostTopology,
    vector: jax.Array,
) -> jax.Array:
    """Apply the same free-node operator through value halo and owner-row reduction."""

    _validate_real_scalar_action(cell_stiffness, topology, vector)
    return owned_ghost_matvec(cell_stiffness, topology.owned_ghost, vector)
