"""Shared exact edge-DOF topology for conforming triangle ``H(curl)`` backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from femx.core.errors import ContractError

TRIANGLE_LOCAL_EDGES: Final = ((0, 1), (1, 2), (2, 0))


def _readonly(value: np.ndarray, *, dtype: np.dtype[np.generic]) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class CanonicalTriangleEdgeMap:
    """Lexicographic global edges and cell-local edge-DOF assembly metadata.

    ``edge_nodes`` always points from the smaller global node id to the larger one.  The local
    triangle convention is ``(0, 1)``, ``(1, 2)``, ``(2, 0)``.  ``cell_edge_signs`` therefore
    converts a basis following that local direction into the canonical global direction.
    """

    edge_nodes: np.ndarray
    cell_edge_dofs: np.ndarray
    cell_edge_signs: np.ndarray

    @property
    def dof_count(self) -> int:
        """Return the number of globally conforming first-order edge DOFs."""

        return int(self.edge_nodes.shape[0])


@dataclass(frozen=True, slots=True)
class CanonicalMixedPortDofPartition:
    """PEC partition for femx's nodal-first, edge-second mixed port ordering."""

    scalar_dofs: np.ndarray
    edge_dofs: np.ndarray
    constrained_dofs: np.ndarray
    free_dofs: np.ndarray


def canonical_triangle_edge_map(
    cells: object,
    edge_signs: object | None,
) -> CanonicalTriangleEdgeMap:
    """Validate and enumerate one canonical first-order triangle edge space.

    The routine belongs to backend preparation rather than a JAX transform: exact integer dtype,
    connectivity, and caller-provided orientation are checked before any compiled numerical kernel
    is entered.  Global edge ids are deterministic lexicographic ranks of sorted node pairs.
    """

    raw_cells = np.asarray(cells)
    if raw_cells.dtype.kind not in "iu":
        raise ContractError("triangle edge connectivity must use an integer dtype")
    if raw_cells.ndim != 2 or raw_cells.shape[1] != 3:
        raise ContractError("triangle edge connectivity must be shaped (cells, 3)")
    if raw_cells.shape[0] == 0:
        raise ContractError("triangle edge connectivity must contain at least one cell")
    cell_nodes = np.asarray(raw_cells, dtype=np.int64)
    if np.any(cell_nodes < 0):
        raise ContractError("triangle edge connectivity contains a negative node id")
    if np.any(
        (cell_nodes[:, 0] == cell_nodes[:, 1])
        | (cell_nodes[:, 1] == cell_nodes[:, 2])
        | (cell_nodes[:, 2] == cell_nodes[:, 0])
    ):
        raise ContractError("triangle edge connectivity contains a repeated node")

    if edge_signs is None:
        raise ContractError("triangle H(curl) requires explicit cell-local edge orientations")
    raw_signs = np.asarray(edge_signs)
    if raw_signs.dtype.kind not in "iu":
        raise ContractError("triangle edge orientations must use an integer dtype")
    if raw_signs.shape != cell_nodes.shape:
        raise ContractError("triangle edge orientations must be shaped (cells, 3)")

    local_edges = cell_nodes[:, TRIANGLE_LOCAL_EDGES]
    expected_signs = np.where(local_edges[:, :, 0] < local_edges[:, :, 1], 1, -1)
    if not np.array_equal(raw_signs, expected_signs):
        raise ContractError(
            "triangle edge orientations disagree with canonical global edge ordering"
        )

    canonical_pairs = np.sort(local_edges, axis=2)
    edge_nodes, inverse = np.unique(
        canonical_pairs.reshape(-1, 2),
        axis=0,
        return_inverse=True,
    )
    return CanonicalTriangleEdgeMap(
        edge_nodes=_readonly(edge_nodes, dtype=np.dtype(np.int64)),
        cell_edge_dofs=_readonly(
            inverse.reshape(cell_nodes.shape[0], 3),
            dtype=np.dtype(np.int64),
        ),
        cell_edge_signs=_readonly(expected_signs, dtype=np.dtype(np.int8)),
    )


def canonical_mixed_port_dof_partition(
    boundary_facets: object,
    edge_map: CanonicalTriangleEdgeMap,
    *,
    node_count: int,
) -> CanonicalMixedPortDofPartition:
    """Map the complete topological boundary to scalar and tangential PEC DOFs.

    The canonical mixed-vector ordering is all nodal scalar DOFs followed by all edge DOFs.
    Boundary facets must equal the complete set of topological boundary edges implied by
    ``edge_map``; callers cannot silently constrain an interior edge or omit a PEC edge.
    """

    if node_count <= 0:
        raise ContractError("mixed port node_count must be positive")
    raw_facets = np.asarray(boundary_facets)
    if raw_facets.dtype.kind not in "iu":
        raise ContractError("mixed port boundary facets must use an integer dtype")
    if raw_facets.ndim != 2 or raw_facets.shape[1] != 2 or raw_facets.shape[0] == 0:
        raise ContractError("mixed port boundary facets must be shaped (facets, 2) and nonempty")
    facets = np.asarray(raw_facets, dtype=np.int64)
    if np.any(facets < 0) or np.any(facets >= node_count):
        raise ContractError("mixed port boundary facets contain an out-of-range node")
    if np.any(facets[:, 0] == facets[:, 1]):
        raise ContractError("mixed port boundary facets contain a repeated node")

    canonical_facets = np.sort(facets, axis=1)
    unique_facets = np.unique(canonical_facets, axis=0)
    if unique_facets.shape[0] != canonical_facets.shape[0]:
        raise ContractError("mixed port boundary facets contain duplicates")

    edge_lookup = {tuple(pair.tolist()): index for index, pair in enumerate(edge_map.edge_nodes)}
    try:
        boundary_edge_dofs = np.asarray(
            [edge_lookup[tuple(pair.tolist())] for pair in canonical_facets],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ContractError("mixed port boundary facet is not a triangle-mesh edge") from error

    incidence = np.bincount(
        edge_map.cell_edge_dofs.reshape(-1),
        minlength=edge_map.dof_count,
    )
    expected_boundary_edge_dofs = np.flatnonzero(incidence == 1)
    if not np.array_equal(np.sort(boundary_edge_dofs), expected_boundary_edge_dofs):
        raise ContractError("mixed port boundary facets do not equal the topological boundary")

    scalar_dofs = np.unique(facets)
    edge_dofs = np.sort(boundary_edge_dofs)
    constrained_dofs = np.concatenate((scalar_dofs, node_count + edge_dofs))
    total_dofs = node_count + edge_map.dof_count
    free_mask = np.ones(total_dofs, dtype=np.bool_)
    free_mask[constrained_dofs] = False
    free_dofs = np.flatnonzero(free_mask)
    if free_dofs.size == 0:
        raise ContractError("mixed port PEC constraints leave no free DOFs")
    return CanonicalMixedPortDofPartition(
        scalar_dofs=_readonly(scalar_dofs, dtype=np.dtype(np.int64)),
        edge_dofs=_readonly(edge_dofs, dtype=np.dtype(np.int64)),
        constrained_dofs=_readonly(constrained_dofs, dtype=np.dtype(np.int64)),
        free_dofs=_readonly(free_dofs, dtype=np.dtype(np.int64)),
    )
