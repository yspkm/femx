"""Deterministic native Elmer mesh lowering for first-order tetrahedra."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from femx.backends.elmer.case import _complete_tag_partition, _format_real
from femx.core.errors import ContractError
from femx.mesh import CellType, Mesh

_TETRAHEDRON_LOCAL_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))


def _triangle_key(nodes: Sequence[int]) -> tuple[int, int, int]:
    first, second, third = sorted(int(node) for node in nodes)
    return first, second, third


@dataclass(frozen=True, slots=True)
class ElmerTet4MeshDeck:
    """Complete serial Elmer native mesh for one tagged Tet4 volume."""

    header: str
    nodes: str
    elements: str
    boundary: str
    node_count: int
    element_count: int
    boundary_count: int
    body_ids: tuple[int, ...]
    boundary_ids: tuple[int, ...]
    body_node_ids: tuple[tuple[int, ...], ...]
    boundary_node_ids: tuple[tuple[int, ...], ...]

    def digest(self) -> str:
        """Hash the exact native files and declared semantic ids."""

        digest = hashlib.sha256()
        for label, content in (
            ("mesh.header", self.header),
            ("mesh.nodes", self.nodes),
            ("mesh.elements", self.elements),
            ("mesh.boundary", self.boundary),
        ):
            digest.update(label.encode("ascii"))
            digest.update(b"\0")
            digest.update(content.encode("utf-8"))
            digest.update(b"\0")
        digest.update(repr(self.body_ids).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(self.boundary_ids).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(self.body_node_ids).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(self.boundary_node_ids).encode("ascii"))
        return digest.hexdigest()


def _normalize_tetrahedra(coordinates: np.ndarray, cells: np.ndarray) -> np.ndarray:
    normalized = np.asarray(cells, dtype=np.int64).copy()
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ContractError("Elmer Tet4 coordinates must have shape (nodes, 3)")
    if normalized.ndim != 2 or normalized.shape[1] != 4:
        raise ContractError("Elmer Tet4 connectivity must have shape (cells, 4)")
    if not np.all(np.isfinite(coordinates)):
        raise ContractError("Elmer Tet4 coordinates must be finite")
    if normalized.size == 0:
        raise ContractError("Elmer Tet4 mesh must contain at least one cell")
    if np.any(normalized < 0) or np.any(normalized >= coordinates.shape[0]):
        raise ContractError("Elmer Tet4 connectivity contains an out-of-range node")
    if any(len(set(int(node) for node in cell)) != 4 for cell in normalized):
        raise ContractError("Elmer Tet4 connectivity contains a repeated cell node")

    points = coordinates[normalized]
    jacobians = np.stack(
        (
            points[:, 1, :] - points[:, 0, :],
            points[:, 2, :] - points[:, 0, :],
            points[:, 3, :] - points[:, 0, :],
        ),
        axis=2,
    )
    determinants = np.linalg.det(jacobians)
    if not np.all(np.isfinite(determinants)) or np.any(determinants == 0.0):
        raise ContractError("Elmer Tet4 mesh contains a non-finite or degenerate cell")
    inverted = determinants < 0.0
    normalized[inverted, 1], normalized[inverted, 2] = (
        normalized[inverted, 2].copy(),
        normalized[inverted, 1].copy(),
    )
    return normalized


def _external_faces(
    cells: np.ndarray,
) -> tuple[dict[tuple[int, int, int], tuple[int, tuple[int, int, int]]], set[tuple[int, int, int]]]:
    occurrences: dict[tuple[int, int, int], list[tuple[int, tuple[int, int, int]]]] = {}
    for element_id, cell in enumerate(cells, start=1):
        for local_face in _TETRAHEDRON_LOCAL_FACES:
            oriented = (
                int(cell[local_face[0]]),
                int(cell[local_face[1]]),
                int(cell[local_face[2]]),
            )
            key = _triangle_key(oriented)
            occurrences.setdefault(key, []).append((element_id, oriented))
    if any(len(parents) > 2 for parents in occurrences.values()):
        raise ContractError("Elmer Tet4 mesh contains a non-manifold face")
    external = {key for key, parents in occurrences.items() if len(parents) == 1}
    parents = {key: occurrences[key][0] for key in external}
    return parents, external


def lower_tagged_tet4_mesh(
    mesh: Mesh,
    *,
    region_tags: tuple[str, ...],
    boundary_tags: tuple[str, ...],
) -> ElmerTet4MeshDeck:
    """Lower complete body and external-boundary tag partitions to native Elmer files."""

    if mesh.geometry.spatial_dimension != 3 or mesh.topology.cell_type is not CellType.TETRAHEDRON:
        raise ContractError("Elmer tagged Tet4 lowering requires a 3D tetrahedron mesh")
    if mesh.boundary_facets is None or mesh.boundary_facets.cell_type is not CellType.TRIANGLE:
        raise ContractError("Elmer tagged Tet4 lowering requires triangular boundary facets")

    region_cells = _complete_tag_partition(
        mesh,
        names=region_tags,
        dimension=3,
        entity_count=mesh.topology.cell_count,
        label="Tet4 region",
    )
    boundary_groups = _complete_tag_partition(
        mesh,
        names=boundary_tags,
        dimension=2,
        entity_count=mesh.boundary_facets.cell_count,
        label="Tet4 boundary",
    )
    coordinates = np.asarray(mesh.geometry.coordinates, dtype=np.float64)
    cells = _normalize_tetrahedra(coordinates, np.asarray(mesh.topology.connectivity))
    facets = np.asarray(mesh.boundary_facets.connectivity, dtype=np.int64)
    if facets.ndim != 2 or facets.shape[1] != 3:
        raise ContractError("Elmer Tet4 boundary connectivity must have shape (facets, 3)")
    if np.any(facets < 0) or np.any(facets >= coordinates.shape[0]):
        raise ContractError("Elmer Tet4 boundary contains an out-of-range node")
    if any(len(set(int(node) for node in facet)) != 3 for facet in facets):
        raise ContractError("Elmer Tet4 boundary contains a repeated facet node")

    parent_by_face, external_faces = _external_faces(cells)
    facet_keys = tuple(_triangle_key(facet) for facet in facets)
    if len(facet_keys) != len(set(facet_keys)):
        raise ContractError("Elmer Tet4 boundary contains duplicate faces")
    if set(facet_keys) != external_faces:
        raise ContractError("Elmer Tet4 boundary must equal the complete external cell boundary")

    body_assignment = np.zeros((cells.shape[0],), dtype=np.int64)
    for body_id, cell_ids in enumerate(region_cells, start=1):
        body_assignment[cell_ids] = body_id
    boundary_assignment = np.zeros((facets.shape[0],), dtype=np.int64)
    for boundary_id, facet_ids in enumerate(boundary_groups, start=1):
        boundary_assignment[facet_ids] = boundary_id

    node_lines = [
        f"{node_id} -1 {_format_real(float(point[0]))} "
        f"{_format_real(float(point[1]))} {_format_real(float(point[2]))}"
        for node_id, point in enumerate(coordinates, start=1)
    ]
    element_lines = [
        f"{element_id} {int(body_assignment[element_id - 1])} 504 "
        + " ".join(str(int(node) + 1) for node in cell)
        for element_id, cell in enumerate(cells, start=1)
    ]
    boundary_lines: list[str] = []
    for facet_id, key in enumerate(facet_keys, start=1):
        parent_id, oriented = parent_by_face[key]
        boundary_lines.append(
            f"{facet_id} {int(boundary_assignment[facet_id - 1])} {parent_id} 0 303 "
            + " ".join(str(node + 1) for node in oriented)
        )

    header = (
        f"{coordinates.shape[0]} {cells.shape[0]} {facets.shape[0]}\n"
        "2\n"
        f"303 {facets.shape[0]}\n"
        f"504 {cells.shape[0]}\n"
    )
    return ElmerTet4MeshDeck(
        header=header,
        nodes="\n".join(node_lines) + "\n",
        elements="\n".join(element_lines) + "\n",
        boundary="\n".join(boundary_lines) + "\n",
        node_count=coordinates.shape[0],
        element_count=cells.shape[0],
        boundary_count=facets.shape[0],
        body_ids=tuple(range(1, len(region_tags) + 1)),
        boundary_ids=tuple(range(1, len(boundary_tags) + 1)),
        body_node_ids=tuple(
            tuple(int(node) for node in np.unique(cells[cell_ids])) for cell_ids in region_cells
        ),
        boundary_node_ids=tuple(
            tuple(int(node) for node in np.unique(facets[facet_ids]))
            for facet_ids in boundary_groups
        ),
    )
