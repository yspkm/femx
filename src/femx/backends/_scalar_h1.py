"""Shared geometric validation for the initial two-dimensional scalar H1 backends."""

from __future__ import annotations

import numpy as np

from femx.core.errors import ContractError
from femx.mesh import CellType, Mesh


def integer_array(value: object, *, label: str) -> np.ndarray:
    """Return one exact integer connectivity array without accepting float coercion."""

    raw = np.asarray(value)
    if raw.dtype.kind not in "iu":
        raise ContractError(f"{label} must use an integer dtype")
    return np.asarray(raw, dtype=np.int64)


def validate_planar_triangle_mesh(
    mesh: Mesh,
    *,
    physics_label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate exact float64 planar first-order triangles and their external boundary."""

    if mesh.geometry.spatial_dimension != 2:
        raise ContractError(f"{physics_label} requires two-dimensional coordinates")
    if mesh.topology.cell_type is not CellType.TRIANGLE:
        raise ContractError(f"{physics_label} currently requires P1 triangle cells")
    if mesh.boundary_facets is None:
        raise ContractError(f"{physics_label} requires explicit boundary facets")
    raw_coordinates = np.asarray(mesh.geometry.coordinates)
    if raw_coordinates.dtype != np.dtype(np.float64):
        raise ContractError(f"{physics_label} mesh coordinates must use the exact float64 dtype")
    coordinates = raw_coordinates
    cells = integer_array(mesh.topology.connectivity, label="cell connectivity")
    facets = integer_array(mesh.boundary_facets.connectivity, label="boundary connectivity")
    if not np.isfinite(coordinates).all():
        raise ContractError("mesh coordinates must be finite")
    for label, connectivity in (("cell", cells), ("boundary", facets)):
        if np.any(connectivity < 0) or np.any(connectivity >= coordinates.shape[0]):
            raise ContractError(f"{label} connectivity contains an out-of-range node")
        if any(len(set(row.tolist())) != row.size for row in connectivity):
            raise ContractError(f"{label} connectivity contains a repeated node")

    points = coordinates[cells]
    first = points[:, 1, :] - points[:, 0, :]
    second = points[:, 2, :] - points[:, 0, :]
    determinant = first[:, 0] * second[:, 1] - second[:, 0] * first[:, 1]
    edge_scale_squared = np.maximum(
        np.max(
            np.stack(
                (
                    np.sum(first * first, axis=1),
                    np.sum(second * second, axis=1),
                    np.sum((points[:, 2, :] - points[:, 1, :]) ** 2, axis=1),
                ),
                axis=1,
            ),
            axis=1,
        ),
        np.finfo(np.float64).tiny,
    )
    if np.any(np.abs(determinant) / edge_scale_squared <= 64.0 * np.finfo(np.float64).eps):
        raise ContractError("mesh contains a degenerate or numerically singular triangle")

    edge_counts: dict[tuple[int, int], int] = {}
    for cell in cells:
        for left, right in ((cell[0], cell[1]), (cell[1], cell[2]), (cell[2], cell[0])):
            left_node, right_node = sorted((int(left), int(right)))
            edge = (left_node, right_node)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    if any(count > 2 for count in edge_counts.values()):
        raise ContractError("triangle connectivity contains a non-manifold edge")
    expected_boundary = {edge for edge, count in edge_counts.items() if count == 1}
    actual_boundary = {tuple(sorted((int(facet[0]), int(facet[1])))) for facet in facets}
    if len(actual_boundary) != facets.shape[0]:
        raise ContractError("boundary connectivity contains duplicate facets")
    if actual_boundary != expected_boundary:
        raise ContractError("explicit boundary facets do not equal the triangle-mesh boundary")
    return coordinates, cells, facets


def validate_scalar_h1_mesh(
    mesh: Mesh,
    *,
    physics_label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible scalar-H1 name for the shared planar triangle validator."""

    return validate_planar_triangle_mesh(mesh, physics_label=physics_label)


def tag_ids(mesh: Mesh, name: str, *, dimension: int, upper_bound: int) -> np.ndarray:
    """Resolve and bounds-check one semantic entity tag."""

    tag = mesh.tag(name)
    if tag.dimension != dimension:
        raise ContractError(
            f"mesh tag {name!r} has dimension {tag.dimension}, expected {dimension}"
        )
    ids = np.asarray(tag.entity_ids, dtype=np.int64)
    if ids.size == 0:
        raise ContractError(f"mesh tag {name!r} cannot be empty")
    if np.any(ids < 0) or np.any(ids >= upper_bound):
        raise ContractError(f"mesh tag {name!r} contains an out-of-range entity id")
    return ids
