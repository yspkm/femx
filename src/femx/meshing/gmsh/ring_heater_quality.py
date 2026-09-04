"""Fail-closed geometric and Tet4 quality report for the public M5 ring heater."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from femx.core.errors import ContractError
from femx.mesh import CellType
from femx.meshing.gmsh.importer import ImportedGmshMesh
from femx.meshing.gmsh.ring_heater import PublicRingHeater3D

_LOCAL_TETRAHEDRON_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_LOCAL_TETRAHEDRON_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))


@dataclass(frozen=True, slots=True)
class PublicRingHeaterMeshReport:
    """Canonical geometry, partition, and shape evidence for one imported mesh."""

    recipe_sha256: str
    import_record_sha256: str
    node_count: int
    tetrahedron_count: int
    boundary_triangle_count: int
    total_volume_m3: float
    full_domain_relative_volume_error: float
    minimum_cell_volume_m3: float
    maximum_cell_volume_m3: float
    minimum_edge_length_m: float
    maximum_edge_length_m: float
    minimum_mean_ratio: float
    percentile_1_mean_ratio: float
    median_mean_ratio: float
    maximum_region_volume_relative_error: float
    region_cell_counts: tuple[tuple[str, int], ...]
    region_volumes_m3: tuple[tuple[str, float], ...]
    region_volume_relative_errors: tuple[tuple[str, float], ...]
    electrical_interface_triangle_counts: tuple[tuple[str, int], ...]
    surface_triangle_counts: tuple[tuple[str, int], ...]
    schema_version: str = "femx.public-ring-heater-mesh-report/v1"

    def canonical_data(self) -> dict[str, object]:
        """Return stable JSON-compatible report content."""

        return {
            "schema_version": self.schema_version,
            "recipe_sha256": self.recipe_sha256,
            "import_record_sha256": self.import_record_sha256,
            "node_count": self.node_count,
            "tetrahedron_count": self.tetrahedron_count,
            "boundary_triangle_count": self.boundary_triangle_count,
            "total_volume_m3": self.total_volume_m3,
            "full_domain_relative_volume_error": self.full_domain_relative_volume_error,
            "minimum_cell_volume_m3": self.minimum_cell_volume_m3,
            "maximum_cell_volume_m3": self.maximum_cell_volume_m3,
            "minimum_edge_length_m": self.minimum_edge_length_m,
            "maximum_edge_length_m": self.maximum_edge_length_m,
            "minimum_mean_ratio": self.minimum_mean_ratio,
            "percentile_1_mean_ratio": self.percentile_1_mean_ratio,
            "median_mean_ratio": self.median_mean_ratio,
            "maximum_region_volume_relative_error": (self.maximum_region_volume_relative_error),
            "region_cell_counts": dict(self.region_cell_counts),
            "region_volumes_m3": dict(self.region_volumes_m3),
            "region_volume_relative_errors": dict(self.region_volume_relative_errors),
            "electrical_interface_triangle_counts": dict(self.electrical_interface_triangle_counts),
            "surface_triangle_counts": dict(self.surface_triangle_counts),
        }

    def digest(self) -> str:
        """Hash the complete report."""

        payload = json.dumps(
            self.canonical_data(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evaluate_public_ring_heater_mesh(
    imported: ImportedGmshMesh,
    recipe: PublicRingHeater3D,
) -> PublicRingHeaterMeshReport:
    """Validate one canonical mesh and calculate deterministic geometry/shape metrics."""

    mesh = imported.mesh
    if mesh.topology.cell_type is not CellType.TETRAHEDRON:
        raise ContractError("public ring-heater quality requires Tet4 volume cells")
    if mesh.boundary_facets is None or mesh.boundary_facets.cell_type is not CellType.TRIANGLE:
        raise ContractError("public ring-heater quality requires triangular boundary facets")
    if mesh.geometry.spatial_dimension != 3:
        raise ContractError("public ring-heater quality requires three-dimensional coordinates")

    expected_names = set((*recipe.VOLUME_GROUPS, *recipe.SURFACE_GROUPS))
    tags_by_name = {tag.name: tag for tag in mesh.tags}
    if set(tags_by_name) != expected_names:
        missing = sorted(expected_names - set(tags_by_name))
        unexpected = sorted(set(tags_by_name) - expected_names)
        raise ContractError(
            "public ring-heater physical groups do not match the recipe: "
            f"missing={missing}, unexpected={unexpected}"
        )

    cell_count = mesh.topology.cell_count
    boundary_count = mesh.boundary_facets.cell_count
    volume_membership = np.zeros(cell_count, dtype=np.int16)
    region_cell_counts: list[tuple[str, int]] = []
    for name in recipe.VOLUME_GROUPS:
        tag = tags_by_name[name]
        if tag.dimension != 3:
            raise ContractError(f"public ring-heater volume group {name!r} must have dimension 3")
        ids = np.asarray(tag.entity_ids, dtype=np.int64)
        if ids.size == 0:
            raise ContractError(f"public ring-heater volume group {name!r} must be non-empty")
        if np.any(ids >= cell_count):
            raise ContractError(f"public ring-heater volume group {name!r} has an invalid cell id")
        volume_membership[ids] += 1
        region_cell_counts.append((name, int(ids.size)))
    if np.any(volume_membership != 1):
        raise ContractError(
            "public ring-heater volume groups must partition every Tet4 exactly once"
        )

    external = tags_by_name["external_boundary"]
    if external.dimension != 2 or set(external.entity_ids) != set(range(boundary_count)):
        raise ContractError("external_boundary must contain every boundary triangle exactly once")
    surface_membership = np.zeros(boundary_count, dtype=np.int16)
    surface_triangle_counts: list[tuple[str, int]] = []
    for name in recipe.SURFACE_GROUPS[1:]:
        tag = tags_by_name[name]
        if tag.dimension != 2:
            raise ContractError(f"public ring-heater surface group {name!r} must have dimension 2")
        ids = np.asarray(tag.entity_ids, dtype=np.int64)
        if ids.size == 0:
            raise ContractError(f"public ring-heater surface group {name!r} must be non-empty")
        if np.any(ids >= boundary_count):
            raise ContractError(
                f"public ring-heater surface group {name!r} has an invalid boundary id"
            )
        surface_membership[ids] += 1
        surface_triangle_counts.append((name, int(ids.size)))
    if np.any(surface_membership != 1):
        raise ContractError(
            "public ring-heater boundary-condition groups must partition the external boundary"
        )
    surface_triangle_counts.insert(0, ("external_boundary", boundary_count))

    coordinates = np.asarray(mesh.geometry.coordinates, dtype=np.float64)
    cells = np.asarray(mesh.topology.connectivity, dtype=np.int64)
    if not np.all(np.isfinite(coordinates)):
        raise ContractError("public ring-heater Tet4 volumes must be finite and positive")
    if np.any(cells < 0) or np.any(cells >= len(coordinates)):
        raise ContractError("public ring-heater Tet4 connectivity contains an invalid node id")
    points = coordinates[cells]
    jacobians = np.stack(
        (
            points[:, 1] - points[:, 0],
            points[:, 2] - points[:, 0],
            points[:, 3] - points[:, 0],
        ),
        axis=2,
    )
    determinants = np.linalg.det(jacobians)
    cell_volumes = determinants / 6.0
    if not np.all(np.isfinite(cell_volumes)) or np.any(cell_volumes <= 0.0):
        raise ContractError("public ring-heater Tet4 volumes must be finite and positive")

    heater_ids = np.asarray(tags_by_name["tin_heater"].entity_ids, dtype=np.int64)
    heater_faces = _canonical_face_keys(cells[heater_ids])
    electrical_interfaces: list[tuple[str, int]] = []
    for contact_name in ("al_contact_negative", "al_contact_positive"):
        contact_ids = np.asarray(tags_by_name[contact_name].entity_ids, dtype=np.int64)
        contact_faces = _canonical_face_keys(cells[contact_ids])
        shared_count = int(np.intersect1d(heater_faces, contact_faces).size)
        if shared_count == 0:
            raise ContractError(
                f"public ring-heater {contact_name!r} must share a conformal face with tin_heater"
            )
        electrical_interfaces.append((f"{contact_name}:tin_heater", shared_count))

    squared_edge_lengths = np.empty((cell_count, len(_LOCAL_TETRAHEDRON_EDGES)), dtype=np.float64)
    for edge_index, (left, right) in enumerate(_LOCAL_TETRAHEDRON_EDGES):
        delta = points[:, right] - points[:, left]
        squared_edge_lengths[:, edge_index] = np.einsum("ij,ij->i", delta, delta)
    mean_ratios = (
        12.0 * np.power(3.0 * cell_volumes, 2.0 / 3.0) / np.sum(squared_edge_lengths, axis=1)
    )
    if (
        not np.all(np.isfinite(mean_ratios))
        or np.any(mean_ratios <= 0.0)
        or np.any(mean_ratios > 1.0 + 1.0e-12)
    ):
        raise ContractError("public ring-heater Tet4 mean-ratio quality is outside (0, 1]")

    expected_volumes = dict(recipe.expected_region_volumes_m3())
    region_volumes: list[tuple[str, float]] = []
    region_errors: list[tuple[str, float]] = []
    for name in recipe.VOLUME_GROUPS:
        ids = np.asarray(tags_by_name[name].entity_ids, dtype=np.int64)
        volume = float(np.sum(cell_volumes[ids], dtype=np.float64))
        error = abs(volume - expected_volumes[name]) / expected_volumes[name]
        region_volumes.append((name, volume))
        region_errors.append((name, error))

    total_volume = float(np.sum(cell_volumes, dtype=np.float64))
    expected_total = sum(expected_volumes.values())
    total_error = abs(total_volume - expected_total) / expected_total
    scalar_metrics = (
        total_volume,
        total_error,
        float(np.min(cell_volumes)),
        float(np.max(cell_volumes)),
        math.sqrt(float(np.min(squared_edge_lengths))),
        math.sqrt(float(np.max(squared_edge_lengths))),
        float(np.min(mean_ratios)),
        float(np.quantile(mean_ratios, 0.01)),
        float(np.median(mean_ratios)),
        max(error for _, error in region_errors),
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in scalar_metrics):
        raise ContractError("public ring-heater mesh report contains a non-finite metric")

    return PublicRingHeaterMeshReport(
        recipe_sha256=recipe.digest(),
        import_record_sha256=imported.record.digest(),
        node_count=mesh.geometry.node_count,
        tetrahedron_count=cell_count,
        boundary_triangle_count=boundary_count,
        total_volume_m3=total_volume,
        full_domain_relative_volume_error=total_error,
        minimum_cell_volume_m3=float(np.min(cell_volumes)),
        maximum_cell_volume_m3=float(np.max(cell_volumes)),
        minimum_edge_length_m=math.sqrt(float(np.min(squared_edge_lengths))),
        maximum_edge_length_m=math.sqrt(float(np.max(squared_edge_lengths))),
        minimum_mean_ratio=float(np.min(mean_ratios)),
        percentile_1_mean_ratio=float(np.quantile(mean_ratios, 0.01)),
        median_mean_ratio=float(np.median(mean_ratios)),
        maximum_region_volume_relative_error=max(error for _, error in region_errors),
        region_cell_counts=tuple(region_cell_counts),
        region_volumes_m3=tuple(region_volumes),
        region_volume_relative_errors=tuple(region_errors),
        electrical_interface_triangle_counts=tuple(electrical_interfaces),
        surface_triangle_counts=tuple(surface_triangle_counts),
    )


def _canonical_face_keys(cells: np.ndarray) -> np.ndarray:
    faces = np.sort(
        cells[:, np.asarray(_LOCAL_TETRAHEDRON_FACES, dtype=np.int64)].reshape(-1, 3),
        axis=1,
    )
    canonical = np.ascontiguousarray(faces, dtype="<i8")
    return canonical.view(np.dtype((np.void, canonical.dtype.itemsize * 3))).reshape(-1)
