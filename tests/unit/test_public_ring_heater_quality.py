from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from femx.core.errors import ContractError
from femx.mesh import CellType, EntityTag, Mesh, MeshGeometry, MeshTopology
from femx.meshing.gmsh import (
    PublicRingHeater3D,
    evaluate_public_ring_heater_mesh,
    ring_heater_mesh_profile,
)
from femx.meshing.gmsh import ring_heater_quality as quality_module

pytestmark = pytest.mark.unit

_OUTWARD_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))


@dataclass(frozen=True)
class _Record:
    def digest(self) -> str:
        return "a" * 64


@dataclass(frozen=True)
class _Imported:
    mesh: Mesh
    record: _Record = _Record()


def _valid_mesh() -> Mesh:
    reference = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    coordinates = np.vstack(
        (
            *(reference + np.asarray((4.0 + 3.0 * cell, 0.0, 0.0)) for cell in range(5)),
            reference,
            np.asarray(((1.0, 1.0, 1.0), (-1.0, 0.0, 0.0))),
        )
    )
    raw_cells = [
        *(tuple(range(4 * cell, 4 * cell + 4)) for cell in range(5)),
        (20, 21, 22, 23),
        (21, 22, 23, 24),
        (20, 22, 23, 25),
    ]
    oriented_cells = []
    for raw_cell in raw_cells:
        cell = list(raw_cell)
        points = coordinates[cell]
        jacobian = np.stack(
            (points[1] - points[0], points[2] - points[0], points[3] - points[0]),
            axis=1,
        )
        if np.linalg.det(jacobian) < 0.0:
            cell[1], cell[2] = cell[2], cell[1]
        oriented_cells.append(tuple(cell))
    cells = np.asarray(oriented_cells, dtype=np.int64)
    face_counts: dict[tuple[int, int, int], int] = {}
    for cell in cells:
        for face in _OUTWARD_FACES:
            key = tuple(sorted(int(cell[index]) for index in face))
            face_counts[key] = face_counts.get(key, 0) + 1
    facets = np.asarray(
        tuple(face for face, count in sorted(face_counts.items()) if count == 1),
        dtype=np.int64,
    )
    recipe = PublicRingHeater3D(ring_heater_mesh_profile("coarse"))
    volume_tags = tuple(
        EntityTag(name, 3, (cell,)) for cell, name in enumerate(recipe.VOLUME_GROUPS)
    )
    surface_tags = (
        EntityTag("external_boundary", 2, tuple(range(28))),
        EntityTag("bottom_temperature", 2, tuple(range(5))),
        EntityTag("top_convection", 2, tuple(range(5, 10))),
        EntityTag("lateral_adiabatic", 2, tuple(range(10, 24))),
        EntityTag("terminal_negative", 2, (24, 25)),
        EntityTag("terminal_positive", 2, (26, 27)),
    )
    return Mesh(
        geometry=MeshGeometry(coordinates),
        topology=MeshTopology(cells, CellType.TETRAHEDRON, len(coordinates)),
        tags=volume_tags + surface_tags,
        boundary_facets=MeshTopology(facets, CellType.TRIANGLE, len(coordinates)),
    )


def _replace_tag(mesh: Mesh, name: str, replacement: EntityTag) -> Mesh:
    return replace(
        mesh,
        tags=tuple(replacement if tag.name == name else tag for tag in mesh.tags),
    )


def _evaluate(mesh: Mesh):
    recipe = PublicRingHeater3D(ring_heater_mesh_profile("coarse"))
    return evaluate_public_ring_heater_mesh(_Imported(mesh), recipe)  # type: ignore[arg-type]


def test_ring_heater_quality_report_is_complete_and_content_addressed() -> None:
    report = _evaluate(_valid_mesh())
    canonical = report.canonical_data()

    assert report.recipe_sha256 == PublicRingHeater3D(ring_heater_mesh_profile("coarse")).digest()
    assert report.import_record_sha256 == "a" * 64
    assert report.node_count == 26
    assert report.tetrahedron_count == 8
    assert report.boundary_triangle_count == 28
    assert report.total_volume_m3 == pytest.approx(1.5)
    assert report.minimum_mean_ratio > 0.0
    assert report.minimum_mean_ratio <= report.percentile_1_mean_ratio <= report.median_mean_ratio
    assert dict(report.region_cell_counts) == {name: 1 for name in PublicRingHeater3D.VOLUME_GROUPS}
    assert dict(report.electrical_interface_triangle_counts) == {
        "al_contact_negative:tin_heater": 1,
        "al_contact_positive:tin_heater": 1,
    }
    assert dict(report.surface_triangle_counts) == {
        "external_boundary": 28,
        "bottom_temperature": 5,
        "top_convection": 5,
        "lateral_adiabatic": 14,
        "terminal_negative": 2,
        "terminal_positive": 2,
    }
    assert canonical["schema_version"] == "femx.public-ring-heater-mesh-report/v1"
    assert canonical["region_cell_counts"] == dict(report.region_cell_counts)
    assert canonical["electrical_interface_triangle_counts"] == dict(
        report.electrical_interface_triangle_counts
    )
    assert len(report.digest()) == 64
    assert report.digest() == _evaluate(_valid_mesh()).digest()


def test_ring_heater_quality_rejects_wrong_mesh_envelope() -> None:
    mesh = _valid_mesh()
    triangle_mesh = Mesh(
        geometry=mesh.geometry,
        topology=MeshTopology(
            mesh.topology.connectivity[:, :3],
            CellType.TRIANGLE,
            mesh.geometry.node_count,
        ),
    )
    with pytest.raises(ContractError, match="Tet4"):
        _evaluate(triangle_mesh)
    with pytest.raises(ContractError, match="boundary facets"):
        _evaluate(replace(mesh, boundary_facets=None))
    with pytest.raises(ContractError, match="three-dimensional"):
        _evaluate(replace(mesh, geometry=MeshGeometry(mesh.geometry.coordinates[:, :2])))

    corrupted = replace(mesh)
    object.__setattr__(
        corrupted,
        "boundary_facets",
        MeshTopology(
            np.asarray(((0, 1),), dtype=np.int64),
            CellType.SEGMENT,
            mesh.geometry.node_count,
        ),
    )
    with pytest.raises(ContractError, match="triangular boundary"):
        _evaluate(corrupted)


def test_ring_heater_quality_rejects_missing_and_unexpected_groups() -> None:
    mesh = _valid_mesh()
    changed = replace(
        mesh,
        tags=(*mesh.tags[:-1], EntityTag("unexpected", 2, (27,))),
    )
    with pytest.raises(ContractError, match=r"missing=.*terminal_positive.*unexpected"):
        _evaluate(changed)


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        (EntityTag("silica", 2, (0,)), "must have dimension 3"),
        (EntityTag("silica", 3, ()), "must be non-empty"),
        (EntityTag("silica", 3, (8,)), "invalid cell id"),
        (EntityTag("silica", 3, (0, 1)), "partition every Tet4"),
    ),
)
def test_ring_heater_quality_rejects_invalid_volume_partition(
    replacement: EntityTag,
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        _evaluate(_replace_tag(_valid_mesh(), "silica", replacement))


def test_ring_heater_quality_rejects_invalid_external_boundary() -> None:
    mesh = _valid_mesh()
    with pytest.raises(ContractError, match="every boundary triangle"):
        _evaluate(_replace_tag(mesh, "external_boundary", EntityTag("external_boundary", 3, (0,))))
    with pytest.raises(ContractError, match="every boundary triangle"):
        _evaluate(
            _replace_tag(
                mesh,
                "external_boundary",
                EntityTag("external_boundary", 2, tuple(range(27))),
            )
        )


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        (EntityTag("bottom_temperature", 3, (0,)), "must have dimension 2"),
        (EntityTag("bottom_temperature", 2, ()), "must be non-empty"),
        (EntityTag("bottom_temperature", 2, (28,)), "invalid boundary id"),
        (EntityTag("bottom_temperature", 2, tuple(range(6))), "partition the external"),
    ),
)
def test_ring_heater_quality_rejects_invalid_surface_partition(
    replacement: EntityTag,
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        _evaluate(_replace_tag(_valid_mesh(), "bottom_temperature", replacement))


@pytest.mark.parametrize("corruption", ("negative", "nonfinite", "connectivity"))
def test_ring_heater_quality_rejects_invalid_cell_geometry(corruption: str) -> None:
    mesh = _valid_mesh()
    if corruption == "negative":
        cells = np.asarray(mesh.topology.connectivity).copy()
        cells[0, 1], cells[0, 2] = cells[0, 2], cells[0, 1]
        changed = replace(
            mesh,
            topology=MeshTopology(cells, CellType.TETRAHEDRON, mesh.geometry.node_count),
        )
    elif corruption == "nonfinite":
        coordinates = np.asarray(mesh.geometry.coordinates).copy()
        coordinates[0, 0] = np.nan
        changed = replace(mesh, geometry=MeshGeometry(coordinates))
    else:
        connectivity = np.asarray(mesh.topology.connectivity).copy()
        connectivity[0, 0] = -1
        topology = replace(mesh.topology)
        object.__setattr__(topology, "connectivity", connectivity)
        changed = replace(mesh, topology=topology)
    message = (
        "invalid node id" if corruption == "connectivity" else "volumes must be finite and positive"
    )
    with pytest.raises(ContractError, match=message):
        _evaluate(changed)


def test_ring_heater_quality_rejects_invalid_shape_metric(monkeypatch) -> None:
    monkeypatch.setattr(
        quality_module.np,
        "power",
        lambda values, exponent: np.zeros_like(values),
    )
    with pytest.raises(ContractError, match="mean-ratio quality"):
        _evaluate(_valid_mesh())


def test_ring_heater_quality_rejects_disconnected_electrical_contact() -> None:
    mesh = _valid_mesh()
    cells = np.asarray(mesh.topology.connectivity).copy()
    cells[6] = cells[0]
    changed = replace(
        mesh,
        topology=MeshTopology(cells, CellType.TETRAHEDRON, mesh.geometry.node_count),
    )
    with pytest.raises(ContractError, match=r"al_contact_negative.*share a conformal face"):
        _evaluate(changed)


def test_ring_heater_quality_rejects_nonfinite_report_metric(monkeypatch) -> None:
    monkeypatch.setattr(quality_module.np, "quantile", lambda values, quantile: float("nan"))
    with pytest.raises(ContractError, match="non-finite metric"):
        _evaluate(_valid_mesh())
