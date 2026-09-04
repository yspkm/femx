from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from femx.core.execution import ExecutionPolicy
from femx.mesh import CellType
from femx.meshing.gmsh import GmshMeshingRequest, read_gmsh_msh_3d

pytestmark = [pytest.mark.integration, pytest.mark.requires_gmsh]

_AUTHORIZED = ExecutionPolicy(execution_authorized=True, allow_external_process=True)
_BOX_GEO = """// femx deterministic Tet4 integration specimen; model unit = 1 um
SetFactory("OpenCASCADE");
Box(1) = {0, 0, 0, 1, 1, 1};
boundary() = Boundary{ Volume{1}; };
Physical Surface("boundary", 101) = {boundary()};
Physical Volume("domain", 201) = {1};
Mesh.MeshSizeMin = 0.35;
Mesh.MeshSizeMax = 0.35;
Mesh.MshFileVersion = 4.1;
Mesh.Binary = 0;
Mesh.ElementOrder = 1;
Mesh.SaveAll = 0;
Mesh.Algorithm3D = 1;
Mesh.RandomFactor = 1e-9;
Mesh.RandomSeed = 1;
"""


def test_real_gmsh_repeat_is_one_canonical_tet4_volume(
    locked_gmsh_runner,
    tmp_path: Path,
) -> None:
    attempts = []
    for attempt_number in (1, 2):
        attempt = tmp_path / f"attempt-{attempt_number:03d}"
        attempt.mkdir()
        (attempt / "box.geo").write_text(_BOX_GEO, encoding="utf-8")
        process = locked_gmsh_runner.run(
            GmshMeshingRequest("box.geo", dimension=3),
            working_directory=attempt,
            policy=_AUTHORIZED,
        )
        assert process.process_succeeded, process.stderr
        assert process.mesh_sha256 is not None
        imported = read_gmsh_msh_3d(attempt / "mesh.msh", coordinate_scale_to_m=1.0e-6)
        attempts.append((process, imported))

    first_process, first = attempts[0]
    second_process, second = attempts[1]
    assert first_process.identity == second_process.identity
    assert first_process.geometry_sha256 == second_process.geometry_sha256
    assert first_process.mesh_sha256 == second_process.mesh_sha256
    assert first.record.digest() == second.record.digest()
    assert first.record.canonical_mesh_sha256 == second.record.canonical_mesh_sha256

    mesh = first.mesh
    assert mesh.topology.cell_type is CellType.TETRAHEDRON
    assert mesh.boundary_facets is not None
    assert mesh.boundary_facets.cell_type is CellType.TRIANGLE
    assert mesh.tag("domain").entity_ids == tuple(range(mesh.topology.cell_count))
    assert mesh.tag("boundary").entity_ids == tuple(range(mesh.boundary_facets.cell_count))
    assert mesh.orientation.edge_signs is not None
    assert mesh.orientation.face_signs is not None
    assert mesh.orientation.edge_signs.shape == (mesh.topology.cell_count, 6)
    assert mesh.orientation.face_signs.shape == (mesh.topology.cell_count, 4)

    coordinates = np.asarray(mesh.geometry.coordinates)
    cells = np.asarray(mesh.topology.connectivity)
    points = coordinates[cells]
    jacobians = np.stack(
        (
            points[:, 1] - points[:, 0],
            points[:, 2] - points[:, 0],
            points[:, 3] - points[:, 0],
        ),
        axis=2,
    )
    assert np.all(np.linalg.det(jacobians) > 0.0)

    external_faces = set()
    face_counts: dict[tuple[int, int, int], int] = {}
    for cell in cells:
        for local_face in ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)):
            face = tuple(sorted(int(cell[index]) for index in local_face))
            face_counts[face] = face_counts.get(face, 0) + 1
    external_faces.update(face for face, count in face_counts.items() if count == 1)
    imported_faces = {
        tuple(sorted(int(node) for node in face))
        for face in np.asarray(mesh.boundary_facets.connectivity)
    }
    assert imported_faces == external_faces
