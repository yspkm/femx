from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from femx.core.errors import ContractError, MeshingError
from femx.meshing.gmsh import (
    GmshImportRecord,
    GmshPhysicalGroup,
    ImportedGmshMesh,
    read_gmsh_msh,
)

pytestmark = pytest.mark.unit


def _minimal_msh(*, triangle: str = "4 1 2 3", z_coordinate: str = "0") -> str:
    return f"""$MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
2
1 10 "boundary"
2 20 "domain"
$EndPhysicalNames
$Entities
0 3 1 0
1 0 0 0 1 0 0 1 10 0
2 0 0 0 1 1 0 1 10 0
3 0 0 0 0 1 0 1 10 0
1 0 0 0 1 1 0 1 20 0
$EndEntities
$Nodes
1 3 1 3
2 1 0 3
1
2
3
0 0 0
1 0 0
0 1 {z_coordinate}
$EndNodes
$Elements
4 4 1 4
1 1 1 1
1 1 2
1 2 1 1
2 2 3
1 3 1 1
3 3 1
2 1 2 1
{triangle}
$EndElements
"""


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _two_triangle_msh_with_internal_segment() -> str:
    return """$MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
3
1 10 "boundary"
1 11 "interface"
2 20 "domain"
$EndPhysicalNames
$Entities
0 5 1 0
1 0 0 0 1 0 0 1 10 0
2 1 0 0 1 1 0 1 10 0
3 0 1 0 1 1 0 1 10 0
4 0 0 0 0 1 0 1 10 0
5 0 0 0 1 1 0 1 11 0
1 0 0 0 1 1 0 1 20 0
$EndEntities
$Nodes
1 4 1 4
2 1 0 4
1
2
3
4
0 0 0
1 0 0
1 1 0
0 1 0
$EndNodes
$Elements
6 7 1 7
1 1 1 1
1 1 2
1 2 1 1
2 2 3
1 3 1 1
3 3 4
1 4 1 1
4 4 1
1 5 1 1
5 1 3
2 1 2 2
6 1 2 3
7 1 3 4
$EndElements
"""


def test_importer_builds_si_mesh_tags_orientation_and_exact_id_record(tmp_path) -> None:
    path = _write(tmp_path / "mesh.msh", _minimal_msh(triangle="4 1 3 2"))

    imported = read_gmsh_msh(path, coordinate_scale_to_m=1.0e-6)
    mesh = imported.mesh

    np.testing.assert_array_equal(mesh.topology.connectivity, [[0, 1, 2]])
    np.testing.assert_allclose(
        mesh.geometry.coordinates,
        [[0.0, 0.0], [1.0e-6, 0.0], [0.0, 1.0e-6]],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(mesh.orientation.edge_signs, [[1, 1, -1]])
    assert mesh.tag("boundary").entity_ids == (0, 1, 2)
    assert mesh.tag("domain").entity_ids == (0,)
    assert imported.record.node_tags == (1, 2, 3)
    assert imported.record.cell_element_tags == (4,)
    assert imported.record.boundary_element_tags == (1, 2, 3)
    assert imported.record.cell_local_node_permutations == ((0, 2, 1),)
    assert imported.record.source_sha256 == imported.record.canonical_data()["source_sha256"]
    assert len(imported.record.canonical_mesh_sha256) == 64
    assert len(imported.record.digest()) == 64


def test_importer_separates_source_permutation_from_canonical_mesh_identity(tmp_path) -> None:
    direct = read_gmsh_msh(
        _write(tmp_path / "direct.msh", _minimal_msh(triangle="4 1 2 3")),
        coordinate_scale_to_m=1.0e-6,
    )
    flipped = read_gmsh_msh(
        _write(tmp_path / "flipped.msh", _minimal_msh(triangle="4 1 3 2")),
        coordinate_scale_to_m=1.0e-6,
    )

    assert direct.record.source_sha256 != flipped.record.source_sha256
    assert direct.record.digest() != flipped.record.digest()
    assert direct.record.cell_local_node_permutations == ((0, 1, 2),)
    assert flipped.record.cell_local_node_permutations == ((0, 2, 1),)
    assert direct.record.canonical_mesh_sha256 == flipped.record.canonical_mesh_sha256
    np.testing.assert_array_equal(
        direct.mesh.topology.connectivity, flipped.mesh.topology.connectivity
    )

    with pytest.raises(ContractError, match="does not match"):
        ImportedGmshMesh(
            direct.mesh,
            replace(direct.record, canonical_mesh_sha256="c" * 64),
        )


def test_importer_rejects_non_si_or_missing_inputs(tmp_path) -> None:
    with pytest.raises(ContractError, match="finite and positive"):
        read_gmsh_msh(tmp_path / "missing.msh", coordinate_scale_to_m=0.0)
    with pytest.raises(ContractError, match="finite and positive"):
        read_gmsh_msh(tmp_path / "missing.msh", coordinate_scale_to_m=float("inf"))
    with pytest.raises(MeshingError, match="does not exist"):
        read_gmsh_msh(tmp_path / "missing.msh", coordinate_scale_to_m=1.0)

    binary = tmp_path / "binary.msh"
    binary.write_bytes(b"\xff\xfe")
    with pytest.raises(MeshingError, match="ASCII"):
        read_gmsh_msh(binary, coordinate_scale_to_m=1.0)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda text: text.replace("4.1 0 8", "4.1 1 8"), "exact ASCII"),
        (lambda text: text.replace("$PhysicalNames\n", ""), "unexpected text"),
        (
            lambda text: text.replace(
                "$EndElements\n", "$EndElements\n$Periodic\n0\n$EndPeriodic\n"
            ),
            "unsupported sections",
        ),
        (lambda text: text.replace("$EndNodes", "$EndBroken"), "not terminated"),
        (
            lambda text: text.replace(
                "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n",
                "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$MeshFormat\n4.1 0 8\n$EndMeshFormat\n",
            ),
            "duplicate",
        ),
    ],
)
def test_importer_rejects_invalid_section_envelopes(tmp_path, mutator, message: str) -> None:
    path = _write(tmp_path / "mesh.msh", mutator(_minimal_msh()))
    with pytest.raises(MeshingError, match=message):
        read_gmsh_msh(path, coordinate_scale_to_m=1.0)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("$PhysicalNames\n2", "$PhysicalNames\n3", "count"),
        ('1 10 "boundary"', "1 10 boundary extra", "physical-name"),
        ('2 20 "domain"', '2 20 "boundary"', "unique"),
        ('1 10 "boundary"', '2 20 "other"', "unique"),
        ('1 10 "boundary"', '3 10 "boundary"', "curve and surface"),
        ('1 10 "boundary"', '1 0 "boundary"', "positive"),
        ('1 10 "boundary"', '1 10 " bad"', "trimmed"),
    ],
)
def test_importer_rejects_invalid_physical_groups(
    tmp_path, old: str, new: str, message: str
) -> None:
    path = _write(tmp_path / "mesh.msh", _minimal_msh().replace(old, new))
    with pytest.raises((ContractError, MeshingError), match=message):
        read_gmsh_msh(path, coordinate_scale_to_m=1.0)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("0 3 1 0", "0 4 1 0", "duplicate"),
        ("0 3 1 0", "0 2 1 0", "unnamed"),
        ("1 0 0 0 1 0 0 1 10 0", "1 0", "too short"),
        ("1 0 0 0 1 0 0 1 10 0", "1 0 0 0 1 0 0 3 10 0", "exceeds"),
        ("1 0 0 0 1 0 0 1 10 0", "1 0 0 0 1 0 0 1 99 0", "unnamed"),
        (
            "1 0 0 0 1 0 0 1 10 0",
            "1 0 0 0 1 0 0 2 10 -10 0",
            "repeats a physical tag",
        ),
        ("1 0 0 0 1 0 0 1 10 0", "1 0 0 0 1 0 0 1 10", "bounding"),
        ("1 0 0 0 1 0 0 1 10 0", "1 0 0 0 1 0 0 1 10 1", "bounding-tag"),
        (
            "0 3 1 0\n",
            "1 3 1 0\n1 0 0 0 0 trailing\n",
            "point entity",
        ),
    ],
)
def test_importer_rejects_invalid_entities(tmp_path, old: str, new: str, message: str) -> None:
    path = _write(tmp_path / "mesh.msh", _minimal_msh().replace(old, new, 1))
    with pytest.raises(MeshingError, match=message):
        read_gmsh_msh(path, coordinate_scale_to_m=1.0)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("1 3 1 3\n", "2 3 1 3\n", "ended before"),
        ("1 3 1 3\n", "0 3 1 3\n", "trailing"),
        ("2 1 0 3", "3 1 0 3", "3D node"),
        ("2 1 0 3", "2 1 1 3", "parametric"),
        ("2 1 0 3", "2 1 0 4", "node tag requires"),
        ("2 1 0 3", "2 1 0 2", "node coordinates requires"),
        ("1\n2\n3\n", "1 2\n2\n3\n", "requires 1"),
        ("1\n2\n3\n", "1\n1\n3\n", "duplicate"),
        ("0 1 0\n", "0 bad 0\n", "floating-point"),
        ("1 3 1 3", "1 4 1 3", "count"),
        ("1 3 1 3", "1 3 2 4", "min/max"),
    ],
)
def test_importer_rejects_invalid_nodes(tmp_path, old: str, new: str, message: str) -> None:
    path = _write(tmp_path / "mesh.msh", _minimal_msh().replace(old, new, 1))
    with pytest.raises(MeshingError, match=message):
        read_gmsh_msh(path, coordinate_scale_to_m=1.0)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("4 4 1 4\n", "5 4 1 4\n", "ended before"),
        ("4 4 1 4\n", "3 4 1 4\n", "trailing"),
        ("1 1 1 1", "1 1 8 1", "unsupported"),
        ("1 1 1 1", "2 1 1 1", "dimension"),
        ("1 1 1 1", "1 99 1 1", "no named"),
        ("1 1 1 1", "1 1 1 2", "element record requires"),
        ("1 1 1 1", "1 1 1 0", "block header requires"),
        ("1 1 2\n", "1 1\n", "requires 3"),
        ("4 4 1 4", "4 5 1 4", "count"),
        ("4 4 1 4", "4 4 2 5", "min/max"),
    ],
)
def test_importer_rejects_invalid_elements(tmp_path, old: str, new: str, message: str) -> None:
    path = _write(tmp_path / "mesh.msh", _minimal_msh().replace(old, new, 1))
    with pytest.raises(MeshingError, match=message):
        read_gmsh_msh(path, coordinate_scale_to_m=1.0)


def test_importer_rejects_invalid_geometry_and_connectivity(tmp_path) -> None:
    z_path = _write(tmp_path / "z.msh", _minimal_msh(z_coordinate="1e-9"))
    with pytest.raises(MeshingError, match="nonzero z"):
        read_gmsh_msh(z_path, coordinate_scale_to_m=1.0)

    missing_node = _write(tmp_path / "missing-node.msh", _minimal_msh(triangle="4 1 2 9"))
    with pytest.raises(MeshingError, match="missing node 9"):
        read_gmsh_msh(missing_node, coordinate_scale_to_m=1.0)

    degenerate = _minimal_msh().replace("0 1 0\n", "2 0 0\n", 1)
    with pytest.raises(MeshingError, match="degenerate"):
        read_gmsh_msh(_write(tmp_path / "degenerate.msh", degenerate), coordinate_scale_to_m=1.0)

    duplicate = _minimal_msh().replace("3 3 1", "3 1 2")
    with pytest.raises(MeshingError, match="duplicate edges"):
        read_gmsh_msh(_write(tmp_path / "duplicate.msh", duplicate), coordinate_scale_to_m=1.0)

    with pytest.raises(MeshingError, match="external boundary"):
        read_gmsh_msh(
            _write(tmp_path / "interior.msh", _two_triangle_msh_with_internal_segment()),
            coordinate_scale_to_m=1.0,
        )


def test_import_record_validates_schema_hash_and_id_maps() -> None:
    group = GmshPhysicalGroup(2, 1, "domain")
    valid = dict(
        source_sha256="a" * 64,
        canonical_mesh_sha256="b" * 64,
        format_version="4.1",
        coordinate_scale_to_m=1.0,
        physical_groups=(group,),
        node_tags=(1, 2, 3),
        cell_element_tags=(4,),
        boundary_element_tags=(1, 2, 3),
        cell_local_node_permutations=((0, 1, 2),),
    )
    planar_record = GmshImportRecord(**valid)
    assert planar_record.schema_version == "femx.gmsh-import/v1"
    assert "topological_dimension" not in planar_record.canonical_data()
    assert "boundary_local_node_permutations" not in planar_record.canonical_data()
    assert planar_record.digest() == (
        "9db3e79ac85bb3d9d96acac7c0128053908d96728a5fd959248077dc60ee8d6c"
    )
    for replacement, message in (
        ({"source_sha256": "bad"}, "SHA-256"),
        ({"canonical_mesh_sha256": "bad"}, "SHA-256"),
        ({"format_version": "2.2"}, "4.1"),
        ({"coordinate_scale_to_m": -1.0}, "finite and positive"),
        ({"schema_version": "future"}, "requires schema"),
        ({"physical_groups": ()}, "non-empty and unique"),
        ({"physical_groups": (group, group)}, "non-empty and unique"),
        (
            {"physical_groups": (group, GmshPhysicalGroup(1, 2, "domain"))},
            "names must be unique",
        ),
        ({"node_tags": ()}, "non-empty unique positive"),
        ({"node_tags": (1, 1)}, "unique positive"),
        ({"cell_element_tags": (0,)}, "unique positive"),
        ({"cell_local_node_permutations": ()}, "canonical cell count"),
        ({"cell_local_node_permutations": ((0, 0, 2),)}, "local node"),
    ):
        with pytest.raises(ContractError, match=message):
            GmshImportRecord(**(valid | replacement))
