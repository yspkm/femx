from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from femx.core.errors import ContractError, MeshingError
from femx.mesh import CellType
from femx.meshing.gmsh import (
    GmshImportRecord,
    GmshPhysicalGroup,
    ImportedGmshMesh,
    read_gmsh_msh_3d,
)

pytestmark = pytest.mark.unit


_OUTWARD_FACETS = ((2, 3, 4), (1, 4, 3), (1, 2, 4), (1, 3, 2))


def _tetrahedron_msh(
    *,
    reverse_cell: bool = False,
    reverse_boundary: bool = False,
    omit_last_boundary: bool = False,
    duplicate_boundary: bool = False,
    duplicate_cell: bool = False,
    fourth_coordinate: str = "0 0 1",
) -> str:
    facets = [
        (facet[0], facet[2], facet[1]) if reverse_boundary else facet for facet in _OUTWARD_FACETS
    ]
    if duplicate_boundary:
        facets[-1] = facets[-2]
    if omit_last_boundary:
        facets.pop()

    blocks: list[str] = []
    for tag, nodes in enumerate(facets, start=1):
        encoded = " ".join(str(node) for node in nodes)
        blocks.append(f"2 {tag} 2 1\n{tag} {encoded}")
    cell_nodes = "1 2 4 3" if reverse_cell else "1 2 3 4"
    blocks.append(f"3 1 4 1\n5 {cell_nodes}")
    if duplicate_cell:
        blocks.append(f"3 1 4 1\n6 {cell_nodes}")
    maximum_tag = 6 if duplicate_cell else 5
    encoded_blocks = "\n".join(blocks)

    return f"""$MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
2
2 10 "boundary"
3 20 "domain"
$EndPhysicalNames
$Entities
0 0 4 1
1 0 0 0 1 1 1 1 -10 0
2 0 0 0 0 1 1 1 10 0
3 0 0 0 1 0 1 1 10 0
4 0 0 0 1 1 0 1 10 0
1 0 0 0 1 1 1 1 20 0
$EndEntities
$Nodes
1 4 1 4
3 1 0 4
1
2
3
4
0 0 0
1 0 0
0 1 0
{fourth_coordinate}
$EndNodes
$Elements
{len(blocks)} {len(blocks)} 1 {maximum_tag}
{encoded_blocks}
$EndElements
"""


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _nonmanifold_tetrahedron_msh() -> str:
    return (
        _tetrahedron_msh()
        .replace("$Nodes\n1 4 1 4\n3 1 0 4", "$Nodes\n1 6 1 6\n3 1 0 6")
        .replace(
            "1\n2\n3\n4\n0 0 0\n1 0 0\n0 1 0\n0 0 1",
            "1\n2\n3\n4\n5\n6\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n0 0 -1\n0 0 2",
        )
        .replace("$Elements\n5 5 1 5", "$Elements\n7 7 1 7")
        .replace(
            "$EndElements",
            "3 1 4 1\n6 1 3 2 5\n3 1 4 1\n7 1 2 3 6\n$EndElements",
        )
    )


def test_volume_importer_preserves_tags_and_normalizes_cell_and_boundary_orientation(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path / "tetra.msh",
        _tetrahedron_msh(reverse_cell=True, reverse_boundary=True),
    )

    imported = read_gmsh_msh_3d(path, coordinate_scale_to_m=1.0e-6)
    mesh = imported.mesh

    assert mesh.topology.cell_type is CellType.TETRAHEDRON
    assert mesh.boundary_facets is not None
    assert mesh.boundary_facets.cell_type is CellType.TRIANGLE
    np.testing.assert_array_equal(mesh.topology.connectivity, ((0, 1, 2, 3),))
    np.testing.assert_array_equal(
        mesh.boundary_facets.connectivity,
        ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)),
    )
    np.testing.assert_allclose(
        mesh.geometry.coordinates,
        np.eye(4, 3, k=-1) * 1.0e-6,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(mesh.orientation.edge_signs, ((1, 1, 1, 1, 1, 1),))
    np.testing.assert_array_equal(mesh.orientation.face_signs, ((1, -1, 1, -1),))
    assert mesh.tag("boundary").entity_ids == (0, 1, 2, 3)
    assert mesh.tag("domain").entity_ids == (0,)
    assert imported.record.topological_dimension == 3
    assert imported.record.schema_version == "femx.gmsh-import/v2"
    assert imported.record.cell_local_node_permutations == ((0, 1, 3, 2),)
    assert imported.record.boundary_local_node_permutations == ((0, 2, 1),) * 4
    assert imported.record.canonical_data()["topological_dimension"] == 3
    assert len(imported.record.digest()) == 64

    cell = np.asarray(mesh.topology.connectivity[0])
    for facet in np.asarray(mesh.boundary_facets.connectivity):
        opposite = next(node for node in cell if node not in facet)
        points = np.asarray(mesh.geometry.coordinates)[facet]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        assert float(np.dot(normal, mesh.geometry.coordinates[opposite] - points[0])) < 0.0


def test_volume_importer_separates_source_permutations_from_canonical_identity(
    tmp_path: Path,
) -> None:
    direct = read_gmsh_msh_3d(
        _write(tmp_path / "direct.msh", _tetrahedron_msh()),
        coordinate_scale_to_m=1.0,
    )
    reversed_source = read_gmsh_msh_3d(
        _write(
            tmp_path / "reversed.msh",
            _tetrahedron_msh(reverse_cell=True, reverse_boundary=True),
        ),
        coordinate_scale_to_m=1.0,
    )
    assert direct.mesh.boundary_facets is not None
    assert reversed_source.mesh.boundary_facets is not None

    assert direct.record.source_sha256 != reversed_source.record.source_sha256
    assert direct.record.digest() != reversed_source.record.digest()
    assert direct.record.canonical_mesh_sha256 == reversed_source.record.canonical_mesh_sha256
    np.testing.assert_array_equal(
        direct.mesh.topology.connectivity,
        reversed_source.mesh.topology.connectivity,
    )
    np.testing.assert_array_equal(
        direct.mesh.boundary_facets.connectivity,
        reversed_source.mesh.boundary_facets.connectivity,
    )

    with pytest.raises(ContractError, match="does not match"):
        ImportedGmshMesh(
            direct.mesh,
            replace(direct.record, canonical_mesh_sha256="c" * 64),
        )


@pytest.mark.parametrize(
    ("mesh_text", "message"),
    (
        (_tetrahedron_msh(omit_last_boundary=True), "every external boundary"),
        (_tetrahedron_msh(duplicate_boundary=True), "duplicate faces"),
        (_tetrahedron_msh(duplicate_cell=True), "duplicate tetrahedra"),
        (_nonmanifold_tetrahedron_msh(), "non-manifold faces"),
        (_tetrahedron_msh(fourth_coordinate="1 1 0"), "degenerate tetrahedron"),
        (
            _tetrahedron_msh().replace('2 10 "boundary"', '1 10 "boundary"'),
            "surface and volume",
        ),
        (
            _tetrahedron_msh().replace('2 10 "boundary"\n', ""),
            "PhysicalNames count",
        ),
        (
            _tetrahedron_msh()
            .replace("$PhysicalNames\n2", "$PhysicalNames\n1")
            .replace('2 10 "boundary"\n', ""),
            "requires named surface and volume",
        ),
        (
            _tetrahedron_msh()
            .replace("$PhysicalNames\n2", "$PhysicalNames\n3")
            .replace('3 20 "domain"\n', '3 20 "domain"\n3 21 "unused"\n'),
            "contain no imported elements",
        ),
        (
            _tetrahedron_msh().replace("3 1 0 4", "4 1 0 4"),
            "node block dimension",
        ),
        (
            _tetrahedron_msh().replace("3 1 4 1", "3 1 11 1"),
            "unsupported Gmsh element type",
        ),
        (
            _tetrahedron_msh().replace("3 1 4 1", "2 1 4 1"),
            "type and entity dimension",
        ),
    ),
)
def test_volume_importer_rejects_invalid_volume_contracts(
    tmp_path: Path,
    mesh_text: str,
    message: str,
) -> None:
    with pytest.raises((ContractError, MeshingError), match=message):
        read_gmsh_msh_3d(_write(tmp_path / "invalid.msh", mesh_text), coordinate_scale_to_m=1.0)


def test_volume_importer_rejects_invalid_input_envelope(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="finite and positive"):
        read_gmsh_msh_3d(tmp_path / "missing.msh", coordinate_scale_to_m=0.0)
    with pytest.raises(MeshingError, match="does not exist"):
        read_gmsh_msh_3d(tmp_path / "missing.msh", coordinate_scale_to_m=1.0)
    binary = tmp_path / "binary.msh"
    binary.write_bytes(b"\xff\xfe")
    with pytest.raises(MeshingError, match="ASCII"):
        read_gmsh_msh_3d(binary, coordinate_scale_to_m=1.0)
    overflow = _write(
        tmp_path / "overflow.msh",
        _tetrahedron_msh(fourth_coordinate="0 0 1e308"),
    )
    with np.errstate(over="ignore"):
        with pytest.raises(MeshingError, match="scaled Gmsh coordinates"):
            read_gmsh_msh_3d(overflow, coordinate_scale_to_m=10.0)


def test_volume_import_record_rejects_dimension_and_permutation_drift() -> None:
    surface = GmshPhysicalGroup(2, 10, "boundary")
    volume = GmshPhysicalGroup(3, 20, "domain")
    valid = dict(
        source_sha256="a" * 64,
        canonical_mesh_sha256="b" * 64,
        format_version="4.1",
        coordinate_scale_to_m=1.0,
        physical_groups=(surface, volume),
        node_tags=(1, 2, 3, 4),
        cell_element_tags=(5,),
        boundary_element_tags=(1, 2, 3, 4),
        cell_local_node_permutations=((0, 1, 2, 3),),
        topological_dimension=3,
        boundary_local_node_permutations=((0, 1, 2),) * 4,
        schema_version="femx.gmsh-import/v2",
    )

    assert GmshImportRecord(**valid).topological_dimension == 3
    for replacement, message in (
        ({"topological_dimension": 4}, "dimension must be 2 or 3"),
        ({"schema_version": "femx.gmsh-import/v1"}, "requires schema"),
        (
            {"physical_groups": (GmshPhysicalGroup(1, 30, "curve"), volume)},
            "dimension disagrees",
        ),
        ({"cell_local_node_permutations": ((0, 1, 2, 2),)}, "every local node"),
        ({"boundary_local_node_permutations": ()}, "boundary count"),
        ({"boundary_local_node_permutations": ((0, 0, 2),) * 4}, "boundary permutation"),
    ):
        with pytest.raises(ContractError, match=message):
            GmshImportRecord(**(valid | replacement))

    with pytest.raises(ContractError, match="curves, surfaces, or volumes"):
        GmshPhysicalGroup(0, 1, "point")


def test_planar_import_record_rejects_boundary_permutation_metadata() -> None:
    with pytest.raises(ContractError, match="do not normalize"):
        GmshImportRecord(
            source_sha256="a" * 64,
            canonical_mesh_sha256="b" * 64,
            format_version="4.1",
            coordinate_scale_to_m=1.0,
            physical_groups=(GmshPhysicalGroup(2, 20, "domain"),),
            node_tags=(1, 2, 3),
            cell_element_tags=(4,),
            boundary_element_tags=(1, 2, 3),
            cell_local_node_permutations=((0, 1, 2),),
            boundary_local_node_permutations=((0, 1),) * 3,
        )
