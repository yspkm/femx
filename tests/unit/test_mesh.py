import pytest
from tests.support import FakeArray

from femx.core.capabilities import FunctionSpaceFamily
from femx.core.errors import ContractError
from femx.mesh import (
    CellType,
    DofLocation,
    DofMap,
    EntityTag,
    FunctionSpace,
    Mesh,
    MeshGeometry,
    MeshPartition,
    MeshTopology,
    OrientationMap,
)

pytestmark = pytest.mark.unit


def test_mesh_keeps_geometry_topology_tags_and_orientation_separate() -> None:
    geometry = MeshGeometry(FakeArray((4, 2)))
    topology = MeshTopology(FakeArray((2, 3)), CellType.TRIANGLE, node_count=4)
    mesh = Mesh(
        geometry=geometry,
        topology=topology,
        tags=(EntityTag("heated_boundary", 1, (0, 2)),),
        orientation=OrientationMap(edge_signs=FakeArray((2, 3))),
    )

    assert mesh.geometry.spatial_dimension == 2
    assert mesh.topology.cell_count == 2
    assert mesh.topology.cell_type.dimension == 2


def test_mesh_rejects_shape_unit_and_dimension_mismatches() -> None:
    with pytest.raises(ContractError, match="shape"):
        MeshGeometry(FakeArray((4,)))
    with pytest.raises(ContractError, match="SI metres"):
        MeshGeometry(FakeArray((4, 2)), coordinate_unit="um")
    with pytest.raises(ContractError, match="requires 3 corners"):
        MeshTopology(FakeArray((2, 4)), CellType.TRIANGLE, node_count=4)

    geometry = MeshGeometry(FakeArray((4, 2)))
    topology = MeshTopology(FakeArray((2, 3)), CellType.TRIANGLE, node_count=5)
    with pytest.raises(ContractError, match="node counts differ"):
        Mesh(geometry, topology)


def test_function_space_and_dof_contracts() -> None:
    hcurl = FunctionSpace(FunctionSpaceFamily.HCURL, order=1, value_shape=(3,))
    dof_map = DofMap(
        cell_dofs=FakeArray((2, 6)),
        dof_count=9,
        locations=frozenset({DofLocation.EDGE}),
    )

    assert hcurl.family is FunctionSpaceFamily.HCURL
    assert dof_map.dof_count == 9
    with pytest.raises(ContractError, match="order"):
        FunctionSpace(FunctionSpaceFamily.H1, order=0)
    with pytest.raises(ContractError, match="rank-two"):
        DofMap(FakeArray((4,)), dof_count=4, locations=frozenset({DofLocation.VERTEX}))


def test_partition_rejects_ambiguous_ownership() -> None:
    partition = MeshPartition(0, 2, owned_dofs=(0, 1), ghost_dofs=(2,))
    assert partition.process_count == 2
    with pytest.raises(ContractError, match="overlap"):
        MeshPartition(0, 2, owned_dofs=(0, 1), ghost_dofs=(1, 2))
    with pytest.raises(ContractError, match="global process range"):
        MeshPartition(2, 2, owned_dofs=())


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: MeshTopology(FakeArray((3,)), CellType.SEGMENT, 3), "rank two"),
        (lambda: MeshTopology(FakeArray((1, 2)), CellType.SEGMENT, 0), "positive"),
        (lambda: EntityTag(" bad", 0, (0,)), "tag name"),
        (lambda: EntityTag("bad_dim", 4, (0,)), "dimension"),
        (lambda: EntityTag("negative", 0, (-1,)), "negative"),
        (lambda: EntityTag("duplicate", 0, (1, 1)), "duplicate"),
        (lambda: OrientationMap(edge_signs=FakeArray((3,))), "rank-two"),
        (lambda: FunctionSpace(FunctionSpaceFamily.H1, 1, value_shape=(0,)), "value shape"),
        (lambda: FunctionSpace(FunctionSpaceFamily.H1, 1, continuity=""), "continuity"),
        (
            lambda: DofMap(FakeArray((1, 2)), 0, frozenset({DofLocation.VERTEX})),
            "dof_count",
        ),
        (lambda: DofMap(FakeArray((1, 2)), 2, frozenset()), "at least one"),
        (lambda: MeshPartition(0, 0, ()), "process_count"),
        (lambda: MeshPartition(0, 1, (-1,)), "cannot be negative"),
        (lambda: MeshPartition(0, 1, (1, 1)), "owned DOF ids"),
        (lambda: MeshPartition(0, 1, (), (2, 2)), "ghost DOF ids"),
    ],
)
def test_mesh_contract_rejects_ambiguous_metadata(factory, message: str) -> None:
    with pytest.raises(ContractError, match=message):
        factory()


def test_mesh_rejects_duplicate_invalid_tags_and_unknown_schema() -> None:
    geometry = MeshGeometry(FakeArray((3, 2)))
    topology = MeshTopology(FakeArray((1, 3)), CellType.TRIANGLE, 3)
    same = EntityTag("same", 1, (0,))
    with pytest.raises(ContractError, match="tag names"):
        Mesh(geometry, topology, tags=(same, same))
    with pytest.raises(ContractError, match="topological dimension"):
        Mesh(geometry, topology, tags=(EntityTag("volume", 3, (0,)),))
    with pytest.raises(ContractError, match="unsupported mesh schema"):
        Mesh(geometry, topology, schema_version="femx.mesh/v2")
    with pytest.raises(ContractError, match="does not define"):
        Mesh(geometry, topology).tag("missing")


def test_mesh_validates_explicit_boundary_facet_topology() -> None:
    geometry = MeshGeometry(FakeArray((3, 2)))
    topology = MeshTopology(FakeArray((1, 3)), CellType.TRIANGLE, 3)
    boundary = MeshTopology(FakeArray((3, 2)), CellType.SEGMENT, 3)

    assert Mesh(geometry, topology, boundary_facets=boundary).boundary_facets is boundary
    with pytest.raises(ContractError, match="boundary-facet and geometry"):
        Mesh(
            geometry,
            topology,
            boundary_facets=MeshTopology(FakeArray((3, 2)), CellType.SEGMENT, 4),
        )
    with pytest.raises(ContractError, match="dimension one below"):
        Mesh(
            geometry,
            topology,
            boundary_facets=MeshTopology(FakeArray((1, 3)), CellType.TRIANGLE, 3),
        )
