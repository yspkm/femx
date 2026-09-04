import numpy as np
import pytest

from femx.backends._hcurl import (
    canonical_mixed_port_dof_partition,
    canonical_triangle_edge_map,
)
from femx.core.errors import ContractError

pytestmark = pytest.mark.unit


def test_canonical_triangle_edge_map_is_lexicographic_and_conforming() -> None:
    cells = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    signs = np.asarray(((1, 1, -1), (1, 1, -1)), dtype=np.int8)

    edge_map = canonical_triangle_edge_map(cells, signs)

    np.testing.assert_array_equal(
        edge_map.edge_nodes,
        ((0, 1), (0, 2), (0, 3), (1, 2), (2, 3)),
    )
    np.testing.assert_array_equal(edge_map.cell_edge_dofs, ((0, 3, 1), (1, 4, 2)))
    np.testing.assert_array_equal(edge_map.cell_edge_signs, signs)
    assert edge_map.dof_count == 5
    assert not edge_map.edge_nodes.flags.writeable
    assert not edge_map.cell_edge_dofs.flags.writeable
    assert not edge_map.cell_edge_signs.flags.writeable


def test_mixed_port_partition_constrains_all_boundary_nodes_and_edges() -> None:
    cells = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    signs = np.asarray(((1, 1, -1), (1, 1, -1)), dtype=np.int8)
    edge_map = canonical_triangle_edge_map(cells, signs)

    partition = canonical_mixed_port_dof_partition(
        np.asarray(((1, 0), (2, 1), (3, 2), (0, 3)), dtype=np.int32),
        edge_map,
        node_count=4,
    )

    np.testing.assert_array_equal(partition.scalar_dofs, (0, 1, 2, 3))
    np.testing.assert_array_equal(partition.edge_dofs, (0, 2, 3, 4))
    np.testing.assert_array_equal(partition.constrained_dofs, (0, 1, 2, 3, 4, 6, 7, 8))
    np.testing.assert_array_equal(partition.free_dofs, (5,))
    for values in (
        partition.scalar_dofs,
        partition.edge_dofs,
        partition.constrained_dofs,
        partition.free_dofs,
    ):
        assert not values.flags.writeable


@pytest.mark.parametrize(
    ("facets", "node_count", "message"),
    [
        (np.asarray(((0, 1),), dtype=np.int64), 0, "node_count"),
        (np.asarray(((0.0, 1.0),)), 4, "integer"),
        (np.asarray((0, 1), dtype=np.int64), 4, "shaped"),
        (np.empty((0, 2), dtype=np.int64), 4, "nonempty"),
        (np.asarray(((-1, 1),), dtype=np.int64), 4, "out-of-range"),
        (np.asarray(((0, 4),), dtype=np.int64), 4, "out-of-range"),
        (np.asarray(((0, 0),), dtype=np.int64), 4, "repeated"),
        (
            np.asarray(((0, 1), (1, 0), (1, 2), (2, 3), (3, 0)), dtype=np.int64),
            4,
            "duplicates",
        ),
        (
            np.asarray(((0, 1), (1, 3), (2, 3), (3, 0)), dtype=np.int64),
            4,
            "not a triangle-mesh edge",
        ),
        (
            np.asarray(((0, 1), (1, 2), (2, 3)), dtype=np.int64),
            4,
            "topological boundary",
        ),
        (
            np.asarray(((0, 1), (1, 2), (2, 3), (0, 2)), dtype=np.int64),
            4,
            "topological boundary",
        ),
    ],
)
def test_mixed_port_partition_rejects_ambiguous_pec_topology(
    facets: np.ndarray,
    node_count: int,
    message: str,
) -> None:
    cells = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    signs = np.asarray(((1, 1, -1), (1, 1, -1)), dtype=np.int8)
    edge_map = canonical_triangle_edge_map(cells, signs)

    with pytest.raises(ContractError, match=message):
        canonical_mixed_port_dof_partition(facets, edge_map, node_count=node_count)


def test_mixed_port_partition_rejects_a_fully_constrained_single_triangle() -> None:
    cells = np.asarray(((0, 1, 2),), dtype=np.int64)
    signs = np.asarray(((1, 1, -1),), dtype=np.int8)
    edge_map = canonical_triangle_edge_map(cells, signs)

    with pytest.raises(ContractError, match="no free DOFs"):
        canonical_mixed_port_dof_partition(
            np.asarray(((0, 1), (1, 2), (2, 0)), dtype=np.int64),
            edge_map,
            node_count=3,
        )


@pytest.mark.parametrize(
    ("cells", "signs", "message"),
    [
        (np.ones((1, 3), dtype=np.float64), np.ones((1, 3), dtype=np.int8), "integer"),
        (np.ones((3,), dtype=np.int64), np.ones((1, 3), dtype=np.int8), "shaped"),
        (np.empty((0, 3), dtype=np.int64), np.empty((0, 3), dtype=np.int8), "at least"),
        (np.asarray(((-1, 0, 1),)), np.asarray(((1, 1, -1),)), "negative"),
        (np.asarray(((0, 0, 1),)), np.asarray(((-1, 1, -1),)), "repeated"),
        (np.asarray(((0, 1, 2),)), None, "explicit"),
        (
            np.asarray(((0, 1, 2),)),
            np.ones((1, 3), dtype=np.float64),
            "integer",
        ),
        (np.asarray(((0, 1, 2),)), np.ones((3,), dtype=np.int8), "shaped"),
        (
            np.asarray(((0, 1, 2),)),
            np.asarray(((1, 1, 1),), dtype=np.int8),
            "canonical global edge",
        ),
    ],
)
def test_canonical_triangle_edge_map_rejects_ambiguous_topology(
    cells: np.ndarray,
    signs: np.ndarray | None,
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        canonical_triangle_edge_map(cells, signs)
