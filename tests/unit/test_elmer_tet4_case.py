from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from femx.backends.elmer.tet4_case import (
    _external_faces,
    _normalize_tetrahedra,
    lower_tagged_tet4_mesh,
)
from femx.backends.elmer.tet4_electrothermal_case import (
    ElmerTet4BoundaryCondition,
    ElmerTet4ElectrothermalBody,
    ElmerTet4ElectrothermalCase,
    render_tet4_electrothermal_sif,
)
from femx.core.errors import ContractError
from femx.mesh import CellType, EntityTag, Mesh, MeshGeometry, MeshTopology

pytestmark = pytest.mark.unit

_LOCAL_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))


def _two_tet_mesh() -> Mesh:
    coordinates = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ),
        dtype=np.float64,
    )
    cells = np.asarray(((0, 1, 2, 3), (0, 2, 1, 4)), dtype=np.int64)
    counts: dict[tuple[int, int, int], int] = {}
    for cell in cells:
        for local_face in _LOCAL_FACES:
            face = tuple(sorted(int(cell[index]) for index in local_face))
            counts[face] = counts.get(face, 0) + 1
    facets = np.asarray(sorted(face for face, count in counts.items() if count == 1))
    by_face = {tuple(int(node) for node in face): index for index, face in enumerate(facets)}
    negative = by_face[(0, 2, 3)]
    positive = by_face[(0, 1, 3)]
    bottom = by_face[(0, 1, 4)]
    assigned = {negative, positive, bottom}
    other = tuple(index for index in range(len(facets)) if index not in assigned)
    return Mesh(
        geometry=MeshGeometry(coordinates),
        topology=MeshTopology(cells, CellType.TETRAHEDRON, len(coordinates)),
        tags=(
            EntityTag("conductor", 3, (0,)),
            EntityTag("dielectric", 3, (1,)),
            EntityTag("terminal_negative", 2, (negative,)),
            EntityTag("terminal_positive", 2, (positive,)),
            EntityTag("bottom_temperature", 2, (bottom,)),
            EntityTag("other", 2, other),
        ),
        boundary_facets=MeshTopology(facets, CellType.TRIANGLE, len(coordinates)),
    )


def _deck():
    return lower_tagged_tet4_mesh(
        _two_tet_mesh(),
        region_tags=("conductor", "dielectric"),
        boundary_tags=(
            "terminal_negative",
            "terminal_positive",
            "bottom_temperature",
            "other",
        ),
    )


def _case() -> ElmerTet4ElectrothermalCase:
    return ElmerTet4ElectrothermalCase(
        mesh=_deck(),
        bodies=(
            ElmerTet4ElectrothermalBody(1, 10.0, 2.0),
            ElmerTet4ElectrothermalBody(2, 1.0),
        ),
        boundaries=(
            ElmerTet4BoundaryCondition(1, potential_v=0.0),
            ElmerTet4BoundaryCondition(
                2,
                potential_v=1.0,
                heat_transfer_coefficient_w_per_m2_k=5.0,
                external_temperature_k=300.0,
            ),
            ElmerTet4BoundaryCondition(3, temperature_k=300.0),
        ),
        initial_temperature_k=300.0,
    )


def test_tagged_tet4_lowering_emits_complete_oriented_native_mesh() -> None:
    deck = _deck()

    assert deck.header == "5 2 6\n2\n303 6\n504 2\n"
    assert deck.node_count == 5
    assert deck.element_count == 2
    assert deck.boundary_count == 6
    assert deck.body_ids == (1, 2)
    assert deck.boundary_ids == (1, 2, 3, 4)
    assert deck.body_node_ids == ((0, 1, 2, 3), (0, 1, 2, 4))
    assert "1 1 504 1 2 3 4" in deck.elements
    assert "2 2 504 1 3 2 5" in deck.elements
    assert all(" 303 " in line for line in deck.boundary.splitlines())
    assert len(deck.digest()) == 64
    assert deck.digest() == _deck().digest()


def test_tet4_renderer_separates_partial_current_from_full_heat() -> None:
    case = _case()
    sif = render_tet4_electrothermal_sif(
        case,
        stat_current_module=Path("/locked/StatCurrentSolve.so"),
        heat_solve_module=Path("/locked/HeatSolve.so"),
        convergence_tolerance=1.0e-12,
    )

    assert case.potential_node_ids == (0, 1, 2, 3)
    assert len(case.digest()) == 64
    assert case.canonical_data()["potential_node_ids"] == [0, 1, 2, 3]
    assert "Coordinate System = Cartesian 3D" in sif
    assert "Body 1\n  Target Bodies(1) = 1\n  Equation = 1" in sif
    assert "Body 2\n  Target Bodies(1) = 2\n  Equation = 2" in sif
    assert "Equation 1\n  Active Solvers(2) = 1 2" in sif
    assert "Equation 2\n  Active Solvers(1) = 2" in sif
    assert sif.count("Joule Heat = Logical True") == 1
    assert sif.count("Body Force 1") == 1
    assert "Body Force 2" not in sif
    assert 'Procedure = File "/locked/StatCurrentSolve.so" "StatCurrentSolver"' in sif
    assert 'Procedure = File "/locked/HeatSolve.so" "HeatSolver"' in sif
    assert "Heat Transfer Coefficient = Real 5.00000000000000000e+00" in sif
    assert "External Temperature = Real 3.00000000000000000e+02" in sif


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"body_id": 0, "heat_conductivity_w_per_m_k": 1.0}, "positive integer"),
        ({"body_id": True, "heat_conductivity_w_per_m_k": 1.0}, "positive integer"),
        ({"body_id": 1, "heat_conductivity_w_per_m_k": True}, "real scalar"),
        ({"body_id": 1, "heat_conductivity_w_per_m_k": 0.0}, "finite and positive"),
        (
            {
                "body_id": 1,
                "heat_conductivity_w_per_m_k": 1.0,
                "electric_conductivity_s_per_m": -1.0,
            },
            "finite and positive",
        ),
    ),
)
def test_tet4_body_rejects_invalid_coefficients(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        ElmerTet4ElectrothermalBody(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"boundary_id": 0, "potential_v": 0.0}, "positive integer"),
        ({"boundary_id": True, "potential_v": 0.0}, "positive integer"),
        ({"boundary_id": 1, "potential_v": True}, "real scalar"),
        ({"boundary_id": 1, "temperature_k": float("nan")}, "finite"),
        (
            {"boundary_id": 1, "heat_transfer_coefficient_w_per_m2_k": 1.0},
            "requires both",
        ),
        (
            {
                "boundary_id": 1,
                "heat_transfer_coefficient_w_per_m2_k": 0.0,
                "external_temperature_k": 300.0,
            },
            "finite and positive",
        ),
        (
            {
                "boundary_id": 1,
                "temperature_k": 300.0,
                "heat_transfer_coefficient_w_per_m2_k": 1.0,
                "external_temperature_k": 300.0,
            },
            "cannot prescribe temperature and Robin",
        ),
        ({"boundary_id": 1}, "at least one value"),
    ),
)
def test_tet4_boundary_rejects_ambiguous_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        ElmerTet4BoundaryCondition(**kwargs)  # type: ignore[arg-type]


def test_tet4_case_rejects_invalid_bindings() -> None:
    valid = _case()
    with pytest.raises(ContractError, match="native Tet4 mesh"):
        replace(valid, mesh=object())  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="match emitted body ids"):
        replace(valid, bodies=valid.bodies[:1])
    with pytest.raises(ContractError, match="requires an electrical body"):
        replace(
            valid,
            bodies=(
                ElmerTet4ElectrothermalBody(1, 10.0),
                ElmerTet4ElectrothermalBody(2, 1.0),
            ),
        )
    with pytest.raises(ContractError, match="unique emitted ids"):
        replace(valid, boundaries=(valid.boundaries[0], valid.boundaries[0]))
    with pytest.raises(ContractError, match="unique emitted ids"):
        replace(
            valid,
            boundaries=(
                *valid.boundaries,
                ElmerTet4BoundaryCondition(99, temperature_k=300.0),
            ),
        )
    with pytest.raises(ContractError, match="distinct terminal potentials"):
        replace(
            valid,
            boundaries=(
                ElmerTet4BoundaryCondition(1, potential_v=0.0, temperature_k=300.0),
                ElmerTet4BoundaryCondition(2, potential_v=0.0),
            ),
        )
    with pytest.raises(ContractError, match="requires a thermal boundary"):
        replace(
            valid,
            boundaries=(
                ElmerTet4BoundaryCondition(1, potential_v=0.0),
                ElmerTet4BoundaryCondition(2, potential_v=1.0),
            ),
        )
    with pytest.raises(ContractError, match="initial temperature"):
        replace(valid, initial_temperature_k=0.0)
    with pytest.raises(ContractError, match="outside electrical bodies"):
        replace(
            valid,
            boundaries=(
                ElmerTet4BoundaryCondition(1, potential_v=0.0),
                ElmerTet4BoundaryCondition(3, potential_v=1.0, temperature_k=300.0),
            ),
        )


@pytest.mark.parametrize(
    ("stat_path", "heat_path", "tolerance", "message"),
    (
        (Path("relative.so"), Path("/heat.so"), 1.0e-12, "must be absolute"),
        (Path("/stat.so"), Path("relative.so"), 1.0e-12, "must be absolute"),
        (Path("/bad path/stat.so"), Path("/heat.so"), 1.0e-12, "whitespace"),
        (Path("/stat.so"), Path('/bad"heat.so'), 1.0e-12, "quotes"),
        (Path("/stat.so"), Path("/heat.so"), 0.0, "finite and positive"),
    ),
)
def test_tet4_renderer_rejects_unsafe_inputs(
    stat_path: Path,
    heat_path: Path,
    tolerance: float,
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        render_tet4_electrothermal_sif(
            _case(),
            stat_current_module=stat_path,
            heat_solve_module=heat_path,
            convergence_tolerance=tolerance,
        )
    if tolerance == 0.0:
        with pytest.raises(ContractError, match="requires an electrothermal case"):
            render_tet4_electrothermal_sif(  # type: ignore[arg-type]
                object(),
                stat_current_module=Path("/stat.so"),
                heat_solve_module=Path("/heat.so"),
                convergence_tolerance=1.0e-12,
            )


def test_tet4_topology_helpers_reject_invalid_cells_and_nonmanifold_faces() -> None:
    coordinates = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    inverted = _normalize_tetrahedra(coordinates, np.asarray(((0, 2, 1, 3),)))
    np.testing.assert_array_equal(inverted, ((0, 1, 2, 3),))
    cases = (
        (coordinates[:, :2], np.asarray(((0, 1, 2, 3),)), "coordinates"),
        (
            np.asarray(((np.nan, 0.0, 0.0), *coordinates[1:])),
            np.asarray(((0, 1, 2, 3),)),
            "coordinates must be finite",
        ),
        (coordinates, np.asarray(((0, 1, 2),)), "connectivity"),
        (coordinates, np.empty((0, 4), dtype=np.int64), "at least one cell"),
        (coordinates, np.asarray(((0, 1, 2, 4),)), "out-of-range"),
        (coordinates, np.asarray(((0, 1, 2, 2),)), "repeated"),
        (
            np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0))),
            np.asarray(((0, 1, 2, 3),)),
            "degenerate",
        ),
    )
    for points, cells, message in cases:
        with pytest.raises(ContractError, match=message):
            _normalize_tetrahedra(points, cells)
    with pytest.raises(ContractError, match="non-manifold"):
        _external_faces(
            np.asarray(
                (
                    (0, 1, 2, 3),
                    (0, 2, 1, 4),
                    (0, 1, 2, 5),
                ),
                dtype=np.int64,
            )
        )
