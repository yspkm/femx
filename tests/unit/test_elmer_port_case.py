from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tests.support import structured_unit_square_mesh

from femx.backends._port_eigenmode import validate_port_eigenmode_problem
from femx.backends.elmer.case import lower_tagged_triangle_mesh
from femx.backends.elmer.port_case import render_port_eigenmode_sif
from femx.core.errors import ContractError
from femx.core.problem import Problem
from femx.mesh import OrientationMap
from femx.physics import (
    VACUUM_SPEED_OF_LIGHT_M_PER_S,
    IsotropicOpticalRegion,
    PerfectElectricBoundary,
    PortEigenmode,
)

pytestmark = pytest.mark.unit


def _case():
    mesh = structured_unit_square_mesh(1)
    cells = np.asarray(mesh.topology.connectivity)
    edges = cells[:, ((0, 1), (1, 2), (2, 0))]
    signs = np.where(edges[:, :, 0] < edges[:, :, 1], 1, -1).astype(np.int8)
    mesh = replace(mesh, orientation=OrientationMap(edge_signs=signs))
    physics = PortEigenmode(
        regions=(IsotropicOpticalRegion("domain", 12.0, 2.0),),
        perfect_electric_boundaries=tuple(
            PerfectElectricBoundary(name) for name in ("bottom", "right", "top", "left")
        ),
        frequency_hz=VACUUM_SPEED_OF_LIGHT_M_PER_S / 1.55e-6,
        eigenmode_count=6,
        selected_mode_index=2,
        target_power_w=1.0,
    )
    problem = Problem("port-case", mesh, physics)
    validated = validate_port_eigenmode_problem(problem)
    deck = lower_tagged_triangle_mesh(
        mesh,
        region_tags=("domain",),
        boundary_tags=("bottom", "right", "top", "left"),
    )
    return validated, deck


def _render(**overrides: object) -> str:
    validated, deck = _case()
    kwargs: dict[str, object] = {
        "convergence_tolerance": 1.0e-10,
        "em_port_module": Path("/locked/elmer/lib/EMPort.so"),
        "result_output_module": Path("/locked/elmer/lib/ResultOutputSolve.so"),
        "save_data_module": Path("/locked/elmer/lib/SaveData.so"),
    }
    kwargs.update(overrides)
    return render_port_eigenmode_sif(validated, deck, **kwargs)  # type: ignore[arg-type]


def test_generic_triangle_lowering_preserves_complete_tag_partitions() -> None:
    _validated, deck = _case()

    assert deck.body_ids == (1,)
    assert deck.boundary_ids == (1, 2, 3, 4)
    assert deck.header == "4 2 4\n2\n202 4\n303 2\n"
    assert deck.elements == "1 1 303 1 2 4\n2 1 303 1 4 3\n"
    assert deck.edge_nodes == ((0, 1), (1, 3), (0, 3), (2, 3), (0, 2))
    assert {int(line.split()[1]) for line in deck.boundary.splitlines()} == {1, 2, 3, 4}


def test_port_sif_renderer_freezes_mixed_formulation_and_output_contract() -> None:
    sif = _render()

    assert 'Mesh DB "." "mesh"' in sif
    assert "Coordinate System = Cartesian 2D" in sif
    assert "Permittivity of Vacuum = Real 8.85418781280000060e-12" in sif
    assert "Permeability of Vacuum = Real 1.25663706212005479e-06" in sif
    assert 'Procedure = File "/locked/elmer/lib/EMPort.so" "EMPortSolver"' in sif
    assert "Variable Output = Logical True" in sif
    assert "Use Piola Transform = Logical True" in sif
    assert "Quadratic Approximation = Logical False" in sif
    assert "Second Kind Basis = Logical False" in sif
    assert "Calculate Nodal Field = Logical True" in sif
    assert "Calculate Impedance = Logical True" in sif
    assert "Eigenfunction Index = Integer 3" in sif
    assert "Eigen System Values = Integer 6" in sif
    assert 'Eigen System Sorting = String "smallest real part"' in sif
    assert "Eigen System Compute Residuals = Logical True" in sif
    assert "post: Linear System Direct Method = UMFPACK" in sif
    assert 'Procedure = File "/locked/elmer/lib/ResultOutputSolve.so" "ResultOutputSolver"' in sif
    assert 'Vector Field 1 = String "EF2D Re"' in sif
    assert 'Vector Field 2 = String "EF2D Im"' in sif
    assert 'Procedure = File "/locked/elmer/lib/SaveData.so" "SaveScalars"' in sif
    assert "Relative Permittivity = Real 1.20000000000000000e+01" in sif
    assert "Relative Reluctivity = Real 5.00000000000000000e-01" in sif
    assert "Target Boundaries(4) = 1 2 3 4" in sif
    assert "Port Ground = Logical True" in sif
    assert sif.endswith("\n")


@pytest.mark.parametrize("tolerance", [0.0, float("inf")])
def test_port_sif_renderer_rejects_invalid_eigen_tolerance(tolerance: float) -> None:
    with pytest.raises(ContractError, match="convergence tolerance"):
        _render(convergence_tolerance=tolerance)


@pytest.mark.parametrize(
    ("key", "path", "label"),
    [
        ("em_port_module", Path("relative/EMPort.so"), "EMPort"),
        ("result_output_module", Path("/with space/ResultOutputSolve.so"), "ResultOutput"),
        ("save_data_module", Path('/bad"quote/SaveData.so'), "SaveData"),
    ],
)
def test_port_sif_renderer_requires_unambiguous_absolute_module_paths(
    key: str, path: Path, label: str
) -> None:
    with pytest.raises(ContractError, match=label):
        _render(**{key: path})


def test_port_sif_renderer_rejects_mesh_material_and_boundary_count_drift() -> None:
    validated, deck = _case()
    common = {
        "convergence_tolerance": 1.0e-10,
        "em_port_module": Path("/locked/EMPort.so"),
        "result_output_module": Path("/locked/ResultOutputSolve.so"),
        "save_data_module": Path("/locked/SaveData.so"),
    }
    with pytest.raises(ContractError, match="body count"):
        render_port_eigenmode_sif(
            validated,
            replace(deck, body_ids=()),
            **common,
        )
    with pytest.raises(ContractError, match="boundary count"):
        render_port_eigenmode_sif(
            validated,
            replace(deck, boundary_ids=(1,)),
            **common,
        )
