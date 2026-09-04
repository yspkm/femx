"""Deterministic Elmer SIF lowering for the v1 electromagnetic port oracle."""

from __future__ import annotations

import math
from pathlib import Path

from femx.backends._port_eigenmode import (
    ResolvedPortMaterials,
    ValidatedPortEigenmode,
    resolve_port_materials,
)
from femx.backends.elmer.case import ElmerTriangleMeshDeck, _format_real
from femx.core.errors import ContractError
from femx.core.parameters import ParameterValues
from femx.physics.port_eigenmode import (
    VACUUM_PERMEABILITY_H_PER_M,
    VACUUM_PERMITTIVITY_F_PER_M,
)


def _module_path(path: Path, *, label: str) -> str:
    if not path.is_absolute():
        raise ContractError(f"Elmer {label} procedure path must be absolute")
    encoded = str(path)
    if any(character.isspace() for character in encoded) or '"' in encoded:
        raise ContractError(f"Elmer {label} procedure path cannot contain whitespace or quotes")
    return encoded


def render_port_eigenmode_sif(
    problem: ValidatedPortEigenmode,
    mesh: ElmerTriangleMeshDeck,
    *,
    convergence_tolerance: float,
    em_port_module: Path,
    result_output_module: Path,
    save_data_module: Path,
    materials: ResolvedPortMaterials | None = None,
) -> str:
    """Render the fixed lossless first-family EMPort subset; raw SIF is never accepted."""

    if not math.isfinite(convergence_tolerance) or convergence_tolerance <= 0.0:
        raise ContractError("Elmer port eigen convergence tolerance must be positive")
    if len(mesh.body_ids) != len(problem.region_cells):
        raise ContractError("Elmer port mesh body count does not match optical regions")
    if len(mesh.boundary_ids) != len(problem.pec_facets):
        raise ContractError("Elmer port mesh boundary count does not match PEC declarations")
    em_port = _module_path(em_port_module, label="EMPort")
    result_output = _module_path(result_output_module, label="ResultOutputSolve")
    save_data = _module_path(save_data_module, label="SaveData")
    angular_frequency = 2.0 * math.pi * problem.frequency_hz
    resolved_materials = (
        resolve_port_materials(problem, ParameterValues()) if materials is None else materials
    )

    lines = [
        "Header",
        "  CHECK KEYWORDS Warn",
        '  Mesh DB "." "mesh"',
        "End",
        "",
        "Simulation",
        "  Coordinate System = Cartesian 2D",
        "  Coordinate Mapping(3) = 1 2 3",
        "  Simulation Type = Steady State",
        "  Steady State Max Iterations = 1",
        "  Max Output Level = 7",
        "  Output Intervals = 1",
        '  Output File = "femx.result"',
        "  Output File Final Only = Logical True",
        "  Binary Output = Logical False",
        "  Omit Unchanged Variables In Output = Logical False",
        "End",
        "",
        "Constants",
        f"  Permittivity of Vacuum = Real {_format_real(VACUUM_PERMITTIVITY_F_PER_M)}",
        f"  Permeability of Vacuum = Real {_format_real(VACUUM_PERMEABILITY_H_PER_M)}",
        "End",
        "",
    ]
    for body_id in mesh.body_ids:
        lines.extend(
            (
                f"Body {body_id}",
                f"  Target Bodies(1) = {body_id}",
                "  Equation = 1",
                f"  Material = {body_id}",
                "End",
                "",
            )
        )
    lines.extend(
        (
            "Equation 1",
            "  Active Solvers(3) = 1 2 3",
            "End",
            "",
            "Solver 1",
            '  Equation = "Port mode"',
            f'  Procedure = File "{em_port}" "EMPortSolver"',
            "  Variable Output = Logical True",
            f"  Angular Frequency = Real {_format_real(angular_frequency)}",
            "  Use Piola Transform = Logical True",
            "  Quadratic Approximation = Logical False",
            "  Second Kind Basis = Logical False",
            "  Calculate Nodal Field = Logical True",
            "  Calculate Impedance = Logical True",
            f"  Eigenfunction Index = Integer {problem.selected_mode_index + 1}",
            "  Linear System Solver = Direct",
            "  Linear System Direct Method = UMFPACK",
            "  Linear System Abort Not Converged = Logical True",
            "  Optimize Bandwidth = Logical False",
            "  Eigen Analysis = Logical True",
            f"  Eigen System Values = Integer {problem.eigenmode_count}",
            '  Eigen System Sorting = String "smallest real part"',
            "  Eigen System Normalize To Unity = Logical True",
            "  Eigen System Shift Automatic = Logical True",
            "  Eigen System Compute Residuals = Logical True",
            f"  Eigen System Convergence Tolerance = Real {_format_real(convergence_tolerance)}",
            "  post: Linear System Solver = Direct",
            "  post: Linear System Direct Method = UMFPACK",
            "  post: Linear System Abort Not Converged = Logical True",
            "  post: Optimize Bandwidth = Logical False",
            "End",
            "",
            "Solver 2",
            '  Equation = "Projected mode output"',
            f'  Procedure = File "{result_output}" "ResultOutputSolver"',
            '  Output File Name = "femx-mode"',
            "  Vtu Format = Logical True",
            "  Save Geometry IDs = Logical True",
            "  Ascii Output = Logical False",
            "  Single Precision = Logical False",
            '  Vector Field 1 = String "EF2D Re"',
            '  Vector Field 2 = String "EF2D Im"',
            "  Eigen Analysis = Logical False",
            "End",
            "",
            "Solver 3",
            '  Equation = "Port scalar evidence"',
            f'  Procedure = File "{save_data}" "SaveScalars"',
            "  Save EigenValues = Logical True",
            f"  Show Norm Index = Integer {problem.selected_mode_index + 1}",
            "End",
            "",
        )
    )
    for body_id, (permittivity, permeability) in enumerate(
        zip(
            resolved_materials.relative_permittivity,
            resolved_materials.relative_permeability,
            strict=True,
        ),
        start=1,
    ):
        lines.extend(
            (
                f"Material {body_id}",
                f"  Relative Permittivity = Real {_format_real(permittivity)}",
                f"  Relative Reluctivity = Real {_format_real(1.0 / permeability)}",
                "End",
                "",
            )
        )
    target_ids = " ".join(str(boundary_id) for boundary_id in mesh.boundary_ids)
    lines.extend(
        (
            "Boundary Condition 1",
            f"  Target Boundaries({len(mesh.boundary_ids)}) = {target_ids}",
            "  Port Ground = Logical True",
            "End",
            "",
        )
    )
    return "\n".join(lines)
