"""Deterministic SIF lowering for the supported Elmer steady-current slice."""

from __future__ import annotations

from pathlib import Path

from femx.backends._steady_current import ValidatedSteadyCurrent, resolve_current_scalar
from femx.backends.elmer.case import ElmerMeshDeck, _format_real
from femx.core.errors import ContractError
from femx.core.parameters import ParameterValues


def _encoded_module_path(path: Path) -> str:
    if not path.is_absolute():
        raise ContractError("Elmer StatCurrentSolve procedure path must be absolute")
    encoded = str(path)
    if any(character.isspace() for character in encoded) or '"' in encoded:
        raise ContractError(
            "Elmer StatCurrentSolve procedure path cannot contain whitespace or quotes"
        )
    return encoded


def render_steady_current_sif(
    problem: ValidatedSteadyCurrent,
    mesh: ElmerMeshDeck,
    parameters: ParameterValues,
    *,
    convergence_tolerance: float,
    stat_current_module: Path,
) -> str:
    """Render the fixed typed current-conduction subset; raw SIF is never accepted."""

    encoded_module = _encoded_module_path(stat_current_module)
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
        "  Steady State Max Iterations = 2",
        "  Output Intervals = 1",
        '  Output File = "femx.result"',
        "  Output File Final Only = Logical True",
        "  Binary Output = Logical False",
        '  Output Variable 1 = String "Potential"',
        "  Omit Unchanged Variables In Output = Logical False",
        '  Post File = "femx.vtu"',
        "  vtu: Binary Output = Logical True",
        "  vtu: No Fileindex = Logical True",
        "  vtu: Save Bulk Only = Logical True",
        "End",
        "",
    ]
    for body_id in range(1, len(problem.region_cells) + 1):
        lines.extend(
            (
                f"Body {body_id}",
                f"  Target Bodies(1) = {body_id}",
                "  Equation = 1",
                f"  Material = {body_id}",
                f"  Body Force = {body_id}",
                "End",
                "",
            )
        )
    lines.extend(
        (
            "Equation 1",
            "  Active Solvers(1) = 1",
            "End",
            "",
            "Solver 1",
            '  Equation = "Static Current"',
            f'  Procedure = File "{encoded_module}" "StatCurrentSolver"',
            '  Variable = "Potential"',
            "  Variable Dofs = 1",
            "  Linear System Solver = Direct",
            "  Linear System Direct Method = UMFPACK",
            "  Linear System Abort Not Converged = Logical True",
            "  Optimize Bandwidth = Logical False",
            "  Nonlinear System Max Iterations = 1",
            f"  Steady State Convergence Tolerance = {_format_real(convergence_tolerance)}",
            "End",
            "",
        )
    )
    for body_id, (conductivity, source) in enumerate(
        zip(problem.region_conductivity, problem.region_source, strict=True),
        start=1,
    ):
        conductivity_value = resolve_current_scalar(
            conductivity,
            parameters,
            strictly_positive=True,
        )
        source_value = resolve_current_scalar(source, parameters)
        lines.extend(
            (
                f"Material {body_id}",
                f"  Electric Conductivity = Real {_format_real(conductivity_value)}",
                "End",
                "",
                f"Body Force {body_id}",
                f"  Current Source = Real {_format_real(source_value)}",
                "End",
                "",
            )
        )
    boundary_number = 1
    for boundary_id, value in zip(
        mesh.essential_boundary_ids,
        problem.potential_values,
        strict=True,
    ):
        lines.extend(
            (
                f"Boundary Condition {boundary_number}",
                f"  Target Boundaries(1) = {boundary_id}",
                f"  Potential = Real {_format_real(resolve_current_scalar(value, parameters))}",
                "End",
                "",
            )
        )
        boundary_number += 1
    for boundary_id, value in zip(
        mesh.natural_boundary_ids,
        problem.flux_values,
        strict=True,
    ):
        lines.extend(
            (
                f"Boundary Condition {boundary_number}",
                f"  Target Boundaries(1) = {boundary_id}",
                "  Current Density BC = Logical True",
                f"  Current Density = Real {_format_real(resolve_current_scalar(value, parameters))}",
                "End",
                "",
            )
        )
        boundary_number += 1
    return "\n".join(lines)
