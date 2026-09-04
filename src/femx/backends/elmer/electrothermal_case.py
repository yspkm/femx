"""Typed Elmer lowering for self-consistent temperature-dependent Joule heating."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from femx.backends._steady_current import (
    ValidatedSteadyCurrent,
    resolve_current_scalar,
)
from femx.backends._steady_heat import ValidatedSteadyHeat, resolve_scalar
from femx.backends.elmer.case import ElmerMeshDeck, _format_real, lower_scalar_h1_mesh
from femx.core.errors import ContractError
from femx.core.parameters import ParameterReference, ParameterValues
from femx.physics._scalar import ScalarCoefficient
from femx.physics.steady_current import SteadyCurrent
from femx.workflows.electrothermal import SelfConsistentJouleHeating


@dataclass(frozen=True, slots=True)
class ElmerCoupledBody:
    """One combined electrical/thermal material partition in emitted body order."""

    body_id: int
    electrical_region_index: int
    thermal_region_index: int


@dataclass(frozen=True, slots=True)
class ElmerCoupledBoundary:
    """One emitted boundary ID and its at-most-one condition from each equation."""

    boundary_id: int
    potential_index: int | None
    current_flux_index: int | None
    temperature_index: int | None
    heat_flux_index: int | None


@dataclass(frozen=True, slots=True)
class ElmerCoupledMeshDeck:
    """Native mesh plus deterministic combined body and boundary assignments."""

    native: ElmerMeshDeck
    bodies: tuple[ElmerCoupledBody, ...]
    boundaries: tuple[ElmerCoupledBoundary, ...]


def _require_same_array(left: np.ndarray, right: np.ndarray, *, label: str) -> None:
    if left.shape != right.shape or not np.array_equal(left, right):
        raise ContractError(f"coupled Elmer lowering requires identical validated {label}")


def _partition_pairs(
    current: ValidatedSteadyCurrent,
    heat: ValidatedSteadyHeat,
) -> tuple[tuple[np.ndarray, ...], tuple[tuple[int, int], ...]]:
    cell_count = current.cells.shape[0]
    current_owner = np.full((cell_count,), -1, dtype=np.int64)
    thermal_owner = np.full((cell_count,), -1, dtype=np.int64)
    for index, ids in enumerate(current.region_cells):
        current_owner[ids] = index
    for index, ids in enumerate(heat.region_cells):
        thermal_owner[ids] = index
    pairs = np.stack((current_owner, thermal_owner), axis=1)
    unique_pairs: list[tuple[int, int]] = []
    grouped_cells: list[list[int]] = []
    pair_index: dict[tuple[int, int], int] = {}
    for cell_id, raw_pair in enumerate(pairs):
        pair = (int(raw_pair[0]), int(raw_pair[1]))
        group = pair_index.get(pair)
        if group is None:
            group = len(unique_pairs)
            pair_index[pair] = group
            unique_pairs.append(pair)
            grouped_cells.append([])
        grouped_cells[group].append(cell_id)
    return (
        tuple(np.asarray(ids, dtype=np.int64) for ids in grouped_cells),
        tuple(unique_pairs),
    )


def _boundary_signatures(
    boundary_count: int,
    groups: tuple[tuple[np.ndarray, ...], ...],
) -> tuple[tuple[np.ndarray, ...], tuple[tuple[int | None, ...], ...]]:
    flattened = tuple(group for category in groups for group in category)
    category_offsets = np.cumsum((0, *(len(category) for category in groups)))
    membership = np.zeros((boundary_count, len(flattened)), dtype=np.bool_)
    for group_index, facet_ids in enumerate(flattened):
        membership[facet_ids, group_index] = True

    signatures: list[tuple[bool, ...]] = []
    grouped_facets: list[list[int]] = []
    signature_index: dict[tuple[bool, ...], int] = {}
    for facet_id, row in enumerate(membership):
        signature = tuple(bool(value) for value in row)
        if not any(signature):
            continue
        group = signature_index.get(signature)
        if group is None:
            group = len(signatures)
            signature_index[signature] = group
            signatures.append(signature)
            grouped_facets.append([])
        grouped_facets[group].append(facet_id)

    assignments: list[tuple[int | None, ...]] = []
    for signature in signatures:
        categories: list[int | None] = []
        for category in range(len(groups)):
            start = int(category_offsets[category])
            stop = int(category_offsets[category + 1])
            active = [index - start for index in range(start, stop) if signature[index]]
            if len(active) > 1:
                raise ContractError("coupled boundary signature has overlapping equation data")
            categories.append(active[0] if active else None)
        assignments.append(tuple(categories))
    return (
        tuple(np.asarray(ids, dtype=np.int64) for ids in grouped_facets),
        tuple(assignments),
    )


def lower_self_consistent_mesh(
    current: ValidatedSteadyCurrent,
    heat: ValidatedSteadyHeat,
) -> ElmerCoupledMeshDeck:
    """Lower two same-mesh partitions without assuming equal material or boundary tags."""

    _require_same_array(current.coordinates, heat.coordinates, label="coordinates")
    _require_same_array(current.cells, heat.cells, label="cells")
    _require_same_array(current.boundary_facets, heat.boundary_facets, label="boundary facets")
    body_cells, body_pairs = _partition_pairs(current, heat)
    boundary_facets, boundary_assignments = _boundary_signatures(
        current.boundary_facets.shape[0],
        (
            current.potential_facets,
            current.flux_facets,
            heat.temperature_facets,
            heat.flux_facets,
        ),
    )
    native = lower_scalar_h1_mesh(
        coordinates=current.coordinates,
        cells=current.cells,
        boundary_facets=current.boundary_facets,
        region_cells=body_cells,
        essential_facets=boundary_facets,
        natural_facets=(),
    )
    bodies = tuple(
        ElmerCoupledBody(body_id, pair[0], pair[1])
        for body_id, pair in enumerate(body_pairs, start=1)
    )
    boundaries = tuple(
        ElmerCoupledBoundary(boundary_id, *assignment)
        for boundary_id, assignment in zip(
            native.essential_boundary_ids,
            boundary_assignments,
            strict=True,
        )
    )
    return ElmerCoupledMeshDeck(native=native, bodies=bodies, boundaries=boundaries)


def _encoded_module_path(path: Path, *, label: str) -> str:
    if not path.is_absolute():
        raise ContractError(f"Elmer {label} procedure path must be absolute")
    encoded = str(path)
    if any(character.isspace() for character in encoded) or '"' in encoded:
        raise ContractError(f"Elmer {label} procedure path cannot contain whitespace or quotes")
    return encoded


def _resolve_feedback_scalar(
    coefficient: ScalarCoefficient,
    parameters: ParameterValues,
) -> float:
    resolved = (
        parameters[coefficient.name] if isinstance(coefficient, ParameterReference) else coefficient
    )
    raw = np.asarray(resolved)
    if raw.shape or raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise ContractError("Elmer feedback coefficients must resolve to finite real scalars")
    return float(raw)


def render_self_consistent_sif(
    feedback: SelfConsistentJouleHeating,
    current: ValidatedSteadyCurrent,
    heat: ValidatedSteadyHeat,
    mesh: ElmerCoupledMeshDeck,
    current_parameters: ParameterValues,
    heat_parameters: ParameterValues,
    feedback_parameters: ParameterValues,
    *,
    stat_current_module: Path,
    heat_solve_module: Path,
    convergence_tolerance: float,
) -> str:
    """Render a closed two-solver SIF matching Elmer's nodal conductivity path."""

    stat_module = _encoded_module_path(stat_current_module, label="StatCurrentSolve")
    heat_module = _encoded_module_path(heat_solve_module, label="HeatSolve")
    if not np.isfinite(convergence_tolerance) or convergence_tolerance <= 0.0:
        raise ContractError("Elmer coupled convergence tolerance must be finite and positive")
    feedback.parameters.bind(feedback_parameters.values)
    physics = feedback.one_way.electrical_problem.physics
    assert isinstance(physics, SteadyCurrent)
    laws = {law.tag: law for law in feedback.conductivity_laws}
    policy = feedback.iteration
    initial_temperature = resolve_scalar(heat.temperature_values[0], heat_parameters)

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
        f"  Steady State Max Iterations = {policy.max_iterations}",
        f"  Steady State Min Iterations = {policy.minimum_iterations}",
        "  Output Intervals = 1",
        '  Output File = "femx.result"',
        "  Output File Final Only = Logical True",
        "  Binary Output = Logical False",
        '  Output Variable 1 = String "Potential"',
        '  Output Variable 2 = String "Temperature"',
        "  Omit Unchanged Variables In Output = Logical False",
        '  Post File = "femx.vtu"',
        "  vtu: Binary Output = Logical True",
        "  vtu: No Fileindex = Logical True",
        "  vtu: Save Bulk Only = Logical True",
        "End",
        "",
    ]
    for body in mesh.bodies:
        lines.extend(
            (
                f"Body {body.body_id}",
                f"  Target Bodies(1) = {body.body_id}",
                "  Equation = 1",
                f"  Material = {body.body_id}",
                f"  Body Force = {body.body_id}",
                "  Initial Condition = 1",
                "End",
                "",
            )
        )
    lines.extend(
        (
            "Equation 1",
            "  Active Solvers(2) = 1 2",
            "End",
            "",
            "Solver 1",
            '  Equation = "Static Current"',
            f'  Procedure = File "{stat_module}" "StatCurrentSolver"',
            '  Variable = "Potential"',
            "  Variable Dofs = 1",
            '  Exec Solver = "Always"',
            "  Linear System Solver = Direct",
            "  Linear System Direct Method = UMFPACK",
            "  Linear System Abort Not Converged = Logical True",
            "  Optimize Bandwidth = Logical False",
            "  Nonlinear System Max Iterations = 1",
            f"  Steady State Relaxation Factor = {_format_real(policy.potential_relaxation)}",
            f"  Steady State Convergence Tolerance = {_format_real(convergence_tolerance)}",
            "End",
            "",
            "Solver 2",
            '  Equation = "Heat Equation"',
            f'  Procedure = File "{heat_module}" "HeatSolver"',
            '  Variable = "Temperature"',
            "  Variable Dofs = 1",
            '  Exec Solver = "Always"',
            "  Linear System Solver = Direct",
            "  Linear System Direct Method = UMFPACK",
            "  Linear System Abort Not Converged = Logical True",
            "  Optimize Bandwidth = Logical False",
            "  Nonlinear System Max Iterations = 1",
            f"  Steady State Relaxation Factor = {_format_real(policy.temperature_relaxation)}",
            f"  Steady State Convergence Tolerance = {_format_real(convergence_tolerance)}",
            "End",
            "",
            "Initial Condition 1",
            f"  Temperature = Real {_format_real(initial_temperature)}",
            "End",
            "",
        )
    )

    for body in mesh.bodies:
        current_region = physics.regions[body.electrical_region_index]
        law = laws.get(current_region.tag)
        conductivity = resolve_current_scalar(
            current.region_conductivity[body.electrical_region_index],
            current_parameters,
            strictly_positive=True,
        )
        thermal_conductivity = resolve_scalar(
            heat.region_conductivity[body.thermal_region_index],
            heat_parameters,
            strictly_positive=True,
        )
        lines.append(f"Material {body.body_id}")
        if law is None:
            lines.append(f"  Electric Conductivity = Real {_format_real(conductivity)}")
        else:
            reference = _resolve_feedback_scalar(law.reference_temperature, feedback_parameters)
            coefficient = _resolve_feedback_scalar(
                law.temperature_coefficient,
                feedback_parameters,
            )
            expression = (
                f"{_format_real(conductivity)}/"
                f"(1.0+{_format_real(coefficient)}*(tx(0)-{_format_real(reference)}))"
            )
            lines.extend(
                (
                    "  Electric Conductivity = Variable Temperature",
                    f'    Real MATC "{expression}"',
                )
            )
        lines.extend(
            (
                f"  Heat Conductivity = Real {_format_real(thermal_conductivity)}",
                "  Density = Real 1.00000000000000000e+00",
                "End",
                "",
            )
        )
        current_source = resolve_current_scalar(
            current.region_source[body.electrical_region_index],
            current_parameters,
        )
        heat_source = resolve_scalar(
            heat.region_source[body.thermal_region_index],
            heat_parameters,
        )
        lines.extend(
            (
                f"Body Force {body.body_id}",
                f"  Current Source = Real {_format_real(current_source)}",
                f"  Volumetric Heat Source = Real {_format_real(heat_source)}",
                "  Joule Heat = Logical True",
                "End",
                "",
            )
        )

    for number, boundary in enumerate(mesh.boundaries, start=1):
        lines.extend(
            (
                f"Boundary Condition {number}",
                f"  Target Boundaries(1) = {boundary.boundary_id}",
            )
        )
        if boundary.potential_index is not None:
            value = resolve_current_scalar(
                current.potential_values[boundary.potential_index],
                current_parameters,
            )
            lines.append(f"  Potential = Real {_format_real(value)}")
        if boundary.current_flux_index is not None:
            value = resolve_current_scalar(
                current.flux_values[boundary.current_flux_index],
                current_parameters,
            )
            lines.extend(
                (
                    "  Current Density BC = Logical True",
                    f"  Current Density = Real {_format_real(value)}",
                )
            )
        if boundary.temperature_index is not None:
            value = resolve_scalar(
                heat.temperature_values[boundary.temperature_index],
                heat_parameters,
            )
            lines.append(f"  Temperature = Real {_format_real(value)}")
        if boundary.heat_flux_index is not None:
            value = resolve_scalar(
                heat.flux_values[boundary.heat_flux_index],
                heat_parameters,
            )
            lines.append(f"  Heat Flux = Real {_format_real(value)}")
        lines.extend(("End", ""))
    return "\n".join(lines)
