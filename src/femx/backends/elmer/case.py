"""Deterministic native-mesh and SIF lowering for the supported Elmer heat slice."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from femx.backends._hcurl import TRIANGLE_LOCAL_EDGES
from femx.backends._steady_heat import ValidatedSteadyHeat, resolve_scalar
from femx.core.errors import ContractError
from femx.core.parameters import ParameterValues
from femx.mesh import CellType, Mesh


@dataclass(frozen=True, slots=True)
class ElmerMeshDeck:
    """Complete serial Elmer native-mesh input held in memory."""

    header: str
    nodes: str
    elements: str
    boundary: str
    essential_boundary_ids: tuple[int, ...]
    natural_boundary_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ElmerTriangleMeshDeck:
    """Complete tagged first-order triangle mesh without equation-specific BC roles."""

    header: str
    nodes: str
    elements: str
    boundary: str
    body_ids: tuple[int, ...]
    boundary_ids: tuple[int, ...]
    edge_nodes: tuple[tuple[int, int], ...]


def _format_real(value: float) -> str:
    if not np.isfinite(value):
        raise ContractError("Elmer input coefficients must be finite")
    return format(float(value), ".17e")


def _normalize_triangle_cells(coordinates: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Return the exact counter-clockwise connectivity emitted to Elmer."""

    normalized = np.asarray(cells, dtype=np.int64).copy()
    points = coordinates[normalized]
    first = points[:, 1, :] - points[:, 0, :]
    second = points[:, 2, :] - points[:, 0, :]
    determinant = first[:, 0] * second[:, 1] - second[:, 0] * first[:, 1]
    clockwise = determinant < 0.0
    normalized[clockwise, 1], normalized[clockwise, 2] = (
        normalized[clockwise, 2].copy(),
        normalized[clockwise, 1].copy(),
    )
    return normalized


def _first_encounter_triangle_edges(
    cells: np.ndarray,
) -> tuple[tuple[int, int], ...]:
    """Mirror Elmer's first-encounter global edge numbering for linear triangles."""

    edge_ids: dict[tuple[int, int], int] = {}
    for cell in cells:
        for left_local, right_local in TRIANGLE_LOCAL_EDGES:
            left = int(cell[left_local])
            right = int(cell[right_local])
            edge = (min(left, right), max(left, right))
            if edge not in edge_ids:
                edge_ids[edge] = len(edge_ids)
    return tuple(edge_ids)


def lower_scalar_h1_mesh(
    *,
    coordinates: np.ndarray,
    cells: np.ndarray,
    boundary_facets: np.ndarray,
    region_cells: tuple[np.ndarray, ...],
    essential_facets: tuple[np.ndarray, ...],
    natural_facets: tuple[np.ndarray, ...],
) -> ElmerMeshDeck:
    """Lower validated scalar-H1 arrays to Elmer's four native serial files."""

    normalized_cells = _normalize_triangle_cells(coordinates, cells)

    body_ids = np.zeros(normalized_cells.shape[0], dtype=np.int64)
    for body_id, cell_ids in enumerate(region_cells, start=1):
        body_ids[cell_ids] = body_id

    boundary_ids = np.zeros(boundary_facets.shape[0], dtype=np.int64)
    next_boundary_id = 1
    essential_boundary_ids: list[int] = []
    for facet_ids in essential_facets:
        boundary_ids[facet_ids] = next_boundary_id
        essential_boundary_ids.append(next_boundary_id)
        next_boundary_id += 1
    natural_boundary_ids: list[int] = []
    for facet_ids in natural_facets:
        boundary_ids[facet_ids] = next_boundary_id
        natural_boundary_ids.append(next_boundary_id)
        next_boundary_id += 1
    if np.any(boundary_ids == 0):
        boundary_ids[boundary_ids == 0] = next_boundary_id

    edge_parents: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for element_id, cell in enumerate(normalized_cells, start=1):
        for left, right in ((cell[0], cell[1]), (cell[1], cell[2]), (cell[2], cell[0])):
            left_node, right_node = sorted((int(left), int(right)))
            edge = (left_node, right_node)
            edge_parents.setdefault(edge, (element_id, (int(left), int(right))))

    node_lines = [
        f"{node_id} -1 {_format_real(float(point[0]))} "
        f"{_format_real(float(point[1]))} {_format_real(0.0)}"
        for node_id, point in enumerate(coordinates, start=1)
    ]
    element_lines = [
        f"{element_id} {int(body_ids[element_id - 1])} 303 "
        + " ".join(str(int(node) + 1) for node in cell)
        for element_id, cell in enumerate(normalized_cells, start=1)
    ]
    boundary_lines: list[str] = []
    for facet_id, facet in enumerate(boundary_facets, start=1):
        left_node, right_node = sorted((int(facet[0]), int(facet[1])))
        edge = (left_node, right_node)
        parent_id, oriented = edge_parents[edge]
        boundary_lines.append(
            f"{facet_id} {int(boundary_ids[facet_id - 1])} {parent_id} 0 202 "
            f"{oriented[0] + 1} {oriented[1] + 1}"
        )

    header = (
        f"{coordinates.shape[0]} {normalized_cells.shape[0]} {boundary_facets.shape[0]}\n"
        "2\n"
        f"202 {boundary_facets.shape[0]}\n"
        f"303 {normalized_cells.shape[0]}\n"
    )
    return ElmerMeshDeck(
        header=header,
        nodes="\n".join(node_lines) + "\n",
        elements="\n".join(element_lines) + "\n",
        boundary="\n".join(boundary_lines) + "\n",
        essential_boundary_ids=tuple(essential_boundary_ids),
        natural_boundary_ids=tuple(natural_boundary_ids),
    )


def lower_tagged_scalar_h1_mesh(
    mesh: Mesh,
    *,
    region_tags: tuple[str, ...],
    essential_boundary_tags: tuple[str, ...],
    natural_boundary_tags: tuple[str, ...] = (),
) -> ElmerMeshDeck:
    """Lower one complete semantic tag partition of a canonical triangle mesh."""

    if mesh.geometry.spatial_dimension != 2 or mesh.topology.cell_type is not CellType.TRIANGLE:
        raise ContractError("Elmer scalar-H1 tagged lowering requires a 2D triangle mesh")
    if mesh.boundary_facets is None:
        raise ContractError("Elmer scalar-H1 tagged lowering requires boundary segments")
    region_cells = _complete_tag_partition(
        mesh,
        names=region_tags,
        dimension=2,
        entity_count=mesh.topology.cell_count,
        label="region",
    )
    boundary_names = essential_boundary_tags + natural_boundary_tags
    boundary_groups = _complete_tag_partition(
        mesh,
        names=boundary_names,
        dimension=1,
        entity_count=mesh.boundary_facets.cell_count,
        label="boundary",
    )
    essential_count = len(essential_boundary_tags)
    return lower_scalar_h1_mesh(
        coordinates=np.asarray(mesh.geometry.coordinates, dtype=np.float64),
        cells=np.asarray(mesh.topology.connectivity, dtype=np.int64),
        boundary_facets=np.asarray(mesh.boundary_facets.connectivity, dtype=np.int64),
        region_cells=region_cells,
        essential_facets=boundary_groups[:essential_count],
        natural_facets=boundary_groups[essential_count:],
    )


def lower_tagged_triangle_mesh(
    mesh: Mesh,
    *,
    region_tags: tuple[str, ...],
    boundary_tags: tuple[str, ...],
) -> ElmerTriangleMeshDeck:
    """Lower complete region and external-boundary partitions without assigning physics roles."""

    if mesh.geometry.spatial_dimension != 2 or mesh.topology.cell_type is not CellType.TRIANGLE:
        raise ContractError("Elmer tagged triangle lowering requires a 2D triangle mesh")
    if mesh.boundary_facets is None:
        raise ContractError("Elmer tagged triangle lowering requires boundary segments")
    region_cells = _complete_tag_partition(
        mesh,
        names=region_tags,
        dimension=2,
        entity_count=mesh.topology.cell_count,
        label="region",
    )
    boundary_groups = _complete_tag_partition(
        mesh,
        names=boundary_tags,
        dimension=1,
        entity_count=mesh.boundary_facets.cell_count,
        label="boundary",
    )
    coordinates = np.asarray(mesh.geometry.coordinates, dtype=np.float64)
    cells = np.asarray(mesh.topology.connectivity, dtype=np.int64)
    normalized_cells = _normalize_triangle_cells(coordinates, cells)
    lowered = lower_scalar_h1_mesh(
        coordinates=coordinates,
        cells=cells,
        boundary_facets=np.asarray(mesh.boundary_facets.connectivity, dtype=np.int64),
        region_cells=region_cells,
        essential_facets=boundary_groups,
        natural_facets=(),
    )
    return ElmerTriangleMeshDeck(
        header=lowered.header,
        nodes=lowered.nodes,
        elements=lowered.elements,
        boundary=lowered.boundary,
        body_ids=tuple(range(1, len(region_tags) + 1)),
        boundary_ids=lowered.essential_boundary_ids,
        edge_nodes=_first_encounter_triangle_edges(normalized_cells),
    )


def _complete_tag_partition(
    mesh: Mesh,
    *,
    names: tuple[str, ...],
    dimension: int,
    entity_count: int,
    label: str,
) -> tuple[np.ndarray, ...]:
    if not names:
        raise ContractError(f"Elmer {label} tag partition must not be empty")
    if len(names) != len(set(names)):
        raise ContractError(f"Elmer {label} tag partition contains duplicate names")
    owner = np.full(entity_count, -1, dtype=np.int64)
    groups: list[np.ndarray] = []
    for group_index, name in enumerate(names):
        tag = mesh.tag(name)
        if tag.dimension != dimension:
            raise ContractError(
                f"Elmer {label} tag {name!r} must have dimension {dimension}, got {tag.dimension}"
            )
        entity_ids = np.asarray(tag.entity_ids, dtype=np.int64)
        if entity_ids.size == 0:
            raise ContractError(f"Elmer {label} tag {name!r} must not be empty")
        if np.any(entity_ids >= entity_count):
            raise ContractError(f"Elmer {label} tag {name!r} contains an out-of-range entity id")
        if np.any(owner[entity_ids] != -1):
            raise ContractError(f"Elmer {label} tag partition overlaps at {name!r}")
        owner[entity_ids] = group_index
        groups.append(entity_ids)
    missing_count = int(np.count_nonzero(owner == -1))
    if missing_count:
        raise ContractError(
            f"Elmer {label} tag partition leaves {missing_count} canonical entities unmapped"
        )
    return tuple(groups)


def lower_elmer_mesh(problem: ValidatedSteadyHeat) -> ElmerMeshDeck:
    """Lower the validated steady-heat specialization to an Elmer native mesh."""

    return lower_scalar_h1_mesh(
        coordinates=problem.coordinates,
        cells=problem.cells,
        boundary_facets=problem.boundary_facets,
        region_cells=problem.region_cells,
        essential_facets=problem.temperature_facets,
        natural_facets=problem.flux_facets,
    )


def render_steady_heat_sif(
    problem: ValidatedSteadyHeat,
    mesh: ElmerMeshDeck,
    parameters: ParameterValues,
    *,
    convergence_tolerance: float,
    heat_solve_module: Path,
) -> str:
    """Render the fixed, typed steady-heat subset; no raw user SIF is accepted."""

    if not heat_solve_module.is_absolute():
        raise ContractError("Elmer HeatSolve procedure path must be absolute")
    encoded_module = str(heat_solve_module)
    if any(character.isspace() for character in encoded_module) or '"' in encoded_module:
        raise ContractError("Elmer HeatSolve procedure path cannot contain whitespace or quotes")

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
        '  Output Variable 1 = String "Temperature"',
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
            '  Equation = "Heat Equation"',
            f'  Procedure = File "{encoded_module}" "HeatSolver"',
            '  Variable = "Temperature"',
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
        conductivity_value = resolve_scalar(conductivity, parameters, strictly_positive=True)
        source_value = resolve_scalar(source, parameters)
        lines.extend(
            (
                f"Material {body_id}",
                f"  Heat Conductivity = Real {_format_real(conductivity_value)}",
                "  Density = Real 1.00000000000000000e+00",
                "End",
                "",
                f"Body Force {body_id}",
                f"  Volumetric Heat Source = Real {_format_real(source_value)}",
                "End",
                "",
            )
        )
    boundary_number = 1
    for boundary_id, value in zip(
        mesh.essential_boundary_ids,
        problem.temperature_values,
        strict=True,
    ):
        lines.extend(
            (
                f"Boundary Condition {boundary_number}",
                f"  Target Boundaries(1) = {boundary_id}",
                f"  Temperature = Real {_format_real(resolve_scalar(value, parameters))}",
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
                f"  Heat Flux = Real {_format_real(resolve_scalar(value, parameters))}",
                "End",
                "",
            )
        )
        boundary_number += 1
    return "\n".join(lines)
