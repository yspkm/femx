"""Shared validation and coefficient binding for steady-current backends."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from femx.backends._scalar_h1 import tag_ids, validate_scalar_h1_mesh
from femx.core.errors import ContractError
from femx.core.parameters import ParameterReference, ParameterValues
from femx.core.problem import Problem
from femx.mesh import Mesh
from femx.physics._scalar import ScalarCoefficient
from femx.physics.steady_current import SteadyCurrent

ELECTRIC_CONDUCTIVITY_UNIT: Final = "S/m"
CURRENT_SOURCE_UNIT: Final = "A/m^3"
POTENTIAL_UNIT: Final = "V"
CURRENT_FLUX_UNIT: Final = "A/m^2"
ELECTRIC_FIELD_UNIT: Final = "V/m"
CURRENT_DENSITY_UNIT: Final = "A/m^2"
JOULE_HEAT_DENSITY_UNIT: Final = "W/m^3"
POWER_PER_DEPTH_UNIT: Final = "W/m"


@dataclass(frozen=True, slots=True)
class ValidatedSteadyCurrent:
    """Backend-independent numerical lowering of the supported current H1 slice."""

    coordinates: np.ndarray
    cells: np.ndarray
    boundary_facets: np.ndarray
    region_cells: tuple[np.ndarray, ...]
    region_conductivity: tuple[ScalarCoefficient, ...]
    region_source: tuple[ScalarCoefficient, ...]
    potential_facets: tuple[np.ndarray, ...]
    potential_values: tuple[ScalarCoefficient, ...]
    flux_facets: tuple[np.ndarray, ...]
    flux_values: tuple[ScalarCoefficient, ...]
    dirichlet_nodes: np.ndarray
    dirichlet_values: tuple[ScalarCoefficient, ...]
    free_nodes: np.ndarray


@dataclass(frozen=True, slots=True)
class CurrentPostprocessing:
    """Independent NumPy reconstruction used to audit an external nodal potential."""

    electric_field: np.ndarray
    current_density: np.ndarray
    joule_heat_density: np.ndarray
    relative_backward_error: float
    joule_power: float
    variational_input_power: float
    energy_balance_relative_error: float


def _parameter_units(physics: SteadyCurrent) -> dict[str, str]:
    expected: dict[str, str] = {}

    def register(value: ScalarCoefficient, unit: str) -> None:
        if not isinstance(value, ParameterReference):
            return
        previous = expected.setdefault(value.name, unit)
        if previous != unit:
            raise ContractError(
                f"parameter {value.name!r} is used with incompatible units {previous!r} and "
                f"{unit!r}"
            )

    for region in physics.regions:
        register(region.electric_conductivity, ELECTRIC_CONDUCTIVITY_UNIT)
        register(region.volumetric_current_source, CURRENT_SOURCE_UNIT)
    for potential_boundary in physics.potential_boundaries:
        register(potential_boundary.potential, POTENTIAL_UNIT)
    for flux_boundary in physics.current_flux_boundaries:
        register(flux_boundary.current_density, CURRENT_FLUX_UNIT)
    return expected


def _validate_parameter_schema(problem: Problem, physics: SteadyCurrent) -> None:
    expected = _parameter_units(physics)
    actual = {spec.name: spec for spec in problem.parameters.specs}
    if set(expected) != set(actual):
        raise ContractError(
            "steady-current coefficient parameters do not match the problem schema: "
            f"expected={sorted(expected)}, actual={sorted(actual)}"
        )
    for name, unit in expected.items():
        spec = actual[name]
        if spec.unit != unit or spec.shape:
            raise ContractError(
                f"parameter {name!r} must be a scalar with unit {unit!r}, "
                f"got unit={spec.unit!r}, shape={spec.shape}"
            )


def resolve_current_scalar(
    value: ScalarCoefficient,
    parameters: ParameterValues,
    *,
    strictly_positive: bool = False,
) -> float:
    """Resolve one finite real scalar under the current-conduction contract."""

    resolved = parameters[value.name] if isinstance(value, ParameterReference) else value
    raw = np.asarray(resolved)
    if raw.shape or raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise ContractError("steady-current coefficients must resolve to finite real scalars")
    numeric = float(raw)
    if strictly_positive and numeric <= 0.0:
        raise ContractError("electric conductivity must resolve to a positive value")
    return numeric


def _resolved_arrays(
    problem: ValidatedSteadyCurrent,
    parameters: ParameterValues,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell_conductivity = np.zeros(problem.cells.shape[0], dtype=np.float64)
    cell_source = np.zeros_like(cell_conductivity)
    for ids, conductivity, source in zip(
        problem.region_cells,
        problem.region_conductivity,
        problem.region_source,
        strict=True,
    ):
        cell_conductivity[ids] = resolve_current_scalar(
            conductivity,
            parameters,
            strictly_positive=True,
        )
        cell_source[ids] = resolve_current_scalar(source, parameters)
    facet_load = np.zeros(problem.boundary_facets.shape[0], dtype=np.float64)
    for ids, value in zip(problem.flux_facets, problem.flux_values, strict=True):
        facet_load[ids] = resolve_current_scalar(value, parameters)
    return cell_conductivity, cell_source, facet_load


def postprocess_current_potential(
    problem: ValidatedSteadyCurrent,
    parameters: ParameterValues,
    potential: np.ndarray,
) -> CurrentPostprocessing:
    """Derive physical current fields and variational audits without JAX or Elmer APIs."""

    potential = np.asarray(potential, dtype=np.float64)
    expected_shape = (problem.coordinates.shape[0],)
    if potential.shape != expected_shape or not np.isfinite(potential).all():
        raise ContractError("steady-current potential must be one finite scalar at every mesh node")
    cell_conductivity, cell_source, facet_load = _resolved_arrays(problem, parameters)

    points = problem.coordinates[problem.cells]
    first = points[:, 1, :] - points[:, 0, :]
    second = points[:, 2, :] - points[:, 0, :]
    determinant = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    areas = 0.5 * np.abs(determinant)
    gradient_numerators = np.stack(
        (
            np.stack(
                (points[:, 1, 1] - points[:, 2, 1], points[:, 2, 0] - points[:, 1, 0]), axis=1
            ),
            np.stack(
                (points[:, 2, 1] - points[:, 0, 1], points[:, 0, 0] - points[:, 2, 0]), axis=1
            ),
            np.stack(
                (points[:, 0, 1] - points[:, 1, 1], points[:, 1, 0] - points[:, 0, 0]), axis=1
            ),
        ),
        axis=1,
    )
    basis_gradients = gradient_numerators / determinant[:, None, None]
    potential_gradient = np.einsum(
        "ci,cid->cd",
        potential[problem.cells],
        basis_gradients,
    )
    electric_field = -potential_gradient
    current_density = cell_conductivity[:, None] * electric_field
    joule_heat_density = np.einsum("cd,cd->c", current_density, electric_field)

    local_stiffness = (
        cell_conductivity[:, None, None]
        * areas[:, None, None]
        * np.einsum("cid,cjd->cij", basis_gradients, basis_gradients)
    )
    local_load = np.broadcast_to(
        cell_source[:, None] * areas[:, None] / 3.0,
        problem.cells.shape,
    )
    node_count = problem.coordinates.shape[0]
    rows = np.repeat(problem.cells, 3, axis=1).reshape(-1)
    columns = np.tile(problem.cells, (1, 3)).reshape(-1)
    stiffness = np.zeros((node_count, node_count), dtype=np.float64)
    np.add.at(stiffness, (rows, columns), local_stiffness.reshape(-1))
    load = np.zeros(node_count, dtype=np.float64)
    np.add.at(load, problem.cells.reshape(-1), local_load.reshape(-1))
    facet_points = problem.coordinates[problem.boundary_facets]
    facet_lengths = np.linalg.norm(facet_points[:, 1, :] - facet_points[:, 0, :], axis=1)
    local_facet_load = facet_load * facet_lengths / 2.0
    np.add.at(load, problem.boundary_facets.reshape(-1), np.repeat(local_facet_load, 2))

    residual = stiffness @ potential - load
    free_residual = residual[problem.free_nodes]
    free_operator = stiffness[problem.free_nodes, :]
    free_load = load[problem.free_nodes]
    residual_norm = float(np.linalg.norm(free_residual))
    backward_scale = float(
        np.linalg.norm(free_operator) * np.linalg.norm(potential) + np.linalg.norm(free_load)
    )
    if backward_scale > 0.0:
        relative_backward_error = residual_norm / backward_scale
    else:
        relative_backward_error = 0.0 if residual_norm == 0.0 else math.inf

    joule_power = float(np.sum(areas * joule_heat_density))
    reaction_power = float(
        np.vdot(potential[problem.dirichlet_nodes], residual[problem.dirichlet_nodes])
    )
    variational_input_power = float(np.vdot(potential, load)) + reaction_power
    energy_difference = abs(joule_power - variational_input_power)
    energy_scale = abs(joule_power) + abs(variational_input_power)
    if energy_scale > 0.0:
        energy_balance_relative_error = energy_difference / energy_scale
    else:
        energy_balance_relative_error = 0.0 if energy_difference == 0.0 else math.inf
    return CurrentPostprocessing(
        electric_field=electric_field,
        current_density=current_density,
        joule_heat_density=joule_heat_density,
        relative_backward_error=relative_backward_error,
        joule_power=joule_power,
        variational_input_power=variational_input_power,
        energy_balance_relative_error=energy_balance_relative_error,
    )


def validate_steady_current_problem(problem: Problem) -> ValidatedSteadyCurrent:
    """Validate the exact current-conduction subset shared by JAX and Elmer."""

    if not isinstance(problem.mesh, Mesh):
        raise ContractError("steady current requires the concrete femx Mesh contract")
    if not isinstance(problem.physics, SteadyCurrent):
        raise ContractError("steady current requires a SteadyCurrent physics specification")
    mesh = problem.mesh
    physics = problem.physics
    physics.validate()
    _validate_parameter_schema(problem, physics)
    coordinates, cells, facets = validate_scalar_h1_mesh(mesh, physics_label="steady current")

    region_cells: list[np.ndarray] = []
    cell_owners = np.zeros(cells.shape[0], dtype=np.int64)
    for region in physics.regions:
        ids = tag_ids(mesh, region.tag, dimension=2, upper_bound=cells.shape[0])
        cell_owners[ids] += 1
        region_cells.append(ids)
    if np.any(cell_owners != 1):
        raise ContractError("conductive region tags must partition every cell exactly once")

    potential_facets: list[np.ndarray] = []
    potential_owners = np.zeros(facets.shape[0], dtype=np.int64)
    node_values: dict[int, ScalarCoefficient] = {}
    for potential_boundary in physics.potential_boundaries:
        ids = tag_ids(mesh, potential_boundary.tag, dimension=1, upper_bound=facets.shape[0])
        potential_owners[ids] += 1
        potential_facets.append(ids)
        for node in np.unique(facets[ids].reshape(-1)):
            node_id = int(node)
            previous = node_values.setdefault(node_id, potential_boundary.potential)
            if previous != potential_boundary.potential:
                raise ContractError(f"potential boundary values conflict at mesh node {node_id}")
    if np.any(potential_owners > 1):
        raise ContractError("potential boundary tags cannot overlap on a facet")

    flux_facets: list[np.ndarray] = []
    flux_owners = np.zeros(facets.shape[0], dtype=np.int64)
    for flux_boundary in physics.current_flux_boundaries:
        ids = tag_ids(mesh, flux_boundary.tag, dimension=1, upper_bound=facets.shape[0])
        flux_owners[ids] += 1
        flux_facets.append(ids)
    if np.any(flux_owners > 1):
        raise ContractError("current-flux boundary tags cannot overlap on a facet")
    if np.any((potential_owners > 0) & (flux_owners > 0)):
        raise ContractError("a boundary facet cannot carry both potential and current flux")

    dirichlet_nodes = np.asarray(sorted(node_values), dtype=np.int64)
    free_nodes = np.setdiff1d(
        np.arange(coordinates.shape[0], dtype=np.int64),
        dirichlet_nodes,
        assume_unique=True,
    )
    return ValidatedSteadyCurrent(
        coordinates=coordinates,
        cells=cells,
        boundary_facets=facets,
        region_cells=tuple(region_cells),
        region_conductivity=tuple(region.electric_conductivity for region in physics.regions),
        region_source=tuple(region.volumetric_current_source for region in physics.regions),
        potential_facets=tuple(potential_facets),
        potential_values=tuple(boundary.potential for boundary in physics.potential_boundaries),
        flux_facets=tuple(flux_facets),
        flux_values=tuple(boundary.current_density for boundary in physics.current_flux_boundaries),
        dirichlet_nodes=dirichlet_nodes,
        dirichlet_values=tuple(node_values[int(node)] for node in dirichlet_nodes),
        free_nodes=free_nodes,
    )
