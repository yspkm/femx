"""Shared validation and coefficient binding for steady-heat backends.

This module deliberately lives above the solver-neutral core.  It may use NumPy, but neither the
core contracts nor importing :mod:`femx` depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from femx.backends._scalar_h1 import tag_ids, validate_scalar_h1_mesh
from femx.core.errors import ContractError
from femx.core.parameters import ParameterReference, ParameterValues
from femx.core.problem import Problem
from femx.mesh import Mesh
from femx.physics._scalar import ScalarCoefficient
from femx.physics.steady_heat import SteadyHeat

CONDUCTIVITY_UNIT: Final = "W/(m*K)"
SOURCE_UNIT: Final = "W/m^3"
TEMPERATURE_UNIT: Final = "K"
HEAT_FLUX_UNIT: Final = "W/m^2"


@dataclass(frozen=True, slots=True)
class ValidatedSteadyHeat:
    """Backend-independent numerical lowering of the supported H1 slice."""

    coordinates: np.ndarray
    cells: np.ndarray
    boundary_facets: np.ndarray
    region_cells: tuple[np.ndarray, ...]
    region_conductivity: tuple[ScalarCoefficient, ...]
    region_source: tuple[ScalarCoefficient, ...]
    temperature_facets: tuple[np.ndarray, ...]
    temperature_values: tuple[ScalarCoefficient, ...]
    flux_facets: tuple[np.ndarray, ...]
    flux_values: tuple[ScalarCoefficient, ...]
    dirichlet_nodes: np.ndarray
    dirichlet_values: tuple[ScalarCoefficient, ...]
    free_nodes: np.ndarray


def _parameter_units(physics: SteadyHeat) -> dict[str, str]:
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
        register(region.conductivity, CONDUCTIVITY_UNIT)
        register(region.volumetric_heat_source, SOURCE_UNIT)
    for temperature_boundary in physics.temperature_boundaries:
        register(temperature_boundary.temperature, TEMPERATURE_UNIT)
    for flux_boundary in physics.heat_flux_boundaries:
        register(flux_boundary.heat_flux, HEAT_FLUX_UNIT)
    return expected


def _validate_parameter_schema(problem: Problem, physics: SteadyHeat) -> None:
    expected = _parameter_units(physics)
    actual = {spec.name: spec for spec in problem.parameters.specs}
    if set(expected) != set(actual):
        raise ContractError(
            "steady-heat coefficient parameters do not match the problem schema: "
            f"expected={sorted(expected)}, actual={sorted(actual)}"
        )
    for name, unit in expected.items():
        spec = actual[name]
        if spec.unit != unit or spec.shape:
            raise ContractError(
                f"parameter {name!r} must be a scalar with unit {unit!r}, "
                f"got unit={spec.unit!r}, shape={spec.shape}"
            )


def resolve_scalar(
    value: ScalarCoefficient,
    parameters: ParameterValues,
    *,
    strictly_positive: bool = False,
) -> float:
    """Resolve one finite real scalar under the shared backend contract."""

    resolved = parameters[value.name] if isinstance(value, ParameterReference) else value
    raw = np.asarray(resolved)
    if raw.shape or raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise ContractError("steady-heat coefficients must resolve to finite real scalars")
    numeric = float(raw)
    if strictly_positive and numeric <= 0.0:
        raise ContractError("thermal conductivity must resolve to a positive value")
    return numeric


def validate_steady_heat_problem(problem: Problem) -> ValidatedSteadyHeat:
    """Validate the exact numerical subset shared by JAX and Elmer."""

    if not isinstance(problem.mesh, Mesh):
        raise ContractError("steady heat requires the concrete femx Mesh contract")
    if not isinstance(problem.physics, SteadyHeat):
        raise ContractError("steady heat requires a SteadyHeat physics specification")
    mesh = problem.mesh
    physics = problem.physics
    physics.validate()
    _validate_parameter_schema(problem, physics)
    coordinates, cells, facets = validate_scalar_h1_mesh(mesh, physics_label="steady heat")

    region_cells: list[np.ndarray] = []
    cell_owners = np.zeros(cells.shape[0], dtype=np.int64)
    for region in physics.regions:
        ids = tag_ids(mesh, region.tag, dimension=2, upper_bound=cells.shape[0])
        cell_owners[ids] += 1
        region_cells.append(ids)
    if np.any(cell_owners != 1):
        raise ContractError("thermal region tags must partition every cell exactly once")

    temperature_facets: list[np.ndarray] = []
    temperature_owners = np.zeros(facets.shape[0], dtype=np.int64)
    node_values: dict[int, ScalarCoefficient] = {}
    for temperature_boundary in physics.temperature_boundaries:
        ids = tag_ids(mesh, temperature_boundary.tag, dimension=1, upper_bound=facets.shape[0])
        temperature_owners[ids] += 1
        temperature_facets.append(ids)
        for node in np.unique(facets[ids].reshape(-1)):
            node_id = int(node)
            previous = node_values.setdefault(node_id, temperature_boundary.temperature)
            if previous != temperature_boundary.temperature:
                raise ContractError(f"temperature boundary values conflict at mesh node {node_id}")
    if np.any(temperature_owners > 1):
        raise ContractError("temperature boundary tags cannot overlap on a facet")

    flux_facets: list[np.ndarray] = []
    flux_owners = np.zeros(facets.shape[0], dtype=np.int64)
    for flux_boundary in physics.heat_flux_boundaries:
        ids = tag_ids(mesh, flux_boundary.tag, dimension=1, upper_bound=facets.shape[0])
        flux_owners[ids] += 1
        flux_facets.append(ids)
    if np.any(flux_owners > 1):
        raise ContractError("heat-flux boundary tags cannot overlap on a facet")
    if np.any((temperature_owners > 0) & (flux_owners > 0)):
        raise ContractError("a boundary facet cannot carry both temperature and heat flux")

    dirichlet_nodes = np.asarray(sorted(node_values), dtype=np.int64)
    free_nodes = np.setdiff1d(
        np.arange(coordinates.shape[0], dtype=np.int64),
        dirichlet_nodes,
        assume_unique=True,
    )
    return ValidatedSteadyHeat(
        coordinates=coordinates,
        cells=cells,
        boundary_facets=facets,
        region_cells=tuple(region_cells),
        region_conductivity=tuple(region.conductivity for region in physics.regions),
        region_source=tuple(region.volumetric_heat_source for region in physics.regions),
        temperature_facets=tuple(temperature_facets),
        temperature_values=tuple(
            boundary.temperature for boundary in physics.temperature_boundaries
        ),
        flux_facets=tuple(flux_facets),
        flux_values=tuple(boundary.heat_flux for boundary in physics.heat_flux_boundaries),
        dirichlet_nodes=dirichlet_nodes,
        dirichlet_values=tuple(node_values[int(node)] for node in dirichlet_nodes),
        free_nodes=free_nodes,
    )
