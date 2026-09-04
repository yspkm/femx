"""Shared numerical validation and normalization for port-eigenmode backends."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from femx.backends._hcurl import (
    CanonicalMixedPortDofPartition,
    canonical_mixed_port_dof_partition,
    canonical_triangle_edge_map,
)
from femx.backends._scalar_h1 import tag_ids, validate_planar_triangle_mesh
from femx.core.errors import BackendError, ContractError
from femx.core.parameters import ParameterReference, ParameterValues
from femx.core.problem import Problem
from femx.mesh import Mesh
from femx.physics._scalar import ScalarCoefficient
from femx.physics.port_eigenmode import PortEigenmode

ELECTRIC_FIELD_UNIT: Final = "V/m"
MAGNETIC_FIELD_UNIT: Final = "A/m"
PROPAGATION_CONSTANT_UNIT: Final = "rad/m"
RELATIVE_MATERIAL_UNIT: Final = "1"


@dataclass(frozen=True, slots=True)
class ValidatedPortEigenmode:
    """Exact lossless planar port problem shared by reference and native backends."""

    coordinates: np.ndarray
    cells: np.ndarray
    boundary_facets: np.ndarray
    edge_nodes: np.ndarray
    cell_edge_dofs: np.ndarray
    edge_signs: np.ndarray
    dof_partition: CanonicalMixedPortDofPartition
    region_cells: tuple[np.ndarray, ...]
    relative_permittivity: tuple[ScalarCoefficient, ...]
    relative_permeability: tuple[ScalarCoefficient, ...]
    pec_facets: tuple[np.ndarray, ...]
    frequency_hz: float
    eigenmode_count: int
    selected_mode_index: int
    target_power_w: float
    parameter_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResolvedPortMaterials:
    """Finite positive region values after solve-time parameter binding."""

    relative_permittivity: tuple[float, ...]
    relative_permeability: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class NormalizedProjectedMode:
    """Power-scaled, deterministically phased Cartesian nodal projection."""

    electric_field: np.ndarray
    anchor_node: int
    anchor_component: int
    phase_factor: complex
    amplitude_scale: float


@dataclass(frozen=True, slots=True)
class NormalizedProjectedElectromagneticMode:
    """Joint physical E/H normalization with one phase and amplitude factor."""

    electric_field: np.ndarray
    magnetic_field: np.ndarray
    anchor_node: int
    anchor_component: int
    phase_factor: complex
    amplitude_scale: float
    normalized_forward_power_w: float


def _validate_parameter_schema(problem: Problem, physics: PortEigenmode) -> None:
    expected: set[str] = set()
    for region in physics.regions:
        for value in (region.relative_permittivity, region.relative_permeability):
            if isinstance(value, ParameterReference):
                expected.add(value.name)
    actual = {spec.name: spec for spec in problem.parameters.specs}
    if expected != set(actual):
        raise ContractError(
            "port material parameters do not match the problem schema: "
            f"expected={sorted(expected)}, actual={sorted(actual)}"
        )
    for name in expected:
        spec = actual[name]
        if spec.unit != RELATIVE_MATERIAL_UNIT or spec.shape:
            raise ContractError(
                f"port material parameter {name!r} must be a dimensionless scalar, "
                f"got unit={spec.unit!r}, shape={spec.shape}"
            )


def _resolve_relative_material_scalar(
    value: ScalarCoefficient,
    parameters: ParameterValues,
) -> float:
    resolved = parameters[value.name] if isinstance(value, ParameterReference) else value
    raw = np.asarray(resolved)
    if raw.shape or raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise ContractError("port relative material values must resolve to finite real scalars")
    numeric = float(raw)
    if numeric <= 0.0:
        raise ContractError("port relative material values must resolve to positive values")
    return numeric


def resolve_port_materials(
    problem: ValidatedPortEigenmode,
    parameters: ParameterValues,
) -> ResolvedPortMaterials:
    """Resolve one exact parameter mapping to positive lossless region properties."""

    if set(parameters.values) != set(problem.parameter_names):
        raise ContractError(
            "port material parameter keys do not match the validated problem: "
            f"expected={sorted(problem.parameter_names)}, actual={sorted(parameters.values)}"
        )
    return ResolvedPortMaterials(
        relative_permittivity=tuple(
            _resolve_relative_material_scalar(value, parameters)
            for value in problem.relative_permittivity
        ),
        relative_permeability=tuple(
            _resolve_relative_material_scalar(value, parameters)
            for value in problem.relative_permeability
        ),
    )


def validate_port_eigenmode_problem(problem: Problem) -> ValidatedPortEigenmode:
    """Validate the complete v1 mesh, tag, orientation, and parameter contract."""

    if not isinstance(problem.mesh, Mesh):
        raise ContractError("port eigenmode requires the concrete femx Mesh contract")
    if not isinstance(problem.physics, PortEigenmode):
        raise ContractError("port eigenmode requires a PortEigenmode physics specification")
    mesh = problem.mesh
    physics = problem.physics
    physics.validate()
    _validate_parameter_schema(problem, physics)
    coordinates, cells, facets = validate_planar_triangle_mesh(
        mesh,
        physics_label="port eigenmode",
    )

    region_cells: list[np.ndarray] = []
    cell_owners = np.zeros(cells.shape[0], dtype=np.int64)
    for region in physics.regions:
        ids = tag_ids(mesh, region.tag, dimension=2, upper_bound=cells.shape[0])
        cell_owners[ids] += 1
        region_cells.append(ids)
    if np.any(cell_owners != 1):
        raise ContractError("optical region tags must partition every cell exactly once")

    pec_facets: list[np.ndarray] = []
    facet_owners = np.zeros(facets.shape[0], dtype=np.int64)
    for boundary in physics.perfect_electric_boundaries:
        ids = tag_ids(mesh, boundary.tag, dimension=1, upper_bound=facets.shape[0])
        facet_owners[ids] += 1
        pec_facets.append(ids)
    if np.any(facet_owners != 1):
        raise ContractError("PEC tags must partition every external boundary facet exactly once")

    edge_map = canonical_triangle_edge_map(cells, mesh.orientation.edge_signs)
    dof_partition = canonical_mixed_port_dof_partition(
        facets,
        edge_map,
        node_count=coordinates.shape[0],
    )

    return ValidatedPortEigenmode(
        coordinates=coordinates,
        cells=cells,
        boundary_facets=facets,
        edge_nodes=edge_map.edge_nodes,
        cell_edge_dofs=edge_map.cell_edge_dofs,
        edge_signs=edge_map.cell_edge_signs,
        dof_partition=dof_partition,
        region_cells=tuple(region_cells),
        relative_permittivity=tuple(region.relative_permittivity for region in physics.regions),
        relative_permeability=tuple(region.relative_permeability for region in physics.regions),
        pec_facets=tuple(pec_facets),
        frequency_hz=float(physics.frequency_hz),
        eigenmode_count=physics.eigenmode_count,
        selected_mode_index=physics.selected_mode_index,
        target_power_w=float(physics.target_power_w),
        parameter_names=problem.parameters.names,
    )


def normalize_projected_mode(
    electric_field: np.ndarray,
    *,
    raw_forward_power_w: float,
    target_forward_power_w: float,
) -> NormalizedProjectedMode:
    """Scale a nonzero complex nodal field and fix its otherwise arbitrary global phase."""

    values = np.asarray(electric_field)
    if values.ndim != 2 or values.shape[1] != 3 or values.dtype.kind != "c":
        raise BackendError("projected port electric field must be a complex (nodes, 3) array")
    if not np.isfinite(values).all():
        raise BackendError("projected port electric field contains non-finite values")
    for label, value in (
        ("raw forward power", raw_forward_power_w),
        ("target forward power", target_forward_power_w),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise BackendError(f"port {label} must be finite and positive")
    magnitudes = np.abs(values).reshape(-1)
    anchor_flat = int(np.argmax(magnitudes))
    anchor_magnitude = float(magnitudes[anchor_flat])
    if anchor_magnitude == 0.0:
        raise BackendError("projected port electric field is identically zero")
    anchor_node, anchor_component = divmod(anchor_flat, 3)
    anchor = complex(values[anchor_node, anchor_component])
    phase_factor = anchor.conjugate() / abs(anchor)
    power_ratio = float(target_forward_power_w) / float(raw_forward_power_w)
    if not math.isfinite(power_ratio):
        raise BackendError("port power-normalization ratio overflowed")
    amplitude_scale = math.sqrt(power_ratio)
    with np.errstate(over="ignore", invalid="ignore"):
        normalized = np.asarray(values * phase_factor * amplitude_scale, dtype=np.complex128)
    if not np.isfinite(normalized).all():
        raise BackendError("port power normalization produced non-finite field values")
    canonical_anchor = complex(normalized[anchor_node, anchor_component])
    tolerance = 32.0 * np.finfo(np.float64).eps * abs(canonical_anchor)
    if canonical_anchor.real <= 0.0 or abs(canonical_anchor.imag) > tolerance:
        raise BackendError("port phase canonicalization failed at its deterministic anchor")
    return NormalizedProjectedMode(
        electric_field=normalized,
        anchor_node=anchor_node,
        anchor_component=anchor_component,
        phase_factor=phase_factor,
        amplitude_scale=amplitude_scale,
    )


def normalize_projected_electromagnetic_mode(
    electric_field: np.ndarray,
    magnetic_field: np.ndarray,
    *,
    raw_forward_power_w: float,
    target_forward_power_w: float,
) -> NormalizedProjectedElectromagneticMode:
    """Apply one power and phase normalization to a physical E/H field pair."""

    electric = normalize_projected_mode(
        electric_field,
        raw_forward_power_w=raw_forward_power_w,
        target_forward_power_w=target_forward_power_w,
    )
    magnetic = np.asarray(magnetic_field)
    if magnetic.shape != electric.electric_field.shape or magnetic.dtype.kind != "c":
        raise BackendError("projected port magnetic field must match the complex electric field")
    if not np.isfinite(magnetic).all():
        raise BackendError("projected port magnetic field contains non-finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        normalized_magnetic = np.asarray(
            magnetic * electric.phase_factor * electric.amplitude_scale,
            dtype=np.complex128,
        )
    if not np.isfinite(normalized_magnetic).all():
        raise BackendError("port power normalization produced non-finite magnetic field values")
    with np.errstate(over="ignore", invalid="ignore"):
        normalized_power = float(
            np.float64(raw_forward_power_w)
            * np.float64(electric.amplitude_scale)
            * np.float64(electric.amplitude_scale)
        )
    if not math.isfinite(normalized_power):
        raise BackendError("normalized port forward power is non-finite")
    return NormalizedProjectedElectromagneticMode(
        electric_field=electric.electric_field,
        magnetic_field=normalized_magnetic,
        anchor_node=electric.anchor_node,
        anchor_component=electric.anchor_component,
        phase_factor=electric.phase_factor,
        amplitude_scale=electric.amplitude_scale,
        normalized_forward_power_w=normalized_power,
    )
