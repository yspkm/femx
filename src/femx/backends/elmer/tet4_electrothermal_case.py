"""Typed 3D Tet4 current-to-Joule-to-heat lowering for the Elmer oracle."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from femx.backends.elmer.case import _format_real
from femx.backends.elmer.tet4_case import ElmerTet4MeshDeck
from femx.core.errors import ContractError


def _positive_finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"Elmer Tet4 {label} must be a real scalar")
    canonical = float(value)
    if not math.isfinite(canonical) or canonical <= 0.0:
        raise ContractError(f"Elmer Tet4 {label} must be finite and positive")
    return canonical


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"Elmer Tet4 {label} must be a real scalar")
    canonical = float(value)
    if not math.isfinite(canonical):
        raise ContractError(f"Elmer Tet4 {label} must be finite")
    return canonical


def _encoded_module_path(path: Path, *, label: str) -> str:
    if not path.is_absolute():
        raise ContractError(f"Elmer {label} procedure path must be absolute")
    encoded = str(path)
    if any(character.isspace() for character in encoded) or '"' in encoded:
        raise ContractError(f"Elmer {label} procedure path cannot contain whitespace or quotes")
    return encoded


@dataclass(frozen=True, slots=True)
class ElmerTet4ElectrothermalBody:
    """One thermal body, optionally participating in current and Joule heating."""

    body_id: int
    heat_conductivity_w_per_m_k: float
    electric_conductivity_s_per_m: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.body_id, bool) or not isinstance(self.body_id, int) or self.body_id <= 0:
            raise ContractError("Elmer Tet4 body id must be a positive integer")
        object.__setattr__(
            self,
            "heat_conductivity_w_per_m_k",
            _positive_finite(
                self.heat_conductivity_w_per_m_k,
                label="heat conductivity",
            ),
        )
        if self.electric_conductivity_s_per_m is not None:
            object.__setattr__(
                self,
                "electric_conductivity_s_per_m",
                _positive_finite(
                    self.electric_conductivity_s_per_m,
                    label="electric conductivity",
                ),
            )

    @property
    def is_electrical(self) -> bool:
        """Whether StaticCurrentSolver and Joule heating are active on this body."""

        return self.electric_conductivity_s_per_m is not None


@dataclass(frozen=True, slots=True)
class ElmerTet4BoundaryCondition:
    """Supported electrical and thermal data on one emitted boundary group."""

    boundary_id: int
    potential_v: float | None = None
    temperature_k: float | None = None
    heat_transfer_coefficient_w_per_m2_k: float | None = None
    external_temperature_k: float | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.boundary_id, bool)
            or not isinstance(self.boundary_id, int)
            or self.boundary_id <= 0
        ):
            raise ContractError("Elmer Tet4 boundary id must be a positive integer")
        for name, label in (
            ("potential_v", "boundary potential"),
            ("temperature_k", "boundary temperature"),
            ("external_temperature_k", "external temperature"),
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(value, label=label))
        coefficient = self.heat_transfer_coefficient_w_per_m2_k
        ambient = self.external_temperature_k
        if (coefficient is None) != (ambient is None):
            raise ContractError(
                "Elmer Tet4 Robin boundary requires both heat-transfer coefficient and "
                "external temperature"
            )
        if coefficient is not None:
            object.__setattr__(
                self,
                "heat_transfer_coefficient_w_per_m2_k",
                _positive_finite(coefficient, label="heat-transfer coefficient"),
            )
        if self.temperature_k is not None and coefficient is not None:
            raise ContractError(
                "Elmer Tet4 boundary cannot prescribe temperature and Robin transfer together"
            )
        if self.potential_v is None and self.temperature_k is None and coefficient is None:
            raise ContractError("Elmer Tet4 boundary condition must prescribe at least one value")


@dataclass(frozen=True, slots=True)
class ElmerTet4ElectrothermalCase:
    """Closed one-way 3D current/Joule/heat case on one native Tet4 mesh."""

    mesh: ElmerTet4MeshDeck
    bodies: tuple[ElmerTet4ElectrothermalBody, ...]
    boundaries: tuple[ElmerTet4BoundaryCondition, ...]
    initial_temperature_k: float

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, ElmerTet4MeshDeck):
            raise ContractError("Elmer Tet4 electrothermal case requires a native Tet4 mesh")
        if tuple(body.body_id for body in self.bodies) != self.mesh.body_ids:
            raise ContractError("Elmer Tet4 electrothermal bodies must match emitted body ids")
        if not any(body.is_electrical for body in self.bodies):
            raise ContractError("Elmer Tet4 electrothermal case requires an electrical body")
        boundary_ids = tuple(boundary.boundary_id for boundary in self.boundaries)
        if len(boundary_ids) != len(set(boundary_ids)) or not set(boundary_ids).issubset(
            self.mesh.boundary_ids
        ):
            raise ContractError(
                "Elmer Tet4 electrothermal boundary conditions must be unique emitted ids"
            )
        potentials = tuple(
            boundary.potential_v for boundary in self.boundaries if boundary.potential_v is not None
        )
        if len(potentials) < 2 or len(set(potentials)) < 2:
            raise ContractError(
                "Elmer Tet4 electrothermal case requires two distinct terminal potentials"
            )
        if not any(
            boundary.temperature_k is not None
            or boundary.heat_transfer_coefficient_w_per_m2_k is not None
            for boundary in self.boundaries
        ):
            raise ContractError("Elmer Tet4 electrothermal case requires a thermal boundary")
        object.__setattr__(
            self,
            "initial_temperature_k",
            _positive_finite(self.initial_temperature_k, label="initial temperature"),
        )

        electrical_nodes = set(self.potential_node_ids)
        for boundary in self.boundaries:
            if boundary.potential_v is None:
                continue
            boundary_nodes = set(self.mesh.boundary_node_ids[boundary.boundary_id - 1])
            if not boundary_nodes.issubset(electrical_nodes):
                raise ContractError(
                    "Elmer Tet4 potential boundary contains a node outside electrical bodies"
                )

    @property
    def potential_node_ids(self) -> tuple[int, ...]:
        """Original zero-based mesh nodes carrying Elmer's partial Potential variable."""

        nodes: set[int] = set()
        for body in self.bodies:
            if body.is_electrical:
                nodes.update(self.mesh.body_node_ids[body.body_id - 1])
        return tuple(sorted(nodes))

    def canonical_data(self) -> dict[str, object]:
        """Return the exact typed case binding without duplicating native files."""

        return {
            "mesh_sha256": self.mesh.digest(),
            "bodies": [
                {
                    "body_id": body.body_id,
                    "heat_conductivity_W_per_m_K": body.heat_conductivity_w_per_m_k,
                    "electric_conductivity_S_per_m": body.electric_conductivity_s_per_m,
                    "joule_heat": body.is_electrical,
                }
                for body in self.bodies
            ],
            "boundaries": [
                {
                    "boundary_id": boundary.boundary_id,
                    "potential_V": boundary.potential_v,
                    "temperature_K": boundary.temperature_k,
                    "heat_transfer_coefficient_W_per_m2_K": (
                        boundary.heat_transfer_coefficient_w_per_m2_k
                    ),
                    "external_temperature_K": boundary.external_temperature_k,
                }
                for boundary in self.boundaries
            ],
            "potential_node_ids": list(self.potential_node_ids),
            "initial_temperature_K": self.initial_temperature_k,
        }

    def digest(self) -> str:
        """Hash mesh, coefficients, boundaries, and active current-node identity."""

        payload = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def render_tet4_electrothermal_sif(
    case: ElmerTet4ElectrothermalCase,
    *,
    stat_current_module: Path,
    heat_solve_module: Path,
    convergence_tolerance: float,
) -> str:
    """Render femx's closed 3D one-way current/Joule/heat Elmer subset."""

    if not isinstance(case, ElmerTet4ElectrothermalCase):
        raise ContractError("Elmer Tet4 renderer requires an electrothermal case")
    stat_module = _encoded_module_path(stat_current_module, label="StatCurrentSolve")
    heat_module = _encoded_module_path(heat_solve_module, label="HeatSolve")
    tolerance = _positive_finite(convergence_tolerance, label="convergence tolerance")

    lines = [
        "Header",
        "  CHECK KEYWORDS Warn",
        '  Mesh DB "." "mesh"',
        "End",
        "",
        "Simulation",
        "  Coordinate System = Cartesian 3D",
        "  Coordinate Mapping(3) = 1 2 3",
        "  Simulation Type = Steady State",
        "  Steady State Max Iterations = 2",
        "  Steady State Min Iterations = 1",
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
    for body in case.bodies:
        lines.extend(
            (
                f"Body {body.body_id}",
                f"  Target Bodies(1) = {body.body_id}",
                f"  Equation = {1 if body.is_electrical else 2}",
                f"  Material = {body.body_id}",
            )
        )
        if body.is_electrical:
            lines.append(f"  Body Force = {body.body_id}")
        lines.extend(("  Initial Condition = 1", "End", ""))
    lines.extend(
        (
            "Equation 1",
            "  Active Solvers(2) = 1 2",
            "End",
            "",
            "Equation 2",
            "  Active Solvers(1) = 2",
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
            f"  Steady State Convergence Tolerance = {_format_real(tolerance)}",
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
            f"  Steady State Convergence Tolerance = {_format_real(tolerance)}",
            "End",
            "",
            "Initial Condition 1",
            f"  Temperature = Real {_format_real(case.initial_temperature_k)}",
            "End",
            "",
        )
    )
    for body in case.bodies:
        lines.append(f"Material {body.body_id}")
        if body.electric_conductivity_s_per_m is not None:
            lines.append(
                f"  Electric Conductivity = Real {_format_real(body.electric_conductivity_s_per_m)}"
            )
        lines.extend(
            (
                f"  Heat Conductivity = Real {_format_real(body.heat_conductivity_w_per_m_k)}",
                "  Density = Real 1.00000000000000000e+00",
                "End",
                "",
            )
        )
        if body.is_electrical:
            lines.extend(
                (
                    f"Body Force {body.body_id}",
                    "  Current Source = Real 0.00000000000000000e+00",
                    "  Volumetric Heat Source = Real 0.00000000000000000e+00",
                    "  Joule Heat = Logical True",
                    "End",
                    "",
                )
            )
    for condition_number, boundary in enumerate(case.boundaries, start=1):
        lines.extend(
            (
                f"Boundary Condition {condition_number}",
                f"  Target Boundaries(1) = {boundary.boundary_id}",
            )
        )
        if boundary.potential_v is not None:
            lines.append(f"  Potential = Real {_format_real(boundary.potential_v)}")
        if boundary.temperature_k is not None:
            lines.append(f"  Temperature = Real {_format_real(boundary.temperature_k)}")
        if boundary.heat_transfer_coefficient_w_per_m2_k is not None:
            assert boundary.external_temperature_k is not None
            lines.extend(
                (
                    "  Heat Transfer Coefficient = Real "
                    f"{_format_real(boundary.heat_transfer_coefficient_w_per_m2_k)}",
                    "  External Temperature = Real "
                    f"{_format_real(boundary.external_temperature_k)}",
                )
            )
        lines.extend(("End", ""))
    return "\n".join(lines)
