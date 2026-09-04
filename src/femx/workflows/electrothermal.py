"""Solver-neutral contract for one-way same-mesh Joule heating."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from femx.core.errors import ContractError
from femx.core.parameters import ParameterReference, ParameterSchema
from femx.core.problem import Problem
from femx.physics._scalar import (
    ScalarCoefficient,
    coefficient_data,
    require_unique_tags,
    validate_coefficient,
    validate_name,
)
from femx.physics.steady_current import SteadyCurrent
from femx.physics.steady_heat import SteadyHeat
from femx.workflows.graph import CouplingEdge, WorkflowGraph, WorkflowNode

JOULE_HEAT_DENSITY_UNIT = "W/m^3"
SAME_MESH_CELL_OPERATOR = "femx.transfer.same_mesh_l2_p0_identity/v1"
SAME_MESH_CELL_LOCAL_P1_OPERATOR = "femx.transfer.same_mesh_cell_local_l2_p1_identity/v1"


@dataclass(frozen=True, slots=True)
class SameMeshJouleHeating:
    """A one-way current-to-heat coupling on one exact cell ordering.

    The current solve produces one constant Joule density per bulk cell. The transfer adds that
    field to the heat problem's existing volumetric source without interpolation, averaging, or
    unit conversion. Requiring the same mesh object is intentional: a distinct mesh, even with
    equal-looking arrays, needs a separately versioned and validated transfer operator.
    """

    electrical_problem: Problem
    thermal_problem: Problem
    schema_version: str = "femx.workflow.electrothermal/v1"

    def __post_init__(self) -> None:
        if not isinstance(self.electrical_problem.physics, SteadyCurrent):
            raise ContractError("electrothermal source problem must use SteadyCurrent")
        if not isinstance(self.thermal_problem.physics, SteadyHeat):
            raise ContractError("electrothermal target problem must use SteadyHeat")
        if self.electrical_problem.mesh is not self.thermal_problem.mesh:
            raise ContractError(
                "same-mesh Joule heating requires the exact shared mesh object and cell ordering"
            )
        if self.schema_version != "femx.workflow.electrothermal/v1":
            raise ContractError(
                f"unsupported electrothermal workflow schema {self.schema_version!r}"
            )

    @property
    def graph(self) -> WorkflowGraph:
        """Return the explicit two-node coupling graph."""

        return WorkflowGraph(
            nodes=(
                WorkflowNode(
                    "electrical",
                    consumes=frozenset(),
                    produces=frozenset({"potential", "joule_heat_density"}),
                ),
                WorkflowNode(
                    "thermal",
                    consumes=frozenset({"additive_cell_heat_source"}),
                    produces=frozenset({"temperature"}),
                ),
            ),
            edges=(
                CouplingEdge(
                    source="electrical",
                    source_quantity="joule_heat_density",
                    target="thermal",
                    target_quantity="additive_cell_heat_source",
                    operator=SAME_MESH_CELL_OPERATOR,
                    source_unit=JOULE_HEAT_DENSITY_UNIT,
                    target_unit=JOULE_HEAT_DENSITY_UNIT,
                ),
            ),
        )

    def canonical_data(self) -> Mapping[str, object]:
        """Return deterministic semantic metadata for provenance and hashing."""

        edge = self.graph.edges[0]
        return {
            "schema_version": self.schema_version,
            "electrical_problem": self.electrical_problem.name,
            "thermal_problem": self.thermal_problem.name,
            "direction": "one_way",
            "feedback": False,
            "source": {
                "quantity": edge.source_quantity,
                "unit": edge.source_unit,
                "function_space": "L2",
                "order": 0,
                "dof_location": "cell",
            },
            "target": {
                "quantity": edge.target_quantity,
                "unit": edge.target_unit,
                "function_space": "L2",
                "order": 0,
                "dof_location": "cell",
                "combination": "add",
            },
            "operator": edge.operator,
            "mesh_relation": "same_object_same_cell_order",
        }


@dataclass(frozen=True, slots=True)
class ResistivityTemperatureLaw:
    r"""Smooth local conductivity law on one electrical material region.

    The region's declared electrical conductivity is the reference value ``sigma_ref`` and

    .. math::

       \sigma(T)=\frac{\sigma_{ref}}{1+\alpha(T-T_{ref})}.

    A nonpositive denominator is outside the law domain and must not be clipped.
    """

    tag: str
    reference_temperature: ScalarCoefficient
    temperature_coefficient: ScalarCoefficient

    def __post_init__(self) -> None:
        validate_name(self.tag, label="resistivity-temperature law tag")
        validate_coefficient(
            self.reference_temperature,
            label=f"reference temperature on {self.tag!r}",
        )
        validate_coefficient(
            self.temperature_coefficient,
            label=f"resistivity temperature coefficient on {self.tag!r}",
        )

    def canonical_data(self) -> Mapping[str, object]:
        """Return the exact formula and coefficient metadata."""

        return {
            "tag": self.tag,
            "formula": "sigma_ref/(1+alpha*(temperature-reference_temperature))",
            "reference_temperature_K": coefficient_data(self.reference_temperature),
            "temperature_coefficient_per_K": coefficient_data(self.temperature_coefficient),
            "invalid_domain": "nonpositive_denominator",
        }


@dataclass(frozen=True, slots=True)
class CoupledIterationPolicy:
    """Numerical contract for the serial block Gauss-Seidel reference solve."""

    max_iterations: int = 100
    minimum_iterations: int = 2
    relative_tolerance: float = 1.0e-10
    residual_tolerance: float = 1.0e-10
    potential_absolute_tolerance: float = 1.0e-12
    temperature_absolute_tolerance: float = 1.0e-10
    potential_relaxation: float = 1.0
    temperature_relaxation: float = 0.5

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ContractError("coupled max_iterations must be positive")
        if self.minimum_iterations <= 0 or self.minimum_iterations > self.max_iterations:
            raise ContractError(
                "coupled minimum_iterations must be positive and not exceed max_iterations"
            )
        for name, value in (
            ("relative_tolerance", self.relative_tolerance),
            ("residual_tolerance", self.residual_tolerance),
            ("potential_absolute_tolerance", self.potential_absolute_tolerance),
            ("temperature_absolute_tolerance", self.temperature_absolute_tolerance),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ContractError(f"coupled {name} must be finite and positive")
        for name, value in (
            ("potential_relaxation", self.potential_relaxation),
            ("temperature_relaxation", self.temperature_relaxation),
        ):
            if not math.isfinite(value) or value <= 0.0 or value > 1.0:
                raise ContractError(f"coupled {name} must lie in (0, 1]")

    def canonical_data(self) -> Mapping[str, object]:
        """Return deterministic stopping and relaxation metadata."""

        return {
            "algorithm": "block_gauss_seidel",
            "max_iterations": self.max_iterations,
            "minimum_iterations": self.minimum_iterations,
            "relative_tolerance": self.relative_tolerance,
            "residual_tolerance": self.residual_tolerance,
            "potential_absolute_tolerance_V": self.potential_absolute_tolerance,
            "temperature_absolute_tolerance_K": self.temperature_absolute_tolerance,
            "potential_relaxation": self.potential_relaxation,
            "temperature_relaxation": self.temperature_relaxation,
        }


@dataclass(frozen=True, slots=True)
class SelfConsistentJouleHeating:
    """Temperature-dependent feedback using Elmer-compatible cell-local P1 transfer."""

    one_way: SameMeshJouleHeating
    conductivity_laws: tuple[ResistivityTemperatureLaw, ...]
    parameters: ParameterSchema
    iteration: CoupledIterationPolicy = CoupledIterationPolicy()
    schema_version: str = "femx.workflow.electrothermal_feedback/v1"

    def __post_init__(self) -> None:
        if not self.conductivity_laws:
            raise ContractError("self-consistent Joule heating requires a conductivity law")
        require_unique_tags(self.conductivity_laws, label="resistivity-temperature law")
        electrical = self.one_way.electrical_problem.physics
        assert isinstance(electrical, SteadyCurrent)
        electrical_tags = {region.tag for region in electrical.regions}
        unknown = sorted(
            law.tag for law in self.conductivity_laws if law.tag not in electrical_tags
        )
        if unknown:
            raise ContractError(
                f"conductivity laws reference unknown electrical regions: {unknown}"
            )

        expected_units: dict[str, str] = {}
        for law in self.conductivity_laws:
            for coefficient, unit in (
                (law.reference_temperature, "K"),
                (law.temperature_coefficient, "1/K"),
            ):
                if not isinstance(coefficient, ParameterReference):
                    continue
                previous = expected_units.setdefault(coefficient.name, unit)
                if previous != unit:
                    raise ContractError(
                        f"feedback parameter {coefficient.name!r} has incompatible units"
                    )
        actual = {spec.name: spec for spec in self.parameters.specs}
        if set(actual) != set(expected_units):
            raise ContractError(
                "feedback coefficient parameters do not match the workflow schema: "
                f"expected={sorted(expected_units)}, actual={sorted(actual)}"
            )
        for name, unit in expected_units.items():
            spec = actual[name]
            if spec.unit != unit or spec.shape:
                raise ContractError(
                    f"feedback parameter {name!r} must be a scalar with unit {unit!r}, "
                    f"got unit={spec.unit!r}, shape={spec.shape}"
                )
        if self.schema_version != "femx.workflow.electrothermal_feedback/v1":
            raise ContractError(
                f"unsupported electrothermal feedback schema {self.schema_version!r}"
            )

    def canonical_data(self) -> Mapping[str, object]:
        """Return deterministic nonlinear-coupling metadata."""

        return {
            "schema_version": self.schema_version,
            "electrical_problem": self.one_way.electrical_problem.name,
            "thermal_problem": self.one_way.thermal_problem.name,
            "mesh_relation": "same_object_same_cell_order",
            "conductivity_laws": [law.canonical_data() for law in self.conductivity_laws],
            "parameter_names": list(self.parameters.names),
            "iteration": self.iteration.canonical_data(),
            "coupled_residual": ["current", "heat_with_joule"],
            "differentiation": "implicit_coupled_residual",
            "joule_transfer": {
                "operator": SAME_MESH_CELL_LOCAL_P1_OPERATOR,
                "quantity": "joule_heat_density",
                "unit": JOULE_HEAT_DENSITY_UNIT,
                "function_space": "L2",
                "order": 1,
                "continuity": "cell_local_discontinuous",
                "dof_location": "cell_local_vertex",
                "conductivity_evaluation": "law_at_temperature_vertex_dofs",
                "electric_field_evaluation": "constant_cell_gradient",
                "heat_load_integration": "consistent_p1_triangle",
            },
        }
