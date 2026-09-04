"""Standalone bounded electrothermal case used by TPU development runners.

This module intentionally does not import from ``tests``.  Deployed evidence builders must be
reconstructible from the committed package and runner sources alone.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from femx.core.capabilities import GradientMethod
from femx.core.parameters import (
    ParameterReference,
    ParameterRole,
    ParameterSchema,
    ParameterSpec,
    ParameterValues,
)
from femx.core.problem import Problem
from femx.mesh import CellType, EntityTag, Mesh, MeshGeometry, MeshTopology
from femx.physics import (
    ConductiveRegion,
    PotentialBoundary,
    SteadyCurrent,
    SteadyHeat,
    TemperatureBoundary,
    ThermalRegion,
)
from femx.workflows import (
    CoupledIterationPolicy,
    ResistivityTemperatureLaw,
    SameMeshJouleHeating,
    SelfConsistentJouleHeating,
)

if TYPE_CHECKING:
    from femx.backends.jax.self_consistent import DifferentiableSelfConsistentElectrothermal


def _structured_unit_square_mesh(intervals: int) -> Mesh:
    if intervals <= 0:
        raise ValueError("intervals must be positive")
    width = intervals + 1
    coordinates = np.asarray(
        [(i / intervals, j / intervals) for j in range(width) for i in range(width)],
        dtype=np.float64,
    )

    def node(i: int, j: int) -> int:
        return j * width + i

    cells: list[tuple[int, int, int]] = []
    for j in range(intervals):
        for i in range(intervals):
            lower_left = node(i, j)
            lower_right = node(i + 1, j)
            upper_left = node(i, j + 1)
            upper_right = node(i + 1, j + 1)
            cells.extend(
                (
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                )
            )

    facets: list[tuple[int, int]] = []
    tag_ids: dict[str, tuple[int, ...]] = {}
    for name, edges in (
        ("bottom", [(node(i, 0), node(i + 1, 0)) for i in range(intervals)]),
        (
            "right",
            [(node(intervals, j), node(intervals, j + 1)) for j in range(intervals)],
        ),
        (
            "top",
            [(node(i + 1, intervals), node(i, intervals)) for i in range(intervals)],
        ),
        ("left", [(node(0, j + 1), node(0, j)) for j in range(intervals)]),
    ):
        start = len(facets)
        facets.extend(edges)
        tag_ids[name] = tuple(range(start, len(facets)))

    cell_array = np.asarray(cells, dtype=np.int32)
    facet_array = np.asarray(facets, dtype=np.int32)
    return Mesh(
        geometry=MeshGeometry(coordinates),
        topology=MeshTopology(cell_array, CellType.TRIANGLE, coordinates.shape[0]),
        tags=(
            EntityTag("domain", 2, tuple(range(cell_array.shape[0]))),
            *(EntityTag(name, 1, ids) for name, ids in tag_ids.items()),
        ),
        boundary_facets=MeshTopology(
            facet_array,
            CellType.SEGMENT,
            coordinates.shape[0],
        ),
    )


def _parameterized_microheater(
    intervals: int,
) -> tuple[SameMeshJouleHeating, ParameterValues, ParameterValues]:
    if intervals <= 0 or intervals % 2 != 0:
        raise ValueError("microheater intervals must be a positive even integer")
    mesh = _structured_unit_square_mesh(intervals)
    coordinates = np.asarray(mesh.geometry.coordinates, dtype=np.float64).copy()
    coordinates[:, 0] *= 2.0e-6
    coordinates[:, 1] *= 0.5e-6
    cells = np.asarray(mesh.topology.connectivity, dtype=np.int64)
    centroids_x = coordinates[cells, 0].mean(axis=1)
    heater_cells = tuple(int(value) for value in np.flatnonzero(centroids_x < 1.0e-6))
    contact_cells = tuple(int(value) for value in np.flatnonzero(centroids_x >= 1.0e-6))
    mesh = replace(
        mesh,
        geometry=MeshGeometry(coordinates),
        tags=(
            *mesh.tags,
            EntityTag("doped_silicon_heater", 2, heater_cells),
            EntityTag("heavily_doped_contact", 2, contact_cells),
        ),
    )

    current_schema = ParameterSchema(
        (
            ParameterSpec(
                "applied_voltage",
                unit="V",
                role=ParameterRole.CONTROL,
                lower_bound=0.01,
                upper_bound=1.0,
            ),
            ParameterSpec(
                "heater_conductivity",
                unit="S/m",
                role=ParameterRole.DESIGN,
                lower_bound=100.0,
                upper_bound=1.0e6,
            ),
            ParameterSpec(
                "contact_conductivity",
                unit="S/m",
                role=ParameterRole.FIXED,
                lower_bound=100.0,
            ),
        )
    )
    electrical = Problem(
        "siph-microheater-current",
        mesh,
        SteadyCurrent(
            regions=(
                ConductiveRegion(
                    "doped_silicon_heater",
                    ParameterReference("heater_conductivity"),
                ),
                ConductiveRegion(
                    "heavily_doped_contact",
                    ParameterReference("contact_conductivity"),
                ),
            ),
            potential_boundaries=(
                PotentialBoundary("left", 0.0),
                PotentialBoundary("right", ParameterReference("applied_voltage")),
            ),
            gradient_method=GradientMethod.ADJOINT,
        ),
        parameters=current_schema,
    )
    current_parameters = current_schema.bind(
        {
            "applied_voltage": 0.2,
            "heater_conductivity": 2.0e3,
            "contact_conductivity": 2.0e5,
        }
    )

    heat_schema = ParameterSchema(
        (
            ParameterSpec(
                "thermal_conductivity",
                unit="W/(m*K)",
                role=ParameterRole.DESIGN,
                lower_bound=1.0,
                upper_bound=300.0,
            ),
        )
    )
    thermal = Problem(
        "siph-microheater-heat",
        mesh,
        SteadyHeat(
            regions=(
                ThermalRegion(
                    "doped_silicon_heater",
                    ParameterReference("thermal_conductivity"),
                ),
                ThermalRegion(
                    "heavily_doped_contact",
                    ParameterReference("thermal_conductivity"),
                ),
            ),
            temperature_boundaries=(
                TemperatureBoundary("left", 300.0),
                TemperatureBoundary("right", 300.0),
            ),
            gradient_method=GradientMethod.ADJOINT,
        ),
        parameters=heat_schema,
    )
    thermal_parameters = heat_schema.bind({"thermal_conductivity": 120.0})
    return SameMeshJouleHeating(electrical, thermal), current_parameters, thermal_parameters


def parameterized_self_consistent_microheater(
    *,
    intervals: int,
    iteration: CoupledIterationPolicy,
) -> tuple[
    SelfConsistentJouleHeating,
    ParameterValues,
    ParameterValues,
    ParameterValues,
]:
    """Return the bounded same-mesh silicon-photonics heater witness."""

    one_way, current_parameters, thermal_parameters = _parameterized_microheater(intervals)
    feedback_schema = ParameterSchema(
        (
            ParameterSpec(
                "heater_temperature_coefficient",
                unit="1/K",
                role=ParameterRole.DESIGN,
                lower_bound=0.0,
                upper_bound=0.02,
            ),
        )
    )
    feedback = SelfConsistentJouleHeating(
        one_way=one_way,
        conductivity_laws=(
            ResistivityTemperatureLaw(
                "doped_silicon_heater",
                reference_temperature=300.0,
                temperature_coefficient=ParameterReference("heater_temperature_coefficient"),
            ),
        ),
        parameters=feedback_schema,
        iteration=iteration,
    )
    feedback_parameters = feedback_schema.bind({"heater_temperature_coefficient": 3.0e-3})
    return feedback, current_parameters, thermal_parameters, feedback_parameters


def distributed_electrothermal_iteration_policy() -> CoupledIterationPolicy:
    """Return the coupled policy shared by controller input and physical TPU runners."""

    return CoupledIterationPolicy(
        max_iterations=100,
        minimum_iterations=2,
        relative_tolerance=2.0e-5,
        residual_tolerance=1.0e-4,
        potential_absolute_tolerance=1.0e-7,
        temperature_absolute_tolerance=1.0e-4,
        potential_relaxation=1.0,
        temperature_relaxation=0.5,
    )


def bind_jax_self_consistent_microheater(
    *,
    intervals: int,
    iteration: CoupledIterationPolicy,
) -> DifferentiableSelfConsistentElectrothermal:
    """Bind the standalone heater case to the native differentiable JAX backends."""

    from femx.backends.jax.self_consistent import DifferentiableSelfConsistentElectrothermal
    from femx.backends.jax.steady_current import JaxSteadyCurrentBackend
    from femx.backends.jax.steady_heat import JaxSteadyHeatBackend
    from femx.runtime import prepare

    feedback, current_parameters, thermal_parameters, feedback_parameters = (
        parameterized_self_consistent_microheater(
            intervals=intervals,
            iteration=iteration,
        )
    )
    current_backend = JaxSteadyCurrentBackend()
    thermal_backend = JaxSteadyHeatBackend()
    current = current_backend.bind_differentiable(
        prepare(feedback.one_way.electrical_problem, current_backend),
        current_parameters,
    )
    thermal = thermal_backend.bind_differentiable(
        prepare(feedback.one_way.thermal_problem, thermal_backend),
        thermal_parameters,
    )
    return DifferentiableSelfConsistentElectrothermal.bind(
        feedback,
        current,
        thermal,
        feedback_parameters,
    )


__all__ = [
    "bind_jax_self_consistent_microheater",
    "distributed_electrothermal_iteration_policy",
    "parameterized_self_consistent_microheater",
]
