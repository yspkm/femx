"""Shared micrometre-scale electrothermal case for JAX and Elmer evidence."""

from dataclasses import replace

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
from femx.mesh import EntityTag, MeshGeometry
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
from tests.support import structured_unit_square_mesh


def parameterized_microheater_coupling(
    *,
    intervals: int,
) -> tuple[SameMeshJouleHeating, ParameterValues, ParameterValues]:
    """Return a shared-mesh, two-conductivity silicon-photonics heater precursor."""

    if intervals <= 0 or intervals % 2 != 0:
        raise ValueError("microheater intervals must be a positive even number")
    mesh = structured_unit_square_mesh(intervals)
    coordinates = np.asarray(mesh.geometry.coordinates).copy()
    coordinates[:, 0] *= 2.0e-6
    coordinates[:, 1] *= 0.5e-6
    cells = np.asarray(mesh.topology.connectivity)
    centroids_x = coordinates[cells, 0].mean(axis=1)
    heater_cells = tuple(np.flatnonzero(centroids_x < 1.0e-6).tolist())
    contact_cells = tuple(np.flatnonzero(centroids_x >= 1.0e-6).tolist())
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
    heat_parameters = heat_schema.bind({"thermal_conductivity": 120.0})
    return SameMeshJouleHeating(electrical, thermal), current_parameters, heat_parameters


def parameterized_self_consistent_microheater(
    *,
    intervals: int,
    iteration: CoupledIterationPolicy | None = None,
) -> tuple[
    SelfConsistentJouleHeating,
    ParameterValues,
    ParameterValues,
    ParameterValues,
]:
    """Return the shared silicon-photonics heater with local resistive feedback."""

    one_way, current_parameters, heat_parameters = parameterized_microheater_coupling(
        intervals=intervals
    )
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
        iteration=iteration or CoupledIterationPolicy(),
    )
    feedback_parameters = feedback_schema.bind({"heater_temperature_coefficient": 3.0e-3})
    return feedback, current_parameters, heat_parameters, feedback_parameters


def triangle_areas(problem: Problem) -> np.ndarray:
    """Return exact physical areas in the problem's bulk-cell order."""

    coordinates = np.asarray(problem.mesh.geometry.coordinates)
    cells = np.asarray(problem.mesh.topology.connectivity)
    points = coordinates[cells]
    first = points[:, 1, :] - points[:, 0, :]
    second = points[:, 2, :] - points[:, 0, :]
    return 0.5 * np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
