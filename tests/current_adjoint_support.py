"""Shared problem construction for current-adjoint scientific evidence."""

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
from femx.physics import (
    ConductiveRegion,
    CurrentFluxBoundary,
    PotentialBoundary,
    SteadyCurrent,
)
from tests.support import structured_unit_square_mesh


def parameterized_current_adjoint_problem(
    *,
    intervals: int,
) -> tuple[Problem, ParameterValues]:
    """Return the shared Joule-adjoint case used by JAX and locked Elmer evidence."""

    schema = ParameterSchema(
        (
            ParameterSpec(
                "conductivity",
                unit="S/m",
                role=ParameterRole.DESIGN,
                lower_bound=0.2,
            ),
            ParameterSpec("source", unit="A/m^3", role=ParameterRole.CONTROL),
            ParameterSpec("current_load", unit="A/m^2", role=ParameterRole.DESIGN),
        )
    )
    physics = SteadyCurrent(
        regions=(
            ConductiveRegion(
                "domain",
                ParameterReference("conductivity"),
                ParameterReference("source"),
            ),
        ),
        potential_boundaries=(PotentialBoundary("left", 0.0),),
        current_flux_boundaries=(CurrentFluxBoundary("right", ParameterReference("current_load")),),
        gradient_method=GradientMethod.ADJOINT,
    )
    problem = Problem(
        "scientific-adjoint-current",
        structured_unit_square_mesh(intervals),
        physics,
        parameters=schema,
    )
    parameters = schema.bind(
        {
            "conductivity": 2.3,
            "source": 1.7,
            "current_load": 0.8,
        }
    )
    return problem, parameters


def triangle_areas(problem: Problem) -> np.ndarray:
    """Return physical P1 triangle areas for an exact cellwise integral."""

    coordinates = np.asarray(problem.mesh.geometry.coordinates)
    cells = np.asarray(problem.mesh.topology.connectivity)
    points = coordinates[cells]
    first = points[:, 1, :] - points[:, 0, :]
    second = points[:, 2, :] - points[:, 0, :]
    return 0.5 * np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
