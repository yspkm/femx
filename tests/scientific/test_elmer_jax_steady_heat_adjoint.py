from dataclasses import replace

import pytest
from tests.support import structured_unit_square_mesh

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from femx.backends.jax.steady_heat import JaxSteadyHeatBackend  # noqa: E402
from femx.backends.protocol import (  # noqa: E402
    ExecutionPolicy,
    PrepareRequest,
    SolveRequest,
)
from femx.core.capabilities import GradientMethod  # noqa: E402
from femx.core.parameters import (  # noqa: E402
    ParameterReference,
    ParameterRole,
    ParameterSchema,
    ParameterSpec,
)
from femx.core.problem import Problem  # noqa: E402
from femx.physics import (  # noqa: E402
    HeatFluxBoundary,
    SteadyHeat,
    TemperatureBoundary,
    ThermalRegion,
)
from femx.runtime import prepare, solve  # noqa: E402

pytestmark = [
    pytest.mark.scientific,
    pytest.mark.requires_elmer,
    pytest.mark.requires_jax,
]


def test_jax_adjoint_matches_locked_elmer_central_differences(
    locked_elmer_backend,
    tmp_path,
) -> None:
    schema = ParameterSchema(
        (
            ParameterSpec(
                "conductivity",
                unit="W/(m*K)",
                role=ParameterRole.DESIGN,
                lower_bound=0.2,
            ),
            ParameterSpec("source", unit="W/m^3", role=ParameterRole.CONTROL),
            ParameterSpec("heat_load", unit="W/m^2", role=ParameterRole.DESIGN),
            ParameterSpec(
                "left_temperature",
                unit="K",
                role=ParameterRole.CONTROL,
                lower_bound=250.0,
                upper_bound=400.0,
            ),
        )
    )
    adjoint_physics = SteadyHeat(
        regions=(
            ThermalRegion(
                "domain",
                ParameterReference("conductivity"),
                ParameterReference("source"),
            ),
        ),
        temperature_boundaries=(
            TemperatureBoundary("left", ParameterReference("left_temperature")),
        ),
        heat_flux_boundaries=(HeatFluxBoundary("right", ParameterReference("heat_load")),),
        gradient_method=GradientMethod.ADJOINT,
    )
    mesh = structured_unit_square_mesh(3)
    jax_problem = Problem("cross-adjoint-jax", mesh, adjoint_physics, parameters=schema)
    elmer_problem = Problem(
        "cross-adjoint-elmer",
        mesh,
        replace(adjoint_physics, gradient_method=GradientMethod.NONE),
        parameters=schema,
    )
    initial_mapping = {
        "conductivity": 2.3,
        "source": 1.7,
        "heat_load": 0.8,
        "left_temperature": 300.0,
    }
    initial_parameters = schema.bind(initial_mapping)
    jax_backend = JaxSteadyHeatBackend()
    bound = jax_backend.bind_differentiable(
        prepare(jax_problem, jax_backend),
        initial_parameters,
    )
    initial = bound.initial_values

    def objective(temperature):
        normalized = temperature - 300.0
        return 0.5 * jnp.mean(normalized * normalized)

    jax_temperature = bound.temperature(initial)
    cotangent = (jax_temperature - 300.0) / jax_temperature.size
    adjoint_gradient = np.asarray(bound.vjp(initial, cotangent).parameter_gradient)

    def elmer_objective(active: np.ndarray, attempt_name: str) -> float:
        values = dict(initial_mapping)
        values.update(
            (name, float(value)) for name, value in zip(bound.parameter_names, active, strict=True)
        )
        parameters = schema.bind(values)
        run_directory = tmp_path / attempt_name
        solution = solve(
            prepare(
                elmer_problem,
                locked_elmer_backend,
                request=PrepareRequest(run_directory=run_directory),
            ),
            locked_elmer_backend,
            request=SolveRequest(
                parameters=parameters,
                run_directory=run_directory,
                policy=ExecutionPolicy(
                    execution_authorized=True,
                    allow_external_process=True,
                ),
            ),
        )
        assert solution.convergence.status.value == "converged"
        temperature = np.asarray(solution.fields["temperature"].values)
        normalized = temperature - 300.0
        return 0.5 * float(np.mean(normalized * normalized))

    initial_numpy = np.asarray(initial)
    elmer_gradient = []
    for index, value in enumerate(initial_numpy):
        step = 2.0e-5 * max(abs(float(value)), 1.0)
        plus = initial_numpy.copy()
        minus = initial_numpy.copy()
        plus[index] += step
        minus[index] -= step
        plus_value = elmer_objective(plus, f"parameter-{index}-plus")
        minus_value = elmer_objective(minus, f"parameter-{index}-minus")
        elmer_gradient.append((plus_value - minus_value) / (2.0 * step))

    baseline_elmer_objective = elmer_objective(initial_numpy, "baseline")
    np.testing.assert_allclose(
        baseline_elmer_objective,
        float(objective(jax_temperature)),
        rtol=2.0e-11,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        adjoint_gradient,
        np.asarray(elmer_gradient),
        rtol=3.0e-7,
        atol=3.0e-9,
    )
