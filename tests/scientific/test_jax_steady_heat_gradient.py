import pytest
from tests.support import structured_unit_square_mesh

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from femx.backends.jax.steady_heat import JaxSteadyHeatBackend  # noqa: E402
from femx.core.capabilities import GradientMethod  # noqa: E402
from femx.core.parameters import (  # noqa: E402
    ParameterReference,
    ParameterRole,
    ParameterSchema,
    ParameterSpec,
    ParameterValues,
)
from femx.core.problem import Problem  # noqa: E402
from femx.physics import (  # noqa: E402
    HeatFluxBoundary,
    SteadyHeat,
    TemperatureBoundary,
    ThermalRegion,
)
from femx.runtime import prepare  # noqa: E402

pytestmark = [pytest.mark.scientific, pytest.mark.requires_jax]


def _parameterized_adjoint_problem() -> tuple[Problem, ParameterValues]:
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
    physics = SteadyHeat(
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
    problem = Problem(
        "scientific-adjoint-heat",
        structured_unit_square_mesh(5),
        physics,
        parameters=schema,
    )
    parameters = schema.bind(
        {
            "conductivity": 2.3,
            "source": 1.7,
            "heat_load": 0.8,
            "left_temperature": 300.0,
        }
    )
    return problem, parameters


def test_implicit_adjoint_matches_reverse_mode_and_central_finite_difference() -> None:
    problem, parameters = _parameterized_adjoint_problem()
    backend = JaxSteadyHeatBackend()
    bound = backend.bind_differentiable(prepare(problem, backend), parameters)
    initial = bound.initial_values

    def objective(active: jax.Array) -> jax.Array:
        normalized_temperature = bound.temperature(active) - 300.0
        return 0.5 * jnp.mean(normalized_temperature * normalized_temperature)

    objective_value, reverse_gradient = jax.jit(jax.value_and_grad(objective))(initial)
    temperature = bound.temperature(initial)
    cotangent = (temperature - 300.0) / temperature.size
    adjoint_result = bound.vjp(initial, cotangent)
    finite_difference = []
    for index, value in enumerate(np.asarray(initial)):
        step = 2.0e-5 * max(abs(float(value)), 1.0)
        plus = float(objective(initial.at[index].add(step)))
        minus = float(objective(initial.at[index].add(-step)))
        finite_difference.append((plus - minus) / (2.0 * step))

    assert float(objective_value) > 0.0
    assert float(adjoint_result.adjoint_backward_error) < 3.0e-16
    np.testing.assert_allclose(
        adjoint_result.parameter_gradient,
        reverse_gradient,
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    np.testing.assert_allclose(
        adjoint_result.parameter_gradient,
        np.asarray(finite_difference),
        rtol=2.0e-8,
        atol=2.0e-9,
    )
