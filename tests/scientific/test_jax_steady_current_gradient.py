import pytest
from tests.current_adjoint_support import (
    parameterized_current_adjoint_problem,
    triangle_areas,
)

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from femx.backends.jax.steady_current import JaxSteadyCurrentBackend  # noqa: E402
from femx.runtime import prepare  # noqa: E402

pytestmark = [pytest.mark.scientific, pytest.mark.requires_jax]


def test_total_joule_adjoint_matches_reverse_mode_and_central_finite_difference() -> None:
    problem, parameters = parameterized_current_adjoint_problem(intervals=5)
    backend = JaxSteadyCurrentBackend()
    bound = backend.bind_differentiable(prepare(problem, backend), parameters)
    initial = bound.initial_values
    areas = jnp.asarray(triangle_areas(problem), dtype=jnp.float64)

    def objective(active: jax.Array) -> jax.Array:
        return jnp.vdot(areas, bound.joule_heat_density(active))

    objective_value, reverse_gradient = jax.jit(jax.value_and_grad(objective))(initial)
    adjoint_result = bound.joule_vjp(initial, areas)
    finite_difference = []
    for index, value in enumerate(np.asarray(initial)):
        step = 2.0e-5 * max(abs(float(value)), 1.0)
        plus = float(objective(initial.at[index].add(step)))
        minus = float(objective(initial.at[index].add(-step)))
        finite_difference.append((plus - minus) / (2.0 * step))

    assert float(objective_value) > 0.0
    assert float(adjoint_result.adjoint_backward_error) < 4.0e-16
    np.testing.assert_allclose(
        adjoint_result.parameter_gradient,
        reverse_gradient,
        rtol=8.0e-12,
        atol=8.0e-12,
    )
    np.testing.assert_allclose(
        adjoint_result.parameter_gradient,
        np.asarray(finite_difference),
        rtol=3.0e-8,
        atol=3.0e-9,
    )
