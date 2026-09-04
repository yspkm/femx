import pytest
from tests.electrothermal_support import parameterized_self_consistent_microheater

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from femx.backends.jax.self_consistent import (  # noqa: E402
    DifferentiableSelfConsistentElectrothermal,
)
from femx.backends.jax.steady_current import JaxSteadyCurrentBackend  # noqa: E402
from femx.backends.jax.steady_heat import JaxSteadyHeatBackend  # noqa: E402
from femx.runtime import prepare  # noqa: E402

pytestmark = [pytest.mark.scientific, pytest.mark.requires_jax]


def test_coupled_residual_adjoint_matches_independent_central_differences() -> None:
    feedback, current_parameters, heat_parameters, feedback_parameters = (
        parameterized_self_consistent_microheater(intervals=4)
    )
    current_backend = JaxSteadyCurrentBackend()
    heat_backend = JaxSteadyHeatBackend()
    current = current_backend.bind_differentiable(
        prepare(feedback.one_way.electrical_problem, current_backend),
        current_parameters,
    )
    thermal = heat_backend.bind_differentiable(
        prepare(feedback.one_way.thermal_problem, heat_backend),
        heat_parameters,
    )
    system = DifferentiableSelfConsistentElectrothermal.bind(
        feedback,
        current,
        thermal,
        feedback_parameters,
    )
    current_values = system.initial_current_values
    thermal_values = system.initial_thermal_values
    feedback_values = system.initial_feedback_values
    weights = jnp.linspace(
        0.75,
        1.25,
        current._engine.payload.coordinates.shape[0],
        dtype=jnp.float64,
    )
    weights /= jnp.sum(weights)
    adjoint = system.vjp(
        current_values,
        thermal_values,
        feedback_values,
        weights,
    )

    def objective(
        electrical: jax.Array,
        thermal_: jax.Array,
        feedback_: jax.Array,
    ) -> float:
        result = system.solve(electrical, thermal_, feedback_)
        assert bool(result.converged)
        return float(jnp.vdot(weights, result.temperature))

    def central_difference(
        argument: int,
        values: jax.Array,
        index: int,
        step: float,
    ) -> float:
        plus = values.at[index].add(step)
        minus = values.at[index].add(-step)
        arguments = [current_values, thermal_values, feedback_values]
        arguments[argument] = plus
        upper = objective(*arguments)
        arguments[argument] = minus
        lower = objective(*arguments)
        return (upper - lower) / (2.0 * step)

    finite_current = np.asarray(
        [
            central_difference(0, current_values, 0, 2.0e-5),
            central_difference(0, current_values, 1, 2.0e-1),
        ]
    )
    finite_thermal = np.asarray([central_difference(1, thermal_values, 0, 1.2e-2)])
    finite_feedback = np.asarray([central_difference(2, feedback_values, 0, 3.0e-7)])

    np.testing.assert_allclose(
        adjoint.current_parameter_gradient,
        finite_current,
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        adjoint.thermal_parameter_gradient,
        finite_thermal,
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        adjoint.feedback_parameter_gradient,
        finite_feedback,
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    assert float(adjoint.state.transfer_relative_error) < 2.0e-15
    assert float(adjoint.state.heat_balance_relative_error) <= feedback.iteration.residual_tolerance
