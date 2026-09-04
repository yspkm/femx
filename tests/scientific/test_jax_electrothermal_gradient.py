import pytest
from tests.electrothermal_support import (
    parameterized_microheater_coupling,
    triangle_areas,
)

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from femx.backends.jax.electrothermal import (  # noqa: E402
    DifferentiableOneWayElectrothermal,
)
from femx.backends.jax.steady_current import JaxSteadyCurrentBackend  # noqa: E402
from femx.backends.jax.steady_heat import JaxSteadyHeatBackend  # noqa: E402
from femx.runtime import prepare  # noqa: E402

pytestmark = [pytest.mark.scientific, pytest.mark.requires_jax]


def test_microheater_chain_conserves_power_and_matches_central_finite_difference() -> None:
    coupling, current_parameters, heat_parameters = parameterized_microheater_coupling(intervals=6)
    current_backend = JaxSteadyCurrentBackend()
    heat_backend = JaxSteadyHeatBackend()
    system = DifferentiableOneWayElectrothermal(
        coupling,
        current_backend.bind_differentiable(
            prepare(coupling.electrical_problem, current_backend),
            current_parameters,
        ),
        heat_backend.bind_differentiable(
            prepare(coupling.thermal_problem, heat_backend),
            heat_parameters,
        ),
    )
    current_initial = system.initial_current_values
    thermal_initial = system.initial_thermal_values
    temperature = system.temperature(current_initial, thermal_initial)
    weights = jnp.asarray(
        triangle_areas(coupling.thermal_problem)[:, None] * np.full((1, 3), 1.0 / 3.0),
        dtype=jnp.float64,
    )
    nodal_weights = jnp.zeros_like(temperature)
    cells = jnp.asarray(
        coupling.thermal_problem.mesh.topology.connectivity,
        dtype=jnp.int32,
    )
    nodal_weights = nodal_weights.at[cells.reshape(-1)].add(weights.reshape(-1))
    nodal_weights /= jnp.sum(nodal_weights)

    def objective(current_values: jax.Array, thermal_values: jax.Array) -> jax.Array:
        return jnp.vdot(
            nodal_weights,
            system.temperature(current_values, thermal_values),
        )

    reverse_current, reverse_thermal = jax.jit(jax.grad(objective, argnums=(0, 1)))(
        current_initial,
        thermal_initial,
    )
    explicit = system.vjp(current_initial, thermal_initial, nodal_weights)

    def central_difference(
        first: jax.Array,
        second: jax.Array,
        *,
        first_argument: bool,
    ) -> np.ndarray:
        differentiated = first if first_argument else second
        values = []
        for index, value in enumerate(np.asarray(differentiated)):
            step = 2.0e-5 * max(abs(float(value)), 1.0)
            plus = differentiated.at[index].add(step)
            minus = differentiated.at[index].add(-step)
            if first_argument:
                plus_value = float(objective(plus, second))
                minus_value = float(objective(minus, second))
            else:
                plus_value = float(objective(first, plus))
                minus_value = float(objective(first, minus))
            values.append((plus_value - minus_value) / (2.0 * step))
        return np.asarray(values)

    finite_current = central_difference(
        current_initial,
        thermal_initial,
        first_argument=True,
    )
    finite_thermal = central_difference(
        current_initial,
        thermal_initial,
        first_argument=False,
    )

    voltage = float(current_initial[0])
    heater_conductivity = float(current_initial[1])
    contact_conductivity = float(current_parameters.values["contact_conductivity"])
    resistance_per_area = 1.0e-6 / heater_conductivity + 1.0e-6 / contact_conductivity
    expected_power = 0.5e-6 * voltage**2 / resistance_per_area

    assert float(jnp.max(temperature) - 300.0) > 0.0
    assert float(explicit.transfer.electrical_joule_power) == pytest.approx(
        expected_power,
        rel=3.0e-11,
    )
    assert float(explicit.transfer.relative_error) == 0.0
    assert float(explicit.thermal_energy.relative_error) < 2.0e-11
    np.testing.assert_allclose(
        explicit.current.parameter_gradient,
        reverse_current,
        rtol=3.0e-11,
        atol=3.0e-11,
    )
    np.testing.assert_allclose(
        explicit.thermal.parameter_gradient,
        reverse_thermal,
        rtol=3.0e-11,
        atol=3.0e-11,
    )
    np.testing.assert_allclose(
        explicit.current.parameter_gradient,
        finite_current,
        rtol=2.0e-7,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        explicit.thermal.parameter_gradient,
        finite_thermal,
        rtol=2.0e-7,
        atol=2.0e-9,
    )
