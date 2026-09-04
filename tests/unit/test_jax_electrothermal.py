import pytest
from tests.electrothermal_support import parameterized_microheater_coupling

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from femx.backends.jax.electrothermal import (  # noqa: E402
    DifferentiableOneWayElectrothermal,
)
from femx.backends.jax.steady_current import JaxSteadyCurrentBackend  # noqa: E402
from femx.backends.jax.steady_heat import JaxSteadyHeatBackend  # noqa: E402
from femx.core.errors import ContractError  # noqa: E402
from femx.runtime import prepare  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def _bound_system(*, intervals: int = 2) -> DifferentiableOneWayElectrothermal:
    coupling, current_parameters, heat_parameters = parameterized_microheater_coupling(
        intervals=intervals
    )
    current_backend = JaxSteadyCurrentBackend()
    heat_backend = JaxSteadyHeatBackend()
    current = current_backend.bind_differentiable(
        prepare(coupling.electrical_problem, current_backend),
        current_parameters,
    )
    thermal = heat_backend.bind_differentiable(
        prepare(coupling.thermal_problem, heat_backend),
        heat_parameters,
    )
    return DifferentiableOneWayElectrothermal(coupling, current, thermal)


def test_composed_electrothermal_map_supports_jit_and_two_adjoint_pullback() -> None:
    system = _bound_system(intervals=4)
    assert system.current.problem is system.coupling.electrical_problem
    assert system.thermal.problem is system.coupling.thermal_problem
    assert system.current.parameter_names == ("applied_voltage", "heater_conductivity")
    assert system.thermal.parameter_names == ("thermal_conductivity",)
    np.testing.assert_array_equal(system.initial_current_values, (0.2, 2.0e3))
    np.testing.assert_array_equal(system.initial_thermal_values, (120.0,))

    temperature = jax.jit(system.temperature)(
        system.initial_current_values,
        system.initial_thermal_values,
    )
    weights = jnp.linspace(0.5, 1.5, temperature.size, dtype=jnp.float64)
    weights /= jnp.sum(weights)

    def objective(current_values: jax.Array, thermal_values: jax.Array) -> jax.Array:
        return jnp.vdot(weights, system.temperature(current_values, thermal_values))

    automatic_current, automatic_thermal = jax.jit(jax.grad(objective, argnums=(0, 1)))(
        system.initial_current_values, system.initial_thermal_values
    )
    result = system.vjp(
        system.initial_current_values,
        system.initial_thermal_values,
        weights,
    )

    assert float(jnp.min(temperature)) >= 300.0
    assert float(jnp.max(temperature)) > 300.0
    np.testing.assert_allclose(result.thermal.temperature, temperature, atol=4.0e-12)
    np.testing.assert_array_equal(
        result.current.joule_cotangent,
        result.thermal.additive_cell_heat_source_gradient,
    )
    np.testing.assert_allclose(
        result.current.parameter_gradient,
        automatic_current,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        result.thermal.parameter_gradient,
        automatic_thermal,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        result.current.parameter_gradient,
        result.current.direct_parameter_gradient + result.current.indirect_parameter_gradient,
        rtol=0.0,
        atol=0.0,
    )
    assert float(result.current.adjoint_backward_error) < 5.0e-16
    assert float(result.thermal.adjoint_backward_error) < 5.0e-16
    assert float(result.transfer.electrical_joule_power) > 0.0
    assert result.transfer.power_unit == "W/m"
    np.testing.assert_allclose(
        result.transfer.electrical_joule_power,
        result.transfer.thermal_source_power,
        rtol=0.0,
        atol=0.0,
    )
    assert float(result.transfer.relative_error) == 0.0
    np.testing.assert_allclose(
        result.thermal_energy.variational_heat_load,
        result.transfer.thermal_source_power,
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        result.thermal_energy.dirichlet_reaction,
        -result.thermal_energy.variational_heat_load,
        rtol=1.0e-11,
        atol=2.0e-12,
    )
    assert result.thermal_energy.power_unit == "W/m"
    assert float(result.thermal_energy.relative_error) < 5.0e-12
    direct_balance = system.thermal_energy_balance(
        system.initial_current_values,
        system.initial_thermal_values,
    )
    np.testing.assert_allclose(
        direct_balance.variational_heat_load,
        result.thermal_energy.variational_heat_load,
        rtol=0.0,
        atol=0.0,
    )


def test_electrothermal_binding_and_zero_power_contracts_are_explicit() -> None:
    first = _bound_system()
    second = _bound_system()

    with pytest.raises(ContractError, match="bound current map"):
        DifferentiableOneWayElectrothermal(
            first.coupling,
            second.current,
            first.thermal,
        )
    with pytest.raises(ContractError, match="bound heat map"):
        DifferentiableOneWayElectrothermal(
            first.coupling,
            first.current,
            second.thermal,
        )

    zero = jnp.zeros(
        (first.coupling.electrical_problem.mesh.topology.cell_count,),
        dtype=jnp.float64,
    )
    balance = first.transfer_balance(zero)
    assert float(balance.electrical_joule_power) == 0.0
    assert float(balance.thermal_source_power) == 0.0
    assert float(balance.relative_error) == 0.0

    constant_temperature = jnp.full(
        (first.coupling.thermal_problem.mesh.geometry.node_count,),
        300.0,
        dtype=jnp.float64,
    )
    zero_thermal = first._thermal_energy_balance(
        first.initial_thermal_values,
        zero,
        constant_temperature,
    )
    assert float(zero_thermal.variational_heat_load) == 0.0
    assert float(zero_thermal.dirichlet_reaction) == 0.0
    assert float(zero_thermal.relative_error) == 0.0
