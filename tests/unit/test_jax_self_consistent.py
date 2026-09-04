from dataclasses import replace

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
from femx.core.errors import ContractError  # noqa: E402
from femx.core.parameters import (  # noqa: E402
    ParameterRole,
    ParameterSchema,
    ParameterSpec,
)
from femx.runtime import prepare  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def _bound_system(*, intervals: int = 2) -> DifferentiableSelfConsistentElectrothermal:
    feedback, current_parameters, heat_parameters, feedback_parameters = (
        parameterized_self_consistent_microheater(intervals=intervals)
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
    return DifferentiableSelfConsistentElectrothermal.bind(
        feedback,
        current,
        thermal,
        feedback_parameters,
    )


def test_self_consistent_forward_converges_and_conserves_transferred_power() -> None:
    system = _bound_system(intervals=4)
    state = system.solve(
        system.initial_current_values,
        system.initial_thermal_values,
        system.initial_feedback_values,
    )

    assert bool(state.converged)
    assert int(state.iterations) >= system.feedback.iteration.minimum_iterations
    assert float(state.update_error) <= 1.0
    assert float(state.current_residual_error) <= system.feedback.iteration.residual_tolerance
    assert float(state.heat_residual_error) <= system.feedback.iteration.residual_tolerance
    assert float(jnp.min(state.temperature)) >= 300.0
    assert float(jnp.max(state.temperature)) > 300.0
    assert state.conductivity_unit == "S/m"
    assert state.joule_heat_density_unit == "W/m^3"
    assert state.power_unit == "W/m"

    heater_cells, contact_cells = system.current._engine.payload.region_cells
    heater_conductivity = state.cell_nodal_conductivity[heater_cells]
    contact_conductivity = state.cell_nodal_conductivity[contact_cells]
    assert float(jnp.min(heater_conductivity)) < 2.0e3
    assert float(jnp.max(heater_conductivity)) <= 2.0e3
    np.testing.assert_allclose(contact_conductivity, 2.0e5, rtol=0.0, atol=0.0)
    assert float(jnp.min(state.cell_nodal_joule_heat_density)) >= 0.0
    assert float(state.electrical_joule_power) > 0.0
    np.testing.assert_allclose(
        state.electrical_joule_power,
        state.thermal_joule_load,
        rtol=2.0e-15,
        atol=0.0,
    )
    assert float(state.transfer_relative_error) < 2.0e-15
    assert float(state.heat_balance_relative_error) <= system.feedback.iteration.residual_tolerance


def test_self_consistent_reverse_rule_matches_explicit_coupled_adjoint() -> None:
    system = _bound_system(intervals=2)
    current_values = system.initial_current_values
    thermal_values = system.initial_thermal_values
    feedback_values = system.initial_feedback_values
    temperature = jax.jit(system.temperature)(
        current_values,
        thermal_values,
        feedback_values,
    )
    potential = jax.jit(system.potential)(
        current_values,
        thermal_values,
        feedback_values,
    )
    assert potential.shape == temperature.shape
    np.testing.assert_allclose(np.asarray(potential)[[0, -1]], (0.0, 0.2), atol=2.0e-15)
    weights = jnp.linspace(0.5, 1.5, temperature.size, dtype=jnp.float64)
    weights /= jnp.sum(weights)

    def objective(
        current: jax.Array,
        thermal: jax.Array,
        feedback: jax.Array,
    ) -> jax.Array:
        return jnp.vdot(weights, system.temperature(current, thermal, feedback))

    automatic = jax.jit(jax.grad(objective, argnums=(0, 1, 2)))(
        current_values,
        thermal_values,
        feedback_values,
    )
    explicit = system.vjp(
        current_values,
        thermal_values,
        feedback_values,
        weights,
    )

    np.testing.assert_allclose(explicit.state.temperature, temperature, rtol=0.0, atol=5.0e-11)
    np.testing.assert_allclose(
        explicit.current_parameter_gradient,
        automatic[0],
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        explicit.thermal_parameter_gradient,
        automatic[1],
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        explicit.feedback_parameter_gradient,
        automatic[2],
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    assert explicit.current_parameter_names == (
        "applied_voltage",
        "heater_conductivity",
    )
    assert explicit.current_parameter_units == ("V", "S/m")
    assert explicit.thermal_parameter_names == ("thermal_conductivity",)
    assert explicit.thermal_parameter_units == ("W/(m*K)",)
    assert explicit.feedback_parameter_names == ("heater_temperature_coefficient",)
    assert explicit.feedback_parameter_units == ("1/K",)
    assert float(explicit.adjoint_backward_error) < 1.0e-14


def test_self_consistent_binding_and_cotangent_contracts_are_explicit() -> None:
    first = _bound_system()
    second = _bound_system()

    with pytest.raises(ContractError, match="bound current map"):
        DifferentiableSelfConsistentElectrothermal.bind(
            first.feedback,
            second.current,
            first.thermal,
            first.feedback.parameters.bind({"heater_temperature_coefficient": 3.0e-3}),
        )
    with pytest.raises(ContractError, match="bound heat map"):
        DifferentiableSelfConsistentElectrothermal.bind(
            first.feedback,
            first.current,
            second.thermal,
            first.feedback.parameters.bind({"heater_temperature_coefficient": 3.0e-3}),
        )

    unbounded_schema = ParameterSchema(
        (
            ParameterSpec(
                "heater_temperature_coefficient",
                unit="1/K",
                role=ParameterRole.DESIGN,
            ),
        )
    )
    complex_feedback = replace(first.feedback, parameters=unbounded_schema)
    with pytest.raises(ContractError, match="finite real scalars"):
        DifferentiableSelfConsistentElectrothermal.bind(
            complex_feedback,
            first.current,
            first.thermal,
            unbounded_schema.bind({"heater_temperature_coefficient": 3.0e-3 + 0.0j}),
        )

    with pytest.raises(ContractError, match="cotangent must have shape"):
        first.vjp(
            first.initial_current_values,
            first.initial_thermal_values,
            first.initial_feedback_values,
            jnp.zeros((1,), dtype=jnp.float64),
        )
    node_count = first.current._engine.payload.coordinates.shape[0]
    with pytest.raises(ContractError, match="exact float64"):
        first.vjp(
            first.initial_current_values,
            first.initial_thermal_values,
            first.initial_feedback_values,
            jnp.zeros((node_count,), dtype=jnp.float32),
        )
