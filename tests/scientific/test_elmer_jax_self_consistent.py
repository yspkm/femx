import pytest
from tests.electrothermal_support import parameterized_self_consistent_microheater

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from femx.backends.elmer.self_consistent import (  # noqa: E402
    ElmerSelfConsistentSolveRequest,
)
from femx.backends.jax.self_consistent import (  # noqa: E402
    DifferentiableSelfConsistentElectrothermal,
)
from femx.backends.jax.steady_current import JaxSteadyCurrentBackend  # noqa: E402
from femx.backends.jax.steady_heat import JaxSteadyHeatBackend  # noqa: E402
from femx.backends.protocol import ExecutionPolicy, PrepareRequest  # noqa: E402
from femx.runtime import prepare  # noqa: E402

pytestmark = [
    pytest.mark.scientific,
    pytest.mark.requires_elmer,
    pytest.mark.requires_jax,
]


def test_locked_elmer_and_jax_match_self_consistent_microheater_fields(
    locked_elmer_electrothermal_backend,
    tmp_path,
) -> None:
    feedback, current_parameters, heat_parameters, feedback_parameters = (
        parameterized_self_consistent_microheater(intervals=2)
    )
    current_backend = JaxSteadyCurrentBackend()
    heat_backend = JaxSteadyHeatBackend()
    jax_system = DifferentiableSelfConsistentElectrothermal.bind(
        feedback,
        current_backend.bind_differentiable(
            prepare(feedback.one_way.electrical_problem, current_backend),
            current_parameters,
        ),
        heat_backend.bind_differentiable(
            prepare(feedback.one_way.thermal_problem, heat_backend),
            heat_parameters,
        ),
        feedback_parameters,
    )
    jax_state = jax_system.solve(
        jax_system.initial_current_values,
        jax_system.initial_thermal_values,
        jax_system.initial_feedback_values,
    )
    assert bool(jax_state.converged)

    def run_elmer(current_values, thermal_values, feedback_values, attempt: str):
        run_directory = tmp_path / attempt
        return locked_elmer_electrothermal_backend.solve(
            locked_elmer_electrothermal_backend.prepare(
                feedback,
                PrepareRequest(run_directory=run_directory),
            ),
            ElmerSelfConsistentSolveRequest(
                current_parameters=current_values,
                thermal_parameters=thermal_values,
                feedback_parameters=feedback_values,
                run_directory=run_directory,
                policy=ExecutionPolicy(
                    execution_authorized=True,
                    allow_external_process=True,
                ),
            ),
        )

    elmer_solution = run_elmer(
        current_parameters,
        heat_parameters,
        feedback_parameters,
        "baseline",
    )

    assert elmer_solution.convergence.status.value == "converged"
    np.testing.assert_allclose(
        elmer_solution.fields["potential"].values,
        jax_state.potential,
        rtol=0.0,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        elmer_solution.fields["temperature"].values,
        jax_state.temperature,
        rtol=0.0,
        atol=2.0e-8,
    )
    np.testing.assert_allclose(
        elmer_solution.fields["electric_conductivity"].values,
        jax_state.cell_nodal_conductivity,
        rtol=2.0e-10,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        elmer_solution.fields["joule_heat_density"].values,
        jax_state.cell_nodal_joule_heat_density,
        rtol=5.0e-9,
        atol=5.0e-1,
    )
    assert elmer_solution.observables["transfer_relative_error"] < 2.0e-15
    assert elmer_solution.observables["current_energy_balance_relative_error"] < 2.0e-11
    assert elmer_solution.observables["heat_balance_relative_error"] < 2.0e-9
    assert (
        elmer_solution.metadata["elmer_source_commit"] == "4f2d7e4b99f8f0dcf2f7ac579e056969373bf594"
    )

    weights = jnp.linspace(0.75, 1.25, jax_state.temperature.size, dtype=jnp.float64)
    weights /= jnp.sum(weights)
    adjoint = jax_system.vjp(
        jax_system.initial_current_values,
        jax_system.initial_thermal_values,
        jax_system.initial_feedback_values,
        weights,
    )
    current_base = dict(current_parameters.values)
    thermal_base = dict(heat_parameters.values)
    feedback_base = dict(feedback_parameters.values)

    def bind_active(schema, base, names, active):
        values = dict(base)
        values.update((name, float(value)) for name, value in zip(names, active, strict=True))
        return schema.bind(values)

    current_initial = np.asarray(jax_system.initial_current_values)
    thermal_initial = np.asarray(jax_system.initial_thermal_values)
    feedback_initial = np.asarray(jax_system.initial_feedback_values)

    def objective(
        current_active: np.ndarray,
        thermal_active: np.ndarray,
        feedback_active: np.ndarray,
        attempt: str,
    ) -> float:
        solution = run_elmer(
            bind_active(
                feedback.one_way.electrical_problem.parameters,
                current_base,
                jax_system.current.parameter_names,
                current_active,
            ),
            bind_active(
                feedback.one_way.thermal_problem.parameters,
                thermal_base,
                jax_system.thermal.parameter_names,
                thermal_active,
            ),
            bind_active(
                feedback.parameters,
                feedback_base,
                jax_system.feedback_parameter_names,
                feedback_active,
            ),
            attempt,
        )
        assert solution.convergence.status.value == "converged"
        return float(np.vdot(np.asarray(weights), solution.fields["temperature"].values))

    def finite_difference(argument: int, index: int, step: float) -> float:
        arguments = [current_initial.copy(), thermal_initial.copy(), feedback_initial.copy()]
        arguments[argument][index] += step
        upper = objective(*arguments, f"fd-{argument}-{index}-plus")
        arguments[argument][index] -= 2.0 * step
        lower = objective(*arguments, f"fd-{argument}-{index}-minus")
        return (upper - lower) / (2.0 * step)

    elmer_current_gradient = np.asarray(
        (
            finite_difference(0, 0, 2.0e-5),
            finite_difference(0, 1, 2.0e-1),
        )
    )
    elmer_thermal_gradient = np.asarray((finite_difference(1, 0, 1.2e-2),))
    elmer_feedback_gradient = np.asarray((finite_difference(2, 0, 3.0e-7),))
    np.testing.assert_allclose(
        adjoint.current_parameter_gradient,
        elmer_current_gradient,
        rtol=2.0e-5,
        atol=3.0e-7,
    )
    np.testing.assert_allclose(
        adjoint.thermal_parameter_gradient,
        elmer_thermal_gradient,
        rtol=2.0e-5,
        atol=3.0e-7,
    )
    np.testing.assert_allclose(
        adjoint.feedback_parameter_gradient,
        elmer_feedback_gradient,
        rtol=2.0e-5,
        atol=3.0e-7,
    )
