from dataclasses import replace

import pytest
from tests.support import DummyPhysics, structured_unit_square_mesh

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from femx.backends.jax.steady_heat import JaxSteadyHeatBackend  # noqa: E402
from femx.backends.protocol import (  # noqa: E402
    BackendDescriptor,
    PreparedProblem,
    SolveRequest,
)
from femx.core.capabilities import GradientMethod  # noqa: E402
from femx.core.errors import BackendError, CapabilityError, ContractError  # noqa: E402
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
from femx.runtime import prepare, solve  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def _adjoint_problem(*, active: bool = True) -> tuple[Problem, ParameterValues]:
    design = ParameterRole.DESIGN if active else ParameterRole.FIXED
    control = ParameterRole.CONTROL if active else ParameterRole.FIXED
    schema = ParameterSchema(
        (
            ParameterSpec("flux", unit="W/m^2", role=design),
            ParameterSpec("source", unit="W/m^3", role=ParameterRole.FIXED),
            ParameterSpec(
                "left_temperature",
                unit="K",
                role=control,
                lower_bound=250.0,
                upper_bound=400.0,
            ),
            ParameterSpec(
                "conductivity",
                unit="W/(m*K)",
                role=design,
                lower_bound=0.1,
                upper_bound=10.0,
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
        heat_flux_boundaries=(HeatFluxBoundary("right", ParameterReference("flux")),),
        gradient_method=GradientMethod.ADJOINT,
    )
    problem = Problem("adjoint-heat", structured_unit_square_mesh(2), physics, parameters=schema)
    parameters = schema.bind(
        {
            "flux": 0.7,
            "source": 1.25,
            "left_temperature": 300.0,
            "conductivity": 2.5,
        }
    )
    return problem, parameters


def test_bound_state_map_supports_jit_reverse_mode_and_explicit_vjp() -> None:
    problem, parameters = _adjoint_problem()
    backend = JaxSteadyHeatBackend()
    prepared = prepare(problem, backend)
    bound = backend.bind_differentiable(prepared, parameters)

    with pytest.raises(ContractError, match="parameter key mismatch"):
        backend.bind_differentiable(prepared, ParameterValues({"flux": 0.7}))

    assert bound.parameter_names == ("flux", "left_temperature", "conductivity")
    assert bound.parameter_units == ("W/m^2", "K", "W/(m*K)")
    np.testing.assert_array_equal(bound.initial_values, (0.7, 300.0, 2.5))

    temperature = jax.jit(bound.temperature)(bound.initial_values)
    normal_solution = solve(
        prepared,
        backend,
        request=SolveRequest(parameters=parameters),
    )
    np.testing.assert_allclose(
        temperature,
        normal_solution.fields["temperature"].values,
        rtol=0.0,
        atol=2.0e-13,
    )

    weights = jnp.linspace(1.0, 2.0, temperature.size, dtype=jnp.float64)
    weights = weights / jnp.sum(weights)
    automatic_gradient = jax.jit(
        jax.grad(lambda values: jnp.vdot(weights, bound.temperature(values)))
    )(bound.initial_values)
    result = bound.vjp(bound.initial_values, weights)

    np.testing.assert_allclose(result.temperature, temperature, rtol=0.0, atol=2.0e-13)
    np.testing.assert_allclose(
        result.temperature_cotangent,
        weights,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.parameter_gradient,
        automatic_gradient,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert result.parameter_names == bound.parameter_names
    assert result.parameter_units == bound.parameter_units
    assert float(result.adjoint_backward_error) < 2.0e-16


def test_bound_heat_map_differentiates_an_additive_cell_source() -> None:
    problem, parameters = _adjoint_problem()
    backend = JaxSteadyHeatBackend()
    bound = backend.bind_differentiable(prepare(problem, backend), parameters)
    source = jnp.linspace(
        0.2,
        0.9,
        problem.mesh.topology.cell_count,
        dtype=jnp.float64,
    )
    weights = jnp.linspace(
        0.5,
        1.5,
        problem.mesh.geometry.node_count,
        dtype=jnp.float64,
    )
    weights /= jnp.sum(weights)

    def objective(active: jax.Array, additive_source: jax.Array) -> jax.Array:
        return jnp.vdot(
            weights,
            bound.temperature_with_cell_source(active, additive_source),
        )

    temperature = jax.jit(bound.temperature_with_cell_source)(bound.initial_values, source)
    automatic_parameter_gradient, automatic_source_gradient = jax.jit(
        jax.grad(objective, argnums=(0, 1))
    )(bound.initial_values, source)
    result = bound.source_vjp(bound.initial_values, source, weights)

    np.testing.assert_allclose(result.temperature, temperature, rtol=0.0, atol=2.0e-13)
    np.testing.assert_array_equal(result.additive_cell_heat_source, source)
    np.testing.assert_allclose(
        result.parameter_gradient,
        automatic_parameter_gradient,
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        result.additive_cell_heat_source_gradient,
        automatic_source_gradient,
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    assert result.source_unit == "W/m^3"
    assert result.parameter_names == bound.parameter_names
    assert result.parameter_units == bound.parameter_units
    assert float(result.adjoint_backward_error) < 2.0e-16


def test_differentiable_binding_and_vector_contracts_fail_explicitly() -> None:
    problem, parameters = _adjoint_problem()
    backend = JaxSteadyHeatBackend()
    prepared = prepare(problem, backend)
    bound = backend.bind_differentiable(prepared, parameters)

    with pytest.raises(ContractError, match="shape"):
        bound.temperature(jnp.ones((2,), dtype=jnp.float64))
    with pytest.raises(ContractError, match="exact float64"):
        bound.temperature(jnp.asarray((0.7, 300.0, 2.5), dtype=jnp.float32))
    with pytest.raises(ContractError, match="cotangent must have shape"):
        bound.vjp(bound.initial_values, jnp.ones((2,), dtype=jnp.float64))
    with pytest.raises(ContractError, match="cotangent must use the exact float64"):
        bound.vjp(
            bound.initial_values,
            jnp.ones((problem.mesh.geometry.node_count,), dtype=jnp.float32),
        )

    outside_temperature_bound = bound.initial_values.at[1].set(401.0)
    assert np.isnan(np.asarray(bound.temperature(outside_temperature_bound))).all()
    invalid_automatic_gradient = jax.grad(lambda values: jnp.sum(bound.temperature(values)))(
        outside_temperature_bound
    )
    assert np.isnan(np.asarray(invalid_automatic_gradient)).all()
    negative_conductivity = bound.initial_values.at[2].set(-1.0)
    invalid_vjp = bound.vjp(
        negative_conductivity,
        jnp.ones((problem.mesh.geometry.node_count,), dtype=jnp.float64),
    )
    assert np.isnan(np.asarray(invalid_vjp.temperature)).all()
    assert np.isnan(np.asarray(invalid_vjp.adjoint)).all()
    assert np.isnan(np.asarray(invalid_vjp.parameter_gradient)).all()
    assert np.isnan(float(invalid_vjp.adjoint_backward_error))


def test_additive_heat_source_contracts_fail_explicitly() -> None:
    problem, parameters = _adjoint_problem()
    backend = JaxSteadyHeatBackend()
    bound = backend.bind_differentiable(prepare(problem, backend), parameters)
    cell_count = problem.mesh.topology.cell_count
    node_count = problem.mesh.geometry.node_count
    source = jnp.ones((cell_count,), dtype=jnp.float64)
    cotangent = jnp.ones((node_count,), dtype=jnp.float64)

    with pytest.raises(ContractError, match="additive cell source must have shape"):
        bound.temperature_with_cell_source(
            bound.initial_values,
            jnp.ones((cell_count - 1,), dtype=jnp.float64),
        )
    with pytest.raises(ContractError, match="additive cell source must use the exact float64"):
        bound.source_vjp(
            bound.initial_values,
            jnp.ones((cell_count,), dtype=jnp.float32),
            cotangent,
        )

    nonfinite_source = source.at[0].set(jnp.inf)
    invalid_temperature = bound.temperature_with_cell_source(
        bound.initial_values,
        nonfinite_source,
    )
    invalid_vjp = bound.source_vjp(
        bound.initial_values,
        nonfinite_source,
        cotangent,
    )
    assert np.isnan(np.asarray(invalid_temperature)).all()
    assert np.isnan(np.asarray(invalid_vjp.temperature)).all()
    assert np.isnan(np.asarray(invalid_vjp.adjoint)).all()
    assert np.isnan(np.asarray(invalid_vjp.parameter_gradient)).all()
    assert np.isnan(np.asarray(invalid_vjp.additive_cell_heat_source_gradient)).all()
    assert np.isnan(float(invalid_vjp.adjoint_backward_error))


def test_differentiable_binding_requires_identity_physics_and_active_roles() -> None:
    problem, parameters = _adjoint_problem()
    backend = JaxSteadyHeatBackend()
    prepared = prepare(problem, backend)
    wrong_descriptor = PreparedProblem(
        BackendDescriptor("other", "1"),
        prepared.problem,
        prepared.payload,
    )
    with pytest.raises(BackendError, match="identity"):
        backend.bind_differentiable(wrong_descriptor, parameters)
    wrong_payload = PreparedProblem(backend.descriptor, prepared.problem, object())
    with pytest.raises(BackendError, match="payload"):
        backend.bind_differentiable(wrong_payload, parameters)
    wrong_physics_problem = Problem("dummy", problem.mesh, DummyPhysics())
    wrong_physics = PreparedProblem(backend.descriptor, wrong_physics_problem, prepared.payload)
    with pytest.raises(BackendError, match="not a steady-heat"):
        backend.bind_differentiable(wrong_physics, parameters)

    forward_physics = replace(problem.physics, gradient_method=GradientMethod.NONE)
    forward_problem = replace(problem, physics=forward_physics)
    forward_prepared = prepare(forward_problem, backend)
    with pytest.raises(CapabilityError, match="gradient_method=adjoint"):
        backend.bind_differentiable(forward_prepared, parameters)

    fixed_problem, fixed_parameters = _adjoint_problem(active=False)
    fixed_prepared = prepare(fixed_problem, backend)
    with pytest.raises(ContractError, match="DESIGN or CONTROL"):
        backend.bind_differentiable(fixed_prepared, fixed_parameters)
