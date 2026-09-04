from dataclasses import replace

import pytest
from tests.support import DummyPhysics, structured_unit_square_mesh

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from femx.backends.jax.steady_current import JaxSteadyCurrentBackend  # noqa: E402
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
    ConductiveRegion,
    CurrentFluxBoundary,
    PotentialBoundary,
    SteadyCurrent,
)
from femx.runtime import prepare, solve  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def _adjoint_problem(*, active: bool = True) -> tuple[Problem, ParameterValues]:
    design = ParameterRole.DESIGN if active else ParameterRole.FIXED
    control = ParameterRole.CONTROL if active else ParameterRole.FIXED
    schema = ParameterSchema(
        (
            ParameterSpec("current_load", unit="A/m^2", role=design),
            ParameterSpec("source", unit="A/m^3", role=ParameterRole.FIXED),
            ParameterSpec(
                "left_potential",
                unit="V",
                role=control,
                lower_bound=-1.0,
                upper_bound=1.0,
            ),
            ParameterSpec(
                "conductivity",
                unit="S/m",
                role=design,
                lower_bound=0.1,
                upper_bound=10.0,
            ),
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
        potential_boundaries=(PotentialBoundary("left", ParameterReference("left_potential")),),
        current_flux_boundaries=(CurrentFluxBoundary("right", ParameterReference("current_load")),),
        gradient_method=GradientMethod.ADJOINT,
    )
    problem = Problem(
        "adjoint-current",
        structured_unit_square_mesh(2),
        physics,
        parameters=schema,
    )
    parameters = schema.bind(
        {
            "current_load": 0.7,
            "source": 1.25,
            "left_potential": 0.1,
            "conductivity": 2.5,
        }
    )
    return problem, parameters


def test_bound_current_map_supports_potential_and_joule_reverse_mode() -> None:
    problem, parameters = _adjoint_problem()
    backend = JaxSteadyCurrentBackend()
    prepared = prepare(problem, backend)
    bound = backend.bind_differentiable(prepared, parameters)

    with pytest.raises(ContractError, match="parameter key mismatch"):
        backend.bind_differentiable(
            prepared,
            ParameterValues({"current_load": 0.7}),
        )

    assert bound.parameter_names == ("current_load", "left_potential", "conductivity")
    assert bound.parameter_units == ("A/m^2", "V", "S/m")
    np.testing.assert_array_equal(bound.initial_values, (0.7, 0.1, 2.5))

    potential = jax.jit(bound.potential)(bound.initial_values)
    normal_solution = solve(
        prepared,
        backend,
        request=SolveRequest(parameters=parameters),
    )
    np.testing.assert_allclose(
        potential,
        normal_solution.fields["potential"].values,
        rtol=0.0,
        atol=2.0e-13,
    )

    potential_weights = jnp.linspace(1.0, 2.0, potential.size, dtype=jnp.float64)
    potential_weights /= jnp.sum(potential_weights)
    automatic_potential_gradient = jax.jit(
        jax.grad(lambda values: jnp.vdot(potential_weights, bound.potential(values)))
    )(bound.initial_values)
    potential_vjp = bound.vjp(bound.initial_values, potential_weights)
    np.testing.assert_allclose(potential_vjp.potential, potential, atol=2.0e-13)
    np.testing.assert_allclose(
        potential_vjp.parameter_gradient,
        automatic_potential_gradient,
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    assert float(potential_vjp.adjoint_backward_error) < 3.0e-16

    joule = jax.jit(bound.joule_heat_density)(bound.initial_values)
    np.testing.assert_allclose(
        joule,
        normal_solution.fields["joule_heat_density"].values,
        rtol=0.0,
        atol=3.0e-13,
    )
    joule_weights = jnp.linspace(0.5, 1.5, joule.size, dtype=jnp.float64)
    joule_weights /= jnp.sum(joule_weights)
    automatic_joule_gradient = jax.jit(
        jax.grad(lambda values: jnp.vdot(joule_weights, bound.joule_heat_density(values)))
    )(bound.initial_values)
    joule_vjp = bound.joule_vjp(bound.initial_values, joule_weights)
    np.testing.assert_allclose(joule_vjp.potential, potential, atol=2.0e-13)
    np.testing.assert_allclose(joule_vjp.joule_heat_density, joule, atol=3.0e-13)
    np.testing.assert_allclose(joule_vjp.joule_cotangent, joule_weights, atol=0.0)
    np.testing.assert_allclose(
        joule_vjp.parameter_gradient,
        joule_vjp.direct_parameter_gradient + joule_vjp.indirect_parameter_gradient,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        joule_vjp.parameter_gradient,
        automatic_joule_gradient,
        rtol=8.0e-13,
        atol=8.0e-13,
    )
    assert joule_vjp.parameter_names == bound.parameter_names
    assert joule_vjp.parameter_units == bound.parameter_units
    assert float(joule_vjp.adjoint_backward_error) < 3.0e-16


def test_current_differentiable_vector_and_cotangent_contracts_fail_explicitly() -> None:
    problem, parameters = _adjoint_problem()
    backend = JaxSteadyCurrentBackend()
    bound = backend.bind_differentiable(prepare(problem, backend), parameters)

    with pytest.raises(ContractError, match="shape"):
        bound.potential(jnp.ones((2,), dtype=jnp.float64))
    with pytest.raises(ContractError, match="exact float64"):
        bound.potential(jnp.asarray((0.7, 0.1, 2.5), dtype=jnp.float32))
    with pytest.raises(ContractError, match="potential cotangent must have shape"):
        bound.vjp(bound.initial_values, jnp.ones((2,), dtype=jnp.float64))
    with pytest.raises(ContractError, match="potential cotangent must use the exact float64"):
        bound.vjp(
            bound.initial_values,
            jnp.ones((problem.mesh.geometry.node_count,), dtype=jnp.float32),
        )
    with pytest.raises(ContractError, match="Joule-density cotangent must have shape"):
        bound.joule_vjp(bound.initial_values, jnp.ones((2,), dtype=jnp.float64))
    with pytest.raises(ContractError, match="Joule-density cotangent must use the exact float64"):
        bound.joule_vjp(
            bound.initial_values,
            jnp.ones((problem.mesh.topology.cell_count,), dtype=jnp.float32),
        )

    outside_potential_bound = bound.initial_values.at[1].set(1.1)
    assert np.isnan(np.asarray(bound.potential(outside_potential_bound))).all()
    assert np.isnan(np.asarray(bound.joule_heat_density(outside_potential_bound))).all()
    invalid_gradient = jax.grad(lambda values: jnp.sum(bound.joule_heat_density(values)))(
        outside_potential_bound
    )
    assert np.isnan(np.asarray(invalid_gradient)).all()
    negative_conductivity = bound.initial_values.at[2].set(-1.0)
    invalid_vjp = bound.joule_vjp(
        negative_conductivity,
        jnp.ones((problem.mesh.topology.cell_count,), dtype=jnp.float64),
    )
    assert np.isnan(np.asarray(invalid_vjp.potential)).all()
    assert np.isnan(np.asarray(invalid_vjp.joule_heat_density)).all()
    assert np.isnan(np.asarray(invalid_vjp.parameter_gradient)).all()
    assert np.isnan(float(invalid_vjp.adjoint_backward_error))


def test_current_differentiable_binding_requires_identity_physics_and_active_roles() -> None:
    problem, parameters = _adjoint_problem()
    backend = JaxSteadyCurrentBackend()
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
    with pytest.raises(BackendError, match="not a steady-current"):
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
