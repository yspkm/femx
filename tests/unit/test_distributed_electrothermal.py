from __future__ import annotations

from dataclasses import replace

import pytest
from tests.electrothermal_support import parameterized_self_consistent_microheater

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from femx.backends.jax.distributed_electrothermal import (  # noqa: E402
    DISTRIBUTED_ELECTROTHERMAL_SCHEMA,
    DistributedElectrothermalPlan,
    ElectrothermalAdjointPolicy,
    PackedDistributedElectrothermalInputs,
    PackedElectrothermalVector,
    build_distributed_electrothermal_runtime,
    pack_distributed_electrothermal_inputs,
    pack_distributed_electrothermal_inputs_host,
    prepare_distributed_electrothermal_plan,
    reconstruct_distributed_electrothermal_state,
)
from femx.backends.jax.scalar_cg import (  # noqa: E402
    ScalarH1CGPolicy,
    ScalarH1JacobiPolicy,
    build_packed_scalar_h1_jacobi_preconditioner_factory,
)
from femx.backends.jax.self_consistent import (  # noqa: E402
    DifferentiableSelfConsistentElectrothermal,
)
from femx.backends.jax.steady_current import JaxSteadyCurrentBackend  # noqa: E402
from femx.backends.jax.steady_heat import JaxSteadyHeatBackend  # noqa: E402
from femx.core.errors import ContractError  # noqa: E402
from femx.runtime import prepare  # noqa: E402
from femx.workflows import CoupledIterationPolicy  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def _bound_system(
    *,
    iteration: CoupledIterationPolicy | None = None,
    intervals: int = 2,
) -> DifferentiableSelfConsistentElectrothermal:
    feedback, current_parameters, thermal_parameters, feedback_parameters = (
        parameterized_self_consistent_microheater(intervals=intervals, iteration=iteration)
    )
    current_backend = JaxSteadyCurrentBackend()
    thermal_backend = JaxSteadyHeatBackend()
    current = current_backend.bind_differentiable(
        prepare(feedback.one_way.electrical_problem, current_backend),
        current_parameters,
    )
    thermal = thermal_backend.bind_differentiable(
        prepare(feedback.one_way.thermal_problem, thermal_backend),
        thermal_parameters,
    )
    return DifferentiableSelfConsistentElectrothermal.bind(
        feedback,
        current,
        thermal,
        feedback_parameters,
    )


def _plan(
    system: DifferentiableSelfConsistentElectrothermal,
    *,
    partition_count: int = 1,
) -> DistributedElectrothermalPlan:
    coordinates = np.asarray(system.current._engine.payload.coordinates)
    cells = np.asarray(system.current._engine.payload.cells)
    centroids = np.mean(coordinates[cells, 0], axis=1)
    width = float(np.max(coordinates[:, 0]))
    owners = np.minimum((partition_count * centroids / width).astype(np.int64), partition_count - 1)
    return prepare_distributed_electrothermal_plan(
        system,
        owners,
        partition_count=partition_count,
    )


def _runtime(plan: DistributedElectrothermalPlan):
    mesh = Mesh(np.asarray((jax.devices("cpu")[0],), dtype=object), ("partition",))
    return build_distributed_electrothermal_runtime(
        plan,
        mesh,
        ScalarH1CGPolicy(1.0e-12, 1.0e-14, 200),
        ElectrothermalAdjointPolicy(1.0e-9, 1.0e-12, 12, 30),
    )


def test_distributed_forward_and_coupled_vjp_match_dense_authority() -> None:
    system = _bound_system()
    plan = _plan(system)
    inputs = pack_distributed_electrothermal_inputs(plan, value_dtype=np.float64)
    runtime = _runtime(plan)
    current = jnp.asarray(plan.current_initial)
    thermal = jnp.asarray(plan.thermal_initial)
    feedback = jnp.asarray(plan.feedback_initial)

    forward = jax.jit(runtime.solve)(inputs, current, thermal, feedback)
    dense = system.solve(current, thermal, feedback)
    potential, temperature = reconstruct_distributed_electrothermal_state(
        plan,
        forward.state,
        current,
        thermal,
    )
    assert bool(forward.converged)
    np.testing.assert_allclose(potential, dense.potential, rtol=2.0e-13, atol=2.0e-14)
    np.testing.assert_allclose(temperature, dense.temperature, rtol=2.0e-13, atol=2.0e-11)
    assert float(forward.transfer_relative_error) < 2.0e-15
    assert float(forward.current_residual_error) <= system.feedback.iteration.residual_tolerance
    assert float(forward.heat_residual_error) <= system.feedback.iteration.residual_tolerance

    free_count = plan.layout.topology.free_dof_count
    owner_weights = jnp.where(inputs.owner_mask, 1.0 / free_count, 0.0)
    cotangent = PackedElectrothermalVector(jnp.zeros_like(owner_weights), owner_weights)
    explicit = jax.jit(runtime.vjp)(inputs, current, thermal, feedback, cotangent)
    full_weights = (
        jnp.zeros((plan.layout.topology.node_count,), dtype=jnp.float64)
        .at[jnp.asarray(plan.layout.topology.free_nodes)]
        .set(1.0 / free_count)
    )
    dense_vjp = system.vjp(current, thermal, feedback, full_weights)
    assert bool(explicit.adjoint_converged)
    assert float(explicit.adjoint_backward_error) < 1.0e-10
    np.testing.assert_allclose(
        explicit.current_parameter_gradient,
        dense_vjp.current_parameter_gradient,
        rtol=2.0e-9,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        explicit.thermal_parameter_gradient,
        dense_vjp.thermal_parameter_gradient,
        rtol=2.0e-9,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        explicit.feedback_parameter_gradient,
        dense_vjp.feedback_parameter_gradient,
        rtol=2.0e-9,
        atol=2.0e-10,
    )

    def objective(
        current_values: jax.Array,
        thermal_values: jax.Array,
        feedback_values: jax.Array,
    ) -> jax.Array:
        state = runtime.state(inputs, current_values, thermal_values, feedback_values)
        return jnp.sum(state.temperature * owner_weights)

    native = jax.jit(jax.grad(objective, argnums=(0, 1, 2)))(current, thermal, feedback)
    for observed, expected in zip(
        native,
        (
            explicit.current_parameter_gradient,
            explicit.thermal_parameter_gradient,
            explicit.feedback_parameter_gradient,
        ),
        strict=True,
    ):
        np.testing.assert_allclose(observed, expected, rtol=2.0e-11, atol=2.0e-12)


def test_cell_temperature_preserves_absolute_p1_values_and_reverse_rule() -> None:
    system = _bound_system()
    plan = _plan(system)
    inputs = pack_distributed_electrothermal_inputs(plan, value_dtype=np.float64)
    runtime = _runtime(plan)
    current = jnp.asarray(plan.current_initial)
    thermal = jnp.asarray(plan.thermal_initial)
    feedback = jnp.asarray(plan.feedback_initial)
    state = runtime.state(inputs, current, thermal, feedback)
    cell_temperature = jax.jit(runtime.cell_temperature)(inputs, state, thermal)
    dense_temperature = system.temperature(current, thermal, feedback)
    cells = np.asarray(system.thermal._engine.payload.cells)

    np.testing.assert_allclose(
        cell_temperature,
        np.asarray(dense_temperature)[cells][None, ...],
        rtol=2.0e-13,
        atol=2.0e-11,
    )

    def objective(thermal_values: jax.Array) -> jax.Array:
        solved = runtime.state(inputs, current, thermal_values, feedback)
        values = runtime.cell_temperature(inputs, solved, thermal_values)
        return jnp.mean(values)

    derivative = jax.grad(objective)(thermal)
    step = 1.0e-4
    finite_difference = (objective(thermal + step) - objective(thermal - step)) / (2.0 * step)
    np.testing.assert_allclose(derivative[0], finite_difference, rtol=2.0e-7, atol=2.0e-8)

    bad_shape = PackedElectrothermalVector(
        state.potential[:, :1],
        state.temperature,
    )
    with pytest.raises(ContractError, match="state must match owner"):
        runtime.cell_temperature(inputs, bad_shape, thermal)
    bad_dtype = PackedElectrothermalVector(
        state.potential.astype(jnp.float32),
        state.temperature.astype(jnp.float32),
    )
    with pytest.raises(ContractError, match="state must match input dtype"):
        runtime.cell_temperature(inputs, bad_dtype, thermal)
    with pytest.raises(ContractError, match="parameters must have shape"):
        runtime.cell_temperature(inputs, state, thermal[:0])
    with pytest.raises(ContractError, match="parameters must match input dtype"):
        runtime.cell_temperature(inputs, state, thermal.astype(jnp.float32))


def test_reference_shift_and_block_preconditioner_admit_float32_forward_and_vjp() -> None:
    iteration = CoupledIterationPolicy(
        max_iterations=100,
        minimum_iterations=2,
        relative_tolerance=2.0e-5,
        residual_tolerance=1.0e-4,
        potential_absolute_tolerance=1.0e-7,
        temperature_absolute_tolerance=1.0e-4,
        potential_relaxation=1.0,
        temperature_relaxation=0.5,
    )
    system = _bound_system(iteration=iteration, intervals=8)
    plan = _plan(system)
    inputs = pack_distributed_electrothermal_inputs(plan, value_dtype=np.float32)
    mesh = Mesh(np.asarray((jax.devices("cpu")[0],), dtype=object), ("partition",))
    runtime = build_distributed_electrothermal_runtime(
        plan,
        mesh,
        ScalarH1CGPolicy(
            2.0e-5,
            0.0,
            1000,
            backward_error_tolerance=5.0e-7,
        ),
        ElectrothermalAdjointPolicy(5.0e-4, 1.0e-6, 20, 60),
        linear_preconditioner_factory=build_packed_scalar_h1_jacobi_preconditioner_factory(
            plan.layout,
            mesh,
            ScalarH1JacobiPolicy(),
        ),
    )
    current = jnp.asarray(plan.current_initial, dtype=jnp.float32)
    thermal = jnp.asarray(plan.thermal_initial, dtype=jnp.float32)
    feedback = jnp.asarray(plan.feedback_initial, dtype=jnp.float32)
    forward = jax.jit(runtime.solve)(inputs, current, thermal, feedback)
    dense = system.solve(
        system.initial_current_values,
        system.initial_thermal_values,
        system.initial_feedback_values,
    )
    potential, temperature = reconstruct_distributed_electrothermal_state(
        plan,
        forward.state,
        current,
        thermal,
    )
    assert bool(forward.converged)
    assert float(forward.current_linear_backward_error) < 5.0e-7
    assert float(forward.heat_linear_backward_error) < 5.0e-7
    assert float(forward.current_residual_error) < 1.0e-4
    assert float(forward.heat_residual_error) < 1.0e-4
    np.testing.assert_allclose(potential, dense.potential, rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(temperature, dense.temperature, rtol=2.0e-6, atol=2.0e-4)

    free_count = plan.layout.topology.free_dof_count
    owner_weights = jnp.where(
        inputs.owner_mask,
        jnp.asarray(1.0 / free_count, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    cotangent = PackedElectrothermalVector(jnp.zeros_like(owner_weights), owner_weights)
    explicit = jax.jit(runtime.vjp)(inputs, current, thermal, feedback, cotangent)
    full_weights = (
        jnp.zeros((plan.layout.topology.node_count,), dtype=jnp.float64)
        .at[jnp.asarray(plan.layout.topology.free_nodes)]
        .set(1.0 / free_count)
    )
    dense_vjp = system.vjp(
        system.initial_current_values,
        system.initial_thermal_values,
        system.initial_feedback_values,
        full_weights,
    )
    assert bool(explicit.adjoint_converged)
    assert float(explicit.adjoint_backward_error) < 5.0e-4
    np.testing.assert_allclose(
        explicit.current_parameter_gradient,
        dense_vjp.current_parameter_gradient,
        rtol=2.0e-3,
        atol=2.0e-5,
    )
    np.testing.assert_allclose(
        explicit.thermal_parameter_gradient,
        dense_vjp.thermal_parameter_gradient,
        rtol=2.0e-3,
        atol=2.0e-5,
    )
    np.testing.assert_allclose(
        explicit.feedback_parameter_gradient,
        dense_vjp.feedback_parameter_gradient,
        rtol=2.0e-3,
        atol=2.0e-5,
    )


def test_host_plan_and_packing_preserve_explicit_identity_and_precision() -> None:
    system = _bound_system()
    first = _plan(system)
    second = _plan(system)
    assert first.schema_version == DISTRIBUTED_ELECTROTHERMAL_SCHEMA
    assert first.digest() == second.digest()
    assert first.current_parameter_names == ("applied_voltage", "heater_conductivity")
    assert first.thermal_parameter_names == ("thermal_conductivity",)
    assert first.feedback_parameter_names == ("heater_temperature_coefficient",)

    host32 = pack_distributed_electrothermal_inputs_host(first, value_dtype=np.float32)
    assert host32.unit_stiffness.dtype == np.float32
    assert host32.cell_local_dofs.dtype == np.int32
    assert host32.owner_mask.dtype == np.bool_
    assert host32.unit_stiffness.shape == (1, 8, 3, 3)
    assert not host32.unit_stiffness.flags.writeable
    packed64 = pack_distributed_electrothermal_inputs(first, value_dtype=np.float64)
    assert packed64.unit_stiffness.dtype == jnp.float64

    with pytest.raises(ContractError, match="prepared plan"):
        pack_distributed_electrothermal_inputs_host(object(), value_dtype=np.float64)  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="float32 or float64"):
        pack_distributed_electrothermal_inputs_host(first, value_dtype=np.int32)
    with pytest.raises(ContractError, match="schema"):
        replace(first, schema_version="femx.test/wrong")
    with pytest.raises(ContractError, match="scalar H1 layout"):
        replace(first, layout=object())  # type: ignore[arg-type]


def test_lowering_rejects_nonreference_and_space_identity_drift() -> None:
    system = _bound_system()
    cells = np.asarray(system.current._engine.payload.cells)
    owners = np.zeros((cells.shape[0],), dtype=np.int64)
    with pytest.raises(ContractError, match="bound dense reference"):
        prepare_distributed_electrothermal_plan(object(), owners, partition_count=1)  # type: ignore[arg-type]

    thermal_engine = system.thermal._engine
    payload = thermal_engine.payload
    variants = (
        ("coordinates", payload.coordinates.at[0, 0].add(1.0e-9), "coordinates"),
        ("cells", payload.cells.at[0, 0].set(1), "cell order"),
        ("free_nodes", payload.free_nodes[1:], "free-node identity"),
        ("dirichlet_nodes", payload.dirichlet_nodes[::-1], "Dirichlet nodes"),
    )
    for field, value, message in variants:
        changed_payload = replace(payload, **{field: value})
        changed_thermal = replace(
            system.thermal, _engine=replace(thermal_engine, payload=changed_payload)
        )
        changed_system = replace(system, thermal=changed_thermal)
        with pytest.raises(ContractError, match=message):
            prepare_distributed_electrothermal_plan(changed_system, owners, partition_count=1)


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        ((True, 0.0, 2, 2), "must be real"),
        ((1.0e-8, float("inf"), 2, 2), "must be finite"),
        ((0.0, 0.0, 2, 2), "must be positive"),
        ((1.0e-8, 0.0, True, 2), "must be positive"),
        ((1.0e-8, 0.0, 2, 0), "must be positive"),
    ),
)
def test_adjoint_policy_rejects_ambiguous_values(
    arguments: tuple[object, object, object, object],
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        ElectrothermalAdjointPolicy(*arguments)  # type: ignore[arg-type]


def test_runtime_contracts_reject_topology_dtype_parameter_and_cotangent_drift() -> None:
    system = _bound_system()
    plan = _plan(system)
    inputs = pack_distributed_electrothermal_inputs(plan, value_dtype=np.float64)
    mesh = Mesh(np.asarray((jax.devices("cpu")[0],), dtype=object), ("partition",))
    cg = ScalarH1CGPolicy(1.0e-12, 1.0e-14, 200)
    adjoint = ElectrothermalAdjointPolicy(1.0e-9, 1.0e-12, 12, 30)
    with pytest.raises(ContractError, match="prepared plan"):
        build_distributed_electrothermal_runtime(object(), mesh, cg, adjoint)  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="scalar CG policy"):
        build_distributed_electrothermal_runtime(plan, mesh, object(), adjoint)  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="adjoint policy"):
        build_distributed_electrothermal_runtime(plan, mesh, cg, object())  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="preconditioner factory"):
        build_distributed_electrothermal_runtime(
            plan,
            mesh,
            cg,
            adjoint,
            linear_preconditioner_factory=object(),  # type: ignore[arg-type]
        )
    runtime = build_distributed_electrothermal_runtime(plan, mesh, cg, adjoint)
    current = jnp.asarray(plan.current_initial)
    thermal = jnp.asarray(plan.thermal_initial)
    feedback = jnp.asarray(plan.feedback_initial)

    with pytest.raises(ContractError, match="packed contract"):
        runtime.solve(object(), current, thermal, feedback)
    bad_inputs: tuple[PackedDistributedElectrothermalInputs, ...] = (
        inputs._replace(cell_local_dofs=inputs.cell_local_dofs[:, :, :2]),
        inputs._replace(unit_stiffness=inputs.unit_stiffness[:, :, :, :2]),
    )
    for changed in bad_inputs:
        with pytest.raises(ValueError, match="inputs disagree"):
            runtime.solve(changed, current, thermal, feedback)
    with pytest.raises(TypeError, match="real floating"):
        runtime.solve(
            inputs._replace(unit_stiffness=inputs.unit_stiffness.astype(jnp.int32)),
            current,
            thermal,
            feedback,
        )
    with pytest.raises(TypeError, match="share one dtype"):
        runtime.solve(
            inputs._replace(
                current_reference_base=inputs.current_reference_base.astype(jnp.float32)
            ),
            current,
            thermal,
            feedback,
        )
    with pytest.raises(TypeError, match="cell map"):
        runtime.solve(
            inputs._replace(cell_local_dofs=inputs.cell_local_dofs.astype(jnp.float64)),
            current,
            thermal,
            feedback,
        )
    with pytest.raises(TypeError, match="activity masks"):
        runtime.solve(
            inputs._replace(owner_mask=inputs.owner_mask.astype(jnp.int32)),
            current,
            thermal,
            feedback,
        )

    with pytest.raises(ContractError, match="current parameters must have shape"):
        runtime.solve(inputs, current[:1], thermal, feedback)
    with pytest.raises(ContractError, match="thermal parameters must match input dtype"):
        runtime.solve(inputs, current, thermal.astype(jnp.float32), feedback)
    invalid_current = current.at[0].set(2.0)
    invalid = jax.jit(runtime.solve)(inputs, invalid_current, thermal, feedback)
    assert not bool(invalid.converged)
    assert int(invalid.iterations) == 0

    shape_bad = PackedElectrothermalVector(
        jnp.zeros((1, 2), dtype=jnp.float64),
        jnp.zeros_like(inputs.owner_mask, dtype=jnp.float64),
    )
    with pytest.raises(ContractError, match="cotangent must match owner"):
        runtime.vjp(inputs, current, thermal, feedback, shape_bad)
    dtype_bad = PackedElectrothermalVector(
        jnp.zeros_like(inputs.owner_mask, dtype=jnp.float32),
        jnp.zeros_like(inputs.owner_mask, dtype=jnp.float32),
    )
    with pytest.raises(ContractError, match="cotangent must match input dtype"):
        runtime.vjp(inputs, current, thermal, feedback, dtype_bad)
