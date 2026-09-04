"""Four-device CPU portability probe for the coupled electrothermal residual/VJP."""

from __future__ import annotations

import json

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from femx.backends.jax.distributed_electrothermal import (  # noqa: E402
    ElectrothermalAdjointPolicy,
    PackedElectrothermalVector,
    build_distributed_electrothermal_runtime,
    pack_distributed_electrothermal_inputs,
    prepare_distributed_electrothermal_plan,
    reconstruct_distributed_electrothermal_state,
)
from femx.backends.jax.scalar_cg import ScalarH1CGPolicy  # noqa: E402
from femx.backends.jax.scalar_collective import (  # noqa: E402
    pack_collective_scalar_h1_owned_vector,
)
from femx.backends.jax.self_consistent import (  # noqa: E402
    DifferentiableSelfConsistentElectrothermal,
)
from femx.backends.jax.steady_current import JaxSteadyCurrentBackend  # noqa: E402
from femx.backends.jax.steady_heat import JaxSteadyHeatBackend  # noqa: E402
from femx.runtime import prepare  # noqa: E402
from tests.electrothermal_support import (  # noqa: E402
    parameterized_self_consistent_microheater,
)


def _relative_difference(observed: jax.Array, expected: jax.Array) -> float:
    difference = float(jnp.linalg.norm(observed - expected))
    scale = float(jnp.linalg.norm(expected))
    return difference / scale if scale > 0.0 else (0.0 if difference == 0.0 else float("inf"))


def _cell_owners(coordinates: np.ndarray, cells: np.ndarray, count: int) -> np.ndarray:
    centroids = np.mean(coordinates[cells, 0], axis=1)
    width = float(np.max(coordinates[:, 0]))
    return np.minimum((count * centroids / width).astype(np.int64), count - 1)


def _system() -> DifferentiableSelfConsistentElectrothermal:
    feedback, current_parameters, thermal_parameters, feedback_parameters = (
        parameterized_self_consistent_microheater(intervals=4)
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


def main() -> int:
    devices = jax.devices("cpu")
    if jax.default_backend() != "cpu" or len(devices) != 4:
        raise RuntimeError("distributed electrothermal probe requires four forced CPU devices")
    system = _system()
    current = system.initial_current_values
    thermal = system.initial_thermal_values
    feedback = system.initial_feedback_values
    dense = system.solve(current, thermal, feedback)
    payload = system.current._engine.payload
    coordinates = np.asarray(payload.coordinates)
    cells = np.asarray(payload.cells)
    free_nodes = np.asarray(payload.free_nodes)
    full_weights = jnp.linspace(0.75, 1.25, coordinates.shape[0], dtype=jnp.float64)
    full_weights = full_weights.at[jnp.asarray(payload.dirichlet_nodes)].set(0.0)
    full_weights /= jnp.sum(full_weights)
    dense_vjp = system.vjp(current, thermal, feedback, full_weights)
    thermal_reference = float(
        np.asarray(system.thermal._engine.resolved_coefficients(thermal)[-1][0])
    )
    dense_objective = float(jnp.vdot(full_weights, dense.temperature - thermal_reference))
    reports: dict[str, object] = {}
    four = None

    for partition_count in (1, 2, 4):
        plan = prepare_distributed_electrothermal_plan(
            system,
            _cell_owners(coordinates, cells, partition_count),
            partition_count=partition_count,
        )
        inputs = pack_distributed_electrothermal_inputs(plan, value_dtype=np.float64)
        mesh = Mesh(np.asarray(devices[:partition_count], dtype=object), ("partition",))
        runtime = build_distributed_electrothermal_runtime(
            plan,
            mesh,
            ScalarH1CGPolicy(1.0e-12, 1.0e-14, 400),
            ElectrothermalAdjointPolicy(2.0e-10, 1.0e-12, 20, 40),
        )
        solve = jax.jit(runtime.solve)
        forward = solve(inputs, current, thermal, feedback)
        forward.state.temperature.block_until_ready()
        potential, temperature = reconstruct_distributed_electrothermal_state(
            plan,
            forward.state,
            current,
            thermal,
        )
        packed_weights = pack_collective_scalar_h1_owned_vector(
            plan.layout,
            full_weights[jnp.asarray(free_nodes)],
        )
        cotangent = PackedElectrothermalVector(
            jnp.zeros_like(packed_weights),
            packed_weights,
        )
        explicit = jax.jit(runtime.vjp)(inputs, current, thermal, feedback, cotangent)
        explicit.current_parameter_gradient.block_until_ready()
        report = {
            "plan_sha256": plan.digest(),
            "halo_link_count": len(plan.layout.transport.halo_links),
            "iterations": int(forward.iterations),
            "current_linear_iterations": int(forward.current_linear_iterations),
            "heat_linear_iterations": int(forward.heat_linear_iterations),
            "converged": bool(forward.converged),
            "adjoint_converged": bool(explicit.adjoint_converged),
            "potential_relative_difference": _relative_difference(potential, dense.potential),
            "temperature_relative_difference": _relative_difference(
                temperature,
                dense.temperature,
            ),
            "current_gradient_relative_difference": _relative_difference(
                explicit.current_parameter_gradient,
                dense_vjp.current_parameter_gradient,
            ),
            "thermal_gradient_relative_difference": _relative_difference(
                explicit.thermal_parameter_gradient,
                dense_vjp.thermal_parameter_gradient,
            ),
            "feedback_gradient_relative_difference": _relative_difference(
                explicit.feedback_parameter_gradient,
                dense_vjp.feedback_parameter_gradient,
            ),
            "current_residual_error": float(forward.current_residual_error),
            "heat_residual_error": float(forward.heat_residual_error),
            "transfer_relative_error": float(forward.transfer_relative_error),
            "adjoint_backward_error": float(explicit.adjoint_backward_error),
        }
        reports[str(partition_count)] = report
        if partition_count == 4:
            four = (plan, inputs, runtime, solve, cotangent, forward, explicit)

    if four is None:
        raise RuntimeError("four-device coupled witness was not constructed")
    plan, inputs, runtime, solve, cotangent, forward, explicit = four
    owner_weights = cotangent.temperature

    def differentiable_objective(
        current_values: jax.Array,
        thermal_values: jax.Array,
        feedback_values: jax.Array,
    ) -> jax.Array:
        state = runtime.state(inputs, current_values, thermal_values, feedback_values)
        reference = inputs.thermal_reference_base + jnp.vdot(
            inputs.thermal_reference_weights,
            thermal_values,
        )
        return jnp.sum((state.temperature - reference) * owner_weights)

    native_function = jax.jit(jax.value_and_grad(differentiable_objective, argnums=(0, 1, 2)))
    objective_value, native_gradients = native_function(current, thermal, feedback)
    jax.tree.map(lambda value: value.block_until_ready(), native_gradients)

    @jax.jit
    def forward_objective(
        current_values: jax.Array,
        thermal_values: jax.Array,
        feedback_values: jax.Array,
    ) -> jax.Array:
        result = runtime.solve(inputs, current_values, thermal_values, feedback_values)
        reference = inputs.thermal_reference_base + jnp.vdot(
            inputs.thermal_reference_weights,
            thermal_values,
        )
        return jnp.sum((result.state.temperature - reference) * owner_weights)

    parameter_sets = (current, thermal, feedback)
    steps = ((2.0e-5, 2.0e-1), (1.2e-2,), (3.0e-6,))
    finite_gradients: list[jax.Array] = []
    for argument_index, (values, argument_steps) in enumerate(
        zip(parameter_sets, steps, strict=True)
    ):
        derivatives: list[jax.Array] = []
        for value_index, step in enumerate(argument_steps):
            plus_values = values.at[value_index].add(step)
            minus_values = values.at[value_index].add(-step)
            plus_arguments = list(parameter_sets)
            minus_arguments = list(parameter_sets)
            plus_arguments[argument_index] = plus_values
            minus_arguments[argument_index] = minus_values
            derivatives.append(
                (forward_objective(*plus_arguments) - forward_objective(*minus_arguments))
                / (2.0 * step)
            )
        finite_gradients.append(jnp.stack(derivatives))

    explicit_gradients = (
        explicit.current_parameter_gradient,
        explicit.thermal_parameter_gradient,
        explicit.feedback_parameter_gradient,
    )
    native_errors = [
        _relative_difference(observed, expected)
        for observed, expected in zip(native_gradients, explicit_gradients, strict=True)
    ]
    finite_errors = [
        _relative_difference(observed, expected)
        for observed, expected in zip(finite_gradients, explicit_gradients, strict=True)
    ]
    forward_hlo = str(solve.lower(inputs, current, thermal, feedback).compiler_ir("stablehlo"))
    explicit_hlo = str(
        jax.jit(runtime.vjp)
        .lower(inputs, current, thermal, feedback, cotangent)
        .compiler_ir("stablehlo")
    )
    native_hlo = str(native_function.lower(current, thermal, feedback).compiler_ir("stablehlo"))

    def hlo_report(text: str) -> dict[str, object]:
        lowered = text.lower()
        return {
            "collective_permute_count": lowered.count("stablehlo.collective_permute"),
            "all_reduce_count": lowered.count("stablehlo.all_reduce"),
            "contains_all_gather": "all_gather" in lowered,
        }

    result = {
        "schema_version": "femx.jax.distributed_electrothermal.cpu_portability/v1",
        "backend": jax.default_backend(),
        "forced_cpu_device_count": len(devices),
        "process_count": jax.process_count(),
        "partition_reports": reports,
        "objective": float(objective_value),
        "dense_objective": dense_objective,
        "objective_relative_difference": abs(float(objective_value) - dense_objective)
        / abs(dense_objective),
        "native_gradient_relative_errors": native_errors,
        "finite_difference_relative_errors": finite_errors,
        "stablehlo": {
            "forward": hlo_report(forward_hlo),
            "explicit_vjp": hlo_report(explicit_hlo),
            "native_reverse": hlo_report(native_hlo),
        },
        "claim_scope": (
            "single-process four-forced-CPU portability for one same-mesh coupled residual; "
            "not physical accelerator, multi-host, scaling, foundry, or FDTDX evidence"
        ),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
