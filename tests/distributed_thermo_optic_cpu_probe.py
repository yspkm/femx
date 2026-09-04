"""Four-device CPU probe for the distributed electrothermal-to-FDTDX material boundary."""

from __future__ import annotations

import json

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from femx.backends.jax.distributed_electrothermal import (  # noqa: E402
    ElectrothermalAdjointPolicy,
    build_distributed_electrothermal_runtime,
    pack_distributed_electrothermal_inputs,
    prepare_distributed_electrothermal_plan,
)
from femx.backends.jax.scalar_cg import ScalarH1CGPolicy  # noqa: E402
from femx.backends.jax.self_consistent import (  # noqa: E402
    DifferentiableSelfConsistentElectrothermal,
)
from femx.backends.jax.steady_current import JaxSteadyCurrentBackend  # noqa: E402
from femx.backends.jax.steady_heat import JaxSteadyHeatBackend  # noqa: E402
from femx.interop.fdtdx import (  # noqa: E402
    FDTDXDeviceParameterContract,
    FDTDXFingerprint,
    ThermoOpticLaw,
    build_distributed_thermo_optic_runtime,
    build_triangle_p1_sampling_plan,
    pack_distributed_thermo_optic_inputs,
    prepare_distributed_triangle_p1_sampling_plan,
    thermo_optic_parameter_state,
)
from femx.runtime import prepare  # noqa: E402
from tests.electrothermal_support import (  # noqa: E402
    parameterized_self_consistent_microheater,
)


def _relative_difference(observed: jax.Array, expected: jax.Array) -> float:
    difference = float(jnp.linalg.norm(observed - expected))
    scale = float(jnp.linalg.norm(expected))
    return difference / scale if scale > 0.0 else (0.0 if difference == 0.0 else float("inf"))


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


def _thermo_optic_contract(system: DifferentiableSelfConsistentElectrothermal):
    payload = system.current._engine.payload
    sampling = build_triangle_p1_sampling_plan(
        np.asarray(payload.coordinates),
        np.asarray(payload.cells),
        (
            np.linspace(0.125e-6, 1.875e-6, 8, dtype=np.float64),
            np.asarray((-0.1e-6, 0.1e-6), dtype=np.float64),
            np.linspace(0.0625e-6, 0.4375e-6, 4, dtype=np.float64),
        ),
        plane_axes=(0, 2),
    )
    law = ThermoOpticLaw(
        material_region="silicon",
        reference_temperature_k=300.0,
        reference_refractive_index=3.48,
        thermo_optic_coefficient_per_k=1.86e-4,
        vacuum_wavelength_m=1.55e-6,
    )
    contract = FDTDXDeviceParameterContract(
        device_name="heated-silicon",
        target_shape=sampling.target_shape,
        plane_axes=sampling.plane_axes,
        lower_relative_permittivity=10.0,
        upper_relative_permittivity=16.0,
        parameter_dtype="float64",
        thermo_optic_law_sha256=law.sha256,
        target_coordinate_sha256=sampling.target_coordinate_sha256,
        transfer_operator_sha256=sampling.operator_sha256,
        fdtdx=FDTDXFingerprint(
            "0.6.2",
            "0c05c4784b2be83b42d9b46ab089265981ba157f",
            "29bed9483c4c2b57fd2f495fdb47534edf6b244206679e34b2de41ec39aaa9fa",
        ),
    )
    return sampling, law, contract


def _hlo_report(text: str) -> dict[str, object]:
    lowered = text.lower()
    return {
        "all_to_all_count": lowered.count("stablehlo.all_to_all"),
        "collective_permute_count": lowered.count("stablehlo.collective_permute"),
        "all_reduce_count": lowered.count("stablehlo.all_reduce"),
        "contains_all_gather": "all_gather" in lowered,
    }


def main() -> int:
    devices = jax.devices("cpu")
    if jax.default_backend() != "cpu" or len(devices) != 4:
        raise RuntimeError("distributed thermo-optic probe requires four forced CPU devices")
    mesh = Mesh(np.asarray(devices, dtype=object), ("shard",))
    system = _system()
    current = system.initial_current_values
    thermal = system.initial_thermal_values
    feedback = system.initial_feedback_values
    payload = system.current._engine.payload
    coordinates = np.asarray(payload.coordinates)
    cells = np.asarray(payload.cells)
    cell_centers_x = np.mean(coordinates[cells, 0], axis=1)
    cell_owners = np.minimum(
        (4 * cell_centers_x / float(np.max(coordinates[:, 0]))).astype(np.int64),
        3,
    )
    electrothermal_plan = prepare_distributed_electrothermal_plan(
        system,
        cell_owners,
        partition_count=4,
    )
    electrothermal_inputs = pack_distributed_electrothermal_inputs(
        electrothermal_plan,
        value_dtype=np.float64,
    )
    electrothermal_runtime = build_distributed_electrothermal_runtime(
        electrothermal_plan,
        mesh,
        ScalarH1CGPolicy(1.0e-12, 1.0e-14, 400),
        ElectrothermalAdjointPolicy(2.0e-10, 1.0e-12, 20, 40),
        axis_name="shard",
    )
    sampling, law, contract = _thermo_optic_contract(system)
    transfer_plan = prepare_distributed_triangle_p1_sampling_plan(
        sampling,
        electrothermal_plan.layout.transport.cell_ids,
        source_layout_sha256=electrothermal_plan.layout.digest(),
        mesh_axis_name="shard",
    )
    transfer_inputs = pack_distributed_thermo_optic_inputs(
        transfer_plan,
        value_dtype=np.float64,
    )
    transfer_runtime = build_distributed_thermo_optic_runtime(
        transfer_plan,
        mesh,
        law,
        contract,
    )

    def distributed_state(
        current_values: jax.Array,
        thermal_values: jax.Array,
        feedback_values: jax.Array,
    ):
        state = electrothermal_runtime.state(
            electrothermal_inputs,
            current_values,
            thermal_values,
            feedback_values,
        )
        cell_temperature = electrothermal_runtime.cell_temperature(
            electrothermal_inputs,
            state,
            thermal_values,
        )
        return transfer_runtime.state(transfer_inputs, cell_temperature)

    def distributed_objective(
        current_values: jax.Array,
        thermal_values: jax.Array,
        feedback_values: jax.Array,
    ) -> jax.Array:
        parameter = distributed_state(current_values, thermal_values, feedback_values).parameter
        return jnp.mean(parameter**2)

    def dense_objective(
        current_values: jax.Array,
        thermal_values: jax.Array,
        feedback_values: jax.Array,
    ) -> jax.Array:
        temperature = system.temperature(current_values, thermal_values, feedback_values)
        parameter = thermo_optic_parameter_state(
            sampling,
            temperature,
            law,
            contract,
        ).parameter
        return jnp.mean(parameter**2)

    compiled_state = jax.jit(distributed_state)
    observed = compiled_state(current, thermal, feedback)
    jax.tree.map(lambda value: value.block_until_ready(), observed)
    dense_temperature = system.temperature(current, thermal, feedback)
    expected = thermo_optic_parameter_state(
        sampling,
        dense_temperature,
        law,
        contract,
    )
    compiled_value_and_grad = jax.jit(jax.value_and_grad(distributed_objective, argnums=(0, 1, 2)))
    objective, gradients = compiled_value_and_grad(current, thermal, feedback)
    jax.tree.map(lambda value: value.block_until_ready(), gradients)
    dense_value_and_grad = jax.jit(jax.value_and_grad(dense_objective, argnums=(0, 1, 2)))
    dense_objective_value, dense_gradients = dense_value_and_grad(current, thermal, feedback)
    jax.tree.map(lambda value: value.block_until_ready(), dense_gradients)

    @jax.jit
    def forward_objective(
        current_values: jax.Array,
        thermal_values: jax.Array,
        feedback_values: jax.Array,
    ) -> jax.Array:
        result = electrothermal_runtime.solve(
            electrothermal_inputs,
            current_values,
            thermal_values,
            feedback_values,
        )
        cell_temperature = electrothermal_runtime.cell_temperature(
            electrothermal_inputs,
            result.state,
            thermal_values,
        )
        parameter = transfer_runtime.state(transfer_inputs, cell_temperature).parameter
        return jnp.mean(parameter**2)

    parameter_sets = (current, thermal, feedback)
    steps = ((2.0e-5, 2.0e-1), (1.2e-2,), (3.0e-6,))
    finite_gradients: list[jax.Array] = []
    for argument_index, (values, argument_steps) in enumerate(
        zip(parameter_sets, steps, strict=True)
    ):
        derivatives: list[jax.Array] = []
        for value_index, step in enumerate(argument_steps):
            plus_arguments = list(parameter_sets)
            minus_arguments = list(parameter_sets)
            plus_arguments[argument_index] = values.at[value_index].add(step)
            minus_arguments[argument_index] = values.at[value_index].add(-step)
            derivatives.append(
                (forward_objective(*plus_arguments) - forward_objective(*minus_arguments))
                / (2.0 * step)
            )
        finite_gradients.append(jnp.stack(derivatives))

    forward_hlo = str(compiled_state.lower(current, thermal, feedback).compiler_ir("stablehlo"))
    reverse_hlo = str(
        compiled_value_and_grad.lower(current, thermal, feedback).compiler_ir("stablehlo")
    )
    result = {
        "schema_version": "femx.fdtdx.distributed_thermo_optic.cpu_portability/v1",
        "backend": jax.default_backend(),
        "forced_cpu_device_count": len(devices),
        "process_count": jax.process_count(),
        "electrothermal_plan_sha256": electrothermal_plan.digest(),
        "source_layout_sha256": electrothermal_plan.layout.digest(),
        "sampling_operator_sha256": sampling.operator_sha256,
        "distributed_operator_sha256": transfer_plan.operator_sha256,
        "source_cell_count": transfer_plan.source_cell_count,
        "target_shape": transfer_plan.target_shape,
        "transfer_capacity": transfer_plan.transfer_capacity,
        "temperature_relative_difference": _relative_difference(
            observed.sampled_temperature_k,
            expected.sampled_temperature_k,
        ),
        "permittivity_relative_difference": _relative_difference(
            observed.relative_permittivity,
            expected.relative_permittivity,
        ),
        "parameter_relative_difference": _relative_difference(
            observed.parameter,
            expected.parameter,
        ),
        "all_valid": bool(observed.all_valid),
        "parameter_sharding": str(observed.parameter.sharding.spec),
        "objective": float(objective),
        "dense_objective": float(dense_objective_value),
        "objective_relative_difference": _relative_difference(objective, dense_objective_value),
        "dense_gradient_relative_errors": [
            _relative_difference(actual, reference)
            for actual, reference in zip(gradients, dense_gradients, strict=True)
        ],
        "finite_difference_relative_errors": [
            _relative_difference(actual, reference)
            for actual, reference in zip(gradients, finite_gradients, strict=True)
        ],
        "stablehlo": {
            "forward": _hlo_report(forward_hlo),
            "native_reverse": _hlo_report(reverse_hlo),
        },
        "claim_scope": (
            "single-process four-forced-CPU portability for sharded electrothermal P1 sampling "
            "and thermo-optic material parameters; not FDTDX time integration, physical TPU, "
            "multi-host, scaling, 3D FEM, calibrated material, or device evidence"
        ),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
