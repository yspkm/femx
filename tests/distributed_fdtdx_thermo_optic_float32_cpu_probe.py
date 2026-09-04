#!/usr/bin/env python3
"""Exercise the physical-gate coupled FDTDX graph on four forced CPU devices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import fdtdx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from scripts._distributed_electrothermal_case import (
    bind_jax_self_consistent_microheater,
    distributed_electrothermal_iteration_policy,
)
from scripts._distributed_fdtdx_thermo_optic_case import (
    DETECTOR_NAME,
    DEVICE_NAME,
    FDTDX_SOURCE_DIGEST,
    FDTDX_SOURCE_REVISION,
    GRID_SHAPE,
    GRID_SPACING_M,
    MESH_AXIS_NAME,
    PHASOR_OBJECTIVE_SCALE,
    RUNTIME_TARGET_COORDINATE_MAX_GRID_FRACTION_ERROR,
    RUNTIME_TARGET_COORDINATE_MAX_ULP_ERROR,
    CoupledRuntimeInputs,
    build_scene,
    coupled_mesh_from_material_sharding,
    device_contract,
    thermo_optic_law,
    verify_locked_fdtdx,
)
from scripts._tpu_distributed_fdtdx_thermo_optic_plan import (
    read_distributed_fdtdx_thermo_optic_artifact,
)

from femx.backends.jax.distributed_electrothermal import (
    ElectrothermalAdjointPolicy,
    PackedElectrothermalVector,
    build_distributed_electrothermal_runtime,
    pack_distributed_electrothermal_inputs,
    prepare_distributed_electrothermal_plan,
    reconstruct_distributed_electrothermal_state,
)
from femx.backends.jax.scalar_cg import (
    ScalarH1CGPolicy,
    ScalarH1JacobiPolicy,
    build_packed_scalar_h1_jacobi_preconditioner_factory,
)
from femx.interop.fdtdx import (
    build_distributed_thermo_optic_runtime,
    build_triangle_p1_sampling_plan,
    pack_distributed_thermo_optic_inputs,
    prepare_distributed_triangle_p1_sampling_plan,
    thermo_optic_parameter_state,
    with_fdtdx_device_parameter,
)


def _relative_difference(observed: jax.Array, expected: jax.Array) -> float:
    difference = float(jnp.linalg.norm(observed - expected))
    scale = float(jnp.linalg.norm(expected))
    return difference / scale if scale > 0.0 else (0.0 if difference == 0.0 else float("inf"))


def _finite_float(value: object) -> float | None:
    converted = float(value)
    return converted if np.isfinite(converted) else None


def _array_report(value: object) -> dict[str, object]:
    array = np.asarray(value)
    finite = bool(np.all(np.isfinite(array)))
    norm_dtype = np.complex128 if np.iscomplexobj(array) else np.float64
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "all_finite": finite,
        "norm": float(np.linalg.norm(array.astype(norm_dtype))) if finite else None,
        "values": array.tolist() if finite else None,
    }


def _hlo_report(text: str) -> dict[str, object]:
    lowered = text.lower()
    return {
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "all_to_all_count": lowered.count("stablehlo.all_to_all"),
        "collective_permute_count": lowered.count("stablehlo.collective_permute"),
        "all_reduce_count": lowered.count("stablehlo.all_reduce"),
        "contains_all_gather": "all_gather" in lowered,
        "contains_float64": "f64" in lowered,
    }


def _ordered_float32_bits(values: np.ndarray) -> np.ndarray:
    """Map finite float32 values to monotonically ordered unsigned integers."""

    array = np.asarray(values)
    if array.dtype != np.float32 or np.any(~np.isfinite(array)):
        raise RuntimeError("runtime target coordinates must be finite float32 arrays")
    bits = array.view(np.uint32).astype(np.uint64)
    sign = np.uint64(1 << 31)
    maximum = np.uint64((1 << 32) - 1)
    return np.where((bits & sign) != 0, maximum - bits, bits + sign)


def _float32_ulp_distance(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Return ULP distance from runtime values to rounded controller values."""

    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected, dtype=np.float64)
    if actual_array.shape != expected_array.shape:
        raise RuntimeError("runtime and controller target-coordinate shapes differ")
    expected_runtime = np.asarray(expected_array, dtype=np.float32)
    actual_ordered = _ordered_float32_bits(actual_array)
    expected_ordered = _ordered_float32_bits(expected_runtime)
    return np.maximum(actual_ordered, expected_ordered) - np.minimum(
        actual_ordered,
        expected_ordered,
    )


def _system():
    return bind_jax_self_consistent_microheater(
        intervals=4,
        iteration=distributed_electrothermal_iteration_policy(),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    devices = jax.devices("cpu")
    if jax.default_backend() != "cpu" or len(devices) != 4:
        raise RuntimeError("float32 coupled FDTDX probe requires four CPU devices")
    if bool(getattr(jax.config, "jax_enable_x64", False)):
        raise RuntimeError("float32 coupled FDTDX probe requires JAX x64 disabled")
    if str(getattr(jax.config, "jax_default_matmul_precision", None)) != "highest":
        raise RuntimeError("float32 coupled FDTDX probe requires highest matmul precision")
    module_hashes = verify_locked_fdtdx(fdtdx)

    scene = build_scene(fdtdx, jax, jnp, backend="cpu")
    mesh = coupled_mesh_from_material_sharding(
        scene.material_array_shardings.inv_permittivities,
        Mesh,
        axis_name=MESH_AXIS_NAME,
        global_device_count=len(devices),
    )
    input_identity: dict[str, object] | None = None
    if arguments.input is None:
        # Production workers use the artifact branch below.  The fallback keeps this probe
        # independently runnable while confining reference construction to a scoped x64 context.
        with jax.enable_x64():
            system = _system()
            payload = system.current._engine.payload
            coordinates = np.asarray(payload.coordinates, dtype=np.float64)
            cells = np.asarray(payload.cells, dtype=np.int64)
            cell_centers_x = np.mean(coordinates[cells, 0], axis=1)
            cell_owners = np.minimum(
                (len(devices) * cell_centers_x / float(np.max(coordinates[:, 0]))).astype(np.int64),
                len(devices) - 1,
            )
            electrothermal_plan = prepare_distributed_electrothermal_plan(
                system,
                cell_owners,
                partition_count=len(devices),
            )
        sampling = None
        transfer_plan = None
        law = None
        contract = None
    else:
        loaded = read_distributed_fdtdx_thermo_optic_artifact(arguments.input)
        electrothermal_plan = loaded.electrothermal.plan
        sampling = loaded.sampling
        transfer_plan = loaded.transfer
        law = loaded.law
        contract = loaded.contract
        coordinates = np.asarray(sampling.source_coordinates, dtype=np.float64)
        cells = np.asarray(sampling.source_cells, dtype=np.int64)
        if electrothermal_plan.layout.partition_count != len(devices):
            raise RuntimeError("controller artifact partition count must equal CPU device count")
        input_identity = {
            "source_commit": loaded.manifest["source_commit"],
            "arrays_sha256": loaded.arrays_sha256,
            "electrothermal_arrays_sha256": loaded.electrothermal.arrays_sha256,
            "sampling_operator_sha256": sampling.operator_sha256,
            "transfer_operator_sha256": transfer_plan.operator_sha256,
        }
    if bool(getattr(jax.config, "jax_enable_x64", False)):
        raise RuntimeError("scoped controller-plan construction leaked x64 into the runtime")
    current = jnp.asarray(electrothermal_plan.current_initial, dtype=jnp.float32)
    thermal = jnp.asarray(electrothermal_plan.thermal_initial, dtype=jnp.float32)
    feedback = jnp.asarray(electrothermal_plan.feedback_initial, dtype=jnp.float32)
    electrothermal_inputs = pack_distributed_electrothermal_inputs(
        electrothermal_plan,
        value_dtype=np.float32,
    )
    preconditioner = build_packed_scalar_h1_jacobi_preconditioner_factory(
        electrothermal_plan.layout,
        mesh,
        ScalarH1JacobiPolicy(),
        axis_name=MESH_AXIS_NAME,
    )
    electrothermal_runtime = build_distributed_electrothermal_runtime(
        electrothermal_plan,
        mesh,
        ScalarH1CGPolicy(
            2.0e-5,
            0.0,
            1000,
            backward_error_tolerance=5.0e-7,
        ),
        ElectrothermalAdjointPolicy(5.0e-4, 0.0, 20, 60),
        axis_name=MESH_AXIS_NAME,
        linear_preconditioner_factory=preconditioner,
    )

    runtime_target_coordinates = tuple(np.asarray(axis) for axis in scene.target_coordinates)
    if sampling is None:
        sampling = build_triangle_p1_sampling_plan(
            coordinates,
            cells,
            runtime_target_coordinates,
            plane_axes=(0, 2),
        )
        law = thermo_optic_law()
        contract = device_contract(sampling, parameter_dtype="float32")
        transfer_plan = prepare_distributed_triangle_p1_sampling_plan(
            sampling,
            electrothermal_plan.layout.transport.cell_ids,
            source_layout_sha256=electrothermal_plan.layout.digest(),
            mesh_axis_name=MESH_AXIS_NAME,
        )
    assert law is not None and contract is not None and transfer_plan is not None
    coordinate_errors: list[float] = []
    coordinate_grid_fraction_errors: list[float] = []
    coordinate_max_ulp_errors: list[int] = []
    coordinate_rounding_exact: list[bool] = []
    coordinate_admitted: list[bool] = []
    for actual, expected in zip(
        runtime_target_coordinates,
        sampling.target_coordinates,
        strict=True,
    ):
        expected_runtime = np.asarray(expected, dtype=np.float32)
        coordinate_rounding_exact.append(bool(np.array_equal(actual, expected_runtime)))
        maximum_error = float(
            np.max(np.abs(actual.astype(np.float64) - np.asarray(expected, dtype=np.float64)))
        )
        maximum_grid_fraction = maximum_error / GRID_SPACING_M
        maximum_ulp = int(np.max(_float32_ulp_distance(actual, expected)))
        coordinate_errors.append(maximum_error)
        coordinate_grid_fraction_errors.append(maximum_grid_fraction)
        coordinate_max_ulp_errors.append(maximum_ulp)
        coordinate_admitted.append(
            maximum_ulp <= RUNTIME_TARGET_COORDINATE_MAX_ULP_ERROR
            and maximum_grid_fraction <= RUNTIME_TARGET_COORDINATE_MAX_GRID_FRACTION_ERROR
        )
    if not all(coordinate_admitted):
        raise RuntimeError("runtime FDTDX cell centers exceed the artifact coordinate tolerance")
    transfer_inputs = pack_distributed_thermo_optic_inputs(
        transfer_plan,
        value_dtype=np.float32,
    )
    transfer_runtime = build_distributed_thermo_optic_runtime(
        transfer_plan,
        mesh,
        law,
        contract,
    )
    runtime_inputs = CoupledRuntimeInputs(
        electrothermal=electrothermal_inputs,
        thermo_optic=transfer_inputs,
        fdtdx_arrays=scene.arrays,
        fdtdx_objects=scene.objects,
        fdtdx_parameters=scene.parameters,
        fdtdx_config=scene.config,
        fdtdx_key=scene.key,
    )
    material_array_shardings = scene.material_array_shardings
    device_contract_value = contract

    def downstream_phasor(
        inputs: CoupledRuntimeInputs,
        cell_temperature: jax.Array,
    ) -> jax.Array:
        thermo_optic = transfer_runtime.state(inputs.thermo_optic, cell_temperature)
        updated_parameters = with_fdtdx_device_parameter(
            inputs.fdtdx_parameters,
            thermo_optic,
            device_contract_value,
        )
        updated_arrays, updated_objects, _application = fdtdx.apply_params(
            arrays=inputs.fdtdx_arrays,
            objects=inputs.fdtdx_objects,
            params=updated_parameters,
            key=inputs.fdtdx_key,
            material_array_shardings=material_array_shardings,
        )
        _completed_step, final_arrays = fdtdx.run_fdtd(
            arrays=updated_arrays,
            objects=updated_objects,
            config=inputs.fdtdx_config,
            key=inputs.fdtdx_key,
            show_progress=False,
        )
        phasor = final_arrays.detector_states[DETECTOR_NAME]["phasor"]
        return jnp.sum(phasor)

    def reference_phasor_function(
        inputs: CoupledRuntimeInputs,
        current_values: jax.Array,
        thermal_values: jax.Array,
        feedback_values: jax.Array,
    ) -> jax.Array:
        state = electrothermal_runtime.state(
            inputs.electrothermal,
            current_values,
            thermal_values,
            feedback_values,
        )
        cell_temperature = electrothermal_runtime.cell_temperature(
            inputs.electrothermal,
            state,
            thermal_values,
        )
        return downstream_phasor(inputs, cell_temperature)

    reference = jax.jit(reference_phasor_function)
    reference_phasor = reference(runtime_inputs, current, thermal, feedback)
    jax.block_until_ready(reference_phasor)
    reference_phasor_value = complex(reference_phasor)
    if not np.isfinite(reference_phasor_value) or abs(reference_phasor_value) == 0.0:
        raise RuntimeError("nominal FDTDX phasor reference must be finite and nonzero")
    frozen_reference_phasor = jax.lax.stop_gradient(reference_phasor)

    def downstream_objective(
        inputs: CoupledRuntimeInputs,
        cell_temperature: jax.Array,
        reference_phasor_value: jax.Array,
    ) -> jax.Array:
        phasor = downstream_phasor(inputs, cell_temperature)
        reference_power = jnp.real(reference_phasor_value * jnp.conj(reference_phasor_value))
        normalized = phasor * jnp.conj(reference_phasor_value) / reference_power
        return PHASOR_OBJECTIVE_SCALE * jnp.imag(normalized)

    def native_objective(
        inputs: CoupledRuntimeInputs,
        current_values: jax.Array,
        thermal_values: jax.Array,
        feedback_values: jax.Array,
        reference_phasor_value: jax.Array,
    ) -> jax.Array:
        state = electrothermal_runtime.state(
            inputs.electrothermal,
            current_values,
            thermal_values,
            feedback_values,
        )
        cell_temperature = electrothermal_runtime.cell_temperature(
            inputs.electrothermal,
            state,
            thermal_values,
        )
        return downstream_objective(inputs, cell_temperature, reference_phasor_value)

    def explicit_objective_and_gradients(
        inputs: CoupledRuntimeInputs,
        current_values: jax.Array,
        thermal_values: jax.Array,
        feedback_values: jax.Array,
        reference_phasor_value: jax.Array,
    ):
        state = electrothermal_runtime.state(
            inputs.electrothermal,
            current_values,
            thermal_values,
            feedback_values,
        )
        cell_temperature = electrothermal_runtime.cell_temperature(
            inputs.electrothermal,
            state,
            thermal_values,
        )
        objective, downstream_pullback = jax.vjp(
            lambda value: downstream_objective(inputs, value, reference_phasor_value),
            cell_temperature,
        )
        (cell_cotangent,) = downstream_pullback(jnp.ones_like(objective))

        def cell_map(
            state_value: PackedElectrothermalVector,
            thermal_parameter_value: jax.Array,
        ) -> jax.Array:
            return electrothermal_runtime.cell_temperature(
                inputs.electrothermal,
                state_value,
                thermal_parameter_value,
            )

        _cell_value, cell_pullback = jax.vjp(cell_map, state, thermal_values)
        state_cotangent, direct_thermal_gradient = cell_pullback(cell_cotangent)
        explicit = electrothermal_runtime.vjp(
            inputs.electrothermal,
            current_values,
            thermal_values,
            feedback_values,
            state_cotangent,
        )
        return (
            objective,
            (
                explicit.current_parameter_gradient,
                explicit.thermal_parameter_gradient + direct_thermal_gradient,
                explicit.feedback_parameter_gradient,
            ),
            explicit.adjoint_backward_error,
            explicit.adjoint_converged,
            jnp.linalg.norm(cell_cotangent),
            jnp.linalg.norm(state_cotangent.potential),
            jnp.linalg.norm(state_cotangent.temperature),
        )

    objective_arguments = (
        runtime_inputs,
        current,
        thermal,
        feedback,
        frozen_reference_phasor,
    )
    native = jax.jit(jax.value_and_grad(native_objective, argnums=(1, 2, 3)))
    native_lowered = native.lower(*objective_arguments)
    native_hlo = str(native_lowered.compiler_ir("stablehlo"))
    objective, native_gradients = native(*objective_arguments)
    jax.block_until_ready((objective, native_gradients))

    explicit = jax.jit(explicit_objective_and_gradients)
    explicit_lowered = explicit.lower(*objective_arguments)
    explicit_hlo = str(explicit_lowered.compiler_ir("stablehlo"))
    (
        explicit_objective,
        explicit_gradients,
        adjoint_error,
        adjoint_converged,
        cell_cotangent_norm,
        potential_cotangent_norm,
        temperature_cotangent_norm,
    ) = explicit(*objective_arguments)
    jax.block_until_ready(
        (
            explicit_objective,
            explicit_gradients,
            adjoint_error,
            adjoint_converged,
            cell_cotangent_norm,
            potential_cotangent_norm,
            temperature_cotangent_norm,
        )
    )

    forward = jax.jit(native_objective)
    finite_difference_errors: dict[str, float | None] = {}
    finite_difference_gradients: dict[str, float | None] = {}
    for step in (1.0e-1, 5.0e-2, 2.0e-2, 1.0e-2):
        plus = current.at[0].add(step)
        minus = current.at[0].add(-step)
        finite_difference = (
            forward(runtime_inputs, plus, thermal, feedback, frozen_reference_phasor)
            - forward(runtime_inputs, minus, thermal, feedback, frozen_reference_phasor)
        ) / (2.0 * step)
        jax.block_until_ready(finite_difference)
        finite_difference_errors[f"{step:.1e}"] = _finite_float(
            _relative_difference(native_gradients[0][0], finite_difference)
        )
        finite_difference_gradients[f"{step:.1e}"] = _finite_float(finite_difference)

    state = electrothermal_runtime.state(
        runtime_inputs.electrothermal,
        current,
        thermal,
        feedback,
    )
    cell_temperature = electrothermal_runtime.cell_temperature(
        runtime_inputs.electrothermal,
        state,
        thermal,
    )
    thermo_optic = transfer_runtime.state(runtime_inputs.thermo_optic, cell_temperature)
    updated_parameters = with_fdtdx_device_parameter(
        runtime_inputs.fdtdx_parameters,
        thermo_optic,
        device_contract_value,
    )
    updated_arrays, _objects, _application = fdtdx.apply_params(
        arrays=runtime_inputs.fdtdx_arrays,
        objects=runtime_inputs.fdtdx_objects,
        params=updated_parameters,
        key=runtime_inputs.fdtdx_key,
        material_array_shardings=material_array_shardings,
    )
    device_inverse_permittivity = updated_arrays.inv_permittivities[
        (slice(None), *scene.placed_device.grid_slice)
    ]
    _canonical_potential, canonical_temperature = reconstruct_distributed_electrothermal_state(
        electrothermal_plan,
        state,
        current,
        thermal,
    )
    canonical_thermo_optic = thermo_optic_parameter_state(
        sampling,
        canonical_temperature,
        law,
        contract,
    )
    jax.block_until_ready((thermo_optic, device_inverse_permittivity, canonical_thermo_optic))

    result = {
        "schema_version": "femx.fdtdx.distributed_thermo_optic.float32_cpu_objective/v1",
        "backend": jax.default_backend(),
        "forced_cpu_device_count": len(devices),
        "process_count": jax.process_count(),
        "x64_enabled": bool(getattr(jax.config, "jax_enable_x64", False)),
        "matmul_precision": str(getattr(jax.config, "jax_default_matmul_precision", None)),
        "fdtdx_source_revision": FDTDX_SOURCE_REVISION,
        "fdtdx_source_digest": FDTDX_SOURCE_DIGEST,
        "fdtdx_module_sha256": module_hashes,
        "grid_shape": list(GRID_SHAPE),
        "device_shape": list(scene.parameters[DEVICE_NAME].shape),
        "time_steps": int(scene.config.time_steps_total),
        "electrothermal_plan_sha256": electrothermal_plan.digest(),
        "transfer_operator_sha256": transfer_plan.operator_sha256,
        "input_artifact": input_identity,
        "runtime_target_coordinate_max_errors_m": coordinate_errors,
        "runtime_target_coordinate_max_grid_fraction_errors": (coordinate_grid_fraction_errors),
        "runtime_target_coordinate_max_ulp_errors": coordinate_max_ulp_errors,
        "runtime_target_coordinate_float32_rounding_exact": coordinate_rounding_exact,
        "runtime_target_coordinate_admitted": coordinate_admitted,
        "runtime_target_coordinate_tolerance": {
            "max_ulp_error": RUNTIME_TARGET_COORDINATE_MAX_ULP_ERROR,
            "max_grid_spacing_fraction_error": (RUNTIME_TARGET_COORDINATE_MAX_GRID_FRACTION_ERROR),
        },
        "material_sharding": str(updated_arrays.inv_permittivities.sharding.spec),
        "thermo_optic_parameter_sharding": str(thermo_optic.parameter.sharding.spec),
        "all_valid": bool(thermo_optic.all_valid),
        "material_relative_difference": _finite_float(
            _relative_difference(
                1.0 / device_inverse_permittivity[0],
                thermo_optic.relative_permittivity,
            )
        ),
        "parameter_canonical_relative_difference": _finite_float(
            _relative_difference(
                thermo_optic.parameter,
                canonical_thermo_optic.parameter,
            )
        ),
        "objective": _finite_float(objective),
        "phasor_objective_scale": PHASOR_OBJECTIVE_SCALE,
        "reference_phasor": {
            "real": reference_phasor_value.real,
            "imag": reference_phasor_value.imag,
            "magnitude": abs(reference_phasor_value),
        },
        "objective_explicit_relative_difference": _finite_float(
            _relative_difference(
                objective,
                explicit_objective,
            )
        ),
        "native_gradients_finite": [
            bool(jnp.all(jnp.isfinite(value))) for value in native_gradients
        ],
        "native_gradient_reports": [_array_report(value) for value in native_gradients],
        "explicit_gradients_finite": [
            bool(jnp.all(jnp.isfinite(value))) for value in explicit_gradients
        ],
        "native_explicit_gradient_relative_differences": [
            _finite_float(_relative_difference(observed, expected))
            for observed, expected in zip(native_gradients, explicit_gradients, strict=True)
        ],
        "applied_voltage_finite_difference_relative_errors": finite_difference_errors,
        "applied_voltage_finite_difference_gradients": finite_difference_gradients,
        "adjoint_backward_error": _finite_float(adjoint_error),
        "adjoint_converged": bool(adjoint_converged),
        "cell_cotangent_norm": _finite_float(cell_cotangent_norm),
        "state_cotangent_norms": {
            "potential": _finite_float(potential_cotangent_norm),
            "temperature": _finite_float(temperature_cotangent_norm),
        },
        "stablehlo": {
            "native": _hlo_report(native_hlo),
            "explicit": _hlo_report(explicit_hlo),
        },
        "claim_scope": (
            "single-process four-forced-CPU float32/complex64 admission of the physical-gate "
            "distributed FEM to all-to-all thermo-optic to checkpointed FDTDX objective graph; "
            "not physical TPU, multi-host, convergence, S-parameters, 3D FEM, calibrated "
            "material, or device evidence"
        ),
    }
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
