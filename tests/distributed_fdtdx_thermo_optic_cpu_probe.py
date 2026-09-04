#!/usr/bin/env python3
"""Exercise the distributed electrothermal-to-FDTDX objective on four CPU devices."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version as distribution_version
from pathlib import Path

import fdtdx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from femx.backends.jax.distributed_electrothermal import (
    ElectrothermalAdjointPolicy,
    build_distributed_electrothermal_runtime,
    pack_distributed_electrothermal_inputs,
    prepare_distributed_electrothermal_plan,
)
from femx.backends.jax.scalar_cg import ScalarH1CGPolicy
from femx.interop.fdtdx import (
    FDTDXDeviceParameterContract,
    FDTDXFingerprint,
    ThermoOpticLaw,
    build_distributed_thermo_optic_runtime,
    build_triangle_p1_sampling_plan,
    pack_distributed_thermo_optic_inputs,
    prepare_distributed_triangle_p1_sampling_plan,
    thermo_optic_parameter_state,
    with_fdtdx_device_parameter,
)
from tests.distributed_thermo_optic_cpu_probe import _relative_difference, _system

FDTDX_SOURCE_REVISION = "0c05c4784b2be83b42d9b46ab089265981ba157f"
FDTDX_SOURCE_DIGEST = "29bed9483c4c2b57fd2f495fdb47534edf6b244206679e34b2de41ec39aaa9fa"
FDTDX_MODULE_SHA256 = {
    "__init__.py": "fcf000b7955c97e7fbe1ccd5901c1f5ba47a5bfd86f0fce3d2dc8be1bfe131cf",
    "core/jax/sharding.py": "a6e07ac439c1c1b48958380812406f090844a1b4924a3b3a9b0a7f49eca8a9c3",
    "fdtd/fdtd.py": "7c654097d43d5062afbef0cf8c479ba2a7db523b64683693fa4e24bc5070e4e0",
    "fdtd/initialization.py": "2b7d56d47789f38c73b96fe7a078521e1146a45e98753af6c5e536ea8f9225a1",
    "fdtd/wrapper.py": "97d562e0a33eeeccd6ce42d12cf8dc29f1e9fd071e561b5b009ac5da8ece7384",
}


def _hlo_report(text: str) -> dict[str, object]:
    lowered = text.lower()
    return {
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "all_to_all_count": lowered.count("stablehlo.all_to_all"),
        "collective_permute_count": lowered.count("stablehlo.collective_permute"),
        "all_reduce_count": lowered.count("stablehlo.all_reduce"),
        "contains_all_gather": "all_gather" in lowered,
    }


def _verify_fdtdx_modules() -> None:
    package_root = Path(fdtdx.__file__).resolve().parent
    for relative_path, expected_sha256 in FDTDX_MODULE_SHA256.items():
        source = package_root / relative_path
        observed_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        if observed_sha256 != expected_sha256:
            raise RuntimeError(f"locked FDTDX module hash differs for {relative_path}")


def main() -> int:
    devices = jax.devices("cpu")
    if jax.default_backend() != "cpu" or len(devices) != 4:
        raise RuntimeError("distributed FDTDX thermo-optic probe requires four CPU devices")
    if distribution_version("fdtdx") != "0.6.2":
        raise RuntimeError("distributed FDTDX thermo-optic probe requires FDTDX 0.6.2")
    if not callable(getattr(fdtdx, "capture_material_array_shardings", None)):
        raise RuntimeError("locked FDTDX does not expose material sharding capture")
    _verify_fdtdx_modules()

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

    wavelength = 1.55e-6
    grid_shape = (16, 4, 4)
    lower_epsilon = 10.0
    upper_epsilon = 16.0
    device_name = "heated-silicon"
    detector_name = "optical-phasor"
    volume = fdtdx.SimulationVolume(
        partial_grid_shape=grid_shape,
        material=fdtdx.Material(permittivity=2.085136),
    )
    boundaries, boundary_constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(
            thickness=2,
            override_types={
                face: "periodic" for face in ("min_x", "max_x", "min_y", "max_y", "min_z", "max_z")
            },
        ),
        volume,
    )
    waveguide = fdtdx.UniformMaterialObject(
        name="silicon-waveguide",
        partial_grid_shape=(16, 2, 2),
        material=fdtdx.Material(permittivity=12.0),
    )
    device = fdtdx.Device(
        name=device_name,
        partial_grid_shape=(8, 2, 2),
        materials={
            "lower": fdtdx.Material(permittivity=lower_epsilon),
            "upper": fdtdx.Material(permittivity=upper_epsilon),
        },
        param_transforms=[],
        partial_voxel_grid_shape=(1, 1, 1),
    )
    source = fdtdx.PointDipoleSource(
        name="optical-source",
        partial_grid_shape=(1, 1, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=wavelength),
        polarization=1,
        amplitude=1.0,
    )
    detector = fdtdx.PhasorDetector(
        name=detector_name,
        partial_grid_shape=(4, 2, 2),
        wave_characters=(fdtdx.WaveCharacter(wavelength=wavelength),),
        components=("Ey",),
        reduce_volume=True,
        dtype=jnp.complex128,
        dft_subsample=1,
        plot=False,
    )
    config = fdtdx.SimulationConfig(
        time=20.0e-15,
        grid=fdtdx.UniformGrid(
            spacing=0.125e-6,
            center=(1.0e-6, 0.0, 0.25e-6),
        ),
        backend="cpu",
        dtype=jnp.float64,
        gradient_config=fdtdx.GradientConfig(method="checkpointed", num_checkpoints=4),
    )
    constraints = [
        *boundary_constraints,
        waveguide.place_at_center(volume, axes=(0, 1, 2)),
        device.place_at_center(volume, axes=(0, 1, 2)),
        source.set_grid_coordinates(
            axes=(0, 1, 2),
            sides=("-", "-", "-"),
            coordinates=(2, 2, 2),
        ),
        detector.set_grid_coordinates(
            axes=(0, 1, 2),
            sides=("-", "-", "-"),
            coordinates=(12, 1, 1),
        ),
    ]
    objects, arrays, parameters, config, _placement = fdtdx.place_objects(
        [volume, *boundaries.values(), waveguide, device, source, detector],
        config,
        constraints,
        key=jax.random.PRNGKey(29),
    )
    placed_device = objects[device_name]
    target_coordinates = tuple(
        np.asarray(config.resolved_grid.centers(axis)[placed_device.grid_slice[axis]])
        for axis in range(3)
    )
    sampling = build_triangle_p1_sampling_plan(
        coordinates,
        cells,
        target_coordinates,
        plane_axes=(0, 2),
    )
    law = ThermoOpticLaw(
        material_region="silicon",
        reference_temperature_k=300.0,
        reference_refractive_index=3.48,
        thermo_optic_coefficient_per_k=1.86e-4,
        vacuum_wavelength_m=wavelength,
    )
    fingerprint = FDTDXFingerprint(
        package_version="0.6.2",
        source_revision=FDTDX_SOURCE_REVISION,
        source_digest=FDTDX_SOURCE_DIGEST,
    )
    contract = FDTDXDeviceParameterContract(
        device_name=device_name,
        target_shape=sampling.target_shape,
        plane_axes=sampling.plane_axes,
        lower_relative_permittivity=lower_epsilon,
        upper_relative_permittivity=upper_epsilon,
        parameter_dtype="float64",
        thermo_optic_law_sha256=law.sha256,
        target_coordinate_sha256=sampling.target_coordinate_sha256,
        transfer_operator_sha256=sampling.operator_sha256,
        fdtdx=fingerprint,
    )
    transfer_plan = prepare_distributed_triangle_p1_sampling_plan(
        sampling,
        electrothermal_plan.layout.transport.cell_ids,
        source_layout_sha256=electrothermal_plan.layout.digest(),
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
    material_shardings = fdtdx.capture_material_array_shardings(arrays)
    parameters = dict(parameters)
    parameters[device_name] = jnp.asarray(parameters[device_name], dtype=jnp.float64)
    key = jax.random.PRNGKey(31)

    def updated_scene(
        arrays_arg,
        objects_arg,
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
        thermo_optic = transfer_runtime.state(transfer_inputs, cell_temperature)
        updated_parameters = with_fdtdx_device_parameter(parameters, thermo_optic, contract)
        updated_arrays, updated_objects, _application = fdtdx.apply_params(
            arrays=arrays_arg,
            objects=objects_arg,
            params=updated_parameters,
            key=key,
            material_array_shardings=material_shardings,
        )
        return updated_arrays, updated_objects, thermo_optic

    def material_state(
        arrays_arg,
        objects_arg,
        current_values: jax.Array,
        thermal_values: jax.Array,
        feedback_values: jax.Array,
    ):
        updated_arrays, _updated_objects, thermo_optic = updated_scene(
            arrays_arg,
            objects_arg,
            current_values,
            thermal_values,
            feedback_values,
        )
        return updated_arrays.inv_permittivities, thermo_optic

    compiled_material = jax.jit(material_state)
    inverse_permittivity, thermo_optic = compiled_material(
        arrays,
        objects,
        current,
        thermal,
        feedback,
    )
    jax.block_until_ready((inverse_permittivity, thermo_optic))
    device_inverse_permittivity = inverse_permittivity[(slice(None), *placed_device.grid_slice)]
    material_relative_difference = _relative_difference(
        1.0 / device_inverse_permittivity[0],
        thermo_optic.relative_permittivity,
    )

    def optical_objective(
        arrays_arg,
        objects_arg,
        config_arg,
        current_values: jax.Array,
        thermal_values: jax.Array,
        feedback_values: jax.Array,
    ) -> jax.Array:
        updated_arrays, updated_objects, _thermo_optic = updated_scene(
            arrays_arg,
            objects_arg,
            current_values,
            thermal_values,
            feedback_values,
        )
        _step, final_arrays = fdtdx.run_fdtd(
            arrays=updated_arrays,
            objects=updated_objects,
            config=config_arg,
            key=key,
            show_progress=False,
        )
        phasor = final_arrays.detector_states[detector_name]["phasor"]
        return jnp.sum(jnp.abs(phasor) ** 2)

    value_and_grad = jax.jit(jax.value_and_grad(optical_objective, argnums=(3, 4, 5)))
    arguments = (arrays, objects, config, current, thermal, feedback)
    lowered = value_and_grad.lower(*arguments)
    stablehlo = str(lowered.compiler_ir("stablehlo"))
    objective, gradients = value_and_grad(*arguments)
    jax.block_until_ready((objective, gradients))

    step = 2.0e-5
    current_plus = current.at[0].add(step)
    current_minus = current.at[0].add(-step)
    forward = jax.jit(optical_objective)
    finite_difference = (
        forward(arrays, objects, config, current_plus, thermal, feedback)
        - forward(arrays, objects, config, current_minus, thermal, feedback)
    ) / (2.0 * step)
    jax.block_until_ready(finite_difference)
    finite_difference_relative_error = _relative_difference(
        gradients[0][0],
        finite_difference,
    )

    dense_temperature = system.temperature(current, thermal, feedback)
    dense_thermo_optic = thermo_optic_parameter_state(
        sampling,
        dense_temperature,
        law,
        contract,
    )
    result = {
        "schema_version": "femx.fdtdx.distributed_thermo_optic.cpu_objective/v1",
        "backend": jax.default_backend(),
        "forced_cpu_device_count": len(devices),
        "process_count": jax.process_count(),
        "fdtdx_version": distribution_version("fdtdx"),
        "fdtdx_source_revision": FDTDX_SOURCE_REVISION,
        "fdtdx_source_digest": FDTDX_SOURCE_DIGEST,
        "fdtdx_module_sha256": FDTDX_MODULE_SHA256,
        "electrothermal_plan_sha256": electrothermal_plan.digest(),
        "transfer_operator_sha256": transfer_plan.operator_sha256,
        "target_shape": transfer_plan.target_shape,
        "material_sharding": str(inverse_permittivity.sharding.spec),
        "thermo_optic_parameter_sharding": str(thermo_optic.parameter.sharding.spec),
        "material_relative_difference": material_relative_difference,
        "parameter_dense_relative_difference": _relative_difference(
            thermo_optic.parameter,
            dense_thermo_optic.parameter,
        ),
        "all_valid": bool(thermo_optic.all_valid),
        "objective": float(objective),
        "gradients_finite": [bool(jnp.all(jnp.isfinite(value))) for value in gradients],
        "current_gradient_nonzero": bool(jnp.any(gradients[0] != 0.0)),
        "finite_difference_relative_error": finite_difference_relative_error,
        "stablehlo": _hlo_report(stablehlo),
        "claim_scope": (
            "single-process four-forced-CPU portability for the distributed coupled FEM "
            "adjoint, all-to-all P1 thermo-optic transfer, sharding-preserving FDTDX "
            "apply_params, checkpointed Maxwell time advance, and phasor objective; not "
            "physical TPU, multi-host, convergence, 3D FEM, S-parameters, calibrated "
            "material, or device evidence"
        ),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
