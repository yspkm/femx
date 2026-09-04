import numpy as np
import pytest
from tests.electrothermal_support import parameterized_self_consistent_microheater

fdtdx = pytest.importorskip("fdtdx")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from femx.backends.jax.self_consistent import (  # noqa: E402
    DifferentiableSelfConsistentElectrothermal,
)
from femx.backends.jax.steady_current import JaxSteadyCurrentBackend  # noqa: E402
from femx.backends.jax.steady_heat import JaxSteadyHeatBackend  # noqa: E402
from femx.interop.fdtdx import (  # noqa: E402
    FDTDXDeviceParameterContract,
    FDTDXFingerprint,
    ThermoOpticLaw,
    apply_thermo_optic_to_fdtdx,
    build_triangle_p1_sampling_plan,
    thermo_optic_parameter_state,
)
from femx.runtime import prepare  # noqa: E402

pytestmark = [
    pytest.mark.scientific,
    pytest.mark.requires_fdtdx,
    pytest.mark.requires_jax,
]


@pytest.mark.parametrize("gradient_method", ["checkpointed", "reversible"])
def test_locked_fdtdx_run_fdtd_objective_backpropagates_to_electrical_design(
    gradient_method: str,
) -> None:
    feedback, current_parameters, thermal_parameters, feedback_parameters = (
        parameterized_self_consistent_microheater(intervals=2)
    )
    current_backend = JaxSteadyCurrentBackend()
    heat_backend = JaxSteadyHeatBackend()
    system = DifferentiableSelfConsistentElectrothermal.bind(
        feedback,
        current_backend.bind_differentiable(
            prepare(feedback.one_way.electrical_problem, current_backend),
            current_parameters,
        ),
        heat_backend.bind_differentiable(
            prepare(feedback.one_way.thermal_problem, heat_backend),
            thermal_parameters,
        ),
        feedback_parameters,
    )

    wavelength = 1.55e-6
    grid_shape = (16, 4, 4)
    spacing = 0.125e-6
    lower_epsilon = 12.0
    upper_epsilon = 12.5
    device_name = "heated-silicon"
    detector_name = "optical-phasor"
    volume = fdtdx.SimulationVolume(
        partial_grid_shape=grid_shape,
        material=fdtdx.Material(permittivity=2.085136),
    )
    boundary_config = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=2,
        override_types={
            face: "periodic" for face in ("min_x", "max_x", "min_y", "max_y", "min_z", "max_z")
        },
    )
    boundaries, boundary_constraints = fdtdx.boundary_objects_from_config(
        boundary_config,
        volume,
    )
    waveguide = fdtdx.UniformMaterialObject(
        name="silicon-waveguide",
        partial_grid_shape=(16, 2, 2),
        material=fdtdx.Material(permittivity=lower_epsilon),
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
    gradient_config = (
        fdtdx.GradientConfig(method="checkpointed", num_checkpoints=4)
        if gradient_method == "checkpointed"
        else fdtdx.GradientConfig(
            method="reversible",
            recorder=fdtdx.Recorder(modules=[]),
        )
    )
    config = fdtdx.SimulationConfig(
        time=20.0e-15,
        grid=fdtdx.UniformGrid(
            spacing=spacing,
            center=(1.0e-6, 0.0, 0.25e-6),
        ),
        backend="cpu",
        dtype=jnp.float64,
        gradient_config=gradient_config,
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
    objects, arrays, parameters, config, _info = fdtdx.place_objects(
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
    mesh = feedback.one_way.electrical_problem.mesh
    plan = build_triangle_p1_sampling_plan(
        mesh.geometry.coordinates,
        mesh.topology.connectivity,
        target_coordinates,
        plane_axes=(0, 2),
    )
    fingerprint = FDTDXFingerprint(
        package_version="0.6.2",
        source_revision="eaab78a42cd1351b7f447f312fa50c9febfe4b99",
        source_digest="cf7bf29a1aa2411f2ffc84dcf2c1806d43d1823e32a60f234134298171da7d08",
    )
    law = ThermoOpticLaw(
        material_region="silicon",
        reference_temperature_k=300.0,
        reference_refractive_index=3.48,
        thermo_optic_coefficient_per_k=1.86e-4,
        vacuum_wavelength_m=wavelength,
    )
    contract = FDTDXDeviceParameterContract(
        device_name=device_name,
        target_shape=plan.target_shape,
        plane_axes=plan.plane_axes,
        lower_relative_permittivity=lower_epsilon,
        upper_relative_permittivity=upper_epsilon,
        parameter_dtype="float64",
        thermo_optic_law_sha256=law.sha256,
        target_coordinate_sha256=plan.target_coordinate_sha256,
        transfer_operator_sha256=plan.operator_sha256,
        fdtdx=fingerprint,
    )
    parameters = dict(parameters)
    parameters[device_name] = jnp.asarray(parameters[device_name], dtype=jnp.float64)
    key = jax.random.PRNGKey(31)

    def objective(current_values: jax.Array) -> jax.Array:
        temperature = system.temperature(
            current_values,
            system.initial_thermal_values,
            system.initial_feedback_values,
        )
        state = thermo_optic_parameter_state(plan, temperature, law, contract)
        updated_arrays, updated_objects, _apply_info = apply_thermo_optic_to_fdtdx(
            arrays,
            objects,
            parameters,
            config,
            state,
            contract,
            verified_fingerprint=fingerprint,
            key=key,
        )
        _step, final_arrays = fdtdx.run_fdtd(
            arrays=updated_arrays,
            objects=updated_objects,
            config=config,
            key=key,
            show_progress=False,
        )
        phasor = final_arrays.detector_states[detector_name]["phasor"]
        return jnp.sum(jnp.abs(phasor) ** 2)

    value, gradient = jax.value_and_grad(objective)(system.initial_current_values)
    assert np.isfinite(value)
    assert value > 0.0
    assert np.all(np.isfinite(gradient))
    assert np.any(np.asarray(gradient) != 0.0)

    initial = np.asarray(system.initial_current_values)

    def central_difference(index: int, step: float) -> float:
        upper = initial.copy()
        lower = initial.copy()
        upper[index] += step
        lower[index] -= step
        return float((objective(jnp.asarray(upper)) - objective(jnp.asarray(lower))) / (2.0 * step))

    finite_difference = np.asarray(
        (
            central_difference(0, 2.0e-5),
            central_difference(1, 2.0e-1),
        )
    )
    np.testing.assert_allclose(gradient, finite_difference, rtol=2.0e-3, atol=1.0e-18)
