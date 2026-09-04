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


def test_locked_fdtdx_material_boundary_backpropagates_through_electrothermal_adjoint() -> None:
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

    grid_shape = (8, 1, 2)
    spacing = 0.25e-6
    lower_epsilon = 12.0
    upper_epsilon = 12.5
    device_name = "heated-silicon"
    volume = fdtdx.SimulationVolume(partial_grid_shape=grid_shape)
    device = fdtdx.Device(
        name=device_name,
        partial_grid_shape=grid_shape,
        materials={
            "lower": fdtdx.Material(permittivity=lower_epsilon),
            "upper": fdtdx.Material(permittivity=upper_epsilon),
        },
        param_transforms=[],
        partial_voxel_grid_shape=(1, 1, 1),
    )
    config = fdtdx.SimulationConfig(
        time=1.0e-15,
        grid=fdtdx.UniformGrid(
            spacing=spacing,
            center=(1.0e-6, 0.0, 0.25e-6),
        ),
        backend="cpu",
        dtype=jnp.float64,
        gradient_config=None,
    )
    objects, arrays, parameters, config, _info = fdtdx.place_objects(
        [volume, device],
        config,
        [device.place_at_center(volume, axes=(0, 1, 2))],
        key=jax.random.PRNGKey(17),
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
        vacuum_wavelength_m=1.55e-6,
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
    weights = jnp.linspace(0.75, 1.25, np.prod(grid_shape), dtype=jnp.float64).reshape(grid_shape)
    weights /= jnp.sum(weights)

    def objective(current_values: jax.Array) -> jax.Array:
        temperature = system.temperature(
            current_values,
            system.initial_thermal_values,
            system.initial_feedback_values,
        )
        state = thermo_optic_parameter_state(plan, temperature, law, contract)
        updated_arrays, _updated_objects, _apply_info = apply_thermo_optic_to_fdtdx(
            arrays,
            objects,
            parameters,
            config,
            state,
            contract,
            verified_fingerprint=fingerprint,
            key=jax.random.PRNGKey(23),
        )
        inverse_permittivity = updated_arrays.inv_permittivities[0, *placed_device.grid_slice]
        return jnp.vdot(weights, inverse_permittivity)

    temperature = system.temperature(
        system.initial_current_values,
        system.initial_thermal_values,
        system.initial_feedback_values,
    )
    state = thermo_optic_parameter_state(plan, temperature, law, contract)
    updated_arrays, _updated_objects, _apply_info = apply_thermo_optic_to_fdtdx(
        arrays,
        objects,
        parameters,
        config,
        state,
        contract,
        verified_fingerprint=fingerprint,
        key=jax.random.PRNGKey(23),
    )
    actual_epsilon = 1.0 / np.asarray(
        updated_arrays.inv_permittivities[0, *placed_device.grid_slice]
    )
    np.testing.assert_allclose(
        actual_epsilon,
        state.relative_permittivity,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    assert bool(state.all_valid)
    assert plan.maximum_partition_error < 2.0e-16

    gradient = np.asarray(jax.grad(objective)(system.initial_current_values))
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
    np.testing.assert_allclose(gradient, finite_difference, rtol=3.0e-5, atol=2.0e-12)
