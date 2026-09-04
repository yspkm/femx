import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from femx.interop.fdtdx import (  # noqa: E402
    FDTDXDeviceParameterContract,
    FDTDXFingerprint,
    ThermoOpticLaw,
    build_triangle_p1_sampling_plan,
    thermo_optic_parameter_state,
)

pytestmark = [pytest.mark.scientific, pytest.mark.requires_jax]


def test_p1_thermo_optic_pullback_matches_analytic_transpose_and_finite_difference() -> None:
    coordinates = np.asarray(
        (
            (0.0, 0.0),
            (2.0e-6, 0.0),
            (0.0, 0.5e-6),
            (2.0e-6, 0.5e-6),
        )
    )
    cells = np.asarray(((0, 1, 2), (1, 3, 2)), dtype=np.int64)
    target_coordinates = (
        np.asarray((0.25e-6, 1.0e-6, 1.75e-6)),
        np.asarray((-0.1e-6, 0.1e-6)),
        np.asarray((0.125e-6, 0.375e-6)),
    )
    plan = build_triangle_p1_sampling_plan(
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
        vacuum_wavelength_m=1.55e-6,
    )
    contract = FDTDXDeviceParameterContract(
        device_name="heated-silicon",
        target_shape=plan.target_shape,
        plane_axes=plan.plane_axes,
        lower_relative_permittivity=11.0,
        upper_relative_permittivity=13.0,
        parameter_dtype="float64",
        thermo_optic_law_sha256=law.sha256,
        target_coordinate_sha256=plan.target_coordinate_sha256,
        transfer_operator_sha256=plan.operator_sha256,
        fdtdx=FDTDXFingerprint(
            package_version="0.6.2",
            source_revision="eaab78a42cd1351b7f447f312fa50c9febfe4b99",
            source_digest=("cf7bf29a1aa2411f2ffc84dcf2c1806d43d1823e32a60f234134298171da7d08"),
        ),
    )
    nodal_temperature = jnp.asarray((300.0, 301.5, 302.0, 303.0), dtype=jnp.float64)
    objective_weights = jnp.linspace(
        0.7,
        1.3,
        np.prod(plan.target_shape),
        dtype=jnp.float64,
    ).reshape(plan.target_shape)
    objective_weights /= jnp.sum(objective_weights)

    def objective(temperature: jax.Array) -> jax.Array:
        state = thermo_optic_parameter_state(plan, temperature, law, contract)
        return jnp.vdot(objective_weights, 1.0 / state.relative_permittivity)

    gradient = np.asarray(jax.grad(objective)(nodal_temperature))
    compiled = jax.jit(objective)(nodal_temperature)
    np.testing.assert_allclose(compiled, objective(nodal_temperature), rtol=0.0, atol=1.0e-15)

    finite_difference = np.empty_like(gradient)
    step = 2.0e-4
    for index in range(nodal_temperature.size):
        upper = np.asarray(nodal_temperature).copy()
        lower = upper.copy()
        upper[index] += step
        lower[index] -= step
        finite_difference[index] = float(
            (objective(jnp.asarray(upper)) - objective(jnp.asarray(lower))) / (2.0 * step)
        )

    state = thermo_optic_parameter_state(plan, nodal_temperature, law, contract)
    local_derivative = (
        -2.0 * law.thermo_optic_coefficient_per_k / np.asarray(state.refractive_index) ** 3
    )
    analytic = np.zeros_like(gradient)
    cell_indices = np.asarray(plan.target_cell_indices)
    barycentric = np.asarray(plan.barycentric_weights)
    for target_index in np.ndindex(plan.target_shape):
        cell = cells[cell_indices[target_index]]
        contribution = (
            float(objective_weights[target_index])
            * local_derivative[target_index]
            * barycentric[target_index]
        )
        np.add.at(analytic, cell, contribution)

    np.testing.assert_allclose(gradient, analytic, rtol=3.0e-13, atol=2.0e-16)
    np.testing.assert_allclose(gradient, finite_difference, rtol=2.0e-8, atol=2.0e-13)
