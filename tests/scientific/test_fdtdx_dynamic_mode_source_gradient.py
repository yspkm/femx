from __future__ import annotations

import numpy as np
import pytest

fdtdx = pytest.importorskip("fdtdx")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from tests.fdtdx_mode_source_support import (  # noqa: E402
    LOCKED_FDTDX_MODE_SOURCE,
    uniform_mode_bundle,
)

from femx.interop.fdtdx import (  # noqa: E402
    build_fdtdx_dynamic_mode_source_contract,
    build_fdtdx_mode_source_contract,
    make_fdtdx_dynamic_mode_source,
    validate_fdtdx_mode_source,
    with_fdtdx_dynamic_mode_profile,
)

pytestmark = [
    pytest.mark.scientific,
    pytest.mark.requires_fdtdx,
    pytest.mark.requires_jax,
]


def test_checkpointed_fdtd_backpropagates_through_dynamic_mode_profile() -> None:
    """A JAX-valued source profile reaches a downstream detector through real FDTD."""

    relative_permittivity = 2.085136
    x_edges = np.arange(5, dtype=np.float64) * 100e-9
    y_edges = np.arange(5, dtype=np.float64) * 100e-9
    z_edges = np.arange(17, dtype=np.float64) * 50e-9
    source_z_index = 2
    bundle = uniform_mode_bundle(
        x_edges=x_edges,
        y_edges=y_edges,
        source_z_edges=z_edges[source_z_index : source_z_index + 2],
        relative_permittivity=relative_permittivity,
    )
    baseline_contract = build_fdtdx_mode_source_contract(
        bundle,
        source_name="dynamic-fem-port",
        expected_inverse_permittivity=np.full(
            (1, *bundle.electric.grid.shape),
            1.0 / relative_permittivity,
            dtype=np.float64,
        ),
        expected_inverse_permeability=np.asarray(1.0, dtype=np.float64),
        fdtdx=LOCKED_FDTDX_MODE_SOURCE,
    )
    dynamic_contract = build_fdtdx_dynamic_mode_source_contract(
        bundle,
        baseline_contract,
        parameter_names=("profile_impedance",),
        parameter_units=("1",),
    )
    source = make_fdtdx_dynamic_mode_source(
        bundle,
        dynamic_contract,
        verified_fingerprint=LOCKED_FDTDX_MODE_SOURCE,
    )
    volume = fdtdx.SimulationVolume(
        partial_grid_shape=(4, 4, 16),
        material=fdtdx.Material(permittivity=relative_permittivity),
    )
    boundaries, boundary_constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(
            thickness=1,
            override_types={
                face: "periodic" for face in ("min_x", "max_x", "min_y", "max_y", "min_z", "max_z")
            },
        ),
        volume,
    )
    detector_name = "dynamic-output-phasor"
    detector = fdtdx.PhasorDetector(
        name=detector_name,
        partial_grid_shape=(4, 4, 1),
        wave_characters=(fdtdx.WaveCharacter(frequency=bundle.frequency_hz),),
        components=("Ex",),
        reduce_volume=True,
        dtype=jnp.complex128,
        dft_subsample=1,
        plot=False,
    )
    config = fdtdx.SimulationConfig(
        time=20e-15,
        grid=fdtdx.RectilinearGrid(
            x_edges=jnp.asarray(x_edges),
            y_edges=jnp.asarray(y_edges),
            z_edges=jnp.asarray(z_edges),
        ),
        backend="cpu",
        dtype=jnp.float64,
        gradient_config=fdtdx.GradientConfig(method="checkpointed", num_checkpoints=4),
    )
    constraints = [
        *boundary_constraints,
        source.same_size(volume, axes=(0, 1)),
        source.place_at_center(volume, axes=(0, 1)),
        fdtdx.RealCoordinateConstraint(
            object=baseline_contract.source_name,
            axes=(2,),
            sides=("-",),
            coordinates=(float(z_edges[source_z_index]),),
        ),
        detector.same_size(volume, axes=(0, 1)),
        detector.place_at_center(volume, axes=(0, 1)),
        fdtdx.RealCoordinateConstraint(
            object=detector_name,
            axes=(2,),
            sides=("-",),
            coordinates=(float(z_edges[12]),),
        ),
    ]
    key = jax.random.PRNGKey(109)
    objects, arrays, _parameters, config, _info = fdtdx.place_objects(
        [volume, *boundaries.values(), source, detector],
        config,
        constraints,
        key=key,
    )
    validate_fdtdx_mode_source(
        objects[baseline_contract.source_name],
        bundle,
        baseline_contract,
    )
    source_index = next(
        index
        for index, item in enumerate(objects.object_list)
        if item.name == baseline_contract.source_name
    )
    baseline_electric = jnp.asarray(bundle.electric.values)
    baseline_magnetic = jnp.asarray(bundle.magnetic.values)
    baseline_neff = jnp.asarray(bundle.effective_index, dtype=jnp.complex128)

    def objective(profile_parameter: jax.Array) -> jax.Array:
        impedance_scale = 1.0 + 0.15 * profile_parameter
        dynamic_source = with_fdtdx_dynamic_mode_profile(
            objects[baseline_contract.source_name],
            dynamic_contract,
            electric_v_per_m=baseline_electric * impedance_scale,
            magnetic_eta0_v_per_m=baseline_magnetic / impedance_scale,
            effective_index=baseline_neff * (1.0 + 0.02 * profile_parameter),
        )
        object_list = list(objects.object_list)
        object_list[source_index] = dynamic_source
        dynamic_objects = objects.aset("object_list", object_list)
        _step, final_arrays = fdtdx.run_fdtd(
            arrays=arrays,
            objects=dynamic_objects,
            config=config,
            key=key,
            show_progress=False,
        )
        phasor = final_arrays.detector_states[detector_name]["phasor"]
        return jnp.sum(jnp.abs(phasor) ** 2)

    parameter = jnp.asarray(0.2, dtype=jnp.float64)
    value, gradient = jax.value_and_grad(objective)(parameter)
    step = jnp.asarray(1e-4, dtype=jnp.float64)
    finite_difference = (objective(parameter + step) - objective(parameter - step)) / (2.0 * step)

    assert bool(jnp.isfinite(value))
    assert float(value) > 0.0
    assert bool(jnp.isfinite(gradient))
    assert float(jnp.abs(gradient)) > 0.0
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-6, atol=0.0)
