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
    build_fdtdx_mode_source_contract,
    make_fdtdx_mode_source,
    validate_fdtdx_mode_source,
)

pytestmark = [
    pytest.mark.scientific,
    pytest.mark.requires_fdtdx,
    pytest.mark.requires_jax,
]


@pytest.mark.parametrize("gradient_method", ["checkpointed", "reversible"])
def test_custom_mode_source_preserves_downstream_fdtd_gradient(gradient_method: str) -> None:
    """Both FDTDX VJPs match finite differences beyond a static imported mode plane."""

    lower_permittivity = 2.085136
    upper_permittivity = 2.35
    x_edges = np.arange(5, dtype=np.float64) * 100e-9
    y_edges = np.arange(5, dtype=np.float64) * 100e-9
    z_edges = np.arange(17, dtype=np.float64) * 50e-9
    source_z_index = 2
    bundle = uniform_mode_bundle(
        x_edges=x_edges,
        y_edges=y_edges,
        source_z_edges=z_edges[source_z_index : source_z_index + 2],
        relative_permittivity=lower_permittivity,
    )
    contract = build_fdtdx_mode_source_contract(
        bundle,
        source_name="fem-port",
        expected_inverse_permittivity=np.full(
            (1, *bundle.electric.grid.shape),
            1.0 / lower_permittivity,
            dtype=np.float64,
        ),
        expected_inverse_permeability=np.asarray(1.0, dtype=np.float64),
        fdtdx=LOCKED_FDTDX_MODE_SOURCE,
    )
    source = make_fdtdx_mode_source(
        bundle,
        contract,
        verified_fingerprint=LOCKED_FDTDX_MODE_SOURCE,
    )

    volume = fdtdx.SimulationVolume(
        partial_grid_shape=(4, 4, 16),
        material=fdtdx.Material(permittivity=lower_permittivity),
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
    device_name = "phase-segment"
    device = fdtdx.Device(
        name=device_name,
        partial_grid_shape=(4, 4, 4),
        materials={
            "lower": fdtdx.Material(permittivity=lower_permittivity),
            "upper": fdtdx.Material(permittivity=upper_permittivity),
        },
        param_transforms=[],
        partial_voxel_grid_shape=(1, 1, 1),
    )
    detector_name = "output-phasor"
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
    gradient_config = (
        fdtdx.GradientConfig(method="checkpointed", num_checkpoints=4)
        if gradient_method == "checkpointed"
        else fdtdx.GradientConfig(
            method="reversible",
            recorder=fdtdx.Recorder(modules=[]),
        )
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
        gradient_config=gradient_config,
    )
    constraints = [
        *boundary_constraints,
        source.same_size(volume, axes=(0, 1)),
        source.place_at_center(volume, axes=(0, 1)),
        fdtdx.RealCoordinateConstraint(
            object=contract.source_name,
            axes=(2,),
            sides=("-",),
            coordinates=(float(z_edges[source_z_index]),),
        ),
        device.place_at_center(volume, axes=(0, 1)),
        fdtdx.RealCoordinateConstraint(
            object=device_name,
            axes=(2,),
            sides=("-",),
            coordinates=(float(z_edges[6]),),
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
    key = jax.random.PRNGKey(107)
    objects, arrays, parameters, config, _info = fdtdx.place_objects(
        [volume, *boundaries.values(), source, device, detector],
        config,
        constraints,
        key=key,
    )
    validate_fdtdx_mode_source(objects[contract.source_name], bundle, contract)
    base_parameters = dict(parameters)

    def objective(value: jax.Array) -> jax.Array:
        candidate_parameters = dict(base_parameters)
        candidate_parameters[device_name] = jnp.full_like(
            base_parameters[device_name],
            value,
        )
        updated_arrays, updated_objects, _apply_info = fdtdx.apply_params(
            arrays=arrays,
            objects=objects,
            params=candidate_parameters,
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

    design = jnp.asarray(0.35, dtype=jnp.float64)
    value, gradient = jax.value_and_grad(objective)(design)
    step = 1e-3
    finite_difference = (objective(design + step) - objective(design - step)) / (2.0 * step)

    assert bool(jnp.isfinite(value))
    assert float(value) > 0.0
    assert bool(jnp.isfinite(gradient))
    assert float(jnp.abs(gradient)) > 0.0
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-3, atol=1e-18)
