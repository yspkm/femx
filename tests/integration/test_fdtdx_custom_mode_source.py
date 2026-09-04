from __future__ import annotations

import math

import numpy as np
import pytest

fdtdx = pytest.importorskip("fdtdx")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from tests.fdtdx_mode_source_support import (  # noqa: E402
    LOCKED_FDTDX_MODE_SOURCE,
    assert_locked_fdtdx_mode_source,
)

from femx.core.axes import Axis, AxisDirection, Direction  # noqa: E402
from femx.interop.fdtdx import (  # noqa: E402
    FieldRepresentation,
    MagneticFieldConvention,
    ModeBundle,
    ModeNormalization,
    SolverFingerprint,
    TransferReport,
    YeeFieldKind,
    YeeVectorField,
    bind_fdtdx_distributed_mode_source,
    build_fdtdx_mode_source_contract,
    build_yee_grid,
    lower_mode_source_inputs_for_tpu,
    make_fdtdx_distributed_mode_source,
    make_fdtdx_mode_source,
    validate_fdtdx_mode_source,
)
from femx.physics import VACUUM_SPEED_OF_LIGHT_M_PER_S  # noqa: E402

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_fdtdx,
    pytest.mark.requires_jax,
]


def _uniform_mode_bundle(
    *,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    source_z_edges: np.ndarray,
    relative_permittivity: float,
) -> ModeBundle:
    grid = build_yee_grid((x_edges, y_edges, source_z_edges))
    area = float((x_edges[-1] - x_edges[0]) * (y_edges[-1] - y_edges[0]))
    effective_index = math.sqrt(relative_permittivity)
    vacuum_impedance = 4.0e-7 * math.pi * VACUUM_SPEED_OF_LIGHT_M_PER_S
    electric_amplitude = math.sqrt(2.0 * vacuum_impedance / (effective_index * area))
    electric = np.zeros((3, *grid.shape), dtype=np.complex128)
    magnetic = np.zeros((3, *grid.shape), dtype=np.complex128)
    electric[0] = electric_amplitude * np.exp(0.125j)
    magnetic[1] = effective_index * electric[0]
    frequency_hz = VACUUM_SPEED_OF_LIGHT_M_PER_S / 1.55e-6
    return ModeBundle(
        frequency_hz=frequency_hz,
        effective_index=effective_index + 0.0j,
        beta_per_m=effective_index * 2.0 * math.pi * frequency_hz / VACUUM_SPEED_OF_LIGHT_M_PER_S,
        electric=YeeVectorField(electric, grid, YeeFieldKind.ELECTRIC, "V/m"),
        magnetic=YeeVectorField(magnetic, grid, YeeFieldKind.MAGNETIC, "V/m"),
        propagation=AxisDirection(Axis.Z, Direction.POSITIVE),
        magnetic_convention=MagneticFieldConvention.ETA0_H,
        normalization=ModeNormalization(target_power_watts=1.0),
        solver=SolverFingerprint(
            "analytic-uniform-port",
            "1",
            "a" * 64,
            "b" * 64,
            "analytic",
        ),
        transfer=TransferReport(
            source_representation=FieldRepresentation.FEM_DOFS,
            target_representation=FieldRepresentation.CARTESIAN_YEE_SAMPLES,
            operator_sha256="c" * 64,
            relative_power_error=0.0,
            source_power_watts=1.0,
            pre_correction_power_watts=1.0,
            relative_pre_correction_power_error=0.0,
            transferred_power_watts=1.0,
            power_correction_scale=1.0,
            target_runtime_name="fdtdx",
            target_runtime_version=LOCKED_FDTDX_MODE_SOURCE.package_version,
            target_source_revision=LOCKED_FDTDX_MODE_SOURCE.source_revision,
            target_source_digest=LOCKED_FDTDX_MODE_SOURCE.source_digest,
        ),
    )


def test_locked_fdtdx_injects_mode_bundle_through_time_domain_source() -> None:
    assert_locked_fdtdx_mode_source()
    relative_permittivity = 2.085136
    x_edges = np.asarray((-200e-9, -100e-9, 0.0, 100e-9, 200e-9), dtype=np.float64)
    y_edges = np.asarray((-180e-9, -80e-9, 20e-9, 120e-9, 220e-9), dtype=np.float64)
    z_edges = np.arange(13, dtype=np.float64) * 40e-9
    source_z_index = 2
    bundle = _uniform_mode_bundle(
        x_edges=x_edges,
        y_edges=y_edges,
        source_z_edges=z_edges[source_z_index : source_z_index + 2],
        relative_permittivity=relative_permittivity,
    )
    plane_area = float((x_edges[-1] - x_edges[0]) * (y_edges[-1] - y_edges[0]))
    electric_sample = bundle.electric.values[0, 0, 0, 0]
    magnetic_sample = bundle.magnetic.values[1, 0, 0, 0]
    source_power = 0.5 * plane_area * float(np.real(electric_sample * magnetic_sample.conjugate()))
    source_power /= 4.0e-7 * math.pi * VACUUM_SPEED_OF_LIGHT_M_PER_S
    assert source_power == pytest.approx(1.0, rel=2e-15)
    inverse_permittivity = np.full(
        (1, *bundle.electric.grid.shape),
        1.0 / relative_permittivity,
        dtype=np.float64,
    )
    contract = build_fdtdx_mode_source_contract(
        bundle,
        source_name="fem-port",
        expected_inverse_permittivity=inverse_permittivity,
        expected_inverse_permeability=np.asarray(1.0, dtype=np.float64),
        fdtdx=LOCKED_FDTDX_MODE_SOURCE,
    )
    source = make_fdtdx_mode_source(
        bundle,
        contract,
        verified_fingerprint=LOCKED_FDTDX_MODE_SOURCE,
    )

    volume = fdtdx.SimulationVolume(
        partial_grid_shape=(4, 4, 12),
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
    ]
    config = fdtdx.SimulationConfig(
        time=15e-15,
        grid=fdtdx.RectilinearGrid(
            x_edges=jnp.asarray(x_edges),
            y_edges=jnp.asarray(y_edges),
            z_edges=jnp.asarray(z_edges),
        ),
        backend="cpu",
        dtype=jnp.float64,
    )
    key = jax.random.PRNGKey(101)
    objects, arrays, parameters, config, _info = fdtdx.place_objects(
        [volume, *boundaries.values(), source],
        config,
        constraints,
        key=key,
    )
    placed_source = objects[contract.source_name]
    validate_fdtdx_mode_source(placed_source, bundle, contract)
    np.testing.assert_array_equal(np.asarray(placed_source._E), bundle.electric.values)
    np.testing.assert_array_equal(np.asarray(placed_source._H), bundle.magnetic.values)

    arrays, objects, _apply_info = fdtdx.apply_params(
        arrays=arrays,
        objects=objects,
        params=parameters,
        key=key,
    )
    _step, final_arrays = fdtdx.run_fdtd(
        arrays=arrays,
        objects=objects,
        config=config,
        key=key,
        show_progress=False,
    )
    assert float(jnp.linalg.norm(final_arrays.fields.E)) > 0.0
    assert float(jnp.linalg.norm(final_arrays.fields.H)) > 0.0
    assert bool(jnp.all(jnp.isfinite(final_arrays.fields.E)))
    assert bool(jnp.all(jnp.isfinite(final_arrays.fields.H)))


def test_explicit_tpu_precision_mode_runs_on_the_locked_float32_path() -> None:
    """Exercise the TPU scalar contract on CPU without making an accelerator claim."""

    with jax.enable_x64(False):
        _assert_explicit_tpu_precision_mode_runs_on_float32()


def _assert_explicit_tpu_precision_mode_runs_on_float32() -> None:
    """Run the locked source while JAX enforces the target TPU scalar policy."""

    assert_locked_fdtdx_mode_source()
    relative_permittivity = 2.085136
    x_edges = np.asarray((-200e-9, -100e-9, 0.0, 100e-9, 200e-9), dtype=np.float64)
    y_edges = np.asarray((-180e-9, -80e-9, 20e-9, 120e-9, 220e-9), dtype=np.float64)
    z_edges = np.arange(9, dtype=np.float64) * 40e-9
    source_z_index = 2
    bundle = _uniform_mode_bundle(
        x_edges=x_edges,
        y_edges=y_edges,
        source_z_edges=z_edges[source_z_index : source_z_index + 2],
        relative_permittivity=relative_permittivity,
    )
    inverse_permittivity = np.full(
        (1, *bundle.electric.grid.shape),
        1.0 / relative_permittivity,
        dtype=np.float64,
    )
    lowered = lower_mode_source_inputs_for_tpu(
        bundle,
        expected_inverse_permittivity=inverse_permittivity,
        expected_inverse_permeability=np.asarray(1.0, dtype=np.float64),
    )
    contract = build_fdtdx_mode_source_contract(
        lowered.bundle,
        source_name="fem-port-fp32",
        expected_inverse_permittivity=lowered.expected_inverse_permittivity,
        expected_inverse_permeability=lowered.expected_inverse_permeability,
        fdtdx=LOCKED_FDTDX_MODE_SOURCE,
    )
    source = make_fdtdx_distributed_mode_source(
        lowered.bundle,
        contract,
        verified_fingerprint=LOCKED_FDTDX_MODE_SOURCE,
    )

    runtime_x_edges, runtime_y_edges, _source_edges = (
        np.asarray(axis) for axis in lowered.bundle.electric.grid.edge_coordinates
    )
    runtime_z_edges = np.asarray(z_edges, dtype=np.float32)
    volume = fdtdx.SimulationVolume(
        partial_grid_shape=(4, 4, 8),
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
    constraints = [
        *boundary_constraints,
        source.same_size(volume, axes=(0, 1)),
        source.place_at_center(volume, axes=(0, 1)),
        fdtdx.RealCoordinateConstraint(
            object=contract.source_name,
            axes=(2,),
            sides=("-",),
            coordinates=(float(runtime_z_edges[source_z_index]),),
        ),
    ]
    config = fdtdx.SimulationConfig(
        time=5e-15,
        grid=fdtdx.RectilinearGrid(
            x_edges=jnp.asarray(runtime_x_edges),
            y_edges=jnp.asarray(runtime_y_edges),
            z_edges=jnp.asarray(runtime_z_edges),
        ),
        backend="cpu",
        dtype=jnp.float32,
    )
    key = jax.random.PRNGKey(103)
    objects, arrays, parameters, config, _info = fdtdx.place_objects(
        [volume, *boundaries.values(), source],
        config,
        constraints,
        key=key,
    )
    placed_source = objects[contract.source_name]
    validate_fdtdx_mode_source(placed_source, lowered.bundle, contract)
    assert placed_source._E.dtype == jnp.complex64
    assert placed_source._H.dtype == jnp.complex64
    assert arrays.fields.E.dtype == jnp.float32
    assert arrays.inv_permittivities.dtype == jnp.float32

    arrays, objects, _apply_info = fdtdx.apply_params(
        arrays=arrays,
        objects=objects,
        params=parameters,
        key=key,
    )
    objects, binding = bind_fdtdx_distributed_mode_source(
        objects,
        lowered.bundle,
        contract,
    )
    bound_source = objects[contract.source_name]
    assert bound_source._E.sharding == arrays.fields.E.sharding
    assert bound_source._H.sharding == arrays.fields.H.sharding
    assert bound_source._time_offset_E.sharding == bound_source._E.sharding
    assert bound_source._time_offset_H.sharding == bound_source._H.sharding
    assert binding.global_device_count == jax.device_count()
    assert binding.local_device_count == jax.local_device_count()
    assert binding.canonical_data()["physical_evidence"] is False

    jitted_run_fdtd = jax.jit(
        fdtdx.run_fdtd,
        static_argnames=("show_progress", "progress_callback"),
    )
    _step, final_arrays = jitted_run_fdtd(
        arrays=arrays,
        objects=objects,
        config=config,
        key=key,
        show_progress=False,
    )
    assert float(jnp.linalg.norm(final_arrays.fields.E)) > 0.0
    assert float(jnp.linalg.norm(final_arrays.fields.H)) > 0.0
    assert bool(jnp.all(jnp.isfinite(final_arrays.fields.E)))
    assert bool(jnp.all(jnp.isfinite(final_arrays.fields.H)))
