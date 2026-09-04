"""Canonical bounded FDTDX scene for the distributed thermo-optic TPU gate.

This module is runner support rather than public API.  It fixes the optical geometry, scalar
precision, source, detector, material bracket, and locked FDTDX source identity used by M2e.7b.
The scene is intentionally small and periodic: it is an execution-integrity witness for the full
coupled derivative graph, not a transmission, convergence, or fabricated-device model.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, NamedTuple

FDTDX_PACKAGE_VERSION = "0.6.2"
FDTDX_SOURCE_REVISION = "0c05c4784b2be83b42d9b46ab089265981ba157f"
FDTDX_SOURCE_DIGEST = "29bed9483c4c2b57fd2f495fdb47534edf6b244206679e34b2de41ec39aaa9fa"
FDTDX_MODULE_SHA256 = {
    "__init__.py": "fcf000b7955c97e7fbe1ccd5901c1f5ba47a5bfd86f0fce3d2dc8be1bfe131cf",
    "core/jax/sharding.py": "a6e07ac439c1c1b48958380812406f090844a1b4924a3b3a9b0a7f49eca8a9c3",
    "fdtd/fdtd.py": "7c654097d43d5062afbef0cf8c479ba2a7db523b64683693fa4e24bc5070e4e0",
    "fdtd/initialization.py": "2b7d56d47789f38c73b96fe7a078521e1146a45e98753af6c5e536ea8f9225a1",
    "fdtd/wrapper.py": "97d562e0a33eeeccd6ce42d12cf8dc29f1e9fd071e561b5b009ac5da8ece7384",
}

GRID_SHAPE = (96, 4, 8)
DEVICE_SHAPE = (32, 2, 4)
DEVICE_GRID_SLICE = (slice(32, 64), slice(1, 3), slice(2, 6))
WAVEGUIDE_SHAPE = (96, 2, 4)
GRID_SPACING_M = 62.5e-9
GRID_CENTER_M = (1.0e-6, 0.0, 0.25e-6)
SIMULATION_TIME_S = 36.0e-15
FDTDX_TIME_STEPS = 302
WAVELENGTH_M = 1.55e-6
LOWER_RELATIVE_PERMITTIVITY = 10.0
UPPER_RELATIVE_PERMITTIVITY = 16.0
CLADDING_RELATIVE_PERMITTIVITY = 2.085136
STATIC_WAVEGUIDE_RELATIVE_PERMITTIVITY = 12.0
DEVICE_NAME = "heated-silicon"
DETECTOR_NAME = "optical-phasor"
SOURCE_NAME = "optical-source"
MESH_AXIS_NAME = "shard"
RANDOM_SEED = 20260903
# FDTDX forms a runtime float32 grid through several rounded multiply/add operations, whereas the
# controller freezes the same cell centers in float64 for P1 point location. The physical gate
# admits only a tiny, explicitly recorded difference between those two representations.
RUNTIME_TARGET_COORDINATE_MAX_ULP_ERROR = 8
RUNTIME_TARGET_COORDINATE_MAX_GRID_FRACTION_ERROR = 4.0e-6
# The fixed scale conditions the float32 residual adjoint; it does not change the frozen phasor
# reference, material law, or scientific claim boundary.
PHASOR_OBJECTIVE_SCALE = 1.0e8


@dataclass(frozen=True, slots=True)
class PreparedThermoOpticScene:
    """Placed FDTDX objects plus exact coordinates of the active material voxels."""

    objects: Any
    arrays: Any
    parameters: dict[str, object]
    config: Any
    placed_device: Any
    target_coordinates: tuple[Any, Any, Any]
    material_array_shardings: Any
    key: Any


class CoupledRuntimeInputs(NamedTuple):
    """Globally sharded FDTDX/FEM state passed through the outer JIT boundary."""

    electrothermal: Any
    thermo_optic: Any
    fdtdx_arrays: Any
    fdtdx_objects: Any
    fdtdx_parameters: Any
    fdtdx_config: Any
    fdtdx_key: Any


def coupled_mesh_from_material_sharding(
    material_sharding: object,
    mesh_type: type[Any],
    *,
    axis_name: str,
    global_device_count: int,
) -> Any:
    """Adopt the concrete FDTDX device order for every coupled input array."""

    mesh = getattr(material_sharding, "mesh", None)
    if not isinstance(mesh, mesh_type):
        raise RuntimeError("FDTDX material arrays must expose a concrete NamedSharding Mesh")
    if bool(mesh.empty) or int(mesh.size) != global_device_count:
        raise RuntimeError("FDTDX material Mesh must contain every global device exactly once")
    if tuple(mesh.axis_names) != (axis_name,) or int(mesh.devices.ndim) != 1:
        raise RuntimeError("FDTDX material Mesh must use the canonical one-dimensional axis")
    return mesh


def verify_locked_fdtdx(fdtdx: Any) -> dict[str, str]:
    """Verify package version and every runtime module used by the coupled graph."""

    if distribution_version("fdtdx") != FDTDX_PACKAGE_VERSION:
        raise RuntimeError(f"locked FDTDX {FDTDX_PACKAGE_VERSION} is required")
    if not callable(getattr(fdtdx, "capture_material_array_shardings", None)):
        raise RuntimeError("locked FDTDX does not expose material sharding capture")
    package_root = Path(fdtdx.__file__).resolve().parent
    observed: dict[str, str] = {}
    for relative_path, expected_sha256 in FDTDX_MODULE_SHA256.items():
        source = package_root / relative_path
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise RuntimeError(f"locked FDTDX module hash differs for {relative_path}")
        observed[relative_path] = digest
    return observed


def thermo_optic_law() -> Any:
    """Return the wavelength-specific linear silicon thermo-optic law for this witness."""

    from femx.interop.fdtdx import ThermoOpticLaw

    return ThermoOpticLaw(
        material_region="silicon",
        reference_temperature_k=300.0,
        reference_refractive_index=3.48,
        thermo_optic_coefficient_per_k=1.86e-4,
        vacuum_wavelength_m=WAVELENGTH_M,
    )


def device_contract(sampling: Any, *, parameter_dtype: str) -> Any:
    """Bind one canonical sampler to the raw FDTDX Device parameter."""

    from femx.interop.fdtdx import FDTDXDeviceParameterContract, FDTDXFingerprint

    return FDTDXDeviceParameterContract(
        device_name=DEVICE_NAME,
        target_shape=sampling.target_shape,
        plane_axes=sampling.plane_axes,
        lower_relative_permittivity=LOWER_RELATIVE_PERMITTIVITY,
        upper_relative_permittivity=UPPER_RELATIVE_PERMITTIVITY,
        parameter_dtype=parameter_dtype,
        thermo_optic_law_sha256=thermo_optic_law().sha256,
        target_coordinate_sha256=sampling.target_coordinate_sha256,
        transfer_operator_sha256=sampling.operator_sha256,
        fdtdx=FDTDXFingerprint(
            package_version=FDTDX_PACKAGE_VERSION,
            source_revision=FDTDX_SOURCE_REVISION,
            source_digest=FDTDX_SOURCE_DIGEST,
        ),
    )


def build_scene(fdtdx: Any, jax: Any, jnp: Any, *, backend: str) -> PreparedThermoOpticScene:
    """Place the bounded 3D scene with an x-sharded raw thermo-optic Device."""

    if backend not in {"cpu", "tpu"}:
        raise ValueError("distributed thermo-optic scene backend must be 'cpu' or 'tpu'")
    verify_locked_fdtdx(fdtdx)
    volume = fdtdx.SimulationVolume(
        partial_grid_shape=GRID_SHAPE,
        material=fdtdx.Material(permittivity=CLADDING_RELATIVE_PERMITTIVITY),
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
        partial_grid_shape=WAVEGUIDE_SHAPE,
        material=fdtdx.Material(permittivity=STATIC_WAVEGUIDE_RELATIVE_PERMITTIVITY),
    )
    device = fdtdx.Device(
        name=DEVICE_NAME,
        partial_grid_shape=DEVICE_SHAPE,
        materials={
            "lower": fdtdx.Material(permittivity=LOWER_RELATIVE_PERMITTIVITY),
            "upper": fdtdx.Material(permittivity=UPPER_RELATIVE_PERMITTIVITY),
        },
        param_transforms=[],
        partial_voxel_grid_shape=(1, 1, 1),
    )
    source = fdtdx.PointDipoleSource(
        name=SOURCE_NAME,
        partial_grid_shape=(1, 1, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=WAVELENGTH_M),
        polarization=1,
        amplitude=1.0,
    )
    detector = fdtdx.PhasorDetector(
        name=DETECTOR_NAME,
        partial_grid_shape=(4, 2, 4),
        wave_characters=(fdtdx.WaveCharacter(wavelength=WAVELENGTH_M),),
        components=("Ey",),
        reduce_volume=True,
        dtype=jnp.complex64,
        dft_subsample=1,
        plot=False,
    )
    config = fdtdx.SimulationConfig(
        time=SIMULATION_TIME_S,
        grid=fdtdx.UniformGrid(spacing=GRID_SPACING_M, center=GRID_CENTER_M),
        backend=backend,
        dtype=jnp.float32,
        gradient_config=fdtdx.GradientConfig(method="checkpointed", num_checkpoints=4),
    )
    constraints = [
        *boundary_constraints,
        waveguide.place_at_center(volume, axes=(0, 1, 2)),
        device.place_at_center(volume, axes=(0, 1, 2)),
        source.set_grid_coordinates(
            axes=(0, 1, 2),
            sides=("-", "-", "-"),
            coordinates=(28, 2, 4),
        ),
        detector.set_grid_coordinates(
            axes=(0, 1, 2),
            sides=("-", "-", "-"),
            coordinates=(68, 1, 2),
        ),
    ]
    key = jax.random.PRNGKey(RANDOM_SEED)
    objects, arrays, parameters, config, _placement = fdtdx.place_objects(
        [volume, *boundaries.values(), waveguide, device, source, detector],
        config,
        constraints,
        key=key,
    )
    if config.time_steps_total != FDTDX_TIME_STEPS:
        raise RuntimeError("locked FDTDX time-step count differs from the canonical scene")
    placed_device = objects[DEVICE_NAME]
    if placed_device.grid_slice != DEVICE_GRID_SLICE:
        raise RuntimeError("placed thermo-optic Device slice differs from the canonical scene")
    target_coordinates = tuple(
        config.resolved_grid.centers(axis)[placed_device.grid_slice[axis]] for axis in range(3)
    )
    parameters = dict(parameters)
    parameters[DEVICE_NAME] = jnp.asarray(parameters[DEVICE_NAME], dtype=jnp.float32)
    if parameters[DEVICE_NAME].shape != DEVICE_SHAPE:
        raise RuntimeError("raw thermo-optic Device parameter shape differs from the scene")
    return PreparedThermoOpticScene(
        objects=objects,
        arrays=arrays,
        parameters=parameters,
        config=config,
        placed_device=placed_device,
        target_coordinates=target_coordinates,
        material_array_shardings=fdtdx.capture_material_array_shardings(arrays),
        key=key,
    )


def scene_metadata(*, time_steps: int) -> dict[str, object]:
    """Return JSON-safe immutable scene metadata used by process-set admission."""

    if time_steps != FDTDX_TIME_STEPS:
        raise ValueError("FDTDX scene time-step count differs from the canonical contract")
    return {
        "grid_shape_xyz": list(GRID_SHAPE),
        "device_shape_xyz": list(DEVICE_SHAPE),
        "device_grid_slice": [[item.start, item.stop] for item in DEVICE_GRID_SLICE],
        "grid_spacing_m": GRID_SPACING_M,
        "grid_center_m": list(GRID_CENTER_M),
        "simulation_time_s": SIMULATION_TIME_S,
        "time_steps": time_steps,
        "wavelength_m": WAVELENGTH_M,
        "boundaries": ["periodic"] * 6,
        "source_grid_index_xyz": [28, 2, 4],
        "detector_grid_slice": [[68, 72], [1, 3], [2, 6]],
        "detector_name": DETECTOR_NAME,
        "phasor_objective": {
            "kind": "frozen_reference_normalized_quadrature",
            "scale": PHASOR_OBJECTIVE_SCALE,
        },
        "runtime_target_coordinate_tolerance": {
            "comparison": "float64_controller_vs_float32_fdtdx_cell_centers",
            "max_ulp_error": RUNTIME_TARGET_COORDINATE_MAX_ULP_ERROR,
            "max_grid_spacing_fraction_error": (RUNTIME_TARGET_COORDINATE_MAX_GRID_FRACTION_ERROR),
        },
        "device_name": DEVICE_NAME,
        "lower_relative_permittivity": LOWER_RELATIVE_PERMITTIVITY,
        "upper_relative_permittivity": UPPER_RELATIVE_PERMITTIVITY,
        "static_waveguide_relative_permittivity": STATIC_WAVEGUIDE_RELATIVE_PERMITTIVITY,
        "cladding_relative_permittivity": CLADDING_RELATIVE_PERMITTIVITY,
        "claim_boundary": (
            "bounded periodic execution-integrity scene; not a converged transmission or "
            "fabricated-device model"
        ),
    }


__all__ = [
    "DETECTOR_NAME",
    "DEVICE_GRID_SLICE",
    "DEVICE_NAME",
    "DEVICE_SHAPE",
    "FDTDX_MODULE_SHA256",
    "FDTDX_PACKAGE_VERSION",
    "FDTDX_SOURCE_DIGEST",
    "FDTDX_SOURCE_REVISION",
    "FDTDX_TIME_STEPS",
    "GRID_SHAPE",
    "LOWER_RELATIVE_PERMITTIVITY",
    "MESH_AXIS_NAME",
    "PHASOR_OBJECTIVE_SCALE",
    "RUNTIME_TARGET_COORDINATE_MAX_GRID_FRACTION_ERROR",
    "RUNTIME_TARGET_COORDINATE_MAX_ULP_ERROR",
    "UPPER_RELATIVE_PERMITTIVITY",
    "CoupledRuntimeInputs",
    "PreparedThermoOpticScene",
    "build_scene",
    "coupled_mesh_from_material_sharding",
    "device_contract",
    "scene_metadata",
    "thermo_optic_law",
    "verify_locked_fdtdx",
]
