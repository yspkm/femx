from __future__ import annotations

import hashlib
import math
from importlib import import_module
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np

from femx.core.axes import Axis, AxisDirection, Direction
from femx.interop.fdtdx import (
    FDTDXFingerprint,
    FieldRepresentation,
    MagneticFieldConvention,
    ModeBundle,
    ModeNormalization,
    SolverFingerprint,
    TransferReport,
    YeeFieldKind,
    YeeVectorField,
    build_yee_grid,
)
from femx.physics import VACUUM_SPEED_OF_LIGHT_M_PER_S

LOCKED_FDTDX_MODE_SOURCE = FDTDXFingerprint(
    package_version="0.6.2",
    source_revision="81a58da9cde4a4ff822f835b63597c0d0d8ba978",
    source_digest="c881f0cf32b4272dc10e0acee716bfda3cf2d0f11973c23d31f29b11f3dce01c",
)
LOCKED_FDTDX_MODE_SOURCE_FILES = {
    "fdtdx.core.grid": "d24739b9229ad8c61a57e4f688e6224eae63a680ff6554ddd7a5ef765edab6dd",
    "fdtdx.objects.object": "24c986b9fa73bf474bce9fefc2145436654be4758e83dbcaf6fb955b7eb8557f",
    "fdtdx.objects.sources.custom_mode": (
        "0c5925a784da33f8d8236a874d4759d4ebe6df29317dcc1ce68877b4a4036df5"
    ),
    "fdtdx.objects.sources.tfsf": (
        "bd270995bffd174c7014adf9a02c7648134547c3bab7a294570e0a179326e611"
    ),
    "fdtdx.fdtd.wrapper": "97d562e0a33eeeccd6ce42d12cf8dc29f1e9fd071e561b5b009ac5da8ece7384",
}


def assert_locked_fdtdx_mode_source() -> None:
    """Assert the package and source files behind the imported-mode witness."""

    assert package_version("fdtdx") == LOCKED_FDTDX_MODE_SOURCE.package_version
    actual_hashes = {}
    for module_name in LOCKED_FDTDX_MODE_SOURCE_FILES:
        module_path = Path(str(import_module(module_name).__file__)).resolve()
        actual_hashes[module_name] = hashlib.sha256(module_path.read_bytes()).hexdigest()
    assert actual_hashes == LOCKED_FDTDX_MODE_SOURCE_FILES


def uniform_mode_bundle(
    *,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    source_z_edges: np.ndarray,
    relative_permittivity: float,
) -> ModeBundle:
    """Return an analytic one-watt +z plane mode on one exact Yee plane."""

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


__all__ = [
    "LOCKED_FDTDX_MODE_SOURCE",
    "LOCKED_FDTDX_MODE_SOURCE_FILES",
    "assert_locked_fdtdx_mode_source",
    "uniform_mode_bundle",
]
