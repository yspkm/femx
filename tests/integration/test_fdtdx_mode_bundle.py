from __future__ import annotations

import hashlib
import math
from importlib import import_module
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
import pytest

fdtdx = pytest.importorskip("fdtdx")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from femx.core.capabilities import FunctionSpaceFamily  # noqa: E402
from femx.core.solution import (  # noqa: E402
    ConvergenceReport,
    ConvergenceStatus,
    Field,
    Solution,
)
from femx.interop.fdtdx import (  # noqa: E402
    FDTDXFingerprint,
    SolverFingerprint,
    build_yee_grid,
    build_yee_port_sampling_plan,
    make_fdtdx_mode_function,
    port_mode_solution_to_bundle,
    read_mode_bundle_hdf5,
    write_mode_bundle_hdf5,
)
from femx.mesh import FunctionSpace  # noqa: E402
from femx.physics import (  # noqa: E402
    PORT_LONGITUDINAL_POTENTIAL_FIELD,
    PORT_LONGITUDINAL_POTENTIAL_UNIT,
    PORT_TRANSVERSE_ELECTRIC_DOF_UNIT,
    PORT_TRANSVERSE_ELECTRIC_FIELD,
    VACUUM_PERMEABILITY_H_PER_M,
    VACUUM_SPEED_OF_LIGHT_M_PER_S,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_fdtdx,
    pytest.mark.requires_jax,
]

_LOCKED_FDTDX_FILES = {
    "fdtdx.constants": "063ffcad38916b48876f917d3312e7cd111eb72b363bca72ae7d7d8ff12ee97b",
    "fdtdx.core.grid": "d24739b9229ad8c61a57e4f688e6224eae63a680ff6554ddd7a5ef765edab6dd",
    "fdtdx.objects.detectors.mode": (
        "f6a37c59a9d63cae4cba430e5c4b9cda23c6ddee94724ae2623b6c8a0b2e5f81"
    ),
}


def _module_sha256(name: str) -> str:
    module_path = Path(str(import_module(name).__file__)).resolve()
    return hashlib.sha256(module_path.read_bytes()).hexdigest()


def test_locked_fdtdx_custom_detector_consumes_exact_yee_mode_bundle(tmp_path: Path) -> None:
    assert package_version("fdtdx") == "0.6.2"
    assert {name: _module_sha256(name) for name in _LOCKED_FDTDX_FILES} == _LOCKED_FDTDX_FILES
    frequency_hz = VACUUM_SPEED_OF_LIGHT_M_PER_S / 1.55e-6
    beta_per_m = 4.0e6
    coordinates = np.asarray(((0.0, 0.0), (2.0e-6, 0.0), (0.0, 2.0e-6)))
    cells = np.asarray(((0, 1, 2),), dtype=np.int64)
    signs = np.asarray(((1, 1, -1),), dtype=np.int8)
    grid = build_yee_grid(
        (
            np.asarray((0.2e-6, 0.4e-6, 0.6e-6)),
            np.asarray((0.2e-6, 0.4e-6, 0.6e-6)),
            np.asarray((0.0, 20.0e-9)),
        )
    )
    plan = build_yee_port_sampling_plan(coordinates, cells, signs, grid)
    scalar = jnp.full((3,), 1j * beta_per_m, dtype=jnp.complex128)
    edge = jnp.asarray((4.0e-6, 6.0e-6, 2.0e-6), dtype=jnp.complex128)
    solution = Solution(
        backend_name="synthetic-mixed-port",
        backend_version="1",
        fields={
            PORT_LONGITUDINAL_POTENTIAL_FIELD: Field(
                PORT_LONGITUDINAL_POTENTIAL_FIELD,
                scalar,
                PORT_LONGITUDINAL_POTENTIAL_UNIT,
                FunctionSpace(FunctionSpaceFamily.H1, order=1),
            ),
            PORT_TRANSVERSE_ELECTRIC_FIELD: Field(
                PORT_TRANSVERSE_ELECTRIC_FIELD,
                edge,
                PORT_TRANSVERSE_ELECTRIC_DOF_UNIT,
                FunctionSpace(FunctionSpaceFamily.HCURL, order=1, value_shape=(2,)),
            ),
        },
        observables={
            "propagation_constant_rad_per_m": beta_per_m + 0.0j,
            "effective_index": beta_per_m
            * VACUUM_SPEED_OF_LIGHT_M_PER_S
            / (2.0 * math.pi * frequency_hz),
            "target_forward_power_W": 1.0,
        },
        convergence=ConvergenceReport(ConvergenceStatus.CONVERGED),
    )
    bundle = port_mode_solution_to_bundle(
        solution,
        plan,
        jnp.asarray((1.0 / VACUUM_PERMEABILITY_H_PER_M,)),
        frequency_hz=frequency_hz,
        solver=SolverFingerprint(
            "synthetic-mixed-port",
            "1",
            "a" * 64,
            plan.source_mesh_sha256,
        ),
        fdtdx=FDTDXFingerprint(
            "0.6.2",
            "eaab78a42cd1351b7f447f312fa50c9febfe4b99",
            "cf7bf29a1aa22911e1b3a523a65cf267079562d408238f45c19bb9cd01b51d70",
        ),
    )
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    written = write_mode_bundle_hdf5(artifact_root, "modes/port.h5", bundle)
    bundle = read_mode_bundle_hdf5(artifact_root, written.reference).bundle

    config = fdtdx.SimulationConfig(
        time=10.0e-15,
        grid=fdtdx.RectilinearGrid.custom(
            x_edges=jnp.asarray(grid.edge_coordinates[0]),
            y_edges=jnp.asarray(grid.edge_coordinates[1]),
            z_edges=jnp.asarray(grid.edge_coordinates[2]),
        ),
        backend="cpu",
        dtype=jnp.float64,
    )
    detector = fdtdx.CustomModeOverlapDetector(
        wave_characters=(
            fdtdx.WaveCharacter(
                wavelength=VACUUM_SPEED_OF_LIGHT_M_PER_S / frequency_hz,
            ),
        ),
        mode_function=make_fdtdx_mode_function(bundle),
        normalize=False,
    )
    key = jax.random.PRNGKey(11)
    detector = detector.place_on_grid(((0, 2), (0, 2), (0, 1)), config, key)
    detector = detector.apply(
        key,
        jnp.ones((3, 2, 2, 1), dtype=jnp.float64),
        1.0,
    )

    np.testing.assert_array_equal(np.asarray(detector._mode_E[0]), bundle.electric.values)
    np.testing.assert_array_equal(np.asarray(detector._mode_H[0]), bundle.magnetic.values)
    assert detector.propagation_axis == 2
