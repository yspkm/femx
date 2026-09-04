from __future__ import annotations

import hashlib
import json
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
import pytest

fdtdx = pytest.importorskip("fdtdx")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from tests.fdtdx_mode_source_support import (  # noqa: E402
    LOCKED_FDTDX_MODE_SOURCE,
    LOCKED_FDTDX_MODE_SOURCE_FILES,
    assert_locked_fdtdx_mode_source,
)

from femx.backends.jax.port_eigenmode import JaxPortEigenmodeBackend  # noqa: E402
from femx.backends.jax.port_operator import lossless_port_coefficients  # noqa: E402
from femx.backends.protocol import ExecutionPolicy, PrepareRequest, SolveRequest  # noqa: E402
from femx.core.problem import Problem  # noqa: E402
from femx.core.solution import ConvergenceStatus  # noqa: E402
from femx.interop.fdtdx import (  # noqa: E402
    SolverFingerprint,
    build_fdtdx_mode_source_contract,
    build_yee_grid,
    build_yee_port_sampling_plan,
    make_fdtdx_mode_source,
    port_mode_solution_to_bundle,
    read_mode_bundle_hdf5,
    validate_fdtdx_mode_source,
    write_mode_bundle_hdf5,
)
from femx.meshing.gmsh import (  # noqa: E402
    GmshMeshingRequest,
    RectangularWaveguideCrossSection,
    read_gmsh_msh,
)
from femx.physics import (  # noqa: E402
    VACUUM_SPEED_OF_LIGHT_M_PER_S,
    IsotropicOpticalRegion,
    PerfectElectricBoundary,
    PortEigenmode,
)
from femx.runtime import prepare, solve  # noqa: E402

pytestmark = [
    pytest.mark.scientific,
    pytest.mark.requires_elmer,
    pytest.mark.requires_gmsh,
    pytest.mark.requires_fdtdx,
    pytest.mark.requires_jax,
    pytest.mark.slow,
]

_AUTHORIZED = ExecutionPolicy(execution_authorized=True, allow_external_process=True)
_WAVELENGTH_M = 1.55e-6
_CLADDING_INDEX = 1.444
_CORE_INDEX = 3.48
_SOURCE_Z_INDEX = 6
_DETECTOR_Z_INDEX = 24
_Z_SPACING_M = 40.0e-9


def _cell_materials(validated) -> tuple[np.ndarray, np.ndarray]:
    relative_permittivity = np.empty(validated.cells.shape[0], dtype=np.float64)
    relative_permeability = np.empty_like(relative_permittivity)
    for cell_ids, epsilon_r, mu_r in zip(
        validated.region_cells,
        validated.relative_permittivity,
        validated.relative_permeability,
        strict=True,
    ):
        relative_permittivity[cell_ids] = epsilon_r
        relative_permeability[cell_ids] = mu_r
    return relative_permittivity, relative_permeability


def _piecewise_edges(
    lower: float,
    core_lower: float,
    core_upper: float,
    upper: float,
    *,
    lower_cells: int,
    core_cells: int,
    upper_cells: int,
) -> np.ndarray:
    return np.concatenate(
        (
            np.linspace(lower, core_lower, lower_cells + 1, dtype=np.float64)[:-1],
            np.linspace(core_lower, core_upper, core_cells + 1, dtype=np.float64)[:-1],
            np.linspace(core_upper, upper, upper_cells + 1, dtype=np.float64),
        )
    )


def _fdtd_edges(validated, recipe: RectangularWaveguideCrossSection):
    coordinate_minimum = np.min(validated.coordinates, axis=0)
    coordinate_maximum = np.max(validated.coordinates, axis=0)
    x_lower = float(coordinate_minimum[0] + 1.1e-12)
    x_upper = float(coordinate_maximum[0] - 2.3e-12)
    y_lower = float(coordinate_minimum[1] + 1.7e-12)
    y_upper = float(coordinate_maximum[1] - 2.9e-12)
    x_center = 0.5 * (x_lower + x_upper)
    y_center = 0.5 * (y_lower + y_upper)
    x_edges = _piecewise_edges(
        x_lower,
        x_center - 0.5 * recipe.core_width_m,
        x_center + 0.5 * recipe.core_width_m,
        x_upper,
        lower_cells=30,
        core_cells=10,
        upper_cells=30,
    )
    y_edges = _piecewise_edges(
        y_lower,
        y_center - 0.5 * recipe.core_height_m,
        y_center + 0.5 * recipe.core_height_m,
        y_upper,
        lower_cells=24,
        core_cells=4,
        upper_cells=24,
    )
    z_edges = np.arange(-6, 31, dtype=np.float64) * _Z_SPACING_M
    return x_edges, y_edges, z_edges


def _fdtd_scene(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    *,
    frequency_hz: float,
    source: object | None,
):
    grid_shape = (x_edges.size - 1, y_edges.size - 1, z_edges.size - 1)
    volume = fdtdx.SimulationVolume(
        partial_grid_shape=grid_shape,
        material=fdtdx.Material(permittivity=_CLADDING_INDEX**2),
    )
    boundaries, boundary_constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(
            thickness=4,
            boundary_type="pml",
            override_types={face: "pec" for face in ("min_x", "max_x", "min_y", "max_y")},
        ),
        volume,
    )
    core = fdtdx.UniformMaterialObject(
        name="silicon-core",
        partial_grid_shape=(10, 4, grid_shape[2]),
        material=fdtdx.Material(permittivity=_CORE_INDEX**2),
    )
    detector = fdtdx.PhasorDetector(
        name="downstream-phasor",
        partial_grid_shape=(grid_shape[0], grid_shape[1], 1),
        wave_characters=(fdtdx.WaveCharacter(frequency=frequency_hz),),
        components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        reduce_volume=False,
        dtype=jnp.complex128,
        dft_subsample="auto",
        plot=False,
    )
    objects = [volume, *boundaries.values(), core, detector]
    constraints = [
        *boundary_constraints,
        core.place_at_center(volume, axes=(0, 1, 2)),
        detector.same_size(volume, axes=(0, 1)),
        detector.place_at_center(volume, axes=(0, 1)),
        fdtdx.RealCoordinateConstraint(
            object=detector.name,
            axes=(2,),
            sides=("-",),
            coordinates=(float(z_edges[_DETECTOR_Z_INDEX]),),
        ),
    ]
    if source is not None:
        objects.append(source)
        constraints.extend(
            (
                source.same_size(volume, axes=(0, 1)),
                source.place_at_center(volume, axes=(0, 1)),
                fdtdx.RealCoordinateConstraint(
                    object=source.name,
                    axes=(2,),
                    sides=("-",),
                    coordinates=(float(z_edges[_SOURCE_Z_INDEX]),),
                ),
            )
        )
    config = fdtdx.SimulationConfig(
        time=30.0e-15,
        grid=fdtdx.RectilinearGrid(
            x_edges=jnp.asarray(x_edges),
            y_edges=jnp.asarray(y_edges),
            z_edges=jnp.asarray(z_edges),
        ),
        backend="cpu",
        dtype=jnp.float64,
    )
    return objects, constraints, config


def _run_fdtd_source(
    bundle,
    expected_inverse_permittivity: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
):
    contract = build_fdtdx_mode_source_contract(
        bundle,
        source_name="fem-port",
        expected_inverse_permittivity=expected_inverse_permittivity,
        expected_inverse_permeability=np.asarray(1.0, dtype=np.float64),
        fdtdx=LOCKED_FDTDX_MODE_SOURCE,
    )
    source = make_fdtdx_mode_source(
        bundle,
        contract,
        verified_fingerprint=LOCKED_FDTDX_MODE_SOURCE,
    )
    objects, constraints, config = _fdtd_scene(
        x_edges,
        y_edges,
        z_edges,
        frequency_hz=bundle.frequency_hz,
        source=source,
    )
    key = jax.random.PRNGKey(211)
    placed_objects, arrays, parameters, config, _info = fdtdx.place_objects(
        objects,
        config,
        constraints,
        key=key,
    )
    placed_source = placed_objects[contract.source_name]
    validate_fdtdx_mode_source(placed_source, bundle, contract)
    np.testing.assert_array_equal(
        np.asarray(arrays.inv_permittivities[:, :, :, _SOURCE_Z_INDEX : _SOURCE_Z_INDEX + 1]),
        expected_inverse_permittivity,
    )
    arrays, placed_objects, _apply_info = fdtdx.apply_params(
        arrays=arrays,
        objects=placed_objects,
        params=parameters,
        key=key,
    )
    _step, final_arrays = fdtdx.run_fdtd(
        arrays=arrays,
        objects=placed_objects,
        config=config,
        key=key,
        show_progress=False,
    )
    return np.asarray(final_arrays.detector_states["downstream-phasor"]["phasor"]), contract


def test_locked_elmer_and_jax_modes_drive_matching_fdtdx_downstream_fields(
    locked_gmsh_runner,
    locked_elmer_port_backend,
    tmp_path: Path,
) -> None:
    """Compare the complete same-mesh FEM-to-FDTD source path at one complex detector."""

    assert_locked_fdtdx_mode_source()
    recipe = RectangularWaveguideCrossSection(
        cladding_mesh_size_m=0.44e-6,
        core_mesh_size_m=0.09e-6,
    )
    meshing_directory = tmp_path / "meshing"
    meshing_directory.mkdir()
    (meshing_directory / "waveguide.geo").write_text(recipe.render_geo(), encoding="utf-8")
    meshing = locked_gmsh_runner.run(
        GmshMeshingRequest("waveguide.geo"),
        working_directory=meshing_directory,
        policy=_AUTHORIZED,
    )
    assert meshing.process_succeeded, meshing.stderr
    imported = read_gmsh_msh(
        meshing_directory / "mesh.msh",
        coordinate_scale_to_m=recipe.coordinate_scale_to_m,
    )
    frequency_hz = VACUUM_SPEED_OF_LIGHT_M_PER_S / _WAVELENGTH_M
    problem = Problem(
        "locked-elmer-jax-fdtdx-silicon-port",
        imported.mesh,
        PortEigenmode(
            regions=(
                IsotropicOpticalRegion("cladding", _CLADDING_INDEX**2),
                IsotropicOpticalRegion("core", _CORE_INDEX**2),
            ),
            perfect_electric_boundaries=tuple(
                PerfectElectricBoundary(name) for name in ("bottom", "right", "top", "left")
            ),
            frequency_hz=frequency_hz,
            eigenmode_count=8,
            selected_mode_index=0,
            target_power_w=1.0,
        ),
    )

    elmer_run_directory = tmp_path / "elmer-attempt-001"
    elmer_prepared = prepare(
        problem,
        locked_elmer_port_backend,
        request=PrepareRequest(run_directory=elmer_run_directory),
    )
    elmer_solution = solve(
        elmer_prepared,
        locked_elmer_port_backend,
        request=SolveRequest(run_directory=elmer_run_directory, policy=_AUTHORIZED),
    )
    jax_backend = JaxPortEigenmodeBackend(relative_residual_tolerance=1.0e-12)
    jax_solution = solve(prepare(problem, jax_backend), jax_backend)
    assert elmer_solution.convergence.status is ConvergenceStatus.CONVERGED
    assert jax_solution.convergence.status is ConvergenceStatus.CONVERGED

    validated = elmer_prepared.payload.validated
    relative_permittivity, relative_permeability = _cell_materials(validated)
    _, cell_reluctivity = lossless_port_coefficients(
        jnp.asarray(relative_permittivity),
        jnp.asarray(relative_permeability),
    )
    x_edges, y_edges, z_edges = _fdtd_edges(validated, recipe)
    source_grid = build_yee_grid((x_edges, y_edges, z_edges[_SOURCE_Z_INDEX : _SOURCE_Z_INDEX + 2]))
    transfer_plan = build_yee_port_sampling_plan(
        validated.coordinates,
        validated.cells,
        validated.edge_signs,
        source_grid,
    )
    assert transfer_plan.target_grid.shape == (70, 52, 1)
    assert transfer_plan.ambiguous_target_point_count == 0
    config_sha256 = hashlib.sha256(
        json.dumps(
            problem.physics.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    elmer_bundle = port_mode_solution_to_bundle(
        elmer_solution,
        transfer_plan,
        cell_reluctivity,
        frequency_hz=frequency_hz,
        solver=SolverFingerprint(
            name=elmer_solution.backend_name,
            version=elmer_solution.backend_version,
            config_sha256=str(elmer_solution.metadata["input_sif_sha256"]),
            mesh_sha256=transfer_plan.source_mesh_sha256,
            source_revision=str(elmer_solution.metadata["elmer_source_commit"]),
        ),
        fdtdx=LOCKED_FDTDX_MODE_SOURCE,
    )
    jax_bundle = port_mode_solution_to_bundle(
        jax_solution,
        transfer_plan,
        cell_reluctivity,
        frequency_hz=frequency_hz,
        solver=SolverFingerprint(
            name=jax_solution.backend_name,
            version=jax_solution.backend_version,
            config_sha256=config_sha256,
            mesh_sha256=transfer_plan.source_mesh_sha256,
        ),
        fdtdx=LOCKED_FDTDX_MODE_SOURCE,
    )

    artifact_root = tmp_path / "mode-artifacts"
    artifact_root.mkdir()
    elmer_artifact = write_mode_bundle_hdf5(artifact_root, "elmer-mode.h5", elmer_bundle)
    jax_artifact = write_mode_bundle_hdf5(artifact_root, "jax-mode.h5", jax_bundle)
    elmer_bundle = read_mode_bundle_hdf5(artifact_root, elmer_artifact.reference).bundle
    jax_bundle = read_mode_bundle_hdf5(artifact_root, jax_artifact.reference).bundle
    electric_relative_error = float(
        np.linalg.norm(jax_bundle.electric.values - elmer_bundle.electric.values)
        / np.linalg.norm(elmer_bundle.electric.values)
    )
    magnetic_relative_error = float(
        np.linalg.norm(jax_bundle.magnetic.values - elmer_bundle.magnetic.values)
        / np.linalg.norm(elmer_bundle.magnetic.values)
    )
    assert electric_relative_error <= 1.0e-10
    assert magnetic_relative_error <= 1.0e-10
    assert elmer_bundle.transfer.relative_pre_correction_power_error is not None
    assert jax_bundle.transfer.relative_pre_correction_power_error is not None
    assert elmer_bundle.transfer.relative_pre_correction_power_error <= 0.06
    assert jax_bundle.transfer.relative_pre_correction_power_error <= 0.06
    assert elmer_bundle.transfer.relative_power_error <= 2.0e-14
    assert jax_bundle.transfer.relative_power_error <= 2.0e-14
    assert elmer_bundle.transfer.pre_correction_power_watts == pytest.approx(
        jax_bundle.transfer.pre_correction_power_watts,
        rel=1.0e-12,
    )

    base_objects, base_constraints, base_config = _fdtd_scene(
        x_edges,
        y_edges,
        z_edges,
        frequency_hz=frequency_hz,
        source=None,
    )
    _base_placed, base_arrays, _base_parameters, _base_config, _base_info = fdtdx.place_objects(
        base_objects,
        base_config,
        base_constraints,
        key=jax.random.PRNGKey(211),
    )
    expected_inverse_permittivity = np.asarray(
        base_arrays.inv_permittivities[:, :, :, _SOURCE_Z_INDEX : _SOURCE_Z_INDEX + 1]
    )
    assert expected_inverse_permittivity.shape == (1, 70, 52, 1)
    source_epsilon = 1.0 / expected_inverse_permittivity[0, :, :, 0]
    assert np.count_nonzero(source_epsilon == _CORE_INDEX**2) == 10 * 4
    assert np.count_nonzero(source_epsilon == _CLADDING_INDEX**2) == 70 * 52 - 10 * 4

    elmer_phasor, elmer_contract = _run_fdtd_source(
        elmer_bundle,
        expected_inverse_permittivity,
        x_edges,
        y_edges,
        z_edges,
    )
    jax_phasor, jax_contract = _run_fdtd_source(
        jax_bundle,
        expected_inverse_permittivity,
        x_edges,
        y_edges,
        z_edges,
    )
    assert np.isfinite(elmer_phasor).all()
    assert np.isfinite(jax_phasor).all()
    assert float(np.linalg.norm(elmer_phasor)) > 0.0
    detector_relative_error = float(
        np.linalg.norm(jax_phasor - elmer_phasor) / np.linalg.norm(elmer_phasor)
    )
    assert detector_relative_error <= 1.0e-9

    evidence = {
        "schema_version": "femx.elmer-jax-fdtdx-source-parity/v1",
        "claim": "same-scene downstream complex-field parity",
        "not_claimed": ["absolute transmission convergence", "S-parameters", "TPU execution"],
        "mesh_sha256": transfer_plan.source_mesh_sha256,
        "gmsh": {
            "version": meshing.identity.version,
            "executable_sha256": meshing.identity.executable_sha256,
            "geometry_sha256": meshing.geometry_sha256,
            "mesh_file_sha256": meshing.mesh_sha256,
        },
        "elmer": {
            "version": elmer_solution.backend_version,
            "source_revision": elmer_solution.metadata["elmer_source_commit"],
            "source_digest": elmer_solution.metadata["elmer_source_digest"],
            "executable_sha256": elmer_solution.metadata["elmer_executable_sha256"],
            "em_port_sha256": elmer_solution.metadata["elmer_em_port_sha256"],
            "result_output_sha256": elmer_solution.metadata["elmer_result_output_sha256"],
            "save_data_sha256": elmer_solution.metadata["elmer_save_data_sha256"],
            "mode_artifact": elmer_artifact.reference.to_dict(),
            "mode_content_sha256": elmer_artifact.content_sha256,
            "source_contract_sha256": elmer_contract.sha256,
        },
        "jax": {
            "version": jax_solution.backend_version,
            "jax_package_version": package_version("jax"),
            "jaxlib_package_version": package_version("jaxlib"),
            "platform": jax.default_backend(),
            "precision": jax_solution.metadata["precision"],
            "mode_artifact": jax_artifact.reference.to_dict(),
            "mode_content_sha256": jax_artifact.content_sha256,
            "source_contract_sha256": jax_contract.sha256,
        },
        "fdtdx": {
            "package_version": LOCKED_FDTDX_MODE_SOURCE.package_version,
            "source_revision": LOCKED_FDTDX_MODE_SOURCE.source_revision,
            "source_digest": LOCKED_FDTDX_MODE_SOURCE.source_digest,
            "locked_module_sha256": LOCKED_FDTDX_MODE_SOURCE_FILES,
            "grid_shape_xyz": [70, 52, 36],
            "source_z_m": float(z_edges[_SOURCE_Z_INDEX]),
            "detector_z_m": float(z_edges[_DETECTOR_Z_INDEX]),
            "simulation_time_s": 30.0e-15,
        },
        "errors": {
            "source_electric_relative_l2": electric_relative_error,
            "source_magnetic_relative_l2": magnetic_relative_error,
            "downstream_phasor_relative_l2": detector_relative_error,
            "elmer_pre_correction_power_relative": (
                elmer_bundle.transfer.relative_pre_correction_power_error
            ),
            "jax_pre_correction_power_relative": (
                jax_bundle.transfer.relative_pre_correction_power_error
            ),
        },
    }
    (tmp_path / "fdtdx-source-parity-evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
