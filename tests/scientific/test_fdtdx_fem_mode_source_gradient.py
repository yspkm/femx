from __future__ import annotations

import hashlib
import json
import math
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
    assert_locked_fdtdx_mode_source,
)

from femx.backends.jax.port_eigenmode import (  # noqa: E402
    JaxPortEigenmodeBackend,
    PreparedJaxPortEigenmode,
)
from femx.backends.protocol import ExecutionPolicy, SolveRequest  # noqa: E402
from femx.core.capabilities import GradientMethod  # noqa: E402
from femx.core.parameters import (  # noqa: E402
    ParameterReference,
    ParameterRole,
    ParameterSchema,
    ParameterSpec,
    ParameterValues,
)
from femx.core.problem import Problem  # noqa: E402
from femx.interop.fdtdx import (  # noqa: E402
    SolverFingerprint,
    build_fdtdx_dynamic_mode_source_contract,
    build_fdtdx_mode_source_contract,
    build_yee_grid,
    build_yee_port_sampling_plan,
    make_fdtdx_dynamic_mode_source,
    port_mode_solution_to_bundle,
    sample_port_mode_to_yee,
    validate_fdtdx_mode_source,
    with_fdtdx_dynamic_mode_profile,
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
    pytest.mark.requires_gmsh,
    pytest.mark.requires_fdtdx,
    pytest.mark.requires_jax,
    pytest.mark.slow,
]

_AUTHORIZED = ExecutionPolicy(execution_authorized=True, allow_external_process=True)
_WAVELENGTH_M = 1.55e-6
_CLADDING_INDEX = 1.444
_CORE_INDEX = 3.48
_SOURCE_Z_INDEX = 3
_DETECTOR_Z_INDEX = 14
_Z_SPACING_M = 50.0e-9


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


def _fdtd_edges(
    payload: PreparedJaxPortEigenmode,
    recipe: RectangularWaveguideCrossSection,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minimum = np.min(payload.validated.coordinates, axis=0)
    maximum = np.max(payload.validated.coordinates, axis=0)
    x_lower = float(minimum[0] + 1.1e-12)
    x_upper = float(maximum[0] - 2.3e-12)
    y_lower = float(minimum[1] + 1.7e-12)
    y_upper = float(maximum[1] - 2.9e-12)
    x_center = 0.5 * (x_lower + x_upper)
    y_center = 0.5 * (y_lower + y_upper)
    x_edges = _piecewise_edges(
        x_lower,
        x_center - 0.5 * recipe.core_width_m,
        x_center + 0.5 * recipe.core_width_m,
        x_upper,
        lower_cells=5,
        core_cells=4,
        upper_cells=5,
    )
    y_edges = _piecewise_edges(
        y_lower,
        y_center - 0.5 * recipe.core_height_m,
        y_center + 0.5 * recipe.core_height_m,
        y_upper,
        lower_cells=4,
        core_cells=2,
        upper_cells=4,
    )
    z_edges = np.arange(-3, 16, dtype=np.float64) * _Z_SPACING_M
    return x_edges, y_edges, z_edges


def _fdtd_scene(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    *,
    frequency_hz: float,
    source: object | None,
    checkpointed: bool,
):
    grid_shape = (x_edges.size - 1, y_edges.size - 1, z_edges.size - 1)
    volume = fdtdx.SimulationVolume(
        partial_grid_shape=grid_shape,
        material=fdtdx.Material(permittivity=_CLADDING_INDEX**2),
    )
    boundaries, boundary_constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(
            thickness=2,
            boundary_type="pml",
            override_types={face: "pec" for face in ("min_x", "max_x", "min_y", "max_y")},
        ),
        volume,
    )
    core = fdtdx.UniformMaterialObject(
        name="silicon-core",
        partial_grid_shape=(4, 2, grid_shape[2]),
        material=fdtdx.Material(permittivity=_CORE_INDEX**2),
    )
    detector = fdtdx.PhasorDetector(
        name="downstream-phasor",
        partial_grid_shape=(grid_shape[0], grid_shape[1], 1),
        wave_characters=(fdtdx.WaveCharacter(frequency=frequency_hz),),
        components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        reduce_volume=True,
        dtype=jnp.complex128,
        dft_subsample=1,
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
    gradient_config = (
        fdtdx.GradientConfig(method="checkpointed", num_checkpoints=4) if checkpointed else None
    )
    config = fdtdx.SimulationConfig(
        time=15.0e-15,
        grid=fdtdx.RectilinearGrid(
            x_edges=jnp.asarray(x_edges),
            y_edges=jnp.asarray(y_edges),
            z_edges=jnp.asarray(z_edges),
        ),
        backend="cpu",
        dtype=jnp.float64,
        gradient_config=gradient_config,
    )
    return objects, constraints, config


def test_silicon_fem_mode_profile_reaches_checkpointed_fdtd_detector(
    locked_gmsh_runner,
    tmp_path: Path,
) -> None:
    """Differentiate a Si/SiO2 FEM mode source through an actual FDTDX run.

    The FDTD material scene is intentionally frozen at the baseline. This isolates the
    eigenmode/profile contribution and does not claim the total derivative of a device
    whose three-dimensional material distribution changes with the FEM parameter.
    """

    assert_locked_fdtdx_mode_source()
    recipe = RectangularWaveguideCrossSection(
        cladding_mesh_size_m=0.44e-6,
        core_mesh_size_m=0.09e-6,
    )
    meshing_directory = tmp_path / "meshing"
    meshing_directory.mkdir()
    (meshing_directory / "waveguide.geo").write_text(
        recipe.render_geo(),
        encoding="utf-8",
    )
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
    core_permittivity = _CORE_INDEX**2
    parameter_schema = ParameterSchema(
        (
            ParameterSpec(
                "core_relative_permittivity",
                unit="1",
                role=ParameterRole.DESIGN,
                lower_bound=8.0,
                upper_bound=16.0,
            ),
        )
    )
    problem = Problem(
        "silicon-fem-to-dynamic-fdtdx-source",
        imported.mesh,
        PortEigenmode(
            regions=(
                IsotropicOpticalRegion("cladding", _CLADDING_INDEX**2),
                IsotropicOpticalRegion(
                    "core",
                    ParameterReference("core_relative_permittivity"),
                ),
            ),
            perfect_electric_boundaries=tuple(
                PerfectElectricBoundary(name) for name in ("bottom", "right", "top", "left")
            ),
            frequency_hz=frequency_hz,
            eigenmode_count=8,
            selected_mode_index=0,
            target_power_w=1.0,
            gradient_method=GradientMethod.ADJOINT,
        ),
        parameters=parameter_schema,
    )
    parameters = ParameterValues({"core_relative_permittivity": core_permittivity})
    backend = JaxPortEigenmodeBackend(relative_residual_tolerance=1.0e-12)
    prepared = prepare(problem, backend)
    assert isinstance(prepared.payload, PreparedJaxPortEigenmode)
    bound = backend.bind_differentiable(prepared, parameters)
    baseline_solution = solve(
        prepared,
        backend,
        request=SolveRequest(parameters=parameters),
    )
    x_edges, y_edges, z_edges = _fdtd_edges(prepared.payload, recipe)
    source_grid = build_yee_grid((x_edges, y_edges, z_edges[_SOURCE_Z_INDEX : _SOURCE_Z_INDEX + 2]))
    transfer_plan = build_yee_port_sampling_plan(
        prepared.payload.validated.coordinates,
        prepared.payload.validated.cells,
        prepared.payload.validated.edge_signs,
        source_grid,
    )
    baseline_mode = bound.mode(bound.initial_values)
    config_sha256 = hashlib.sha256(
        json.dumps(
            problem.physics.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    bundle = port_mode_solution_to_bundle(
        baseline_solution,
        transfer_plan,
        np.asarray(baseline_mode.cell_reluctivity_per_henry_m),
        frequency_hz=frequency_hz,
        solver=SolverFingerprint(
            name=baseline_solution.backend_name,
            version=baseline_solution.backend_version,
            config_sha256=config_sha256,
            mesh_sha256=transfer_plan.source_mesh_sha256,
        ),
        fdtdx=LOCKED_FDTDX_MODE_SOURCE,
    )

    base_objects, base_constraints, base_config = _fdtd_scene(
        x_edges,
        y_edges,
        z_edges,
        frequency_hz=frequency_hz,
        source=None,
        checkpointed=False,
    )
    key = jax.random.PRNGKey(307)
    _base_objects, base_arrays, _base_parameters, _base_config, _base_info = fdtdx.place_objects(
        base_objects,
        base_config,
        base_constraints,
        key=key,
    )
    expected_inverse_permittivity = np.asarray(
        base_arrays.inv_permittivities[:, :, :, _SOURCE_Z_INDEX : _SOURCE_Z_INDEX + 1]
    )
    baseline_contract = build_fdtdx_mode_source_contract(
        bundle,
        source_name="dynamic-fem-port",
        expected_inverse_permittivity=expected_inverse_permittivity,
        expected_inverse_permeability=np.asarray(1.0, dtype=np.float64),
        fdtdx=LOCKED_FDTDX_MODE_SOURCE,
    )
    dynamic_contract = build_fdtdx_dynamic_mode_source_contract(
        bundle,
        baseline_contract,
        parameter_names=bound.parameter_names,
        parameter_units=bound.parameter_units,
    )
    source = make_fdtdx_dynamic_mode_source(
        bundle,
        dynamic_contract,
        verified_fingerprint=LOCKED_FDTDX_MODE_SOURCE,
    )
    objects, constraints, config = _fdtd_scene(
        x_edges,
        y_edges,
        z_edges,
        frequency_hz=frequency_hz,
        source=source,
        checkpointed=True,
    )
    placed_objects, arrays, fdtd_parameters, config, _info = fdtdx.place_objects(
        objects,
        config,
        constraints,
        key=key,
    )
    arrays, placed_objects, _apply_info = fdtdx.apply_params(
        arrays=arrays,
        objects=placed_objects,
        params=fdtd_parameters,
        key=key,
    )
    validate_fdtdx_mode_source(
        placed_objects[baseline_contract.source_name],
        bundle,
        baseline_contract,
    )
    source_index = next(
        index
        for index, item in enumerate(placed_objects.object_list)
        if item.name == baseline_contract.source_name
    )

    def objective(active: jax.Array) -> jax.Array:
        mode = bound.mode(active)
        samples = sample_port_mode_to_yee(
            transfer_plan,
            mode.scalar_coefficients,
            mode.edge_coefficients,
            mode.propagation_constant_per_m,
            mode.cell_reluctivity_per_henry_m,
            bound.angular_frequency_rad_per_s,
            bound.target_power_w,
        )
        dynamic_source = with_fdtdx_dynamic_mode_profile(
            placed_objects[baseline_contract.source_name],
            dynamic_contract,
            electric_v_per_m=samples.electric_v_per_m,
            magnetic_eta0_v_per_m=samples.magnetic_eta0_v_per_m,
            effective_index=mode.effective_index,
        )
        object_list = list(placed_objects.object_list)
        object_list[source_index] = dynamic_source
        dynamic_objects = placed_objects.aset("object_list", object_list)
        _step, final_arrays = fdtdx.run_fdtd(
            arrays=arrays,
            objects=dynamic_objects,
            config=config,
            key=key,
            show_progress=False,
        )
        phasor = final_arrays.detector_states["downstream-phasor"]["phasor"]
        signal = jnp.sum(phasor)
        return jnp.real(signal * jnp.exp(0.37j))

    initial = bound.initial_values
    value, gradient = jax.jit(jax.value_and_grad(objective))(initial)
    step = 2.0e-3
    central_difference = (
        float(objective(initial.at[0].add(step))) - float(objective(initial.at[0].add(-step)))
    ) / (2.0 * step)

    assert transfer_plan.target_grid.shape == (14, 10, 1)
    assert transfer_plan.ambiguous_target_point_count == 0
    assert bool(baseline_mode.is_valid)
    assert math.isfinite(float(value))
    assert math.isfinite(float(gradient[0]))
    assert abs(float(gradient[0])) > 0.0
    assert float(gradient[0]) == pytest.approx(central_difference, rel=2.0e-5)

    relative_error = abs(float(gradient[0]) - central_difference) / abs(central_difference)
    evidence = {
        "schema_version": "femx.fdtdx.fem-mode-source-gradient-evidence/v1",
        "claim_scope": (
            "Si/SiO2 simple-mode source-profile derivative through exact Yee sampling "
            "and checkpointed FDTDX"
        ),
        "not_claimed": [
            "total derivative of a changing three-dimensional material scene",
            "open-boundary port convergence",
            "S-parameters",
            "GPU or TPU execution",
        ],
        "fixed_fdtd_material_scene": True,
        "parameter": bound.parameter_names[0],
        "parameter_value": float(initial[0]),
        "central_difference_step": step,
        "objective_value": float(value),
        "jax_reverse_derivative": float(gradient[0]),
        "central_difference_derivative": central_difference,
        "relative_error": relative_error,
        "fem": {
            "node_count": int(prepared.payload.validated.coordinates.shape[0]),
            "triangle_count": int(prepared.payload.validated.cells.shape[0]),
            "effective_index": float(baseline_mode.effective_index),
            "propagation_constant_per_m": float(baseline_mode.propagation_constant_per_m),
            "simple_mode_relative_gap": float(bound.baseline_diagnostics.relative_eigenvalue_gap),
            "simple_mode_relative_residual": float(bound.baseline_diagnostics.relative_residual),
        },
        "transfer": {
            "operator_sha256": transfer_plan.operator_sha256,
            "source_mesh_sha256": transfer_plan.source_mesh_sha256,
            "grid_shape_xyz": list(transfer_plan.target_grid.shape),
            "ambiguous_target_point_count": (transfer_plan.ambiguous_target_point_count),
            "baseline_source_contract_sha256": baseline_contract.sha256,
            "dynamic_source_contract_sha256": dynamic_contract.sha256,
        },
        "fdtdx": {
            "package_version": LOCKED_FDTDX_MODE_SOURCE.package_version,
            "source_revision": LOCKED_FDTDX_MODE_SOURCE.source_revision,
            "source_digest": LOCKED_FDTDX_MODE_SOURCE.source_digest,
            "gradient_method": "checkpointed",
            "checkpoint_count": 4,
            "grid_shape_xyz": [
                x_edges.size - 1,
                y_edges.size - 1,
                z_edges.size - 1,
            ],
            "simulation_time_s": 15.0e-15,
            "source_z_m": float(z_edges[_SOURCE_Z_INDEX]),
            "detector_z_m": float(z_edges[_DETECTOR_Z_INDEX]),
        },
        "runtime": {
            "jax_version": package_version("jax"),
            "jaxlib_version": package_version("jaxlib"),
            "platform": jax.default_backend(),
            "precision": "float64/complex128",
        },
        "gmsh": {
            "version": meshing.identity.version,
            "executable_sha256": meshing.identity.executable_sha256,
            "geometry_sha256": meshing.geometry_sha256,
            "mesh_sha256": meshing.mesh_sha256,
        },
    }
    (tmp_path / "fdtdx-fem-mode-source-gradient-evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
