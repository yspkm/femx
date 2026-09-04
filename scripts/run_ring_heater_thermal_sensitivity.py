"""Run the bounded CPU/Gmsh ring-heater thermal-envelope sensitivity study."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from importlib.metadata import version as package_version
from pathlib import Path

if os.environ.get("JAX_PLATFORMS") != "cpu":
    raise RuntimeError("ring-heater thermal sensitivity requires explicit JAX_PLATFORMS=cpu")

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from femx.applications import (  # noqa: E402
    RingHeaterThermalSensitivityCase,
    prepare_ring_heater_thermal_sensitivity_plan,
    project_public_ring_heater_current,
    public_ring_heater_operating_point,
    ring_heater_thermal_sensitivity_cases,
)
from femx.artifacts import sha256_file  # noqa: E402
from femx.backends.jax.scalar_cg import (  # noqa: E402
    ScalarH1CGPolicy,
    ScalarH1JacobiPolicy,
    build_packed_scalar_h1_jacobi_preconditioner_factory,
)
from femx.backends.jax.tet4_electrothermal import (  # noqa: E402
    Tet4ElectrothermalAdmissionPolicy,
    Tet4ElectrothermalParameters,
    build_tet4_electrothermal_runtime,
    pack_tet4_electrothermal_inputs,
    reconstruct_tet4_electrothermal_state,
)
from femx.core.execution import ExecutionPolicy  # noqa: E402
from femx.meshing.gmsh import (  # noqa: E402
    GmshInstallation,
    GmshMeshingRequest,
    GmshRunner,
    ImportedGmshMesh,
    read_gmsh_msh_3d,
)

SCHEMA_VERSION = "femx.ring-heater-thermal-sensitivity-evidence/v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "src/femx/applications/ring_heater.py",
    "src/femx/backends/jax/tet4_electrothermal.py",
    "src/femx/meshing/gmsh/ring_heater.py",
    "scripts/run_ring_heater_thermal_sensitivity.py",
    "uv.lock",
)


def _scalar(value: object) -> float:
    return float(np.asarray(jax.device_get(value)))


def _integer(value: object) -> int:
    return int(np.asarray(jax.device_get(value)))


def _region_temperature(
    temperature: np.ndarray,
    cells: np.ndarray,
    volumes: np.ndarray,
    cell_ids: np.ndarray,
) -> float:
    cell_mean = np.mean(temperature[cells[cell_ids]], axis=1)
    weights = volumes[cell_ids]
    return float(np.sum(cell_mean * weights) / np.sum(weights))


def _temperature_metrics(
    temperature: np.ndarray,
    *,
    imported: ImportedGmshMesh,
    thermal_cell_volumes: np.ndarray,
    power_w: float,
    ambient_k: float,
) -> dict[str, float]:
    cells = np.asarray(imported.mesh.topology.connectivity, dtype=np.int64)
    ring_ids = np.asarray(imported.mesh.tag("silicon_ring").entity_ids, dtype=np.int64)
    heater_ids = np.asarray(imported.mesh.tag("tin_heater").entity_ids, dtype=np.int64)
    rises = {
        "peak": float(np.max(temperature)) - ambient_k,
        "ring_mean": _region_temperature(
            temperature,
            cells,
            thermal_cell_volumes,
            ring_ids,
        )
        - ambient_k,
        "heater_mean": _region_temperature(
            temperature,
            cells,
            thermal_cell_volumes,
            heater_ids,
        )
        - ambient_k,
    }
    power_mw = power_w * 1.0e3
    return {
        "minimum_temperature_K": float(np.min(temperature)),
        "maximum_temperature_K": float(np.max(temperature)),
        "peak_rise_K": rises["peak"],
        "ring_mean_rise_K": rises["ring_mean"],
        "heater_mean_rise_K": rises["heater_mean"],
        "peak_K_per_mW": rises["peak"] / power_mw,
        "ring_mean_K_per_mW": rises["ring_mean"] / power_mw,
        "heater_mean_K_per_mW": rises["heater_mean"] / power_mw,
    }


def _solve_case(
    case: RingHeaterThermalSensitivityCase,
    *,
    case_directory: Path,
    gmsh_runner: GmshRunner,
    execution_policy: ExecutionPolicy,
) -> dict[str, object]:
    case_directory.mkdir()
    geometry_path = case_directory / "geometry.geo"
    mesh_path = case_directory / "mesh.msh"
    geometry_path.write_text(case.recipe.render_geo(), encoding="utf-8", newline="\n")
    gmsh_result = gmsh_runner.run(
        GmshMeshingRequest(
            geometry_path.name,
            mesh_filename=mesh_path.name,
            timeout_seconds=600.0,
            dimension=3,
        ),
        working_directory=case_directory,
        policy=execution_policy,
    )
    (case_directory / "gmsh.stdout.log").write_text(
        gmsh_result.stdout,
        encoding="utf-8",
        newline="\n",
    )
    (case_directory / "gmsh.stderr.log").write_text(
        gmsh_result.stderr,
        encoding="utf-8",
        newline="\n",
    )
    if not gmsh_result.process_succeeded:
        raise RuntimeError(f"Gmsh failed for {case.name}: {gmsh_result.stderr.strip()}")

    imported = read_gmsh_msh_3d(
        mesh_path,
        coordinate_scale_to_m=case.recipe.coordinate_scale_to_m,
    )
    plan = prepare_ring_heater_thermal_sensitivity_plan(
        imported,
        case.recipe,
        np.zeros((imported.mesh.topology.cell_count,), dtype=np.int64),
        partition_count=1,
        boundary=case.boundary,
    )
    devices = jax.devices("cpu")
    if len(devices) != 1 or not bool(getattr(jax.config, "jax_enable_x64", False)):
        raise RuntimeError("ring-heater sensitivity requires one CPU device with float64 enabled")
    jax_mesh = Mesh(np.asarray((devices[0],), dtype=object), ("partition",))
    jacobi = ScalarH1JacobiPolicy(1.0e-15)
    current_preconditioner = build_packed_scalar_h1_jacobi_preconditioner_factory(
        plan.tet4.current_layout,
        jax_mesh,
        jacobi,
    )
    thermal_preconditioner = build_packed_scalar_h1_jacobi_preconditioner_factory(
        plan.tet4.thermal_layout,
        jax_mesh,
        jacobi,
    )
    cg = ScalarH1CGPolicy(
        relative_tolerance=1.0e-10,
        absolute_tolerance=0.0,
        max_iterations=20_000,
        backward_error_tolerance=1.0e-9,
    )
    runtime = build_tet4_electrothermal_runtime(
        plan.tet4,
        jax_mesh,
        cg,
        cg,
        Tet4ElectrothermalAdmissionPolicy(1.0e-7, 1.0e-7, 1.0e-12, 1.0e-7),
        current_preconditioner_factory=current_preconditioner,
        thermal_preconditioner_factory=thermal_preconditioner,
    )
    inputs = pack_tet4_electrothermal_inputs(plan.tet4, value_dtype=np.float64)
    solve = jax.jit(runtime.solve)
    unit_parameters = Tet4ElectrothermalParameters(
        jnp.asarray(1.0, dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
    )
    solve_started = time.perf_counter()
    unit = solve(inputs, unit_parameters)
    unit.numerically_admitted.block_until_ready()
    unit_seconds = time.perf_counter() - solve_started
    if not bool(np.asarray(jax.device_get(unit.numerically_admitted))):
        raise RuntimeError(f"unit-voltage solve was not numerically admitted for {case.name}")

    operating_point = public_ring_heater_operating_point("low_temperature_projection")
    calibration = project_public_ring_heater_current(
        _scalar(unit.electrical_joule_power),
        operating_point=operating_point,
    )
    target_parameters = Tet4ElectrothermalParameters(
        jnp.asarray(calibration.target_voltage_v, dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
    )
    target_started = time.perf_counter()
    target = solve(inputs, target_parameters)
    target.numerically_admitted.block_until_ready()
    target_seconds = time.perf_counter() - target_started
    if not bool(np.asarray(jax.device_get(target.numerically_admitted))):
        raise RuntimeError(f"5 mA solve was not numerically admitted for {case.name}")
    _potential, temperature_device = reconstruct_tet4_electrothermal_state(
        plan.tet4,
        target.state,
        target_parameters,
    )
    temperature = np.asarray(jax.device_get(temperature_device), dtype=np.float64)
    if not np.all(np.isfinite(temperature)):
        raise RuntimeError(f"temperature was not finite for {case.name}")
    power_w = _scalar(target.electrical_joule_power)
    metrics = _temperature_metrics(
        temperature,
        imported=imported,
        thermal_cell_volumes=np.asarray(plan.tet4.thermal_cell_volumes, dtype=np.float64),
        power_w=power_w,
        ambient_k=case.boundary.ambient_temperature_k,
    )
    if metrics["minimum_temperature_K"] < case.boundary.ambient_temperature_k - 1.0e-7:
        raise RuntimeError(f"temperature fell below ambient for {case.name}")

    result: dict[str, object] = {
        "case": case.canonical_data(),
        "case_sha256": case.digest(),
        "plan_sha256": plan.digest(),
        "plan": plan.canonical_data(),
        "boundary": case.boundary.canonical_data(),
        "geometry": {
            "recipe_sha256": case.recipe.digest(),
            "source_msh_sha256": imported.record.source_sha256,
            "canonical_mesh_sha256": imported.record.canonical_mesh_sha256,
            "mesh_report_sha256": plan.mesh_report.digest(),
        },
        "gmsh": {
            "version": gmsh_result.identity.version,
            "executable_sha256": gmsh_result.identity.executable_sha256,
            "elapsed_seconds": gmsh_result.elapsed_seconds,
        },
        "mesh": {
            "node_count": plan.mesh_report.node_count,
            "tetrahedron_count": plan.mesh_report.tetrahedron_count,
            "minimum_mean_ratio": plan.mesh_report.minimum_mean_ratio,
            "maximum_region_volume_relative_error": (
                plan.mesh_report.maximum_region_volume_relative_error
            ),
        },
        "excitation": {
            "operating_point": operating_point.canonical_data(),
            "calibration": calibration.canonical_data(),
            "field_evaluation": {
                "method": "direct JAX solve at the calibrated 5 mA terminal voltage",
                "interpretation": (
                    "the projection role selects the low-temperature current; recomputing the "
                    "linear field does not add calibration or fabricated-device evidence"
                ),
            },
        },
        "numerics": {
            "unit_current_iterations": _integer(unit.current_linear.iterations),
            "unit_thermal_iterations": _integer(unit.thermal_linear.iterations),
            "target_current_iterations": _integer(target.current_linear.iterations),
            "target_thermal_iterations": _integer(target.thermal_linear.iterations),
            "current_backward_error": _scalar(target.current_linear.backward_error),
            "thermal_backward_error": _scalar(target.thermal_linear.backward_error),
            "charge_balance_relative_error": _scalar(target.charge_balance_relative_error),
            "electrical_energy_relative_error": _scalar(target.electrical_energy_relative_error),
            "joule_transfer_relative_error": _scalar(target.joule_transfer_relative_error),
            "thermal_balance_relative_error": _scalar(target.thermal_balance_relative_error),
            "electrical_joule_power_W": power_w,
            "robin_outward_power_W": _scalar(target.convection_outward_power),
            "dirichlet_outward_power_W": _scalar(target.dirichlet_outward_power),
        },
        "temperature": metrics,
        "timing": {
            "unit_solve_seconds_including_compile": unit_seconds,
            "target_solve_seconds": target_seconds,
        },
        "status": "passed",
    }
    (case_directory / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def _comparisons(records: list[dict[str, object]]) -> dict[str, object]:
    by_name = {str(record["case"]["name"]): record for record in records}  # type: ignore[index]
    baseline = by_name.get("source_envelope")
    if baseline is None:
        return {}
    baseline_temperature = baseline["temperature"]
    assert isinstance(baseline_temperature, dict)
    result: dict[str, object] = {}
    keys = ("peak_K_per_mW", "ring_mean_K_per_mW", "heater_mean_K_per_mW")
    for name, record in by_name.items():
        temperature = record["temperature"]
        assert isinstance(temperature, dict)
        result[name] = {
            f"{key}_relative_to_source_envelope": (
                float(temperature[key]) / float(baseline_temperature[key])
            )
            for key in keys
        }
    return result


def run(
    *,
    gmsh_executable: Path,
    run_directory: Path,
    selected_names: tuple[str, ...],
    authorize_external_process: bool,
) -> Path:
    """Create one immutable local sensitivity bundle and return its evidence path."""

    if not run_directory.is_absolute() or run_directory.exists() or run_directory.is_symlink():
        raise RuntimeError("run directory must be a new absolute path")
    if not gmsh_executable.is_absolute() or not gmsh_executable.is_file():
        raise RuntimeError("Gmsh executable must be an existing absolute file")
    available = {case.name: case for case in ring_heater_thermal_sensitivity_cases()}
    unknown = sorted(set(selected_names) - set(available))
    if unknown:
        raise RuntimeError(f"unknown sensitivity cases: {unknown}")
    names = selected_names or tuple(available)
    if "source_envelope" not in names:
        names = ("source_envelope", *names)
    run_directory.mkdir(parents=True)
    runner = GmshRunner(GmshInstallation(gmsh_executable.resolve()))
    policy = ExecutionPolicy(
        execution_authorized=authorize_external_process,
        allow_external_process=authorize_external_process,
    )
    started = time.perf_counter()
    records = [
        _solve_case(
            available[name],
            case_directory=run_directory / name,
            gmsh_runner=runner,
            execution_policy=policy,
        )
        for name in names
    ]
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "runtime": {
            "backend": jax.default_backend(),
            "device_count": len(jax.devices()),
            "x64_enabled": bool(getattr(jax.config, "jax_enable_x64", False)),
            "python_version": platform.python_version(),
            "jax_version": jax.__version__,
            "jaxlib_version": package_version("jaxlib"),
            "numpy_version": np.__version__,
        },
        "source_files": {
            relative_path: sha256_file(REPOSITORY_ROOT / relative_path)
            for relative_path in SOURCE_FILES
        },
        "cases": records,
        "comparisons": _comparisons(records),
        "elapsed_seconds": time.perf_counter() - started,
        "claim_scope": (
            "bounded constant-property CPU sensitivity; not mesh-converged, package-calibrated, "
            "or fabricated-device evidence"
        ),
    }
    evidence_path = run_directory / "evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return evidence_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gmsh", required=True, type=Path)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        choices=tuple(case.name for case in ring_heater_thermal_sensitivity_cases()),
        help="repeat to run a subset; source_envelope is added automatically",
    )
    parser.add_argument("--authorize-external-process", action="store_true")
    arguments = parser.parse_args()
    evidence_path = run(
        gmsh_executable=arguments.gmsh,
        run_directory=arguments.run_directory,
        selected_names=tuple(arguments.case),
        authorize_external_process=arguments.authorize_external_process,
    )
    print(evidence_path)


if __name__ == "__main__":
    main()
