#!/usr/bin/env python3
"""Generate the public 3D ring-heater JAX/Elmer parity figure bundle.

The generator builds the admitted coarse Gmsh model, solves the exact same first-order
tetrahedral current/Joule/heat problem with native JAX and a separately installed locked Elmer,
checks complete-field parity, and renders only from the retained open CSV tables. The source
operating point remains 15 mA; ``--operating-point low_temperature_projection`` instead performs
new JAX and Elmer solves at the separately declared 5 mA current. External execution is disabled
unless ``--allow-external`` is supplied. Nothing is downloaded or installed.
``--render-existing`` rebuilds the presentation from a checked bundle without starting a solver.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "femx-matplotlib"))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib as mpl  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Circle, Patch, Rectangle  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from femx.applications import (  # noqa: E402
    calibrate_public_ring_heater_current,
    prepare_public_ring_heater_elmer_plan,
    prepare_public_ring_heater_forward_plan,
    project_public_ring_heater_current,
    public_ring_heater_operating_point,
)
from femx.artifacts import sha256_file  # noqa: E402
from femx.backends.elmer.runner import ElmerInstallation  # noqa: E402
from femx.backends.elmer.steady_current import ElmerSteadyCurrentIdentity  # noqa: E402
from femx.backends.elmer.steady_heat import ElmerSteadyHeatIdentity  # noqa: E402
from femx.backends.elmer.tet4_electrothermal import (  # noqa: E402
    ElmerTet4ElectrothermalOracle,
)
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
    PublicRingHeater3D,
    read_gmsh_msh_3d,
    ring_heater_mesh_profile,
)

SCHEMA_VERSION = "femx.figure.3d-ring-heater-reference/v2"
SUPPORTED_INPUT_SCHEMA_VERSIONS = (
    "femx.figure.3d-ring-heater-reference/v1",
    SCHEMA_VERSION,
)
RUN_IDENTITY_SCHEMA_VERSION = "femx.figure.3d-ring-heater-run-identity/v1"
FIGURE_IDENTITY_SCHEMA_VERSION = "femx.figure.3d-ring-heater-render-identity/v1"
RUN_PREFIX = "3d-ring-heater-run"
FIGURE_PREFIX = "3d-ring-heater-figure"
FIGURE_RENDERING_REVISION = "paper-figure-v5"
GENERATION_LOCK_FILENAME = "generation.uv.lock"
AMBIENT_TEMPERATURE_K = 300.0
HORIZONTAL_SLICE_Z_M = 0.11e-6
TEMPERATURE_COLORMAP = "inferno"
DIFFERENCE_COLORMAP = "RdBu_r"
SUBSTRATE_HANDLE_DISPLAY_DEPTH_UM = 1.5
NOMINAL_HANDLE_WAFER_THICKNESS_UM = 725.0
DEVICE_OUTLINE_COLOR = "#E7ECF2"
DEVICE_OUTLINE_WIDTH_PT = 0.34
OUTLINE_CONTRAST_COLOR = "#263241"
OUTLINE_CONTRAST_STROKE_WIDTH_PT = 0.72
VERTICAL_OUTLINE_COLOR = "#F8FAFC"
VERTICAL_OUTLINE_WIDTH_PT = 0.50
VERTICAL_OUTLINE_CONTRAST_COLOR = "#111827"
VERTICAL_OUTLINE_CONTRAST_STROKE_WIDTH_PT = 1.18
MATERIAL_PALETTE_ID = "femx-semantic-materials-v1.1"
MATERIAL_PALETTE_PROFILE = "light"
MATERIAL_PALETTE_POLICY = (
    "v1.1 light-canvas fill and frame pairs; same material keeps one hue and display role uses "
    "geometry, outline, and opacity"
)
MATERIAL_LEGEND_ANCHOR = (0.125, 0.845)
MATERIAL_LEGEND_COLUMN_SPACING = 0.85
MATERIAL_LEGEND_HANDLE_LENGTH = 1.25
MATERIAL_LEGEND_HANDLE_TEXT_PADDING = 0.4
MATERIAL_FILL_COLORS = {
    "si_substrate": "#4B3F72",
    "si_device": "#685AB8",
    "sio2": "#167786",
    "tin": "#604900",
    "al": "#6F7885",
}
MATERIAL_FRAME_COLORS = {
    "si_substrate": "#2B2344",
    "si_device": "#352B76",
    "sio2": "#083F48",
    "tin": "#332600",
    "al": "#303741",
}
MATERIAL_RENDER_ORDER = (
    "continuous_silicon_substrate_with_truncated_handle_context",
    "solved_substrate_depth_boundary",
    "silica_box",
    "silica_cladding_extent_backdrops",
    "silicon_bus_upper_far",
    "silicon_ring",
    "tin_heater",
    "silicon_bus_lower_near",
    "al_contacts",
)
DIRECT_FIELD_EVALUATION = (
    "direct native JAX and external Elmer solves at the recorded target voltage; "
    "no temperature or parity field is obtained by algebraic rescaling"
)
DIRECT_FIELD_EVIDENCE_TIER = "direct_jax_elmer_same_mesh_solve"

PARITY_THRESHOLDS = {
    "maximum_absolute_potential_difference_V": 2.0e-8,
    "relative_l2_potential_difference": 1.0e-8,
    "maximum_absolute_temperature_difference_K": 2.0e-5,
    "relative_l2_temperature_rise_difference": 1.0e-8,
    "jax_electrical_energy_relative_error": 1.0e-10,
    "jax_charge_balance_relative_error": 1.0e-7,
    "jax_joule_transfer_relative_error": 1.0e-12,
    "jax_thermal_balance_relative_error": 1.0e-10,
    "target_current_relative_error": 1.0e-10,
}

REGION_COLORS = {
    "silica": MATERIAL_FILL_COLORS["sio2"],
    "silica_box": MATERIAL_FILL_COLORS["sio2"],
    "silica_cladding": MATERIAL_FILL_COLORS["sio2"],
    "silicon_substrate": MATERIAL_FILL_COLORS["si_substrate"],
    "silicon_ring": MATERIAL_FILL_COLORS["si_device"],
    "silicon_bus_upper": MATERIAL_FILL_COLORS["si_device"],
    "silicon_bus_lower": MATERIAL_FILL_COLORS["si_device"],
    "tin_heater": MATERIAL_FILL_COLORS["tin"],
    "al_contact_negative": MATERIAL_FILL_COLORS["al"],
    "al_contact_positive": MATERIAL_FILL_COLORS["al"],
}

LOCAL_TETRAHEDRON_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


@dataclass(frozen=True, slots=True)
class SolvedRing:
    """Complete fields and evidence required by the public figure bundle."""

    coordinates_m: np.ndarray
    cells: np.ndarray
    region_ids: np.ndarray
    region_names: tuple[str, ...]
    potential_node_ids: np.ndarray
    jax_potential_v: np.ndarray
    elmer_potential_v: np.ndarray
    jax_temperature_k: np.ndarray
    elmer_temperature_k: np.ndarray
    metrics: dict[str, float | int]
    provenance: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PlaneSlice:
    """Piecewise-linear Tet4 intersection with one physical plane."""

    first_um: np.ndarray
    second_um: np.ndarray
    triangles: np.ndarray
    jax_rise_k: np.ndarray
    elmer_rise_k: np.ndarray
    difference_k: np.ndarray


def _scalar(value: object) -> float:
    return float(np.asarray(jax.device_get(value)))


def _integer(value: object) -> int:
    return int(np.asarray(jax.device_get(value)))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _field_sha256(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype="<f8")
    return _sha256_bytes(canonical.tobytes())


def _git_head() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )
    head = completed.stdout.strip()
    if completed.returncode != 0 or len(head) != 40:
        raise RuntimeError("could not identify the femx Git commit")
    return head


def _git_head_file_bytes(relative_path: str) -> bytes:
    completed = subprocess.run(
        ("git", "show", f"HEAD:{relative_path}"),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"could not read {relative_path!r} from the femx Git commit")
    return completed.stdout


def _git_head_file_sha256(relative_path: str) -> str:
    return _sha256_bytes(_git_head_file_bytes(relative_path))


def _source_report(elmer_source: Path | None) -> dict[str, Any]:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "check_source_checkouts.py"),
        "--source",
        "elmer",
        "--require-clean",
        "--timeout-seconds",
        "300",
        "--json",
    ]
    if elmer_source is not None:
        command.extend(("--elmer", str(elmer_source.resolve())))
    completed = subprocess.run(
        tuple(command),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"strict Elmer source check failed: {completed.stderr.strip()}")
    payload = json.loads(completed.stdout)
    if not payload.get("valid") or len(payload.get("sources", [])) != 1:
        raise RuntimeError("strict Elmer source identity is not valid")
    source = dict(payload["sources"][0])
    if source.get("worktree_state") != "clean":
        raise RuntimeError("Elmer publication evidence requires a clean source worktree")
    return source


def _next_run_directory(root: Path, recipe_sha256: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 1000):
        candidate = root / f"{recipe_sha256[:16]}-attempt-{attempt:03d}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("no fresh 3D ring-heater run directory is available")


def _region_assignment(imported: Any, region_names: tuple[str, ...]) -> np.ndarray:
    cell_count = imported.mesh.topology.cell_count
    assignment = np.full((cell_count,), -1, dtype=np.int64)
    for region_id, name in enumerate(region_names):
        cell_ids = np.asarray(imported.mesh.tag(name).entity_ids, dtype=np.int64)
        if np.any(assignment[cell_ids] != -1):
            raise RuntimeError(f"3D region {name!r} overlaps a prior region")
        assignment[cell_ids] = region_id
    if np.any(assignment < 0):
        raise RuntimeError("3D material regions do not partition every Tet4 cell")
    return assignment


def _region_temperature(
    temperature: np.ndarray,
    cells: np.ndarray,
    volumes: np.ndarray,
    cell_ids: np.ndarray,
) -> float:
    means = np.mean(temperature[cells[cell_ids]], axis=1)
    return float(np.sum(means * volumes[cell_ids]) / np.sum(volumes[cell_ids]))


def solve_case(
    *,
    gmsh_executable: Path,
    elmer_executable: Path,
    elmer_source: Path | None,
    run_root: Path,
    operating_point_name: str,
) -> SolvedRing:
    """Generate, solve, and admit one public coarse 3D same-mesh comparison."""

    devices = jax.devices()
    if jax.default_backend() != "cpu" or len(devices) != 1:
        raise RuntimeError("README 3D reference generation requires exactly one CPU JAX device")
    if not bool(getattr(jax.config, "jax_enable_x64", False)):
        raise RuntimeError("README 3D reference generation requires JAX float64")

    recipe = PublicRingHeater3D(ring_heater_mesh_profile("coarse"))
    run_directory = _next_run_directory(run_root, recipe.digest())
    meshing_directory = run_directory / "gmsh"
    meshing_directory.mkdir()
    geometry_path = meshing_directory / "public_ring_heater.geo"
    geometry_path.write_text(recipe.render_geo(), encoding="utf-8", newline="\n")
    authorized = ExecutionPolicy(execution_authorized=True, allow_external_process=True)
    gmsh_started = time.perf_counter()
    gmsh = GmshRunner(GmshInstallation(gmsh_executable.resolve())).run(
        GmshMeshingRequest(geometry_path.name, dimension=3, timeout_seconds=300.0),
        working_directory=meshing_directory,
        policy=authorized,
    )
    gmsh_seconds = time.perf_counter() - gmsh_started
    (meshing_directory / "gmsh.stdout.log").write_text(gmsh.stdout, encoding="utf-8", newline="\n")
    (meshing_directory / "gmsh.stderr.log").write_text(gmsh.stderr, encoding="utf-8", newline="\n")
    if not gmsh.process_succeeded:
        raise RuntimeError(f"Gmsh failed: {gmsh.stderr.strip()}")

    mesh_path = meshing_directory / "mesh.msh"
    imported = read_gmsh_msh_3d(mesh_path, coordinate_scale_to_m=recipe.coordinate_scale_to_m)
    cells = np.asarray(imported.mesh.topology.connectivity, dtype=np.int64)
    coordinates = np.asarray(imported.mesh.geometry.coordinates, dtype=np.float64)
    region_names = tuple(recipe.VOLUME_GROUPS)
    region_ids = _region_assignment(imported, region_names)
    forward = prepare_public_ring_heater_forward_plan(
        imported,
        recipe,
        np.zeros((cells.shape[0],), dtype=np.int64),
        partition_count=1,
    )

    jax_mesh = Mesh(np.asarray((devices[0],), dtype=object), ("partition",))
    jacobi = ScalarH1JacobiPolicy(1.0e-15)
    current_preconditioner = build_packed_scalar_h1_jacobi_preconditioner_factory(
        forward.tet4.current_layout,
        jax_mesh,
        jacobi,
    )
    thermal_preconditioner = build_packed_scalar_h1_jacobi_preconditioner_factory(
        forward.tet4.thermal_layout,
        jax_mesh,
        jacobi,
    )
    cg = ScalarH1CGPolicy(1.0e-10, 0.0, 10_000, backward_error_tolerance=1.0e-9)
    runtime = build_tet4_electrothermal_runtime(
        forward.tet4,
        jax_mesh,
        cg,
        cg,
        Tet4ElectrothermalAdmissionPolicy(1.0e-7, 1.0e-7, 1.0e-12, 1.0e-7),
        current_preconditioner_factory=current_preconditioner,
        thermal_preconditioner_factory=thermal_preconditioner,
    )
    inputs = pack_tet4_electrothermal_inputs(forward.tet4, value_dtype=np.float64)

    def parameters(voltage_v: float) -> Tet4ElectrothermalParameters:
        return Tet4ElectrothermalParameters(
            jnp.asarray(voltage_v, dtype=jnp.float64),
            jnp.asarray(1.0, dtype=jnp.float64),
            jnp.asarray(1.0, dtype=jnp.float64),
        )

    solve = jax.jit(runtime.solve)
    unit_started = time.perf_counter()
    unit = solve(inputs, parameters(1.0))
    unit.numerically_admitted.block_until_ready()
    unit_seconds = time.perf_counter() - unit_started
    if not bool(unit.numerically_admitted):
        raise RuntimeError("unit-voltage JAX solve was not numerically admitted")
    operating_point = public_ring_heater_operating_point(operating_point_name)
    if operating_point.name == "source_reproduction":
        calibration = calibrate_public_ring_heater_current(
            _scalar(unit.electrical_joule_power),
            reference=forward.reference,
        )
    else:
        calibration = project_public_ring_heater_current(
            _scalar(unit.electrical_joule_power),
            operating_point=operating_point,
        )
    target_parameters = parameters(calibration.target_voltage_v)
    target_started = time.perf_counter()
    target = solve(inputs, target_parameters)
    target.numerically_admitted.block_until_ready()
    target_seconds = time.perf_counter() - target_started
    if not bool(target.numerically_admitted):
        raise RuntimeError("target-current JAX solve was not numerically admitted")
    potential_device, temperature_device = reconstruct_tet4_electrothermal_state(
        forward.tet4,
        target.state,
        target_parameters,
    )
    jax_potential = np.asarray(jax.device_get(potential_device), dtype=np.float64)
    jax_temperature = np.asarray(jax.device_get(temperature_device), dtype=np.float64)

    source = _source_report(elmer_source)
    elmer_home = elmer_executable.resolve().parent.parent
    module_directory = elmer_home / "share" / "elmersolver" / "lib"
    common_identity = {
        "version": "26.2-devel",
        "revision": str(source["head_commit"])[:9],
        "executable_sha256": sha256_file(elmer_executable.resolve()),
        "source_commit": str(source["head_commit"]),
        "source_digest": str(source["source_digest"]),
        "source_worktree_state": str(source["worktree_state"]),
    }
    current_identity = ElmerSteadyCurrentIdentity(
        **common_identity,
        stat_current_solve_sha256=sha256_file(module_directory / "StatCurrentSolve.so"),
    )
    heat_identity = ElmerSteadyHeatIdentity(
        **common_identity,
        heat_solve_sha256=sha256_file(module_directory / "HeatSolve.so"),
    )
    oracle = ElmerTet4ElectrothermalOracle(
        ElmerInstallation(elmer_executable.resolve()),
        current_identity,
        heat_identity,
        timeout_seconds=300.0,
    )
    elmer_plan = prepare_public_ring_heater_elmer_plan(
        imported,
        recipe,
        forward,
        applied_voltage_v=calibration.target_voltage_v,
    )
    elmer_started = time.perf_counter()
    elmer = oracle.run(
        elmer_plan.case,
        run_directory=run_directory / "elmer",
        policy=authorized,
    )
    elmer_seconds = time.perf_counter() - elmer_started
    if not elmer.numerically_converged:
        raise RuntimeError("Elmer process completed without admitted two-equation convergence")
    np.testing.assert_array_equal(elmer.potential_node_ids, forward.tet4.current_parent_node_ids)

    potential_difference = elmer.potential_v - jax_potential
    temperature_difference = elmer.temperature_k - jax_temperature
    jax_temperature_rise = jax_temperature - forward.reference.ambient_temperature_k
    relative_potential = float(np.linalg.norm(potential_difference) / np.linalg.norm(jax_potential))
    relative_temperature = float(
        np.linalg.norm(temperature_difference) / np.linalg.norm(jax_temperature_rise)
    )
    joule_power = _scalar(target.electrical_joule_power)
    inferred_current = joule_power / calibration.target_voltage_v
    target_current_error = abs(inferred_current - operating_point.target_current_a) / (
        operating_point.target_current_a
    )
    volumes = np.asarray(forward.tet4.thermal_cell_volumes, dtype=np.float64)
    ring_cells = np.asarray(imported.mesh.tag("silicon_ring").entity_ids, dtype=np.int64)
    heater_cells = np.asarray(imported.mesh.tag("tin_heater").entity_ids, dtype=np.int64)

    metrics: dict[str, float | int] = {
        "node_count": int(coordinates.shape[0]),
        "tetrahedron_count": int(cells.shape[0]),
        "conductor_node_count": int(elmer.potential_node_ids.size),
        "conductor_tetrahedron_count": int(forward.tet4.current_layout.topology.cell_count),
        "target_current_A": operating_point.target_current_a,
        "target_voltage_V": calibration.target_voltage_v,
        "electrical_joule_power_W": joule_power,
        "inferred_current_A": inferred_current,
        "target_current_relative_error": target_current_error,
        "minimum_temperature_K": float(np.min(jax_temperature)),
        "maximum_temperature_K": float(np.max(jax_temperature)),
        "maximum_temperature_rise_K": float(np.max(jax_temperature_rise)),
        "ring_volume_weighted_temperature_rise_K": _region_temperature(
            jax_temperature,
            cells,
            volumes,
            ring_cells,
        )
        - forward.reference.ambient_temperature_k,
        "heater_volume_weighted_temperature_rise_K": _region_temperature(
            jax_temperature,
            cells,
            volumes,
            heater_cells,
        )
        - forward.reference.ambient_temperature_k,
        "maximum_absolute_potential_difference_V": float(np.max(np.abs(potential_difference))),
        "rms_potential_difference_V": float(np.sqrt(np.mean(potential_difference**2))),
        "relative_l2_potential_difference": relative_potential,
        "maximum_absolute_temperature_difference_K": float(np.max(np.abs(temperature_difference))),
        "rms_temperature_difference_K": float(np.sqrt(np.mean(temperature_difference**2))),
        "relative_l2_temperature_rise_difference": relative_temperature,
        "jax_current_iterations": _integer(target.current_linear.iterations),
        "jax_thermal_iterations": _integer(target.thermal_linear.iterations),
        "jax_current_backward_error": _scalar(target.current_linear.backward_error),
        "jax_thermal_backward_error": _scalar(target.thermal_linear.backward_error),
        "jax_electrical_energy_relative_error": _scalar(target.electrical_energy_relative_error),
        "jax_charge_balance_relative_error": _scalar(target.charge_balance_relative_error),
        "jax_joule_transfer_relative_error": _scalar(target.joule_transfer_relative_error),
        "jax_thermal_balance_relative_error": _scalar(target.thermal_balance_relative_error),
        "jax_convection_outward_power_W": _scalar(target.convection_outward_power),
        "jax_bottom_outward_power_W": _scalar(target.dirichlet_outward_power),
        "minimum_tet4_mean_ratio": forward.mesh_report.minimum_mean_ratio,
        "maximum_region_volume_relative_error": (
            forward.mesh_report.maximum_region_volume_relative_error
        ),
    }
    for name, threshold in PARITY_THRESHOLDS.items():
        if float(metrics[name]) > threshold:
            raise RuntimeError(
                f"scientific threshold failed: {name}={float(metrics[name]):.9e} > {threshold:.9e}"
            )
    if not np.isfinite(jax_temperature).all() or not np.isfinite(elmer.temperature_k).all():
        raise RuntimeError("3D temperature fields contain non-finite values")
    if not np.isfinite(jax_potential).all() or not np.isfinite(elmer.potential_v).all():
        raise RuntimeError("3D potential fields contain non-finite values")
    peak_rise = float(metrics["maximum_temperature_rise_K"])
    expected_peak_bracket = (
        (160.0, 170.0) if operating_point.name == "source_reproduction" else (18.0, 19.0)
    )
    if not expected_peak_bracket[0] < peak_rise < expected_peak_bracket[1]:
        raise RuntimeError(
            "3D coarse-ring peak temperature is outside the operating-point regression bracket"
        )

    generator_sha256 = sha256_file(Path(__file__).resolve())
    provenance = {
        "model": {
            "dimension": 3,
            "element": "first-order affine Tet4 H1/P1",
            "profile": "coarse",
            "coordinate_unit": "m in solver data; um in figure axes",
            "electrical_space": "TiN plus two femx aluminum contacts",
            "thermal_space": "all eight material volumes",
            "joule_transfer": "exact parent-cell identity",
            "boundary_conditions": forward.canonical_data()["thermal_boundary_conditions"],
            "material_status": "source-pinned public benchmark; not foundry calibrated",
            "operating_point": operating_point.canonical_data(),
            "field_evaluation": DIRECT_FIELD_EVALUATION,
            "field_evidence": {
                "evidence_tier": DIRECT_FIELD_EVIDENCE_TIER,
                "operating_point_selection_tier": operating_point.evidence_tier,
                "jax_target_solve": "new native solve at the recorded target voltage",
                "elmer_target_solve": "new external solve at the recorded target voltage",
                "algebraic_field_rescaling_used": False,
                "claim_scope": (
                    "direct same-discretization solver parity at the selected current; not a "
                    "thermal-domain correction, material calibration, or fabricated-device claim"
                ),
            },
            "same_discretization_parity_claimed": True,
            "mesh_convergence_claimed": False,
            "fabricated_device_prediction_claimed": False,
        },
        "public_source": recipe.canonical_data()["public_source"],
        "femx_geometry_extension": recipe.canonical_data()["femx_extension"],
        "gmsh": {
            "version": gmsh.identity.version,
            "executable_sha256": gmsh.identity.executable_sha256,
            "geometry_sha256": gmsh.geometry_sha256,
            "source_msh_sha256": imported.record.source_sha256,
            "canonical_mesh_sha256": imported.record.canonical_mesh_sha256,
            "import_record_sha256": imported.record.digest(),
            "recipe_sha256": recipe.digest(),
            "mesh_report_sha256": forward.mesh_report.digest(),
            "process_succeeded": gmsh.process_succeeded,
        },
        "jax": {
            "version": jax.__version__,
            "jaxlib_version": package_version("jaxlib"),
            "backend": jax.default_backend(),
            "device_kind": str(getattr(devices[0], "device_kind", devices[0])),
            "device_count": len(devices),
            "float64": bool(getattr(jax.config, "jax_enable_x64", False)),
            "forward_plan_sha256": forward.digest(),
            "potential_sha256_float64": _field_sha256(jax_potential),
            "temperature_sha256_float64": _field_sha256(jax_temperature),
        },
        "elmer": {
            **dict(elmer.provenance),
            "oracle_version": oracle.version,
            "plan_sha256": elmer_plan.digest(),
            "potential_sha256_float64": _field_sha256(elmer.potential_v),
            "temperature_sha256_float64": _field_sha256(elmer.temperature_k),
            "current_steady_change": list(elmer.current_steady_change or ()),
            "heat_steady_change": list(elmer.heat_steady_change or ()),
            "numerically_converged": elmer.numerically_converged,
        },
        "femx": {
            "commit": _git_head(),
            "generator_sha256": generator_sha256,
            "field_generator_sha256": generator_sha256,
            "presentation_generator_sha256": generator_sha256,
            "uv_lock_sha256": _git_head_file_sha256("uv.lock"),
            "uv_lock_source": "HEAD:uv.lock",
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "matplotlib_version": mpl.__version__,
        },
        "timing_seconds": {
            "gmsh": gmsh_seconds,
            "jax_unit_including_compile": unit_seconds,
            "jax_target": target_seconds,
            "elmer": elmer_seconds,
        },
        "raw_run_artifacts": {
            "retained_outside_git": True,
            "relative_root": ".femx/readme-3d-ring-heater/",
            "run_id": run_directory.name,
            "contains": [
                "GEO and MSH",
                "Gmsh stdout and stderr",
                "Elmer native mesh and SIF",
                "Elmer indexed result and VTU",
                "Elmer stdout and stderr",
            ],
        },
    }
    return SolvedRing(
        coordinates_m=coordinates,
        cells=cells,
        region_ids=region_ids,
        region_names=region_names,
        potential_node_ids=np.asarray(elmer.potential_node_ids, dtype=np.int64),
        jax_potential_v=jax_potential,
        elmer_potential_v=np.asarray(elmer.potential_v, dtype=np.float64),
        jax_temperature_k=jax_temperature,
        elmer_temperature_k=np.asarray(elmer.temperature_k, dtype=np.float64),
        metrics=metrics,
        provenance=provenance,
    )


def _write_nodes_csv(path: Path, solved: SolvedRing) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "node_id",
                "x_um",
                "y_um",
                "z_um",
                "jax_temperature_K",
                "elmer_temperature_K",
                "elmer_minus_jax_K",
            )
        )
        for node_id, (coordinate, jax_value, elmer_value) in enumerate(
            zip(
                solved.coordinates_m,
                solved.jax_temperature_k,
                solved.elmer_temperature_k,
                strict=True,
            )
        ):
            writer.writerow(
                (
                    node_id,
                    format(coordinate[0] * 1.0e6, ".17e"),
                    format(coordinate[1] * 1.0e6, ".17e"),
                    format(coordinate[2] * 1.0e6, ".17e"),
                    format(jax_value, ".17e"),
                    format(elmer_value, ".17e"),
                    format(elmer_value - jax_value, ".17e"),
                )
            )


def _write_cells_csv(path: Path, solved: SolvedRing) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "cell_id",
                "node_0",
                "node_1",
                "node_2",
                "node_3",
                "region_id",
                "region_name",
            )
        )
        for cell_id, (cell, region_id) in enumerate(
            zip(solved.cells, solved.region_ids, strict=True)
        ):
            writer.writerow(
                (
                    cell_id,
                    int(cell[0]),
                    int(cell[1]),
                    int(cell[2]),
                    int(cell[3]),
                    int(region_id),
                    solved.region_names[int(region_id)],
                )
            )


def _write_potential_csv(path: Path, solved: SolvedRing) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "field_index",
                "source_node_id",
                "jax_potential_V",
                "elmer_potential_V",
                "elmer_minus_jax_V",
            )
        )
        for field_index, (node_id, jax_value, elmer_value) in enumerate(
            zip(
                solved.potential_node_ids,
                solved.jax_potential_v,
                solved.elmer_potential_v,
                strict=True,
            )
        ):
            writer.writerow(
                (
                    field_index,
                    int(node_id),
                    format(jax_value, ".17e"),
                    format(elmer_value, ".17e"),
                    format(elmer_value - jax_value, ".17e"),
                )
            )


def load_solved_ring_from_bundle(bundle_dir: Path) -> SolvedRing:
    """Load a checked figure bundle without rerunning Gmsh, JAX, or Elmer."""

    bundle_dir = bundle_dir.resolve()
    evidence_path = bundle_dir / "evidence.json"
    if not evidence_path.is_file():
        raise RuntimeError(f"existing bundle has no evidence file: {evidence_path}")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("schema_version") not in SUPPORTED_INPUT_SCHEMA_VERSIONS:
        raise RuntimeError("existing bundle has an unsupported schema version")
    for name in ("nodes.csv", "cells.csv", "potential.csv"):
        expected = evidence.get("artifacts", {}).get(name)
        if not isinstance(expected, str) or sha256_file(bundle_dir / name) != expected:
            raise RuntimeError(f"existing bundle {name} does not match its recorded SHA-256")

    provenance_data = evidence.get("provenance", {})
    femx_provenance = provenance_data.get("femx", {})
    generation_lock_name = femx_provenance.get("uv_lock_artifact", GENERATION_LOCK_FILENAME)
    if generation_lock_name != GENERATION_LOCK_FILENAME:
        raise RuntimeError("existing bundle names an unsupported generation lock artifact")
    generation_lock_path = bundle_dir / generation_lock_name
    expected_lock_sha256 = femx_provenance.get("uv_lock_sha256")
    if (
        not isinstance(expected_lock_sha256, str)
        or not generation_lock_path.is_file()
        or sha256_file(generation_lock_path) != expected_lock_sha256
    ):
        raise RuntimeError("existing bundle generation lock does not match its provenance")
    recorded_lock_sha256 = evidence.get("artifacts", {}).get(generation_lock_name)
    if recorded_lock_sha256 is not None and recorded_lock_sha256 != expected_lock_sha256:
        raise RuntimeError("existing bundle records conflicting generation lock hashes")

    coordinates_m: list[tuple[float, float, float]] = []
    jax_temperature_k: list[float] = []
    elmer_temperature_k: list[float] = []
    with (bundle_dir / "nodes.csv").open(encoding="utf-8", newline="") as stream:
        for expected_node_id, row in enumerate(csv.DictReader(stream)):
            if int(row["node_id"]) != expected_node_id:
                raise RuntimeError("existing nodes.csv is not in canonical node order")
            coordinates_m.append(
                (
                    float(row["x_um"]) * 1.0e-6,
                    float(row["y_um"]) * 1.0e-6,
                    float(row["z_um"]) * 1.0e-6,
                )
            )
            jax_temperature_k.append(float(row["jax_temperature_K"]))
            elmer_temperature_k.append(float(row["elmer_temperature_K"]))

    cells: list[tuple[int, int, int, int]] = []
    region_ids: list[int] = []
    names_by_id: dict[int, str] = {}
    with (bundle_dir / "cells.csv").open(encoding="utf-8", newline="") as stream:
        for expected_cell_id, row in enumerate(csv.DictReader(stream)):
            if int(row["cell_id"]) != expected_cell_id:
                raise RuntimeError("existing cells.csv is not in canonical cell order")
            cells.append(
                (
                    int(row["node_0"]),
                    int(row["node_1"]),
                    int(row["node_2"]),
                    int(row["node_3"]),
                )
            )
            region_id = int(row["region_id"])
            region_name = row["region_name"]
            if region_id in names_by_id and names_by_id[region_id] != region_name:
                raise RuntimeError("existing cells.csv maps one region id to multiple names")
            names_by_id[region_id] = region_name
            region_ids.append(region_id)
    if set(names_by_id) != set(range(len(names_by_id))):
        raise RuntimeError("existing cells.csv region ids are not contiguous from zero")
    region_names = tuple(names_by_id[index] for index in range(len(names_by_id)))

    potential_node_ids: list[int] = []
    jax_potential_v: list[float] = []
    elmer_potential_v: list[float] = []
    with (bundle_dir / "potential.csv").open(encoding="utf-8", newline="") as stream:
        for expected_field_index, row in enumerate(csv.DictReader(stream)):
            if int(row["field_index"]) != expected_field_index:
                raise RuntimeError("existing potential.csv is not in canonical field order")
            potential_node_ids.append(int(row["source_node_id"]))
            jax_potential_v.append(float(row["jax_potential_V"]))
            elmer_potential_v.append(float(row["elmer_potential_V"]))

    metrics = dict(evidence["metrics"])
    if len(coordinates_m) != int(metrics["node_count"]):
        raise RuntimeError("existing nodes.csv row count does not match evidence")
    if len(cells) != int(metrics["tetrahedron_count"]):
        raise RuntimeError("existing cells.csv row count does not match evidence")
    if len(potential_node_ids) != int(metrics["conductor_node_count"]):
        raise RuntimeError("existing potential.csv row count does not match evidence")

    provenance = dict(evidence["provenance"])
    provenance["model"] = dict(provenance["model"])
    model = provenance["model"]
    if "operating_point" not in model:
        target_current_a = float(metrics["target_current_A"])
        if abs(target_current_a - 0.015) <= 1.0e-15:
            operating_point_name = "source_reproduction"
        elif abs(target_current_a - 0.005) <= 1.0e-15:
            operating_point_name = "low_temperature_projection"
        else:
            raise RuntimeError(
                "existing bundle target current does not identify a supported operating point"
            )
        model["operating_point"] = public_ring_heater_operating_point(
            operating_point_name
        ).canonical_data()
    model["field_evaluation"] = DIRECT_FIELD_EVALUATION
    operating_point = dict(model["operating_point"])
    model["field_evidence"] = {
        "evidence_tier": DIRECT_FIELD_EVIDENCE_TIER,
        "operating_point_selection_tier": operating_point["evidence_tier"],
        "jax_target_solve": "retained native solve at the recorded target voltage",
        "elmer_target_solve": "retained external solve at the recorded target voltage",
        "algebraic_field_rescaling_used": False,
        "claim_scope": (
            "direct same-discretization solver parity at the selected current; not a "
            "thermal-domain correction, material calibration, or fabricated-device claim"
        ),
    }
    provenance["femx"] = dict(provenance["femx"])
    field_generator_sha256 = provenance["femx"].get(
        "field_generator_sha256", provenance["femx"]["generator_sha256"]
    )
    presentation_generator_sha256 = sha256_file(Path(__file__).resolve())
    provenance["femx"]["field_generator_sha256"] = field_generator_sha256
    provenance["femx"]["presentation_generator_sha256"] = presentation_generator_sha256
    provenance["femx"]["generator_sha256"] = presentation_generator_sha256
    provenance["femx"]["uv_lock_artifact"] = GENERATION_LOCK_FILENAME
    provenance["figure_rendering"] = {
        "mode": "retained-open-csv-tables",
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "matplotlib_version": mpl.__version__,
    }
    return SolvedRing(
        coordinates_m=np.asarray(coordinates_m, dtype=np.float64),
        cells=np.asarray(cells, dtype=np.int64),
        region_ids=np.asarray(region_ids, dtype=np.int64),
        region_names=region_names,
        potential_node_ids=np.asarray(potential_node_ids, dtype=np.int64),
        jax_potential_v=np.asarray(jax_potential_v, dtype=np.float64),
        elmer_potential_v=np.asarray(elmer_potential_v, dtype=np.float64),
        jax_temperature_k=np.asarray(jax_temperature_k, dtype=np.float64),
        elmer_temperature_k=np.asarray(elmer_temperature_k, dtype=np.float64),
        metrics=metrics,
        provenance=provenance,
    )


def _identity_digest(identity: dict[str, Any]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_identity(output_dir: Path, solved: SolvedRing) -> dict[str, Any]:
    """Return the numerical-field identity, excluding presentation implementation."""

    return {
        "schema_version": RUN_IDENTITY_SCHEMA_VERSION,
        "nodes_sha256": sha256_file(output_dir / "nodes.csv"),
        "cells_sha256": sha256_file(output_dir / "cells.csv"),
        "potential_sha256": sha256_file(output_dir / "potential.csv"),
        "canonical_mesh_sha256": solved.provenance["gmsh"]["canonical_mesh_sha256"],
        "field_generator_sha256": solved.provenance["femx"]["field_generator_sha256"],
    }


def _run_identifier(run_identity: dict[str, Any]) -> str:
    return f"{RUN_PREFIX}-{_identity_digest(run_identity)[:16]}"


def _figure_rendering_identity() -> dict[str, Any]:
    """Return the explicit rendering policy used to distinguish redraws from reruns."""

    return {
        "schema_version": FIGURE_IDENTITY_SCHEMA_VERSION,
        "rendering_revision": FIGURE_RENDERING_REVISION,
        "runtime": {
            "numpy_version": np.__version__,
            "matplotlib_version": mpl.__version__,
        },
        "canvas": {
            "physical_size_inches": [7.2, 4.8],
            "png_dpi": 300,
            "grid_width_ratios": [1.28, 1.0, 1.0],
        },
        "temperature": {
            "colormap": TEMPERATURE_COLORMAP,
            "normalization": "linear complete retained range",
            "panel_scope": ["b", "c"],
            "panel_a_overlay": "none; categorical material geometry only",
            "horizontal_contours": (
                "none; scalar isolines are omitted so they cannot be confused with CAD boundaries"
            ),
        },
        "difference": {
            "colormap": DIFFERENCE_COLORMAP,
            "normalization": "zero-centered symmetric complete retained range",
        },
        "material_palette": {
            "id": MATERIAL_PALETTE_ID,
            "profile": MATERIAL_PALETTE_PROFILE,
            "fill_colors": dict(MATERIAL_FILL_COLORS),
            "frame_colors": dict(MATERIAL_FRAME_COLORS),
        },
        "panel_a": {
            "projection": "orthographic axonometric",
            "view_elevation_deg": 26.0,
            "view_azimuth_deg": -56.0,
            "box_zoom": 0.98,
            "displayed_handle_depth_um": SUBSTRATE_HANDLE_DISPLAY_DEPTH_UM,
            "scalar_overlay": "none",
            "legend": {
                "anchor_figure_fraction": list(MATERIAL_LEGEND_ANCHOR),
                "columns": 2,
                "column_spacing": MATERIAL_LEGEND_COLUMN_SPACING,
                "handle_length": MATERIAL_LEGEND_HANDLE_LENGTH,
                "handle_text_padding": MATERIAL_LEGEND_HANDLE_TEXT_PADDING,
                "layout": "left-shifted to preserve the panel-a-to-b label gutter",
            },
            "silicon_device_painter_order": [
                "silicon_bus_upper_far",
                "silicon_ring",
                "silicon_bus_lower_near",
            ],
            "projected_device_painter_order": [
                "silicon_bus_upper_far",
                "silicon_ring",
                "tin_heater",
                "silicon_bus_lower_near",
                "al_contacts",
            ],
            "ring_and_bus_alpha": 1.0,
            "projected_device_occlusion": (
                "opaque far-bus, ring assembly, near-bus painter layers; the ring hides the "
                "rear bus and the near bus remains foreground"
            ),
        },
        "field_outlines": {
            "horizontal_color": DEVICE_OUTLINE_COLOR,
            "horizontal_width_pt": DEVICE_OUTLINE_WIDTH_PT,
            "horizontal_contrast_width_pt": OUTLINE_CONTRAST_STROKE_WIDTH_PT,
            "horizontal_semantics": "source-CAD Si-device boundary; no scalar isolines",
            "vertical_color": VERTICAL_OUTLINE_COLOR,
            "vertical_width_pt": VERTICAL_OUTLINE_WIDTH_PT,
            "vertical_contrast_color": VERTICAL_OUTLINE_CONTRAST_COLOR,
            "vertical_contrast_width_pt": VERTICAL_OUTLINE_CONTRAST_STROKE_WIDTH_PT,
        },
        "parity_metrics_placement": "compact two-line subtitle above the plotting area",
    }


def _figure_identifier(run_id: str, rendering_identity: dict[str, Any]) -> str:
    identity = {
        "schema_version": FIGURE_IDENTITY_SCHEMA_VERSION,
        "run_id": run_id,
        "rendering": rendering_identity,
    }
    return f"{FIGURE_PREFIX}-{_identity_digest(identity)[:16]}"


def _slice_tetrahedra(
    solved: SolvedRing,
    *,
    fixed_axis: int,
    fixed_value_m: float,
) -> PlaneSlice:
    projected_axes = tuple(axis for axis in range(3) if axis != fixed_axis)
    first: list[float] = []
    second: list[float] = []
    triangles: list[tuple[int, int, int]] = []
    jax_values: list[float] = []
    elmer_values: list[float] = []
    difference_values: list[float] = []
    tolerance = 1.0e-15

    for cell in solved.cells:
        points = solved.coordinates_m[cell]
        signed = points[:, fixed_axis] - fixed_value_m
        if float(np.min(signed)) > tolerance or float(np.max(signed)) < -tolerance:
            continue
        local_jax = solved.jax_temperature_k[cell] - AMBIENT_TEMPERATURE_K
        local_elmer = solved.elmer_temperature_k[cell] - AMBIENT_TEMPERATURE_K
        candidates: dict[tuple[float, float], tuple[np.ndarray, float, float]] = {}

        for local_node, distance in enumerate(signed):
            if abs(float(distance)) <= tolerance:
                projected = points[local_node, list(projected_axes)]
                key = (round(float(projected[0]), 15), round(float(projected[1]), 15))
                candidates[key] = (
                    projected,
                    float(local_jax[local_node]),
                    float(local_elmer[local_node]),
                )
        for left, right in LOCAL_TETRAHEDRON_EDGES:
            left_distance = float(signed[left])
            right_distance = float(signed[right])
            if left_distance * right_distance >= 0.0:
                continue
            fraction = left_distance / (left_distance - right_distance)
            point = points[left] + fraction * (points[right] - points[left])
            jax_value = float(local_jax[left] + fraction * (local_jax[right] - local_jax[left]))
            elmer_value = float(
                local_elmer[left] + fraction * (local_elmer[right] - local_elmer[left])
            )
            projected = point[list(projected_axes)]
            key = (round(float(projected[0]), 15), round(float(projected[1]), 15))
            candidates[key] = (projected, jax_value, elmer_value)
        if len(candidates) < 3:
            continue
        polygon = list(candidates.values())
        centroid = np.mean(np.stack([item[0] for item in polygon]), axis=0)
        polygon.sort(
            key=lambda item: float(np.arctan2(item[0][1] - centroid[1], item[0][0] - centroid[0]))
        )
        start = len(first)
        for projected, jax_value, elmer_value in polygon:
            first.append(float(projected[0] * 1.0e6))
            second.append(float(projected[1] * 1.0e6))
            jax_values.append(jax_value)
            elmer_values.append(elmer_value)
            difference_values.append(elmer_value - jax_value)
        for index in range(1, len(polygon) - 1):
            candidate = (start, start + index, start + index + 1)
            triangle_points = np.asarray(
                [
                    (first[candidate[0]], second[candidate[0]]),
                    (first[candidate[1]], second[candidate[1]]),
                    (first[candidate[2]], second[candidate[2]]),
                ]
            )
            first_edge = triangle_points[1] - triangle_points[0]
            second_edge = triangle_points[2] - triangle_points[0]
            signed_area = float(first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0])
            if abs(signed_area) > 1.0e-18:
                if signed_area > 0.0:
                    triangles.append(candidate)
                else:
                    triangles.append((candidate[0], candidate[2], candidate[1]))
    if not triangles:
        raise RuntimeError("requested 3D slice did not intersect any nondegenerate Tet4 cell")
    return PlaneSlice(
        first_um=np.asarray(first, dtype=np.float64),
        second_um=np.asarray(second, dtype=np.float64),
        triangles=np.asarray(triangles, dtype=np.int64),
        jax_rise_k=np.asarray(jax_values, dtype=np.float64),
        elmer_rise_k=np.asarray(elmer_values, dtype=np.float64),
        difference_k=np.asarray(difference_values, dtype=np.float64),
    )


def _box_faces(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> dict[str, np.ndarray]:
    vertices = np.asarray(
        (
            (x_min, y_min, z_min),
            (x_max, y_min, z_min),
            (x_max, y_max, z_min),
            (x_min, y_max, z_min),
            (x_min, y_min, z_max),
            (x_max, y_min, z_max),
            (x_max, y_max, z_max),
            (x_min, y_max, z_max),
        ),
        dtype=np.float64,
    )
    return {
        "bottom": vertices[[0, 3, 2, 1]],
        "top": vertices[[4, 5, 6, 7]],
        "y_min": vertices[[0, 1, 5, 4]],
        "x_max": vertices[[1, 2, 6, 5]],
        "y_max": vertices[[2, 3, 7, 6]],
        "x_min": vertices[[3, 0, 4, 7]],
    }


def _add_faces(
    axis: Any,
    faces: list[np.ndarray],
    *,
    color: str,
    alpha: float,
    edgecolor: str,
    linewidth: float,
    shade: bool,
    zorder: float,
) -> None:
    shaded_edgecolor = color if shade and edgecolor == "none" else edgecolor
    collection = Poly3DCollection(
        np.asarray(faces, dtype=np.float64),
        facecolors=color,
        edgecolors=shaded_edgecolor,
        linewidths=linewidth,
        alpha=alpha,
        shade=shade,
        lightsource=mpl.colors.LightSource(azdeg=315.0, altdeg=42.0),
        antialiased=True,
        rasterized=True,
        zsort="average",
        zorder=zorder,
    )
    axis.add_collection3d(collection)


def _add_box(
    axis: Any,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    color: str,
    alpha: float,
    edgecolor: str = "none",
    linewidth: float = 0.0,
    visible_faces: tuple[str, ...] | None = None,
    shade: bool = True,
    zorder: float,
) -> None:
    face_map = _box_faces(x_min, x_max, y_min, y_max, z_min, z_max)
    names = tuple(face_map) if visible_faces is None else visible_faces
    _add_faces(
        axis,
        [face_map[name] for name in names],
        color=color,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
        shade=shade,
        zorder=zorder,
    )


def _annular_prism_faces(
    *,
    inner_radius: float,
    outer_radius: float,
    z_min: float,
    z_max: float,
    notch_half_x: float | None = None,
    angular_steps: int = 192,
    radial_steps: int = 5,
) -> list[np.ndarray]:
    faces: list[np.ndarray] = []
    if notch_half_x is None:
        theta = np.linspace(0.0, 2.0 * np.pi, angular_steps + 1)
        for index in range(angular_steps):
            left = theta[index]
            right = theta[index + 1]
            inner_left = (inner_radius * np.cos(left), inner_radius * np.sin(left))
            outer_left = (outer_radius * np.cos(left), outer_radius * np.sin(left))
            inner_right = (inner_radius * np.cos(right), inner_radius * np.sin(right))
            outer_right = (outer_radius * np.cos(right), outer_radius * np.sin(right))
            faces.extend(
                (
                    np.asarray(
                        (
                            (*inner_left, z_max),
                            (*outer_left, z_max),
                            (*outer_right, z_max),
                            (*inner_right, z_max),
                        )
                    ),
                    np.asarray(
                        (
                            (*inner_right, z_min),
                            (*outer_right, z_min),
                            (*outer_left, z_min),
                            (*inner_left, z_min),
                        )
                    ),
                    np.asarray(
                        (
                            (*outer_left, z_min),
                            (*outer_right, z_min),
                            (*outer_right, z_max),
                            (*outer_left, z_max),
                        )
                    ),
                    np.asarray(
                        (
                            (*inner_right, z_min),
                            (*inner_left, z_min),
                            (*inner_left, z_max),
                            (*inner_right, z_max),
                        )
                    ),
                )
            )
        return faces

    if not 0.0 < notch_half_x < inner_radius:
        raise RuntimeError("heater notch must be narrower than the annulus inner radius")
    radii = np.linspace(inner_radius, outer_radius, radial_steps + 1)
    unit_coordinates = np.linspace(0.0, 1.0, angular_steps + 1)
    theta = np.empty((radial_steps + 1, angular_steps + 1), dtype=np.float64)
    for radial_index, radius in enumerate(radii):
        half_angle = float(np.arcsin(notch_half_x / radius))
        start = 1.5 * np.pi + half_angle
        stop = 3.5 * np.pi - half_angle
        theta[radial_index] = start + unit_coordinates * (stop - start)

    def point(radial_index: int, angular_index: int, z_value: float) -> tuple[float, float, float]:
        radius = radii[radial_index]
        angle = theta[radial_index, angular_index]
        return (float(radius * np.cos(angle)), float(radius * np.sin(angle)), z_value)

    for radial_index in range(radial_steps):
        for angular_index in range(angular_steps):
            faces.extend(
                (
                    np.asarray(
                        (
                            point(radial_index, angular_index, z_max),
                            point(radial_index + 1, angular_index, z_max),
                            point(radial_index + 1, angular_index + 1, z_max),
                            point(radial_index, angular_index + 1, z_max),
                        )
                    ),
                    np.asarray(
                        (
                            point(radial_index, angular_index + 1, z_min),
                            point(radial_index + 1, angular_index + 1, z_min),
                            point(radial_index + 1, angular_index, z_min),
                            point(radial_index, angular_index, z_min),
                        )
                    ),
                )
            )
    for angular_index in range(angular_steps):
        faces.extend(
            (
                np.asarray(
                    (
                        point(radial_steps, angular_index, z_min),
                        point(radial_steps, angular_index + 1, z_min),
                        point(radial_steps, angular_index + 1, z_max),
                        point(radial_steps, angular_index, z_max),
                    )
                ),
                np.asarray(
                    (
                        point(0, angular_index + 1, z_min),
                        point(0, angular_index, z_min),
                        point(0, angular_index, z_max),
                        point(0, angular_index + 1, z_max),
                    )
                ),
            )
        )
    for radial_index in range(radial_steps):
        faces.extend(
            (
                np.asarray(
                    (
                        point(radial_index, 0, z_min),
                        point(radial_index + 1, 0, z_min),
                        point(radial_index + 1, 0, z_max),
                        point(radial_index, 0, z_max),
                    )
                ),
                np.asarray(
                    (
                        point(radial_index + 1, angular_steps, z_min),
                        point(radial_index, angular_steps, z_min),
                        point(radial_index, angular_steps, z_max),
                        point(radial_index + 1, angular_steps, z_max),
                    )
                ),
            )
        )
    return faces


def _add_annular_prism(
    axis: Any,
    *,
    inner_radius: float,
    outer_radius: float,
    z_min: float,
    z_max: float,
    color: str,
    notch_half_x: float | None = None,
    alpha: float = 0.94,
    zorder: float,
) -> None:
    _add_faces(
        axis,
        _annular_prism_faces(
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            z_min=z_min,
            z_max=z_max,
            notch_half_x=notch_half_x,
        ),
        color=color,
        alpha=alpha,
        edgecolor="none",
        linewidth=0.0,
        shade=True,
        zorder=zorder,
    )


def _plot_material_surfaces(
    axis: Any,
    solved: SolvedRing,
) -> PublicRingHeater3D:
    recipe = PublicRingHeater3D(ring_heater_mesh_profile("coarse"))
    if solved.provenance["gmsh"]["recipe_sha256"] != recipe.digest():
        raise RuntimeError("figure geometry recipe does not match the retained solved mesh")

    scale = 1.0e6
    half_x = recipe.domain_x_m * scale / 2.0
    half_y = recipe.domain_y_m * scale / 2.0
    substrate_bottom = recipe.substrate_bottom_z_m * scale
    substrate_top = recipe.substrate_top_z_m * scale
    silicon_bottom = 0.0
    silicon_top = recipe.waveguide_height_m * scale
    heater_bottom = recipe.heater_bottom_z_m * scale
    heater_top = recipe.heater_top_z_m * scale
    cladding_top = recipe.cladding_top_z_m * scale
    display_substrate_bottom = substrate_bottom - SUBSTRATE_HANDLE_DISPLAY_DEPTH_UM

    # Draw the physical stack from bottom to top. With computed z-order disabled, the explicit
    # painter order keeps the BOX and cladding-extent guides behind every device solid.
    # Panel a depicts the modeled substrate and truncated handle-wafer context as one continuous
    # Si solid. The dashed line marks where the 0.5 um solved substrate stops; the lower displayed
    # depth is deliberately not proportional to the approximately 725 um handle wafer.
    _add_box(
        axis,
        x_min=-half_x,
        x_max=half_x,
        y_min=-half_y,
        y_max=half_y,
        z_min=display_substrate_bottom,
        z_max=substrate_top,
        color=REGION_COLORS["silicon_substrate"],
        alpha=1.0,
        edgecolor=MATERIAL_FRAME_COLORS["si_substrate"],
        linewidth=0.28,
        zorder=10.0,
    )
    solved_boundary_style = {
        "color": DEVICE_OUTLINE_COLOR,
        "linewidth": 0.36,
        "linestyle": (0.0, (2.2, 1.5)),
        "alpha": 0.52,
        "zorder": 12.0,
    }
    axis.plot(
        (-half_x, half_x),
        (-half_y, -half_y),
        (substrate_bottom, substrate_bottom),
        **solved_boundary_style,
    )
    axis.plot(
        (half_x, half_x),
        (-half_y, half_y),
        (substrate_bottom, substrate_bottom),
        **solved_boundary_style,
    )
    _add_box(
        axis,
        x_min=-half_x,
        x_max=half_x,
        y_min=-half_y,
        y_max=half_y,
        z_min=substrate_top,
        z_max=silicon_bottom,
        color=REGION_COLORS["silica_box"],
        alpha=0.76,
        edgecolor=MATERIAL_FRAME_COLORS["sio2"],
        linewidth=0.32,
        shade=True,
        zorder=20.0,
    )

    # The solved upper silica remains volumetric, but panel a omits that solid so it cannot mask
    # the device. Two far-boundary faces mark its extent as a quiet 2D backdrop at the BOX layer.
    _add_faces(
        axis,
        [
            np.asarray(
                (
                    (-half_x, -half_y, silicon_bottom),
                    (-half_x, half_y, silicon_bottom),
                    (-half_x, half_y, cladding_top),
                    (-half_x, -half_y, cladding_top),
                ),
                dtype=np.float64,
            ),
            np.asarray(
                (
                    (-half_x, half_y, silicon_bottom),
                    (half_x, half_y, silicon_bottom),
                    (half_x, half_y, cladding_top),
                    (-half_x, half_y, cladding_top),
                ),
                dtype=np.float64,
            ),
        ],
        color=REGION_COLORS["silica_cladding"],
        alpha=0.09,
        edgecolor=MATERIAL_FRAME_COLORS["sio2"],
        linewidth=0.28,
        shade=False,
        zorder=21.0,
    )

    bus_half_width = recipe.waveguide_width_m * scale / 2.0
    bus_center = recipe.bus_center_y_m * scale

    def add_silicon_bus(center: float, *, zorder: float) -> None:
        _add_box(
            axis,
            x_min=-half_x,
            x_max=half_x,
            y_min=center - bus_half_width,
            y_max=center + bus_half_width,
            z_min=silicon_bottom,
            z_max=silicon_top,
            color=REGION_COLORS["silicon_ring"],
            alpha=1.0,
            edgecolor=MATERIAL_FRAME_COLORS["si_device"],
            linewidth=0.12,
            zorder=zorder,
        )

    # For the fixed panel-a camera, positive y is the far bus and negative y is the near bus.
    # Explicit opaque far-bus -> ring assembly -> near-bus painter layers preserve the intended
    # projected occlusion: the ring hides the rear bus and the front bus hides the ring.
    add_silicon_bus(bus_center, zorder=29.0)
    ring_inner = (recipe.ring_radius_m - recipe.waveguide_width_m / 2.0) * scale
    ring_outer = (recipe.ring_radius_m + recipe.waveguide_width_m / 2.0) * scale
    _add_annular_prism(
        axis,
        inner_radius=ring_inner,
        outer_radius=ring_outer,
        z_min=silicon_bottom,
        z_max=silicon_top,
        color=REGION_COLORS["silicon_ring"],
        alpha=1.0,
        zorder=30.0,
    )

    heater_inner = (recipe.ring_radius_m - recipe.heater_width_m / 2.0) * scale
    heater_outer = (recipe.ring_radius_m + recipe.heater_width_m / 2.0) * scale
    _add_annular_prism(
        axis,
        inner_radius=heater_inner,
        outer_radius=heater_outer,
        z_min=heater_bottom,
        z_max=heater_top,
        color=REGION_COLORS["tin_heater"],
        notch_half_x=recipe.heater_notch_x_m * scale / 2.0,
        alpha=1.0,
        zorder=40.0,
    )
    add_silicon_bus(-bus_center, zorder=41.0)

    contact_inner_x = recipe.heater_notch_x_m * scale / 2.0
    contact_width = recipe.contact_width_x_m * scale
    contact_y_min = (-recipe.ring_radius_m - recipe.contact_length_y_m / 2.0) * scale
    contact_y_max = contact_y_min + recipe.contact_length_y_m * scale
    for x_min in (-contact_inner_x - contact_width, contact_inner_x):
        _add_box(
            axis,
            x_min=x_min,
            x_max=x_min + contact_width,
            y_min=contact_y_min,
            y_max=contact_y_max,
            z_min=heater_top,
            z_max=cladding_top,
            color=REGION_COLORS["al_contact_negative"],
            alpha=0.97,
            edgecolor=MATERIAL_FRAME_COLORS["al"],
            linewidth=0.2,
            zorder=45.0,
        )

    for x_value, label in (
        (-contact_inner_x - contact_width / 2.0, "-"),
        (contact_inner_x + contact_width / 2.0, "+"),
    ):
        axis.text(
            x_value,
            (contact_y_min + contact_y_max) / 2.0,
            cladding_top - 0.03,
            label,
            color="#374151",
            fontsize=7.0,
            ha="center",
            va="top",
            zorder=60.0,
        )

    axis.set(
        xlim=(-half_x, half_x),
        ylim=(-half_y, half_y),
        zlim=(display_substrate_bottom, cladding_top),
    )
    axis.set_box_aspect(
        (
            2.0 * half_x,
            2.0 * half_y,
            cladding_top - display_substrate_bottom,
        ),
        zoom=0.98,
    )
    axis.set_proj_type("ortho")
    axis.view_init(elev=26.0, azim=-56.0, roll=0.0)
    axis.set_axis_off()
    figure = axis.get_figure()
    panel_title = figure.text(
        0.018,
        0.922,
        "a  Modeled material stack",
        ha="left",
        va="top",
        fontsize=7.4,
    )
    panel_title.set_in_layout(False)
    scale_note = figure.text(
        0.15,
        0.255,
        (
            "modeled layers 1:1 | solved $x,y=\\pm10$ µm (adiabatic)\n"
            "$z$ (µm): substrate 0.50 | BOX 2.00 | Si 0.22\n"
            "Si-to-TiN oxide gap 2.00 | TiN 0.14\n"
            "TiN-to-top SiO₂ 0.44 | Al vias 0.44 (same $z$ span)\n"
            "Si handle ≈725 µm; truncated (top 0.50 µm solved)"
        ),
        ha="center",
        va="center",
        fontsize=4.35,
    )
    scale_note.set_in_layout(False)
    material_legend = figure.legend(
        handles=(
            Patch(
                facecolor=REGION_COLORS["silicon_substrate"],
                edgecolor=MATERIAL_FRAME_COLORS["si_substrate"],
                label="Si substrate",
            ),
            Patch(
                facecolor=REGION_COLORS["silica_box"],
                edgecolor=MATERIAL_FRAME_COLORS["sio2"],
                label="SiO₂ BOX",
            ),
            Patch(
                facecolor=REGION_COLORS["silicon_ring"],
                edgecolor=MATERIAL_FRAME_COLORS["si_device"],
                label="Si ring + buses",
            ),
            Patch(
                facecolor=REGION_COLORS["tin_heater"],
                edgecolor=MATERIAL_FRAME_COLORS["tin"],
                label="TiN heater",
            ),
            Patch(
                facecolor=REGION_COLORS["al_contact_negative"],
                edgecolor=MATERIAL_FRAME_COLORS["al"],
                label="Al contacts",
            ),
            Patch(
                facecolor=REGION_COLORS["silica_cladding"],
                edgecolor=MATERIAL_FRAME_COLORS["sio2"],
                alpha=0.20,
                label="SiO₂ cladding (2D)",
            ),
        ),
        loc="upper center",
        bbox_to_anchor=MATERIAL_LEGEND_ANCHOR,
        ncol=2,
        columnspacing=MATERIAL_LEGEND_COLUMN_SPACING,
        handlelength=MATERIAL_LEGEND_HANDLE_LENGTH,
        handletextpad=MATERIAL_LEGEND_HANDLE_TEXT_PADDING,
        frameon=False,
        fontsize=5.0,
        title="Materials / geometry",
        title_fontsize=5.0,
    )
    material_legend.set_in_layout(False)
    return recipe


def _plot_optical_outline(axis: Any) -> None:
    for radius_um in (4.75, 5.25):
        circle = Circle(
            (0.0, 0.0),
            radius_um,
            fill=False,
            edgecolor=DEVICE_OUTLINE_COLOR,
            linewidth=DEVICE_OUTLINE_WIDTH_PT,
            alpha=0.64,
        )
        circle.set_path_effects(
            [
                pe.withStroke(
                    linewidth=OUTLINE_CONTRAST_STROKE_WIDTH_PT,
                    foreground=mpl.colors.to_rgba(OUTLINE_CONTRAST_COLOR, 0.62),
                )
            ]
        )
        axis.add_patch(circle)
    for center_y_um in (-5.6, 5.6):
        for edge_y_um in (center_y_um - 0.25, center_y_um + 0.25):
            line = axis.plot(
                (-10.0, 10.0),
                (edge_y_um, edge_y_um),
                color=DEVICE_OUTLINE_COLOR,
                linewidth=DEVICE_OUTLINE_WIDTH_PT,
                alpha=0.64,
            )[0]
            line.set_path_effects(
                [
                    pe.withStroke(
                        linewidth=OUTLINE_CONTRAST_STROKE_WIDTH_PT,
                        foreground=mpl.colors.to_rgba(OUTLINE_CONTRAST_COLOR, 0.62),
                    )
                ]
            )


def _annulus_y_intervals(
    *,
    fixed_x_um: float,
    inner_radius_um: float,
    outer_radius_um: float,
) -> tuple[tuple[float, float], ...]:
    absolute_x = abs(fixed_x_um)
    if absolute_x >= outer_radius_um:
        return ()
    outer_y = float(np.sqrt(outer_radius_um**2 - absolute_x**2))
    if absolute_x >= inner_radius_um:
        return ((-outer_y, outer_y),)
    inner_y = float(np.sqrt(inner_radius_um**2 - absolute_x**2))
    return ((-outer_y, -inner_y), (inner_y, outer_y))


def _subtract_y_interval(
    intervals: tuple[tuple[float, float], ...],
    *,
    cut_min: float,
    cut_max: float,
) -> tuple[tuple[float, float], ...]:
    retained: list[tuple[float, float]] = []
    for start, stop in intervals:
        if stop <= cut_min or start >= cut_max:
            retained.append((start, stop))
            continue
        if start < cut_min:
            retained.append((start, cut_min))
        if stop > cut_max:
            retained.append((cut_max, stop))
    return tuple(interval for interval in retained if interval[1] > interval[0])


def _plot_vertical_device_outline(
    axis: Any,
    recipe: PublicRingHeater3D,
    *,
    fixed_x_m: float,
) -> tuple[str, ...]:
    """Overlay the source-CAD solids intersected by one physical x plane."""

    scale = 1.0e6
    fixed_x_um = fixed_x_m * scale
    half_y = recipe.domain_y_m * scale / 2.0
    silicon_bottom = 0.0
    silicon_top = recipe.waveguide_height_m * scale
    heater_bottom = recipe.heater_bottom_z_m * scale
    heater_top = recipe.heater_top_z_m * scale
    visible_regions: list[str] = []

    def add_outline(
        y_min: float,
        y_max: float,
        z_min: float,
        z_max: float,
        *,
        alpha: float,
        linestyle: str | tuple[float, tuple[float, ...]],
    ) -> None:
        rectangle = Rectangle(
            (y_min, z_min),
            y_max - y_min,
            z_max - z_min,
            facecolor="none",
            edgecolor=VERTICAL_OUTLINE_COLOR,
            linewidth=VERTICAL_OUTLINE_WIDTH_PT,
            linestyle=linestyle,
            alpha=alpha,
            zorder=4.0,
        )
        rectangle.set_path_effects(
            [
                pe.withStroke(
                    linewidth=VERTICAL_OUTLINE_CONTRAST_STROKE_WIDTH_PT,
                    foreground=mpl.colors.to_rgba(VERTICAL_OUTLINE_CONTRAST_COLOR, 0.88),
                )
            ]
        )
        axis.add_patch(rectangle)

    add_outline(
        -half_y,
        half_y,
        recipe.substrate_bottom_z_m * scale,
        recipe.substrate_top_z_m * scale,
        alpha=0.52,
        linestyle=(0.0, (1.2, 1.4)),
    )
    visible_regions.append("silicon_substrate")

    ring_intervals = _annulus_y_intervals(
        fixed_x_um=fixed_x_um,
        inner_radius_um=(recipe.ring_radius_m - recipe.waveguide_width_m / 2.0) * scale,
        outer_radius_um=(recipe.ring_radius_m + recipe.waveguide_width_m / 2.0) * scale,
    )
    for y_min, y_max in ring_intervals:
        add_outline(y_min, y_max, silicon_bottom, silicon_top, alpha=0.88, linestyle="solid")
    if ring_intervals:
        visible_regions.append("silicon_ring")

    bus_half_width = recipe.waveguide_width_m * scale / 2.0
    bus_center = recipe.bus_center_y_m * scale
    for region_name, center in (
        ("silicon_bus_lower", -bus_center),
        ("silicon_bus_upper", bus_center),
    ):
        add_outline(
            center - bus_half_width,
            center + bus_half_width,
            silicon_bottom,
            silicon_top,
            alpha=0.88,
            linestyle="solid",
        )
        visible_regions.append(region_name)

    heater_intervals = _annulus_y_intervals(
        fixed_x_um=fixed_x_um,
        inner_radius_um=(recipe.ring_radius_m - recipe.heater_width_m / 2.0) * scale,
        outer_radius_um=(recipe.ring_radius_m + recipe.heater_width_m / 2.0) * scale,
    )
    if abs(fixed_x_um) <= recipe.heater_notch_x_m * scale / 2.0:
        notch_y_min = (-recipe.ring_radius_m - recipe.heater_notch_y_m / 2.0) * scale
        heater_intervals = _subtract_y_interval(
            heater_intervals,
            cut_min=notch_y_min,
            cut_max=notch_y_min + recipe.heater_notch_y_m * scale,
        )
    for y_min, y_max in heater_intervals:
        add_outline(
            y_min,
            y_max,
            heater_bottom,
            heater_top,
            alpha=0.92,
            linestyle=(0.0, (2.4, 1.8)),
        )
    if heater_intervals:
        visible_regions.append("tin_heater")

    contact_inner_x = recipe.heater_notch_x_m * scale / 2.0
    contact_width = recipe.contact_width_x_m * scale
    contact_y_min = (-recipe.ring_radius_m - recipe.contact_length_y_m / 2.0) * scale
    contact_y_max = contact_y_min + recipe.contact_length_y_m * scale
    for region_name, x_min in (
        ("al_contact_negative", -contact_inner_x - contact_width),
        ("al_contact_positive", contact_inner_x),
    ):
        if x_min <= fixed_x_um <= x_min + contact_width:
            add_outline(
                contact_y_min,
                contact_y_max,
                heater_top,
                recipe.cladding_top_z_m * scale,
                alpha=0.92,
                linestyle="dashdot",
            )
            visible_regions.append(region_name)

    return tuple(visible_regions)


def _plane_triangulation(section: PlaneSlice) -> mtri.Triangulation:
    return mtri.Triangulation(
        section.first_um,
        section.second_um,
        section.triangles,
    )


def _tripcolor(
    axis: Any,
    section: PlaneSlice,
    values: np.ndarray,
    *,
    cmap: str,
    norm: mpl.colors.Normalize,
) -> Any:
    return axis.tripcolor(
        _plane_triangulation(section),
        values,
        shading="gouraud",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )


def render_figure(
    output_dir: Path,
    solved: SolvedRing,
    run_id: str,
    figure_id: str,
    rendering_identity: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    """Render one final-size, full-range view of the retained direct solves."""

    target_current_a = float(solved.metrics["target_current_A"])
    target_current_ma = target_current_a * 1.0e3
    operating_point = dict(solved.provenance["model"]["operating_point"])
    temperature_difference = solved.elmer_temperature_k - solved.jax_temperature_k
    hottest_node = int(np.argmax(solved.jax_temperature_k))
    largest_error_node = int(np.argmax(np.abs(temperature_difference)))
    hottest_x_m = float(solved.coordinates_m[hottest_node, 0])
    largest_error_x_m = float(solved.coordinates_m[largest_error_node, 0])
    horizontal = _slice_tetrahedra(
        solved,
        fixed_axis=2,
        fixed_value_m=HORIZONTAL_SLICE_Z_M,
    )
    hot_vertical = _slice_tetrahedra(
        solved,
        fixed_axis=0,
        fixed_value_m=hottest_x_m,
    )
    error_vertical = _slice_tetrahedra(
        solved,
        fixed_axis=0,
        fixed_value_m=largest_error_x_m,
    )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 7.0,
            "axes.titlesize": 7.4,
            "axes.labelsize": 7.0,
            "axes.linewidth": 0.75,
            "lines.linewidth": 0.75,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "legend.fontsize": 6.0,
            "savefig.dpi": 300,
            "svg.fonttype": "none",
            "svg.hashsalt": figure_id,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    jax_rise = solved.jax_temperature_k - AMBIENT_TEMPERATURE_K
    elmer_rise = solved.elmer_temperature_k - AMBIENT_TEMPERATURE_K
    relative_l2_difference = float(
        np.linalg.norm(temperature_difference) / np.linalg.norm(jax_rise)
    )
    rise_limit = max(float(np.max(jax_rise)), float(np.max(elmer_rise)))
    difference_limit = max(float(np.max(np.abs(temperature_difference))), np.finfo(float).eps)
    rise_norm = mpl.colors.Normalize(vmin=0.0, vmax=rise_limit)
    difference_norm = mpl.colors.TwoSlopeNorm(
        vmin=-difference_limit,
        vcenter=0.0,
        vmax=difference_limit,
    )

    figure = plt.figure(figsize=(7.2, 4.8), constrained_layout=True)
    grid = GridSpec(2, 3, figure=figure, width_ratios=(1.28, 1.0, 1.0))
    material_axis = figure.add_subplot(
        grid[:, 0],
        projection="3d",
        computed_zorder=False,
    )
    horizontal_axis = figure.add_subplot(grid[0, 1])
    vertical_axis = figure.add_subplot(grid[0, 2])
    difference_axis = figure.add_subplot(grid[1, 1])
    parity_axis = figure.add_subplot(grid[1, 2])

    recipe = _plot_material_surfaces(material_axis, solved)
    temperature_artist = _tripcolor(
        horizontal_axis,
        horizontal,
        horizontal.jax_rise_k,
        cmap=TEMPERATURE_COLORMAP,
        norm=rise_norm,
    )
    _plot_optical_outline(horizontal_axis)
    horizontal_axis.set(
        xlim=(-10.0, 10.0),
        ylim=(-10.0, 10.0),
        aspect="equal",
        xlabel="$x$ (µm)",
        ylabel="$y$ (µm)",
        title=("b  JAX $\\Delta T$ at optical mid-plane\n$z=0.11$ µm; Si-device boundary overlaid"),
    )
    _tripcolor(
        vertical_axis,
        hot_vertical,
        hot_vertical.jax_rise_k,
        cmap=TEMPERATURE_COLORMAP,
        norm=rise_norm,
    )
    vertical_axis.set(
        xlim=(-10.0, 10.0),
        ylim=(-2.5, 2.8),
        xlabel="$y$ (µm)",
        ylabel="$z$ (µm)",
        title=f"c  JAX $\\Delta T$ through hottest node\n$x={hottest_x_m * 1.0e6:.3f}$ µm",
    )
    vertical_outline_regions = _plot_vertical_device_outline(
        vertical_axis,
        recipe,
        fixed_x_m=hottest_x_m,
    )
    vertical_axis.text(
        0.02,
        0.04,
        "$z$ enlarged",
        transform=vertical_axis.transAxes,
        color="white",
        fontsize=5.8,
        bbox={"facecolor": "#111827", "edgecolor": "none", "alpha": 0.55, "pad": 1.0},
    )
    temperature_colorbar = figure.colorbar(
        temperature_artist,
        ax=(horizontal_axis, vertical_axis),
        fraction=0.035,
        pad=0.02,
    )
    temperature_colorbar.set_label("JAX temperature rise $\\Delta T$ (K)")
    temperature_colorbar.ax.set_title(
        "b, c",
        fontsize=5.0,
        pad=3.0,
    )

    difference_artist = _tripcolor(
        difference_axis,
        error_vertical,
        error_vertical.difference_k,
        cmap=DIFFERENCE_COLORMAP,
        norm=difference_norm,
    )
    difference_axis.set(
        xlim=(-10.0, 10.0),
        ylim=(-2.5, 2.8),
        xlabel="$y$ (µm)",
        ylabel="$z$ (µm)",
        title=(
            "d  Elmer $-$ JAX on worst-error plane\n"
            f"$x={largest_error_x_m * 1.0e6:.3f}$ µm; symmetric full range"
        ),
    )
    difference_axis.text(
        0.02,
        0.04,
        "$z$ enlarged",
        transform=difference_axis.transAxes,
        color="#111827",
        fontsize=5.8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
    )
    difference_colorbar = figure.colorbar(
        difference_artist,
        ax=difference_axis,
        fraction=0.045,
        pad=0.02,
    )
    difference_colorbar.set_label("Temperature difference (K)")
    difference_colorbar.formatter.set_powerlimits((-2, 2))
    difference_colorbar.update_ticks()

    parity_axis.scatter(
        jax_rise,
        elmer_rise,
        s=3.0,
        alpha=0.34,
        color="#355C7D",
        edgecolors="none",
        rasterized=True,
    )
    parity_axis.plot((0.0, rise_limit), (0.0, rise_limit), color="#111827", linestyle="--")
    parity_axis.set(
        xlim=(0.0, rise_limit),
        ylim=(0.0, rise_limit),
        aspect="equal",
        xlabel="JAX $\\Delta T$ (K)",
        ylabel="Elmer $\\Delta T$ (K)",
    )
    parity_axis.set_title(
        f"e  All {solved.coordinates_m.shape[0]:,} thermal nodes",
        loc="left",
        pad=22.0,
    )
    parity_axis.text(
        0.0,
        1.018,
        (
            f"max |difference| = {difference_limit:.2e} K\n"
            f"relative L2 = {relative_l2_difference:.2e}"
        ),
        transform=parity_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=4.8,
        clip_on=False,
    )

    figure.suptitle(
        (
            f"Public 3D ring heater | {target_current_ma:g} mA direct solves | "
            "JAX-Elmer same-mesh parity"
        ),
        fontsize=8.0,
        fontweight="bold",
    )
    figure.text(
        0.995,
        0.004,
        (
            f"run-{run_id.rsplit('-', 1)[-1][:8]} / "
            f"fig-{figure_id.rsplit('-', 1)[-1][:8]}  |  "
            "retained direct fields; coarse uncalibrated model"
        ),
        ha="right",
        va="bottom",
        fontsize=4.8,
        color="#4B5563",
    )
    png_path = output_dir / "figure.png"
    svg_path = output_dir / "figure.svg"
    figure.savefig(
        png_path,
        dpi=300,
        metadata={"Software": "femx", "Description": f"{run_id} / {figure_id}"},
    )
    figure.savefig(
        svg_path,
        metadata={"Date": None, "Description": f"{run_id} / {figure_id}"},
    )
    plt.close(figure)
    svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    figure_metadata = {
        "run_id": run_id,
        "figure_id": figure_id,
        "rendering_identity": rendering_identity,
        "rendering_identity_sha256": _identity_digest(rendering_identity),
        "physical_size_inches": [7.2, 4.8],
        "png_pixels": [2160, 1440],
        "png_dpi": 300,
        "font": "7 pt sans serif",
        "line_width_pt": 0.75,
        "panels": {
            "a": (
                "source-pinned analytic CAD surfaces drawn in physical z proportion as silicon "
                "substrate within a continuous but depth-truncated handle-wafer depiction, full "
                "BOX plus two far-side 2D silica-cladding extent backdrops, silicon device, TiN "
                "heater, and aluminum contacts; no scalar field is overlaid"
            ),
            "b": (
                "direct-solve P1 JAX temperature-rise intersection at z=0.11 um with only the "
                "source-CAD Si-device boundary overlaid; no scalar isolines"
            ),
            "c": (
                "direct-solve P1 JAX temperature-rise intersection through the hottest node with "
                "the high-contrast source-CAD material silhouette"
            ),
            "d": (
                "signed direct-solve P1 Elmer-minus-JAX intersection through the largest-error node"
            ),
            "e": (
                "all-node direct-solve JAX-versus-Elmer temperature-rise parity with metrics "
                "outside the plotting area"
            ),
        },
        "operating_point": {
            "name": operating_point["name"],
            "target_current_A": target_current_a,
            "target_voltage_V": float(solved.metrics["target_voltage_V"]),
            "electrical_joule_power_W": float(solved.metrics["electrical_joule_power_W"]),
            "field_evaluation": solved.provenance["model"]["field_evaluation"],
            "field_evidence_tier": solved.provenance["model"]["field_evidence"]["evidence_tier"],
            "operating_point_selection_tier": operating_point["evidence_tier"],
            "algebraic_field_rescaling_used": False,
        },
        "temperature_colormap": TEMPERATURE_COLORMAP,
        "temperature_limits_K": [0.0, rise_limit],
        "temperature_palette_semantics": (
            "linear perceptually uniform sequential inferno; shared by panels b and c; "
            "complete retained range without clipping"
        ),
        "temperature_colormap_source": (
            "Matplotlib built-in inferno; exact Matplotlib version retained in provenance"
        ),
        "temperature_panel_scope": ["b", "c"],
        "panel_a_scalar_overlay": "none; categorical material geometry only",
        "horizontal_temperature_contours": (
            "none; omitted so scalar isolines cannot be confused with source-CAD boundaries"
        ),
        "horizontal_device_boundary_overlay": (
            "unfilled Si ring and bus boundaries in one muted-neutral line system; identified "
            "directly in the panel subtitle and never used to encode temperature"
        ),
        "difference_colormap": DIFFERENCE_COLORMAP,
        "difference_limits_K": [-difference_limit, difference_limit],
        "difference_palette_semantics": (
            "zero-centered ColorBrewer RdBu diverging map with red for positive Elmer-minus-JAX "
            "and blue for negative; symmetric complete retained range"
        ),
        "difference_colormap_source": (
            "Matplotlib built-in RdBu_r; exact Matplotlib version retained in provenance"
        ),
        "clipping": "none; every field scale uses the complete nodal range",
        "material_palette": {
            "id": MATERIAL_PALETTE_ID,
            "profile": MATERIAL_PALETTE_PROFILE,
            "policy": MATERIAL_PALETTE_POLICY,
            "fill_colors": dict(MATERIAL_FILL_COLORS),
            "frame_colors": dict(MATERIAL_FRAME_COLORS),
            "region_mapping": {
                "silicon_substrate": "si_substrate",
                "silicon_ring_and_buses": "si_device",
                "silica_box_and_cladding": "sio2",
                "tin_heater": "tin",
                "al_contacts": "al",
            },
            "role_modulation": (
                "BOX is a solid fill; upper-cladding extent uses the same SiO2 base colour "
                "with reduced opacity"
            ),
        },
        "surface_visibility": (
            "one continuous opaque Si substrate solid includes the solved top 0.5 um and a "
            "depth-truncated handle-wafer context below its dashed solve boundary; the full BOX "
            "is visible, while upper silica is reduced to faint x-min and y-max backdrops"
        ),
        "material_render_order": list(MATERIAL_RENDER_ORDER),
        "depth_sorting": (
            "manual painter order; cladding backdrops share the BOX background stage before "
            "device solids; Axes3D computed z-order disabled"
        ),
        "geometry_source": "femx.meshing.gmsh.PublicRingHeater3D",
        "geometry_recipe_sha256": recipe.digest(),
        "material_z_extents_um": {
            "domain": [
                round(recipe.substrate_bottom_z_m * 1.0e6, 12),
                round(recipe.cladding_top_z_m * 1.0e6, 12),
            ],
            "silicon_substrate": [
                round(recipe.substrate_bottom_z_m * 1.0e6, 12),
                round(recipe.substrate_top_z_m * 1.0e6, 12),
            ],
            "silica_box": [round(recipe.substrate_top_z_m * 1.0e6, 12), 0.0],
            "silicon_device": [0.0, round(recipe.waveguide_height_m * 1.0e6, 12)],
            "tin_heater": [
                round(recipe.heater_bottom_z_m * 1.0e6, 12),
                round(recipe.heater_top_z_m * 1.0e6, 12),
            ],
            "al_contacts": [
                round(recipe.heater_top_z_m * 1.0e6, 12),
                round(recipe.cladding_top_z_m * 1.0e6, 12),
            ],
            "silica_cladding": [0.0, round(recipe.cladding_top_z_m * 1.0e6, 12)],
        },
        "heater_vertical_gap_um": round(recipe.heater_vertical_gap_m * 1.0e6, 12),
        "substrate_display_extension_below_model_um": SUBSTRATE_HANDLE_DISPLAY_DEPTH_UM,
        "substrate_display_z_extent_um": [
            round(recipe.substrate_bottom_z_m * 1.0e6 - SUBSTRATE_HANDLE_DISPLAY_DEPTH_UM, 12),
            round(recipe.substrate_top_z_m * 1.0e6, 12),
        ],
        "substrate_display_policy": (
            "one continuous Si solid; its top 0.5 um is solved and the lower 1.5 um display "
            "segment is a depth-truncated, not-to-scale depiction of an approximately 725 um "
            "nominal handle wafer"
        ),
        "substrate_handle_context": {
            "material_continuity": "same Si fill and frame; no intervening layer",
            "modeled_top_thickness_um": round(recipe.substrate_thickness_m * 1.0e6, 12),
            "solve_boundary_z_um": round(recipe.substrate_bottom_z_m * 1.0e6, 12),
            "displayed_unsolved_depth_um": SUBSTRATE_HANDLE_DISPLAY_DEPTH_UM,
            "nominal_full_thickness_um": NOMINAL_HANDLE_WAFER_THICKNESS_UM,
            "nominal_thickness_qualifier": (
                "approximate contextual value; not source-pinned solved geometry"
            ),
            "display_scale_below_solve_boundary": "depth truncated; not to scale",
        },
        "silica_cladding_backdrop": {
            "kind": "two far-side two-dimensional extent planes",
            "planes": ["x_min", "y_max"],
            "z_extent_m": [0.0, recipe.cladding_top_z_m],
            "render_layer": "with silica BOX, before device solids",
            "volumetric_cladding_rendered": False,
            "physical_silica_volume_changed": False,
        },
        "physical_scale_policy": (
            "panel-a solved geometry uses a one-to-one micrometre drawing scale; the continuous "
            "Si handle below the dashed solve boundary is deliberately depth-truncated and not "
            "drawn in proportion to its approximately 725 um nominal thickness"
        ),
        "panel_a_projection": "orthographic axonometric",
        "panel_a_dimension_annotation": {
            "equal_physical_axis_scale": True,
            "lateral_extent_um": {
                "x": [
                    round(-recipe.domain_x_m * 0.5e6, 12),
                    round(recipe.domain_x_m * 0.5e6, 12),
                ],
                "y": [
                    round(-recipe.domain_y_m * 0.5e6, 12),
                    round(recipe.domain_y_m * 0.5e6, 12),
                ],
            },
            "lateral_boundary": "adiabatic sides",
            "z_thicknesses_um": {
                "silicon_substrate": round(recipe.substrate_thickness_m * 1.0e6, 12),
                "silica_box": round(recipe.buried_oxide_thickness_m * 1.0e6, 12),
                "silicon_device": round(recipe.waveguide_height_m * 1.0e6, 12),
                "silicon_to_tin_oxide_gap": round(recipe.heater_vertical_gap_m * 1.0e6, 12),
                "tin_heater": round(recipe.heater_height_m * 1.0e6, 12),
                "tin_to_model_top_silica": round(
                    (recipe.cladding_top_z_m - recipe.heater_top_z_m) * 1.0e6, 12
                ),
                "al_vias": round((recipe.cladding_top_z_m - recipe.heater_top_z_m) * 1.0e6, 12),
            },
            "tin_to_top_silica_and_al_vias_share_z_span": True,
            "scale_scope": "solved geometry only",
            "substrate_handle_nominal_thickness_um": NOMINAL_HANDLE_WAFER_THICKNESS_UM,
            "substrate_handle_depth_truncated": True,
        },
        "optical_outline_overlay": (
            "admitted analytic CAD boundaries for the 5 um-radius, 0.5 um-wide ring and buses; "
            "muted neutral dual-contrast lines without scalar encoding"
        ),
        "vertical_device_outline_overlay": {
            "panel": "c",
            "fixed_x_m": hottest_x_m,
            "geometry_source": "femx.meshing.gmsh.PublicRingHeater3D",
            "visible_regions": list(vertical_outline_regions),
            "style": (
                "unfilled white 0.50 pt outlines with a 1.18 pt dark contrast stroke; material "
                "identity additionally uses line style, never scalar colour"
            ),
        },
        "three_dimensional_z_display_exaggeration": 1.0,
        "horizontal_slice_z_m": HORIZONTAL_SLICE_Z_M,
        "hot_vertical_slice_x_m": hottest_x_m,
        "difference_vertical_slice_x_m": largest_error_x_m,
        "hottest_node_id": hottest_node,
        "largest_temperature_difference_node_id": largest_error_node,
        "heat_flux_overlay": "not displayed; no direction arrows are inferred from this figure",
        "exact_plotted_sources": [
            "nodes.csv direct JAX and Elmer fields",
            "cells.csv",
            "PublicRingHeater3D canonical geometry",
        ],
    }
    return png_path, svg_path, figure_metadata


def _write_bundle_readme(
    path: Path,
    solved: SolvedRing,
    run_id: str,
    figure_id: str,
) -> None:
    metrics = solved.metrics
    target_current_a = float(metrics["target_current_A"])
    target_current_ma = target_current_a * 1.0e3
    operating_point = dict(solved.provenance["model"]["operating_point"])
    role_description = (
        "source-pinned reproduction operating point"
        if operating_point["name"] == "source_reproduction"
        else "separately declared low-temperature operating point"
    )
    canonical_bundle = (
        "3d_ring_heater_reference"
        if operating_point["name"] == "source_reproduction"
        else "3d_ring_heater_5ma_reference"
    )
    content = f"""# Public 3D ring-heater reference

![3D ring-heater material stack and direct JAX-Elmer thermal fields](figure.png)

This bundle is figure `{figure_id}`, rendered from numerical run `{run_id}`. It uses the public
coarse ring recipe, one Gmsh mesh with {int(metrics["node_count"]):,} nodes and
{int(metrics["tetrahedron_count"]):,} first-order tetrahedra, and the {target_current_ma:g} mA
{role_description}. Native JAX and locked external Elmer independently solve the same discrete
current/Joule/heat problem at {float(metrics["target_voltage_V"]):.6f} V. No displayed temperature
or parity field is obtained by rescaling another operating point.

The JAX peak temperature rise is {float(metrics["maximum_temperature_rise_K"]):.3f} K. Across all
thermal nodes, the maximum direct-solve Elmer-JAX difference is
{float(metrics["maximum_absolute_temperature_difference_K"]):.3e} K and the temperature-rise
relative L2 difference is {float(metrics["relative_l2_temperature_rise_difference"]):.3e}.
Across the partial conductor field, the maximum potential difference is
{float(metrics["maximum_absolute_potential_difference_V"]):.3e} V.

## Files

- `nodes.csv`: every 3D coordinate and both nodal temperature fields;
- `cells.csv`: every Tet4 connectivity row and material-region identity;
- `potential.csv`: every conductor node and both potential fields;
- `generation.uv.lock`: the exact dependency lockfile recorded for the numerical run;
- `evidence.json`: thresholds, process/numerical/scientific states, source identities, hashes,
  figure rules, and raw-run references;
- `figure.svg` and `figure.png`: publication-scale vector container and 300 dpi preview.

## Reproduce

Re-render the presentation from the checked open fields without starting an external process:

```bash
uv run python examples/readme_3d_ring_heater_reference.py \\
  --render-existing docs/assets/readme/{canonical_bundle} \\
  --output /temporary/new/rendered-bundle
```

To rebuild the mesh, numerical fields, and presentation, run this from a locked femx checkout
with a clean locked Elmer source checkout:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run python \\
  examples/readme_3d_ring_heater_reference.py \\
  --allow-external \\
  --gmsh /absolute/path/to/gmsh \\
  --elmer /absolute/path/to/ElmerSolver \\
  --elmer-source /absolute/path/to/elmerfem \\
  --operating-point {operating_point["name"]} \\
  --output /temporary/empty/output/directory
```

The command starts external processes and therefore requires the explicit flag. It never downloads
or installs dependencies. Raw GEO/MSH, native Elmer mesh/SIF/result/VTU, and process logs are kept
under the ignored `.femx/readme-3d-ring-heater/` run directory and are bound by hashes in the
evidence file.

## Claim boundary

This is a same-discretization parity and conservation result for one coarse, constant-property,
uncalibrated public 3D benchmark. It is not formal mesh convergence, a continuum solution, a
physical TPU run, an FDTDX resonance response, a foundry model, or a fabricated-device prediction.
The 3D panel preserves the source-pinned `PublicRingHeater3D` layer elevations and renders the
materials explicitly from bottom to top: the 0.5 um modeled silicon substrate, the complete 2.0 um
SiO2 BOX, the 0.22 um silicon ring and buses, and the TiN heater and aluminum contacts. The panel
continues the modeled substrate downward as one uninterrupted Si solid rather than adding a second
plate. A dashed line marks the lower boundary of the solved 0.5 um Si region; below it, a short
depth-truncated segment stands for an approximately 725 um nominal handle wafer and is explicitly
not to scale or solved. The upper cladding volume is omitted from panel a; two faint far-side planes
mark its x-min and y-max extent as a 2D backdrop. Those planes are drawn with the BOX before every
device solid and are not rendered solids; these choices change only presentation, not the solved
geometry. Panel a uses an orthographic axonometric view, with one-to-one micrometre scale restricted
to solved geometry. Its annotation states the +/-10 um solved lateral extent and adiabatic sides and
includes the 0.44 um TiN-to-top silica and 0.44 um aluminum-via heights; those two features share
the same z span rather than stacking. Panel a contains categorical material geometry only: no
temperature field or isosurface is overlaid on the TiN or any other material. Panels b and c carry
the direct JAX temperature field on one shared linear `inferno` scale. Panel b deliberately omits
scalar isolines and identifies its single muted-neutral overlay as the source-CAD Si-device
boundary. Panel c uses
white device silhouettes with a thin dark contrast stroke so the geometry remains visible across
the complete inferno luminance range. Field axes retain physical coordinates and all color scales
use the complete retained range without clipping. Panels b and c use the linear perceptually
uniform `inferno` sequential map; the signed panel-d difference uses a zero-centered, symmetric `RdBu_r`
diverging scale. Panel-a materials use the bundled light-canvas categorical material palette
v1.1: Si substrate `#4B3F72`, device Si `#685AB8`, SiO2/TEOS `#167786`,
TiN `#604900`, and Al `#6F7885`, with their paired frame colours retained in figure metadata. BOX
and cladding extent share the same SiO2 fill and differ by opacity. Panels b and c add the matching
source-CAD geometry as restrained muted-neutral outlines with a thin dark contrast stroke; line
style distinguishes material roles in panel c, and outline colour never encodes a scalar value.
The content-addressed run identifier excludes presentation code; the figure identifier combines
that run identifier with the explicit rendering policy. A redraw can therefore change the figure
identifier without implying a new numerical solve.
"""
    path.write_text(content, encoding="utf-8", newline="\n")


def write_bundle(
    output_dir: Path,
    solved: SolvedRing,
    *,
    retained_table_source: Path | None = None,
) -> dict[str, Any]:
    """Write open numerical tables, the figure, and self-checking evidence."""

    output_dir.mkdir(parents=True, exist_ok=False)
    if retained_table_source is None:
        _write_nodes_csv(output_dir / "nodes.csv", solved)
        _write_cells_csv(output_dir / "cells.csv", solved)
        _write_potential_csv(output_dir / "potential.csv", solved)
        (output_dir / GENERATION_LOCK_FILENAME).write_bytes(_git_head_file_bytes("uv.lock"))
    else:
        for name in ("nodes.csv", "cells.csv", "potential.csv"):
            shutil.copyfile(retained_table_source / name, output_dir / name)
        shutil.copyfile(
            retained_table_source / GENERATION_LOCK_FILENAME,
            output_dir / GENERATION_LOCK_FILENAME,
        )
    expected_lock_sha256 = solved.provenance["femx"]["uv_lock_sha256"]
    if sha256_file(output_dir / GENERATION_LOCK_FILENAME) != expected_lock_sha256:
        raise RuntimeError("generation lock does not match the numerical-run provenance")
    solved.provenance["femx"]["uv_lock_artifact"] = GENERATION_LOCK_FILENAME
    run_identity = _run_identity(output_dir, solved)
    run_id = _run_identifier(run_identity)
    rendering_identity = _figure_rendering_identity()
    figure_id = _figure_identifier(run_id, rendering_identity)
    _, _, figure_metadata = render_figure(
        output_dir,
        solved,
        run_id,
        figure_id,
        rendering_identity,
    )
    _write_bundle_readme(
        output_dir / "README.md",
        solved,
        run_id,
        figure_id,
    )

    artifact_names = (
        "README.md",
        "nodes.csv",
        "cells.csv",
        "potential.csv",
        GENERATION_LOCK_FILENAME,
        "figure.png",
        "figure.svg",
    )
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "figure_id": figure_id,
        "identity": {
            "run": run_identity,
            "figure": {
                "schema_version": FIGURE_IDENTITY_SCHEMA_VERSION,
                "run_id": run_id,
                "rendering": rendering_identity,
            },
            "policy": (
                "run_id identifies retained numerical fields and their field generator; "
                "figure_id identifies that run plus explicit rendering rules"
            ),
        },
        "status": {
            "gmsh_process": "succeeded",
            "jax_unit_solve": "numerically_admitted",
            "jax_target_solve": "numerically_admitted",
            "elmer_process": "succeeded",
            "elmer_convergence": "converged",
            "scientific_parity": "passed",
        },
        "metrics": solved.metrics,
        "thresholds": PARITY_THRESHOLDS,
        "figure": figure_metadata,
        "provenance": solved.provenance,
        "artifacts": {name: sha256_file(output_dir / name) for name in artifact_names},
        "claim_scope": (
            "same-discretization JAX/Elmer field parity and conservation on one coarse public 3D "
            "ring-heater benchmark; not formal convergence, TPU, FDTDX response, foundry "
            "calibration, measurement, or fabricated-device prediction"
        ),
    }
    (output_dir / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gmsh", type=Path, help="absolute Gmsh executable path")
    parser.add_argument(
        "--elmer",
        type=Path,
        help="absolute ElmerSolver executable path",
    )
    parser.add_argument(
        "--elmer-source",
        type=Path,
        help="optional absolute locked Elmer source-checkout path",
    )
    parser.add_argument("--output", required=True, type=Path, help="new empty bundle directory")
    parser.add_argument(
        "--render-existing",
        type=Path,
        help="existing checked bundle to re-render from its retained CSV fields",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=REPOSITORY_ROOT / ".femx" / "readme-3d-ring-heater",
        help="ignored parent for immutable raw attempts",
    )
    parser.add_argument(
        "--operating-point",
        choices=("source_reproduction", "low_temperature_projection"),
        default="source_reproduction",
        help="target current role; both JAX and Elmer are run directly at the selected current",
    )
    parser.add_argument(
        "--allow-external",
        action="store_true",
        help="explicitly authorize local Gmsh, Git source-check, and Elmer processes",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    if arguments.output.exists():
        raise SystemExit("output must name a path that does not yet exist")
    if arguments.render_existing is not None:
        if any(
            value is not None for value in (arguments.gmsh, arguments.elmer, arguments.elmer_source)
        ) or bool(arguments.allow_external):
            raise SystemExit("render-existing cannot be combined with external-solver options")
        if arguments.operating_point != "source_reproduction":
            raise SystemExit("render-existing reads its operating point from the retained bundle")
        if not arguments.render_existing.is_dir():
            raise SystemExit("render-existing must name an existing bundle directory")
        solved = load_solved_ring_from_bundle(arguments.render_existing)
        evidence = write_bundle(
            arguments.output.resolve(),
            solved,
            retained_table_source=arguments.render_existing.resolve(),
        )
        print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
        return 0
    if not arguments.allow_external:
        raise SystemExit("external execution is disabled; pass --allow-external explicitly")
    if arguments.gmsh is None or arguments.elmer is None:
        raise SystemExit("full regeneration requires both --gmsh and --elmer")
    for label, executable in (("Gmsh", arguments.gmsh), ("ElmerSolver", arguments.elmer)):
        if not executable.is_absolute() or not executable.is_file():
            raise SystemExit(f"{label} executable must be an existing absolute file")
    if arguments.elmer_source is not None:
        if not arguments.elmer_source.is_absolute() or not arguments.elmer_source.is_dir():
            raise SystemExit("Elmer source checkout must be an existing absolute directory")
    solved = solve_case(
        gmsh_executable=arguments.gmsh,
        elmer_executable=arguments.elmer,
        elmer_source=arguments.elmer_source,
        run_root=arguments.run_root.resolve(),
        operating_point_name=arguments.operating_point,
    )
    evidence = write_bundle(arguments.output.resolve(), solved)
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
