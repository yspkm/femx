#!/usr/bin/env python3
"""Generate the README Silicon Photonics thermal FEM reference bundle.

The case is deliberately narrow: a public thermally tuned ring-resonator design is reduced to
its x=0 vertical solid section. Gmsh creates one canonical P1 triangle mesh, then the native JAX
and separately installed Elmer backends solve the same steady-heat problem on those exact nodes
and elements. The committed figure is a presentation of the retained CSV data, not a substitute
for the numerical checks performed here.

This script requires explicit permission because it starts Gmsh, Git source checks, and Elmer.
It never downloads, installs, or vendors any of them.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "femx-matplotlib"))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Patch, Rectangle, Wedge  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from femx.artifacts import sha256_file  # noqa: E402
from femx.backends.elmer.runner import ElmerInstallation  # noqa: E402
from femx.backends.elmer.steady_heat import (  # noqa: E402
    ElmerSteadyHeatBackend,
    ElmerSteadyHeatIdentity,
)
from femx.backends.jax.operators import solve_steady_heat, triangle_p1_geometry  # noqa: E402
from femx.backends.jax.steady_heat import JaxSteadyHeatBackend  # noqa: E402
from femx.backends.protocol import PrepareRequest, SolveRequest  # noqa: E402
from femx.core.capabilities import GradientMethod  # noqa: E402
from femx.core.execution import ExecutionPolicy  # noqa: E402
from femx.core.parameters import (  # noqa: E402
    ParameterReference,
    ParameterRole,
    ParameterSchema,
    ParameterSpec,
)
from femx.core.problem import Problem  # noqa: E402
from femx.meshing.gmsh import (  # noqa: E402
    GmshInstallation,
    GmshMeshingRequest,
    GmshRunner,
    read_gmsh_msh,
)
from femx.physics import (  # noqa: E402
    HeatFluxBoundary,
    SteadyHeat,
    TemperatureBoundary,
    ThermalRegion,
)
from femx.runtime import prepare, solve  # noqa: E402

SOURCE_PAGE = "https://www.flexcompute.com/tidy3d/examples/notebooks/ThermallyTunedRingResonator/"
SOURCE_REPOSITORY = "https://github.com/flexcompute/tidy3d-notebooks"
SCHEMA_VERSION = "femx.figure.siph-thermal-reference/v1"
AMBIENT_TEMPERATURE_K = 300.0
CURRENT_A = 15.0e-3
TIN_ELECTRICAL_CONDUCTIVITY_S_PER_M = 2.3e6
HEATER_WIDTH_M = 2.0e-6
HEATER_HEIGHT_M = 0.14e-6

Y_COORDINATES_UM = (
    -10.0,
    -6.0,
    -5.85,
    -5.35,
    -5.25,
    -4.75,
    -4.0,
    0.0,
    4.0,
    4.75,
    5.25,
    5.35,
    5.85,
    6.0,
    10.0,
)
Z_COORDINATES_UM = (-2.5, -2.0, 0.0, 0.22, 2.22, 2.36, 2.8)

REGION_ORDER = (
    "silica",
    "silicon_wafer",
    "silicon_ring",
    "silicon_buses",
    "tin_heater",
)
REGION_CONDUCTIVITY_W_PER_MK = {
    "silica": 1.38,
    "silicon_wafer": 148.0,
    "silicon_ring": 148.0,
    "silicon_buses": 148.0,
    "tin_heater": 28.0,
}
REGION_COLORS = {
    "silica": "#B3CDE3",
    "silicon_wafer": "#6B7280",
    "silicon_ring": "#009E73",
    "silicon_buses": "#009E73",
    "tin_heater": "#E69F00",
}

PARITY_THRESHOLDS = {
    "maximum_absolute_temperature_difference_K": 1.0e-7,
    "relative_l2_temperature_rise_difference": 1.0e-10,
    "jax_relative_residual": 1.0e-10,
    "jax_discrete_energy_balance_relative_error": 1.0e-10,
    "adjoint_vs_elmer_fd_relative_error": 5.0e-5,
}


@dataclass(frozen=True, slots=True)
class SolvedCase:
    """Arrays and evidence needed to write the public figure bundle."""

    coordinates_m: np.ndarray
    cells: np.ndarray
    region_ids: np.ndarray
    region_names: tuple[str, ...]
    jax_temperature_K: np.ndarray
    elmer_temperature_K: np.ndarray
    cell_heat_flux_W_per_m2: np.ndarray
    metrics: dict[str, float]
    provenance: dict[str, Any]


def _format_real(value: float) -> str:
    return format(float(value), ".17g")


def _surface_region(y_mid_um: float, z_mid_um: float) -> str:
    if z_mid_um < -2.0:
        return "silicon_wafer"
    if 0.0 < z_mid_um < 0.22:
        absolute_y = abs(y_mid_um)
        if 4.75 < absolute_y < 5.25:
            return "silicon_ring"
        if 5.35 < absolute_y < 5.85:
            return "silicon_buses"
    if 4.0 < y_mid_um < 6.0 and 2.22 < z_mid_um < 2.36:
        return "tin_heater"
    return "silica"


def render_gmsh_geometry() -> str:
    """Return a deterministic boundary-aligned Gmsh model in micrometre coordinates."""

    lines = [
        "// femx README SiPh thermal reference; coordinate unit = 1e-6 m",
        'SetFactory("Built-in");',
        "lc = 0.65;",
        "",
    ]
    point_ids: dict[tuple[int, int], int] = {}
    next_point = 1
    for iz, z_um in enumerate(Z_COORDINATES_UM):
        for iy, y_um in enumerate(Y_COORDINATES_UM):
            point_ids[(iy, iz)] = next_point
            lines.append(
                f"Point({next_point}) = {{{_format_real(y_um)}, {_format_real(z_um)}, 0, lc}};"
            )
            next_point += 1

    lines.append("")
    horizontal_ids: dict[tuple[int, int], int] = {}
    vertical_ids: dict[tuple[int, int], int] = {}
    next_curve = 1
    for iz in range(len(Z_COORDINATES_UM)):
        for iy in range(len(Y_COORDINATES_UM) - 1):
            horizontal_ids[(iy, iz)] = next_curve
            lines.append(
                f"Line({next_curve}) = {{{point_ids[(iy, iz)]}, {point_ids[(iy + 1, iz)]}}};"
            )
            next_curve += 1
    for iy in range(len(Y_COORDINATES_UM)):
        for iz in range(len(Z_COORDINATES_UM) - 1):
            vertical_ids[(iy, iz)] = next_curve
            lines.append(
                f"Line({next_curve}) = {{{point_ids[(iy, iz)]}, {point_ids[(iy, iz + 1)]}}};"
            )
            next_curve += 1

    lines.append("")
    surfaces_by_region: dict[str, list[int]] = {name: [] for name in REGION_ORDER}
    next_loop = 1001
    next_surface = 2001
    for iz in range(len(Z_COORDINATES_UM) - 1):
        for iy in range(len(Y_COORDINATES_UM) - 1):
            bottom = horizontal_ids[(iy, iz)]
            right = vertical_ids[(iy + 1, iz)]
            top = horizontal_ids[(iy, iz + 1)]
            left = vertical_ids[(iy, iz)]
            lines.append(f"Curve Loop({next_loop}) = {{{bottom}, {right}, -{top}, -{left}}};")
            lines.append(f"Plane Surface({next_surface}) = {{{next_loop}}};")
            y_mid = 0.5 * (Y_COORDINATES_UM[iy] + Y_COORDINATES_UM[iy + 1])
            z_mid = 0.5 * (Z_COORDINATES_UM[iz] + Z_COORDINATES_UM[iz + 1])
            surfaces_by_region[_surface_region(y_mid, z_mid)].append(next_surface)
            next_loop += 1
            next_surface += 1

    lines.append("")
    for physical_id, name in enumerate(REGION_ORDER, start=101):
        encoded = ", ".join(str(surface) for surface in surfaces_by_region[name])
        lines.append(f'Physical Surface("{name}", {physical_id}) = {{{encoded}}};')
    bottom_curves = [horizontal_ids[(iy, 0)] for iy in range(len(Y_COORDINATES_UM) - 1)]
    top_curves = [
        horizontal_ids[(iy, len(Z_COORDINATES_UM) - 1)] for iy in range(len(Y_COORDINATES_UM) - 1)
    ]
    left_curves = [vertical_ids[(0, iz)] for iz in range(len(Z_COORDINATES_UM) - 1)]
    right_curves = [
        vertical_ids[(len(Y_COORDINATES_UM) - 1, iz)] for iz in range(len(Z_COORDINATES_UM) - 1)
    ]
    lines.append(
        'Physical Curve("bottom_sink", 201) = {'
        + ", ".join(str(curve) for curve in bottom_curves)
        + "};"
    )
    lines.append(
        'Physical Curve("adiabatic", 202) = {'
        + ", ".join(str(curve) for curve in (*top_curves, *left_curves, *right_curves))
        + "};"
    )
    lines.extend(
        (
            "",
            "Field[1] = Box;",
            "Field[1].VIn = 0.075;",
            "Field[1].VOut = 0.65;",
            "Field[1].XMin = -6.10;",
            "Field[1].XMax = 6.10;",
            "Field[1].YMin = -0.12;",
            "Field[1].YMax = 0.36;",
            "Field[1].Thickness = 0.55;",
            "Field[2] = Box;",
            "Field[2].VIn = 0.065;",
            "Field[2].VOut = 0.65;",
            "Field[2].XMin = 3.80;",
            "Field[2].XMax = 6.20;",
            "Field[2].YMin = 2.05;",
            "Field[2].YMax = 2.52;",
            "Field[2].Thickness = 0.65;",
            "Field[3] = Min;",
            "Field[3].FieldsList = {1, 2};",
            "Background Field = 3;",
            "Mesh.MeshSizeExtendFromBoundary = 0;",
            "Mesh.MeshSizeFromCurvature = 0;",
            "Mesh.MshFileVersion = 4.1;",
            "Mesh.Binary = 0;",
            "Mesh.ElementOrder = 1;",
            "Mesh.SaveAll = 0;",
            "Mesh.Algorithm = 6;",
            "Mesh.AlgorithmSwitchOnFailure = 0;",
            "Mesh.RandomFactor = 1e-9;",
            "Mesh.RandomSeed = 1;",
            "",
        )
    )
    return "\n".join(lines)


def _source_report() -> dict[str, Any]:
    command = (
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "check_source_checkouts.py"),
        "--source",
        "elmer",
        "--json",
    )
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Elmer source check failed: {completed.stderr.strip()}")
    payload = json.loads(completed.stdout)
    if not payload.get("valid") or len(payload.get("sources", [])) != 1:
        raise RuntimeError("Elmer source identity is not valid")
    return dict(payload["sources"][0])


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


def _git_head_file_sha256(relative_path: str) -> str:
    completed = subprocess.run(
        ("git", "show", f"HEAD:{relative_path}"),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        shell=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"could not read {relative_path!r} from the femx Git commit")
    return hashlib.sha256(completed.stdout).hexdigest()


def _make_problem(mesh: Any, *, gradient_method: GradientMethod) -> tuple[Problem, Any]:
    schema = ParameterSchema(
        (
            ParameterSpec(
                "heater_source",
                unit="W/m^3",
                role=ParameterRole.CONTROL,
                lower_bound=0.0,
            ),
        )
    )
    source = _volumetric_heat_source(CURRENT_A)
    physics = SteadyHeat(
        regions=tuple(
            ThermalRegion(
                name,
                REGION_CONDUCTIVITY_W_PER_MK[name],
                ParameterReference("heater_source") if name == "tin_heater" else 0.0,
            )
            for name in REGION_ORDER
        ),
        temperature_boundaries=(TemperatureBoundary("bottom_sink", AMBIENT_TEMPERATURE_K),),
        heat_flux_boundaries=(HeatFluxBoundary("adiabatic", 0.0),),
        gradient_method=gradient_method,
    )
    parameters = schema.bind({"heater_source": source})
    return Problem("readme-siph-thermal-reference", mesh, physics, parameters=schema), parameters


def _volumetric_heat_source(current_A: float) -> float:
    current_density = current_A / (HEATER_WIDTH_M * HEATER_HEIGHT_M)
    return current_density**2 / TIN_ELECTRICAL_CONDUCTIVITY_S_PER_M


def _region_arrays(mesh: Any) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    cell_count = int(mesh.topology.cell_count)
    region_ids = np.full(cell_count, -1, dtype=np.int64)
    region_names = tuple(REGION_ORDER)
    conductivity = np.empty(cell_count, dtype=np.float64)
    for region_id, name in enumerate(region_names):
        cells = np.asarray(mesh.tag(name).entity_ids, dtype=np.int64)
        if np.any(region_ids[cells] != -1):
            raise RuntimeError(f"region tag {name!r} overlaps a previous region")
        region_ids[cells] = region_id
        conductivity[cells] = REGION_CONDUCTIVITY_W_PER_MK[name]
    if np.any(region_ids == -1):
        raise RuntimeError("material tags do not partition the canonical mesh")
    return region_ids, region_names, conductivity


def _node_area_weights(
    cells: np.ndarray,
    cell_areas: np.ndarray,
    selected_cells: np.ndarray,
    node_count: int,
) -> np.ndarray:
    weights = np.zeros(node_count, dtype=np.float64)
    local = np.repeat(cell_areas[selected_cells] / 3.0, 3)
    np.add.at(weights, cells[selected_cells].reshape(-1), local)
    total = float(weights.sum())
    if total <= 0.0:
        raise RuntimeError("observable region has zero area")
    return weights / total


def _solve_elmer(
    problem: Problem,
    parameters: Any,
    backend: ElmerSteadyHeatBackend,
    run_directory: Path,
) -> Any:
    request_policy = ExecutionPolicy(execution_authorized=True, allow_external_process=True)
    return solve(
        prepare(problem, backend, request=PrepareRequest(run_directory=run_directory)),
        backend,
        request=SolveRequest(
            parameters=parameters,
            run_directory=run_directory,
            policy=request_policy,
        ),
    )


def solve_case(
    *,
    gmsh_executable: Path,
    elmer_executable: Path,
    run_root: Path,
) -> SolvedCase:
    """Generate the canonical mesh and execute JAX, Elmer, and gradient checks."""

    run_root.mkdir(parents=True, exist_ok=False)
    meshing_directory = run_root / "gmsh"
    meshing_directory.mkdir()
    geometry_path = meshing_directory / "siph_thermal_reference.geo"
    geometry_path.write_text(render_gmsh_geometry(), encoding="utf-8", newline="\n")

    execution_policy = ExecutionPolicy(execution_authorized=True, allow_external_process=True)
    gmsh_result = GmshRunner(GmshInstallation(gmsh_executable.resolve())).run(
        GmshMeshingRequest(geometry_path.name),
        working_directory=meshing_directory,
        policy=execution_policy,
    )
    (meshing_directory / "gmsh.stdout.log").write_text(
        gmsh_result.stdout, encoding="utf-8", newline="\n"
    )
    (meshing_directory / "gmsh.stderr.log").write_text(
        gmsh_result.stderr, encoding="utf-8", newline="\n"
    )
    if not gmsh_result.process_succeeded:
        raise RuntimeError(f"Gmsh failed: {gmsh_result.stderr.strip()}")
    imported = read_gmsh_msh(meshing_directory / "mesh.msh", coordinate_scale_to_m=1.0e-6)
    mesh = imported.mesh
    coordinates = np.asarray(mesh.geometry.coordinates, dtype=np.float64)
    cells = np.asarray(mesh.topology.connectivity, dtype=np.int64)
    facets = np.asarray(mesh.boundary_facets.connectivity, dtype=np.int64)
    region_ids, region_names, cell_conductivity = _region_arrays(mesh)

    jax_problem, baseline_parameters = _make_problem(mesh, gradient_method=GradientMethod.ADJOINT)
    elmer_problem, _ = _make_problem(mesh, gradient_method=GradientMethod.NONE)
    problem_payload = {
        "mesh_sha256": imported.record.canonical_mesh_sha256,
        "physics": jax_problem.physics.canonical_data(),
        "parameters": {"heater_source": _volumetric_heat_source(CURRENT_A)},
    }
    problem_digest = hashlib.sha256(
        json.dumps(problem_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    jax_backend = JaxSteadyHeatBackend(relative_residual_tolerance=1.0e-10)
    prepared_jax = prepare(jax_problem, jax_backend)
    jax_solution = solve(
        prepared_jax,
        jax_backend,
        request=SolveRequest(parameters=baseline_parameters),
    )
    if jax_solution.convergence.status.value != "converged":
        raise RuntimeError(f"JAX heat solve did not converge: {jax_solution.convergence}")
    jax_temperature = np.asarray(
        jax.device_get(jax_solution.fields["temperature"].values), dtype=np.float64
    )

    source_report = _source_report()
    heat_solve_module = (
        elmer_executable.resolve().parent.parent / "share" / "elmersolver" / "lib" / "HeatSolve.so"
    )
    identity = ElmerSteadyHeatIdentity(
        version="26.2-devel",
        revision=str(source_report["head_commit"])[:9],
        executable_sha256=sha256_file(elmer_executable.resolve()),
        heat_solve_sha256=sha256_file(heat_solve_module),
        source_commit=str(source_report["head_commit"]),
        source_digest=str(source_report["source_digest"]),
        source_worktree_state=str(source_report["worktree_state"]),
    )
    elmer_backend = ElmerSteadyHeatBackend(
        ElmerInstallation(elmer_executable.resolve()),
        identity,
    )
    elmer_baseline = _solve_elmer(
        elmer_problem,
        baseline_parameters,
        elmer_backend,
        run_root / "elmer-baseline",
    )
    if elmer_baseline.convergence.status.value != "converged":
        raise RuntimeError(f"Elmer heat solve did not converge: {elmer_baseline.convergence}")
    elmer_temperature = np.asarray(elmer_baseline.fields["temperature"].values, dtype=np.float64)

    cell_areas_jax, cell_gradients_jax = triangle_p1_geometry(
        jnp.asarray(coordinates), jnp.asarray(cells, dtype=jnp.int32)
    )
    cell_areas = np.asarray(jax.device_get(cell_areas_jax), dtype=np.float64)
    cell_gradients = np.asarray(jax.device_get(cell_gradients_jax), dtype=np.float64)
    ring_cells = np.asarray(mesh.tag("silicon_ring").entity_ids, dtype=np.int64)
    ring_weights = _node_area_weights(
        cells,
        cell_areas,
        ring_cells,
        coordinates.shape[0],
    )

    differentiable = jax_backend.bind_differentiable(prepared_jax, baseline_parameters)
    initial = differentiable.initial_values
    cotangent = jnp.asarray(ring_weights, dtype=jnp.float64)
    vjp = differentiable.vjp(initial, cotangent)
    source_gradient = float(jax.device_get(vjp.parameter_gradient[0]))

    def ring_objective(active: jax.Array) -> jax.Array:
        return jnp.dot(cotangent, differentiable.temperature(active)) - AMBIENT_TEMPERATURE_K

    reverse_gradient = float(jax.device_get(jax.grad(ring_objective)(initial)[0]))
    current_step_A = CURRENT_A * 1.0e-4

    def elmer_ring_objective(current_A: float, run_name: str) -> float:
        parameters = elmer_problem.parameters.bind(
            {"heater_source": _volumetric_heat_source(current_A)}
        )
        solution = _solve_elmer(
            elmer_problem,
            parameters,
            elmer_backend,
            run_root / run_name,
        )
        if solution.convergence.status.value != "converged":
            raise RuntimeError(f"Elmer finite-difference solve did not converge: {run_name}")
        temperature = np.asarray(solution.fields["temperature"].values, dtype=np.float64)
        return float(ring_weights @ temperature - AMBIENT_TEMPERATURE_K)

    elmer_plus = elmer_ring_objective(CURRENT_A + current_step_A, "elmer-fd-plus")
    elmer_minus = elmer_ring_objective(CURRENT_A - current_step_A, "elmer-fd-minus")
    elmer_current_gradient = (elmer_plus - elmer_minus) / (2.0 * current_step_A)
    source_chain_rule = 2.0 * _volumetric_heat_source(CURRENT_A) / CURRENT_A
    jax_current_gradient = source_gradient * source_chain_rule

    cell_source = np.zeros(cells.shape[0], dtype=np.float64)
    heater_cells = np.asarray(mesh.tag("tin_heater").entity_ids, dtype=np.int64)
    cell_source[heater_cells] = _volumetric_heat_source(CURRENT_A)
    facet_heat_load = np.zeros(facets.shape[0], dtype=np.float64)
    bottom_facets = np.asarray(mesh.tag("bottom_sink").entity_ids, dtype=np.int64)
    bottom_nodes = np.unique(facets[bottom_facets].reshape(-1))
    bottom_values = np.full(bottom_nodes.shape, AMBIENT_TEMPERATURE_K, dtype=np.float64)
    assembled_temperature, unconstrained_system = solve_steady_heat(
        jnp.asarray(coordinates),
        jnp.asarray(cells, dtype=jnp.int32),
        jnp.asarray(cell_conductivity),
        jnp.asarray(cell_source),
        jnp.asarray(facets, dtype=jnp.int32),
        jnp.asarray(facet_heat_load),
        jnp.asarray(bottom_nodes, dtype=jnp.int32),
        jnp.asarray(bottom_values),
    )
    assembled_temperature_np = np.asarray(jax.device_get(assembled_temperature))
    np.testing.assert_allclose(assembled_temperature_np, jax_temperature, rtol=0.0, atol=1.0e-11)
    reaction = np.asarray(
        jax.device_get(
            unconstrained_system.stiffness @ assembled_temperature - unconstrained_system.load
        )
    )
    input_line_power = float(np.sum(cell_source * cell_areas))
    bottom_outward_line_power = float(-np.sum(reaction[bottom_nodes]))
    energy_error = abs(bottom_outward_line_power - input_line_power) / input_line_power

    cell_temperature_gradient = np.einsum("ci,cid->cd", jax_temperature[cells], cell_gradients)
    cell_heat_flux = -cell_conductivity[:, None] * cell_temperature_gradient
    difference = elmer_temperature - jax_temperature
    reference_rise = elmer_temperature - AMBIENT_TEMPERATURE_K
    ring_mean_jax = float(ring_weights @ jax_temperature - AMBIENT_TEMPERATURE_K)
    ring_mean_elmer = float(ring_weights @ elmer_temperature - AMBIENT_TEMPERATURE_K)
    adjoint_fd_scale = max(abs(jax_current_gradient), abs(elmer_current_gradient), 1.0e-30)
    metrics = {
        "ambient_temperature_K": AMBIENT_TEMPERATURE_K,
        "current_A": CURRENT_A,
        "volumetric_heat_source_W_per_m3": _volumetric_heat_source(CURRENT_A),
        "input_line_power_W_per_m": input_line_power,
        "node_count": float(coordinates.shape[0]),
        "triangle_count": float(cells.shape[0]),
        "maximum_temperature_rise_K": float(np.max(jax_temperature) - AMBIENT_TEMPERATURE_K),
        "ring_mean_temperature_rise_jax_K": ring_mean_jax,
        "ring_mean_temperature_rise_elmer_K": ring_mean_elmer,
        "maximum_absolute_temperature_difference_K": float(np.max(np.abs(difference))),
        "rms_temperature_difference_K": float(np.sqrt(np.mean(difference**2))),
        "relative_l2_temperature_rise_difference": float(
            np.linalg.norm(difference) / np.linalg.norm(reference_rise)
        ),
        "jax_relative_residual": float(jax_solution.convergence.residual_norm),
        "jax_discrete_energy_balance_relative_error": energy_error,
        "jax_adjoint_backward_error": float(jax.device_get(vjp.adjoint_backward_error)),
        "jax_reverse_vs_explicit_adjoint_relative_error": abs(reverse_gradient - source_gradient)
        / max(abs(reverse_gradient), abs(source_gradient), 1.0e-30),
        "jax_adjoint_ring_temperature_gradient_K_per_A": jax_current_gradient,
        "elmer_central_fd_ring_temperature_gradient_K_per_A": elmer_current_gradient,
        "adjoint_vs_elmer_fd_relative_error": abs(jax_current_gradient - elmer_current_gradient)
        / adjoint_fd_scale,
        "elmer_fd_current_step_A": current_step_A,
    }

    for metric, threshold in PARITY_THRESHOLDS.items():
        if metrics[metric] > threshold:
            raise RuntimeError(
                f"scientific threshold failed: {metric}={metrics[metric]:.9e} > {threshold:.9e}"
            )
    if not np.isfinite(jax_temperature).all() or not np.isfinite(elmer_temperature).all():
        raise RuntimeError("temperature fields contain a non-finite value")
    if metrics["maximum_temperature_rise_K"] <= 0.0 or ring_mean_jax <= 0.0:
        raise RuntimeError("the powered heater did not raise the monitored temperature")

    provenance = {
        "source": {
            "design_page": SOURCE_PAGE,
            "notebook_repository": SOURCE_REPOSITORY,
            "use": "independent reconstruction of published dimensions and tutorial parameters",
        },
        "model": {
            "dimension": 2,
            "slice": "x=0 vertical y-z solid section",
            "coordinate_unit": "m in solver data; um in figure axes",
            "out_of_plane_convention": "per_unit_depth",
            "boundary_conditions": {
                "bottom_sink": "T = 300 K",
                "remaining_external_boundary": "adiabatic",
            },
            "material_status": (
                "public tutorial parameters; not foundry-calibrated thin-film properties"
            ),
            "full_3d_claimed": False,
            "measured_device_prediction_claimed": False,
        },
        "gmsh": {
            "version": gmsh_result.identity.version,
            "executable_sha256": gmsh_result.identity.executable_sha256,
            "geometry_sha256": gmsh_result.geometry_sha256,
            "mesh_sha256": gmsh_result.mesh_sha256,
            "canonical_mesh_sha256": imported.record.canonical_mesh_sha256,
            "import_record_sha256": imported.record.digest(),
            "process_succeeded": gmsh_result.process_succeeded,
        },
        "jax": {
            "version": jax.__version__,
            "backend": jax.default_backend(),
            "device_kind": str(getattr(jax.devices()[0], "device_kind", jax.devices()[0])),
            "float64": bool(jax.config.x64_enabled),
            "backend_name": jax_solution.backend_name,
            "backend_version": jax_solution.backend_version,
            "convergence": jax_solution.convergence.status.value,
        },
        "elmer": {
            "version": elmer_baseline.metadata["elmer_version"],
            "revision": elmer_baseline.metadata["elmer_revision"],
            "source_commit": elmer_baseline.metadata["elmer_source_commit"],
            "source_digest": elmer_baseline.metadata["elmer_source_digest"],
            "source_worktree_state": elmer_baseline.metadata["elmer_source_worktree_state"],
            "executable_sha256": elmer_baseline.metadata["elmer_executable_sha256"],
            "heat_solve_sha256": elmer_baseline.metadata["elmer_heat_solve_sha256"],
            "input_sif_sha256": elmer_baseline.metadata["input_sif_sha256"],
            "raw_vtu_sha256": elmer_baseline.metadata["raw_vtu_sha256"],
            "result_sha256": elmer_baseline.metadata["result_sha256"],
            "stdout_sha256": elmer_baseline.metadata["stdout_sha256"],
            "stderr_sha256": elmer_baseline.metadata["stderr_sha256"],
            "convergence": elmer_baseline.convergence.status.value,
        },
        "femx": {
            "commit": _git_head(),
            "generator_sha256": sha256_file(Path(__file__).resolve()),
            "uv_lock_sha256": _git_head_file_sha256("uv.lock"),
            "uv_lock_source": "HEAD:uv.lock",
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "matplotlib_version": mpl.__version__,
        },
        "raw_run_artifacts": {
            "retained_outside_git": True,
            "relative_root": ".femx/readme-thermal-reference/",
            "run_id": run_root.name,
        },
        "run_identity": {
            "problem_digest": problem_digest,
            "jax_run_id": f"jax-{problem_digest[:16]}",
            "elmer_run_id": f"{run_root.name}/elmer-baseline",
        },
    }
    return SolvedCase(
        coordinates_m=coordinates,
        cells=cells,
        region_ids=region_ids,
        region_names=region_names,
        jax_temperature_K=jax_temperature,
        elmer_temperature_K=elmer_temperature,
        cell_heat_flux_W_per_m2=cell_heat_flux,
        metrics=metrics,
        provenance=provenance,
    )


def _write_nodes_csv(path: Path, solved: SolvedCase) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "node_id",
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
                solved.jax_temperature_K,
                solved.elmer_temperature_K,
                strict=True,
            )
        ):
            writer.writerow(
                (
                    node_id,
                    format(coordinate[0] * 1.0e6, ".17e"),
                    format(coordinate[1] * 1.0e6, ".17e"),
                    format(jax_value, ".17e"),
                    format(elmer_value, ".17e"),
                    format(elmer_value - jax_value, ".17e"),
                )
            )


def _write_cells_csv(path: Path, solved: SolvedCase) -> None:
    coordinates_um = solved.coordinates_m * 1.0e6
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "cell_id",
                "node_0",
                "node_1",
                "node_2",
                "region_id",
                "region_name",
                "centroid_y_um",
                "centroid_z_um",
                "heat_flux_y_W_per_m2",
                "heat_flux_z_W_per_m2",
            )
        )
        for cell_id, (cell, region_id, heat_flux) in enumerate(
            zip(solved.cells, solved.region_ids, solved.cell_heat_flux_W_per_m2, strict=True)
        ):
            centroid = coordinates_um[cell].mean(axis=0)
            writer.writerow(
                (
                    cell_id,
                    int(cell[0]),
                    int(cell[1]),
                    int(cell[2]),
                    int(region_id),
                    solved.region_names[int(region_id)],
                    format(centroid[0], ".17e"),
                    format(centroid[1], ".17e"),
                    format(heat_flux[0], ".17e"),
                    format(heat_flux[1], ".17e"),
                )
            )


def _bundle_identifier(nodes_path: Path, cells_path: Path, solved: SolvedCase) -> str:
    identity = {
        "schema_version": SCHEMA_VERSION,
        "nodes_sha256": sha256_file(nodes_path),
        "cells_sha256": sha256_file(cells_path),
        "canonical_mesh_sha256": solved.provenance["gmsh"]["canonical_mesh_sha256"],
        "generator_sha256": solved.provenance["femx"]["generator_sha256"],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "siph-thermal-reference-" + hashlib.sha256(encoded).hexdigest()[:16]


def _plot_plan_view(axis: Any) -> None:
    silicon_color = REGION_COLORS["silicon_ring"]
    heater_color = REGION_COLORS["tin_heater"]
    outline = "#263238"
    axis.add_patch(
        Wedge(
            (0.0, 0.0),
            5.25,
            0.0,
            360.0,
            width=0.5,
            facecolor=silicon_color,
            edgecolor=outline,
            linewidth=0.75,
            zorder=2,
        )
    )
    for center_y in (-5.6, 5.6):
        axis.add_patch(
            Rectangle(
                (-7.0, center_y - 0.25),
                14.0,
                0.5,
                facecolor=silicon_color,
                edgecolor=outline,
                linewidth=0.75,
                zorder=2,
            )
        )
    axis.add_patch(
        Wedge(
            (0.0, 0.0),
            6.0,
            0.0,
            360.0,
            width=2.0,
            facecolor=heater_color,
            edgecolor=outline,
            linewidth=0.75,
            alpha=0.72,
            zorder=3,
        )
    )
    axis.add_patch(
        Rectangle(
            (-0.5, -6.5),
            1.0,
            3.0,
            facecolor="white",
            edgecolor="#8C1D40",
            linewidth=0.75,
            linestyle="--",
            zorder=4,
        )
    )
    axis.axvline(0.0, color="#111827", linewidth=0.75, linestyle=":", zorder=5)
    axis.text(
        0.97,
        0.97,
        "$x=0$ FEM cut",
        transform=axis.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
    )
    axis.set(
        xlim=(-7.0, 7.0),
        ylim=(-7.0, 7.0),
        aspect="equal",
        xlabel="$x$ (µm)",
        ylabel="$y$ (µm)",
        title="a  Public MRR geometry",
    )
    axis.legend(
        handles=(
            Patch(facecolor=silicon_color, edgecolor=outline, label="Si ring + buses"),
            Patch(facecolor=heater_color, edgecolor=outline, label="TiN heater"),
        ),
        loc="center",
        frameon=True,
        framealpha=0.82,
        edgecolor="none",
        fontsize=6.2,
    )


def _field_panel(
    axis: Any,
    triangulation: mtri.Triangulation,
    values: np.ndarray,
    *,
    title: str,
    cmap: str,
    norm: mpl.colors.Normalize,
) -> Any:
    plotted = axis.tripcolor(
        triangulation,
        values,
        shading="gouraud",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    axis.set(
        xlim=(-10.0, 10.0),
        ylim=(-2.5, 2.8),
        xlabel="$y$ (µm)",
        ylabel="$z$ (µm)",
        title=title,
    )
    axis.text(
        0.02,
        0.04,
        "$z$ enlarged",
        transform=axis.transAxes,
        color="white",
        fontsize=5.8,
        bbox={"facecolor": "#111827", "edgecolor": "none", "alpha": 0.55, "pad": 1.2},
    )
    return plotted


def render_figure(output_dir: Path, solved: SolvedCase, figure_id: str) -> tuple[Path, Path]:
    """Render the policy-compliant SVG and 300 dpi PNG preview."""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 7.0,
            "axes.titlesize": 7.4,
            "axes.labelsize": 7.0,
            "axes.linewidth": 0.75,
            "lines.linewidth": 0.75,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "svg.fonttype": "none",
            "svg.hashsalt": figure_id,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    coordinates_um = solved.coordinates_m * 1.0e6
    triangulation = mtri.Triangulation(coordinates_um[:, 0], coordinates_um[:, 1], solved.cells)
    jax_rise = solved.jax_temperature_K - AMBIENT_TEMPERATURE_K
    elmer_rise = solved.elmer_temperature_K - AMBIENT_TEMPERATURE_K
    difference = solved.elmer_temperature_K - solved.jax_temperature_K
    rise_max = max(float(np.max(jax_rise)), float(np.max(elmer_rise)))
    rise_norm = mpl.colors.Normalize(vmin=0.0, vmax=rise_max)
    difference_limit = max(float(np.max(np.abs(difference))), np.finfo(np.float64).eps)
    difference_norm = mpl.colors.TwoSlopeNorm(
        vmin=-difference_limit, vcenter=0.0, vmax=difference_limit
    )

    figure = plt.figure(figsize=(7.2, 4.75), constrained_layout=True)
    grid = GridSpec(2, 3, figure=figure, width_ratios=(1.08, 1.65, 1.65))
    plan_axis = figure.add_subplot(grid[0, 0])
    mesh_axis = figure.add_subplot(grid[1, 0])
    jax_axis = figure.add_subplot(grid[0, 1])
    elmer_axis = figure.add_subplot(grid[0, 2])
    difference_axis = figure.add_subplot(grid[1, 1])
    parity_axis = figure.add_subplot(grid[1, 2])

    _plot_plan_view(plan_axis)

    region_cmap = mpl.colors.ListedColormap([REGION_COLORS[name] for name in solved.region_names])
    mesh_axis.tripcolor(
        triangulation,
        facecolors=solved.region_ids,
        shading="flat",
        cmap=region_cmap,
        vmin=-0.5,
        vmax=len(solved.region_names) - 0.5,
    )
    mesh_axis.triplot(triangulation, color="white", linewidth=0.22, alpha=0.82)
    mesh_axis.set(
        xlim=(3.72, 6.28),
        ylim=(-0.20, 2.57),
        xlabel="$y$ (µm)",
        ylabel="$z$ (µm)",
        title=(
            "b  Same P1 mesh detail\n"
            f"{int(solved.metrics['node_count']):,} nodes · "
            f"{int(solved.metrics['triangle_count']):,} triangles"
        ),
    )
    mesh_axis.legend(
        handles=(
            Patch(facecolor=REGION_COLORS["silica"], label="SiO₂"),
            Patch(facecolor=REGION_COLORS["silicon_ring"], label="Si"),
            Patch(facecolor=REGION_COLORS["tin_heater"], label="TiN"),
        ),
        loc="upper left",
        frameon=False,
        ncols=3,
        columnspacing=0.7,
        handlelength=1.0,
    )

    jax_plot = _field_panel(
        jax_axis,
        triangulation,
        jax_rise,
        title="c  JAX $\\Delta T$ · shared scale",
        cmap="magma",
        norm=rise_norm,
    )
    jax_colorbar = figure.colorbar(jax_plot, ax=jax_axis, pad=0.02, fraction=0.05)
    jax_colorbar.set_label("$\\Delta T$ (K)")

    elmer_plot = _field_panel(
        elmer_axis,
        triangulation,
        elmer_rise,
        title="d  Elmer $\\Delta T$ · shared scale",
        cmap="magma",
        norm=rise_norm,
    )
    elmer_colorbar = figure.colorbar(elmer_plot, ax=elmer_axis, pad=0.02, fraction=0.05)
    elmer_colorbar.set_label("$\\Delta T$ (K)")

    difference_plot = _field_panel(
        difference_axis,
        triangulation,
        difference,
        title="e  Elmer - JAX · zero-centred",
        cmap="RdBu_r",
        norm=difference_norm,
    )
    difference_colorbar = figure.colorbar(
        difference_plot, ax=difference_axis, pad=0.02, fraction=0.05
    )
    difference_colorbar.set_label("Temperature difference (K)")

    parity_axis.scatter(
        jax_rise,
        elmer_rise,
        s=4.0,
        color="#0072B2",
        edgecolors="none",
        alpha=0.34,
        rasterized=True,
    )
    parity_axis.plot((0.0, rise_max), (0.0, rise_max), color="#111827", linestyle="--")
    parity_axis.set(
        xlim=(0.0, rise_max),
        ylim=(0.0, rise_max),
        aspect="equal",
        xlabel="JAX $\\Delta T$ (K)",
        ylabel="Elmer $\\Delta T$ (K)",
        title="f  Nodal parity + implicit adjoint",
    )
    metrics = solved.metrics
    evidence_text = "\n".join(
        (
            f"max |Elmer - JAX|  {metrics['maximum_absolute_temperature_difference_K']:.2e} K",
            f"relative $L^2$ error       {metrics['relative_l2_temperature_rise_difference']:.2e}",
            f"energy balance          {metrics['jax_discrete_energy_balance_relative_error']:.2e}",
            f"adjoint vs Elmer FD      {metrics['adjoint_vs_elmer_fd_relative_error']:.2e}",
        )
    )
    parity_axis.text(
        0.04,
        0.96,
        evidence_text,
        transform=parity_axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.0,
        linespacing=1.3,
        bbox={"facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.9, "pad": 2.4},
    )

    figure.suptitle(
        "Silicon Photonics thermal FEM reference — one mesh, JAX and Elmer",
        fontsize=8.5,
        fontweight="bold",
    )
    svg_path = output_dir / "figure.svg"
    png_path = output_dir / "figure.png"
    metadata = {
        "Title": "Silicon Photonics thermal FEM same-mesh JAX and Elmer reference",
        "Description": (
            "Public MRR geometry and one 2D Gmsh P1 mesh solved by JAX and Elmer; "
            f"deterministic figure identifier {figure_id}."
        ),
        "Creator": "femx examples/readme_siph_thermal_reference.py",
    }
    figure.savefig(svg_path, format="svg", metadata={**metadata, "Date": None})
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    figure.savefig(png_path, format="png", dpi=300, metadata=metadata)
    plt.close(figure)
    return svg_path, png_path


def write_bundle(output_dir: Path, solved: SolvedCase) -> dict[str, Any]:
    """Write exact open data, render the figure, and finalize evidence hashes."""

    output_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = output_dir / "nodes.csv"
    cells_path = output_dir / "cells.csv"
    _write_nodes_csv(nodes_path, solved)
    _write_cells_csv(cells_path, solved)
    figure_id = _bundle_identifier(nodes_path, cells_path, solved)
    svg_path, png_path = render_figure(output_dir, solved, figure_id)

    evidence = {
        "schema_version": SCHEMA_VERSION,
        "figure_id": figure_id,
        "status": {
            "gmsh_process": "succeeded",
            "jax_convergence": "converged",
            "elmer_convergence": "converged",
            "scientific_parity": "passed",
        },
        "metrics": solved.metrics,
        "thresholds": PARITY_THRESHOLDS,
        "provenance": solved.provenance,
        "figure": {
            "coordinate_axes": "horizontal y, vertical z",
            "coordinate_unit": "um",
            "temperature_unit": "K",
            "temperature_scale": "linear, full data range, no clipping",
            "difference_scale": "linear, symmetric about zero, no clipping",
            "spatial_aspect": "vertical axis enlarged for readability and explicitly labelled",
            "heat_flux_overlay": (
                "not displayed so the JAX and Elmer temperature panels use identical encodings; "
                "cell heat flux remains available in cells.csv"
            ),
            "svg_rendering": (
                "axes and text remain vector; dense spatial fields are embedded at 300 dpi"
            ),
            "font": "7 pt sans serif at final 7.2 x 4.75 inch size",
            "line_width_pt": 0.75,
            "preview_dpi": 300,
        },
        "data_columns": {
            "nodes.csv": {
                "node_id": "zero-based canonical FEM node id",
                "y_um": "horizontal coordinate in micrometres",
                "z_um": "vertical coordinate in micrometres",
                "jax_temperature_K": "JAX nodal temperature in kelvin",
                "elmer_temperature_K": "Elmer nodal temperature in kelvin",
                "elmer_minus_jax_K": "signed nodal temperature difference in kelvin",
            },
            "cells.csv": {
                "cell_id": "zero-based canonical triangle id",
                "node_0,node_1,node_2": "counter-clockwise canonical node ids",
                "region_id,region_name": "piecewise-constant material tag",
                "centroid_y_um,centroid_z_um": "triangle centroid in micrometres",
                "heat_flux_y_W_per_m2,heat_flux_z_W_per_m2": (
                    "JAX P1 cell heat-flux vector in watts per square metre"
                ),
            },
        },
        "artifacts": {
            "nodes.csv": sha256_file(nodes_path),
            "cells.csv": sha256_file(cells_path),
            "figure.svg": sha256_file(svg_path),
            "figure.png": sha256_file(png_path),
        },
        "generation_command": (
            "uv run --with matplotlib==3.11.0 python "
            "examples/readme_siph_thermal_reference.py --allow-external "
            "--elmer-executable /path/to/ElmerSolver"
        ),
    }
    evidence_path = output_dir / "evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-external",
        action="store_true",
        help="authorize the Gmsh, Git source-check, and Elmer subprocesses",
    )
    parser.add_argument(
        "--gmsh-executable",
        type=Path,
        help="absolute Gmsh executable; defaults to the executable on PATH",
    )
    parser.add_argument(
        "--elmer-executable",
        type=Path,
        required=True,
        help="absolute separately installed ElmerSolver executable",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "docs" / "assets" / "readme" / "siph_thermal_reference",
    )
    parser.add_argument(
        "--run-parent",
        type=Path,
        default=REPOSITORY_ROOT / ".femx" / "readme-thermal-reference",
        help="ignored parent directory used to retain raw mesh and Elmer provenance",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.allow_external:
        raise SystemExit("refusing external execution without --allow-external")
    if not args.elmer_executable.is_absolute():
        raise SystemExit("--elmer-executable must be absolute")
    gmsh_executable = args.gmsh_executable
    if gmsh_executable is None:
        discovered = shutil.which("gmsh")
        if discovered is None:
            raise SystemExit("Gmsh is not available on PATH; pass --gmsh-executable")
        gmsh_executable = Path(discovered).resolve()
    if not gmsh_executable.is_absolute():
        raise SystemExit("--gmsh-executable must be absolute")
    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        raise SystemExit("the parity generator requires strict JAX CPU float64")

    args.run_parent.mkdir(parents=True, exist_ok=True)
    input_digest = hashlib.sha256(render_gmsh_geometry().encode("utf-8")).hexdigest()[:12]
    attempt = 1
    while True:
        run_root = args.run_parent / f"{input_digest}-attempt-{attempt:03d}"
        if not run_root.exists():
            break
        attempt += 1
    solved = solve_case(
        gmsh_executable=gmsh_executable,
        elmer_executable=args.elmer_executable,
        run_root=run_root,
    )
    evidence = write_bundle(args.output_dir, solved)
    print(f"figure_id : {evidence['figure_id']}")
    print(f"run_root  : {run_root.relative_to(REPOSITORY_ROOT)}")
    for name, value in solved.metrics.items():
        print(f"{name:52s}: {value:.9e}")


if __name__ == "__main__":
    main()
