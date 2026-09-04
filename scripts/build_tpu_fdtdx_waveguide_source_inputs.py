#!/usr/bin/env python3
"""Build hash-bound Elmer/JAX silicon-waveguide ModeBundle inputs for a TPU FDTDX run."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, cast

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from femx.artifacts import sha256_file
from femx.backends.elmer.port_eigenmode import (
    ElmerPortEigenmodeBackend,
    ElmerPortEigenmodeIdentity,
    PreparedElmerPortEigenmode,
)
from femx.backends.elmer.runner import ElmerInstallation
from femx.backends.jax.port_eigenmode import JaxPortEigenmodeBackend
from femx.backends.jax.port_operator import lossless_port_coefficients
from femx.backends.protocol import ExecutionPolicy, PrepareRequest, SolveRequest
from femx.core.problem import Problem
from femx.core.solution import ConvergenceStatus
from femx.interop.fdtdx import (
    FDTDXFingerprint,
    SolverFingerprint,
    build_fdtdx_mode_source_contract,
    build_yee_grid,
    build_yee_port_sampling_plan,
    port_mode_solution_to_bundle,
    read_mode_bundle_hdf5,
    write_mode_bundle_hdf5,
)
from femx.meshing.gmsh import (
    GmshInstallation,
    GmshMeshingRequest,
    GmshRunner,
    RectangularWaveguideCrossSection,
    read_gmsh_msh,
)
from femx.physics import (
    VACUUM_SPEED_OF_LIGHT_M_PER_S,
    IsotropicOpticalRegion,
    PerfectElectricBoundary,
    PortEigenmode,
)
from femx.runtime import prepare, solve
from femx.validation.tpu_fdtdx_waveguide_source_evidence import INPUT_MANIFEST_SCHEMA
from scripts.check_source_checkouts import inspect_source_checkout, load_source_specs

FDTDX_FINGERPRINT = FDTDXFingerprint(
    package_version="0.6.2",
    source_revision="81a58da9cde4a4ff822f835b63597c0d0d8ba978",
    source_digest="c881f0cf32b4272dc10e0acee716bfda3cf2d0f11973c23d31f29b11f3dce01c",
)
FDTDX_SOURCE_FILES = {
    "src/fdtdx/core/grid.py": ("d24739b9229ad8c61a57e4f688e6224eae63a680ff6554ddd7a5ef765edab6dd"),
    "src/fdtdx/fdtd/wrapper.py": (
        "97d562e0a33eeeccd6ce42d12cf8dc29f1e9fd071e561b5b009ac5da8ece7384"
    ),
    "src/fdtdx/objects/object.py": (
        "24c986b9fa73bf474bce9fefc2145436654be4758e83dbcaf6fb955b7eb8557f"
    ),
    "src/fdtdx/objects/sources/custom_mode.py": (
        "0c5925a784da33f8d8236a874d4759d4ebe6df29317dcc1ce68877b4a4036df5"
    ),
    "src/fdtdx/objects/sources/tfsf.py": (
        "bd270995bffd174c7014adf9a02c7648134547c3bab7a294570e0a179326e611"
    ),
}
WAVELENGTH_M = 1.55e-6
CLADDING_INDEX = 1.444
CORE_INDEX = 3.48
SOURCE_Z_INDEX = 6
DETECTOR_Z_INDEX = 24
Z_SPACING_M = 40.0e-9
AUTHORIZED = ExecutionPolicy(execution_authorized=True, allow_external_process=True)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


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


def _fdtd_edges(validated: Any, recipe: RectangularWaveguideCrossSection) -> tuple[np.ndarray, ...]:
    minimum = np.min(validated.coordinates, axis=0)
    maximum = np.max(validated.coordinates, axis=0)
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
        lower_cells=28,
        core_cells=8,
        upper_cells=28,
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
    z_edges = np.arange(-SOURCE_Z_INDEX, 31, dtype=np.float64) * Z_SPACING_M
    if (x_edges.size - 1, y_edges.size - 1, z_edges.size - 1) != (64, 52, 36):
        raise RuntimeError("waveguide TPU scene grid no longer matches its admitted topology")
    return x_edges, y_edges, z_edges


def _cell_materials(validated: Any) -> tuple[np.ndarray, np.ndarray]:
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


def _source_inverse_permittivity() -> np.ndarray:
    epsilon = np.full((64, 52), CLADDING_INDEX**2, dtype=np.float64)
    epsilon[28:36, 24:28] = CORE_INDEX**2
    return np.ascontiguousarray((1.0 / epsilon)[None, :, :, None])


def _source_report(source_root: Path, name: str) -> dict[str, object]:
    specs = load_source_specs(checkout_overrides={name: source_root})
    spec = next(item for item in specs if item.name == name)
    report = inspect_source_checkout(spec, include_worktree=True, require_clean=True)
    if not report.valid:
        raise RuntimeError(f"locked {name} source checkout is invalid: {list(report.errors)}")
    return report.to_dict()


def _verify_fdtdx_source(source_root: Path) -> dict[str, str]:
    hashes = {path: sha256_file(source_root / path) for path in FDTDX_SOURCE_FILES}
    if hashes != FDTDX_SOURCE_FILES:
        raise RuntimeError("locked FDTDX source files differ from the waveguide input contract")
    return hashes


def _elmer_identity(
    executable: Path, source_report: dict[str, object]
) -> ElmerPortEigenmodeIdentity:
    home = executable.parent.parent
    modules = home / "share" / "elmersolver" / "lib"
    return ElmerPortEigenmodeIdentity(
        version="26.2-devel",
        revision="4f2d7e4b9",
        executable_sha256=sha256_file(executable),
        em_port_sha256=sha256_file(modules / "EMPort.so"),
        result_output_sha256=sha256_file(modules / "ResultOutputSolve.so"),
        save_data_sha256=sha256_file(modules / "SaveData.so"),
        source_commit=cast(str, source_report["head_commit"]),
        source_digest=cast(str, source_report["source_digest"]),
        source_worktree_state=cast(str, source_report["worktree_state"]),
    )


def build_inputs(
    *,
    output_root: Path,
    gmsh_executable: Path,
    elmer_executable: Path,
    elmer_source: Path,
    fdtdx_source: Path,
) -> dict[str, object]:
    import jax.numpy as jnp

    for label, path in (
        ("Gmsh executable", gmsh_executable),
        ("Elmer executable", elmer_executable),
    ):
        if not path.is_absolute() or not path.is_file() or path.is_symlink():
            raise ValueError(f"{label} must be an absolute regular non-symlink file")
    if not output_root.is_absolute() or output_root.exists() or output_root.is_symlink():
        raise ValueError("output root must be a new absolute path")
    output_root.mkdir(parents=True)
    modes = output_root / "modes"
    meshing_directory = output_root / "gmsh"
    elmer_run_directory = output_root / "elmer"
    modes.mkdir()
    meshing_directory.mkdir()

    elmer_report = _source_report(elmer_source, "elmer")
    fdtdx_report = _source_report(fdtdx_source, "fdtdx")
    fdtdx_hashes = _verify_fdtdx_source(fdtdx_source)
    identity = _elmer_identity(elmer_executable, elmer_report)
    elmer_backend = ElmerPortEigenmodeBackend(ElmerInstallation(elmer_executable), identity)
    gmsh_runner = GmshRunner(GmshInstallation(gmsh_executable))

    recipe = RectangularWaveguideCrossSection(
        cladding_mesh_size_m=0.44e-6,
        core_mesh_size_m=0.09e-6,
    )
    geometry = recipe.render_geo()
    (meshing_directory / "waveguide.geo").write_text(geometry, encoding="utf-8")
    meshing = gmsh_runner.run(
        GmshMeshingRequest("waveguide.geo"),
        working_directory=meshing_directory,
        policy=AUTHORIZED,
    )
    if not meshing.process_succeeded:
        raise RuntimeError(f"Gmsh failed: {meshing.stderr}")
    imported = read_gmsh_msh(
        meshing_directory / "mesh.msh",
        coordinate_scale_to_m=recipe.coordinate_scale_to_m,
    )
    frequency_hz = VACUUM_SPEED_OF_LIGHT_M_PER_S / WAVELENGTH_M
    problem = Problem(
        "locked-elmer-jax-tpu-fdtdx-silicon-port",
        imported.mesh,
        PortEigenmode(
            regions=(
                IsotropicOpticalRegion("cladding", CLADDING_INDEX**2),
                IsotropicOpticalRegion("core", CORE_INDEX**2),
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
    elmer_prepared = prepare(
        problem,
        elmer_backend,
        request=PrepareRequest(run_directory=elmer_run_directory),
    )
    elmer_solution = solve(
        elmer_prepared,
        elmer_backend,
        request=SolveRequest(run_directory=elmer_run_directory, policy=AUTHORIZED),
    )
    jax_backend = JaxPortEigenmodeBackend(relative_residual_tolerance=1.0e-12)
    jax_solution = solve(prepare(problem, jax_backend), jax_backend)
    if (
        elmer_solution.convergence.status is not ConvergenceStatus.CONVERGED
        or jax_solution.convergence.status is not ConvergenceStatus.CONVERGED
    ):
        raise RuntimeError("Elmer and JAX port modes must both converge before TPU input admission")

    elmer_payload = cast(PreparedElmerPortEigenmode, elmer_prepared.payload)
    validated = elmer_payload.validated
    relative_permittivity, relative_permeability = _cell_materials(validated)
    _, cell_reluctivity = lossless_port_coefficients(
        jnp.asarray(relative_permittivity),
        jnp.asarray(relative_permeability),
    )
    x_edges, y_edges, z_edges = _fdtd_edges(validated, recipe)
    source_grid = build_yee_grid((x_edges, y_edges, z_edges[SOURCE_Z_INDEX : SOURCE_Z_INDEX + 2]))
    transfer_plan = build_yee_port_sampling_plan(
        validated.coordinates,
        validated.cells,
        validated.edge_signs,
        source_grid,
    )
    if transfer_plan.ambiguous_target_point_count != 0:
        raise RuntimeError("waveguide source grid has ambiguous FEM point locations")
    config_sha256 = hashlib.sha256(
        _canonical_json(problem.physics.canonical_data()).encode("utf-8")
    ).hexdigest()
    bundles = {
        "elmer": port_mode_solution_to_bundle(
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
            fdtdx=FDTDX_FINGERPRINT,
        ),
        "jax": port_mode_solution_to_bundle(
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
            fdtdx=FDTDX_FINGERPRINT,
        ),
    }
    medium = _source_inverse_permittivity()
    artifacts: dict[str, dict[str, object]] = {}
    contracts: dict[str, Any] = {}
    for solver, bundle in bundles.items():
        artifact = write_mode_bundle_hdf5(output_root, f"modes/{solver}-mode.h5", bundle)
        decoded = read_mode_bundle_hdf5(output_root, artifact.reference)
        contract = build_fdtdx_mode_source_contract(
            decoded.bundle,
            source_name="femx-waveguide-port",
            expected_inverse_permittivity=medium,
            expected_inverse_permeability=np.asarray(1.0, dtype=np.float64),
            fdtdx=FDTDX_FINGERPRINT,
        )
        contracts[solver] = contract
        artifacts[solver] = {
            "reference": artifact.reference.to_dict(),
            "content_sha256": artifact.content_sha256,
            "logical_data_bytes": artifact.logical_data_bytes,
            "bundle_sha256": contract.mode_bundle_sha256,
        }

    electric_error = float(
        np.linalg.norm(
            np.asarray(bundles["jax"].electric.values)
            - np.asarray(bundles["elmer"].electric.values)
        )
        / np.linalg.norm(np.asarray(bundles["elmer"].electric.values))
    )
    magnetic_error = float(
        np.linalg.norm(
            np.asarray(bundles["jax"].magnetic.values)
            - np.asarray(bundles["elmer"].magnetic.values)
        )
        / np.linalg.norm(np.asarray(bundles["elmer"].magnetic.values))
    )
    if max(electric_error, magnetic_error) > 1.0e-10:
        raise RuntimeError("canonical Elmer/JAX source parity exceeds 1e-10")
    pre_correction_errors = {
        solver: bundle.transfer.relative_pre_correction_power_error
        for solver, bundle in bundles.items()
    }
    if any(value is None or value > 0.06 for value in pre_correction_errors.values()):
        raise RuntimeError("FEM-to-Yee raw power error exceeds the admitted 6% input bound")

    manifest: dict[str, object] = {
        "schema_version": INPUT_MANIFEST_SCHEMA,
        "status": "passed",
        "geometry": {
            "kind": "centered rectangular silicon core in silica cladding",
            "wavelength_m": WAVELENGTH_M,
            "core_width_m": recipe.core_width_m,
            "core_height_m": recipe.core_height_m,
            "cladding_width_m": recipe.cladding_width_m,
            "cladding_height_m": recipe.cladding_height_m,
            "core_refractive_index": CORE_INDEX,
            "cladding_refractive_index": CLADDING_INDEX,
            "grid_shape_xyz": [64, 52, 36],
            "source_z_index": SOURCE_Z_INDEX,
            "detector_z_index": DETECTOR_Z_INDEX,
            "z_spacing_m": Z_SPACING_M,
            "core_cells_xy": [8, 4],
        },
        "sources": {
            "elmer": elmer_report,
            "fdtdx": fdtdx_report,
            "fdtdx_module_sha256": fdtdx_hashes,
        },
        "runtime": {
            "gmsh": {
                "version": meshing.identity.version,
                "executable_sha256": meshing.identity.executable_sha256,
                "geometry_sha256": meshing.geometry_sha256,
                "mesh_file_sha256": meshing.mesh_sha256,
            },
            "elmer": {
                "version": identity.version,
                "revision": identity.revision,
                "executable_sha256": identity.executable_sha256,
                "em_port_sha256": identity.em_port_sha256,
                "result_output_sha256": identity.result_output_sha256,
                "save_data_sha256": identity.save_data_sha256,
                "result_output_file_sha256": elmer_solution.metadata["elmer_result_output_sha256"],
                "save_data_file_sha256": elmer_solution.metadata["elmer_save_data_sha256"],
            },
            "jax": {
                "jax_version": package_version("jax"),
                "jaxlib_version": package_version("jaxlib"),
                "backend": "cpu",
                "x64_enabled": True,
                "precision": jax_solution.metadata["precision"],
            },
            "fdtdx_fingerprint": {
                "package_version": FDTDX_FINGERPRINT.package_version,
                "source_revision": FDTDX_FINGERPRINT.source_revision,
                "source_digest": FDTDX_FINGERPRINT.source_digest,
            },
        },
        "mesh": {
            "source_mesh_sha256": transfer_plan.source_mesh_sha256,
            "node_count": int(validated.coordinates.shape[0]),
            "triangle_count": int(validated.cells.shape[0]),
            "ambiguous_target_point_count": transfer_plan.ambiguous_target_point_count,
        },
        "artifacts": artifacts,
        "contracts": {
            solver: {
                "source_contract_sha256": contract.sha256,
                "bundle_sha256": contract.mode_bundle_sha256,
            }
            for solver, contract in contracts.items()
        },
        "errors": {
            "canonical_source_electric_relative_l2": electric_error,
            "canonical_source_magnetic_relative_l2": magnetic_error,
            "pre_correction_power_relative": pre_correction_errors,
        },
        "claim_scope": (
            "hash-bound same-mesh Elmer/JAX silicon-waveguide source inputs for a later physical "
            "TPU FDTDX process-set run; this file alone is not accelerator or propagation evidence"
        ),
    }
    _atomic_json(output_root / "input-manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--gmsh-executable", required=True, type=Path)
    parser.add_argument("--elmer-executable", required=True, type=Path)
    parser.add_argument("--elmer-source", required=True, type=Path)
    parser.add_argument("--fdtdx-source", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_inputs(
        output_root=args.output_root.resolve(strict=False),
        gmsh_executable=args.gmsh_executable.resolve(strict=True),
        elmer_executable=args.elmer_executable.resolve(strict=True),
        elmer_source=args.elmer_source.resolve(strict=True),
        fdtdx_source=args.fdtdx_source.resolve(strict=True),
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root.resolve()),
                "manifest_sha256": sha256_file(args.output_root / "input-manifest.json"),
                "source_electric_relative_l2": cast(dict[str, object], manifest["errors"])[
                    "canonical_source_electric_relative_l2"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
