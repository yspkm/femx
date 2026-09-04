#!/usr/bin/env python3
"""Run locked Elmer/JAX silicon-waveguide modes through one physical TPU FDTDX scene."""

# ruff: noqa: E402

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from collections.abc import Callable
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from femx.artifacts import ArtifactRef, sha256_file
from femx.validation.tpu_fdtdx_waveguide_source_evidence import (
    DETECTOR_Z_INDEX,
    GRID_SHAPE,
    INPUT_MANIFEST_SCHEMA,
    PROCESS_EVIDENCE_SCHEMA,
    SOLVERS,
    SOURCE_Z_INDEX,
)
from scripts.run_tpu_fdtdx_mode_source_evidence import (
    FDTDX_PACKAGE_VERSION,
    FDTDX_SOURCE_DIGEST,
    FDTDX_SOURCE_REVISION,
    SCALAR_CONTRACT,
    _atomic_json,
    _atomic_text,
    _claim_worker_entry,
    _manifest_provenance,
    _memory_report,
    _positive_environment_integer,
    _publish_process_zero_compatibility,
    _run_compiled_fdtdx,
    _runtime,
    _scalar,
    _verify_fdtdx_source,
)

INPUT_RELATIVE_PATH = Path("inputs") / "tpu-waveguide"
INPUT_MANIFEST_NAME = "input-manifest.json"
SIMULATION_TIME_S = 30.0e-15
Z_SPACING_M = 40.0e-9
CLADDING_INDEX = 1.444
CORE_INDEX = 3.48
MAXIMUM_MANIFEST_BYTES = 4 * 1024 * 1024
_RUNTIME_MODE_SOURCE_FIELDS = ("_E", "_H", "_neff", "_time_offset_E", "_time_offset_H")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _input_root() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    root = project_root / INPUT_RELATIVE_PATH
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"waveguide input root is missing or unsafe: {root}")
    return root.resolve(strict=True)


def _load_input_manifest(root: Path) -> tuple[dict[str, object], str]:
    path = root / INPUT_MANIFEST_NAME
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("waveguide input manifest must be a regular non-symlink file")
    size = path.stat().st_size
    if size <= 0 or size > MAXIMUM_MANIFEST_BYTES:
        raise RuntimeError("waveguide input manifest size is outside the admitted range")
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise RuntimeError("waveguide input manifest root must have string keys")
    manifest = cast(dict[str, object], value)
    if (
        manifest.get("schema_version") != INPUT_MANIFEST_SCHEMA
        or manifest.get("status") != "passed"
    ):
        raise RuntimeError("waveguide input manifest is unsupported or non-passing")
    return manifest, sha256_file(path)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise RuntimeError(f"waveguide input {label} must be an object with string keys")
    return cast(dict[str, object], value)


def _artifact(
    root: Path, manifest: dict[str, object], solver: str
) -> tuple[Any, dict[str, object]]:
    from femx.interop.fdtdx import read_mode_bundle_hdf5

    artifacts = _mapping(manifest.get("artifacts"), "artifacts")
    record = _mapping(artifacts.get(solver), f"artifacts.{solver}")
    reference = ArtifactRef.from_dict(_mapping(record.get("reference"), "artifact reference"))
    if reference.path != f"modes/{solver}-mode.h5":
        raise RuntimeError(f"waveguide {solver} artifact path is not canonical")
    decoded = read_mode_bundle_hdf5(root, reference)
    if decoded.content_sha256 != record.get(
        "content_sha256"
    ) or decoded.logical_data_bytes != record.get("logical_data_bytes"):
        raise RuntimeError(f"waveguide {solver} HDF5 logical identity differs from the manifest")
    return decoded.bundle, record


def _verify_manifest_contract(manifest: dict[str, object]) -> None:
    geometry = _mapping(manifest.get("geometry"), "geometry")
    if (
        geometry.get("grid_shape_xyz") != list(GRID_SHAPE)
        or geometry.get("source_z_index") != SOURCE_Z_INDEX
        or geometry.get("detector_z_index") != DETECTOR_Z_INDEX
        or geometry.get("core_cells_xy") != [8, 4]
        or geometry.get("core_width_m") != 0.5e-6
        or geometry.get("core_height_m") != 0.22e-6
        or geometry.get("core_refractive_index") != CORE_INDEX
        or geometry.get("cladding_refractive_index") != CLADDING_INDEX
    ):
        raise RuntimeError("waveguide manifest geometry differs from the admitted Si/SiO2 scene")
    runtime = _mapping(manifest.get("runtime"), "runtime")
    fingerprint = _mapping(runtime.get("fdtdx_fingerprint"), "runtime.fdtdx_fingerprint")
    if fingerprint != {
        "package_version": FDTDX_PACKAGE_VERSION,
        "source_revision": FDTDX_SOURCE_REVISION,
        "source_digest": FDTDX_SOURCE_DIGEST,
    }:
        raise RuntimeError("waveguide manifest FDTDX fingerprint differs from the locked runtime")
    errors = _mapping(manifest.get("errors"), "errors")
    for name in (
        "canonical_source_electric_relative_l2",
        "canonical_source_magnetic_relative_l2",
    ):
        value = errors.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise RuntimeError(f"waveguide manifest {name} is not finite")
        if float(value) > 1.0e-10:
            raise RuntimeError(f"waveguide manifest {name} exceeds the canonical parity bound")


def _source_inverse_permittivity() -> Any:
    import numpy as np

    epsilon = np.full(GRID_SHAPE[:2], CLADDING_INDEX**2, dtype=np.float64)
    epsilon[28:36, 24:28] = CORE_INDEX**2
    return np.ascontiguousarray((1.0 / epsilon)[None, :, :, None])


def _relative_l2(observed: Any, reference: Any) -> float:
    import numpy as np

    observed_array = np.asarray(observed, dtype=np.complex128)
    reference_array = np.asarray(reference, dtype=np.complex128)
    denominator = float(np.linalg.norm(reference_array))
    if denominator <= 0.0:
        raise RuntimeError("waveguide parity reference has zero norm")
    return float(np.linalg.norm(observed_array - reference_array) / denominator)


def _scene(
    fdtdx: Any,
    jnp: Any,
    *,
    x_edges: Any,
    y_edges: Any,
    z_edges: Any,
    frequency_hz: float,
    source: Any,
) -> tuple[list[Any], list[Any], Any]:
    volume = fdtdx.SimulationVolume(
        partial_grid_shape=GRID_SHAPE,
        material=fdtdx.Material(permittivity=CLADDING_INDEX**2),
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
        partial_grid_shape=(8, 4, GRID_SHAPE[2]),
        material=fdtdx.Material(permittivity=CORE_INDEX**2),
    )
    detector = fdtdx.PhasorDetector(
        name="downstream-phasor",
        partial_grid_shape=(GRID_SHAPE[0], GRID_SHAPE[1], 1),
        wave_characters=(fdtdx.WaveCharacter(frequency=frequency_hz),),
        components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        reduce_volume=False,
        dtype=jnp.complex64,
        dft_subsample="auto",
        plot=False,
    )
    objects = [volume, *boundaries.values(), core, detector, source]
    constraints = [
        *boundary_constraints,
        core.place_at_center(volume, axes=(0, 1, 2)),
        detector.same_size(volume, axes=(0, 1)),
        detector.place_at_center(volume, axes=(0, 1)),
        fdtdx.RealCoordinateConstraint(
            object=detector.name,
            axes=(2,),
            sides=("-",),
            coordinates=(float(z_edges[DETECTOR_Z_INDEX]),),
        ),
        source.same_size(volume, axes=(0, 1)),
        source.place_at_center(volume, axes=(0, 1)),
        fdtdx.RealCoordinateConstraint(
            object=source.name,
            axes=(2,),
            sides=("-",),
            coordinates=(float(z_edges[SOURCE_Z_INDEX]),),
        ),
    ]
    config = fdtdx.SimulationConfig(
        time=SIMULATION_TIME_S,
        grid=fdtdx.RectilinearGrid(
            x_edges=jnp.asarray(x_edges, dtype=jnp.float32),
            y_edges=jnp.asarray(y_edges, dtype=jnp.float32),
            z_edges=jnp.asarray(z_edges, dtype=jnp.float32),
        ),
        backend="tpu",
        dtype=jnp.float32,
    )
    return objects, constraints, config


def _prepare_source_run(
    *,
    fdtdx: Any,
    jax: Any,
    jnp: Any,
    bundle: Any,
    fingerprint: Any,
    expected_inverse_permittivity: Any,
    x_edges: Any,
    y_edges: Any,
    z_edges: Any,
    key: Any,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    import numpy as np

    from femx.interop.fdtdx import (
        bind_fdtdx_distributed_mode_source,
        build_fdtdx_mode_source_contract,
        lower_mode_source_inputs_for_tpu,
        make_fdtdx_distributed_mode_source,
    )

    lowered = lower_mode_source_inputs_for_tpu(
        bundle,
        expected_inverse_permittivity=expected_inverse_permittivity,
        expected_inverse_permeability=np.asarray(1.0, dtype=np.float64),
    )
    contract = build_fdtdx_mode_source_contract(
        lowered.bundle,
        source_name="femx-waveguide-port",
        expected_inverse_permittivity=lowered.expected_inverse_permittivity,
        expected_inverse_permeability=lowered.expected_inverse_permeability,
        fdtdx=fingerprint,
    )
    source = make_fdtdx_distributed_mode_source(
        lowered.bundle,
        contract,
        verified_fingerprint=fingerprint,
    )
    objects_to_place, constraints, config = _scene(
        fdtdx,
        jnp,
        x_edges=x_edges,
        y_edges=y_edges,
        z_edges=z_edges,
        frequency_hz=lowered.bundle.frequency_hz,
        source=source,
    )
    objects, arrays, parameters, config, _placement = fdtdx.place_objects(
        objects_to_place,
        config,
        constraints,
        key=key,
    )
    arrays, objects, _application = fdtdx.apply_params(
        arrays=arrays,
        objects=objects,
        params=parameters,
        key=key,
    )
    objects, binding = bind_fdtdx_distributed_mode_source(objects, lowered.bundle, contract)
    if _scalar(jax, jnp.linalg.norm(arrays.fields.E)) != 0.0:
        raise RuntimeError("waveguide FDTDX electric field is not initially zero")
    if _scalar(jax, jnp.linalg.norm(arrays.fields.H)) != 0.0:
        raise RuntimeError("waveguide FDTDX magnetic field is not initially zero")
    return arrays, objects, config, lowered, contract, binding


def _same_addressable_runtime_leaf(left: object, right: object) -> bool:
    """Compare immutable JAX leaves without gathering a global array to one process."""

    import numpy as np

    if left is right:
        return True
    left_shape = getattr(left, "shape", None)
    right_shape = getattr(right, "shape", None)
    left_dtype = getattr(left, "dtype", None)
    right_dtype = getattr(right, "dtype", None)
    if left_shape is None or right_shape is None or left_dtype is None or right_dtype is None:
        return False
    if tuple(left_shape) != tuple(right_shape) or np.dtype(left_dtype) != np.dtype(right_dtype):
        return False
    left_sharding = getattr(left, "sharding", None)
    right_sharding = getattr(right, "sharding", None)
    if (left_sharding is None) != (right_sharding is None) or left_sharding != right_sharding:
        return False
    left_shards = getattr(left, "addressable_shards", None)
    right_shards = getattr(right, "addressable_shards", None)
    if left_shards is None and right_shards is None:
        return bool(np.array_equal(np.asarray(left), np.asarray(right)))
    if left_shards is None or right_shards is None or len(left_shards) != len(right_shards):
        return False
    for left_shard, right_shard in zip(left_shards, right_shards, strict=True):
        if (
            left_shard.index != right_shard.index
            or left_shard.replica_id != right_shard.replica_id
            or not np.array_equal(np.asarray(left_shard.data), np.asarray(right_shard.data))
        ):
            return False
    return True


def _reuse_scene_with_candidate_source(
    baseline: tuple[Any, Any, Any, Any, Any, Any],
    candidate: tuple[Any, Any, Any, Any, Any, Any],
    *,
    tree_structure: Callable[[object], object],
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Move only independently prepared source leaves into one static FDTDX scene pytree."""

    baseline_arrays, baseline_objects, baseline_config, *_ = baseline
    _candidate_arrays, candidate_objects, _candidate_config, lowered, contract, binding = candidate
    baseline_contract = baseline[4]
    source_name = getattr(contract, "source_name", None)
    if not isinstance(source_name, str) or source_name != getattr(
        baseline_contract, "source_name", None
    ):
        raise RuntimeError("waveguide source contracts do not share one canonical source name")
    baseline_source = baseline_objects[source_name]
    candidate_source = candidate_objects[source_name]
    for field in _RUNTIME_MODE_SOURCE_FIELDS:
        value = getattr(candidate_source, field, None)
        if value is None:
            raise RuntimeError(f"candidate waveguide source has no runtime field {field}")
        baseline_source = baseline_source.aset(field, value, create_new_ok=True)
    source_index = baseline_objects.index(source_name)
    compatible_objects = baseline_objects.aset(f"object_list->[{source_index}]", baseline_source)
    baseline_structure = tree_structure((baseline_arrays, baseline_objects, baseline_config))
    compatible_structure = tree_structure((baseline_arrays, compatible_objects, baseline_config))
    if compatible_structure != baseline_structure:
        raise RuntimeError("runtime source rebinding changed the compiled FDTDX pytree structure")
    rebound = compatible_objects[source_name]
    if any(
        not _same_addressable_runtime_leaf(
            getattr(rebound, field, None), getattr(candidate_source, field, None)
        )
        for field in _RUNTIME_MODE_SOURCE_FIELDS
    ):
        raise RuntimeError("runtime source rebinding did not preserve candidate mode leaves")
    return baseline_arrays, compatible_objects, baseline_config, lowered, contract, binding


def _result_metrics(jax: Any, jnp: Any, state: Any) -> tuple[int, dict[str, object], Any]:
    import numpy as np

    step, arrays = state
    phasor = arrays.detector_states["downstream-phasor"]["phasor"]
    metrics: dict[str, object] = {
        "final_e_l2": _scalar(jax, jnp.linalg.norm(arrays.fields.E)),
        "final_h_l2": _scalar(jax, jnp.linalg.norm(arrays.fields.H)),
        "downstream_phasor_l2": _scalar(jax, jnp.linalg.norm(phasor)),
        "all_fields_finite": bool(
            np.asarray(
                jax.device_get(
                    jnp.all(jnp.isfinite(arrays.fields.E))
                    & jnp.all(jnp.isfinite(arrays.fields.H))
                    & jnp.all(jnp.isfinite(phasor))
                )
            )
        ),
    }
    return int(np.asarray(jax.device_get(step))), metrics, phasor


def main() -> int:
    remote_run = Path(os.environ["PHOXLA_REMOTE_RUN_DIR"])
    provenance = _manifest_provenance(remote_run)
    launch_claim = _claim_worker_entry(remote_run, provenance)
    jax = _runtime()
    import fdtdx  # type: ignore[import-not-found]
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import multihost_utils

    from femx.interop.fdtdx import FDTDXFingerprint

    process_index = int(jax.process_index())
    output_root = Path(os.environ["PHOXLA_OUTPUT_DIR"])
    if cast(int, launch_claim["process_index"]) != process_index:
        raise RuntimeError("worker entry claim disagrees with initialized JAX process identity")
    module_hashes = _verify_fdtdx_source()
    fingerprint = FDTDXFingerprint(
        package_version=FDTDX_PACKAGE_VERSION,
        source_revision=FDTDX_SOURCE_REVISION,
        source_digest=FDTDX_SOURCE_DIGEST,
    )
    input_root = _input_root()
    manifest, manifest_sha256 = _load_input_manifest(input_root)
    _verify_manifest_contract(manifest)
    canonical_bundles: dict[str, Any] = {}
    artifact_records: dict[str, dict[str, object]] = {}
    for solver in SOLVERS:
        canonical_bundles[solver], artifact_records[solver] = _artifact(
            input_root, manifest, solver
        )
    elmer_grid = canonical_bundles["elmer"].electric.grid
    jax_grid = canonical_bundles["jax"].electric.grid
    for elmer_axis, jax_axis in zip(
        elmer_grid.edge_coordinates,
        jax_grid.edge_coordinates,
        strict=True,
    ):
        if not np.array_equal(elmer_axis, jax_axis):
            raise RuntimeError("Elmer and JAX source artifacts do not share one Yee grid")
    if elmer_grid.shape != (64, 52, 1):
        raise RuntimeError("waveguide source artifacts do not match the admitted source plane")
    x_edges = np.asarray(elmer_grid.edge_coordinates[0], dtype=np.float64)
    y_edges = np.asarray(elmer_grid.edge_coordinates[1], dtype=np.float64)
    z_edges = np.arange(-SOURCE_Z_INDEX, 31, dtype=np.float64) * Z_SPACING_M
    if not np.array_equal(
        np.asarray(elmer_grid.edge_coordinates[2], dtype=np.float64),
        z_edges[SOURCE_Z_INDEX : SOURCE_Z_INDEX + 2],
    ):
        raise RuntimeError("waveguide source artifact z plane differs from the FDTDX scene")

    key = jax.random.PRNGKey(20260902)
    expected_inverse_permittivity = _source_inverse_permittivity()
    prepared: dict[str, tuple[Any, Any, Any, Any, Any, Any]] = {}
    for solver in SOLVERS:
        prepared[solver] = _prepare_source_run(
            fdtdx=fdtdx,
            jax=jax,
            jnp=jnp,
            bundle=canonical_bundles[solver],
            fingerprint=fingerprint,
            expected_inverse_permittivity=expected_inverse_permittivity,
            x_edges=x_edges,
            y_edges=y_edges,
            z_edges=z_edges,
            key=key,
        )
        canonical_hash = artifact_records[solver].get("bundle_sha256")
        if prepared[solver][3].report.source_bundle_sha256 != canonical_hash:
            raise RuntimeError(f"{solver} precision report disagrees with the input artifact")
    prepared["jax"] = _reuse_scene_with_candidate_source(
        prepared["elmer"],
        prepared["jax"],
        tree_structure=jax.tree_util.tree_structure,
    )

    source_e_error = _relative_l2(
        prepared["jax"][3].bundle.electric.values,
        prepared["elmer"][3].bundle.electric.values,
    )
    source_h_error = _relative_l2(
        prepared["jax"][3].bundle.magnetic.values,
        prepared["elmer"][3].bundle.magnetic.values,
    )

    run = jax.jit(fdtdx.run_fdtd, static_argnames=("show_progress", "progress_callback"))
    elmer_arrays, elmer_objects, elmer_config, *_ = prepared["elmer"]
    started = time.perf_counter()
    lowered_executable = run.lower(
        arrays=elmer_arrays,
        objects=elmer_objects,
        config=elmer_config,
        key=key,
        show_progress=False,
        progress_callback=None,
    )
    lowering_seconds = time.perf_counter() - started
    stablehlo = str(lowered_executable.compiler_ir("stablehlo"))
    started = time.perf_counter()
    compiled = lowered_executable.compile()
    compilation_seconds = time.perf_counter() - started
    warmup_seconds: dict[str, float] = {}
    execution_seconds: dict[str, float] = {}
    final_states: dict[str, Any] = {}
    for solver in SOLVERS:
        arrays, objects, config, *_ = prepared[solver]
        started = time.perf_counter()
        warmup = _run_compiled_fdtdx(
            compiled,
            arrays=arrays,
            objects=objects,
            config=config,
            key=key,
        )
        jax.block_until_ready(warmup)
        warmup_seconds[solver] = time.perf_counter() - started
        started = time.perf_counter()
        final_states[solver] = _run_compiled_fdtdx(
            compiled,
            arrays=arrays,
            objects=objects,
            config=config,
            key=key,
        )
        jax.block_until_ready(final_states[solver])
        execution_seconds[solver] = time.perf_counter() - started

    completed: dict[str, int] = {}
    metrics: dict[str, dict[str, object]] = {}
    phasors: dict[str, Any] = {}
    for solver in SOLVERS:
        completed[solver], metrics[solver], phasors[solver] = _result_metrics(
            jax, jnp, final_states[solver]
        )
    detector_denominator = jnp.linalg.norm(phasors["elmer"])
    detector_error = _scalar(
        jax,
        jnp.linalg.norm(phasors["jax"] - phasors["elmer"])
        / jnp.maximum(detector_denominator, 1.0e-30),
    )

    hbm_capacity = _positive_environment_integer("FEMX_HBM_BYTES_PER_DEVICE")
    if hbm_capacity is None:
        raise RuntimeError("FEMX_HBM_BYTES_PER_DEVICE is required for physical evidence")
    compiler_memory = _memory_report(compiled, hbm_capacity)
    time_steps = int(elmer_config.time_steps_total)
    passed = (
        source_e_error <= 2.0e-5
        and source_h_error <= 2.0e-5
        and detector_error <= 2.0e-5
        and all(completed[solver] == time_steps for solver in SOLVERS)
        and all(cast(bool, metrics[solver]["all_fields_finite"]) for solver in SOLVERS)
        and all(cast(float, metrics[solver]["final_e_l2"]) > 0.0 for solver in SOLVERS)
        and all(cast(float, metrics[solver]["final_h_l2"]) > 0.0 for solver in SOLVERS)
        and all(cast(float, metrics[solver]["downstream_phasor_l2"]) > 0.0 for solver in SOLVERS)
    )
    process_payload: dict[str, object] = {
        "schema_version": PROCESS_EVIDENCE_SCHEMA,
        "status": "passed" if passed else "failed",
        "provenance": provenance,
        "runtime": {
            "backend": jax.default_backend(),
            "jax_version": jax.__version__,
            "jaxlib_version": distribution_version("jaxlib"),
            "fdtdx_version": distribution_version("fdtdx"),
            "x64_enabled": bool(getattr(jax.config, "jax_enable_x64", False)),
            "process_index": process_index,
            "process_count": int(jax.process_count()),
            "local_device_count": int(jax.local_device_count()),
            "global_device_count": int(jax.device_count()),
            "device_kinds": sorted({str(device.device_kind) for device in jax.devices()}),
            "scalar_contract": SCALAR_CONTRACT,
        },
        "launch_claim": launch_claim,
        "inputs": {
            "schema_version": INPUT_MANIFEST_SCHEMA,
            "manifest_sha256": manifest_sha256,
            "artifacts": artifact_records,
            "fdtdx_fingerprint": {
                "package_version": fingerprint.package_version,
                "source_revision": fingerprint.source_revision,
                "source_digest": fingerprint.source_digest,
            },
            "runtime_module_sha256": module_hashes,
        },
        "sources": {
            solver: {
                "binding": prepared[solver][5].canonical_data(),
                "binding_sha256": prepared[solver][5].sha256,
                "canonical_bundle_sha256": prepared[solver][3].report.source_bundle_sha256,
                "runtime_bundle_sha256": prepared[solver][4].mode_bundle_sha256,
                "precision_report_sha256": prepared[solver][3].report.sha256,
            }
            for solver in SOLVERS
        },
        "simulation": {
            "grid_shape_xyz": list(GRID_SHAPE),
            "source_z_index": SOURCE_Z_INDEX,
            "detector_z_index": DETECTOR_Z_INDEX,
            "frequency_hz": prepared["elmer"][3].bundle.frequency_hz,
            "simulation_time_s": SIMULATION_TIME_S,
            "time_steps": time_steps,
            "boundaries": ["pec", "pec", "pec", "pec", "pml", "pml"],
            "core_cell_count": 32,
            "cladding_relative_permittivity": CLADDING_INDEX**2,
            "core_relative_permittivity": CORE_INDEX**2,
        },
        "numerics": {
            "completed_step": completed,
            "all_fields_finite": {
                solver: metrics[solver]["all_fields_finite"] for solver in SOLVERS
            },
            "final_e_l2": {solver: metrics[solver]["final_e_l2"] for solver in SOLVERS},
            "final_h_l2": {solver: metrics[solver]["final_h_l2"] for solver in SOLVERS},
            "downstream_phasor_l2": {
                solver: metrics[solver]["downstream_phasor_l2"] for solver in SOLVERS
            },
            "source_electric_relative_l2": source_e_error,
            "source_magnetic_relative_l2": source_h_error,
            "downstream_phasor_relative_l2": detector_error,
        },
        "execution": {
            "shared_compiled_pytree": True,
            "lowering_seconds": lowering_seconds,
            "compilation_seconds": compilation_seconds,
            "warmup_seconds": warmup_seconds,
            "execution_seconds": execution_seconds,
            "compiler_memory": compiler_memory,
            "stablehlo_all_gather_count": len(re.findall(r"all[_-]gather", stablehlo.lower())),
        },
        "claim_scope": (
            "one process-local record for a physical multi-host TPU Elmer/JAX waveguide-source "
            "FDTDX run; the complete claim requires every initialized JAX process"
        ),
    }
    hlo_path = output_root / "hlo" / "fdtdx-waveguide-time-advance.stablehlo.mlir"
    hlo_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(hlo_path, stablehlo)
    _atomic_json(output_root / "results" / "process-metrics.json", process_payload)
    _publish_process_zero_compatibility(
        remote_run,
        process_index=process_index,
        process_payload=process_payload,
        stablehlo=stablehlo,
    )
    multihost_utils.sync_global_devices(f"femx-waveguide-evidence-written-{provenance['run_id']}")
    print(json.dumps({"status": process_payload["status"], "run_id": provenance["run_id"]}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
