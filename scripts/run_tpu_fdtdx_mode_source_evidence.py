#!/usr/bin/env python3
"""Run a bounded process-complete FDTDX mode-source witness on physical TPU hardware.

The Phoxla bootstrap initializes JAX distribution before evaluating this file.  The witness uses
an analytic one-watt homogeneous port so that it isolates the distributed ModeBundle/FDTDX
boundary from FEM eigensolve error.  It is therefore infrastructure evidence, not Elmer parity or
a silicon-waveguide accuracy result.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from importlib import import_module
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, cast

EVIDENCE_SCHEMA = "femx.fdtdx.mode_source.tpu_process/v1"
WORKER_ENTRY_CLAIM_SCHEMA = "femx.fdtdx.mode_source.worker_entry_claim/v1"
FDTDX_PACKAGE_VERSION = "0.6.2"
FDTDX_SOURCE_REVISION = "81a58da9cde4a4ff822f835b63597c0d0d8ba978"
FDTDX_SOURCE_DIGEST = "c881f0cf32b4272dc10e0acee716bfda3cf2d0f11973c23d31f29b11f3dce01c"
FDTDX_MODULE_SHA256 = {
    "fdtdx.core.grid": "d24739b9229ad8c61a57e4f688e6224eae63a680ff6554ddd7a5ef765edab6dd",
    "fdtdx.fdtd.wrapper": "97d562e0a33eeeccd6ce42d12cf8dc29f1e9fd071e561b5b009ac5da8ece7384",
    "fdtdx.objects.object": "24c986b9fa73bf474bce9fefc2145436654be4758e83dbcaf6fb955b7eb8557f",
    "fdtdx.objects.sources.custom_mode": (
        "0c5925a784da33f8d8236a874d4759d4ebe6df29317dcc1ce68877b4a4036df5"
    ),
    "fdtdx.objects.sources.tfsf": (
        "bd270995bffd174c7014adf9a02c7648134547c3bab7a294570e0a179326e611"
    ),
}
SCALAR_CONTRACT = {
    "field_dtype": "float32",
    "mode_dtype": "complex64",
    "time_offset_dtype": "float32",
    "x64_enabled": False,
    "precision_fallback": False,
}
RELATIVE_PERMITTIVITY = 2.085136
SOURCE_Z_INDEX = 4
GRID_SPACING_M = 40.0e-9
SIMULATION_TIME_S = 5.0e-15


def _positive_environment_integer(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def _nonnegative_environment_integer(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a nonnegative integer") from error
    if value < 0:
        raise RuntimeError(f"{name} must be a nonnegative integer")
    return value


def _require_expected_count(name: str, observed: int) -> None:
    expected = _positive_environment_integer(name)
    if expected is not None and expected != observed:
        raise RuntimeError(f"{name} requires {expected}, observed {observed}")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(
        path,
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
    )


def _publish_process_zero_compatibility(
    remote_run: Path,
    *,
    process_index: int,
    process_payload: object,
    stablehlo: str,
) -> None:
    """Publish controller-visible copies without replacing process-local authorities."""

    if process_index != 0:
        return
    _atomic_json(remote_run / "results" / "metrics.json", process_payload)
    _atomic_text(
        remote_run / "hlo" / "fdtdx-time-advance.stablehlo.mlir",
        stablehlo,
    )


def _manifest_provenance(remote_run: Path) -> dict[str, object]:
    manifest_path = remote_run / ".phoxla" / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "run_id": manifest["run_id"],
            "profile": manifest["profile"],
            "source_digest": manifest["source"]["digest"],
            "config_digest": manifest["config"]["digest"],
        }
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid deployed Phoxla manifest: {manifest_path}") from error


def _claim_worker_entry(remote_run: Path, provenance: dict[str, object]) -> dict[str, object]:
    if not remote_run.is_absolute() or not remote_run.is_dir() or remote_run.is_symlink():
        raise RuntimeError("PHOXLA_REMOTE_RUN_DIR must be an absolute non-symlink directory")
    process_index = _nonnegative_environment_integer("PHOXLA_PROCESS_INDEX")
    worker_index = _nonnegative_environment_integer("PHOXLA_GCLOUD_WORKER_INDEX")
    if process_index is None or worker_index is None:
        raise RuntimeError("Phoxla process and worker indexes are required")
    if os.environ.get("PHOXLA_RUN_ID") != provenance.get("run_id"):
        raise RuntimeError("PHOXLA_RUN_ID disagrees with the deployed manifest")
    claim_path = remote_run / "logs" / "femx-fdtdx-entry.claim"
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        claim_path.mkdir()
    except FileExistsError as error:
        raise RuntimeError("duplicate FDTDX evidence entry refused on this worker") from error
    claim = {
        "schema_version": WORKER_ENTRY_CLAIM_SCHEMA,
        "run_id": provenance["run_id"],
        "worker_index": worker_index,
        "process_index": process_index,
        "source_sha256": provenance["source_digest"],
        "config_sha256": provenance["config_digest"],
        "scope": "one immutable worker-local entry after distributed JAX initialization",
    }
    _atomic_json(claim_path / "identity.json", claim)
    return claim


def _runtime() -> Any:
    if os.environ.get("JAX_PLATFORMS") != "tpu,cpu":
        raise RuntimeError("JAX_PLATFORMS=tpu,cpu must be set before Python starts")
    import jax

    if "PHOXLA_PROCESS_INDEX" not in os.environ:
        jax.distributed.initialize()
    if jax.default_backend() != "tpu" or any(device.platform != "tpu" for device in jax.devices()):
        raise RuntimeError(f"physical TPU backend required, observed {jax.default_backend()!r}")
    if bool(getattr(jax.config, "jax_enable_x64", False)):
        raise RuntimeError("physical FDTDX witness requires float32/complex64 with x64 disabled")
    if jax.process_count() < 2 or jax.device_count() < 2:
        raise RuntimeError("physical FDTDX witness requires multiple processes and devices")
    if jax.process_count() * jax.local_device_count() != jax.device_count():
        raise RuntimeError("uniform one-process-per-worker device accounting is required")
    _require_expected_count("FEMX_EXPECTED_PROCESS_COUNT", jax.process_count())
    _require_expected_count("FEMX_EXPECTED_GLOBAL_DEVICE_COUNT", jax.device_count())
    _require_expected_count("FEMX_EXPECTED_LOCAL_DEVICE_COUNT", jax.local_device_count())
    return jax


def _verify_fdtdx_source() -> dict[str, str]:
    if distribution_version("fdtdx") != FDTDX_PACKAGE_VERSION:
        raise RuntimeError("installed FDTDX package version differs from the locked witness")
    actual: dict[str, str] = {}
    for module_name in FDTDX_MODULE_SHA256:
        module_path = Path(str(import_module(module_name).__file__)).resolve()
        actual[module_name] = hashlib.sha256(module_path.read_bytes()).hexdigest()
    if actual != FDTDX_MODULE_SHA256:
        raise RuntimeError("installed FDTDX source files differ from the locked witness")
    return actual


def _uniform_mode_bundle(
    *,
    x_edges: Any,
    y_edges: Any,
    source_z_edges: Any,
    fdtdx_fingerprint: Any,
) -> Any:
    import numpy as np

    from femx.core.axes import Axis, AxisDirection, Direction
    from femx.interop.fdtdx import (
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

    grid = build_yee_grid((x_edges, y_edges, source_z_edges))
    area = float((x_edges[-1] - x_edges[0]) * (y_edges[-1] - y_edges[0]))
    effective_index = math.sqrt(RELATIVE_PERMITTIVITY)
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
        beta_per_m=(effective_index * 2.0 * math.pi * frequency_hz / VACUUM_SPEED_OF_LIGHT_M_PER_S),
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
            target_runtime_version=fdtdx_fingerprint.package_version,
            target_source_revision=fdtdx_fingerprint.source_revision,
            target_source_digest=fdtdx_fingerprint.source_digest,
        ),
    )


def _memory_report(compiled: Any, hbm_capacity_bytes: int) -> dict[str, object]:
    analysis = compiled.memory_analysis()
    if analysis is None:
        raise RuntimeError("JAX executable did not expose compiler memory analysis")

    def measured(name: str) -> int:
        value = getattr(analysis, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"invalid JAX compiler memory statistic {name!r}: {value!r}")
        return value

    argument = measured("argument_size_in_bytes")
    output = measured("output_size_in_bytes")
    temporary = measured("temp_size_in_bytes")
    alias = measured("alias_size_in_bytes")
    peak = max(0, argument + output + temporary - alias)
    return {
        "generated_code_bytes": measured("generated_code_size_in_bytes"),
        "argument_bytes": argument,
        "output_bytes": output,
        "alias_bytes": alias,
        "temporary_bytes": temporary,
        "compiler_peak_bytes": peak,
        "hbm_capacity_bytes_per_device": hbm_capacity_bytes,
        "hbm_fraction": peak / hbm_capacity_bytes,
        "claim_scope": "compiler estimate; not live HBM usage",
    }


def _scalar(jax: Any, value: Any) -> float:
    import numpy as np

    return float(np.asarray(jax.device_get(value)))


def _run_compiled_fdtdx(
    compiled: Any,
    *,
    arrays: Any,
    objects: Any,
    config: Any,
    key: Any,
) -> Any:
    """Call an AOT executable with only the dynamic pytree used during lowering."""

    return compiled(arrays=arrays, objects=objects, config=config, key=key)


def main() -> int:
    remote_run = Path(os.environ["PHOXLA_REMOTE_RUN_DIR"])
    provenance = _manifest_provenance(remote_run)
    launch_claim = _claim_worker_entry(remote_run, provenance)
    jax = _runtime()
    import fdtdx  # type: ignore[import-not-found]
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import multihost_utils

    from femx.interop.fdtdx import (
        FDTDXFingerprint,
        bind_fdtdx_distributed_mode_source,
        build_fdtdx_mode_source_contract,
        lower_mode_source_inputs_for_tpu,
        make_fdtdx_distributed_mode_source,
    )

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

    global_device_count = int(jax.device_count())
    x_cells = 2 * global_device_count
    y_cells = 8
    z_cells = 20
    x_edges = np.arange(x_cells + 1, dtype=np.float64) * GRID_SPACING_M
    y_edges = np.arange(y_cells + 1, dtype=np.float64) * GRID_SPACING_M
    z_edges = np.arange(z_cells + 1, dtype=np.float64) * GRID_SPACING_M
    canonical_bundle = _uniform_mode_bundle(
        x_edges=x_edges,
        y_edges=y_edges,
        source_z_edges=z_edges[SOURCE_Z_INDEX : SOURCE_Z_INDEX + 2],
        fdtdx_fingerprint=fingerprint,
    )
    inverse_permittivity = np.full(
        (1, *canonical_bundle.electric.grid.shape),
        1.0 / RELATIVE_PERMITTIVITY,
        dtype=np.float64,
    )
    lowered = lower_mode_source_inputs_for_tpu(
        canonical_bundle,
        expected_inverse_permittivity=inverse_permittivity,
        expected_inverse_permeability=np.asarray(1.0, dtype=np.float64),
    )
    contract = build_fdtdx_mode_source_contract(
        lowered.bundle,
        source_name="femx-distributed-port",
        expected_inverse_permittivity=lowered.expected_inverse_permittivity,
        expected_inverse_permeability=lowered.expected_inverse_permeability,
        fdtdx=fingerprint,
    )
    source = cast(
        Any,
        make_fdtdx_distributed_mode_source(
            lowered.bundle,
            contract,
            verified_fingerprint=fingerprint,
        ),
    )

    volume = fdtdx.SimulationVolume(
        partial_grid_shape=(x_cells, y_cells, z_cells),
        material=fdtdx.Material(permittivity=RELATIVE_PERMITTIVITY),
    )
    boundaries, boundary_constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(
            thickness=1,
            override_types={
                face: "periodic" for face in ("min_x", "max_x", "min_y", "max_y", "min_z", "max_z")
            },
        ),
        volume,
    )
    constraints = [
        *boundary_constraints,
        source.same_size(volume, axes=(0, 1)),
        source.place_at_center(volume, axes=(0, 1)),
        fdtdx.RealCoordinateConstraint(
            object=contract.source_name,
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
    key = jax.random.PRNGKey(20260902)
    objects, arrays, parameters, config, _placement = fdtdx.place_objects(
        [volume, *boundaries.values(), source],
        config,
        constraints,
        key=key,
    )
    # The v1 eager validator intentionally requires a fully host-addressable source.  On a
    # multi-controller runtime the distributed callback has already checked each addressable
    # material/field shard; bind_fdtdx_distributed_mode_source repeats that shard-wise contract.
    arrays, objects, _application = fdtdx.apply_params(
        arrays=arrays,
        objects=objects,
        params=parameters,
        key=key,
    )
    objects, binding = bind_fdtdx_distributed_mode_source(
        objects,
        lowered.bundle,
        contract,
    )
    initial_e_l2 = _scalar(jax, jnp.linalg.norm(arrays.fields.E))
    initial_h_l2 = _scalar(jax, jnp.linalg.norm(arrays.fields.H))

    run = jax.jit(
        fdtdx.run_fdtd,
        static_argnames=("show_progress", "progress_callback"),
    )
    started = time.perf_counter()
    lowered_executable = run.lower(
        arrays=arrays,
        objects=objects,
        config=config,
        key=key,
        show_progress=False,
        progress_callback=None,
    )
    lowering_seconds = time.perf_counter() - started
    stablehlo = str(lowered_executable.compiler_ir("stablehlo"))
    started = time.perf_counter()
    compiled = lowered_executable.compile()
    compilation_seconds = time.perf_counter() - started
    started = time.perf_counter()
    warmup_state = _run_compiled_fdtdx(
        compiled,
        arrays=arrays,
        objects=objects,
        config=config,
        key=key,
    )
    jax.block_until_ready(warmup_state)
    warmup_seconds = time.perf_counter() - started
    started = time.perf_counter()
    final_step, final_arrays = _run_compiled_fdtdx(
        compiled,
        arrays=arrays,
        objects=objects,
        config=config,
        key=key,
    )
    jax.block_until_ready((final_step, final_arrays))
    execution_seconds = time.perf_counter() - started

    final_e_l2 = _scalar(jax, jnp.linalg.norm(final_arrays.fields.E))
    final_h_l2 = _scalar(jax, jnp.linalg.norm(final_arrays.fields.H))
    downstream_e_l2 = _scalar(
        jax,
        jnp.linalg.norm(final_arrays.fields.E[:, :, :, SOURCE_Z_INDEX + 2 :]),
    )
    all_fields_finite = bool(
        np.asarray(
            jax.device_get(
                jnp.all(jnp.isfinite(final_arrays.fields.E))
                & jnp.all(jnp.isfinite(final_arrays.fields.H))
            )
        )
    )
    completed_step = int(np.asarray(jax.device_get(final_step)))
    hbm_capacity_bytes = _positive_environment_integer("FEMX_HBM_BYTES_PER_DEVICE")
    if hbm_capacity_bytes is None:
        raise RuntimeError("FEMX_HBM_BYTES_PER_DEVICE is required for physical evidence")
    compiler_memory = _memory_report(compiled, hbm_capacity_bytes)
    binding_data = dict(binding.canonical_data())
    passed = (
        initial_e_l2 == 0.0
        and initial_h_l2 == 0.0
        and final_e_l2 > 0.0
        and final_h_l2 > 0.0
        and downstream_e_l2 > 0.0
        and all_fields_finite
        and completed_step == int(config.time_steps_total)
        and binding.field_dtype == "complex64"
        and binding.time_offset_dtype == "float32"
    )
    process_payload = {
        "schema_version": EVIDENCE_SCHEMA,
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
            "global_device_count": global_device_count,
            "device_kinds": sorted({str(device.device_kind) for device in jax.devices()}),
            "scalar_contract": SCALAR_CONTRACT,
        },
        "launch_claim": launch_claim,
        "source": {
            "binding": binding_data,
            "binding_sha256": binding.sha256,
            "bundle_sha256": contract.mode_bundle_sha256,
            "precision_report": lowered.report.canonical_data(),
            "fdtdx_fingerprint": {
                "package_version": fingerprint.package_version,
                "source_revision": fingerprint.source_revision,
                "source_digest": fingerprint.source_digest,
            },
            "module_sha256": module_hashes,
            "profile": "analytic one-watt homogeneous +z port",
        },
        "simulation": {
            "grid_shape_xyz": [x_cells, y_cells, z_cells],
            "source_z_index": SOURCE_Z_INDEX,
            "simulation_time_s": SIMULATION_TIME_S,
            "time_steps": int(config.time_steps_total),
            "relative_permittivity": RELATIVE_PERMITTIVITY,
            "boundaries": ["periodic"] * 6,
        },
        "numerics": {
            "completed_step": completed_step,
            "initial_e_l2": initial_e_l2,
            "initial_h_l2": initial_h_l2,
            "final_e_l2": final_e_l2,
            "final_h_l2": final_h_l2,
            "downstream_e_l2": downstream_e_l2,
            "all_fields_finite": all_fields_finite,
        },
        "execution": {
            "lowering_seconds": lowering_seconds,
            "compilation_seconds": compilation_seconds,
            "warmup_seconds": warmup_seconds,
            "execution_seconds": execution_seconds,
            "compiler_memory": compiler_memory,
            "stablehlo_all_gather_count": len(re.findall(r"all[_-]gather", stablehlo.lower())),
        },
        "claim_scope": (
            "one process-local record for a physical multi-host TPU FDTDX source run; the "
            "complete claim requires aggregation of every initialized JAX process"
        ),
    }
    hlo_path = output_root / "hlo" / "fdtdx-time-advance.stablehlo.mlir"
    hlo_path.parent.mkdir(parents=True, exist_ok=True)
    hlo_path.write_text(stablehlo, encoding="utf-8")
    _atomic_json(output_root / "results" / "process-metrics.json", process_payload)
    if process_index == 0:
        _atomic_json(output_root / "results" / "metrics.json", process_payload)
    _publish_process_zero_compatibility(
        remote_run,
        process_index=process_index,
        process_payload=process_payload,
        stablehlo=stablehlo,
    )
    multihost_utils.sync_global_devices(f"femx-fdtdx-evidence-written-{provenance['run_id']}")
    print(json.dumps({"status": process_payload["status"], "run_id": provenance["run_id"]}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
