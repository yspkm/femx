#!/usr/bin/env python3
"""Run one physical-TPU distributed electrothermal-to-FDTDX gradient witness."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SOURCE_ROOT):
    import_path = str(import_root)
    while import_path in sys.path:
        sys.path.remove(import_path)
    sys.path.insert(0, import_path)

from femx.validation.tpu_distributed_fdtdx_thermo_optic_evidence import (
    CRITICAL_ARRAY_REPORT_SCHEMA,
    EXECUTABLE_NAMES,
    EXPECTED_GLOBAL_DEVICE_COUNT,
    EXPECTED_LOCAL_DEVICE_COUNT,
    EXPECTED_PROCESS_COUNT,
    FINITE_DIFFERENCE_STEPS,
    PROCESS_EVIDENCE_SCHEMA,
    REPLICATION_INTENT,
    SCALAR_CONTRACT,
    TOLERANCES,
    WORKER_ENTRY_CLAIM_SCHEMA,
)

EXECUTION_SAMPLES = 3
_ELECTROTHERMAL_PARTITIONED_FIELDS = {
    "cell_local_dofs",
    "owner_mask",
    "cell_mask",
    "unit_stiffness",
    "basis_gradients",
    "cell_areas",
    "current_conductivity_base",
    "current_conductivity_weights",
    "current_cell_load_base",
    "current_cell_load_weights",
    "current_dirichlet_base",
    "current_dirichlet_weights",
    "thermal_conductivity_base",
    "thermal_conductivity_weights",
    "thermal_cell_load_base",
    "thermal_cell_load_weights",
    "thermal_dirichlet_base",
    "thermal_dirichlet_weights",
    "feedback_reference_base",
    "feedback_reference_weights",
    "feedback_coefficient_base",
    "feedback_coefficient_weights",
}


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


def _runtime() -> Any:
    if os.environ.get("JAX_PLATFORMS") != "tpu,cpu":
        raise RuntimeError("JAX_PLATFORMS=tpu,cpu must be set before Python starts")
    if os.environ.get("JAX_DEFAULT_MATMUL_PRECISION") != "highest":
        raise RuntimeError("JAX_DEFAULT_MATMUL_PRECISION=highest must be set before Python starts")
    import jax

    if "PHOXLA_PROCESS_INDEX" not in os.environ:
        jax.distributed.initialize()
    if jax.default_backend() != "tpu" or any(device.platform != "tpu" for device in jax.devices()):
        raise RuntimeError(f"physical TPU backend required, observed {jax.default_backend()!r}")
    if bool(getattr(jax.config, "jax_enable_x64", False)):
        raise RuntimeError("physical coupled FDTDX evidence requires JAX x64 disabled")
    if str(getattr(jax.config, "jax_default_matmul_precision", None)) != "highest":
        raise RuntimeError("JAX default matmul precision must resolve to 'highest'")
    observed = (
        int(jax.process_count()),
        int(jax.local_device_count()),
        int(jax.device_count()),
    )
    expected = (
        EXPECTED_PROCESS_COUNT,
        EXPECTED_LOCAL_DEVICE_COUNT,
        EXPECTED_GLOBAL_DEVICE_COUNT,
    )
    if observed != expected:
        raise RuntimeError(
            f"physical coupled FDTDX topology must be {expected}, observed {observed}"
        )
    for name, count in (
        ("FEMX_EXPECTED_PROCESS_COUNT", observed[0]),
        ("FEMX_EXPECTED_LOCAL_DEVICE_COUNT", observed[1]),
        ("FEMX_EXPECTED_GLOBAL_DEVICE_COUNT", observed[2]),
    ):
        configured = _positive_environment_integer(name)
        if configured is not None and configured != count:
            raise RuntimeError(f"{name} requires {configured}, observed {count}")
    return jax


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n")


def _publish_process_zero_compatibility(
    remote_run: Path,
    *,
    process_index: int,
    process_payload: object,
    stablehlo_by_name: Mapping[str, str],
) -> None:
    """Publish controller-visible copies without replacing process-local authorities."""

    if process_index != 0:
        return
    _atomic_json(remote_run / "results" / "metrics.json", process_payload)
    for name, stablehlo in stablehlo_by_name.items():
        _atomic_text(remote_run / "hlo" / f"{name}.stablehlo.mlir", stablehlo)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_provenance(remote_run: Path) -> dict[str, object]:
    manifest_path = remote_run / ".phoxla" / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = manifest["source"]
        return {
            "run_id": manifest["run_id"],
            "profile": manifest["profile"],
            "source_commit": source["commit"],
            "source_digest": source["digest"],
            "config_digest": manifest["config"]["digest"],
        }
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid deployed Phoxla manifest: {manifest_path}") from error


def _claim_worker_entry(remote_run: Path, provenance: Mapping[str, object]) -> dict[str, object]:
    if not remote_run.is_absolute() or not remote_run.is_dir() or remote_run.is_symlink():
        raise RuntimeError("PHOXLA_REMOTE_RUN_DIR must be an absolute non-symlink directory")
    process_index = _nonnegative_environment_integer("PHOXLA_PROCESS_INDEX")
    worker_index = _nonnegative_environment_integer("PHOXLA_GCLOUD_WORKER_INDEX")
    if process_index is None or worker_index is None:
        raise RuntimeError("Phoxla process and worker indexes are required before entry")
    if os.environ.get("PHOXLA_RUN_ID") != provenance.get("run_id"):
        raise RuntimeError("PHOXLA_RUN_ID disagrees with the deployed manifest")
    claim_path = remote_run / "logs" / "femx-fdtdx-thermo-optic-entry.claim"
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        claim_path.mkdir()
    except FileExistsError as error:
        raise RuntimeError("duplicate coupled FDTDX entry refused on this worker") from error
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


def _json_safe(value: object) -> object:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _relative_difference(jax: Any, observed: Any, expected: Any) -> float:
    import jax.numpy as jnp
    import numpy as np

    numerator = jnp.linalg.norm(observed - expected)
    denominator = jnp.linalg.norm(expected)
    result = jnp.where(
        denominator > 0.0,
        numerator / denominator,
        jnp.where(numerator == 0.0, 0.0, jnp.inf),
    )
    return float(np.asarray(jax.device_get(result)))


def _memory_report(compiled: Any, hbm_capacity_bytes: int) -> dict[str, object]:
    from femx.backends.jax.collective_runtime import CollectiveCompilerMemoryReport

    analysis = compiled.memory_analysis()
    if analysis is None:
        raise RuntimeError("JAX executable did not expose compiler memory analysis")

    def measured(name: str) -> int:
        value = getattr(analysis, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"invalid JAX compiler memory statistic {name!r}: {value!r}")
        return value

    return CollectiveCompilerMemoryReport(
        generated_code_bytes=measured("generated_code_size_in_bytes"),
        argument_bytes=measured("argument_size_in_bytes"),
        output_bytes=measured("output_size_in_bytes"),
        alias_bytes=measured("alias_size_in_bytes"),
        temporary_bytes=measured("temp_size_in_bytes"),
        hbm_capacity_bytes_per_device=hbm_capacity_bytes,
    ).canonical_data()


def _compile_and_time(
    jax: Any,
    function: Any,
    arguments: tuple[Any, ...],
) -> tuple[Any, dict[str, object], str]:
    from femx.backends.jax.collective_runtime import CollectiveTimingReport

    started = time.perf_counter()
    lowered = jax.jit(function).lower(*arguments)
    lowering_seconds = time.perf_counter() - started
    stablehlo = str(lowered.compiler_ir("stablehlo"))
    started = time.perf_counter()
    compiled = lowered.compile()
    compilation_seconds = time.perf_counter() - started
    started = time.perf_counter()
    warmup = compiled(*arguments)
    jax.block_until_ready(warmup)
    warmup_seconds = time.perf_counter() - started
    samples: list[float] = []
    for _ in range(EXECUTION_SAMPLES):
        started = time.perf_counter()
        result = compiled(*arguments)
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - started)
    timing = CollectiveTimingReport(
        lowering_seconds=lowering_seconds,
        compilation_seconds=compilation_seconds,
        warmup_seconds=warmup_seconds,
        execution_seconds=tuple(samples),
    )
    return compiled, timing.canonical_data(), stablehlo


def _stablehlo_report(text: str) -> dict[str, object]:
    lowered = text.lower()
    return {
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "all_to_all_count": lowered.count("stablehlo.all_to_all"),
        "collective_permute_count": lowered.count("stablehlo.collective_permute"),
        "all_reduce_count": lowered.count("stablehlo.all_reduce"),
        "contains_all_gather": "all_gather" in lowered,
        "contains_float64": "f64" in lowered,
    }


def _host_pack_owned(layout: Any, full_node_values: Any) -> Any:
    import numpy as np

    full = np.asarray(full_node_values)
    free = full[np.asarray(layout.topology.free_nodes)]
    extended = np.concatenate((free, np.zeros((1,), dtype=free.dtype)))
    return np.ascontiguousarray(extended[layout.transport.owned_dof_ids])


def _host_pack_cell_temperature(plan: Any, full_node_temperature: Any) -> Any:
    import numpy as np

    nodal = np.asarray(full_node_temperature, dtype=np.float64)
    canonical = nodal[np.asarray(plan.layout.topology.cells, dtype=np.int64)]
    extended = np.concatenate((canonical, np.zeros((1, 3), dtype=np.float64)), axis=0)
    return np.ascontiguousarray(extended[np.asarray(plan.layout.transport.cell_ids)])


def _partition_spec(sharding: object) -> list[object]:
    spec = getattr(sharding, "spec", None)
    if spec is None:
        raise RuntimeError("critical FDTDX array must use NamedSharding")
    result: list[object] = []
    for entry in tuple(spec):
        if entry is None or isinstance(entry, str):
            result.append(entry)
        elif isinstance(entry, tuple) and all(isinstance(item, str) for item in entry):
            result.append(list(entry))
        else:
            raise RuntimeError(f"unsupported critical FDTDX PartitionSpec entry: {entry!r}")
    return result


def _slice_bounds(index: Sequence[object], shape: Sequence[int]) -> list[list[int]]:
    if len(index) != len(shape):
        raise RuntimeError("critical FDTDX shard index rank differs from the array")
    bounds: list[list[int]] = []
    for item, extent in zip(index, shape, strict=True):
        if not isinstance(item, slice) or item.step not in (None, 1):
            raise RuntimeError("critical FDTDX shard index must use contiguous slices")
        start = 0 if item.start is None else int(item.start)
        stop = int(extent) if item.stop is None else int(item.stop)
        bounds.append([start, stop])
    return bounds


def _critical_array_report(
    name: str,
    array: Any,
    mesh: Any,
    *,
    process_index: int,
    process_count: int,
) -> dict[str, object]:
    device_partitions = {device: index for index, device in enumerate(mesh.devices.reshape(-1))}
    shards = []
    for shard in array.addressable_shards:
        if shard.device not in device_partitions:
            raise RuntimeError("critical FDTDX shard device is outside the declared Mesh")
        shards.append(
            {
                "partition_index": device_partitions[shard.device],
                "process_index": process_index,
                "device_kind": str(shard.device.device_kind),
                "index": _slice_bounds(tuple(shard.index), tuple(array.shape)),
                "local_shape": list(shard.data.shape),
            }
        )
    return {
        "schema_version": CRITICAL_ARRAY_REPORT_SCHEMA,
        "name": name,
        "global_shape": list(array.shape),
        "dtype": str(array.dtype),
        "partition_spec": _partition_spec(array.sharding),
        "process_index": process_index,
        "process_count": process_count,
        "global_device_count": mesh.size,
        "addressable_shards": sorted(shards, key=lambda item: cast(int, item["partition_index"])),
    }


def _coordinate_admission(
    jax: Any, target_coordinates: object, expected: object
) -> dict[str, object]:
    import jax.numpy as jnp
    import numpy as np

    from scripts._distributed_fdtdx_thermo_optic_case import (
        GRID_SPACING_M,
        RUNTIME_TARGET_COORDINATE_MAX_GRID_FRACTION_ERROR,
        RUNTIME_TARGET_COORDINATE_MAX_ULP_ERROR,
    )

    actual_axes = tuple(cast(Sequence[object], target_coordinates))
    expected_axes = tuple(cast(Sequence[object], expected))
    if len(actual_axes) != 3 or len(expected_axes) != 3:
        raise RuntimeError("runtime coordinate admission requires exactly three axes")

    absolute_errors: list[float] = []
    fractions: list[float] = []
    ulps: list[int] = []
    exact: list[bool] = []
    admitted: list[bool] = []
    for actual_raw, expected_raw in zip(actual_axes, expected_axes, strict=True):
        expected64 = np.asarray(expected_raw, dtype=np.float64)
        expected32 = np.asarray(expected64, dtype=np.float32)
        actual = jnp.asarray(actual_raw, dtype=jnp.float32)
        rounded = jnp.asarray(expected32, dtype=jnp.float32)
        if tuple(actual.shape) != tuple(rounded.shape):
            raise RuntimeError("runtime and controller target-coordinate shapes differ")
        actual_bits = jax.lax.bitcast_convert_type(actual, jnp.uint32)
        rounded_bits = jax.lax.bitcast_convert_type(rounded, jnp.uint32)
        sign = jnp.asarray(1 << 31, dtype=jnp.uint32)
        maximum = jnp.asarray((1 << 32) - 1, dtype=jnp.uint32)
        actual_ordered = jnp.where(
            (actual_bits & sign) != 0,
            maximum - actual_bits,
            actual_bits + sign,
        )
        rounded_ordered = jnp.where(
            (rounded_bits & sign) != 0,
            maximum - rounded_bits,
            rounded_bits + sign,
        )
        finite = jnp.isfinite(actual) & jnp.isfinite(rounded)
        bit_distance = jnp.where(
            finite,
            jnp.maximum(actual_ordered, rounded_ordered)
            - jnp.minimum(actual_ordered, rounded_ordered),
            maximum,
        )
        runtime_rounding_error = float(
            np.asarray(jax.device_get(jnp.max(jnp.abs(actual - rounded))))
        )
        controller_rounding_error = float(
            np.max(np.abs(expected64 - expected32.astype(np.float64)))
        )
        maximum_error = controller_rounding_error + runtime_rounding_error
        maximum_fraction = maximum_error / GRID_SPACING_M
        maximum_ulp = int(np.asarray(jax.device_get(jnp.max(bit_distance))))
        is_exact = bool(np.asarray(jax.device_get(jnp.all(actual == rounded))))
        is_admitted = (
            maximum_ulp <= RUNTIME_TARGET_COORDINATE_MAX_ULP_ERROR
            and maximum_fraction <= RUNTIME_TARGET_COORDINATE_MAX_GRID_FRACTION_ERROR
        )
        absolute_errors.append(maximum_error)
        fractions.append(maximum_fraction)
        ulps.append(maximum_ulp)
        exact.append(is_exact)
        admitted.append(is_admitted)
    return {
        "maximum_absolute_errors_m": absolute_errors,
        "maximum_grid_fraction_errors": fractions,
        "maximum_ulp_errors": ulps,
        "float32_rounding_exact": exact,
        "admitted": admitted,
    }


def _material_relative_difference(
    jax: Any,
    applied_inverse_permittivity: Any,
    expected_relative_permittivity: object,
    *,
    device_grid_slice: Sequence[slice],
) -> float:
    import numpy as np
    from jax.experimental import multihost_utils

    expected = np.asarray(expected_relative_permittivity, dtype=np.float32)
    device_x = device_grid_slice[0]
    numerator_squared = 0.0
    denominator_squared = 0.0
    for shard in applied_inverse_permittivity.addressable_shards:
        x_slice = cast(slice, shard.index[1])
        x_start = 0 if x_slice.start is None else int(x_slice.start)
        x_stop = (
            applied_inverse_permittivity.shape[1] if x_slice.stop is None else int(x_slice.stop)
        )
        overlap_start = max(x_start, cast(int, device_x.start))
        overlap_stop = min(x_stop, cast(int, device_x.stop))
        if overlap_start >= overlap_stop:
            continue
        local_start = overlap_start - x_start
        local_stop = overlap_stop - x_start
        expected_start = overlap_start - cast(int, device_x.start)
        expected_stop = overlap_stop - cast(int, device_x.start)
        local_inverse = np.asarray(shard.data)[
            0,
            local_start:local_stop,
            device_grid_slice[1],
            device_grid_slice[2],
        ]
        local_observed = 1.0 / local_inverse
        local_expected = expected[expected_start:expected_stop]
        numerator_squared += float(
            np.sum((local_observed.astype(np.float64) - local_expected.astype(np.float64)) ** 2)
        )
        denominator_squared += float(np.sum(local_expected.astype(np.float64) ** 2))
    gathered = np.asarray(
        multihost_utils.process_allgather(
            np.asarray([numerator_squared, denominator_squared], dtype=np.float32),
            tiled=False,
        )
    ).reshape(jax.process_count(), 2)
    numerator = float(np.sum(gathered[:, 0]))
    denominator = float(np.sum(gathered[:, 1]))
    if denominator <= 0.0:
        raise RuntimeError("canonical thermo-optic material has zero norm")
    return math.sqrt(numerator / denominator)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    remote_run = Path(os.environ["PHOXLA_REMOTE_RUN_DIR"])
    provenance = _manifest_provenance(remote_run)
    jax = _runtime()
    launch_claim = _claim_worker_entry(remote_run, provenance)
    import fdtdx  # type: ignore[import-not-found]
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import multihost_utils
    from jax.sharding import Mesh

    from femx.backends.jax.collective_runtime import (
        describe_collective_mesh,
        make_collective_array_from_process_local_data,
        make_replicated_array_from_process_local_data,
    )
    from femx.backends.jax.distributed_electrothermal import (
        ElectrothermalAdjointPolicy,
        PackedDistributedElectrothermalInputs,
        build_distributed_electrothermal_runtime,
        pack_distributed_electrothermal_inputs_host,
    )
    from femx.backends.jax.scalar_cg import (
        ScalarH1CGPolicy,
        ScalarH1JacobiPolicy,
        build_packed_scalar_h1_jacobi_preconditioner_factory,
    )
    from femx.interop.fdtdx import (
        PackedDistributedThermoOpticInputs,
        build_distributed_thermo_optic_runtime,
        pack_distributed_thermo_optic_inputs_host,
        thermo_optic_parameter_state,
        with_fdtdx_device_parameter,
    )
    from scripts._distributed_fdtdx_thermo_optic_case import (
        DETECTOR_NAME,
        DEVICE_GRID_SLICE,
        FDTDX_SOURCE_DIGEST,
        FDTDX_SOURCE_REVISION,
        MESH_AXIS_NAME,
        PHASOR_OBJECTIVE_SCALE,
        CoupledRuntimeInputs,
        build_scene,
        coupled_mesh_from_material_sharding,
        verify_locked_fdtdx,
    )
    from scripts._tpu_distributed_fdtdx_thermo_optic_plan import (
        MANIFEST_FILENAME,
        read_distributed_fdtdx_thermo_optic_artifact,
    )

    process_index = int(jax.process_index())
    process_count = int(jax.process_count())
    global_device_count = int(jax.device_count())
    if cast(int, launch_claim["process_index"]) != process_index:
        raise RuntimeError("worker entry claim disagrees with initialized JAX process identity")
    input_root = (remote_run / arguments.input).resolve(strict=True)
    if remote_run.resolve() not in input_root.parents:
        raise RuntimeError("coupled FDTDX input must remain inside the deployed run")
    loaded = read_distributed_fdtdx_thermo_optic_artifact(input_root)
    if loaded.manifest.get("source_commit") != provenance.get("source_commit"):
        raise RuntimeError("controller artifact source commit differs from deployed femx")
    if not loaded.electrothermal.authority.forward_converged:
        raise RuntimeError("controller electrothermal authority did not converge")
    plan = loaded.electrothermal.plan
    authority = loaded.electrothermal.authority
    if plan.layout.partition_count != global_device_count:
        raise RuntimeError("controller artifact partition count must equal TPU device count")

    module_hashes = verify_locked_fdtdx(fdtdx)
    scene = build_scene(fdtdx, jax, jnp, backend="tpu")
    coordinates = _coordinate_admission(
        jax,
        scene.target_coordinates,
        loaded.sampling.target_coordinates,
    )
    if not all(cast(list[bool], coordinates["admitted"])):
        raise RuntimeError("runtime FDTDX coordinates exceed the controller artifact tolerance")
    mesh = coupled_mesh_from_material_sharding(
        scene.material_array_shardings.inv_permittivities,
        Mesh,
        axis_name=MESH_AXIS_NAME,
        global_device_count=global_device_count,
    )
    mesh_report = describe_collective_mesh(
        plan.layout.transport,
        mesh,
        axis_name=MESH_AXIS_NAME,
        layout_sha256=plan.layout.digest(),
    )
    partitioned_reports: dict[str, object] = {}
    replicated_reports: dict[str, object] = {}

    def load_partitioned(name: str, value: Any) -> Any:
        array, report = make_collective_array_from_process_local_data(
            name,
            value,
            mesh,
            axis_name=MESH_AXIS_NAME,
        )
        partitioned_reports[name] = report.canonical_data()
        return array

    def load_replicated(name: str, value: Any) -> Any:
        raw = np.asarray(value)
        transport = raw.reshape((1,)) if raw.ndim == 0 else raw
        array, report = make_replicated_array_from_process_local_data(
            name,
            transport,
            mesh,
            replication_intent=REPLICATION_INTENT,
        )
        replicated_reports[name] = report.canonical_data()
        return array[0] if raw.ndim == 0 else array

    host_electrothermal = pack_distributed_electrothermal_inputs_host(
        plan,
        value_dtype=np.float32,
    )
    packed_values = []
    for name, value in zip(host_electrothermal._fields, host_electrothermal, strict=True):
        report_name = f"input-{name.replace('_', '-')}"
        loader = load_partitioned if name in _ELECTROTHERMAL_PARTITIONED_FIELDS else load_replicated
        packed_values.append(loader(report_name, value))
    electrothermal_inputs = PackedDistributedElectrothermalInputs(*packed_values)
    current = load_replicated("current-parameters", plan.current_initial.astype(np.float32))
    thermal = load_replicated("thermal-parameters", plan.thermal_initial.astype(np.float32))
    feedback = load_replicated("feedback-parameters", plan.feedback_initial.astype(np.float32))
    expected_potential = load_partitioned(
        "authority-potential",
        _host_pack_owned(plan.layout, authority.potential).astype(np.float32),
    )
    expected_temperature = load_partitioned(
        "authority-temperature",
        _host_pack_owned(plan.layout, authority.temperature).astype(np.float32),
    )
    expected_cell_temperature = load_partitioned(
        "authority-cell-temperature",
        _host_pack_cell_temperature(plan, authority.temperature).astype(np.float32),
    )
    canonical_thermo_optic = thermo_optic_parameter_state(
        loaded.sampling,
        authority.temperature,
        loaded.law,
        loaded.contract,
    )
    expected_parameter = load_partitioned(
        "authority-thermo-optic-parameter",
        np.asarray(canonical_thermo_optic.parameter, dtype=np.float32),
    )

    host_transfer = pack_distributed_thermo_optic_inputs_host(
        loaded.transfer,
        value_dtype=np.float32,
    )
    transfer_values = [
        load_partitioned(f"transfer-{name.replace('_', '-')}", value)
        for name, value in zip(host_transfer._fields, host_transfer, strict=True)
    ]
    transfer_inputs = PackedDistributedThermoOpticInputs(*transfer_values)

    preconditioner = build_packed_scalar_h1_jacobi_preconditioner_factory(
        plan.layout,
        mesh,
        ScalarH1JacobiPolicy(),
        axis_name=MESH_AXIS_NAME,
    )
    electrothermal_runtime = build_distributed_electrothermal_runtime(
        plan,
        mesh,
        ScalarH1CGPolicy(2.0e-5, 0.0, 1000, backward_error_tolerance=5.0e-7),
        ElectrothermalAdjointPolicy(5.0e-4, 0.0, 20, 60),
        axis_name=MESH_AXIS_NAME,
        linear_preconditioner_factory=preconditioner,
    )
    transfer_runtime = build_distributed_thermo_optic_runtime(
        loaded.transfer,
        mesh,
        loaded.law,
        loaded.contract,
    )
    runtime_inputs = CoupledRuntimeInputs(
        electrothermal=electrothermal_inputs,
        thermo_optic=transfer_inputs,
        fdtdx_arrays=scene.arrays,
        fdtdx_objects=scene.objects,
        fdtdx_parameters=scene.parameters,
        fdtdx_config=scene.config,
        fdtdx_key=scene.key,
    )
    material_array_shardings = scene.material_array_shardings
    device_contract = loaded.contract

    def downstream_phasor(inputs: CoupledRuntimeInputs, cell_temperature: Any) -> Any:
        thermo_optic = transfer_runtime.state(inputs.thermo_optic, cell_temperature)
        parameters = with_fdtdx_device_parameter(
            inputs.fdtdx_parameters,
            thermo_optic,
            device_contract,
        )
        arrays, objects, _application = fdtdx.apply_params(
            arrays=inputs.fdtdx_arrays,
            objects=inputs.fdtdx_objects,
            params=parameters,
            key=inputs.fdtdx_key,
            material_array_shardings=material_array_shardings,
        )
        _completed_step, final_arrays = fdtdx.run_fdtd(
            arrays=arrays,
            objects=objects,
            config=inputs.fdtdx_config,
            key=inputs.fdtdx_key,
            show_progress=False,
        )
        return jnp.sum(final_arrays.detector_states[DETECTOR_NAME]["phasor"])

    def reference_phasor_function(
        inputs: CoupledRuntimeInputs,
        current_values: Any,
        thermal_values: Any,
        feedback_values: Any,
    ) -> Any:
        state = electrothermal_runtime.state(
            inputs.electrothermal,
            current_values,
            thermal_values,
            feedback_values,
        )
        cell_temperature = electrothermal_runtime.cell_temperature(
            inputs.electrothermal,
            state,
            thermal_values,
        )
        return downstream_phasor(inputs, cell_temperature)

    reference_arguments = (runtime_inputs, current, thermal, feedback)
    compiled_reference, reference_timing, reference_hlo = _compile_and_time(
        jax,
        reference_phasor_function,
        reference_arguments,
    )
    reference_phasor = compiled_reference(*reference_arguments)
    jax.block_until_ready(reference_phasor)
    reference_value = complex(np.asarray(jax.device_get(reference_phasor)))
    if not math.isfinite(reference_value.real) or not math.isfinite(reference_value.imag):
        raise RuntimeError("nominal FDTDX reference phasor must be finite")
    if abs(reference_value) == 0.0:
        raise RuntimeError("nominal FDTDX reference phasor must be nonzero")
    frozen_reference = jax.lax.stop_gradient(reference_phasor)

    def downstream_objective(
        inputs: CoupledRuntimeInputs,
        cell_temperature: Any,
        reference: Any,
    ) -> Any:
        phasor = downstream_phasor(inputs, cell_temperature)
        reference_power = jnp.real(reference * jnp.conj(reference))
        normalized = phasor * jnp.conj(reference) / reference_power
        return PHASOR_OBJECTIVE_SCALE * jnp.imag(normalized)

    def objective_function(
        inputs: CoupledRuntimeInputs,
        current_values: Any,
        thermal_values: Any,
        feedback_values: Any,
        reference: Any,
    ) -> Any:
        state = electrothermal_runtime.state(
            inputs.electrothermal,
            current_values,
            thermal_values,
            feedback_values,
        )
        cell_temperature = electrothermal_runtime.cell_temperature(
            inputs.electrothermal,
            state,
            thermal_values,
        )
        return downstream_objective(inputs, cell_temperature, reference)

    def explicit_objective_and_gradients(
        inputs: CoupledRuntimeInputs,
        current_values: Any,
        thermal_values: Any,
        feedback_values: Any,
        reference: Any,
    ) -> Any:
        state = electrothermal_runtime.state(
            inputs.electrothermal,
            current_values,
            thermal_values,
            feedback_values,
        )
        cell_temperature = electrothermal_runtime.cell_temperature(
            inputs.electrothermal,
            state,
            thermal_values,
        )
        objective, downstream_pullback = jax.vjp(
            lambda value: downstream_objective(inputs, value, reference),
            cell_temperature,
        )
        (cell_cotangent,) = downstream_pullback(jnp.ones_like(objective))

        def cell_map(state_value: Any, thermal_parameter_value: Any) -> Any:
            return electrothermal_runtime.cell_temperature(
                inputs.electrothermal,
                state_value,
                thermal_parameter_value,
            )

        _cell_value, cell_pullback = jax.vjp(cell_map, state, thermal_values)
        state_cotangent, direct_thermal_gradient = cell_pullback(cell_cotangent)
        explicit = electrothermal_runtime.vjp(
            inputs.electrothermal,
            current_values,
            thermal_values,
            feedback_values,
            state_cotangent,
        )
        gradients = (
            explicit.current_parameter_gradient,
            explicit.thermal_parameter_gradient + direct_thermal_gradient,
            explicit.feedback_parameter_gradient,
        )
        cotangent_norms = (
            jnp.linalg.norm(cell_cotangent),
            jnp.linalg.norm(state_cotangent.potential),
            jnp.linalg.norm(state_cotangent.temperature),
        )
        return objective, gradients, explicit, cell_temperature, cotangent_norms

    objective_arguments = (runtime_inputs, current, thermal, feedback, frozen_reference)
    compiled_forward, forward_timing, forward_hlo = _compile_and_time(
        jax,
        objective_function,
        objective_arguments,
    )
    native_function = jax.value_and_grad(objective_function, argnums=(1, 2, 3))
    compiled_native, native_timing, native_hlo = _compile_and_time(
        jax,
        native_function,
        objective_arguments,
    )
    compiled_explicit, explicit_timing, explicit_hlo = _compile_and_time(
        jax,
        explicit_objective_and_gradients,
        objective_arguments,
    )
    objective, native_gradients = compiled_native(*objective_arguments)
    (
        explicit_objective,
        explicit_gradients,
        explicit_result,
        cell_temperature,
        cotangent_norms,
    ) = compiled_explicit(*objective_arguments)
    jax.block_until_ready(
        (
            objective,
            native_gradients,
            explicit_objective,
            explicit_gradients,
            explicit_result,
            cell_temperature,
            cotangent_norms,
        )
    )

    finite_difference_gradients: list[float] = []
    finite_difference_errors: list[float] = []
    for step in FINITE_DIFFERENCE_STEPS:
        plus = current.at[0].add(step)
        minus = current.at[0].add(-step)
        plus_value = compiled_forward(runtime_inputs, plus, thermal, feedback, frozen_reference)
        minus_value = compiled_forward(runtime_inputs, minus, thermal, feedback, frozen_reference)
        finite_difference = (plus_value - minus_value) / (2.0 * step)
        jax.block_until_ready(finite_difference)
        finite_difference_gradients.append(float(np.asarray(jax.device_get(finite_difference))))
        finite_difference_errors.append(
            _relative_difference(jax, native_gradients[0][0], finite_difference)
        )

    thermo_optic = transfer_runtime.state(transfer_inputs, cell_temperature)
    updated_parameters = with_fdtdx_device_parameter(
        scene.parameters,
        thermo_optic,
        loaded.contract,
    )
    updated_arrays, _updated_objects, _application = fdtdx.apply_params(
        arrays=scene.arrays,
        objects=scene.objects,
        params=updated_parameters,
        key=scene.key,
        material_array_shardings=scene.material_array_shardings,
    )
    jax.block_until_ready((thermo_optic, updated_arrays.inv_permittivities))
    material_sharding_preserved = updated_arrays.inv_permittivities.sharding.is_equivalent_to(
        scene.material_array_shardings.inv_permittivities,
        updated_arrays.inv_permittivities.ndim,
    )
    material_difference = _material_relative_difference(
        jax,
        updated_arrays.inv_permittivities,
        canonical_thermo_optic.relative_permittivity,
        device_grid_slice=DEVICE_GRID_SLICE,
    )

    potential_difference = _relative_difference(
        jax,
        explicit_result.forward.state.potential,
        expected_potential,
    )
    temperature_difference = _relative_difference(
        jax,
        explicit_result.forward.state.temperature,
        expected_temperature,
    )
    cell_temperature_difference = _relative_difference(
        jax,
        cell_temperature,
        expected_cell_temperature,
    )
    parameter_difference = _relative_difference(jax, thermo_optic.parameter, expected_parameter)
    objective_difference = _relative_difference(jax, objective, explicit_objective)
    gradient_differences = [
        _relative_difference(jax, observed, expected)
        for observed, expected in zip(native_gradients, explicit_gradients, strict=True)
    ]
    gradient_norms = [
        float(np.asarray(jax.device_get(jnp.linalg.norm(value)))) for value in native_gradients
    ]
    finite = bool(
        np.asarray(
            jax.device_get(
                jnp.all(
                    jnp.stack(
                        (
                            jnp.isfinite(objective),
                            jnp.isfinite(explicit_objective),
                            jnp.all(jnp.isfinite(explicit_result.forward.state.potential)),
                            jnp.all(jnp.isfinite(explicit_result.forward.state.temperature)),
                            jnp.all(jnp.isfinite(cell_temperature)),
                            jnp.all(jnp.isfinite(cast(Any, thermo_optic.parameter))),
                            *(jnp.all(jnp.isfinite(value)) for value in native_gradients),
                            *(jnp.all(jnp.isfinite(value)) for value in explicit_gradients),
                        )
                    )
                )
            )
        )
    )

    first_report = cast(dict[str, object], partitioned_reports["input-cell-local-dofs"])
    local_mask = np.zeros((global_device_count,), dtype=np.int32)
    for shard in cast(list[dict[str, object]], first_report["addressable_shards"]):
        local_mask[cast(int, shard["partition_index"])] = 1
    gathered_masks = np.asarray(multihost_utils.process_allgather(local_mask, tiled=False)).reshape(
        process_count, global_device_count
    )
    addressability_counts = np.sum(gathered_masks, axis=0)
    exact_addressability = bool(
        np.array_equal(addressability_counts, np.ones(global_device_count, dtype=np.int64))
    )

    critical_reports = {
        "applied-inverse-permittivity": _critical_array_report(
            "applied-inverse-permittivity",
            updated_arrays.inv_permittivities,
            mesh,
            process_index=process_index,
            process_count=process_count,
        ),
        "thermo-optic-parameter": _critical_array_report(
            "thermo-optic-parameter",
            thermo_optic.parameter,
            mesh,
            process_index=process_index,
            process_count=process_count,
        ),
    }
    hlo_by_name = {
        "reference_phasor": reference_hlo,
        "forward": forward_hlo,
        "explicit_vjp": explicit_hlo,
        "native_reverse": native_hlo,
    }
    if tuple(hlo_by_name) != EXECUTABLE_NAMES:
        raise RuntimeError("compiled executable names differ from the evidence contract")
    hbm_capacity = _positive_environment_integer("FEMX_HBM_BYTES_PER_DEVICE")
    if hbm_capacity is None:
        raise RuntimeError("FEMX_HBM_BYTES_PER_DEVICE is required for physical TPU evidence")
    compiled_by_name = {
        "reference_phasor": compiled_reference,
        "forward": compiled_forward,
        "explicit_vjp": compiled_explicit,
        "native_reverse": compiled_native,
    }
    timing_by_name = {
        "reference_phasor": reference_timing,
        "forward": forward_timing,
        "explicit_vjp": explicit_timing,
        "native_reverse": native_timing,
    }
    executables = {
        name: {
            "timing": timing_by_name[name],
            "memory": _memory_report(compiled_by_name[name], hbm_capacity),
            "stablehlo": _stablehlo_report(hlo_by_name[name]),
        }
        for name in EXECUTABLE_NAMES
    }

    forward = explicit_result.forward
    raw_numerics = {
        "finite": finite,
        "forward_converged": bool(np.asarray(jax.device_get(forward.converged))),
        "adjoint_converged": bool(np.asarray(jax.device_get(explicit_result.adjoint_converged))),
        "thermo_optic_all_valid": bool(np.asarray(jax.device_get(thermo_optic.all_valid))),
        "material_destination_sharding_preserved": bool(material_sharding_preserved),
        "reference_phasor_real": reference_value.real,
        "reference_phasor_imag": reference_value.imag,
        "reference_phasor_magnitude": abs(reference_value),
        "objective": float(np.asarray(jax.device_get(objective))),
        "potential_relative_difference": potential_difference,
        "temperature_relative_difference": temperature_difference,
        "cell_temperature_relative_difference": cell_temperature_difference,
        "parameter_relative_difference": parameter_difference,
        "material_relative_difference": material_difference,
        "objective_explicit_relative_difference": objective_difference,
        "native_explicit_gradient_relative_differences": gradient_differences,
        "native_gradient_norms": gradient_norms,
        "applied_voltage_finite_difference": {
            "steps": list(FINITE_DIFFERENCE_STEPS),
            "gradients": finite_difference_gradients,
            "relative_errors": finite_difference_errors,
        },
        "iterations": int(np.asarray(jax.device_get(forward.iterations))),
        "current_residual_error": float(np.asarray(jax.device_get(forward.current_residual_error))),
        "heat_residual_error": float(np.asarray(jax.device_get(forward.heat_residual_error))),
        "current_linear_backward_error": float(
            np.asarray(jax.device_get(forward.current_linear_backward_error))
        ),
        "heat_linear_backward_error": float(
            np.asarray(jax.device_get(forward.heat_linear_backward_error))
        ),
        "transfer_relative_error": float(
            np.asarray(jax.device_get(forward.transfer_relative_error))
        ),
        "adjoint_backward_error": float(
            np.asarray(jax.device_get(explicit_result.adjoint_backward_error))
        ),
        "electrical_joule_power_W_per_m": float(
            np.asarray(jax.device_get(forward.electrical_joule_power))
        ),
        "thermal_joule_load_W_per_m": float(np.asarray(jax.device_get(forward.thermal_joule_load))),
        "cell_cotangent_norm": float(np.asarray(jax.device_get(cotangent_norms[0]))),
        "potential_cotangent_norm": float(np.asarray(jax.device_get(cotangent_norms[1]))),
        "temperature_cotangent_norm": float(np.asarray(jax.device_get(cotangent_norms[2]))),
    }
    admitted_numerics = (
        finite
        and cast(bool, raw_numerics["forward_converged"])
        and cast(bool, raw_numerics["adjoint_converged"])
        and cast(bool, raw_numerics["thermo_optic_all_valid"])
        and material_sharding_preserved
        and potential_difference <= TOLERANCES["potential_relative_difference"]
        and temperature_difference <= TOLERANCES["temperature_relative_difference"]
        and cell_temperature_difference <= TOLERANCES["cell_temperature_relative_difference"]
        and parameter_difference <= TOLERANCES["parameter_relative_difference"]
        and material_difference <= TOLERANCES["material_relative_difference"]
        and objective_difference <= TOLERANCES["objective_explicit_relative_difference"]
        and max(gradient_differences) <= TOLERANCES["native_explicit_gradient_relative_difference"]
        and max(finite_difference_errors) <= TOLERANCES["finite_difference_relative_error"]
        and cast(float, raw_numerics["current_residual_error"])
        <= TOLERANCES["current_residual_error"]
        and cast(float, raw_numerics["heat_residual_error"]) <= TOLERANCES["heat_residual_error"]
        and cast(float, raw_numerics["current_linear_backward_error"])
        <= TOLERANCES["linear_backward_error"]
        and cast(float, raw_numerics["heat_linear_backward_error"])
        <= TOLERANCES["linear_backward_error"]
        and cast(float, raw_numerics["transfer_relative_error"])
        <= TOLERANCES["transfer_relative_error"]
        and cast(float, raw_numerics["adjoint_backward_error"])
        <= TOLERANCES["adjoint_backward_error"]
        and cast(float, raw_numerics["cell_cotangent_norm"]) > 0.0
        and cast(float, raw_numerics["potential_cotangent_norm"]) == 0.0
        and cast(float, raw_numerics["temperature_cotangent_norm"]) > 0.0
        and all(value > 0.0 and math.isfinite(value) for value in gradient_norms)
        and all(
            value > 0.0 and math.isfinite(value) for value in map(abs, finite_difference_gradients)
        )
    )
    collectives_admitted = all(
        not cast(bool, record["stablehlo"]["contains_all_gather"])
        and not cast(bool, record["stablehlo"]["contains_float64"])
        and cast(int, record["stablehlo"]["all_to_all_count"]) > 0
        and cast(int, record["stablehlo"]["collective_permute_count"]) > 0
        and cast(int, record["stablehlo"]["all_reduce_count"]) > 0
        and cast(float, record["memory"]["hbm_fraction"])
        < TOLERANCES["maximum_compiler_hbm_fraction"]
        for name, record in executables.items()
    )
    passed = admitted_numerics and exact_addressability and collectives_admitted
    numerics = json.loads(json.dumps(_json_safe(raw_numerics), allow_nan=False))

    scene_record = {
        "grid_shape_xyz": loaded.scene["grid_shape_xyz"],
        "device_shape_xyz": list(loaded.contract.target_shape),
        "time_steps": loaded.scene["time_steps"],
        "sha256": _canonical_digest(dict(loaded.scene)),
    }
    process_payload = {
        "schema_version": PROCESS_EVIDENCE_SCHEMA,
        "status": "passed" if passed else "failed",
        "provenance": provenance,
        "runtime": {
            "backend": jax.default_backend(),
            "jax_version": jax.__version__,
            "jaxlib_version": package_version("jaxlib"),
            "fdtdx_version": package_version("fdtdx"),
            "x64_enabled": bool(getattr(jax.config, "jax_enable_x64", False)),
            "default_matmul_precision": str(
                getattr(jax.config, "jax_default_matmul_precision", None)
            ),
            "process_index": process_index,
            "process_count": process_count,
            "local_device_count": int(jax.local_device_count()),
            "global_device_count": global_device_count,
            "device_kinds": sorted({str(device.device_kind) for device in jax.devices()}),
            "scalar_contract": dict(SCALAR_CONTRACT),
        },
        "launch_claim": launch_claim,
        "input": {
            "manifest_sha256": _sha256_file(input_root / MANIFEST_FILENAME),
            "arrays_sha256": loaded.arrays_sha256,
            "electrothermal_arrays_sha256": loaded.electrothermal.arrays_sha256,
            "source_commit": loaded.manifest["source_commit"],
            "sampling_operator_sha256": loaded.sampling.operator_sha256,
            "transfer_operator_sha256": loaded.transfer.operator_sha256,
            "scene_sha256": scene_record["sha256"],
            "fdtdx_package_version": package_version("fdtdx"),
            "fdtdx_source_revision": FDTDX_SOURCE_REVISION,
            "fdtdx_source_digest": FDTDX_SOURCE_DIGEST,
            "fdtdx_module_sha256": module_hashes,
        },
        "plan": {
            "sha256": plan.digest(),
            "layout_sha256": plan.layout.digest(),
            "partition_count": plan.layout.partition_count,
            "node_count": plan.layout.topology.node_count,
            "triangle_count": plan.layout.topology.cell_count,
            "free_dof_count": plan.layout.topology.free_dof_count,
        },
        "mesh_report": mesh_report.canonical_data(),
        "addressability": {
            "process_local_partition_mask": local_mask.tolist(),
            "partition_addressability_counts": addressability_counts.tolist(),
            "every_partition_addressable_once": exact_addressability,
        },
        "partitioned_array_reports": partitioned_reports,
        "replicated_array_reports": replicated_reports,
        "critical_array_reports": critical_reports,
        "coordinate_admission": coordinates,
        "scene": scene_record,
        "numerics": numerics,
        "tolerances": dict(TOLERANCES),
        "executables": executables,
        "claim_scope": (
            "one process-local record from a bounded physical multi-host TPU execution of the "
            "2D distributed electrothermal residual adjoint through all-to-all thermo-optic "
            "transfer and checkpointed FDTDX objective; complete claims require all eight "
            "process records and exclude 3D FEM, ring convergence, S-parameters, scaling, live "
            "HBM, measured-device, foundry, and preemption-recovery evidence"
        ),
    }
    output_root = Path(os.environ["PHOXLA_OUTPUT_DIR"])
    for name, stablehlo in hlo_by_name.items():
        _atomic_text(output_root / "hlo" / f"{name}.stablehlo.mlir", stablehlo)
    _atomic_json(output_root / "results" / "process-metrics.json", process_payload)
    if process_index == 0:
        _atomic_json(output_root / "results" / "metrics.json", process_payload)
    _publish_process_zero_compatibility(
        remote_run,
        process_index=process_index,
        process_payload=process_payload,
        stablehlo_by_name=hlo_by_name,
    )
    multihost_utils.sync_global_devices(f"femx-fdtdx-thermo-optic-written-{provenance['run_id']}")
    print(json.dumps({"status": process_payload["status"], "run_id": provenance["run_id"]}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
