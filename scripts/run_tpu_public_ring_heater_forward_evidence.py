#!/usr/bin/env python3
"""Run the source-pinned fine public-ring forward solve on physical TPU v4-64."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Mapping
from functools import partial
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, NamedTuple, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SOURCE_ROOT):
    import_path = str(import_root)
    while import_path in sys.path:
        sys.path.remove(import_path)
    sys.path.insert(0, import_path)

from femx.validation.tpu_public_ring_heater_evidence import (
    CONSERVATION_TOLERANCES,
    EXPECTED_CONDUCTOR_TETRAHEDRON_COUNT,
    EXPECTED_FINE_MSH_SHA256,
    EXPECTED_GLOBAL_DEVICE_COUNT,
    EXPECTED_LOCAL_DEVICE_COUNT,
    EXPECTED_NODE_COUNT,
    EXPECTED_PROCESS_COUNT,
    EXPECTED_TETRAHEDRON_COUNT,
    PARITY_TOLERANCES,
    PROCESS_EVIDENCE_SCHEMA,
    REAL_SCALAR_CONTRACT,
    SCALAR_CG_POLICY,
    WORKER_ENTRY_CLAIM_SCHEMA,
)

NUMERICAL_DIAGNOSTIC_SCHEMA = "femx.public-ring-heater.tpu_forward_diagnostic/v1"
ONE_SHOT_TIMING_SCHEMA = "femx.public-ring-heater.tpu_forward_timing/v1"
EVIDENCE_SCHEMA = PROCESS_EVIDENCE_SCHEMA
ENTRY_CLAIM_SCHEMA = WORKER_ENTRY_CLAIM_SCHEMA
RUNTIME_SCALAR_CONTRACT = dict(REAL_SCALAR_CONTRACT)
REPLICATION_INTENT = "three bounded scalar controls replicated identically on all TPU devices"
_SHA256_HEX = frozenset("0123456789abcdef")


class ForwardObservation(NamedTuple):
    """Minimal forward state and replicated scientific diagnostics."""

    potential: Any
    temperature_rise: Any
    current_iterations: Any
    thermal_iterations: Any
    current_recursive_residual: Any
    thermal_recursive_residual: Any
    current_recomputed_residual: Any
    thermal_recomputed_residual: Any
    current_relative_residual: Any
    thermal_relative_residual: Any
    current_backward_error: Any
    thermal_backward_error: Any
    current_converged: Any
    thermal_converged: Any
    current_breakdown: Any
    thermal_breakdown: Any
    electrical_joule_power: Any
    electrical_variational_power: Any
    electrical_energy_relative_error: Any
    charge_balance_relative_error: Any
    thermal_joule_load: Any
    joule_transfer_relative_error: Any
    thermal_input_power: Any
    convection_outward_power: Any
    dirichlet_outward_power: Any
    thermal_balance_relative_error: Any
    numerically_admitted: Any
    potential_relative_l2_difference: Any
    potential_normalized_max_difference: Any
    temperature_rise_relative_l2_difference: Any
    temperature_rise_normalized_max_difference: Any
    minimum_temperature: Any
    maximum_temperature: Any
    silicon_ring_mean_temperature: Any
    tin_heater_mean_temperature: Any
    silicon_ring_volume: Any
    tin_heater_volume: Any
    all_finite: Any


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


def _require_expected_count(name: str, observed: int, required: int) -> None:
    configured = _positive_environment_integer(name)
    if configured is None:
        raise RuntimeError(f"{name} must be set for physical evidence")
    if configured != required or observed != required:
        raise RuntimeError(f"{name} requires {required}, observed {observed}")


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
        raise RuntimeError("public-ring physical witness requires the explicit float32 path")
    if str(getattr(jax.config, "jax_default_matmul_precision", None)) != "highest":
        raise RuntimeError("JAX default matmul precision must resolve to 'highest'")
    _require_expected_count(
        "FEMX_EXPECTED_PROCESS_COUNT",
        int(jax.process_count()),
        EXPECTED_PROCESS_COUNT,
    )
    _require_expected_count(
        "FEMX_EXPECTED_LOCAL_DEVICE_COUNT",
        int(jax.local_device_count()),
        EXPECTED_LOCAL_DEVICE_COUNT,
    )
    _require_expected_count(
        "FEMX_EXPECTED_GLOBAL_DEVICE_COUNT",
        int(jax.device_count()),
        EXPECTED_GLOBAL_DEVICE_COUNT,
    )
    if jax.process_count() * jax.local_device_count() != jax.device_count():
        raise RuntimeError("uniform one-process-per-worker device accounting is required")
    return jax


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise RuntimeError(f"{label} must be an object with string keys")
    return value


def _integer(value: object, *, label: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if positive else "nonnegative"
        raise RuntimeError(f"{label} must be a {qualifier} integer")
    return value


def _number(value: object, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted) or (positive and converted <= 0.0):
        qualifier = "finite positive" if positive else "finite"
        raise RuntimeError(f"{label} must be a {qualifier} number")
    return converted


def _sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise RuntimeError(f"{label} must be a lowercase SHA-256")
    return value


def _atomic_json(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite immutable evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, allow_nan=False, indent=2, sort_keys=True))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite immutable evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _manifest_provenance(remote_run: Path) -> dict[str, object]:
    manifest_path = remote_run / ".phoxla" / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = _mapping(manifest["source"], label="Phoxla source")
        config = _mapping(manifest["config"], label="Phoxla config")
        return {
            "run_id": manifest["run_id"],
            "profile": manifest["profile"],
            "source_digest": _sha256(source["digest"], label="Phoxla source digest"),
            "source_commit": source["commit"],
            "config_digest": _sha256(config["digest"], label="Phoxla config digest"),
        }
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid deployed Phoxla manifest: {manifest_path}") from error


def _claim_worker_entry(remote_run: Path, provenance: Mapping[str, object]) -> dict[str, object]:
    if not remote_run.is_absolute() or not remote_run.is_dir() or remote_run.is_symlink():
        raise RuntimeError("PHOXLA_REMOTE_RUN_DIR must be an absolute non-symlink directory")
    process_index = _nonnegative_environment_integer("PHOXLA_PROCESS_INDEX")
    worker_index = _nonnegative_environment_integer("PHOXLA_GCLOUD_WORKER_INDEX")
    if process_index is None or worker_index is None:
        raise RuntimeError("Phoxla process and worker indexes are required before entry execution")
    run_id = os.environ.get("PHOXLA_RUN_ID")
    if run_id != provenance.get("run_id"):
        raise RuntimeError("PHOXLA_RUN_ID disagrees with the deployed manifest")
    claim_path = remote_run / "logs" / "femx-ring-forward-entry.claim"
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        claim_path.mkdir()
    except FileExistsError as error:
        raise RuntimeError(
            "duplicate public-ring TPU entry refused for this immutable run"
        ) from error
    claim = {
        "schema_version": ENTRY_CLAIM_SCHEMA,
        "run_id": run_id,
        "worker_index": worker_index,
        "process_index": process_index,
        "source_sha256": provenance.get("source_digest"),
        "config_sha256": provenance.get("config_digest"),
        "scope": "worker-local immutable entry fence after distributed bootstrap",
    }
    _atomic_json(claim_path / "identity.json", claim)
    return claim


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


def _stablehlo_report(text: str) -> dict[str, object]:
    lowered = text.lower()
    return {
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "collective_permute_count": lowered.count("stablehlo.collective_permute"),
        "all_reduce_count": lowered.count("stablehlo.all_reduce"),
        "contains_all_gather": "all_gather" in lowered,
        "contains_f64": "f64" in lowered,
    }


def _compile_and_execute(
    jax: Any,
    function: Any,
    arguments: tuple[Any, ...],
) -> tuple[Any, Any, dict[str, object], str]:
    started = time.perf_counter()
    lowered = jax.jit(function).lower(*arguments)
    lowering_seconds = time.perf_counter() - started
    stablehlo = str(lowered.compiler_ir("stablehlo"))
    started = time.perf_counter()
    compiled = lowered.compile()
    compilation_seconds = time.perf_counter() - started
    started = time.perf_counter()
    result = compiled(*arguments)
    jax.block_until_ready(result)
    execution_seconds = time.perf_counter() - started
    timing = {
        "schema_version": ONE_SHOT_TIMING_SCHEMA,
        "lowering_seconds": lowering_seconds,
        "compilation_seconds": compilation_seconds,
        "execution_seconds": execution_seconds,
        "execution_count": 1,
        "synchronization": "the sole target-voltage result blocked until ready",
        "benchmark_claimed": False,
    }
    return compiled, result, timing, stablehlo


def _host_scalar(jax: Any, value: object) -> float:
    import numpy as np

    return float(np.asarray(jax.device_get(value)))


def _host_integer(jax: Any, value: object) -> int:
    import numpy as np

    return int(np.asarray(jax.device_get(value)))


def _host_boolean(jax: Any, value: object) -> bool:
    import numpy as np

    return bool(np.asarray(jax.device_get(value)))


def _relative_error(observed: float, expected: float) -> float:
    return abs(observed - expected) / max(abs(expected), float.fromhex("0x1p-126"))


def _json_nonfinite_paths(value: object, *, path: str = "$") -> list[str]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, float):
        return [] if math.isfinite(value) else [path]
    if isinstance(value, Mapping):
        result: list[str] = []
        for key, item in value.items():
            result.extend(_json_nonfinite_paths(item, path=f"{path}.{key}"))
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for index, item in enumerate(value):
            result.extend(_json_nonfinite_paths(item, path=f"{path}[{index}]"))
        return result
    return []


def _shard_hashes(jax: Any, array: Any) -> list[dict[str, object]]:
    import numpy as np

    records = []
    for shard in array.addressable_shards:
        index = tuple(shard.index)
        leading = cast(slice, index[0])
        partition = leading.indices(int(array.shape[0]))[0]
        values = np.ascontiguousarray(jax.device_get(shard.data), dtype="<f4")
        records.append(
            {
                "partition_index": partition,
                "shape": list(values.shape),
                "dtype": values.dtype.name,
                "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                "finite": bool(np.all(np.isfinite(values))),
            }
        )
    return sorted(records, key=lambda item: cast(int, item["partition_index"]))


def _diagnostic_payload(
    provenance: Mapping[str, object],
    process_index: int,
    values: Mapping[str, object],
    nonfinite_paths: list[str],
) -> dict[str, object]:
    classifications = {}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        converted = float(value)
        classifications[name] = {
            "finite": math.isfinite(converted),
            "classification": (
                "finite"
                if math.isfinite(converted)
                else "nan"
                if math.isnan(converted)
                else "positive_infinity"
                if converted > 0.0
                else "negative_infinity"
            ),
            "value": converted if math.isfinite(converted) else None,
        }
    return {
        "schema_version": NUMERICAL_DIAGNOSTIC_SCHEMA,
        "status": "failed",
        "provenance": dict(provenance),
        "process_index": process_index,
        "nonfinite_paths": nonfinite_paths,
        "scalar_classifications": classifications,
        "failure": "non-finite physical TPU public-ring forward result",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    remote_run = Path(os.environ["PHOXLA_REMOTE_RUN_DIR"])
    provenance = _manifest_provenance(remote_run)
    launch_claim = _claim_worker_entry(remote_run, provenance)
    jax = _runtime()

    import jax.numpy as jnp
    import numpy as np
    from jax import lax
    from jax.experimental import multihost_utils
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from femx.backends.jax.collective_runtime import (
        describe_collective_mesh,
        make_collective_array_from_process_local_data,
        make_replicated_array_from_process_local_data,
    )
    from femx.backends.jax.scalar_cg import (
        ScalarH1CGPolicy,
        ScalarH1JacobiPolicy,
        build_packed_scalar_h1_jacobi_preconditioner_factory,
    )
    from femx.backends.jax.tet4_electrothermal import (
        HostPackedTet4ElectrothermalInputs,
        PackedTet4ElectrothermalInputs,
        Tet4ElectrothermalAdmissionPolicy,
        Tet4ElectrothermalParameters,
        build_tet4_electrothermal_runtime,
    )
    from scripts._tpu_public_ring_heater_plan import (
        read_public_ring_heater_tpu_artifact,
    )

    process_index = int(jax.process_index())
    process_count = int(jax.process_count())
    global_device_count = int(jax.device_count())
    local_device_count = int(jax.local_device_count())
    if _integer(launch_claim.get("process_index"), label="entry process index") != process_index:
        raise RuntimeError("worker entry claim disagrees with initialized JAX process identity")

    input_root = (remote_run / arguments.input).resolve(strict=True)
    if remote_run.resolve() not in input_root.parents:
        raise RuntimeError("public-ring TPU input must remain inside the deployed run")
    loaded = read_public_ring_heater_tpu_artifact(input_root)
    manifest = loaded.manifest
    if manifest.get("source_commit") != provenance.get("source_commit"):
        raise RuntimeError("controller artifact source commit disagrees with deployed source")
    if manifest.get("source_msh_sha256") != EXPECTED_FINE_MSH_SHA256:
        raise RuntimeError("controller artifact does not bind the admitted fine MSH")
    model = _mapping(manifest.get("model"), label="artifact model")
    if (
        _integer(model.get("node_count"), label="model node count", positive=True)
        != EXPECTED_NODE_COUNT
        or _integer(
            model.get("tetrahedron_count"),
            label="model tetrahedron count",
            positive=True,
        )
        != EXPECTED_TETRAHEDRON_COUNT
        or _integer(
            model.get("conductor_tetrahedron_count"),
            label="model conductor tetrahedron count",
            positive=True,
        )
        != EXPECTED_CONDUCTOR_TETRAHEDRON_COUNT
    ):
        raise RuntimeError("controller artifact model counts changed")

    plan = loaded.runtime_plan
    if plan.thermal_layout.partition_count != EXPECTED_GLOBAL_DEVICE_COUNT:
        raise RuntimeError("runtime plan must bind exactly one partition per TPU device")
    mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("partition",))
    mesh_report = describe_collective_mesh(
        plan.thermal_layout.transport,
        mesh,
        layout_sha256=plan.thermal_layout.digest(),
    )
    partitioned_reports: dict[str, object] = {}

    def load_partitioned(name: str, value: np.ndarray) -> Any:
        array, report = make_collective_array_from_process_local_data(name, value, mesh)
        partitioned_reports[name] = report.canonical_data()
        return array

    packed_values = []
    for name in HostPackedTet4ElectrothermalInputs._fields:
        packed_values.append(load_partitioned(f"input-{name}", loaded.arrays[name]))
    inputs = PackedTet4ElectrothermalInputs(*packed_values)
    authority_potential = load_partitioned(
        "authority-potential",
        loaded.arrays["authority_potential"],
    )
    authority_temperature_rise = load_partitioned(
        "authority-temperature-rise",
        loaded.arrays["authority_temperature_rise"],
    )
    silicon_ring_mask = load_partitioned(
        "silicon-ring-cell-mask",
        loaded.arrays["silicon_ring_cell_mask"],
    )
    tin_heater_mask = load_partitioned(
        "tin-heater-cell-mask",
        loaded.arrays["tin_heater_cell_mask"],
    )

    authority = _mapping(manifest.get("authority"), label="artifact authority")
    authority_record = _mapping(authority.get("record"), label="artifact authority record")
    authority_numerics = _mapping(
        authority_record.get("numerics"),
        label="artifact authority numerics",
    )
    authority_regions = _mapping(
        authority_numerics.get("region_temperature"),
        label="artifact authority region temperatures",
    )
    target_voltage = _number(
        authority.get("target_voltage_V"),
        label="target voltage",
        positive=True,
    )
    parameter_host = np.asarray((target_voltage, 1.0, 1.0), dtype=np.float32)
    replicated_parameters, replicated_report = make_replicated_array_from_process_local_data(
        "electrothermal-controls",
        parameter_host,
        mesh,
        replication_intent=REPLICATION_INTENT,
    )
    parameters = Tet4ElectrothermalParameters(
        *tuple(replicated_parameters[index] for index in range(3))
    )

    jacobi_policy = ScalarH1JacobiPolicy(
        cast(
            float,
            cast(Mapping[str, object], SCALAR_CG_POLICY["preconditioner"])[
                "minimum_relative_diagonal"
            ],
        )
    )
    current_preconditioner = build_packed_scalar_h1_jacobi_preconditioner_factory(
        plan.current_layout,
        mesh,
        jacobi_policy,
    )
    thermal_preconditioner = build_packed_scalar_h1_jacobi_preconditioner_factory(
        plan.thermal_layout,
        mesh,
        jacobi_policy,
    )
    cg_policy = ScalarH1CGPolicy(
        relative_tolerance=cast(float, SCALAR_CG_POLICY["relative_tolerance"]),
        absolute_tolerance=cast(float, SCALAR_CG_POLICY["absolute_tolerance"]),
        max_iterations=cast(int, SCALAR_CG_POLICY["max_iterations"]),
        backward_error_tolerance=cast(float, SCALAR_CG_POLICY["backward_error_tolerance"]),
    )
    runtime = build_tet4_electrothermal_runtime(
        plan,
        mesh,
        cg_policy,
        cg_policy,
        Tet4ElectrothermalAdmissionPolicy(
            CONSERVATION_TOLERANCES["charge_balance_relative_error"],
            CONSERVATION_TOLERANCES["electrical_energy_relative_error"],
            CONSERVATION_TOLERANCES["joule_transfer_relative_error"],
            CONSERVATION_TOLERANCES["thermal_balance_relative_error"],
        ),
        current_preconditioner_factory=current_preconditioner,
        thermal_preconditioner_factory=thermal_preconditioner,
    )

    current_owner_spec = P("partition", None)  # type: ignore[no-untyped-call]
    thermal_owner_spec = P("partition", None)  # type: ignore[no-untyped-call]
    thermal_cell_spec = P("partition", None)  # type: ignore[no-untyped-call]
    thermal_nodal_cell_spec = P("partition", None, None)  # type: ignore[no-untyped-call]

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(current_owner_spec, current_owner_spec, current_owner_spec),
        out_specs=(P(), P(), P()),  # type: ignore[no-untyped-call]
        check_vma=True,
    )
    def current_vector_metrics(observed: Any, expected: Any, active: Any) -> tuple[Any, ...]:
        difference = jnp.where(active[0], observed[0] - expected[0], 0.0)
        reference = jnp.where(active[0], expected[0], 0.0)
        numerator = lax.psum(  # type: ignore[no-untyped-call]
            jnp.sum(difference * difference), "partition"
        )
        denominator = lax.psum(  # type: ignore[no-untyped-call]
            jnp.sum(reference * reference), "partition"
        )
        maximum = lax.pmax(  # type: ignore[no-untyped-call]
            jnp.max(jnp.abs(difference)), "partition"
        )
        scale = lax.pmax(  # type: ignore[no-untyped-call]
            jnp.max(jnp.abs(reference)), "partition"
        )
        relative_l2 = jnp.where(denominator > 0.0, jnp.sqrt(numerator / denominator), jnp.inf)
        normalized_max = jnp.where(scale > 0.0, maximum / scale, jnp.inf)
        finite = lax.pmin(  # type: ignore[no-untyped-call]
            jnp.all(jnp.isfinite(observed[0])).astype(jnp.int32), "partition"
        )
        return relative_l2, normalized_max, finite

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(thermal_owner_spec, thermal_owner_spec, thermal_owner_spec),
        out_specs=(P(), P(), P(), P(), P()),  # type: ignore[no-untyped-call]
        check_vma=True,
    )
    def thermal_vector_metrics(observed: Any, expected: Any, active: Any) -> tuple[Any, ...]:
        difference = jnp.where(active[0], observed[0] - expected[0], 0.0)
        reference = jnp.where(active[0], expected[0], 0.0)
        numerator = lax.psum(  # type: ignore[no-untyped-call]
            jnp.sum(difference * difference), "partition"
        )
        denominator = lax.psum(  # type: ignore[no-untyped-call]
            jnp.sum(reference * reference), "partition"
        )
        maximum = lax.pmax(  # type: ignore[no-untyped-call]
            jnp.max(jnp.abs(difference)), "partition"
        )
        scale = lax.pmax(  # type: ignore[no-untyped-call]
            jnp.max(jnp.abs(reference)), "partition"
        )
        minimum = lax.pmin(  # type: ignore[no-untyped-call]
            jnp.min(jnp.where(active[0], observed[0], jnp.inf)), "partition"
        )
        maximum_value = lax.pmax(  # type: ignore[no-untyped-call]
            jnp.max(jnp.where(active[0], observed[0], -jnp.inf)), "partition"
        )
        relative_l2 = jnp.where(denominator > 0.0, jnp.sqrt(numerator / denominator), jnp.inf)
        normalized_max = jnp.where(scale > 0.0, maximum / scale, jnp.inf)
        return (
            relative_l2,
            normalized_max,
            minimum,
            maximum_value,
            jnp.isfinite(minimum + maximum_value),
        )

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(thermal_nodal_cell_spec, thermal_cell_spec, thermal_cell_spec, thermal_cell_spec),
        out_specs=(P(), P()),  # type: ignore[no-untyped-call]
        check_vma=True,
    )
    def region_mean(
        cell_temperature: Any,
        volumes: Any,
        region: Any,
        active: Any,
    ) -> tuple[Any, ...]:
        selected = region[0] & active[0]
        weights = jnp.where(selected, volumes[0], 0.0)
        weighted = jnp.sum(jnp.mean(cell_temperature[0], axis=1) * weights)
        total_weight = jnp.sum(weights)
        global_weighted = lax.psum(weighted, "partition")  # type: ignore[no-untyped-call]
        global_weight = lax.psum(total_weight, "partition")  # type: ignore[no-untyped-call]
        return jnp.where(
            global_weight > 0.0, global_weighted / global_weight, jnp.nan
        ), global_weight

    def solve_and_observe(
        packed_inputs: Any,
        controls: Any,
        expected_potential: Any,
        expected_temperature_rise: Any,
        ring_mask: Any,
        heater_mask: Any,
    ) -> ForwardObservation:
        result = runtime.solve(packed_inputs, controls)
        potential_metrics = current_vector_metrics(
            result.state.potential,
            expected_potential,
            packed_inputs.current_owner_mask,
        )
        temperature_metrics = thermal_vector_metrics(
            result.state.temperature_rise,
            expected_temperature_rise,
            packed_inputs.thermal_owner_mask,
        )
        cell_temperature = runtime.thermal_cell_temperature(packed_inputs, result.state)
        ring_mean, ring_volume = region_mean(
            cell_temperature,
            packed_inputs.thermal_cell_volumes,
            ring_mask,
            packed_inputs.thermal_cell_mask,
        )
        heater_mean, heater_volume = region_mean(
            cell_temperature,
            packed_inputs.thermal_cell_volumes,
            heater_mask,
            packed_inputs.thermal_cell_mask,
        )
        reference = jnp.asarray(plan.thermal_reference, dtype=result.state.temperature_rise.dtype)
        all_finite = (
            potential_metrics[2].astype(jnp.bool_)
            & temperature_metrics[4]
            & jnp.all(
                jnp.isfinite(
                    jnp.stack(
                        (
                            result.electrical_joule_power,
                            result.electrical_variational_power,
                            result.electrical_energy_relative_error,
                            result.charge_balance_relative_error,
                            result.thermal_joule_load,
                            result.joule_transfer_relative_error,
                            result.thermal_input_power,
                            result.convection_outward_power,
                            result.dirichlet_outward_power,
                            result.thermal_balance_relative_error,
                            potential_metrics[0],
                            potential_metrics[1],
                            temperature_metrics[0],
                            temperature_metrics[1],
                            ring_mean,
                            heater_mean,
                            ring_volume,
                            heater_volume,
                        )
                    )
                )
            )
        )
        return ForwardObservation(
            result.state.potential,
            result.state.temperature_rise,
            result.current_linear.iterations,
            result.thermal_linear.iterations,
            result.current_linear.recursive_residual_norm,
            result.thermal_linear.recursive_residual_norm,
            result.current_linear.recomputed_residual_norm,
            result.thermal_linear.recomputed_residual_norm,
            result.current_linear.relative_residual,
            result.thermal_linear.relative_residual,
            result.current_linear.backward_error,
            result.thermal_linear.backward_error,
            result.current_linear.converged,
            result.thermal_linear.converged,
            result.current_linear.breakdown,
            result.thermal_linear.breakdown,
            result.electrical_joule_power,
            result.electrical_variational_power,
            result.electrical_energy_relative_error,
            result.charge_balance_relative_error,
            result.thermal_joule_load,
            result.joule_transfer_relative_error,
            result.thermal_input_power,
            result.convection_outward_power,
            result.dirichlet_outward_power,
            result.thermal_balance_relative_error,
            result.numerically_admitted,
            potential_metrics[0],
            potential_metrics[1],
            temperature_metrics[0],
            temperature_metrics[1],
            reference + jnp.minimum(temperature_metrics[2], 0.0),
            reference + jnp.maximum(temperature_metrics[3], 0.0),
            ring_mean,
            heater_mean,
            ring_volume,
            heater_volume,
            all_finite,
        )

    arguments_tuple = (
        inputs,
        parameters,
        authority_potential,
        authority_temperature_rise,
        silicon_ring_mask,
        tin_heater_mask,
    )
    compiled, result, timing, stablehlo = _compile_and_execute(
        jax,
        solve_and_observe,
        arguments_tuple,
    )
    hbm_capacity = _positive_environment_integer("FEMX_HBM_BYTES_PER_DEVICE")
    if hbm_capacity is None:
        raise RuntimeError("FEMX_HBM_BYTES_PER_DEVICE is required for physical evidence")
    memory = _memory_report(compiled, hbm_capacity)
    hlo = _stablehlo_report(stablehlo)

    representative = cast(
        dict[str, object],
        partitioned_reports["input-thermal_cell_local_dofs"],
    )
    local_partition_mask = np.zeros((global_device_count,), dtype=np.int32)
    for shard in cast(list[dict[str, object]], representative["addressable_shards"]):
        local_partition_mask[cast(int, shard["partition_index"])] = 1
    gathered_masks = np.asarray(
        multihost_utils.process_allgather(local_partition_mask, tiled=False)
    ).reshape(process_count, global_device_count)
    addressability_counts = np.sum(gathered_masks, axis=0)
    exact_addressability = bool(
        np.array_equal(addressability_counts, np.ones(global_device_count, dtype=np.int64))
    )

    numerics = {
        "all_finite": _host_boolean(jax, result.all_finite),
        "numerically_admitted": _host_boolean(jax, result.numerically_admitted),
        "current_converged": _host_boolean(jax, result.current_converged),
        "thermal_converged": _host_boolean(jax, result.thermal_converged),
        "current_breakdown": _host_boolean(jax, result.current_breakdown),
        "thermal_breakdown": _host_boolean(jax, result.thermal_breakdown),
        "current_iterations": _host_integer(jax, result.current_iterations),
        "thermal_iterations": _host_integer(jax, result.thermal_iterations),
        "current_recursive_residual": _host_scalar(jax, result.current_recursive_residual),
        "thermal_recursive_residual": _host_scalar(jax, result.thermal_recursive_residual),
        "current_recomputed_residual": _host_scalar(jax, result.current_recomputed_residual),
        "thermal_recomputed_residual": _host_scalar(jax, result.thermal_recomputed_residual),
        "current_relative_residual": _host_scalar(jax, result.current_relative_residual),
        "thermal_relative_residual": _host_scalar(jax, result.thermal_relative_residual),
        "current_backward_error": _host_scalar(jax, result.current_backward_error),
        "thermal_backward_error": _host_scalar(jax, result.thermal_backward_error),
        "electrical_joule_power_W": _host_scalar(jax, result.electrical_joule_power),
        "electrical_variational_power_W": _host_scalar(
            jax,
            result.electrical_variational_power,
        ),
        "electrical_energy_relative_error": _host_scalar(
            jax,
            result.electrical_energy_relative_error,
        ),
        "charge_balance_relative_error": _host_scalar(
            jax,
            result.charge_balance_relative_error,
        ),
        "thermal_joule_load_W": _host_scalar(jax, result.thermal_joule_load),
        "joule_transfer_relative_error": _host_scalar(
            jax,
            result.joule_transfer_relative_error,
        ),
        "thermal_input_power_W": _host_scalar(jax, result.thermal_input_power),
        "convection_outward_power_W": _host_scalar(jax, result.convection_outward_power),
        "dirichlet_outward_power_W": _host_scalar(jax, result.dirichlet_outward_power),
        "thermal_balance_relative_error": _host_scalar(
            jax,
            result.thermal_balance_relative_error,
        ),
        "potential_relative_l2_difference": _host_scalar(
            jax,
            result.potential_relative_l2_difference,
        ),
        "potential_normalized_max_difference": _host_scalar(
            jax,
            result.potential_normalized_max_difference,
        ),
        "temperature_rise_relative_l2_difference": _host_scalar(
            jax,
            result.temperature_rise_relative_l2_difference,
        ),
        "temperature_rise_normalized_max_difference": _host_scalar(
            jax,
            result.temperature_rise_normalized_max_difference,
        ),
        "minimum_temperature_K": _host_scalar(jax, result.minimum_temperature),
        "maximum_temperature_K": _host_scalar(jax, result.maximum_temperature),
        "silicon_ring_mean_temperature_K": _host_scalar(
            jax,
            result.silicon_ring_mean_temperature,
        ),
        "tin_heater_mean_temperature_K": _host_scalar(
            jax,
            result.tin_heater_mean_temperature,
        ),
        "silicon_ring_volume_m3": _host_scalar(jax, result.silicon_ring_volume),
        "tin_heater_volume_m3": _host_scalar(jax, result.tin_heater_volume),
    }
    predicted_power = _number(
        authority.get("predicted_joule_power_W"),
        label="authority predicted power",
        positive=True,
    )
    target_current = _number(
        authority.get("target_current_A"),
        label="authority target current",
        positive=True,
    )
    inferred_current = numerics["electrical_joule_power_W"] / target_voltage
    numerics["inferred_current_A"] = inferred_current
    numerics["target_current_relative_error"] = _relative_error(inferred_current, target_current)
    numerics["target_power_relative_error"] = _relative_error(
        numerics["electrical_joule_power_W"],
        predicted_power,
    )
    authority_ring = _mapping(authority_regions.get("silicon_ring"), label="authority ring")
    authority_heater = _mapping(authority_regions.get("tin_heater"), label="authority heater")
    parity_observables = {
        "maximum_temperature_relative_difference": _relative_error(
            numerics["maximum_temperature_K"],
            _number(authority_numerics.get("maximum_temperature_K"), label="authority maximum"),
        ),
        "silicon_ring_mean_temperature_rise_relative_difference": _relative_error(
            numerics["silicon_ring_mean_temperature_K"] - plan.thermal_reference,
            _number(
                authority_ring.get("volume_weighted_cell_mean_K"),
                label="authority ring mean",
            )
            - plan.thermal_reference,
        ),
        "tin_heater_mean_temperature_rise_relative_difference": _relative_error(
            numerics["tin_heater_mean_temperature_K"] - plan.thermal_reference,
            _number(
                authority_heater.get("volume_weighted_cell_mean_K"),
                label="authority heater mean",
            )
            - plan.thermal_reference,
        ),
    }
    numerics.update(parity_observables)

    potential_shards = _shard_hashes(jax, result.potential)
    temperature_shards = _shard_hashes(jax, result.temperature_rise)
    outputs_finite = all(
        cast(bool, item["finite"]) for item in (*potential_shards, *temperature_shards)
    )
    hlo_admitted = (
        not cast(bool, hlo["contains_all_gather"])
        and not cast(bool, hlo["contains_f64"])
        and cast(int, hlo["collective_permute_count"]) > 0
        and cast(int, hlo["all_reduce_count"]) > 0
    )
    memory_admitted = (
        memory.get("risk") in {"safe", "elevated"}
        and _number(memory.get("hbm_fraction"), label="compiler HBM fraction") < 0.85
    )
    passed = (
        cast(bool, numerics["all_finite"])
        and outputs_finite
        and cast(bool, numerics["numerically_admitted"])
        and cast(bool, numerics["current_converged"])
        and cast(bool, numerics["thermal_converged"])
        and not cast(bool, numerics["current_breakdown"])
        and not cast(bool, numerics["thermal_breakdown"])
        and exact_addressability
        and hlo_admitted
        and memory_admitted
        and numerics["current_backward_error"]
        <= cast(float, SCALAR_CG_POLICY["backward_error_tolerance"])
        and numerics["thermal_backward_error"]
        <= cast(float, SCALAR_CG_POLICY["backward_error_tolerance"])
        and all(numerics[name] <= tolerance for name, tolerance in CONSERVATION_TOLERANCES.items())
        and all(numerics[name] <= tolerance for name, tolerance in PARITY_TOLERANCES.items())
    )

    process_payload = {
        "schema_version": EVIDENCE_SCHEMA,
        "status": "passed" if passed else "failed",
        "provenance": provenance,
        "runtime": {
            "backend": jax.default_backend(),
            "jax_version": jax.__version__,
            "jaxlib_version": package_version("jaxlib"),
            "x64_enabled": bool(getattr(jax.config, "jax_enable_x64", False)),
            "default_matmul_precision": str(
                getattr(jax.config, "jax_default_matmul_precision", None)
            ),
            "process_index": process_index,
            "process_count": process_count,
            "local_device_count": local_device_count,
            "global_device_count": global_device_count,
            "device_kinds": sorted({str(device.device_kind) for device in jax.devices()}),
            "real_scalar_contract": RUNTIME_SCALAR_CONTRACT,
        },
        "launch_claim": launch_claim,
        "artifact": {
            "schema_version": manifest.get("schema_version"),
            "logical_sha256": loaded.logical_sha256,
            "runtime_plan_sha256": plan.digest(),
            "source_plan_sha256": plan.source_plan_sha256,
            "source_msh_sha256": manifest.get("source_msh_sha256"),
            "partition_owner_sha256": _mapping(
                manifest.get("partition"),
                label="artifact partition",
            ).get("owner_sha256"),
            "total_array_file_bytes": manifest.get("total_array_file_bytes"),
            "host_storage": (
                "verified memory-mapped global topology and arrays; only process-addressable "
                "partition slices transferred to TPU HBM"
            ),
        },
        "model": {
            "node_count": EXPECTED_NODE_COUNT,
            "tetrahedron_count": EXPECTED_TETRAHEDRON_COUNT,
            "conductor_tetrahedron_count": EXPECTED_CONDUCTOR_TETRAHEDRON_COUNT,
            "dimension": 3,
            "element": "first-order Tet4 H1",
            "coupling": "current to cell-local Joule density to steady heat",
            "target_voltage_V": target_voltage,
            "target_current_A": target_current,
            "authority_predicted_power_W": predicted_power,
        },
        "mesh_report": mesh_report.canonical_data(),
        "addressability": {
            "process_local_partition_mask": local_partition_mask.tolist(),
            "partition_addressability_counts": addressability_counts.tolist(),
            "every_partition_addressable_once": exact_addressability,
            "check_scope": "metadata-only process allgather outside the compiled forward solve",
        },
        "partitioned_array_reports": partitioned_reports,
        "replicated_parameter_report": replicated_report.canonical_data(),
        "output_shards": {
            "potential": potential_shards,
            "temperature_rise": temperature_shards,
        },
        "policies": {
            "scalar_cg": {
                **dict(SCALAR_CG_POLICY),
                "preconditioner": dict(
                    cast(Mapping[str, object], SCALAR_CG_POLICY["preconditioner"])
                ),
            },
            "conservation_tolerances": dict(CONSERVATION_TOLERANCES),
            "parity_tolerances": dict(PARITY_TOLERANCES),
            "target_voltage_source": (
                "fixed by the CPU float64 unit-voltage linear calibration; no repeated TPU "
                "calibration solve"
            ),
        },
        "numerics": numerics,
        "executable": {
            "timing": timing,
            "compiler_memory": memory,
            "stablehlo": hlo,
            "hlo_admitted": hlo_admitted,
            "memory_admitted": memory_admitted,
        },
        "claim_scope": (
            "process-complete physical eight-process, 32-device TPU v4 float32 forward witness "
            "for the source-pinned fine public 3D ring current/Joule/heat model against its CPU "
            "float64 same-mesh authority; not fresh Elmer execution, mesh convergence, FDTDX, "
            "inverse design, scaling, live HBM, preemption recovery, foundry calibration, or "
            "fabricated-device validation"
        ),
    }
    nonfinite_paths = _json_nonfinite_paths(process_payload)
    output_root = Path(os.environ["PHOXLA_OUTPUT_DIR"])
    if nonfinite_paths or not cast(bool, numerics["all_finite"]) or not outputs_finite:
        diagnostic = _diagnostic_payload(
            provenance,
            process_index,
            numerics,
            nonfinite_paths,
        )
        _atomic_json(output_root / "results" / "numerical-diagnostic.json", diagnostic)
        if process_index == 0:
            _atomic_json(remote_run / "results" / "numerical-diagnostic.json", diagnostic)
        multihost_utils.sync_global_devices(f"femx-ring-forward-nonfinite-{provenance['run_id']}")
        print(json.dumps({"status": "failed", "run_id": provenance["run_id"]}, sort_keys=True))
        return 2

    hlo_path = output_root / "hlo" / "forward.stablehlo.mlir"
    _atomic_text(hlo_path, stablehlo)
    _atomic_json(output_root / "results" / "process-metrics.json", process_payload)
    if process_index == 0:
        _atomic_json(remote_run / "results" / "metrics.json", process_payload)
    multihost_utils.sync_global_devices(f"femx-ring-forward-written-{provenance['run_id']}")
    print(
        json.dumps(
            {"status": process_payload["status"], "run_id": provenance["run_id"]},
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
