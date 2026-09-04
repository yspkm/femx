"""Process-complete admission for a physical TPU FDTDX mode-source witness.

The remote runner writes one record per initialized JAX process.  A successful record from
process zero is not multi-host evidence: this module admits only a mutually consistent, complete
process set whose addressable source shards cover the global source plane exactly once.  It has no
JAX, FDTDX, or cloud dependency and therefore remains usable in portable CI.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NoReturn, cast

from femx.core.errors import ValidationError

PROCESS_EVIDENCE_SCHEMA = "femx.fdtdx.mode_source.tpu_process/v1"
PROCESS_SET_EVIDENCE_SCHEMA = "femx.validation.fdtdx_mode_source.tpu_process_set/v1"
_BINDING_SCHEMA = "femx.fdtdx.distributed_mode_source/v1"
_WORKER_ENTRY_CLAIM_SCHEMA = "femx.fdtdx.mode_source.worker_entry_claim/v1"
_FDTDX_PACKAGE_VERSION = "0.6.2"
_FDTDX_SOURCE_REVISION = "81a58da9cde4a4ff822f835b63597c0d0d8ba978"
_FDTDX_SOURCE_DIGEST = "c881f0cf32b4272dc10e0acee716bfda3cf2d0f11973c23d31f29b11f3dce01c"
_SCALAR_CONTRACT = {
    "field_dtype": "float32",
    "mode_dtype": "complex64",
    "time_offset_dtype": "float32",
    "x64_enabled": False,
    "precision_fallback": False,
}


def _fail(message: str) -> NoReturn:
    raise ValidationError(f"physical TPU FDTDX mode-source evidence {message}")


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        _fail(f"requires {label} to be an object with string keys")
    return value


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        _fail(f"requires {label} to be an array")
    return value


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        _fail(f"requires {label} to be a nonempty trimmed string")
    return value


def _integer(value: object, *, label: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if positive else "nonnegative"
        _fail(f"requires {label} to be a {qualifier} integer")
    return value


def _number(value: object, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"requires {label} to be a finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0 or (positive and converted <= 0.0):
        _fail(f"requires {label} to be a finite {'positive' if positive else 'nonnegative'} number")
    return converted


def _boolean(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"requires {label} to be boolean")
    return value


def _sha256(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        _fail(f"requires {label} to be a canonical lowercase SHA-256")
    return text


def _canonical_digest(record: Mapping[str, object]) -> str:
    try:
        encoded = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValidationError(
            "physical TPU FDTDX mode-source evidence record is not canonical JSON"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class _ProcessEvidence:
    process_index: int
    worker_index: int
    digest: str
    process_count: int
    local_device_count: int
    global_device_count: int
    provenance_identity: tuple[object, ...]
    runtime_identity: tuple[object, ...]
    source_identity: tuple[object, ...]
    binding_digest: str
    simulation_identity: tuple[object, ...]
    result_identity: tuple[object, ...]
    addressable_x_ranges: tuple[tuple[int, int], ...]
    lowering_seconds: float
    compilation_seconds: float
    warmup_seconds: float
    execution_seconds: float
    compiler_peak_bytes: int
    hbm_capacity_bytes: int
    stablehlo_all_gather_count: int


def _binding(
    value: object,
    *,
    process_index: int,
    process_count: int,
    local_device_count: int,
    global_device_count: int,
) -> tuple[tuple[object, ...], tuple[tuple[int, int], ...]]:
    binding = _mapping(value, label="source.binding")
    if binding.get("schema_version") != _BINDING_SCHEMA:
        _fail("has an unsupported distributed source binding schema")
    if binding.get("physical_evidence") is not False:
        _fail("must preserve the binding record's process-local claim scope")
    source_name = _text(binding.get("source_name"), label="source.binding.source_name")
    contract_sha256 = _sha256(
        binding.get("source_contract_sha256"),
        label="source.binding.source_contract_sha256",
    )
    mesh_axis_name = _text(binding.get("mesh_axis_name"), label="source.binding.mesh_axis_name")
    if mesh_axis_name != "shard" or list(
        _sequence(binding.get("partition_spec"), label="source.binding.partition_spec")
    ) != ["replicated", "shard", "replicated", "replicated"]:
        _fail("requires the canonical FDTDX first-spatial-axis sharding")
    shape = tuple(
        _integer(item, label="source.binding.global_shape[]", positive=True)
        for item in _sequence(binding.get("global_shape"), label="source.binding.global_shape")
    )
    if len(shape) != 4 or shape[0] != 3 or shape[3] != 1:
        _fail("requires one three-component source plane")
    if shape[1] % global_device_count != 0:
        _fail("requires the source x extent to be divisible by the global device count")
    if binding.get("field_dtype") != "complex64" or binding.get("time_offset_dtype") != "float32":
        _fail("requires complex64 source fields and float32 time offsets")
    for key, expected in (
        ("process_index", process_index),
        ("process_count", process_count),
        ("local_device_count", local_device_count),
        ("global_device_count", global_device_count),
    ):
        if _integer(binding.get(key), label=f"source.binding.{key}") != expected:
            _fail(f"has inconsistent binding {key.replace('_', ' ')}")
    raw_ranges = _sequence(
        binding.get("addressable_x_ranges"),
        label="source.binding.addressable_x_ranges",
    )
    ranges: list[tuple[int, int]] = []
    for raw_range in raw_ranges:
        pair = _sequence(raw_range, label="source.binding.addressable_x_ranges[]")
        if len(pair) != 2:
            _fail("requires every addressable x range to contain start and stop")
        start = _integer(pair[0], label="source.binding.addressable_x_ranges[].start")
        stop = _integer(pair[1], label="source.binding.addressable_x_ranges[].stop", positive=True)
        if not start < stop <= shape[1]:
            _fail("has an addressable x range outside the global source plane")
        ranges.append((start, stop))
    if len(ranges) != local_device_count or tuple(sorted(ranges)) != tuple(ranges):
        _fail("requires one canonical addressable x range per local device")
    if binding.get("profile_distribution") != "identical_full_snapshot_per_process":
        _fail("requires the declared identical host snapshot distribution")
    if binding.get("execution_policy") != "outer_jit_with_arrays_objects_config_as_arguments":
        _fail("requires the explicit outer-JIT execution policy")
    identity = (source_name, contract_sha256, mesh_axis_name, shape)
    return identity, tuple(ranges)


def _process(record: Mapping[str, object]) -> _ProcessEvidence:
    if record.get("schema_version") != PROCESS_EVIDENCE_SCHEMA or record.get("status") != "passed":
        _fail("contains an unsupported or non-passing process record")
    provenance = _mapping(record.get("provenance"), label="provenance")
    runtime = _mapping(record.get("runtime"), label="runtime")
    launch_claim = _mapping(record.get("launch_claim"), label="launch_claim")
    source = _mapping(record.get("source"), label="source")
    simulation = _mapping(record.get("simulation"), label="simulation")
    numerics = _mapping(record.get("numerics"), label="numerics")
    execution = _mapping(record.get("execution"), label="execution")

    process_index = _integer(runtime.get("process_index"), label="runtime.process_index")
    process_count = _integer(
        runtime.get("process_count"), label="runtime.process_count", positive=True
    )
    local_device_count = _integer(
        runtime.get("local_device_count"), label="runtime.local_device_count", positive=True
    )
    global_device_count = _integer(
        runtime.get("global_device_count"), label="runtime.global_device_count", positive=True
    )
    if process_count < 2 or process_index >= process_count:
        _fail("requires a valid multi-process JAX identity")
    if global_device_count != process_count * local_device_count:
        _fail("has inconsistent global and process-local device counts")
    device_kinds = tuple(
        _text(item, label="runtime.device_kinds[]")
        for item in _sequence(runtime.get("device_kinds"), label="runtime.device_kinds")
    )
    if (
        runtime.get("backend") != "tpu"
        or runtime.get("x64_enabled") is not False
        or not device_kinds
        or dict(_mapping(runtime.get("scalar_contract"), label="runtime.scalar_contract"))
        != _SCALAR_CONTRACT
    ):
        _fail("requires the exact physical TPU float32/complex64 runtime contract")
    runtime_identity = (
        _text(runtime.get("jax_version"), label="runtime.jax_version"),
        _text(runtime.get("jaxlib_version"), label="runtime.jaxlib_version"),
        _text(runtime.get("fdtdx_version"), label="runtime.fdtdx_version"),
        device_kinds,
        process_count,
        local_device_count,
        global_device_count,
    )
    if runtime_identity[2] != _FDTDX_PACKAGE_VERSION:
        _fail("was not produced by the locked FDTDX package version")

    provenance_identity = (
        _text(provenance.get("run_id"), label="provenance.run_id"),
        _text(provenance.get("profile"), label="provenance.profile"),
        _sha256(provenance.get("source_digest"), label="provenance.source_digest"),
        _sha256(provenance.get("config_digest"), label="provenance.config_digest"),
    )
    if launch_claim.get("schema_version") != _WORKER_ENTRY_CLAIM_SCHEMA:
        _fail("has an unsupported worker-entry claim schema")
    worker_index = _integer(launch_claim.get("worker_index"), label="launch_claim.worker_index")
    if worker_index >= process_count:
        _fail("has a worker-entry claim outside the declared topology")
    if (
        launch_claim.get("run_id") != provenance_identity[0]
        or launch_claim.get("source_sha256") != provenance_identity[2]
        or launch_claim.get("config_sha256") != provenance_identity[3]
        or _integer(launch_claim.get("process_index"), label="launch_claim.process_index")
        != process_index
    ):
        _fail("has a worker-entry claim inconsistent with provenance")

    binding_record = _mapping(source.get("binding"), label="source.binding")
    binding_identity, ranges = _binding(
        binding_record,
        process_index=process_index,
        process_count=process_count,
        local_device_count=local_device_count,
        global_device_count=global_device_count,
    )
    binding_digest = _sha256(source.get("binding_sha256"), label="source.binding_sha256")
    if binding_digest != _canonical_digest(binding_record):
        _fail("has inconsistent distributed binding digests")
    fingerprint = _mapping(source.get("fdtdx_fingerprint"), label="source.fdtdx_fingerprint")
    if (
        fingerprint.get("package_version") != _FDTDX_PACKAGE_VERSION
        or fingerprint.get("source_revision") != _FDTDX_SOURCE_REVISION
        or fingerprint.get("source_digest") != _FDTDX_SOURCE_DIGEST
    ):
        _fail("does not match the locked FDTDX source fingerprint")
    module_hashes = _mapping(source.get("module_sha256"), label="source.module_sha256")
    if not module_hashes:
        _fail("requires at least one locked FDTDX module hash")
    canonical_module_hashes = tuple(
        (name, _sha256(value, label=f"source.module_sha256.{name}"))
        for name, value in sorted(module_hashes.items())
    )
    source_identity = (
        *binding_identity,
        _sha256(source.get("bundle_sha256"), label="source.bundle_sha256"),
        canonical_module_hashes,
    )

    grid_shape = tuple(
        _integer(item, label="simulation.grid_shape_xyz[]", positive=True)
        for item in _sequence(simulation.get("grid_shape_xyz"), label="simulation.grid_shape_xyz")
    )
    binding_shape = cast(tuple[int, ...], binding_identity[3])
    if len(grid_shape) != 3 or grid_shape[:2] != binding_shape[1:3]:
        _fail("has a simulation grid inconsistent with the source plane")
    source_z_index = _integer(simulation.get("source_z_index"), label="simulation.source_z_index")
    if source_z_index >= grid_shape[2] - 1:
        _fail("places the source outside the admitted simulation interior")
    time_steps = _integer(
        simulation.get("time_steps"), label="simulation.time_steps", positive=True
    )
    simulation_identity = (
        grid_shape,
        source_z_index,
        _number(
            simulation.get("simulation_time_s"), label="simulation.simulation_time_s", positive=True
        ),
        time_steps,
        _number(
            simulation.get("relative_permittivity"),
            label="simulation.relative_permittivity",
            positive=True,
        ),
        tuple(
            _text(item, label="simulation.boundaries[]")
            for item in _sequence(simulation.get("boundaries"), label="simulation.boundaries")
        ),
    )
    if simulation_identity[-1] != ("periodic",) * 6:
        _fail("requires the bounded six-face periodic infrastructure witness")

    step = _integer(numerics.get("completed_step"), label="numerics.completed_step", positive=True)
    if step != time_steps:
        _fail("did not complete the declared number of FDTD time steps")
    initial_e = _number(numerics.get("initial_e_l2"), label="numerics.initial_e_l2")
    initial_h = _number(numerics.get("initial_h_l2"), label="numerics.initial_h_l2")
    final_e = _number(numerics.get("final_e_l2"), label="numerics.final_e_l2", positive=True)
    final_h = _number(numerics.get("final_h_l2"), label="numerics.final_h_l2", positive=True)
    downstream_e = _number(
        numerics.get("downstream_e_l2"), label="numerics.downstream_e_l2", positive=True
    )
    if initial_e != 0.0 or initial_h != 0.0:
        _fail("requires an exactly zero initial field")
    if not _boolean(numerics.get("all_fields_finite"), label="numerics.all_fields_finite"):
        _fail("contains non-finite final fields")
    result_identity = (step, initial_e, initial_h, final_e, final_h, downstream_e)

    memory = _mapping(execution.get("compiler_memory"), label="execution.compiler_memory")
    compiler_peak_bytes = _integer(
        memory.get("compiler_peak_bytes"), label="execution.compiler_memory.compiler_peak_bytes"
    )
    hbm_capacity_bytes = _integer(
        memory.get("hbm_capacity_bytes_per_device"),
        label="execution.compiler_memory.hbm_capacity_bytes_per_device",
        positive=True,
    )
    if compiler_peak_bytes > hbm_capacity_bytes:
        _fail("reports a compiler peak larger than the admitted per-device HBM capacity")
    return _ProcessEvidence(
        process_index=process_index,
        worker_index=worker_index,
        digest=_canonical_digest(record),
        process_count=process_count,
        local_device_count=local_device_count,
        global_device_count=global_device_count,
        provenance_identity=provenance_identity,
        runtime_identity=runtime_identity,
        source_identity=source_identity,
        binding_digest=binding_digest,
        simulation_identity=simulation_identity,
        result_identity=result_identity,
        addressable_x_ranges=ranges,
        lowering_seconds=_number(
            execution.get("lowering_seconds"), label="execution.lowering_seconds"
        ),
        compilation_seconds=_number(
            execution.get("compilation_seconds"), label="execution.compilation_seconds"
        ),
        warmup_seconds=_number(execution.get("warmup_seconds"), label="execution.warmup_seconds"),
        execution_seconds=_number(
            execution.get("execution_seconds"), label="execution.execution_seconds"
        ),
        compiler_peak_bytes=compiler_peak_bytes,
        hbm_capacity_bytes=hbm_capacity_bytes,
        stablehlo_all_gather_count=_integer(
            execution.get("stablehlo_all_gather_count"),
            label="execution.stablehlo_all_gather_count",
        ),
    )


def _range(values: Sequence[float]) -> dict[str, float]:
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


@dataclass(frozen=True, slots=True)
class TpuFdtdxModeSourceProcessSetEvidence:
    """Canonical admission summary for every process in one physical TPU FDTDX run."""

    payload: Mapping[str, object]

    def canonical_data(self) -> dict[str, object]:
        """Return a detached JSON-compatible copy of the aggregate."""

        return cast(dict[str, object], json.loads(self.canonical_json()))

    def canonical_json(self) -> str:
        """Serialize the aggregate deterministically."""

        return json.dumps(
            self.payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    def digest(self) -> str:
        """Return the logical SHA-256 of the admitted process set."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def aggregate_tpu_fdtdx_mode_source_process_evidence(
    records: Sequence[Mapping[str, object]],
) -> TpuFdtdxModeSourceProcessSetEvidence:
    """Validate and aggregate one complete physical TPU FDTDX process set."""

    if not records:
        _fail("requires at least one process record")
    extracted = tuple(_process(_mapping(record, label="process record")) for record in records)
    ordered = tuple(sorted(extracted, key=lambda record: record.process_index))
    baseline = ordered[0]
    if len(ordered) != baseline.process_count or tuple(
        record.process_index for record in ordered
    ) != tuple(range(baseline.process_count)):
        _fail("requires exactly one record for every declared process index")
    if tuple(sorted(record.worker_index for record in ordered)) != tuple(
        range(baseline.process_count)
    ):
        _fail("requires exactly one immutable worker-entry claim per TPU worker")
    for record in ordered[1:]:
        for label, observed, expected in (
            ("provenance", record.provenance_identity, baseline.provenance_identity),
            ("runtime", record.runtime_identity, baseline.runtime_identity),
            ("source", record.source_identity, baseline.source_identity),
            ("simulation", record.simulation_identity, baseline.simulation_identity),
            ("numerics", record.result_identity, baseline.result_identity),
        ):
            if observed != expected:
                _fail(f"has inconsistent per-process {label}")

    all_ranges = tuple(sorted(item for record in ordered for item in record.addressable_x_ranges))
    expected_start = 0
    for start, stop in all_ranges:
        if start != expected_start:
            _fail("does not cover the global source x extent exactly once")
        expected_start = stop
    global_x = cast(tuple[int, ...], baseline.source_identity[3])[1]
    if len(all_ranges) != baseline.global_device_count or expected_start != global_x:
        _fail("does not cover one source shard per global TPU device")

    run_id, profile, source_digest, config_digest = baseline.provenance_identity
    jax_version, jaxlib_version, fdtdx_version, device_kinds, _, _, _ = baseline.runtime_identity
    source_name, contract_sha256, mesh_axis_name, global_shape, bundle_sha256, module_hashes = (
        baseline.source_identity
    )
    grid_shape, source_z_index, simulation_time, time_steps, relative_permittivity, boundaries = (
        baseline.simulation_identity
    )
    _, _, _, final_e, final_h, downstream_e = baseline.result_identity
    payload: dict[str, object] = {
        "schema_version": PROCESS_SET_EVIDENCE_SCHEMA,
        "status": "passed",
        "provenance": {
            "run_id": run_id,
            "profile": profile,
            "source_digest": source_digest,
            "config_digest": config_digest,
            "process_records": [
                {"process_index": record.process_index, "sha256": record.digest}
                for record in ordered
            ],
        },
        "runtime": {
            "backend": "tpu",
            "jax_version": jax_version,
            "jaxlib_version": jaxlib_version,
            "fdtdx_version": fdtdx_version,
            "x64_enabled": False,
            "process_indexes": list(range(baseline.process_count)),
            "worker_indexes": sorted(record.worker_index for record in ordered),
            "process_count": baseline.process_count,
            "local_device_count": baseline.local_device_count,
            "global_device_count": baseline.global_device_count,
            "device_kinds": list(cast(tuple[str, ...], device_kinds)),
            "scalar_contract": _SCALAR_CONTRACT,
        },
        "source": {
            "source_name": source_name,
            "source_contract_sha256": contract_sha256,
            "bundle_sha256": bundle_sha256,
            "mesh_axis_name": mesh_axis_name,
            "global_shape": list(cast(tuple[int, ...], global_shape)),
            "combined_addressable_x_ranges": [list(item) for item in all_ranges],
            "every_global_source_shard_addressable_once": True,
            "process_bindings": [
                {"process_index": record.process_index, "sha256": record.binding_digest}
                for record in ordered
            ],
            "fdtdx_fingerprint": {
                "package_version": _FDTDX_PACKAGE_VERSION,
                "source_revision": _FDTDX_SOURCE_REVISION,
                "source_digest": _FDTDX_SOURCE_DIGEST,
                "module_sha256": dict(cast(tuple[tuple[str, str], ...], module_hashes)),
            },
        },
        "simulation": {
            "grid_shape_xyz": list(cast(tuple[int, ...], grid_shape)),
            "source_z_index": source_z_index,
            "simulation_time_s": simulation_time,
            "time_steps": time_steps,
            "relative_permittivity": relative_permittivity,
            "boundaries": list(cast(tuple[str, ...], boundaries)),
        },
        "numerics": {
            "all_processes_completed_same_step": True,
            "all_process_fields_finite": True,
            "final_e_l2": final_e,
            "final_h_l2": final_h,
            "downstream_e_l2": downstream_e,
        },
        "execution": {
            "lowering_seconds_across_processes": _range(
                tuple(record.lowering_seconds for record in ordered)
            ),
            "compilation_seconds_across_processes": _range(
                tuple(record.compilation_seconds for record in ordered)
            ),
            "warmup_seconds_across_processes": _range(
                tuple(record.warmup_seconds for record in ordered)
            ),
            "execution_seconds_across_processes": _range(
                tuple(record.execution_seconds for record in ordered)
            ),
            "maximum_compiler_peak_bytes": max(record.compiler_peak_bytes for record in ordered),
            "hbm_capacity_bytes_per_device": min(record.hbm_capacity_bytes for record in ordered),
            "stablehlo_all_gather_counts": [
                record.stablehlo_all_gather_count for record in ordered
            ],
        },
        "claim_scope": (
            "process-complete physical multi-host TPU float32/complex64 execution of one "
            "bounded homogeneous-port ModeBundle through the locked FDTDX time-domain source; "
            "this proves source sharding, injection, and finite nonzero time advance, not Elmer "
            "parity, waveguide accuracy, convergence, S-parameters, scaling, adjoint, or "
            "preemption recovery"
        ),
    }
    return TpuFdtdxModeSourceProcessSetEvidence(payload)


__all__ = [
    "PROCESS_EVIDENCE_SCHEMA",
    "PROCESS_SET_EVIDENCE_SCHEMA",
    "TpuFdtdxModeSourceProcessSetEvidence",
    "aggregate_tpu_fdtdx_mode_source_process_evidence",
]
