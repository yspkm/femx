"""Process-complete aggregation for physical TPU collective evidence.

The physical runner writes one independent metrics record per JAX process.  A multi-host claim
cannot use process zero alone: this module admits the complete process set, proves that every
process and FEM partition appears exactly once, and reports ordinal critical-path timings as the
maximum observed across processes.  It imports neither JAX nor a cloud control plane.
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

PROCESS_EVIDENCE_SCHEMA = "femx.jax.port_collective.tpu_evidence/v4"
PROCESS_SET_EVIDENCE_SCHEMA = "femx.validation.tpu_collective.process_set/v3"
_MESH_REPORT_SCHEMA = "femx.jax.port_collective.mesh_report/v1"
_CHECKPOINT_REPORT_SCHEMA = "femx.jax.port_collective.checkpoint_fragment_report/v1"
_TIMING_REPORT_SCHEMA = "femx.jax.port_collective.timing_report/v1"
_MEMORY_REPORT_SCHEMA = "femx.jax.port_collective.memory_report/v1"
_WORKER_ENTRY_CLAIM_SCHEMA = "femx.jax.port_collective.worker_entry_claim/v1"
_EXECUTABLE_NAMES = ("complex_forward", "complex_vjp", "real_forward", "real_vjp")
_CHECKPOINT_ARRAY_NAMES = (
    "cell-local-dof-map",
    "complex-owned-cotangent",
    "complex-owned-vector",
    "mass-cell-blocks",
    "real-owned-cotangent",
    "real-owned-vector",
    "shifted-cell-blocks",
    "stiffness-cell-blocks",
)
_RISK_RANK = {"not_assessed": 0, "safe": 1, "elevated": 2, "high": 3, "extreme": 4}
_COMPLEX_SCALAR_CONTRACT = {
    "logical_dtype": "complex64",
    "matrix_dtype": "float32",
    "index_dtype": "int32",
    "execution_representation": "native complex64",
    "matmul_precision": "highest",
    "host_reference_dtype": "complex128",
    "precision_fallback": False,
}


def _fail(message: str) -> NoReturn:
    raise ValidationError(f"physical TPU process-set evidence {message}")


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


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"requires {label} to be a finite nonnegative number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        _fail(f"requires {label} to be a finite nonnegative number")
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
        payload = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValidationError(
            "physical TPU process-set evidence record is not canonical JSON"
        ) from error
    return hashlib.sha256(payload).hexdigest()


def _risk_for_fraction(fraction: float) -> str:
    if fraction < 0.70:
        return "safe"
    if fraction < 0.85:
        return "elevated"
    if fraction < 0.95:
        return "high"
    return "extreme"


def _all_true(value: object, *, label: str) -> bool:
    mapping = _mapping(value, label=label)
    if not mapping:
        _fail(f"requires {label} to be nonempty")
    result = True
    for key, item in mapping.items():
        if isinstance(item, Mapping):
            result = _all_true(item, label=f"{label}.{key}") and result
        else:
            result = _boolean(item, label=f"{label}.{key}") and result
    return result


def _maximum_number_mapping(value: object, *, label: str) -> float:
    mapping = _mapping(value, label=label)
    if not mapping:
        _fail(f"requires {label} to be nonempty")
    return max(_number(item, label=f"{label}.{key}") for key, item in mapping.items())


@dataclass(frozen=True, slots=True)
class _ExecutableProcessEvidence:
    lowering_seconds: float
    compilation_seconds: float
    warmup_seconds: float
    execution_seconds: tuple[float, ...]
    compiler_peak_bytes: int
    hbm_capacity_bytes: int
    hbm_fraction: float
    risk: str
    collective_permute_count: int


@dataclass(frozen=True, slots=True)
class _ProcessEvidence:
    process_index: int
    worker_index: int
    digest: str
    provenance_identity: tuple[str, str, str, str]
    runtime_identity: tuple[object, ...]
    physics_identity: tuple[object, ...]
    problem_identity: tuple[object, ...]
    checkpoint_identity: tuple[object, ...]
    assignment_identity: tuple[tuple[object, ...], ...]
    local_partition_mask: tuple[int, ...]
    reported_partition_counts: tuple[int, ...]
    maximum_action_difference: float
    maximum_vjp_difference: float
    maximum_host_precision_difference: float
    action_tolerance: float
    vjp_tolerance: float
    host_precision_tolerance: float
    executables: tuple[tuple[str, _ExecutableProcessEvidence], ...]


def _executable(record: Mapping[str, object], *, name: str) -> _ExecutableProcessEvidence:
    timing = _mapping(record.get("timing"), label=f"executables.{name}.timing")
    if timing.get("schema_version") != _TIMING_REPORT_SCHEMA:
        _fail(f"has unsupported {name} timing schema")
    if timing.get("synchronization") != "every timed result blocked until ready":
        _fail(f"does not prove synchronized {name} results")
    samples = tuple(
        _number(value, label=f"executables.{name}.timing.execution_seconds")
        for value in _sequence(
            timing.get("execution_seconds"),
            label=f"executables.{name}.timing.execution_seconds",
        )
    )
    if len(samples) != 5:
        _fail(f"requires exactly five {name} execution samples")

    memory = _mapping(record.get("memory"), label=f"executables.{name}.memory")
    if memory.get("schema_version") != _MEMORY_REPORT_SCHEMA:
        _fail(f"has unsupported {name} memory schema")
    peak = _integer(
        memory.get("compiler_peak_bytes"),
        label=f"executables.{name}.memory.compiler_peak_bytes",
    )
    capacity = _integer(
        memory.get("hbm_capacity_bytes_per_device"),
        label=f"executables.{name}.memory.hbm_capacity_bytes_per_device",
        positive=True,
    )
    fraction = _number(memory.get("hbm_fraction"), label=f"executables.{name}.memory.hbm_fraction")
    if not math.isclose(fraction, peak / capacity, rel_tol=1.0e-12, abs_tol=1.0e-15):
        _fail(f"has inconsistent {name} compiler-memory fraction")
    risk = _text(memory.get("risk"), label=f"executables.{name}.memory.risk")
    if risk != _risk_for_fraction(fraction):
        _fail(f"has inconsistent {name} compiler-memory risk")
    if memory.get("claim_scope") != "compiler estimate; not live HBM usage":
        _fail(f"overstates {name} compiler memory as live HBM")
    if _boolean(
        record.get("stablehlo_contains_all_gather"),
        label=f"executables.{name}.stablehlo_contains_all_gather",
    ):
        _fail(f"contains an all-gather in {name} StableHLO")

    return _ExecutableProcessEvidence(
        lowering_seconds=_number(
            timing.get("lowering_seconds"),
            label=f"executables.{name}.timing.lowering_seconds",
        ),
        compilation_seconds=_number(
            timing.get("compilation_seconds"),
            label=f"executables.{name}.timing.compilation_seconds",
        ),
        warmup_seconds=_number(
            timing.get("warmup_seconds"),
            label=f"executables.{name}.timing.warmup_seconds",
        ),
        execution_seconds=samples,
        compiler_peak_bytes=peak,
        hbm_capacity_bytes=capacity,
        hbm_fraction=fraction,
        risk=risk,
        collective_permute_count=_integer(
            record.get("stablehlo_collective_permute_count"),
            label=f"executables.{name}.stablehlo_collective_permute_count",
        ),
    )


def _process(record: Mapping[str, object]) -> _ProcessEvidence:
    if record.get("schema_version") != PROCESS_EVIDENCE_SCHEMA:
        _fail("has an unsupported process-record schema")
    if record.get("status") != "passed":
        _fail("contains a process record that did not pass")
    provenance = _mapping(record.get("provenance"), label="provenance")
    runtime = _mapping(record.get("runtime"), label="runtime")
    physics = _mapping(record.get("physics"), label="physics")
    problem = _mapping(record.get("problem"), label="problem")
    mesh = _mapping(record.get("mesh_report"), label="mesh_report")
    addressability = _mapping(record.get("addressability"), label="addressability")
    checkpoint = _mapping(record.get("checkpoint"), label="checkpoint")
    numerics = _mapping(record.get("numerics"), label="numerics")
    launch_claim = _mapping(record.get("launch_claim"), label="launch_claim")

    process_index = _integer(runtime.get("process_index"), label="runtime.process_index")
    process_count = _integer(
        runtime.get("process_count"), label="runtime.process_count", positive=True
    )
    local_device_count = _integer(
        runtime.get("local_device_count"),
        label="runtime.local_device_count",
        positive=True,
    )
    global_device_count = _integer(
        runtime.get("global_device_count"),
        label="runtime.global_device_count",
        positive=True,
    )
    if process_index >= process_count:
        _fail("has a process index outside the declared process count")
    if (
        runtime.get("backend") != "tpu"
        or runtime.get("x64_enabled") is not False
        or runtime.get("default_matmul_precision") != "highest"
    ):
        _fail("requires physical TPU float32/complex64 with highest matmul precision")
    if process_count < 2 or global_device_count != process_count * local_device_count:
        _fail("has inconsistent multi-process device counts")
    device_kinds = tuple(
        _text(value, label="runtime.device_kinds")
        for value in _sequence(runtime.get("device_kinds"), label="runtime.device_kinds")
    )
    if not device_kinds:
        _fail("requires at least one TPU device kind")
    scalar_contract = _mapping(
        runtime.get("complex_scalar_contract"),
        label="runtime.complex_scalar_contract",
    )
    if dict(scalar_contract) != _COMPLEX_SCALAR_CONTRACT:
        _fail("requires the exact TPU complex64 and host complex128 scalar contract")

    layout_sha256 = _sha256(problem.get("layout_sha256"), label="problem.layout_sha256")
    partition_count = _integer(
        problem.get("partition_count"), label="problem.partition_count", positive=True
    )
    if partition_count != global_device_count:
        _fail("requires one FEM partition per global TPU device")

    if mesh.get("schema_version") != _MESH_REPORT_SCHEMA:
        _fail("has an unsupported Mesh-report schema")
    if mesh.get("axis_name") != "partition" or mesh.get("is_multi_process") is not True:
        _fail("requires the declared multi-process partition Mesh")
    if mesh.get("layout_sha256") != layout_sha256:
        _fail("has inconsistent problem and Mesh layout identities")
    for key, expected in (
        ("partition_count", partition_count),
        ("global_device_count", global_device_count),
        ("addressable_device_count", local_device_count),
        ("process_count", process_count),
    ):
        if _integer(mesh.get(key), label=f"mesh_report.{key}", positive=True) != expected:
            _fail(f"has inconsistent Mesh {key.replace('_', ' ')}")

    local_mask = tuple(
        _integer(value, label="addressability.process_local_partition_mask")
        for value in _sequence(
            addressability.get("process_local_partition_mask"),
            label="addressability.process_local_partition_mask",
        )
    )
    reported_counts = tuple(
        _integer(value, label="addressability.partition_addressability_counts")
        for value in _sequence(
            addressability.get("partition_addressability_counts"),
            label="addressability.partition_addressability_counts",
        )
    )
    if len(local_mask) != partition_count or any(value not in (0, 1) for value in local_mask):
        _fail("has a noncanonical process-local partition mask")
    if sum(local_mask) != local_device_count:
        _fail("has a process-local partition mask inconsistent with local devices")
    if reported_counts != (1,) * partition_count:
        _fail("does not report every partition addressable exactly once")
    if addressability.get("every_partition_addressable_once") is not True:
        _fail("does not attest unique partition addressability")

    assignments: list[tuple[object, ...]] = []
    device_keys: list[tuple[int, int]] = []
    assigned_processes: list[int] = []
    raw_assignments = _sequence(mesh.get("assignments"), label="mesh_report.assignments")
    if len(raw_assignments) != partition_count:
        _fail("requires one Mesh assignment per FEM partition")
    for expected_partition, raw_assignment in enumerate(raw_assignments):
        assignment = _mapping(raw_assignment, label="mesh_report.assignments[]")
        partition = _integer(
            assignment.get("partition_index"),
            label="mesh_report.assignments[].partition_index",
        )
        if partition != expected_partition:
            _fail("requires Mesh assignments in canonical partition order")
        assignment_process = _integer(
            assignment.get("process_index"),
            label="mesh_report.assignments[].process_index",
        )
        if assignment_process >= process_count:
            _fail("has a Mesh assignment outside the declared process count")
        device_id = _integer(
            assignment.get("device_id"), label="mesh_report.assignments[].device_id"
        )
        platform = _text(assignment.get("platform"), label="mesh_report.assignments[].platform")
        device_kind = _text(
            assignment.get("device_kind"), label="mesh_report.assignments[].device_kind"
        )
        addressable = _boolean(
            assignment.get("addressable"), label="mesh_report.assignments[].addressable"
        )
        if platform != "tpu" or device_kind not in device_kinds:
            _fail("has a Mesh assignment inconsistent with the TPU runtime")
        if addressable != bool(local_mask[partition]) or addressable != (
            assignment_process == process_index
        ):
            _fail("has Mesh addressability inconsistent with its process-local mask")
        device_keys.append((assignment_process, device_id))
        assigned_processes.append(assignment_process)
        assignments.append((partition, assignment_process, device_id, platform, device_kind))
    if len(set(device_keys)) != partition_count:
        _fail("requires one unique TPU device identity per FEM partition")
    if set(assigned_processes) != set(range(process_count)) or any(
        assigned_processes.count(index) != local_device_count for index in range(process_count)
    ):
        _fail("has Mesh assignments inconsistent with the declared process topology")

    fragment = _mapping(checkpoint.get("fragment"), label="checkpoint.fragment")
    if checkpoint.get("mode") not in {
        "fresh-process-roundtrip",
        "restored-external-fragment",
    }:
        _fail("has an unsupported checkpoint mode")
    if checkpoint.get("restored_state_consumed_by_operator") is not True:
        _fail("did not consume restored checkpoint state")
    if checkpoint.get("cross_topology_restore") is not False:
        _fail("claims unsupported cross-topology restore")
    if _boolean(
        checkpoint.get("actual_preemption_event"),
        label="checkpoint.actual_preemption_event",
    ):
        _fail("mixes preemption-recovery claims into the correctness evidence")
    if fragment.get("schema_version") != _CHECKPOINT_REPORT_SCHEMA:
        _fail("has an unsupported checkpoint-fragment report schema")
    if _integer(fragment.get("process_index"), label="checkpoint.fragment.process_index") != (
        process_index
    ):
        _fail("has a checkpoint fragment for the wrong process")
    if (
        _integer(
            fragment.get("process_count"),
            label="checkpoint.fragment.process_count",
            positive=True,
        )
        != process_count
    ):
        _fail("has a checkpoint fragment with the wrong process count")
    if fragment.get("layout_sha256") != layout_sha256:
        _fail("has a checkpoint fragment with the wrong layout identity")
    _sha256(fragment.get("manifest_sha256"), label="checkpoint.fragment.manifest_sha256")
    array_names = tuple(
        _text(value, label="checkpoint.fragment.array_names")
        for value in _sequence(
            fragment.get("array_names"),
            label="checkpoint.fragment.array_names",
        )
    )
    if array_names != _CHECKPOINT_ARRAY_NAMES:
        _fail("has an incomplete or noncanonical checkpoint array set")
    if (
        fragment.get("completion_scope") != "one process-local fragment"
        or fragment.get("restore_policy") != "exact same topology only; no resharding"
    ):
        _fail("has unsupported checkpoint completion or restore semantics")

    maximum_action = _number(
        numerics.get("maximum_action_relative_difference"),
        label="numerics.maximum_action_relative_difference",
    )
    maximum_vjp = _number(
        numerics.get("maximum_vjp_relative_difference"),
        label="numerics.maximum_vjp_relative_difference",
    )
    action_tolerance = _number(numerics.get("action_tolerance"), label="numerics.action_tolerance")
    vjp_tolerance = _number(numerics.get("vjp_tolerance"), label="numerics.vjp_tolerance")
    maximum_host_precision = _number(
        numerics.get("maximum_host_c64_vs_c128_relative_difference"),
        label="numerics.maximum_host_c64_vs_c128_relative_difference",
    )
    reported_host_precision = max(
        _maximum_number_mapping(
            numerics.get("host_c64_vs_c128_action_relative_differences"),
            label="numerics.host_c64_vs_c128_action_relative_differences",
        ),
        _maximum_number_mapping(
            numerics.get("host_c64_vs_c128_vjp_relative_differences"),
            label="numerics.host_c64_vs_c128_vjp_relative_differences",
        ),
    )
    if not math.isclose(maximum_host_precision, reported_host_precision, rel_tol=0.0, abs_tol=0.0):
        _fail("has an inconsistent maximum host c64-to-c128 precision difference")
    host_precision_tolerance = _number(
        numerics.get("host_precision_tolerance"),
        label="numerics.host_precision_tolerance",
    )
    if maximum_action > action_tolerance or maximum_vjp > vjp_tolerance:
        _fail("contains a numerical difference above its declared tolerance")
    if maximum_host_precision > host_precision_tolerance:
        _fail("contains a host c64-to-c128 precision difference above tolerance")
    if numerics.get("host_precision_scope") != (
        "operator arithmetic and vector/cotangent quantization with float32 cell "
        "coefficients held fixed; not float64 FEM assembly parity"
    ):
        _fail("overstates or changes the host precision-reference scope")
    if not _all_true(numerics.get("action_finite"), label="numerics.action_finite"):
        _fail("contains a non-finite action result")
    if not _all_true(numerics.get("vjp_finite"), label="numerics.vjp_finite"):
        _fail("contains a non-finite VJP result")

    raw_executables = _mapping(record.get("executables"), label="executables")
    if tuple(sorted(raw_executables)) != _EXECUTABLE_NAMES:
        _fail("does not contain the exact physical executable set")
    executables = tuple(
        (name, _executable(_mapping(raw_executables[name], label=f"executables.{name}"), name=name))
        for name in _EXECUTABLE_NAMES
    )

    provenance_identity = (
        _text(provenance.get("run_id"), label="provenance.run_id"),
        _text(provenance.get("profile"), label="provenance.profile"),
        _sha256(provenance.get("source_digest"), label="provenance.source_digest"),
        _sha256(provenance.get("config_digest"), label="provenance.config_digest"),
    )
    if launch_claim.get("schema_version") != _WORKER_ENTRY_CLAIM_SCHEMA:
        _fail("has an unsupported worker-entry claim schema")
    worker_index = _integer(
        launch_claim.get("worker_index"),
        label="launch_claim.worker_index",
    )
    if worker_index >= process_count:
        _fail("has a worker-entry claim outside the declared process count")
    if (
        launch_claim.get("run_id") != provenance_identity[0]
        or launch_claim.get("source_sha256") != provenance_identity[2]
        or launch_claim.get("config_sha256") != provenance_identity[3]
        or _integer(
            launch_claim.get("process_index"),
            label="launch_claim.process_index",
        )
        != process_index
    ):
        _fail("has a worker-entry claim inconsistent with process provenance")
    if launch_claim.get("scope") != (
        "worker-local femx entry fence after Phoxla bootstrap; prevents duplicate "
        "scientific execution but does not claim controller-level launch ownership"
    ):
        _fail("overstates or changes the worker-entry claim scope")
    runtime_identity = (
        "tpu",
        _text(runtime.get("jax_version"), label="runtime.jax_version"),
        _text(runtime.get("jaxlib_version"), label="runtime.jaxlib_version"),
        False,
        process_count,
        local_device_count,
        global_device_count,
        device_kinds,
        tuple(
            (key, tuple(value) if isinstance(value, list) else value)
            for key, value in sorted(_COMPLEX_SCALAR_CONTRACT.items())
        ),
    )
    model = _text(physics.get("model"), label="physics.model")
    if model != "2D lossless Si/SiO2 mixed H(curl)/H1 port operator":
        _fail("has an unsupported physical witness")
    shift = physics.get("shift_per_m2")
    if (
        isinstance(shift, bool)
        or not isinstance(shift, (int, float))
        or not math.isfinite(float(shift))
        or float(shift) >= 0.0
    ):
        _fail("requires physics.shift_per_m2 to be a finite negative number")
    physics_identity = (
        model,
        _number(physics.get("frequency_hz"), label="physics.frequency_hz"),
        _number(
            physics.get("silicon_refractive_index"),
            label="physics.silicon_refractive_index",
        ),
        _number(
            physics.get("silica_refractive_index"),
            label="physics.silica_refractive_index",
        ),
        _number(physics.get("core_width_m"), label="physics.core_width_m"),
        _number(physics.get("core_height_m"), label="physics.core_height_m"),
        _number(
            physics.get("cross_section_width_m"),
            label="physics.cross_section_width_m",
        ),
        _number(
            physics.get("cross_section_height_m"),
            label="physics.cross_section_height_m",
        ),
        float(shift),
    )
    if any(value <= 0.0 for value in physics_identity[1:8]):
        _fail("requires positive frequency, indices, and physical dimensions")
    problem_identity = (
        _integer(problem.get("node_count"), label="problem.node_count", positive=True),
        _integer(problem.get("triangle_count"), label="problem.triangle_count", positive=True),
        _integer(
            problem.get("free_mixed_dof_count"),
            label="problem.free_mixed_dof_count",
            positive=True,
        ),
        partition_count,
        layout_sha256,
        _integer(problem.get("halo_link_count"), label="problem.halo_link_count", positive=True),
        _integer(problem.get("halo_value_count"), label="problem.halo_value_count", positive=True),
    )
    checkpoint_identity = (
        checkpoint.get("mode"),
        _text(fragment.get("checkpoint_id"), label="checkpoint.fragment.checkpoint_id"),
        _integer(fragment.get("step"), label="checkpoint.fragment.step"),
        _sha256(fragment.get("source_sha256"), label="checkpoint.fragment.source_sha256"),
        _sha256(fragment.get("config_sha256"), label="checkpoint.fragment.config_sha256"),
    )
    if checkpoint_identity[3:] != provenance_identity[2:]:
        _fail("has checkpoint source/configuration identities inconsistent with provenance")

    return _ProcessEvidence(
        process_index=process_index,
        worker_index=worker_index,
        digest=_canonical_digest(record),
        provenance_identity=provenance_identity,
        runtime_identity=runtime_identity,
        physics_identity=physics_identity,
        problem_identity=problem_identity,
        checkpoint_identity=checkpoint_identity,
        assignment_identity=tuple(assignments),
        local_partition_mask=local_mask,
        reported_partition_counts=reported_counts,
        maximum_action_difference=maximum_action,
        maximum_vjp_difference=maximum_vjp,
        maximum_host_precision_difference=maximum_host_precision,
        action_tolerance=action_tolerance,
        vjp_tolerance=vjp_tolerance,
        host_precision_tolerance=host_precision_tolerance,
        executables=executables,
    )


def _range(values: Sequence[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


@dataclass(frozen=True, slots=True)
class TpuCollectiveProcessSetEvidence:
    """Canonical admission summary for every process in one physical TPU run."""

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
        """Return the logical SHA-256 of the aggregate."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def aggregate_tpu_collective_process_evidence(
    records: Sequence[Mapping[str, object]],
) -> TpuCollectiveProcessSetEvidence:
    """Validate and aggregate one complete set of physical TPU process records."""

    if not records:
        _fail("requires at least one process record")
    extracted = tuple(_process(_mapping(record, label="process record")) for record in records)
    ordered = tuple(sorted(extracted, key=lambda record: record.process_index))
    process_count = cast(int, ordered[0].runtime_identity[4])
    if len(ordered) != process_count or tuple(record.process_index for record in ordered) != tuple(
        range(process_count)
    ):
        _fail("requires exactly one record for every declared process index")
    if tuple(sorted(record.worker_index for record in ordered)) != tuple(range(process_count)):
        _fail("requires exactly one immutable worker-entry claim per TPU worker")

    baseline = ordered[0]
    for record in ordered[1:]:
        for label, observed, expected in (
            ("provenance", record.provenance_identity, baseline.provenance_identity),
            ("runtime", record.runtime_identity, baseline.runtime_identity),
            ("physics", record.physics_identity, baseline.physics_identity),
            ("problem", record.problem_identity, baseline.problem_identity),
            ("checkpoint", record.checkpoint_identity, baseline.checkpoint_identity),
            ("Mesh assignment", record.assignment_identity, baseline.assignment_identity),
            ("action tolerance", record.action_tolerance, baseline.action_tolerance),
            ("VJP tolerance", record.vjp_tolerance, baseline.vjp_tolerance),
            (
                "host precision tolerance",
                record.host_precision_tolerance,
                baseline.host_precision_tolerance,
            ),
        ):
            if observed != expected:
                _fail(f"has inconsistent per-process {label}")

    partition_count = cast(int, baseline.problem_identity[3])
    combined_addressability = tuple(
        sum(record.local_partition_mask[index] for record in ordered)
        for index in range(partition_count)
    )
    # Per-record Mesh/process checks imply this equality; keep the aggregate assertion explicit so
    # a future relaxation cannot silently weaken the process-set contract.
    if combined_addressability != (1,) * partition_count:  # pragma: no cover - defensive invariant
        _fail("does not cover every partition exactly once across process-local masks")

    executable_payload: dict[str, object] = {}
    executable_maps = [dict(record.executables) for record in ordered]
    for name in _EXECUTABLE_NAMES:
        process_executables = tuple(mapping[name] for mapping in executable_maps)
        sample_count = len(process_executables[0].execution_seconds)
        permutation_counts = {
            executable.collective_permute_count for executable in process_executables
        }
        if len(permutation_counts) != 1:
            _fail(f"has inconsistent per-process {name} collective counts")
        critical_path = tuple(
            max(executable.execution_seconds[index] for executable in process_executables)
            for index in range(sample_count)
        )
        risks = tuple(executable.risk for executable in process_executables)
        executable_payload[name] = {
            "process_count": process_count,
            "lowering_seconds_across_processes": _range(
                tuple(executable.lowering_seconds for executable in process_executables)
            ),
            "compilation_seconds_across_processes": _range(
                tuple(executable.compilation_seconds for executable in process_executables)
            ),
            "warmup_seconds_across_processes": _range(
                tuple(executable.warmup_seconds for executable in process_executables)
            ),
            "process_execution_median_seconds": [
                statistics.median(executable.execution_seconds)
                for executable in process_executables
            ],
            "execution_ordinal_critical_path_seconds": list(critical_path),
            "execution_ordinal_critical_path_summary_seconds": _range(critical_path),
            "sample_alignment": (
                "same program and collective order; ordinal maximum across process-local "
                "block_until_ready samples"
            ),
            "maximum_compiler_peak_bytes": max(
                executable.compiler_peak_bytes for executable in process_executables
            ),
            "hbm_capacity_bytes_per_device": min(
                executable.hbm_capacity_bytes for executable in process_executables
            ),
            "maximum_compiler_hbm_fraction": max(
                executable.hbm_fraction for executable in process_executables
            ),
            "worst_compiler_hbm_risk": max(risks, key=_RISK_RANK.__getitem__),
            "stablehlo_collective_permute_count": permutation_counts.pop(),
            "stablehlo_all_gathers_absent_on_every_process": True,
        }

    run_id, profile, source_digest, config_digest = baseline.provenance_identity
    (
        _,
        jax_version,
        jaxlib_version,
        _,
        _,
        local_device_count,
        global_device_count,
        device_kinds,
        _,
    ) = baseline.runtime_identity
    (
        node_count,
        triangle_count,
        free_dof_count,
        _,
        layout_sha256,
        halo_link_count,
        halo_value_count,
    ) = baseline.problem_identity
    (
        model,
        frequency_hz,
        silicon_refractive_index,
        silica_refractive_index,
        core_width_m,
        core_height_m,
        cross_section_width_m,
        cross_section_height_m,
        shift_per_m2,
    ) = baseline.physics_identity
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
            "x64_enabled": False,
            "default_matmul_precision": "highest",
            "process_indexes": list(range(process_count)),
            "worker_indexes": sorted(record.worker_index for record in ordered),
            "process_count": process_count,
            "local_device_count": local_device_count,
            "global_device_count": global_device_count,
            "device_kinds": list(cast(tuple[str, ...], device_kinds)),
            "complex_scalar_contract": _COMPLEX_SCALAR_CONTRACT,
        },
        "physics": {
            "model": model,
            "frequency_hz": frequency_hz,
            "silicon_refractive_index": silicon_refractive_index,
            "silica_refractive_index": silica_refractive_index,
            "core_width_m": core_width_m,
            "core_height_m": core_height_m,
            "cross_section_width_m": cross_section_width_m,
            "cross_section_height_m": cross_section_height_m,
            "shift_per_m2": shift_per_m2,
        },
        "problem": {
            "node_count": node_count,
            "triangle_count": triangle_count,
            "free_mixed_dof_count": free_dof_count,
            "partition_count": partition_count,
            "layout_sha256": layout_sha256,
            "halo_link_count": halo_link_count,
            "halo_value_count": halo_value_count,
        },
        "addressability": {
            "combined_partition_addressability_counts": list(combined_addressability),
            "every_partition_addressable_once": True,
        },
        "checkpoint": {
            "mode": baseline.checkpoint_identity[0],
            "checkpoint_id": baseline.checkpoint_identity[1],
            "step": baseline.checkpoint_identity[2],
            "complete_process_fragment_count": process_count,
            "same_topology_only": True,
        },
        "numerics": {
            "maximum_action_relative_difference_across_processes": max(
                record.maximum_action_difference for record in ordered
            ),
            "maximum_vjp_relative_difference_across_processes": max(
                record.maximum_vjp_difference for record in ordered
            ),
            "maximum_host_c64_vs_c128_relative_difference_across_processes": max(
                record.maximum_host_precision_difference for record in ordered
            ),
            "action_tolerance": baseline.action_tolerance,
            "vjp_tolerance": baseline.vjp_tolerance,
            "host_precision_tolerance": baseline.host_precision_tolerance,
            "all_process_actions_finite": True,
            "all_process_vjps_finite": True,
        },
        "executables": executable_payload,
        "claim_scope": (
            "process-complete physical multi-host TPU action/VJP evidence for one bounded "
            "lossless Si/SiO2 float32/complex64 correctness witness with a bounded host "
            "complex128 arithmetic reference; ordinal critical-path timing is not a "
            "scaling result, compiler memory is not live HBM, and this is not eigensolve "
            "scaling, Elmer parity, FDTDX integration, or preemption recovery"
        ),
    }
    return TpuCollectiveProcessSetEvidence(payload=payload)
