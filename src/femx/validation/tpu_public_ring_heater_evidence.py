"""Process-complete admission for the fine public-ring physical TPU forward witness."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import NoReturn, cast

from femx.core.errors import ValidationError

PROCESS_EVIDENCE_SCHEMA = "femx.validation.public_ring_heater.tpu_forward_process/v1"
PROCESS_SET_EVIDENCE_SCHEMA = "femx.validation.public_ring_heater.tpu_forward_process_set/v1"
WORKER_ENTRY_CLAIM_SCHEMA = "femx.public-ring-heater.tpu_forward_entry_claim/v1"

EXPECTED_PROCESS_COUNT = 8
EXPECTED_LOCAL_DEVICE_COUNT = 4
EXPECTED_GLOBAL_DEVICE_COUNT = 32
EXPECTED_NODE_COUNT = 521_442
EXPECTED_TETRAHEDRON_COUNT = 3_179_879
EXPECTED_CONDUCTOR_TETRAHEDRON_COUNT = 134_331
EXPECTED_FINE_MSH_SHA256 = "c484d4be5f52a59b93ba0904f79bef98d7dea0aceb8976e269b49cdc739d0a69"

SCALAR_CG_POLICY = MappingProxyType(
    {
        "relative_tolerance": 5.0e-7,
        "absolute_tolerance": 0.0,
        "max_iterations": 5_000,
        "backward_error_tolerance": 2.0e-5,
        "preconditioner": MappingProxyType(
            {
                "name": "stopped_positive_diagonal_jacobi",
                "minimum_relative_diagonal": 1.0e-15,
            }
        ),
    }
)
CONSERVATION_TOLERANCES = MappingProxyType(
    {
        "charge_balance_relative_error": 2.0e-4,
        "electrical_energy_relative_error": 2.0e-4,
        "joule_transfer_relative_error": 2.0e-6,
        "thermal_balance_relative_error": 2.0e-4,
    }
)
PARITY_TOLERANCES = MappingProxyType(
    {
        "potential_relative_l2_difference": 1.0e-3,
        "potential_normalized_max_difference": 2.0e-3,
        "temperature_rise_relative_l2_difference": 1.0e-3,
        "temperature_rise_normalized_max_difference": 2.0e-3,
        "target_current_relative_error": 1.0e-3,
        "target_power_relative_error": 1.0e-3,
        "maximum_temperature_relative_difference": 1.0e-3,
        "silicon_ring_mean_temperature_rise_relative_difference": 1.0e-3,
        "tin_heater_mean_temperature_rise_relative_difference": 1.0e-3,
    }
)
REAL_SCALAR_CONTRACT = MappingProxyType(
    {
        "state_dtype": "float32",
        "index_dtype": "int32",
        "jax_x64_enabled": False,
        "default_matmul_precision": "highest",
        "backend": "tpu",
        "fallback_allowed": False,
    }
)

_PACKED_INPUT_NAMES = (
    "current_cell_local_dofs",
    "current_owner_mask",
    "current_cell_mask",
    "current_conduction_stiffness",
    "current_basis_gradients",
    "current_cell_volumes",
    "current_conductivity",
    "current_cell_load",
    "current_cell_dirichlet_base",
    "current_cell_dirichlet_scale",
    "current_to_thermal_slots",
    "thermal_cell_local_dofs",
    "thermal_owner_mask",
    "thermal_cell_mask",
    "thermal_conduction_stiffness",
    "thermal_robin_matrix",
    "thermal_cell_volumes",
    "thermal_nonrobin_load",
    "thermal_robin_ambient_load",
    "thermal_cell_dirichlet_shifted",
)
EXPECTED_PARTITIONED_ARRAY_NAMES = tuple(
    [
        *(f"input-{name}" for name in _PACKED_INPUT_NAMES),
        "authority-potential",
        "authority-temperature-rise",
        "silicon-ring-cell-mask",
        "tin-heater-cell-mask",
    ]
)


def _fail(message: str) -> NoReturn:
    raise ValidationError(f"public-ring physical TPU process-set evidence {message}")


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        _fail(f"requires {label} to be an object with string keys")
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, (tuple, list)):
        _fail(f"requires {label} to be an array")
    return cast(Sequence[object], value)


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
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite positive" if positive else "finite"
        _fail(f"requires {label} to be a {qualifier} number")
    return result


def _boolean(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"requires {label} to be boolean")
    return value


def _hex(value: object, *, label: str, length: int) -> str:
    result = _text(value, label=label)
    if len(result) != length or any(character not in "0123456789abcdef" for character in result):
        _fail(f"requires {label} to be lowercase hexadecimal with length {length}")
    return result


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValidationError(
            "public-ring physical TPU process-set evidence must be finite JSON"
        ) from error


def _range(values: Sequence[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
    }


def _expected_policies() -> dict[str, object]:
    return {
        "scalar_cg": {
            **dict(SCALAR_CG_POLICY),
            "preconditioner": dict(cast(Mapping[str, object], SCALAR_CG_POLICY["preconditioner"])),
        },
        "conservation_tolerances": dict(CONSERVATION_TOLERANCES),
        "parity_tolerances": dict(PARITY_TOLERANCES),
        "target_voltage_source": (
            "fixed by the CPU float64 unit-voltage linear calibration; no repeated TPU "
            "calibration solve"
        ),
    }


@dataclass(frozen=True, slots=True)
class _ProcessRecord:
    process_index: int
    worker_index: int
    partitions: tuple[int, ...]
    common_identity: tuple[str, ...]
    digest: str
    lowering_seconds: float
    compilation_seconds: float
    execution_seconds: float
    compiler_peak_bytes: int
    hbm_capacity_bytes: int
    hbm_fraction: float


def _mesh_partitions(value: object, process_index: int) -> tuple[int, ...]:
    report = _mapping(value, label="mesh report")
    expected_counts = {
        "partition_count": EXPECTED_GLOBAL_DEVICE_COUNT,
        "global_device_count": EXPECTED_GLOBAL_DEVICE_COUNT,
        "addressable_device_count": EXPECTED_LOCAL_DEVICE_COUNT,
        "process_count": EXPECTED_PROCESS_COUNT,
    }
    if (
        report.get("schema_version") != "femx.jax.collective.mesh_report/v1"
        or report.get("axis_name") != "partition"
        or report.get("is_multi_process") is not True
        or any(
            _integer(report.get(name), label=f"mesh {name}", positive=True) != expected
            for name, expected in expected_counts.items()
        )
    ):
        _fail("does not use the exact 8-process, 32-device collective mesh")
    _hex(report.get("layout_sha256"), label="mesh layout SHA-256", length=64)
    assignments = _sequence(report.get("assignments"), label="mesh assignments")
    if len(assignments) != EXPECTED_GLOBAL_DEVICE_COUNT:
        _fail("must retain one assignment per global partition")
    local = []
    devices = set()
    for expected_partition, raw in enumerate(assignments):
        assignment = _mapping(raw, label="mesh assignment")
        partition = _integer(assignment.get("partition_index"), label="assigned partition")
        assigned_process = _integer(assignment.get("process_index"), label="assigned process")
        device_id = _integer(assignment.get("device_id"), label="assigned device")
        addressable = _boolean(assignment.get("addressable"), label="addressable flag")
        if (
            partition != expected_partition
            or assigned_process >= EXPECTED_PROCESS_COUNT
            or assignment.get("platform") != "tpu"
            or "TPU" not in _text(assignment.get("device_kind"), label="device kind")
            or addressable != (assigned_process == process_index)
        ):
            _fail("contains an inconsistent partition/device/process assignment")
        devices.add((assigned_process, device_id))
        if addressable:
            local.append(partition)
    if len(devices) != EXPECTED_GLOBAL_DEVICE_COUNT or len(local) != EXPECTED_LOCAL_DEVICE_COUNT:
        _fail("does not identify 32 unique devices and four local partitions")
    return tuple(local)


def _array_partitions(value: object, name: str, process_index: int) -> tuple[int, ...]:
    report = _mapping(value, label=f"partitioned array {name}")
    shape = _sequence(report.get("global_shape"), label=f"{name} global shape")
    if (
        report.get("schema_version") != "femx.jax.collective.array_report/v1"
        or report.get("name") != name
        or report.get("partition_axis_name") != "partition"
        or _integer(shape[0], label=f"{name} leading extent", positive=True)
        != EXPECTED_GLOBAL_DEVICE_COUNT
        or _integer(report.get("process_index"), label=f"{name} process") != process_index
        or _integer(report.get("process_count"), label=f"{name} process count", positive=True)
        != EXPECTED_PROCESS_COUNT
        or _integer(report.get("global_device_count"), label=f"{name} devices", positive=True)
        != EXPECTED_GLOBAL_DEVICE_COUNT
        or report.get("dtype") not in {"bool", "int32", "float32"}
        or report.get("replication_intent") != "none; one leading FEM partition per device"
    ):
        _fail(f"contains an invalid partitioned array report for {name}")
    return tuple(
        _integer(
            _mapping(shard, label=f"{name} shard").get("partition_index"),
            label=f"{name} shard partition",
        )
        for shard in _sequence(report.get("addressable_shards"), label=f"{name} shards")
    )


def _output_partitions(value: object, name: str) -> tuple[int, ...]:
    result = []
    for raw in _sequence(value, label=f"{name} output shards"):
        shard = _mapping(raw, label=f"{name} output shard")
        shape = _sequence(shard.get("shape"), label=f"{name} output shape")
        if (
            len(shape) != 2
            or _integer(shape[0], label=f"{name} output leading extent", positive=True) != 1
            or shard.get("dtype") != "float32"
            or shard.get("finite") is not True
        ):
            _fail(f"contains an invalid {name} output shard")
        _hex(shard.get("sha256"), label=f"{name} output SHA-256", length=64)
        result.append(_integer(shard.get("partition_index"), label=f"{name} output partition"))
    return tuple(result)


def _numerics(value: object) -> Mapping[str, object]:
    result = _mapping(value, label="numerics")
    for name in ("all_finite", "numerically_admitted", "current_converged", "thermal_converged"):
        if _boolean(result.get(name), label=name) is not True:
            _fail(f"requires passing {name}")
    for name in ("current_breakdown", "thermal_breakdown"):
        if _boolean(result.get(name), label=name) is not False:
            _fail(f"rejects {name}")
    for name in ("current_iterations", "thermal_iterations"):
        if _integer(result.get(name), label=name) > cast(int, SCALAR_CG_POLICY["max_iterations"]):
            _fail(f"exceeds the admitted {name}")
    signed_finite = (
        "current_recursive_residual",
        "thermal_recursive_residual",
        "current_recomputed_residual",
        "thermal_recomputed_residual",
        "current_relative_residual",
        "thermal_relative_residual",
        "electrical_variational_power_W",
        "convection_outward_power_W",
        "dirichlet_outward_power_W",
    )
    positive_finite = (
        "electrical_joule_power_W",
        "thermal_joule_load_W",
        "thermal_input_power_W",
        "maximum_temperature_K",
        "silicon_ring_mean_temperature_K",
        "tin_heater_mean_temperature_K",
        "silicon_ring_volume_m3",
        "tin_heater_volume_m3",
        "inferred_current_A",
    )
    for name in signed_finite:
        _number(result.get(name), label=name)
    for name in positive_finite:
        _number(result.get(name), label=name, positive=True)
    if _number(result.get("minimum_temperature_K"), label="minimum temperature") < 299.999:
        _fail("violates the passive ambient lower bound")
    for name in ("current_backward_error", "thermal_backward_error"):
        if _number(result.get(name), label=name) > cast(
            float, SCALAR_CG_POLICY["backward_error_tolerance"]
        ):
            _fail(f"exceeds the admitted {name}")
    for tolerances in (CONSERVATION_TOLERANCES, PARITY_TOLERANCES):
        for name, maximum in tolerances.items():
            if _number(result.get(name), label=name) > maximum:
                _fail(f"exceeds the admitted {name}")
    return result


def _process(record: Mapping[str, object]) -> _ProcessRecord:
    if record.get("schema_version") != PROCESS_EVIDENCE_SCHEMA or record.get("status") != "passed":
        _fail("contains an unsupported or non-passing process record")
    provenance = _mapping(record.get("provenance"), label="provenance")
    runtime = _mapping(record.get("runtime"), label="runtime")
    claim = _mapping(record.get("launch_claim"), label="launch claim")
    artifact = _mapping(record.get("artifact"), label="artifact")
    model = _mapping(record.get("model"), label="model")
    addressability = _mapping(record.get("addressability"), label="addressability")
    reports = _mapping(record.get("partitioned_array_reports"), label="array reports")
    outputs = _mapping(record.get("output_shards"), label="outputs")
    policies = _mapping(record.get("policies"), label="policies")
    execution = _mapping(record.get("executable"), label="executable")

    process_index = _integer(runtime.get("process_index"), label="process index")
    exact_runtime = (
        runtime.get("backend") == "tpu"
        and runtime.get("x64_enabled") is False
        and runtime.get("default_matmul_precision") == "highest"
        and runtime.get("process_count") == EXPECTED_PROCESS_COUNT
        and runtime.get("local_device_count") == EXPECTED_LOCAL_DEVICE_COUNT
        and runtime.get("global_device_count") == EXPECTED_GLOBAL_DEVICE_COUNT
        and dict(_mapping(runtime.get("real_scalar_contract"), label="scalar contract"))
        == dict(REAL_SCALAR_CONTRACT)
    )
    if process_index >= EXPECTED_PROCESS_COUNT or not exact_runtime:
        _fail("requires the exact physical TPU float32 runtime contract")
    versions = (
        _text(runtime.get("jax_version"), label="JAX version"),
        _text(runtime.get("jaxlib_version"), label="jaxlib version"),
        tuple(
            _text(item, label="device kind")
            for item in _sequence(runtime.get("device_kinds"), label="device kinds")
        ),
    )
    if not versions[2] or any("TPU" not in kind for kind in versions[2]):
        _fail("does not identify physical TPU devices")

    run_id = _text(provenance.get("run_id"), label="run ID")
    source_digest = _hex(provenance.get("source_digest"), label="source digest", length=64)
    source_commit = _hex(provenance.get("source_commit"), label="source commit", length=40)
    config_digest = _hex(provenance.get("config_digest"), label="config digest", length=64)
    if provenance.get("profile") != "v4-od-32":
        _fail("was not produced by the declared v4-od-32 profile")
    worker_index = _integer(claim.get("worker_index"), label="worker index")
    if (
        claim.get("schema_version") != WORKER_ENTRY_CLAIM_SCHEMA
        or claim.get("run_id") != run_id
        or claim.get("process_index") != process_index
        or worker_index >= EXPECTED_PROCESS_COUNT
        or claim.get("source_sha256") != source_digest
        or claim.get("config_sha256") != config_digest
    ):
        _fail("contains an inconsistent immutable worker-entry claim")

    artifact_identity = (
        _text(artifact.get("schema_version"), label="artifact schema"),
        _hex(artifact.get("logical_sha256"), label="artifact digest", length=64),
        _hex(artifact.get("runtime_plan_sha256"), label="runtime-plan digest", length=64),
        _hex(artifact.get("source_plan_sha256"), label="source-plan digest", length=64),
        _hex(artifact.get("source_msh_sha256"), label="source MSH digest", length=64),
        _hex(artifact.get("partition_owner_sha256"), label="owner digest", length=64),
        _integer(artifact.get("total_array_file_bytes"), label="artifact bytes", positive=True),
    )
    if artifact_identity[4] != EXPECTED_FINE_MSH_SHA256:
        _fail("does not bind the admitted fine public-ring MSH")
    exact_model = (
        model.get("node_count") == EXPECTED_NODE_COUNT
        and model.get("tetrahedron_count") == EXPECTED_TETRAHEDRON_COUNT
        and model.get("conductor_tetrahedron_count") == EXPECTED_CONDUCTOR_TETRAHEDRON_COUNT
        and model.get("dimension") == 3
        and model.get("element") == "first-order Tet4 H1"
    )
    if not exact_model:
        _fail("does not identify the exact fine 3D Tet4 model")
    for name in ("target_voltage_V", "target_current_A", "authority_predicted_power_W"):
        _number(model.get(name), label=name, positive=True)

    partitions = _mesh_partitions(record.get("mesh_report"), process_index)
    local_mask = tuple(
        _integer(item, label="local partition mask")
        for item in _sequence(addressability.get("process_local_partition_mask"), label="mask")
    )
    counts = tuple(
        _integer(item, label="addressability count")
        for item in _sequence(
            addressability.get("partition_addressability_counts"), label="addressability counts"
        )
    )
    if (
        len(local_mask) != EXPECTED_GLOBAL_DEVICE_COUNT
        or tuple(index for index, active in enumerate(local_mask) if active == 1) != partitions
        or any(active not in (0, 1) for active in local_mask)
        or counts != (1,) * EXPECTED_GLOBAL_DEVICE_COUNT
        or addressability.get("every_partition_addressable_once") is not True
    ):
        _fail("does not establish exact process-set partition addressability")
    if set(reports) != set(EXPECTED_PARTITIONED_ARRAY_NAMES):
        _fail("does not contain the exact partitioned input set")
    if any(
        _array_partitions(reports[name], name, process_index) != partitions
        for name in EXPECTED_PARTITIONED_ARRAY_NAMES
    ):
        _fail("transfers a nonlocal partitioned input")

    replicated = _mapping(record.get("replicated_parameter_report"), label="replicated controls")
    if (
        replicated.get("schema_version") != "femx.jax.collective.replicated_array_report/v1"
        or replicated.get("name") != "electrothermal-controls"
        or replicated.get("global_shape") != [3]
        or replicated.get("dtype") != "float32"
        or replicated.get("partition_spec") != []
        or replicated.get("process_index") != process_index
        or replicated.get("process_count") != EXPECTED_PROCESS_COUNT
        or replicated.get("global_device_count") != EXPECTED_GLOBAL_DEVICE_COUNT
        or replicated.get("addressable_device_count") != EXPECTED_LOCAL_DEVICE_COUNT
    ):
        _fail("does not use the exact bounded replicated control contract")
    if (
        _output_partitions(outputs.get("potential"), "potential") != partitions
        or _output_partitions(outputs.get("temperature_rise"), "temperature") != partitions
    ):
        _fail("does not retain exactly the local output shards")
    if _canonical_json(policies) != _canonical_json(_expected_policies()):
        _fail("does not use the committed solver and admission policies")
    numerics = _numerics(record.get("numerics"))

    timing = _mapping(execution.get("timing"), label="timing")
    memory = _mapping(execution.get("compiler_memory"), label="compiler memory")
    hlo = _mapping(execution.get("stablehlo"), label="StableHLO")
    if (
        timing.get("schema_version") != "femx.public-ring-heater.tpu_forward_timing/v1"
        or timing.get("execution_count") != 1
        or timing.get("benchmark_claimed") is not False
        or execution.get("hlo_admitted") is not True
        or execution.get("memory_admitted") is not True
        or hlo.get("contains_all_gather") is not False
        or hlo.get("contains_f64") is not False
        or _integer(hlo.get("collective_permute_count"), label="ppermute count", positive=True) <= 0
        or _integer(hlo.get("all_reduce_count"), label="all-reduce count", positive=True) <= 0
    ):
        _fail("does not admit the one-shot compiler or StableHLO contract")
    _hex(hlo.get("sha256"), label="StableHLO digest", length=64)
    lowering = _number(timing.get("lowering_seconds"), label="lowering time")
    compilation = _number(timing.get("compilation_seconds"), label="compilation time")
    execution_seconds = _number(timing.get("execution_seconds"), label="execution time")
    compiler_peak = _integer(
        memory.get("compiler_peak_bytes"), label="compiler peak", positive=True
    )
    hbm_capacity = _integer(
        memory.get("hbm_capacity_bytes_per_device"), label="HBM capacity", positive=True
    )
    hbm_fraction = _number(memory.get("hbm_fraction"), label="HBM fraction")
    if (
        hbm_fraction >= 0.85
        or memory.get("risk") not in {"safe", "elevated"}
        or memory.get("claim_scope") != "compiler estimate; not live HBM usage"
    ):
        _fail("exceeds or overstates the compiler-memory estimate")
    scope = _text(record.get("claim_scope"), label="claim scope")
    if any(
        term not in scope for term in ("not fresh Elmer", "FDTDX", "inverse design", "live HBM")
    ):
        _fail("does not retain the required scientific scope exclusions")

    common = (
        _canonical_json((run_id, "v4-od-32", source_digest, source_commit, config_digest)),
        _canonical_json(versions),
        _canonical_json(artifact_identity),
        _canonical_json(model),
        _canonical_json(policies),
        _canonical_json(numerics),
        _canonical_json(hlo),
    )
    return _ProcessRecord(
        process_index,
        worker_index,
        partitions,
        common,
        hashlib.sha256(_canonical_json(record).encode()).hexdigest(),
        lowering,
        compilation,
        execution_seconds,
        compiler_peak,
        hbm_capacity,
        hbm_fraction,
    )


@dataclass(frozen=True, slots=True)
class TpuPublicRingHeaterProcessSetEvidence:
    """Canonical admitted view of all eight immutable TPU process records."""

    _encoded: str

    def __init__(self, payload: Mapping[str, object]) -> None:
        if payload.get("schema_version") != PROCESS_SET_EVIDENCE_SCHEMA:
            _fail("aggregate uses an unsupported schema")
        if payload.get("status") != "passed":
            _fail("aggregate must be passing")
        object.__setattr__(self, "_encoded", _canonical_json(payload))

    def canonical_data(self) -> dict[str, object]:
        """Return detached JSON-compatible aggregate data."""

        return cast(dict[str, object], json.loads(self._encoded))

    def canonical_json(self) -> str:
        """Return deterministic compact JSON."""

        return self._encoded

    def digest(self) -> str:
        """Return the logical aggregate SHA-256."""

        return hashlib.sha256(self._encoded.encode()).hexdigest()


def aggregate_tpu_public_ring_heater_process_evidence(
    records: Sequence[Mapping[str, object]],
) -> TpuPublicRingHeaterProcessSetEvidence:
    """Require one mutually consistent record from every TPU v4-64 process."""

    if not records:
        _fail("requires process records")
    parsed = tuple(
        sorted(
            (_process(_mapping(record, label="process record")) for record in records),
            key=lambda item: item.process_index,
        )
    )
    if len(parsed) != EXPECTED_PROCESS_COUNT or tuple(
        item.process_index for item in parsed
    ) != tuple(range(EXPECTED_PROCESS_COUNT)):
        _fail("requires exactly one record for each process index")
    if tuple(sorted(item.worker_index for item in parsed)) != tuple(range(EXPECTED_PROCESS_COUNT)):
        _fail("requires exactly one immutable claim per TPU worker")
    baseline = parsed[0]
    if any(item.common_identity != baseline.common_identity for item in parsed[1:]):
        _fail("contains inconsistent per-process provenance, policy, or numerics")
    combined = tuple(index for item in parsed for index in item.partitions)
    if tuple(sorted(combined)) != tuple(range(EXPECTED_GLOBAL_DEVICE_COUNT)):
        _fail("does not cover every input and output partition exactly once")

    provenance = cast(list[object], json.loads(baseline.common_identity[0]))
    versions = cast(list[object], json.loads(baseline.common_identity[1]))
    artifact = cast(list[object], json.loads(baseline.common_identity[2]))
    model = cast(dict[str, object], json.loads(baseline.common_identity[3]))
    policies = cast(dict[str, object], json.loads(baseline.common_identity[4]))
    numerics = cast(dict[str, object], json.loads(baseline.common_identity[5]))
    stablehlo = cast(dict[str, object], json.loads(baseline.common_identity[6]))
    payload: dict[str, object] = {
        "schema_version": PROCESS_SET_EVIDENCE_SCHEMA,
        "status": "passed",
        "provenance": {
            "run_id": provenance[0],
            "profile": provenance[1],
            "source_digest": provenance[2],
            "source_commit": provenance[3],
            "config_digest": provenance[4],
            "process_records": [
                {
                    "process_index": item.process_index,
                    "worker_index": item.worker_index,
                    "sha256": item.digest,
                }
                for item in parsed
            ],
        },
        "runtime": {
            "backend": "tpu",
            "jax_version": versions[0],
            "jaxlib_version": versions[1],
            "device_kinds": versions[2],
            "x64_enabled": False,
            "default_matmul_precision": "highest",
            "process_count": EXPECTED_PROCESS_COUNT,
            "local_device_count": EXPECTED_LOCAL_DEVICE_COUNT,
            "global_device_count": EXPECTED_GLOBAL_DEVICE_COUNT,
            "real_scalar_contract": dict(REAL_SCALAR_CONTRACT),
        },
        "artifact": {
            "schema_version": artifact[0],
            "logical_sha256": artifact[1],
            "runtime_plan_sha256": artifact[2],
            "source_plan_sha256": artifact[3],
            "source_msh_sha256": artifact[4],
            "partition_owner_sha256": artifact[5],
            "total_array_file_bytes": artifact[6],
        },
        "model": model,
        "partitioning": {
            "partition_count": EXPECTED_GLOBAL_DEVICE_COUNT,
            "process_local_partitions": [
                {"process_index": item.process_index, "partitions": list(item.partitions)}
                for item in parsed
            ],
            "every_input_partition_addressable_once": True,
            "every_potential_partition_retained_once": True,
            "every_temperature_partition_retained_once": True,
        },
        "policies": policies,
        "numerics": numerics,
        "execution": {
            "execution_count_per_process": 1,
            "benchmark_claimed": False,
            "lowering_seconds_across_processes": _range(
                tuple(item.lowering_seconds for item in parsed)
            ),
            "compilation_seconds_across_processes": _range(
                tuple(item.compilation_seconds for item in parsed)
            ),
            "execution_seconds_across_processes": _range(
                tuple(item.execution_seconds for item in parsed)
            ),
            "maximum_compiler_peak_bytes": max(item.compiler_peak_bytes for item in parsed),
            "minimum_hbm_capacity_bytes_per_device": min(
                item.hbm_capacity_bytes for item in parsed
            ),
            "maximum_compiler_hbm_fraction": max(item.hbm_fraction for item in parsed),
            "compiler_memory_scope": "compiler estimate; not live HBM usage",
            "stablehlo": stablehlo,
        },
        "claim_scope": (
            "process-complete physical eight-process, 32-device TPU v4 float32 forward evidence "
            "for the source-pinned fine public 3D ring current/Joule/heat model against one CPU "
            "float64 same-mesh authority; not fresh Elmer execution, formal mesh convergence, "
            "FDTDX response, inverse design, performance scaling, live HBM, preemption recovery, "
            "foundry calibration, or fabricated-device validation"
        ),
    }
    return TpuPublicRingHeaterProcessSetEvidence(payload)


__all__ = [
    "CONSERVATION_TOLERANCES",
    "EXPECTED_CONDUCTOR_TETRAHEDRON_COUNT",
    "EXPECTED_FINE_MSH_SHA256",
    "EXPECTED_GLOBAL_DEVICE_COUNT",
    "EXPECTED_LOCAL_DEVICE_COUNT",
    "EXPECTED_NODE_COUNT",
    "EXPECTED_PARTITIONED_ARRAY_NAMES",
    "EXPECTED_PROCESS_COUNT",
    "EXPECTED_TETRAHEDRON_COUNT",
    "PARITY_TOLERANCES",
    "PROCESS_EVIDENCE_SCHEMA",
    "PROCESS_SET_EVIDENCE_SCHEMA",
    "REAL_SCALAR_CONTRACT",
    "SCALAR_CG_POLICY",
    "WORKER_ENTRY_CLAIM_SCHEMA",
    "TpuPublicRingHeaterProcessSetEvidence",
    "aggregate_tpu_public_ring_heater_process_evidence",
]
