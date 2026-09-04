"""Fail-closed process-set admission for distributed electrothermal TPU evidence.

The raw runner writes one record on every initialized JAX process.  This module admits a
scientific result only when those records describe one immutable input, one complete physical
TPU topology, exactly-once FEM partition addressability, matching distributed-array layouts, and
the same finite forward/adjoint result.  The aggregate intentionally omits private infrastructure
identifiers while retaining hashes of every raw process record.
"""

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

PROCESS_EVIDENCE_SCHEMA = "femx.jax.distributed_electrothermal.tpu_evidence/v1"
PROCESS_SET_EVIDENCE_SCHEMA = "femx.validation.tpu_distributed_electrothermal.process_set/v1"
WORKER_ENTRY_CLAIM_SCHEMA = "femx.jax.distributed_electrothermal.worker_entry_claim/v1"
PLAN_SCHEMA = "femx.jax.distributed_electrothermal/v1"
MESH_REPORT_SCHEMA = "femx.jax.collective.mesh_report/v1"
ARRAY_REPORT_SCHEMA = "femx.jax.collective.array_report/v1"
REPLICATED_ARRAY_REPORT_SCHEMA = "femx.jax.collective.replicated_array_report/v1"
TIMING_REPORT_SCHEMA = "femx.jax.collective.timing_report/v1"
MEMORY_REPORT_SCHEMA = "femx.jax.collective.memory_report/v1"

EXECUTABLE_NAMES = ("forward", "explicit_vjp", "native_reverse")
PARTITIONED_ARRAY_REPORT_NAMES = (
    "authority-potential",
    "authority-temperature",
    "input-basis-gradients",
    "input-cell-areas",
    "input-cell-local-dofs",
    "input-cell-mask",
    "input-current-cell-load-base",
    "input-current-cell-load-weights",
    "input-current-conductivity-base",
    "input-current-conductivity-weights",
    "input-current-dirichlet-base",
    "input-current-dirichlet-weights",
    "input-feedback-coefficient-base",
    "input-feedback-coefficient-weights",
    "input-feedback-reference-base",
    "input-feedback-reference-weights",
    "input-owner-mask",
    "input-thermal-cell-load-base",
    "input-thermal-cell-load-weights",
    "input-thermal-conductivity-base",
    "input-thermal-conductivity-weights",
    "input-thermal-dirichlet-base",
    "input-thermal-dirichlet-weights",
    "input-unit-stiffness",
    "temperature-cotangent",
)
REPLICATED_ARRAY_REPORT_NAMES = (
    "authority-current-gradient",
    "authority-feedback-gradient",
    "authority-thermal-gradient",
    "current-parameters",
    "feedback-parameters",
    "input-current-lower-bounds",
    "input-current-reference-base",
    "input-current-reference-weights",
    "input-current-upper-bounds",
    "input-feedback-lower-bounds",
    "input-feedback-upper-bounds",
    "input-thermal-lower-bounds",
    "input-thermal-reference-base",
    "input-thermal-reference-weights",
    "input-thermal-upper-bounds",
    "thermal-parameters",
)
REAL_SCALAR_CONTRACT: Mapping[str, object] = MappingProxyType(
    {
        "logical_dtype": "float32",
        "index_dtype": "int32",
        "mask_dtype": "bool",
        "controller_authority_dtype": "float64",
        "matmul_precision": "highest",
        "precision_fallback": False,
    }
)
TOLERANCES: Mapping[str, float] = MappingProxyType(
    {
        "potential_relative_difference": 5.0e-4,
        "temperature_relative_difference": 2.0e-4,
        "gradient_relative_difference": 5.0e-3,
        "native_explicit_gradient_relative_difference": 1.0e-3,
        "objective_relative_difference": 5.0e-4,
        "transfer_relative_error": 2.0e-5,
    }
)
COUPLED_ITERATION_POLICY: Mapping[str, object] = MappingProxyType(
    {
        "algorithm": "block_gauss_seidel",
        "max_iterations": 100,
        "minimum_iterations": 2,
        "relative_tolerance": 2.0e-5,
        "potential_absolute_tolerance_V": 1.0e-7,
        "temperature_absolute_tolerance_K": 1.0e-4,
        "potential_relaxation": 1.0,
        "temperature_relaxation": 0.5,
        "residual_tolerance": 1.0e-4,
    }
)
SCALAR_CG_POLICY: Mapping[str, object] = MappingProxyType(
    {
        "relative_tolerance": 2.0e-5,
        "absolute_tolerance": 0.0,
        "max_iterations": 1000,
        "admission_metric": "componentwise_normwise_backward_error",
        "backward_error_tolerance": 5.0e-7,
        "preconditioner": {
            "name": "stopped_positive_diagonal_jacobi",
            "minimum_relative_diagonal": 1.0e-14,
        },
    }
)
COUPLED_ADJOINT_POLICY: Mapping[str, object] = MappingProxyType(
    {
        "relative_tolerance": 5.0e-4,
        "absolute_tolerance": 1.0e-6,
        "restart": 20,
        "max_restarts": 60,
        "preconditioner": "stopped uncoupled current/heat right block inverse",
        "preconditioning_side": "right",
    }
)
REPLICATION_INTENT = "bounded plan scalars, parameter vectors, and float64-authority projections"
HOST_INPUT_REPLICATION = (
    "bounded complete plan file exists on every worker; only addressable partition slices are "
    "transferred for partition-leading device arrays"
)
RISK_RANK = {"safe": 0, "elevated": 1, "high": 2, "extreme": 3}
_DTYPE_BYTES = {"bool": 1, "float32": 4, "int32": 4}


def _fail(message: str) -> NoReturn:
    raise ValidationError(f"distributed electrothermal physical TPU evidence {message}")


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        _fail(f"requires {label} to be an object with string keys")
    return value


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
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
        _fail(f"requires {label} to be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (positive and result == 0.0):
        qualifier = "positive" if positive else "nonnegative"
        _fail(f"requires {label} to be a finite {qualifier} number")
    return result


def _boolean(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"requires {label} to be boolean")
    return value


def _sha256(value: object, *, label: str) -> str:
    result = _text(value, label=label)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        _fail(f"requires {label} to be a canonical lowercase SHA-256")
    return result


def _git_revision(value: object, *, label: str) -> str:
    result = _text(value, label=label)
    if len(result) != 40 or any(character not in "0123456789abcdef" for character in result):
        _fail(f"requires {label} to be a full lowercase Git revision")
    return result


def _keys(value: Mapping[str, object], expected: Sequence[str], *, label: str) -> None:
    if tuple(sorted(value)) != tuple(sorted(expected)):
        _fail(f"requires the exact {label} key set")


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
            "distributed electrothermal physical TPU record is not canonical JSON"
        ) from error


def _canonical_digest(record: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(record).encode("utf-8")).hexdigest()


def _exact_json_object(
    value: object,
    expected: Mapping[str, object],
    *,
    label: str,
) -> Mapping[str, object]:
    record = _mapping(value, label=label)
    if _canonical_json(record) != _canonical_json(dict(expected)):
        _fail(f"changes the locked {label}")
    return record


def _shape(value: object, *, label: str) -> tuple[int, ...]:
    result = tuple(
        _integer(item, label=f"{label}[]", positive=True) for item in _sequence(value, label=label)
    )
    if not result:
        _fail(f"requires {label} to be nonempty")
    return result


def _logical_bytes(shape: Sequence[int], dtype: str) -> int:
    if dtype not in _DTYPE_BYTES:
        _fail(f"uses unsupported distributed-array dtype {dtype!r}")
    result = _DTYPE_BYTES[dtype]
    for extent in shape:
        result *= extent
    return result


def _risk_for_fraction(fraction: float) -> str:
    if fraction < 0.70:
        return "safe"
    if fraction < 0.85:
        return "elevated"
    if fraction < 0.95:
        return "high"
    return "extreme"


@dataclass(frozen=True, slots=True)
class _Executable:
    lowering_seconds: float
    compilation_seconds: float
    warmup_seconds: float
    execution_seconds: tuple[float, ...]
    compiler_peak_bytes: int
    hbm_capacity_bytes: int
    hbm_fraction: float
    risk: str
    stablehlo_sha256: str
    collective_permute_count: int
    all_reduce_count: int

    @property
    def static_identity(self) -> tuple[object, ...]:
        return (
            self.compiler_peak_bytes,
            self.hbm_capacity_bytes,
            self.hbm_fraction,
            self.risk,
            self.stablehlo_sha256,
            self.collective_permute_count,
            self.all_reduce_count,
        )


@dataclass(frozen=True, slots=True)
class _Numerics:
    iterations: int
    current_linear_iterations: int
    heat_linear_iterations: int
    current_linear_backward_error: float
    heat_linear_backward_error: float
    current_linear_recursive_residual: float
    current_linear_recomputed_residual: float
    current_linear_relative_residual: float
    heat_linear_recursive_residual: float
    heat_linear_recomputed_residual: float
    heat_linear_relative_residual: float
    current_residual_error: float
    heat_residual_error: float
    adjoint_backward_error: float
    potential_relative_difference: float
    temperature_relative_difference: float
    objective: float
    authority_objective: float
    objective_relative_difference: float
    electrical_joule_power: float
    thermal_joule_load: float
    transfer_relative_error: float
    explicit_gradient_differences: tuple[float, float, float]
    native_authority_gradient_differences: tuple[float, float, float]
    native_explicit_gradient_differences: tuple[float, float, float]

    def canonical_data(self) -> dict[str, object]:
        return {
            "all_processes_forward_converged_and_finite": True,
            "all_processes_adjoint_converged_and_finite": True,
            "coupled_iterations": self.iterations,
            "current_linear_iterations": self.current_linear_iterations,
            "heat_linear_iterations": self.heat_linear_iterations,
            "current_linear_backward_error": self.current_linear_backward_error,
            "heat_linear_backward_error": self.heat_linear_backward_error,
            "current_linear_recursive_residual": self.current_linear_recursive_residual,
            "current_linear_recomputed_residual": self.current_linear_recomputed_residual,
            "current_linear_relative_residual": self.current_linear_relative_residual,
            "heat_linear_recursive_residual": self.heat_linear_recursive_residual,
            "heat_linear_recomputed_residual": self.heat_linear_recomputed_residual,
            "heat_linear_relative_residual": self.heat_linear_relative_residual,
            "current_coupled_residual_error": self.current_residual_error,
            "heat_coupled_residual_error": self.heat_residual_error,
            "coupled_adjoint_backward_error": self.adjoint_backward_error,
            "potential_relative_difference": self.potential_relative_difference,
            "temperature_relative_difference": self.temperature_relative_difference,
            "objective": self.objective,
            "authority_objective": self.authority_objective,
            "objective_relative_difference": self.objective_relative_difference,
            "electrical_joule_power_W_per_m": self.electrical_joule_power,
            "thermal_joule_load_W_per_m": self.thermal_joule_load,
            "transfer_relative_error": self.transfer_relative_error,
            "explicit_gradient_relative_differences": list(self.explicit_gradient_differences),
            "native_gradient_authority_relative_differences": list(
                self.native_authority_gradient_differences
            ),
            "native_gradient_explicit_relative_differences": list(
                self.native_explicit_gradient_differences
            ),
            "gradient_namespace_order": ["current", "thermal", "feedback"],
        }


@dataclass(frozen=True, slots=True)
class _Process:
    process_index: int
    worker_index: int
    digest: str
    provenance_identity: tuple[str, str, str, str, str]
    runtime_identity: tuple[object, ...]
    plan_identity: tuple[object, ...]
    assignment_identity: tuple[tuple[object, ...], ...]
    local_partition_mask: tuple[int, ...]
    partitioned_array_identity: tuple[tuple[object, ...], ...]
    replicated_array_identity: tuple[tuple[object, ...], ...]
    numerics: _Numerics
    executables: tuple[tuple[str, _Executable], ...]


def _timed_executable(value: object, *, name: str) -> _Executable:
    record = _mapping(value, label=f"executables.{name}")
    _keys(record, ("timing", "memory", "stablehlo"), label=f"executables.{name}")
    timing = _mapping(record["timing"], label=f"executables.{name}.timing")
    _keys(
        timing,
        (
            "schema_version",
            "lowering_seconds",
            "compilation_seconds",
            "warmup_seconds",
            "execution_seconds",
            "execution_min_seconds",
            "execution_median_seconds",
            "execution_max_seconds",
            "synchronization",
        ),
        label=f"executables.{name}.timing",
    )
    if timing.get("schema_version") != TIMING_REPORT_SCHEMA:
        _fail(f"has unsupported {name} timing schema")
    if timing.get("synchronization") != "every timed result blocked until ready":
        _fail(f"does not prove synchronized {name} timings")
    samples = tuple(
        _number(item, label=f"executables.{name}.timing.execution_seconds[]", positive=True)
        for item in _sequence(
            timing.get("execution_seconds"),
            label=f"executables.{name}.timing.execution_seconds",
        )
    )
    if len(samples) != 5:
        _fail(f"requires exactly five {name} execution samples")
    for key, expected in (
        ("execution_min_seconds", min(samples)),
        ("execution_median_seconds", statistics.median(samples)),
        ("execution_max_seconds", max(samples)),
    ):
        observed = _number(timing.get(key), label=f"executables.{name}.timing.{key}")
        if not math.isclose(observed, expected, rel_tol=1.0e-12, abs_tol=1.0e-15):
            _fail(f"has inconsistent {name} execution timing summary")

    memory = _mapping(record["memory"], label=f"executables.{name}.memory")
    _keys(
        memory,
        (
            "schema_version",
            "argument_bytes",
            "output_bytes",
            "alias_bytes",
            "temporary_bytes",
            "generated_code_bytes",
            "compiler_peak_bytes",
            "hbm_capacity_bytes_per_device",
            "hbm_fraction",
            "risk",
            "claim_scope",
        ),
        label=f"executables.{name}.memory",
    )
    if memory.get("schema_version") != MEMORY_REPORT_SCHEMA:
        _fail(f"has unsupported {name} compiler-memory schema")
    byte_values = {
        key: _integer(memory.get(key), label=f"executables.{name}.memory.{key}")
        for key in (
            "argument_bytes",
            "output_bytes",
            "alias_bytes",
            "temporary_bytes",
            "generated_code_bytes",
            "compiler_peak_bytes",
        )
    }
    peak = byte_values["compiler_peak_bytes"]
    expected_peak = (
        byte_values["argument_bytes"]
        + byte_values["output_bytes"]
        + byte_values["temporary_bytes"]
        - byte_values["alias_bytes"]
    )
    if peak != expected_peak:
        _fail(f"has inconsistent {name} compiler-memory byte accounting")
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
    if risk not in {"safe", "elevated"}:
        _fail(f"exceeds the admitted compiler-memory risk for {name}")
    if memory.get("claim_scope") != "compiler estimate; not live HBM usage":
        _fail(f"overstates {name} compiler memory as live HBM")

    stablehlo = _mapping(record["stablehlo"], label=f"executables.{name}.stablehlo")
    _keys(
        stablehlo,
        ("sha256", "contains_all_gather", "collective_permute_count", "all_reduce_count"),
        label=f"executables.{name}.stablehlo",
    )
    if _boolean(
        stablehlo.get("contains_all_gather"),
        label=f"executables.{name}.stablehlo.contains_all_gather",
    ):
        _fail(f"contains an all-gather in {name} StableHLO")
    return _Executable(
        lowering_seconds=_number(
            timing.get("lowering_seconds"),
            label=f"executables.{name}.timing.lowering_seconds",
            positive=True,
        ),
        compilation_seconds=_number(
            timing.get("compilation_seconds"),
            label=f"executables.{name}.timing.compilation_seconds",
            positive=True,
        ),
        warmup_seconds=_number(
            timing.get("warmup_seconds"),
            label=f"executables.{name}.timing.warmup_seconds",
            positive=True,
        ),
        execution_seconds=samples,
        compiler_peak_bytes=peak,
        hbm_capacity_bytes=capacity,
        hbm_fraction=fraction,
        risk=risk,
        stablehlo_sha256=_sha256(
            stablehlo.get("sha256"), label=f"executables.{name}.stablehlo.sha256"
        ),
        collective_permute_count=_integer(
            stablehlo.get("collective_permute_count"),
            label=f"executables.{name}.stablehlo.collective_permute_count",
            positive=True,
        ),
        all_reduce_count=_integer(
            stablehlo.get("all_reduce_count"),
            label=f"executables.{name}.stablehlo.all_reduce_count",
            positive=True,
        ),
    )


def _partitioned_array_report(
    value: object,
    *,
    name: str,
    process_index: int,
    process_count: int,
    local_device_count: int,
    global_device_count: int,
    partition_count: int,
    local_mask: tuple[int, ...],
    assignments: tuple[tuple[object, ...], ...],
) -> tuple[object, ...]:
    report = _mapping(value, label=f"partitioned_array_reports.{name}")
    _keys(
        report,
        (
            "schema_version",
            "name",
            "global_shape",
            "dtype",
            "partition_axis_name",
            "partition_count",
            "global_device_count",
            "process_index",
            "process_count",
            "global_logical_bytes",
            "addressable_logical_bytes",
            "replication_intent",
            "addressable_shards",
        ),
        label=f"partitioned_array_reports.{name}",
    )
    if report.get("schema_version") != ARRAY_REPORT_SCHEMA or report.get("name") != name:
        _fail(f"has unsupported or mismatched partitioned array report {name}")
    if report.get("partition_axis_name") != "partition":
        _fail(f"does not partition {name} on the FEM partition axis")
    for key, expected in (
        ("partition_count", partition_count),
        ("global_device_count", global_device_count),
        ("process_index", process_index),
        ("process_count", process_count),
    ):
        if _integer(report.get(key), label=f"partitioned_array_reports.{name}.{key}") != expected:
            _fail(f"has inconsistent {name} {key.replace('_', ' ')}")
    shape = _shape(
        report.get("global_shape"), label=f"partitioned_array_reports.{name}.global_shape"
    )
    if shape[0] != partition_count:
        _fail(f"has inconsistent leading partition extent for {name}")
    dtype = _text(report.get("dtype"), label=f"partitioned_array_reports.{name}.dtype")
    global_bytes = _integer(
        report.get("global_logical_bytes"),
        label=f"partitioned_array_reports.{name}.global_logical_bytes",
        positive=True,
    )
    if global_bytes != _logical_bytes(shape, dtype):
        _fail(f"has inconsistent global byte accounting for {name}")
    expected_dtype = (
        "int32"
        if name == "input-cell-local-dofs"
        else "bool"
        if name in {"input-cell-mask", "input-owner-mask"}
        else "float32"
    )
    if dtype != expected_dtype:
        _fail(f"has the wrong semantic dtype for {name}")
    if report.get("replication_intent") != "none; one leading FEM partition per device":
        _fail(f"does not prove non-replicated storage for {name}")

    expected_partitions = tuple(index for index, active in enumerate(local_mask) if active)
    shards = _sequence(
        report.get("addressable_shards"),
        label=f"partitioned_array_reports.{name}.addressable_shards",
    )
    if len(shards) != local_device_count:
        _fail(f"requires one addressable {name} shard per local TPU device")
    observed_partitions: list[int] = []
    observed_bytes = 0
    for raw_shard in shards:
        shard = _mapping(raw_shard, label=f"partitioned_array_reports.{name}.addressable_shards[]")
        _keys(
            shard,
            (
                "partition_index",
                "process_index",
                "device_id",
                "device_kind",
                "local_shape",
                "logical_bytes",
            ),
            label=f"partitioned_array_reports.{name}.addressable_shards[]",
        )
        partition = _integer(
            shard.get("partition_index"),
            label=f"partitioned_array_reports.{name}.addressable_shards[].partition_index",
        )
        if partition >= partition_count:
            _fail(f"has an out-of-range partition shard for {name}")
        if (
            _integer(
                shard.get("process_index"),
                label=f"partitioned_array_reports.{name}.addressable_shards[].process_index",
            )
            != process_index
        ):
            _fail(f"has a foreign-process shard for {name}")
        assignment = assignments[partition]
        if (
            _integer(
                shard.get("device_id"),
                label=f"partitioned_array_reports.{name}.addressable_shards[].device_id",
            )
            != assignment[2]
            or _text(
                shard.get("device_kind"),
                label=f"partitioned_array_reports.{name}.addressable_shards[].device_kind",
            )
            != assignment[4]
        ):
            _fail(f"has a shard device inconsistent with the Mesh assignment for {name}")
        local_shape = _shape(
            shard.get("local_shape"),
            label=f"partitioned_array_reports.{name}.addressable_shards[].local_shape",
        )
        if local_shape != (1, *shape[1:]):
            _fail(f"has a noncanonical local shard shape for {name}")
        logical_bytes = _integer(
            shard.get("logical_bytes"),
            label=f"partitioned_array_reports.{name}.addressable_shards[].logical_bytes",
            positive=True,
        )
        if logical_bytes != _logical_bytes(local_shape, dtype):
            _fail(f"has inconsistent local byte accounting for {name}")
        observed_partitions.append(partition)
        observed_bytes += logical_bytes
    if tuple(sorted(observed_partitions)) != expected_partitions or len(
        set(observed_partitions)
    ) != len(observed_partitions):
        _fail(f"has shards inconsistent with the process-local partition mask for {name}")
    if (
        _integer(
            report.get("addressable_logical_bytes"),
            label=f"partitioned_array_reports.{name}.addressable_logical_bytes",
            positive=True,
        )
        != observed_bytes
    ):
        _fail(f"has inconsistent addressable byte accounting for {name}")
    return (name, dtype, shape, global_bytes)


def _replicated_array_report(
    value: object,
    *,
    name: str,
    process_index: int,
    process_count: int,
    local_device_count: int,
    global_device_count: int,
) -> tuple[object, ...]:
    report = _mapping(value, label=f"replicated_array_reports.{name}")
    _keys(
        report,
        (
            "schema_version",
            "name",
            "global_shape",
            "dtype",
            "partition_spec",
            "global_device_count",
            "addressable_device_count",
            "process_index",
            "process_count",
            "logical_bytes_per_replica",
            "addressable_logical_bytes",
            "global_replica_logical_bytes",
            "replication_intent",
        ),
        label=f"replicated_array_reports.{name}",
    )
    if report.get("schema_version") != REPLICATED_ARRAY_REPORT_SCHEMA or report.get("name") != name:
        _fail(f"has unsupported or mismatched replicated array report {name}")
    if list(
        _sequence(
            report.get("partition_spec"), label=f"replicated_array_reports.{name}.partition_spec"
        )
    ):
        _fail(f"requires a fully replicated PartitionSpec for {name}")
    for key, expected in (
        ("global_device_count", global_device_count),
        ("addressable_device_count", local_device_count),
        ("process_index", process_index),
        ("process_count", process_count),
    ):
        if _integer(report.get(key), label=f"replicated_array_reports.{name}.{key}") != expected:
            _fail(f"has inconsistent replicated {name} {key.replace('_', ' ')}")
    shape = _shape(
        report.get("global_shape"), label=f"replicated_array_reports.{name}.global_shape"
    )
    dtype = _text(report.get("dtype"), label=f"replicated_array_reports.{name}.dtype")
    if dtype != "float32":
        _fail(f"requires float32 replicated values for {name}")
    per_replica = _integer(
        report.get("logical_bytes_per_replica"),
        label=f"replicated_array_reports.{name}.logical_bytes_per_replica",
        positive=True,
    )
    if per_replica != _logical_bytes(shape, dtype):
        _fail(f"has inconsistent per-replica byte accounting for {name}")
    if (
        _integer(
            report.get("addressable_logical_bytes"),
            label=f"replicated_array_reports.{name}.addressable_logical_bytes",
            positive=True,
        )
        != per_replica * local_device_count
    ):
        _fail(f"has inconsistent addressable replica byte accounting for {name}")
    if (
        _integer(
            report.get("global_replica_logical_bytes"),
            label=f"replicated_array_reports.{name}.global_replica_logical_bytes",
            positive=True,
        )
        != per_replica * global_device_count
    ):
        _fail(f"has inconsistent global replica byte accounting for {name}")
    if report.get("replication_intent") != REPLICATION_INTENT:
        _fail(f"does not preserve the bounded replication intent for {name}")
    return (name, dtype, shape, per_replica)


def _gradient_triplet(value: object, *, label: str) -> tuple[float, float, float]:
    result = tuple(_number(item, label=f"{label}[]") for item in _sequence(value, label=label))
    if len(result) != 3:
        _fail(f"requires {label} in current, thermal, feedback order")
    return result


def _numerics(value: object) -> _Numerics:
    record = _mapping(value, label="numerics")
    _keys(
        record,
        (
            "iterations",
            "forward_converged",
            "finite",
            "current_linear_iterations",
            "current_linear_recursive_residual",
            "current_linear_recomputed_residual",
            "current_linear_relative_residual",
            "current_linear_backward_error",
            "current_linear_converged",
            "current_linear_breakdown",
            "heat_linear_iterations",
            "heat_linear_recursive_residual",
            "heat_linear_recomputed_residual",
            "heat_linear_relative_residual",
            "heat_linear_backward_error",
            "heat_linear_converged",
            "heat_linear_breakdown",
            "current_residual_error",
            "heat_residual_error",
            "adjoint_converged",
            "adjoint_backward_error",
            "potential_relative_difference",
            "temperature_relative_difference",
            "objective",
            "authority_objective",
            "objective_relative_difference",
            "electrical_joule_power_W_per_m",
            "thermal_joule_load_W_per_m",
            "transfer_relative_error",
            "explicit_gradient_relative_differences",
            "native_gradient_authority_relative_differences",
            "native_gradient_explicit_relative_differences",
            "authority",
        ),
        label="numerics",
    )
    for key in ("forward_converged", "finite", "adjoint_converged"):
        if not _boolean(record.get(key), label=f"numerics.{key}"):
            _fail(f"does not admit finite converged {key.replace('_', ' ')}")
    for prefix in ("current", "heat"):
        if not _boolean(
            record.get(f"{prefix}_linear_converged"),
            label=f"numerics.{prefix}_linear_converged",
        ) or _boolean(
            record.get(f"{prefix}_linear_breakdown"),
            label=f"numerics.{prefix}_linear_breakdown",
        ):
            _fail(f"does not admit the final {prefix} scalar solve")
    iterations = _integer(record.get("iterations"), label="numerics.iterations", positive=True)
    if not (
        cast(int, COUPLED_ITERATION_POLICY["minimum_iterations"])
        <= iterations
        <= cast(int, COUPLED_ITERATION_POLICY["max_iterations"])
    ):
        _fail("reports coupled iterations outside the locked policy")
    current_iterations = _integer(
        record.get("current_linear_iterations"),
        label="numerics.current_linear_iterations",
        positive=True,
    )
    heat_iterations = _integer(
        record.get("heat_linear_iterations"),
        label="numerics.heat_linear_iterations",
        positive=True,
    )
    if max(current_iterations, heat_iterations) > cast(int, SCALAR_CG_POLICY["max_iterations"]):
        _fail("reports scalar iterations outside the locked CG policy")
    current_backward = _number(
        record.get("current_linear_backward_error"),
        label="numerics.current_linear_backward_error",
    )
    heat_backward = _number(
        record.get("heat_linear_backward_error"),
        label="numerics.heat_linear_backward_error",
    )
    if max(current_backward, heat_backward) > cast(
        float, SCALAR_CG_POLICY["backward_error_tolerance"]
    ):
        _fail("exceeds the locked scalar backward-error admission")
    current_residual = _number(
        record.get("current_residual_error"), label="numerics.current_residual_error"
    )
    heat_residual = _number(record.get("heat_residual_error"), label="numerics.heat_residual_error")
    if max(current_residual, heat_residual) > cast(
        float, COUPLED_ITERATION_POLICY["residual_tolerance"]
    ):
        _fail("exceeds the locked coupled forward-residual admission")
    adjoint_backward = _number(
        record.get("adjoint_backward_error"), label="numerics.adjoint_backward_error"
    )
    if adjoint_backward > cast(float, COUPLED_ADJOINT_POLICY["relative_tolerance"]):
        _fail("exceeds the locked coupled-transpose backward-error admission")

    potential_difference = _number(
        record.get("potential_relative_difference"),
        label="numerics.potential_relative_difference",
    )
    temperature_difference = _number(
        record.get("temperature_relative_difference"),
        label="numerics.temperature_relative_difference",
    )
    objective_difference = _number(
        record.get("objective_relative_difference"),
        label="numerics.objective_relative_difference",
    )
    transfer_error = _number(
        record.get("transfer_relative_error"), label="numerics.transfer_relative_error"
    )
    for observed, key in (
        (potential_difference, "potential_relative_difference"),
        (temperature_difference, "temperature_relative_difference"),
        (objective_difference, "objective_relative_difference"),
        (transfer_error, "transfer_relative_error"),
    ):
        if observed > TOLERANCES[key]:
            _fail(f"exceeds the locked {key.replace('_', ' ')}")
    explicit = _gradient_triplet(
        record.get("explicit_gradient_relative_differences"),
        label="numerics.explicit_gradient_relative_differences",
    )
    native_authority = _gradient_triplet(
        record.get("native_gradient_authority_relative_differences"),
        label="numerics.native_gradient_authority_relative_differences",
    )
    native_explicit = _gradient_triplet(
        record.get("native_gradient_explicit_relative_differences"),
        label="numerics.native_gradient_explicit_relative_differences",
    )
    if max((*explicit, *native_authority)) > TOLERANCES["gradient_relative_difference"]:
        _fail("exceeds the locked dense-authority gradient admission")
    if max(native_explicit) > TOLERANCES["native_explicit_gradient_relative_difference"]:
        _fail("exceeds the locked native-versus-explicit gradient admission")

    objective = _number(record.get("objective"), label="numerics.objective", positive=True)
    authority_objective = _number(
        record.get("authority_objective"),
        label="numerics.authority_objective",
        positive=True,
    )
    recomputed_objective_difference = abs(objective - authority_objective) / abs(
        authority_objective
    )
    if not math.isclose(
        objective_difference,
        recomputed_objective_difference,
        rel_tol=1.0e-9,
        abs_tol=1.0e-12,
    ):
        _fail("has an inconsistent objective relative difference")
    electrical_power = _number(
        record.get("electrical_joule_power_W_per_m"),
        label="numerics.electrical_joule_power_W_per_m",
        positive=True,
    )
    thermal_power = _number(
        record.get("thermal_joule_load_W_per_m"),
        label="numerics.thermal_joule_load_W_per_m",
        positive=True,
    )
    recomputed_transfer = abs(electrical_power - thermal_power) / max(
        electrical_power, thermal_power
    )
    if not math.isclose(transfer_error, recomputed_transfer, rel_tol=1.0e-9, abs_tol=1.0e-12):
        _fail("has an inconsistent Joule-transfer relative error")
    authority = _text(record.get("authority"), label="numerics.authority")
    if "dense float64 same-discretization forward and coupled residual VJP" not in authority:
        _fail("does not identify the immutable dense float64 authority")
    return _Numerics(
        iterations=iterations,
        current_linear_iterations=current_iterations,
        heat_linear_iterations=heat_iterations,
        current_linear_backward_error=current_backward,
        heat_linear_backward_error=heat_backward,
        current_linear_recursive_residual=_number(
            record.get("current_linear_recursive_residual"),
            label="numerics.current_linear_recursive_residual",
        ),
        current_linear_recomputed_residual=_number(
            record.get("current_linear_recomputed_residual"),
            label="numerics.current_linear_recomputed_residual",
        ),
        current_linear_relative_residual=_number(
            record.get("current_linear_relative_residual"),
            label="numerics.current_linear_relative_residual",
        ),
        heat_linear_recursive_residual=_number(
            record.get("heat_linear_recursive_residual"),
            label="numerics.heat_linear_recursive_residual",
        ),
        heat_linear_recomputed_residual=_number(
            record.get("heat_linear_recomputed_residual"),
            label="numerics.heat_linear_recomputed_residual",
        ),
        heat_linear_relative_residual=_number(
            record.get("heat_linear_relative_residual"),
            label="numerics.heat_linear_relative_residual",
        ),
        current_residual_error=current_residual,
        heat_residual_error=heat_residual,
        adjoint_backward_error=adjoint_backward,
        potential_relative_difference=potential_difference,
        temperature_relative_difference=temperature_difference,
        objective=objective,
        authority_objective=authority_objective,
        objective_relative_difference=objective_difference,
        electrical_joule_power=electrical_power,
        thermal_joule_load=thermal_power,
        transfer_relative_error=transfer_error,
        explicit_gradient_differences=explicit,
        native_authority_gradient_differences=native_authority,
        native_explicit_gradient_differences=native_explicit,
    )


def _process(record: Mapping[str, object]) -> _Process:
    _keys(
        record,
        (
            "schema_version",
            "status",
            "provenance",
            "runtime",
            "launch_claim",
            "plan",
            "mesh_report",
            "addressability",
            "partitioned_array_reports",
            "replicated_array_reports",
            "tolerances",
            "policies",
            "numerics",
            "executables",
            "claim_scope",
        ),
        label="process record",
    )
    if record.get("schema_version") != PROCESS_EVIDENCE_SCHEMA:
        _fail("has an unsupported process-record schema")
    if record.get("status") != "passed":
        _fail("contains a process record that did not pass")

    provenance = _mapping(record.get("provenance"), label="provenance")
    _keys(
        provenance,
        ("run_id", "profile", "source_commit", "source_digest", "config_digest"),
        label="provenance",
    )
    run_id = _text(provenance.get("run_id"), label="provenance.run_id")
    profile = _text(provenance.get("profile"), label="provenance.profile")
    source_commit = _git_revision(provenance.get("source_commit"), label="provenance.source_commit")
    source_digest = _sha256(provenance.get("source_digest"), label="provenance.source_digest")
    config_digest = _sha256(provenance.get("config_digest"), label="provenance.config_digest")

    runtime = _mapping(record.get("runtime"), label="runtime")
    _keys(
        runtime,
        (
            "backend",
            "jax_version",
            "jaxlib_version",
            "x64_enabled",
            "default_matmul_precision",
            "process_index",
            "process_count",
            "local_device_count",
            "global_device_count",
            "device_kinds",
            "real_scalar_contract",
        ),
        label="runtime",
    )
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
    if (
        process_count < 2
        or process_index >= process_count
        or global_device_count != process_count * local_device_count
    ):
        _fail("has inconsistent physical multi-process TPU counts")
    if (
        runtime.get("backend") != "tpu"
        or runtime.get("x64_enabled") is not False
        or runtime.get("default_matmul_precision") != "highest"
    ):
        _fail("requires the physical TPU float32 highest-precision runtime")
    device_kinds = tuple(
        _text(item, label="runtime.device_kinds[]")
        for item in _sequence(runtime.get("device_kinds"), label="runtime.device_kinds")
    )
    if not device_kinds or tuple(sorted(set(device_kinds))) != device_kinds:
        _fail("requires canonical nonempty TPU device kinds")
    _exact_json_object(
        runtime.get("real_scalar_contract"), REAL_SCALAR_CONTRACT, label="real scalar contract"
    )
    runtime_identity = (
        _text(runtime.get("jax_version"), label="runtime.jax_version"),
        _text(runtime.get("jaxlib_version"), label="runtime.jaxlib_version"),
        process_count,
        local_device_count,
        global_device_count,
        device_kinds,
    )

    launch = _mapping(record.get("launch_claim"), label="launch_claim")
    _keys(
        launch,
        (
            "schema_version",
            "run_id",
            "worker_index",
            "process_index",
            "source_sha256",
            "config_sha256",
            "scope",
        ),
        label="launch_claim",
    )
    worker_index = _integer(launch.get("worker_index"), label="launch_claim.worker_index")
    if (
        launch.get("schema_version") != WORKER_ENTRY_CLAIM_SCHEMA
        or worker_index >= process_count
        or launch.get("run_id") != run_id
        or _integer(launch.get("process_index"), label="launch_claim.process_index")
        != process_index
        or launch.get("source_sha256") != source_digest
        or launch.get("config_sha256") != config_digest
    ):
        _fail("has a worker-entry claim inconsistent with provenance or runtime")
    if "worker-local coupled electrothermal entry fence" not in _text(
        launch.get("scope"), label="launch_claim.scope"
    ):
        _fail("does not preserve the worker-entry fence scope")

    plan = _mapping(record.get("plan"), label="plan")
    _keys(
        plan,
        (
            "schema_version",
            "sha256",
            "arrays_sha256",
            "layout_sha256",
            "partition_count",
            "node_count",
            "triangle_count",
            "free_dof_count",
            "host_input_replication",
        ),
        label="plan",
    )
    if plan.get("schema_version") != PLAN_SCHEMA:
        _fail("has an unsupported electrothermal plan schema")
    plan_sha256 = _sha256(plan.get("sha256"), label="plan.sha256")
    arrays_sha256 = _sha256(plan.get("arrays_sha256"), label="plan.arrays_sha256")
    layout_sha256 = _sha256(plan.get("layout_sha256"), label="plan.layout_sha256")
    partition_count = _integer(
        plan.get("partition_count"), label="plan.partition_count", positive=True
    )
    node_count = _integer(plan.get("node_count"), label="plan.node_count", positive=True)
    triangle_count = _integer(
        plan.get("triangle_count"), label="plan.triangle_count", positive=True
    )
    free_dof_count = _integer(
        plan.get("free_dof_count"), label="plan.free_dof_count", positive=True
    )
    if (
        partition_count != global_device_count
        or triangle_count < partition_count
        or free_dof_count >= node_count
        or plan.get("host_input_replication") != HOST_INPUT_REPLICATION
    ):
        _fail("has an inconsistent bounded electrothermal plan")
    plan_identity = (
        plan_sha256,
        arrays_sha256,
        layout_sha256,
        partition_count,
        node_count,
        triangle_count,
        free_dof_count,
    )

    mesh = _mapping(record.get("mesh_report"), label="mesh_report")
    _keys(
        mesh,
        (
            "schema_version",
            "axis_name",
            "partition_count",
            "global_device_count",
            "addressable_device_count",
            "process_count",
            "is_multi_process",
            "layout_sha256",
            "assignments",
        ),
        label="mesh_report",
    )
    if (
        mesh.get("schema_version") != MESH_REPORT_SCHEMA
        or mesh.get("axis_name") != "partition"
        or mesh.get("is_multi_process") is not True
        or mesh.get("layout_sha256") != layout_sha256
    ):
        _fail("has an inconsistent collective Mesh identity")
    for key, expected in (
        ("partition_count", partition_count),
        ("global_device_count", global_device_count),
        ("addressable_device_count", local_device_count),
        ("process_count", process_count),
    ):
        if _integer(mesh.get(key), label=f"mesh_report.{key}", positive=True) != expected:
            _fail(f"has inconsistent Mesh {key.replace('_', ' ')}")

    addressability = _mapping(record.get("addressability"), label="addressability")
    _keys(
        addressability,
        (
            "process_local_partition_mask",
            "partition_addressability_counts",
            "every_partition_addressable_once",
        ),
        label="addressability",
    )
    local_mask = tuple(
        _integer(item, label="addressability.process_local_partition_mask[]")
        for item in _sequence(
            addressability.get("process_local_partition_mask"),
            label="addressability.process_local_partition_mask",
        )
    )
    counts = tuple(
        _integer(item, label="addressability.partition_addressability_counts[]")
        for item in _sequence(
            addressability.get("partition_addressability_counts"),
            label="addressability.partition_addressability_counts",
        )
    )
    if (
        len(local_mask) != partition_count
        or any(item not in (0, 1) for item in local_mask)
        or sum(local_mask) != local_device_count
    ):
        _fail("has a noncanonical process-local partition mask")
    if counts != (1,) * partition_count or not _boolean(
        addressability.get("every_partition_addressable_once"),
        label="addressability.every_partition_addressable_once",
    ):
        _fail("does not report every global partition addressable exactly once")

    raw_assignments = _sequence(mesh.get("assignments"), label="mesh_report.assignments")
    if len(raw_assignments) != partition_count:
        _fail("requires one Mesh assignment per partition")
    assignments: list[tuple[object, ...]] = []
    for expected_partition, raw_assignment in enumerate(raw_assignments):
        assignment = _mapping(raw_assignment, label="mesh_report.assignments[]")
        _keys(
            assignment,
            (
                "partition_index",
                "process_index",
                "device_id",
                "platform",
                "device_kind",
                "addressable",
            ),
            label="mesh_report.assignments[]",
        )
        partition = _integer(
            assignment.get("partition_index"),
            label="mesh_report.assignments[].partition_index",
        )
        assigned_process = _integer(
            assignment.get("process_index"), label="mesh_report.assignments[].process_index"
        )
        device_id = _integer(
            assignment.get("device_id"), label="mesh_report.assignments[].device_id"
        )
        platform = _text(assignment.get("platform"), label="mesh_report.assignments[].platform")
        device_kind = _text(
            assignment.get("device_kind"), label="mesh_report.assignments[].device_kind"
        )
        is_addressable = _boolean(
            assignment.get("addressable"), label="mesh_report.assignments[].addressable"
        )
        if partition != expected_partition or assigned_process >= process_count:
            _fail("has a noncanonical Mesh assignment order or process")
        if platform != "tpu" or device_kind not in device_kinds:
            _fail("has a Mesh assignment inconsistent with the TPU runtime")
        if is_addressable != bool(local_mask[partition]) or is_addressable != (
            assigned_process == process_index
        ):
            _fail("has Mesh addressability inconsistent with the process-local mask")
        assignments.append((partition, assigned_process, device_id, platform, device_kind))
    assignment_identity = tuple(assignments)
    if len({(item[1], item[2]) for item in assignment_identity}) != partition_count:
        _fail("requires one unique TPU device per FEM partition")

    partitioned = _mapping(
        record.get("partitioned_array_reports"), label="partitioned_array_reports"
    )
    if tuple(sorted(partitioned)) != tuple(sorted(PARTITIONED_ARRAY_REPORT_NAMES)):
        _fail("requires the exact partitioned electrothermal array-report set")
    partitioned_identity = tuple(
        _partitioned_array_report(
            partitioned[name],
            name=name,
            process_index=process_index,
            process_count=process_count,
            local_device_count=local_device_count,
            global_device_count=global_device_count,
            partition_count=partition_count,
            local_mask=local_mask,
            assignments=assignment_identity,
        )
        for name in PARTITIONED_ARRAY_REPORT_NAMES
    )
    replicated = _mapping(record.get("replicated_array_reports"), label="replicated_array_reports")
    if tuple(sorted(replicated)) != tuple(sorted(REPLICATED_ARRAY_REPORT_NAMES)):
        _fail("requires the exact replicated electrothermal array-report set")
    replicated_identity = tuple(
        _replicated_array_report(
            replicated[name],
            name=name,
            process_index=process_index,
            process_count=process_count,
            local_device_count=local_device_count,
            global_device_count=global_device_count,
        )
        for name in REPLICATED_ARRAY_REPORT_NAMES
    )

    _exact_json_object(record.get("tolerances"), TOLERANCES, label="electrothermal tolerances")
    policies = _mapping(record.get("policies"), label="policies")
    _keys(
        policies,
        ("coupled_iteration", "scalar_cg", "coupled_adjoint"),
        label="policies",
    )
    _exact_json_object(
        policies.get("coupled_iteration"),
        COUPLED_ITERATION_POLICY,
        label="coupled iteration policy",
    )
    _exact_json_object(policies.get("scalar_cg"), SCALAR_CG_POLICY, label="scalar CG policy")
    _exact_json_object(
        policies.get("coupled_adjoint"),
        COUPLED_ADJOINT_POLICY,
        label="coupled adjoint policy",
    )
    numerics = _numerics(record.get("numerics"))
    raw_executables = _mapping(record.get("executables"), label="executables")
    if tuple(sorted(raw_executables)) != tuple(sorted(EXECUTABLE_NAMES)):
        _fail("requires the exact forward, explicit-VJP, and native-reverse executable set")
    executables = tuple(
        (name, _timed_executable(raw_executables[name], name=name)) for name in EXECUTABLE_NAMES
    )
    claim_scope = _text(record.get("claim_scope"), label="claim_scope")
    for phrase in (
        "bounded process-local physical multi-host TPU",
        "not Elmer re-execution",
        "scaling",
        "live HBM",
        "3D production FEM",
        "foundry",
        "FDTDX",
        "recovery evidence",
    ):
        if phrase not in claim_scope:
            _fail(f"omits the bounded claim-scope phrase {phrase!r}")
    return _Process(
        process_index=process_index,
        worker_index=worker_index,
        digest=_canonical_digest(record),
        provenance_identity=(run_id, profile, source_commit, source_digest, config_digest),
        runtime_identity=runtime_identity,
        plan_identity=plan_identity,
        assignment_identity=assignment_identity,
        local_partition_mask=local_mask,
        partitioned_array_identity=partitioned_identity,
        replicated_array_identity=replicated_identity,
        numerics=numerics,
        executables=executables,
    )


@dataclass(frozen=True, slots=True)
class TpuDistributedElectrothermalProcessSetEvidence:
    """Canonical public-safe projection of one complete physical TPU process set."""

    payload: Mapping[str, object]

    def canonical_data(self) -> dict[str, object]:
        """Return a detached JSON-compatible aggregate."""

        return cast(dict[str, object], json.loads(self.canonical_json()))

    def canonical_json(self) -> str:
        """Return canonical UTF-8 JSON text."""

        return _canonical_json(self.payload)

    def digest(self) -> str:
        """Hash the canonical aggregate payload."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def aggregate_tpu_distributed_electrothermal_process_evidence(
    records: object,
) -> TpuDistributedElectrothermalProcessSetEvidence:
    """Admit exactly one immutable record per initialized JAX process."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence) or not records:
        _fail("requires a nonempty sequence of process records")
    parsed = tuple(
        _process(_mapping(record, label="process_records[]"))
        for record in cast(Sequence[object], records)
    )
    ordered = tuple(sorted(parsed, key=lambda item: item.process_index))
    baseline = ordered[0]
    (
        jax_version,
        jaxlib_version,
        process_count_raw,
        local_device_count_raw,
        global_device_count_raw,
        device_kinds,
    ) = baseline.runtime_identity
    process_count = cast(int, process_count_raw)
    local_device_count = cast(int, local_device_count_raw)
    global_device_count = cast(int, global_device_count_raw)
    if len(ordered) != process_count or tuple(item.process_index for item in ordered) != tuple(
        range(process_count)
    ):
        _fail("requires one unique record for every declared JAX process index")
    if tuple(sorted(item.worker_index for item in ordered)) != tuple(range(process_count)):
        _fail("requires one unique worker-entry claim for every TPU worker")
    for item in ordered[1:]:
        if item.provenance_identity != baseline.provenance_identity:
            _fail("mixes records from different deployed source or configuration inputs")
        if item.runtime_identity != baseline.runtime_identity:
            _fail("mixes records from different physical TPU runtimes")
        if item.plan_identity != baseline.plan_identity:
            _fail("mixes records from different electrothermal plans")
        if item.assignment_identity != baseline.assignment_identity:
            _fail("mixes inconsistent global Mesh assignments")
        if item.partitioned_array_identity != baseline.partitioned_array_identity:
            _fail("mixes inconsistent partitioned array layouts")
        if item.replicated_array_identity != baseline.replicated_array_identity:
            _fail("mixes inconsistent replicated array layouts")
        if item.numerics != baseline.numerics:
            _fail("mixes inconsistent reduced forward or adjoint numerics")
        for (name, executable), (baseline_name, baseline_executable) in zip(
            item.executables, baseline.executables, strict=True
        ):
            if (
                name != baseline_name
                or executable.static_identity != baseline_executable.static_identity
            ):
                _fail("mixes inconsistent compiled executable identities")

    combined_mask = tuple(
        sum(item.local_partition_mask[index] for item in ordered)
        for index in range(global_device_count)
    )
    if combined_mask != (1,) * global_device_count:  # pragma: no cover - parser invariant
        _fail("does not cover every global FEM partition exactly once across processes")

    _, _, source_commit, source_digest, config_digest = baseline.provenance_identity
    (
        plan_sha256,
        arrays_sha256,
        layout_sha256,
        partition_count,
        node_count,
        triangle_count,
        free_dof_count,
    ) = baseline.plan_identity
    executable_payload: dict[str, object] = {}
    for name in EXECUTABLE_NAMES:
        values = [dict(item.executables)[name] for item in ordered]
        ordinal_critical_path = [
            max(value.execution_seconds[index] for value in values) for index in range(5)
        ]
        worst_fraction = max(value.hbm_fraction for value in values)
        executable_payload[name] = {
            "process_lowering_seconds": [value.lowering_seconds for value in values],
            "process_compilation_seconds": [value.compilation_seconds for value in values],
            "process_warmup_seconds": [value.warmup_seconds for value in values],
            "process_execution_median_seconds": [
                statistics.median(value.execution_seconds) for value in values
            ],
            "execution_ordinal_critical_path_seconds": ordinal_critical_path,
            "execution_ordinal_critical_path_summary_seconds": {
                "min": min(ordinal_critical_path),
                "median": statistics.median(ordinal_critical_path),
                "max": max(ordinal_critical_path),
            },
            "maximum_compiler_peak_bytes": max(value.compiler_peak_bytes for value in values),
            "hbm_capacity_bytes_per_device": values[0].hbm_capacity_bytes,
            "maximum_compiler_hbm_fraction": worst_fraction,
            "worst_compiler_hbm_risk": max(
                (value.risk for value in values), key=RISK_RANK.__getitem__
            ),
            "stablehlo_sha256": values[0].stablehlo_sha256,
            "stablehlo_collective_permute_count": values[0].collective_permute_count,
            "stablehlo_all_reduce_count": values[0].all_reduce_count,
            "stablehlo_all_gathers_absent_on_every_process": True,
            "sample_alignment": (
                "ordinal maximum across synchronized process-local samples; not a scaling result"
            ),
            "memory_scope": "compiler estimate; not live HBM usage",
        }

    payload: dict[str, object] = {
        "schema_version": PROCESS_SET_EVIDENCE_SCHEMA,
        "status": "passed",
        "provenance": {
            "source_commit": source_commit,
            "source_digest": source_digest,
            "config_digest": config_digest,
            "plan_sha256": plan_sha256,
            "arrays_sha256": arrays_sha256,
            "process_records": [
                {"process_index": item.process_index, "sha256": item.digest} for item in ordered
            ],
            "raw_artifacts": "retained_outside_git",
            "redacted_fields": [
                "cloud project",
                "hostname",
                "machine address",
                "profile",
                "resource name",
                "run ID",
                "worker mapping",
                "zone",
            ],
        },
        "runtime": {
            "backend": "tpu",
            "jax_version": jax_version,
            "jaxlib_version": jaxlib_version,
            "x64_enabled": False,
            "default_matmul_precision": "highest",
            "process_indexes": list(range(process_count)),
            "process_count": process_count,
            "local_device_count": local_device_count,
            "global_device_count": global_device_count,
            "device_kinds": list(cast(tuple[str, ...], device_kinds)),
            "real_scalar_contract": dict(REAL_SCALAR_CONTRACT),
        },
        "problem": {
            "model": (
                "bounded 2D same-mesh H1/P1 current-to-Joule-to-heat self-consistent residual"
            ),
            "node_count": node_count,
            "triangle_count": triangle_count,
            "free_dof_count": free_dof_count,
            "partition_count": partition_count,
            "layout_sha256": layout_sha256,
        },
        "addressability": {
            "combined_partition_addressability_counts": list(combined_mask),
            "every_partition_addressable_once": True,
        },
        "array_storage": {
            "partitioned": [
                {
                    "name": cast(str, name),
                    "dtype": cast(str, dtype),
                    "global_shape": list(cast(tuple[int, ...], shape)),
                }
                for name, dtype, shape, _ in baseline.partitioned_array_identity
            ],
            "replicated": [
                {
                    "name": cast(str, name),
                    "dtype": cast(str, dtype),
                    "global_shape": list(cast(tuple[int, ...], shape)),
                }
                for name, dtype, shape, _ in baseline.replicated_array_identity
            ],
            "partitioned_arrays_are_device_local": True,
            "replication_intent": REPLICATION_INTENT,
        },
        "policies": {
            "coupled_iteration": dict(COUPLED_ITERATION_POLICY),
            "scalar_cg": json.loads(_canonical_json(dict(SCALAR_CG_POLICY))),
            "coupled_adjoint": dict(COUPLED_ADJOINT_POLICY),
        },
        "tolerances": dict(TOLERANCES),
        "numerics": baseline.numerics.canonical_data(),
        "executables": executable_payload,
        "claim_scope": (
            "process-complete physical multi-host TPU float32 execution of one bounded 2D "
            "same-mesh H1/P1 current-to-Joule-to-heat forward solve, right-preconditioned "
            "coupled residual adjoint, and native JAX reverse path against an immutable dense "
            "float64 same-discretization authority; ordinal timings are not a scaling result, "
            "compiler memory is not live HBM, and this is not fresh Elmer execution, FDTDX "
            "composition, 3D production FEM, foundry or measured-device validation, or "
            "preemption-recovery evidence"
        ),
    }
    return TpuDistributedElectrothermalProcessSetEvidence(payload=payload)


__all__ = [
    "COUPLED_ADJOINT_POLICY",
    "COUPLED_ITERATION_POLICY",
    "EXECUTABLE_NAMES",
    "PARTITIONED_ARRAY_REPORT_NAMES",
    "PROCESS_EVIDENCE_SCHEMA",
    "PROCESS_SET_EVIDENCE_SCHEMA",
    "REAL_SCALAR_CONTRACT",
    "REPLICATED_ARRAY_REPORT_NAMES",
    "SCALAR_CG_POLICY",
    "TOLERANCES",
    "TpuDistributedElectrothermalProcessSetEvidence",
    "aggregate_tpu_distributed_electrothermal_process_evidence",
]
