"""Process-set admission for the physical distributed FEM-to-FDTDX TPU graph.

The input artifact is generated on an x64 controller.  Every TPU worker independently reloads
that artifact, executes the same float32/complex64 graph, and writes one raw process record.  This
module admits a claim only when all eight records describe one immutable 32-device execution.
Private run, profile, worker, and device identifiers are intentionally omitted from the aggregate.
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

PROCESS_EVIDENCE_SCHEMA = "femx.fdtdx.distributed_thermo_optic.tpu_process/v1"
PROCESS_SET_EVIDENCE_SCHEMA = "femx.validation.fdtdx_distributed_thermo_optic.tpu_process_set/v1"
WORKER_ENTRY_CLAIM_SCHEMA = "femx.fdtdx.distributed_thermo_optic.worker_entry_claim/v1"
CRITICAL_ARRAY_REPORT_SCHEMA = "femx.fdtdx.distributed_thermo_optic.array_report/v1"

EXPECTED_PROCESS_COUNT = 8
EXPECTED_LOCAL_DEVICE_COUNT = 4
EXPECTED_GLOBAL_DEVICE_COUNT = 32
EXPECTED_DEVICE_KINDS = ("TPU v4",)
EXPECTED_GRID_SHAPE = (96, 4, 8)
EXPECTED_DEVICE_SHAPE = (32, 2, 4)
GRID_SPACING_M = 62.5e-9
FINITE_DIFFERENCE_STEPS = (1.0e-1, 5.0e-2)
EXECUTABLE_NAMES = ("reference_phasor", "forward", "explicit_vjp", "native_reverse")
REPLICATION_INTENT = "bounded plan scalars and parameter vectors"
MAXIMUM_REPLICATED_LOGICAL_BYTES_PER_REPLICA = 4096

FDTDX_PACKAGE_VERSION = "0.6.2"
FDTDX_SOURCE_REVISION = "0c05c4784b2be83b42d9b46ab089265981ba157f"
FDTDX_SOURCE_DIGEST = "29bed9483c4c2b57fd2f495fdb47534edf6b244206679e34b2de41ec39aaa9fa"
FDTDX_MODULE_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "__init__.py": "fcf000b7955c97e7fbe1ccd5901c1f5ba47a5bfd86f0fce3d2dc8be1bfe131cf",
        "core/jax/sharding.py": (
            "a6e07ac439c1c1b48958380812406f090844a1b4924a3b3a9b0a7f49eca8a9c3"
        ),
        "fdtd/fdtd.py": "7c654097d43d5062afbef0cf8c479ba2a7db523b64683693fa4e24bc5070e4e0",
        "fdtd/initialization.py": (
            "2b7d56d47789f38c73b96fe7a078521e1146a45e98753af6c5e536ea8f9225a1"
        ),
        "fdtd/wrapper.py": ("97d562e0a33eeeccd6ce42d12cf8dc29f1e9fd071e561b5b009ac5da8ece7384"),
    }
)

SCALAR_CONTRACT: Mapping[str, object] = MappingProxyType(
    {
        "real_dtype": "float32",
        "complex_dtype": "complex64",
        "index_dtype": "int32",
        "x64_enabled": False,
        "matmul_precision": "highest",
        "precision_fallback": False,
    }
)
TOLERANCES: Mapping[str, float] = MappingProxyType(
    {
        "potential_relative_difference": 5.0e-4,
        "temperature_relative_difference": 2.0e-4,
        "cell_temperature_relative_difference": 2.0e-4,
        "parameter_relative_difference": 2.0e-4,
        "material_relative_difference": 2.0e-4,
        "objective_explicit_relative_difference": 1.0e-3,
        "native_explicit_gradient_relative_difference": 2.0e-3,
        "finite_difference_relative_error": 2.0e-2,
        "current_residual_error": 1.0e-4,
        "heat_residual_error": 1.0e-4,
        "linear_backward_error": 5.0e-7,
        "adjoint_backward_error": 5.0e-4,
        "transfer_relative_error": 2.0e-5,
        "maximum_compiler_hbm_fraction": 0.85,
        "runtime_coordinate_max_ulp_error": 8.0,
        "runtime_coordinate_max_grid_fraction_error": 4.0e-6,
    }
)

PARTITIONED_ARRAY_REPORT_NAMES = (
    "authority-cell-temperature",
    "authority-potential",
    "authority-temperature",
    "authority-thermo-optic-parameter",
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
    "transfer-receive-active",
    "transfer-receive-target-local-indices",
    "transfer-send-active",
    "transfer-send-barycentric-weights",
    "transfer-send-source-cell-slots",
)
REPLICATED_ARRAY_REPORT_NAMES = (
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
CRITICAL_ARRAY_SPECS: Mapping[str, tuple[tuple[int, ...], str, tuple[object, ...]]] = (
    MappingProxyType(
        {
            "applied-inverse-permittivity": (
                (1, *EXPECTED_GRID_SHAPE),
                "float32",
                (None, "shard", None, None),
            ),
            "thermo-optic-parameter": (
                EXPECTED_DEVICE_SHAPE,
                "float32",
                ("shard", None, None),
            ),
        }
    )
)
_DTYPE_BYTES = {"bool": 1, "float32": 4, "int32": 4}
_RISK_RANK = {"safe": 0, "elevated": 1, "high": 2, "extreme": 3}


def _fail(message: str) -> NoReturn:
    raise ValidationError(f"distributed FEM-to-FDTDX physical TPU evidence {message}")


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
        _fail(f"requires {label} to be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (positive and result == 0.0):
        qualifier = "positive" if positive else "nonnegative"
        _fail(f"requires {label} to be a finite {qualifier} number")
    return result


def _signed_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"requires {label} to be a finite number")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"requires {label} to be a finite number")
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
        raise ValidationError("physical TPU evidence is not canonical JSON") from error


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


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


def _exact_json(value: object, expected: object, *, label: str) -> None:
    if _canonical_json(value) != _canonical_json(expected):
        _fail(f"changes the locked {label}")


@dataclass(frozen=True, slots=True)
class _Executable:
    static_identity: tuple[object, ...]
    lowering_seconds: float
    compilation_seconds: float
    warmup_seconds: float
    execution_seconds: tuple[float, ...]
    compiler_peak_bytes: int
    hbm_fraction: float
    risk: str


@dataclass(frozen=True, slots=True)
class _Process:
    process_index: int
    worker_index: int
    digest: str
    provenance_identity: tuple[object, ...]
    runtime_identity: tuple[object, ...]
    input_identity: tuple[object, ...]
    plan_identity: tuple[object, ...]
    mesh_identity: tuple[object, ...]
    local_partition_mask: tuple[int, ...]
    partitioned_identity: tuple[object, ...]
    replicated_identity: tuple[object, ...]
    critical_identity: tuple[object, ...]
    coordinate_identity: tuple[
        tuple[float, ...],
        tuple[float, ...],
        tuple[int, ...],
        tuple[bool, ...],
        tuple[bool, ...],
    ]
    numerics: Mapping[str, object]
    executables: tuple[tuple[str, _Executable], ...]


def _partitioned_report(
    value: object,
    *,
    expected_name: str,
    process_index: int,
) -> tuple[tuple[object, ...], tuple[int, ...]]:
    report = _mapping(value, label=f"partitioned_array_reports.{expected_name}")
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
        label=f"{expected_name} partitioned-array report",
    )
    if (
        report.get("schema_version") != "femx.jax.collective.array_report/v1"
        or report.get("name") != expected_name
        or report.get("partition_axis_name") != "shard"
        or report.get("replication_intent") != "none; one leading FEM partition per device"
    ):
        _fail(f"has an invalid {expected_name} partitioned-array contract")
    shape = _shape(report.get("global_shape"), label=f"{expected_name}.global_shape")
    dtype = _text(report.get("dtype"), label=f"{expected_name}.dtype")
    if shape[0] != EXPECTED_GLOBAL_DEVICE_COUNT:
        _fail(f"requires {expected_name} to have one leading partition per device")
    if (
        _integer(report.get("partition_count"), label=f"{expected_name}.partition_count")
        != EXPECTED_GLOBAL_DEVICE_COUNT
        or _integer(report.get("global_device_count"), label=f"{expected_name}.devices")
        != EXPECTED_GLOBAL_DEVICE_COUNT
        or _integer(report.get("process_index"), label=f"{expected_name}.process_index")
        != process_index
        or _integer(report.get("process_count"), label=f"{expected_name}.process_count")
        != EXPECTED_PROCESS_COUNT
        or _integer(report.get("global_logical_bytes"), label=f"{expected_name}.bytes")
        != _logical_bytes(shape, dtype)
    ):
        _fail(f"has inconsistent {expected_name} partitioned-array metadata")
    shards = _sequence(report.get("addressable_shards"), label=f"{expected_name}.shards")
    if len(shards) != EXPECTED_LOCAL_DEVICE_COUNT:
        _fail(f"requires {expected_name} on every local device")
    mask = [0] * EXPECTED_GLOBAL_DEVICE_COUNT
    local_bytes = 0
    for raw in shards:
        shard = _mapping(raw, label=f"{expected_name}.shards[]")
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
            label=f"{expected_name} shard",
        )
        partition = _integer(shard.get("partition_index"), label="partition index")
        if partition >= EXPECTED_GLOBAL_DEVICE_COUNT or mask[partition]:
            _fail(f"repeats or exceeds an {expected_name} partition")
        if _integer(shard.get("process_index"), label="shard process index") != process_index:
            _fail(f"assigns an {expected_name} shard to the wrong process")
        _integer(shard.get("device_id"), label="device id")
        if _text(shard.get("device_kind"), label="device kind") not in EXPECTED_DEVICE_KINDS:
            _fail(f"places {expected_name} on a non-v4 TPU device")
        local_shape = _shape(shard.get("local_shape"), label="local shape")
        if local_shape != (1, *shape[1:]):
            _fail(f"has an invalid {expected_name} local shard shape")
        shard_bytes = _integer(shard.get("logical_bytes"), label="shard bytes", positive=True)
        if shard_bytes != _logical_bytes(local_shape, dtype):
            _fail(f"has invalid {expected_name} local byte accounting")
        mask[partition] = 1
        local_bytes += shard_bytes
    if _integer(report.get("addressable_logical_bytes"), label="addressable bytes") != local_bytes:
        _fail(f"has invalid {expected_name} addressable byte accounting")
    return (expected_name, shape, dtype, _logical_bytes(shape, dtype)), tuple(mask)


def _replicated_report(
    value: object,
    *,
    expected_name: str,
    process_index: int,
) -> tuple[object, ...]:
    report = _mapping(value, label=f"replicated_array_reports.{expected_name}")
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
        label=f"{expected_name} replicated-array report",
    )
    if (
        report.get("schema_version") != "femx.jax.collective.replicated_array_report/v1"
        or report.get("name") != expected_name
        or report.get("partition_spec") != []
    ):
        _fail(f"has an invalid {expected_name} replicated-array contract")
    shape = _shape(report.get("global_shape"), label=f"{expected_name}.global_shape")
    dtype = _text(report.get("dtype"), label=f"{expected_name}.dtype")
    logical_bytes = _logical_bytes(shape, dtype)
    if (
        _integer(report.get("global_device_count"), label="replicated devices")
        != EXPECTED_GLOBAL_DEVICE_COUNT
        or _integer(report.get("addressable_device_count"), label="addressable devices")
        != EXPECTED_LOCAL_DEVICE_COUNT
        or _integer(report.get("process_index"), label="replicated process") != process_index
        or _integer(report.get("process_count"), label="replicated processes")
        != EXPECTED_PROCESS_COUNT
        or _integer(report.get("logical_bytes_per_replica"), label="replica bytes") != logical_bytes
        or _integer(report.get("addressable_logical_bytes"), label="local replica bytes")
        != logical_bytes * EXPECTED_LOCAL_DEVICE_COUNT
        or _integer(report.get("global_replica_logical_bytes"), label="global replica bytes")
        != logical_bytes * EXPECTED_GLOBAL_DEVICE_COUNT
    ):
        _fail(f"has inconsistent {expected_name} replicated-array metadata")
    intent = _text(report.get("replication_intent"), label="replication intent")
    if intent != REPLICATION_INTENT or logical_bytes > MAXIMUM_REPLICATED_LOGICAL_BYTES_PER_REPLICA:
        _fail(f"exceeds the bounded {expected_name} replication contract")
    return expected_name, shape, dtype, logical_bytes, intent


def _critical_report(
    value: object,
    *,
    expected_name: str,
    process_index: int,
) -> tuple[tuple[object, ...], tuple[int, ...]]:
    report = _mapping(value, label=f"critical_array_reports.{expected_name}")
    _keys(
        report,
        (
            "schema_version",
            "name",
            "global_shape",
            "dtype",
            "partition_spec",
            "process_index",
            "process_count",
            "global_device_count",
            "addressable_shards",
        ),
        label=f"{expected_name} critical-array report",
    )
    expected_shape, expected_dtype, expected_spec = CRITICAL_ARRAY_SPECS[expected_name]
    if (
        report.get("schema_version") != CRITICAL_ARRAY_REPORT_SCHEMA
        or report.get("name") != expected_name
        or _shape(report.get("global_shape"), label=f"{expected_name}.global_shape")
        != expected_shape
        or report.get("dtype") != expected_dtype
        or tuple(_sequence(report.get("partition_spec"), label="partition spec")) != expected_spec
        or _integer(report.get("process_index"), label="critical process") != process_index
        or _integer(report.get("process_count"), label="critical processes")
        != EXPECTED_PROCESS_COUNT
        or _integer(report.get("global_device_count"), label="critical devices")
        != EXPECTED_GLOBAL_DEVICE_COUNT
    ):
        _fail(f"has an invalid {expected_name} critical-array contract")
    shards = _sequence(report.get("addressable_shards"), label=f"{expected_name}.shards")
    if len(shards) != EXPECTED_LOCAL_DEVICE_COUNT:
        _fail(f"requires {expected_name} on every local device")
    mask = [0] * EXPECTED_GLOBAL_DEVICE_COUNT
    for raw in shards:
        shard = _mapping(raw, label=f"{expected_name}.shards[]")
        _keys(
            shard,
            ("partition_index", "process_index", "device_kind", "index", "local_shape"),
            label=f"{expected_name} critical shard",
        )
        partition = _integer(shard.get("partition_index"), label="critical partition")
        if partition >= EXPECTED_GLOBAL_DEVICE_COUNT or mask[partition]:
            _fail(f"repeats or exceeds an {expected_name} critical partition")
        if _integer(shard.get("process_index"), label="critical shard process") != process_index:
            _fail(f"assigns an {expected_name} critical shard to the wrong process")
        if (
            _text(shard.get("device_kind"), label="critical shard device kind")
            not in EXPECTED_DEVICE_KINDS
        ):
            _fail(f"places {expected_name} on a non-v4 TPU device")
        index = tuple(
            tuple(
                _integer(bound, label="critical index bound")
                for bound in _sequence(axis, label="critical index axis")
            )
            for axis in _sequence(shard.get("index"), label="critical index")
        )
        local_shape = _shape(shard.get("local_shape"), label="critical local shape")
        if len(index) != len(expected_shape) or any(len(axis) != 2 for axis in index):
            _fail(f"has a malformed {expected_name} critical index")
        if tuple(stop - start for start, stop in index) != local_shape:
            _fail(f"has an inconsistent {expected_name} critical index and shape")
        sharded_axis = 1 if expected_name == "applied-inverse-permittivity" else 0
        expected_width = expected_shape[sharded_axis] // EXPECTED_GLOBAL_DEVICE_COUNT
        if index[sharded_axis] != (
            partition * expected_width,
            (partition + 1) * expected_width,
        ):
            _fail(f"has an unexpected {expected_name} partition range")
        for axis, extent in enumerate(expected_shape):
            if axis != sharded_axis and index[axis] != (0, extent):
                _fail(f"partitions an unexpected {expected_name} axis")
        mask[partition] = 1
    return (expected_name, expected_shape, expected_dtype, expected_spec), tuple(mask)


def _executable(value: object, *, name: str) -> _Executable:
    record = _mapping(value, label=f"executables.{name}")
    _keys(record, ("timing", "memory", "stablehlo"), label=f"{name} executable")
    timing = _mapping(record.get("timing"), label=f"{name}.timing")
    memory = _mapping(record.get("memory"), label=f"{name}.memory")
    hlo = _mapping(record.get("stablehlo"), label=f"{name}.stablehlo")
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
        label=f"{name} timing report",
    )
    _keys(
        memory,
        (
            "schema_version",
            "generated_code_bytes",
            "argument_bytes",
            "output_bytes",
            "alias_bytes",
            "temporary_bytes",
            "compiler_peak_bytes",
            "hbm_capacity_bytes_per_device",
            "hbm_fraction",
            "risk",
            "claim_scope",
        ),
        label=f"{name} memory report",
    )
    _keys(
        hlo,
        (
            "sha256",
            "all_to_all_count",
            "collective_permute_count",
            "all_reduce_count",
            "contains_all_gather",
            "contains_float64",
        ),
        label=f"{name} StableHLO report",
    )
    executions = tuple(
        _number(item, label=f"{name}.execution_seconds[]", positive=True)
        for item in _sequence(timing.get("execution_seconds"), label="execution seconds")
    )
    if len(executions) < 3:
        _fail(f"requires at least three synchronized {name} execution samples")
    lowering = _number(timing.get("lowering_seconds"), label="lowering seconds")
    compilation = _number(timing.get("compilation_seconds"), label="compilation seconds")
    warmup = _number(timing.get("warmup_seconds"), label="warmup seconds")
    observed_timing_summary = (
        _number(timing.get("execution_min_seconds"), label="execution minimum", positive=True),
        _number(timing.get("execution_median_seconds"), label="execution median", positive=True),
        _number(timing.get("execution_max_seconds"), label="execution maximum", positive=True),
    )
    expected_timing_summary = (
        min(executions),
        statistics.median(executions),
        max(executions),
    )
    if (
        timing.get("schema_version") != "femx.jax.collective.timing_report/v1"
        or timing.get("synchronization") != "every timed result blocked until ready"
        or any(
            not math.isclose(observed, expected, rel_tol=1.0e-12, abs_tol=1.0e-12)
            for observed, expected in zip(
                observed_timing_summary,
                expected_timing_summary,
                strict=True,
            )
        )
    ):
        _fail(f"has an inconsistent {name} timing report")

    generated = _integer(memory.get("generated_code_bytes"), label="generated code bytes")
    arguments = _integer(memory.get("argument_bytes"), label="argument bytes")
    outputs = _integer(memory.get("output_bytes"), label="output bytes")
    aliases = _integer(memory.get("alias_bytes"), label="alias bytes")
    temporaries = _integer(memory.get("temporary_bytes"), label="temporary bytes")
    peak = _integer(memory.get("compiler_peak_bytes"), label="compiler peak bytes")
    capacity = _integer(
        memory.get("hbm_capacity_bytes_per_device"), label="HBM capacity", positive=True
    )
    fraction = _number(memory.get("hbm_fraction"), label="HBM fraction")
    risk = _text(memory.get("risk"), label="HBM risk")
    expected_peak = max(0, arguments + outputs + temporaries - aliases)
    if fraction < 0.70:
        expected_risk = "safe"
    elif fraction < 0.85:
        expected_risk = "elevated"
    elif fraction < 0.95:
        expected_risk = "high"
    else:
        expected_risk = "extreme"
    if (
        memory.get("schema_version") != "femx.jax.collective.memory_report/v1"
        or memory.get("claim_scope") != "compiler estimate; not live HBM usage"
        or peak != expected_peak
        or not math.isclose(fraction, peak / capacity, rel_tol=1.0e-12, abs_tol=1.0e-12)
        or fraction >= TOLERANCES["maximum_compiler_hbm_fraction"]
        or risk != expected_risk
        or _boolean(hlo.get("contains_all_gather"), label="contains_all_gather")
        or _boolean(hlo.get("contains_float64"), label="contains_float64")
    ):
        _fail(f"does not admit the {name} compiler-memory or StableHLO contract")
    all_to_all = _integer(hlo.get("all_to_all_count"), label="all-to-all count")
    permute = _integer(hlo.get("collective_permute_count"), label="collective-permute count")
    reduce = _integer(hlo.get("all_reduce_count"), label="all-reduce count")
    if all_to_all == 0 or permute == 0 or reduce == 0:
        _fail(f"does not expose the required {name} distributed collectives")
    static = (
        _sha256(hlo.get("sha256"), label="StableHLO SHA-256"),
        all_to_all,
        permute,
        reduce,
        peak,
        capacity,
        fraction,
        risk,
        generated,
        arguments,
        outputs,
        aliases,
        temporaries,
    )
    return _Executable(static, lowering, compilation, warmup, executions, peak, fraction, risk)


def _validated_numerics(value: object) -> Mapping[str, object]:
    numerics = _mapping(value, label="numerics")
    _keys(
        numerics,
        (
            "finite",
            "forward_converged",
            "adjoint_converged",
            "thermo_optic_all_valid",
            "material_destination_sharding_preserved",
            "reference_phasor_real",
            "reference_phasor_imag",
            "reference_phasor_magnitude",
            "objective",
            "potential_relative_difference",
            "temperature_relative_difference",
            "cell_temperature_relative_difference",
            "parameter_relative_difference",
            "material_relative_difference",
            "objective_explicit_relative_difference",
            "native_explicit_gradient_relative_differences",
            "native_gradient_norms",
            "applied_voltage_finite_difference",
            "iterations",
            "current_residual_error",
            "heat_residual_error",
            "current_linear_backward_error",
            "heat_linear_backward_error",
            "transfer_relative_error",
            "adjoint_backward_error",
            "electrical_joule_power_W_per_m",
            "thermal_joule_load_W_per_m",
            "cell_cotangent_norm",
            "potential_cotangent_norm",
            "temperature_cotangent_norm",
        ),
        label="numerical report",
    )
    required_flags = (
        "finite",
        "forward_converged",
        "adjoint_converged",
        "thermo_optic_all_valid",
        "material_destination_sharding_preserved",
    )
    if any(not _boolean(numerics.get(name), label=f"numerics.{name}") for name in required_flags):
        _fail("contains a false numerical admission flag")
    positive_names = (
        "reference_phasor_magnitude",
        "cell_cotangent_norm",
        "temperature_cotangent_norm",
    )
    for name in positive_names:
        _number(numerics.get(name), label=f"numerics.{name}", positive=True)
    potential_cotangent_norm = _number(
        numerics.get("potential_cotangent_norm"),
        label="numerics.potential_cotangent_norm",
    )
    if potential_cotangent_norm != 0.0:
        _fail("requires the direct sampled-cell potential cotangent norm to be zero")
    reference_real = _signed_number(
        numerics.get("reference_phasor_real"), label="numerics.reference_phasor_real"
    )
    reference_imag = _signed_number(
        numerics.get("reference_phasor_imag"), label="numerics.reference_phasor_imag"
    )
    reference_magnitude = _number(
        numerics.get("reference_phasor_magnitude"),
        label="numerics.reference_phasor_magnitude",
        positive=True,
    )
    if not math.isclose(
        reference_magnitude,
        math.hypot(reference_real, reference_imag),
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        _fail("has an inconsistent reference phasor magnitude")
    _signed_number(numerics.get("objective"), label="numerics.objective")
    _integer(numerics.get("iterations"), label="numerics.iterations", positive=True)
    for name in ("electrical_joule_power_W_per_m", "thermal_joule_load_W_per_m"):
        _number(numerics.get(name), label=f"numerics.{name}", positive=True)
    bounded = {
        "potential_relative_difference": "potential_relative_difference",
        "temperature_relative_difference": "temperature_relative_difference",
        "cell_temperature_relative_difference": "cell_temperature_relative_difference",
        "parameter_relative_difference": "parameter_relative_difference",
        "material_relative_difference": "material_relative_difference",
        "objective_explicit_relative_difference": "objective_explicit_relative_difference",
        "current_residual_error": "current_residual_error",
        "heat_residual_error": "heat_residual_error",
        "current_linear_backward_error": "linear_backward_error",
        "heat_linear_backward_error": "linear_backward_error",
        "adjoint_backward_error": "adjoint_backward_error",
        "transfer_relative_error": "transfer_relative_error",
    }
    for field, tolerance in bounded.items():
        if _number(numerics.get(field), label=f"numerics.{field}") > TOLERANCES[tolerance]:
            _fail(f"exceeds the {field} tolerance")
    gradient_differences = tuple(
        _number(item, label="native-explicit gradient difference")
        for item in _sequence(
            numerics.get("native_explicit_gradient_relative_differences"),
            label="native-explicit gradient differences",
        )
    )
    gradient_norms = tuple(
        _number(item, label="native gradient norm", positive=True)
        for item in _sequence(numerics.get("native_gradient_norms"), label="native gradient norms")
    )
    if (
        len(gradient_differences) != 3
        or len(gradient_norms) != 3
        or max(gradient_differences) > TOLERANCES["native_explicit_gradient_relative_difference"]
    ):
        _fail("does not admit all three native/explicit parameter gradients")
    finite_difference = _mapping(
        numerics.get("applied_voltage_finite_difference"),
        label="applied-voltage finite difference",
    )
    _keys(
        finite_difference,
        ("steps", "gradients", "relative_errors"),
        label="applied-voltage finite-difference report",
    )
    errors = tuple(
        _number(item, label="finite-difference relative error")
        for item in _sequence(finite_difference.get("relative_errors"), label="FD errors")
    )
    steps = tuple(
        _number(item, label="finite-difference step", positive=True)
        for item in _sequence(finite_difference.get("steps"), label="FD steps")
    )
    gradients = tuple(
        _number(
            abs(_signed_number(item, label="finite-difference gradient")),
            label="absolute FD gradient",
            positive=True,
        )
        for item in _sequence(finite_difference.get("gradients"), label="FD gradients")
    )
    if (
        len(errors) != 2
        or len(steps) != 2
        or len(gradients) != 2
        or steps != FINITE_DIFFERENCE_STEPS
    ):
        _fail("requires exactly two applied-voltage finite-difference checks")
    if max(errors) > TOLERANCES["finite_difference_relative_error"]:
        _fail("does not admit both applied-voltage finite-difference checks")
    return cast(Mapping[str, object], json.loads(_canonical_json(numerics)))


def _process(record: Mapping[str, object]) -> _Process:
    _keys(
        record,
        (
            "schema_version",
            "status",
            "provenance",
            "runtime",
            "launch_claim",
            "input",
            "plan",
            "mesh_report",
            "addressability",
            "partitioned_array_reports",
            "replicated_array_reports",
            "critical_array_reports",
            "coordinate_admission",
            "scene",
            "numerics",
            "tolerances",
            "executables",
            "claim_scope",
        ),
        label="process record",
    )
    if record.get("schema_version") != PROCESS_EVIDENCE_SCHEMA or record.get("status") != "passed":
        _fail("contains an unsupported or non-passing process record")
    _exact_json(record.get("tolerances"), dict(TOLERANCES), label="tolerance policy")
    _text(record.get("claim_scope"), label="claim scope")

    provenance = _mapping(record.get("provenance"), label="provenance")
    _keys(
        provenance,
        ("run_id", "profile", "source_commit", "source_digest", "config_digest"),
        label="provenance",
    )
    provenance_identity = (
        _text(provenance.get("run_id"), label="run id"),
        _text(provenance.get("profile"), label="profile"),
        _git_revision(provenance.get("source_commit"), label="source commit"),
        _sha256(provenance.get("source_digest"), label="source digest"),
        _sha256(provenance.get("config_digest"), label="config digest"),
    )
    runtime = _mapping(record.get("runtime"), label="runtime")
    _keys(
        runtime,
        (
            "backend",
            "jax_version",
            "jaxlib_version",
            "fdtdx_version",
            "x64_enabled",
            "default_matmul_precision",
            "process_index",
            "process_count",
            "local_device_count",
            "global_device_count",
            "device_kinds",
            "scalar_contract",
        ),
        label="runtime",
    )
    process_index = _integer(runtime.get("process_index"), label="process index")
    if (
        process_index >= EXPECTED_PROCESS_COUNT
        or runtime.get("backend") != "tpu"
        or runtime.get("x64_enabled") is not False
        or runtime.get("default_matmul_precision") != "highest"
        or _integer(runtime.get("process_count"), label="process count") != EXPECTED_PROCESS_COUNT
        or _integer(runtime.get("local_device_count"), label="local device count")
        != EXPECTED_LOCAL_DEVICE_COUNT
        or _integer(runtime.get("global_device_count"), label="global device count")
        != EXPECTED_GLOBAL_DEVICE_COUNT
    ):
        _fail("requires the exact physical v4-64 TPU topology and scalar mode")
    _exact_json(runtime.get("scalar_contract"), dict(SCALAR_CONTRACT), label="scalar contract")
    kinds = tuple(
        _text(item, label="device kind")
        for item in _sequence(runtime.get("device_kinds"), label="device kinds")
    )
    if kinds != EXPECTED_DEVICE_KINDS:
        _fail("requires the exact TPU v4 device kind")
    runtime_identity = (
        _text(runtime.get("jax_version"), label="JAX version"),
        _text(runtime.get("jaxlib_version"), label="jaxlib version"),
        _text(runtime.get("fdtdx_version"), label="FDTDX version"),
        kinds,
    )
    if runtime_identity[2] != FDTDX_PACKAGE_VERSION:
        _fail("does not use the locked FDTDX package version")

    claim = _mapping(record.get("launch_claim"), label="launch claim")
    _keys(
        claim,
        (
            "schema_version",
            "run_id",
            "worker_index",
            "process_index",
            "source_sha256",
            "config_sha256",
            "scope",
        ),
        label="launch claim",
    )
    worker_index = _integer(claim.get("worker_index"), label="worker index")
    if (
        claim.get("schema_version") != WORKER_ENTRY_CLAIM_SCHEMA
        or worker_index >= EXPECTED_PROCESS_COUNT
        or _integer(claim.get("process_index"), label="claim process index") != process_index
        or claim.get("run_id") != provenance_identity[0]
        or claim.get("source_sha256") != provenance_identity[3]
        or claim.get("config_sha256") != provenance_identity[4]
    ):
        _fail("has an inconsistent worker-entry claim")
    _text(claim.get("scope"), label="worker-entry scope")

    input_record = _mapping(record.get("input"), label="input")
    _keys(
        input_record,
        (
            "manifest_sha256",
            "arrays_sha256",
            "electrothermal_arrays_sha256",
            "source_commit",
            "sampling_operator_sha256",
            "transfer_operator_sha256",
            "scene_sha256",
            "fdtdx_package_version",
            "fdtdx_source_revision",
            "fdtdx_source_digest",
            "fdtdx_module_sha256",
        ),
        label="input",
    )
    input_identity = (
        _sha256(input_record.get("manifest_sha256"), label="input manifest SHA-256"),
        _sha256(input_record.get("arrays_sha256"), label="input arrays SHA-256"),
        _sha256(
            input_record.get("electrothermal_arrays_sha256"),
            label="electrothermal arrays SHA-256",
        ),
        _git_revision(input_record.get("source_commit"), label="input source commit"),
        _sha256(input_record.get("sampling_operator_sha256"), label="sampling SHA-256"),
        _sha256(input_record.get("transfer_operator_sha256"), label="transfer SHA-256"),
        _sha256(input_record.get("scene_sha256"), label="scene SHA-256"),
    )
    if input_identity[3] != provenance_identity[2]:
        _fail("mixes controller and deployed femx commits")
    if (
        input_record.get("fdtdx_package_version") != FDTDX_PACKAGE_VERSION
        or input_record.get("fdtdx_source_revision") != FDTDX_SOURCE_REVISION
        or input_record.get("fdtdx_source_digest") != FDTDX_SOURCE_DIGEST
    ):
        _fail("does not bind the locked FDTDX source")
    _exact_json(
        input_record.get("fdtdx_module_sha256"),
        dict(FDTDX_MODULE_SHA256),
        label="FDTDX module hashes",
    )

    plan = _mapping(record.get("plan"), label="plan")
    _keys(
        plan,
        (
            "sha256",
            "layout_sha256",
            "partition_count",
            "node_count",
            "triangle_count",
            "free_dof_count",
        ),
        label="plan",
    )
    plan_identity = (
        _sha256(plan.get("sha256"), label="plan SHA-256"),
        _sha256(plan.get("layout_sha256"), label="layout SHA-256"),
        _integer(plan.get("partition_count"), label="plan partitions", positive=True),
        _integer(plan.get("node_count"), label="plan nodes", positive=True),
        _integer(plan.get("triangle_count"), label="plan triangles", positive=True),
        _integer(plan.get("free_dof_count"), label="plan free DOFs", positive=True),
    )
    if plan_identity[2] != EXPECTED_GLOBAL_DEVICE_COUNT:
        _fail("requires one immutable FEM partition per TPU device")

    mesh = _mapping(record.get("mesh_report"), label="mesh report")
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
        label="mesh report",
    )
    assignments = _sequence(mesh.get("assignments"), label="mesh assignments")
    if (
        mesh.get("schema_version") != "femx.jax.collective.mesh_report/v1"
        or mesh.get("axis_name") != "shard"
        or _integer(mesh.get("partition_count"), label="mesh partitions")
        != EXPECTED_GLOBAL_DEVICE_COUNT
        or _integer(mesh.get("global_device_count"), label="mesh devices")
        != EXPECTED_GLOBAL_DEVICE_COUNT
        or _integer(mesh.get("addressable_device_count"), label="mesh local devices")
        != EXPECTED_LOCAL_DEVICE_COUNT
        or _integer(mesh.get("process_count"), label="mesh processes") != EXPECTED_PROCESS_COUNT
        or mesh.get("is_multi_process") is not True
        or _sha256(mesh.get("layout_sha256"), label="mesh layout SHA-256") != plan_identity[1]
        or len(assignments) != EXPECTED_GLOBAL_DEVICE_COUNT
    ):
        _fail("has an inconsistent global Mesh report")
    assignment_identity: list[tuple[object, ...]] = []
    local_from_mesh = [0] * EXPECTED_GLOBAL_DEVICE_COUNT
    for expected_partition, raw in enumerate(assignments):
        assignment = _mapping(raw, label="mesh assignment")
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
            label="mesh assignment",
        )
        partition = _integer(assignment.get("partition_index"), label="assignment partition")
        owner = _integer(assignment.get("process_index"), label="assignment process")
        addressable = _boolean(assignment.get("addressable"), label="assignment addressable")
        if partition != expected_partition or owner >= EXPECTED_PROCESS_COUNT:
            _fail("has an invalid global Mesh assignment")
        if addressable != (owner == process_index):
            _fail("has an incorrect process-addressable Mesh assignment")
        if addressable:
            local_from_mesh[partition] = 1
        platform = _text(assignment.get("platform"), label="assignment platform")
        device_kind = _text(assignment.get("device_kind"), label="assignment device kind")
        if platform != "tpu" or device_kind not in EXPECTED_DEVICE_KINDS:
            _fail("assigns a FEM partition outside the locked TPU v4 topology")
        assignment_identity.append(
            (
                partition,
                owner,
                platform,
                device_kind,
            )
        )
    mesh_identity = tuple(assignment_identity)

    partitioned = _mapping(record.get("partitioned_array_reports"), label="partitioned reports")
    if tuple(sorted(partitioned)) != PARTITIONED_ARRAY_REPORT_NAMES:
        _fail("requires every named partitioned FEM and transfer input report")
    partitioned_identity: list[object] = []
    masks: list[tuple[int, ...]] = []
    for name in PARTITIONED_ARRAY_REPORT_NAMES:
        identity, mask = _partitioned_report(
            partitioned[name], expected_name=name, process_index=process_index
        )
        partitioned_identity.append(identity)
        masks.append(mask)
    if any(mask != tuple(local_from_mesh) for mask in masks):
        _fail("mixes partitioned-array and Mesh addressability")

    replicated = _mapping(record.get("replicated_array_reports"), label="replicated reports")
    if tuple(sorted(replicated)) != REPLICATED_ARRAY_REPORT_NAMES:
        _fail("requires every named bounded replicated input report")
    replicated_identity = tuple(
        _replicated_report(replicated[name], expected_name=name, process_index=process_index)
        for name in REPLICATED_ARRAY_REPORT_NAMES
    )

    critical = _mapping(record.get("critical_array_reports"), label="critical reports")
    if tuple(sorted(critical)) != tuple(sorted(CRITICAL_ARRAY_SPECS)):
        _fail("requires every critical FDTDX array report")
    critical_identity: list[object] = []
    for name in sorted(CRITICAL_ARRAY_SPECS):
        identity, mask = _critical_report(
            critical[name], expected_name=name, process_index=process_index
        )
        critical_identity.append(identity)
        if mask != tuple(local_from_mesh):
            _fail("mixes FDTDX array and global Mesh addressability")

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
        _integer(item, label="local partition mask")
        for item in _sequence(
            addressability.get("process_local_partition_mask"), label="local mask"
        )
    )
    counts = tuple(
        _integer(item, label="partition addressability count")
        for item in _sequence(
            addressability.get("partition_addressability_counts"), label="addressability counts"
        )
    )
    if (
        local_mask != tuple(local_from_mesh)
        or counts != (1,) * EXPECTED_GLOBAL_DEVICE_COUNT
        or addressability.get("every_partition_addressable_once") is not True
    ):
        _fail("does not prove exact once-only global partition addressability")

    coordinates = _mapping(record.get("coordinate_admission"), label="coordinate admission")
    _keys(
        coordinates,
        (
            "maximum_absolute_errors_m",
            "maximum_grid_fraction_errors",
            "maximum_ulp_errors",
            "float32_rounding_exact",
            "admitted",
        ),
        label="coordinate admission",
    )
    errors = tuple(
        _number(item, label="coordinate absolute error")
        for item in _sequence(
            coordinates.get("maximum_absolute_errors_m"), label="coordinate errors"
        )
    )
    fractions = tuple(
        _number(item, label="coordinate grid fraction")
        for item in _sequence(
            coordinates.get("maximum_grid_fraction_errors"), label="coordinate fractions"
        )
    )
    ulps = tuple(
        _integer(item, label="coordinate ULP error")
        for item in _sequence(coordinates.get("maximum_ulp_errors"), label="coordinate ULPs")
    )
    admitted = tuple(
        _boolean(item, label="coordinate admitted")
        for item in _sequence(coordinates.get("admitted"), label="coordinate admitted flags")
    )
    rounding_exact = tuple(
        _boolean(item, label="coordinate exact-rounding flag")
        for item in _sequence(
            coordinates.get("float32_rounding_exact"),
            label="coordinate exact-rounding flags",
        )
    )
    if (
        len(errors) != 3
        or len(fractions) != 3
        or len(ulps) != 3
        or len(rounding_exact) != 3
        or admitted != (True, True, True)
        or max(fractions) > TOLERANCES["runtime_coordinate_max_grid_fraction_error"]
        or max(ulps) > TOLERANCES["runtime_coordinate_max_ulp_error"]
        or any((ulp == 0) != exact for ulp, exact in zip(ulps, rounding_exact, strict=True))
        or any(
            not math.isclose(
                fraction,
                error / GRID_SPACING_M,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
            for error, fraction in zip(errors, fractions, strict=True)
        )
    ):
        _fail("does not admit the controller/runtime target coordinates")
    coordinate_identity = (errors, fractions, ulps, rounding_exact, admitted)

    scene = _mapping(record.get("scene"), label="scene")
    _keys(
        scene,
        ("grid_shape_xyz", "device_shape_xyz", "time_steps", "sha256"),
        label="scene",
    )
    if (
        _shape(scene.get("grid_shape_xyz"), label="scene grid shape") != EXPECTED_GRID_SHAPE
        or _shape(scene.get("device_shape_xyz"), label="scene device shape")
        != EXPECTED_DEVICE_SHAPE
        or _integer(scene.get("time_steps"), label="scene time steps", positive=True) != 302
        or _sha256(scene.get("sha256"), label="scene SHA-256") != input_identity[6]
    ):
        _fail("changes the immutable bounded FDTDX scene")

    numerics = _validated_numerics(record.get("numerics"))
    executables_record = _mapping(record.get("executables"), label="executables")
    if tuple(sorted(executables_record)) != tuple(sorted(EXECUTABLE_NAMES)):
        _fail("requires every declared compiled executable")
    executables = tuple(
        (name, _executable(executables_record[name], name=name)) for name in EXECUTABLE_NAMES
    )
    return _Process(
        process_index=process_index,
        worker_index=worker_index,
        digest=_canonical_digest(record),
        provenance_identity=provenance_identity,
        runtime_identity=runtime_identity,
        input_identity=input_identity,
        plan_identity=plan_identity,
        mesh_identity=mesh_identity,
        local_partition_mask=local_mask,
        partitioned_identity=tuple(partitioned_identity),
        replicated_identity=replicated_identity,
        critical_identity=tuple(critical_identity),
        coordinate_identity=coordinate_identity,
        numerics=numerics,
        executables=executables,
    )


@dataclass(frozen=True, slots=True)
class TpuDistributedFDTDXThermoOpticProcessSetEvidence:
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


def aggregate_tpu_distributed_fdtdx_thermo_optic_process_evidence(
    records: object,
) -> TpuDistributedFDTDXThermoOpticProcessSetEvidence:
    """Admit exactly one immutable record per initialized TPU process."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence) or not records:
        _fail("requires a nonempty sequence of process records")
    parsed = tuple(
        _process(_mapping(record, label="process_records[]"))
        for record in cast(Sequence[object], records)
    )
    ordered = tuple(sorted(parsed, key=lambda item: item.process_index))
    if len(ordered) != EXPECTED_PROCESS_COUNT or tuple(
        item.process_index for item in ordered
    ) != tuple(range(EXPECTED_PROCESS_COUNT)):
        _fail("requires one unique record for each of eight JAX process indexes")
    if tuple(sorted(item.worker_index for item in ordered)) != tuple(range(EXPECTED_PROCESS_COUNT)):
        _fail("requires one unique worker-entry claim for each TPU worker")
    baseline = ordered[0]
    common_fields = (
        "provenance_identity",
        "runtime_identity",
        "input_identity",
        "plan_identity",
        "mesh_identity",
        "partitioned_identity",
        "replicated_identity",
        "critical_identity",
        "coordinate_identity",
        "numerics",
    )
    for item in ordered[1:]:
        for field in common_fields:
            if getattr(item, field) != getattr(baseline, field):
                _fail(f"mixes inconsistent process records at {field}")
        for (name, executable), (baseline_name, baseline_executable) in zip(
            item.executables, baseline.executables, strict=True
        ):
            if (
                name != baseline_name
                or executable.static_identity != baseline_executable.static_identity
            ):
                _fail("mixes inconsistent compiled executable identities")
    combined = tuple(
        sum(item.local_partition_mask[index] for item in ordered)
        for index in range(EXPECTED_GLOBAL_DEVICE_COUNT)
    )
    if combined != (1,) * EXPECTED_GLOBAL_DEVICE_COUNT:  # pragma: no cover - parser invariant
        _fail("does not cover every global partition exactly once")

    _, _, source_commit, source_digest, config_digest = baseline.provenance_identity
    jax_version, jaxlib_version, fdtdx_version, device_kinds = baseline.runtime_identity
    executable_payload: dict[str, object] = {}
    for name in EXECUTABLE_NAMES:
        values = [dict(item.executables)[name] for item in ordered]
        sample_count = len(values[0].execution_seconds)
        ordinal_critical = [
            max(value.execution_seconds[index] for value in values) for index in range(sample_count)
        ]
        executable_payload[name] = {
            "process_lowering_seconds": [value.lowering_seconds for value in values],
            "process_compilation_seconds": [value.compilation_seconds for value in values],
            "process_warmup_seconds": [value.warmup_seconds for value in values],
            "process_execution_median_seconds": [
                statistics.median(value.execution_seconds) for value in values
            ],
            "execution_ordinal_critical_path_seconds": ordinal_critical,
            "execution_ordinal_critical_path_summary_seconds": {
                "min": min(ordinal_critical),
                "median": statistics.median(ordinal_critical),
                "max": max(ordinal_critical),
            },
            "maximum_compiler_peak_bytes": max(value.compiler_peak_bytes for value in values),
            "maximum_compiler_hbm_fraction": max(value.hbm_fraction for value in values),
            "worst_compiler_hbm_risk": max(
                (value.risk for value in values), key=_RISK_RANK.__getitem__
            ),
            "stablehlo_sha256": values[0].static_identity[0],
            "stablehlo_all_to_all_count": values[0].static_identity[1],
            "stablehlo_collective_permute_count": values[0].static_identity[2],
            "stablehlo_all_reduce_count": values[0].static_identity[3],
            "stablehlo_all_gathers_absent_on_every_process": True,
            "stablehlo_float64_absent_on_every_process": True,
            "memory_scope": "compiler estimate; not live HBM usage",
            "timing_scope": "ordinal maximum across synchronized processes; not scaling",
        }
    payload: dict[str, object] = {
        "schema_version": PROCESS_SET_EVIDENCE_SCHEMA,
        "status": "passed",
        "source": {
            "commit": source_commit,
            "source_sha256": source_digest,
            "config_sha256": config_digest,
        },
        "runtime": {
            "backend": "tpu",
            "jax_version": jax_version,
            "jaxlib_version": jaxlib_version,
            "fdtdx_version": fdtdx_version,
            "process_count": EXPECTED_PROCESS_COUNT,
            "local_device_count": EXPECTED_LOCAL_DEVICE_COUNT,
            "global_device_count": EXPECTED_GLOBAL_DEVICE_COUNT,
            "device_kinds": list(cast(tuple[object, ...], device_kinds)),
            "scalar_contract": dict(SCALAR_CONTRACT),
        },
        "input": {
            "manifest_sha256": baseline.input_identity[0],
            "arrays_sha256": baseline.input_identity[1],
            "electrothermal_arrays_sha256": baseline.input_identity[2],
            "sampling_operator_sha256": baseline.input_identity[4],
            "transfer_operator_sha256": baseline.input_identity[5],
            "scene_sha256": baseline.input_identity[6],
            "fdtdx_source_revision": FDTDX_SOURCE_REVISION,
            "fdtdx_source_digest": FDTDX_SOURCE_DIGEST,
            "fdtdx_module_sha256": dict(FDTDX_MODULE_SHA256),
        },
        "plan": {
            "sha256": baseline.plan_identity[0],
            "layout_sha256": baseline.plan_identity[1],
            "partition_count": baseline.plan_identity[2],
            "node_count": baseline.plan_identity[3],
            "triangle_count": baseline.plan_identity[4],
            "free_dof_count": baseline.plan_identity[5],
        },
        "array_admission": {
            "partitioned_names": list(PARTITIONED_ARRAY_REPORT_NAMES),
            "replicated_names": list(REPLICATED_ARRAY_REPORT_NAMES),
            "critical_names": list(sorted(CRITICAL_ARRAY_SPECS)),
            "every_partition_addressable_once": True,
        },
        "coordinate_admission": {
            "maximum_absolute_errors_m": list(baseline.coordinate_identity[0]),
            "maximum_grid_fraction_errors": list(baseline.coordinate_identity[1]),
            "maximum_ulp_errors": list(baseline.coordinate_identity[2]),
            "float32_rounding_exact": list(baseline.coordinate_identity[3]),
            "admitted": list(baseline.coordinate_identity[4]),
        },
        "numerics": baseline.numerics,
        "tolerances": dict(TOLERANCES),
        "executables": executable_payload,
        "process_records": [
            {"process_index": item.process_index, "sha256": item.digest} for item in ordered
        ],
        "claim_scope": (
            "one bounded eight-process, 32-device physical TPU execution of the 2D distributed "
            "electrothermal residual adjoint through one all-to-all thermo-optic transfer and "
            "checkpointed FDTDX objective; not 3D FEM, ring convergence, S-parameters, scaling, "
            "live HBM, measured-device, foundry, or preemption-recovery evidence"
        ),
    }
    return TpuDistributedFDTDXThermoOpticProcessSetEvidence(payload)


__all__ = [
    "CRITICAL_ARRAY_REPORT_SCHEMA",
    "CRITICAL_ARRAY_SPECS",
    "EXECUTABLE_NAMES",
    "EXPECTED_DEVICE_KINDS",
    "EXPECTED_GLOBAL_DEVICE_COUNT",
    "EXPECTED_LOCAL_DEVICE_COUNT",
    "EXPECTED_PROCESS_COUNT",
    "FDTDX_MODULE_SHA256",
    "FDTDX_PACKAGE_VERSION",
    "FDTDX_SOURCE_DIGEST",
    "FDTDX_SOURCE_REVISION",
    "FINITE_DIFFERENCE_STEPS",
    "GRID_SPACING_M",
    "MAXIMUM_REPLICATED_LOGICAL_BYTES_PER_REPLICA",
    "PARTITIONED_ARRAY_REPORT_NAMES",
    "PROCESS_EVIDENCE_SCHEMA",
    "PROCESS_SET_EVIDENCE_SCHEMA",
    "REPLICATED_ARRAY_REPORT_NAMES",
    "REPLICATION_INTENT",
    "SCALAR_CONTRACT",
    "TOLERANCES",
    "WORKER_ENTRY_CLAIM_SCHEMA",
    "TpuDistributedFDTDXThermoOpticProcessSetEvidence",
    "aggregate_tpu_distributed_fdtdx_thermo_optic_process_evidence",
]
