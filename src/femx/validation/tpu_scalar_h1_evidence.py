"""Fail-closed admission of process-complete scalar H1 physical-TPU evidence."""

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

PROCESS_EVIDENCE_SCHEMA = "femx.jax.scalar_h1_collective.tpu_evidence/v1"
PROCESS_SET_EVIDENCE_SCHEMA = "femx.validation.tpu_scalar_h1_collective.process_set/v1"
ARRAY_REPORT_SCHEMA = "femx.jax.collective.array_report/v1"
MESH_REPORT_SCHEMA = "femx.jax.collective.mesh_report/v1"
MEMORY_REPORT_SCHEMA = "femx.jax.collective.memory_report/v1"
TIMING_REPORT_SCHEMA = "femx.jax.collective.timing_report/v1"
WORKER_ENTRY_CLAIM_SCHEMA = "femx.jax.scalar_h1_collective.worker_entry_claim/v1"
EXECUTABLE_NAMES = ("current_forward", "current_vjp", "heat_forward", "heat_vjp")
CASE_NAMES = ("current", "heat")
ARRAY_REPORT_NAMES = (
    "cell_local_dofs",
    "current_cell_rhs",
    "current_cell_stiffness",
    "current_owned_cotangent",
    "heat_cell_rhs",
    "heat_cell_stiffness",
    "heat_owned_cotangent",
    "owner_mask",
)
REAL_SCALAR_CONTRACT = {
    "logical_dtype": "float32",
    "matrix_dtype": "float32",
    "load_dtype": "float32",
    "index_dtype": "int32",
    "mask_dtype": "bool",
    "matmul_precision": "highest",
    "host_reference_dtype": "float64",
    "precision_fallback": False,
}
TOLERANCES: Mapping[str, float] = MappingProxyType(
    {
        "solution_relative_difference": 4.0e-4,
        "rhs_relative_difference": 2.0e-6,
        "vjp_relative_difference": 2.0e-3,
        "host_precision_relative_difference": 2.0e-3,
    }
)
RISK_RANK = {"safe": 0, "elevated": 1, "high": 2, "extreme": 3}


def _fail(message: str) -> NoReturn:
    raise ValidationError(f"scalar H1 physical TPU process-set evidence {message}")


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
    if not math.isfinite(converted) or converted < 0.0:
        _fail(f"requires {label} to be a finite nonnegative number")
    if positive and converted == 0.0:
        _fail(f"requires {label} to be positive")
    return converted


def _boolean(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"requires {label} to be boolean")
    return value


def _sha256(value: object, *, label: str) -> str:
    result = _text(value, label=label)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        _fail(f"requires {label} to be a canonical lowercase SHA-256")
    return result


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
            "scalar H1 physical TPU process record is not canonical JSON"
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
    collective_permute_count: int
    all_reduce_count: int


@dataclass(frozen=True, slots=True)
class _Case:
    identity: tuple[object, ...]
    iterations: int
    relative_residual: float
    solution_difference: float
    rhs_difference: float
    matrix_vjp_difference: float
    cell_rhs_vjp_difference: float
    host_precision_difference: float


@dataclass(frozen=True, slots=True)
class _Process:
    process_index: int
    worker_index: int
    digest: str
    provenance_identity: tuple[str, str, str, str]
    runtime_identity: tuple[object, ...]
    problem_identity: tuple[object, ...]
    assignment_identity: tuple[tuple[object, ...], ...]
    local_partition_mask: tuple[int, ...]
    cases: tuple[tuple[str, _Case], ...]
    executables: tuple[tuple[str, _Executable], ...]


def _executable(value: object, *, name: str) -> _Executable:
    record = _mapping(value, label=f"executables.{name}")
    timing = _mapping(record.get("timing"), label=f"executables.{name}.timing")
    if timing.get("schema_version") != TIMING_REPORT_SCHEMA:
        _fail(f"has unsupported {name} timing schema")
    if timing.get("synchronization") != "every timed result blocked until ready":
        _fail(f"does not prove synchronized {name} results")
    samples = tuple(
        _number(item, label=f"executables.{name}.timing.execution_seconds")
        for item in _sequence(
            timing.get("execution_seconds"),
            label=f"executables.{name}.timing.execution_seconds",
        )
    )
    if len(samples) != 5:
        _fail(f"requires exactly five {name} execution samples")
    memory = _mapping(record.get("memory"), label=f"executables.{name}.memory")
    if memory.get("schema_version") != MEMORY_REPORT_SCHEMA:
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
    fraction = _number(
        memory.get("hbm_fraction"),
        label=f"executables.{name}.memory.hbm_fraction",
    )
    if not math.isclose(fraction, peak / capacity, rel_tol=1.0e-12, abs_tol=1.0e-15):
        _fail(f"has inconsistent {name} compiler-memory fraction")
    risk = _text(memory.get("risk"), label=f"executables.{name}.memory.risk")
    if risk != _risk_for_fraction(fraction):
        _fail(f"has inconsistent {name} compiler-memory risk")
    if risk not in {"safe", "elevated"}:
        _fail(f"exceeds the admitted compiler-memory risk for {name}")
    if memory.get("claim_scope") != "compiler estimate; not live HBM usage":
        _fail(f"overstates {name} compiler memory as live HBM")
    if _boolean(
        record.get("stablehlo_contains_all_gather"),
        label=f"executables.{name}.stablehlo_contains_all_gather",
    ):
        _fail(f"contains an all-gather in {name} StableHLO")
    permutes = _integer(
        record.get("stablehlo_collective_permute_count"),
        label=f"executables.{name}.stablehlo_collective_permute_count",
        positive=True,
    )
    reductions = _integer(
        record.get("stablehlo_all_reduce_count"),
        label=f"executables.{name}.stablehlo_all_reduce_count",
        positive=True,
    )
    return _Executable(
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
        collective_permute_count=permutes,
        all_reduce_count=reductions,
    )


def _case(value: object, *, name: str, tolerances: Mapping[str, float]) -> _Case:
    record = _mapping(value, label=f"cases.{name}")
    if record.get("status") != "passed":
        _fail(f"contains a {name} case that did not pass")
    physics = _mapping(record.get("physics"), label=f"cases.{name}.physics")
    model = _text(physics.get("model"), label=f"cases.{name}.physics.model")
    units = _mapping(physics.get("units"), label=f"cases.{name}.physics.units")
    unit_identity = tuple(
        _text(units.get(key), label=f"cases.{name}.physics.units.{key}")
        for key in ("state", "coefficient", "source", "facet_load")
    )
    coefficient_minimum = _number(
        physics.get("coefficient_minimum"),
        label=f"cases.{name}.physics.coefficient_minimum",
        positive=True,
    )
    coefficient_maximum = _number(
        physics.get("coefficient_maximum"),
        label=f"cases.{name}.physics.coefficient_maximum",
        positive=True,
    )
    if coefficient_maximum < coefficient_minimum:
        _fail(f"has inverted {name} coefficient bounds")
    source_maximum = _number(
        physics.get("source_maximum"),
        label=f"cases.{name}.physics.source_maximum",
    )
    dirichlet_minimum = _number(
        physics.get("dirichlet_minimum"),
        label=f"cases.{name}.physics.dirichlet_minimum",
    )
    dirichlet_maximum = _number(
        physics.get("dirichlet_maximum"),
        label=f"cases.{name}.physics.dirichlet_maximum",
    )
    if dirichlet_maximum < dirichlet_minimum:
        _fail(f"has inverted {name} Dirichlet bounds")
    if (
        physics.get("material_scope")
        != "representative values; not foundry-calibrated material data"
    ):
        _fail(f"overstates {name} material authority")
    if (name == "heat" and source_maximum <= 0.0) or (name == "current" and source_maximum != 0.0):
        _fail(f"does not preserve the declared {name} source case")

    cg = _mapping(record.get("cg"), label=f"cases.{name}.cg")
    relative_tolerance = _number(
        cg.get("relative_tolerance"),
        label=f"cases.{name}.cg.relative_tolerance",
        positive=True,
    )
    absolute_tolerance = _number(
        cg.get("absolute_tolerance"),
        label=f"cases.{name}.cg.absolute_tolerance",
    )
    max_iterations = _integer(
        cg.get("max_iterations"),
        label=f"cases.{name}.cg.max_iterations",
        positive=True,
    )
    iterations = _integer(
        cg.get("iterations"),
        label=f"cases.{name}.cg.iterations",
        positive=True,
    )
    if iterations > max_iterations:
        _fail(f"reports impossible {name} CG iterations")
    rhs_norm = _number(cg.get("rhs_norm"), label=f"cases.{name}.cg.rhs_norm", positive=True)
    _number(
        cg.get("recursive_residual_norm"),
        label=f"cases.{name}.cg.recursive_residual_norm",
    )
    recomputed_residual = _number(
        cg.get("recomputed_residual_norm"),
        label=f"cases.{name}.cg.recomputed_residual_norm",
    )
    relative_residual = _number(
        cg.get("relative_residual"),
        label=f"cases.{name}.cg.relative_residual",
    )
    if relative_residual > relative_tolerance or recomputed_residual > max(
        absolute_tolerance,
        relative_tolerance * rhs_norm,
    ):
        _fail(f"does not satisfy the recomputed {name} CG residual policy")
    if not _boolean(cg.get("converged"), label=f"cases.{name}.cg.converged"):
        _fail(f"does not admit {name} CG convergence")
    if _boolean(cg.get("breakdown"), label=f"cases.{name}.cg.breakdown"):
        _fail(f"reports a {name} CG breakdown")

    numerics = _mapping(record.get("numerics"), label=f"cases.{name}.numerics")
    finite = _mapping(numerics.get("finite"), label=f"cases.{name}.numerics.finite")
    if set(finite) != {"solution", "right_hand_side", "matrix_vjp", "cell_rhs_vjp"} or not all(
        _boolean(item, label=f"cases.{name}.numerics.finite.{key}") for key, item in finite.items()
    ):
        _fail(f"does not prove finite {name} forward and VJP arrays")
    solution_difference = _number(
        numerics.get("solution_relative_difference"),
        label=f"cases.{name}.numerics.solution_relative_difference",
    )
    rhs_difference = _number(
        numerics.get("rhs_relative_difference"),
        label=f"cases.{name}.numerics.rhs_relative_difference",
    )
    matrix_vjp_difference = _number(
        numerics.get("matrix_vjp_relative_difference"),
        label=f"cases.{name}.numerics.matrix_vjp_relative_difference",
    )
    cell_rhs_vjp_difference = _number(
        numerics.get("cell_rhs_vjp_relative_difference"),
        label=f"cases.{name}.numerics.cell_rhs_vjp_relative_difference",
    )
    host_precision_difference = _number(
        numerics.get("host_float32_input_vs_float64_assembly_solution_relative_difference"),
        label=f"cases.{name}.numerics.host_precision_relative_difference",
    )
    if solution_difference > tolerances["solution_relative_difference"]:
        _fail(f"exceeds the admitted {name} solution difference")
    if rhs_difference > tolerances["rhs_relative_difference"]:
        _fail(f"exceeds the admitted {name} RHS difference")
    if max(matrix_vjp_difference, cell_rhs_vjp_difference) > tolerances["vjp_relative_difference"]:
        _fail(f"exceeds the admitted {name} VJP difference")
    if host_precision_difference > tolerances["host_precision_relative_difference"]:
        _fail(f"exceeds the admitted {name} host-precision difference")
    for key in (
        "numpy_input_authority_iterations",
        "numpy_float64_authority_iterations",
        "numpy_adjoint_iterations",
    ):
        _integer(numerics.get(key), label=f"cases.{name}.numerics.{key}", positive=True)
    for key in (
        "numpy_input_authority_residual_norm",
        "numpy_float64_authority_residual_norm",
        "numpy_adjoint_residual_norm",
    ):
        _number(numerics.get(key), label=f"cases.{name}.numerics.{key}")
    authority = _text(numerics.get("authority"), label=f"cases.{name}.numerics.authority")
    if "independent NumPy float64 matrix-free CG" not in authority:
        _fail(f"does not declare the independent {name} numerical authority")
    return _Case(
        identity=(
            model,
            unit_identity,
            coefficient_minimum,
            coefficient_maximum,
            source_maximum,
            dirichlet_minimum,
            dirichlet_maximum,
            relative_tolerance,
            absolute_tolerance,
            max_iterations,
        ),
        iterations=iterations,
        relative_residual=relative_residual,
        solution_difference=solution_difference,
        rhs_difference=rhs_difference,
        matrix_vjp_difference=matrix_vjp_difference,
        cell_rhs_vjp_difference=cell_rhs_vjp_difference,
        host_precision_difference=host_precision_difference,
    )


def _array_report(
    value: object,
    *,
    label: str,
    process_index: int,
    process_count: int,
    partition_count: int,
    global_device_count: int,
    local_partition_mask: tuple[int, ...],
) -> None:
    report = _mapping(value, label=label)
    if report.get("schema_version") != ARRAY_REPORT_SCHEMA:
        _fail(f"has unsupported {label} schema")
    _text(report.get("name"), label=f"{label}.name")
    _text(report.get("dtype"), label=f"{label}.dtype")
    if report.get("partition_axis_name") != "partition":
        _fail(f"requires the partition axis for {label}")
    for key, expected in (
        ("partition_count", partition_count),
        ("global_device_count", global_device_count),
        ("process_index", process_index),
        ("process_count", process_count),
    ):
        if _integer(report.get(key), label=f"{label}.{key}") != expected:
            _fail(f"has inconsistent {label} {key.replace('_', ' ')}")
    shape = tuple(
        _integer(item, label=f"{label}.global_shape", positive=True)
        for item in _sequence(report.get("global_shape"), label=f"{label}.global_shape")
    )
    if not shape or shape[0] != partition_count:
        _fail(f"has inconsistent {label} leading shape")
    global_bytes = _integer(
        report.get("global_logical_bytes"),
        label=f"{label}.global_logical_bytes",
        positive=True,
    )
    addressable_bytes = _integer(
        report.get("addressable_logical_bytes"),
        label=f"{label}.addressable_logical_bytes",
        positive=True,
    )
    if addressable_bytes > global_bytes:
        _fail(f"has impossible {label} addressable bytes")
    if report.get("replication_intent") != "none; one leading FEM partition per device":
        _fail(f"does not prove non-replicated storage for {label}")
    shard_partitions: list[int] = []
    shard_bytes = 0
    for raw_shard in _sequence(
        report.get("addressable_shards"),
        label=f"{label}.addressable_shards",
    ):
        shard = _mapping(raw_shard, label=f"{label}.addressable_shards[]")
        partition = _integer(
            shard.get("partition_index"),
            label=f"{label}.addressable_shards[].partition_index",
        )
        if partition >= partition_count:
            _fail(f"has an out-of-range shard in {label}")
        if (
            _integer(
                shard.get("process_index"),
                label=f"{label}.addressable_shards[].process_index",
            )
            != process_index
        ):
            _fail(f"has a foreign-process shard in {label}")
        _integer(shard.get("device_id"), label=f"{label}.addressable_shards[].device_id")
        _text(shard.get("device_kind"), label=f"{label}.addressable_shards[].device_kind")
        local_shape = tuple(
            _integer(item, label=f"{label}.addressable_shards[].local_shape", positive=True)
            for item in _sequence(
                shard.get("local_shape"),
                label=f"{label}.addressable_shards[].local_shape",
            )
        )
        if not local_shape or local_shape[0] != 1:
            _fail(f"has a noncanonical local shard shape in {label}")
        shard_bytes += _integer(
            shard.get("logical_bytes"),
            label=f"{label}.addressable_shards[].logical_bytes",
            positive=True,
        )
        shard_partitions.append(partition)
    expected_partitions = tuple(
        index for index, active in enumerate(local_partition_mask) if active
    )
    if tuple(sorted(shard_partitions)) != expected_partitions or len(shard_partitions) != len(
        set(shard_partitions)
    ):
        _fail(f"has shards inconsistent with the process-local mask for {label}")
    if shard_bytes != addressable_bytes:
        _fail(f"has inconsistent addressable byte accounting for {label}")


def _process(record: Mapping[str, object]) -> _Process:
    if record.get("schema_version") != PROCESS_EVIDENCE_SCHEMA:
        _fail("has an unsupported process-record schema")
    if record.get("status") != "passed":
        _fail("contains a process record that did not pass")
    provenance = _mapping(record.get("provenance"), label="provenance")
    runtime = _mapping(record.get("runtime"), label="runtime")
    problem = _mapping(record.get("problem"), label="problem")
    mesh = _mapping(record.get("mesh_report"), label="mesh_report")
    addressability = _mapping(record.get("addressability"), label="addressability")
    launch = _mapping(record.get("launch_claim"), label="launch_claim")

    run_id = _text(provenance.get("run_id"), label="provenance.run_id")
    profile = _text(provenance.get("profile"), label="provenance.profile")
    source_digest = _sha256(provenance.get("source_digest"), label="provenance.source_digest")
    config_digest = _sha256(provenance.get("config_digest"), label="provenance.config_digest")
    process_index = _integer(runtime.get("process_index"), label="runtime.process_index")
    process_count = _integer(
        runtime.get("process_count"),
        label="runtime.process_count",
        positive=True,
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
    if process_count < 2 or global_device_count != process_count * local_device_count:
        _fail("has inconsistent multi-process TPU device counts")
    if (
        runtime.get("backend") != "tpu"
        or runtime.get("x64_enabled") is not False
        or runtime.get("default_matmul_precision") != "highest"
    ):
        _fail("requires the physical TPU float32 highest-precision runtime")
    jax_version = _text(runtime.get("jax_version"), label="runtime.jax_version")
    jaxlib_version = _text(runtime.get("jaxlib_version"), label="runtime.jaxlib_version")
    device_kinds = tuple(
        _text(item, label="runtime.device_kinds")
        for item in _sequence(runtime.get("device_kinds"), label="runtime.device_kinds")
    )
    if not device_kinds or tuple(sorted(set(device_kinds))) != device_kinds:
        _fail("requires canonical nonempty TPU device kinds")
    scalar_contract = _mapping(
        runtime.get("real_scalar_contract"),
        label="runtime.real_scalar_contract",
    )
    if dict(scalar_contract) != REAL_SCALAR_CONTRACT:
        _fail("requires the exact physical TPU float32 scalar contract")

    if launch.get("schema_version") != WORKER_ENTRY_CLAIM_SCHEMA:
        _fail("has unsupported worker-entry claim schema")
    worker_index = _integer(launch.get("worker_index"), label="launch_claim.worker_index")
    if (
        launch.get("run_id") != run_id
        or _integer(launch.get("process_index"), label="launch_claim.process_index")
        != process_index
        or launch.get("source_sha256") != source_digest
        or launch.get("config_sha256") != config_digest
    ):
        _fail("has a worker-entry claim inconsistent with provenance or runtime")
    if "prevents duplicate scientific execution" not in _text(
        launch.get("scope"),
        label="launch_claim.scope",
    ):
        _fail("does not preserve the duplicate-entry fence scope")

    if (
        problem.get("model")
        != "two bounded scalar H1/P1 diffusion systems on one exact triangular mesh"
    ):
        _fail("has an unsupported scalar problem model")
    x_intervals = _integer(
        problem.get("x_intervals"),
        label="problem.x_intervals",
        positive=True,
    )
    y_intervals = _integer(
        problem.get("y_intervals"),
        label="problem.y_intervals",
        positive=True,
    )
    node_count = _integer(problem.get("node_count"), label="problem.node_count", positive=True)
    triangle_count = _integer(
        problem.get("triangle_count"),
        label="problem.triangle_count",
        positive=True,
    )
    free_dof_count = _integer(
        problem.get("free_dof_count"),
        label="problem.free_dof_count",
        positive=True,
    )
    partition_count = _integer(
        problem.get("partition_count"),
        label="problem.partition_count",
        positive=True,
    )
    layout_sha256 = _sha256(problem.get("layout_sha256"), label="problem.layout_sha256")
    halo_link_count = _integer(
        problem.get("halo_link_count"),
        label="problem.halo_link_count",
        positive=True,
    )
    halo_value_count = _integer(
        problem.get("halo_value_count"),
        label="problem.halo_value_count",
        positive=True,
    )
    if partition_count != global_device_count or triangle_count < 2 * partition_count:
        _fail("does not assign a bounded nonempty scalar mesh partition to every TPU device")
    for key in (
        "cell_padding_fraction",
        "owned_dof_padding_fraction",
        "ghost_dof_padding_fraction",
    ):
        if _number(problem.get(key), label=f"problem.{key}") > 1.0:
            _fail(f"has invalid problem {key.replace('_', ' ')}")

    if mesh.get("schema_version") != MESH_REPORT_SCHEMA:
        _fail("has unsupported collective Mesh-report schema")
    if mesh.get("axis_name") != "partition" or mesh.get("is_multi_process") is not True:
        _fail("requires an explicit multi-process partition Mesh")
    for key, expected in (
        ("partition_count", partition_count),
        ("global_device_count", global_device_count),
        ("addressable_device_count", local_device_count),
        ("process_count", process_count),
    ):
        if _integer(mesh.get(key), label=f"mesh_report.{key}", positive=True) != expected:
            _fail(f"has inconsistent Mesh {key.replace('_', ' ')}")
    if mesh.get("layout_sha256") != layout_sha256:
        _fail("has inconsistent problem and Mesh layout identities")

    local_mask = tuple(
        _integer(item, label="addressability.process_local_partition_mask")
        for item in _sequence(
            addressability.get("process_local_partition_mask"),
            label="addressability.process_local_partition_mask",
        )
    )
    counts = tuple(
        _integer(item, label="addressability.partition_addressability_counts")
        for item in _sequence(
            addressability.get("partition_addressability_counts"),
            label="addressability.partition_addressability_counts",
        )
    )
    if len(local_mask) != partition_count or any(item not in (0, 1) for item in local_mask):
        _fail("has a noncanonical process-local partition mask")
    if sum(local_mask) != local_device_count:
        _fail("has a process-local mask inconsistent with local device count")
    if (
        counts != (1,) * partition_count
        or addressability.get("every_partition_addressable_once") is not True
    ):
        _fail("does not report every scalar partition addressable exactly once")

    assignments: list[tuple[object, ...]] = []
    raw_assignments = _sequence(mesh.get("assignments"), label="mesh_report.assignments")
    if len(raw_assignments) != partition_count:
        _fail("requires one Mesh assignment per scalar partition")
    for expected_partition, raw_assignment in enumerate(raw_assignments):
        assignment = _mapping(raw_assignment, label="mesh_report.assignments[]")
        partition = _integer(
            assignment.get("partition_index"),
            label="mesh_report.assignments[].partition_index",
        )
        assigned_process = _integer(
            assignment.get("process_index"),
            label="mesh_report.assignments[].process_index",
        )
        device_id = _integer(
            assignment.get("device_id"),
            label="mesh_report.assignments[].device_id",
        )
        platform = _text(
            assignment.get("platform"),
            label="mesh_report.assignments[].platform",
        )
        device_kind = _text(
            assignment.get("device_kind"),
            label="mesh_report.assignments[].device_kind",
        )
        addressable = _boolean(
            assignment.get("addressable"),
            label="mesh_report.assignments[].addressable",
        )
        if partition != expected_partition or assigned_process >= process_count:
            _fail("has a noncanonical Mesh assignment order or process")
        if platform != "tpu" or device_kind not in device_kinds:
            _fail("has a Mesh assignment inconsistent with the TPU runtime")
        if addressable != bool(local_mask[partition]) or addressable != (
            assigned_process == process_index
        ):
            _fail("has Mesh addressability inconsistent with the process-local mask")
        assignments.append((partition, assigned_process, device_id, platform, device_kind))
    if len({(item[1], item[2]) for item in assignments}) != partition_count:
        _fail("requires one unique TPU device per scalar partition")

    reports = _mapping(record.get("array_reports"), label="array_reports")
    if tuple(sorted(reports)) != ARRAY_REPORT_NAMES:
        _fail("requires the exact scalar distributed-array report set")
    for name in ARRAY_REPORT_NAMES:
        _array_report(
            reports[name],
            label=f"array_reports.{name}",
            process_index=process_index,
            process_count=process_count,
            partition_count=partition_count,
            global_device_count=global_device_count,
            local_partition_mask=local_mask,
        )

    tolerances_record = _mapping(record.get("tolerances"), label="tolerances")
    tolerances = {
        name: _number(tolerances_record.get(name), label=f"tolerances.{name}", positive=True)
        for name in TOLERANCES
    }
    if tolerances != TOLERANCES:
        _fail("changes the locked scalar TPU admission tolerances")
    raw_cases = _mapping(record.get("cases"), label="cases")
    if tuple(sorted(raw_cases)) != CASE_NAMES:
        _fail("requires exactly the heat and current scalar cases")
    cases = tuple(
        (name, _case(raw_cases[name], name=name, tolerances=tolerances)) for name in CASE_NAMES
    )
    raw_executables = _mapping(record.get("executables"), label="executables")
    if tuple(sorted(raw_executables)) != EXECUTABLE_NAMES:
        _fail("requires the exact scalar forward/VJP executable set")
    executables = tuple(
        (name, _executable(raw_executables[name], name=name)) for name in EXECUTABLE_NAMES
    )
    claim_scope = _text(record.get("claim_scope"), label="claim_scope")
    for required in (
        "process-complete",
        "not Elmer parity",
        "not a foundry prediction",
        "Spot-preemption recovery",
    ):
        if required not in claim_scope:
            _fail(f"omits the bounded claim-scope phrase {required!r}")
    return _Process(
        process_index=process_index,
        worker_index=worker_index,
        digest=_canonical_digest(record),
        provenance_identity=(run_id, profile, source_digest, config_digest),
        runtime_identity=(
            jax_version,
            jaxlib_version,
            process_count,
            local_device_count,
            global_device_count,
            device_kinds,
        ),
        problem_identity=(
            x_intervals,
            y_intervals,
            node_count,
            triangle_count,
            free_dof_count,
            partition_count,
            layout_sha256,
            halo_link_count,
            halo_value_count,
        ),
        assignment_identity=tuple(assignments),
        local_partition_mask=local_mask,
        cases=cases,
        executables=executables,
    )


@dataclass(frozen=True, slots=True)
class TpuScalarH1ProcessSetEvidence:
    """Canonical admitted aggregate for one exact physical process set."""

    payload: Mapping[str, object]

    def canonical_data(self) -> dict[str, object]:
        """Return a detached JSON-compatible aggregate."""

        return cast(
            dict[str, object],
            json.loads(
                json.dumps(
                    self.payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            ),
        )

    def canonical_json(self) -> str:
        """Return canonical UTF-8 JSON text."""

        return json.dumps(
            self.payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    def digest(self) -> str:
        """Hash the canonical aggregate payload."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def aggregate_tpu_scalar_h1_process_evidence(
    records: object,
) -> TpuScalarH1ProcessSetEvidence:
    """Admit one immutable record per JAX process and aggregate critical-path evidence."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence) or not records:
        _fail("requires a nonempty sequence of process records")
    raw_records = cast(Sequence[object], records)
    parsed = tuple(_process(_mapping(record, label="process_records[]")) for record in raw_records)
    ordered = tuple(sorted(parsed, key=lambda record: record.process_index))
    baseline = ordered[0]
    (
        jax_version,
        jaxlib_version,
        raw_process_count,
        raw_local_device_count,
        raw_global_device_count,
        device_kinds,
    ) = baseline.runtime_identity
    process_count = cast(int, raw_process_count)
    local_device_count = cast(int, raw_local_device_count)
    global_device_count = cast(int, raw_global_device_count)
    if len(ordered) != process_count:
        _fail("requires exactly one process record per declared JAX process")
    if tuple(record.process_index for record in ordered) != tuple(range(process_count)):
        _fail("requires a complete unique canonical JAX process index set")
    if len({record.worker_index for record in ordered}) != process_count:
        _fail("requires one unique worker-entry claim per JAX process")
    for record in ordered[1:]:
        if record.provenance_identity != baseline.provenance_identity:
            _fail("mixes process records from different deployed inputs")
        if record.runtime_identity != baseline.runtime_identity:
            _fail("mixes process records from different TPU runtimes")
        if record.problem_identity != baseline.problem_identity:
            _fail("mixes process records from different scalar problems")
        if record.assignment_identity != baseline.assignment_identity:
            _fail("mixes inconsistent global Mesh assignments")
        if tuple((name, case.identity) for name, case in record.cases) != tuple(
            (name, case.identity) for name, case in baseline.cases
        ):
            _fail("mixes inconsistent heat/current problem identities")
        for (name, executable), (baseline_name, baseline_executable) in zip(
            record.executables,
            baseline.executables,
            strict=True,
        ):
            if name != baseline_name or (
                executable.hbm_capacity_bytes,
                executable.collective_permute_count,
                executable.all_reduce_count,
            ) != (
                baseline_executable.hbm_capacity_bytes,
                baseline_executable.collective_permute_count,
                baseline_executable.all_reduce_count,
            ):
                _fail("mixes inconsistent compiled collective identities")

    combined = tuple(
        sum(record.local_partition_mask[index] for record in ordered)
        for index in range(global_device_count)
    )
    if combined != (1,) * global_device_count:  # pragma: no cover - defensive invariant
        _fail("does not cover every global scalar partition exactly once across processes")

    run_id, profile, source_digest, config_digest = baseline.provenance_identity
    (
        x_intervals,
        y_intervals,
        node_count,
        triangle_count,
        free_dof_count,
        partition_count,
        layout_sha256,
        halo_link_count,
        halo_value_count,
    ) = baseline.problem_identity
    case_payload: dict[str, object] = {}
    for case_name in CASE_NAMES:
        case_records = [dict(record.cases)[case_name] for record in ordered]
        case_payload[case_name] = {
            "maximum_iterations_across_processes": max(case.iterations for case in case_records),
            "maximum_relative_residual_across_processes": max(
                case.relative_residual for case in case_records
            ),
            "maximum_solution_relative_difference_across_processes": max(
                case.solution_difference for case in case_records
            ),
            "maximum_rhs_relative_difference_across_processes": max(
                case.rhs_difference for case in case_records
            ),
            "maximum_matrix_vjp_relative_difference_across_processes": max(
                case.matrix_vjp_difference for case in case_records
            ),
            "maximum_cell_rhs_vjp_relative_difference_across_processes": max(
                case.cell_rhs_vjp_difference for case in case_records
            ),
            "maximum_host_precision_relative_difference_across_processes": max(
                case.host_precision_difference for case in case_records
            ),
            "all_processes_converged_and_finite": True,
        }

    executable_payload: dict[str, object] = {}
    for executable_name in EXECUTABLE_NAMES:
        executable_records = [dict(record.executables)[executable_name] for record in ordered]
        ordinal_critical_path = [
            max(executable.execution_seconds[index] for executable in executable_records)
            for index in range(5)
        ]
        maximum_fraction = max(executable.hbm_fraction for executable in executable_records)
        executable_payload[executable_name] = {
            "process_count": process_count,
            "process_lowering_seconds": [
                executable.lowering_seconds for executable in executable_records
            ],
            "process_compilation_seconds": [
                executable.compilation_seconds for executable in executable_records
            ],
            "process_warmup_seconds": [
                executable.warmup_seconds for executable in executable_records
            ],
            "process_execution_median_seconds": [
                statistics.median(executable.execution_seconds) for executable in executable_records
            ],
            "execution_ordinal_critical_path_seconds": ordinal_critical_path,
            "execution_ordinal_critical_path_summary_seconds": {
                "min": min(ordinal_critical_path),
                "median": statistics.median(ordinal_critical_path),
                "max": max(ordinal_critical_path),
            },
            "maximum_compiler_peak_bytes": max(
                executable.compiler_peak_bytes for executable in executable_records
            ),
            "hbm_capacity_bytes_per_device": executable_records[0].hbm_capacity_bytes,
            "maximum_compiler_hbm_fraction": maximum_fraction,
            "worst_compiler_hbm_risk": max(
                (executable.risk for executable in executable_records),
                key=RISK_RANK.__getitem__,
            ),
            "stablehlo_collective_permute_count": executable_records[0].collective_permute_count,
            "stablehlo_all_reduce_count": executable_records[0].all_reduce_count,
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
            "real_scalar_contract": REAL_SCALAR_CONTRACT,
        },
        "problem": {
            "model": "two bounded scalar H1/P1 diffusion systems on one exact triangular mesh",
            "x_intervals": x_intervals,
            "y_intervals": y_intervals,
            "node_count": node_count,
            "triangle_count": triangle_count,
            "free_dof_count": free_dof_count,
            "partition_count": partition_count,
            "layout_sha256": layout_sha256,
            "halo_link_count": halo_link_count,
            "halo_value_count": halo_value_count,
        },
        "addressability": {
            "combined_partition_addressability_counts": list(combined),
            "every_partition_addressable_once": True,
        },
        "tolerances": dict(TOLERANCES),
        "cases": case_payload,
        "executables": executable_payload,
        "claim_scope": (
            "process-complete physical multi-host TPU scalar H1/P1 RHS, unpreconditioned CG, "
            "and residual-defined implicit-VJP correctness evidence for bounded heat and current "
            "systems; ordinal critical-path timing is not a scaling result, compiler memory is "
            "not live HBM, and this is not Elmer parity, coupled electrothermal execution, "
            "or Spot-preemption recovery; it is not a foundry prediction"
        ),
    }
    return TpuScalarH1ProcessSetEvidence(payload=payload)
