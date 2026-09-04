"""Admission of the opt-in multilevel extension to scalar physical-TPU evidence."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import cast

from femx.core.errors import ValidationError

from .tpu_scalar_h1_evidence import (
    ARRAY_REPORT_SCHEMA,
    CASE_NAMES,
    RISK_RANK,
    TOLERANCES,
    TpuScalarH1ProcessSetEvidence,
    _array_report,
    _boolean,
    _Executable,
    _executable,
    _integer,
    _mapping,
    _number,
    _process,
    _sequence,
    _sha256,
    _text,
    aggregate_tpu_scalar_h1_process_evidence,
)

MULTILEVEL_EXTENSION_SCHEMA = "femx.jax.scalar_h1_collective.multilevel_extension/v1"
MULTILEVEL_PROCESS_SET_EVIDENCE_SCHEMA = (
    "femx.validation.tpu_scalar_h1_collective.multilevel_process_set/v1"
)
MULTILEVEL_HIERARCHY_SCHEMA = "femx.jax.scalar_h1_multilevel_hierarchy/v1"
REPLICATED_ARRAY_REPORT_SCHEMA = "femx.jax.collective.replicated_array_report/v1"
REPLICATION_INTENT = "bounded multilevel coarse interpolation"
PARTITIONED_TRANSFER_NAMES = (
    "multilevel-cell-columns",
    "multilevel-cell-weights",
    "multilevel-owner-columns",
    "multilevel-owner-weights",
)
MULTILEVEL_EXECUTABLE_NAMES = (
    "current_forward",
    "current_setup",
    "current_vjp",
    "heat_forward",
    "heat_setup",
    "heat_vjp",
)
LOCKED_POLICY: Mapping[str, float] = {
    "diagonal_weight": 1.0,
    "minimum_relative_diagonal": 1.0e-14,
    "maximum_relative_symmetry_error": 2.0e-6,
    "maximum_coarse_condition_number": 1.0e8,
}
ITERATION_ADMISSION = (
    "PCG iterations must be strictly below same-run unpreconditioned CG for both heat and current"
)


def _fail(message: str) -> None:
    raise ValidationError(f"scalar H1 physical TPU multilevel evidence {message}")


@dataclass(frozen=True, slots=True)
class _MultilevelCase:
    baseline_iterations: int
    iterations: int
    relative_residual: float
    minimum_relative_diagonal: float
    maximum_relative_symmetry_error: float
    maximum_coarse_condition_number: float
    solution_difference: float
    baseline_solution_difference: float
    rhs_difference: float
    matrix_vjp_difference: float
    cell_rhs_vjp_difference: float


@dataclass(frozen=True, slots=True)
class _MultilevelProcess:
    process_index: int
    hierarchy_identity: tuple[object, ...]
    policy_identity: tuple[float, ...]
    partitioned_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    replicated_bytes_per_device: int
    cases: tuple[tuple[str, _MultilevelCase], ...]
    executables: tuple[tuple[str, _Executable], ...]


def _replicated_report(
    value: object,
    *,
    label: str,
    expected_name: str,
    expected_shape: tuple[int, int],
    expected_dtype: str,
    process_index: int,
    process_count: int,
    local_device_count: int,
    global_device_count: int,
) -> int:
    report = _mapping(value, label=label)
    if report.get("schema_version") != REPLICATED_ARRAY_REPORT_SCHEMA:
        _fail(f"has unsupported {label} schema")
    if report.get("name") != expected_name or report.get("dtype") != expected_dtype:
        _fail(f"has inconsistent {label} name or dtype")
    if report.get("partition_spec") != []:
        _fail(f"does not prove full replication for {label}")
    shape = tuple(
        _integer(item, label=f"{label}.global_shape", positive=True)
        for item in _sequence(report.get("global_shape"), label=f"{label}.global_shape")
    )
    if shape != expected_shape:
        _fail(f"has inconsistent {label} shape")
    for key, expected in (
        ("global_device_count", global_device_count),
        ("addressable_device_count", local_device_count),
        ("process_index", process_index),
        ("process_count", process_count),
    ):
        if (
            _integer(report.get(key), label=f"{label}.{key}", positive=key != "process_index")
            != expected
        ):
            _fail(f"has inconsistent {label} {key.replace('_', ' ')}")
    per_replica = _integer(
        report.get("logical_bytes_per_replica"),
        label=f"{label}.logical_bytes_per_replica",
        positive=True,
    )
    if per_replica != math.prod(expected_shape) * 4:
        _fail(f"has inconsistent shape-derived byte accounting for {label}")
    if (
        _integer(
            report.get("addressable_logical_bytes"),
            label=f"{label}.addressable_logical_bytes",
            positive=True,
        )
        != per_replica * local_device_count
    ):
        _fail(f"has inconsistent addressable byte accounting for {label}")
    if (
        _integer(
            report.get("global_replica_logical_bytes"),
            label=f"{label}.global_replica_logical_bytes",
            positive=True,
        )
        != per_replica * global_device_count
    ):
        _fail(f"has inconsistent global byte accounting for {label}")
    if report.get("replication_intent") != REPLICATION_INTENT:
        _fail(f"does not declare the bounded replication intent for {label}")
    return per_replica


def _case(
    value: object,
    *,
    name: str,
    baseline_iterations: int,
    policy: Mapping[str, float],
) -> _MultilevelCase:
    record = _mapping(value, label=f"multilevel.cases.{name}")
    if record.get("status") != "passed" or record.get("iteration_improved") is not True:
        _fail(f"does not admit the {name} PCG case")
    if (
        _integer(
            record.get("baseline_cg_iterations"),
            label=f"multilevel.cases.{name}.baseline_cg_iterations",
            positive=True,
        )
        != baseline_iterations
    ):
        _fail(f"changes the same-run {name} unpreconditioned baseline")
    pcg = _mapping(record.get("pcg"), label=f"multilevel.cases.{name}.pcg")
    relative_tolerance = _number(
        pcg.get("relative_tolerance"),
        label=f"multilevel.cases.{name}.pcg.relative_tolerance",
        positive=True,
    )
    absolute_tolerance = _number(
        pcg.get("absolute_tolerance"),
        label=f"multilevel.cases.{name}.pcg.absolute_tolerance",
    )
    max_iterations = _integer(
        pcg.get("max_iterations"),
        label=f"multilevel.cases.{name}.pcg.max_iterations",
        positive=True,
    )
    iterations = _integer(
        pcg.get("iterations"),
        label=f"multilevel.cases.{name}.pcg.iterations",
        positive=True,
    )
    rhs_norm = _number(
        pcg.get("rhs_norm"),
        label=f"multilevel.cases.{name}.pcg.rhs_norm",
        positive=True,
    )
    recomputed_residual = _number(
        pcg.get("recomputed_residual_norm"),
        label=f"multilevel.cases.{name}.pcg.recomputed_residual_norm",
    )
    _number(
        pcg.get("recursive_residual_norm"),
        label=f"multilevel.cases.{name}.pcg.recursive_residual_norm",
    )
    relative_residual = _number(
        pcg.get("relative_residual"),
        label=f"multilevel.cases.{name}.pcg.relative_residual",
    )
    if (
        iterations >= baseline_iterations
        or iterations > max_iterations
        or relative_residual > relative_tolerance
        or recomputed_residual > max(absolute_tolerance, relative_tolerance * rhs_norm)
        or not _boolean(pcg.get("converged"), label=f"multilevel.cases.{name}.pcg.converged")
        or _boolean(pcg.get("breakdown"), label=f"multilevel.cases.{name}.pcg.breakdown")
    ):
        _fail(f"does not satisfy the {name} PCG convergence and iteration policy")

    setup = _mapping(record.get("setup"), label=f"multilevel.cases.{name}.setup")
    minimum_diagonal = _number(
        setup.get("minimum_relative_diagonal"),
        label=f"multilevel.cases.{name}.setup.minimum_relative_diagonal",
    )
    maximum_symmetry = _number(
        setup.get("maximum_relative_symmetry_error"),
        label=f"multilevel.cases.{name}.setup.maximum_relative_symmetry_error",
    )
    maximum_condition = _number(
        setup.get("maximum_coarse_condition_number"),
        label=f"multilevel.cases.{name}.setup.maximum_coarse_condition_number",
        positive=True,
    )
    if (
        not _boolean(setup.get("valid"), label=f"multilevel.cases.{name}.setup.valid")
        or minimum_diagonal < policy["minimum_relative_diagonal"]
        or maximum_symmetry > policy["maximum_relative_symmetry_error"]
        or maximum_condition > policy["maximum_coarse_condition_number"]
    ):
        _fail(f"does not satisfy the {name} multilevel setup policy")

    numerics = _mapping(record.get("numerics"), label=f"multilevel.cases.{name}.numerics")
    finite = _mapping(
        numerics.get("finite"),
        label=f"multilevel.cases.{name}.numerics.finite",
    )
    if set(finite) != {"solution", "right_hand_side", "matrix_vjp", "cell_rhs_vjp"} or not all(
        _boolean(item, label=f"multilevel.cases.{name}.numerics.finite.{key}")
        for key, item in finite.items()
    ):
        _fail(f"does not prove finite {name} PCG forward and VJP arrays")
    differences = {
        key: _number(numerics.get(key), label=f"multilevel.cases.{name}.numerics.{key}")
        for key in (
            "solution_relative_difference",
            "solution_vs_unpreconditioned_relative_difference",
            "rhs_relative_difference",
            "matrix_vjp_relative_difference",
            "cell_rhs_vjp_relative_difference",
        )
    }
    if (
        max(
            differences["solution_relative_difference"],
            differences["solution_vs_unpreconditioned_relative_difference"],
        )
        > TOLERANCES["solution_relative_difference"]
        or differences["rhs_relative_difference"] > TOLERANCES["rhs_relative_difference"]
        or max(
            differences["matrix_vjp_relative_difference"],
            differences["cell_rhs_vjp_relative_difference"],
        )
        > TOLERANCES["vjp_relative_difference"]
    ):
        _fail(f"exceeds the admitted {name} PCG forward or VJP difference")
    if "same independent NumPy float64 matrix-free CG" not in _text(
        numerics.get("authority"),
        label=f"multilevel.cases.{name}.numerics.authority",
    ):
        _fail(f"does not preserve the independent {name} numerical authority")
    return _MultilevelCase(
        baseline_iterations=baseline_iterations,
        iterations=iterations,
        relative_residual=relative_residual,
        minimum_relative_diagonal=minimum_diagonal,
        maximum_relative_symmetry_error=maximum_symmetry,
        maximum_coarse_condition_number=maximum_condition,
        solution_difference=differences["solution_relative_difference"],
        baseline_solution_difference=differences[
            "solution_vs_unpreconditioned_relative_difference"
        ],
        rhs_difference=differences["rhs_relative_difference"],
        matrix_vjp_difference=differences["matrix_vjp_relative_difference"],
        cell_rhs_vjp_difference=differences["cell_rhs_vjp_relative_difference"],
    )


def _multilevel_process(record: Mapping[str, object]) -> _MultilevelProcess:
    base = _process(record)
    extension = _mapping(record.get("multilevel"), label="multilevel")
    if (
        extension.get("schema_version") != MULTILEVEL_EXTENSION_SCHEMA
        or extension.get("status") != "passed"
        or extension.get("collectives_valid") is not True
    ):
        _fail("requires a passed multilevel extension and collective gate")
    hierarchy = _mapping(extension.get("hierarchy"), label="multilevel.hierarchy")
    if hierarchy.get("schema_version") != MULTILEVEL_HIERARCHY_SCHEMA:
        _fail("has an unsupported hierarchy schema")
    hierarchy_sha256 = _sha256(hierarchy.get("sha256"), label="multilevel.hierarchy.sha256")
    layout_sha256 = _sha256(
        hierarchy.get("layout_sha256"),
        label="multilevel.hierarchy.layout_sha256",
    )
    if layout_sha256 != cast(str, base.problem_identity[6]):
        _fail("does not bind the hierarchy to the exact scalar layout")
    level_counts = tuple(
        _integer(item, label="multilevel.hierarchy.level_dof_counts", positive=True)
        for item in _sequence(
            hierarchy.get("level_dof_counts"),
            label="multilevel.hierarchy.level_dof_counts",
        )
    )
    if (
        len(level_counts) < 2
        or level_counts[0] != cast(int, base.problem_identity[4])
        or any(fine <= coarse for fine, coarse in pairwise(level_counts))
    ):
        _fail("has a noncanonical strictly decreasing hierarchy")
    maximum_replicated = _integer(
        hierarchy.get("maximum_replicated_dofs"),
        label="multilevel.hierarchy.maximum_replicated_dofs",
        positive=True,
    )
    if maximum_replicated != 2048 or level_counts[1] > maximum_replicated:
        _fail("changes or exceeds the bounded replicated-DOF policy")
    prolongation_hashes = tuple(
        _sha256(item, label="multilevel.hierarchy.prolongation_sha256")
        for item in _sequence(
            hierarchy.get("prolongation_sha256"),
            label="multilevel.hierarchy.prolongation_sha256",
        )
    )
    if len(prolongation_hashes) != len(level_counts) - 1:
        _fail("has inconsistent hierarchy prolongation identities")

    policy_record = _mapping(extension.get("policy"), label="multilevel.policy")
    policy = {
        key: _number(policy_record.get(key), label=f"multilevel.policy.{key}", positive=True)
        for key in LOCKED_POLICY
    }
    if policy != LOCKED_POLICY or policy_record.get("iteration_admission") != ITERATION_ADMISSION:
        _fail("changes the locked float32 multilevel admission policy")

    runtime = _mapping(record.get("runtime"), label="runtime")
    process_index = base.process_index
    process_count = cast(int, base.runtime_identity[2])
    local_device_count = cast(int, base.runtime_identity[3])
    global_device_count = cast(int, base.runtime_identity[4])
    local_mask = base.local_partition_mask
    partitioned = _mapping(
        extension.get("partitioned_transfer_reports"),
        label="multilevel.partitioned_transfer_reports",
    )
    if tuple(sorted(partitioned)) != PARTITIONED_TRANSFER_NAMES:
        _fail("requires the exact partitioned multilevel transfer report set")
    base_reports = _mapping(record.get("array_reports"), label="array_reports")
    cell_map_shape = tuple(
        _integer(item, label="array_reports.cell_local_dofs.global_shape", positive=True)
        for item in _sequence(
            _mapping(
                base_reports.get("cell_local_dofs"),
                label="array_reports.cell_local_dofs",
            ).get("global_shape"),
            label="array_reports.cell_local_dofs.global_shape",
        )
    )
    owner_mask_shape = tuple(
        _integer(item, label="array_reports.owner_mask.global_shape", positive=True)
        for item in _sequence(
            _mapping(
                base_reports.get("owner_mask"),
                label="array_reports.owner_mask",
            ).get("global_shape"),
            label="array_reports.owner_mask.global_shape",
        )
    )
    partitioned_shapes = []
    for name in PARTITIONED_TRANSFER_NAMES:
        report = _mapping(
            partitioned[name],
            label=f"multilevel.partitioned_transfer_reports.{name}",
        )
        _array_report(
            report,
            label=f"multilevel.partitioned_transfer_reports.{name}",
            process_index=process_index,
            process_count=process_count,
            partition_count=global_device_count,
            global_device_count=global_device_count,
            local_partition_mask=local_mask,
        )
        shape = tuple(cast(Sequence[int], report["global_shape"]))
        expected_dtype = "int32" if name.endswith("columns") else "float32"
        expected_shape = (*cell_map_shape, 3) if "cell" in name else (*owner_mask_shape, 3)
        if (
            report.get("schema_version") != ARRAY_REPORT_SCHEMA
            or report.get("name") != name
            or report.get("dtype") != expected_dtype
            or shape != expected_shape
        ):
            _fail(f"has inconsistent partitioned transfer semantics for {name}")
        partitioned_shapes.append((name, shape))

    replicated = _mapping(
        extension.get("replicated_transfer_reports"),
        label="multilevel.replicated_transfer_reports",
    )
    expected_replicated_names = tuple(
        sorted(
            f"multilevel-coarse-{index}-{kind}"
            for index in range(1, len(level_counts) - 1)
            for kind in ("columns", "weights")
        )
    )
    if tuple(sorted(replicated)) != expected_replicated_names:
        _fail("requires the exact replicated coarse-transfer report set")
    replicated_bytes = 0
    for index in range(1, len(level_counts) - 1):
        for kind in ("columns", "weights"):
            name = f"multilevel-coarse-{index}-{kind}"
            replicated_bytes += _replicated_report(
                replicated[name],
                label=f"multilevel.replicated_transfer_reports.{name}",
                expected_name=name,
                expected_shape=(level_counts[index], 3),
                expected_dtype="int32" if kind == "columns" else "float32",
                process_index=process_index,
                process_count=process_count,
                local_device_count=local_device_count,
                global_device_count=global_device_count,
            )

    raw_base_cases = _mapping(record.get("cases"), label="cases")
    raw_cases = _mapping(extension.get("cases"), label="multilevel.cases")
    if tuple(sorted(raw_cases)) != CASE_NAMES:
        _fail("requires exactly the heat and current multilevel cases")
    cases = []
    for name in CASE_NAMES:
        baseline_case = _mapping(raw_base_cases[name], label=f"cases.{name}")
        baseline_cg = _mapping(baseline_case.get("cg"), label=f"cases.{name}.cg")
        cases.append(
            (
                name,
                _case(
                    raw_cases[name],
                    name=name,
                    baseline_iterations=_integer(
                        baseline_cg.get("iterations"),
                        label=f"cases.{name}.cg.iterations",
                        positive=True,
                    ),
                    policy=policy,
                ),
            )
        )

    raw_executables = _mapping(extension.get("executables"), label="multilevel.executables")
    if tuple(sorted(raw_executables)) != MULTILEVEL_EXECUTABLE_NAMES:
        _fail("requires the exact multilevel setup/forward/VJP executable set")
    executables = tuple(
        (name, _executable(raw_executables[name], name=f"multilevel.{name}"))
        for name in MULTILEVEL_EXECUTABLE_NAMES
    )
    claim = _text(extension.get("claim_scope"), label="multilevel.claim_scope")
    for phrase in (
        "physical process-local record",
        "not a scaling result",
        "not Elmer parity",
        "not live HBM",
        "Spot-preemption recovery",
    ):
        if phrase not in claim:
            _fail(f"omits the bounded multilevel claim phrase {phrase!r}")
    del runtime
    return _MultilevelProcess(
        process_index=process_index,
        hierarchy_identity=(
            hierarchy_sha256,
            layout_sha256,
            level_counts,
            maximum_replicated,
            prolongation_hashes,
        ),
        policy_identity=tuple(policy[key] for key in LOCKED_POLICY),
        partitioned_shapes=tuple(partitioned_shapes),
        replicated_bytes_per_device=replicated_bytes,
        cases=tuple(cases),
        executables=executables,
    )


def aggregate_tpu_scalar_h1_multilevel_process_evidence(
    records: object,
) -> TpuScalarH1ProcessSetEvidence:
    """Admit base scalar evidence plus explicit multilevel setup, PCG, and implicit VJP."""

    base = aggregate_tpu_scalar_h1_process_evidence(records)
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        _fail("requires a process-record sequence")  # pragma: no cover - base rejects first
    parsed = tuple(
        sorted(
            (
                _multilevel_process(_mapping(record, label="process_records[]"))
                for record in cast(Sequence[object], records)
            ),
            key=lambda item: item.process_index,
        )
    )
    baseline = parsed[0]
    for process in parsed[1:]:
        if (
            process.hierarchy_identity != baseline.hierarchy_identity
            or process.policy_identity != baseline.policy_identity
            or process.partitioned_shapes != baseline.partitioned_shapes
            or process.replicated_bytes_per_device != baseline.replicated_bytes_per_device
            or process.cases != baseline.cases
        ):
            _fail("mixes inconsistent multilevel hierarchy, policy, transfer, or cases")
        for (name, executable), (baseline_name, baseline_executable) in zip(
            process.executables,
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
                _fail("mixes inconsistent multilevel compiled collective identities")

    payload = base.canonical_data()
    payload["schema_version"] = MULTILEVEL_PROCESS_SET_EVIDENCE_SCHEMA
    hierarchy_sha256, layout_sha256, level_counts, maximum_replicated, prolongation_hashes = (
        baseline.hierarchy_identity
    )
    case_payload: dict[str, object] = {}
    for name in CASE_NAMES:
        cases = [dict(process.cases)[name] for process in parsed]
        case_payload[name] = {
            "maximum_unpreconditioned_cg_iterations_across_processes": max(
                case.baseline_iterations for case in cases
            ),
            "maximum_multilevel_pcg_iterations_across_processes": max(
                case.iterations for case in cases
            ),
            "strict_iteration_improvement_on_every_process": True,
            "maximum_relative_residual_across_processes": max(
                case.relative_residual for case in cases
            ),
            "minimum_relative_diagonal_across_processes": min(
                case.minimum_relative_diagonal for case in cases
            ),
            "maximum_relative_symmetry_error_across_processes": max(
                case.maximum_relative_symmetry_error for case in cases
            ),
            "maximum_coarse_condition_number_across_processes": max(
                case.maximum_coarse_condition_number for case in cases
            ),
            "maximum_solution_relative_difference_across_processes": max(
                case.solution_difference for case in cases
            ),
            "maximum_solution_vs_unpreconditioned_relative_difference_across_processes": max(
                case.baseline_solution_difference for case in cases
            ),
            "maximum_rhs_relative_difference_across_processes": max(
                case.rhs_difference for case in cases
            ),
            "maximum_matrix_vjp_relative_difference_across_processes": max(
                case.matrix_vjp_difference for case in cases
            ),
            "maximum_cell_rhs_vjp_relative_difference_across_processes": max(
                case.cell_rhs_vjp_difference for case in cases
            ),
            "all_processes_converged_setup_valid_and_finite": True,
        }

    executable_payload: dict[str, object] = {}
    for name in MULTILEVEL_EXECUTABLE_NAMES:
        executables = [dict(process.executables)[name] for process in parsed]
        critical_path = [
            max(executable.execution_seconds[index] for executable in executables)
            for index in range(5)
        ]
        maximum_fraction = max(executable.hbm_fraction for executable in executables)
        executable_payload[name] = {
            "process_count": len(parsed),
            "execution_ordinal_critical_path_seconds": critical_path,
            "execution_ordinal_critical_path_summary_seconds": {
                "min": min(critical_path),
                "median": statistics.median(critical_path),
                "max": max(critical_path),
            },
            "maximum_compiler_peak_bytes": max(
                executable.compiler_peak_bytes for executable in executables
            ),
            "hbm_capacity_bytes_per_device": executables[0].hbm_capacity_bytes,
            "maximum_compiler_hbm_fraction": maximum_fraction,
            "worst_compiler_hbm_risk": max(
                (executable.risk for executable in executables),
                key=RISK_RANK.__getitem__,
            ),
            "stablehlo_collective_permute_count": executables[0].collective_permute_count,
            "stablehlo_all_reduce_count": executables[0].all_reduce_count,
            "stablehlo_all_gathers_absent_on_every_process": True,
            "sample_alignment": (
                "ordinal maximum across synchronized process-local samples; not a scaling result"
            ),
            "memory_scope": "compiler estimate; not live HBM usage",
        }

    payload["multilevel"] = {
        "status": "passed",
        "hierarchy": {
            "schema_version": MULTILEVEL_HIERARCHY_SCHEMA,
            "sha256": hierarchy_sha256,
            "layout_sha256": layout_sha256,
            "level_dof_counts": list(cast(tuple[int, ...], level_counts)),
            "maximum_replicated_dofs": maximum_replicated,
            "prolongation_sha256": list(cast(tuple[str, ...], prolongation_hashes)),
        },
        "policy": {
            **LOCKED_POLICY,
            "iteration_admission": ITERATION_ADMISSION,
        },
        "transfer": {
            "partitioned_report_names": list(PARTITIONED_TRANSFER_NAMES),
            "partitioned_global_shapes": {
                name: list(shape) for name, shape in baseline.partitioned_shapes
            },
            "replicated_report_names": [
                f"multilevel-coarse-{index}-{kind}"
                for index in range(1, len(cast(tuple[int, ...], level_counts)) - 1)
                for kind in ("columns", "weights")
            ],
            "replicated_logical_bytes_per_device": baseline.replicated_bytes_per_device,
            "replication_intent": REPLICATION_INTENT,
        },
        "cases": case_payload,
        "executables": executable_payload,
        "claim_scope": (
            "process-complete physical multi-host TPU explicit multilevel-PCG setup, forward, "
            "and residual-defined implicit-VJP evidence on the same bounded heat/current systems; "
            "ordinal critical-path timing is not a scaling result, compiler memory is not live "
            "HBM, and this is not Elmer parity, coupled electrothermal execution, a foundry "
            "prediction, or Spot-preemption recovery"
        ),
    }
    payload["claim_scope"] = cast(Mapping[str, object], payload["multilevel"])["claim_scope"]
    if not math.isfinite(float(baseline.replicated_bytes_per_device)):
        _fail("has invalid replicated byte accounting")  # pragma: no cover - integer invariant
    return TpuScalarH1ProcessSetEvidence(payload=payload)
