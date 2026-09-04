from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping

import pytest

from femx.core.errors import ValidationError
from femx.validation.tpu_scalar_h1_evidence import (
    ARRAY_REPORT_NAMES,
    EXECUTABLE_NAMES,
    PROCESS_EVIDENCE_SCHEMA,
    PROCESS_SET_EVIDENCE_SCHEMA,
    REAL_SCALAR_CONTRACT,
    TOLERANCES,
    _boolean,
    _integer,
    _mapping,
    _number,
    _risk_for_fraction,
    _sequence,
    _text,
    aggregate_tpu_scalar_h1_process_evidence,
)
from femx.validation.tpu_scalar_h1_multilevel_evidence import (
    ITERATION_ADMISSION,
    LOCKED_POLICY,
    MULTILEVEL_EXECUTABLE_NAMES,
    MULTILEVEL_EXTENSION_SCHEMA,
    MULTILEVEL_PROCESS_SET_EVIDENCE_SCHEMA,
    PARTITIONED_TRANSFER_NAMES,
    REPLICATION_INTENT,
    aggregate_tpu_scalar_h1_multilevel_process_evidence,
)

pytestmark = pytest.mark.unit


def _array_report(name: str, process_index: int) -> dict[str, object]:
    partitions = (0, 1) if process_index == 0 else (2, 3)
    if name == "cell_local_dofs" or name.endswith("_cell_rhs"):
        tail = (2, 3)
    elif name.endswith("_cell_stiffness"):
        tail = (2, 3, 3)
    else:
        tail = (2,)
    dtype = "int32" if name == "cell_local_dofs" else "bool" if name == "owner_mask" else "float32"
    itemsize = 1 if dtype == "bool" else 4
    per_shard_bytes = itemsize
    for extent in tail:
        per_shard_bytes *= extent
    return {
        "schema_version": "femx.jax.collective.array_report/v1",
        "name": name.replace("_", "-"),
        "global_shape": [4, *tail],
        "dtype": dtype,
        "partition_axis_name": "partition",
        "partition_count": 4,
        "global_device_count": 4,
        "process_index": process_index,
        "process_count": 2,
        "global_logical_bytes": 4 * per_shard_bytes,
        "addressable_logical_bytes": 2 * per_shard_bytes,
        "replication_intent": "none; one leading FEM partition per device",
        "addressable_shards": [
            {
                "partition_index": partition,
                "process_index": process_index,
                "device_id": partition % 2,
                "device_kind": "TPU v5 lite",
                "local_shape": [1, *tail],
                "logical_bytes": per_shard_bytes,
            }
            for partition in partitions
        ],
    }


def _timing() -> dict[str, object]:
    return {
        "schema_version": "femx.jax.collective.timing_report/v1",
        "lowering_seconds": 0.1,
        "compilation_seconds": 0.2,
        "warmup_seconds": 0.3,
        "execution_seconds": [0.5, 0.4, 0.6, 0.45, 0.55],
        "synchronization": "every timed result blocked until ready",
    }


def _memory() -> dict[str, object]:
    return {
        "schema_version": "femx.jax.collective.memory_report/v1",
        "compiler_peak_bytes": 10,
        "hbm_capacity_bytes_per_device": 100,
        "hbm_fraction": 0.1,
        "risk": "safe",
        "claim_scope": "compiler estimate; not live HBM usage",
    }


def _case(name: str) -> dict[str, object]:
    heat = name == "heat"
    return {
        "status": "passed",
        "physics": {
            "model": (
                "2D per-unit-depth representative Si/SiO2 steady heat diffusion"
                if heat
                else "2D per-unit-depth representative steady electrical conduction"
            ),
            "units": {
                "state": "K" if heat else "V",
                "coefficient": "W/(m*K)" if heat else "S/m",
                "source": "W/m^3" if heat else "A/m^3",
                "facet_load": "W/m^2" if heat else "A/m^2",
            },
            "coefficient_minimum": 1.4 if heat else 5.0e4,
            "coefficient_maximum": 130.0 if heat else 2.0e5,
            "source_maximum": 4.0e13 if heat else 0.0,
            "dirichlet_minimum": 300.0 if heat else 0.0,
            "dirichlet_maximum": 305.0 if heat else 1.0,
            "material_scope": "representative values; not foundry-calibrated material data",
        },
        "cg": {
            "relative_tolerance": 2.0e-5,
            "absolute_tolerance": 0.0,
            "max_iterations": 4000,
            "iterations": 20,
            "rhs_norm": 10.0,
            "recursive_residual_norm": 1.0e-5,
            "recomputed_residual_norm": 1.0e-5,
            "relative_residual": 1.0e-6,
            "converged": True,
            "breakdown": False,
        },
        "numerics": {
            "solution_relative_difference": 1.0e-5,
            "rhs_relative_difference": 1.0e-7,
            "matrix_vjp_relative_difference": 1.0e-4,
            "cell_rhs_vjp_relative_difference": 2.0e-4,
            "host_float32_input_vs_float64_assembly_solution_relative_difference": 3.0e-5,
            "finite": {
                "solution": True,
                "right_hand_side": True,
                "matrix_vjp": True,
                "cell_rhs_vjp": True,
            },
            "numpy_input_authority_iterations": 21,
            "numpy_input_authority_residual_norm": 1.0e-10,
            "numpy_float64_authority_iterations": 22,
            "numpy_float64_authority_residual_norm": 1.0e-11,
            "numpy_adjoint_iterations": 23,
            "numpy_adjoint_residual_norm": 1.0e-12,
            "authority": (
                "independent NumPy float64 matrix-free CG and analytic residual adjoint "
                "applied to the explicit float32 FEM inputs"
            ),
        },
    }


def _record(process_index: int) -> dict[str, object]:
    local_mask = [1, 1, 0, 0] if process_index == 0 else [0, 0, 1, 1]
    assignments = []
    for partition in range(4):
        assigned_process = partition // 2
        assignments.append(
            {
                "partition_index": partition,
                "process_index": assigned_process,
                "device_id": partition % 2,
                "platform": "tpu",
                "device_kind": "TPU v5 lite",
                "addressable": assigned_process == process_index,
            }
        )
    executable = {
        "timing": _timing(),
        "memory": _memory(),
        "stablehlo_collective_permute_count": 6,
        "stablehlo_all_reduce_count": 4,
        "stablehlo_contains_all_gather": False,
    }
    return {
        "schema_version": PROCESS_EVIDENCE_SCHEMA,
        "status": "passed",
        "provenance": {
            "run_id": "run-1",
            "profile": "spot-v5e-4",
            "source_digest": "a" * 64,
            "config_digest": "b" * 64,
        },
        "runtime": {
            "backend": "tpu",
            "jax_version": "0.10.1",
            "jaxlib_version": "0.10.1",
            "x64_enabled": False,
            "default_matmul_precision": "highest",
            "process_index": process_index,
            "process_count": 2,
            "local_device_count": 2,
            "global_device_count": 4,
            "device_kinds": ["TPU v5 lite"],
            "real_scalar_contract": REAL_SCALAR_CONTRACT,
        },
        "launch_claim": {
            "schema_version": "femx.jax.scalar_h1_collective.worker_entry_claim/v1",
            "run_id": "run-1",
            "worker_index": process_index,
            "process_index": process_index,
            "source_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "scope": (
                "worker-local scalar-H1 entry fence after Phoxla bootstrap; prevents duplicate "
                "scientific execution but does not claim controller-level launch ownership"
            ),
        },
        "problem": {
            "model": "two bounded scalar H1/P1 diffusion systems on one exact triangular mesh",
            "x_intervals": 8,
            "y_intervals": 8,
            "node_count": 81,
            "triangle_count": 128,
            "free_dof_count": 63,
            "partition_count": 4,
            "layout_sha256": "c" * 64,
            "halo_link_count": 3,
            "halo_value_count": 27,
            "cell_padding_fraction": 0.0,
            "owned_dof_padding_fraction": 0.1,
            "ghost_dof_padding_fraction": 0.2,
        },
        "mesh_report": {
            "schema_version": "femx.jax.collective.mesh_report/v1",
            "axis_name": "partition",
            "partition_count": 4,
            "global_device_count": 4,
            "addressable_device_count": 2,
            "process_count": 2,
            "is_multi_process": True,
            "layout_sha256": "c" * 64,
            "assignments": assignments,
        },
        "addressability": {
            "process_local_partition_mask": local_mask,
            "partition_addressability_counts": [1, 1, 1, 1],
            "every_partition_addressable_once": True,
        },
        "array_reports": {name: _array_report(name, process_index) for name in ARRAY_REPORT_NAMES},
        "tolerances": dict(TOLERANCES),
        "cases": {"heat": _case("heat"), "current": _case("current")},
        "executables": {name: copy.deepcopy(executable) for name in EXECUTABLE_NAMES},
        "claim_scope": (
            "physical process-complete multi-process TPU scalar H1/P1 RHS, unpreconditioned CG, "
            "and residual-defined implicit-VJP correctness evidence for bounded representative "
            "heat and current systems; not Elmer parity, coupled electrothermal execution, "
            "not a foundry prediction, or Spot-preemption recovery"
        ),
    }


def _multilevel_partitioned_report(
    name: str,
    process_index: int,
) -> dict[str, object]:
    partitions = (0, 1) if process_index == 0 else (2, 3)
    tail = (2, 3, 3) if "cell" in name else (2, 3)
    per_shard_bytes = 4
    for extent in tail:
        per_shard_bytes *= extent
    return {
        "schema_version": "femx.jax.collective.array_report/v1",
        "name": name,
        "global_shape": [4, *tail],
        "dtype": "int32" if name.endswith("columns") else "float32",
        "partition_axis_name": "partition",
        "partition_count": 4,
        "global_device_count": 4,
        "process_index": process_index,
        "process_count": 2,
        "global_logical_bytes": 4 * per_shard_bytes,
        "addressable_logical_bytes": 2 * per_shard_bytes,
        "replication_intent": "none; one leading FEM partition per device",
        "addressable_shards": [
            {
                "partition_index": partition,
                "process_index": process_index,
                "device_id": partition % 2,
                "device_kind": "TPU v5 lite",
                "local_shape": [1, *tail],
                "logical_bytes": per_shard_bytes,
            }
            for partition in partitions
        ],
    }


def _multilevel_replicated_report(name: str, process_index: int) -> dict[str, object]:
    return {
        "schema_version": "femx.jax.collective.replicated_array_report/v1",
        "name": name,
        "global_shape": [15, 3],
        "dtype": "int32" if name.endswith("columns") else "float32",
        "partition_spec": [],
        "global_device_count": 4,
        "addressable_device_count": 2,
        "process_index": process_index,
        "process_count": 2,
        "logical_bytes_per_replica": 180,
        "addressable_logical_bytes": 360,
        "global_replica_logical_bytes": 720,
        "replication_intent": REPLICATION_INTENT,
    }


def _multilevel_case() -> dict[str, object]:
    return {
        "status": "passed",
        "baseline_cg_iterations": 20,
        "iteration_improved": True,
        "pcg": {
            "relative_tolerance": 2.0e-5,
            "absolute_tolerance": 0.0,
            "max_iterations": 4000,
            "iterations": 10,
            "rhs_norm": 10.0,
            "recursive_residual_norm": 1.0e-5,
            "recomputed_residual_norm": 1.0e-5,
            "relative_residual": 1.0e-6,
            "converged": True,
            "breakdown": False,
        },
        "setup": {
            "valid": True,
            "minimum_relative_diagonal": 1.0e-3,
            "maximum_relative_symmetry_error": 1.0e-7,
            "maximum_coarse_condition_number": 100.0,
        },
        "numerics": {
            "solution_relative_difference": 1.0e-5,
            "solution_vs_unpreconditioned_relative_difference": 2.0e-5,
            "rhs_relative_difference": 1.0e-7,
            "matrix_vjp_relative_difference": 1.0e-4,
            "cell_rhs_vjp_relative_difference": 2.0e-4,
            "finite": {
                "solution": True,
                "right_hand_side": True,
                "matrix_vjp": True,
                "cell_rhs_vjp": True,
            },
            "authority": (
                "the same independent NumPy float64 matrix-free CG and analytic residual adjoint "
                "used by the unpreconditioned physical witness"
            ),
        },
    }


def _multilevel_record(process_index: int) -> dict[str, object]:
    record = _record(process_index)
    executable = {
        "timing": _timing(),
        "memory": _memory(),
        "stablehlo_collective_permute_count": 12,
        "stablehlo_all_reduce_count": 8,
        "stablehlo_contains_all_gather": False,
    }
    record["multilevel"] = {
        "schema_version": MULTILEVEL_EXTENSION_SCHEMA,
        "status": "passed",
        "hierarchy": {
            "schema_version": "femx.jax.scalar_h1_multilevel_hierarchy/v1",
            "sha256": "d" * 64,
            "layout_sha256": "c" * 64,
            "level_dof_counts": [63, 15, 3],
            "maximum_replicated_dofs": 2048,
            "prolongation_sha256": ["e" * 64, "f" * 64],
        },
        "policy": {**LOCKED_POLICY, "iteration_admission": ITERATION_ADMISSION},
        "partitioned_transfer_reports": {
            name: _multilevel_partitioned_report(name, process_index)
            for name in PARTITIONED_TRANSFER_NAMES
        },
        "replicated_transfer_reports": {
            name: _multilevel_replicated_report(name, process_index)
            for name in (
                "multilevel-coarse-1-columns",
                "multilevel-coarse-1-weights",
            )
        },
        "cases": {
            "heat": _multilevel_case(),
            "current": _multilevel_case(),
        },
        "executables": {name: copy.deepcopy(executable) for name in MULTILEVEL_EXECUTABLE_NAMES},
        "collectives_valid": True,
        "claim_scope": (
            "physical process-local record for explicit multilevel-PCG setup, forward, and "
            "residual-defined implicit VJP; timing is not a scaling result, compiler memory is "
            "not live HBM, and this is not Elmer parity or Spot-preemption recovery"
        ),
    }
    return record


def _set(record: dict[str, object], path: str, value: object) -> None:
    parts = path.split(".")
    current: object = record
    for part in parts[:-1]:
        assert isinstance(current, dict)
        current = current[part]
    assert isinstance(current, dict)
    current[parts[-1]] = value


def test_process_set_aggregate_is_canonical_process_complete_and_detached() -> None:
    evidence = aggregate_tpu_scalar_h1_process_evidence([_record(1), _record(0)])
    data = evidence.canonical_data()
    assert data["schema_version"] == PROCESS_SET_EVIDENCE_SCHEMA
    assert data["status"] == "passed"
    assert data["runtime"]["process_indexes"] == [0, 1]  # type: ignore[index]
    assert data["addressability"]["combined_partition_addressability_counts"] == [  # type: ignore[index]
        1,
        1,
        1,
        1,
    ]
    assert set(data["cases"]) == {"heat", "current"}  # type: ignore[arg-type]
    assert set(data["executables"]) == set(EXECUTABLE_NAMES)  # type: ignore[arg-type]
    assert len(evidence.digest()) == 64
    assert json.loads(evidence.canonical_json()) == data
    data["status"] = "mutated"
    assert evidence.canonical_data()["status"] == "passed"


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: _mapping({1: "value"}, label="value"), "string keys"),
        (lambda: _sequence("value", label="value"), "to be an array"),
        (lambda: _text(" ", label="value"), "nonempty trimmed"),
        (lambda: _integer(True, label="value"), "nonnegative integer"),
        (lambda: _integer(0, label="value", positive=True), "positive integer"),
        (lambda: _number(True, label="value"), "finite number"),
        (lambda: _number(float("nan"), label="value"), "finite nonnegative"),
        (lambda: _number(0.0, label="value", positive=True), "to be positive"),
        (lambda: _boolean(1, label="value"), "to be boolean"),
    ],
)
def test_strict_json_primitive_decoders_fail_closed(
    call: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        call()


def test_memory_risk_bands_are_exact() -> None:
    assert _risk_for_fraction(0.69) == "safe"
    assert _risk_for_fraction(0.70) == "elevated"
    assert _risk_for_fraction(0.85) == "high"
    assert _risk_for_fraction(0.95) == "extreme"


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("schema_version", "v2", "unsupported process-record schema"),
        ("status", "failed", "did not pass"),
        ("provenance.source_digest", "bad", "canonical lowercase SHA-256"),
        ("runtime.process_index", 2, "outside"),
        ("runtime.process_count", 1, "multi-process"),
        ("runtime.backend", "cpu", "physical TPU"),
        ("runtime.x64_enabled", True, "physical TPU"),
        ("runtime.device_kinds", [], "canonical nonempty"),
        ("runtime.real_scalar_contract", {}, "exact physical TPU float32"),
        ("launch_claim.schema_version", "v2", "worker-entry"),
        ("launch_claim.run_id", "other", "inconsistent with provenance"),
        ("launch_claim.scope", "missing", "duplicate-entry"),
        ("problem.model", "other", "unsupported scalar problem"),
        ("problem.partition_count", 3, "every TPU device"),
        ("problem.cell_padding_fraction", 2.0, "invalid problem"),
        ("mesh_report.schema_version", "v2", "Mesh-report"),
        ("mesh_report.axis_name", "other", "partition Mesh"),
        ("mesh_report.layout_sha256", "d" * 64, "layout identities"),
        ("addressability.process_local_partition_mask", [1, 2, 0, 0], "noncanonical"),
        ("addressability.partition_addressability_counts", [1, 1, 1, 0], "exactly once"),
        ("tolerances.solution_relative_difference", 1.0, "locked scalar TPU"),
        ("cases.heat.status", "failed", "heat case"),
        ("cases.heat.physics.material_scope", "calibrated", "material authority"),
        ("cases.heat.physics.source_maximum", 0.0, "declared heat source"),
        ("cases.current.physics.source_maximum", 1.0, "declared current source"),
        ("cases.heat.cg.iterations", 4001, "impossible heat CG"),
        ("cases.heat.cg.relative_residual", 1.0, "recomputed heat CG"),
        ("cases.heat.cg.converged", False, "heat CG convergence"),
        ("cases.heat.cg.breakdown", True, "heat CG breakdown"),
        ("cases.heat.numerics.solution_relative_difference", 1.0, "solution difference"),
        ("cases.heat.numerics.rhs_relative_difference", 1.0, "RHS difference"),
        ("cases.heat.numerics.matrix_vjp_relative_difference", 1.0, "VJP difference"),
        (
            "cases.heat.numerics.host_float32_input_vs_float64_assembly_solution_relative_difference",
            1.0,
            "host-precision",
        ),
        ("cases.heat.numerics.authority", "unknown", "independent heat"),
        ("executables.heat_forward.timing.schema_version", "v2", "timing schema"),
        ("executables.heat_forward.timing.synchronization", "none", "synchronized"),
        ("executables.heat_forward.timing.execution_seconds", [0.1], "exactly five"),
        ("executables.heat_forward.memory.schema_version", "v2", "memory schema"),
        ("executables.heat_forward.memory.hbm_fraction", 0.2, "memory fraction"),
        ("executables.heat_forward.memory.risk", "high", "memory risk"),
        ("executables.heat_forward.memory.claim_scope", "live HBM", "overstates"),
        ("executables.heat_forward.stablehlo_contains_all_gather", True, "all-gather"),
        ("executables.heat_forward.stablehlo_all_reduce_count", 0, "positive integer"),
        ("claim_scope", "process-complete only", "claim-scope"),
    ],
)
def test_process_record_mutations_fail_closed(path: str, value: object, message: str) -> None:
    records = [_record(0), _record(1)]
    _set(records[0], path, value)
    with pytest.raises(ValidationError, match=message):
        aggregate_tpu_scalar_h1_process_evidence(records)


def test_array_reports_and_mesh_assignments_fail_closed() -> None:
    records = [_record(0), _record(1)]
    reports = records[0]["array_reports"]
    assert isinstance(reports, dict)
    reports.pop("owner_mask")
    with pytest.raises(ValidationError, match="exact scalar distributed-array report set"):
        aggregate_tpu_scalar_h1_process_evidence(records)

    records = [_record(0), _record(1)]
    report = records[0]["array_reports"]
    assert isinstance(report, dict)
    first = report["owner_mask"]
    assert isinstance(first, dict)
    first["replication_intent"] = "replicated"
    with pytest.raises(ValidationError, match="non-replicated"):
        aggregate_tpu_scalar_h1_process_evidence(records)

    records = [_record(0), _record(1)]
    mesh = records[0]["mesh_report"]
    assert isinstance(mesh, dict)
    assignments = mesh["assignments"]
    assert isinstance(assignments, list)
    assignment = assignments[0]
    assert isinstance(assignment, dict)
    assignment["addressable"] = False
    with pytest.raises(ValidationError, match="process-local mask"):
        aggregate_tpu_scalar_h1_process_evidence(records)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("array_reports.owner_mask.schema_version", "v2", "unsupported"),
        ("array_reports.owner_mask.partition_axis_name", "other", "partition axis"),
        ("array_reports.owner_mask.partition_count", 3, "inconsistent"),
        ("array_reports.owner_mask.global_shape", [3, 2], "leading shape"),
        ("array_reports.owner_mask.addressable_logical_bytes", 64, "impossible"),
        (
            "array_reports.owner_mask.addressable_shards",
            [],
            "process-local mask",
        ),
    ],
)
def test_array_report_mutations_fail_closed(path: str, value: object, message: str) -> None:
    records = [_record(0), _record(1)]
    _set(records[0], path, value)
    with pytest.raises(ValidationError, match=message):
        aggregate_tpu_scalar_h1_process_evidence(records)


def test_array_shard_identity_shape_and_bytes_fail_closed() -> None:
    for key, value, message in (
        ("partition_index", 4, "out-of-range"),
        ("process_index", 1, "foreign-process"),
        ("local_shape", [2, 2], "local shard shape"),
        ("logical_bytes", 7, "byte accounting"),
    ):
        records = [_record(0), _record(1)]
        reports = records[0]["array_reports"]
        assert isinstance(reports, dict)
        report = reports["owner_mask"]
        assert isinstance(report, dict)
        shards = report["addressable_shards"]
        assert isinstance(shards, list)
        shard = shards[0]
        assert isinstance(shard, dict)
        shard[key] = value
        with pytest.raises(ValidationError, match=message):
            aggregate_tpu_scalar_h1_process_evidence(records)


def test_case_and_mesh_structure_mutations_fail_closed() -> None:
    mutations = (
        ("cases.heat.physics.coefficient_minimum", 200.0, "coefficient bounds"),
        ("cases.heat.physics.dirichlet_minimum", 400.0, "Dirichlet bounds"),
        ("cases.heat.numerics.finite.solution", False, "finite heat"),
        ("mesh_report.addressable_device_count", 1, "inconsistent Mesh"),
        ("addressability.process_local_partition_mask", [1, 0, 0, 0], "local device count"),
        ("mesh_report.assignments", [], "one Mesh assignment"),
        ("cases", {"heat": _case("heat")}, "exactly the heat and current"),
        ("executables", {}, "exact scalar forward/VJP"),
    )
    for path, value, message in mutations:
        records = [_record(0), _record(1)]
        _set(records[0], path, value)
        with pytest.raises(ValidationError, match=message):
            aggregate_tpu_scalar_h1_process_evidence(records)


def test_mesh_assignment_order_platform_and_uniqueness_fail_closed() -> None:
    for key, value, message in (
        ("partition_index", 1, "noncanonical Mesh assignment"),
        ("platform", "cpu", "TPU runtime"),
        ("device_id", 1, "unique TPU device"),
    ):
        records = [_record(0), _record(1)]
        mesh = records[0]["mesh_report"]
        assert isinstance(mesh, dict)
        assignments = mesh["assignments"]
        assert isinstance(assignments, list)
        assignment = assignments[0]
        assert isinstance(assignment, dict)
        assignment[key] = value
        with pytest.raises(ValidationError, match=message):
            aggregate_tpu_scalar_h1_process_evidence(records)


def test_process_set_rejects_missing_duplicate_or_mixed_records() -> None:
    with pytest.raises(ValidationError, match="nonempty sequence"):
        aggregate_tpu_scalar_h1_process_evidence([])
    with pytest.raises(ValidationError, match="exactly one process record"):
        aggregate_tpu_scalar_h1_process_evidence([_record(0)])
    with pytest.raises(ValidationError, match="canonical JAX process index set"):
        aggregate_tpu_scalar_h1_process_evidence([_record(0), _record(0)])
    mixed = [_record(0), _record(1)]
    provenance = mixed[1]["provenance"]
    assert isinstance(provenance, dict)
    provenance["run_id"] = "other"
    launch = mixed[1]["launch_claim"]
    assert isinstance(launch, dict)
    launch["run_id"] = "other"
    with pytest.raises(ValidationError, match="different deployed inputs"):
        aggregate_tpu_scalar_h1_process_evidence(mixed)


def test_process_set_rejects_duplicate_workers_and_mixed_identities() -> None:
    records = [_record(0), _record(1)]
    launch = records[1]["launch_claim"]
    assert isinstance(launch, dict)
    launch["worker_index"] = 0
    with pytest.raises(ValidationError, match="unique worker-entry"):
        aggregate_tpu_scalar_h1_process_evidence(records)

    mutations = (
        ("runtime.jax_version", "different", "different TPU runtimes"),
        ("problem.x_intervals", 9, "different scalar problems"),
        ("mesh_report.assignments", None, "placeholder"),
        ("cases.heat.physics.coefficient_maximum", 131.0, "heat/current problem identities"),
        (
            "executables.heat_forward.memory.hbm_capacity_bytes_per_device",
            200,
            "placeholder",
        ),
    )
    for path, value, message in mutations:
        records = [_record(0), _record(1)]
        if path == "mesh_report.assignments":
            mesh = records[1]["mesh_report"]
            assert isinstance(mesh, dict)
            assignments = mesh["assignments"]
            assert isinstance(assignments, list)
            assignment = assignments[0]
            assert isinstance(assignment, dict)
            assignment["device_id"] = 7
            expected = "global Mesh assignments"
        elif path.endswith("hbm_capacity_bytes_per_device"):
            _set(records[1], path, value)
            _set(records[1], "executables.heat_forward.memory.compiler_peak_bytes", 20)
            expected = "compiled collective identities"
        else:
            _set(records[1], path, value)
            expected = message
        with pytest.raises(ValidationError, match=expected):
            aggregate_tpu_scalar_h1_process_evidence(records)


@pytest.mark.parametrize(
    ("fraction", "risk"),
    [(0.75, "elevated"), (0.90, "high"), (0.96, "extreme")],
)
def test_compiler_memory_risk_admission(fraction: float, risk: str) -> None:
    records = [_record(0), _record(1)]
    for record in records:
        for executable_name in EXECUTABLE_NAMES:
            _set(
                record,
                f"executables.{executable_name}.memory.compiler_peak_bytes",
                int(100 * fraction),
            )
            _set(record, f"executables.{executable_name}.memory.hbm_fraction", fraction)
            _set(record, f"executables.{executable_name}.memory.risk", risk)
    if risk == "elevated":
        evidence = aggregate_tpu_scalar_h1_process_evidence(records)
        assert (
            evidence.canonical_data()["executables"]["heat_forward"][  # type: ignore[index]
                "worst_compiler_hbm_risk"
            ]
            == "elevated"
        )
    else:
        with pytest.raises(ValidationError, match="exceeds the admitted"):
            aggregate_tpu_scalar_h1_process_evidence(records)


def test_process_set_rejects_noncanonical_json_record() -> None:
    record = _record(0)
    record["extra"] = {"bad": object()}
    with pytest.raises(ValidationError, match="not canonical JSON"):
        aggregate_tpu_scalar_h1_process_evidence([record, _record(1)])


def test_mapping_root_is_required() -> None:
    with pytest.raises(ValidationError, match=r"process_records\[\]"):
        aggregate_tpu_scalar_h1_process_evidence([None])  # type: ignore[list-item]


def test_records_type_contract_rejects_text() -> None:
    with pytest.raises(ValidationError, match="nonempty sequence"):
        aggregate_tpu_scalar_h1_process_evidence("records")  # type: ignore[arg-type]


def test_process_record_is_a_mapping_fixture() -> None:
    assert isinstance(_record(0), Mapping)


def test_multilevel_process_set_aggregate_preserves_base_and_strategy_evidence() -> None:
    evidence = aggregate_tpu_scalar_h1_multilevel_process_evidence(
        [_multilevel_record(1), _multilevel_record(0)]
    )
    data = evidence.canonical_data()
    assert data["schema_version"] == MULTILEVEL_PROCESS_SET_EVIDENCE_SCHEMA
    assert data["status"] == "passed"
    multilevel = data["multilevel"]
    assert isinstance(multilevel, dict)
    assert multilevel["status"] == "passed"
    assert multilevel["hierarchy"]["level_dof_counts"] == [63, 15, 3]  # type: ignore[index]
    assert multilevel["transfer"]["replicated_logical_bytes_per_device"] == 360  # type: ignore[index]
    assert multilevel["transfer"]["partitioned_global_shapes"] == {  # type: ignore[index]
        "multilevel-cell-columns": [4, 2, 3, 3],
        "multilevel-cell-weights": [4, 2, 3, 3],
        "multilevel-owner-columns": [4, 2, 3],
        "multilevel-owner-weights": [4, 2, 3],
    }
    for case in multilevel["cases"].values():  # type: ignore[union-attr]
        assert case["maximum_unpreconditioned_cg_iterations_across_processes"] == 20
        assert case["maximum_multilevel_pcg_iterations_across_processes"] == 10
        assert case["strict_iteration_improvement_on_every_process"] is True
    assert set(multilevel["executables"]) == set(MULTILEVEL_EXECUTABLE_NAMES)  # type: ignore[arg-type]
    assert "process-complete physical multi-host TPU" in data["claim_scope"]


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("multilevel.schema_version", "v2", "passed multilevel extension"),
        ("multilevel.status", "failed", "passed multilevel extension"),
        ("multilevel.collectives_valid", False, "passed multilevel extension"),
        ("multilevel.hierarchy.schema_version", "v2", "hierarchy schema"),
        ("multilevel.hierarchy.layout_sha256", "a" * 64, "exact scalar layout"),
        ("multilevel.hierarchy.level_dof_counts", [63, 15, 15], "strictly decreasing"),
        ("multilevel.hierarchy.maximum_replicated_dofs", 16, "bounded replicated-DOF"),
        ("multilevel.hierarchy.prolongation_sha256", [], "prolongation identities"),
        ("multilevel.policy.maximum_relative_symmetry_error", 1.0e-4, "locked float32"),
        ("multilevel.policy.iteration_admission", "automatic", "locked float32"),
        ("multilevel.partitioned_transfer_reports", {}, "exact partitioned"),
        (
            "multilevel.partitioned_transfer_reports.multilevel-owner-columns.dtype",
            "float32",
            "transfer semantics",
        ),
        (
            "multilevel.partitioned_transfer_reports.multilevel-owner-columns.addressable_logical_bytes",
            1,
            "byte accounting",
        ),
        ("multilevel.replicated_transfer_reports", {}, "exact replicated"),
        (
            "multilevel.replicated_transfer_reports.multilevel-coarse-1-columns.schema_version",
            "v2",
            "unsupported",
        ),
        (
            "multilevel.replicated_transfer_reports.multilevel-coarse-1-columns.name",
            "other",
            "name or dtype",
        ),
        (
            "multilevel.replicated_transfer_reports.multilevel-coarse-1-columns.partition_spec",
            ["partition"],
            "full replication",
        ),
        (
            "multilevel.replicated_transfer_reports.multilevel-coarse-1-columns.global_shape",
            [14, 3],
            "shape",
        ),
        (
            "multilevel.replicated_transfer_reports.multilevel-coarse-1-columns.global_device_count",
            3,
            "global device count",
        ),
        (
            "multilevel.replicated_transfer_reports.multilevel-coarse-1-columns.addressable_logical_bytes",
            1,
            "addressable byte accounting",
        ),
        (
            "multilevel.replicated_transfer_reports.multilevel-coarse-1-columns.logical_bytes_per_replica",
            4,
            "shape-derived byte accounting",
        ),
        (
            "multilevel.replicated_transfer_reports.multilevel-coarse-1-columns.global_replica_logical_bytes",
            1,
            "global byte accounting",
        ),
        (
            "multilevel.replicated_transfer_reports.multilevel-coarse-1-columns.replication_intent",
            "implicit",
            "bounded replication intent",
        ),
        ("multilevel.cases", {"heat": _multilevel_case()}, "exactly the heat and current"),
        ("multilevel.cases.heat.status", "failed", "heat PCG case"),
        ("multilevel.cases.heat.baseline_cg_iterations", 21, "same-run heat"),
        ("multilevel.cases.heat.pcg.iterations", 20, "heat PCG convergence"),
        ("multilevel.cases.heat.pcg.relative_residual", 1.0, "heat PCG convergence"),
        ("multilevel.cases.heat.pcg.converged", False, "heat PCG convergence"),
        ("multilevel.cases.heat.pcg.breakdown", True, "heat PCG convergence"),
        ("multilevel.cases.heat.setup.valid", False, "heat multilevel setup"),
        (
            "multilevel.cases.heat.setup.maximum_relative_symmetry_error",
            3.0e-6,
            "heat multilevel setup",
        ),
        (
            "multilevel.cases.heat.setup.maximum_coarse_condition_number",
            2.0e8,
            "heat multilevel setup",
        ),
        ("multilevel.cases.heat.numerics.finite.solution", False, "finite heat PCG"),
        (
            "multilevel.cases.heat.numerics.solution_relative_difference",
            1.0,
            "forward or VJP difference",
        ),
        ("multilevel.cases.heat.numerics.authority", "unknown", "independent heat"),
        ("multilevel.executables", {}, "exact multilevel"),
        (
            "multilevel.executables.heat_forward.stablehlo_contains_all_gather",
            True,
            "all-gather",
        ),
        ("multilevel.claim_scope", "physical only", "bounded multilevel claim"),
    ],
)
def test_multilevel_process_record_mutations_fail_closed(
    path: str,
    value: object,
    message: str,
) -> None:
    records = [_multilevel_record(0), _multilevel_record(1)]
    _set(records[0], path, value)
    with pytest.raises(ValidationError, match=message):
        aggregate_tpu_scalar_h1_multilevel_process_evidence(records)


def test_multilevel_process_set_rejects_mixed_strategy_identities() -> None:
    mutations = (
        ("multilevel.hierarchy.sha256", "0" * 64, "hierarchy, policy, transfer, or cases"),
        ("multilevel.cases.heat.pcg.iterations", 9, "hierarchy, policy, transfer, or cases"),
        (
            "multilevel.executables.heat_forward.stablehlo_all_reduce_count",
            9,
            "compiled collective identities",
        ),
    )
    for path, value, message in mutations:
        records = [_multilevel_record(0), _multilevel_record(1)]
        _set(records[1], path, value)
        with pytest.raises(ValidationError, match=message):
            aggregate_tpu_scalar_h1_multilevel_process_evidence(records)
