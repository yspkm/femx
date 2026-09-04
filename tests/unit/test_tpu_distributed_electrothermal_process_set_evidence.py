from __future__ import annotations

import copy
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from scripts.aggregate_tpu_distributed_electrothermal_evidence import main

from femx.core.errors import ValidationError
from femx.validation.tpu_distributed_electrothermal_evidence import (
    COUPLED_ADJOINT_POLICY,
    COUPLED_ITERATION_POLICY,
    EXECUTABLE_NAMES,
    PARTITIONED_ARRAY_REPORT_NAMES,
    PROCESS_EVIDENCE_SCHEMA,
    PROCESS_SET_EVIDENCE_SCHEMA,
    REAL_SCALAR_CONTRACT,
    REPLICATED_ARRAY_REPORT_NAMES,
    SCALAR_CG_POLICY,
    TOLERANCES,
    _boolean,
    _canonical_json,
    _git_revision,
    _integer,
    _mapping,
    _number,
    _risk_for_fraction,
    _sequence,
    _sha256,
    _shape,
    _text,
    aggregate_tpu_distributed_electrothermal_process_evidence,
)

pytestmark = pytest.mark.unit

_PARTITIONED_SHAPES = {
    "authority-potential": (4, 3),
    "authority-temperature": (4, 3),
    "input-basis-gradients": (4, 2, 3, 2),
    "input-cell-areas": (4, 2),
    "input-cell-local-dofs": (4, 2, 3),
    "input-cell-mask": (4, 2),
    "input-current-cell-load-base": (4, 2, 3),
    "input-current-cell-load-weights": (4, 2, 3, 2),
    "input-current-conductivity-base": (4, 2),
    "input-current-conductivity-weights": (4, 2, 2),
    "input-current-dirichlet-base": (4, 2, 3),
    "input-current-dirichlet-weights": (4, 2, 3, 2),
    "input-feedback-coefficient-base": (4, 2, 3),
    "input-feedback-coefficient-weights": (4, 2, 3, 1),
    "input-feedback-reference-base": (4, 2, 3),
    "input-feedback-reference-weights": (4, 2, 3, 1),
    "input-owner-mask": (4, 3),
    "input-thermal-cell-load-base": (4, 2, 3),
    "input-thermal-cell-load-weights": (4, 2, 3, 1),
    "input-thermal-conductivity-base": (4, 2),
    "input-thermal-conductivity-weights": (4, 2, 1),
    "input-thermal-dirichlet-base": (4, 2, 3),
    "input-thermal-dirichlet-weights": (4, 2, 3, 1),
    "input-unit-stiffness": (4, 2, 3, 3),
    "temperature-cotangent": (4, 3),
}
_REPLICATED_SHAPES = {
    "authority-current-gradient": (2,),
    "authority-feedback-gradient": (1,),
    "authority-thermal-gradient": (1,),
    "current-parameters": (2,),
    "feedback-parameters": (1,),
    "input-current-lower-bounds": (2,),
    "input-current-reference-base": (1,),
    "input-current-reference-weights": (2,),
    "input-current-upper-bounds": (2,),
    "input-feedback-lower-bounds": (1,),
    "input-feedback-upper-bounds": (1,),
    "input-thermal-lower-bounds": (1,),
    "input-thermal-reference-base": (1,),
    "input-thermal-reference-weights": (1,),
    "input-thermal-upper-bounds": (1,),
    "thermal-parameters": (1,),
}


def _nested(record: dict[str, object], *keys: str) -> Any:
    value: Any = record
    for key in keys:
        value = value[key]
    return value


def _dtype(name: str) -> str:
    if name == "input-cell-local-dofs":
        return "int32"
    if name in {"input-cell-mask", "input-owner-mask"}:
        return "bool"
    return "float32"


def _bytes(shape: tuple[int, ...], dtype: str) -> int:
    value = 1 if dtype == "bool" else 4
    for extent in shape:
        value *= extent
    return value


def _partitioned_report(name: str, process_index: int) -> dict[str, object]:
    shape = _PARTITIONED_SHAPES[name]
    dtype = _dtype(name)
    local_shape = (1, *shape[1:])
    local_bytes = _bytes(local_shape, dtype)
    partitions = (0, 1) if process_index == 0 else (2, 3)
    return {
        "schema_version": "femx.jax.collective.array_report/v1",
        "name": name,
        "global_shape": list(shape),
        "dtype": dtype,
        "partition_axis_name": "partition",
        "partition_count": 4,
        "global_device_count": 4,
        "process_index": process_index,
        "process_count": 2,
        "global_logical_bytes": _bytes(shape, dtype),
        "addressable_logical_bytes": 2 * local_bytes,
        "replication_intent": "none; one leading FEM partition per device",
        "addressable_shards": [
            {
                "partition_index": partition,
                "process_index": process_index,
                "device_id": partition,
                "device_kind": "TPU v4",
                "local_shape": list(local_shape),
                "logical_bytes": local_bytes,
            }
            for partition in partitions
        ],
    }


def _replicated_report(name: str, process_index: int) -> dict[str, object]:
    shape = _REPLICATED_SHAPES[name]
    logical_bytes = _bytes(shape, "float32")
    return {
        "schema_version": "femx.jax.collective.replicated_array_report/v1",
        "name": name,
        "global_shape": list(shape),
        "dtype": "float32",
        "partition_spec": [],
        "global_device_count": 4,
        "addressable_device_count": 2,
        "process_index": process_index,
        "process_count": 2,
        "logical_bytes_per_replica": logical_bytes,
        "addressable_logical_bytes": 2 * logical_bytes,
        "global_replica_logical_bytes": 4 * logical_bytes,
        "replication_intent": (
            "bounded plan scalars, parameter vectors, and float64-authority projections"
        ),
    }


def _executable(name: str, process_index: int) -> dict[str, object]:
    samples = [0.5, 0.4, 0.6, 0.45, 0.55]
    hlo_index = EXECUTABLE_NAMES.index(name) + 1
    return {
        "timing": {
            "schema_version": "femx.jax.collective.timing_report/v1",
            "lowering_seconds": 0.1 + process_index * 0.01,
            "compilation_seconds": 0.2 + process_index * 0.01,
            "warmup_seconds": 0.3 + process_index * 0.01,
            "execution_seconds": samples,
            "execution_min_seconds": 0.4,
            "execution_median_seconds": 0.5,
            "execution_max_seconds": 0.6,
            "synchronization": "every timed result blocked until ready",
        },
        "memory": {
            "schema_version": "femx.jax.collective.memory_report/v1",
            "argument_bytes": 100,
            "output_bytes": 20,
            "alias_bytes": 0,
            "temporary_bytes": 80,
            "generated_code_bytes": 50,
            "compiler_peak_bytes": 200,
            "hbm_capacity_bytes_per_device": 1000,
            "hbm_fraction": 0.2,
            "risk": "safe",
            "claim_scope": "compiler estimate; not live HBM usage",
        },
        "stablehlo": {
            "sha256": str(hlo_index) * 64,
            "contains_all_gather": False,
            "collective_permute_count": 10 * hlo_index,
            "all_reduce_count": 5 * hlo_index,
        },
    }


def _record(process_index: int) -> dict[str, object]:
    local_mask = [1, 1, 0, 0] if process_index == 0 else [0, 0, 1, 1]
    assignments = [
        {
            "partition_index": partition,
            "process_index": partition // 2,
            "device_id": partition,
            "platform": "tpu",
            "device_kind": "TPU v4",
            "addressable": partition // 2 == process_index,
        }
        for partition in range(4)
    ]
    objective = 1.0
    authority_objective = 1.0001
    return {
        "schema_version": PROCESS_EVIDENCE_SCHEMA,
        "status": "passed",
        "provenance": {
            "run_id": "private-run",
            "profile": "private-profile",
            "source_commit": "a" * 40,
            "source_digest": "b" * 64,
            "config_digest": "c" * 64,
        },
        "runtime": {
            "backend": "tpu",
            "jax_version": "0.11.0",
            "jaxlib_version": "0.11.0",
            "x64_enabled": False,
            "default_matmul_precision": "highest",
            "process_index": process_index,
            "process_count": 2,
            "local_device_count": 2,
            "global_device_count": 4,
            "device_kinds": ["TPU v4"],
            "real_scalar_contract": dict(REAL_SCALAR_CONTRACT),
        },
        "launch_claim": {
            "schema_version": ("femx.jax.distributed_electrothermal.worker_entry_claim/v1"),
            "run_id": "private-run",
            "worker_index": process_index,
            "process_index": process_index,
            "source_sha256": "b" * 64,
            "config_sha256": "c" * 64,
            "scope": (
                "worker-local coupled electrothermal entry fence after distributed bootstrap"
            ),
        },
        "plan": {
            "schema_version": "femx.jax.distributed_electrothermal/v1",
            "sha256": "d" * 64,
            "arrays_sha256": "e" * 64,
            "layout_sha256": "f" * 64,
            "partition_count": 4,
            "node_count": 9,
            "triangle_count": 8,
            "free_dof_count": 3,
            "host_input_replication": (
                "bounded complete plan file exists on every worker; only addressable partition "
                "slices are transferred for partition-leading device arrays"
            ),
        },
        "mesh_report": {
            "schema_version": "femx.jax.collective.mesh_report/v1",
            "axis_name": "partition",
            "partition_count": 4,
            "global_device_count": 4,
            "addressable_device_count": 2,
            "process_count": 2,
            "is_multi_process": True,
            "layout_sha256": "f" * 64,
            "assignments": assignments,
        },
        "addressability": {
            "process_local_partition_mask": local_mask,
            "partition_addressability_counts": [1, 1, 1, 1],
            "every_partition_addressable_once": True,
        },
        "partitioned_array_reports": {
            name: _partitioned_report(name, process_index)
            for name in PARTITIONED_ARRAY_REPORT_NAMES
        },
        "replicated_array_reports": {
            name: _replicated_report(name, process_index) for name in REPLICATED_ARRAY_REPORT_NAMES
        },
        "tolerances": dict(TOLERANCES),
        "policies": {
            "coupled_iteration": dict(COUPLED_ITERATION_POLICY),
            "scalar_cg": json.loads(json.dumps(dict(SCALAR_CG_POLICY))),
            "coupled_adjoint": dict(COUPLED_ADJOINT_POLICY),
        },
        "numerics": {
            "iterations": 10,
            "forward_converged": True,
            "finite": True,
            "current_linear_iterations": 30,
            "current_linear_recursive_residual": 1.0e-4,
            "current_linear_recomputed_residual": 2.0e-4,
            "current_linear_relative_residual": 1.0e-5,
            "current_linear_backward_error": 1.0e-7,
            "current_linear_converged": True,
            "current_linear_breakdown": False,
            "heat_linear_iterations": 40,
            "heat_linear_recursive_residual": 2.0e-4,
            "heat_linear_recomputed_residual": 3.0e-4,
            "heat_linear_relative_residual": 3.0e-5,
            "heat_linear_backward_error": 2.0e-7,
            "heat_linear_converged": True,
            "heat_linear_breakdown": False,
            "current_residual_error": 1.0e-5,
            "heat_residual_error": 2.0e-5,
            "adjoint_converged": True,
            "adjoint_backward_error": 1.0e-4,
            "potential_relative_difference": 1.0e-5,
            "temperature_relative_difference": 2.0e-5,
            "objective": objective,
            "authority_objective": authority_objective,
            "objective_relative_difference": abs(objective - authority_objective)
            / authority_objective,
            "electrical_joule_power_W_per_m": 2.0,
            "thermal_joule_load_W_per_m": 2.0,
            "transfer_relative_error": 0.0,
            "explicit_gradient_relative_differences": [1.0e-4, 2.0e-4, 3.0e-4],
            "native_gradient_authority_relative_differences": [
                1.0e-4,
                2.0e-4,
                3.0e-4,
            ],
            "native_gradient_explicit_relative_differences": [0.0, 0.0, 0.0],
            "authority": (
                "controller-generated dense float64 same-discretization forward and coupled "
                "residual VJP from the immutable input artifact"
            ),
        },
        "executables": {name: _executable(name, process_index) for name in EXECUTABLE_NAMES},
        "claim_scope": (
            "bounded process-local physical multi-host TPU current-to-Joule-to-heat forward, "
            "coupled residual adjoint, and native JAX reverse correctness witness against an "
            "immutable dense float64 authority; not Elmer re-execution, scaling, live HBM, 3D "
            "production FEM, measured-device, foundry, FDTDX, or recovery evidence"
        ),
    }


def _records() -> list[dict[str, object]]:
    return [_record(0), _record(1)]


def _change_deployed_identity(records: list[dict[str, object]]) -> None:
    _nested(records[1], "provenance")["run_id"] = "other-run"
    _nested(records[1], "launch_claim")["run_id"] = "other-run"


def _change_partitioned_layout(records: list[dict[str, object]]) -> None:
    report = _nested(records[1], "partitioned_array_reports", "input-cell-areas")
    report["global_shape"] = [4, 3]
    report["global_logical_bytes"] = 48
    report["addressable_logical_bytes"] = 24
    for shard in report["addressable_shards"]:
        shard["local_shape"] = [1, 3]
        shard["logical_bytes"] = 12


def _change_replicated_layout(records: list[dict[str, object]]) -> None:
    report = _nested(records[1], "replicated_array_reports", "current-parameters")
    report["global_shape"] = [3]
    report["logical_bytes_per_replica"] = 12
    report["addressable_logical_bytes"] = 24
    report["global_replica_logical_bytes"] = 48


def _change_valid_numerics(records: list[dict[str, object]]) -> None:
    numerics = _nested(records[1], "numerics")
    numerics["objective"] = 1.00005
    numerics["objective_relative_difference"] = abs(1.00005 - 1.0001) / 1.0001


def _duplicate_assignment_device(record: dict[str, object]) -> None:
    _nested(record, "mesh_report", "assignments")[1]["device_id"] = 0
    for report in _nested(record, "partitioned_array_reports").values():
        for shard in report["addressable_shards"]:
            if shard["partition_index"] == 1:
                shard["device_id"] = 0


def _set_extreme_memory_risk(record: dict[str, object]) -> None:
    memory = _nested(record, "executables", "forward", "memory")
    memory["hbm_capacity_bytes_per_device"] = 210
    memory["hbm_fraction"] = 200 / 210
    memory["risk"] = "extreme"


def test_process_set_is_complete_order_independent_detached_and_public_safe() -> None:
    records = _records()
    forward = aggregate_tpu_distributed_electrothermal_process_evidence(records)
    reverse = aggregate_tpu_distributed_electrothermal_process_evidence(list(reversed(records)))
    data = forward.canonical_data()

    assert data["schema_version"] == PROCESS_SET_EVIDENCE_SCHEMA
    assert data["status"] == "passed"
    assert data["runtime"]["global_device_count"] == 4  # type: ignore[index]
    assert data["problem"]["triangle_count"] == 8  # type: ignore[index]
    assert data["addressability"][  # type: ignore[index]
        "combined_partition_addressability_counts"
    ] == [1, 1, 1, 1]
    assert data["numerics"]["coupled_adjoint_backward_error"] == 1.0e-4  # type: ignore[index]
    assert len(data["array_storage"]["partitioned"]) == len(  # type: ignore[index]
        PARTITIONED_ARRAY_REPORT_NAMES
    )
    assert len(data["array_storage"]["replicated"]) == len(  # type: ignore[index]
        REPLICATED_ARRAY_REPORT_NAMES
    )
    serialized = forward.canonical_json()
    for private_value in ("private-run", "private-profile"):
        assert private_value not in serialized
    for private_key in ("run_id", "profile", "worker_index", "hostname", "zone"):
        assert f'"{private_key}":' not in serialized
    assert forward.canonical_json() == reverse.canonical_json()
    assert forward.digest() == reverse.digest()
    data["status"] = "tampered"
    assert forward.canonical_data()["status"] == "passed"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda records: records.pop(),
        lambda records: records.__setitem__(1, copy.deepcopy(records[0])),
        lambda records: _nested(records[1], "launch_claim").__setitem__("worker_index", 0),
        _change_deployed_identity,
        lambda records: _nested(records[1], "runtime").__setitem__("jax_version", "other"),
        lambda records: _nested(records[1], "plan").__setitem__("sha256", "0" * 64),
        lambda records: _nested(records[1], "mesh_report", "assignments")[0].__setitem__(
            "device_id", 99
        ),
        lambda records: _nested(
            records[1], "partitioned_array_reports", "input-cell-areas"
        ).__setitem__("global_shape", [4, 3]),
        _change_partitioned_layout,
        lambda records: _nested(
            records[1], "replicated_array_reports", "current-parameters"
        ).__setitem__("global_shape", [3]),
        _change_replicated_layout,
        lambda records: _nested(records[1], "tolerances").__setitem__(
            "gradient_relative_difference", 4.0e-3
        ),
        lambda records: _nested(records[1], "numerics").__setitem__("objective", 0.999),
        _change_valid_numerics,
        lambda records: _nested(records[1], "executables", "forward", "stablehlo").__setitem__(
            "sha256", "9" * 64
        ),
    ],
)
def test_process_set_rejects_incomplete_duplicate_or_inconsistent_records(
    mutate: Callable[[list[dict[str, object]]], object],
) -> None:
    records = _records()
    mutate(records)
    with pytest.raises(ValidationError, match="distributed electrothermal physical TPU evidence"):
        aggregate_tpu_distributed_electrothermal_process_evidence(records)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.__setitem__("unexpected", True),
        lambda r: r.__setitem__("schema_version", "v0"),
        lambda r: r.__setitem__("status", "failed"),
        lambda r: r.__setitem__("runtime", None),
        lambda r: _nested(r, "provenance").__setitem__("source_commit", "a" * 39),
        lambda r: _nested(r, "provenance").__setitem__("source_digest", "A" * 64),
        lambda r: _nested(r, "runtime").__setitem__("process_count", 1),
        lambda r: _nested(r, "runtime").__setitem__("process_index", 2),
        lambda r: _nested(r, "runtime").__setitem__("global_device_count", 3),
        lambda r: _nested(r, "runtime").__setitem__("backend", "cpu"),
        lambda r: _nested(r, "runtime").__setitem__("x64_enabled", True),
        lambda r: _nested(r, "runtime").__setitem__("device_kinds", []),
        lambda r: _nested(r, "runtime").__setitem__("real_scalar_contract", {}),
        lambda r: _nested(r, "launch_claim").__setitem__("schema_version", "v0"),
        lambda r: _nested(r, "launch_claim").__setitem__("scope", "wrong"),
        lambda r: _nested(r, "plan").__setitem__("schema_version", "v0"),
        lambda r: _nested(r, "plan").__setitem__("partition_count", 3),
        lambda r: _nested(r, "plan").__setitem__("free_dof_count", 9),
        lambda r: _nested(r, "plan").__setitem__("host_input_replication", "wrong"),
        lambda r: _nested(r, "mesh_report").__setitem__("schema_version", "v0"),
        lambda r: _nested(r, "mesh_report").__setitem__("addressable_device_count", 1),
        lambda r: _nested(r, "addressability").__setitem__(
            "process_local_partition_mask", [1, 2, 0, 0]
        ),
        lambda r: _nested(r, "addressability").__setitem__(
            "partition_addressability_counts", [1, 1, 0, 1]
        ),
        lambda r: _nested(r, "mesh_report").__setitem__("assignments", []),
        lambda r: _nested(r, "mesh_report", "assignments")[0].__setitem__("partition_index", 1),
        lambda r: _nested(r, "mesh_report", "assignments")[0].__setitem__("platform", "cpu"),
        lambda r: _nested(r, "mesh_report", "assignments")[0].__setitem__("addressable", False),
        _duplicate_assignment_device,
        lambda r: _nested(r, "partitioned_array_reports").pop("input-cell-areas"),
        lambda r: _nested(r, "partitioned_array_reports", "input-cell-areas").__setitem__(
            "schema_version", "v0"
        ),
        lambda r: _nested(r, "partitioned_array_reports", "input-cell-areas").__setitem__(
            "partition_axis_name", "wrong"
        ),
        lambda r: _nested(r, "partitioned_array_reports", "input-cell-areas").__setitem__(
            "global_logical_bytes", 1
        ),
        lambda r: _nested(r, "partitioned_array_reports", "input-cell-areas").__setitem__(
            "dtype", "float64"
        ),
        lambda r: _nested(r, "partitioned_array_reports", "input-cell-areas").__setitem__(
            "dtype", "int32"
        ),
        lambda r: _nested(r, "partitioned_array_reports", "input-cell-areas").__setitem__(
            "replication_intent", "replicated"
        ),
        lambda r: _nested(r, "partitioned_array_reports", "input-cell-areas").__setitem__(
            "addressable_shards", []
        ),
        lambda r: _nested(r, "partitioned_array_reports", "input-cell-areas", "addressable_shards")[
            0
        ].__setitem__("process_index", 1),
        lambda r: _nested(r, "partitioned_array_reports", "input-cell-areas", "addressable_shards")[
            0
        ].__setitem__("device_id", 99),
        lambda r: _nested(r, "partitioned_array_reports", "input-cell-areas", "addressable_shards")[
            0
        ].__setitem__("local_shape", [1, 3]),
        lambda r: _nested(r, "partitioned_array_reports", "input-cell-areas", "addressable_shards")[
            0
        ].__setitem__("logical_bytes", 1),
        lambda r: _nested(r, "replicated_array_reports").pop("current-parameters"),
        lambda r: _nested(r, "replicated_array_reports", "current-parameters").__setitem__(
            "partition_spec", ["partition"]
        ),
        lambda r: _nested(r, "replicated_array_reports", "current-parameters").__setitem__(
            "logical_bytes_per_replica", 1
        ),
        lambda r: _nested(r, "replicated_array_reports", "current-parameters").__setitem__(
            "dtype", "int32"
        ),
        lambda r: _nested(r, "replicated_array_reports", "current-parameters").__setitem__(
            "replication_intent", "wrong"
        ),
        lambda r: r.__setitem__("tolerances", {}),
        lambda r: _nested(r, "policies", "scalar_cg").__setitem__("max_iterations", 999),
        lambda r: _nested(r, "numerics").__setitem__("finite", False),
        lambda r: _nested(r, "numerics").__setitem__("current_linear_breakdown", True),
        lambda r: _nested(r, "numerics").__setitem__("iterations", 101),
        lambda r: _nested(r, "numerics").__setitem__("heat_linear_iterations", 1001),
        lambda r: _nested(r, "numerics").__setitem__("current_linear_backward_error", 1.0e-6),
        lambda r: _nested(r, "numerics").__setitem__("heat_residual_error", 2.0e-4),
        lambda r: _nested(r, "numerics").__setitem__("adjoint_backward_error", 1.0e-3),
        lambda r: _nested(r, "numerics").__setitem__("potential_relative_difference", 1.0e-3),
        lambda r: _nested(r, "numerics").__setitem__(
            "explicit_gradient_relative_differences", [1.0e-2, 0.0, 0.0]
        ),
        lambda r: _nested(r, "numerics").__setitem__(
            "explicit_gradient_relative_differences", [0.0, 0.0]
        ),
        lambda r: _nested(r, "numerics").__setitem__(
            "native_gradient_explicit_relative_differences", [2.0e-3, 0.0, 0.0]
        ),
        lambda r: _nested(r, "numerics").__setitem__("objective_relative_difference", 0.0),
        lambda r: _nested(r, "numerics").__setitem__("thermal_joule_load_W_per_m", 1.0),
        lambda r: _nested(r, "numerics").__setitem__("authority", "wrong"),
        lambda r: _nested(r, "executables").pop("forward"),
        lambda r: _nested(r, "executables", "forward", "timing").__setitem__(
            "schema_version", "v0"
        ),
        lambda r: _nested(r, "executables", "forward", "timing").__setitem__(
            "synchronization", "not blocked"
        ),
        lambda r: _nested(r, "executables", "forward", "timing").__setitem__(
            "execution_seconds", [0.1]
        ),
        lambda r: _nested(r, "executables", "forward", "timing").__setitem__(
            "execution_min_seconds", 0.1
        ),
        lambda r: _nested(r, "executables", "forward", "memory").__setitem__(
            "compiler_peak_bytes", 201
        ),
        lambda r: _nested(r, "executables", "forward", "memory").__setitem__(
            "schema_version", "v0"
        ),
        lambda r: _nested(r, "executables", "forward", "memory").__setitem__("hbm_fraction", 0.3),
        lambda r: _nested(r, "executables", "forward", "memory").__setitem__("risk", "elevated"),
        _set_extreme_memory_risk,
        lambda r: _nested(r, "executables", "forward", "memory").__setitem__(
            "claim_scope", "live HBM"
        ),
        lambda r: _nested(r, "executables", "forward", "stablehlo").__setitem__(
            "contains_all_gather", True
        ),
        lambda r: r.__setitem__("claim_scope", "too broad"),
    ],
)
def test_process_record_fails_closed(
    mutate: Callable[[dict[str, object]], object],
) -> None:
    records = _records()
    mutate(records[0])
    with pytest.raises(ValidationError, match="distributed electrothermal physical TPU evidence"):
        aggregate_tpu_distributed_electrothermal_process_evidence(records)


def test_helpers_fail_closed_on_wrong_json_types_and_ranges() -> None:
    with pytest.raises(ValidationError):
        _mapping([], label="item")
    with pytest.raises(ValidationError):
        _sequence("item", label="item")
    with pytest.raises(ValidationError):
        _text(" spaced ", label="item")
    with pytest.raises(ValidationError):
        _integer(True, label="item")
    with pytest.raises(ValidationError):
        _number(float("nan"), label="item")
    with pytest.raises(ValidationError):
        _number("not-a-number", label="item")
    with pytest.raises(ValidationError):
        _boolean(1, label="item")
    with pytest.raises(ValidationError):
        _sha256("0" * 63, label="item")
    with pytest.raises(ValidationError):
        _git_revision("0" * 39, label="item")
    with pytest.raises(ValidationError):
        _shape([], label="item")
    assert _risk_for_fraction(0.70) == "elevated"
    assert _risk_for_fraction(0.85) == "high"
    assert _risk_for_fraction(0.95) == "extreme"


def test_process_set_rejects_nonsequence_and_noncanonical_json() -> None:
    invalid_inputs: tuple[object, ...] = ([], "record")
    for records in invalid_inputs:
        with pytest.raises(ValidationError, match="nonempty sequence"):
            aggregate_tpu_distributed_electrothermal_process_evidence(records)
    records = _records()
    records[0]["claim_scope"] = {"not-json"}
    with pytest.raises(ValidationError, match="claim_scope"):
        aggregate_tpu_distributed_electrothermal_process_evidence(records)
    with pytest.raises(ValidationError, match="not canonical JSON"):
        _canonical_json({"not-json": {"set"}})


def test_aggregate_cli_publishes_once_and_rejects_duplicate_inputs(
    tmp_path: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inputs = []
    for index, record in enumerate(_records()):
        path = tmp_path / f"process-{index}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        inputs.append(path)
    output = tmp_path / "aggregate.json"

    assert main([*(str(path) for path in inputs), "--output", str(output)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["process_count"] == 2
    assert printed["global_device_count"] == 4
    assert printed["status"] == "passed"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main([*(str(path) for path in inputs), "--output", str(output)])
    with pytest.raises(ValueError, match="must be unique"):
        main([str(inputs[0]), str(inputs[0]), "--output", str(tmp_path / "duplicate.json")])


def test_aggregate_cli_bootstraps_source_tree_without_site_packages() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            str(
                repository_root / "scripts" / "aggregate_tpu_distributed_electrothermal_evidence.py"
            ),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "process_metrics" in completed.stdout
