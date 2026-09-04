from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from scripts.aggregate_tpu_public_ring_heater_forward_evidence import main

from femx.core.errors import ValidationError
from femx.validation.tpu_public_ring_heater_evidence import (
    CONSERVATION_TOLERANCES,
    EXPECTED_FINE_MSH_SHA256,
    EXPECTED_PARTITIONED_ARRAY_NAMES,
    PARITY_TOLERANCES,
    PROCESS_EVIDENCE_SCHEMA,
    PROCESS_SET_EVIDENCE_SCHEMA,
    REAL_SCALAR_CONTRACT,
    SCALAR_CG_POLICY,
    WORKER_ENTRY_CLAIM_SCHEMA,
    TpuPublicRingHeaterProcessSetEvidence,
    aggregate_tpu_public_ring_heater_process_evidence,
)

pytestmark = pytest.mark.unit


def _policies() -> dict[str, object]:
    preconditioner = cast(dict[str, object], dict(SCALAR_CG_POLICY)["preconditioner"])
    return {
        "scalar_cg": {
            **dict(SCALAR_CG_POLICY),
            "preconditioner": dict(preconditioner),
        },
        "conservation_tolerances": dict(CONSERVATION_TOLERANCES),
        "parity_tolerances": dict(PARITY_TOLERANCES),
        "target_voltage_source": (
            "fixed by the CPU float64 unit-voltage linear calibration; no repeated TPU "
            "calibration solve"
        ),
    }


def _record(process_index: int) -> dict[str, object]:
    partitions = tuple(range(4 * process_index, 4 * process_index + 4))
    assignments = [
        {
            "partition_index": partition,
            "process_index": partition // 4,
            "device_id": partition,
            "platform": "tpu",
            "device_kind": "TPU v4",
            "addressable": partition in partitions,
        }
        for partition in range(32)
    ]
    shards = [
        {
            "partition_index": partition,
            "process_index": process_index,
            "device_id": partition,
            "device_kind": "TPU v4",
            "local_shape": [1, 2],
            "logical_bytes": 8,
        }
        for partition in partitions
    ]
    reports = {
        name: {
            "schema_version": "femx.jax.collective.array_report/v1",
            "name": name,
            "global_shape": [32, 2],
            "dtype": "float32",
            "partition_axis_name": "partition",
            "partition_count": 32,
            "global_device_count": 32,
            "process_index": process_index,
            "process_count": 8,
            "global_logical_bytes": 256,
            "addressable_logical_bytes": 32,
            "replication_intent": "none; one leading FEM partition per device",
            "addressable_shards": copy.deepcopy(shards),
        }
        for name in EXPECTED_PARTITIONED_ARRAY_NAMES
    }
    outputs = [
        {
            "partition_index": partition,
            "shape": [1, 2],
            "dtype": "float32",
            "sha256": hashlib.sha256(f"output-{partition}".encode()).hexdigest(),
            "finite": True,
        }
        for partition in partitions
    ]
    numerics: dict[str, object] = {
        "all_finite": True,
        "numerically_admitted": True,
        "current_converged": True,
        "thermal_converged": True,
        "current_breakdown": False,
        "thermal_breakdown": False,
        "current_iterations": 950,
        "thermal_iterations": 1519,
        "current_recursive_residual": 1.0e-5,
        "thermal_recursive_residual": 2.0e-5,
        "current_recomputed_residual": 1.1e-5,
        "thermal_recomputed_residual": 2.1e-5,
        "current_relative_residual": 4.0e-7,
        "thermal_relative_residual": 4.0e-7,
        "current_backward_error": 1.0e-6,
        "thermal_backward_error": 1.0e-6,
        "electrical_joule_power_W": 0.01036,
        "electrical_variational_power_W": 0.01036,
        "thermal_joule_load_W": 0.01036,
        "thermal_input_power_W": 0.01036,
        "convection_outward_power_W": 0.001,
        "dirichlet_outward_power_W": 0.00936,
        "minimum_temperature_K": 300.0,
        "maximum_temperature_K": 469.6,
        "silicon_ring_mean_temperature_K": 360.7,
        "tin_heater_mean_temperature_K": 458.1,
        "silicon_ring_volume_m3": 1.0e-18,
        "tin_heater_volume_m3": 2.0e-18,
        "inferred_current_A": 0.015,
    }
    numerics.update({name: 1.0e-6 for name in CONSERVATION_TOLERANCES})
    numerics.update({name: 1.0e-5 for name in PARITY_TOLERANCES})
    local_mask = [int(partition in partitions) for partition in range(32)]
    return {
        "schema_version": PROCESS_EVIDENCE_SCHEMA,
        "status": "passed",
        "provenance": {
            "run_id": "run-1",
            "profile": "v4-od-32",
            "source_digest": "a" * 64,
            "source_commit": "b" * 40,
            "config_digest": "c" * 64,
        },
        "runtime": {
            "backend": "tpu",
            "jax_version": "0.10.1",
            "jaxlib_version": "0.10.1",
            "x64_enabled": False,
            "default_matmul_precision": "highest",
            "process_index": process_index,
            "process_count": 8,
            "local_device_count": 4,
            "global_device_count": 32,
            "device_kinds": ["TPU v4"],
            "real_scalar_contract": dict(REAL_SCALAR_CONTRACT),
        },
        "launch_claim": {
            "schema_version": WORKER_ENTRY_CLAIM_SCHEMA,
            "run_id": "run-1",
            "worker_index": process_index,
            "process_index": process_index,
            "source_sha256": "a" * 64,
            "config_sha256": "c" * 64,
        },
        "artifact": {
            "schema_version": "femx.public-ring-heater.tpu_forward_input/v1",
            "logical_sha256": "d" * 64,
            "runtime_plan_sha256": "e" * 64,
            "source_plan_sha256": "f" * 64,
            "source_msh_sha256": EXPECTED_FINE_MSH_SHA256,
            "partition_owner_sha256": "1" * 64,
            "total_array_file_bytes": 500_000_000,
        },
        "model": {
            "node_count": 521_442,
            "tetrahedron_count": 3_179_879,
            "conductor_tetrahedron_count": 134_331,
            "dimension": 3,
            "element": "first-order Tet4 H1",
            "coupling": "current to cell-local Joule density to steady heat",
            "target_voltage_V": 0.691,
            "target_current_A": 0.015,
            "authority_predicted_power_W": 0.01036,
        },
        "mesh_report": {
            "schema_version": "femx.jax.collective.mesh_report/v1",
            "axis_name": "partition",
            "partition_count": 32,
            "global_device_count": 32,
            "addressable_device_count": 4,
            "process_count": 8,
            "is_multi_process": True,
            "layout_sha256": "2" * 64,
            "assignments": assignments,
        },
        "addressability": {
            "process_local_partition_mask": local_mask,
            "partition_addressability_counts": [1] * 32,
            "every_partition_addressable_once": True,
        },
        "partitioned_array_reports": reports,
        "replicated_parameter_report": {
            "schema_version": "femx.jax.collective.replicated_array_report/v1",
            "name": "electrothermal-controls",
            "global_shape": [3],
            "dtype": "float32",
            "partition_spec": [],
            "process_index": process_index,
            "process_count": 8,
            "global_device_count": 32,
            "addressable_device_count": 4,
        },
        "output_shards": {
            "potential": copy.deepcopy(outputs),
            "temperature_rise": copy.deepcopy(outputs),
        },
        "policies": _policies(),
        "numerics": numerics,
        "executable": {
            "timing": {
                "schema_version": "femx.public-ring-heater.tpu_forward_timing/v1",
                "lowering_seconds": 2.0 + process_index,
                "compilation_seconds": 5.0 + process_index,
                "execution_seconds": 3.0 + process_index,
                "execution_count": 1,
                "benchmark_claimed": False,
            },
            "compiler_memory": {
                "compiler_peak_bytes": 1000 + process_index,
                "hbm_capacity_bytes_per_device": 16_000,
                "hbm_fraction": (1000 + process_index) / 16_000,
                "risk": "safe",
                "claim_scope": "compiler estimate; not live HBM usage",
            },
            "stablehlo": {
                "sha256": "3" * 64,
                "collective_permute_count": 10,
                "all_reduce_count": 20,
                "contains_all_gather": False,
                "contains_f64": False,
            },
            "hlo_admitted": True,
            "memory_admitted": True,
        },
        "claim_scope": (
            "physical TPU forward; not fresh Elmer execution, no FDTDX response, no inverse "
            "design, and compiler estimate is not live HBM"
        ),
    }


def _records() -> list[dict[str, object]]:
    return [_record(index) for index in range(8)]


def _nested(record: dict[str, object], *keys: str) -> dict[str, object]:
    current: object = record
    for key in keys:
        current = cast(dict[str, object], current)[key]
    return cast(dict[str, object], current)


def test_complete_process_set_is_deterministic_and_detached() -> None:
    records = _records()
    forward = aggregate_tpu_public_ring_heater_process_evidence(records)
    reverse = aggregate_tpu_public_ring_heater_process_evidence(list(reversed(records)))

    assert forward.canonical_json() == reverse.canonical_json()
    assert forward.digest() == reverse.digest()
    data = forward.canonical_data()
    assert data["schema_version"] == PROCESS_SET_EVIDENCE_SCHEMA
    assert data["status"] == "passed"
    assert _nested(data, "runtime")["global_device_count"] == 32
    assert _nested(data, "partitioning")["every_temperature_partition_retained_once"] is True
    assert _nested(data, "execution")["maximum_compiler_peak_bytes"] == 1007
    data["status"] = "changed"
    assert forward.canonical_data()["status"] == "passed"


Mutation = Callable[[list[dict[str, object]]], None]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda records: records.pop(),
        lambda records: _nested(records[0], "runtime").__setitem__("backend", "cpu"),
        lambda records: _nested(records[0], "runtime").__setitem__("device_kinds", []),
        lambda records: _nested(records[0], "provenance").__setitem__("profile", "other"),
        lambda records: _nested(records[0], "launch_claim").__setitem__("run_id", "other"),
        lambda records: _nested(records[0], "artifact").__setitem__("source_msh_sha256", "0" * 64),
        lambda records: _nested(records[0], "model").__setitem__("dimension", 2),
        lambda records: _nested(records[0], "mesh_report").__setitem__("process_count", 4),
        lambda records: _nested(records[0], "addressability").__setitem__(
            "every_partition_addressable_once", False
        ),
        lambda records: _nested(records[0], "partitioned_array_reports").pop(
            EXPECTED_PARTITIONED_ARRAY_NAMES[0]
        ),
        lambda records: _nested(records[0], "replicated_parameter_report").__setitem__(
            "dtype", "float64"
        ),
        lambda records: cast(
            list[dict[str, object]], _nested(records[0], "output_shards")["potential"]
        )[0].__setitem__("finite", False),
        lambda records: _nested(records[0], "policies").__setitem__(
            "target_voltage_source", "changed"
        ),
        lambda records: _nested(records[0], "numerics").__setitem__("all_finite", False),
        lambda records: _nested(records[0], "numerics").__setitem__(
            "potential_relative_l2_difference", 1.0
        ),
        lambda records: _nested(records[0], "executable", "stablehlo").__setitem__(
            "contains_all_gather", True
        ),
        lambda records: _nested(records[0], "executable", "compiler_memory").__setitem__(
            "hbm_fraction", 0.9
        ),
        lambda records: records[0].__setitem__("claim_scope", "too broad"),
        lambda records: _nested(records[1], "numerics").__setitem__("maximum_temperature_K", 468.0),
        lambda records: _nested(records[1], "launch_claim").__setitem__("worker_index", 0),
    ],
)
def test_process_set_rejects_incomplete_or_drifted_evidence(mutation: Mutation) -> None:
    records = _records()
    mutation(records)
    with pytest.raises(ValidationError, match="public-ring physical TPU"):
        aggregate_tpu_public_ring_heater_process_evidence(records)


def test_empty_nonfinite_and_invalid_aggregate_fail_closed() -> None:
    with pytest.raises(ValidationError, match="requires process records"):
        aggregate_tpu_public_ring_heater_process_evidence([])
    records = _records()
    _nested(records[0], "numerics")["maximum_temperature_K"] = float("nan")
    with pytest.raises(ValidationError, match=r"finite|number"):
        aggregate_tpu_public_ring_heater_process_evidence(records)
    with pytest.raises(ValidationError, match="unsupported schema"):
        TpuPublicRingHeaterProcessSetEvidence({"schema_version": "bad", "status": "passed"})
    with pytest.raises(ValidationError, match="must be passing"):
        TpuPublicRingHeaterProcessSetEvidence(
            {"schema_version": PROCESS_SET_EVIDENCE_SCHEMA, "status": "failed"}
        )


def test_aggregator_cli_publishes_once(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths = []
    for index, record in enumerate(_records()):
        path = tmp_path / f"process-{index}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        paths.append(path)
    output = tmp_path / "aggregate.json"
    assert main([*(str(path) for path in paths), "--output", str(output)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "passed"
    assert summary["process_count"] == 8
    assert json.loads(output.read_text())["status"] == "passed"
    with pytest.raises(FileExistsError, match="overwrite"):
        main([*(str(path) for path in paths), "--output", str(output)])
    with pytest.raises(ValueError, match="unique"):
        main([str(paths[0]), str(paths[0]), "--output", str(tmp_path / "other.json")])
