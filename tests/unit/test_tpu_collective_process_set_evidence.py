from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from scripts.aggregate_tpu_collective_port_evidence import (
    MAX_PROCESS_RECORD_BYTES,
    _load_process_record,
    _publish,
    main,
)

from femx.core.errors import ValidationError
from femx.validation.tpu_collective_evidence import (
    PROCESS_EVIDENCE_SCHEMA,
    PROCESS_SET_EVIDENCE_SCHEMA,
    aggregate_tpu_collective_process_evidence,
)

pytestmark = pytest.mark.unit


def _executable(process_index: int, *, risk_fraction: float = 0.10) -> dict[str, object]:
    peak = round(1000 * risk_fraction)
    return {
        "timing": {
            "schema_version": "femx.jax.port_collective.timing_report/v1",
            "lowering_seconds": 1.0 + process_index,
            "compilation_seconds": 2.0 + process_index,
            "warmup_seconds": 3.0 + process_index,
            "execution_seconds": [
                1.0 + process_index,
                2.0 + process_index,
                3.0 + process_index,
                4.0 + process_index,
                5.0 + process_index,
            ],
            "execution_min_seconds": 1.0 + process_index,
            "execution_median_seconds": 3.0 + process_index,
            "execution_max_seconds": 5.0 + process_index,
            "synchronization": "every timed result blocked until ready",
        },
        "memory": {
            "schema_version": "femx.jax.port_collective.memory_report/v1",
            "generated_code_bytes": 1,
            "argument_bytes": peak,
            "output_bytes": 0,
            "alias_bytes": 0,
            "temporary_bytes": 0,
            "compiler_peak_bytes": peak,
            "hbm_capacity_bytes_per_device": 1000,
            "hbm_fraction": peak / 1000,
            "risk": (
                "safe"
                if risk_fraction < 0.70
                else "elevated"
                if risk_fraction < 0.85
                else "high"
                if risk_fraction < 0.95
                else "extreme"
            ),
            "claim_scope": "compiler estimate; not live HBM usage",
        },
        "stablehlo_collective_permute_count": 4,
        "stablehlo_contains_all_gather": False,
    }


def _process_record(
    process_index: int,
    *,
    process_count: int = 2,
    risk_fraction: float = 0.10,
) -> dict[str, object]:
    local_device_count = 2
    global_device_count = process_count * local_device_count
    local_mask = [
        int(partition // local_device_count == process_index)
        for partition in range(global_device_count)
    ]
    assignments = [
        {
            "partition_index": partition,
            "process_index": partition // local_device_count,
            "device_id": partition % local_device_count,
            "platform": "tpu",
            "device_kind": "TPU test",
            "addressable": bool(local_mask[partition]),
        }
        for partition in range(global_device_count)
    ]
    executables = {
        name: _executable(process_index, risk_fraction=risk_fraction)
        for name in ("real_forward", "complex_forward", "real_vjp", "complex_vjp")
    }
    return {
        "schema_version": PROCESS_EVIDENCE_SCHEMA,
        "status": "passed",
        "provenance": {
            "run_id": "run-physical-1",
            "profile": "spot-test",
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
            "process_count": process_count,
            "local_device_count": local_device_count,
            "global_device_count": global_device_count,
            "device_kinds": ["TPU test"],
            "complex_scalar_contract": {
                "logical_dtype": "complex64",
                "matrix_dtype": "float32",
                "index_dtype": "int32",
                "execution_representation": "native complex64",
                "matmul_precision": "highest",
                "host_reference_dtype": "complex128",
                "precision_fallback": False,
            },
        },
        "launch_claim": {
            "schema_version": "femx.jax.port_collective.worker_entry_claim/v1",
            "run_id": "run-physical-1",
            "worker_index": process_index,
            "process_index": process_index,
            "source_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "scope": (
                "worker-local femx entry fence after Phoxla bootstrap; prevents duplicate "
                "scientific execution but does not claim controller-level launch ownership"
            ),
        },
        "physics": {
            "model": "2D lossless Si/SiO2 mixed H(curl)/H1 port operator",
            "frequency_hz": 193.414e12,
            "silicon_refractive_index": 3.48,
            "silica_refractive_index": 1.444,
            "core_width_m": 0.5e-6,
            "core_height_m": 0.22e-6,
            "cross_section_width_m": 2.0e-6,
            "cross_section_height_m": 1.0e-6,
            "shift_per_m2": -4.0,
        },
        "problem": {
            "node_count": 21,
            "triangle_count": 32,
            "free_mixed_dof_count": 49,
            "partition_count": global_device_count,
            "layout_sha256": "c" * 64,
            "halo_link_count": 2,
            "halo_value_count": 7,
        },
        "mesh_report": {
            "schema_version": "femx.jax.port_collective.mesh_report/v1",
            "axis_name": "partition",
            "partition_count": global_device_count,
            "global_device_count": global_device_count,
            "addressable_device_count": local_device_count,
            "process_count": process_count,
            "is_multi_process": True,
            "layout_sha256": "c" * 64,
            "assignments": assignments,
        },
        "addressability": {
            "process_local_partition_mask": local_mask,
            "partition_addressability_counts": [1] * global_device_count,
            "every_partition_addressable_once": True,
        },
        "checkpoint": {
            "mode": "fresh-process-roundtrip",
            "fragment": {
                "schema_version": "femx.jax.port_collective.checkpoint_fragment_report/v1",
                "checkpoint_id": "port-collective-step-000000",
                "step": 0,
                "source_sha256": "a" * 64,
                "config_sha256": "b" * 64,
                "layout_sha256": "c" * 64,
                "process_index": process_index,
                "process_count": process_count,
                "manifest_sha256": f"{process_index + 1:x}" * 64,
                "array_names": [
                    "cell-local-dof-map",
                    "complex-owned-cotangent",
                    "complex-owned-vector",
                    "mass-cell-blocks",
                    "real-owned-cotangent",
                    "real-owned-vector",
                    "shifted-cell-blocks",
                    "stiffness-cell-blocks",
                ],
                "completion_scope": "one process-local fragment",
                "restore_policy": "exact same topology only; no resharding",
            },
            "restored_state_consumed_by_operator": True,
            "actual_preemption_event": False,
            "cross_topology_restore": False,
        },
        "numerics": {
            "maximum_action_relative_difference": 1.0e-13 * (process_index + 1),
            "maximum_vjp_relative_difference": 2.0e-13 * (process_index + 1),
            "action_finite": {
                "stiffness": {"real": True, "complex": True},
                "mass": {"real": True, "complex": True},
            },
            "vjp_finite": {"real": True, "complex": True},
            "action_tolerance": 1.0e-10,
            "vjp_tolerance": 2.0e-10,
            "host_c64_vs_c128_action_relative_differences": {
                "stiffness": 4.0e-8,
                "mass": 5.0e-8,
            },
            "host_c64_vs_c128_vjp_relative_differences": {
                "complex_cell_matrix": 6.0e-8,
                "complex_owned_vector": 7.0e-8,
            },
            "maximum_host_c64_vs_c128_relative_difference": 7.0e-8,
            "host_precision_tolerance": 2.0e-6,
            "host_precision_scope": (
                "operator arithmetic and vector/cotangent quantization with float32 cell "
                "coefficients held fixed; not float64 FEM assembly parity"
            ),
        },
        "executables": executables,
        "claim_scope": "process record",
    }


def _records() -> list[dict[str, object]]:
    return [_process_record(0), _process_record(1)]


def _nested(record: dict[str, object], *keys: str) -> dict[str, Any]:
    value: Any = record
    for key in keys:
        value = value[key]
    assert isinstance(value, dict)
    return value


def test_aggregate_is_order_independent_and_reports_process_critical_path() -> None:
    records = _records()
    forward = aggregate_tpu_collective_process_evidence(records)
    reverse = aggregate_tpu_collective_process_evidence(list(reversed(records)))
    data = forward.canonical_data()

    assert data["schema_version"] == PROCESS_SET_EVIDENCE_SCHEMA
    assert data["status"] == "passed"
    assert data["runtime"] == {
        "backend": "tpu",
        "jax_version": "0.10.1",
        "jaxlib_version": "0.10.1",
        "x64_enabled": False,
        "default_matmul_precision": "highest",
        "process_indexes": [0, 1],
        "worker_indexes": [0, 1],
        "process_count": 2,
        "local_device_count": 2,
        "global_device_count": 4,
        "device_kinds": ["TPU test"],
        "complex_scalar_contract": {
            "logical_dtype": "complex64",
            "matrix_dtype": "float32",
            "index_dtype": "int32",
            "execution_representation": "native complex64",
            "matmul_precision": "highest",
            "host_reference_dtype": "complex128",
            "precision_fallback": False,
        },
    }
    assert data["physics"] == _process_record(0)["physics"]
    assert data["addressability"]["combined_partition_addressability_counts"] == [1] * 4
    assert data["checkpoint"]["complete_process_fragment_count"] == 2
    assert data["numerics"]["maximum_action_relative_difference_across_processes"] == (
        pytest.approx(2.0e-13)
    )
    executable = data["executables"]["real_forward"]
    assert executable["compilation_seconds_across_processes"] == {
        "min": 2.0,
        "median": 2.5,
        "max": 3.0,
    }
    assert executable["process_execution_median_seconds"] == [3.0, 4.0]
    assert executable["execution_ordinal_critical_path_seconds"] == [2.0, 3.0, 4.0, 5.0, 6.0]
    assert executable["execution_ordinal_critical_path_summary_seconds"]["median"] == 4.0
    assert executable["maximum_compiler_hbm_fraction"] == pytest.approx(0.1)
    assert executable["worst_compiler_hbm_risk"] == "safe"
    assert "not a scaling result" in data["claim_scope"]
    assert forward.canonical_json() == reverse.canonical_json()
    assert forward.digest() == reverse.digest()
    assert len(forward.digest()) == 64
    data["status"] = "tampered"
    assert forward.canonical_data()["status"] == "passed"


@pytest.mark.parametrize(
    ("fractions", "risk"),
    [
        ((0.69, 0.70), "elevated"),
        ((0.70, 0.85), "high"),
        ((0.85, 0.95), "extreme"),
        ((0.95, 1.05), "extreme"),
    ],
)
def test_aggregate_reports_worst_compiler_memory_risk(
    fractions: tuple[float, float], risk: str
) -> None:
    records = [
        _process_record(0, risk_fraction=fractions[0]),
        _process_record(1, risk_fraction=fractions[1]),
    ]
    data = aggregate_tpu_collective_process_evidence(records).canonical_data()
    executable = data["executables"]["complex_vjp"]
    assert executable["worst_compiler_hbm_risk"] == risk
    assert executable["maximum_compiler_hbm_fraction"] == pytest.approx(fractions[1])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda records: records.pop(),
        lambda records: records.__setitem__(1, copy.deepcopy(records[0])),
        lambda records: _nested(records[1], "provenance").__setitem__("run_id", "other"),
        lambda records: _nested(records[1], "runtime").__setitem__("jax_version", "other"),
        lambda records: _nested(records[1], "launch_claim").__setitem__("worker_index", 0),
        lambda records: _nested(records[1], "physics").__setitem__("frequency_hz", 200.0e12),
        lambda records: _nested(records[1], "problem").__setitem__("node_count", 22),
        lambda records: _nested(records[1], "checkpoint").__setitem__(
            "mode", "restored-external-fragment"
        ),
        lambda records: _nested(records[1], "mesh_report")["assignments"][0].__setitem__(
            "device_id", 99
        ),
        lambda records: _nested(records[1], "numerics").__setitem__("action_tolerance", 2.0e-10),
        lambda records: _nested(records[1], "executables", "real_forward", "timing").__setitem__(
            "execution_seconds", [1.0, 2.0, 3.0, 4.0]
        ),
        lambda records: _nested(records[1], "executables", "real_forward").__setitem__(
            "stablehlo_collective_permute_count", 5
        ),
        lambda records: _nested(records[1], "addressability").__setitem__(
            "process_local_partition_mask", [0, 0, 1, 0]
        ),
    ],
)
def test_process_set_rejects_missing_duplicate_or_inconsistent_records(
    mutate: Callable[[list[dict[str, object]]], object],
) -> None:
    records = _records()
    mutate(records)
    with pytest.raises(ValidationError, match="physical TPU process-set evidence"):
        aggregate_tpu_collective_process_evidence(records)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda record: record.__setitem__("schema_version", "v1"),
        lambda record: record.__setitem__("status", "failed"),
        lambda record: _nested(record, "runtime").__setitem__("backend", "cpu"),
        lambda record: _nested(record, "runtime").__setitem__("x64_enabled", True),
        lambda record: _nested(record, "runtime").__setitem__(
            "default_matmul_precision", "default"
        ),
        lambda record: _nested(record, "runtime").__setitem__("process_index", 2),
        lambda record: _nested(record, "runtime").__setitem__("global_device_count", 3),
        lambda record: _nested(record, "runtime").__setitem__("device_kinds", []),
        lambda record: _nested(record, "runtime").__setitem__(
            "complex_scalar_contract", {"logical_dtype": "complex64"}
        ),
        lambda record: _nested(record, "launch_claim").__setitem__("schema_version", "v0"),
        lambda record: _nested(record, "launch_claim").__setitem__("run_id", "other"),
        lambda record: _nested(record, "launch_claim").__setitem__("worker_index", 2),
        lambda record: _nested(record, "launch_claim").__setitem__("process_index", 1),
        lambda record: _nested(record, "launch_claim").__setitem__("scope", "whole launcher"),
        lambda record: _nested(record, "problem").__setitem__("partition_count", 3),
        lambda record: _nested(record, "problem").__setitem__("layout_sha256", "X" * 64),
        lambda record: _nested(record, "mesh_report").__setitem__("schema_version", "v0"),
        lambda record: _nested(record, "mesh_report").__setitem__("axis_name", "wrong"),
        lambda record: _nested(record, "mesh_report").__setitem__("is_multi_process", False),
        lambda record: _nested(record, "mesh_report").__setitem__("layout_sha256", "d" * 64),
        lambda record: _nested(record, "mesh_report").__setitem__("addressable_device_count", 1),
        lambda record: _nested(record, "addressability").__setitem__(
            "process_local_partition_mask", [1, 0]
        ),
        lambda record: _nested(record, "addressability").__setitem__(
            "process_local_partition_mask", [2, 0, 0, 0]
        ),
        lambda record: _nested(record, "addressability").__setitem__(
            "partition_addressability_counts", [1, 1, 0, 2]
        ),
        lambda record: _nested(record, "addressability").__setitem__(
            "every_partition_addressable_once", False
        ),
        lambda record: _nested(record, "mesh_report").__setitem__("assignments", []),
        lambda record: _nested(record, "mesh_report")["assignments"][0].__setitem__(
            "partition_index", 1
        ),
        lambda record: _nested(record, "mesh_report")["assignments"][0].__setitem__(
            "addressable", False
        ),
        lambda record: _nested(record, "mesh_report")["assignments"][0].__setitem__(
            "addressable", "yes"
        ),
        lambda record: _nested(record, "mesh_report")["assignments"][0].__setitem__(
            "process_index", 2
        ),
        lambda record: _nested(record, "mesh_report")["assignments"][0].__setitem__(
            "process_index", 1
        ),
        lambda record: _nested(record, "mesh_report")["assignments"][0].__setitem__(
            "platform", "cpu"
        ),
        lambda record: _nested(record, "mesh_report")["assignments"][0].__setitem__(
            "device_kind", "other"
        ),
        lambda record: _nested(record, "mesh_report")["assignments"][1].__setitem__("device_id", 0),
        lambda record: _nested(record, "checkpoint").__setitem__("mode", "cross-topology"),
        lambda record: _nested(record, "checkpoint").__setitem__(
            "restored_state_consumed_by_operator", False
        ),
        lambda record: _nested(record, "checkpoint").__setitem__("cross_topology_restore", True),
        lambda record: _nested(record, "checkpoint").__setitem__("actual_preemption_event", True),
        lambda record: _nested(record, "checkpoint", "fragment").__setitem__(
            "schema_version", "v0"
        ),
        lambda record: _nested(record, "checkpoint", "fragment").__setitem__("process_index", 1),
        lambda record: _nested(record, "checkpoint", "fragment").__setitem__("process_count", 3),
        lambda record: _nested(record, "checkpoint", "fragment").__setitem__(
            "layout_sha256", "d" * 64
        ),
        lambda record: _nested(record, "checkpoint", "fragment").__setitem__(
            "source_sha256", "d" * 64
        ),
        lambda record: _nested(record, "checkpoint", "fragment").__setitem__(
            "manifest_sha256", "bad"
        ),
        lambda record: _nested(record, "checkpoint", "fragment").__setitem__("array_names", []),
        lambda record: _nested(record, "checkpoint", "fragment").__setitem__(
            "completion_scope", "whole cluster"
        ),
        lambda record: _nested(record, "checkpoint", "fragment").__setitem__(
            "restore_policy", "reshard"
        ),
        lambda record: _nested(record, "physics").__setitem__("model", "other"),
        lambda record: _nested(record, "physics").__setitem__("shift_per_m2", True),
        lambda record: _nested(record, "physics").__setitem__("shift_per_m2", "bad"),
        lambda record: _nested(record, "physics").__setitem__("shift_per_m2", 0.0),
        lambda record: _nested(record, "physics").__setitem__("shift_per_m2", float("nan")),
        lambda record: _nested(record, "physics").__setitem__("frequency_hz", 0.0),
        lambda record: _nested(record, "numerics").__setitem__(
            "maximum_action_relative_difference", 2.0
        ),
        lambda record: _nested(record, "numerics").__setitem__(
            "maximum_host_c64_vs_c128_relative_difference", 3.0e-6
        ),
        lambda record: _nested(record, "numerics").__setitem__(
            "maximum_host_c64_vs_c128_relative_difference", 1.0e-8
        ),
        lambda record: _nested(record, "numerics").__setitem__(
            "host_c64_vs_c128_action_relative_differences", {}
        ),
        lambda record: _nested(record, "numerics").__setitem__(
            "host_precision_scope", "full FEM parity"
        ),
        lambda record: _nested(record, "numerics").__setitem__("action_finite", {}),
        lambda record: _nested(record, "numerics", "action_finite", "mass").__setitem__(
            "real", False
        ),
        lambda record: _nested(record, "numerics", "vjp_finite").__setitem__("real", False),
        lambda record: record.__setitem__("executables", {}),
    ],
)
def test_process_record_fails_closed_on_invalid_physical_contract(
    mutate: Callable[[dict[str, object]], object],
) -> None:
    records = _records()
    mutate(records[0])
    with pytest.raises(ValidationError, match="physical TPU process-set evidence"):
        aggregate_tpu_collective_process_evidence(records)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda executable: _nested(executable, "timing").__setitem__("schema_version", "v0"),
        lambda executable: _nested(executable, "timing").__setitem__("synchronization", "async"),
        lambda executable: _nested(executable, "timing").__setitem__("execution_seconds", [1.0]),
        lambda executable: _nested(executable, "memory").__setitem__("schema_version", "v0"),
        lambda executable: _nested(executable, "memory").__setitem__(
            "hbm_capacity_bytes_per_device", 0
        ),
        lambda executable: _nested(executable, "memory").__setitem__("hbm_fraction", 0.5),
        lambda executable: _nested(executable, "memory").__setitem__("risk", "extreme"),
        lambda executable: _nested(executable, "memory").__setitem__("claim_scope", "live HBM"),
        lambda executable: executable.__setitem__("stablehlo_contains_all_gather", True),
        lambda executable: _nested(executable, "timing").__setitem__(
            "lowering_seconds", float("nan")
        ),
    ],
)
def test_executable_record_fails_closed_on_invalid_timing_memory_or_hlo(
    mutate: Callable[[dict[str, object]], object],
) -> None:
    records = _records()
    executable = _nested(records[0], "executables", "real_forward")
    mutate(executable)
    with pytest.raises(ValidationError, match="physical TPU process-set evidence"):
        aggregate_tpu_collective_process_evidence(records)


@pytest.mark.parametrize(
    ("value", "path"),
    [
        (None, ("runtime", "jax_version")),
        (True, ("runtime", "process_count")),
        (-1, ("runtime", "process_index")),
        (float("inf"), ("numerics", "action_tolerance")),
        ("bad", ("numerics", "action_tolerance")),
        ("bad", ("addressability", "process_local_partition_mask")),
        ({1: "bad"}, ("provenance",)),
    ],
)
def test_strict_json_types_are_required(value: object, path: tuple[str, ...]) -> None:
    records = _records()
    target: Any = records[0]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValidationError, match="physical TPU process-set evidence"):
        aggregate_tpu_collective_process_evidence(records)


def test_noncanonical_record_json_is_rejected() -> None:
    records = _records()
    records[0]["not_json"] = {1, 2, 3}
    with pytest.raises(ValidationError, match="not canonical JSON"):
        aggregate_tpu_collective_process_evidence(records)


def test_process_set_rejects_partition_overlap_before_aggregation() -> None:
    records = _records()
    addressability = _nested(records[1], "addressability")
    addressability["process_local_partition_mask"] = [1, 0, 0, 1]
    assignments = _nested(records[1], "mesh_report")["assignments"]
    for partition, assignment in enumerate(assignments):
        assignment["addressable"] = bool(addressability["process_local_partition_mask"][partition])
    with pytest.raises(ValidationError, match="Mesh addressability"):
        aggregate_tpu_collective_process_evidence(records)


def test_process_record_rejects_a_skewed_global_process_assignment() -> None:
    records = [_process_record(index, process_count=3) for index in range(3)]
    assignments = _nested(records[0], "mesh_report")["assignments"]
    assignments[2]["process_index"] = 2
    assignments[2]["device_id"] = 9
    with pytest.raises(ValidationError, match="declared process topology"):
        aggregate_tpu_collective_process_evidence(records)


def test_empty_process_set_is_rejected() -> None:
    with pytest.raises(ValidationError, match="requires at least one"):
        aggregate_tpu_collective_process_evidence([])


def _write_record(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, allow_nan=True), encoding="utf-8")


def test_aggregate_cli_publishes_once_and_reports_logical_digest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inputs = [tmp_path / "process-0.json", tmp_path / "process-1.json"]
    for path, record in zip(inputs, _records(), strict=True):
        _write_record(path, record)
    output = tmp_path / "results" / "aggregate.json"

    assert main([*(str(path) for path in inputs), "--output", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == payload["status"] == "passed"
    assert report["process_count"] == payload["runtime"]["process_count"] == 2
    assert len(report["logical_sha256"]) == 64
    assert not list(output.parent.glob("*.incomplete"))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main([*(str(path) for path in inputs), "--output", str(output)])
    with pytest.raises(ValueError, match="must be unique"):
        main([str(inputs[0]), str(inputs[0]), "--output", str(tmp_path / "duplicate.json")])


def test_process_record_loader_rejects_unsafe_or_noncanonical_inputs(tmp_path: Path) -> None:
    empty = tmp_path / "empty.json"
    empty.touch()
    with pytest.raises(ValueError, match="size"):
        _load_process_record(empty)

    oversized = tmp_path / "oversized.json"
    with oversized.open("wb") as stream:
        stream.truncate(MAX_PROCESS_RECORD_BYTES + 1)
    with pytest.raises(ValueError, match="size"):
        _load_process_record(oversized)

    scalar = tmp_path / "scalar.json"
    _write_record(scalar, [1, 2])
    with pytest.raises(ValueError, match="root"):
        _load_process_record(scalar)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        _load_process_record(nonfinite)

    symlink = tmp_path / "link.json"
    symlink.symlink_to(scalar.name)
    with pytest.raises(ValueError, match="non-symlink"):
        _load_process_record(symlink)


def test_atomic_publisher_rejects_a_preexisting_temporary_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("scripts.aggregate_tpu_collective_port_evidence.os.getpid", lambda: 7)
    destination = tmp_path / "aggregate.json"
    temporary = tmp_path / ".aggregate.json.7.incomplete"
    temporary.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError, match="temporary"):
        _publish(destination, "{}")
