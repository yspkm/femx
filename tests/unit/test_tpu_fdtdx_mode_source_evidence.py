from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from scripts.aggregate_tpu_fdtdx_mode_source_evidence import (
    MAX_PROCESS_RECORD_BYTES,
    _load_process_record,
    _publish,
    main,
)

from femx.core.errors import ValidationError
from femx.validation.tpu_fdtdx_mode_source_evidence import (
    PROCESS_EVIDENCE_SCHEMA,
    PROCESS_SET_EVIDENCE_SCHEMA,
    aggregate_tpu_fdtdx_mode_source_process_evidence,
)

pytestmark = pytest.mark.unit


def _digest(value: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _process_record(process_index: int, *, process_count: int = 2) -> dict[str, object]:
    local_device_count = 2
    global_device_count = process_count * local_device_count
    start = process_index * 4
    binding = {
        "schema_version": "femx.fdtdx.distributed_mode_source/v1",
        "source_name": "femx-distributed-port",
        "source_contract_sha256": "a" * 64,
        "mesh_axis_name": "shard",
        "partition_spec": ["replicated", "shard", "replicated", "replicated"],
        "global_shape": [3, 4 * process_count, 4, 1],
        "field_dtype": "complex64",
        "time_offset_dtype": "float32",
        "global_device_count": global_device_count,
        "local_device_count": local_device_count,
        "process_count": process_count,
        "process_index": process_index,
        "addressable_x_ranges": [[start, start + 2], [start + 2, start + 4]],
        "profile_distribution": "identical_full_snapshot_per_process",
        "execution_policy": "outer_jit_with_arrays_objects_config_as_arguments",
        "physical_evidence": False,
    }
    return {
        "schema_version": PROCESS_EVIDENCE_SCHEMA,
        "status": "passed",
        "provenance": {
            "run_id": "fdtdx-physical-1",
            "profile": "femx-v5e-us-16",
            "source_digest": "1" * 64,
            "config_digest": "2" * 64,
        },
        "runtime": {
            "backend": "tpu",
            "jax_version": "0.10.1",
            "jaxlib_version": "0.10.1",
            "fdtdx_version": "0.6.2",
            "x64_enabled": False,
            "process_index": process_index,
            "process_count": process_count,
            "local_device_count": local_device_count,
            "global_device_count": global_device_count,
            "device_kinds": ["TPU test"],
            "scalar_contract": {
                "field_dtype": "float32",
                "mode_dtype": "complex64",
                "time_offset_dtype": "float32",
                "x64_enabled": False,
                "precision_fallback": False,
            },
        },
        "launch_claim": {
            "schema_version": "femx.fdtdx.mode_source.worker_entry_claim/v1",
            "run_id": "fdtdx-physical-1",
            "worker_index": process_index,
            "process_index": process_index,
            "source_sha256": "1" * 64,
            "config_sha256": "2" * 64,
            "scope": "one immutable worker-local entry after distributed JAX initialization",
        },
        "source": {
            "binding": binding,
            "binding_sha256": _digest(binding),
            "bundle_sha256": "b" * 64,
            "fdtdx_fingerprint": {
                "package_version": "0.6.2",
                "source_revision": "81a58da9cde4a4ff822f835b63597c0d0d8ba978",
                "source_digest": (
                    "c881f0cf32b4272dc10e0acee716bfda3cf2d0f11973c23d31f29b11f3dce01c"
                ),
            },
            "module_sha256": {"fdtdx.objects.sources.custom_mode": "c" * 64},
        },
        "simulation": {
            "grid_shape_xyz": [4 * process_count, 4, 10],
            "source_z_index": 2,
            "simulation_time_s": 5.0e-15,
            "time_steps": 30,
            "relative_permittivity": 2.085136,
            "boundaries": ["periodic"] * 6,
        },
        "numerics": {
            "completed_step": 30,
            "initial_e_l2": 0.0,
            "initial_h_l2": 0.0,
            "final_e_l2": 10.0,
            "final_h_l2": 11.0,
            "downstream_e_l2": 3.0,
            "all_fields_finite": True,
        },
        "execution": {
            "lowering_seconds": 1.0 + process_index,
            "compilation_seconds": 2.0 + process_index,
            "warmup_seconds": 3.0 + process_index,
            "execution_seconds": 4.0 + process_index,
            "compiler_memory": {
                "compiler_peak_bytes": 100 + process_index,
                "hbm_capacity_bytes_per_device": 1000,
            },
            "stablehlo_all_gather_count": process_index,
        },
    }


def _records() -> list[dict[str, object]]:
    return [_process_record(0), _process_record(1)]


def _nested(record: dict[str, object], *keys: str) -> dict[str, Any]:
    value: Any = record
    for key in keys:
        value = value[key]
    assert isinstance(value, dict)
    return value


def _refresh_binding_digest(record: dict[str, object]) -> None:
    source = _nested(record, "source")
    binding = _nested(record, "source", "binding")
    source["binding_sha256"] = _digest(binding)


def test_aggregate_is_order_independent_and_process_complete() -> None:
    records = _records()
    forward = aggregate_tpu_fdtdx_mode_source_process_evidence(records)
    reverse = aggregate_tpu_fdtdx_mode_source_process_evidence(list(reversed(records)))
    data = forward.canonical_data()

    assert data["schema_version"] == PROCESS_SET_EVIDENCE_SCHEMA
    assert data["status"] == "passed"
    assert data["runtime"] == {
        "backend": "tpu",
        "jax_version": "0.10.1",
        "jaxlib_version": "0.10.1",
        "fdtdx_version": "0.6.2",
        "x64_enabled": False,
        "process_indexes": [0, 1],
        "worker_indexes": [0, 1],
        "process_count": 2,
        "local_device_count": 2,
        "global_device_count": 4,
        "device_kinds": ["TPU test"],
        "scalar_contract": {
            "field_dtype": "float32",
            "mode_dtype": "complex64",
            "time_offset_dtype": "float32",
            "x64_enabled": False,
            "precision_fallback": False,
        },
    }
    source = data["source"]
    assert isinstance(source, dict)
    assert source["combined_addressable_x_ranges"] == [[0, 2], [2, 4], [4, 6], [6, 8]]
    assert source["every_global_source_shard_addressable_once"] is True
    assert len(source["process_bindings"]) == 2
    execution = data["execution"]
    assert isinstance(execution, dict)
    assert execution["compilation_seconds_across_processes"] == {
        "min": 2.0,
        "median": 2.5,
        "max": 3.0,
    }
    assert execution["stablehlo_all_gather_counts"] == [0, 1]
    assert "not Elmer parity" in str(data["claim_scope"])
    assert forward.canonical_json() == reverse.canonical_json()
    assert forward.digest() == reverse.digest()
    assert len(forward.digest()) == 64
    data["status"] = "tampered"
    assert forward.canonical_data()["status"] == "passed"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda records: records.pop(),
        lambda records: records.__setitem__(1, copy.deepcopy(records[0])),
        lambda records: _nested(records[1], "provenance").__setitem__("run_id", "other"),
        lambda records: _nested(records[1], "runtime").__setitem__("jax_version", "other"),
        lambda records: _nested(records[1], "source").__setitem__("bundle_sha256", "d" * 64),
        lambda records: _nested(records[1], "simulation").__setitem__("time_steps", 31),
        lambda records: _nested(records[1], "numerics").__setitem__("final_e_l2", 12.0),
        lambda records: _nested(records[1], "launch_claim").__setitem__("worker_index", 0),
        lambda records: _nested(records[1], "source", "binding").__setitem__(
            "addressable_x_ranges", [[2, 4], [4, 6]]
        ),
    ],
)
def test_process_set_rejects_missing_duplicate_inconsistent_or_overlapping_records(
    mutate: Callable[[list[dict[str, object]]], object],
) -> None:
    records = _records()
    mutate(records)
    if len(records) == 2:
        _refresh_binding_digest(records[1])
    with pytest.raises(ValidationError, match="physical TPU FDTDX mode-source evidence"):
        aggregate_tpu_fdtdx_mode_source_process_evidence(records)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda record: record.__setitem__("schema_version", "v0"),
        lambda record: record.__setitem__("status", "failed"),
        lambda record: _nested(record, "runtime").__setitem__("backend", "cpu"),
        lambda record: record.__setitem__("runtime", None),
        lambda record: _nested(record, "runtime").__setitem__("x64_enabled", True),
        lambda record: _nested(record, "runtime").__setitem__("device_kinds", "TPU test"),
        lambda record: _nested(record, "runtime").__setitem__("process_index", False),
        lambda record: _nested(record, "runtime").__setitem__("process_index", 2),
        lambda record: _nested(record, "runtime").__setitem__("process_count", 1),
        lambda record: _nested(record, "runtime").__setitem__("global_device_count", 3),
        lambda record: _nested(record, "runtime").__setitem__("device_kinds", []),
        lambda record: _nested(record, "runtime").__setitem__("fdtdx_version", "0.6.1"),
        lambda record: _nested(record, "runtime").__setitem__("scalar_contract", {}),
        lambda record: _nested(record, "launch_claim").__setitem__("schema_version", "v0"),
        lambda record: _nested(record, "launch_claim").__setitem__("worker_index", 2),
        lambda record: _nested(record, "launch_claim").__setitem__("run_id", "other"),
        lambda record: _nested(record, "source").__setitem__("binding_sha256", "d" * 64),
        lambda record: _nested(record, "source").__setitem__("module_sha256", {}),
        lambda record: _nested(record, "source").__setitem__(
            "fdtdx_fingerprint", {"package_version": "0.6.2"}
        ),
        lambda record: _nested(record, "simulation").__setitem__("grid_shape_xyz", [7, 4, 10]),
        lambda record: _nested(record, "simulation").__setitem__("source_z_index", 9),
        lambda record: _nested(record, "simulation").__setitem__("simulation_time_s", "bad"),
        lambda record: _nested(record, "simulation").__setitem__("boundaries", ["pml"] * 6),
        lambda record: _nested(record, "numerics").__setitem__("completed_step", 29),
        lambda record: _nested(record, "numerics").__setitem__("initial_e_l2", 1.0),
        lambda record: _nested(record, "numerics").__setitem__("final_h_l2", 0.0),
        lambda record: _nested(record, "numerics").__setitem__("all_fields_finite", False),
        lambda record: _nested(record, "numerics").__setitem__("all_fields_finite", "yes"),
        lambda record: _nested(record, "execution", "compiler_memory").__setitem__(
            "compiler_peak_bytes", 1001
        ),
        lambda record: _nested(record, "execution").__setitem__("execution_seconds", float("nan")),
    ],
)
def test_process_record_fails_closed_on_invalid_physical_contract(
    mutate: Callable[[dict[str, object]], object],
) -> None:
    records = _records()
    mutate(records[0])
    with pytest.raises(ValidationError, match="physical TPU FDTDX mode-source evidence"):
        aggregate_tpu_fdtdx_mode_source_process_evidence(records)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda binding: binding.__setitem__("schema_version", "v0"),
        lambda binding: binding.__setitem__("physical_evidence", True),
        lambda binding: binding.__setitem__("source_name", " "),
        lambda binding: binding.__setitem__("source_contract_sha256", "bad"),
        lambda binding: binding.__setitem__("mesh_axis_name", "x"),
        lambda binding: binding.__setitem__("global_shape", [3, 7, 4, 1]),
        lambda binding: binding.__setitem__("global_shape", [2, 8, 4, 1]),
        lambda binding: binding.__setitem__("field_dtype", "complex128"),
        lambda binding: binding.__setitem__("process_count", 3),
        lambda binding: binding.__setitem__("addressable_x_ranges", [[0], [2, 4]]),
        lambda binding: binding.__setitem__("addressable_x_ranges", [[0, 9], [2, 4]]),
        lambda binding: binding.__setitem__("addressable_x_ranges", [[2, 4], [0, 2]]),
        lambda binding: binding.__setitem__("profile_distribution", "local-only"),
        lambda binding: binding.__setitem__("execution_policy", "closed-over-objects"),
    ],
)
def test_binding_record_fails_closed_on_invalid_sharding_contract(
    mutate: Callable[[dict[str, object]], object],
) -> None:
    records = _records()
    binding = _nested(records[0], "source", "binding")
    mutate(binding)
    _refresh_binding_digest(records[0])
    with pytest.raises(ValidationError, match="physical TPU FDTDX mode-source evidence"):
        aggregate_tpu_fdtdx_mode_source_process_evidence(records)


def test_process_record_rejects_noncanonical_json_value() -> None:
    records = _records()
    records[0]["unused"] = {"not-json"}
    with pytest.raises(ValidationError, match="not canonical JSON"):
        aggregate_tpu_fdtdx_mode_source_process_evidence(records)


def test_process_set_rejects_a_contiguous_but_incomplete_global_x_cover() -> None:
    records = _records()
    _nested(records[0], "source", "binding")["addressable_x_ranges"] = [[0, 1], [1, 2]]
    _nested(records[1], "source", "binding")["addressable_x_ranges"] = [[2, 3], [3, 4]]
    for record in records:
        _refresh_binding_digest(record)
    with pytest.raises(ValidationError, match="one source shard per global TPU device"):
        aggregate_tpu_fdtdx_mode_source_process_evidence(records)


def test_aggregate_requires_a_record() -> None:
    with pytest.raises(ValidationError, match="requires at least one"):
        aggregate_tpu_fdtdx_mode_source_process_evidence([])


def test_aggregate_cli_loads_and_publishes_once(
    tmp_path: Path,
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
    assert printed["sha256"] == hashlib.sha256(output.read_text().strip().encode()).hexdigest()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main([*(str(path) for path in inputs), "--output", str(output)])


def test_aggregate_cli_rejects_duplicate_or_invalid_inputs(tmp_path: Path) -> None:
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps(_process_record(0)), encoding="utf-8")
    with pytest.raises(ValueError, match="must be unique"):
        main([str(valid), str(valid), "--output", str(tmp_path / "out.json")])

    invalid = tmp_path / "invalid.json"
    invalid.write_text("NaN", encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        _load_process_record(invalid)

    empty = tmp_path / "empty.json"
    empty.touch()
    with pytest.raises(ValueError, match="size is outside"):
        _load_process_record(empty)

    large = tmp_path / "large.json"
    large.write_bytes(b" " * (MAX_PROCESS_RECORD_BYTES + 1))
    with pytest.raises(ValueError, match="size is outside"):
        _load_process_record(large)


def test_load_and_publish_reject_symlinks_or_existing_temporaries(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="regular non-symlink"):
        _load_process_record(link)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _publish(link, "{}")

    output = tmp_path / "new.json"
    temporary = output.with_name(f".{output.name}.{__import__('os').getpid()}.incomplete")
    temporary.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError, match="temporary path"):
        _publish(output, "{}")
