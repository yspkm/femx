from __future__ import annotations

import json
import math
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from scripts.aggregate_tpu_distributed_fdtdx_thermo_optic_evidence import main

from femx.core.errors import ValidationError
from femx.validation.tpu_distributed_fdtdx_thermo_optic_evidence import (
    CRITICAL_ARRAY_REPORT_SCHEMA,
    CRITICAL_ARRAY_SPECS,
    EXECUTABLE_NAMES,
    EXPECTED_GLOBAL_DEVICE_COUNT,
    EXPECTED_LOCAL_DEVICE_COUNT,
    EXPECTED_PROCESS_COUNT,
    FDTDX_MODULE_SHA256,
    FDTDX_PACKAGE_VERSION,
    FDTDX_SOURCE_DIGEST,
    FDTDX_SOURCE_REVISION,
    PARTITIONED_ARRAY_REPORT_NAMES,
    PROCESS_EVIDENCE_SCHEMA,
    PROCESS_SET_EVIDENCE_SCHEMA,
    REPLICATED_ARRAY_REPORT_NAMES,
    SCALAR_CONTRACT,
    TOLERANCES,
    WORKER_ENTRY_CLAIM_SCHEMA,
    _boolean,
    _canonical_json,
    _git_revision,
    _integer,
    _mapping,
    _number,
    _sequence,
    _sha256,
    _shape,
    _signed_number,
    _text,
    aggregate_tpu_distributed_fdtdx_thermo_optic_process_evidence,
)

pytestmark = pytest.mark.unit


def _dtype(name: str) -> str:
    if "local-dofs" in name or "indices" in name or "slots" in name:
        return "int32"
    if "mask" in name or name.endswith("active"):
        return "bool"
    return "float32"


def _bytes(shape: tuple[int, ...], dtype: str) -> int:
    result = 1 if dtype == "bool" else 4
    for extent in shape:
        result *= extent
    return result


def _partitions(process_index: int) -> tuple[int, ...]:
    first = process_index * EXPECTED_LOCAL_DEVICE_COUNT
    return tuple(range(first, first + EXPECTED_LOCAL_DEVICE_COUNT))


def _partitioned_report(name: str, process_index: int) -> dict[str, object]:
    dtype = _dtype(name)
    shape = (EXPECTED_GLOBAL_DEVICE_COUNT, 2)
    local_shape = (1, 2)
    local_bytes = _bytes(local_shape, dtype)
    return {
        "schema_version": "femx.jax.collective.array_report/v1",
        "name": name,
        "global_shape": list(shape),
        "dtype": dtype,
        "partition_axis_name": "shard",
        "partition_count": EXPECTED_GLOBAL_DEVICE_COUNT,
        "global_device_count": EXPECTED_GLOBAL_DEVICE_COUNT,
        "process_index": process_index,
        "process_count": EXPECTED_PROCESS_COUNT,
        "global_logical_bytes": _bytes(shape, dtype),
        "addressable_logical_bytes": EXPECTED_LOCAL_DEVICE_COUNT * local_bytes,
        "replication_intent": "none; one leading FEM partition per device",
        "addressable_shards": [
            {
                "partition_index": partition,
                "process_index": process_index,
                "device_id": partition % EXPECTED_LOCAL_DEVICE_COUNT,
                "device_kind": "TPU v4",
                "local_shape": list(local_shape),
                "logical_bytes": local_bytes,
            }
            for partition in _partitions(process_index)
        ],
    }


def _replicated_report(name: str, process_index: int) -> dict[str, object]:
    shape = (2,)
    logical_bytes = _bytes(shape, "float32")
    return {
        "schema_version": "femx.jax.collective.replicated_array_report/v1",
        "name": name,
        "global_shape": list(shape),
        "dtype": "float32",
        "partition_spec": [],
        "global_device_count": EXPECTED_GLOBAL_DEVICE_COUNT,
        "addressable_device_count": EXPECTED_LOCAL_DEVICE_COUNT,
        "process_index": process_index,
        "process_count": EXPECTED_PROCESS_COUNT,
        "logical_bytes_per_replica": logical_bytes,
        "addressable_logical_bytes": EXPECTED_LOCAL_DEVICE_COUNT * logical_bytes,
        "global_replica_logical_bytes": EXPECTED_GLOBAL_DEVICE_COUNT * logical_bytes,
        "replication_intent": "bounded plan scalars and parameter vectors",
    }


def _critical_report(name: str, process_index: int) -> dict[str, object]:
    shape, dtype, spec = CRITICAL_ARRAY_SPECS[name]
    sharded_axis = 1 if name == "applied-inverse-permittivity" else 0
    width = shape[sharded_axis] // EXPECTED_GLOBAL_DEVICE_COUNT
    shards = []
    for partition in _partitions(process_index):
        index = [[0, extent] for extent in shape]
        index[sharded_axis] = [partition * width, (partition + 1) * width]
        local_shape = list(shape)
        local_shape[sharded_axis] = width
        shards.append(
            {
                "partition_index": partition,
                "process_index": process_index,
                "device_kind": "TPU v4",
                "index": index,
                "local_shape": local_shape,
            }
        )
    return {
        "schema_version": CRITICAL_ARRAY_REPORT_SCHEMA,
        "name": name,
        "global_shape": list(shape),
        "dtype": dtype,
        "partition_spec": list(spec),
        "process_index": process_index,
        "process_count": EXPECTED_PROCESS_COUNT,
        "global_device_count": EXPECTED_GLOBAL_DEVICE_COUNT,
        "addressable_shards": shards,
    }


def _executable(name: str, process_index: int) -> dict[str, object]:
    index = EXECUTABLE_NAMES.index(name) + 1
    samples = [0.4 + process_index * 0.01, 0.5, 0.6]
    peak = 2_000_000 * index
    capacity = 32_000_000_000
    return {
        "timing": {
            "schema_version": "femx.jax.collective.timing_report/v1",
            "lowering_seconds": 0.1 + process_index * 0.01,
            "compilation_seconds": 0.2 + process_index * 0.01,
            "warmup_seconds": 0.3 + process_index * 0.01,
            "execution_seconds": samples,
            "execution_min_seconds": min(samples),
            "execution_median_seconds": 0.5,
            "execution_max_seconds": max(samples),
            "synchronization": "every timed result blocked until ready",
        },
        "memory": {
            "schema_version": "femx.jax.collective.memory_report/v1",
            "generated_code_bytes": 100,
            "argument_bytes": peak // 4,
            "output_bytes": peak // 4,
            "alias_bytes": 0,
            "temporary_bytes": peak // 2,
            "compiler_peak_bytes": peak,
            "hbm_capacity_bytes_per_device": capacity,
            "hbm_fraction": peak / capacity,
            "risk": "safe",
            "claim_scope": "compiler estimate; not live HBM usage",
        },
        "stablehlo": {
            "sha256": f"{index:x}" * 64,
            "all_to_all_count": index,
            "collective_permute_count": 2 * index,
            "all_reduce_count": 3 * index,
            "contains_all_gather": False,
            "contains_float64": False,
        },
    }


def _record(process_index: int) -> dict[str, object]:
    local_mask = [
        int(partition in _partitions(process_index))
        for partition in range(EXPECTED_GLOBAL_DEVICE_COUNT)
    ]
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
            "fdtdx_version": FDTDX_PACKAGE_VERSION,
            "x64_enabled": False,
            "default_matmul_precision": "highest",
            "process_index": process_index,
            "process_count": EXPECTED_PROCESS_COUNT,
            "local_device_count": EXPECTED_LOCAL_DEVICE_COUNT,
            "global_device_count": EXPECTED_GLOBAL_DEVICE_COUNT,
            "device_kinds": ["TPU v4"],
            "scalar_contract": dict(SCALAR_CONTRACT),
        },
        "launch_claim": {
            "schema_version": WORKER_ENTRY_CLAIM_SCHEMA,
            "run_id": "private-run",
            "worker_index": process_index,
            "process_index": process_index,
            "source_sha256": "b" * 64,
            "config_sha256": "c" * 64,
            "scope": "one immutable worker-local entry after distributed JAX initialization",
        },
        "input": {
            "manifest_sha256": "d" * 64,
            "arrays_sha256": "e" * 64,
            "electrothermal_arrays_sha256": "f" * 64,
            "source_commit": "a" * 40,
            "sampling_operator_sha256": "1" * 64,
            "transfer_operator_sha256": "2" * 64,
            "scene_sha256": "3" * 64,
            "fdtdx_package_version": FDTDX_PACKAGE_VERSION,
            "fdtdx_source_revision": FDTDX_SOURCE_REVISION,
            "fdtdx_source_digest": FDTDX_SOURCE_DIGEST,
            "fdtdx_module_sha256": dict(FDTDX_MODULE_SHA256),
        },
        "plan": {
            "sha256": "4" * 64,
            "layout_sha256": "5" * 64,
            "partition_count": EXPECTED_GLOBAL_DEVICE_COUNT,
            "node_count": 289,
            "triangle_count": 512,
            "free_dof_count": 255,
        },
        "mesh_report": {
            "schema_version": "femx.jax.collective.mesh_report/v1",
            "axis_name": "shard",
            "partition_count": EXPECTED_GLOBAL_DEVICE_COUNT,
            "global_device_count": EXPECTED_GLOBAL_DEVICE_COUNT,
            "addressable_device_count": EXPECTED_LOCAL_DEVICE_COUNT,
            "process_count": EXPECTED_PROCESS_COUNT,
            "is_multi_process": True,
            "layout_sha256": "5" * 64,
            "assignments": [
                {
                    "partition_index": partition,
                    "process_index": partition // EXPECTED_LOCAL_DEVICE_COUNT,
                    "device_id": partition % EXPECTED_LOCAL_DEVICE_COUNT,
                    "platform": "tpu",
                    "device_kind": "TPU v4",
                    "addressable": bool(local_mask[partition]),
                }
                for partition in range(EXPECTED_GLOBAL_DEVICE_COUNT)
            ],
        },
        "addressability": {
            "process_local_partition_mask": local_mask,
            "partition_addressability_counts": [1] * EXPECTED_GLOBAL_DEVICE_COUNT,
            "every_partition_addressable_once": True,
        },
        "partitioned_array_reports": {
            name: _partitioned_report(name, process_index)
            for name in PARTITIONED_ARRAY_REPORT_NAMES
        },
        "replicated_array_reports": {
            name: _replicated_report(name, process_index) for name in REPLICATED_ARRAY_REPORT_NAMES
        },
        "critical_array_reports": {
            name: _critical_report(name, process_index) for name in CRITICAL_ARRAY_SPECS
        },
        "coordinate_admission": {
            "maximum_absolute_errors_m": [1.0e-13, 2.0e-14, 2.0e-14],
            "maximum_grid_fraction_errors": [1.6e-6, 3.2e-7, 3.2e-7],
            "maximum_ulp_errors": [5, 1, 1],
            "float32_rounding_exact": [False, False, False],
            "admitted": [True, True, True],
        },
        "scene": {
            "grid_shape_xyz": [96, 4, 8],
            "device_shape_xyz": [32, 2, 4],
            "time_steps": 302,
            "sha256": "3" * 64,
        },
        "numerics": {
            "finite": True,
            "forward_converged": True,
            "adjoint_converged": True,
            "thermo_optic_all_valid": True,
            "material_destination_sharding_preserved": True,
            "reference_phasor_real": 1.0,
            "reference_phasor_imag": -0.5,
            "reference_phasor_magnitude": math.sqrt(1.25),
            "objective": 0.0,
            "potential_relative_difference": 1.0e-6,
            "temperature_relative_difference": 1.0e-7,
            "cell_temperature_relative_difference": 1.0e-7,
            "parameter_relative_difference": 1.0e-7,
            "material_relative_difference": 1.0e-7,
            "objective_explicit_relative_difference": 0.0,
            "native_explicit_gradient_relative_differences": [0.0, 0.0, 0.0],
            "native_gradient_norms": [2.0, 3.0, 4.0],
            "applied_voltage_finite_difference": {
                "steps": [0.1, 0.05],
                "gradients": [-2.0, -2.01],
                "relative_errors": [5.0e-3, 6.0e-3],
            },
            "iterations": 14,
            "current_residual_error": 1.0e-6,
            "heat_residual_error": 1.0e-6,
            "current_linear_backward_error": 1.0e-8,
            "heat_linear_backward_error": 1.0e-8,
            "transfer_relative_error": 0.0,
            "adjoint_backward_error": 1.0e-4,
            "electrical_joule_power_W_per_m": 39.5,
            "thermal_joule_load_W_per_m": 39.5,
            "cell_cotangent_norm": 2.0,
            "potential_cotangent_norm": 0.0,
            "temperature_cotangent_norm": 4.0,
        },
        "tolerances": dict(TOLERANCES),
        "executables": {name: _executable(name, process_index) for name in EXECUTABLE_NAMES},
        "claim_scope": "bounded physical coupled FDTDX process record",
    }


def _records() -> list[dict[str, object]]:
    return [_record(index) for index in range(EXPECTED_PROCESS_COUNT)]


def _nested(record: dict[str, object], *keys: str) -> Any:
    value: Any = record
    for key in keys:
        value = value[key]
    return value


def test_complete_process_set_is_public_safe_and_deterministic() -> None:
    evidence = aggregate_tpu_distributed_fdtdx_thermo_optic_process_evidence(_records())
    payload = evidence.canonical_data()
    assert payload["schema_version"] == PROCESS_SET_EVIDENCE_SCHEMA
    assert payload["status"] == "passed"
    assert payload["runtime"]["process_count"] == EXPECTED_PROCESS_COUNT  # type: ignore[index]
    assert payload["array_admission"]["every_partition_addressable_once"] is True  # type: ignore[index]
    assert len(payload["process_records"]) == EXPECTED_PROCESS_COUNT  # type: ignore[arg-type]
    assert (
        evidence.digest()
        == aggregate_tpu_distributed_fdtdx_thermo_optic_process_evidence(
            list(reversed(_records()))
        ).digest()
    )
    encoded = evidence.canonical_json()
    assert "private-run" not in encoded
    assert "private-profile" not in encoded
    assert json.loads(encoded) == payload


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), "bad"),
        (("status",), "failed"),
        (("tolerances", "parameter_relative_difference"), 9.0),
        (("runtime", "backend"), "cpu"),
        (("runtime", "process_count"), 7),
        (("runtime", "local_device_count"), 3),
        (("runtime", "global_device_count"), 31),
        (("runtime", "x64_enabled"), True),
        (("runtime", "default_matmul_precision"), "default"),
        (("runtime", "fdtdx_version"), "0.0.0"),
        (("runtime", "scalar_contract", "real_dtype"), "float64"),
        (("launch_claim", "schema_version"), "bad"),
        (("launch_claim", "process_index"), 1),
        (("launch_claim", "source_sha256"), "9" * 64),
        (("input", "source_commit"), "8" * 40),
        (("input", "fdtdx_source_revision"), "8" * 40),
        (("input", "fdtdx_source_digest"), "8" * 64),
        (("plan", "partition_count"), 31),
        (("mesh_report", "axis_name"), "partition"),
        (("mesh_report", "layout_sha256"), "8" * 64),
        (("addressability", "every_partition_addressable_once"), False),
        (("coordinate_admission", "maximum_ulp_errors"), [9, 1, 1]),
        (("coordinate_admission", "admitted"), [False, True, True]),
        (("scene", "time_steps"), 301),
        (("numerics", "finite"), False),
        (("numerics", "parameter_relative_difference"), 1.0),
        (("numerics", "cell_cotangent_norm"), 0.0),
        (("numerics", "potential_cotangent_norm"), 1.0),
        (("numerics", "temperature_cotangent_norm"), 0.0),
        (("numerics", "native_gradient_norms"), [2.0, 0.0, 4.0]),
        (("numerics", "native_explicit_gradient_relative_differences"), [0.0, 1.0, 0.0]),
        (("numerics", "applied_voltage_finite_difference", "relative_errors"), [0.5, 0.5]),
        (("executables", "forward", "stablehlo", "contains_all_gather"), True),
        (("executables", "forward", "stablehlo", "contains_float64"), True),
        (("executables", "forward", "stablehlo", "all_to_all_count"), 0),
        (("executables", "forward", "stablehlo", "collective_permute_count"), 0),
        (("executables", "reference_phasor", "stablehlo", "collective_permute_count"), 0),
        (("executables", "forward", "memory", "hbm_fraction"), 0.9),
    ],
)
def test_invalid_process_records_fail_closed(path: tuple[str, ...], value: object) -> None:
    records = _records()
    parent = _nested(records[0], *path[:-1])
    parent[path[-1]] = value
    with pytest.raises(ValidationError):
        aggregate_tpu_distributed_fdtdx_thermo_optic_process_evidence(records)


def test_partitioned_replicated_and_critical_layout_drift_fails_closed() -> None:
    mutations: list[Callable[[dict[str, object]], None]] = [
        lambda record: _nested(
            record,
            "partitioned_array_reports",
            PARTITIONED_ARRAY_REPORT_NAMES[0],
        ).update({"partition_axis_name": "partition"}),
        lambda record: _nested(
            record,
            "partitioned_array_reports",
            PARTITIONED_ARRAY_REPORT_NAMES[0],
            "addressable_shards",
        )[0].update({"local_shape": [2, 2]}),
        lambda record: _nested(
            record,
            "replicated_array_reports",
            REPLICATED_ARRAY_REPORT_NAMES[0],
        ).update({"partition_spec": ["shard"]}),
        lambda record: _nested(
            record,
            "critical_array_reports",
            "thermo-optic-parameter",
        ).update({"partition_spec": [None, None, None]}),
        lambda record: _nested(
            record,
            "critical_array_reports",
            "applied-inverse-permittivity",
            "addressable_shards",
        )[0].update({"index": [[0, 1], [1, 4], [0, 4], [0, 8]]}),
    ]
    for mutate in mutations:
        records = _records()
        mutate(records[0])
        with pytest.raises(ValidationError):
            aggregate_tpu_distributed_fdtdx_thermo_optic_process_evidence(records)


def test_process_set_completeness_and_cross_process_identity_fail_closed() -> None:
    with pytest.raises(ValidationError):
        aggregate_tpu_distributed_fdtdx_thermo_optic_process_evidence([])
    with pytest.raises(ValidationError):
        aggregate_tpu_distributed_fdtdx_thermo_optic_process_evidence("not records")
    with pytest.raises(ValidationError):
        aggregate_tpu_distributed_fdtdx_thermo_optic_process_evidence(_records()[:-1])

    records = _records()
    records[1]["launch_claim"]["worker_index"] = 0  # type: ignore[index]
    with pytest.raises(ValidationError):
        aggregate_tpu_distributed_fdtdx_thermo_optic_process_evidence(records)

    records = _records()
    records[1]["numerics"]["objective"] = 1.0  # type: ignore[index]
    with pytest.raises(ValidationError):
        aggregate_tpu_distributed_fdtdx_thermo_optic_process_evidence(records)

    records = _records()
    records[1]["executables"]["forward"]["stablehlo"]["sha256"] = "f" * 64  # type: ignore[index]
    with pytest.raises(ValidationError):
        aggregate_tpu_distributed_fdtdx_thermo_optic_process_evidence(records)


def test_helpers_reject_wrong_json_types() -> None:
    for function, value in (
        (_mapping, []),
        (_sequence, "x"),
        (_text, " x"),
        (_integer, True),
        (_number, -1.0),
        (_signed_number, float("nan")),
        (_boolean, 1),
        (_sha256, "a"),
        (_git_revision, "a"),
        (_shape, []),
    ):
        with pytest.raises(ValidationError):
            function(value, label="test")
    with pytest.raises(ValidationError):
        _canonical_json(float("nan"))


def test_remaining_structural_failures_are_rejected() -> None:
    direct_calls: tuple[tuple[Callable[..., object], object], ...] = (
        (_number, "1.0"),
        (_signed_number, "1.0"),
    )
    for function, value in direct_calls:
        with pytest.raises(ValidationError):
            function(value, label="test")

    mutations: list[Callable[[dict[str, object]], None]] = [
        lambda record: record.update({"unexpected": True}),
        lambda record: _nested(
            record,
            "partitioned_array_reports",
            PARTITIONED_ARRAY_REPORT_NAMES[0],
        ).update({"dtype": "float64"}),
        lambda record: _nested(record, "executables", "forward", "timing").update(
            {"execution_seconds": [0.1, 0.2]}
        ),
        lambda record: _nested(record, "executables", "forward", "timing").update(
            {"execution_min_seconds": 0.2}
        ),
        lambda record: _nested(record, "runtime").update({"device_kinds": []}),
        lambda record: _nested(record, "mesh_report", "assignments")[0].update(
            {"partition_index": 1}
        ),
        lambda record: _nested(record, "mesh_report", "assignments")[0].update(
            {"addressable": False}
        ),
        lambda record: _nested(record, "mesh_report", "assignments")[0].update({"platform": "cpu"}),
        lambda record: _nested(record, "partitioned_array_reports").pop(
            PARTITIONED_ARRAY_REPORT_NAMES[0]
        ),
        lambda record: _nested(
            record,
            "partitioned_array_reports",
            PARTITIONED_ARRAY_REPORT_NAMES[0],
            "addressable_shards",
        )[-1].update({"partition_index": 4}),
        lambda record: _nested(record, "replicated_array_reports").pop(
            REPLICATED_ARRAY_REPORT_NAMES[0]
        ),
        lambda record: _nested(record, "critical_array_reports").pop("thermo-optic-parameter"),
        lambda record: _nested(
            record,
            "critical_array_reports",
            "thermo-optic-parameter",
            "addressable_shards",
        )[-1].update(
            {
                "partition_index": 4,
                "index": [[4, 5], [0, 2], [0, 4]],
            }
        ),
        lambda record: _nested(record, "executables").pop("forward"),
        lambda record: _nested(
            record,
            "numerics",
            "applied_voltage_finite_difference",
        ).update({"relative_errors": [0.005]}),
        lambda record: _nested(record, "numerics").update({"reference_phasor_magnitude": 2.0}),
        lambda record: _nested(record, "numerics").update({"private_machine": "must-not-leak"}),
        lambda record: _nested(
            record,
            "numerics",
            "applied_voltage_finite_difference",
        ).update({"relative_errors": [0.005, 0.5]}),
        lambda record: _nested(record, "coordinate_admission").update(
            {"float32_rounding_exact": [True, False, False]}
        ),
        lambda record: _nested(record, "coordinate_admission").update(
            {"maximum_grid_fraction_errors": [1.0e-9, 3.2e-7, 3.2e-7]}
        ),
    ]
    for mutate in mutations:
        records = _records()
        mutate(records[0])
        with pytest.raises(ValidationError):
            aggregate_tpu_distributed_fdtdx_thermo_optic_process_evidence(records)


def test_bounded_elevated_memory_is_admitted_but_high_and_extreme_are_not() -> None:
    elevated = _records()
    for record in elevated:
        for executable in EXECUTABLE_NAMES:
            memory = _nested(record, "executables", executable, "memory")
            capacity = 32_000_000_000
            peak = 24_000_000_000
            memory.update(
                {
                    "argument_bytes": peak,
                    "output_bytes": 0,
                    "alias_bytes": 0,
                    "temporary_bytes": 0,
                    "compiler_peak_bytes": peak,
                    "hbm_capacity_bytes_per_device": capacity,
                    "hbm_fraction": 0.75,
                    "risk": "elevated",
                }
            )
    assert (
        aggregate_tpu_distributed_fdtdx_thermo_optic_process_evidence(elevated).canonical_data()[
            "status"
        ]
        == "passed"
    )

    for fraction, risk in ((0.9, "high"), (0.96, "extreme")):
        records = _records()
        memory = _nested(records[0], "executables", "forward", "memory")
        capacity = 100_000_000
        peak = int(capacity * fraction)
        memory.update(
            {
                "argument_bytes": peak,
                "output_bytes": 0,
                "alias_bytes": 0,
                "temporary_bytes": 0,
                "compiler_peak_bytes": peak,
                "hbm_capacity_bytes_per_device": capacity,
                "hbm_fraction": fraction,
                "risk": risk,
            }
        )
        with pytest.raises(ValidationError):
            aggregate_tpu_distributed_fdtdx_thermo_optic_process_evidence(records)


def test_replicated_array_size_and_intent_are_bounded() -> None:
    for mutation in ("size", "intent"):
        records = _records()
        report = _nested(
            records[0],
            "replicated_array_reports",
            REPLICATED_ARRAY_REPORT_NAMES[0],
        )
        if mutation == "size":
            report.update(
                {
                    "global_shape": [1025],
                    "logical_bytes_per_replica": 4100,
                    "addressable_logical_bytes": 4100 * EXPECTED_LOCAL_DEVICE_COUNT,
                    "global_replica_logical_bytes": 4100 * EXPECTED_GLOBAL_DEVICE_COUNT,
                }
            )
        else:
            report["replication_intent"] = "unbounded"
        with pytest.raises(ValidationError):
            aggregate_tpu_distributed_fdtdx_thermo_optic_process_evidence(records)


def test_aggregator_cli_refuses_duplicates_and_publishes_once(tmp_path: Path) -> None:
    paths = []
    for index, record in enumerate(_records()):
        path = tmp_path / f"process-{index}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        paths.append(path)
    output = tmp_path / "aggregate.json"
    assert main([*(str(path) for path in paths), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "passed"
    with pytest.raises(FileExistsError):
        main([*(str(path) for path in paths), "--output", str(output)])
    with pytest.raises(ValueError, match="must be unique"):
        main([str(paths[0]), str(paths[0]), "--output", str(tmp_path / "duplicate.json")])

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/aggregate_tpu_distributed_fdtdx_thermo_optic_evidence.py",
            *(str(path) for path in paths),
            "--output",
            str(tmp_path / "subprocess.json"),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "passed"
