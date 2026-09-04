from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from typing import Any

import pytest
from scripts.aggregate_tpu_fdtdx_waveguide_source_evidence import main

from femx.core.errors import ValidationError
from femx.validation.tpu_fdtdx_waveguide_source_evidence import (
    INPUT_MANIFEST_SCHEMA,
    PROCESS_EVIDENCE_SCHEMA,
    PROCESS_SET_EVIDENCE_SCHEMA,
    aggregate_tpu_fdtdx_waveguide_source_process_evidence,
)

pytestmark = pytest.mark.unit


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _binding(process_index: int, solver: str) -> dict[str, object]:
    start = process_index * 32
    return {
        "schema_version": "femx.fdtdx.distributed_mode_source/v1",
        "source_name": "femx-waveguide-port",
        "source_contract_sha256": ("a" if solver == "elmer" else "b") * 64,
        "mesh_axis_name": "shard",
        "partition_spec": ["replicated", "shard", "replicated", "replicated"],
        "global_shape": [3, 64, 52, 1],
        "field_dtype": "complex64",
        "time_offset_dtype": "float32",
        "global_device_count": 4,
        "local_device_count": 2,
        "process_count": 2,
        "process_index": process_index,
        "addressable_x_ranges": [[start, start + 16], [start + 16, start + 32]],
        "profile_distribution": "identical_full_snapshot_per_process",
        "execution_policy": "outer_jit_with_arrays_objects_config_as_arguments",
        "physical_evidence": False,
    }


def _process(process_index: int) -> dict[str, object]:
    bundle_hashes = {"elmer": "e" * 64, "jax": "f" * 64}
    bindings = {solver: _binding(process_index, solver) for solver in ("elmer", "jax")}
    return {
        "schema_version": PROCESS_EVIDENCE_SCHEMA,
        "status": "passed",
        "provenance": {
            "run_id": "waveguide-physical-1",
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
            "process_count": 2,
            "local_device_count": 2,
            "global_device_count": 4,
            "device_kinds": ["TPU v5 lite"],
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
            "run_id": "waveguide-physical-1",
            "worker_index": process_index,
            "process_index": process_index,
            "source_sha256": "1" * 64,
            "config_sha256": "2" * 64,
        },
        "inputs": {
            "schema_version": INPUT_MANIFEST_SCHEMA,
            "manifest_sha256": "3" * 64,
            "artifacts": {
                solver: {
                    "reference": {
                        "path": f"modes/{solver}-mode.h5",
                        "sha256": ("4" if solver == "elmer" else "5") * 64,
                    },
                    "content_sha256": ("6" if solver == "elmer" else "7") * 64,
                    "bundle_sha256": bundle_hashes[solver],
                }
                for solver in ("elmer", "jax")
            },
            "fdtdx_fingerprint": {
                "package_version": "0.6.2",
                "source_revision": "81a58da9cde4a4ff822f835b63597c0d0d8ba978",
                "source_digest": (
                    "c881f0cf32b4272dc10e0acee716bfda3cf2d0f11973c23d31f29b11f3dce01c"
                ),
            },
            "runtime_module_sha256": {
                "fdtdx.core.grid": (
                    "d24739b9229ad8c61a57e4f688e6224eae63a680ff6554ddd7a5ef765edab6dd"
                ),
                "fdtdx.fdtd.wrapper": (
                    "97d562e0a33eeeccd6ce42d12cf8dc29f1e9fd071e561b5b009ac5da8ece7384"
                ),
                "fdtdx.objects.object": (
                    "24c986b9fa73bf474bce9fefc2145436654be4758e83dbcaf6fb955b7eb8557f"
                ),
                "fdtdx.objects.sources.custom_mode": (
                    "0c5925a784da33f8d8236a874d4759d4ebe6df29317dcc1ce68877b4a4036df5"
                ),
                "fdtdx.objects.sources.tfsf": (
                    "bd270995bffd174c7014adf9a02c7648134547c3bab7a294570e0a179326e611"
                ),
            },
        },
        "sources": {
            solver: {
                "binding": bindings[solver],
                "binding_sha256": _digest(bindings[solver]),
                "canonical_bundle_sha256": bundle_hashes[solver],
                "runtime_bundle_sha256": ("c" if solver == "elmer" else "d") * 64,
                "precision_report_sha256": ("8" if solver == "elmer" else "9") * 64,
            }
            for solver in ("elmer", "jax")
        },
        "simulation": {
            "grid_shape_xyz": [64, 52, 36],
            "source_z_index": 6,
            "detector_z_index": 24,
            "frequency_hz": 193_414_489_032_258.06,
            "simulation_time_s": 30.0e-15,
            "time_steps": 180,
            "boundaries": ["pec", "pec", "pec", "pec", "pml", "pml"],
            "core_cell_count": 32,
            "cladding_relative_permittivity": 1.444**2,
            "core_relative_permittivity": 3.48**2,
        },
        "numerics": {
            "completed_step": {"elmer": 180, "jax": 180},
            "all_fields_finite": {"elmer": True, "jax": True},
            "final_e_l2": {"elmer": 10.0, "jax": 10.00001},
            "final_h_l2": {"elmer": 11.0, "jax": 11.00001},
            "downstream_phasor_l2": {"elmer": 2.0, "jax": 2.00001},
            "source_electric_relative_l2": 1.0e-6,
            "source_magnetic_relative_l2": 2.0e-6,
            "downstream_phasor_relative_l2": 3.0e-6,
        },
        "execution": {
            "shared_compiled_pytree": True,
            "lowering_seconds": 1.0 + process_index,
            "compilation_seconds": 2.0 + process_index,
            "warmup_seconds": {"elmer": 3.0 + process_index, "jax": 4.0 + process_index},
            "execution_seconds": {"elmer": 5.0 + process_index, "jax": 6.0 + process_index},
            "compiler_memory": {
                "compiler_peak_bytes": 100 + process_index,
                "hbm_capacity_bytes_per_device": 1000,
            },
            "stablehlo_all_gather_count": process_index,
        },
    }


def _records() -> list[dict[str, object]]:
    return [_process(0), _process(1)]


def _nested(record: dict[str, object], *keys: str) -> dict[str, Any]:
    value: Any = record
    for key in keys:
        value = value[key]
    assert isinstance(value, dict)
    return value


def _refresh_binding(record: dict[str, object], solver: str) -> None:
    source = _nested(record, "sources", solver)
    source["binding_sha256"] = _digest(source["binding"])


def test_waveguide_process_set_is_complete_order_independent_and_detached() -> None:
    records = _records()
    forward = aggregate_tpu_fdtdx_waveguide_source_process_evidence(records)
    reverse = aggregate_tpu_fdtdx_waveguide_source_process_evidence(list(reversed(records)))
    data = forward.canonical_data()

    assert data["schema_version"] == PROCESS_SET_EVIDENCE_SCHEMA
    assert data["status"] == "passed"
    assert data["runtime"]["global_device_count"] == 4  # type: ignore[index]
    assert data["simulation"]["grid_shape_xyz"] == [64, 52, 36]  # type: ignore[index]
    assert data["sources"]["elmer"][  # type: ignore[index]
        "combined_addressable_x_ranges"
    ] == [[0, 16], [16, 32], [32, 48], [48, 64]]
    assert data["numerics"]["downstream_phasor_relative_l2"] == 3.0e-6  # type: ignore[index]
    assert "independently generated Elmer and JAX" in str(data["claim_scope"])
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
        lambda records: _nested(records[1], "provenance").__setitem__("run_id", "different"),
        lambda records: _nested(records[1], "numerics").__setitem__(
            "downstream_phasor_relative_l2", 4.0e-6
        ),
        lambda records: _nested(records[1], "sources", "jax", "binding").__setitem__(
            "addressable_x_ranges", [[16, 32], [32, 48]]
        ),
    ],
)
def test_waveguide_process_set_rejects_incomplete_duplicate_or_inconsistent_records(
    mutate: Callable[[list[dict[str, object]]], object],
) -> None:
    records = _records()
    mutate(records)
    if len(records) == 2:
        _refresh_binding(records[1], "jax")
    with pytest.raises(ValidationError, match="physical TPU FDTDX mode-source evidence"):
        aggregate_tpu_fdtdx_waveguide_source_process_evidence(records)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.__setitem__("schema_version", "v0"),
        lambda r: r.__setitem__("status", "failed"),
        lambda r: r.__setitem__("runtime", None),
        lambda r: _nested(r, "runtime").__setitem__("process_count", 1),
        lambda r: _nested(r, "runtime").__setitem__("process_index", 2),
        lambda r: _nested(r, "runtime").__setitem__("global_device_count", 3),
        lambda r: _nested(r, "runtime").__setitem__("backend", "cpu"),
        lambda r: _nested(r, "runtime").__setitem__("x64_enabled", True),
        lambda r: _nested(r, "runtime").__setitem__("device_kinds", []),
        lambda r: _nested(r, "runtime").__setitem__("scalar_contract", {}),
        lambda r: _nested(r, "runtime").__setitem__("fdtdx_version", "0.6.1"),
        lambda r: _nested(r, "launch_claim").__setitem__("schema_version", "v0"),
        lambda r: _nested(r, "launch_claim").__setitem__("worker_index", 2),
        lambda r: _nested(r, "launch_claim").__setitem__("run_id", "different"),
        lambda r: _nested(r, "inputs").__setitem__("schema_version", "v0"),
        lambda r: _nested(r, "inputs", "artifacts", "elmer", "reference").__setitem__(
            "path", "modes/wrong.h5"
        ),
        lambda r: _nested(r, "inputs", "artifacts", "elmer").__setitem__("bundle_sha256", "0" * 64),
        lambda r: _nested(r, "inputs", "fdtdx_fingerprint").__setitem__(
            "source_revision", "0" * 40
        ),
        lambda r: _nested(r, "inputs").__setitem__("runtime_module_sha256", {}),
        lambda r: _nested(r, "sources", "elmer").__setitem__("binding_sha256", "0" * 64),
        lambda r: _nested(r, "sources", "elmer", "binding").__setitem__("source_name", "other"),
        lambda r: _nested(r, "simulation").__setitem__("grid_shape_xyz", [32, 52, 36]),
        lambda r: _nested(r, "simulation").__setitem__("source_z_index", 5),
        lambda r: _nested(r, "simulation").__setitem__("detector_z_index", 23),
        lambda r: _nested(r, "simulation").__setitem__("boundaries", ["periodic"] * 6),
        lambda r: _nested(r, "simulation").__setitem__("core_cell_count", 40),
        lambda r: _nested(r, "numerics", "completed_step").__setitem__("elmer", 179),
        lambda r: _nested(r, "numerics", "all_fields_finite").__setitem__("jax", False),
        lambda r: _nested(r, "numerics", "final_e_l2").__setitem__("jax", 0.0),
        lambda r: _nested(r, "numerics").__setitem__("source_electric_relative_l2", 3.0e-5),
        lambda r: _nested(r, "numerics").__setitem__("source_magnetic_relative_l2", 3.0e-5),
        lambda r: _nested(r, "numerics").__setitem__("downstream_phasor_relative_l2", 3.0e-5),
        lambda r: _nested(r, "execution", "compiler_memory").__setitem__(
            "compiler_peak_bytes", 1001
        ),
        lambda r: _nested(r, "execution").__setitem__("execution_seconds", {}),
        lambda r: _nested(r, "execution").__setitem__("shared_compiled_pytree", False),
    ],
)
def test_waveguide_process_record_fails_closed(
    mutate: Callable[[dict[str, object]], object],
) -> None:
    records = _records()
    mutate(records[0])
    with pytest.raises(ValidationError, match="physical TPU FDTDX mode-source evidence"):
        aggregate_tpu_fdtdx_waveguide_source_process_evidence(records)


def test_waveguide_process_set_requires_records_and_canonical_json() -> None:
    with pytest.raises(ValidationError, match="requires at least one waveguide"):
        aggregate_tpu_fdtdx_waveguide_source_process_evidence([])
    records = _records()
    records[0]["not_json"] = {"set"}
    with pytest.raises(ValidationError, match="not canonical JSON"):
        aggregate_tpu_fdtdx_waveguide_source_process_evidence(records)


def test_waveguide_aggregate_cli_publishes_once(
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
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main([*(str(path) for path in inputs), "--output", str(output)])
    with pytest.raises(ValueError, match="must be unique"):
        main([str(inputs[0]), str(inputs[0]), "--output", str(tmp_path / "duplicate.json")])
