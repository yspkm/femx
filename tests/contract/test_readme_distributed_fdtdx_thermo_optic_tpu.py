from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from femx.validation.tpu_distributed_fdtdx_thermo_optic_evidence import (
    EXECUTABLE_NAMES,
    FDTDX_SOURCE_REVISION,
    PROCESS_SET_EVIDENCE_SCHEMA,
    SCALAR_CONTRACT,
    TOLERANCES,
)

pytestmark = pytest.mark.contract

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_ROOT = REPOSITORY_ROOT / "docs/assets/readme/distributed_fdtdx_thermo_optic_tpu"
EVIDENCE_PATH = BUNDLE_ROOT / "evidence.json"
SCOPE_PATH = BUNDLE_ROOT / "README.md"
EXPECTED_LOGICAL_SHA256 = "1dd42ac8f51bff53a17814e7f923581f08fd2dd2f1aec11604223b33354e8654"
FEMX_EVIDENCE_REVISION = "e5860a99e4bb3c49f6665f8c0f8e38d4d533fd3c"


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _evidence() -> dict[str, Any]:
    value = json.loads(
        EVIDENCE_PATH.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )
    assert isinstance(value, dict)
    return value


def test_retained_distributed_fdtdx_tpu_evidence_is_canonical_and_complete() -> None:
    evidence = _evidence()
    canonical = json.dumps(
        evidence,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == EXPECTED_LOGICAL_SHA256
    assert evidence["schema_version"] == PROCESS_SET_EVIDENCE_SCHEMA
    assert evidence["status"] == "passed"
    assert evidence["source"]["commit"] == FEMX_EVIDENCE_REVISION
    assert evidence["input"]["fdtdx_source_revision"] == FDTDX_SOURCE_REVISION

    runtime = evidence["runtime"]
    assert runtime["backend"] == "tpu"
    assert runtime["device_kinds"] == ["TPU v4"]
    assert runtime["process_count"] == 8
    assert runtime["local_device_count"] == 4
    assert runtime["global_device_count"] == 32
    assert runtime["scalar_contract"] == dict(SCALAR_CONTRACT)

    plan = evidence["plan"]
    assert plan["partition_count"] == 32
    assert plan["node_count"] == 289
    assert plan["triangle_count"] == 512
    assert plan["free_dof_count"] == 255
    assert evidence["array_admission"]["every_partition_addressable_once"] is True

    numerics = evidence["numerics"]
    assert numerics["finite"] is True
    assert numerics["forward_converged"] is True
    assert numerics["adjoint_converged"] is True
    assert numerics["potential_cotangent_norm"] == 0.0
    assert numerics["cell_cotangent_norm"] > 0.0
    assert numerics["temperature_cotangent_norm"] > 0.0
    assert (
        max(numerics["native_explicit_gradient_relative_differences"])
        <= (TOLERANCES["native_explicit_gradient_relative_difference"])
    )
    assert (
        max(numerics["applied_voltage_finite_difference"]["relative_errors"])
        <= (TOLERANCES["finite_difference_relative_error"])
    )

    assert set(evidence["executables"]) == set(EXECUTABLE_NAMES)
    for executable in evidence["executables"].values():
        assert executable["stablehlo_all_to_all_count"] > 0
        assert executable["stablehlo_collective_permute_count"] > 0
        assert executable["stablehlo_all_reduce_count"] > 0
        assert executable["stablehlo_all_gathers_absent_on_every_process"] is True
        assert executable["stablehlo_float64_absent_on_every_process"] is True
        assert executable["maximum_compiler_hbm_fraction"] < 0.85

    assert [record["process_index"] for record in evidence["process_records"]] == list(range(8))
    assert all(len(record["sha256"]) == 64 for record in evidence["process_records"])


def test_public_bundle_documents_scope_and_omits_private_infrastructure() -> None:
    scope = SCOPE_PATH.read_text(encoding="utf-8")
    serialized = EVIDENCE_PATH.read_text(encoding="utf-8")
    assert FEMX_EVIDENCE_REVISION in scope
    assert FDTDX_SOURCE_REVISION in scope
    assert EXPECTED_LOGICAL_SHA256 in scope
    assert "not 3D FEM" in scope
    assert "distributed_fdtdx_thermo_optic_tpu/evidence.json" in (
        REPOSITORY_ROOT / "README.md"
    ).read_text(encoding="utf-8")
    roadmap = " ".join((REPOSITORY_ROOT / "docs/ROADMAP.md").read_text(encoding="utf-8").split())
    assert "physical eight-process, 32-device TPU v4 process set complete" in roadmap
    for relative in ("docs/VERIFICATION.md", "docs/INTEROPERABILITY.md"):
        assert EXPECTED_LOGICAL_SHA256 in (REPOSITORY_ROOT / relative).read_text(encoding="utf-8")

    for forbidden in (
        '"hostname"',
        '"run_id"',
        '"profile"',
        '"worker_index"',
        "/home/",
        "/tmp/",
        "t1v-",
        "v4-od-32",
        "europe-west",
        "us-central",
        "us-east",
    ):
        assert forbidden not in serialized
