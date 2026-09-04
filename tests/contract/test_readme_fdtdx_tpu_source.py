from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from femx.validation.tpu_fdtdx_mode_source_evidence import PROCESS_SET_EVIDENCE_SCHEMA

pytestmark = pytest.mark.contract

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_ROOT = REPO_ROOT / "docs" / "assets" / "readme" / "fdtdx_tpu_source"
EVIDENCE_PATH = BUNDLE_ROOT / "evidence.json"
LOGICAL_SHA256 = "4bdd3e2642b8e0fb86340a0b2f9f87df3f156912698853d4e131b2abf432c189"
FEMX_EVIDENCE_REVISION = "6c21321006302a81972efc29c7d3128672cf460e"
FDTDX_REVISION = "81a58da9cde4a4ff822f835b63597c0d0d8ba978"


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


def test_retained_tpu_fdtdx_source_process_set_is_canonical_and_complete() -> None:
    evidence = _evidence()
    canonical = json.dumps(
        evidence,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()

    assert evidence["schema_version"] == PROCESS_SET_EVIDENCE_SCHEMA
    assert evidence["status"] == "passed"
    assert hashlib.sha256(canonical).hexdigest() == LOGICAL_SHA256

    runtime = evidence["runtime"]
    assert runtime["backend"] == "tpu"
    assert runtime["device_kinds"] == ["TPU v5 lite"]
    assert runtime["process_count"] == 4
    assert runtime["local_device_count"] == 4
    assert runtime["global_device_count"] == 16
    assert runtime["process_indexes"] == [0, 1, 2, 3]
    assert runtime["worker_indexes"] == [0, 1, 2, 3]
    assert runtime["scalar_contract"] == {
        "field_dtype": "float32",
        "mode_dtype": "complex64",
        "precision_fallback": False,
        "time_offset_dtype": "float32",
        "x64_enabled": False,
    }

    records = evidence["provenance"]["process_records"]
    assert [record["process_index"] for record in records] == [0, 1, 2, 3]
    assert all(len(record["sha256"]) == 64 for record in records)
    assert evidence["source"]["fdtdx_fingerprint"]["source_revision"] == FDTDX_REVISION
    assert evidence["source"]["global_shape"] == [3, 32, 8, 1]
    assert evidence["source"]["combined_addressable_x_ranges"] == [
        [start, start + 2] for start in range(0, 32, 2)
    ]
    assert evidence["source"]["every_global_source_shard_addressable_once"] is True

    assert evidence["simulation"]["grid_shape_xyz"] == [32, 8, 20]
    assert evidence["simulation"]["time_steps"] == 66
    assert evidence["numerics"]["all_process_fields_finite"] is True
    assert evidence["numerics"]["all_processes_completed_same_step"] is True
    assert evidence["numerics"]["downstream_e_l2"] > 0.0
    assert evidence["execution"]["stablehlo_all_gather_counts"] == [0, 0, 0, 0]

    assert "process-complete" in evidence["claim_scope"]
    for excluded_claim in ("Elmer parity", "convergence", "scaling", "adjoint", "recovery"):
        assert excluded_claim in evidence["claim_scope"]


def test_public_tpu_evidence_bundle_documents_revision_scope_and_omits_machine_identity() -> None:
    readme = (BUNDLE_ROOT / "README.md").read_text(encoding="utf-8")
    serialized = EVIDENCE_PATH.read_text(encoding="utf-8")

    assert FEMX_EVIDENCE_REVISION in readme
    assert LOGICAL_SHA256 in readme
    assert "not an Elmer or JAX waveguide" in readme
    for forbidden in (
        '"hostname"',
        '"ip_address"',
        '"project_id"',
        '"service_account"',
        "femx-v5e16-us-01",
        "us-central1-a",
    ):
        assert forbidden not in serialized
