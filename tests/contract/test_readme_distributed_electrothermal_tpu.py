from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from femx.validation.tpu_distributed_electrothermal_evidence import (
    PROCESS_SET_EVIDENCE_SCHEMA,
)

pytestmark = pytest.mark.contract

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = REPOSITORY_ROOT / "docs/assets/readme/distributed_electrothermal_tpu/evidence.json"
SCOPE_PATH = REPOSITORY_ROOT / "docs/assets/readme/distributed_electrothermal_tpu/README.md"
EXPECTED_LOGICAL_SHA256 = "ba48ad3d6d6334ecae01db1effa63989d96118f6102729c3a91e10a4ae424b7f"


def _all_mapping_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key) for key in value}
        for item in value.values():
            keys.update(_all_mapping_keys(item))
        return keys
    if isinstance(value, list):
        list_keys: set[str] = set()
        for item in value:
            list_keys.update(_all_mapping_keys(item))
        return list_keys
    return set()


def test_tracked_distributed_electrothermal_tpu_evidence_is_bound_and_public_safe() -> None:
    evidence: dict[str, Any] = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    canonical = json.dumps(
        evidence,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == EXPECTED_LOGICAL_SHA256
    assert evidence["schema_version"] == PROCESS_SET_EVIDENCE_SCHEMA
    assert evidence["status"] == "passed"
    assert evidence["runtime"]["process_count"] == 8
    assert evidence["runtime"]["global_device_count"] == 32
    assert len(evidence["provenance"]["process_records"]) == 8
    assert evidence["addressability"]["combined_partition_addressability_counts"] == [1] * 32

    forbidden_keys = {
        "cloud_project",
        "hostname",
        "machine_address",
        "profile",
        "resource_name",
        "run_id",
        "worker_index",
        "worker_mapping",
        "zone",
    }
    assert _all_mapping_keys(evidence).isdisjoint(forbidden_keys)
    scope = SCOPE_PATH.read_text(encoding="utf-8")
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    verification = (REPOSITORY_ROOT / "docs/VERIFICATION.md").read_text(encoding="utf-8")
    adr = (
        REPOSITORY_ROOT
        / "docs/adr/0048-process-complete-tpu-distributed-electrothermal-evidence.md"
    ).read_text(encoding="utf-8")
    for document in (scope, verification, adr):
        assert EXPECTED_LOGICAL_SHA256 in document
    assert "distributed_electrothermal_tpu/evidence.json" in readme
    assert "3D production" in scope
    assert "not live HBM" in evidence["claim_scope"]
