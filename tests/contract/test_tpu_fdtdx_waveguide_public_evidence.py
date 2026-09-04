from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.contract

EVIDENCE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "assets"
    / "readme"
    / "fdtdx_tpu_waveguide_source"
    / "evidence.json"
)


def _contains_key(value: Any, forbidden: set[str]) -> bool:
    if isinstance(value, dict):
        return any(key in forbidden for key in value) or any(
            _contains_key(item, forbidden) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(item, forbidden) for item in value)
    return False


def test_public_tpu_waveguide_evidence_is_bounded_and_redacted() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    assert evidence["schema_version"] == (
        "femx.validation.fdtdx_waveguide_source.tpu_public_summary/v1"
    )
    assert evidence["process_set_schema_version"] == (
        "femx.validation.fdtdx_waveguide_source.tpu_process_set/v1"
    )
    assert evidence["status"] == "passed"
    assert evidence["runtime"]["backend"] == "tpu"
    assert evidence["runtime"]["process_count"] == 4
    assert evidence["runtime"]["global_device_count"] == 16
    assert evidence["simulation"]["time_steps"] == 316
    assert evidence["numerics"]["downstream_phasor_relative_l2"] < 2.0e-5
    assert all(
        source["every_global_source_shard_addressable_once"] is True
        for source in evidence["sources"].values()
    )
    assert evidence["public_provenance"]["canonical_process_set_sha256"] == (
        "e909db1632769775ee15c3927e48e636568746ef8f878140456a78a188e1cf56"
    )
    assert not _contains_key(
        evidence,
        {"cloud_project", "hostname", "ip", "node", "profile", "resource_name", "run_id", "zone"},
    )
    serialized = json.dumps(evidence, sort_keys=True).lower()
    assert "europe-west" not in serialized
    assert "euw4" not in serialized
