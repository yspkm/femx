from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from femx.validation.tpu_fdtdx_waveguide_source_evidence import PROCESS_SET_EVIDENCE_SCHEMA

pytestmark = [
    pytest.mark.scientific,
    pytest.mark.requires_jax,
    pytest.mark.requires_fdtdx,
    pytest.mark.requires_accelerator,
    pytest.mark.multihost,
]

EVIDENCE_ENVIRONMENT = "FEMX_TPU_WAVEGUIDE_SOURCE_EVIDENCE"


def _evidence() -> dict[str, Any]:
    configured = os.environ.get(EVIDENCE_ENVIRONMENT)
    if configured is None:
        pytest.skip(f"set {EVIDENCE_ENVIRONMENT} to an admitted physical TPU process set")
    path = Path(configured).resolve()
    if not path.is_file():
        pytest.fail(f"configured physical TPU waveguide evidence does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        pytest.fail("physical TPU waveguide evidence root must be an object")
    return value


def test_physical_tpu_waveguide_evidence_admits_both_fem_sources_and_downstream_parity() -> None:
    evidence = _evidence()
    assert evidence["schema_version"] == PROCESS_SET_EVIDENCE_SCHEMA
    assert evidence["status"] == "passed"
    assert evidence["runtime"]["backend"] == "tpu"
    assert evidence["runtime"]["process_count"] >= 2
    assert evidence["runtime"]["global_device_count"] == (
        evidence["runtime"]["process_count"] * evidence["runtime"]["local_device_count"]
    )
    assert evidence["simulation"]["grid_shape_xyz"] == [64, 52, 36]
    assert evidence["simulation"]["boundaries"] == ["pec", "pec", "pec", "pec", "pml", "pml"]
    for solver in ("elmer", "jax"):
        source = evidence["sources"][solver]
        assert source["every_global_source_shard_addressable_once"] is True
        assert (
            len(source["combined_addressable_x_ranges"])
            == evidence["runtime"]["global_device_count"]
        )
        assert len(source["process_bindings"]) == evidence["runtime"]["process_count"]
    numerics = evidence["numerics"]
    assert numerics["all_processes_completed_same_step"] is True
    assert numerics["all_process_fields_finite"] is True
    assert (
        numerics["source_electric_relative_l2"]
        <= numerics["thresholds"]["maximum_source_relative_l2"]
    )
    assert (
        numerics["source_magnetic_relative_l2"]
        <= numerics["thresholds"]["maximum_source_relative_l2"]
    )
    assert (
        numerics["downstream_phasor_relative_l2"]
        <= numerics["thresholds"]["maximum_downstream_relative_l2"]
    )
    assert "same-mesh silicon-waveguide" in evidence["claim_scope"]
    assert "S-parameters" in evidence["claim_scope"]
