from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from femx.validation.tpu_scalar_h1_evidence import (
    EXECUTABLE_NAMES,
    PROCESS_SET_EVIDENCE_SCHEMA,
    REAL_SCALAR_CONTRACT,
)

pytestmark = [
    pytest.mark.scientific,
    pytest.mark.requires_jax,
    pytest.mark.requires_accelerator,
    pytest.mark.multihost,
]

EVIDENCE_ENVIRONMENT = "FEMX_TPU_SCALAR_H1_EVIDENCE"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _evidence() -> dict[str, Any]:
    configured = os.environ.get(EVIDENCE_ENVIRONMENT)
    if configured is None:
        pytest.skip(f"set {EVIDENCE_ENVIRONMENT} to an admitted physical TPU process-set aggregate")
    path = Path(configured).resolve()
    if not path.is_file():
        pytest.fail(f"configured scalar physical TPU evidence does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        pytest.fail("scalar physical TPU evidence root must be an object")
    return value


def test_physical_tpu_scalar_h1_evidence_admits_heat_current_cg_and_vjp() -> None:
    evidence = _evidence()
    assert evidence["schema_version"] == PROCESS_SET_EVIDENCE_SCHEMA
    assert evidence["status"] == "passed"
    provenance = evidence["provenance"]
    assert SHA256_PATTERN.fullmatch(provenance["source_digest"])
    assert SHA256_PATTERN.fullmatch(provenance["config_digest"])
    runtime = evidence["runtime"]
    problem = evidence["problem"]
    assert runtime["backend"] == "tpu"
    assert runtime["x64_enabled"] is False
    assert runtime["default_matmul_precision"] == "highest"
    assert runtime["real_scalar_contract"] == REAL_SCALAR_CONTRACT
    assert runtime["process_count"] >= 2
    assert runtime["global_device_count"] == (
        runtime["process_count"] * runtime["local_device_count"]
    )
    assert runtime["process_indexes"] == list(range(runtime["process_count"]))
    assert len(provenance["process_records"]) == runtime["process_count"]
    assert all(
        SHA256_PATTERN.fullmatch(record["sha256"]) for record in provenance["process_records"]
    )
    assert problem["partition_count"] == runtime["global_device_count"]
    assert problem["triangle_count"] >= 2 * problem["partition_count"]
    assert problem["halo_link_count"] > 0
    assert evidence["addressability"]["every_partition_addressable_once"] is True
    assert (
        evidence["addressability"]["combined_partition_addressability_counts"]
        == [1] * runtime["global_device_count"]
    )

    tolerances = evidence["tolerances"]
    for name in ("heat", "current"):
        case = evidence["cases"][name]
        assert case["all_processes_converged_and_finite"] is True
        assert (
            case["maximum_solution_relative_difference_across_processes"]
            <= tolerances["solution_relative_difference"]
        )
        assert (
            case["maximum_rhs_relative_difference_across_processes"]
            <= tolerances["rhs_relative_difference"]
        )
        assert (
            max(
                case["maximum_matrix_vjp_relative_difference_across_processes"],
                case["maximum_cell_rhs_vjp_relative_difference_across_processes"],
            )
            <= tolerances["vjp_relative_difference"]
        )
        assert (
            case["maximum_host_precision_relative_difference_across_processes"]
            <= tolerances["host_precision_relative_difference"]
        )

    assert set(evidence["executables"]) == set(EXECUTABLE_NAMES)
    for executable in evidence["executables"].values():
        assert executable["process_count"] == runtime["process_count"]
        assert executable["stablehlo_collective_permute_count"] > 0
        assert executable["stablehlo_all_reduce_count"] > 0
        assert executable["stablehlo_all_gathers_absent_on_every_process"] is True
        assert len(executable["execution_ordinal_critical_path_seconds"]) == 5
        assert executable["maximum_compiler_hbm_fraction"] < 0.85
        assert executable["worst_compiler_hbm_risk"] in {"safe", "elevated"}

    claim = evidence["claim_scope"]
    assert "process-complete" in claim
    assert "not a scaling result" in claim
    assert "not Elmer parity" in claim
    assert "foundry prediction" in claim
    assert "Spot-preemption recovery" in claim
