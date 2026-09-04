from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from femx.validation.tpu_collective_evidence import PROCESS_SET_EVIDENCE_SCHEMA

pytestmark = [
    pytest.mark.scientific,
    pytest.mark.requires_jax,
    pytest.mark.requires_accelerator,
    pytest.mark.multihost,
]

EVIDENCE_ENVIRONMENT = "FEMX_TPU_COLLECTIVE_EVIDENCE"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _evidence() -> dict[str, Any]:
    configured = os.environ.get(EVIDENCE_ENVIRONMENT)
    if configured is None:
        pytest.skip(f"set {EVIDENCE_ENVIRONMENT} to an admitted physical TPU process-set aggregate")
    path = Path(configured).resolve()
    if not path.is_file():
        pytest.fail(f"configured physical TPU evidence does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        pytest.fail("physical TPU evidence root must be an object")
    return value


def test_physical_tpu_collective_evidence_admits_exact_topology_action_and_vjp() -> None:
    evidence = _evidence()
    assert evidence["schema_version"] == PROCESS_SET_EVIDENCE_SCHEMA
    assert evidence["status"] == "passed"

    provenance = evidence["provenance"]
    assert isinstance(provenance["run_id"], str) and provenance["run_id"]
    assert SHA256_PATTERN.fullmatch(provenance["source_digest"])
    assert SHA256_PATTERN.fullmatch(provenance["config_digest"])

    runtime = evidence["runtime"]
    problem = evidence["problem"]
    assert runtime["backend"] == "tpu"
    assert runtime["x64_enabled"] is False
    assert runtime["default_matmul_precision"] == "highest"
    assert runtime["complex_scalar_contract"] == {
        "logical_dtype": "complex64",
        "matrix_dtype": "float32",
        "index_dtype": "int32",
        "execution_representation": "native complex64",
        "matmul_precision": "highest",
        "host_reference_dtype": "complex128",
        "precision_fallback": False,
    }
    assert runtime["process_count"] >= 2
    assert runtime["local_device_count"] >= 1
    assert runtime["global_device_count"] == (
        runtime["process_count"] * runtime["local_device_count"]
    )
    assert runtime["process_indexes"] == list(range(runtime["process_count"]))
    assert len(provenance["process_records"]) == runtime["process_count"]
    assert [record["process_index"] for record in provenance["process_records"]] == (
        runtime["process_indexes"]
    )
    assert all(
        SHA256_PATTERN.fullmatch(record["sha256"]) for record in provenance["process_records"]
    )
    assert problem["partition_count"] == runtime["global_device_count"]
    assert problem["triangle_count"] >= 2 * runtime["global_device_count"]
    assert problem["halo_link_count"] > 0
    assert problem["halo_value_count"] > 0

    assert evidence["addressability"]["every_partition_addressable_once"] is True
    assert (
        evidence["addressability"]["combined_partition_addressability_counts"]
        == [1] * runtime["global_device_count"]
    )

    checkpoint = evidence["checkpoint"]
    assert checkpoint["mode"] in {"fresh-process-roundtrip", "restored-external-fragment"}
    assert checkpoint["checkpoint_id"] == "port-collective-step-000000"
    assert checkpoint["step"] == 0
    assert checkpoint["complete_process_fragment_count"] == runtime["process_count"]
    assert checkpoint["same_topology_only"] is True

    numerics = evidence["numerics"]
    assert (
        numerics["maximum_action_relative_difference_across_processes"]
        <= numerics["action_tolerance"]
    )
    assert numerics["maximum_vjp_relative_difference_across_processes"] <= numerics["vjp_tolerance"]
    assert (
        numerics["maximum_host_c64_vs_c128_relative_difference_across_processes"]
        <= numerics["host_precision_tolerance"]
    )
    assert numerics["all_process_actions_finite"] is True
    assert numerics["all_process_vjps_finite"] is True

    executables = evidence["executables"]
    assert set(executables) == {"real_forward", "complex_forward", "real_vjp", "complex_vjp"}
    expected_forward_permutations = 2 * problem["halo_link_count"]
    for name, record in executables.items():
        assert record["process_count"] == runtime["process_count"]
        assert record["stablehlo_all_gathers_absent_on_every_process"] is True
        assert len(record["process_execution_median_seconds"]) == runtime["process_count"]
        critical_path = record["execution_ordinal_critical_path_seconds"]
        assert len(critical_path) == 5
        assert record["execution_ordinal_critical_path_summary_seconds"]["max"] == max(
            critical_path
        )
        assert record["worst_compiler_hbm_risk"] in {"safe", "elevated"}
        assert record["maximum_compiler_hbm_fraction"] < 0.85
        assert "ordinal maximum" in record["sample_alignment"]
        if name.endswith("forward"):
            assert record["stablehlo_collective_permute_count"] == (expected_forward_permutations)

    assert "process-complete" in evidence["claim_scope"]
    assert "not a scaling result" in evidence["claim_scope"]
    assert "not eigensolve scaling" in evidence["claim_scope"]
    assert "preemption recovery" in evidence["claim_scope"]
    assert "not" in evidence["claim_scope"] and "Elmer parity" in evidence["claim_scope"]
