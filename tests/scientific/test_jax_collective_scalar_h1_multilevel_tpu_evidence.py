from __future__ import annotations

import json
import os
import re
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from femx.validation.tpu_scalar_h1_multilevel_evidence import (
    ITERATION_ADMISSION,
    LOCKED_POLICY,
    MULTILEVEL_EXECUTABLE_NAMES,
    MULTILEVEL_PROCESS_SET_EVIDENCE_SCHEMA,
    PARTITIONED_TRANSFER_NAMES,
    REPLICATION_INTENT,
)

pytestmark = [
    pytest.mark.scientific,
    pytest.mark.requires_jax,
    pytest.mark.requires_accelerator,
    pytest.mark.multihost,
]

EVIDENCE_ENVIRONMENT = "FEMX_TPU_SCALAR_H1_MULTILEVEL_EVIDENCE"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _evidence() -> dict[str, Any]:
    configured = os.environ.get(EVIDENCE_ENVIRONMENT)
    if configured is None:
        pytest.skip(f"set {EVIDENCE_ENVIRONMENT} to an admitted physical TPU process-set aggregate")
    path = Path(configured).resolve()
    if not path.is_file():
        pytest.fail(f"configured multilevel physical TPU evidence does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        pytest.fail("multilevel physical TPU evidence root must be an object")
    return value


def test_physical_tpu_multilevel_evidence_admits_setup_pcg_and_implicit_vjp() -> None:
    evidence = _evidence()
    assert evidence["schema_version"] == MULTILEVEL_PROCESS_SET_EVIDENCE_SCHEMA
    assert evidence["status"] == "passed"
    runtime = evidence["runtime"]
    assert runtime["backend"] == "tpu"
    assert runtime["x64_enabled"] is False
    assert runtime["process_count"] >= 2
    assert runtime["global_device_count"] == (
        runtime["process_count"] * runtime["local_device_count"]
    )

    multilevel = evidence["multilevel"]
    hierarchy = multilevel["hierarchy"]
    assert multilevel["status"] == "passed"
    assert SHA256_PATTERN.fullmatch(hierarchy["sha256"])
    assert hierarchy["layout_sha256"] == evidence["problem"]["layout_sha256"]
    assert len(hierarchy["level_dof_counts"]) >= 2
    assert all(fine > coarse for fine, coarse in pairwise(hierarchy["level_dof_counts"]))
    assert hierarchy["level_dof_counts"][1] <= hierarchy["maximum_replicated_dofs"]
    assert all(SHA256_PATTERN.fullmatch(value) for value in hierarchy["prolongation_sha256"])
    assert multilevel["policy"] == {**LOCKED_POLICY, "iteration_admission": ITERATION_ADMISSION}

    transfer = multilevel["transfer"]
    assert transfer["partitioned_report_names"] == list(PARTITIONED_TRANSFER_NAMES)
    assert set(transfer["partitioned_global_shapes"]) == set(PARTITIONED_TRANSFER_NAMES)
    assert transfer["replicated_logical_bytes_per_device"] > 0
    assert transfer["replication_intent"] == REPLICATION_INTENT
    for case in multilevel["cases"].values():
        assert case["all_processes_converged_setup_valid_and_finite"] is True
        assert case["strict_iteration_improvement_on_every_process"] is True
        assert (
            case["maximum_multilevel_pcg_iterations_across_processes"]
            < case["maximum_unpreconditioned_cg_iterations_across_processes"]
        )
        assert case["maximum_relative_residual_across_processes"] <= 2.0e-5
        assert case["minimum_relative_diagonal_across_processes"] >= 1.0e-14
        assert case["maximum_relative_symmetry_error_across_processes"] <= 2.0e-6
        assert case["maximum_coarse_condition_number_across_processes"] <= 1.0e8

    assert set(multilevel["executables"]) == set(MULTILEVEL_EXECUTABLE_NAMES)
    for executable in multilevel["executables"].values():
        assert executable["process_count"] == runtime["process_count"]
        assert executable["stablehlo_collective_permute_count"] > 0
        assert executable["stablehlo_all_reduce_count"] > 0
        assert executable["stablehlo_all_gathers_absent_on_every_process"] is True
        assert len(executable["execution_ordinal_critical_path_seconds"]) == 5
        assert executable["maximum_compiler_hbm_fraction"] < 0.85

    claim = evidence["claim_scope"]
    assert "process-complete physical multi-host TPU" in claim
    assert "not a scaling result" in claim
    assert "not Elmer parity" in claim
    assert "foundry prediction" in claim
    assert "Spot-preemption recovery" in claim
