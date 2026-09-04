from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from femx.validation.tpu_distributed_electrothermal_evidence import (
    EXECUTABLE_NAMES,
    PROCESS_SET_EVIDENCE_SCHEMA,
    REAL_SCALAR_CONTRACT,
    TOLERANCES,
)

pytestmark = [
    pytest.mark.scientific,
    pytest.mark.requires_jax,
    pytest.mark.requires_accelerator,
    pytest.mark.multihost,
]

EVIDENCE_ENVIRONMENT = "FEMX_TPU_DISTRIBUTED_ELECTROTHERMAL_EVIDENCE"
EXPECTED_LOGICAL_SHA256 = "ba48ad3d6d6334ecae01db1effa63989d96118f6102729c3a91e10a4ae424b7f"


def _evidence() -> dict[str, Any]:
    configured = os.environ.get(EVIDENCE_ENVIRONMENT)
    if configured is None:
        pytest.skip(f"set {EVIDENCE_ENVIRONMENT} to the admitted physical process-set aggregate")
    path = Path(configured).resolve()
    if not path.is_file():
        pytest.fail(f"configured distributed electrothermal TPU evidence does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        pytest.fail("distributed electrothermal TPU evidence root must be an object")
    return value


def test_physical_tpu_distributed_electrothermal_forward_and_adjoint_evidence() -> None:
    evidence = _evidence()
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
    assert evidence["provenance"]["source_commit"] == ("6c344613f1bfacaaf39ebfa0751e0b1e85581b5e")

    runtime = evidence["runtime"]
    assert runtime["backend"] == "tpu"
    assert runtime["x64_enabled"] is False
    assert runtime["default_matmul_precision"] == "highest"
    assert runtime["process_count"] == 8
    assert runtime["local_device_count"] == 4
    assert runtime["global_device_count"] == 32
    assert runtime["device_kinds"] == ["TPU v4"]
    assert runtime["process_indexes"] == list(range(8))
    assert runtime["real_scalar_contract"] == REAL_SCALAR_CONTRACT

    problem = evidence["problem"]
    assert problem["node_count"] == 289
    assert problem["triangle_count"] == 512
    assert problem["free_dof_count"] == 255
    assert problem["partition_count"] == 32
    assert evidence["addressability"]["every_partition_addressable_once"] is True
    assert evidence["addressability"]["combined_partition_addressability_counts"] == [1] * 32

    numerics = evidence["numerics"]
    assert numerics["all_processes_forward_converged_and_finite"] is True
    assert numerics["all_processes_adjoint_converged_and_finite"] is True
    assert numerics["current_linear_backward_error"] <= 5.0e-7
    assert numerics["heat_linear_backward_error"] <= 5.0e-7
    assert numerics["current_coupled_residual_error"] <= 1.0e-4
    assert numerics["heat_coupled_residual_error"] <= 1.0e-4
    assert numerics["coupled_adjoint_backward_error"] <= 5.0e-4
    assert numerics["potential_relative_difference"] <= TOLERANCES["potential_relative_difference"]
    assert (
        numerics["temperature_relative_difference"] <= TOLERANCES["temperature_relative_difference"]
    )
    assert numerics["objective_relative_difference"] <= TOLERANCES["objective_relative_difference"]
    assert (
        max(numerics["explicit_gradient_relative_differences"])
        <= TOLERANCES["gradient_relative_difference"]
    )
    assert (
        max(numerics["native_gradient_explicit_relative_differences"])
        <= TOLERANCES["native_explicit_gradient_relative_difference"]
    )
    assert numerics["transfer_relative_error"] <= TOLERANCES["transfer_relative_error"]

    assert set(evidence["executables"]) == set(EXECUTABLE_NAMES)
    for executable in evidence["executables"].values():
        assert executable["stablehlo_collective_permute_count"] > 0
        assert executable["stablehlo_all_reduce_count"] > 0
        assert executable["stablehlo_all_gathers_absent_on_every_process"] is True
        assert len(executable["execution_ordinal_critical_path_seconds"]) == 5
        assert executable["maximum_compiler_hbm_fraction"] < 0.85
        assert executable["memory_scope"] == "compiler estimate; not live HBM usage"

    claim = evidence["claim_scope"]
    for boundary in (
        "not a scaling result",
        "not live HBM",
        "not fresh Elmer execution",
        "FDTDX composition",
        "3D production FEM",
        "foundry",
        "preemption-recovery",
    ):
        assert boundary in claim
