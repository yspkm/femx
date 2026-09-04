from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from femx.validation.tpu_distributed_fdtdx_thermo_optic_evidence import (
    EXECUTABLE_NAMES,
    EXPECTED_DEVICE_KINDS,
    EXPECTED_GLOBAL_DEVICE_COUNT,
    EXPECTED_LOCAL_DEVICE_COUNT,
    EXPECTED_PROCESS_COUNT,
    FDTDX_SOURCE_DIGEST,
    FDTDX_SOURCE_REVISION,
    PROCESS_SET_EVIDENCE_SCHEMA,
    SCALAR_CONTRACT,
    TOLERANCES,
)

pytestmark = [
    pytest.mark.scientific,
    pytest.mark.requires_jax,
    pytest.mark.requires_fdtdx,
    pytest.mark.requires_accelerator,
    pytest.mark.multihost,
]

EVIDENCE_ENVIRONMENT = "FEMX_TPU_DISTRIBUTED_FDTDX_THERMO_OPTIC_EVIDENCE"
EXPECTED_LOGICAL_SHA256: str | None = (
    "1dd42ac8f51bff53a17814e7f923581f08fd2dd2f1aec11604223b33354e8654"
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


def _evidence() -> dict[str, Any]:
    configured = os.environ.get(EVIDENCE_ENVIRONMENT)
    if configured is None:
        pytest.skip(f"set {EVIDENCE_ENVIRONMENT} to the admitted physical process-set aggregate")
    supplied = Path(configured)
    path = supplied.resolve()
    if supplied.is_symlink() or not path.is_file():
        pytest.fail(f"configured distributed FEM-to-FDTDX TPU evidence is invalid: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        pytest.fail("distributed FEM-to-FDTDX TPU evidence root must be an object")
    return value


def test_physical_tpu_distributed_fdtdx_forward_and_adjoint_evidence() -> None:
    evidence = _evidence()
    if EXPECTED_LOGICAL_SHA256 is None:
        pytest.fail("pin the admitted physical aggregate digest before accepting M2e.7b")
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

    source = evidence["source"]
    assert GIT_REVISION_PATTERN.fullmatch(source["commit"])
    assert SHA256_PATTERN.fullmatch(source["source_sha256"])
    assert SHA256_PATTERN.fullmatch(source["config_sha256"])

    runtime = evidence["runtime"]
    assert runtime["backend"] == "tpu"
    assert runtime["process_count"] == EXPECTED_PROCESS_COUNT
    assert runtime["local_device_count"] == EXPECTED_LOCAL_DEVICE_COUNT
    assert runtime["global_device_count"] == EXPECTED_GLOBAL_DEVICE_COUNT
    assert runtime["device_kinds"] == list(EXPECTED_DEVICE_KINDS)
    assert runtime["scalar_contract"] == dict(SCALAR_CONTRACT)

    inputs = evidence["input"]
    assert inputs["fdtdx_source_revision"] == FDTDX_SOURCE_REVISION
    assert inputs["fdtdx_source_digest"] == FDTDX_SOURCE_DIGEST
    for name in (
        "manifest_sha256",
        "arrays_sha256",
        "electrothermal_arrays_sha256",
        "sampling_operator_sha256",
        "transfer_operator_sha256",
        "scene_sha256",
    ):
        assert SHA256_PATTERN.fullmatch(inputs[name])

    plan = evidence["plan"]
    assert plan["partition_count"] == EXPECTED_GLOBAL_DEVICE_COUNT
    assert plan["node_count"] == 289
    assert plan["triangle_count"] == 512
    assert plan["free_dof_count"] == 255
    assert evidence["array_admission"]["every_partition_addressable_once"] is True
    assert evidence["coordinate_admission"]["admitted"] == [True, True, True]
    assert max(evidence["coordinate_admission"]["maximum_ulp_errors"]) <= 8

    numerics = evidence["numerics"]
    assert numerics["finite"] is True
    assert numerics["forward_converged"] is True
    assert numerics["adjoint_converged"] is True
    assert numerics["thermo_optic_all_valid"] is True
    assert numerics["material_destination_sharding_preserved"] is True
    for field in (
        "potential_relative_difference",
        "temperature_relative_difference",
        "cell_temperature_relative_difference",
        "parameter_relative_difference",
        "material_relative_difference",
        "objective_explicit_relative_difference",
        "current_residual_error",
        "heat_residual_error",
        "current_linear_backward_error",
        "heat_linear_backward_error",
        "adjoint_backward_error",
        "transfer_relative_error",
    ):
        tolerance_name = field
        if field in {"current_linear_backward_error", "heat_linear_backward_error"}:
            tolerance_name = "linear_backward_error"
        assert numerics[field] <= TOLERANCES[tolerance_name]
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
        assert (
            executable["maximum_compiler_hbm_fraction"]
            < (TOLERANCES["maximum_compiler_hbm_fraction"])
        )
        assert len(executable["execution_ordinal_critical_path_seconds"]) == 3

    process_records = evidence["process_records"]
    assert [record["process_index"] for record in process_records] == list(
        range(EXPECTED_PROCESS_COUNT)
    )
    assert all(SHA256_PATTERN.fullmatch(record["sha256"]) for record in process_records)
    for excluded_claim in (
        "not 3D FEM",
        "ring convergence",
        "S-parameters",
        "scaling",
        "live HBM",
        "measured-device",
        "foundry",
        "preemption-recovery",
    ):
        assert excluded_claim in evidence["claim_scope"]
