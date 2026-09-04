from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("jax")

pytestmark = [pytest.mark.scientific, pytest.mark.requires_jax]

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROBE = REPOSITORY_ROOT / "tests" / "collective_scalar_tpu_harness_cpu_probe.py"


def test_four_device_cpu_smoke_preserves_physical_runner_input_and_vjp_contract() -> None:
    environment = os.environ.copy()
    environment["JAX_ENABLE_X64"] = "0"
    environment["JAX_PLATFORMS"] = "cpu"
    environment["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    python_path = (str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "src"))
    if environment.get("PYTHONPATH"):
        python_path = (*python_path, environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    completed = subprocess.run(
        (sys.executable, str(PROBE)),
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=180.0,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["schema_version"].endswith("tpu_harness_cpu_smoke/v1")
    assert payload["backend"] == "cpu"
    assert payload["x64_enabled"] is False
    assert payload["process_count"] == 1
    assert payload["global_device_count"] == 4
    assert len(payload["layout_sha256"]) == 64
    for case in payload["cases"].values():
        assert case["converged"] is True
        assert case["breakdown"] is False
        assert case["iterations"] <= 4000
        assert case["relative_residual"] <= 2.0e-5
        assert case["solution_relative_difference"] <= payload["tolerances"]["solution"]
        assert case["rhs_relative_difference"] <= payload["tolerances"]["rhs"]
        assert (
            max(
                case["matrix_vjp_relative_difference"],
                case["cell_rhs_vjp_relative_difference"],
            )
            <= payload["tolerances"]["vjp"]
        )
        assert case["host_precision_relative_difference"] <= payload["tolerances"]["host_precision"]
    for executable in payload["stablehlo"].values():
        assert executable["collective_permute_count"] > 0
        assert executable["all_reduce_count"] > 0
        assert executable["contains_all_gather"] is False
    assert payload["claim_scope"].endswith("not accelerator or multi-host evidence")
