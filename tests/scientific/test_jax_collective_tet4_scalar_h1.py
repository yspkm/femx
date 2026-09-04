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
PROBE = REPOSITORY_ROOT / "tests" / "collective_tet4_scalar_cpu_probe.py"


def test_four_device_cpu_tet4_scalar_solve_and_vjp_preserve_dense_authority() -> None:
    environment = os.environ.copy()
    environment["JAX_ENABLE_X64"] = "1"
    environment["JAX_PLATFORMS"] = "cpu"
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

    assert payload["schema_version"] == "femx.jax.tet4_scalar_collective.cpu_portability/v1"
    assert payload["layout_schema_version"] == "femx.jax.scalar_h1_collective/v2"
    assert payload["backend"] == "cpu"
    assert payload["forced_cpu_device_count"] == 4
    assert payload["process_count"] == 1
    assert payload["local_device_count"] == payload["global_device_count"] == 4
    assert payload["node_count"] == 45
    assert payload["tet4_cell_count"] == 96
    assert set(payload["reports"]) == {"1", "2", "4"}
    for report in payload["reports"].values():
        assert 0 < report["iterations"] <= 100
        assert report["relative_residual"] < 2.0e-10
        assert report["solution_relative_difference"] < 2.0e-10
        assert report["rhs_relative_difference"] < 2.0e-15
        assert len(report["layout_sha256"]) == 64
    assert payload["maximum_solution_relative_difference"] < 2.0e-10
    assert payload["maximum_rhs_relative_difference"] < 2.0e-15
    assert payload["gradient_relative_error"] < 2.0e-7
    assert payload["stablehlo_collective_permute_count"] > 0
    assert payload["stablehlo_all_reduce_count"] > 0
    assert not payload["stablehlo_contains_all_gather"]
    assert payload["claim_scope"] == (
        "forced multi-CPU Tet4 scalar operator/RHS/CG/VJP portability; not accelerator, "
        "multi-host, ring-heater, or Elmer parity evidence"
    )
