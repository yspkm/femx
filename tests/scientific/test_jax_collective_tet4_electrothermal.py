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
PROBE = REPOSITORY_ROOT / "tests" / "collective_tet4_electrothermal_cpu_probe.py"


def test_four_device_cpu_tet4_electrothermal_path_preserves_partition_authority() -> None:
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
        timeout=240.0,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert payload["schema_version"] == "femx.jax.tet4_electrothermal.cpu_portability/v1"
    assert payload["backend"] == "cpu"
    assert payload["forced_cpu_device_count"] == 4
    assert payload["process_count"] == 1
    assert payload["local_device_count"] == payload["global_device_count"] == 4
    assert payload["thermal_node_count"] == 125
    assert payload["thermal_tet4_cell_count"] == 384
    assert payload["current_node_count"] == 75
    assert payload["current_tet4_cell_count"] == 192
    assert set(payload["partition_reports"]) == {"1", "2", "4"}
    for report in payload["partition_reports"].values():
        assert 0 < report["current_iterations"] <= 600
        assert 0 < report["thermal_iterations"] <= 600
        assert report["current_backward_error"] < 2.0e-12
        assert report["thermal_backward_error"] < 2.0e-12
        assert report["charge_balance_relative_error"] < 2.0e-10
        assert report["electrical_energy_relative_error"] < 2.0e-10
        assert report["joule_transfer_relative_error"] < 2.0e-14
        assert report["thermal_balance_relative_error"] < 2.0e-10
        assert len(report["current_layout_sha256"]) == 64
        assert len(report["thermal_layout_sha256"]) == 64
        assert len(report["plan_sha256"]) == 64
    assert payload["maximum_potential_relative_difference"] < 2.0e-10
    assert payload["maximum_temperature_relative_difference"] < 2.0e-10
    assert payload["gradient_relative_error"] < 2.0e-7
    assert payload["stablehlo_collective_permute_count"] > 0
    assert payload["stablehlo_all_reduce_count"] > 0
    assert not payload["stablehlo_contains_all_gather"]
    assert payload["claim_scope"] == (
        "forced single-process four-CPU-device Tet4 current/Joule/heat/VJP portability with "
        "an exact parent-cell transfer; not TPU, multi-host, public-ring, or Elmer evidence"
    )
