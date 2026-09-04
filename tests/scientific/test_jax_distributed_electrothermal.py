from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.scientific, pytest.mark.requires_jax]


def test_four_device_coupled_forward_adjoint_and_stablehlo_are_partition_invariant() -> None:
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    environment["JAX_ENABLE_X64"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root)))
    completed = subprocess.run(
        [sys.executable, str(root / "tests" / "distributed_electrothermal_cpu_probe.py")],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=240,
    )
    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == "femx.jax.distributed_electrothermal.cpu_portability/v1"
    assert payload["backend"] == "cpu"
    assert payload["forced_cpu_device_count"] == 4
    assert payload["process_count"] == 1
    assert set(payload["partition_reports"]) == {"1", "2", "4"}
    for count, report in payload["partition_reports"].items():
        assert report["converged"] is True
        assert report["adjoint_converged"] is True
        assert report["potential_relative_difference"] < 2.0e-12
        assert report["temperature_relative_difference"] < 2.0e-12
        assert report["current_gradient_relative_difference"] < 2.0e-8
        assert report["thermal_gradient_relative_difference"] < 2.0e-8
        assert report["feedback_gradient_relative_difference"] < 2.0e-8
        assert report["current_residual_error"] < 1.0e-10
        assert report["heat_residual_error"] < 1.0e-10
        assert report["transfer_relative_error"] < 2.0e-15
        assert report["adjoint_backward_error"] < 2.0e-10
        if count != "1":
            assert report["halo_link_count"] > 0
    assert payload["objective_relative_difference"] < 1.0e-11
    assert max(payload["native_gradient_relative_errors"]) < 2.0e-10
    assert max(payload["finite_difference_relative_errors"]) < 3.0e-6
    for report in payload["stablehlo"].values():
        assert report["collective_permute_count"] > 0
        assert report["all_reduce_count"] > 0
        assert report["contains_all_gather"] is False
    assert payload["claim_scope"] == (
        "single-process four-forced-CPU portability for one same-mesh coupled residual; "
        "not physical accelerator, multi-host, scaling, foundry, or FDTDX evidence"
    )
