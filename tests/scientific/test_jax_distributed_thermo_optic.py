from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.scientific, pytest.mark.requires_jax]


def test_four_device_thermo_optic_transfer_is_partitioned_and_differentiable() -> None:
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    environment["JAX_ENABLE_X64"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root)))
    completed = subprocess.run(
        [sys.executable, str(root / "tests" / "distributed_thermo_optic_cpu_probe.py")],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == ("femx.fdtdx.distributed_thermo_optic.cpu_portability/v1")
    assert payload["backend"] == "cpu"
    assert payload["forced_cpu_device_count"] == 4
    assert payload["process_count"] == 1
    assert payload["source_cell_count"] == 32
    assert payload["target_shape"] == [8, 2, 4]
    assert payload["transfer_capacity"] > 0
    assert payload["all_valid"] is True
    assert payload["parameter_sharding"] == "P('shard',)"
    assert payload["temperature_relative_difference"] < 2.0e-12
    assert payload["permittivity_relative_difference"] < 2.0e-12
    assert payload["parameter_relative_difference"] < 2.0e-12
    assert payload["objective_relative_difference"] < 2.0e-12
    assert max(payload["dense_gradient_relative_errors"]) < 2.0e-8
    assert max(payload["finite_difference_relative_errors"]) < 3.0e-6
    for report in payload["stablehlo"].values():
        assert report["all_to_all_count"] > 0
        assert report["collective_permute_count"] > 0
        assert report["all_reduce_count"] > 0
        assert report["contains_all_gather"] is False
    assert payload["claim_scope"] == (
        "single-process four-forced-CPU portability for sharded electrothermal P1 sampling "
        "and thermo-optic material parameters; not FDTDX time integration, physical TPU, "
        "multi-host, scaling, 3D FEM, calibrated material, or device evidence"
    )
