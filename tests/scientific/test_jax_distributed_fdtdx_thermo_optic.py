from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("fdtdx")

pytestmark = [
    pytest.mark.scientific,
    pytest.mark.requires_jax,
    pytest.mark.requires_fdtdx,
]

_FDTDX_SOURCE_REVISION = "0c05c4784b2be83b42d9b46ab089265981ba157f"
_FDTDX_SOURCE_DIGEST = "29bed9483c4c2b57fd2f495fdb47534edf6b244206679e34b2de41ec39aaa9fa"
_FDTDX_MODULE_SHA256 = {
    "__init__.py": "fcf000b7955c97e7fbe1ccd5901c1f5ba47a5bfd86f0fce3d2dc8be1bfe131cf",
    "core/jax/sharding.py": "a6e07ac439c1c1b48958380812406f090844a1b4924a3b3a9b0a7f49eca8a9c3",
    "fdtd/fdtd.py": "7c654097d43d5062afbef0cf8c479ba2a7db523b64683693fa4e24bc5070e4e0",
    "fdtd/initialization.py": "2b7d56d47789f38c73b96fe7a078521e1146a45e98753af6c5e536ea8f9225a1",
    "fdtd/wrapper.py": "97d562e0a33eeeccd6ce42d12cf8dc29f1e9fd071e561b5b009ac5da8ece7384",
}


def test_four_device_coupled_fdtdx_objective_is_partitioned_and_differentiable(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    environment["JAX_PLATFORMS"] = "cpu"
    environment["JAX_ENABLE_X64"] = "1"
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    python_path = (str(root / "src"), str(root))
    existing_python_path = environment.get("PYTHONPATH")
    if existing_python_path:
        python_path += (existing_python_path,)
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "tests" / "distributed_fdtdx_thermo_optic_cpu_probe.py"),
        ],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    payload = json.loads(completed.stdout)

    assert payload["schema_version"] == "femx.fdtdx.distributed_thermo_optic.cpu_objective/v1"
    assert payload["backend"] == "cpu"
    assert payload["forced_cpu_device_count"] == 4
    assert payload["process_count"] == 1
    assert payload["fdtdx_version"] == "0.6.2"
    assert payload["fdtdx_source_revision"] == _FDTDX_SOURCE_REVISION
    assert payload["fdtdx_source_digest"] == _FDTDX_SOURCE_DIGEST
    assert payload["fdtdx_module_sha256"] == _FDTDX_MODULE_SHA256
    assert payload["target_shape"] == [8, 2, 2]
    assert payload["material_sharding"] == "P(None, 'shard', None, None)"
    assert payload["thermo_optic_parameter_sharding"] == "P('shard',)"
    assert payload["all_valid"] is True
    assert payload["material_relative_difference"] < 2.0e-12
    assert payload["parameter_dense_relative_difference"] < 2.0e-12
    assert math.isfinite(payload["objective"])
    assert payload["objective"] > 0.0
    assert payload["gradients_finite"] == [True, True, True]
    assert payload["current_gradient_nonzero"] is True
    assert payload["finite_difference_relative_error"] < 2.0e-6
    assert payload["stablehlo"]["all_to_all_count"] > 0
    assert payload["stablehlo"]["collective_permute_count"] > 0
    assert payload["stablehlo"]["all_reduce_count"] > 0
    assert payload["stablehlo"]["contains_all_gather"] is False
    assert payload["claim_scope"] == (
        "single-process four-forced-CPU portability for the distributed coupled FEM "
        "adjoint, all-to-all P1 thermo-optic transfer, sharding-preserving FDTDX "
        "apply_params, checkpointed Maxwell time advance, and phasor objective; not "
        "physical TPU, multi-host, convergence, 3D FEM, S-parameters, calibrated "
        "material, or device evidence"
    )
