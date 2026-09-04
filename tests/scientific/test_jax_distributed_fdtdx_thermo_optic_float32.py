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


def test_float32_coupled_fdtdx_objective_has_an_admitted_residual_adjoint(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    environment["JAX_PLATFORMS"] = "cpu"
    environment["JAX_ENABLE_X64"] = "0"
    environment["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    python_path = (str(root / "src"), str(root))
    existing_python_path = environment.get("PYTHONPATH")
    if existing_python_path:
        python_path += (existing_python_path,)
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    input_root = tmp_path / "coupled-input"
    builder_environment = dict(environment)
    builder_environment["JAX_ENABLE_X64"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from pathlib import Path; "
                "from scripts.build_tpu_distributed_fdtdx_thermo_optic_inputs import "
                "build_inputs; build_inputs(Path(sys.argv[1]), intervals=4, "
                "partition_count=4, require_clean=False)"
            ),
            str(input_root),
        ],
        cwd=root,
        env=builder_environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "tests" / "distributed_fdtdx_thermo_optic_float32_cpu_probe.py"),
            "--input",
            str(input_root),
        ],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=900,
    )
    payload = json.loads(completed.stdout)

    assert (
        payload["schema_version"] == "femx.fdtdx.distributed_thermo_optic.float32_cpu_objective/v1"
    )
    assert payload["backend"] == "cpu"
    assert payload["forced_cpu_device_count"] == 4
    assert payload["process_count"] == 1
    assert payload["x64_enabled"] is False
    assert payload["matmul_precision"] == "highest"
    assert payload["fdtdx_source_revision"] == _FDTDX_SOURCE_REVISION
    assert payload["fdtdx_source_digest"] == _FDTDX_SOURCE_DIGEST
    assert payload["grid_shape"] == [96, 4, 8]
    assert payload["device_shape"] == [32, 2, 4]
    assert payload["time_steps"] == 302
    assert payload["input_artifact"] is not None
    assert (
        payload["input_artifact"]["transfer_operator_sha256"] == payload["transfer_operator_sha256"]
    )
    assert payload["runtime_target_coordinate_admitted"] == [True, True, True]
    assert max(payload["runtime_target_coordinate_max_ulp_errors"]) <= 8
    assert max(payload["runtime_target_coordinate_max_grid_fraction_errors"]) <= 4.0e-6
    assert max(payload["runtime_target_coordinate_max_errors_m"]) <= 2.5e-13
    assert payload["material_sharding"] == "P(None, 'shard', None, None)"
    assert payload["thermo_optic_parameter_sharding"] == "P('shard', None, None)"
    assert payload["all_valid"] is True
    assert payload["material_relative_difference"] < 1.0e-6
    assert payload["parameter_canonical_relative_difference"] < 1.0e-6

    assert math.isfinite(payload["objective"])
    assert abs(payload["objective"]) > 1.0e-3
    assert payload["phasor_objective_scale"] == 1.0e8
    assert payload["reference_phasor"]["magnitude"] > 1.0e-6
    assert payload["native_gradients_finite"] == [True, True, True]
    assert payload["explicit_gradients_finite"] == [True, True, True]
    assert all(
        report["norm"] is not None and report["norm"] > 0.0
        for report in payload["native_gradient_reports"]
    )
    assert payload["objective_explicit_relative_difference"] < 1.0e-6
    assert all(
        difference < 1.0e-6
        for difference in payload["native_explicit_gradient_relative_differences"]
    )
    assert payload["adjoint_converged"] is True
    assert payload["adjoint_backward_error"] < 5.0e-4
    assert payload["cell_cotangent_norm"] > 0.0
    assert payload["state_cotangent_norms"]["potential"] == 0.0
    assert payload["state_cotangent_norms"]["temperature"] > 0.0
    finite_difference_errors = payload["applied_voltage_finite_difference_relative_errors"]
    assert finite_difference_errors["1.0e-01"] < 1.0e-2
    assert finite_difference_errors["5.0e-02"] < 1.0e-2

    for report in payload["stablehlo"].values():
        assert report["all_to_all_count"] > 0
        assert report["collective_permute_count"] > 0
        assert report["all_reduce_count"] > 0
        assert report["contains_all_gather"] is False
        assert report["contains_float64"] is False
    assert payload["claim_scope"] == (
        "single-process four-forced-CPU float32/complex64 admission of the physical-gate "
        "distributed FEM to all-to-all thermo-optic to checkpointed FDTDX objective graph; "
        "not physical TPU, multi-host, convergence, S-parameters, 3D FEM, calibrated "
        "material, or device evidence"
    )
