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
PROBE = REPOSITORY_ROOT / "tests" / "collective_scalar_multilevel_cpu_probe.py"


def test_four_device_multilevel_pcg_controls_refinement_contrast_and_adjoint() -> None:
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
        timeout=300.0,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert payload["schema_version"] == "femx.jax.scalar_h1_multilevel_pcg.cpu_portability/v1"
    assert payload["backend"] == "cpu"
    assert payload["forced_cpu_device_count"] == 4
    assert payload["process_count"] == 1
    assert payload["local_device_count"] == payload["global_device_count"] == 4

    refinement = payload["refinement_reports"]
    assert set(refinement) == {"8", "16", "32"}
    assert refinement["8"]["pcg_iterations"] > refinement["8"]["cg_iterations"]
    assert refinement["32"]["pcg_iterations"] <= 0.55 * refinement["32"]["cg_iterations"]
    assert refinement["32"]["pcg_iterations"] <= 1.3 * refinement["16"]["pcg_iterations"]
    assert refinement["32"]["cg_iterations"] >= 1.8 * refinement["16"]["cg_iterations"]

    contrast = payload["coefficient_contrast_reports"]
    assert set(contrast) == {"1e+00", "1e+02", "1e+04"}
    assert contrast["1e+04"]["pcg_iterations"] <= 0.1 * contrast["1e+04"]["cg_iterations"]
    assert contrast["1e+04"]["pcg_iterations"] <= 1.3 * contrast["1e+00"]["pcg_iterations"]

    for report in (*refinement.values(), *contrast.values()):
        assert report["cg_relative_residual"] < 1.0e-9
        assert report["pcg_relative_residual"] < 1.0e-9
        assert report["solution_relative_difference"] < 2.0e-8
    assert all(len(value) == 64 for value in payload["hierarchy_sha256"].values())
    assert payload["gradient_relative_error"] < 5.0e-7
    assert payload["stablehlo_collective_permute_count"] > 0
    assert payload["stablehlo_all_reduce_count"] > 0
    assert not payload["stablehlo_contains_all_gather"]
    assert payload["claim_scope"] == (
        "forced multi-CPU multilevel-PCG portability, refinement, contrast, and adjoint; "
        "not accelerator or multi-host evidence"
    )
