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
PROBE = REPOSITORY_ROOT / "tests" / "collective_port_cpu_probe.py"


def test_four_device_cpu_portability_preserves_action_vjp_and_pairwise_stablehlo() -> None:
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
        timeout=120.0,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert payload["schema_version"] == "femx.jax.port_collective.cpu_portability/v1"
    assert payload["backend"] == "cpu"
    assert payload["forced_cpu_device_count"] == 4
    assert payload["process_count"] == 1
    assert payload["local_device_count"] == payload["global_device_count"] == 4
    assert payload["partition_count"] == 4
    assert payload["global_cell_count"] == 32
    assert payload["global_dof_count"] == 49
    assert len(payload["layout_sha256"]) == 64
    mesh_report = payload["mesh_report"]
    assert mesh_report["layout_sha256"] == payload["layout_sha256"]
    assert mesh_report["axis_name"] == "partition"
    assert mesh_report["partition_count"] == 4
    assert mesh_report["global_device_count"] == 4
    assert mesh_report["addressable_device_count"] == 4
    assert mesh_report["process_count"] == 1
    assert not mesh_report["is_multi_process"]
    assert [assignment["partition_index"] for assignment in mesh_report["assignments"]] == [
        0,
        1,
        2,
        3,
    ]
    assert all(assignment["addressable"] for assignment in mesh_report["assignments"])
    assert set(payload["array_reports"]) == {
        "cell_blocks",
        "cell_local_dofs",
        "owned_vector",
    }
    for report in payload["array_reports"].values():
        assert report["partition_count"] == 4
        assert report["global_device_count"] == 4
        assert report["process_count"] == 1
        assert len(report["addressable_shards"]) == 4
        assert report["replication_intent"] == "none; one leading FEM partition per device"
    assert payload["halo_link_count"] > 0
    assert payload["halo_value_count"] > 0
    assert 0.0 <= payload["cell_padding_fraction"] < 1.0
    assert 0.0 <= payload["owned_dof_padding_fraction"] < 1.0
    assert 0.0 <= payload["ghost_dof_padding_fraction"] < 1.0
    assert set(payload["action_relative_differences"]) == {"stiffness", "mass", "shifted"}
    assert payload["maximum_real_action_relative_difference"] < 3.0e-15
    assert payload["maximum_complex_action_relative_difference"] < 3.0e-15
    assert payload["matrix_vjp_relative_difference"] < 3.0e-15
    assert payload["vector_vjp_relative_difference"] < 3.0e-15
    assert payload["complex_matrix_vjp_relative_difference"] < 3.0e-15
    assert payload["complex_vector_vjp_relative_difference"] < 3.0e-15
    assert payload["packed_process_local_action_relative_difference"] < 3.0e-15
    assert (
        payload["stablehlo_collective_permute_count"]
        == payload["expected_collective_permute_count"]
    )
    assert not payload["stablehlo_contains_all_gather"]
    assert payload["claim_scope"] == (
        "forced multi-CPU portability; not accelerator or multi-host evidence"
    )
