from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("jax")

pytestmark = [pytest.mark.scientific, pytest.mark.requires_jax, pytest.mark.slow]

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROBE = REPOSITORY_ROOT / "tests" / "public_ring_heater_forward_cpu_probe.py"
EXPECTED_MESH_SHA256 = {
    "coarse": "92189a8903aac73e8fd37f45387a449751441373b65f892ab1c4f01e0d7875a8",
    "medium": "8f392f32233a786311674f548cae341b066ffd6d9eb24885a955ac0d86baad14",
    "fine": "c484d4be5f52a59b93ba0904f79bef98d7dea0aceb8976e269b49cdc739d0a69",
}
EXPECTED_CELL_COUNTS = {
    "coarse": (71_808, 6_831),
    "medium": (435_574, 24_688),
    "fine": (3_179_879, 134_331),
}


def _mesh_path(profile: str) -> Path:
    variable = f"FEMX_PUBLIC_RING_HEATER_{profile.upper()}_MSH"
    raw = os.environ.get(variable)
    if raw is None:
        pytest.skip(f"set {variable} to an admitted public ring-heater MSH file")
    path = Path(raw).resolve()
    if not path.is_file():
        pytest.fail(f"{variable} does not identify a file: {path}")
    return path


def _run(profile: str) -> dict[str, object]:
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    python_path = (str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "src"))
    if environment.get("PYTHONPATH"):
        python_path = (*python_path, environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    completed = subprocess.run(
        (sys.executable, str(PROBE), profile, str(_mesh_path(profile))),
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=600.0 if profile == "fine" else 300.0,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _nested(mapping: dict[str, object], key: str) -> dict[str, object]:
    value = mapping[key]
    assert isinstance(value, dict)
    return value


def _relative_difference(first: float, second: float, *, reference: float = 0.0) -> float:
    return abs(first - second) / max(abs(first - reference), abs(second - reference))


def _refinement_observables(record: dict[str, object]) -> dict[str, tuple[float, float]]:
    numerics = _nested(record, "numerics")
    excitation = _nested(record, "excitation")
    regions = _nested(numerics, "region_temperature")
    return {
        "conductance": (float(excitation["conductance_S"]), 0.0),
        "peak_temperature_rise": (float(numerics["maximum_temperature_K"]), 300.0),
        "ring_mean_temperature_rise": (
            float(_nested(regions, "silicon_ring")["volume_weighted_cell_mean_K"]),
            300.0,
        ),
        "heater_mean_temperature_rise": (
            float(_nested(regions, "tin_heater")["volume_weighted_cell_mean_K"]),
            300.0,
        ),
    }


def test_three_level_public_ring_forward_is_admitted_and_changes_contract() -> None:
    records = {profile: _run(profile) for profile in ("coarse", "medium", "fine")}

    for profile, record in records.items():
        assert record["schema_version"] == "femx.public-ring-heater-forward.cpu-witness/v1"
        assert record["status"] == "passed"
        assert record["profile"] == profile
        assert "not Elmer parity" in str(record["claim_scope"])
        runtime = _nested(record, "runtime")
        assert runtime["backend"] == "cpu"
        assert runtime["device_count"] == runtime["partition_count"] == 1
        assert runtime["x64_enabled"] is True
        provenance = _nested(record, "provenance")
        assert provenance["source_msh_sha256"] == EXPECTED_MESH_SHA256[profile]
        assert all(len(str(value)) == 64 for value in provenance.values())
        mesh = _nested(record, "mesh")
        assert (
            int(mesh["tetrahedron_count"]),
            int(mesh["conductor_tetrahedron_count"]),
        ) == EXPECTED_CELL_COUNTS[profile]
        numerics = _nested(record, "numerics")
        assert int(numerics["target_current_iterations"]) <= 10_000
        assert int(numerics["target_thermal_iterations"]) <= 10_000
        assert float(numerics["current_backward_error"]) < 1.0e-9
        assert float(numerics["thermal_backward_error"]) < 1.0e-9
        assert float(numerics["charge_balance_relative_error"]) < 1.0e-7
        assert float(numerics["electrical_energy_relative_error"]) < 1.0e-7
        assert float(numerics["joule_transfer_relative_error"]) < 1.0e-12
        assert float(numerics["thermal_balance_relative_error"]) < 1.0e-7
        assert float(numerics["target_current_relative_error"]) < 1.0e-10
        assert float(numerics["target_power_relative_error"]) < 1.0e-10
        assert float(numerics["minimum_temperature_K"]) >= 300.0 - 1.0e-7
        assert float(numerics["maximum_temperature_K"]) > 300.0
        assert len(str(numerics["potential_sha256_float64"])) == 64
        assert len(str(numerics["temperature_sha256_float64"])) == 64

    observables = {profile: _refinement_observables(record) for profile, record in records.items()}
    for name in observables["coarse"]:
        coarse, reference = observables["coarse"][name]
        medium, medium_reference = observables["medium"][name]
        fine, fine_reference = observables["fine"][name]
        assert medium_reference == fine_reference == reference
        limit = 5.0e-3 if name == "conductance" else 3.0e-2
        coarse_to_medium = _relative_difference(coarse, medium, reference=reference)
        medium_to_fine = _relative_difference(medium, fine, reference=reference)
        assert coarse_to_medium < limit
        assert medium_to_fine < limit

        first_increment = medium - coarse
        second_increment = fine - medium
        assert first_increment * second_increment > 0.0
        assert abs(second_increment) < abs(first_increment)
