from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_ROOT = REPOSITORY_ROOT / "docs/assets/readme/ring_heater_thermal_sensitivity"
EVIDENCE_PATH = BUNDLE_ROOT / "evidence.json"
GENERATOR_PATH = REPOSITORY_ROOT / "examples/readme_ring_heater_thermal_sensitivity.py"

pytestmark = pytest.mark.contract


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence() -> dict[str, Any]:
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def test_thermal_sensitivity_bundle_is_content_addressed_and_reproducible() -> None:
    evidence = _evidence()
    assert evidence["schema_version"] == "femx.figure.ring-heater-thermal-sensitivity/v1"
    assert evidence["figure_id"].startswith("ring-heater-thermal-sensitivity-")
    assert evidence["generator_sha256"] == _sha256(GENERATOR_PATH)
    for name, expected in evidence["artifacts"].items():
        assert _sha256(BUNDLE_ROOT / name) == expected

    source = evidence["source_evidence"]
    assert source["schema_version"] == "femx.ring-heater-thermal-sensitivity-evidence/v1"
    assert source["status"] == "passed"
    assert source["runtime"] == {
        "backend": "cpu",
        "device_count": 1,
        "jax_version": "0.10.1",
        "jaxlib_version": "0.10.1",
        "numpy_version": "2.4.6",
        "python_version": "3.13.13",
        "x64_enabled": True,
    }
    for relative_path, expected in source["source_files"].items():
        assert _sha256(REPOSITORY_ROOT / relative_path) == expected


def test_thermal_sensitivity_factorial_and_sidewall_bound_are_retained() -> None:
    evidence = _evidence()
    records = {record["case"]["name"]: record for record in evidence["source_evidence"]["cases"]}
    assert len(records) == 10
    assert all(record["status"] == "passed" for record in records.values())
    assert all(
        record["excitation"]["operating_point"]["target_current_A"] == 0.005
        for record in records.values()
    )

    source = records["source_envelope"]["temperature"]
    deep_narrow = records["substrate_50um"]["temperature"]
    wide_deep = records["domain_80um_substrate_50um"]["temperature"]
    sidewall_reference = records["domain_40um_substrate_5um"]["temperature"]
    sidewall_bound = records["ideal_isothermal_sidewall_bound"]["temperature"]

    assert source["peak_K_per_mW"] == pytest.approx(15.908548435198913)
    assert source["ring_mean_K_per_mW"] == pytest.approx(5.726992263817455)
    assert deep_narrow["peak_K_per_mW"] / source["peak_K_per_mW"] - 1.0 == pytest.approx(
        0.0521875259205125
    )
    assert deep_narrow["ring_mean_K_per_mW"] / source["ring_mean_K_per_mW"] - 1.0 == pytest.approx(
        0.149027923511301
    )
    assert wide_deep["peak_K_per_mW"] / source["peak_K_per_mW"] - 1.0 == pytest.approx(
        -0.014143541060333553
    )
    assert wide_deep["ring_mean_K_per_mW"] / source["ring_mean_K_per_mW"] - 1.0 == pytest.approx(
        -0.03234748558580747
    )
    assert sidewall_bound["peak_K_per_mW"] / sidewall_reference[
        "peak_K_per_mW"
    ] - 1.0 == pytest.approx(-0.00013590116913353434)
    assert (
        records["ideal_isothermal_sidewall_bound"]["boundary"]["lateral"]["condition"]
        == "isothermal"
    )


def test_thermal_sensitivity_figure_and_open_summary_match_evidence() -> None:
    evidence = _evidence()
    figure = evidence["figure"]
    assert figure["physical_size_inches"] == [7.2, 3.0]
    assert figure["png_pixels"] == [2160, 900]
    assert figure["png_dpi"] == 300
    assert figure["heatmap_colormap"] == "inferno"
    assert figure["non_color_encoding"].startswith("every heatmap cell")
    assert figure["widths_um"] == [20.0, 40.0, 80.0]
    assert figure["substrate_depths_um"] == [0.5, 5.0, 50.0]
    assert figure["peak_K_per_mW"][0][0] == pytest.approx(15.908548435198913)
    assert figure["ring_mean_K_per_mW"][2][2] == pytest.approx(5.541738464113589)

    with (BUNDLE_ROOT / "summary.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    assert {row["case"] for row in rows} == {
        record["case"]["name"] for record in evidence["source_evidence"]["cases"]
    }

    svg = (BUNDLE_ROOT / "figure.svg").read_text(encoding="utf-8")
    readme = (BUNDLE_ROOT / "README.md").read_text(encoding="utf-8")
    assert evidence["figure_id"] in svg
    assert "Bounded ring-heater thermal-envelope sensitivity" in svg
    assert "not formal domain convergence" in readme
    assert "physical TPU evidence" in readme
