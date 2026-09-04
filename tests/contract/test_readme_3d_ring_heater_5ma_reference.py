import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.contract

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_BUNDLE = REPO_ROOT / "docs" / "assets" / "readme" / "3d_ring_heater_reference"
BUNDLE_ROOT = REPO_ROOT / "docs" / "assets" / "readme" / "3d_ring_heater_5ma_reference"
EVIDENCE_PATH = BUNDLE_ROOT / "evidence.json"
GENERATOR_PATH = REPO_ROOT / "examples" / "readme_3d_ring_heater_reference.py"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _readme_paths() -> tuple[Path, ...]:
    paths = [REPO_ROOT / "README.md"]
    public_readme = REPO_ROOT / "README_PUBLIC.md"
    if public_readme.is_file():
        paths.append(public_readme)
    return tuple(paths)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(path: Path = EVIDENCE_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_row_count(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as stream:
        return sum(1 for _ in csv.reader(stream)) - 1


def test_direct_5ma_bundle_is_complete_content_addressed_and_same_mesh() -> None:
    evidence = _evidence()
    source = _evidence(SOURCE_BUNDLE / "evidence.json")
    assert evidence["schema_version"] == "femx.figure.3d-ring-heater-reference/v2"
    assert re.fullmatch(r"3d-ring-heater-run-[0-9a-f]{16}", evidence["run_id"])
    assert re.fullmatch(r"3d-ring-heater-figure-[0-9a-f]{16}", evidence["figure_id"])
    for name, expected_hash in evidence["artifacts"].items():
        assert SHA256_PATTERN.fullmatch(expected_hash)
        assert _sha256(BUNDLE_ROOT / name) == expected_hash
    femx = evidence["provenance"]["femx"]
    assert femx["uv_lock_artifact"] == "generation.uv.lock"
    assert evidence["artifacts"][femx["uv_lock_artifact"]] == femx["uv_lock_sha256"]
    assert _sha256(BUNDLE_ROOT / femx["uv_lock_artifact"]) == femx["uv_lock_sha256"]
    assert femx["generator_sha256"] == _sha256(GENERATOR_PATH)
    assert femx["presentation_generator_sha256"] == _sha256(GENERATOR_PATH)
    assert femx["field_generator_sha256"] == (
        "7b8bcd3bd78f33e003c6b474cb2bc9b2cc3602f2de3fc801cf43c28e46b8e040"
    )
    assert femx["field_generator_sha256"] != femx["presentation_generator_sha256"]
    assert evidence["identity"]["run"]["field_generator_sha256"] == femx["field_generator_sha256"]
    assert "presentation_generator_sha256" not in evidence["identity"]["run"]
    assert evidence["identity"]["figure"]["run_id"] == evidence["run_id"]
    assert evidence["identity"]["figure"]["rendering"] == evidence["figure"]["rendering_identity"]
    assert (
        evidence["provenance"]["gmsh"]["canonical_mesh_sha256"]
        == (source["provenance"]["gmsh"]["canonical_mesh_sha256"])
    )
    assert evidence["artifacts"]["cells.csv"] == source["artifacts"]["cells.csv"]
    assert _csv_row_count(BUNDLE_ROOT / "nodes.csv") == 12_761
    assert _csv_row_count(BUNDLE_ROOT / "cells.csv") == 71_808
    assert _csv_row_count(BUNDLE_ROOT / "potential.csv") == 2_484


def test_direct_5ma_run_has_independent_solver_evidence_and_passes_thresholds() -> None:
    evidence = _evidence()
    assert evidence["status"] == {
        "elmer_convergence": "converged",
        "elmer_process": "succeeded",
        "gmsh_process": "succeeded",
        "jax_target_solve": "numerically_admitted",
        "jax_unit_solve": "numerically_admitted",
        "scientific_parity": "passed",
    }
    metrics = evidence["metrics"]
    for metric_name, threshold in evidence["thresholds"].items():
        assert float(metrics[metric_name]) <= float(threshold)
    assert metrics["target_current_A"] == pytest.approx(0.005)
    assert metrics["inferred_current_A"] == pytest.approx(0.005)
    assert metrics["target_voltage_V"] == pytest.approx(0.22957269281112186)
    assert metrics["electrical_joule_power_W"] == pytest.approx(0.0011478634640556093)
    assert metrics["maximum_temperature_rise_K"] == pytest.approx(18.260841514923868)
    assert metrics["ring_volume_weighted_temperature_rise_K"] == pytest.approx(6.5738051785651805)
    assert metrics["heater_volume_weighted_temperature_rise_K"] == pytest.approx(16.95407905003583)
    assert metrics["maximum_absolute_temperature_difference_K"] == pytest.approx(
        2.1169114461372374e-08
    )
    assert metrics["relative_l2_temperature_rise_difference"] == pytest.approx(
        2.3072763065874994e-10
    )
    assert metrics["maximum_absolute_potential_difference_V"] == pytest.approx(
        7.718213568264076e-11
    )

    model = evidence["provenance"]["model"]
    assert model["operating_point"]["name"] == "low_temperature_projection"
    assert model["operating_point"]["evidence_tier"] == "derived_linear_projection"
    assert model["field_evidence"] == {
        "algebraic_field_rescaling_used": False,
        "claim_scope": (
            "direct same-discretization solver parity at the selected current; not a thermal-"
            "domain correction, material calibration, or fabricated-device claim"
        ),
        "elmer_target_solve": "retained external solve at the recorded target voltage",
        "evidence_tier": "direct_jax_elmer_same_mesh_solve",
        "jax_target_solve": "retained native solve at the recorded target voltage",
        "operating_point_selection_tier": "derived_linear_projection",
    }
    assert model["field_evaluation"].startswith("direct native JAX")
    assert "no temperature or parity field" in model["field_evaluation"]
    assert model["same_discretization_parity_claimed"] is True
    assert model["fabricated_device_prediction_claimed"] is False


def test_direct_5ma_fields_are_not_a_relabelled_15ma_bundle() -> None:
    source = _evidence(SOURCE_BUNDLE / "evidence.json")
    direct = _evidence()
    source_metrics = source["metrics"]
    direct_metrics = direct["metrics"]
    projection_scale = (0.005 / 0.015) ** 2

    assert direct["run_id"] != source["run_id"]
    assert direct["figure_id"] != source["figure_id"]
    assert direct["identity"]["figure"]["rendering"] == source["identity"]["figure"]["rendering"]
    assert direct["artifacts"]["nodes.csv"] != source["artifacts"]["nodes.csv"]
    assert direct["artifacts"]["potential.csv"] != source["artifacts"]["potential.csv"]
    assert direct_metrics["maximum_temperature_rise_K"] == pytest.approx(
        source_metrics["maximum_temperature_rise_K"] * projection_scale,
        abs=5.0e-11,
    )
    assert direct_metrics["maximum_absolute_temperature_difference_K"] != pytest.approx(
        source_metrics["maximum_absolute_temperature_difference_K"] * projection_scale,
        rel=1.0e-4,
    )
    assert direct_metrics["relative_l2_temperature_rise_difference"] != pytest.approx(
        source_metrics["relative_l2_temperature_rise_difference"],
        rel=1.0e-4,
        abs=0.0,
    )
    assert direct_metrics["jax_thermal_iterations"] == 545
    assert source_metrics["jax_thermal_iterations"] == 544

    figure = direct["figure"]
    assert figure["operating_point"]["name"] == "low_temperature_projection"
    assert figure["operating_point"]["algebraic_field_rescaling_used"] is False
    assert figure["temperature_colormap"] == "inferno"
    assert figure["temperature_panel_scope"] == ["b", "c"]
    assert figure["panel_a_scalar_overlay"] == "none; categorical material geometry only"
    assert figure["horizontal_temperature_contours"].startswith("none")
    assert figure["material_palette"] == source["figure"]["material_palette"]
    assert (
        figure["panel_a_dimension_annotation"] == source["figure"]["panel_a_dimension_annotation"]
    )
    svg = (BUNDLE_ROOT / "figure.svg").read_text(encoding="utf-8")
    assert "5 mA direct solves" in svg


def test_readmes_link_the_direct_5ma_evidence_without_upgrading_device_claim() -> None:
    for path in _readme_paths():
        text = path.read_text(encoding="utf-8")
        assert "docs/assets/readme/3d_ring_heater_5ma_reference/figure.png" in text
        assert re.search(r"no (?:displayed )?temperature or parity field", text, re.IGNORECASE)
        assert "fabricated-device prediction" in text
    bundle_readme = (BUNDLE_ROOT / "README.md").read_text(encoding="utf-8")
    assert _evidence()["run_id"] in bundle_readme
    assert _evidence()["figure_id"] in bundle_readme
    assert "Native JAX and locked external Elmer independently solve" in bundle_readme
    assert "No displayed temperature" in bundle_readme
    assert "--operating-point low_temperature_projection" in bundle_readme
