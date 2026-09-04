import json
import re
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.contract

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples"
NOTEBOOK_PATH = EXAMPLES_ROOT / "thermally_tuned_ring_fem.ipynb"
BUILDER_PATH = EXAMPLES_ROOT / "build_thermally_tuned_ring_fem.py"


def _notebook() -> dict[str, Any]:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _source_text(notebook: dict[str, Any]) -> str:
    return "\n".join("".join(cell.get("source", ())) for cell in notebook["cells"])


def _output_text(notebook: dict[str, Any]) -> str:
    text_parts: list[str] = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", ()):
            text = output.get("text", ())
            text_parts.append(text if isinstance(text, str) else "".join(text))
    return "\n".join(text_parts)


def _evidence(notebook: dict[str, Any]) -> dict[str, Any]:
    prefix = "FEMX_THERMALLY_TUNED_RING_EVIDENCE="
    evidence_lines = [
        line for line in _output_text(notebook).splitlines() if line.startswith(prefix)
    ]
    assert len(evidence_lines) == 1
    return json.loads(evidence_lines[0][len(prefix) :])


def test_notebook_metadata_preserves_published_design_contract() -> None:
    metadata = _notebook()["metadata"]["femx_example"]
    geometry = metadata["geometry_um"]

    assert metadata["schema_version"] == "femx.example.thermally_tuned_ring_fem/v1"
    assert geometry == {
        "box_thickness": 2.0,
        "bus_center_y": [-5.6, 5.6],
        "cladding_top_z": 2.8,
        "coupling_gap": 0.1,
        "heat_domain_xy": [20.0, 20.0],
        "heater_center_z": 2.29,
        "heater_height": 0.14,
        "heater_notch_center": [0.0, -5.0, 2.29],
        "heater_notch_size": [1.0, 3.0, 0.21],
        "heater_vertical_gap": 2.0,
        "heater_width": 2.0,
        "ring_radius": 5.0,
        "wafer_thickness": 0.5,
        "waveguide_height": 0.22,
        "waveguide_width": 0.5,
    }
    assert metadata["numerical_model"] == {
        "backend": "native JAX dense P1 triangle tutorial operator",
        "dimension": "2D",
        "elmer_parity_claimed": False,
        "full_3d_parity_claimed": False,
        "language": "English",
        "slice": "x=0 vertical y-z solid slice",
    }
    assert metadata["optical_handoff_target"] == {
        "domain_xyz_um": [14.0, 14.0, 3.0],
        "drop_monitor_center_um": [-6.8, -5.6, 0.11],
        "mode_source_center_um": [-6.8, 5.6, 0.11],
        "ring_mode_bend_radius_um": 5.0,
        "ring_mode_plane_center_um": [-5.0, 0.0, 0.11],
        "through_monitor_center_um": [6.8, 5.6, 0.11],
        "wavelength_range_um": [1.5, 1.6],
    }


def test_notebook_contains_executed_accelerator_evidence() -> None:
    notebook = _notebook()
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert code_cells
    assert all(cell.get("execution_count") is not None for cell in code_cells)
    assert not any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", ())
    )
    image_count = sum(
        "image/png" in output.get("data", {})
        for cell in code_cells
        for output in cell.get("outputs", ())
    )
    assert image_count >= 2

    evidence = _evidence(notebook)
    assert evidence["schema_version"] == ("femx.example.thermally_tuned_ring_fem.evidence/v1")
    assert evidence["backend"] in {"cpu", "gpu"}
    assert evidence["float64"] is True
    assert evidence["model_dimension"] == "2D"
    assert evidence["full_3d_parity_claimed"] is False
    assert evidence["elmer_parity_claimed"] is False
    assert evidence["model_slice"] == "x=0 vertical y-z solid slice"
    assert evidence["mesh_profile"] == "standard"
    assert evidence["current_mA"] == pytest.approx(15.0)
    assert evidence["node_count"] == 1881
    assert evidence["triangle_count"] == 3584

    checks = evidence["checks"]
    assert checks["relative_free_residual"] < 2.0e-9
    assert checks["relative_energy_error"] < 2.0e-9
    assert checks["bottom_constraint_error_K"] < 2.0e-10
    assert checks["minimum_temperature_K"] >= 300.0 - 2.0e-8
    assert checks["upper_ring_mean_delta_K"] > checks["lower_ring_mean_delta_K"] > 0.0
    assert checks["upper_bus_mean_delta_K"] > checks["lower_bus_mean_delta_K"] > 0.0

    adjoint = evidence["adjoint"]
    assert adjoint["relative_gradient_error"] < 2.0e-8
    assert adjoint["quadratic_identity_error"] < 2.0e-8


def test_notebook_is_english_public_safe_and_scientifically_scoped() -> None:
    notebook = _notebook()
    source = _source_text(notebook)
    output = _output_text(notebook)
    combined = source + "\n" + output

    assert "jax.value_and_grad" in source
    assert "implicit_linear_solve" in source
    assert "completed 3D Elmer parity" in source
    assert "FDTDX" in source
    assert "central finite difference" in source
    assert "$$" in source
    assert re.search(r"(?<!\$)\$[^$\n]+\$(?!\$)", source)
    assert not re.search(r"[가-힣]", combined)
    assert not any(token in source for token in (r"\[", r"\]", r"\(", r"\)"))

    forbidden_patterns = (
        r"[A-Za-z]:\\Users\\",
        r"/mnt/[A-Za-z]/",
        r"/home/[^/]+/",
        r"tidy3d\.simulation\.cloud",
        r"(?:task|folder|resource)_id",
        r"(?:asia|australia|europe|northamerica|southamerica|us)-[a-z]+[0-9]-[a-z]",
        r"(?:api|access|secret)[_-]?key\s*[=:]",
    )
    assert not any(
        re.search(pattern, combined, flags=re.IGNORECASE) for pattern in forbidden_patterns
    )


def test_public_examples_contain_one_notebook_and_its_builder() -> None:
    public_notebooks = sorted(
        path.relative_to(EXAMPLES_ROOT)
        for path in EXAMPLES_ROOT.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in path.parts
    )
    assert public_notebooks == [Path("thermally_tuned_ring_fem.ipynb")]
    builder = BUILDER_PATH.read_text(encoding="utf-8")
    assert 'OUTPUT = Path(__file__).with_name("thermally_tuned_ring_fem.ipynb")' in builder
    assert "femx.example.thermally_tuned_ring_fem/v1" in builder
