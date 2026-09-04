import csv
import hashlib
import json
import math
import re
import struct
import subprocess
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.contract

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_ROOT = REPO_ROOT / "docs" / "assets" / "readme" / "3d_ring_heater_reference"
EVIDENCE_PATH = BUNDLE_ROOT / "evidence.json"
GENERATOR_PATH = REPO_ROOT / "examples" / "readme_3d_ring_heater_reference.py"
README_PATH = REPO_ROOT / "README.md"
PUBLIC_README_PATH = REPO_ROOT / "README_PUBLIC.md"
THERMAL_SCOPE_PATH = REPO_ROOT / "docs" / "physics" / "PUBLIC_RING_HEATER_THERMAL_SCOPE.md"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence() -> dict[str, Any]:
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def _readme_paths() -> tuple[Path, ...]:
    paths = [README_PATH]
    if PUBLIC_README_PATH.is_file():
        paths.append(PUBLIC_README_PATH)
    return tuple(paths)


def _provenance_lock_bytes(commit: str, artifact_name: str) -> bytes:
    assert artifact_name == "generation.uv.lock"
    retained = (BUNDLE_ROOT / artifact_name).read_bytes()
    if (REPO_ROOT / ".git").exists():
        result = subprocess.run(
            ("git", "show", f"{commit}:uv.lock"),
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
            shell=False,
        )
        if result.returncode == 0:
            assert result.stdout == retained
    return retained


def _csv_row_count(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as stream:
        return sum(1 for _ in csv.reader(stream)) - 1


def _png_metadata(path: Path) -> tuple[int, int, float | None]:
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", data[16:24])
    dpi: float | None = None
    offset = 8
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        if chunk_type == b"pHYs":
            x_per_metre, y_per_metre, unit = struct.unpack(">IIB", chunk_data)
            assert x_per_metre == y_per_metre
            if unit == 1:
                dpi = x_per_metre * 0.0254
        offset += 12 + length
    return width, height, dpi


def test_3d_bundle_identity_and_open_data_are_self_consistent() -> None:
    evidence = _evidence()
    assert evidence["schema_version"] == "femx.figure.3d-ring-heater-reference/v2"
    assert re.fullmatch(r"3d-ring-heater-run-[0-9a-f]{16}", evidence["run_id"])
    assert re.fullmatch(r"3d-ring-heater-figure-[0-9a-f]{16}", evidence["figure_id"])
    for name, expected_hash in evidence["artifacts"].items():
        assert SHA256_PATTERN.fullmatch(expected_hash)
        assert _sha256(BUNDLE_ROOT / name) == expected_hash

    metrics = evidence["metrics"]
    assert _csv_row_count(BUNDLE_ROOT / "nodes.csv") == metrics["node_count"]
    assert _csv_row_count(BUNDLE_ROOT / "cells.csv") == metrics["tetrahedron_count"]
    assert _csv_row_count(BUNDLE_ROOT / "potential.csv") == metrics["conductor_node_count"]
    run_identity = {
        "schema_version": "femx.figure.3d-ring-heater-run-identity/v1",
        "nodes_sha256": evidence["artifacts"]["nodes.csv"],
        "cells_sha256": evidence["artifacts"]["cells.csv"],
        "potential_sha256": evidence["artifacts"]["potential.csv"],
        "canonical_mesh_sha256": evidence["provenance"]["gmsh"]["canonical_mesh_sha256"],
        "field_generator_sha256": evidence["provenance"]["femx"]["field_generator_sha256"],
    }
    assert evidence["identity"]["run"] == run_identity
    encoded = json.dumps(run_identity, sort_keys=True, separators=(",", ":")).encode()
    assert evidence["run_id"] == "3d-ring-heater-run-" + hashlib.sha256(encoded).hexdigest()[:16]

    figure_identity = evidence["identity"]["figure"]
    assert figure_identity["schema_version"] == "femx.figure.3d-ring-heater-render-identity/v1"
    assert figure_identity["run_id"] == evidence["run_id"]
    assert figure_identity["rendering"] == evidence["figure"]["rendering_identity"]
    encoded = json.dumps(figure_identity, sort_keys=True, separators=(",", ":")).encode()
    assert evidence["figure_id"] == (
        "3d-ring-heater-figure-" + hashlib.sha256(encoded).hexdigest()[:16]
    )
    assert evidence["figure"]["run_id"] == evidence["run_id"]
    assert evidence["figure"]["figure_id"] == evidence["figure_id"]
    assert evidence["provenance"]["femx"]["presentation_generator_sha256"] == _sha256(
        GENERATOR_PATH
    )
    assert "presentation_generator_sha256" not in run_identity


def test_3d_tables_retain_complete_finite_fields_and_material_partition() -> None:
    evidence = _evidence()
    metrics = evidence["metrics"]
    node_count = int(metrics["node_count"])
    maximum_difference = 0.0
    maximum_temperature = -math.inf
    with (BUNDLE_ROOT / "nodes.csv").open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream)
        for expected_node_id, row in enumerate(rows):
            assert int(row["node_id"]) == expected_node_id
            values = tuple(
                float(row[name])
                for name in (
                    "x_um",
                    "y_um",
                    "z_um",
                    "jax_temperature_K",
                    "elmer_temperature_K",
                    "elmer_minus_jax_K",
                )
            )
            assert all(math.isfinite(value) for value in values)
            maximum_temperature = max(maximum_temperature, values[3])
            maximum_difference = max(maximum_difference, abs(values[5]))
    assert maximum_temperature == pytest.approx(metrics["maximum_temperature_K"])
    assert maximum_difference == pytest.approx(metrics["maximum_absolute_temperature_difference_K"])

    region_names: set[str] = set()
    with (BUNDLE_ROOT / "cells.csv").open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream)
        for expected_cell_id, row in enumerate(rows):
            assert int(row["cell_id"]) == expected_cell_id
            nodes = tuple(int(row[f"node_{index}"]) for index in range(4))
            assert len(set(nodes)) == 4
            assert all(0 <= node < node_count for node in nodes)
            region_names.add(row["region_name"])
    assert region_names == {
        "silica",
        "silicon_substrate",
        "silicon_ring",
        "silicon_bus_upper",
        "silicon_bus_lower",
        "tin_heater",
        "al_contact_negative",
        "al_contact_positive",
    }

    source_nodes: set[int] = set()
    with (BUNDLE_ROOT / "potential.csv").open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream)
        for expected_field_index, row in enumerate(rows):
            assert int(row["field_index"]) == expected_field_index
            source_node = int(row["source_node_id"])
            assert 0 <= source_node < node_count
            assert source_node not in source_nodes
            source_nodes.add(source_node)
            assert all(
                math.isfinite(float(row[name]))
                for name in (
                    "jax_potential_V",
                    "elmer_potential_V",
                    "elmer_minus_jax_V",
                )
            )
    assert len(source_nodes) == metrics["conductor_node_count"]


def test_3d_process_convergence_and_scientific_parity_are_separate() -> None:
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
    assert metrics["node_count"] == 12_761
    assert metrics["tetrahedron_count"] == 71_808
    assert metrics["conductor_node_count"] == 2_484
    assert metrics["conductor_tetrahedron_count"] == 6_831
    assert metrics["target_current_A"] == pytest.approx(0.015)
    assert metrics["target_voltage_V"] == pytest.approx(0.6887180784333656)
    assert metrics["maximum_temperature_rise_K"] == pytest.approx(164.3475736346429)
    assert metrics["maximum_absolute_temperature_difference_K"] < 2.0e-7
    assert metrics["relative_l2_temperature_rise_difference"] < 3.0e-10
    assert metrics["maximum_absolute_potential_difference_V"] < 3.0e-10

    provenance = evidence["provenance"]
    model = provenance["model"]
    assert model["dimension"] == 3
    assert model["profile"] == "coarse"
    assert model["same_discretization_parity_claimed"] is True
    assert model["mesh_convergence_claimed"] is False
    assert model["fabricated_device_prediction_claimed"] is False
    assert provenance["gmsh"]["source_msh_sha256"] == (
        "92189a8903aac73e8fd37f45387a449751441373b65f892ab1c4f01e0d7875a8"
    )
    assert provenance["jax"]["backend"] == "cpu"
    assert provenance["jax"]["device_count"] == 1
    assert provenance["jax"]["float64"] is True
    assert provenance["jax"]["version"] == "0.10.1"
    assert provenance["elmer"]["elmer_source_commit"] == (
        "4f2d7e4b99f8f0dcf2f7ac579e056969373bf594"
    )
    assert provenance["elmer"]["elmer_source_worktree_state"] == "clean"
    assert provenance["elmer"]["elmer_executable_sha256"] == (
        "a8aceea1fde474a427fe4cf3fd9bd32ca61e03d702ec98811f4b374947e80b84"
    )
    assert provenance["femx"]["uv_lock_source"] == "HEAD:uv.lock"
    assert SHA256_PATTERN.fullmatch(provenance["femx"]["uv_lock_sha256"])
    assert provenance["femx"]["uv_lock_artifact"] == "generation.uv.lock"
    committed_lock = _provenance_lock_bytes(
        str(provenance["femx"]["commit"]), provenance["femx"]["uv_lock_artifact"]
    )
    assert hashlib.sha256(committed_lock).hexdigest() == provenance["femx"]["uv_lock_sha256"]


def test_3d_operating_point_projection_keeps_device_claim_open() -> None:
    metrics = _evidence()["metrics"]
    power_mw = float(metrics["electrical_joule_power_W"]) * 1.0e3
    current_ratio = 0.005 / float(metrics["target_current_A"])
    power_ratio = current_ratio * current_ratio

    assert float(metrics["target_voltage_V"]) / float(metrics["target_current_A"]) == (
        pytest.approx(45.91453856222438)
    )
    assert float(metrics["maximum_temperature_rise_K"]) / power_mw == pytest.approx(
        15.908548435230667
    )
    assert float(metrics["ring_volume_weighted_temperature_rise_K"]) / power_mw == (
        pytest.approx(5.726992263817338)
    )
    assert float(metrics["heater_volume_weighted_temperature_rise_K"]) / power_mw == (
        pytest.approx(14.770118207381643)
    )
    assert float(metrics["target_voltage_V"]) * current_ratio == pytest.approx(0.2295726928111219)
    assert power_mw * power_ratio == pytest.approx(1.1478634640556098)
    assert float(metrics["maximum_temperature_rise_K"]) * power_ratio == pytest.approx(
        18.260841514960322
    )
    assert float(metrics["ring_volume_weighted_temperature_rise_K"]) * power_ratio == (
        pytest.approx(6.573805178565048)
    )
    assert float(metrics["heater_volume_weighted_temperature_rise_K"]) * power_ratio == (
        pytest.approx(16.954079050035926)
    )

    bottom_fraction = float(metrics["jax_bottom_outward_power_W"]) / float(
        metrics["electrical_joule_power_W"]
    )
    assert bottom_fraction == pytest.approx(0.9999695202817231)

    scope = THERMAL_SCOPE_PATH.read_text(encoding="utf-8")
    assert "0.5 um of silicon substrate is represented" in scope
    assert "zero normal heat flux on all four lateral sides" in scope
    assert "does not make the finite-element field mathematically one-dimensional" in scope
    assert "5 mA column is the exact algebraic prediction" in scope
    assert "JAX and locked external Elmer were then run again" in scope
    assert "device-representative only after" in scope
    for readme_path in _readme_paths():
        readme = readme_path.read_text(encoding="utf-8")
        assert "99.997 percent" in readme
        assert "5 mA" in readme
        assert "PUBLIC_RING_HEATER_THERMAL_SCOPE.md" in readme


def test_3d_figure_is_professional_public_safe_and_readme_visible() -> None:
    evidence = _evidence()
    figure = evidence["figure"]
    width, height, dpi = _png_metadata(BUNDLE_ROOT / "figure.png")
    assert (width, height) == (2160, 1440)
    assert dpi == pytest.approx(300.0, abs=0.1)
    assert figure["font"] == "7 pt sans serif"
    assert figure["line_width_pt"] == 0.75
    assert figure["temperature_colormap"] == "inferno"
    assert figure["temperature_palette_semantics"].startswith(
        "linear perceptually uniform sequential inferno"
    )
    assert figure["temperature_colormap_source"].startswith("Matplotlib built-in inferno")
    assert figure["temperature_panel_scope"] == ["b", "c"]
    assert figure["panel_a_scalar_overlay"] == "none; categorical material geometry only"
    assert figure["horizontal_temperature_contours"].startswith("none")
    assert figure["horizontal_device_boundary_overlay"].startswith(
        "unfilled Si ring and bus boundaries"
    )
    assert figure["rendering_identity"]["rendering_revision"] == "paper-figure-v5"
    assert figure["rendering_identity"]["panel_a"]["box_zoom"] == pytest.approx(0.98)
    assert figure["rendering_identity"]["panel_a"]["legend"] == {
        "anchor_figure_fraction": [0.125, 0.845],
        "columns": 2,
        "column_spacing": 0.85,
        "handle_length": 1.25,
        "handle_text_padding": 0.4,
        "layout": "left-shifted to preserve the panel-a-to-b label gutter",
    }
    assert figure["rendering_identity"]["panel_a"]["silicon_device_painter_order"] == [
        "silicon_bus_upper_far",
        "silicon_ring",
        "silicon_bus_lower_near",
    ]
    assert figure["rendering_identity"]["panel_a"]["projected_device_painter_order"] == [
        "silicon_bus_upper_far",
        "silicon_ring",
        "tin_heater",
        "silicon_bus_lower_near",
        "al_contacts",
    ]
    assert figure["rendering_identity"]["panel_a"]["ring_and_bus_alpha"] == 1.0
    assert figure["rendering_identity"]["panel_a"]["projected_device_occlusion"].startswith(
        "opaque far-bus, ring assembly, near-bus painter layers"
    )
    assert figure["rendering_identity"]["temperature"] == {
        "colormap": "inferno",
        "normalization": "linear complete retained range",
        "panel_scope": ["b", "c"],
        "panel_a_overlay": "none; categorical material geometry only",
        "horizontal_contours": (
            "none; scalar isolines are omitted so they cannot be confused with CAD boundaries"
        ),
    }
    assert figure["rendering_identity"]["parity_metrics_placement"] == (
        "compact two-line subtitle above the plotting area"
    )
    assert (
        figure["rendering_identity_sha256"]
        == hashlib.sha256(
            json.dumps(figure["rendering_identity"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    operating_point = figure["operating_point"]
    assert operating_point["name"] == "source_reproduction"
    assert operating_point["target_current_A"] == pytest.approx(0.015)
    assert operating_point["target_voltage_V"] == pytest.approx(0.6887180784333656)
    assert operating_point["field_evidence_tier"] == "direct_jax_elmer_same_mesh_solve"
    assert operating_point["operating_point_selection_tier"] == "source_pinned_parity"
    assert operating_point["algebraic_field_rescaling_used"] is False
    assert operating_point["field_evaluation"].startswith("direct native JAX")
    assert figure["difference_colormap"] == "RdBu_r"
    assert figure["difference_limits_K"][0] == -figure["difference_limits_K"][1]
    assert figure["difference_palette_semantics"].startswith(
        "zero-centered ColorBrewer RdBu diverging map"
    )
    assert figure["difference_colormap_source"].startswith("Matplotlib built-in RdBu_r")
    assert figure["clipping"].startswith("none")
    assert figure["material_palette"] == {
        "id": "femx-semantic-materials-v1.1",
        "profile": "light",
        "policy": (
            "v1.1 light-canvas fill and frame pairs; same material keeps one hue and display "
            "role uses geometry, outline, and opacity"
        ),
        "fill_colors": {
            "si_substrate": "#4B3F72",
            "si_device": "#685AB8",
            "sio2": "#167786",
            "tin": "#604900",
            "al": "#6F7885",
        },
        "frame_colors": {
            "si_substrate": "#2B2344",
            "si_device": "#352B76",
            "sio2": "#083F48",
            "tin": "#332600",
            "al": "#303741",
        },
        "region_mapping": {
            "silicon_substrate": "si_substrate",
            "silicon_ring_and_buses": "si_device",
            "silica_box_and_cladding": "sio2",
            "tin_heater": "tin",
            "al_contacts": "al",
        },
        "role_modulation": (
            "BOX is a solid fill; upper-cladding extent uses the same SiO2 base colour with "
            "reduced opacity"
        ),
    }
    assert figure["three_dimensional_z_display_exaggeration"] == 1.0
    assert figure["surface_visibility"].startswith(
        "one continuous opaque Si substrate solid includes the solved top 0.5 um"
    )
    assert figure["material_render_order"] == [
        "continuous_silicon_substrate_with_truncated_handle_context",
        "solved_substrate_depth_boundary",
        "silica_box",
        "silica_cladding_extent_backdrops",
        "silicon_bus_upper_far",
        "silicon_ring",
        "tin_heater",
        "silicon_bus_lower_near",
        "al_contacts",
    ]
    assert figure["depth_sorting"].startswith("manual painter order")
    assert figure["geometry_source"] == "femx.meshing.gmsh.PublicRingHeater3D"
    assert figure["geometry_recipe_sha256"] == evidence["provenance"]["gmsh"]["recipe_sha256"]
    assert figure["heater_vertical_gap_um"] == pytest.approx(2.0)
    assert figure["material_z_extents_um"] == {
        "domain": [-2.5, 2.8],
        "silicon_substrate": [-2.5, -2.0],
        "silica_box": [-2.0, 0.0],
        "silicon_device": [0.0, 0.22],
        "tin_heater": [2.22, 2.36],
        "al_contacts": [2.36, 2.8],
        "silica_cladding": [0.0, 2.8],
    }
    assert figure["heat_flux_overlay"].startswith("not displayed")
    assert figure["substrate_display_extension_below_model_um"] == pytest.approx(1.5)
    assert figure["substrate_display_z_extent_um"] == [-4.0, -2.0]
    assert figure["substrate_display_policy"].startswith(
        "one continuous Si solid; its top 0.5 um is solved"
    )
    assert figure["substrate_handle_context"] == {
        "material_continuity": "same Si fill and frame; no intervening layer",
        "modeled_top_thickness_um": 0.5,
        "solve_boundary_z_um": -2.5,
        "displayed_unsolved_depth_um": 1.5,
        "nominal_full_thickness_um": 725.0,
        "nominal_thickness_qualifier": (
            "approximate contextual value; not source-pinned solved geometry"
        ),
        "display_scale_below_solve_boundary": "depth truncated; not to scale",
    }
    assert figure["physical_scale_policy"].startswith(
        "panel-a solved geometry uses a one-to-one micrometre drawing scale"
    )
    assert figure["panel_a_projection"] == "orthographic axonometric"
    assert figure["panel_a_dimension_annotation"] == {
        "equal_physical_axis_scale": True,
        "lateral_extent_um": {
            "x": [-10.0, 10.0],
            "y": [-10.0, 10.0],
        },
        "lateral_boundary": "adiabatic sides",
        "z_thicknesses_um": {
            "silicon_substrate": 0.5,
            "silica_box": 2.0,
            "silicon_device": 0.22,
            "silicon_to_tin_oxide_gap": 2.0,
            "tin_heater": 0.14,
            "tin_to_model_top_silica": 0.44,
            "al_vias": 0.44,
        },
        "tin_to_top_silica_and_al_vias_share_z_span": True,
        "scale_scope": "solved geometry only",
        "substrate_handle_nominal_thickness_um": 725.0,
        "substrate_handle_depth_truncated": True,
    }
    assert figure["silica_cladding_backdrop"] == {
        "kind": "two far-side two-dimensional extent planes",
        "planes": ["x_min", "y_max"],
        "z_extent_m": [0.0, pytest.approx(2.8e-6)],
        "render_layer": "with silica BOX, before device solids",
        "volumetric_cladding_rendered": False,
        "physical_silica_volume_changed": False,
    }
    vertical_outline = figure["vertical_device_outline_overlay"]
    assert vertical_outline["panel"] == "c"
    assert vertical_outline["fixed_x_m"] == pytest.approx(figure["hot_vertical_slice_x_m"])
    assert vertical_outline["geometry_source"] == "femx.meshing.gmsh.PublicRingHeater3D"
    assert vertical_outline["visible_regions"] == [
        "silicon_substrate",
        "silicon_ring",
        "silicon_bus_lower",
        "silicon_bus_upper",
        "tin_heater",
    ]
    assert vertical_outline["style"] == (
        "unfilled white 0.50 pt outlines with a 1.18 pt dark contrast stroke; material identity "
        "additionally uses line style, never scalar colour"
    )

    svg = (BUNDLE_ROOT / "figure.svg").read_text(encoding="utf-8")
    assert "<svg" in svg
    assert evidence["run_id"] in svg
    assert evidence["figure_id"] in svg
    assert "15 mA direct solves" in svg
    assert "a  Modeled material stack" in svg
    assert "Si-device boundary overlaid" in svg
    assert "isosurface" not in svg
    assert "contours" not in svg
    assert "TiN-to-top SiO₂ 0.44" in svg
    assert "Si handle ≈725 µm; truncated (top 0.50 µm solved)" in svg
    generator = GENERATOR_PATH.read_text(encoding="utf-8")
    assert ".quiver(" not in generator
    assert 'TEMPERATURE_COLORMAP = "inferno"' in generator
    assert 'TEMPERATURE_COLORMAP = "turbo"' not in generator
    assert ".tricontour(" not in generator
    assert 'DIFFERENCE_COLORMAP = "RdBu_r"' in generator
    readme = README_PATH.read_text(encoding="utf-8")
    assert "docs/assets/readme/3d_ring_heater_5ma_reference/figure.png" in readme
    assert "15 mA source-reproduction" in readme
    assert "12,761" in readme
    assert "71,808" in readme
    assert "3D same-discretization parity" in readme

    public_text = "\n".join(
        (
            readme,
            (BUNDLE_ROOT / "README.md").read_text(encoding="utf-8"),
            EVIDENCE_PATH.read_text(encoding="utf-8"),
            svg,
            generator,
        )
    )
    forbidden_patterns = (
        r"[A-Za-z]:\\Users\\",
        r"/mnt/[A-Za-z]/",
        r"/home/[^/]+/",
        r"(?:api|access|secret)[_-]?key\s*[=:]",
        r"(?:asia|australia|europe|northamerica|southamerica|us)-[a-z]+[0-9]-[a-z]",
    )
    assert not any(
        re.search(pattern, public_text, flags=re.IGNORECASE) for pattern in forbidden_patterns
    )
