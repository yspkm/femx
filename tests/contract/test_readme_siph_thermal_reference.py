import csv
import hashlib
import json
import re
import struct
import subprocess
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.contract

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_ROOT = REPO_ROOT / "docs" / "assets" / "readme" / "siph_thermal_reference"
EVIDENCE_PATH = BUNDLE_ROOT / "evidence.json"
GENERATOR_PATH = REPO_ROOT / "examples" / "readme_siph_thermal_reference.py"
README_PATH = REPO_ROOT / "README.md"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence() -> dict[str, Any]:
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


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


def test_bundle_identity_and_open_data_are_self_consistent() -> None:
    evidence = _evidence()
    assert evidence["schema_version"] == "femx.figure.siph-thermal-reference/v1"
    assert re.fullmatch(r"siph-thermal-reference-[0-9a-f]{16}", evidence["figure_id"])

    for name, expected_hash in evidence["artifacts"].items():
        assert SHA256_PATTERN.fullmatch(expected_hash)
        assert _sha256(BUNDLE_ROOT / name) == expected_hash

    metrics = evidence["metrics"]
    assert _csv_row_count(BUNDLE_ROOT / "nodes.csv") == int(metrics["node_count"])
    assert _csv_row_count(BUNDLE_ROOT / "cells.csv") == int(metrics["triangle_count"])

    identity = {
        "schema_version": evidence["schema_version"],
        "nodes_sha256": evidence["artifacts"]["nodes.csv"],
        "cells_sha256": evidence["artifacts"]["cells.csv"],
        "canonical_mesh_sha256": evidence["provenance"]["gmsh"]["canonical_mesh_sha256"],
        "generator_sha256": evidence["provenance"]["femx"]["generator_sha256"],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    expected_id = "siph-thermal-reference-" + hashlib.sha256(encoded).hexdigest()[:16]
    assert evidence["figure_id"] == expected_id
    assert identity["generator_sha256"] == _sha256(GENERATOR_PATH)


def test_bundle_records_process_convergence_and_scientific_parity_separately() -> None:
    evidence = _evidence()
    assert evidence["status"] == {
        "elmer_convergence": "converged",
        "gmsh_process": "succeeded",
        "jax_convergence": "converged",
        "scientific_parity": "passed",
    }

    metrics = evidence["metrics"]
    for metric_name, threshold in evidence["thresholds"].items():
        assert metrics[metric_name] <= threshold

    assert metrics["node_count"] == 3240
    assert metrics["triangle_count"] == 6369
    assert metrics["maximum_temperature_rise_K"] == pytest.approx(164.35810376058305)
    assert metrics["maximum_absolute_temperature_difference_K"] < 1.0e-9
    assert metrics["adjoint_vs_elmer_fd_relative_error"] < 1.0e-10

    provenance = evidence["provenance"]
    assert provenance["model"]["dimension"] == 2
    assert provenance["model"]["full_3d_claimed"] is False
    assert provenance["model"]["measured_device_prediction_claimed"] is False
    assert provenance["model"]["out_of_plane_convention"] == "per_unit_depth"
    assert provenance["jax"]["backend"] == "cpu"
    assert provenance["jax"]["float64"] is True
    assert provenance["elmer"]["source_commit"] == ("4f2d7e4b99f8f0dcf2f7ac579e056969373bf594")
    assert provenance["femx"]["uv_lock_source"] == "HEAD:uv.lock"
    assert SHA256_PATTERN.fullmatch(provenance["femx"]["uv_lock_sha256"])

    if (REPO_ROOT / ".git").exists():
        committed_lock = subprocess.run(
            ("git", "show", f"{provenance['femx']['commit']}:uv.lock"),
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
            shell=False,
        ).stdout
        assert hashlib.sha256(committed_lock).hexdigest() == provenance["femx"]["uv_lock_sha256"]


def test_figure_is_public_safe_and_readme_visible() -> None:
    evidence = _evidence()
    png_path = BUNDLE_ROOT / "figure.png"
    svg_path = BUNDLE_ROOT / "figure.svg"
    width, height, dpi = _png_metadata(png_path)
    assert (width, height) == (2160, 1425)
    assert dpi == pytest.approx(300.0, abs=0.1)

    svg = svg_path.read_text(encoding="utf-8")
    assert "<svg" in svg
    assert evidence["figure_id"] in svg
    assert "heat-flow direction" not in svg

    generator = GENERATOR_PATH.read_text(encoding="utf-8")
    assert ".quiver(" not in generator
    assert "heat_flux_arrows" not in evidence["figure"]
    assert evidence["figure"]["heat_flux_overlay"].startswith("not displayed")

    readme = README_PATH.read_text(encoding="utf-8")
    assert "docs/assets/readme/siph_thermal_reference/README.md" in readme
    assert "2D adjoint reference" in readme

    public_text = "\n".join(
        [
            readme,
            (BUNDLE_ROOT / "README.md").read_text(encoding="utf-8"),
            EVIDENCE_PATH.read_text(encoding="utf-8"),
            svg,
            generator,
        ]
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
