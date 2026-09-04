from pathlib import Path

import pytest

from femx.core.execution import ExecutionPolicy
from femx.meshing.gmsh import (
    GmshMeshingRequest,
    PublicRingHeater3D,
    RingHeaterMeshProfile,
    RingHeaterThermalSensitivity3D,
    evaluate_public_ring_heater_mesh,
    read_gmsh_msh_3d,
    ring_heater_mesh_profile,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_gmsh]

_AUTHORIZED = ExecutionPolicy(execution_authorized=True, allow_external_process=True)


def _run_recipe(locked_gmsh_runner, root: Path, profile_name: str):
    recipe = PublicRingHeater3D(ring_heater_mesh_profile(profile_name))
    root.mkdir()
    geometry_path = root / "public_ring_heater.geo"
    geometry_path.write_text(recipe.render_geo(), encoding="utf-8", newline="\n")
    process = locked_gmsh_runner.run(
        GmshMeshingRequest(geometry_path.name, dimension=3, timeout_seconds=300.0),
        working_directory=root,
        policy=_AUTHORIZED,
    )
    assert process.process_succeeded, process.stderr
    imported = read_gmsh_msh_3d(
        root / "mesh.msh",
        coordinate_scale_to_m=recipe.coordinate_scale_to_m,
    )
    report = evaluate_public_ring_heater_mesh(imported, recipe)
    return process, imported.record.digest(), report


@pytest.mark.slow
def test_public_ring_heater_is_deterministic_conformal_and_refined(
    locked_gmsh_runner,
    tmp_path: Path,
) -> None:
    coarse_first = _run_recipe(locked_gmsh_runner, tmp_path / "coarse-first", "coarse")
    coarse_repeat = _run_recipe(locked_gmsh_runner, tmp_path / "coarse-repeat", "coarse")
    medium = _run_recipe(locked_gmsh_runner, tmp_path / "medium", "medium")
    fine = _run_recipe(locked_gmsh_runner, tmp_path / "fine", "fine")

    first_process, first_import_digest, first_report = coarse_first
    repeat_process, repeat_import_digest, repeat_report = coarse_repeat
    assert first_process.identity == repeat_process.identity
    assert first_process.geometry_sha256 == repeat_process.geometry_sha256
    assert first_process.mesh_sha256 == repeat_process.mesh_sha256
    assert first_import_digest == repeat_import_digest
    assert first_report.digest() == repeat_report.digest()

    reports = (first_report, medium[2], fine[2])
    assert [report.tetrahedron_count for report in reports] == sorted(
        report.tetrahedron_count for report in reports
    )
    assert len({report.tetrahedron_count for report in reports}) == len(reports)
    assert [report.maximum_edge_length_m for report in reports] == sorted(
        (report.maximum_edge_length_m for report in reports), reverse=True
    )
    assert [report.maximum_region_volume_relative_error for report in reports] == sorted(
        (report.maximum_region_volume_relative_error for report in reports), reverse=True
    )
    assert all(report.minimum_mean_ratio > 0.05 for report in reports)
    assert all(report.percentile_1_mean_ratio > 0.35 for report in reports)
    assert all(
        all(count > 0 for _, count in report.electrical_interface_triangle_counts)
        for report in reports
    )
    assert all(report.maximum_region_volume_relative_error < 1.0e-4 for report in reports)
    assert all(report.full_domain_relative_volume_error < 1.0e-12 for report in reports)


@pytest.mark.slow
def test_ring_heater_sensitivity_envelope_has_generic_admitted_boundaries(
    locked_gmsh_runner,
    tmp_path: Path,
) -> None:
    recipe = RingHeaterThermalSensitivity3D(
        RingHeaterMeshProfile("sensitivity-integration", 0.28e-6, 1.28e-6),
        domain_x_m=40.0e-6,
        domain_y_m=40.0e-6,
        substrate_thickness_m=5.0e-6,
    )
    geometry_path = tmp_path / "ring_heater_sensitivity.geo"
    geometry_path.write_text(recipe.render_geo(), encoding="utf-8", newline="\n")
    process = locked_gmsh_runner.run(
        GmshMeshingRequest(geometry_path.name, dimension=3, timeout_seconds=300.0),
        working_directory=tmp_path,
        policy=_AUTHORIZED,
    )
    assert process.process_succeeded, process.stderr
    imported = read_gmsh_msh_3d(
        tmp_path / "mesh.msh",
        coordinate_scale_to_m=recipe.coordinate_scale_to_m,
    )
    report = evaluate_public_ring_heater_mesh(imported, recipe)

    assert tuple(name for name, _count in report.surface_triangle_counts) == recipe.SURFACE_GROUPS
    assert all(count > 0 for _name, count in report.surface_triangle_counts)
    assert report.minimum_mean_ratio > 0.05
    assert report.maximum_region_volume_relative_error < 1.0e-4
    assert report.full_domain_relative_volume_error < 1.0e-12
