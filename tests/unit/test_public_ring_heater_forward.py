from __future__ import annotations

from dataclasses import replace

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

from femx.applications import (  # noqa: E402
    LinearCurrentCalibration,
    PublicRingHeaterMeshAdmissionPolicy,
    PublicRingHeaterOperatingPoint,
    PublicRingHeaterReferenceParameters,
    RingHeaterThermalBoundaryPolicy,
    RingHeaterThermalSensitivityCase,
    RingHeaterThermalSensitivityPlan,
    calibrate_public_ring_heater_current,
    prepare_public_ring_heater_forward_plan,
    prepare_ring_heater_thermal_sensitivity_plan,
    project_public_ring_heater_current,
    public_ring_heater_operating_point,
    ring_heater_thermal_sensitivity_cases,
)
from femx.applications import ring_heater as forward_module  # noqa: E402
from femx.backends.jax.elements.tetrahedron_h1 import (  # noqa: E402
    tetrahedron_p1_diffusion_cell_matrices,
)
from femx.core.errors import ContractError  # noqa: E402
from femx.mesh import CellType, EntityTag, Mesh, MeshGeometry, MeshTopology  # noqa: E402
from femx.meshing.gmsh import (  # noqa: E402
    GmshImportRecord,
    GmshPhysicalGroup,
    ImportedGmshMesh,
    PublicRingHeater3D,
    PublicRingHeaterMeshReport,
    RingHeaterThermalSensitivity3D,
    ring_heater_mesh_profile,
)
from femx.meshing.gmsh import importer as importer_module  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]

_LOCAL_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))


def _oriented_cell(coordinates: np.ndarray, raw: tuple[int, int, int, int]) -> tuple[int, ...]:
    cell = list(raw)
    points = coordinates[cell]
    jacobian = np.stack(
        (points[1] - points[0], points[2] - points[0], points[3] - points[0]),
        axis=1,
    )
    if np.linalg.det(jacobian) < 0.0:
        cell[1], cell[2] = cell[2], cell[1]
    return tuple(cell)


def _imported_mesh() -> ImportedGmshMesh:
    reference = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    coordinates = np.vstack(
        (
            *(reference + np.asarray((4.0 + 3.0 * cell, 0.0, 0.0)) for cell in range(5)),
            reference,
            np.asarray(((1.0, 1.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0))),
        )
    )
    raw_cells = (
        *(tuple(range(4 * cell, 4 * cell + 4)) for cell in range(5)),
        (20, 21, 22, 23),
        (21, 22, 23, 24),
        (20, 22, 23, 25),
        (20, 21, 23, 26),
    )
    cells = np.asarray(
        tuple(_oriented_cell(coordinates, raw) for raw in raw_cells),
        dtype=np.int64,
    )

    face_counts: dict[tuple[int, int, int], int] = {}
    faces_by_cell: list[tuple[tuple[int, int, int], ...]] = []
    for cell in cells:
        local = tuple(
            tuple(sorted(int(cell[index]) for index in local_face)) for local_face in _LOCAL_FACES
        )
        faces_by_cell.append(local)
        for face in local:
            face_counts[face] = face_counts.get(face, 0) + 1
    external_keys = tuple(sorted(face for face, count in face_counts.items() if count == 1))
    facets = np.asarray(external_keys, dtype=np.int64)
    by_key = {key: index for index, key in enumerate(external_keys)}

    negative_key = tuple(sorted((21, 22, 24)))
    positive_key = tuple(sorted((20, 23, 25)))
    terminal_ids = {by_key[negative_key], by_key[positive_key]}
    bottom_ids: list[int] = []
    for cell_id in (*range(5), 8):
        candidate = next(
            by_key[face]
            for face in faces_by_cell[cell_id]
            if face in by_key and by_key[face] not in terminal_ids
        )
        bottom_ids.append(candidate)
    remaining = sorted(set(range(len(facets))) - terminal_ids - set(bottom_ids))
    top_id = remaining[0]
    lateral_ids = tuple(remaining[1:])

    recipe = PublicRingHeater3D(ring_heater_mesh_profile("coarse"))
    volume_ids = {
        "silica": (0,),
        "silicon_substrate": (1,),
        "silicon_ring": (2,),
        "silicon_bus_upper": (3,),
        "silicon_bus_lower": (4,),
        "tin_heater": (5, 8),
        "al_contact_negative": (6,),
        "al_contact_positive": (7,),
    }
    tags = (
        *(EntityTag(name, 3, volume_ids[name]) for name in recipe.VOLUME_GROUPS),
        EntityTag("external_boundary", 2, tuple(range(len(facets)))),
        EntityTag("bottom_temperature", 2, tuple(sorted(bottom_ids))),
        EntityTag("top_convection", 2, (top_id,)),
        EntityTag("lateral_adiabatic", 2, lateral_ids),
        EntityTag("terminal_negative", 2, (by_key[negative_key],)),
        EntityTag("terminal_positive", 2, (by_key[positive_key],)),
    )
    mesh = Mesh(
        geometry=MeshGeometry(coordinates),
        topology=MeshTopology(cells, CellType.TETRAHEDRON, len(coordinates)),
        tags=tags,
        boundary_facets=MeshTopology(facets, CellType.TRIANGLE, len(coordinates)),
    )
    physical_groups = tuple(
        GmshPhysicalGroup(tag.dimension, index + 1, tag.name) for index, tag in enumerate(tags)
    )
    record = GmshImportRecord(
        source_sha256="a" * 64,
        canonical_mesh_sha256=importer_module._canonical_mesh_sha256(mesh),
        format_version="4.1",
        coordinate_scale_to_m=1.0,
        physical_groups=physical_groups,
        node_tags=tuple(range(1, len(coordinates) + 1)),
        cell_element_tags=tuple(range(1, len(cells) + 1)),
        boundary_element_tags=tuple(range(1, len(facets) + 1)),
        cell_local_node_permutations=((0, 1, 2, 3),) * len(cells),
        topological_dimension=3,
        boundary_local_node_permutations=((0, 1, 2),) * len(facets),
        schema_version="femx.gmsh-import/v2",
    )
    return ImportedGmshMesh(mesh, record)


def _sensitivity_imported_mesh() -> ImportedGmshMesh:
    source = _imported_mesh()
    renames = {
        "bottom_temperature": "bottom_boundary",
        "top_convection": "top_boundary",
        "lateral_adiabatic": "lateral_boundary",
    }
    tags = tuple(
        EntityTag(renames.get(tag.name, tag.name), tag.dimension, tag.entity_ids)
        for tag in source.mesh.tags
    )
    mesh = Mesh(
        geometry=source.mesh.geometry,
        topology=source.mesh.topology,
        tags=tags,
        boundary_facets=source.mesh.boundary_facets,
    )
    record = GmshImportRecord(
        source_sha256="b" * 64,
        canonical_mesh_sha256=importer_module._canonical_mesh_sha256(mesh),
        format_version=source.record.format_version,
        coordinate_scale_to_m=source.record.coordinate_scale_to_m,
        physical_groups=tuple(
            GmshPhysicalGroup(group.dimension, group.tag, renames.get(group.name, group.name))
            for group in source.record.physical_groups
        ),
        node_tags=source.record.node_tags,
        cell_element_tags=source.record.cell_element_tags,
        boundary_element_tags=source.record.boundary_element_tags,
        cell_local_node_permutations=source.record.cell_local_node_permutations,
        topological_dimension=source.record.topological_dimension,
        boundary_local_node_permutations=source.record.boundary_local_node_permutations,
        schema_version=source.record.schema_version,
    )
    return ImportedGmshMesh(mesh, record)


def _report(imported: ImportedGmshMesh, recipe: PublicRingHeater3D):
    counts = tuple((name, len(imported.mesh.tag(name).entity_ids)) for name in recipe.VOLUME_GROUPS)
    surfaces = tuple(
        (name, len(imported.mesh.tag(name).entity_ids)) for name in recipe.SURFACE_GROUPS
    )
    volumes = recipe.expected_region_volumes_m3()
    return PublicRingHeaterMeshReport(
        recipe_sha256=recipe.digest(),
        import_record_sha256=imported.record.digest(),
        node_count=imported.mesh.geometry.node_count,
        tetrahedron_count=imported.mesh.topology.cell_count,
        boundary_triangle_count=imported.mesh.boundary_facets.cell_count,  # type: ignore[union-attr]
        total_volume_m3=sum(value for _, value in volumes),
        full_domain_relative_volume_error=0.0,
        minimum_cell_volume_m3=1.0e-22,
        maximum_cell_volume_m3=1.0e-18,
        minimum_edge_length_m=1.0e-7,
        maximum_edge_length_m=1.0e-6,
        minimum_mean_ratio=0.2,
        percentile_1_mean_ratio=0.3,
        median_mean_ratio=0.7,
        maximum_region_volume_relative_error=1.0e-6,
        region_cell_counts=counts,
        region_volumes_m3=volumes,
        region_volume_relative_errors=tuple((name, 0.0) for name, _ in volumes),
        electrical_interface_triangle_counts=(
            ("al_contact_negative:tin_heater", 1),
            ("al_contact_positive:tin_heater", 1),
        ),
        surface_triangle_counts=surfaces,
    )


def _prepare(monkeypatch):
    imported = _imported_mesh()
    recipe = PublicRingHeater3D(ring_heater_mesh_profile("coarse"))
    report = _report(imported, recipe)
    monkeypatch.setattr(forward_module, "evaluate_public_ring_heater_mesh", lambda *_: report)
    plan = prepare_public_ring_heater_forward_plan(
        imported,
        recipe,
        np.zeros((imported.mesh.topology.cell_count,), dtype=np.int64),
        partition_count=1,
    )
    return imported, recipe, report, plan


def _prepare_sensitivity(monkeypatch, boundary: RingHeaterThermalBoundaryPolicy):
    imported = _sensitivity_imported_mesh()
    recipe = RingHeaterThermalSensitivity3D(ring_heater_mesh_profile("coarse"))
    report = _report(imported, recipe)
    monkeypatch.setattr(forward_module, "evaluate_public_ring_heater_mesh", lambda *_: report)
    plan = prepare_ring_heater_thermal_sensitivity_plan(
        imported,
        recipe,
        np.zeros((imported.mesh.topology.cell_count,), dtype=np.int64),
        partition_count=1,
        boundary=boundary,
    )
    return imported, recipe, report, plan


def test_reference_parameters_are_source_pinned_and_content_addressed() -> None:
    reference = PublicRingHeaterReferenceParameters()
    data = reference.canonical_data()

    assert data["evidence_tier"] == "public_benchmark_uncalibrated"
    assert data["values_si"]["target_current_A"] == 0.015  # type: ignore[index]
    assert data["sources"]["public_tidy3d_tutorial"]["revision"] == (  # type: ignore[index]
        "c37c785d52e9258c9d048a781524b8e8d7c758ca"
    )
    assert data["sources"]["elmer_gui_aluminum_compatibility"]["status"] == (  # type: ignore[index]
        "legacy_unverified"
    )
    assert len(reference.digest()) == 64
    assert reference.digest() == PublicRingHeaterReferenceParameters().digest()


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        ({"ambient_temperature_k": True}, "real scalar"),
        ({"target_current_a": float("nan")}, "finite and positive"),
        ({"tin_electrical_conductivity_s_per_m": 1.0}, "source-pinned"),
        ({"schema_version": "wrong"}, "reference schema"),
    ),
)
def test_reference_parameters_reject_source_drift(
    replacement: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        PublicRingHeaterReferenceParameters(**replacement)  # type: ignore[arg-type]


def test_public_operating_points_separate_source_reproduction_from_projection() -> None:
    source = public_ring_heater_operating_point("source_reproduction")
    low_temperature = public_ring_heater_operating_point("low_temperature_projection")

    assert isinstance(source, PublicRingHeaterOperatingPoint)
    assert source.target_current_a == 0.015
    assert source.evidence_tier == "source_pinned_parity"
    assert source.canonical_data()["joule_power_and_temperature_rise_scale"] == 1.0
    assert low_temperature.target_current_a == 0.005
    assert low_temperature.evidence_tier == "derived_linear_projection"
    assert low_temperature.canonical_data()["current_and_voltage_scale"] == pytest.approx(1.0 / 3.0)
    assert low_temperature.canonical_data()["joule_power_and_temperature_rise_scale"] == (
        pytest.approx(1.0 / 9.0)
    )
    assert "not an independent solve" in low_temperature.canonical_data()["claim_scope"]
    assert source.digest() != low_temperature.digest()

    calibration = project_public_ring_heater_current(
        0.003,
        operating_point=low_temperature,
    )
    assert calibration.target_current_a == 0.005
    assert calibration.target_voltage_v == pytest.approx(5.0 / 3.0)
    assert calibration.predicted_joule_power_w == pytest.approx(1.0 / 120.0)


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        ({"name": 1}, "non-empty and trimmed"),
        ({"name": ""}, "non-empty and trimmed"),
        ({"name": " low_temperature_projection"}, "non-empty and trimmed"),
        ({"name": "unknown"}, "must be source_reproduction"),
        ({"target_current_a": True}, "real scalar"),
        ({"target_current_a": 0.006}, "pins target current"),
        ({"evidence_tier": "wrong"}, "requires evidence tier"),
        ({"schema_version": "wrong"}, "operating-point schema"),
    ),
)
def test_public_operating_points_reject_role_drift(
    replacement: dict[str, object],
    message: str,
) -> None:
    valid = public_ring_heater_operating_point("low_temperature_projection")
    with pytest.raises(ContractError, match=message):
        replace(valid, **replacement)

    with pytest.raises(ContractError, match="must be a string"):
        public_ring_heater_operating_point(1)  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="must be source_reproduction"):
        public_ring_heater_operating_point("unknown")
    with pytest.raises(ContractError, match="requires an operating point"):
        project_public_ring_heater_current(0.003, operating_point=object())  # type: ignore[arg-type]


def test_thermal_boundary_policy_is_explicit_content_addressed_and_fail_closed() -> None:
    source_envelope = RingHeaterThermalBoundaryPolicy()
    robin_envelope = RingHeaterThermalBoundaryPolicy(
        top_transfer_w_per_m2_k=20.0,
        bottom_condition="robin",
        bottom_transfer_w_per_m2_k=80_000.0,
        lateral_condition="robin",
        lateral_transfer_w_per_m2_k=10.0,
    )

    assert source_envelope.canonical_data()["bottom"] == {
        "condition": "isothermal",
        "transfer_W_per_m2_K": 0.0,
    }
    assert source_envelope.canonical_data()["lateral"] == {
        "condition": "adiabatic",
        "transfer_W_per_m2_K": 0.0,
    }
    assert robin_envelope.canonical_data()["top"]["transfer_W_per_m2_K"] == 20.0  # type: ignore[index]
    assert "fixture-specific" in robin_envelope.canonical_data()["claim_scope"]
    assert len(source_envelope.digest()) == 64
    assert source_envelope.digest() != robin_envelope.digest()

    invalid = (
        ({"ambient_temperature_k": True}, "real scalar"),
        ({"top_transfer_w_per_m2_k": 0.0}, "finite and positive"),
        ({"bottom_condition": "fixed"}, "adiabatic, isothermal, or robin"),
        ({"bottom_condition": 1}, "adiabatic, isothermal, or robin"),
        ({"bottom_transfer_w_per_m2_k": -1.0}, "finite and nonnegative"),
        ({"bottom_condition": "robin"}, "requires positive transfer"),
        ({"bottom_transfer_w_per_m2_k": 1.0}, "must be zero"),
        (
            {"lateral_condition": "robin", "lateral_transfer_w_per_m2_k": 0.0},
            "requires positive transfer",
        ),
        ({"schema_version": "wrong"}, "thermal-boundary schema"),
    )
    for replacement, message in invalid:
        with pytest.raises(ContractError, match=message):
            replace(source_envelope, **replacement)


def test_initial_thermal_sensitivity_cases_form_a_bounded_factorial_and_sidewall_bound() -> None:
    cases = ring_heater_thermal_sensitivity_cases()
    by_name = {case.name: case for case in cases}

    assert tuple(by_name) == (
        "source_envelope",
        "substrate_5um",
        "substrate_50um",
        "domain_40um",
        "domain_80um",
        "domain_40um_substrate_5um",
        "domain_40um_substrate_50um",
        "domain_80um_substrate_5um",
        "domain_80um_substrate_50um",
        "ideal_isothermal_sidewall_bound",
    )
    assert all(isinstance(case, RingHeaterThermalSensitivityCase) for case in cases)
    assert len({case.digest() for case in cases}) == len(cases)
    baseline = by_name["source_envelope"]
    assert baseline.varied_axis == "baseline"
    assert baseline.reference_case is None
    assert baseline.recipe.domain_x_m == 20.0e-6
    assert baseline.recipe.substrate_thickness_m == 0.5e-6
    assert by_name["substrate_5um"].recipe.domain_x_m == baseline.recipe.domain_x_m
    assert by_name["substrate_5um"].recipe.substrate_thickness_m == 5.0e-6
    assert by_name["substrate_50um"].recipe.substrate_thickness_m == 50.0e-6
    assert by_name["domain_40um"].recipe.domain_x_m == 40.0e-6
    assert by_name["domain_40um"].recipe.substrate_thickness_m == (
        baseline.recipe.substrate_thickness_m
    )
    assert by_name["domain_80um"].recipe.domain_x_m == 80.0e-6
    assert by_name["domain_40um_substrate_50um"].recipe.domain_x_m == 40.0e-6
    assert by_name["domain_80um_substrate_5um"].recipe.substrate_thickness_m == 5.0e-6
    assert by_name["domain_80um_substrate_50um"].recipe.substrate_thickness_m == 50.0e-6
    assert by_name["ideal_isothermal_sidewall_bound"].boundary.lateral_condition == ("isothermal")
    assert "not a calibrated" in baseline.canonical_data()["claim_scope"]

    invalid = (
        ({"name": ""}, "non-empty and trimmed"),
        ({"varied_axis": "unknown"}, "axis must be"),
        ({"recipe": object()}, "geometry recipe"),
        ({"boundary": object()}, "boundary policy"),
        ({"reference_case": "source_envelope"}, "baseline cannot reference"),
        (
            {"varied_axis": "substrate_depth", "reference_case": None},
            "must reference source_envelope",
        ),
        ({"schema_version": "wrong"}, "sensitivity-case schema"),
    )
    for replacement, message in invalid:
        with pytest.raises(ContractError, match=message):
            replace(baseline, **replacement)


def test_thermal_sensitivity_plan_separates_source_and_robin_boundary_roles(monkeypatch) -> None:
    source_boundary = RingHeaterThermalBoundaryPolicy()
    _imported, recipe, report, source_plan = _prepare_sensitivity(
        monkeypatch,
        source_boundary,
    )
    assert isinstance(source_plan, RingHeaterThermalSensitivityPlan)
    assert source_plan.mesh_report == report
    assert source_plan.tet4.thermal_layout.topology.constrained_nodes.size > 0
    assert np.any(source_plan.tet4.thermal_robin_matrix != 0.0)
    assert source_plan.canonical_data()["recipe_sha256"] == recipe.digest()
    assert source_plan.canonical_data()["boundary_sha256"] == source_boundary.digest()
    assert "not source-reproduction parity" in source_plan.canonical_data()["claim_scope"]

    robin_boundary = RingHeaterThermalBoundaryPolicy(
        bottom_condition="robin",
        bottom_transfer_w_per_m2_k=80_000.0,
        lateral_condition="robin",
        lateral_transfer_w_per_m2_k=10.0,
    )
    _other_imported, _other_recipe, _other_report, robin_plan = _prepare_sensitivity(
        monkeypatch,
        robin_boundary,
    )
    assert robin_plan.tet4.thermal_layout.topology.constrained_nodes.size == 0
    assert robin_plan.tet4.thermal_layout.topology.free_dof_count == (
        robin_plan.tet4.thermal_layout.topology.node_count
    )
    assert robin_plan.tet4.thermal_dirichlet_shifted.size == 0
    assert source_plan.digest() != robin_plan.digest()


def test_thermal_sensitivity_preparation_rejects_mixed_identities(monkeypatch) -> None:
    imported = _sensitivity_imported_mesh()
    recipe = RingHeaterThermalSensitivity3D(ring_heater_mesh_profile("coarse"))
    report = _report(imported, recipe)
    monkeypatch.setattr(forward_module, "evaluate_public_ring_heater_mesh", lambda *_: report)
    owners = np.zeros((imported.mesh.topology.cell_count,), dtype=np.int64)
    boundary = RingHeaterThermalBoundaryPolicy()

    with pytest.raises(ContractError, match="imported Gmsh mesh"):
        prepare_ring_heater_thermal_sensitivity_plan(  # type: ignore[arg-type]
            object(), recipe, owners, partition_count=1, boundary=boundary
        )
    with pytest.raises(ContractError, match="geometry recipe"):
        prepare_ring_heater_thermal_sensitivity_plan(  # type: ignore[arg-type]
            imported,
            PublicRingHeater3D(ring_heater_mesh_profile("coarse")),
            owners,
            partition_count=1,
            boundary=boundary,
        )
    with pytest.raises(ContractError, match="boundary policy"):
        prepare_ring_heater_thermal_sensitivity_plan(  # type: ignore[arg-type]
            imported, recipe, owners, partition_count=1, boundary=object()
        )
    with pytest.raises(ContractError, match="reference materials"):
        prepare_ring_heater_thermal_sensitivity_plan(  # type: ignore[arg-type]
            imported,
            recipe,
            owners,
            partition_count=1,
            boundary=boundary,
            reference=object(),
        )
    with pytest.raises(ContractError, match="ambient must match"):
        prepare_ring_heater_thermal_sensitivity_plan(
            imported,
            recipe,
            owners,
            partition_count=1,
            boundary=replace(boundary, ambient_temperature_k=301.0),
        )
    with pytest.raises(ContractError, match="mesh admission"):
        prepare_ring_heater_thermal_sensitivity_plan(  # type: ignore[arg-type]
            imported,
            recipe,
            owners,
            partition_count=1,
            boundary=boundary,
            mesh_admission=object(),
        )


def test_mesh_admission_records_thresholds_and_rejects_failed_metrics() -> None:
    imported = _imported_mesh()
    recipe = PublicRingHeater3D(ring_heater_mesh_profile("coarse"))
    report = _report(imported, recipe)
    policy = PublicRingHeaterMeshAdmissionPolicy()

    policy.require(report)
    assert policy.canonical_data()["minimum_mean_ratio"] == 0.05
    assert len(policy.digest()) == 64
    for changed, message in (
        (replace(report, minimum_mean_ratio=0.01), "minimum mean ratio"),
        (
            replace(report, maximum_region_volume_relative_error=2.0e-4),
            "maximum region volume error",
        ),
        (
            replace(report, full_domain_relative_volume_error=2.0e-10),
            "full-domain volume error",
        ),
    ):
        with pytest.raises(ContractError, match=message):
            policy.require(changed)
    with pytest.raises(ContractError, match="mesh report"):
        policy.require(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        ({"minimum_mean_ratio": True}, "real scalar"),
        ({"minimum_mean_ratio": 0.0}, "finite and positive"),
        ({"minimum_mean_ratio": 1.1}, "cannot exceed one"),
        ({"maximum_region_volume_relative_error": 1.0}, "must be below one"),
        ({"maximum_full_domain_relative_volume_error": 1.0}, "must be below one"),
    ),
)
def test_mesh_admission_rejects_ambiguous_thresholds(
    replacement: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        PublicRingHeaterMeshAdmissionPolicy(**replacement)  # type: ignore[arg-type]


def test_linear_current_calibration_preserves_current_voltage_power_identity() -> None:
    reference = PublicRingHeaterReferenceParameters()
    calibration = calibrate_public_ring_heater_current(0.003, reference=reference)

    assert calibration.conductance_s == pytest.approx(0.003)
    assert calibration.target_voltage_v == pytest.approx(5.0)
    assert calibration.predicted_joule_power_w == pytest.approx(0.075)
    assert calibration.canonical_data()["unit_voltage_V"] == 1.0
    assert len(calibration.digest()) == 64


def test_linear_current_calibration_rejects_invalid_input_and_inconsistent_records() -> None:
    reference = PublicRingHeaterReferenceParameters()
    with pytest.raises(ContractError, match="reference parameters"):
        calibrate_public_ring_heater_current(1.0, reference=object())  # type: ignore[arg-type]
    for value, message in ((True, "real scalar"), (0.0, "finite and positive")):
        with pytest.raises(ContractError, match=message):
            calibrate_public_ring_heater_current(value, reference=reference)

    valid = calibrate_public_ring_heater_current(0.003, reference=reference)
    for changed, message in (
        ({"schema_version": "wrong"}, "calibration schema"),
        ({"unit_voltage_joule_power_w": -1.0}, "finite and positive"),
        ({"conductance_s": 0.004}, "must equal"),
        ({"target_voltage_v": 6.0}, "target voltage"),
        ({"predicted_joule_power_w": 0.08}, "predicted Joule power"),
    ):
        with pytest.raises(ContractError, match=message):
            replace(valid, **changed)


def test_public_ring_plan_binds_regions_boundaries_and_exact_parent_transfer(monkeypatch) -> None:
    imported, recipe, report, plan = _prepare(monkeypatch)

    assert plan.mesh_report == report
    assert plan.current_region_cell_counts == (
        ("tin_heater", 2),
        ("al_contact_negative", 1),
        ("al_contact_positive", 1),
    )
    assert plan.terminal_node_counts == (("terminal_negative", 3), ("terminal_positive", 3))
    assert plan.tet4.current_layout.topology.cell_count == 4
    assert plan.tet4.thermal_layout.topology.cell_count == 9
    np.testing.assert_array_equal(plan.tet4.current_parent_cell_ids, (5, 6, 7, 8))
    np.testing.assert_allclose(
        plan.tet4.current_conductivity,
        (2.3e6, 37.73e6, 37.73e6, 2.3e6),
    )

    conductivity = np.asarray((1.38, 148.0, 148.0, 148.0, 148.0, 28.0, 237.0, 237.0, 28.0))
    expected_stiffness = tetrahedron_p1_diffusion_cell_matrices(
        np.asarray(imported.mesh.geometry.coordinates),
        np.asarray(imported.mesh.topology.connectivity),
        conductivity,
    )
    np.testing.assert_allclose(plan.tet4.thermal_conduction_stiffness, expected_stiffness)
    assert np.any(plan.tet4.thermal_robin_matrix != 0.0)
    assert np.all(plan.tet4.thermal_dirichlet_shifted == 0.0)

    constrained_current = plan.tet4.current_layout.topology.constrained_nodes
    constrained_full = plan.tet4.current_parent_node_ids[constrained_current]
    by_full_node = dict(
        zip(
            constrained_full.tolist(),
            plan.tet4.current_dirichlet_scale.tolist(),
            strict=True,
        )
    )
    negative_nodes = np.unique(
        imported.mesh.boundary_facets.connectivity[  # type: ignore[union-attr]
            imported.mesh.tag("terminal_negative").entity_ids
        ]
    )
    positive_nodes = np.unique(
        imported.mesh.boundary_facets.connectivity[  # type: ignore[union-attr]
            imported.mesh.tag("terminal_positive").entity_ids
        ]
    )
    assert {by_full_node[int(node)] for node in negative_nodes} == {0.0}
    assert {by_full_node[int(node)] for node in positive_nodes} == {1.0}

    data = plan.canonical_data()
    assert data["recipe_sha256"] == recipe.digest()
    assert data["tet4_plan_sha256"] == plan.tet4.digest()
    assert "terminal tops" in plan.reference.canonical_data()["model_extensions"]["thermal"]  # type: ignore[index]
    assert len(plan.digest()) == 64
    assert plan.digest() == _prepare(monkeypatch)[3].digest()


def test_public_ring_preparation_rejects_wrong_envelopes(monkeypatch) -> None:
    imported = _imported_mesh()
    recipe = PublicRingHeater3D(ring_heater_mesh_profile("coarse"))
    report = _report(imported, recipe)
    monkeypatch.setattr(forward_module, "evaluate_public_ring_heater_mesh", lambda *_: report)
    owners = np.zeros((imported.mesh.topology.cell_count,), dtype=np.int64)

    with pytest.raises(ContractError, match="imported Gmsh mesh"):
        prepare_public_ring_heater_forward_plan(  # type: ignore[arg-type]
            object(), recipe, owners, partition_count=1
        )
    with pytest.raises(ContractError, match="geometry recipe"):
        prepare_public_ring_heater_forward_plan(  # type: ignore[arg-type]
            imported, object(), owners, partition_count=1
        )
    with pytest.raises(ContractError, match="reference parameters"):
        prepare_public_ring_heater_forward_plan(  # type: ignore[arg-type]
            imported, recipe, owners, partition_count=1, reference=object()
        )
    with pytest.raises(ContractError, match="mesh admission policy"):
        prepare_public_ring_heater_forward_plan(  # type: ignore[arg-type]
            imported, recipe, owners, partition_count=1, mesh_admission=object()
        )


def test_public_ring_preparation_rejects_tag_and_boundary_drift(monkeypatch) -> None:
    imported = _imported_mesh()
    recipe = PublicRingHeater3D(ring_heater_mesh_profile("coarse"))
    report = _report(imported, recipe)
    monkeypatch.setattr(forward_module, "evaluate_public_ring_heater_mesh", lambda *_: report)
    owners = np.zeros((imported.mesh.topology.cell_count,), dtype=np.int64)

    def changed_imported(*, tags=None, boundary_facets=...):
        mesh = replace(
            imported.mesh,
            tags=imported.mesh.tags if tags is None else tags,
            boundary_facets=(
                imported.mesh.boundary_facets if boundary_facets is ... else boundary_facets
            ),
        )
        record = replace(
            imported.record,
            canonical_mesh_sha256=importer_module._canonical_mesh_sha256(mesh),
        )
        return ImportedGmshMesh(mesh, record)

    missing = tuple(tag for tag in imported.mesh.tags if tag.name != "silica")
    with pytest.raises(ContractError, match="exactly one dimension-3 tag"):
        prepare_public_ring_heater_forward_plan(
            changed_imported(tags=missing), recipe, owners, partition_count=1
        )
    wrong_dimension = tuple(
        EntityTag("silica", 2, (0,)) if tag.name == "silica" else tag for tag in imported.mesh.tags
    )
    with pytest.raises(ContractError, match="exactly one dimension-3 tag"):
        prepare_public_ring_heater_forward_plan(
            changed_imported(tags=wrong_dimension), recipe, owners, partition_count=1
        )
    empty = tuple(
        EntityTag("silica", 3, ()) if tag.name == "silica" else tag for tag in imported.mesh.tags
    )
    with pytest.raises(ContractError, match="must be non-empty"):
        prepare_public_ring_heater_forward_plan(
            changed_imported(tags=empty), recipe, owners, partition_count=1
        )
    missing_boundary = _imported_mesh()
    object.__setattr__(missing_boundary.mesh, "boundary_facets", None)
    with pytest.raises(ContractError, match="requires boundary facets"):
        prepare_public_ring_heater_forward_plan(missing_boundary, recipe, owners, partition_count=1)

    negative = imported.mesh.tag("terminal_negative")
    positive = imported.mesh.tag("terminal_positive")
    overlap = tuple(
        replace(tag, entity_ids=negative.entity_ids) if tag.name == positive.name else tag
        for tag in imported.mesh.tags
    )
    with pytest.raises(ContractError, match="terminal nodes must be disjoint"):
        prepare_public_ring_heater_forward_plan(
            changed_imported(tags=overlap), recipe, owners, partition_count=1
        )

    incomplete = tuple(
        replace(tag, entity_ids=(5,)) if tag.name == "tin_heater" else tag
        for tag in imported.mesh.tags
    )
    with pytest.raises(ContractError, match="thermal cells are not completely materialized"):
        prepare_public_ring_heater_forward_plan(
            changed_imported(tags=incomplete), recipe, owners, partition_count=1
        )


def test_forward_plan_rejects_internal_identity_drift(monkeypatch) -> None:
    _imported, _recipe, _report_value, plan = _prepare(monkeypatch)
    for changed, message in (
        ({"reference": object()}, "reference parameters"),
        ({"mesh_admission": object()}, "mesh admission policy"),
        ({"tet4": object()}, "Tet4 numerical plan"),
        ({"schema_version": "wrong"}, "forward schema"),
        ({"current_region_cell_counts": (("wrong", 4),)}, "wrong names or order"),
        (
            {
                "current_region_cell_counts": (
                    ("tin_heater", 0),
                    ("al_contact_negative", 1),
                    ("al_contact_positive", 1),
                )
            },
            "must contain cells",
        ),
        (
            {
                "current_region_cell_counts": (
                    ("tin_heater", 1),
                    ("al_contact_negative", 1),
                    ("al_contact_positive", 1),
                )
            },
            "disagree with the plan",
        ),
        ({"mesh_report": replace(plan.mesh_report, tetrahedron_count=8)}, "thermal cell count"),
        ({"terminal_node_counts": (("wrong", 3),)}, "wrong names or order"),
        (
            {"terminal_node_counts": (("terminal_negative", 0), ("terminal_positive", 3))},
            "must contain nodes",
        ),
    ):
        with pytest.raises(ContractError, match=message):
            replace(plan, **changed)


def test_linear_current_record_rejects_non_scalar_value() -> None:
    with pytest.raises(ContractError, match="real scalar"):
        LinearCurrentCalibration(
            unit_voltage_joule_power_w=[],  # type: ignore[arg-type]
            conductance_s=1.0,
            target_current_a=1.0,
            target_voltage_v=1.0,
            predicted_joule_power_w=1.0,
        )
