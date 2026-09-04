from dataclasses import replace

import pytest

from femx.core.errors import ContractError
from femx.meshing.gmsh import (
    PUBLIC_TIDY3D_NOTEBOOK_REPOSITORY,
    PUBLIC_TIDY3D_NOTEBOOK_REVISION,
    PUBLIC_TIDY3D_NOTEBOOK_SHA256,
    PUBLIC_TIDY3D_RING_PAGE,
    RING_HEATER_THERMAL_SENSITIVITY_GEOMETRY_SCHEMA,
    PublicRingHeater3D,
    RingHeaterMeshProfile,
    RingHeaterThermalSensitivity3D,
    ring_heater_mesh_profile,
)

pytestmark = pytest.mark.unit


def test_ring_heater_profiles_are_explicit_two_to_one_refinements() -> None:
    coarse = ring_heater_mesh_profile("coarse")
    medium = ring_heater_mesh_profile("medium")
    fine = ring_heater_mesh_profile("fine")

    assert (coarse.interface_size_m, medium.interface_size_m, fine.interface_size_m) == (
        0.28e-6,
        0.14e-6,
        0.07e-6,
    )
    assert (coarse.bulk_size_m, medium.bulk_size_m, fine.bulk_size_m) == (
        1.28e-6,
        0.64e-6,
        0.32e-6,
    )
    assert coarse.interface_size_m == 2.0 * medium.interface_size_m
    assert medium.interface_size_m == 2.0 * fine.interface_size_m
    assert coarse.bulk_size_m == 2.0 * medium.bulk_size_m
    assert medium.bulk_size_m == 2.0 * fine.bulk_size_m


@pytest.mark.parametrize(
    ("args", "message"),
    (
        (("", 1.0, 1.0), "name"),
        ((" coarse ", 1.0, 1.0), "name"),
        (("bad", 0.0, 1.0), "finite and positive"),
        (("bad", float("nan"), 1.0), "finite and positive"),
        (("bad", 2.0, 1.0), "must not exceed"),
    ),
)
def test_ring_heater_profile_constructor_rejects_invalid_contract(
    args: tuple[object, ...],
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        RingHeaterMeshProfile(*args)  # type: ignore[arg-type]


def test_ring_heater_profile_lookup_rejects_unknown_name() -> None:
    with pytest.raises(ContractError, match="coarse, medium, or fine"):
        ring_heater_mesh_profile("preview")


def test_public_ring_heater_matches_pinned_public_dimensions_and_separates_extension() -> None:
    recipe = PublicRingHeater3D(ring_heater_mesh_profile("coarse"))
    canonical = recipe.canonical_data()

    assert recipe.coordinate_scale_to_m == 1.0e-6
    assert recipe.substrate_bottom_z_m == pytest.approx(-2.5e-6, rel=0.0, abs=1.0e-21)
    assert recipe.substrate_top_z_m == -2.0e-6
    assert recipe.bus_center_y_m == pytest.approx(5.6e-6, rel=0.0, abs=1.0e-21)
    assert recipe.heater_bottom_z_m == pytest.approx(2.22e-6, rel=0.0, abs=1.0e-21)
    assert recipe.heater_top_z_m == pytest.approx(2.36e-6, rel=0.0, abs=1.0e-21)
    assert canonical["public_source"] == {
        "kind": "independent reconstruction of published dimensions",
        "page": PUBLIC_TIDY3D_RING_PAGE,
        "repository": PUBLIC_TIDY3D_NOTEBOOK_REPOSITORY,
        "revision": PUBLIC_TIDY3D_NOTEBOOK_REVISION,
        "notebook_sha256": PUBLIC_TIDY3D_NOTEBOOK_SHA256,
    }
    assert canonical["femx_extension"] == {
        "kind": "two aluminum top-contact vias for a future current solve",
        "part_of_public_source": False,
    }
    assert canonical["gmsh_policy"] == {
        "factory": "OpenCASCADE",
        "format": "msh41-ascii",
        "element_order": 1,
        "algorithm_3d": 10,
        "algorithm_name": "HXT",
        "algorithm_fallback": False,
        "optimize": True,
        "netgen_optimizer": False,
        "random_factor": 1.0e-9,
        "random_seed": 1,
        "thread_count": 1,
    }


def test_public_ring_heater_expected_regions_partition_the_analytic_domain() -> None:
    recipe = PublicRingHeater3D(ring_heater_mesh_profile("coarse"))
    volumes = dict(recipe.expected_region_volumes_m3())
    expected_total = (
        recipe.domain_x_m
        * recipe.domain_y_m
        * (recipe.cladding_top_z_m - recipe.substrate_bottom_z_m)
    )

    assert tuple(volumes) == recipe.VOLUME_GROUPS
    assert sum(volumes.values()) == pytest.approx(expected_total, rel=1.0e-15)
    assert volumes["silicon_substrate"] == pytest.approx(200.0e-18, rel=1.0e-15)
    assert volumes["silicon_bus_upper"] == pytest.approx(2.2e-18, rel=1.0e-15)
    assert volumes["silicon_bus_lower"] == volumes["silicon_bus_upper"]
    assert volumes["al_contact_negative"] == pytest.approx(0.165e-18, rel=1.0e-15)
    assert volumes["al_contact_positive"] == volumes["al_contact_negative"]
    assert volumes["silicon_ring"] == pytest.approx(3.455751918948773e-18, rel=1.0e-15)
    assert volumes["tin_heater"] == pytest.approx(8.515970897003623e-18, rel=1.0e-15)


def test_public_ring_heater_geo_is_deterministic_exact_and_public_safe() -> None:
    recipe = PublicRingHeater3D(ring_heater_mesh_profile("coarse"))
    rendered = recipe.render_geo()

    assert rendered == PublicRingHeater3D(ring_heater_mesh_profile("coarse")).render_geo()
    assert len(recipe.digest()) == 64
    assert f"// recipe SHA-256 = {recipe.digest()}" in rendered
    assert PUBLIC_TIDY3D_RING_PAGE in rendered
    assert "Box(busUpper) = {-1e+1, 5.35, 0, 2e+1, 0.5, 0.22};" in rendered
    assert "5.600000000000001" not in rendered
    assert "Mesh.Algorithm3D = 10;" in rendered
    assert "Mesh.AlgorithmSwitchOnFailure = 0;" in rendered
    assert "Mesh.OptimizeNetgen = 0;" in rendered
    for tag, name in enumerate(recipe.VOLUME_GROUPS, start=101):
        assert f'Physical Volume("{name}", {tag})' in rendered
    for tag, name in enumerate(recipe.SURFACE_GROUPS, start=201):
        assert f'Physical Surface("{name}", {tag})' in rendered
    for private_fragment in ("/home/", "/mnt/", "C:\\", "tailscale", "TPU", "TRC"):
        assert private_fragment not in rendered


def test_thermal_sensitivity_recipe_separates_envelope_variants_from_source_geometry() -> None:
    source = PublicRingHeater3D(ring_heater_mesh_profile("coarse"))
    sensitivity = RingHeaterThermalSensitivity3D(
        RingHeaterMeshProfile("sensitivity", 0.28e-6, 5.0e-6),
        domain_x_m=80.0e-6,
        domain_y_m=80.0e-6,
        substrate_thickness_m=50.0e-6,
    )

    data = sensitivity.canonical_data()
    geometry = sensitivity.render_geo()
    assert data["schema_version"] == RING_HEATER_THERMAL_SENSITIVITY_GEOMETRY_SCHEMA
    assert data["study_scope"]["varied_geometry"] == [  # type: ignore[index]
        "domain_x_m",
        "domain_y_m",
        "substrate_thickness_m",
    ]
    assert "not source-reproduction parity" in data["study_scope"]["claim_scope"]  # type: ignore[index]
    assert sensitivity.SURFACE_GROUPS == (
        "external_boundary",
        "bottom_boundary",
        "top_boundary",
        "lateral_boundary",
        "terminal_negative",
        "terminal_positive",
    )
    assert 'Physical Surface("bottom_boundary", 202)' in geometry
    assert 'Physical Surface("top_boundary", 203)' in geometry
    assert 'Physical Surface("lateral_boundary", 204)' in geometry
    assert "bottom_temperature" not in geometry
    assert "top_convection" not in geometry
    assert "lateral_adiabatic" not in geometry
    assert source.digest() != sensitivity.digest()
    assert len(sensitivity.digest()) == 64


def test_public_ring_heater_digest_changes_with_geometry_and_mesh_policy() -> None:
    coarse = PublicRingHeater3D(ring_heater_mesh_profile("coarse"))
    medium = PublicRingHeater3D(ring_heater_mesh_profile("medium"))
    changed_geometry = replace(coarse, coupling_gap_m=0.11e-6)

    assert len({coarse.digest(), medium.digest(), changed_geometry.digest()}) == 3


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        ({"ring_radius_m": 0.0}, "finite and positive"),
        ({"ring_radius_m": float("nan")}, "finite and positive"),
        ({"waveguide_width_m": 10.0e-6}, "waveguide width"),
        ({"heater_width_m": 10.0e-6}, "heater width"),
        ({"heater_notch_x_m": 8.1e-6}, "narrower than the heater inner diameter"),
        ({"heater_notch_y_m": 1.0e-6}, "complete annulus intersection"),
        ({"heater_notch_height_m": 0.1e-6}, "complete heater thickness"),
        ({"contact_width_x_m": 0.6e-6}, "opposite sides"),
        ({"contact_length_y_m": 3.1e-6}, "notch span"),
        ({"domain_x_m": 12.0e-6}, "complete heater"),
        ({"domain_y_m": 11.7e-6, "heater_width_m": 0.2e-6}, "bus waveguides"),
        ({"cladding_top_z_m": 2.3e-6}, "upper-cladding"),
        ({"heater_width_m": 0.2e-6}, "contact footprint"),
        (
            {"mesh_profile": RingHeaterMeshProfile("oversized", 0.3e-6, 0.3e-6)},
            "twice the heater thickness",
        ),
    ),
)
def test_public_ring_heater_rejects_invalid_geometry(
    replacement: dict[str, object],
    message: str,
) -> None:
    valid = PublicRingHeater3D(ring_heater_mesh_profile("coarse"))
    with pytest.raises(ContractError, match=message):
        replace(valid, **replacement)
