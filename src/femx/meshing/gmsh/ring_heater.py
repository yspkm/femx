"""Public three-dimensional ring-heater geometry for the M5 validation pipeline."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import ClassVar

from femx.core.errors import ContractError

PUBLIC_TIDY3D_RING_PAGE = (
    "https://www.flexcompute.com/tidy3d/examples/notebooks/ThermallyTunedRingResonator/"
)
PUBLIC_TIDY3D_NOTEBOOK_REPOSITORY = "https://github.com/flexcompute/tidy3d-notebooks"
PUBLIC_TIDY3D_NOTEBOOK_REVISION = "c37c785d52e9258c9d048a781524b8e8d7c758ca"
PUBLIC_TIDY3D_NOTEBOOK_SHA256 = "4ed3c6dbd8021d40ba03a5d44bdad5f684e60f3732118a0ba11bf8e4669550d6"
RING_HEATER_THERMAL_SENSITIVITY_GEOMETRY_SCHEMA = "femx.ring-heater-thermal-sensitivity-geometry/v1"


@dataclass(frozen=True, slots=True)
class RingHeaterMeshProfile:
    """Named first-order Tet4 size policy expressed in SI metres."""

    name: str
    interface_size_m: float
    bulk_size_m: float

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ContractError("ring-heater mesh profile name must be non-empty and trimmed")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in (self.interface_size_m, self.bulk_size_m)
        ):
            raise ContractError("ring-heater mesh sizes must be finite and positive")
        if self.interface_size_m > self.bulk_size_m:
            raise ContractError("ring-heater interface size must not exceed bulk size")


def ring_heater_mesh_profile(name: str) -> RingHeaterMeshProfile:
    """Return one explicit two-to-one M5 geometry-refinement level."""

    profiles = {
        "coarse": RingHeaterMeshProfile("coarse", 0.28e-6, 1.28e-6),
        "medium": RingHeaterMeshProfile("medium", 0.14e-6, 0.64e-6),
        "fine": RingHeaterMeshProfile("fine", 0.07e-6, 0.32e-6),
    }
    try:
        return profiles[name]
    except KeyError as error:
        raise ContractError("ring-heater mesh profile must be coarse, medium, or fine") from error


@dataclass(frozen=True, slots=True)
class PublicRingHeater3D:
    """Reproducible public SOI ring/heater solid with explicit electrical access."""

    mesh_profile: RingHeaterMeshProfile
    model_unit_m: float = 1.0e-6
    domain_x_m: float = 20.0e-6
    domain_y_m: float = 20.0e-6
    substrate_thickness_m: float = 0.5e-6
    buried_oxide_thickness_m: float = 2.0e-6
    cladding_top_z_m: float = 2.8e-6
    ring_radius_m: float = 5.0e-6
    waveguide_width_m: float = 0.5e-6
    waveguide_height_m: float = 0.22e-6
    coupling_gap_m: float = 0.1e-6
    heater_width_m: float = 2.0e-6
    heater_height_m: float = 0.14e-6
    heater_vertical_gap_m: float = 2.0e-6
    heater_notch_x_m: float = 1.0e-6
    heater_notch_y_m: float = 3.0e-6
    heater_notch_height_m: float = 0.21e-6
    contact_width_x_m: float = 0.25e-6
    contact_length_y_m: float = 1.5e-6

    VOLUME_GROUPS: ClassVar[tuple[str, ...]] = (
        "silica",
        "silicon_substrate",
        "silicon_ring",
        "silicon_bus_upper",
        "silicon_bus_lower",
        "tin_heater",
        "al_contact_negative",
        "al_contact_positive",
    )
    SURFACE_GROUPS: ClassVar[tuple[str, ...]] = (
        "external_boundary",
        "bottom_temperature",
        "top_convection",
        "lateral_adiabatic",
        "terminal_negative",
        "terminal_positive",
    )

    def __post_init__(self) -> None:
        dimensional_values = tuple(
            value
            for name, value in asdict(self).items()
            if name != "mesh_profile" and isinstance(value, float)
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in dimensional_values):
            raise ContractError("ring-heater dimensions must be finite and positive")
        if self.waveguide_width_m >= 2.0 * self.ring_radius_m:
            raise ContractError("ring waveguide width must be smaller than its diameter")
        if self.heater_width_m >= 2.0 * self.ring_radius_m:
            raise ContractError("heater width must leave a positive inner radius")
        heater_inner_radius = self.ring_radius_m - self.heater_width_m / 2.0
        notch_half_x = self.heater_notch_x_m / 2.0
        if notch_half_x >= heater_inner_radius:
            raise ContractError("heater notch must remain narrower than the heater inner diameter")
        required_notch_half_y = max(
            self.heater_width_m / 2.0,
            self.ring_radius_m
            - math.sqrt(heater_inner_radius * heater_inner_radius - notch_half_x * notch_half_x),
        )
        if self.heater_notch_y_m / 2.0 < required_notch_half_y:
            raise ContractError("heater notch must span the complete annulus intersection")
        if self.heater_notch_height_m < self.heater_height_m:
            raise ContractError("heater notch must span the complete heater thickness")
        if self.contact_width_x_m * 2.0 > self.heater_notch_x_m:
            raise ContractError("contact vias must remain on opposite sides of the heater notch")
        if self.contact_length_y_m > self.heater_notch_y_m:
            raise ContractError("contact length must fit inside the heater notch span")

        half_x = self.domain_x_m / 2.0
        half_y = self.domain_y_m / 2.0
        bus_center = self.bus_center_y_m
        if self.ring_radius_m + self.heater_width_m / 2.0 >= min(half_x, half_y):
            raise ContractError("thermal domain must contain the complete heater annulus")
        if bus_center + self.waveguide_width_m / 2.0 >= half_y:
            raise ContractError("thermal domain must contain both bus waveguides")
        if self.heater_top_z_m >= self.cladding_top_z_m:
            raise ContractError("heater and contact access require positive upper-cladding height")
        if self.mesh_profile.interface_size_m > self.heater_height_m * 2.0:
            raise ContractError("interface mesh size may not exceed twice the heater thickness")
        contact_inner_x = self.heater_notch_x_m / 2.0
        contact_outer_x = contact_inner_x + self.contact_width_x_m
        contact_near_y = self.ring_radius_m - self.contact_length_y_m / 2.0
        contact_far_y = self.ring_radius_m + self.contact_length_y_m / 2.0
        if math.hypot(contact_inner_x, contact_near_y) <= (
            self.ring_radius_m - self.heater_width_m / 2.0
        ) or math.hypot(contact_outer_x, contact_far_y) >= (
            self.ring_radius_m + self.heater_width_m / 2.0
        ):
            raise ContractError("contact footprint must lie strictly inside the heater annulus")

    @property
    def coordinate_scale_to_m(self) -> float:
        """Scale required when importing the rendered Gmsh model."""

        return self.model_unit_m

    @property
    def substrate_bottom_z_m(self) -> float:
        return -(self.buried_oxide_thickness_m + self.substrate_thickness_m)

    @property
    def substrate_top_z_m(self) -> float:
        return -self.buried_oxide_thickness_m

    @property
    def bus_center_y_m(self) -> float:
        return self.ring_radius_m + self.waveguide_width_m + self.coupling_gap_m

    @property
    def heater_bottom_z_m(self) -> float:
        return self.waveguide_height_m + self.heater_vertical_gap_m

    @property
    def heater_top_z_m(self) -> float:
        return self.heater_bottom_z_m + self.heater_height_m

    def expected_region_volumes_m3(self) -> tuple[tuple[str, float], ...]:
        """Return analytic solid volumes for geometry-convergence checks."""

        ring_outer = self.ring_radius_m + self.waveguide_width_m / 2.0
        ring_inner = self.ring_radius_m - self.waveguide_width_m / 2.0
        heater_outer = self.ring_radius_m + self.heater_width_m / 2.0
        heater_inner = self.ring_radius_m - self.heater_width_m / 2.0
        notch_half_x = self.heater_notch_x_m / 2.0

        def circle_integral(radius: float, x: float) -> float:
            return 0.5 * (
                x * math.sqrt(radius * radius - x * x) + radius * radius * math.asin(x / radius)
            )

        notch_area = 2.0 * (
            circle_integral(heater_outer, notch_half_x)
            - circle_integral(heater_inner, notch_half_x)
        )
        heater_area = math.pi * (heater_outer**2 - heater_inner**2) - notch_area
        contact_height = self.cladding_top_z_m - self.heater_top_z_m
        volumes = {
            "silicon_substrate": (self.domain_x_m * self.domain_y_m * self.substrate_thickness_m),
            "silicon_ring": (math.pi * (ring_outer**2 - ring_inner**2) * self.waveguide_height_m),
            "silicon_bus_upper": (
                self.domain_x_m * self.waveguide_width_m * self.waveguide_height_m
            ),
            "silicon_bus_lower": (
                self.domain_x_m * self.waveguide_width_m * self.waveguide_height_m
            ),
            "tin_heater": heater_area * self.heater_height_m,
            "al_contact_negative": (
                self.contact_width_x_m * self.contact_length_y_m * contact_height
            ),
            "al_contact_positive": (
                self.contact_width_x_m * self.contact_length_y_m * contact_height
            ),
        }
        full_domain = (
            self.domain_x_m * self.domain_y_m * (self.cladding_top_z_m - self.substrate_bottom_z_m)
        )
        volumes["silica"] = full_domain - sum(volumes.values())
        return tuple((name, volumes[name]) for name in self.VOLUME_GROUPS)

    def canonical_data(self) -> dict[str, object]:
        """Return the complete public-source, extension, geometry, and mesh contract."""

        return {
            "schema_version": "femx.public-ring-heater-geometry/v1",
            "public_source": {
                "kind": "independent reconstruction of published dimensions",
                "page": PUBLIC_TIDY3D_RING_PAGE,
                "repository": PUBLIC_TIDY3D_NOTEBOOK_REPOSITORY,
                "revision": PUBLIC_TIDY3D_NOTEBOOK_REVISION,
                "notebook_sha256": PUBLIC_TIDY3D_NOTEBOOK_SHA256,
            },
            "femx_extension": {
                "kind": "two aluminum top-contact vias for a future current solve",
                "part_of_public_source": False,
            },
            "geometry_m": {
                key: value for key, value in asdict(self).items() if key != "mesh_profile"
            },
            "mesh_profile": asdict(self.mesh_profile),
            "gmsh_policy": {
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
            },
            "volume_groups": list(self.VOLUME_GROUPS),
            "surface_groups": list(self.SURFACE_GROUPS),
        }

    def digest(self) -> str:
        """Hash the recipe without depending on a Gmsh installation."""

        payload = json.dumps(
            self.canonical_data(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def render_geo(self) -> str:
        """Render a deterministic OpenCASCADE solid and first-order mesh policy."""

        unit = Decimal(str(self.model_unit_m))

        def model(value: float) -> str:
            return _format_real(Decimal(str(value)) / unit)

        def model_decimal(value: Decimal) -> str:
            return _format_real(value / unit)

        decimal = {
            field: Decimal(str(value))
            for field, value in asdict(self).items()
            if field != "mesh_profile"
        }
        half_x = decimal["domain_x_m"] / 2
        half_y = decimal["domain_y_m"] / 2
        ring_outer = decimal["ring_radius_m"] + decimal["waveguide_width_m"] / 2
        ring_inner = decimal["ring_radius_m"] - decimal["waveguide_width_m"] / 2
        heater_outer = decimal["ring_radius_m"] + decimal["heater_width_m"] / 2
        heater_inner = decimal["ring_radius_m"] - decimal["heater_width_m"] / 2
        bus_center = (
            decimal["ring_radius_m"] + decimal["waveguide_width_m"] + decimal["coupling_gap_m"]
        )
        heater_bottom = decimal["waveguide_height_m"] + decimal["heater_vertical_gap_m"]
        heater_top = heater_bottom + decimal["heater_height_m"]
        contact_inner_x = decimal["heater_notch_x_m"] / 2
        contact_y_min = -decimal["ring_radius_m"] - decimal["contact_length_y_m"] / 2
        contact_y_max = contact_y_min + decimal["contact_length_y_m"]
        contact_height = decimal["cladding_top_z_m"] - heater_top
        selection_epsilon = unit * Decimal("1e-7")

        values = {
            "unit": _format_real(unit),
            "digest": self.digest(),
            "xmin": model_decimal(-half_x),
            "ymin": model_decimal(-half_y),
            "xlength": model(self.domain_x_m),
            "ylength": model(self.domain_y_m),
            "zmin": model_decimal(
                -(decimal["buried_oxide_thickness_m"] + decimal["substrate_thickness_m"])
            ),
            "zlength": model_decimal(
                decimal["cladding_top_z_m"]
                + decimal["buried_oxide_thickness_m"]
                + decimal["substrate_thickness_m"]
            ),
            "substrate_height": model(self.substrate_thickness_m),
            "ring_outer": model_decimal(ring_outer),
            "ring_inner": model_decimal(ring_inner),
            "waveguide_height": model(self.waveguide_height_m),
            "bus_center": model_decimal(bus_center),
            "bus_ymin_upper": model_decimal(bus_center - decimal["waveguide_width_m"] / 2),
            "bus_ymin_lower": model_decimal(-bus_center - decimal["waveguide_width_m"] / 2),
            "bus_width": model(self.waveguide_width_m),
            "heater_z": model_decimal(heater_bottom),
            "heater_outer": model_decimal(heater_outer),
            "heater_inner": model_decimal(heater_inner),
            "heater_height": model(self.heater_height_m),
            "notch_xmin": model(-self.heater_notch_x_m / 2.0),
            "notch_ymin": model_decimal(
                -decimal["ring_radius_m"] - decimal["heater_notch_y_m"] / 2
            ),
            "notch_zmin": model_decimal(
                heater_bottom - (decimal["heater_notch_height_m"] - decimal["heater_height_m"]) / 2
            ),
            "notch_x": model(self.heater_notch_x_m),
            "notch_y": model(self.heater_notch_y_m),
            "notch_height": model(self.heater_notch_height_m),
            "contact_negative_x": model_decimal(-contact_inner_x - decimal["contact_width_x_m"]),
            "contact_negative_xmax": model_decimal(-contact_inner_x),
            "contact_positive_x": model_decimal(contact_inner_x),
            "contact_positive_xmax": model_decimal(contact_inner_x + decimal["contact_width_x_m"]),
            "contact_ymin": model_decimal(contact_y_min),
            "contact_ymax": model_decimal(contact_y_max),
            "contact_width": model(self.contact_width_x_m),
            "contact_length": model(self.contact_length_y_m),
            "contact_z": model_decimal(heater_top),
            "contact_height": model_decimal(contact_height),
            "top": model(self.cladding_top_z_m),
            "half_x": model_decimal(half_x),
            "half_y": model_decimal(half_y),
            "epsilon": model_decimal(selection_epsilon),
            "lc_interface": model(self.mesh_profile.interface_size_m),
            "lc_bulk": model(self.mesh_profile.bulk_size_m),
        }
        return f"""// femx public ring heater; model coordinate unit = {values["unit"]} m
// recipe schema = femx.public-ring-heater-geometry/v1
// recipe SHA-256 = {values["digest"]}
// published dimensions: {PUBLIC_TIDY3D_RING_PAGE}
// femx adds two explicit top-contact vias; they are not part of the published source design.
SetFactory("OpenCASCADE");

Geometry.Tolerance = 1e-8;
Geometry.OCCBoundsUseStl = 1;
eps = {values["epsilon"]};
lcInterface = {values["lc_interface"]};
lcBulk = {values["lc_bulk"]};

domain = newv;
Box(domain) = {{{values["xmin"]}, {values["ymin"]}, {values["zmin"]}, {values["xlength"]}, {values["ylength"]}, {values["zlength"]}}};
substrate = newv;
Box(substrate) = {{{values["xmin"]}, {values["ymin"]}, {values["zmin"]}, {values["xlength"]}, {values["ylength"]}, {values["substrate_height"]}}};

ringOuter = newv;
Cylinder(ringOuter) = {{0, 0, 0, 0, 0, {values["waveguide_height"]}, {values["ring_outer"]}}};
ringInner = newv;
Cylinder(ringInner) = {{0, 0, 0, 0, 0, {values["waveguide_height"]}, {values["ring_inner"]}}};
ringVolumes() = BooleanDifference{{ Volume{{ringOuter}}; Delete; }}{{ Volume{{ringInner}}; Delete; }};

busUpper = newv;
Box(busUpper) = {{{values["xmin"]}, {values["bus_ymin_upper"]}, 0, {values["xlength"]}, {values["bus_width"]}, {values["waveguide_height"]}}};
busLower = newv;
Box(busLower) = {{{values["xmin"]}, {values["bus_ymin_lower"]}, 0, {values["xlength"]}, {values["bus_width"]}, {values["waveguide_height"]}}};

heaterOuter = newv;
Cylinder(heaterOuter) = {{0, 0, {values["heater_z"]}, 0, 0, {values["heater_height"]}, {values["heater_outer"]}}};
heaterInner = newv;
Cylinder(heaterInner) = {{0, 0, {values["heater_z"]}, 0, 0, {values["heater_height"]}, {values["heater_inner"]}}};
heaterNotch = newv;
Box(heaterNotch) = {{{values["notch_xmin"]}, {values["notch_ymin"]}, {values["notch_zmin"]}, {values["notch_x"]}, {values["notch_y"]}, {values["notch_height"]}}};
heaterVolumes() = BooleanDifference{{ Volume{{heaterOuter}}; Delete; }}{{ Volume{{heaterInner, heaterNotch}}; Delete; }};

contactNegative = newv;
Box(contactNegative) = {{{values["contact_negative_x"]}, {values["contact_ymin"]}, {values["contact_z"]}, {values["contact_width"]}, {values["contact_length"]}, {values["contact_height"]}}};
contactPositive = newv;
Box(contactPositive) = {{{values["contact_positive_x"]}, {values["contact_ymin"]}, {values["contact_z"]}, {values["contact_width"]}, {values["contact_length"]}, {values["contact_height"]}}};

silicaVolumes() = BooleanDifference{{ Volume{{domain}}; Delete; }}{{ Volume{{substrate, ringVolumes(), busUpper, busLower, heaterVolumes(), contactNegative, contactPositive}}; }};
Coherence;

allVolumes() = {{silicaVolumes(), substrate, ringVolumes(), busUpper, busLower, heaterVolumes(), contactNegative, contactPositive}};
externalSurfaces() = CombinedBoundary{{ Volume{{allVolumes()}}; }};
bottomSurfaces() = Surface In BoundingBox {{{values["xmin"]} - eps, {values["ymin"]} - eps, {values["zmin"]} - eps, {values["half_x"]} + eps, {values["half_y"]} + eps, {values["zmin"]} + eps}};
topSurfaces() = Surface In BoundingBox {{{values["xmin"]} - eps, {values["ymin"]} - eps, {values["top"]} - eps, {values["half_x"]} + eps, {values["half_y"]} + eps, {values["top"]} + eps}};
terminalNegative() = Surface In BoundingBox {{{values["contact_negative_x"]} - eps, {values["contact_ymin"]} - eps, {values["top"]} - eps, {values["contact_negative_xmax"]} + eps, {values["contact_ymax"]} + eps, {values["top"]} + eps}};
terminalPositive() = Surface In BoundingBox {{{values["contact_positive_x"]} - eps, {values["contact_ymin"]} - eps, {values["top"]} - eps, {values["contact_positive_xmax"]} + eps, {values["contact_ymax"]} + eps, {values["top"]} + eps}};
topConvection() = {{topSurfaces()}};
topConvection() -= {{terminalNegative(), terminalPositive()}};
xLowSurfaces() = Surface In BoundingBox {{{values["xmin"]} - eps, {values["ymin"]} - eps, {values["zmin"]} - eps, {values["xmin"]} + eps, {values["half_y"]} + eps, {values["top"]} + eps}};
xHighSurfaces() = Surface In BoundingBox {{{values["half_x"]} - eps, {values["ymin"]} - eps, {values["zmin"]} - eps, {values["half_x"]} + eps, {values["half_y"]} + eps, {values["top"]} + eps}};
yLowSurfaces() = Surface In BoundingBox {{{values["xmin"]} - eps, {values["ymin"]} - eps, {values["zmin"]} - eps, {values["half_x"]} + eps, {values["ymin"]} + eps, {values["top"]} + eps}};
yHighSurfaces() = Surface In BoundingBox {{{values["xmin"]} - eps, {values["half_y"]} - eps, {values["zmin"]} - eps, {values["half_x"]} + eps, {values["half_y"]} + eps, {values["top"]} + eps}};
lateralSurfaces() = {{xLowSurfaces(), xHighSurfaces(), yLowSurfaces(), yHighSurfaces()}};

Physical Volume("silica", 101) = {{silicaVolumes()}};
Physical Volume("silicon_substrate", 102) = {{substrate}};
Physical Volume("silicon_ring", 103) = {{ringVolumes()}};
Physical Volume("silicon_bus_upper", 104) = {{busUpper}};
Physical Volume("silicon_bus_lower", 105) = {{busLower}};
Physical Volume("tin_heater", 106) = {{heaterVolumes()}};
Physical Volume("al_contact_negative", 107) = {{contactNegative}};
Physical Volume("al_contact_positive", 108) = {{contactPositive}};

Physical Surface("external_boundary", 201) = {{externalSurfaces()}};
Physical Surface("bottom_temperature", 202) = {{bottomSurfaces()}};
Physical Surface("top_convection", 203) = {{topConvection()}};
Physical Surface("lateral_adiabatic", 204) = {{lateralSurfaces()}};
Physical Surface("terminal_negative", 205) = {{terminalNegative()}};
Physical Surface("terminal_positive", 206) = {{terminalPositive()}};

MeshSize {{ PointsOf{{ Volume{{ringVolumes(), busUpper, busLower, heaterVolumes(), contactNegative, contactPositive}}; }} }} = lcInterface;
Mesh.MeshSizeMin = lcInterface;
Mesh.MeshSizeMax = lcBulk;
Mesh.MeshSizeFromPoints = 1;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 1;
Mesh.MshFileVersion = 4.1;
Mesh.Binary = 0;
Mesh.ElementOrder = 1;
Mesh.SaveAll = 0;
Mesh.Algorithm3D = 10;
Mesh.AlgorithmSwitchOnFailure = 0;
Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 0;
Mesh.RandomFactor = 1e-9;
Mesh.RandomSeed = 1;
"""


@dataclass(frozen=True, slots=True)
class RingHeaterThermalSensitivity3D(PublicRingHeater3D):
    """Public device solids inside a separately identified computational envelope study."""

    SURFACE_GROUPS: ClassVar[tuple[str, ...]] = (
        "external_boundary",
        "bottom_boundary",
        "top_boundary",
        "lateral_boundary",
        "terminal_negative",
        "terminal_positive",
    )

    def canonical_data(self) -> dict[str, object]:
        """Return geometry provenance without relabeling a variant as source reproduction."""

        data = PublicRingHeater3D.canonical_data(self)
        data["schema_version"] = RING_HEATER_THERMAL_SENSITIVITY_GEOMETRY_SCHEMA
        data["study_scope"] = {
            "kind": "computational-envelope sensitivity around the public device solids",
            "varied_geometry": [
                "domain_x_m",
                "domain_y_m",
                "substrate_thickness_m",
            ],
            "boundary_assignment": "selected separately by the application-layer study policy",
            "claim_scope": (
                "numerical sensitivity only; not source-reproduction parity, package calibration, "
                "or fabricated-device prediction"
            ),
        }
        return data

    def render_geo(self) -> str:
        """Render generic thermal-boundary groups while preserving the device solids."""

        geometry = PublicRingHeater3D.render_geo(self)
        replacements = (
            ("// femx public ring heater", "// femx ring-heater thermal sensitivity"),
            (
                "// recipe schema = femx.public-ring-heater-geometry/v1",
                f"// recipe schema = {RING_HEATER_THERMAL_SENSITIVITY_GEOMETRY_SCHEMA}",
            ),
            (
                'Physical Surface("bottom_temperature", 202)',
                'Physical Surface("bottom_boundary", 202)',
            ),
            ('Physical Surface("top_convection", 203)', 'Physical Surface("top_boundary", 203)'),
            (
                'Physical Surface("lateral_adiabatic", 204)',
                'Physical Surface("lateral_boundary", 204)',
            ),
        )
        for source, replacement in replacements:
            if geometry.count(source) != 1:
                raise ContractError(
                    "ring-heater sensitivity renderer could not identify one source boundary"
                )
            geometry = geometry.replace(source, replacement)
        return geometry


def _format_real(value: Decimal) -> str:
    return format(value, ".17g")
