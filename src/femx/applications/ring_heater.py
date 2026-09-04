"""Source-pinned 3D electrothermal assembly for the public ring-heater benchmark.

This application layer is intentionally above meshing and solver backends.  It binds one admitted
public Gmsh mesh to explicit benchmark-only material values and boundary conditions, then delegates
the numerical preparation to the native JAX Tet4 backend.  It does not promote tutorial or legacy
compatibility values to universal material defaults and does not claim foundry calibration.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from femx.backends.jax.tet4_electrothermal import (
    Tet4ElectrothermalPlan,
    prepare_tet4_electrothermal_plan,
)
from femx.core.errors import ContractError
from femx.meshing.gmsh.importer import ImportedGmshMesh
from femx.meshing.gmsh.ring_heater import (
    PUBLIC_TIDY3D_NOTEBOOK_REPOSITORY,
    PUBLIC_TIDY3D_NOTEBOOK_REVISION,
    PUBLIC_TIDY3D_NOTEBOOK_SHA256,
    PUBLIC_TIDY3D_RING_PAGE,
    PublicRingHeater3D,
    RingHeaterMeshProfile,
    RingHeaterThermalSensitivity3D,
)
from femx.meshing.gmsh.ring_heater_quality import (
    PublicRingHeaterMeshReport,
    evaluate_public_ring_heater_mesh,
)

PUBLIC_RING_HEATER_REFERENCE_SCHEMA = "femx.public-ring-heater-reference/v1"
PUBLIC_RING_HEATER_OPERATING_POINT_SCHEMA = "femx.public-ring-heater-operating-point/v1"
PUBLIC_RING_HEATER_FORWARD_SCHEMA = "femx.public-ring-heater-forward/v1"
RING_HEATER_THERMAL_BOUNDARY_SCHEMA = "femx.ring-heater-thermal-boundary/v1"
RING_HEATER_THERMAL_SENSITIVITY_CASE_SCHEMA = "femx.ring-heater-thermal-sensitivity-case/v1"
RING_HEATER_THERMAL_SENSITIVITY_SCHEMA = "femx.ring-heater-thermal-sensitivity/v1"
LINEAR_CURRENT_CALIBRATION_SCHEMA = "femx.linear-current-calibration/v1"

ELMER_REFERENCE_REVISION = "4f2d7e4b99f8f0dcf2f7ac579e056969373bf594"
ELMER_GUI_MATERIAL_XML_SHA256 = "50793332435448c2a7249e03ab32d1f4ae28c2413b6ffd927a10c4aae8662d29"
ELMER_GUI_MATERIAL_URL = (
    "https://github.com/ElmerCSC/elmerfem/blob/"
    f"{ELMER_REFERENCE_REVISION}/ElmerGUI/Application/edf/egmaterials.xml"
)

_CURRENT_REGIONS = ("tin_heater", "al_contact_negative", "al_contact_positive")
_SILICON_REGIONS = (
    "silicon_substrate",
    "silicon_ring",
    "silicon_bus_upper",
    "silicon_bus_lower",
)
_THERMAL_ROBIN_SURFACES = (
    "top_convection",
    "terminal_negative",
    "terminal_positive",
)
_PUBLIC_RING_HEATER_OPERATING_POINTS = {
    "source_reproduction": (0.015, "source_pinned_parity"),
    "low_temperature_projection": (0.005, "derived_linear_projection"),
}


def _canonical_payload_sha256(data: Mapping[str, object]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _positive_finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be a real scalar")
    canonical = float(value)
    if not math.isfinite(canonical) or canonical <= 0.0:
        raise ContractError(f"{label} must be finite and positive")
    return canonical


def _nonnegative_finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be a real scalar")
    canonical = float(value)
    if not math.isfinite(canonical) or canonical < 0.0:
        raise ContractError(f"{label} must be finite and nonnegative")
    return canonical


@dataclass(frozen=True, slots=True)
class PublicRingHeaterReferenceParameters:
    """Exact uncalibrated values used by the public M5 forward benchmark.

    The first seven values reproduce the public Tidy3D tutorial.  Aluminum is a declared femx-only
    geometry extension, so its two values are separately bound to the locked ElmerGUI compatibility
    observation.  Neither source is represented as a calibrated fabrication-process model.
    """

    ambient_temperature_k: float = 300.0
    target_current_a: float = 0.015
    convection_w_per_m2_k: float = 10.0
    silicon_thermal_conductivity_w_per_m_k: float = 148.0
    silica_thermal_conductivity_w_per_m_k: float = 1.38
    tin_thermal_conductivity_w_per_m_k: float = 28.0
    tin_electrical_conductivity_s_per_m: float = 2.3e6
    aluminum_thermal_conductivity_w_per_m_k: float = 237.0
    aluminum_electrical_conductivity_s_per_m: float = 37.73e6
    schema_version: str = PUBLIC_RING_HEATER_REFERENCE_SCHEMA

    def __post_init__(self) -> None:
        expected = {
            "ambient_temperature_k": 300.0,
            "target_current_a": 0.015,
            "convection_w_per_m2_k": 10.0,
            "silicon_thermal_conductivity_w_per_m_k": 148.0,
            "silica_thermal_conductivity_w_per_m_k": 1.38,
            "tin_thermal_conductivity_w_per_m_k": 28.0,
            "tin_electrical_conductivity_s_per_m": 2.3e6,
            "aluminum_thermal_conductivity_w_per_m_k": 237.0,
            "aluminum_electrical_conductivity_s_per_m": 37.73e6,
        }
        for name, source_value in expected.items():
            canonical = _positive_finite(getattr(self, name), label=name.replace("_", " "))
            if canonical != source_value:
                raise ContractError(
                    f"public ring-heater reference parameter {name!r} is source-pinned to "
                    f"{source_value!r}"
                )
            object.__setattr__(self, name, canonical)
        if self.schema_version != PUBLIC_RING_HEATER_REFERENCE_SCHEMA:
            raise ContractError(
                "public ring-heater reference schema must be "
                f"{PUBLIC_RING_HEATER_REFERENCE_SCHEMA!r}"
            )

    def canonical_data(self) -> dict[str, object]:
        """Return the complete values, source split, and scientific limitation."""

        return {
            "schema_version": self.schema_version,
            "parameter_set_id": "public_tidy3d_ring_with_femx_al_contacts",
            "evidence_tier": "public_benchmark_uncalibrated",
            "values_si": {
                "ambient_temperature_K": self.ambient_temperature_k,
                "target_current_A": self.target_current_a,
                "convection_W_per_m2_K": self.convection_w_per_m2_k,
                "silicon_thermal_conductivity_W_per_m_K": (
                    self.silicon_thermal_conductivity_w_per_m_k
                ),
                "silica_thermal_conductivity_W_per_m_K": (
                    self.silica_thermal_conductivity_w_per_m_k
                ),
                "tin_thermal_conductivity_W_per_m_K": (self.tin_thermal_conductivity_w_per_m_k),
                "tin_electrical_conductivity_S_per_m": (self.tin_electrical_conductivity_s_per_m),
                "aluminum_thermal_conductivity_W_per_m_K": (
                    self.aluminum_thermal_conductivity_w_per_m_k
                ),
                "aluminum_electrical_conductivity_S_per_m": (
                    self.aluminum_electrical_conductivity_s_per_m
                ),
            },
            "sources": {
                "public_tidy3d_tutorial": {
                    "page": PUBLIC_TIDY3D_RING_PAGE,
                    "repository": PUBLIC_TIDY3D_NOTEBOOK_REPOSITORY,
                    "revision": PUBLIC_TIDY3D_NOTEBOOK_REVISION,
                    "notebook_sha256": PUBLIC_TIDY3D_NOTEBOOK_SHA256,
                    "parameters": [
                        "ambient_temperature_K",
                        "target_current_A",
                        "convection_W_per_m2_K",
                        "silicon_thermal_conductivity_W_per_m_K",
                        "silica_thermal_conductivity_W_per_m_K",
                        "tin_thermal_conductivity_W_per_m_K",
                        "tin_electrical_conductivity_S_per_m",
                    ],
                    "original_excitation": "uniform TiN heat source inferred from 15 mA",
                },
                "elmer_gui_aluminum_compatibility": {
                    "url": ELMER_GUI_MATERIAL_URL,
                    "revision": ELMER_REFERENCE_REVISION,
                    "xml_sha256": ELMER_GUI_MATERIAL_XML_SHA256,
                    "material_name": "Aluminium (generic)",
                    "parameters": [
                        "aluminum_thermal_conductivity_W_per_m_K",
                        "aluminum_electrical_conductivity_S_per_m",
                    ],
                    "status": "legacy_unverified",
                },
            },
            "model_extensions": {
                "electrical": (
                    "solve 3D voltage-driven TiN plus aluminum conduction, then scale the linear "
                    "unit-voltage result to 15 mA"
                ),
                "thermal": (
                    "apply the published convection coefficient to the complete top plane, "
                    "including the two femx-only terminal tops"
                ),
            },
            "claim_scope": (
                "source-pinned public comparison parameters; not a universal material library, "
                "foundry calibration, or fabricated-device prediction"
            ),
        }

    def digest(self) -> str:
        """Hash every value, source identity, extension, and limitation."""

        return _canonical_payload_sha256(self.canonical_data())


@dataclass(frozen=True, slots=True)
class PublicRingHeaterOperatingPoint:
    """One explicit current role without changing the source-pinned material record."""

    name: str
    target_current_a: float
    evidence_tier: str
    schema_version: str = PUBLIC_RING_HEATER_OPERATING_POINT_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or self.name.strip() != self.name:
            raise ContractError(
                "public ring-heater operating-point name must be non-empty and trimmed"
            )
        try:
            expected_current, expected_tier = _PUBLIC_RING_HEATER_OPERATING_POINTS[self.name]
        except KeyError as error:
            raise ContractError(
                "public ring-heater operating point must be source_reproduction or "
                "low_temperature_projection"
            ) from error
        current = _positive_finite(self.target_current_a, label="target current")
        if current != expected_current:
            raise ContractError(
                f"public ring-heater operating point {self.name!r} pins target current to "
                f"{expected_current!r}"
            )
        object.__setattr__(self, "target_current_a", current)
        if self.evidence_tier != expected_tier:
            raise ContractError(
                f"public ring-heater operating point {self.name!r} requires evidence tier "
                f"{expected_tier!r}"
            )
        if self.schema_version != PUBLIC_RING_HEATER_OPERATING_POINT_SCHEMA:
            raise ContractError(
                "public ring-heater operating-point schema must be "
                f"{PUBLIC_RING_HEATER_OPERATING_POINT_SCHEMA!r}"
            )

    def canonical_data(self) -> dict[str, object]:
        """Return the current scaling and the claim boundary for this role."""

        source_current_a = _PUBLIC_RING_HEATER_OPERATING_POINTS["source_reproduction"][0]
        current_ratio = self.target_current_a / source_current_a
        if self.name == "source_reproduction":
            claim_scope = (
                "source-pinned current for same-discretization parity; not a recommended or "
                "calibrated fabricated-device operating point"
            )
        else:
            claim_scope = (
                "near-ambient linear projection of the source-pinned benchmark; not an "
                "independent solve, domain correction, or fabricated-device calibration"
            )
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "evidence_tier": self.evidence_tier,
            "target_current_A": self.target_current_a,
            "source_reproduction_current_A": source_current_a,
            "current_and_voltage_scale": current_ratio,
            "joule_power_and_temperature_rise_scale": current_ratio * current_ratio,
            "assumptions": [
                "linear temperature-independent electrical conductivity",
                "constant thermal conductivity and linear thermal boundaries",
            ],
            "claim_scope": claim_scope,
        }

    def digest(self) -> str:
        """Hash the selected role, current, scaling rule, and limitation."""

        return _canonical_payload_sha256(self.canonical_data())


def public_ring_heater_operating_point(name: str) -> PublicRingHeaterOperatingPoint:
    """Return one closed public operating-point role."""

    if not isinstance(name, str):
        raise ContractError("public ring-heater operating-point name must be a string")
    try:
        target_current_a, evidence_tier = _PUBLIC_RING_HEATER_OPERATING_POINTS[name]
    except KeyError as error:
        raise ContractError(
            "public ring-heater operating point must be source_reproduction or "
            "low_temperature_projection"
        ) from error
    return PublicRingHeaterOperatingPoint(
        name=name,
        target_current_a=target_current_a,
        evidence_tier=evidence_tier,
    )


@dataclass(frozen=True, slots=True)
class RingHeaterThermalBoundaryPolicy:
    """Explicit thermal sinks for a non-calibrated computational-envelope study."""

    ambient_temperature_k: float = 300.0
    top_transfer_w_per_m2_k: float = 10.0
    bottom_condition: str = "isothermal"
    bottom_transfer_w_per_m2_k: float = 0.0
    lateral_condition: str = "adiabatic"
    lateral_transfer_w_per_m2_k: float = 0.0
    schema_version: str = RING_HEATER_THERMAL_BOUNDARY_SCHEMA

    def __post_init__(self) -> None:
        ambient = _positive_finite(self.ambient_temperature_k, label="thermal ambient temperature")
        top_transfer = _positive_finite(
            self.top_transfer_w_per_m2_k,
            label="top thermal transfer",
        )
        object.__setattr__(self, "ambient_temperature_k", ambient)
        object.__setattr__(self, "top_transfer_w_per_m2_k", top_transfer)
        allowed = ("adiabatic", "isothermal", "robin")
        for boundary in ("bottom", "lateral"):
            condition_name = f"{boundary}_condition"
            transfer_name = f"{boundary}_transfer_w_per_m2_k"
            condition = getattr(self, condition_name)
            if not isinstance(condition, str) or condition not in allowed:
                raise ContractError(
                    f"ring-heater {boundary} condition must be adiabatic, isothermal, or robin"
                )
            transfer = _nonnegative_finite(
                getattr(self, transfer_name),
                label=f"{boundary} thermal transfer",
            )
            if condition == "robin" and transfer == 0.0:
                raise ContractError(
                    f"ring-heater {boundary} Robin condition requires positive transfer"
                )
            if condition != "robin" and transfer != 0.0:
                raise ContractError(
                    f"ring-heater {boundary} transfer must be zero unless condition is robin"
                )
            object.__setattr__(self, transfer_name, transfer)
        if self.schema_version != RING_HEATER_THERMAL_BOUNDARY_SCHEMA:
            raise ContractError(
                f"ring-heater thermal-boundary schema must be {RING_HEATER_THERMAL_BOUNDARY_SCHEMA!r}"
            )

    def canonical_data(self) -> dict[str, object]:
        """Return boundary values and the limit on their interpretation."""

        return {
            "schema_version": self.schema_version,
            "ambient_temperature_K": self.ambient_temperature_k,
            "top": {
                "condition": "robin",
                "transfer_W_per_m2_K": self.top_transfer_w_per_m2_k,
            },
            "bottom": {
                "condition": self.bottom_condition,
                "transfer_W_per_m2_K": self.bottom_transfer_w_per_m2_k,
            },
            "lateral": {
                "condition": self.lateral_condition,
                "transfer_W_per_m2_K": self.lateral_transfer_w_per_m2_k,
            },
            "claim_scope": (
                "declared numerical boundary scenario; Robin coefficients require package- or "
                "fixture-specific justification before device interpretation"
            ),
        }

    def digest(self) -> str:
        """Hash every thermal sink and its interpretation boundary."""

        return _canonical_payload_sha256(self.canonical_data())


@dataclass(frozen=True, slots=True)
class RingHeaterThermalSensitivityCase:
    """One named, one-factor-at-a-time envelope or boundary perturbation."""

    name: str
    varied_axis: str
    recipe: RingHeaterThermalSensitivity3D
    boundary: RingHeaterThermalBoundaryPolicy
    reference_case: str | None
    schema_version: str = RING_HEATER_THERMAL_SENSITIVITY_CASE_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or self.name.strip() != self.name:
            raise ContractError("ring-heater sensitivity case name must be non-empty and trimmed")
        allowed_axes = (
            "baseline",
            "substrate_depth",
            "lateral_extent",
            "depth_extent_interaction",
            "sidewall_boundary",
        )
        if not isinstance(self.varied_axis, str) or self.varied_axis not in allowed_axes:
            raise ContractError(
                "ring-heater sensitivity axis must be baseline, substrate_depth, lateral_extent, "
                "depth_extent_interaction, or sidewall_boundary"
            )
        if not isinstance(self.recipe, RingHeaterThermalSensitivity3D):
            raise ContractError("ring-heater sensitivity case requires its geometry recipe")
        if not isinstance(self.boundary, RingHeaterThermalBoundaryPolicy):
            raise ContractError("ring-heater sensitivity case requires a boundary policy")
        if self.varied_axis == "baseline":
            if self.reference_case is not None:
                raise ContractError(
                    "ring-heater sensitivity baseline cannot reference another case"
                )
        elif self.reference_case != "source_envelope":
            raise ContractError(
                "ring-heater sensitivity perturbations must reference source_envelope"
            )
        if self.schema_version != RING_HEATER_THERMAL_SENSITIVITY_CASE_SCHEMA:
            raise ContractError(
                "ring-heater sensitivity-case schema must be "
                f"{RING_HEATER_THERMAL_SENSITIVITY_CASE_SCHEMA!r}"
            )

    def canonical_data(self) -> dict[str, object]:
        """Return the named perturbation and its two content identities."""

        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "varied_axis": self.varied_axis,
            "reference_case": self.reference_case,
            "recipe_sha256": self.recipe.digest(),
            "boundary_sha256": self.boundary.digest(),
            "geometry_m": {
                "domain_x": self.recipe.domain_x_m,
                "domain_y": self.recipe.domain_y_m,
                "substrate_thickness": self.recipe.substrate_thickness_m,
            },
            "claim_scope": (
                "one controlled computational sensitivity factor; values are not a calibrated "
                "wafer, die, package, or operating-point recommendation"
            ),
        }

    def digest(self) -> str:
        """Hash the complete named perturbation."""

        return _canonical_payload_sha256(self.canonical_data())


def ring_heater_thermal_sensitivity_cases() -> tuple[RingHeaterThermalSensitivityCase, ...]:
    """Return the bounded initial factorial study without claiming full-domain convergence."""

    profile = RingHeaterMeshProfile("thermal-sensitivity-coarse", 0.28e-6, 1.28e-6)
    boundary = RingHeaterThermalBoundaryPolicy()

    def recipe(width_m: float, substrate_thickness_m: float) -> RingHeaterThermalSensitivity3D:
        return RingHeaterThermalSensitivity3D(
            profile,
            domain_x_m=width_m,
            domain_y_m=width_m,
            substrate_thickness_m=substrate_thickness_m,
        )

    return (
        RingHeaterThermalSensitivityCase(
            "source_envelope",
            "baseline",
            recipe(20.0e-6, 0.5e-6),
            boundary,
            None,
        ),
        RingHeaterThermalSensitivityCase(
            "substrate_5um",
            "substrate_depth",
            recipe(20.0e-6, 5.0e-6),
            boundary,
            "source_envelope",
        ),
        RingHeaterThermalSensitivityCase(
            "substrate_50um",
            "substrate_depth",
            recipe(20.0e-6, 50.0e-6),
            boundary,
            "source_envelope",
        ),
        RingHeaterThermalSensitivityCase(
            "domain_40um",
            "lateral_extent",
            recipe(40.0e-6, 0.5e-6),
            boundary,
            "source_envelope",
        ),
        RingHeaterThermalSensitivityCase(
            "domain_80um",
            "lateral_extent",
            recipe(80.0e-6, 0.5e-6),
            boundary,
            "source_envelope",
        ),
        RingHeaterThermalSensitivityCase(
            "domain_40um_substrate_5um",
            "depth_extent_interaction",
            recipe(40.0e-6, 5.0e-6),
            boundary,
            "source_envelope",
        ),
        RingHeaterThermalSensitivityCase(
            "domain_40um_substrate_50um",
            "depth_extent_interaction",
            recipe(40.0e-6, 50.0e-6),
            boundary,
            "source_envelope",
        ),
        RingHeaterThermalSensitivityCase(
            "domain_80um_substrate_5um",
            "depth_extent_interaction",
            recipe(80.0e-6, 5.0e-6),
            boundary,
            "source_envelope",
        ),
        RingHeaterThermalSensitivityCase(
            "domain_80um_substrate_50um",
            "depth_extent_interaction",
            recipe(80.0e-6, 50.0e-6),
            boundary,
            "source_envelope",
        ),
        RingHeaterThermalSensitivityCase(
            "ideal_isothermal_sidewall_bound",
            "sidewall_boundary",
            recipe(40.0e-6, 5.0e-6),
            RingHeaterThermalBoundaryPolicy(lateral_condition="isothermal"),
            "source_envelope",
        ),
    )


@dataclass(frozen=True, slots=True)
class PublicRingHeaterMeshAdmissionPolicy:
    """Fixed-recipe geometric checks required before physics binding."""

    minimum_mean_ratio: float = 0.05
    maximum_region_volume_relative_error: float = 1.0e-4
    maximum_full_domain_relative_volume_error: float = 1.0e-10

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            canonical = _positive_finite(getattr(self, name), label=name.replace("_", " "))
            object.__setattr__(self, name, canonical)
        if self.minimum_mean_ratio > 1.0:
            raise ContractError("public ring-heater minimum mean ratio cannot exceed one")
        for name in (
            "maximum_region_volume_relative_error",
            "maximum_full_domain_relative_volume_error",
        ):
            if getattr(self, name) >= 1.0:
                raise ContractError(
                    f"public ring-heater {name.replace('_', ' ')} must be below one"
                )

    def require(self, report: PublicRingHeaterMeshReport) -> None:
        """Fail unless one report satisfies every fixed-recipe threshold."""

        if not isinstance(report, PublicRingHeaterMeshReport):
            raise ContractError("public ring-heater mesh admission requires a mesh report")
        failures: list[str] = []
        if report.minimum_mean_ratio < self.minimum_mean_ratio:
            failures.append("minimum mean ratio")
        if report.maximum_region_volume_relative_error > self.maximum_region_volume_relative_error:
            failures.append("maximum region volume error")
        if (
            report.full_domain_relative_volume_error
            > self.maximum_full_domain_relative_volume_error
        ):
            failures.append("full-domain volume error")
        if failures:
            raise ContractError(
                "public ring-heater mesh fails physics admission: " + ", ".join(failures)
            )

    def canonical_data(self) -> dict[str, float]:
        """Return deterministic threshold metadata."""

        return {
            "minimum_mean_ratio": self.minimum_mean_ratio,
            "maximum_region_volume_relative_error": self.maximum_region_volume_relative_error,
            "maximum_full_domain_relative_volume_error": (
                self.maximum_full_domain_relative_volume_error
            ),
        }

    def digest(self) -> str:
        """Hash the exact admission thresholds."""

        return _canonical_payload_sha256(self.canonical_data())


@dataclass(frozen=True, slots=True)
class LinearCurrentCalibration:
    """Target-current voltage inferred from a linear unit-voltage conduction solve."""

    unit_voltage_joule_power_w: float
    conductance_s: float
    target_current_a: float
    target_voltage_v: float
    predicted_joule_power_w: float
    schema_version: str = LINEAR_CURRENT_CALIBRATION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "unit_voltage_joule_power_w",
            "conductance_s",
            "target_current_a",
            "target_voltage_v",
            "predicted_joule_power_w",
        ):
            canonical = _positive_finite(getattr(self, name), label=name.replace("_", " "))
            object.__setattr__(self, name, canonical)
        if self.schema_version != LINEAR_CURRENT_CALIBRATION_SCHEMA:
            raise ContractError(
                f"linear current calibration schema must be {LINEAR_CURRENT_CALIBRATION_SCHEMA!r}"
            )
        expected_voltage = self.target_current_a / self.conductance_s
        expected_power = self.target_current_a * expected_voltage
        if not math.isclose(self.unit_voltage_joule_power_w, self.conductance_s, rel_tol=1.0e-13):
            raise ContractError(
                "unit-voltage Joule power must equal linear conductance numerically"
            )
        if not math.isclose(self.target_voltage_v, expected_voltage, rel_tol=1.0e-13):
            raise ContractError("target voltage disagrees with the linear conductance")
        if not math.isclose(self.predicted_joule_power_w, expected_power, rel_tol=1.0e-13):
            raise ContractError("predicted Joule power disagrees with target current and voltage")

    def canonical_data(self) -> dict[str, object]:
        """Return the exact linear-rescaling record."""

        return {
            "schema_version": self.schema_version,
            "unit_voltage_V": 1.0,
            "unit_voltage_joule_power_W": self.unit_voltage_joule_power_w,
            "conductance_S": self.conductance_s,
            "target_current_A": self.target_current_a,
            "target_voltage_V": self.target_voltage_v,
            "predicted_joule_power_W": self.predicted_joule_power_w,
            "assumption": "linear temperature-independent electrical conductivity",
        }

    def digest(self) -> str:
        """Hash the complete current calibration."""

        return _canonical_payload_sha256(self.canonical_data())


def _calibrate_linear_current(
    unit_voltage_joule_power_w: object,
    *,
    target_current_a: float,
) -> LinearCurrentCalibration:
    conductance = _positive_finite(
        unit_voltage_joule_power_w,
        label="unit-voltage Joule power",
    )
    voltage = target_current_a / conductance
    return LinearCurrentCalibration(
        unit_voltage_joule_power_w=conductance,
        conductance_s=conductance,
        target_current_a=target_current_a,
        target_voltage_v=voltage,
        predicted_joule_power_w=target_current_a * voltage,
    )


def calibrate_public_ring_heater_current(
    unit_voltage_joule_power_w: object,
    *,
    reference: PublicRingHeaterReferenceParameters,
) -> LinearCurrentCalibration:
    """Map a finite admitted 1 V Joule result to the source-pinned target current."""

    if not isinstance(reference, PublicRingHeaterReferenceParameters):
        raise ContractError("public ring-heater current calibration requires reference parameters")
    return _calibrate_linear_current(
        unit_voltage_joule_power_w,
        target_current_a=reference.target_current_a,
    )


def project_public_ring_heater_current(
    unit_voltage_joule_power_w: object,
    *,
    operating_point: PublicRingHeaterOperatingPoint,
) -> LinearCurrentCalibration:
    """Map an admitted 1 V Joule result to one explicit public current role."""

    if not isinstance(operating_point, PublicRingHeaterOperatingPoint):
        raise ContractError("public ring-heater projection requires an operating point")
    return _calibrate_linear_current(
        unit_voltage_joule_power_w,
        target_current_a=operating_point.target_current_a,
    )


@dataclass(frozen=True, slots=True)
class PublicRingHeaterForwardPlan:
    """Physics-bound public benchmark plus the exact native Tet4 numerical plan."""

    reference: PublicRingHeaterReferenceParameters
    mesh_report: PublicRingHeaterMeshReport
    mesh_admission: PublicRingHeaterMeshAdmissionPolicy
    tet4: Tet4ElectrothermalPlan
    current_region_cell_counts: tuple[tuple[str, int], ...]
    terminal_node_counts: tuple[tuple[str, int], ...]
    schema_version: str = PUBLIC_RING_HEATER_FORWARD_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.reference, PublicRingHeaterReferenceParameters):
            raise ContractError("public ring-heater forward plan requires reference parameters")
        if not isinstance(self.mesh_admission, PublicRingHeaterMeshAdmissionPolicy):
            raise ContractError("public ring-heater forward plan requires mesh admission policy")
        if not isinstance(self.tet4, Tet4ElectrothermalPlan):
            raise ContractError("public ring-heater forward plan requires a Tet4 numerical plan")
        self.mesh_admission.require(self.mesh_report)
        if self.schema_version != PUBLIC_RING_HEATER_FORWARD_SCHEMA:
            raise ContractError(
                f"public ring-heater forward schema must be {PUBLIC_RING_HEATER_FORWARD_SCHEMA!r}"
            )
        expected_region_names = _CURRENT_REGIONS
        if tuple(name for name, _ in self.current_region_cell_counts) != expected_region_names:
            raise ContractError(
                "public ring-heater current-region counts have wrong names or order"
            )
        if any(count <= 0 for _, count in self.current_region_cell_counts):
            raise ContractError("public ring-heater current regions must contain cells")
        if sum(count for _, count in self.current_region_cell_counts) != (
            self.tet4.current_layout.topology.cell_count
        ):
            raise ContractError("public ring-heater current-region counts disagree with the plan")
        if self.tet4.thermal_layout.topology.cell_count != self.mesh_report.tetrahedron_count:
            raise ContractError("public ring-heater thermal cell count disagrees with the report")
        if tuple(name for name, _ in self.terminal_node_counts) != (
            "terminal_negative",
            "terminal_positive",
        ):
            raise ContractError("public ring-heater terminal counts have wrong names or order")
        if any(count <= 0 for _, count in self.terminal_node_counts):
            raise ContractError("public ring-heater terminals must contain nodes")

    def canonical_data(self) -> dict[str, object]:
        """Return the content-addressed geometry-to-physics binding."""

        return {
            "schema_version": self.schema_version,
            "reference_sha256": self.reference.digest(),
            "mesh_report_sha256": self.mesh_report.digest(),
            "mesh_admission_sha256": self.mesh_admission.digest(),
            "tet4_plan_sha256": self.tet4.digest(),
            "recipe_sha256": self.mesh_report.recipe_sha256,
            "import_record_sha256": self.mesh_report.import_record_sha256,
            "node_count": self.mesh_report.node_count,
            "tetrahedron_count": self.mesh_report.tetrahedron_count,
            "current_region_cell_counts": dict(self.current_region_cell_counts),
            "terminal_node_counts": dict(self.terminal_node_counts),
            "electrical_boundary_conditions": {
                "terminal_negative": "0 V",
                "terminal_positive": "applied_voltage",
                "remaining_conductor_boundary": "zero_normal_current",
            },
            "thermal_boundary_conditions": {
                "bottom_temperature": f"{self.reference.ambient_temperature_k:g} K",
                "top_convection_and_terminals": (
                    f"h={self.reference.convection_w_per_m2_k:g} W m^-2 K^-1 at "
                    f"{self.reference.ambient_temperature_k:g} K"
                ),
                "lateral_adiabatic": "zero_normal_heat_flux",
            },
            "joule_transfer": "exact conductor-parent-cell identity",
            "claim_scope": (
                "prepared source-pinned public 3D ring current/Joule/heat benchmark; numerical "
                "solution, mesh convergence, Elmer parity, TPU execution, FDTDX response, and "
                "foundry prediction require separate evidence"
            ),
        }

    def digest(self) -> str:
        """Hash the complete application binding without duplicating its arrays."""

        return _canonical_payload_sha256(self.canonical_data())


@dataclass(frozen=True, slots=True)
class RingHeaterThermalSensitivityPlan:
    """One geometry-and-boundary perturbation outside source-reproduction evidence."""

    reference: PublicRingHeaterReferenceParameters
    boundary: RingHeaterThermalBoundaryPolicy
    mesh_report: PublicRingHeaterMeshReport
    mesh_admission: PublicRingHeaterMeshAdmissionPolicy
    tet4: Tet4ElectrothermalPlan
    current_region_cell_counts: tuple[tuple[str, int], ...]
    terminal_node_counts: tuple[tuple[str, int], ...]
    schema_version: str = RING_HEATER_THERMAL_SENSITIVITY_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.reference, PublicRingHeaterReferenceParameters):
            raise ContractError("ring-heater thermal sensitivity requires reference materials")
        if not isinstance(self.boundary, RingHeaterThermalBoundaryPolicy):
            raise ContractError("ring-heater thermal sensitivity requires a boundary policy")
        if not isinstance(self.mesh_admission, PublicRingHeaterMeshAdmissionPolicy):
            raise ContractError("ring-heater thermal sensitivity requires mesh admission")
        if not isinstance(self.tet4, Tet4ElectrothermalPlan):
            raise ContractError("ring-heater thermal sensitivity requires a Tet4 plan")
        self.mesh_admission.require(self.mesh_report)
        if self.schema_version != RING_HEATER_THERMAL_SENSITIVITY_SCHEMA:
            raise ContractError(
                "ring-heater thermal-sensitivity schema must be "
                f"{RING_HEATER_THERMAL_SENSITIVITY_SCHEMA!r}"
            )
        if tuple(name for name, _ in self.current_region_cell_counts) != _CURRENT_REGIONS:
            raise ContractError("ring-heater sensitivity current-region names or order are wrong")
        if any(count <= 0 for _, count in self.current_region_cell_counts):
            raise ContractError("ring-heater sensitivity current regions must contain cells")
        if sum(count for _, count in self.current_region_cell_counts) != (
            self.tet4.current_layout.topology.cell_count
        ):
            raise ContractError("ring-heater sensitivity current-region counts disagree")
        if self.tet4.thermal_layout.topology.cell_count != self.mesh_report.tetrahedron_count:
            raise ContractError("ring-heater sensitivity thermal cell count disagrees")
        if tuple(name for name, _ in self.terminal_node_counts) != (
            "terminal_negative",
            "terminal_positive",
        ):
            raise ContractError("ring-heater sensitivity terminal names or order are wrong")
        if any(count <= 0 for _, count in self.terminal_node_counts):
            raise ContractError("ring-heater sensitivity terminals must contain nodes")

    def canonical_data(self) -> dict[str, object]:
        """Return content-addressed inputs and the sensitivity-only claim boundary."""

        return {
            "schema_version": self.schema_version,
            "reference_sha256": self.reference.digest(),
            "boundary_sha256": self.boundary.digest(),
            "boundary": self.boundary.canonical_data(),
            "mesh_report_sha256": self.mesh_report.digest(),
            "mesh_admission_sha256": self.mesh_admission.digest(),
            "tet4_plan_sha256": self.tet4.digest(),
            "recipe_sha256": self.mesh_report.recipe_sha256,
            "import_record_sha256": self.mesh_report.import_record_sha256,
            "node_count": self.mesh_report.node_count,
            "tetrahedron_count": self.mesh_report.tetrahedron_count,
            "current_region_cell_counts": dict(self.current_region_cell_counts),
            "terminal_node_counts": dict(self.terminal_node_counts),
            "electrical_boundary_conditions": {
                "terminal_negative": "0 V",
                "terminal_positive": "applied_voltage",
                "remaining_conductor_boundary": "zero_normal_current",
            },
            "joule_transfer": "exact conductor-parent-cell identity",
            "claim_scope": (
                "one numerical envelope or boundary perturbation; not source-reproduction parity, "
                "a calibrated package model, or fabricated-device prediction"
            ),
        }

    def digest(self) -> str:
        """Hash the complete sensitivity binding without duplicating arrays."""

        return _canonical_payload_sha256(self.canonical_data())


def _tag_ids(imported: ImportedGmshMesh, name: str, *, dimension: int) -> np.ndarray:
    matching = tuple(tag for tag in imported.mesh.tags if tag.name == name)
    if len(matching) != 1 or matching[0].dimension != dimension:
        raise ContractError(
            f"public ring-heater requires exactly one dimension-{dimension} tag {name!r}"
        )
    result = np.asarray(matching[0].entity_ids, dtype=np.int64)
    if result.size == 0:
        raise ContractError(f"public ring-heater tag {name!r} must be non-empty")
    return result


def _boundary_facets(imported: ImportedGmshMesh, name: str) -> np.ndarray:
    boundary = imported.mesh.boundary_facets
    if boundary is None:
        raise ContractError("public ring-heater physics requires boundary facets")
    return np.array(
        np.asarray(boundary.connectivity, dtype=np.int64)[_tag_ids(imported, name, dimension=2)],
        dtype=np.int64,
        copy=True,
    )


def _prepare_ring_heater_tet4(
    imported: ImportedGmshMesh,
    recipe: PublicRingHeater3D,
    cell_owners: object,
    *,
    partition_count: int,
    reference: PublicRingHeaterReferenceParameters,
    dirichlet_surface_names: tuple[str, ...],
    robin_surface_transfers: tuple[tuple[str, float], ...],
) -> tuple[
    Tet4ElectrothermalPlan,
    tuple[tuple[str, int], ...],
    tuple[tuple[str, int], ...],
]:
    """Prepare common device materials while keeping thermal boundary selection explicit."""

    mesh = imported.mesh
    coordinates = np.asarray(mesh.geometry.coordinates, dtype=np.float64)
    cells = np.asarray(mesh.topology.connectivity, dtype=np.int64)
    cell_count = cells.shape[0]
    region_ids = {name: _tag_ids(imported, name, dimension=3) for name in recipe.VOLUME_GROUPS}
    current_parent_cells = np.sort(
        np.concatenate(tuple(region_ids[name] for name in _CURRENT_REGIONS))
    )
    electrical_conductivity = np.full((cell_count,), np.nan, dtype=np.float64)
    electrical_conductivity[region_ids["tin_heater"]] = (
        reference.tin_electrical_conductivity_s_per_m
    )
    for name in ("al_contact_negative", "al_contact_positive"):
        electrical_conductivity[region_ids[name]] = (
            reference.aluminum_electrical_conductivity_s_per_m
        )
    current_conductivity = electrical_conductivity[current_parent_cells]

    thermal_conductivity = np.full((cell_count,), np.nan, dtype=np.float64)
    thermal_conductivity[region_ids["silica"]] = reference.silica_thermal_conductivity_w_per_m_k
    for name in _SILICON_REGIONS:
        thermal_conductivity[region_ids[name]] = reference.silicon_thermal_conductivity_w_per_m_k
    thermal_conductivity[region_ids["tin_heater"]] = reference.tin_thermal_conductivity_w_per_m_k
    for name in ("al_contact_negative", "al_contact_positive"):
        thermal_conductivity[region_ids[name]] = reference.aluminum_thermal_conductivity_w_per_m_k
    if not np.all(np.isfinite(thermal_conductivity)):
        raise ContractError("ring-heater thermal cells are not completely materialized")

    negative_nodes = np.unique(_boundary_facets(imported, "terminal_negative"))
    positive_nodes = np.unique(_boundary_facets(imported, "terminal_positive"))
    if np.intersect1d(negative_nodes, positive_nodes).size:
        raise ContractError("ring-heater electrical terminal nodes must be disjoint")
    current_dirichlet_nodes = np.concatenate((negative_nodes, positive_nodes))
    current_dirichlet_base = np.zeros(current_dirichlet_nodes.shape, dtype=np.float64)
    current_dirichlet_scale = np.concatenate(
        (
            np.zeros(negative_nodes.shape, dtype=np.float64),
            np.ones(positive_nodes.shape, dtype=np.float64),
        )
    )

    dirichlet_node_sets = tuple(
        np.unique(_boundary_facets(imported, name)) for name in dirichlet_surface_names
    )
    thermal_dirichlet_nodes = (
        np.unique(np.concatenate(dirichlet_node_sets))
        if dirichlet_node_sets
        else np.empty((0,), dtype=np.int64)
    )
    robin_facet_sets = tuple(
        (_boundary_facets(imported, name), transfer) for name, transfer in robin_surface_transfers
    )
    thermal_robin_facets = (
        np.concatenate(tuple(facets for facets, _ in robin_facet_sets), axis=0)
        if robin_facet_sets
        else np.empty((0, 3), dtype=np.int64)
    )
    thermal_robin_transfer = (
        np.concatenate(
            tuple(
                np.full((facets.shape[0],), transfer, dtype=np.float64)
                for facets, transfer in robin_facet_sets
            )
        )
        if robin_facet_sets
        else np.empty((0,), dtype=np.float64)
    )
    empty_facets = np.empty((0, 3), dtype=np.int64)
    empty_values = np.empty((0,), dtype=np.float64)
    tet4 = prepare_tet4_electrothermal_plan(
        coordinates,
        cells,
        cell_owners,
        current_parent_cells,
        current_conductivity=current_conductivity,
        current_cell_source=np.zeros(current_parent_cells.shape, dtype=np.float64),
        current_flux_facets=empty_facets,
        current_facet_flux=empty_values,
        current_dirichlet_nodes=current_dirichlet_nodes,
        current_dirichlet_base=current_dirichlet_base,
        current_dirichlet_voltage_scale=current_dirichlet_scale,
        thermal_conductivity=thermal_conductivity,
        thermal_cell_source=np.zeros((cell_count,), dtype=np.float64),
        thermal_flux_facets=empty_facets,
        thermal_facet_flux=empty_values,
        thermal_robin_facets=thermal_robin_facets,
        thermal_robin_transfer=thermal_robin_transfer,
        thermal_robin_ambient=np.full(
            (thermal_robin_facets.shape[0],),
            reference.ambient_temperature_k,
            dtype=np.float64,
        ),
        thermal_dirichlet_nodes=thermal_dirichlet_nodes,
        thermal_dirichlet_values=np.full(
            thermal_dirichlet_nodes.shape,
            reference.ambient_temperature_k,
            dtype=np.float64,
        ),
        thermal_reference=reference.ambient_temperature_k,
        partition_count=partition_count,
    )
    current_counts = tuple((name, int(region_ids[name].size)) for name in _CURRENT_REGIONS)
    terminal_counts = (
        ("terminal_negative", int(negative_nodes.size)),
        ("terminal_positive", int(positive_nodes.size)),
    )
    return tet4, current_counts, terminal_counts


def prepare_public_ring_heater_forward_plan(
    imported: ImportedGmshMesh,
    recipe: PublicRingHeater3D,
    cell_owners: object,
    *,
    partition_count: int,
    reference: PublicRingHeaterReferenceParameters | None = None,
    mesh_admission: PublicRingHeaterMeshAdmissionPolicy | None = None,
) -> PublicRingHeaterForwardPlan:
    """Bind one admitted public mesh to explicit current/Joule/heat coefficients and boundaries."""

    if not isinstance(imported, ImportedGmshMesh):
        raise ContractError("public ring-heater preparation requires an imported Gmsh mesh")
    if not isinstance(recipe, PublicRingHeater3D):
        raise ContractError("public ring-heater preparation requires the public geometry recipe")
    bound_reference = PublicRingHeaterReferenceParameters() if reference is None else reference
    if not isinstance(bound_reference, PublicRingHeaterReferenceParameters):
        raise ContractError("public ring-heater preparation requires reference parameters")
    bound_admission = (
        PublicRingHeaterMeshAdmissionPolicy() if mesh_admission is None else mesh_admission
    )
    if not isinstance(bound_admission, PublicRingHeaterMeshAdmissionPolicy):
        raise ContractError("public ring-heater preparation requires a mesh admission policy")

    report = evaluate_public_ring_heater_mesh(imported, recipe)
    bound_admission.require(report)
    tet4, current_counts, terminal_counts = _prepare_ring_heater_tet4(
        imported,
        recipe,
        cell_owners,
        partition_count=partition_count,
        reference=bound_reference,
        dirichlet_surface_names=("bottom_temperature",),
        robin_surface_transfers=tuple(
            (name, bound_reference.convection_w_per_m2_k) for name in _THERMAL_ROBIN_SURFACES
        ),
    )
    return PublicRingHeaterForwardPlan(
        reference=bound_reference,
        mesh_report=report,
        mesh_admission=bound_admission,
        tet4=tet4,
        current_region_cell_counts=current_counts,
        terminal_node_counts=terminal_counts,
    )


def prepare_ring_heater_thermal_sensitivity_plan(
    imported: ImportedGmshMesh,
    recipe: RingHeaterThermalSensitivity3D,
    cell_owners: object,
    *,
    partition_count: int,
    boundary: RingHeaterThermalBoundaryPolicy,
    reference: PublicRingHeaterReferenceParameters | None = None,
    mesh_admission: PublicRingHeaterMeshAdmissionPolicy | None = None,
) -> RingHeaterThermalSensitivityPlan:
    """Bind one independently identified envelope and thermal-boundary perturbation."""

    if not isinstance(imported, ImportedGmshMesh):
        raise ContractError("ring-heater sensitivity preparation requires an imported Gmsh mesh")
    if not isinstance(recipe, RingHeaterThermalSensitivity3D):
        raise ContractError("ring-heater sensitivity preparation requires its geometry recipe")
    if not isinstance(boundary, RingHeaterThermalBoundaryPolicy):
        raise ContractError("ring-heater sensitivity preparation requires a boundary policy")
    bound_reference = PublicRingHeaterReferenceParameters() if reference is None else reference
    if not isinstance(bound_reference, PublicRingHeaterReferenceParameters):
        raise ContractError("ring-heater sensitivity preparation requires reference materials")
    if boundary.ambient_temperature_k != bound_reference.ambient_temperature_k:
        raise ContractError(
            "ring-heater sensitivity boundary ambient must match the reference ambient"
        )
    bound_admission = (
        PublicRingHeaterMeshAdmissionPolicy() if mesh_admission is None else mesh_admission
    )
    if not isinstance(bound_admission, PublicRingHeaterMeshAdmissionPolicy):
        raise ContractError("ring-heater sensitivity preparation requires mesh admission")

    report = evaluate_public_ring_heater_mesh(imported, recipe)
    bound_admission.require(report)
    dirichlet_surfaces: list[str] = []
    robin_surfaces: list[tuple[str, float]] = [
        ("top_boundary", boundary.top_transfer_w_per_m2_k),
        ("terminal_negative", boundary.top_transfer_w_per_m2_k),
        ("terminal_positive", boundary.top_transfer_w_per_m2_k),
    ]
    for name, condition, transfer in (
        (
            "bottom_boundary",
            boundary.bottom_condition,
            boundary.bottom_transfer_w_per_m2_k,
        ),
        (
            "lateral_boundary",
            boundary.lateral_condition,
            boundary.lateral_transfer_w_per_m2_k,
        ),
    ):
        if condition == "isothermal":
            dirichlet_surfaces.append(name)
        elif condition == "robin":
            robin_surfaces.append((name, transfer))

    tet4, current_counts, terminal_counts = _prepare_ring_heater_tet4(
        imported,
        recipe,
        cell_owners,
        partition_count=partition_count,
        reference=bound_reference,
        dirichlet_surface_names=tuple(dirichlet_surfaces),
        robin_surface_transfers=tuple(robin_surfaces),
    )
    return RingHeaterThermalSensitivityPlan(
        reference=bound_reference,
        boundary=boundary,
        mesh_report=report,
        mesh_admission=bound_admission,
        tet4=tet4,
        current_region_cell_counts=current_counts,
        terminal_node_counts=terminal_counts,
    )
