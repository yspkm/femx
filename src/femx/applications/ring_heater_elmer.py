"""Elmer oracle binding for the public 3D ring-heater forward benchmark."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from femx.applications.ring_heater import (
    PublicRingHeaterForwardPlan,
    PublicRingHeaterReferenceParameters,
)
from femx.backends.elmer.tet4_case import lower_tagged_tet4_mesh
from femx.backends.elmer.tet4_electrothermal_case import (
    ElmerTet4BoundaryCondition,
    ElmerTet4ElectrothermalBody,
    ElmerTet4ElectrothermalCase,
)
from femx.core.errors import ContractError
from femx.meshing.gmsh.importer import ImportedGmshMesh
from femx.meshing.gmsh.ring_heater import PublicRingHeater3D

PUBLIC_RING_HEATER_ELMER_SCHEMA = "femx.public-ring-heater-elmer/v1"

_BOUNDARY_GROUPS = (
    "bottom_temperature",
    "top_convection",
    "lateral_adiabatic",
    "terminal_negative",
    "terminal_positive",
)


def _canonical_sha256(data: Mapping[str, object]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PublicRingHeaterElmerPlan:
    """Content-addressed Elmer case bound to one already-admitted JAX forward plan."""

    case: ElmerTet4ElectrothermalCase
    applied_voltage_v: float
    recipe_sha256: str
    import_record_sha256: str
    forward_plan_sha256: str
    reference_sha256: str
    region_body_ids: tuple[tuple[str, int], ...]
    boundary_ids: tuple[tuple[str, int], ...]
    schema_version: str = PUBLIC_RING_HEATER_ELMER_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.case, ElmerTet4ElectrothermalCase):
            raise ContractError("public ring-heater Elmer plan requires a typed Tet4 case")
        if (
            isinstance(self.applied_voltage_v, bool)
            or not isinstance(self.applied_voltage_v, (int, float))
            or not math.isfinite(float(self.applied_voltage_v))
            or float(self.applied_voltage_v) <= 0.0
        ):
            raise ContractError("public ring-heater Elmer voltage must be finite and positive")
        object.__setattr__(self, "applied_voltage_v", float(self.applied_voltage_v))
        for label, digest in (
            ("recipe", self.recipe_sha256),
            ("import record", self.import_record_sha256),
            ("forward plan", self.forward_plan_sha256),
            ("reference", self.reference_sha256),
        ):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ContractError(f"public ring-heater Elmer {label} digest must be SHA-256")
        if tuple(identifier for _, identifier in self.region_body_ids) != self.case.mesh.body_ids:
            raise ContractError("public ring-heater Elmer region/body map differs from its mesh")
        if tuple(name for name, _ in self.region_body_ids) != PublicRingHeater3D.VOLUME_GROUPS:
            raise ContractError("public ring-heater Elmer region/body names are noncanonical")
        if tuple(identifier for _, identifier in self.boundary_ids) != self.case.mesh.boundary_ids:
            raise ContractError("public ring-heater Elmer boundary map differs from its mesh")
        if tuple(name for name, _ in self.boundary_ids) != _BOUNDARY_GROUPS:
            raise ContractError("public ring-heater Elmer boundary names are noncanonical")
        if self.schema_version != PUBLIC_RING_HEATER_ELMER_SCHEMA:
            raise ContractError(
                f"public ring-heater Elmer schema must be {PUBLIC_RING_HEATER_ELMER_SCHEMA!r}"
            )

    def canonical_data(self) -> dict[str, object]:
        """Return the exact same-mesh oracle binding and its limited claim scope."""

        return {
            "schema_version": self.schema_version,
            "case_sha256": self.case.digest(),
            "applied_voltage_V": self.applied_voltage_v,
            "recipe_sha256": self.recipe_sha256,
            "import_record_sha256": self.import_record_sha256,
            "forward_plan_sha256": self.forward_plan_sha256,
            "reference_sha256": self.reference_sha256,
            "region_body_ids": dict(self.region_body_ids),
            "boundary_ids": dict(self.boundary_ids),
            "potential_node_ids": list(self.case.potential_node_ids),
            "claim_scope": (
                "prepared same-mesh Elmer oracle input; process success, convergence, numerical "
                "parity, mesh convergence, TPU execution, FDTDX response, and fabricated-device "
                "accuracy require separate evidence"
            ),
        }

    def digest(self) -> str:
        """Hash the complete public application-to-Elmer binding."""

        return _canonical_sha256(self.canonical_data())


def _body_coefficients(
    reference: PublicRingHeaterReferenceParameters,
) -> dict[str, tuple[float, float | None]]:
    return {
        "silica": (reference.silica_thermal_conductivity_w_per_m_k, None),
        "silicon_substrate": (reference.silicon_thermal_conductivity_w_per_m_k, None),
        "silicon_ring": (reference.silicon_thermal_conductivity_w_per_m_k, None),
        "silicon_bus_upper": (reference.silicon_thermal_conductivity_w_per_m_k, None),
        "silicon_bus_lower": (reference.silicon_thermal_conductivity_w_per_m_k, None),
        "tin_heater": (
            reference.tin_thermal_conductivity_w_per_m_k,
            reference.tin_electrical_conductivity_s_per_m,
        ),
        "al_contact_negative": (
            reference.aluminum_thermal_conductivity_w_per_m_k,
            reference.aluminum_electrical_conductivity_s_per_m,
        ),
        "al_contact_positive": (
            reference.aluminum_thermal_conductivity_w_per_m_k,
            reference.aluminum_electrical_conductivity_s_per_m,
        ),
    }


def prepare_public_ring_heater_elmer_plan(
    imported: ImportedGmshMesh,
    recipe: PublicRingHeater3D,
    forward: PublicRingHeaterForwardPlan,
    *,
    applied_voltage_v: float,
) -> PublicRingHeaterElmerPlan:
    """Bind the admitted public JAX plan to a distinct-space Elmer Tet4 oracle case."""

    if not isinstance(imported, ImportedGmshMesh):
        raise ContractError("public ring-heater Elmer preparation requires an imported Gmsh mesh")
    if not isinstance(recipe, PublicRingHeater3D):
        raise ContractError("public ring-heater Elmer preparation requires the public recipe")
    if not isinstance(forward, PublicRingHeaterForwardPlan):
        raise ContractError("public ring-heater Elmer preparation requires a JAX forward plan")
    if isinstance(applied_voltage_v, bool) or not isinstance(applied_voltage_v, (int, float)):
        raise ContractError("public ring-heater Elmer voltage must be finite and positive")
    voltage = float(applied_voltage_v)
    if not math.isfinite(voltage) or voltage <= 0.0:
        raise ContractError("public ring-heater Elmer voltage must be finite and positive")
    recipe_sha256 = recipe.digest()
    import_sha256 = imported.record.digest()
    if forward.mesh_report.recipe_sha256 != recipe_sha256:
        raise ContractError("public ring-heater Elmer recipe differs from the JAX forward plan")
    if forward.mesh_report.import_record_sha256 != import_sha256:
        raise ContractError("public ring-heater Elmer mesh differs from the JAX forward plan")

    mesh = lower_tagged_tet4_mesh(
        imported.mesh,
        region_tags=recipe.VOLUME_GROUPS,
        boundary_tags=_BOUNDARY_GROUPS,
    )
    coefficients = _body_coefficients(forward.reference)
    bodies = tuple(
        ElmerTet4ElectrothermalBody(
            body_id,
            heat_conductivity_w_per_m_k=coefficients[name][0],
            electric_conductivity_s_per_m=coefficients[name][1],
        )
        for body_id, name in enumerate(recipe.VOLUME_GROUPS, start=1)
    )
    boundary_by_name = dict(zip(_BOUNDARY_GROUPS, mesh.boundary_ids, strict=True))
    reference = forward.reference
    boundaries = (
        ElmerTet4BoundaryCondition(
            boundary_by_name["bottom_temperature"],
            temperature_k=reference.ambient_temperature_k,
        ),
        ElmerTet4BoundaryCondition(
            boundary_by_name["top_convection"],
            heat_transfer_coefficient_w_per_m2_k=reference.convection_w_per_m2_k,
            external_temperature_k=reference.ambient_temperature_k,
        ),
        ElmerTet4BoundaryCondition(
            boundary_by_name["terminal_negative"],
            potential_v=0.0,
            heat_transfer_coefficient_w_per_m2_k=reference.convection_w_per_m2_k,
            external_temperature_k=reference.ambient_temperature_k,
        ),
        ElmerTet4BoundaryCondition(
            boundary_by_name["terminal_positive"],
            potential_v=voltage,
            heat_transfer_coefficient_w_per_m2_k=reference.convection_w_per_m2_k,
            external_temperature_k=reference.ambient_temperature_k,
        ),
    )
    case = ElmerTet4ElectrothermalCase(
        mesh=mesh,
        bodies=bodies,
        boundaries=boundaries,
        initial_temperature_k=reference.ambient_temperature_k,
    )
    expected_potential_nodes = tuple(
        int(node) for node in np.asarray(forward.tet4.current_parent_node_ids)
    )
    if case.potential_node_ids != expected_potential_nodes:
        raise ContractError(
            "public ring-heater Elmer electrical nodes differ from the JAX conductor submesh"
        )
    return PublicRingHeaterElmerPlan(
        case=case,
        applied_voltage_v=voltage,
        recipe_sha256=recipe_sha256,
        import_record_sha256=import_sha256,
        forward_plan_sha256=forward.digest(),
        reference_sha256=reference.digest(),
        region_body_ids=tuple(zip(recipe.VOLUME_GROUPS, mesh.body_ids, strict=True)),
        boundary_ids=tuple(zip(_BOUNDARY_GROUPS, mesh.boundary_ids, strict=True)),
    )


__all__ = [
    "PUBLIC_RING_HEATER_ELMER_SCHEMA",
    "PublicRingHeaterElmerPlan",
    "prepare_public_ring_heater_elmer_plan",
]
