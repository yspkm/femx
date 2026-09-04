from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from tests.unit.test_public_ring_heater_forward import _prepare

from femx.applications import (
    PUBLIC_RING_HEATER_ELMER_SCHEMA,
    prepare_public_ring_heater_elmer_plan,
)
from femx.applications import ring_heater_elmer as elmer_application
from femx.core.errors import ContractError

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def test_public_ring_elmer_plan_reuses_exact_jax_mesh_physics_and_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported, recipe, _report, forward = _prepare(monkeypatch)
    plan = prepare_public_ring_heater_elmer_plan(
        imported,
        recipe,
        forward,
        applied_voltage_v=1.25,
    )

    assert plan.schema_version == PUBLIC_RING_HEATER_ELMER_SCHEMA
    assert plan.applied_voltage_v == 1.25
    assert plan.region_body_ids == tuple(
        (name, index) for index, name in enumerate(recipe.VOLUME_GROUPS, start=1)
    )
    assert tuple(name for name, _ in plan.boundary_ids) == (
        "bottom_temperature",
        "top_convection",
        "lateral_adiabatic",
        "terminal_negative",
        "terminal_positive",
    )
    assert plan.case.mesh.node_count == imported.mesh.geometry.node_count
    assert plan.case.mesh.element_count == imported.mesh.topology.cell_count
    assert plan.case.potential_node_ids == tuple(
        int(node) for node in forward.tet4.current_parent_node_ids
    )
    body_by_name = dict(zip(recipe.VOLUME_GROUPS, plan.case.bodies, strict=True))
    assert body_by_name["silica"].electric_conductivity_s_per_m is None
    assert body_by_name["tin_heater"].electric_conductivity_s_per_m == 2.3e6
    assert body_by_name["al_contact_negative"].electric_conductivity_s_per_m == 37.73e6
    assert body_by_name["silicon_ring"].heat_conductivity_w_per_m_k == 148.0
    conditions = {condition.boundary_id: condition for condition in plan.case.boundaries}
    boundary_ids = dict(plan.boundary_ids)
    assert boundary_ids["lateral_adiabatic"] not in conditions
    assert conditions[boundary_ids["terminal_negative"]].potential_v == 0.0
    assert conditions[boundary_ids["terminal_positive"]].potential_v == 1.25
    assert conditions[boundary_ids["top_convection"]].external_temperature_k == 300.0
    data = plan.canonical_data()
    assert data["forward_plan_sha256"] == forward.digest()
    assert "same-mesh" in data["claim_scope"]
    assert len(plan.digest()) == 64
    assert (
        plan.digest()
        == prepare_public_ring_heater_elmer_plan(
            imported,
            recipe,
            forward,
            applied_voltage_v=1.25,
        ).digest()
    )


@pytest.mark.parametrize("voltage", (True, "1.0", 0.0, -1.0, float("nan")))
def test_public_ring_elmer_plan_rejects_invalid_voltage(
    monkeypatch: pytest.MonkeyPatch,
    voltage: object,
) -> None:
    imported, recipe, _report, forward = _prepare(monkeypatch)
    with pytest.raises(ContractError, match="voltage must be finite and positive"):
        prepare_public_ring_heater_elmer_plan(
            imported,
            recipe,
            forward,
            applied_voltage_v=voltage,  # type: ignore[arg-type]
        )


def test_public_ring_elmer_preparation_rejects_cross_model_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported, recipe, _report, forward = _prepare(monkeypatch)
    with pytest.raises(ContractError, match="imported Gmsh mesh"):
        prepare_public_ring_heater_elmer_plan(  # type: ignore[arg-type]
            object(), recipe, forward, applied_voltage_v=1.0
        )
    with pytest.raises(ContractError, match="public recipe"):
        prepare_public_ring_heater_elmer_plan(  # type: ignore[arg-type]
            imported, object(), forward, applied_voltage_v=1.0
        )
    with pytest.raises(ContractError, match="JAX forward plan"):
        prepare_public_ring_heater_elmer_plan(  # type: ignore[arg-type]
            imported, recipe, object(), applied_voltage_v=1.0
        )
    with pytest.raises(ContractError, match="recipe differs"):
        prepare_public_ring_heater_elmer_plan(
            imported,
            replace(recipe, domain_x_m=recipe.domain_x_m * 1.01),
            forward,
            applied_voltage_v=1.0,
        )
    changed_record = replace(imported.record, source_sha256="b" * 64)
    with pytest.raises(ContractError, match="mesh differs"):
        prepare_public_ring_heater_elmer_plan(
            replace(imported, record=changed_record),
            recipe,
            forward,
            applied_voltage_v=1.0,
        )


def test_public_ring_elmer_plan_rejects_corrupt_provenance_and_maps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported, recipe, _report, forward = _prepare(monkeypatch)
    plan = prepare_public_ring_heater_elmer_plan(
        imported,
        recipe,
        forward,
        applied_voltage_v=1.0,
    )
    with pytest.raises(ContractError, match="typed Tet4 case"):
        replace(plan, case=object())  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="digest must be SHA-256"):
        replace(plan, recipe_sha256="bad")
    with pytest.raises(ContractError, match="digest must be SHA-256"):
        replace(plan, reference_sha256=object())  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="region/body map differs"):
        replace(plan, region_body_ids=plan.region_body_ids[:-1])
    renamed = (("wrong", 1), *plan.region_body_ids[1:])
    with pytest.raises(ContractError, match="region/body names"):
        replace(plan, region_body_ids=renamed)
    with pytest.raises(ContractError, match="boundary map differs"):
        replace(plan, boundary_ids=plan.boundary_ids[:-1])
    renamed_boundaries = (("wrong", 1), *plan.boundary_ids[1:])
    with pytest.raises(ContractError, match="boundary names"):
        replace(plan, boundary_ids=renamed_boundaries)
    with pytest.raises(ContractError, match="schema must be"):
        replace(plan, schema_version="wrong")
    with pytest.raises(ContractError, match="voltage must be finite and positive"):
        replace(plan, applied_voltage_v=np.inf)


def test_public_ring_elmer_plan_checks_jax_conductor_node_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported, recipe, _report, forward = _prepare(monkeypatch)
    real_lower = elmer_application.lower_tagged_tet4_mesh
    deck = real_lower(
        imported.mesh,
        region_tags=recipe.VOLUME_GROUPS,
        boundary_tags=(
            "bottom_temperature",
            "top_convection",
            "lateral_adiabatic",
            "terminal_negative",
            "terminal_positive",
        ),
    )
    expected = set(int(node) for node in forward.tet4.current_parent_node_ids)
    extra = next(node for node in range(deck.node_count) if node not in expected)
    body_nodes = list(deck.body_node_ids)
    heater_index = recipe.VOLUME_GROUPS.index("tin_heater")
    body_nodes[heater_index] = tuple(sorted((*body_nodes[heater_index], extra)))
    changed_deck = replace(deck, body_node_ids=tuple(body_nodes))
    monkeypatch.setattr(
        elmer_application,
        "lower_tagged_tet4_mesh",
        lambda *_args, **_kwargs: changed_deck,
    )

    with pytest.raises(ContractError, match="electrical nodes differ"):
        prepare_public_ring_heater_elmer_plan(
            imported,
            recipe,
            forward,
            applied_voltage_v=1.0,
        )
