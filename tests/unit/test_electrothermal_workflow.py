from dataclasses import replace

import pytest
from tests.support import DummyPhysics, structured_unit_square_mesh

from femx.core.errors import ContractError
from femx.core.parameters import (
    ParameterReference,
    ParameterRole,
    ParameterSchema,
    ParameterSpec,
)
from femx.core.problem import Problem
from femx.physics import (
    ConductiveRegion,
    PotentialBoundary,
    SteadyCurrent,
    SteadyHeat,
    TemperatureBoundary,
    ThermalRegion,
)
from femx.workflows import (
    JOULE_HEAT_DENSITY_UNIT,
    SAME_MESH_CELL_LOCAL_P1_OPERATOR,
    SAME_MESH_CELL_OPERATOR,
    CoupledIterationPolicy,
    ResistivityTemperatureLaw,
    SameMeshJouleHeating,
    SelfConsistentJouleHeating,
)

pytestmark = pytest.mark.unit


def _problems() -> tuple[Problem, Problem]:
    mesh = structured_unit_square_mesh(1)
    electrical = Problem(
        "electrical",
        mesh,
        SteadyCurrent(
            regions=(ConductiveRegion("domain", 2.0),),
            potential_boundaries=(PotentialBoundary("left", 0.0),),
        ),
    )
    thermal = Problem(
        "thermal",
        mesh,
        SteadyHeat(
            regions=(ThermalRegion("domain", 3.0),),
            temperature_boundaries=(TemperatureBoundary("left", 300.0),),
        ),
    )
    return electrical, thermal


def test_same_mesh_joule_heating_declares_exact_typed_identity_transfer() -> None:
    electrical, thermal = _problems()
    coupling = SameMeshJouleHeating(electrical, thermal)

    assert coupling.graph.topological_order() == ("electrical", "thermal")
    edge = coupling.graph.edges[0]
    assert edge.operator == SAME_MESH_CELL_OPERATOR
    assert edge.source_unit == edge.target_unit == JOULE_HEAT_DENSITY_UNIT
    assert coupling.canonical_data() == {
        "schema_version": "femx.workflow.electrothermal/v1",
        "electrical_problem": "electrical",
        "thermal_problem": "thermal",
        "direction": "one_way",
        "feedback": False,
        "source": {
            "quantity": "joule_heat_density",
            "unit": "W/m^3",
            "function_space": "L2",
            "order": 0,
            "dof_location": "cell",
        },
        "target": {
            "quantity": "additive_cell_heat_source",
            "unit": "W/m^3",
            "function_space": "L2",
            "order": 0,
            "dof_location": "cell",
            "combination": "add",
        },
        "operator": "femx.transfer.same_mesh_l2_p0_identity/v1",
        "mesh_relation": "same_object_same_cell_order",
    }


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda electrical, thermal: SameMeshJouleHeating(
                replace(electrical, physics=DummyPhysics()), thermal
            ),
            "source problem",
        ),
        (
            lambda electrical, thermal: SameMeshJouleHeating(
                electrical, replace(thermal, physics=DummyPhysics())
            ),
            "target problem",
        ),
        (
            lambda electrical, thermal: SameMeshJouleHeating(
                electrical,
                replace(thermal, mesh=structured_unit_square_mesh(1)),
            ),
            "exact shared mesh object",
        ),
        (
            lambda electrical, thermal: SameMeshJouleHeating(
                electrical,
                thermal,
                schema_version="future",
            ),
            "unsupported electrothermal workflow schema",
        ),
    ],
)
def test_same_mesh_joule_heating_rejects_ambiguous_contracts(mutator, message: str) -> None:
    electrical, thermal = _problems()
    with pytest.raises(ContractError, match=message):
        mutator(electrical, thermal)


def _feedback() -> SelfConsistentJouleHeating:
    electrical, thermal = _problems()
    schema = ParameterSchema(
        (
            ParameterSpec(
                "alpha",
                unit="1/K",
                role=ParameterRole.DESIGN,
            ),
        )
    )
    return SelfConsistentJouleHeating(
        SameMeshJouleHeating(electrical, thermal),
        (
            ResistivityTemperatureLaw(
                "domain",
                reference_temperature=300.0,
                temperature_coefficient=ParameterReference("alpha"),
            ),
        ),
        parameters=schema,
    )


def test_feedback_contract_declares_local_law_iteration_and_implicit_residual() -> None:
    feedback = _feedback()

    assert feedback.conductivity_laws[0].canonical_data() == {
        "tag": "domain",
        "formula": "sigma_ref/(1+alpha*(temperature-reference_temperature))",
        "reference_temperature_K": 300.0,
        "temperature_coefficient_per_K": {"parameter": "alpha"},
        "invalid_domain": "nonpositive_denominator",
    }
    canonical = feedback.canonical_data()
    assert canonical["schema_version"] == "femx.workflow.electrothermal_feedback/v1"
    assert canonical["electrical_problem"] == "electrical"
    assert canonical["thermal_problem"] == "thermal"
    assert canonical["mesh_relation"] == "same_object_same_cell_order"
    assert canonical["parameter_names"] == ["alpha"]
    assert canonical["coupled_residual"] == ["current", "heat_with_joule"]
    assert canonical["differentiation"] == "implicit_coupled_residual"
    assert canonical["joule_transfer"] == {
        "operator": SAME_MESH_CELL_LOCAL_P1_OPERATOR,
        "quantity": "joule_heat_density",
        "unit": "W/m^3",
        "function_space": "L2",
        "order": 1,
        "continuity": "cell_local_discontinuous",
        "dof_location": "cell_local_vertex",
        "conductivity_evaluation": "law_at_temperature_vertex_dofs",
        "electric_field_evaluation": "constant_cell_gradient",
        "heat_load_integration": "consistent_p1_triangle",
    }
    assert canonical["iteration"] == {
        "algorithm": "block_gauss_seidel",
        "max_iterations": 100,
        "minimum_iterations": 2,
        "relative_tolerance": 1.0e-10,
        "residual_tolerance": 1.0e-10,
        "potential_absolute_tolerance_V": 1.0e-12,
        "temperature_absolute_tolerance_K": 1.0e-10,
        "potential_relaxation": 1.0,
        "temperature_relaxation": 0.5,
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_iterations": 0}, "max_iterations"),
        ({"max_iterations": 2, "minimum_iterations": 3}, "minimum_iterations"),
        ({"minimum_iterations": 0}, "minimum_iterations"),
        ({"relative_tolerance": 0.0}, "relative_tolerance"),
        ({"residual_tolerance": float("inf")}, "residual_tolerance"),
        ({"potential_absolute_tolerance": -1.0}, "potential_absolute_tolerance"),
        ({"temperature_absolute_tolerance": float("nan")}, "temperature_absolute_tolerance"),
        ({"potential_relaxation": 0.0}, "potential_relaxation"),
        ({"temperature_relaxation": 1.1}, "temperature_relaxation"),
    ],
)
def test_coupled_iteration_policy_rejects_invalid_controls(kwargs, message: str) -> None:
    with pytest.raises(ContractError, match=message):
        CoupledIterationPolicy(**kwargs)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda base: SelfConsistentJouleHeating(base, (), ParameterSchema()),
            "requires a conductivity law",
        ),
        (
            lambda base: SelfConsistentJouleHeating(
                base,
                (
                    ResistivityTemperatureLaw("domain", 300.0, 0.01),
                    ResistivityTemperatureLaw("domain", 300.0, 0.02),
                ),
                ParameterSchema(),
            ),
            "law tags must be unique",
        ),
        (
            lambda base: SelfConsistentJouleHeating(
                base,
                (ResistivityTemperatureLaw("missing", 300.0, 0.01),),
                ParameterSchema(),
            ),
            "unknown electrical regions",
        ),
        (
            lambda base: SelfConsistentJouleHeating(
                base,
                (ResistivityTemperatureLaw("domain", 300.0, ParameterReference("alpha")),),
                ParameterSchema(),
            ),
            "do not match",
        ),
        (
            lambda base: SelfConsistentJouleHeating(
                base,
                (ResistivityTemperatureLaw("domain", 300.0, ParameterReference("alpha")),),
                ParameterSchema((ParameterSpec("alpha", unit="K"),)),
            ),
            "unit '1/K'",
        ),
        (
            lambda base: SelfConsistentJouleHeating(
                base,
                (
                    ResistivityTemperatureLaw(
                        "domain",
                        ParameterReference("shared"),
                        ParameterReference("shared"),
                    ),
                ),
                ParameterSchema((ParameterSpec("shared", unit="K"),)),
            ),
            "incompatible units",
        ),
        (
            lambda base: SelfConsistentJouleHeating(
                base,
                (ResistivityTemperatureLaw("domain", 300.0, 0.01),),
                ParameterSchema(),
                schema_version="future",
            ),
            "unsupported electrothermal feedback schema",
        ),
    ],
)
def test_feedback_contract_rejects_ambiguous_laws(factory, message: str) -> None:
    electrical, thermal = _problems()
    base = SameMeshJouleHeating(electrical, thermal)
    with pytest.raises(ContractError, match=message):
        factory(base)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ResistivityTemperatureLaw(" bad", 300.0, 0.01), "law tag"),
        (lambda: ResistivityTemperatureLaw("domain", float("nan"), 0.01), "must be finite"),
        (lambda: ResistivityTemperatureLaw("domain", 300.0, True), "real scalar"),
    ],
)
def test_resistivity_temperature_law_rejects_invalid_coefficients(factory, message: str) -> None:
    with pytest.raises(ContractError, match=message):
        factory()
