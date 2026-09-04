import pytest
from tests.support import DummyPhysics, FakeArray, dummy_mesh

from femx.core.arrays import shape_of
from femx.core.errors import ContractError
from femx.core.problem import ObservableSpec, Problem
from femx.core.solution import ConvergenceReport, ConvergenceStatus, Field, Solution

pytestmark = pytest.mark.unit


def test_problem_validates_physics_and_observable_names() -> None:
    problem = Problem(
        name="heat",
        mesh=dummy_mesh(),
        physics=DummyPhysics(),
        observables=(ObservableSpec("max_temperature", "K", "max"),),
    )
    assert problem.requirements == problem.physics.requirements

    with pytest.raises(ContractError, match="unique"):
        Problem(
            "heat",
            dummy_mesh(),
            DummyPhysics(),
            observables=(ObservableSpec("x", "K"), ObservableSpec("x", "K")),
        )
    with pytest.raises(ValueError, match="invalid dummy"):
        Problem("heat", dummy_mesh(), DummyPhysics(valid=False))


def test_solution_freezes_mappings_and_separates_convergence() -> None:
    field = Field("temperature", FakeArray((3,)), "K", function_space="H1-P1")
    solution = Solution(
        backend_name="fake",
        backend_version="1",
        fields={"temperature": field},
        observables={"max_temperature": 301.0},
        convergence=ConvergenceReport(
            ConvergenceStatus.CONVERGED,
            iterations=4,
            residual_norm=1e-9,
            tolerance=1e-8,
        ),
    )
    assert solution.observables["max_temperature"] == 301.0
    with pytest.raises(TypeError):
        solution.fields["new"] = field  # type: ignore[index]


def test_solution_and_array_metadata_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        shape_of(FakeArray((-1, 2)))
    with pytest.raises(ContractError, match="negative"):
        ConvergenceReport(ConvergenceStatus.NOT_CONVERGED, iterations=-1)
    with pytest.raises(ContractError, match="mapping key"):
        Solution(
            "fake",
            "1",
            {"wrong": Field("right", FakeArray((1,)), "K", "H1")},
            {},
            ConvergenceReport(ConvergenceStatus.NOT_EVALUATED),
        )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ObservableSpec(" bad", "K"), "observable name"),
        (lambda: ObservableSpec("x", ""), "declare a unit"),
        (lambda: ObservableSpec("x", "K", ""), "declare a reduction"),
        (lambda: Problem(" bad", dummy_mesh(), DummyPhysics()), "problem name"),
        (lambda: Problem("bad", object(), DummyPhysics()), "MeshSpec"),
        (lambda: Problem("bad", dummy_mesh(), object()), "PhysicsSpec"),
        (
            lambda: Problem(
                "bad", dummy_mesh(), DummyPhysics(), schema_version="femx.problem/unknown"
            ),
            "unsupported problem schema",
        ),
        (lambda: Field(" bad", FakeArray((1,)), "K", "H1"), "field name"),
        (lambda: Field("x", FakeArray((1,)), "", "H1"), "declare a unit"),
        (lambda: Field("x", FakeArray((1,)), "K", None), "function space"),
    ],
)
def test_semantic_envelopes_reject_ambiguous_metadata(factory, message: str) -> None:
    with pytest.raises(ContractError, match=message):
        factory()


def test_convergence_and_solution_identity_validation() -> None:
    with pytest.raises(ContractError, match="residual norm"):
        ConvergenceReport(ConvergenceStatus.NOT_CONVERGED, residual_norm=-1)
    with pytest.raises(ContractError, match="tolerance"):
        ConvergenceReport(ConvergenceStatus.NOT_CONVERGED, tolerance=0)
    with pytest.raises(ContractError, match="backend_name"):
        Solution("", "1", {}, {}, ConvergenceReport(ConvergenceStatus.NOT_EVALUATED))
    with pytest.raises(ContractError, match="backend_version"):
        Solution("fake", "", {}, {}, ConvergenceReport(ConvergenceStatus.NOT_EVALUATED))
