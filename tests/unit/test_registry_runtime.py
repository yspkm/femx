import pytest
from tests.support import DummyPhysics, FakeBackend, dummy_mesh

from femx.backends import BackendDescriptor, BackendRegistry, PreparedProblem
from femx.core.errors import BackendError, BackendUnavailableError, ContractError
from femx.core.parameters import ParameterSchema, ParameterSpec
from femx.core.problem import Problem
from femx.core.solution import ConvergenceReport, ConvergenceStatus, Solution
from femx.runtime import prepare, solve

pytestmark = pytest.mark.unit


def test_registry_is_explicit_and_never_falls_back() -> None:
    registry = BackendRegistry()
    registry.register("fake", FakeBackend)

    assert registry.names() == ("fake",)
    assert registry.create("fake").descriptor.name == "fake"
    with pytest.raises(ContractError, match="already"):
        registry.register("fake", FakeBackend)
    with pytest.raises(BackendUnavailableError, match="available: fake"):
        registry.create("unknown")


def test_prepare_and_solve_lock_backend_identity() -> None:
    backend = FakeBackend()
    problem = Problem("heat", dummy_mesh(), DummyPhysics())

    prepared = prepare(problem, backend)
    solution = solve(prepared, backend)

    assert solution.backend_name == "fake"
    assert backend.prepare_calls == 1
    assert backend.solve_calls == 1


def test_solve_rejects_a_different_backend_descriptor() -> None:
    backend = FakeBackend()
    problem = Problem("heat", dummy_mesh(), DummyPhysics())
    prepared = PreparedProblem(BackendDescriptor("other", "1"), problem, payload=None)

    with pytest.raises(BackendError, match="does not match"):
        solve(prepared, backend)


def test_registry_rejects_invalid_names_and_mislabeled_factories() -> None:
    with pytest.raises(ValueError, match="name and version"):
        BackendDescriptor("", "1")
    registry = BackendRegistry()
    with pytest.raises(ContractError, match="trimmed"):
        registry.register(" bad", FakeBackend)
    registry.register("alias", FakeBackend)
    with pytest.raises(ContractError, match="returned 'fake'"):
        registry.create("alias")


def test_runtime_rejects_backend_rewriting_and_parameter_mismatch() -> None:
    problem = Problem("heat", dummy_mesh(), DummyPhysics())

    class WrongPreparedBackend(FakeBackend):
        def prepare(self, problem, request):
            return PreparedProblem(BackendDescriptor("wrong", "1"), problem, None)

    with pytest.raises(BackendError, match="different backend descriptor"):
        prepare(problem, WrongPreparedBackend())

    class ReplacingBackend(FakeBackend):
        def prepare(self, problem, request):
            replacement = Problem("replacement", dummy_mesh(), DummyPhysics())
            return PreparedProblem(self.descriptor, replacement, None)

    with pytest.raises(BackendError, match="replaced"):
        prepare(problem, ReplacingBackend())

    prepared = prepare(problem, FakeBackend())
    problem_with_parameter = Problem(
        "parameterized",
        dummy_mesh(),
        DummyPhysics(),
        parameters=ParameterSchema((ParameterSpec("x"),)),
    )
    prepared_with_parameter = prepare(problem_with_parameter, FakeBackend())
    with pytest.raises(ContractError, match="parameter keys"):
        solve(prepared_with_parameter, FakeBackend())
    assert prepared.problem is problem


def test_runtime_rejects_mislabeled_solution() -> None:
    class WrongSolutionBackend(FakeBackend):
        def solve(self, prepared, request):
            return Solution(
                "wrong",
                "1",
                {},
                {},
                ConvergenceReport(ConvergenceStatus.CONVERGED),
            )

    backend = WrongSolutionBackend()
    prepared = prepare(Problem("heat", dummy_mesh(), DummyPhysics()), backend)
    with pytest.raises(BackendError, match="solution backend identity"):
        solve(prepared, backend)
