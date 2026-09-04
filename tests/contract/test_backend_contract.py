import pytest
from tests.support import DummyPhysics, FakeBackend, dummy_mesh

from femx.backends.protocol import Backend
from femx.core.problem import Problem
from femx.runtime import prepare, solve

pytestmark = pytest.mark.contract


def test_backend_protocol_and_lifecycle_contract() -> None:
    backend = FakeBackend()
    assert isinstance(backend, Backend)

    problem = Problem("contract-heat", dummy_mesh(), DummyPhysics())
    prepared = prepare(problem, backend)
    solution = solve(prepared, backend)

    assert prepared.problem is problem
    assert solution.convergence.status.value == "converged"
