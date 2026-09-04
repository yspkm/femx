"""Public backend-neutral prepare and solve lifecycle."""

from femx.backends.protocol import Backend, PreparedProblem, PrepareRequest, SolveRequest
from femx.core.errors import BackendError, ContractError
from femx.core.problem import Problem
from femx.core.solution import Solution


def prepare(
    problem: Problem,
    backend: Backend,
    *,
    request: PrepareRequest | None = None,
) -> PreparedProblem:
    """Validate capabilities and lower ``problem`` for exactly ``backend``."""

    actual_request = request if request is not None else PrepareRequest()
    backend.capabilities.require(problem.requirements, backend_name=backend.descriptor.name)
    prepared = backend.prepare(problem, actual_request)
    if prepared.backend != backend.descriptor:
        raise BackendError(
            "backend returned a prepared problem with a different backend descriptor"
        )
    if prepared.problem is not problem:
        raise BackendError("backend replaced the solver-neutral problem during preparation")
    return prepared


def solve(
    prepared: PreparedProblem,
    backend: Backend,
    *,
    request: SolveRequest | None = None,
) -> Solution:
    """Run a prepared problem without substituting another backend."""

    if prepared.backend != backend.descriptor:
        raise BackendError(
            f"prepared backend {prepared.backend.name!r} does not match "
            f"solver backend {backend.descriptor.name!r}"
        )
    actual_request = request if request is not None else SolveRequest()
    expected_parameters = set(prepared.problem.parameters.names)
    actual_parameters = set(actual_request.parameters.values)
    if expected_parameters != actual_parameters:
        raise ContractError(
            "solve parameter keys do not match the prepared problem: "
            f"expected={sorted(expected_parameters)}, actual={sorted(actual_parameters)}"
        )
    prepared.problem.parameters.bind(actual_request.parameters.values)
    solution = backend.solve(prepared, actual_request)
    if (
        solution.backend_name != backend.descriptor.name
        or solution.backend_version != backend.descriptor.version
    ):
        raise BackendError("solution backend identity does not match the executing backend")
    return solution
