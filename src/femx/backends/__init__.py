"""Backend protocols and explicit registration."""

from femx.backends.protocol import (
    Backend,
    BackendDescriptor,
    PreparedProblem,
    PrepareRequest,
    SolveRequest,
)
from femx.backends.registry import BackendRegistry
from femx.core.execution import ExecutionPolicy

__all__ = [
    "Backend",
    "BackendDescriptor",
    "BackendRegistry",
    "ExecutionPolicy",
    "PrepareRequest",
    "PreparedProblem",
    "SolveRequest",
]
