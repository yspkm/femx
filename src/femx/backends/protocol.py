"""Backend lifecycle and execution-policy contracts."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from femx.core.capabilities import CapabilitySet
from femx.core.execution import ExecutionPolicy as ExecutionPolicy
from femx.core.parameters import ParameterValues
from femx.core.problem import Problem
from femx.core.solution import Solution


@dataclass(frozen=True, slots=True)
class BackendDescriptor:
    """Stable identity for one backend implementation."""

    name: str
    version: str

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ValueError("backend name and version must be non-empty")


@dataclass(frozen=True, slots=True)
class PrepareRequest:
    """Inputs controlling deterministic backend lowering."""

    run_directory: Path | None = None
    policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)


@dataclass(frozen=True, slots=True)
class SolveRequest:
    """Inputs controlling one solve of a prepared problem."""

    parameters: ParameterValues = field(default_factory=ParameterValues)
    run_directory: Path | None = None
    policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)


@dataclass(frozen=True, slots=True)
class PreparedProblem:
    """Opaque backend lowering tied to an exact backend identity."""

    backend: BackendDescriptor
    problem: Problem
    payload: object


@runtime_checkable
class Backend(Protocol):
    """Required lifecycle for every femx backend."""

    @property
    def descriptor(self) -> BackendDescriptor:
        """Exact backend identity used for compatibility checks."""

    @property
    def capabilities(self) -> CapabilitySet:
        """Capabilities supported by this implementation and version."""

    def prepare(self, problem: Problem, request: PrepareRequest) -> PreparedProblem:
        """Lower a backend-neutral problem without solving it."""

    def solve(self, prepared: PreparedProblem, request: SolveRequest) -> Solution:
        """Solve an already lowered problem."""
