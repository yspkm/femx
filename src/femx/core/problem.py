"""Solver-neutral problem container and physics protocol."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from femx.core.capabilities import CapabilityRequest
from femx.core.errors import ContractError
from femx.core.parameters import ParameterSchema


@runtime_checkable
class PhysicsSpec(Protocol):
    """Backend-neutral equation metadata supplied by a concrete physics package."""

    @property
    def kind(self) -> str:
        """Stable physics identifier, such as ``steady_heat``."""

    @property
    def requirements(self) -> CapabilityRequest:
        """Capabilities required to lower and solve this specification."""

    def validate(self) -> None:
        """Raise when the specification is internally inconsistent."""

    def canonical_data(self) -> Mapping[str, object]:
        """Return deterministic, JSON-compatible semantic input metadata."""


@runtime_checkable
class MeshSpec(Protocol):
    """Minimum structural identity required from a validated mesh implementation."""

    @property
    def geometry(self) -> object:
        """Physical geometry contract."""

    @property
    def topology(self) -> object:
        """Cell topology contract."""

    @property
    def schema_version(self) -> str:
        """Versioned mesh schema identifier."""


@dataclass(frozen=True, slots=True)
class ObservableSpec:
    """A named quantity requested from a solution."""

    name: str
    unit: str
    reduction: str = "identity"

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ContractError("observable name must be non-empty and trimmed")
        if not self.unit:
            raise ContractError(f"observable {self.name!r} must declare a unit")
        if not self.reduction:
            raise ContractError(f"observable {self.name!r} must declare a reduction")


@dataclass(frozen=True, slots=True)
class Problem:
    """A backend-independent problem envelope.

    ``mesh`` is structural here. Concrete validation lives in ``femx.mesh``; the protocol also
    permits future lazy or distributed mesh handles without weakening the lifecycle contract.
    """

    name: str
    mesh: MeshSpec
    physics: PhysicsSpec
    parameters: ParameterSchema = field(default_factory=ParameterSchema)
    observables: tuple[ObservableSpec, ...] = ()
    schema_version: str = "femx.problem/v1"

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ContractError("problem name must be non-empty and trimmed")
        if not isinstance(self.mesh, MeshSpec):
            raise ContractError("mesh object does not implement MeshSpec")
        if not isinstance(self.physics, PhysicsSpec):
            raise ContractError("physics object does not implement PhysicsSpec")
        self.physics.validate()
        names = tuple(observable.name for observable in self.observables)
        if len(names) != len(set(names)):
            raise ContractError("observable names must be unique")
        if self.schema_version != "femx.problem/v1":
            raise ContractError(f"unsupported problem schema {self.schema_version!r}")

    @property
    def requirements(self) -> CapabilityRequest:
        """Return the capabilities required by the physics specification."""

        return self.physics.requirements
