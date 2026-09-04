"""Backend-neutral in-memory solution contracts."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from femx.core.arrays import ArrayLike
from femx.core.errors import ContractError


class ConvergenceStatus(StrEnum):
    """Numerical convergence state, separate from process execution state."""

    CONVERGED = "converged"
    NOT_CONVERGED = "not_converged"
    NOT_EVALUATED = "not_evaluated"


@dataclass(frozen=True, slots=True)
class ConvergenceReport:
    """A backend's explicit numerical convergence report."""

    status: ConvergenceStatus
    iterations: int | None = None
    residual_norm: float | None = None
    tolerance: float | None = None
    message: str = ""

    def __post_init__(self) -> None:
        if self.iterations is not None and self.iterations < 0:
            raise ContractError("iteration count cannot be negative")
        if self.residual_norm is not None and self.residual_norm < 0:
            raise ContractError("residual norm cannot be negative")
        if self.tolerance is not None and self.tolerance <= 0:
            raise ContractError("convergence tolerance must be positive")


@dataclass(frozen=True, slots=True)
class Field:
    """A numerical field plus the metadata needed to interpret it."""

    name: str
    values: ArrayLike
    unit: str
    function_space: object

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ContractError("field name must be non-empty and trimmed")
        if not self.unit:
            raise ContractError(f"field {self.name!r} must declare a unit")
        if self.function_space is None:
            raise ContractError(f"field {self.name!r} must declare a function space")


ObservableValue = int | float | complex


@dataclass(frozen=True, slots=True)
class Solution:
    """A backend-neutral solution envelope.

    This type records convergence but intentionally does not claim scientific validity. A
    ``ValidationReport`` is required for such a claim.
    """

    backend_name: str
    backend_version: str
    fields: Mapping[str, Field]
    observables: Mapping[str, ObservableValue]
    convergence: ConvergenceReport
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.backend_name:
            raise ContractError("solution backend_name cannot be empty")
        if not self.backend_version:
            raise ContractError("solution backend_version cannot be empty")
        fields = dict(self.fields)
        for name, field_value in fields.items():
            if name != field_value.name:
                raise ContractError(
                    f"field mapping key {name!r} does not match field name {field_value.name!r}"
                )
        object.__setattr__(self, "fields", MappingProxyType(fields))
        object.__setattr__(self, "observables", MappingProxyType(dict(self.observables)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
