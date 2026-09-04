"""Typed parameter schemas that replace unstructured nested dictionaries."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from femx.core.arrays import ArrayLike, shape_of
from femx.core.errors import ContractError

ParameterValue = int | float | complex | ArrayLike


@dataclass(frozen=True, slots=True)
class ParameterReference:
    """A typed physics coefficient resolved from a problem parameter at solve time."""

    name: str

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ContractError("parameter reference name must be non-empty and trimmed")


class ParameterRole(StrEnum):
    """How a parameter participates in a solve or optimization."""

    FIXED = "fixed"
    DESIGN = "design"
    CONTROL = "control"


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """A named, unit-bearing parameter contract."""

    name: str
    unit: str = "1"
    shape: tuple[int, ...] = ()
    role: ParameterRole = ParameterRole.FIXED
    lower_bound: float | None = None
    upper_bound: float | None = None

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ContractError("parameter name must be non-empty and have no surrounding space")
        if not self.unit:
            raise ContractError(f"parameter {self.name!r} must declare a unit")
        if any(size <= 0 for size in self.shape):
            raise ContractError(f"parameter {self.name!r} has invalid shape {self.shape}")
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ContractError(f"parameter {self.name!r} has reversed bounds")

    def validate_value(self, value: ParameterValue) -> None:
        """Validate shape and scalar bounds without importing an array package."""

        if self.shape:
            if not isinstance(value, ArrayLike):
                raise ContractError(f"parameter {self.name!r} requires array shape {self.shape}")
            actual_shape = shape_of(value)
            if actual_shape != self.shape:
                raise ContractError(
                    f"parameter {self.name!r} expected shape {self.shape}, got {actual_shape}"
                )
            return

        if isinstance(value, ArrayLike):
            raise ContractError(f"scalar parameter {self.name!r} cannot receive an array")
        if isinstance(value, bool):
            raise ContractError(f"parameter {self.name!r} must be numeric, not boolean")
        if isinstance(value, complex):
            if not math.isfinite(value.real) or not math.isfinite(value.imag):
                raise ContractError(f"parameter {self.name!r} must be finite")
            if self.lower_bound is not None or self.upper_bound is not None:
                raise ContractError(f"complex parameter {self.name!r} cannot use ordered bounds")
            return
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ContractError(f"parameter {self.name!r} must be finite")
        if self.lower_bound is not None and numeric < self.lower_bound:
            raise ContractError(f"parameter {self.name!r} is below its lower bound")
        if self.upper_bound is not None and numeric > self.upper_bound:
            raise ContractError(f"parameter {self.name!r} is above its upper bound")


@dataclass(frozen=True, slots=True)
class ParameterSchema:
    """An ordered set of unique parameter specifications."""

    specs: tuple[ParameterSpec, ...] = ()

    def __post_init__(self) -> None:
        names = tuple(spec.name for spec in self.specs)
        if len(names) != len(set(names)):
            raise ContractError("parameter names must be unique")

    @property
    def names(self) -> tuple[str, ...]:
        """Return schema names in canonical order."""

        return tuple(spec.name for spec in self.specs)

    def bind(self, values: Mapping[str, ParameterValue]) -> ParameterValues:
        """Validate an exact set of values and return an immutable mapping wrapper."""

        expected = set(self.names)
        actual = set(values)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            raise ContractError(
                f"parameter key mismatch: missing={missing}, unexpected={unexpected}"
            )
        by_name = {spec.name: spec for spec in self.specs}
        for name, value in values.items():
            by_name[name].validate_value(value)
        return ParameterValues(values)


@dataclass(frozen=True, slots=True)
class ParameterValues:
    """A read-only top-level parameter mapping."""

    values: Mapping[str, ParameterValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))

    def __getitem__(self, name: str) -> ParameterValue:
        return self.values[name]
