"""Shared scalar-coefficient validation for solver-neutral physics contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Protocol, TypeAlias

from femx.core.errors import ContractError
from femx.core.parameters import ParameterReference

ScalarCoefficient: TypeAlias = float | ParameterReference


class Tagged(Protocol):
    """Structural contract for declarations owned by one mesh tag."""

    @property
    def tag(self) -> str: ...


def validate_name(value: str, *, label: str) -> None:
    """Require one stable, trimmed semantic name."""

    if not value or value.strip() != value:
        raise ContractError(f"{label} must be non-empty and trimmed")


def validate_coefficient(
    value: ScalarCoefficient,
    *,
    label: str,
    strictly_positive: bool = False,
) -> None:
    """Validate a literal real coefficient while preserving parameter references."""

    if isinstance(value, ParameterReference):
        return
    if isinstance(value, bool):
        raise ContractError(f"{label} must be a real scalar coefficient")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ContractError(f"{label} must be finite")
    if strictly_positive and numeric <= 0.0:
        raise ContractError(f"{label} must be strictly positive")


def coefficient_data(value: ScalarCoefficient) -> float | Mapping[str, str]:
    """Return deterministic literal or parameter-reference metadata."""

    if isinstance(value, ParameterReference):
        return {"parameter": value.name}
    return float(value)


def require_unique_tags(values: Sequence[Tagged], *, label: str) -> None:
    """Reject duplicate semantic ownership within one declaration family."""

    tags = tuple(value.tag for value in values)
    if len(tags) != len(set(tags)):
        raise ContractError(f"{label} tags must be unique")
