"""Coordinate-axis and propagation-direction contracts."""

from dataclasses import dataclass
from enum import StrEnum


class Axis(StrEnum):
    """A right-handed Cartesian axis."""

    X = "x"
    Y = "y"
    Z = "z"


class Direction(StrEnum):
    """Positive or negative orientation along an axis."""

    POSITIVE = "+"
    NEGATIVE = "-"

    @property
    def sign(self) -> int:
        """Return the algebraic sign associated with the direction."""

        return 1 if self is Direction.POSITIVE else -1


@dataclass(frozen=True, slots=True)
class AxisDirection:
    """An oriented Cartesian axis used by ports and mode handoffs."""

    axis: Axis
    direction: Direction = Direction.POSITIVE
