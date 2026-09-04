"""Small structural array protocol that does not import NumPy or JAX."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class ArrayLike(Protocol):
    """The array metadata femx contracts need without owning an array library."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Global logical shape."""

    @property
    def ndim(self) -> int:
        """Number of logical dimensions."""

    @property
    def dtype(self) -> object:
        """Scalar dtype descriptor."""


def shape_of(value: ArrayLike) -> tuple[int, ...]:
    """Return a normalized shape and reject negative dimensions."""

    shape = tuple(int(size) for size in value.shape)
    if any(size < 0 for size in shape):
        raise ValueError(f"array shape must be non-negative, got {shape}")
    if value.ndim != len(shape):
        raise ValueError(f"array ndim={value.ndim} does not match shape {shape}")
    return shape
