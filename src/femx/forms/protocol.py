"""Backend-neutral weak-form metadata protocol."""

from collections.abc import Mapping
from enum import StrEnum
from typing import Protocol, runtime_checkable

from femx.mesh import FunctionSpace


class FormKind(StrEnum):
    """Algebraic role of a weak form."""

    BILINEAR = "bilinear"
    LINEAR = "linear"
    RESIDUAL = "residual"
    FUNCTIONAL = "functional"


@runtime_checkable
class WeakForm(Protocol):
    """Small lowering contract for registered equation formulations."""

    @property
    def name(self) -> str:
        """Stable formulation name."""

    @property
    def kind(self) -> FormKind:
        """Algebraic form role."""

    @property
    def spaces(self) -> tuple[FunctionSpace, ...]:
        """Ordered trial/test/field spaces used by this form."""

    def canonical_data(self) -> Mapping[str, object]:
        """Return deterministic backend-neutral coefficient metadata."""
