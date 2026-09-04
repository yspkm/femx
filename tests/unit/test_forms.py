from collections.abc import Mapping
from dataclasses import dataclass

import pytest

from femx.core.capabilities import FunctionSpaceFamily
from femx.forms import FormKind, WeakForm
from femx.mesh import FunctionSpace

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class DummyForm:
    name: str = "poisson"
    kind: FormKind = FormKind.BILINEAR
    spaces: tuple[FunctionSpace, ...] = (FunctionSpace(FunctionSpaceFamily.H1, 1),)

    def canonical_data(self) -> Mapping[str, object]:
        return {"operator": "grad_dot_grad"}


def test_weak_form_protocol_is_structural() -> None:
    assert isinstance(DummyForm(), WeakForm)
