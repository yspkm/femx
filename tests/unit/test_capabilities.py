import pytest

from femx.core.axes import Axis, AxisDirection, Direction
from femx.core.capabilities import (
    AnalysisKind,
    CapabilityRequest,
    CapabilitySet,
    FunctionSpaceFamily,
    GradientMethod,
    ParallelModel,
    ScalarKind,
)
from femx.core.errors import CapabilityError

pytestmark = pytest.mark.unit


def test_capability_set_accepts_exact_request() -> None:
    request = CapabilityRequest(
        analysis=AnalysisKind.HARMONIC,
        function_spaces=frozenset({FunctionSpaceFamily.HCURL}),
        scalar_kind=ScalarKind.COMPLEX,
        gradient=GradientMethod.ADJOINT,
        parallel=ParallelModel.MPI,
    )
    capabilities = CapabilitySet(
        analyses=frozenset({AnalysisKind.HARMONIC}),
        function_spaces=frozenset({FunctionSpaceFamily.HCURL}),
        scalar_kinds=frozenset({ScalarKind.COMPLEX}),
        gradients=frozenset({GradientMethod.ADJOINT}),
        parallel_models=frozenset({ParallelModel.MPI}),
    )

    assert capabilities.missing(request) == ()
    capabilities.require(request, backend_name="reference")


def test_capability_set_reports_every_missing_axis_deterministically() -> None:
    request = CapabilityRequest(
        analysis=AnalysisKind.EIGENMODE,
        function_spaces=frozenset({FunctionSpaceFamily.H1, FunctionSpaceFamily.HCURL}),
        scalar_kind=ScalarKind.COMPLEX,
        gradient=GradientMethod.IMPLICIT,
        parallel=ParallelModel.JAX_SPMD,
    )

    with pytest.raises(CapabilityError, match=r"analysis=eigenmode.*H1,Hcurl.*complex"):
        CapabilitySet().require(request, backend_name="empty")


def test_capability_request_requires_a_function_space() -> None:
    with pytest.raises(ValueError, match="at least one"):
        CapabilityRequest(analysis=AnalysisKind.STEADY, function_spaces=frozenset())


def test_axis_direction_sign_is_explicit() -> None:
    assert AxisDirection(Axis.X, Direction.POSITIVE).direction.sign == 1
    assert AxisDirection(Axis.X, Direction.NEGATIVE).direction.sign == -1
