from __future__ import annotations

from dataclasses import replace

import pytest

from femx.core.axes import Axis, AxisDirection, Direction
from femx.core.capabilities import (
    AnalysisKind,
    FunctionSpaceFamily,
    GradientMethod,
    ParallelModel,
    ScalarKind,
)
from femx.core.errors import ContractError
from femx.core.parameters import ParameterReference
from femx.physics import (
    VACUUM_PERMEABILITY_H_PER_M,
    VACUUM_PERMITTIVITY_F_PER_M,
    VACUUM_SPEED_OF_LIGHT_M_PER_S,
    IsotropicOpticalRegion,
    PerfectElectricBoundary,
    PortEigenmode,
)

pytestmark = pytest.mark.unit


def _physics() -> PortEigenmode:
    return PortEigenmode(
        regions=(
            IsotropicOpticalRegion("cladding", 2.1),
            IsotropicOpticalRegion("core", 12.1, relative_permeability=1.01),
        ),
        perfect_electric_boundaries=(PerfectElectricBoundary("outer"),),
        frequency_hz=VACUUM_SPEED_OF_LIGHT_M_PER_S / 1.55e-6,
        eigenmode_count=12,
        selected_mode_index=2,
        target_power_w=0.5,
    )


def test_port_eigenmode_exposes_exact_solver_neutral_contract() -> None:
    physics = _physics()
    physics.validate()

    assert physics.kind == "port_eigenmode_mixed_hcurl_h1_2d"
    assert physics.vacuum_wavelength_m == pytest.approx(1.55e-6)
    assert physics.requirements.analysis is AnalysisKind.EIGENMODE
    assert physics.requirements.function_spaces == frozenset(
        {FunctionSpaceFamily.HCURL, FunctionSpaceFamily.H1}
    )
    assert physics.requirements.scalar_kind is ScalarKind.COMPLEX
    assert physics.requirements.gradient is GradientMethod.NONE
    assert physics.requirements.parallel is ParallelModel.SERIAL

    data = physics.canonical_data()
    assert data["mode_ordering"] == "decreasing_real_propagation_constant"
    assert data["coordinate_system"] == "cartesian_xy"
    assert data["target_forward_power_W"] == 0.5
    assert data["regions"] == [
        {
            "tag": "cladding",
            "relative_permittivity": 2.1,
            "relative_permeability": 1.0,
        },
        {
            "tag": "core",
            "relative_permittivity": 12.1,
            "relative_permeability": 1.01,
        },
    ]
    constants = data["vacuum_constants"]
    assert isinstance(constants, dict)
    assert constants == {
        "speed_of_light_m_per_s": VACUUM_SPEED_OF_LIGHT_M_PER_S,
        "permittivity_F_per_m": VACUUM_PERMITTIVITY_F_PER_M,
        "permeability_H_per_m": VACUUM_PERMEABILITY_H_PER_M,
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"regions": ()}, "at least one optical region"),
        ({"perfect_electric_boundaries": ()}, "complete PEC"),
        (
            {
                "regions": (
                    IsotropicOpticalRegion("core", 2.0),
                    IsotropicOpticalRegion("core", 3.0),
                )
            },
            "unique",
        ),
        (
            {
                "perfect_electric_boundaries": (
                    PerfectElectricBoundary("outer"),
                    PerfectElectricBoundary("outer"),
                )
            },
            "unique",
        ),
        ({"frequency_hz": 0.0}, "frequency"),
        ({"frequency_hz": float("inf")}, "frequency"),
        ({"eigenmode_count": True}, "positive integer"),
        ({"eigenmode_count": 2.5}, "positive integer"),
        ({"eigenmode_count": 0}, "positive integer"),
        ({"eigenmode_count": 257}, "limit"),
        ({"selected_mode_index": True}, "within"),
        ({"selected_mode_index": 0.5}, "within"),
        ({"selected_mode_index": -1}, "within"),
        ({"selected_mode_index": 12}, "within"),
        ({"target_power_w": 0.0}, "target power"),
        ({"target_power_w": float("nan")}, "target power"),
        (
            {"propagation": AxisDirection(Axis.X, Direction.POSITIVE)},
            "positive-z",
        ),
    ],
)
def test_port_eigenmode_rejects_ambiguous_or_unsupported_contracts(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ContractError, match=message):
        replace(_physics(), **kwargs).validate()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tag": " bad"}, "tag"),
        ({"relative_permittivity": True}, "real scalar"),
        ({"relative_permittivity": "12"}, "real scalar"),
        ({"relative_permittivity": 0.0}, "finite and positive"),
        ({"relative_permeability": float("nan")}, "finite and positive"),
    ],
)
def test_isotropic_optical_region_rejects_invalid_material_values(
    kwargs: dict[str, object], message: str
) -> None:
    defaults: dict[str, object] = {"tag": "core", "relative_permittivity": 12.0}
    with pytest.raises(ContractError, match=message):
        IsotropicOpticalRegion(**(defaults | kwargs))  # type: ignore[arg-type]


def test_perfect_electric_boundary_requires_a_stable_tag() -> None:
    with pytest.raises(ContractError, match="tag"):
        PerfectElectricBoundary("")


def test_optical_region_preserves_material_parameter_references() -> None:
    physics = replace(
        _physics(),
        regions=(
            IsotropicOpticalRegion(
                "core",
                ParameterReference("epsilon_r"),
                ParameterReference("mu_r"),
            ),
        ),
        gradient_method=GradientMethod.ADJOINT,
    )

    physics.validate()

    assert physics.requirements.gradient is GradientMethod.ADJOINT
    assert physics.canonical_data()["regions"] == [
        {
            "tag": "core",
            "relative_permittivity": {"parameter": "epsilon_r"},
            "relative_permeability": {"parameter": "mu_r"},
        }
    ]
