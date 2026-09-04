import math

import pytest

from femx.core.capabilities import GradientMethod
from femx.core.errors import ContractError
from femx.core.parameters import ParameterReference
from femx.physics import (
    ConductiveRegion,
    CurrentFluxBoundary,
    PotentialBoundary,
    SteadyCurrent,
)

pytestmark = pytest.mark.unit


def test_steady_current_exposes_elmer_compatible_weak_form_metadata() -> None:
    physics = SteadyCurrent(
        regions=(
            ConductiveRegion(
                "heater",
                ParameterReference("sigma"),
                volumetric_current_source=2.0,
            ),
        ),
        potential_boundaries=(PotentialBoundary("electrode", 1.5),),
        current_flux_boundaries=(
            CurrentFluxBoundary("terminal", ParameterReference("terminal_current")),
        ),
        gradient_method=GradientMethod.ADJOINT,
    )

    physics.validate()
    canonical = physics.canonical_data()

    assert physics.kind == "steady_current_h1_2d"
    assert physics.requirements.gradient is GradientMethod.ADJOINT
    assert canonical["weak_form_flux_sign"] == "positive_rhs"
    assert canonical["physical_current_density"] == "J=-sigma*grad(phi)"
    assert canonical["out_of_plane_convention"] == "per_unit_depth"
    assert canonical["regions"] == [
        {
            "tag": "heater",
            "electric_conductivity_S_per_m": {"parameter": "sigma"},
            "volumetric_current_source_A_per_m3": 2.0,
        }
    ]
    assert canonical["potential_boundaries"] == [{"tag": "electrode", "potential_V": 1.5}]
    assert canonical["current_flux_boundaries"] == [
        {
            "tag": "terminal",
            "positive_rhs_current_density_A_per_m2": {"parameter": "terminal_current"},
        }
    ]


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ConductiveRegion("", 1.0), "region tag"),
        (lambda: ConductiveRegion("domain", 0.0), "strictly positive"),
        (lambda: ConductiveRegion("domain", True), "real scalar"),
        (lambda: ConductiveRegion("domain", math.inf), "finite"),
        (lambda: ConductiveRegion("domain", 1.0, math.nan), "finite"),
        (lambda: PotentialBoundary(" bad", 0.0), "boundary tag"),
        (lambda: PotentialBoundary("electrode", math.inf), "finite"),
        (lambda: CurrentFluxBoundary("", 1.0), "boundary tag"),
        (lambda: CurrentFluxBoundary("terminal", math.nan), "finite"),
    ],
)
def test_steady_current_coefficients_reject_ambiguous_values(factory, message: str) -> None:
    with pytest.raises(ContractError, match=message):
        factory()


@pytest.mark.parametrize(
    ("physics", "message"),
    [
        (SteadyCurrent((), (PotentialBoundary("electrode", 0.0),)), "conductive region"),
        (SteadyCurrent((ConductiveRegion("domain", 1.0),), ()), "potential boundary"),
        (
            SteadyCurrent(
                (ConductiveRegion("domain", 1.0), ConductiveRegion("domain", 2.0)),
                (PotentialBoundary("electrode", 0.0),),
            ),
            "region tags",
        ),
        (
            SteadyCurrent(
                (ConductiveRegion("domain", 1.0),),
                (PotentialBoundary("electrode", 0.0), PotentialBoundary("electrode", 1.0)),
            ),
            "potential boundary tags",
        ),
        (
            SteadyCurrent(
                (ConductiveRegion("domain", 1.0),),
                (PotentialBoundary("electrode", 0.0),),
                (CurrentFluxBoundary("terminal", 1.0), CurrentFluxBoundary("terminal", 2.0)),
            ),
            "current-flux boundary tags",
        ),
        (
            SteadyCurrent(
                (ConductiveRegion("domain", 1.0),),
                (PotentialBoundary("electrode", 0.0),),
                (CurrentFluxBoundary("electrode", 1.0),),
            ),
            "both potential and current flux",
        ),
    ],
)
def test_steady_current_rejects_singular_or_ambiguous_declarations(
    physics: SteadyCurrent,
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        physics.validate()
