import math

import pytest

from femx.core.capabilities import GradientMethod
from femx.core.errors import ContractError
from femx.core.parameters import ParameterReference
from femx.physics import (
    HeatFluxBoundary,
    SteadyHeat,
    TemperatureBoundary,
    ThermalRegion,
)

pytestmark = pytest.mark.unit


def test_steady_heat_exposes_elmer_compatible_weak_form_metadata() -> None:
    physics = SteadyHeat(
        regions=(
            ThermalRegion(
                "silicon",
                ParameterReference("silicon_k"),
                volumetric_heat_source=2.0,
            ),
        ),
        temperature_boundaries=(TemperatureBoundary("substrate", 300.0),),
        heat_flux_boundaries=(HeatFluxBoundary("heater", ParameterReference("heater_flux")),),
        gradient_method=GradientMethod.NONE,
    )

    physics.validate()
    canonical = physics.canonical_data()

    assert physics.kind == "steady_heat_h1_2d"
    assert physics.requirements.gradient is GradientMethod.NONE
    assert canonical["weak_form_flux_sign"] == "positive_rhs"
    assert canonical["out_of_plane_convention"] == "per_unit_depth"
    assert canonical["regions"] == [
        {
            "tag": "silicon",
            "conductivity_W_per_mK": {"parameter": "silicon_k"},
            "volumetric_heat_source_W_per_m3": 2.0,
        }
    ]


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ParameterReference(" bad"), "reference name"),
        (lambda: ThermalRegion("", 1.0), "region tag"),
        (lambda: ThermalRegion("domain", 0.0), "strictly positive"),
        (lambda: ThermalRegion("domain", True), "real scalar"),
        (lambda: ThermalRegion("domain", math.inf), "finite"),
        (lambda: ThermalRegion("domain", 1.0, math.nan), "finite"),
        (lambda: TemperatureBoundary(" bad", 300.0), "boundary tag"),
        (lambda: TemperatureBoundary("wall", math.inf), "finite"),
        (lambda: HeatFluxBoundary("", 1.0), "boundary tag"),
        (lambda: HeatFluxBoundary("heater", math.nan), "finite"),
    ],
)
def test_steady_heat_coefficients_reject_ambiguous_values(factory, message: str) -> None:
    with pytest.raises(ContractError, match=message):
        factory()


@pytest.mark.parametrize(
    ("physics", "message"),
    [
        (SteadyHeat((), (TemperatureBoundary("wall", 300.0),)), "thermal region"),
        (SteadyHeat((ThermalRegion("domain", 1.0),), ()), "temperature boundary"),
        (
            SteadyHeat(
                (ThermalRegion("domain", 1.0), ThermalRegion("domain", 2.0)),
                (TemperatureBoundary("wall", 300.0),),
            ),
            "region tags",
        ),
        (
            SteadyHeat(
                (ThermalRegion("domain", 1.0),),
                (TemperatureBoundary("wall", 300.0), TemperatureBoundary("wall", 300.0)),
            ),
            "temperature boundary tags",
        ),
        (
            SteadyHeat(
                (ThermalRegion("domain", 1.0),),
                (TemperatureBoundary("wall", 300.0),),
                (HeatFluxBoundary("heater", 1.0), HeatFluxBoundary("heater", 2.0)),
            ),
            "heat-flux boundary tags",
        ),
        (
            SteadyHeat(
                (ThermalRegion("domain", 1.0),),
                (TemperatureBoundary("wall", 300.0),),
                (HeatFluxBoundary("wall", 1.0),),
            ),
            "both temperature and heat flux",
        ),
    ],
)
def test_steady_heat_rejects_singular_or_ambiguous_declarations(
    physics: SteadyHeat, message: str
) -> None:
    with pytest.raises(ContractError, match=message):
        physics.validate()
