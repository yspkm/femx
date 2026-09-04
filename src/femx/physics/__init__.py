"""Solver-neutral equation specifications."""

from femx.physics._scalar import ScalarCoefficient
from femx.physics.port_eigenmode import (
    PORT_LONGITUDINAL_POTENTIAL_FIELD,
    PORT_LONGITUDINAL_POTENTIAL_UNIT,
    PORT_TRANSVERSE_ELECTRIC_DOF_UNIT,
    PORT_TRANSVERSE_ELECTRIC_FIELD,
    VACUUM_PERMEABILITY_H_PER_M,
    VACUUM_PERMITTIVITY_F_PER_M,
    VACUUM_SPEED_OF_LIGHT_M_PER_S,
    IsotropicOpticalRegion,
    PerfectElectricBoundary,
    PortEigenmode,
)
from femx.physics.steady_current import (
    ConductiveRegion,
    CurrentFluxBoundary,
    PotentialBoundary,
    SteadyCurrent,
)
from femx.physics.steady_heat import (
    HeatFluxBoundary,
    SteadyHeat,
    TemperatureBoundary,
    ThermalRegion,
)

__all__ = [
    "PORT_LONGITUDINAL_POTENTIAL_FIELD",
    "PORT_LONGITUDINAL_POTENTIAL_UNIT",
    "PORT_TRANSVERSE_ELECTRIC_DOF_UNIT",
    "PORT_TRANSVERSE_ELECTRIC_FIELD",
    "VACUUM_PERMEABILITY_H_PER_M",
    "VACUUM_PERMITTIVITY_F_PER_M",
    "VACUUM_SPEED_OF_LIGHT_M_PER_S",
    "ConductiveRegion",
    "CurrentFluxBoundary",
    "HeatFluxBoundary",
    "IsotropicOpticalRegion",
    "PerfectElectricBoundary",
    "PortEigenmode",
    "PotentialBoundary",
    "ScalarCoefficient",
    "SteadyCurrent",
    "SteadyHeat",
    "TemperatureBoundary",
    "ThermalRegion",
]
