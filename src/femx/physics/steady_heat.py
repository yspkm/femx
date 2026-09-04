"""Steady scalar heat-conduction contracts shared by all backends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from femx.core.capabilities import (
    AnalysisKind,
    CapabilityRequest,
    FunctionSpaceFamily,
    GradientMethod,
    ParallelModel,
    ScalarKind,
)
from femx.core.errors import ContractError
from femx.physics._scalar import (
    ScalarCoefficient,
    coefficient_data,
    require_unique_tags,
    validate_coefficient,
    validate_name,
)


@dataclass(frozen=True, slots=True)
class ThermalRegion:
    """Piecewise-constant conductivity and volumetric source on a cell tag."""

    tag: str
    conductivity: ScalarCoefficient
    volumetric_heat_source: ScalarCoefficient = 0.0

    def __post_init__(self) -> None:
        validate_name(self.tag, label="thermal region tag")
        validate_coefficient(
            self.conductivity,
            label=f"thermal conductivity on {self.tag!r}",
            strictly_positive=True,
        )
        validate_coefficient(
            self.volumetric_heat_source,
            label=f"volumetric heat source on {self.tag!r}",
        )


@dataclass(frozen=True, slots=True)
class TemperatureBoundary:
    """Strong temperature constraint on every node of a boundary-facet tag."""

    tag: str
    temperature: ScalarCoefficient

    def __post_init__(self) -> None:
        validate_name(self.tag, label="temperature boundary tag")
        validate_coefficient(self.temperature, label=f"temperature on {self.tag!r}")


@dataclass(frozen=True, slots=True)
class HeatFluxBoundary:
    r"""Elmer-compatible variational heat load on a boundary-facet tag.

    ``heat_flux`` is the scalar :math:`g` in the positive right-hand-side term
    :math:`\int_{\Gamma} v g\,d\Gamma`. With an outward normal and Fourier flux
    :math:`q=-k\nabla T`, the physical outward flux is :math:`q\cdot n=-g`.
    """

    tag: str
    heat_flux: ScalarCoefficient

    def __post_init__(self) -> None:
        validate_name(self.tag, label="heat-flux boundary tag")
        validate_coefficient(self.heat_flux, label=f"heat flux on {self.tag!r}")


@dataclass(frozen=True, slots=True)
class SteadyHeat:
    r"""Two-dimensional Cartesian H1 heat conduction per unit out-of-plane depth.

    The initial form is

    .. math::

       \int_\Omega \nabla v^T k\nabla T\,d\Omega
       = \int_\Omega vQ\,d\Omega + \int_{\Gamma_g}vg\,d\Gamma.

    Regions must partition all bulk cells. Boundary conditions refer to explicit boundary-facet
    tags, and at least one strong temperature boundary is required to remove the constant null
    space.
    """

    regions: tuple[ThermalRegion, ...]
    temperature_boundaries: tuple[TemperatureBoundary, ...]
    heat_flux_boundaries: tuple[HeatFluxBoundary, ...] = ()
    gradient_method: GradientMethod = GradientMethod.NONE

    @property
    def kind(self) -> str:
        """Return the stable equation identifier."""

        return "steady_heat_h1_2d"

    @property
    def requirements(self) -> CapabilityRequest:
        """Return the exact analysis capabilities requested from a backend."""

        return CapabilityRequest(
            analysis=AnalysisKind.STEADY,
            function_spaces=frozenset({FunctionSpaceFamily.H1}),
            scalar_kind=ScalarKind.REAL,
            gradient=self.gradient_method,
            parallel=ParallelModel.SERIAL,
        )

    def validate(self) -> None:
        """Reject singular or ambiguous region and boundary declarations."""

        if not self.regions:
            raise ContractError("steady heat requires at least one thermal region")
        if not self.temperature_boundaries:
            raise ContractError("steady heat requires at least one temperature boundary")
        require_unique_tags(self.regions, label="thermal region")
        require_unique_tags(self.temperature_boundaries, label="temperature boundary")
        require_unique_tags(self.heat_flux_boundaries, label="heat-flux boundary")
        temperature_tags = {boundary.tag for boundary in self.temperature_boundaries}
        flux_tags = {boundary.tag for boundary in self.heat_flux_boundaries}
        overlap = sorted(temperature_tags & flux_tags)
        if overlap:
            raise ContractError(
                f"boundary tags cannot carry both temperature and heat flux: {overlap}"
            )

    def canonical_data(self) -> Mapping[str, object]:
        """Return deterministic backend-neutral metadata for hashing and provenance."""

        return {
            "kind": self.kind,
            "dimension": 2,
            "coordinate_system": "cartesian",
            "out_of_plane_convention": "per_unit_depth",
            "weak_form_flux_sign": "positive_rhs",
            "gradient_method": self.gradient_method.value,
            "regions": [
                {
                    "tag": region.tag,
                    "conductivity_W_per_mK": coefficient_data(region.conductivity),
                    "volumetric_heat_source_W_per_m3": coefficient_data(
                        region.volumetric_heat_source
                    ),
                }
                for region in self.regions
            ],
            "temperature_boundaries": [
                {
                    "tag": boundary.tag,
                    "temperature_K": coefficient_data(boundary.temperature),
                }
                for boundary in self.temperature_boundaries
            ],
            "heat_flux_boundaries": [
                {
                    "tag": boundary.tag,
                    "positive_rhs_heat_load_W_per_m2": coefficient_data(boundary.heat_flux),
                }
                for boundary in self.heat_flux_boundaries
            ],
        }
