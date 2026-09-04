"""Steady scalar electric-current contracts shared by all backends."""

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
class ConductiveRegion:
    """Piecewise-constant isotropic conductivity and current source on a cell tag."""

    tag: str
    electric_conductivity: ScalarCoefficient
    volumetric_current_source: ScalarCoefficient = 0.0

    def __post_init__(self) -> None:
        validate_name(self.tag, label="conductive region tag")
        validate_coefficient(
            self.electric_conductivity,
            label=f"electric conductivity on {self.tag!r}",
            strictly_positive=True,
        )
        validate_coefficient(
            self.volumetric_current_source,
            label=f"volumetric current source on {self.tag!r}",
        )


@dataclass(frozen=True, slots=True)
class PotentialBoundary:
    """Strong electric-potential constraint on every node of a boundary-facet tag."""

    tag: str
    potential: ScalarCoefficient

    def __post_init__(self) -> None:
        validate_name(self.tag, label="potential boundary tag")
        validate_coefficient(self.potential, label=f"potential on {self.tag!r}")


@dataclass(frozen=True, slots=True)
class CurrentFluxBoundary:
    r"""Elmer-compatible variational current load on a boundary-facet tag.

    ``current_density`` is :math:`g` in :math:`\int_\Gamma vg\,d\Gamma`, equivalently
    :math:`\sigma\nabla\phi\cdot n=g`. Since physical current density is
    :math:`J=-\sigma\nabla\phi`, positive physical outward current is :math:`J\cdot n=-g`.
    """

    tag: str
    current_density: ScalarCoefficient

    def __post_init__(self) -> None:
        validate_name(self.tag, label="current-flux boundary tag")
        validate_coefficient(
            self.current_density,
            label=f"current density on {self.tag!r}",
        )


@dataclass(frozen=True, slots=True)
class SteadyCurrent:
    r"""Two-dimensional Cartesian H1 current conduction per unit out-of-plane depth.

    The initial isotropic form is

    .. math::

       \int_\Omega \nabla v^T\sigma\nabla\phi\,d\Omega
       = \int_\Omega vs\,d\Omega + \int_{\Gamma_g}vg\,d\Gamma.

    Regions partition the bulk cells. At least one strong potential boundary removes the constant
    null space. The derived physical fields are :math:`E=-\nabla\phi`,
    :math:`J=\sigma E`, and :math:`q_J=J\cdot E`.
    """

    regions: tuple[ConductiveRegion, ...]
    potential_boundaries: tuple[PotentialBoundary, ...]
    current_flux_boundaries: tuple[CurrentFluxBoundary, ...] = ()
    gradient_method: GradientMethod = GradientMethod.NONE

    @property
    def kind(self) -> str:
        """Return the stable equation identifier."""

        return "steady_current_h1_2d"

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
            raise ContractError("steady current requires at least one conductive region")
        if not self.potential_boundaries:
            raise ContractError("steady current requires at least one potential boundary")
        require_unique_tags(self.regions, label="conductive region")
        require_unique_tags(self.potential_boundaries, label="potential boundary")
        require_unique_tags(self.current_flux_boundaries, label="current-flux boundary")
        potential_tags = {boundary.tag for boundary in self.potential_boundaries}
        flux_tags = {boundary.tag for boundary in self.current_flux_boundaries}
        overlap = sorted(potential_tags & flux_tags)
        if overlap:
            raise ContractError(
                f"boundary tags cannot carry both potential and current flux: {overlap}"
            )

    def canonical_data(self) -> Mapping[str, object]:
        """Return deterministic backend-neutral metadata for hashing and provenance."""

        return {
            "kind": self.kind,
            "dimension": 2,
            "coordinate_system": "cartesian",
            "out_of_plane_convention": "per_unit_depth",
            "weak_form_flux_sign": "positive_rhs",
            "physical_current_density": "J=-sigma*grad(phi)",
            "gradient_method": self.gradient_method.value,
            "regions": [
                {
                    "tag": region.tag,
                    "electric_conductivity_S_per_m": coefficient_data(region.electric_conductivity),
                    "volumetric_current_source_A_per_m3": coefficient_data(
                        region.volumetric_current_source
                    ),
                }
                for region in self.regions
            ],
            "potential_boundaries": [
                {
                    "tag": boundary.tag,
                    "potential_V": coefficient_data(boundary.potential),
                }
                for boundary in self.potential_boundaries
            ],
            "current_flux_boundaries": [
                {
                    "tag": boundary.tag,
                    "positive_rhs_current_density_A_per_m2": coefficient_data(
                        boundary.current_density
                    ),
                }
                for boundary in self.current_flux_boundaries
            ],
        }
