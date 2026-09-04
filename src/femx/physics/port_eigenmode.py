"""Solver-neutral two-dimensional electromagnetic port-eigenmode contract."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from femx.core.axes import Axis, AxisDirection, Direction
from femx.core.capabilities import (
    AnalysisKind,
    CapabilityRequest,
    FunctionSpaceFamily,
    GradientMethod,
    ParallelModel,
    ScalarKind,
)
from femx.core.errors import ContractError
from femx.core.parameters import ParameterReference
from femx.physics._scalar import (
    ScalarCoefficient,
    coefficient_data,
    require_unique_tags,
    validate_name,
)

VACUUM_SPEED_OF_LIGHT_M_PER_S: Final = 299_792_458.0
VACUUM_PERMITTIVITY_F_PER_M: Final = 8.854_187_812_8e-12
VACUUM_PERMEABILITY_H_PER_M: Final = 1.0 / (
    VACUUM_PERMITTIVITY_F_PER_M * VACUUM_SPEED_OF_LIGHT_M_PER_S**2
)
PORT_LONGITUDINAL_POTENTIAL_FIELD: Final = "port_longitudinal_potential_coefficients"
PORT_TRANSVERSE_ELECTRIC_FIELD: Final = "port_transverse_electric_edge_coefficients"
PORT_LONGITUDINAL_POTENTIAL_UNIT: Final = "V/m^2"
PORT_TRANSVERSE_ELECTRIC_DOF_UNIT: Final = "V"
_POSITIVE_Z_PROPAGATION: Final = AxisDirection(Axis.Z, Direction.POSITIVE)


@dataclass(frozen=True, slots=True)
class IsotropicOpticalRegion:
    """Lossless isotropic relative material properties on one bulk-cell tag."""

    tag: str
    relative_permittivity: ScalarCoefficient
    relative_permeability: ScalarCoefficient = 1.0

    def __post_init__(self) -> None:
        validate_name(self.tag, label="optical region tag")
        for label, value in (
            ("relative permittivity", self.relative_permittivity),
            ("relative permeability", self.relative_permeability),
        ):
            if isinstance(value, ParameterReference):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ContractError(f"{label} on {self.tag!r} must be a real scalar")
            if not math.isfinite(value) or value <= 0.0:
                raise ContractError(f"{label} on {self.tag!r} must be finite and positive")


@dataclass(frozen=True, slots=True)
class PerfectElectricBoundary:
    """Homogeneous perfect-electric-conductor condition on a boundary-facet tag."""

    tag: str

    def __post_init__(self) -> None:
        validate_name(self.tag, label="perfect-electric boundary tag")


@dataclass(frozen=True, slots=True)
class PortEigenmode:
    r"""Lossless mixed :math:`H(\mathrm{curl})`--:math:`H^1` port cross-section.

    The v1 contract uses a planar ``x-y`` mesh, positive ``z`` propagation, piecewise-constant
    isotropic real material properties, and homogeneous PEC on the complete external boundary.
    Modes are ordered by decreasing real propagation constant.  Complex fields are required even
    though the current material subset is lossless.
    """

    regions: tuple[IsotropicOpticalRegion, ...]
    perfect_electric_boundaries: tuple[PerfectElectricBoundary, ...]
    frequency_hz: float
    eigenmode_count: int = 8
    selected_mode_index: int = 0
    target_power_w: float = 1.0
    propagation: AxisDirection = _POSITIVE_Z_PROPAGATION
    gradient_method: GradientMethod = GradientMethod.NONE

    @property
    def kind(self) -> str:
        """Return the stable equation identifier."""

        return "port_eigenmode_mixed_hcurl_h1_2d"

    @property
    def requirements(self) -> CapabilityRequest:
        """Return the exact mathematical and execution capabilities required."""

        return CapabilityRequest(
            analysis=AnalysisKind.EIGENMODE,
            function_spaces=frozenset({FunctionSpaceFamily.HCURL, FunctionSpaceFamily.H1}),
            scalar_kind=ScalarKind.COMPLEX,
            gradient=self.gradient_method,
            parallel=ParallelModel.SERIAL,
        )

    @property
    def vacuum_wavelength_m(self) -> float:
        """Return the SI vacuum wavelength implied by the frequency."""

        return VACUUM_SPEED_OF_LIGHT_M_PER_S / float(self.frequency_hz)

    def validate(self) -> None:
        """Reject material, boundary, selection, and propagation ambiguity."""

        if not self.regions:
            raise ContractError("port eigenmode requires at least one optical region")
        if not self.perfect_electric_boundaries:
            raise ContractError("port eigenmode requires a complete PEC boundary declaration")
        require_unique_tags(self.regions, label="optical region")
        require_unique_tags(
            self.perfect_electric_boundaries,
            label="perfect-electric boundary",
        )
        if not math.isfinite(self.frequency_hz) or self.frequency_hz <= 0.0:
            raise ContractError("port eigenmode frequency must be finite and positive")
        if not isinstance(self.eigenmode_count, int) or isinstance(self.eigenmode_count, bool):
            raise ContractError("port eigenmode count must be a positive integer")
        if self.eigenmode_count <= 0:
            raise ContractError("port eigenmode count must be a positive integer")
        if self.eigenmode_count > 256:
            raise ContractError("port eigenmode count exceeds the v1 limit of 256")
        if (
            not isinstance(self.selected_mode_index, int)
            or isinstance(self.selected_mode_index, bool)
            or self.selected_mode_index < 0
            or self.selected_mode_index >= self.eigenmode_count
        ):
            raise ContractError("selected port mode index must be within the requested spectrum")
        if not math.isfinite(self.target_power_w) or self.target_power_w <= 0.0:
            raise ContractError("port eigenmode target power must be finite and positive")
        if self.propagation != _POSITIVE_Z_PROPAGATION:
            raise ContractError("port eigenmode v1 requires positive-z propagation")

    def canonical_data(self) -> Mapping[str, object]:
        """Return deterministic JSON-compatible physics metadata."""

        return {
            "kind": self.kind,
            "dimension": 2,
            "coordinate_system": "cartesian_xy",
            "formulation": "mixed_hcurl_h1",
            "material_model": "lossless_piecewise_constant_isotropic",
            "boundary_model": "homogeneous_pec_complete_external_boundary",
            "mode_ordering": "decreasing_real_propagation_constant",
            "frequency_hz": float(self.frequency_hz),
            "vacuum_wavelength_m": self.vacuum_wavelength_m,
            "eigenmode_count": self.eigenmode_count,
            "selected_mode_index_zero_based": self.selected_mode_index,
            "target_forward_power_W": float(self.target_power_w),
            "propagation": {
                "axis": self.propagation.axis.value,
                "direction": self.propagation.direction.value,
            },
            "gradient_method": self.gradient_method.value,
            "vacuum_constants": {
                "speed_of_light_m_per_s": VACUUM_SPEED_OF_LIGHT_M_PER_S,
                "permittivity_F_per_m": VACUUM_PERMITTIVITY_F_PER_M,
                "permeability_H_per_m": VACUUM_PERMEABILITY_H_PER_M,
            },
            "regions": [
                {
                    "tag": region.tag,
                    "relative_permittivity": coefficient_data(region.relative_permittivity),
                    "relative_permeability": coefficient_data(region.relative_permeability),
                }
                for region in self.regions
            ],
            "perfect_electric_boundaries": [
                {"tag": boundary.tag} for boundary in self.perfect_electric_boundaries
            ],
        }
