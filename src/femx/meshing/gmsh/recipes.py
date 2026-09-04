"""Deterministic Gmsh geometry recipes used by Silicon Photonics evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal

from femx.core.errors import ContractError


@dataclass(frozen=True, slots=True)
class RectangularWaveguideCrossSection:
    """Conformal rectangular Si-core/cladding cross-section expressed in SI."""

    cladding_width_m: float = 4.0e-6
    cladding_height_m: float = 3.0e-6
    core_width_m: float = 0.5e-6
    core_height_m: float = 0.22e-6
    cladding_mesh_size_m: float = 0.4e-6
    core_mesh_size_m: float = 0.08e-6
    model_unit_m: float = 1.0e-6

    def __post_init__(self) -> None:
        values = (
            self.cladding_width_m,
            self.cladding_height_m,
            self.core_width_m,
            self.core_height_m,
            self.cladding_mesh_size_m,
            self.core_mesh_size_m,
            self.model_unit_m,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ContractError("waveguide dimensions and mesh sizes must be finite and positive")
        if self.core_width_m >= self.cladding_width_m:
            raise ContractError("waveguide core width must be smaller than cladding width")
        if self.core_height_m >= self.cladding_height_m:
            raise ContractError("waveguide core height must be smaller than cladding height")
        if self.core_mesh_size_m > self.cladding_mesh_size_m:
            raise ContractError("core mesh size must not exceed cladding mesh size")

    @property
    def coordinate_scale_to_m(self) -> float:
        """Scale required when importing the rendered dimensionless Gmsh model."""

        return self.model_unit_m

    def render_geo(self) -> str:
        """Render a stable Built-in-kernel geometry with named physical groups."""

        model_unit = Decimal(str(self.model_unit_m))
        two_model_units = Decimal(2) * model_unit
        values = {
            "cx": _format_real(Decimal(str(self.cladding_width_m)) / two_model_units),
            "cy": _format_real(Decimal(str(self.cladding_height_m)) / two_model_units),
            "wx": _format_real(Decimal(str(self.core_width_m)) / two_model_units),
            "wy": _format_real(Decimal(str(self.core_height_m)) / two_model_units),
            "lc_cladding": _format_real(Decimal(str(self.cladding_mesh_size_m)) / model_unit),
            "lc_core": _format_real(Decimal(str(self.core_mesh_size_m)) / model_unit),
            "unit": _format_real(model_unit),
        }
        return f"""// femx rectangular waveguide; model coordinate unit = {values["unit"]} m
SetFactory("Built-in");

lc_cladding = {values["lc_cladding"]};
lc_core = {values["lc_core"]};

Point(1) = {{-{values["cx"]}, -{values["cy"]}, 0, lc_cladding}};
Point(2) = {{ {values["cx"]}, -{values["cy"]}, 0, lc_cladding}};
Point(3) = {{ {values["cx"]},  {values["cy"]}, 0, lc_cladding}};
Point(4) = {{-{values["cx"]},  {values["cy"]}, 0, lc_cladding}};
Point(5) = {{-{values["wx"]}, -{values["wy"]}, 0, lc_core}};
Point(6) = {{ {values["wx"]}, -{values["wy"]}, 0, lc_core}};
Point(7) = {{ {values["wx"]},  {values["wy"]}, 0, lc_core}};
Point(8) = {{-{values["wx"]},  {values["wy"]}, 0, lc_core}};

Line(1) = {{1, 2}};
Line(2) = {{2, 3}};
Line(3) = {{3, 4}};
Line(4) = {{4, 1}};
Line(5) = {{5, 6}};
Line(6) = {{6, 7}};
Line(7) = {{7, 8}};
Line(8) = {{8, 5}};

Curve Loop(11) = {{1, 2, 3, 4}};
Curve Loop(12) = {{5, 6, 7, 8}};
Plane Surface(21) = {{11, 12}};
Plane Surface(22) = {{12}};

Physical Surface("cladding", 101) = {{21}};
Physical Surface("core", 102) = {{22}};
Physical Curve("bottom", 201) = {{1}};
Physical Curve("right", 202) = {{2}};
Physical Curve("top", 203) = {{3}};
Physical Curve("left", 204) = {{4}};

Mesh.MshFileVersion = 4.1;
Mesh.Binary = 0;
Mesh.ElementOrder = 1;
Mesh.SaveAll = 0;
Mesh.Algorithm = 6;
Mesh.AlgorithmSwitchOnFailure = 0;
Mesh.RandomFactor = 1e-9;
Mesh.RandomSeed = 1;
"""


def _format_real(value: Decimal) -> str:
    return format(value, ".17g")
