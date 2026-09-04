"""Canonical optical-mode handoff between FEM backends and FDTDX."""

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, cast

from femx.core.arrays import ArrayLike, shape_of
from femx.core.axes import AxisDirection
from femx.core.errors import ContractError

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class FieldRepresentation(StrEnum):
    """Numerical representation of a vector field in a handoff."""

    CARTESIAN_SAMPLES = "cartesian_samples"
    CARTESIAN_YEE_SAMPLES = "cartesian_yee_samples"
    FEM_DOFS = "fem_dofs"


class YeeFieldKind(StrEnum):
    """Electric or magnetic staggering on an FDTDX Yee lattice."""

    ELECTRIC = "electric"
    MAGNETIC = "magnetic"


YEE_SPATIAL_OFFSETS: Final = {
    YeeFieldKind.ELECTRIC: (
        (0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 0.5),
    ),
    YeeFieldKind.MAGNETIC: (
        (0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5),
        (0.5, 0.5, 0.0),
    ),
}


class MagneticFieldConvention(StrEnum):
    """Scaling convention used for the magnetic field values."""

    PHYSICAL_H = "physical_H"
    ETA0_H = "eta0_H"


class NormalizationKind(StrEnum):
    """Normalization rule applied to a mode."""

    FORWARD_POWER = "forward_power"


@dataclass(frozen=True, slots=True)
class SampledVectorField:
    """Complex Cartesian vector samples or backend-native FEM coefficients."""

    values: ArrayLike
    coordinates: tuple[ArrayLike, ...]
    representation: FieldRepresentation
    unit: str
    function_space: str

    def __post_init__(self) -> None:
        value_shape = shape_of(self.values)
        if not value_shape or value_shape[0] != 3:
            raise ContractError(
                f"vector field values must start with three components, got {value_shape}"
            )
        if not self.coordinates:
            raise ContractError("vector field must retain physical coordinates")
        if self.representation is not FieldRepresentation.CARTESIAN_SAMPLES:
            raise ContractError("ModeBundle vector fields must be Cartesian samples")
        for coordinate in self.coordinates:
            if len(shape_of(coordinate)) != 1:
                raise ContractError("mode coordinates must be one-dimensional arrays")
        sample_shape = value_shape[1:]
        coordinate_shape = tuple(int(coordinate.shape[0]) for coordinate in self.coordinates)
        if coordinate_shape != sample_shape:
            raise ContractError(
                f"coordinate lengths {coordinate_shape} do not match samples {sample_shape}"
            )
        if not self.unit or not self.function_space:
            raise ContractError("vector field must declare unit and function-space provenance")
        dtype_kind = getattr(self.values.dtype, "kind", None)
        if dtype_kind is not None and dtype_kind != "c":
            raise ContractError("mode field values must use a complex dtype")


@dataclass(frozen=True, slots=True)
class YeeGrid:
    """Three physical edge-coordinate axes shared by all staggered Yee components."""

    edge_coordinates: tuple[ArrayLike, ArrayLike, ArrayLike]
    coordinate_sha256: str
    coordinate_unit: str = "m"

    def __post_init__(self) -> None:
        if len(self.edge_coordinates) != 3:
            raise ContractError("Yee grid requires exactly three physical edge axes")
        for axis, coordinates in enumerate(self.edge_coordinates):
            shape = shape_of(coordinates)
            if len(shape) != 1 or shape[0] < 2:
                raise ContractError(
                    f"Yee edge axis {axis} must be a vector with at least two entries"
                )
        if not _SHA256_PATTERN.fullmatch(self.coordinate_sha256):
            raise ContractError("Yee coordinate digest must be a lowercase SHA-256")
        if self.coordinate_unit != "m":
            raise ContractError("Yee coordinates must use SI metres")

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the common logical array shape of every Yee component."""

        return tuple(int(axis.shape[0]) - 1 for axis in self.edge_coordinates)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class YeeVectorField:
    """Complex Cartesian values at the exact electric or magnetic Yee offsets."""

    values: ArrayLike
    grid: YeeGrid
    field_kind: YeeFieldKind
    unit: str
    function_space: str = "FDTDX Yee lattice"
    representation: FieldRepresentation = FieldRepresentation.CARTESIAN_YEE_SAMPLES

    def __post_init__(self) -> None:
        value_shape = shape_of(self.values)
        expected_shape = (3, *self.grid.shape)
        if value_shape != expected_shape:
            raise ContractError(
                f"Yee vector field values must have shape {expected_shape}, got {value_shape}"
            )
        if self.representation is not FieldRepresentation.CARTESIAN_YEE_SAMPLES:
            raise ContractError("Yee vector fields require the Cartesian Yee representation")
        if not self.unit or not self.function_space:
            raise ContractError("Yee vector field must declare unit and function-space provenance")
        dtype_kind = getattr(self.values.dtype, "kind", None)
        if dtype_kind is not None and dtype_kind != "c":
            raise ContractError("Yee mode field values must use a complex dtype")

    @property
    def spatial_offsets(self) -> tuple[tuple[float, float, float], ...]:
        """Return component offsets matching FDTDX ``calculate_spatial_offsets_yee``."""

        return YEE_SPATIAL_OFFSETS[self.field_kind]


@dataclass(frozen=True, slots=True)
class ModeNormalization:
    """Power and phase normalization contract."""

    kind: NormalizationKind = NormalizationKind.FORWARD_POWER
    target_power_watts: float = 1.0
    phase_reference: str = "largest_E_component_real_positive"

    def __post_init__(self) -> None:
        if not math.isfinite(self.target_power_watts) or self.target_power_watts <= 0:
            raise ContractError("mode normalization power must be positive")
        if not self.phase_reference:
            raise ContractError("mode normalization must specify a deterministic phase reference")


@dataclass(frozen=True, slots=True)
class SolverFingerprint:
    """Solver and input identity carried with a mode."""

    name: str
    version: str
    config_sha256: str
    mesh_sha256: str
    source_revision: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ContractError("mode solver name and version must be non-empty")
        for label, digest in (
            ("config_sha256", self.config_sha256),
            ("mesh_sha256", self.mesh_sha256),
        ):
            if not _SHA256_PATTERN.fullmatch(digest):
                raise ContractError(f"{label} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class TransferReport:
    """Evidence for a numerical field transfer between representations."""

    source_representation: FieldRepresentation
    target_representation: FieldRepresentation
    operator_sha256: str
    relative_power_error: float
    relative_interpolation_error: float | None = None
    source_power_watts: float | None = None
    pre_correction_power_watts: float | None = None
    relative_pre_correction_power_error: float | None = None
    transferred_power_watts: float | None = None
    power_correction_scale: float | None = None
    target_runtime_name: str | None = None
    target_runtime_version: str | None = None
    target_source_revision: str | None = None
    target_source_digest: str | None = None

    def __post_init__(self) -> None:
        if not _SHA256_PATTERN.fullmatch(self.operator_sha256):
            raise ContractError("transfer operator digest must be a lowercase SHA-256")
        if not math.isfinite(self.relative_power_error) or self.relative_power_error < 0:
            raise ContractError("relative power error cannot be negative")
        if self.relative_interpolation_error is not None and (
            not math.isfinite(self.relative_interpolation_error)
            or self.relative_interpolation_error < 0
        ):
            raise ContractError("relative interpolation error cannot be negative")
        for label, value in (
            ("source power", self.source_power_watts),
            ("pre-correction power", self.pre_correction_power_watts),
            ("transferred power", self.transferred_power_watts),
            ("power correction scale", self.power_correction_scale),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ContractError(f"transfer {label} must be positive")
        if self.relative_pre_correction_power_error is not None and (
            not math.isfinite(self.relative_pre_correction_power_error)
            or self.relative_pre_correction_power_error < 0
        ):
            raise ContractError("relative pre-correction power error cannot be negative")
        target_identity = (
            self.target_runtime_name,
            self.target_runtime_version,
            self.target_source_revision,
            self.target_source_digest,
        )
        if any(value is not None for value in target_identity):
            if any(value is None or not value for value in target_identity):
                raise ContractError("transfer target runtime identity must be complete")
            if _GIT_REVISION_PATTERN.fullmatch(cast(str, self.target_source_revision)) is None:
                raise ContractError("transfer target source revision must be a lowercase Git id")
            if _SHA256_PATTERN.fullmatch(cast(str, self.target_source_digest)) is None:
                raise ContractError("transfer target source digest must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class ModeBundle:
    """Versioned, convention-complete optical mode artifact."""

    frequency_hz: float
    effective_index: complex
    beta_per_m: complex
    electric: YeeVectorField
    magnetic: YeeVectorField
    propagation: AxisDirection
    magnetic_convention: MagneticFieldConvention
    normalization: ModeNormalization
    solver: SolverFingerprint
    transfer: TransferReport
    schema_version: str = "femx.mode/v1"

    def __post_init__(self) -> None:
        if not math.isfinite(self.frequency_hz) or self.frequency_hz <= 0:
            raise ContractError("mode frequency must be finite and positive")
        if not all(
            math.isfinite(value)
            for value in (
                self.effective_index.real,
                self.effective_index.imag,
                self.beta_per_m.real,
                self.beta_per_m.imag,
            )
        ):
            raise ContractError("mode effective index and propagation constant must be finite")
        if not isinstance(self.electric, YeeVectorField) or not isinstance(
            self.magnetic, YeeVectorField
        ):
            raise ContractError("ModeBundle v1 requires exact staggered Yee vector fields")
        if self.electric.field_kind is not YeeFieldKind.ELECTRIC:
            raise ContractError("ModeBundle electric field must use electric Yee offsets")
        if self.magnetic.field_kind is not YeeFieldKind.MAGNETIC:
            raise ContractError("ModeBundle magnetic field must use magnetic Yee offsets")
        if (
            self.electric.grid.shape != self.magnetic.grid.shape
            or self.electric.grid.coordinate_sha256 != self.magnetic.grid.coordinate_sha256
        ):
            raise ContractError("electric and magnetic mode fields use different Yee grids")
        if self.electric.values.dtype != self.magnetic.values.dtype:
            raise ContractError("electric and magnetic mode fields use different scalar dtypes")
        propagation_axis = {"x": 0, "y": 1, "z": 2}[self.propagation.axis.value]
        if self.electric.grid.shape[propagation_axis] != 1:
            raise ContractError(
                "ModeBundle Yee grid must be one cell thick on the propagation axis"
            )
        if self.electric.unit != "V/m":
            raise ContractError("ModeBundle electric Yee field must use unit 'V/m'")
        expected_magnetic_unit = (
            "V/m" if self.magnetic_convention is MagneticFieldConvention.ETA0_H else "A/m"
        )
        if self.magnetic.unit != expected_magnetic_unit:
            raise ContractError(
                f"{self.magnetic_convention.value} requires magnetic unit "
                f"{expected_magnetic_unit!r}"
            )
        if self.transfer.source_representation is not FieldRepresentation.FEM_DOFS:
            raise ContractError("ModeBundle transfer must originate from FEM DOFs")
        if self.transfer.target_representation is not FieldRepresentation.CARTESIAN_YEE_SAMPLES:
            raise ContractError("ModeBundle transfer must target Cartesian Yee samples")
        if self.transfer.target_runtime_name != "fdtdx":
            raise ContractError("ModeBundle transfer must identify the FDTDX target runtime")
        power_evidence = (
            self.transfer.source_power_watts,
            self.transfer.pre_correction_power_watts,
            self.transfer.relative_pre_correction_power_error,
            self.transfer.transferred_power_watts,
            self.transfer.power_correction_scale,
        )
        if any(value is None for value in power_evidence):
            raise ContractError("ModeBundle transfer must retain complete signed-power evidence")
        source_power = cast(float, self.transfer.source_power_watts)
        pre_correction_power = cast(float, self.transfer.pre_correction_power_watts)
        transferred_power = cast(float, self.transfer.transferred_power_watts)
        correction_scale = cast(float, self.transfer.power_correction_scale)
        if source_power != self.normalization.target_power_watts:
            raise ContractError("ModeBundle source power differs from its normalization target")
        expected_power_error = abs(transferred_power - source_power) / source_power
        if not math.isclose(
            self.transfer.relative_power_error,
            expected_power_error,
            rel_tol=64.0 * math.ulp(1.0),
            abs_tol=64.0 * math.ulp(1.0),
        ):
            raise ContractError("ModeBundle transferred-power error is internally inconsistent")
        expected_pre_error = abs(pre_correction_power - source_power) / source_power
        if not math.isclose(
            cast(float, self.transfer.relative_pre_correction_power_error),
            expected_pre_error,
            rel_tol=64.0 * math.ulp(1.0),
            abs_tol=64.0 * math.ulp(1.0),
        ):
            raise ContractError("ModeBundle pre-correction power error is internally inconsistent")
        if not math.isclose(
            pre_correction_power * correction_scale**2,
            transferred_power,
            rel_tol=128.0 * math.ulp(1.0),
            abs_tol=128.0 * math.ulp(1.0) * source_power,
        ):
            raise ContractError("ModeBundle power-correction scale is internally inconsistent")
        if self.schema_version != "femx.mode/v1":
            raise ContractError(f"unsupported mode schema {self.schema_version!r}")
