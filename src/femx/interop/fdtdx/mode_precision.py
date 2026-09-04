"""Explicit float32/complex64 lowering for FDTDX TPU mode sources."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, cast

from femx.core.arrays import ArrayLike
from femx.core.axes import Axis, AxisDirection, Direction
from femx.core.errors import ContractError
from femx.interop.fdtdx.mode_bundle import (
    MagneticFieldConvention,
    ModeBundle,
    ModeNormalization,
    YeeFieldKind,
    YeeGrid,
    YeeVectorField,
)
from femx.interop.fdtdx.mode_source import _mode_bundle_sha256

_LOWERING_SCHEMA = "femx.fdtdx.mode_precision_lowering/v1"
_RUNTIME_GRID_SCHEMA = b"femx.fdtdx.tpu_runtime_grid/v1"
_RUNTIME_OPERATOR_SCHEMA = b"femx.fdtdx.tpu_mode_precision_operator/v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ETA0_OHM = 4.0e-7 * math.pi * 299_792_458.0
_MAXIMUM_COORDINATE_ERROR_CELL_FRACTION = 1.0e-5
_MAXIMUM_SCALAR_RELATIVE_ERROR = 1.0e-6
_MAXIMUM_FIELD_RELATIVE_ERROR = 5.0e-6
_MAXIMUM_MEDIUM_RELATIVE_ERROR = 1.0e-6
_MAXIMUM_POWER_RELATIVE_ERROR = 5.0e-6


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ContractError("TPU mode precision evidence must be canonical JSON") from error


def _framed_update(hasher: Any, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, "big"))
    hasher.update(value)


def _relative_l2(observed: Any, reference: Any) -> float:
    import numpy as np

    observed_array = np.asarray(observed, dtype=np.complex128)
    reference_array = np.asarray(reference, dtype=np.complex128)
    denominator = float(np.linalg.norm(reference_array))
    numerator = float(np.linalg.norm(observed_array - reference_array))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def _relative_maximum(observed: Any, reference: Any) -> float:
    import numpy as np

    observed_array = np.asarray(observed, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    denominator = np.maximum(np.abs(reference_array), np.finfo(np.float64).tiny)
    return float(np.max(np.abs(observed_array - reference_array) / denominator, initial=0.0))


def _signed_power_watts(electric: Any, magnetic_eta0: Any, grid: YeeGrid) -> float:
    import numpy as np

    electric_array = np.asarray(electric, dtype=np.complex128)
    magnetic_array = np.asarray(magnetic_eta0, dtype=np.complex128)
    x_edges = np.asarray(grid.edge_coordinates[0], dtype=np.float64)
    y_edges = np.asarray(grid.edge_coordinates[1], dtype=np.float64)
    area = np.diff(x_edges)[:, None, None] * np.diff(y_edges)[None, :, None]
    flux = 0.5 * np.real(
        electric_array[0] * np.conj(magnetic_array[1])
        - electric_array[1] * np.conj(magnetic_array[0])
    )
    return float(np.sum(area * flux, dtype=np.float64) / _ETA0_OHM)


def _runtime_grid(bundle: ModeBundle) -> tuple[YeeGrid, float, float]:
    import numpy as np

    axes: list[Any] = []
    maximum_error = 0.0
    maximum_cell_fraction = 0.0
    hasher = hashlib.sha256(_RUNTIME_GRID_SCHEMA)
    for axis_index, source_axis in enumerate(bundle.electric.grid.edge_coordinates):
        source = np.asarray(source_axis, dtype=np.float64)
        runtime = np.asarray(source, dtype=np.dtype("<f4"))
        if not np.isfinite(runtime).all() or np.any(np.diff(runtime) <= 0.0):
            raise ContractError(
                f"TPU float32 coordinates collapse or become invalid on axis {axis_index}"
            )
        error = np.abs(runtime.astype(np.float64) - source)
        widths = np.diff(source)
        maximum_error = max(maximum_error, float(np.max(error, initial=0.0)))
        maximum_cell_fraction = max(
            maximum_cell_fraction,
            float(np.max(error, initial=0.0) / np.min(widths)),
        )
        runtime = np.ascontiguousarray(runtime)
        runtime.setflags(write=False)
        _framed_update(hasher, str(axis_index).encode("ascii"))
        _framed_update(hasher, runtime.dtype.str.encode("ascii"))
        _framed_update(hasher, _canonical_json(list(runtime.shape)))
        _framed_update(hasher, runtime.tobytes(order="C"))
        axes.append(runtime)
    grid = YeeGrid(
        edge_coordinates=cast(tuple[ArrayLike, ArrayLike, ArrayLike], tuple(axes)),
        coordinate_sha256=hasher.hexdigest(),
    )
    return grid, maximum_error, maximum_cell_fraction


def _runtime_operator_sha256(
    source_bundle_sha256: str,
    grid: YeeGrid,
    electric: Any,
    magnetic: Any,
) -> str:
    import numpy as np

    hasher = hashlib.sha256(_RUNTIME_OPERATOR_SCHEMA)
    _framed_update(hasher, source_bundle_sha256.encode("ascii"))
    _framed_update(hasher, grid.coordinate_sha256.encode("ascii"))
    for label, values in (("electric", electric), ("magnetic", magnetic)):
        array = np.ascontiguousarray(values, dtype=np.dtype("<c8"))
        _framed_update(hasher, label.encode("ascii"))
        _framed_update(hasher, array.dtype.str.encode("ascii"))
        _framed_update(hasher, _canonical_json(list(array.shape)))
        _framed_update(hasher, array.tobytes(order="C"))
    return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class FDTDXModePrecisionReport:
    """Auditable error report for one explicit TPU scalar lowering."""

    source_bundle_sha256: str
    runtime_bundle_sha256: str
    lowering_operator_sha256: str
    source_field_dtype: str
    maximum_coordinate_error_m: float
    maximum_coordinate_error_cell_fraction: float
    relative_frequency_error: float
    relative_effective_index_error: float
    relative_beta_error: float
    electric_relative_l2_error: float
    magnetic_relative_l2_error: float
    inverse_permittivity_maximum_relative_error: float
    pre_correction_power_watts: float
    transferred_power_watts: float
    relative_pre_correction_power_error: float
    relative_power_error: float
    power_correction_scale: float
    runtime_real_dtype: str = "float32"
    runtime_field_dtype: str = "complex64"
    target_backend: str = "tpu"
    precision_fallback: bool = False
    schema_version: str = _LOWERING_SCHEMA

    def __post_init__(self) -> None:
        for digest_label, digest_value in (
            ("source bundle digest", self.source_bundle_sha256),
            ("runtime bundle digest", self.runtime_bundle_sha256),
            ("lowering operator digest", self.lowering_operator_sha256),
        ):
            if not _SHA256_PATTERN.fullmatch(digest_value):
                raise ContractError(
                    f"TPU mode precision {digest_label} must be a lowercase SHA-256"
                )
        if self.source_field_dtype not in {"complex64", "complex128"}:
            raise ContractError("TPU mode precision source field dtype is unsupported")
        for error_label, error_value in (
            ("coordinate error", self.maximum_coordinate_error_m),
            ("coordinate cell-fraction error", self.maximum_coordinate_error_cell_fraction),
            ("frequency error", self.relative_frequency_error),
            ("effective-index error", self.relative_effective_index_error),
            ("beta error", self.relative_beta_error),
            ("electric-field error", self.electric_relative_l2_error),
            ("magnetic-field error", self.magnetic_relative_l2_error),
            ("inverse-permittivity error", self.inverse_permittivity_maximum_relative_error),
            ("pre-correction power error", self.relative_pre_correction_power_error),
            ("transferred-power error", self.relative_power_error),
        ):
            if not math.isfinite(error_value) or error_value < 0.0:
                raise ContractError(
                    f"TPU mode precision {error_label} must be finite and nonnegative"
                )
        for positive_label, positive_value in (
            ("pre-correction power", self.pre_correction_power_watts),
            ("transferred power", self.transferred_power_watts),
            ("power-correction scale", self.power_correction_scale),
        ):
            if not math.isfinite(positive_value) or positive_value <= 0.0:
                raise ContractError(
                    f"TPU mode precision {positive_label} must be finite and positive"
                )
        if (
            self.runtime_real_dtype != "float32"
            or self.runtime_field_dtype != "complex64"
            or self.target_backend != "tpu"
            or self.precision_fallback is not False
        ):
            raise ContractError(
                "TPU mode precision requires explicit float32/complex64 without fallback"
            )
        if self.schema_version != _LOWERING_SCHEMA:
            raise ContractError(f"unsupported TPU mode precision schema {self.schema_version!r}")
        for admission_label, observed, threshold in (
            (
                "coordinate cell-fraction error",
                self.maximum_coordinate_error_cell_fraction,
                _MAXIMUM_COORDINATE_ERROR_CELL_FRACTION,
            ),
            ("frequency error", self.relative_frequency_error, _MAXIMUM_SCALAR_RELATIVE_ERROR),
            (
                "effective-index error",
                self.relative_effective_index_error,
                _MAXIMUM_SCALAR_RELATIVE_ERROR,
            ),
            ("beta error", self.relative_beta_error, _MAXIMUM_SCALAR_RELATIVE_ERROR),
            (
                "electric-field error",
                self.electric_relative_l2_error,
                _MAXIMUM_FIELD_RELATIVE_ERROR,
            ),
            (
                "magnetic-field error",
                self.magnetic_relative_l2_error,
                _MAXIMUM_FIELD_RELATIVE_ERROR,
            ),
            (
                "inverse-permittivity error",
                self.inverse_permittivity_maximum_relative_error,
                _MAXIMUM_MEDIUM_RELATIVE_ERROR,
            ),
            (
                "pre-correction power error",
                self.relative_pre_correction_power_error,
                _MAXIMUM_POWER_RELATIVE_ERROR,
            ),
            (
                "transferred-power error",
                self.relative_power_error,
                _MAXIMUM_POWER_RELATIVE_ERROR,
            ),
        ):
            if observed > threshold:
                raise ContractError(
                    f"TPU mode precision {admission_label} {observed} exceeds {threshold}"
                )

    def canonical_data(self) -> dict[str, object]:
        """Return the deterministic, JSON-safe precision evidence."""

        return {
            "schema_version": self.schema_version,
            "source_bundle_sha256": self.source_bundle_sha256,
            "runtime_bundle_sha256": self.runtime_bundle_sha256,
            "lowering_operator_sha256": self.lowering_operator_sha256,
            "source_field_dtype": self.source_field_dtype,
            "runtime_real_dtype": self.runtime_real_dtype,
            "runtime_field_dtype": self.runtime_field_dtype,
            "target_backend": self.target_backend,
            "precision_fallback": self.precision_fallback,
            "maximum_coordinate_error_m": self.maximum_coordinate_error_m,
            "maximum_coordinate_error_cell_fraction": (self.maximum_coordinate_error_cell_fraction),
            "relative_frequency_error": self.relative_frequency_error,
            "relative_effective_index_error": self.relative_effective_index_error,
            "relative_beta_error": self.relative_beta_error,
            "electric_relative_l2_error": self.electric_relative_l2_error,
            "magnetic_relative_l2_error": self.magnetic_relative_l2_error,
            "inverse_permittivity_maximum_relative_error": (
                self.inverse_permittivity_maximum_relative_error
            ),
            "pre_correction_power_watts": self.pre_correction_power_watts,
            "transferred_power_watts": self.transferred_power_watts,
            "relative_pre_correction_power_error": self.relative_pre_correction_power_error,
            "relative_power_error": self.relative_power_error,
            "power_correction_scale": self.power_correction_scale,
            "admission_thresholds": {
                "maximum_coordinate_error_cell_fraction": (_MAXIMUM_COORDINATE_ERROR_CELL_FRACTION),
                "maximum_scalar_relative_error": _MAXIMUM_SCALAR_RELATIVE_ERROR,
                "maximum_field_relative_error": _MAXIMUM_FIELD_RELATIVE_ERROR,
                "maximum_medium_relative_error": _MAXIMUM_MEDIUM_RELATIVE_ERROR,
                "maximum_power_relative_error": _MAXIMUM_POWER_RELATIVE_ERROR,
            },
            "claim_scope": (
                "explicit runtime precision conversion; not TPU execution, numerical parity, "
                "or accelerator evidence"
            ),
        }

    @property
    def sha256(self) -> str:
        """Return the logical identity of the precision evidence."""

        return hashlib.sha256(_canonical_json(self.canonical_data())).hexdigest()


@dataclass(frozen=True, slots=True)
class FDTDXTPUModeSourceInputs:
    """Runtime-only TPU mode and material values derived from a canonical mode."""

    bundle: ModeBundle
    expected_inverse_permittivity: ArrayLike
    expected_inverse_permeability: object
    report: FDTDXModePrecisionReport

    def __post_init__(self) -> None:
        import numpy as np

        permittivity = np.asarray(self.expected_inverse_permittivity)
        permeability = np.asarray(self.expected_inverse_permeability)
        if str(self.bundle.electric.values.dtype) != "complex64":
            raise ContractError("TPU runtime mode fields must use complex64")
        if any(
            np.asarray(axis).dtype != np.dtype("<f4")
            for axis in self.bundle.electric.grid.edge_coordinates
        ):
            raise ContractError("TPU runtime mode coordinates must use float32")
        if permittivity.dtype != np.dtype("<f4") or permittivity.shape != (
            1,
            *self.bundle.electric.grid.shape,
        ):
            raise ContractError("TPU runtime inverse permittivity has the wrong dtype or shape")
        if (
            permeability.dtype != np.dtype("<f8")
            or permeability.shape != ()
            or float(permeability) != 1.0
        ):
            raise ContractError(
                "TPU runtime inverse permeability must be FDTDX's float64 scalar-one sentinel"
            )
        runtime_arrays = (
            self.bundle.electric.values,
            self.bundle.magnetic.values,
            *self.bundle.electric.grid.edge_coordinates,
            permittivity,
            permeability,
        )
        if any(np.asarray(values).flags.writeable for values in runtime_arrays):
            raise ContractError("TPU runtime mode-source arrays must be read-only")
        if _mode_bundle_sha256(self.bundle) != self.report.runtime_bundle_sha256:
            raise ContractError("TPU runtime mode differs from its precision report")


def lower_mode_source_inputs_for_tpu(
    bundle: ModeBundle,
    *,
    expected_inverse_permittivity: ArrayLike,
    expected_inverse_permeability: object,
) -> FDTDXTPUModeSourceInputs:
    """Lower one canonical +z ``eta0_H`` mode to explicit TPU runtime scalars.

    The canonical artifact is not modified. Coordinates, fields, optical scalars, and source-plane
    material are converted deliberately, the signed power is recomputed on the float32 grid, and
    E/H receive one shared correction before a runtime-only ``ModeBundle`` is returned.
    """

    import numpy as np

    if bundle.propagation != AxisDirection(Axis.Z, Direction.POSITIVE):
        raise ContractError("TPU FDTDX mode precision v1 supports positive-z propagation only")
    if bundle.magnetic_convention is not MagneticFieldConvention.ETA0_H:
        raise ContractError("TPU FDTDX mode precision requires the eta0_H convention")
    if bundle.effective_index.real <= 0.0 or bundle.beta_per_m.real <= 0.0:
        raise ContractError("TPU FDTDX mode precision requires positive forward optical scalars")
    source_field_dtype = str(np.asarray(bundle.electric.values).dtype)
    if source_field_dtype not in {"complex64", "complex128"}:
        raise ContractError(
            "TPU FDTDX mode precision requires complex64 or complex128 source fields"
        )

    source_permittivity = np.asarray(expected_inverse_permittivity)
    source_permeability = np.asarray(expected_inverse_permeability)
    expected_shape = (1, *bundle.electric.grid.shape)
    if source_permittivity.shape != expected_shape or source_permittivity.dtype.kind not in "fiu":
        raise ContractError(
            f"TPU source inverse permittivity must have real shape {expected_shape}"
        )
    if not np.isfinite(source_permittivity).all() or not np.all(source_permittivity > 0.0):
        raise ContractError("TPU source inverse permittivity must be finite and positive")
    if source_permeability.shape != () or source_permeability.dtype.kind not in "fiu":
        raise ContractError("TPU source inverse permeability must be one real scalar")
    if not np.isfinite(source_permeability).all() or float(source_permeability) != 1.0:
        raise ContractError("TPU source inverse permeability must be finite scalar one")

    runtime_grid, coordinate_error, coordinate_cell_fraction = _runtime_grid(bundle)
    raw_electric = np.ascontiguousarray(bundle.electric.values, dtype=np.dtype("<c8"))
    raw_magnetic = np.ascontiguousarray(bundle.magnetic.values, dtype=np.dtype("<c8"))
    if not np.isfinite(raw_electric).all() or not np.isfinite(raw_magnetic).all():
        raise ContractError("TPU runtime mode fields must be finite")
    pre_correction_power = _signed_power_watts(raw_electric, raw_magnetic, runtime_grid)
    if not math.isfinite(pre_correction_power) or pre_correction_power <= 0.0:
        raise ContractError("TPU precision lowering produced zero, non-finite, or backward power")
    target_power = bundle.normalization.target_power_watts
    correction_scale = np.float32(math.sqrt(target_power / pre_correction_power))
    if not math.isfinite(float(correction_scale)) or float(correction_scale) <= 0.0:
        raise ContractError("TPU precision lowering produced an invalid power correction")
    electric = np.ascontiguousarray(raw_electric * correction_scale, dtype=np.dtype("<c8"))
    magnetic = np.ascontiguousarray(raw_magnetic * correction_scale, dtype=np.dtype("<c8"))
    electric.setflags(write=False)
    magnetic.setflags(write=False)
    transferred_power = _signed_power_watts(electric, magnetic, runtime_grid)
    if not math.isfinite(transferred_power) or transferred_power <= 0.0:
        raise ContractError("TPU precision lowering produced invalid corrected power")

    source_bundle_sha256 = _mode_bundle_sha256(bundle)
    operator_sha256 = _runtime_operator_sha256(
        source_bundle_sha256,
        runtime_grid,
        electric,
        magnetic,
    )
    runtime_frequency = float(np.float32(bundle.frequency_hz))
    runtime_effective_index = complex(np.complex64(bundle.effective_index))
    runtime_beta = complex(np.complex64(bundle.beta_per_m))
    runtime_bundle = ModeBundle(
        frequency_hz=runtime_frequency,
        effective_index=runtime_effective_index,
        beta_per_m=runtime_beta,
        electric=YeeVectorField(
            values=electric,
            grid=runtime_grid,
            field_kind=YeeFieldKind.ELECTRIC,
            unit=bundle.electric.unit,
            function_space=bundle.electric.function_space,
        ),
        magnetic=YeeVectorField(
            values=magnetic,
            grid=runtime_grid,
            field_kind=YeeFieldKind.MAGNETIC,
            unit=bundle.magnetic.unit,
            function_space=bundle.magnetic.function_space,
        ),
        propagation=bundle.propagation,
        magnetic_convention=bundle.magnetic_convention,
        normalization=ModeNormalization(
            kind=bundle.normalization.kind,
            target_power_watts=target_power,
            phase_reference=bundle.normalization.phase_reference,
        ),
        solver=bundle.solver,
        # The original transfer remains the FEM-to-Yee stage.  The second, runtime-only scalar
        # conversion is represented by FDTDXModePrecisionReport instead of rewriting that evidence.
        transfer=bundle.transfer,
    )
    # FDTDX stores forward relative permittivity in Material, casts that value to config.dtype,
    # and only then forms the inverse material array.  Directly casting an already inverted
    # float64 snapshot can differ by one float32 ULP, so reproduce the public runtime arithmetic.
    source_relative_permittivity = np.reciprocal(np.asarray(source_permittivity, dtype=np.float64))
    runtime_relative_permittivity = np.asarray(
        source_relative_permittivity,
        dtype=np.dtype("<f4"),
    )
    runtime_permittivity = np.ascontiguousarray(
        np.float32(1.0) / runtime_relative_permittivity,
        dtype=np.dtype("<f4"),
    )
    # FDTDX represents the nonmagnetic path as a Python scalar 1.0 rather than a material array.
    # Preserve that float64 host sentinel; JAX treats it as a weak scalar at field-update time.
    runtime_permeability = np.asarray(source_permeability, dtype=np.dtype("<f8"))
    runtime_permittivity.setflags(write=False)
    runtime_permeability.setflags(write=False)

    report = FDTDXModePrecisionReport(
        source_bundle_sha256=source_bundle_sha256,
        runtime_bundle_sha256=_mode_bundle_sha256(runtime_bundle),
        lowering_operator_sha256=operator_sha256,
        source_field_dtype=source_field_dtype,
        maximum_coordinate_error_m=coordinate_error,
        maximum_coordinate_error_cell_fraction=coordinate_cell_fraction,
        relative_frequency_error=abs(runtime_frequency - bundle.frequency_hz) / bundle.frequency_hz,
        relative_effective_index_error=(
            abs(runtime_effective_index - bundle.effective_index) / abs(bundle.effective_index)
        ),
        relative_beta_error=abs(runtime_beta - bundle.beta_per_m) / abs(bundle.beta_per_m),
        electric_relative_l2_error=_relative_l2(electric, bundle.electric.values),
        magnetic_relative_l2_error=_relative_l2(magnetic, bundle.magnetic.values),
        inverse_permittivity_maximum_relative_error=_relative_maximum(
            runtime_permittivity,
            source_permittivity,
        ),
        pre_correction_power_watts=pre_correction_power,
        transferred_power_watts=transferred_power,
        relative_pre_correction_power_error=abs(pre_correction_power - target_power) / target_power,
        relative_power_error=abs(transferred_power - target_power) / target_power,
        power_correction_scale=float(correction_scale),
    )
    return FDTDXTPUModeSourceInputs(
        bundle=runtime_bundle,
        expected_inverse_permittivity=runtime_permittivity,
        expected_inverse_permeability=runtime_permeability,
        report=report,
    )


__all__ = [
    "FDTDXModePrecisionReport",
    "FDTDXTPUModeSourceInputs",
    "lower_mode_source_inputs_for_tpu",
]
