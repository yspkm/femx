"""Fail-closed binding from a canonical mode bundle to an FDTDX plane source."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any, cast

from femx.core.arrays import ArrayLike
from femx.core.errors import ContractError
from femx.interop.fdtdx.mode_bundle import ModeBundle
from femx.interop.fdtdx.mode_transfer import make_fdtdx_mode_function
from femx.interop.fdtdx.thermo_optic import FDTDXFingerprint

_MODE_SOURCE_SCHEMA = "femx.fdtdx.mode_source/v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _framed_update(hasher: Any, payload: bytes) -> None:
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


def _canonical_array(values: object, *, label: str) -> tuple[Any, str]:
    import numpy as np

    array = np.asarray(values)
    if array.dtype.kind not in "fcu" or array.dtype.itemsize not in (4, 8, 16):
        raise ContractError(f"{label} uses unsupported dtype {array.dtype}")
    canonical_dtype = array.dtype.newbyteorder("<")
    canonical = np.array(array, dtype=canonical_dtype, order="C", copy=True)
    canonical.setflags(write=False)
    hasher = hashlib.sha256()
    _framed_update(hasher, label.encode("utf-8"))
    _framed_update(hasher, canonical.dtype.str.encode("ascii"))
    _framed_update(
        hasher,
        json.dumps(list(canonical.shape), separators=(",", ":")).encode("ascii"),
    )
    _framed_update(hasher, canonical.tobytes(order="C"))
    return canonical, hasher.hexdigest()


def _mode_bundle_sha256(bundle: ModeBundle) -> str:
    array_hashes: dict[str, str] = {}
    for name, values in (
        ("grid/x_edges_m", bundle.electric.grid.edge_coordinates[0]),
        ("grid/y_edges_m", bundle.electric.grid.edge_coordinates[1]),
        ("grid/z_edges_m", bundle.electric.grid.edge_coordinates[2]),
        ("fields/electric", bundle.electric.values),
        ("fields/magnetic", bundle.magnetic.values),
    ):
        _snapshot, array_hashes[name] = _canonical_array(values, label=name)
    metadata = {
        "schema_version": bundle.schema_version,
        "frequency_hz": bundle.frequency_hz,
        "effective_index": [bundle.effective_index.real, bundle.effective_index.imag],
        "beta_per_m": [bundle.beta_per_m.real, bundle.beta_per_m.imag],
        "propagation_axis": bundle.propagation.axis.value,
        "propagation_direction": bundle.propagation.direction.value,
        "magnetic_convention": bundle.magnetic_convention.value,
        "normalization_kind": bundle.normalization.kind.value,
        "target_power_watts": bundle.normalization.target_power_watts,
        "phase_reference": bundle.normalization.phase_reference,
        "coordinate_sha256": bundle.electric.grid.coordinate_sha256,
        "coordinate_unit": bundle.electric.grid.coordinate_unit,
        "field_dtype": str(bundle.electric.values.dtype),
        "electric_field": {
            "field_kind": bundle.electric.field_kind.value,
            "representation": bundle.electric.representation.value,
            "unit": bundle.electric.unit,
            "function_space": bundle.electric.function_space,
        },
        "magnetic_field": {
            "field_kind": bundle.magnetic.field_kind.value,
            "representation": bundle.magnetic.representation.value,
            "unit": bundle.magnetic.unit,
            "function_space": bundle.magnetic.function_space,
        },
        "solver": {
            "name": bundle.solver.name,
            "version": bundle.solver.version,
            "source_revision": bundle.solver.source_revision,
            "config_sha256": bundle.solver.config_sha256,
            "mesh_sha256": bundle.solver.mesh_sha256,
        },
        "transfer": {
            "source_representation": bundle.transfer.source_representation.value,
            "target_representation": bundle.transfer.target_representation.value,
            "operator_sha256": bundle.transfer.operator_sha256,
            "relative_power_error": bundle.transfer.relative_power_error,
            "relative_interpolation_error": bundle.transfer.relative_interpolation_error,
            "source_power_watts": bundle.transfer.source_power_watts,
            "pre_correction_power_watts": bundle.transfer.pre_correction_power_watts,
            "relative_pre_correction_power_error": (
                bundle.transfer.relative_pre_correction_power_error
            ),
            "transferred_power_watts": bundle.transfer.transferred_power_watts,
            "power_correction_scale": bundle.transfer.power_correction_scale,
            "target_runtime_name": bundle.transfer.target_runtime_name,
            "target_runtime_version": bundle.transfer.target_runtime_version,
            "target_source_revision": bundle.transfer.target_source_revision,
            "target_source_digest": bundle.transfer.target_source_digest,
        },
        "array_sha256": array_hashes,
    }
    encoded = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: str, *, label: str) -> None:
    if not _SHA256_PATTERN.fullmatch(value):
        raise ContractError(f"{label} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class FDTDXModeSourceContract:
    """Static identity and source-plane medium required by one imported mode."""

    source_name: str
    grid_shape: tuple[int, int, int]
    frequency_hz: float
    effective_index: complex
    field_dtype: str
    coordinate_sha256: str
    mode_bundle_sha256: str
    expected_inverse_permittivity: ArrayLike
    expected_inverse_permeability: object
    inverse_permittivity_sha256: str
    inverse_permeability_sha256: str
    fdtdx: FDTDXFingerprint
    propagation_axis: int = 2
    propagation_direction: str = "+"
    schema_version: str = _MODE_SOURCE_SCHEMA

    def __post_init__(self) -> None:
        import numpy as np

        if not self.source_name or self.source_name.strip() != self.source_name:
            raise ContractError("FDTDX mode source name must be non-empty and trimmed")
        if len(self.grid_shape) != 3 or any(size <= 0 for size in self.grid_shape):
            raise ContractError("FDTDX mode source grid shape must contain three positive sizes")
        if self.propagation_axis not in (0, 1, 2):
            raise ContractError("FDTDX mode source propagation axis must be 0, 1, or 2")
        if self.grid_shape[self.propagation_axis] != 1:
            raise ContractError("FDTDX mode source must be one cell thick on its propagation axis")
        if self.propagation_axis != 2 or self.propagation_direction != "+":
            raise ContractError("FDTDX mode source v1 supports positive-z propagation only")
        if not math.isfinite(self.frequency_hz) or self.frequency_hz <= 0.0:
            raise ContractError("FDTDX mode source frequency must be finite and positive")
        if (
            not all(
                math.isfinite(value)
                for value in (self.effective_index.real, self.effective_index.imag)
            )
            or self.effective_index.real <= 0.0
        ):
            raise ContractError(
                "FDTDX mode source effective index must have a positive finite real part"
            )
        if self.field_dtype not in {"complex64", "complex128"}:
            raise ContractError("FDTDX mode source field dtype must be complex64 or complex128")
        for value, label in (
            (self.coordinate_sha256, "FDTDX mode source coordinate digest"),
            (self.mode_bundle_sha256, "FDTDX mode source bundle digest"),
            (self.inverse_permittivity_sha256, "FDTDX mode source permittivity digest"),
            (self.inverse_permeability_sha256, "FDTDX mode source permeability digest"),
        ):
            _require_sha256(value, label=label)

        inverse_permittivity, permittivity_sha256 = _canonical_array(
            self.expected_inverse_permittivity,
            label="source_plane/inverse_permittivity",
        )
        if inverse_permittivity.shape != (1, *self.grid_shape):
            raise ContractError(
                "FDTDX mode source v1 requires isotropic inverse permittivity with shape "
                f"{(1, *self.grid_shape)}"
            )
        expected_real_dtype = np.dtype("<f4" if self.field_dtype == "complex64" else "<f8")
        if inverse_permittivity.dtype != expected_real_dtype:
            raise ContractError(
                "FDTDX mode source inverse permittivity precision differs from its field precision"
            )
        if not np.all(np.isfinite(inverse_permittivity)) or not np.all(inverse_permittivity > 0.0):
            raise ContractError(
                "FDTDX mode source inverse permittivity must be finite and positive"
            )
        if permittivity_sha256 != self.inverse_permittivity_sha256:
            raise ContractError("FDTDX mode source inverse-permittivity digest is inconsistent")

        inverse_permeability, permeability_sha256 = _canonical_array(
            self.expected_inverse_permeability,
            label="source_plane/inverse_permeability",
        )
        if inverse_permeability.shape != () or inverse_permeability.dtype != np.dtype("<f8"):
            raise ContractError(
                "FDTDX mode source v1 requires the nonmagnetic float64 scalar sentinel"
            )
        if float(inverse_permeability) != 1.0:
            raise ContractError("FDTDX mode source v1 requires a non-magnetic source plane")
        if permeability_sha256 != self.inverse_permeability_sha256:
            raise ContractError("FDTDX mode source inverse-permeability digest is inconsistent")
        if self.schema_version != _MODE_SOURCE_SCHEMA:
            raise ContractError(f"unsupported FDTDX mode source schema {self.schema_version!r}")

        object.__setattr__(self, "expected_inverse_permittivity", inverse_permittivity)
        object.__setattr__(self, "expected_inverse_permeability", inverse_permeability)

    def canonical_data(self) -> Mapping[str, object]:
        """Return the deterministic, array-free source binding metadata."""

        return {
            "schema_version": self.schema_version,
            "source_name": self.source_name,
            "grid_shape_xyz": list(self.grid_shape),
            "frequency_hz": self.frequency_hz,
            "effective_index": [self.effective_index.real, self.effective_index.imag],
            "field_dtype": self.field_dtype,
            "propagation_axis": self.propagation_axis,
            "propagation_direction": self.propagation_direction,
            "coordinate_sha256": self.coordinate_sha256,
            "mode_bundle_sha256": self.mode_bundle_sha256,
            "inverse_permittivity_sha256": self.inverse_permittivity_sha256,
            "inverse_permeability_sha256": self.inverse_permeability_sha256,
            "fdtdx_package_version": self.fdtdx.package_version,
            "fdtdx_source_revision": self.fdtdx.source_revision,
            "fdtdx_source_digest": self.fdtdx.source_digest,
            "normalization_policy": "preserve_mode_bundle",
            "source_mode_gradient_policy": "constant_stop_gradient",
            "device_overlap_policy": "forbidden",
            "setup_addressability": "host_addressable",
        }

    @property
    def sha256(self) -> str:
        """Hash the complete source binding contract."""

        encoded = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _bundle_target_fingerprint(bundle: ModeBundle) -> FDTDXFingerprint:
    transfer = bundle.transfer
    if (
        transfer.target_runtime_name != "fdtdx"
        or transfer.target_runtime_version is None
        or transfer.target_source_revision is None
        or transfer.target_source_digest is None
    ):
        raise ContractError("ModeBundle has no complete FDTDX target identity")
    return FDTDXFingerprint(
        package_version=transfer.target_runtime_version,
        source_revision=transfer.target_source_revision,
        source_digest=transfer.target_source_digest,
    )


def build_fdtdx_mode_source_contract(
    bundle: ModeBundle,
    *,
    source_name: str,
    expected_inverse_permittivity: ArrayLike,
    expected_inverse_permeability: object,
    fdtdx: FDTDXFingerprint,
) -> FDTDXModeSourceContract:
    """Snapshot the exact lossless source-plane material and bundle identity."""

    if _bundle_target_fingerprint(bundle) != fdtdx:
        raise ContractError("ModeBundle FDTDX target identity differs from the source contract")
    inverse_permittivity, permittivity_sha256 = _canonical_array(
        expected_inverse_permittivity,
        label="source_plane/inverse_permittivity",
    )
    inverse_permeability, permeability_sha256 = _canonical_array(
        expected_inverse_permeability,
        label="source_plane/inverse_permeability",
    )
    propagation_axis = {"x": 0, "y": 1, "z": 2}[bundle.propagation.axis.value]
    propagation_direction = bundle.propagation.direction.value
    return FDTDXModeSourceContract(
        source_name=source_name,
        grid_shape=bundle.electric.grid.shape,
        frequency_hz=bundle.frequency_hz,
        effective_index=bundle.effective_index,
        field_dtype=str(bundle.electric.values.dtype),
        coordinate_sha256=bundle.electric.grid.coordinate_sha256,
        mode_bundle_sha256=_mode_bundle_sha256(bundle),
        expected_inverse_permittivity=inverse_permittivity,
        expected_inverse_permeability=inverse_permeability,
        inverse_permittivity_sha256=permittivity_sha256,
        inverse_permeability_sha256=permeability_sha256,
        fdtdx=fdtdx,
        propagation_axis=propagation_axis,
        propagation_direction=propagation_direction,
    )


def _validate_bundle_contract(
    bundle: ModeBundle,
    contract: FDTDXModeSourceContract,
) -> None:
    if _bundle_target_fingerprint(bundle) != contract.fdtdx:
        raise ContractError("ModeBundle FDTDX target identity differs from the source contract")
    if _mode_bundle_sha256(bundle) != contract.mode_bundle_sha256:
        raise ContractError("ModeBundle content differs from the source contract")
    if bundle.electric.grid.shape != contract.grid_shape:
        raise ContractError("ModeBundle grid shape differs from the source contract")
    if bundle.electric.grid.coordinate_sha256 != contract.coordinate_sha256:
        raise ContractError("ModeBundle coordinate identity differs from the source contract")
    if str(bundle.electric.values.dtype) != contract.field_dtype:
        raise ContractError("ModeBundle field precision differs from the source contract")
    if bundle.frequency_hz != contract.frequency_hz:
        raise ContractError("ModeBundle frequency differs from the source contract")
    if bundle.effective_index != contract.effective_index:
        raise ContractError("ModeBundle effective index differs from the source contract")


def _runtime_array(values: object, *, label: str) -> Any:
    import numpy as np

    try:
        return np.asarray(values)
    except Exception as error:
        raise ContractError(
            f"{label} is not host-addressable; mode-source v1 requires static eager setup"
        ) from error


def make_fdtdx_mode_source_function(
    bundle: ModeBundle,
    contract: FDTDXModeSourceContract,
) -> Callable[..., tuple[ArrayLike, ArrayLike]]:
    """Return the FDTDX callback with exact source-plane material validation."""

    import numpy as np

    _validate_bundle_contract(bundle, contract)
    mode_function = make_fdtdx_mode_function(bundle)

    def source_mode_function(
        *,
        coordinates: tuple[ArrayLike, ArrayLike, ArrayLike],
        frequency: float,
        propagation_axis: int,
        inv_permittivity: ArrayLike,
        inv_permeability: object,
    ) -> tuple[ArrayLike, ArrayLike]:
        electric, magnetic = mode_function(
            coordinates=coordinates,
            frequency=frequency,
            propagation_axis=propagation_axis,
            inv_permittivity=inv_permittivity,
        )
        actual_inverse_permittivity = _runtime_array(
            inv_permittivity,
            label="FDTDX source-plane inverse permittivity",
        )
        expected_inverse_permittivity = np.asarray(contract.expected_inverse_permittivity)
        if (
            actual_inverse_permittivity.shape != expected_inverse_permittivity.shape
            or actual_inverse_permittivity.dtype != expected_inverse_permittivity.dtype
            or not np.array_equal(actual_inverse_permittivity, expected_inverse_permittivity)
        ):
            raise ContractError(
                "FDTDX source-plane inverse permittivity differs from the mode-source contract"
            )
        actual_inverse_permeability = _runtime_array(
            inv_permeability,
            label="FDTDX source-plane inverse permeability",
        )
        expected_inverse_permeability = np.asarray(contract.expected_inverse_permeability)
        if (
            actual_inverse_permeability.shape != expected_inverse_permeability.shape
            or actual_inverse_permeability.dtype != expected_inverse_permeability.dtype
            or not np.array_equal(actual_inverse_permeability, expected_inverse_permeability)
        ):
            raise ContractError(
                "FDTDX source-plane inverse permeability differs from the mode-source contract"
            )
        return electric, magnetic

    return source_mode_function


def _construct_fdtdx_mode_source(
    bundle: ModeBundle,
    contract: FDTDXModeSourceContract,
    *,
    verified_fingerprint: FDTDXFingerprint,
    temporal_profile: object | None = None,
    allow_profile_updates: bool,
    mode_function: Callable[..., tuple[ArrayLike, ArrayLike]] | None = None,
) -> object:
    """Construct the locked public source with an explicit profile-update policy."""

    _validate_bundle_contract(bundle, contract)
    if verified_fingerprint != contract.fdtdx:
        raise ContractError("verified FDTDX source identity differs from the mode-source contract")
    try:
        installed_version = package_version("fdtdx")
        module = import_module("fdtdx")
    except (ModuleNotFoundError, PackageNotFoundError) as error:
        raise ContractError("FDTDX is not installed in the active environment") from error
    if installed_version != contract.fdtdx.package_version:
        raise ContractError(
            f"FDTDX package version mismatch: expected {contract.fdtdx.package_version!r}, "
            f"got {installed_version!r}"
        )
    source_constructor = getattr(module, "CustomModePlaneSource", None)
    wave_constructor = getattr(module, "WaveCharacter", None)
    if not callable(source_constructor):
        raise ContractError("installed FDTDX does not expose CustomModePlaneSource")
    if not callable(wave_constructor):
        raise ContractError("installed FDTDX does not expose WaveCharacter")
    source_kwargs: dict[str, object] = {
        "name": contract.source_name,
        "partial_grid_shape": contract.grid_shape,
        "wave_character": wave_constructor(frequency=contract.frequency_hz),
        "direction": contract.propagation_direction,
        "mode_function": (
            make_fdtdx_mode_source_function(bundle, contract)
            if mode_function is None
            else mode_function
        ),
        "effective_index": contract.effective_index,
        "normalize": False,
        "allow_device_overlap": False,
    }
    if allow_profile_updates:
        source_kwargs["allow_profile_updates"] = True
    if temporal_profile is not None:
        source_kwargs["temporal_profile"] = temporal_profile
    constructor = cast(Callable[..., object], source_constructor)
    return constructor(**source_kwargs)


def make_fdtdx_mode_source(
    bundle: ModeBundle,
    contract: FDTDXModeSourceContract,
    *,
    verified_fingerprint: FDTDXFingerprint,
    temporal_profile: object | None = None,
) -> object:
    """Construct the locked static ``CustomModePlaneSource`` without fallback."""

    return _construct_fdtdx_mode_source(
        bundle,
        contract,
        verified_fingerprint=verified_fingerprint,
        temporal_profile=temporal_profile,
        allow_profile_updates=False,
    )


def validate_fdtdx_mode_source(
    source: object,
    bundle: ModeBundle,
    contract: FDTDXModeSourceContract,
) -> None:
    """Validate a placed and applied FDTDX source against its complete binding."""

    import numpy as np

    _validate_bundle_contract(bundle, contract)
    if getattr(source, "name", None) != contract.source_name:
        raise ContractError("placed FDTDX mode source name differs from the contract")
    if tuple(getattr(source, "grid_shape", ())) != contract.grid_shape:
        raise ContractError("placed FDTDX mode source shape differs from the contract")
    if getattr(source, "propagation_axis", None) != contract.propagation_axis:
        raise ContractError("placed FDTDX mode source axis differs from the contract")
    if getattr(source, "direction", None) != contract.propagation_direction:
        raise ContractError("placed FDTDX mode source direction differs from the contract")
    if getattr(source, "normalize", None) is not False:
        raise ContractError("placed FDTDX mode source changed the ModeBundle normalization")
    if getattr(source, "allow_device_overlap", None) is not False:
        raise ContractError("placed FDTDX mode source permits an unvalidated Device overlap")
    wave_character = getattr(source, "wave_character", None)
    get_frequency = getattr(wave_character, "get_frequency", None)
    if not callable(get_frequency) or get_frequency() != contract.frequency_hz:
        raise ContractError("placed FDTDX mode source frequency differs from the contract")

    for label, actual, expected in (
        ("electric field", getattr(source, "_E", None), bundle.electric.values),
        ("magnetic field", getattr(source, "_H", None), bundle.magnetic.values),
        (
            "inverse permittivity",
            getattr(source, "_inv_permittivity", None),
            contract.expected_inverse_permittivity,
        ),
        (
            "inverse permeability",
            getattr(source, "_inv_permeability", None),
            contract.expected_inverse_permeability,
        ),
    ):
        actual_array = _runtime_array(actual, label=f"placed FDTDX mode source {label}")
        expected_array = np.asarray(expected)
        if (
            actual_array.shape != expected_array.shape
            or actual_array.dtype != expected_array.dtype
            or not np.array_equal(actual_array, expected_array)
        ):
            raise ContractError(f"placed FDTDX mode source {label} differs from the contract")

    config = getattr(source, "_config", None)
    grid = getattr(config, "resolved_grid", None)
    grid_slice_tuple = getattr(source, "grid_slice_tuple", None)
    if grid is None or not isinstance(grid_slice_tuple, tuple) or len(grid_slice_tuple) != 3:
        raise ContractError("placed FDTDX mode source has no resolved three-axis grid")
    for axis, expected_edges in enumerate(bundle.electric.grid.edge_coordinates):
        lower, upper = grid_slice_tuple[axis]
        actual_edges = _runtime_array(
            grid.edges(axis)[lower : upper + 1],
            label=f"placed FDTDX mode source axis {axis}",
        )
        expected_array = np.asarray(expected_edges)
        if actual_edges.dtype != expected_array.dtype or not np.array_equal(
            actual_edges,
            expected_array,
        ):
            raise ContractError(f"placed FDTDX mode source edge coordinates differ on axis {axis}")


__all__ = [
    "FDTDXModeSourceContract",
    "build_fdtdx_mode_source_contract",
    "make_fdtdx_mode_source",
    "make_fdtdx_mode_source_function",
    "validate_fdtdx_mode_source",
]
