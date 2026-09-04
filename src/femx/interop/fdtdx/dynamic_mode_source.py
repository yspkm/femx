"""Explicit JAX-valued mode-profile binding for the locked FDTDX source."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

from femx.core.arrays import ArrayLike, shape_of
from femx.core.errors import ContractError
from femx.interop.fdtdx.mode_bundle import ModeBundle
from femx.interop.fdtdx.mode_source import (
    FDTDXModeSourceContract,
    _construct_fdtdx_mode_source,
    _validate_bundle_contract,
)
from femx.interop.fdtdx.thermo_optic import FDTDXFingerprint

_DYNAMIC_MODE_SOURCE_SCHEMA = "femx.fdtdx.dynamic_mode_source/v1"


@dataclass(frozen=True, slots=True)
class FDTDXDynamicModeSourceContract:
    """Static identity around a JAX-valued E/H and effective-index source profile."""

    baseline: FDTDXModeSourceContract
    parameter_names: tuple[str, ...]
    parameter_units: tuple[str, ...]
    transfer_operator_sha256: str
    target_power_watts: float
    schema_version: str = _DYNAMIC_MODE_SOURCE_SCHEMA

    def __post_init__(self) -> None:
        if not self.parameter_names:
            raise ContractError("dynamic FDTDX mode source requires at least one parameter")
        if len(self.parameter_names) != len(self.parameter_units):
            raise ContractError("dynamic FDTDX mode-source parameter names and units must align")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ContractError("dynamic FDTDX mode-source parameter names must be unique")
        for name in self.parameter_names:
            if not name or name.strip() != name:
                raise ContractError(
                    "dynamic FDTDX mode-source parameter names must be non-empty and trimmed"
                )
        for unit in self.parameter_units:
            if not unit or unit.strip() != unit:
                raise ContractError(
                    "dynamic FDTDX mode-source parameter units must be non-empty and trimmed"
                )
        if len(self.transfer_operator_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.transfer_operator_sha256
        ):
            raise ContractError("dynamic FDTDX mode-source transfer identity must be a SHA-256")
        if not math.isfinite(self.target_power_watts) or self.target_power_watts <= 0.0:
            raise ContractError(
                "dynamic FDTDX mode-source target power must be finite and positive"
            )
        if self.schema_version != _DYNAMIC_MODE_SOURCE_SCHEMA:
            raise ContractError(
                f"unsupported dynamic FDTDX mode-source schema {self.schema_version!r}"
            )

    def canonical_data(self) -> Mapping[str, object]:
        """Return deterministic metadata without any parameter-dependent field arrays."""

        return {
            "schema_version": self.schema_version,
            "baseline_source_contract_sha256": self.baseline.sha256,
            "source_name": self.baseline.source_name,
            "grid_shape_xyz": list(self.baseline.grid_shape),
            "coordinate_sha256": self.baseline.coordinate_sha256,
            "field_dtype": self.baseline.field_dtype,
            "frequency_hz": self.baseline.frequency_hz,
            "parameter_names": list(self.parameter_names),
            "parameter_units": list(self.parameter_units),
            "transfer_operator_sha256": self.transfer_operator_sha256,
            "target_power_watts": self.target_power_watts,
            "normalization_policy": "differentiable_fem_to_yee_target_power",
            "source_mode_gradient_policy": "dynamic_profile_checkpointed_reverse",
            "effective_index_policy": "dynamic_rebuild_yee_time_offsets",
            "source_plane_medium_policy": "fixed_baseline_snapshot",
            "device_overlap_policy": "forbidden",
            "reversible_gradient_policy": "unsupported_source_object_cotangent",
        }

    @property
    def sha256(self) -> str:
        """Hash the complete dynamic-profile contract."""

        encoded = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def build_fdtdx_dynamic_mode_source_contract(
    bundle: ModeBundle,
    baseline: FDTDXModeSourceContract,
    *,
    parameter_names: tuple[str, ...],
    parameter_units: tuple[str, ...],
) -> FDTDXDynamicModeSourceContract:
    """Bind a baseline bundle to one parameterized in-memory source-profile path."""

    _validate_bundle_contract(bundle, baseline)
    if bundle.transfer.operator_sha256 is None:
        raise ContractError("dynamic FDTDX mode source requires a transfer-operator identity")
    return FDTDXDynamicModeSourceContract(
        baseline=baseline,
        parameter_names=parameter_names,
        parameter_units=parameter_units,
        transfer_operator_sha256=bundle.transfer.operator_sha256,
        target_power_watts=bundle.normalization.target_power_watts,
    )


def make_fdtdx_dynamic_mode_source(
    bundle: ModeBundle,
    contract: FDTDXDynamicModeSourceContract,
    *,
    verified_fingerprint: FDTDXFingerprint,
    temporal_profile: object | None = None,
) -> object:
    """Construct the locked source with explicit runtime profile updates enabled."""

    _validate_bundle_contract(bundle, contract.baseline)
    source = _construct_fdtdx_mode_source(
        bundle,
        contract.baseline,
        verified_fingerprint=verified_fingerprint,
        temporal_profile=temporal_profile,
        allow_profile_updates=True,
    )
    if not callable(getattr(source, "with_mode_profile", None)):
        raise ContractError("installed FDTDX source has no public dynamic-profile method")
    return source


def with_fdtdx_dynamic_mode_profile(
    source: object,
    contract: FDTDXDynamicModeSourceContract,
    *,
    electric_v_per_m: ArrayLike,
    magnetic_eta0_v_per_m: ArrayLike,
    effective_index: object,
) -> object:
    """Return a placed source rebound to JAX-valued fields without host conversion."""

    baseline = contract.baseline
    if getattr(source, "name", None) != baseline.source_name:
        raise ContractError("dynamic FDTDX mode-source name differs from its contract")
    if tuple(getattr(source, "grid_shape", ())) != baseline.grid_shape:
        raise ContractError("dynamic FDTDX mode-source shape differs from its contract")
    if getattr(source, "propagation_axis", None) != baseline.propagation_axis:
        raise ContractError("dynamic FDTDX mode-source axis differs from its contract")
    if getattr(source, "direction", None) != baseline.propagation_direction:
        raise ContractError("dynamic FDTDX mode-source direction differs from its contract")
    if getattr(source, "normalize", None) is not False:
        raise ContractError("dynamic FDTDX mode source must preserve femx normalization")
    if getattr(source, "allow_device_overlap", None) is not False:
        raise ContractError("dynamic FDTDX mode source cannot overlap a Device")
    if getattr(source, "allow_profile_updates", None) is not True:
        raise ContractError("dynamic FDTDX mode source did not enable profile updates")
    expected_shape = (3, *baseline.grid_shape)
    for label, values in (
        ("electric", electric_v_per_m),
        ("magnetic", magnetic_eta0_v_per_m),
    ):
        if shape_of(values) != expected_shape:
            raise ContractError(f"dynamic FDTDX {label} profile must have shape {expected_shape}")
        if str(values.dtype) != baseline.field_dtype:
            raise ContractError(
                f"dynamic FDTDX {label} profile precision differs from its contract"
            )
    effective_index_shape = tuple(getattr(effective_index, "shape", ()))
    if effective_index_shape:
        raise ContractError("dynamic FDTDX effective index must be a scalar")
    effective_index_dtype = str(getattr(effective_index, "dtype", ""))
    expected_real_dtype = "float32" if baseline.field_dtype == "complex64" else "float64"
    if effective_index_dtype not in {baseline.field_dtype, expected_real_dtype}:
        raise ContractError("dynamic FDTDX effective-index precision differs from its contract")
    update = getattr(source, "with_mode_profile", None)
    if not callable(update):
        raise ContractError("installed FDTDX source has no public dynamic-profile method")
    return update(
        mode_E=electric_v_per_m,
        mode_H=magnetic_eta0_v_per_m,
        effective_index=effective_index,
    )


__all__ = [
    "FDTDXDynamicModeSourceContract",
    "build_fdtdx_dynamic_mode_source_contract",
    "make_fdtdx_dynamic_mode_source",
    "with_fdtdx_dynamic_mode_profile",
]
