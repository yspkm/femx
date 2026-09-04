"""Explicit global-sharding boundary for an imported FDTDX mode source.

FDTDX places its full-domain fields and source-plane material on a global
``NamedSharding``.  A canonical :class:`~femx.interop.fdtdx.ModeBundle`, however, is an
ordinary host artifact.  This module makes that transition explicit: every controller must own
the same complete, hash-bound mode snapshot, while JAX materializes only the shards addressable
by that controller.  No global field is gathered back to a host.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from femx.core.arrays import ArrayLike, shape_of
from femx.core.errors import ContractError
from femx.interop.fdtdx.mode_bundle import ModeBundle
from femx.interop.fdtdx.mode_source import (
    FDTDXModeSourceContract,
    _construct_fdtdx_mode_source,
    _runtime_array,
    _validate_bundle_contract,
)
from femx.interop.fdtdx.mode_transfer import make_fdtdx_mode_function
from femx.interop.fdtdx.thermo_optic import FDTDXFingerprint

_DISTRIBUTED_MODE_SOURCE_SCHEMA = "femx.fdtdx.distributed_mode_source/v1"
_PROFILE_DISTRIBUTION = "identical_full_snapshot_per_process"
_EXECUTION_POLICY = "outer_jit_with_arrays_objects_config_as_arguments"


def _canonical_digest(data: Mapping[str, object]) -> str:
    payload = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: str, *, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ContractError(f"{label} must be a canonical lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class FDTDXDistributedModeSourceBinding:
    """Process-local record of one globally sharded imported-mode binding.

    A record made on one process is not multi-host evidence.  A physical claim additionally
    requires one mutually consistent record from every initialized JAX process.
    """

    source_name: str
    source_contract_sha256: str
    mesh_axis_name: str
    partition_spec: tuple[str, str, str, str]
    global_shape: tuple[int, int, int, int]
    field_dtype: str
    time_offset_dtype: str
    global_device_count: int
    local_device_count: int
    process_count: int
    process_index: int
    addressable_x_ranges: tuple[tuple[int, int], ...]
    schema_version: str = _DISTRIBUTED_MODE_SOURCE_SCHEMA

    def __post_init__(self) -> None:
        if not self.source_name or self.source_name.strip() != self.source_name:
            raise ContractError("distributed FDTDX source name must be non-empty and trimmed")
        _require_sha256(
            self.source_contract_sha256,
            label="distributed FDTDX source-contract digest",
        )
        if not self.mesh_axis_name or self.mesh_axis_name.strip() != self.mesh_axis_name:
            raise ContractError("distributed FDTDX mesh axis name must be non-empty and trimmed")
        expected_spec = ("replicated", self.mesh_axis_name, "replicated", "replicated")
        if self.partition_spec != expected_spec:
            raise ContractError("distributed FDTDX source must partition the first spatial axis")
        if len(self.global_shape) != 4 or any(size <= 0 for size in self.global_shape):
            raise ContractError("distributed FDTDX source shape must contain four positive sizes")
        if self.field_dtype not in {"complex64", "complex128"}:
            raise ContractError(
                "distributed FDTDX source field dtype must be complex64 or complex128"
            )
        if self.time_offset_dtype not in {"float32", "float64"}:
            raise ContractError("distributed FDTDX time-offset dtype must be float32 or float64")
        if (
            self.global_device_count <= 0
            or self.local_device_count <= 0
            or self.process_count <= 0
            or not 0 <= self.process_index < self.process_count
        ):
            raise ContractError("distributed FDTDX runtime counts are inconsistent")
        if self.local_device_count > self.global_device_count:
            raise ContractError("distributed FDTDX local device count exceeds the global count")
        if len(self.addressable_x_ranges) != self.local_device_count:
            raise ContractError("distributed FDTDX source requires one local x range per device")
        previous_stop = -1
        for start, stop in self.addressable_x_ranges:
            if not 0 <= start < stop <= self.global_shape[1] or start < previous_stop:
                raise ContractError("distributed FDTDX addressable x ranges are invalid")
            previous_stop = stop
        if self.schema_version != _DISTRIBUTED_MODE_SOURCE_SCHEMA:
            raise ContractError(
                f"unsupported distributed FDTDX source schema {self.schema_version!r}"
            )

    def canonical_data(self) -> Mapping[str, object]:
        """Return deterministic process-local binding metadata."""

        return {
            "schema_version": self.schema_version,
            "source_name": self.source_name,
            "source_contract_sha256": self.source_contract_sha256,
            "mesh_axis_name": self.mesh_axis_name,
            "partition_spec": list(self.partition_spec),
            "global_shape": list(self.global_shape),
            "field_dtype": self.field_dtype,
            "time_offset_dtype": self.time_offset_dtype,
            "global_device_count": self.global_device_count,
            "local_device_count": self.local_device_count,
            "process_count": self.process_count,
            "process_index": self.process_index,
            "addressable_x_ranges": [list(item) for item in self.addressable_x_ranges],
            "profile_distribution": _PROFILE_DISTRIBUTION,
            "execution_policy": _EXECUTION_POLICY,
            "physical_evidence": False,
        }

    @property
    def sha256(self) -> str:
        """Return the canonical digest of this process-local binding record."""

        return _canonical_digest(self.canonical_data())


def _named_sharding(values: object, *, mesh_axis_name: str, label: str) -> Any:
    import jax
    from jax.sharding import NamedSharding, PartitionSpec

    if not isinstance(values, jax.Array) or not isinstance(values.sharding, NamedSharding):
        raise ContractError(f"{label} must use JAX NamedSharding")
    sharding = values.sharding
    if tuple(sharding.mesh.axis_names) != (mesh_axis_name,):
        raise ContractError(f"{label} mesh axes differ from the declared axis {mesh_axis_name!r}")
    expected_spec = PartitionSpec(  # type: ignore[no-untyped-call]
        None,
        mesh_axis_name,
        None,
        None,
    )
    if sharding.spec != expected_spec:
        raise ContractError(f"{label} must shard only its first spatial axis")
    if len(sharding.device_set) != jax.device_count():
        raise ContractError(f"{label} sharding does not cover every global JAX device")
    if len(sharding.addressable_devices) != jax.local_device_count():
        raise ContractError(f"{label} sharding does not cover every local JAX device")
    if values.ndim != 4 or values.shape[1] % jax.device_count() != 0:
        raise ContractError(f"{label} first spatial axis is not divisible by the device count")
    return sharding


def _normalized_index(index: Sequence[object], shape: Sequence[int]) -> tuple[slice, ...]:
    result: list[slice] = []
    if len(index) != len(shape):
        raise ContractError("distributed FDTDX shard index rank differs from its array")
    for axis, (item, size) in enumerate(zip(index, shape, strict=True)):
        if not isinstance(item, slice):
            raise ContractError(f"distributed FDTDX shard index on axis {axis} is not a slice")
        start, stop, step = item.indices(size)
        if step != 1 or start >= stop:
            raise ContractError(f"distributed FDTDX shard slice on axis {axis} is invalid")
        result.append(slice(start, stop, 1))
    return tuple(result)


def _validate_addressable_snapshot(
    values: object,
    expected: object,
    *,
    mesh_axis_name: str,
    label: str,
) -> tuple[Any, tuple[tuple[int, int], ...]]:
    import numpy as np

    expected_array = np.asarray(expected)
    actual = cast(Any, values)
    if shape_of(cast(ArrayLike, values)) != expected_array.shape:
        raise ContractError(f"{label} shape differs from the distributed source contract")
    actual_dtype = getattr(values, "dtype", None)
    if actual_dtype != expected_array.dtype:
        raise ContractError(f"{label} dtype differs from the distributed source contract")
    sharding = _named_sharding(values, mesh_axis_name=mesh_axis_name, label=label)
    shards = tuple(actual.addressable_shards)
    if len(shards) != len(sharding.addressable_devices):
        raise ContractError(f"{label} has incomplete process-local shard addressability")
    ranges: list[tuple[int, int]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for shard in shards:
        index = _normalized_index(shard.index, expected_array.shape)
        token = tuple((item.start or 0, item.stop or 0) for item in index)
        if token in seen:
            raise ContractError(f"{label} unexpectedly replicates a process-local shard")
        seen.add(token)
        actual_local = np.asarray(shard.data)
        expected_local = expected_array[index]
        if actual_local.dtype != expected_local.dtype or not np.array_equal(
            actual_local,
            expected_local,
        ):
            raise ContractError(f"{label} addressable shard differs from the source contract")
        ranges.append((index[1].start or 0, index[1].stop or 0))
    return sharding, tuple(sorted(ranges))


def _global_array_from_identical_snapshot(snapshot: object, sharding: Any, *, label: str) -> Any:
    import jax
    import numpy as np

    host_array = np.asarray(snapshot)
    try:
        result = jax.make_array_from_process_local_data(  # type: ignore[no-untyped-call]
            sharding,
            host_array,
            global_shape=host_array.shape,
        )
    except (TypeError, ValueError) as error:
        raise ContractError(
            f"cannot distribute {label} from the identical host snapshot"
        ) from error
    if result.shape != host_array.shape or result.dtype != host_array.dtype:
        raise ContractError(f"distributed {label} changed shape or dtype")
    return result


def make_fdtdx_distributed_mode_source_function(
    bundle: ModeBundle,
    contract: FDTDXModeSourceContract,
    *,
    mesh_axis_name: str = "shard",
) -> Callable[..., tuple[ArrayLike, ArrayLike]]:
    """Return a callback that materializes E/H directly on FDTDX's global sharding."""

    import numpy as np

    _validate_bundle_contract(bundle, contract)
    if not mesh_axis_name or mesh_axis_name.strip() != mesh_axis_name:
        raise ContractError("distributed FDTDX mesh axis name must be non-empty and trimmed")
    validate_callback = make_fdtdx_mode_function(bundle)

    def source_mode_function(
        *,
        coordinates: tuple[ArrayLike, ArrayLike, ArrayLike],
        frequency: float,
        propagation_axis: int,
        inv_permittivity: ArrayLike,
        inv_permeability: object,
    ) -> tuple[ArrayLike, ArrayLike]:
        validate_callback(
            coordinates=coordinates,
            frequency=frequency,
            propagation_axis=propagation_axis,
            inv_permittivity=inv_permittivity,
        )
        sharding, _ranges = _validate_addressable_snapshot(
            inv_permittivity,
            contract.expected_inverse_permittivity,
            mesh_axis_name=mesh_axis_name,
            label="FDTDX distributed source-plane inverse permittivity",
        )
        actual_inverse_permeability = _runtime_array(
            inv_permeability,
            label="FDTDX distributed source-plane inverse permeability",
        )
        expected_inverse_permeability = np.asarray(contract.expected_inverse_permeability)
        if (
            actual_inverse_permeability.shape != expected_inverse_permeability.shape
            or actual_inverse_permeability.dtype != expected_inverse_permeability.dtype
            or not np.array_equal(actual_inverse_permeability, expected_inverse_permeability)
        ):
            raise ContractError(
                "FDTDX distributed source-plane inverse permeability differs from the contract"
            )
        electric = _global_array_from_identical_snapshot(
            bundle.electric.values,
            sharding,
            label="mode electric field",
        )
        magnetic = _global_array_from_identical_snapshot(
            bundle.magnetic.values,
            sharding,
            label="mode magnetic field",
        )
        return electric, magnetic

    return source_mode_function


def make_fdtdx_distributed_mode_source(
    bundle: ModeBundle,
    contract: FDTDXModeSourceContract,
    *,
    verified_fingerprint: FDTDXFingerprint,
    temporal_profile: object | None = None,
    mesh_axis_name: str = "shard",
) -> object:
    """Construct the locked static source with an explicit global-sharding callback."""

    mode_function = make_fdtdx_distributed_mode_source_function(
        bundle,
        contract,
        mesh_axis_name=mesh_axis_name,
    )
    return _construct_fdtdx_mode_source(
        bundle,
        contract,
        verified_fingerprint=verified_fingerprint,
        temporal_profile=temporal_profile,
        allow_profile_updates=False,
        mode_function=mode_function,
    )


def bind_fdtdx_distributed_mode_source(
    objects: object,
    bundle: ModeBundle,
    contract: FDTDXModeSourceContract,
    *,
    mesh_axis_name: str = "shard",
) -> tuple[object, FDTDXDistributedModeSourceBinding]:
    """Shard the placed source's time offsets and return an auditable local record.

    ``objects`` must be passed as an explicit argument to an outer ``jax.jit(run_fdtd)`` call
    after this binding.  Closing over the container would turn global arrays into constants and
    is not an admitted multi-controller execution path.
    """

    import jax
    import numpy as np

    _validate_bundle_contract(bundle, contract)
    getitem = getattr(objects, "__getitem__", None)
    index_method = getattr(objects, "index", None)
    aset = getattr(objects, "aset", None)
    if not callable(getitem) or not callable(index_method) or not callable(aset):
        raise ContractError("distributed FDTDX binding requires an immutable ObjectContainer")
    source = getitem(contract.source_name)
    if getattr(source, "name", None) != contract.source_name:
        raise ContractError("distributed FDTDX source name differs from the contract")
    if tuple(getattr(source, "grid_shape", ())) != contract.grid_shape:
        raise ContractError("distributed FDTDX source shape differs from the contract")
    if getattr(source, "propagation_axis", None) != contract.propagation_axis:
        raise ContractError("distributed FDTDX source axis differs from the contract")
    if getattr(source, "direction", None) != contract.propagation_direction:
        raise ContractError("distributed FDTDX source direction differs from the contract")
    if getattr(source, "normalize", None) is not False:
        raise ContractError("distributed FDTDX source changed the ModeBundle normalization")
    if getattr(source, "allow_device_overlap", None) is not False:
        raise ContractError("distributed FDTDX source permits an unvalidated Device overlap")
    wave_character = getattr(source, "wave_character", None)
    get_frequency = getattr(wave_character, "get_frequency", None)
    if not callable(get_frequency) or get_frequency() != contract.frequency_hz:
        raise ContractError("distributed FDTDX source frequency differs from the contract")

    actual_inverse_permeability = _runtime_array(
        getattr(source, "_inv_permeability", None),
        label="distributed FDTDX source inverse permeability",
    )
    expected_inverse_permeability = np.asarray(contract.expected_inverse_permeability)
    if (
        actual_inverse_permeability.shape != expected_inverse_permeability.shape
        or actual_inverse_permeability.dtype != expected_inverse_permeability.dtype
        or not np.array_equal(actual_inverse_permeability, expected_inverse_permeability)
    ):
        raise ContractError(
            "distributed FDTDX source inverse permeability differs from the contract"
        )

    config = getattr(source, "_config", None)
    grid = getattr(config, "resolved_grid", None)
    grid_slice_tuple = getattr(source, "grid_slice_tuple", None)
    if grid is None or not isinstance(grid_slice_tuple, tuple) or len(grid_slice_tuple) != 3:
        raise ContractError("distributed FDTDX source has no resolved three-axis grid")
    for axis, expected_edges in enumerate(bundle.electric.grid.edge_coordinates):
        lower, upper = grid_slice_tuple[axis]
        actual_edges = _runtime_array(
            grid.edges(axis)[lower : upper + 1],
            label=f"distributed FDTDX source axis {axis}",
        )
        expected_edges_array = np.asarray(expected_edges)
        if actual_edges.dtype != expected_edges_array.dtype or not np.array_equal(
            actual_edges,
            expected_edges_array,
        ):
            raise ContractError(f"distributed FDTDX source edge coordinates differ on axis {axis}")

    source_sharding, addressable_x_ranges = _validate_addressable_snapshot(
        getattr(source, "_E", None),
        bundle.electric.values,
        mesh_axis_name=mesh_axis_name,
        label="distributed FDTDX source electric field",
    )
    magnetic_sharding, magnetic_ranges = _validate_addressable_snapshot(
        getattr(source, "_H", None),
        bundle.magnetic.values,
        mesh_axis_name=mesh_axis_name,
        label="distributed FDTDX source magnetic field",
    )
    material_sharding, material_ranges = _validate_addressable_snapshot(
        getattr(source, "_inv_permittivity", None),
        contract.expected_inverse_permittivity,
        mesh_axis_name=mesh_axis_name,
        label="distributed FDTDX source inverse permittivity",
    )
    if (
        magnetic_sharding != source_sharding
        or material_sharding != source_sharding
        or magnetic_ranges != addressable_x_ranges
        or material_ranges != addressable_x_ranges
    ):
        raise ContractError("distributed FDTDX source leaves do not share one global sharding")

    offset_e_snapshot = _runtime_array(
        getattr(source, "_time_offset_E", None),
        label="distributed FDTDX electric time offset",
    )
    offset_h_snapshot = _runtime_array(
        getattr(source, "_time_offset_H", None),
        label="distributed FDTDX magnetic time offset",
    )
    expected_shape = shape_of(bundle.electric.values)
    if offset_e_snapshot.shape != expected_shape or offset_h_snapshot.shape != expected_shape:
        raise ContractError("distributed FDTDX source time-offset shape differs from its fields")
    if offset_e_snapshot.dtype != offset_h_snapshot.dtype or offset_e_snapshot.dtype not in (
        np.dtype("float32"),
        np.dtype("float64"),
    ):
        raise ContractError("distributed FDTDX source time offsets have incompatible precision")
    sharded_offset_e = _global_array_from_identical_snapshot(
        offset_e_snapshot,
        source_sharding,
        label="electric time offset",
    )
    sharded_offset_h = _global_array_from_identical_snapshot(
        offset_h_snapshot,
        source_sharding,
        label="magnetic time offset",
    )
    source_aset = getattr(source, "aset", None)
    if not callable(source_aset):
        raise ContractError("distributed FDTDX source does not support immutable replacement")
    source = source_aset("_time_offset_E", sharded_offset_e, create_new_ok=True)
    source = source.aset("_time_offset_H", sharded_offset_h, create_new_ok=True)
    source_index = index_method(contract.source_name)
    updated_objects = aset(f"object_list->[{source_index}]", source)

    partition_spec = ("replicated", mesh_axis_name, "replicated", "replicated")
    binding = FDTDXDistributedModeSourceBinding(
        source_name=contract.source_name,
        source_contract_sha256=contract.sha256,
        mesh_axis_name=mesh_axis_name,
        partition_spec=partition_spec,
        global_shape=cast(tuple[int, int, int, int], expected_shape),
        field_dtype=str(source._E.dtype),
        time_offset_dtype=str(sharded_offset_e.dtype),
        global_device_count=jax.device_count(),
        local_device_count=jax.local_device_count(),
        process_count=jax.process_count(),
        process_index=jax.process_index(),
        addressable_x_ranges=addressable_x_ranges,
    )
    return updated_objects, binding


__all__ = [
    "FDTDXDistributedModeSourceBinding",
    "bind_fdtdx_distributed_mode_source",
    "make_fdtdx_distributed_mode_source",
    "make_fdtdx_distributed_mode_source_function",
]
