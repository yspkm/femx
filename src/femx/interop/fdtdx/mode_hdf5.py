"""Durable HDF5 codec for the versioned FDTDX ``ModeBundle`` boundary."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

from femx.artifacts import ArtifactRef, ArtifactRole, sha256_file
from femx.core.axes import Axis, AxisDirection, Direction
from femx.core.errors import ArtifactError, ContractError
from femx.interop.fdtdx.mode_bundle import (
    FieldRepresentation,
    MagneticFieldConvention,
    ModeBundle,
    ModeNormalization,
    NormalizationKind,
    SolverFingerprint,
    TransferReport,
    YeeFieldKind,
    YeeVectorField,
)
from femx.interop.fdtdx.mode_transfer import build_yee_grid

MODE_HDF5_SCHEMA: Final = "femx.mode.hdf5/v1"
MODE_HDF5_MEDIA_TYPE: Final = "application/x-hdf5; profile=femx.mode.v1"
DEFAULT_MAXIMUM_MODE_DATA_BYTES: Final = 2 * 1024**3

_CONTENT_SCHEMA = b"femx.mode.hdf5.content/v1"
_ARRAY_SCHEMA = b"femx.mode.hdf5.array/v1"
_MAXIMUM_METADATA_BYTES = 1024 * 1024
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ROOT_MEMBERS = frozenset(("fields", "grid", "metadata_json"))
_GRID_MEMBERS = ("x_edges_m", "y_edges_m", "z_edges_m")
_FIELD_MEMBERS = ("electric", "magnetic")


@dataclass(frozen=True, slots=True)
class ModeBundleHDF5Artifact:
    """A verified HDF5 file, its logical content identity, and decoded mode."""

    bundle: ModeBundle
    reference: ArtifactRef
    content_sha256: str
    logical_data_bytes: int
    container_schema: str = MODE_HDF5_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.bundle, ModeBundle):
            raise ArtifactError("mode HDF5 artifact must contain a ModeBundle")
        if self.reference.role is not ArtifactRole.CANONICAL_NUMERICAL:
            raise ArtifactError("mode HDF5 artifact must have the canonical numerical role")
        if self.reference.media_type != MODE_HDF5_MEDIA_TYPE:
            raise ArtifactError("mode HDF5 artifact has the wrong media type")
        if not _SHA256_PATTERN.fullmatch(self.content_sha256):
            raise ArtifactError("mode HDF5 content digest must be a lowercase SHA-256")
        if self.logical_data_bytes <= 0:
            raise ArtifactError("mode HDF5 logical data size must be positive")
        if self.container_schema != MODE_HDF5_SCHEMA:
            raise ArtifactError(f"unsupported mode HDF5 schema {self.container_schema!r}")


def _numeric_modules() -> tuple[Any, Any]:
    try:
        return import_module("h5py"), import_module("numpy")
    except ImportError as exc:
        raise ArtifactError(
            "ModeBundle HDF5 support requires the optional 'femx[artifacts]' dependencies"
        ) from exc


def _require_safe_relative_path(relative_path: str) -> PurePosixPath:
    parsed = PurePosixPath(relative_path)
    if (
        not relative_path
        or relative_path.strip() != relative_path
        or parsed.as_posix() != relative_path
        or "\\" in relative_path
        or parsed.is_absolute()
        or any(part in ("", ".", "..") for part in parsed.parts)
    ):
        raise ArtifactError(f"mode artifact path must be safe and relative: {relative_path!r}")
    if parsed.suffix.lower() not in (".h5", ".hdf5"):
        raise ArtifactError("mode HDF5 artifact path must end in .h5 or .hdf5")
    return parsed


def _resolve_run_path(
    run_root: Path,
    relative_path: str,
    *,
    create_parent: bool,
) -> Path:
    if not run_root.is_dir():
        raise ArtifactError(f"mode artifact run root is not a directory: {run_root}")
    root = run_root.resolve(strict=True)
    parsed = _require_safe_relative_path(relative_path)
    target = root.joinpath(*parsed.parts)
    if create_parent:
        target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.parent.resolve(strict=True).relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise ArtifactError("mode artifact path escapes its run root") from exc
    return target


def _canonical_axis(np: Any, values: Any) -> Any:
    array = np.asarray(values)
    if array.ndim != 1 or array.size < 2 or array.dtype.kind not in "fiu":
        raise ArtifactError("mode HDF5 edge coordinates must be real one-dimensional arrays")
    result = np.array(array, dtype=np.dtype("<f8"), order="C", copy=True)
    if not np.isfinite(result).all() or np.any(np.diff(result) <= 0.0):
        raise ArtifactError("mode HDF5 edge coordinates must be finite and strictly increasing")
    result.setflags(write=False)
    return result


def _canonical_field(np: Any, values: Any) -> Any:
    array = np.asarray(values)
    if array.dtype.kind != "c" or array.dtype.itemsize not in (8, 16):
        raise ArtifactError("mode HDF5 fields require complex64 or complex128 values")
    dtype = np.dtype("<c8" if array.dtype.itemsize == 8 else "<c16")
    result = np.array(array, dtype=dtype, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ArtifactError("mode HDF5 fields must contain only finite values")
    result.setflags(write=False)
    return result


def _framed_update(hasher: Any, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, byteorder="little", signed=False))
    hasher.update(value)


def _array_sha256(name: str, values: Any) -> str:
    hasher = hashlib.sha256()
    hasher.update(_ARRAY_SCHEMA)
    _framed_update(hasher, name.encode("ascii"))
    _framed_update(hasher, values.dtype.str.encode("ascii"))
    _framed_update(hasher, json.dumps(list(values.shape), separators=(",", ":")).encode("ascii"))
    _framed_update(hasher, values.tobytes(order="C"))
    return hasher.hexdigest()


def _array_record(name: str, values: Any) -> dict[str, object]:
    return {
        "path": name,
        "dtype": values.dtype.str,
        "shape": list(values.shape),
        "sha256": _array_sha256(name, values),
    }


def _transfer_metadata(transfer: TransferReport) -> dict[str, object]:
    return {
        "source_representation": transfer.source_representation.value,
        "target_representation": transfer.target_representation.value,
        "operator_sha256": transfer.operator_sha256,
        "relative_power_error": transfer.relative_power_error,
        "relative_interpolation_error": transfer.relative_interpolation_error,
        "source_power_watts": transfer.source_power_watts,
        "pre_correction_power_watts": transfer.pre_correction_power_watts,
        "relative_pre_correction_power_error": transfer.relative_pre_correction_power_error,
        "transferred_power_watts": transfer.transferred_power_watts,
        "power_correction_scale": transfer.power_correction_scale,
        "target_runtime_name": transfer.target_runtime_name,
        "target_runtime_version": transfer.target_runtime_version,
        "target_source_revision": transfer.target_source_revision,
        "target_source_digest": transfer.target_source_digest,
    }


def _mode_metadata(
    bundle: ModeBundle,
    axes: Sequence[Any],
    electric: Any,
    magnetic: Any,
) -> dict[str, object]:
    return {
        "schema_version": bundle.schema_version,
        "frequency_hz": bundle.frequency_hz,
        "effective_index": {
            "real": bundle.effective_index.real,
            "imag": bundle.effective_index.imag,
        },
        "beta_per_m": {"real": bundle.beta_per_m.real, "imag": bundle.beta_per_m.imag},
        "propagation": {
            "axis": bundle.propagation.axis.value,
            "direction": bundle.propagation.direction.value,
        },
        "magnetic_convention": bundle.magnetic_convention.value,
        "normalization": {
            "kind": bundle.normalization.kind.value,
            "target_power_watts": bundle.normalization.target_power_watts,
            "phase_reference": bundle.normalization.phase_reference,
        },
        "solver": {
            "name": bundle.solver.name,
            "version": bundle.solver.version,
            "config_sha256": bundle.solver.config_sha256,
            "mesh_sha256": bundle.solver.mesh_sha256,
            "source_revision": bundle.solver.source_revision,
        },
        "transfer": _transfer_metadata(bundle.transfer),
        "grid": {
            "coordinate_unit": bundle.electric.grid.coordinate_unit,
            "coordinate_sha256": bundle.electric.grid.coordinate_sha256,
            "axes": [
                _array_record(f"grid/{name}", axis)
                for name, axis in zip(_GRID_MEMBERS, axes, strict=True)
            ],
        },
        "fields": {
            "electric": {
                **_array_record("fields/electric", electric),
                "field_kind": bundle.electric.field_kind.value,
                "representation": bundle.electric.representation.value,
                "unit": bundle.electric.unit,
                "function_space": bundle.electric.function_space,
            },
            "magnetic": {
                **_array_record("fields/magnetic", magnetic),
                "field_kind": bundle.magnetic.field_kind.value,
                "representation": bundle.magnetic.representation.value,
                "unit": bundle.magnetic.unit,
                "function_space": bundle.magnetic.function_space,
            },
        },
    }


def _canonical_json(data: Mapping[str, object]) -> bytes:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_sha256(metadata_json: bytes, arrays: Sequence[tuple[str, Any]]) -> str:
    hasher = hashlib.sha256()
    hasher.update(_CONTENT_SCHEMA)
    _framed_update(hasher, metadata_json)
    for name, values in arrays:
        _framed_update(hasher, name.encode("ascii"))
        _framed_update(hasher, values.tobytes(order="C"))
    return hasher.hexdigest()


def _canonical_bundle_arrays(bundle: ModeBundle, np: Any) -> tuple[tuple[Any, Any, Any], Any, Any]:
    electric_axes = tuple(
        _canonical_axis(np, axis) for axis in bundle.electric.grid.edge_coordinates
    )
    magnetic_axes = tuple(
        _canonical_axis(np, axis) for axis in bundle.magnetic.grid.edge_coordinates
    )
    if any(
        not np.array_equal(electric_axis, magnetic_axis)
        for electric_axis, magnetic_axis in zip(electric_axes, magnetic_axes, strict=True)
    ):
        raise ArtifactError("electric and magnetic mode fields retain different Yee coordinates")
    rebuilt_grid = build_yee_grid(electric_axes)
    if rebuilt_grid.coordinate_sha256 != bundle.electric.grid.coordinate_sha256:
        raise ArtifactError("mode HDF5 Yee coordinate digest does not match its coordinates")
    electric = _canonical_field(np, bundle.electric.values)
    magnetic = _canonical_field(np, bundle.magnetic.values)
    return cast(tuple[Any, Any, Any], electric_axes), electric, magnetic


def _logical_data_bytes(arrays: Sequence[tuple[str, Any]]) -> int:
    return sum(int(values.size) * int(values.dtype.itemsize) for _, values in arrays)


def _artifact_reference(path: str, target: Path) -> ArtifactRef:
    return ArtifactRef(
        path=path,
        role=ArtifactRole.CANONICAL_NUMERICAL,
        media_type=MODE_HDF5_MEDIA_TYPE,
        sha256=sha256_file(target),
        size_bytes=target.stat().st_size,
    )


def write_mode_bundle_hdf5(
    run_root: Path,
    relative_path: str,
    bundle: ModeBundle,
) -> ModeBundleHDF5Artifact:
    """Write one mode atomically without overwriting an existing artifact."""

    if not isinstance(bundle, ModeBundle):
        raise ArtifactError("mode HDF5 writer requires a ModeBundle")
    h5py, np = _numeric_modules()
    target = _resolve_run_path(run_root, relative_path, create_parent=True)
    if target.exists() or target.is_symlink():
        raise ArtifactError(f"mode HDF5 artifact already exists: {relative_path}")

    axes, electric, magnetic = _canonical_bundle_arrays(bundle, np)
    arrays = (
        *((f"grid/{name}", axis) for name, axis in zip(_GRID_MEMBERS, axes, strict=True)),
        ("fields/electric", electric),
        ("fields/magnetic", magnetic),
    )
    metadata = _mode_metadata(bundle, axes, electric, magnetic)
    metadata_json = _canonical_json(metadata)
    content_sha256 = _content_sha256(metadata_json, arrays)
    canonical_bundle = _read_mode_bundle(metadata, axes, (electric, magnetic))
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        with h5py.File(temporary_path, "w", libver="latest") as handle:
            handle.attrs["container_schema"] = MODE_HDF5_SCHEMA
            handle.attrs["content_sha256"] = content_sha256
            handle.create_dataset(
                "metadata_json",
                data=np.frombuffer(metadata_json, dtype=np.uint8),
                dtype=np.dtype("u1"),
                track_times=False,
            )
            grid_group = handle.create_group("grid", track_order=False)
            for name, axis in zip(_GRID_MEMBERS, axes, strict=True):
                grid_group.create_dataset(
                    name,
                    data=axis,
                    dtype=np.dtype("<f8"),
                    fletcher32=True,
                    track_times=False,
                )
            field_group = handle.create_group("fields", track_order=False)
            for name, values in zip(_FIELD_MEMBERS, (electric, magnetic), strict=True):
                field_group.create_dataset(
                    name,
                    data=values,
                    dtype=values.dtype,
                    chunks=True,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                    fletcher32=True,
                    track_times=False,
                )
            handle.flush()
        if target.exists() or target.is_symlink():
            raise ArtifactError(f"mode HDF5 artifact appeared during write: {relative_path}")
        try:
            os.link(temporary_path, target)
        except FileExistsError as exc:
            raise ArtifactError(
                f"mode HDF5 artifact appeared during write: {relative_path}"
            ) from exc
        temporary_path.unlink()
        temporary_path = None
    except ArtifactError:
        raise
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise ArtifactError(f"failed to write mode HDF5 artifact: {exc}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    reference = _artifact_reference(relative_path, target)
    return ModeBundleHDF5Artifact(
        bundle=canonical_bundle,
        reference=reference,
        content_sha256=content_sha256,
        logical_data_bytes=_logical_data_bytes(arrays),
    )


def _require_exact_members(container: Any, expected: Sequence[str] | frozenset[str]) -> None:
    actual = frozenset(str(name) for name in container.keys())
    if actual != frozenset(expected):
        raise ArtifactError(
            f"mode HDF5 members differ: expected {sorted(expected)}, got {sorted(actual)}"
        )


def _hard_object(container: Any, name: str, expected_type: Any, h5py: Any) -> Any:
    link = container.get(name, getlink=True)
    if not isinstance(link, h5py.HardLink):
        raise ArtifactError(f"mode HDF5 member {name!r} must be an internal hard link")
    value = container.get(name)
    if not isinstance(value, expected_type):
        raise ArtifactError(f"mode HDF5 member {name!r} has the wrong object type")
    return value


def _reject_external_storage(dataset: Any) -> None:
    if dataset.id.get_create_plist().get_external_count() != 0:
        raise ArtifactError(f"mode HDF5 dataset {dataset.name!r} uses external storage")
    if dataset.is_virtual:
        raise ArtifactError(f"mode HDF5 dataset {dataset.name!r} uses virtual storage")


def _metadata_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ArtifactError(f"mode HDF5 metadata {label} must be an object")
    return cast(Mapping[str, object], value)


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


def _read_transfer(metadata: Mapping[str, object]) -> TransferReport:
    return TransferReport(
        source_representation=FieldRepresentation(str(metadata["source_representation"])),
        target_representation=FieldRepresentation(str(metadata["target_representation"])),
        operator_sha256=str(metadata["operator_sha256"]),
        relative_power_error=float(cast(float, metadata["relative_power_error"])),
        relative_interpolation_error=(
            None
            if metadata["relative_interpolation_error"] is None
            else float(cast(float, metadata["relative_interpolation_error"]))
        ),
        source_power_watts=float(cast(float, metadata["source_power_watts"])),
        pre_correction_power_watts=float(cast(float, metadata["pre_correction_power_watts"])),
        relative_pre_correction_power_error=float(
            cast(float, metadata["relative_pre_correction_power_error"])
        ),
        transferred_power_watts=float(cast(float, metadata["transferred_power_watts"])),
        power_correction_scale=float(cast(float, metadata["power_correction_scale"])),
        target_runtime_name=_optional_text(metadata["target_runtime_name"]),
        target_runtime_version=_optional_text(metadata["target_runtime_version"]),
        target_source_revision=_optional_text(metadata["target_source_revision"]),
        target_source_digest=_optional_text(metadata["target_source_digest"]),
    )


def _read_mode_bundle(
    metadata: Mapping[str, object], axes: tuple[Any, Any, Any], fields: Any
) -> ModeBundle:
    grid_metadata = _metadata_mapping(metadata["grid"], label="grid")
    field_metadata = _metadata_mapping(metadata["fields"], label="fields")
    electric_metadata = _metadata_mapping(field_metadata["electric"], label="fields.electric")
    magnetic_metadata = _metadata_mapping(field_metadata["magnetic"], label="fields.magnetic")
    normalization = _metadata_mapping(metadata["normalization"], label="normalization")
    propagation = _metadata_mapping(metadata["propagation"], label="propagation")
    solver = _metadata_mapping(metadata["solver"], label="solver")
    effective_index = _metadata_mapping(metadata["effective_index"], label="effective_index")
    beta_per_m = _metadata_mapping(metadata["beta_per_m"], label="beta_per_m")
    transfer = _metadata_mapping(metadata["transfer"], label="transfer")
    rebuilt_grid = build_yee_grid(axes)
    if rebuilt_grid.coordinate_sha256 != str(grid_metadata["coordinate_sha256"]):
        raise ArtifactError("mode HDF5 coordinate digest is inconsistent")
    electric_values, magnetic_values = fields
    return ModeBundle(
        frequency_hz=float(cast(float, metadata["frequency_hz"])),
        effective_index=complex(
            float(cast(float, effective_index["real"])),
            float(cast(float, effective_index["imag"])),
        ),
        beta_per_m=complex(
            float(cast(float, beta_per_m["real"])),
            float(cast(float, beta_per_m["imag"])),
        ),
        electric=YeeVectorField(
            values=electric_values,
            grid=rebuilt_grid,
            field_kind=YeeFieldKind(str(electric_metadata["field_kind"])),
            unit=str(electric_metadata["unit"]),
            function_space=str(electric_metadata["function_space"]),
            representation=FieldRepresentation(str(electric_metadata["representation"])),
        ),
        magnetic=YeeVectorField(
            values=magnetic_values,
            grid=rebuilt_grid,
            field_kind=YeeFieldKind(str(magnetic_metadata["field_kind"])),
            unit=str(magnetic_metadata["unit"]),
            function_space=str(magnetic_metadata["function_space"]),
            representation=FieldRepresentation(str(magnetic_metadata["representation"])),
        ),
        propagation=AxisDirection(
            Axis(str(propagation["axis"])),
            Direction(str(propagation["direction"])),
        ),
        magnetic_convention=MagneticFieldConvention(str(metadata["magnetic_convention"])),
        normalization=ModeNormalization(
            kind=NormalizationKind(str(normalization["kind"])),
            target_power_watts=float(cast(float, normalization["target_power_watts"])),
            phase_reference=str(normalization["phase_reference"]),
        ),
        solver=SolverFingerprint(
            name=str(solver["name"]),
            version=str(solver["version"]),
            config_sha256=str(solver["config_sha256"]),
            mesh_sha256=str(solver["mesh_sha256"]),
            source_revision=_optional_text(solver["source_revision"]),
        ),
        transfer=_read_transfer(transfer),
        schema_version=str(metadata["schema_version"]),
    )


def _dataset_array(
    dataset: Any, *, expected_kind: str, expected_itemsizes: tuple[int, ...], np: Any
) -> Any:
    _reject_external_storage(dataset)
    if dataset.dtype.kind != expected_kind or dataset.dtype.itemsize not in expected_itemsizes:
        raise ArtifactError(f"mode HDF5 dataset {dataset.name!r} has an unsupported dtype")
    values = dataset[...]
    return _canonical_axis(np, values) if expected_kind == "f" else _canonical_field(np, values)


def _verify_array_metadata(record: object, name: str, values: Any) -> None:
    metadata = _metadata_mapping(record, label=name)
    array_keys = ("path", "dtype", "shape", "sha256")
    if {key: metadata.get(key) for key in array_keys} != _array_record(name, values):
        raise ArtifactError(f"mode HDF5 array metadata is inconsistent for {name}")


def read_mode_bundle_hdf5(
    run_root: Path,
    reference: ArtifactRef,
    *,
    maximum_data_bytes: int = DEFAULT_MAXIMUM_MODE_DATA_BYTES,
) -> ModeBundleHDF5Artifact:
    """Read and verify one canonical mode artifact against its external file reference."""

    if (
        isinstance(maximum_data_bytes, bool)
        or not isinstance(maximum_data_bytes, int)
        or maximum_data_bytes <= 0
    ):
        raise ArtifactError("maximum mode HDF5 data bytes must be positive")
    if reference.role is not ArtifactRole.CANONICAL_NUMERICAL:
        raise ArtifactError("mode HDF5 reference must have the canonical numerical role")
    if reference.media_type != MODE_HDF5_MEDIA_TYPE:
        raise ArtifactError("mode HDF5 reference has the wrong media type")
    target = _resolve_run_path(run_root, reference.path, create_parent=False)
    if target.is_symlink() or not target.is_file():
        raise ArtifactError(
            f"mode HDF5 reference is not a regular non-symlink file: {reference.path}"
        )
    if target.stat().st_size != reference.size_bytes or sha256_file(target) != reference.sha256:
        raise ArtifactError("mode HDF5 file size or SHA-256 differs from its ArtifactRef")

    h5py, np = _numeric_modules()
    try:
        with h5py.File(target, "r") as handle:
            _require_exact_members(handle, _ROOT_MEMBERS)
            if str(handle.attrs.get("container_schema", "")) != MODE_HDF5_SCHEMA:
                raise ArtifactError("mode HDF5 container schema is missing or unsupported")
            recorded_content_sha256 = str(handle.attrs.get("content_sha256", ""))
            if not _SHA256_PATTERN.fullmatch(recorded_content_sha256):
                raise ArtifactError("mode HDF5 content digest is missing or invalid")

            metadata_dataset = _hard_object(handle, "metadata_json", h5py.Dataset, h5py)
            grid_group = _hard_object(handle, "grid", h5py.Group, h5py)
            field_group = _hard_object(handle, "fields", h5py.Group, h5py)
            _require_exact_members(grid_group, _GRID_MEMBERS)
            _require_exact_members(field_group, _FIELD_MEMBERS)
            _reject_external_storage(metadata_dataset)
            if (
                metadata_dataset.ndim != 1
                or metadata_dataset.dtype != np.dtype("u1")
                or not 0 < metadata_dataset.size <= _MAXIMUM_METADATA_BYTES
            ):
                raise ArtifactError("mode HDF5 metadata dataset has an invalid layout")

            axis_datasets = tuple(
                _hard_object(grid_group, name, h5py.Dataset, h5py) for name in _GRID_MEMBERS
            )
            field_datasets = tuple(
                _hard_object(field_group, name, h5py.Dataset, h5py) for name in _FIELD_MEMBERS
            )
            logical_bytes = sum(
                int(dataset.size) * int(dataset.dtype.itemsize)
                for dataset in (*axis_datasets, *field_datasets)
            )
            if logical_bytes > maximum_data_bytes:
                raise ArtifactError(
                    f"mode HDF5 logical data size {logical_bytes} exceeds limit {maximum_data_bytes}"
                )

            metadata_json = bytes(np.asarray(metadata_dataset[...], dtype=np.uint8).tobytes())
            try:
                decoded = json.loads(metadata_json.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ArtifactError("mode HDF5 metadata is not canonical UTF-8 JSON") from exc
            metadata = _metadata_mapping(decoded, label="root")
            if _canonical_json(metadata) != metadata_json:
                raise ArtifactError("mode HDF5 metadata JSON is not canonical")

            axes = cast(
                tuple[Any, Any, Any],
                tuple(
                    _dataset_array(
                        dataset,
                        expected_kind="f",
                        expected_itemsizes=(8,),
                        np=np,
                    )
                    for dataset in axis_datasets
                ),
            )
            fields = tuple(
                _dataset_array(
                    dataset,
                    expected_kind="c",
                    expected_itemsizes=(8, 16),
                    np=np,
                )
                for dataset in field_datasets
            )
            grid_metadata = _metadata_mapping(metadata["grid"], label="grid")
            axis_metadata = cast(Sequence[object], grid_metadata["axes"])
            if len(axis_metadata) != 3:
                raise ArtifactError("mode HDF5 metadata must describe three grid axes")
            for name, values, record in zip(
                (f"grid/{name}" for name in _GRID_MEMBERS), axes, axis_metadata, strict=True
            ):
                _verify_array_metadata(record, name, values)
            field_metadata = _metadata_mapping(metadata["fields"], label="fields")
            for name, values in zip(_FIELD_MEMBERS, fields, strict=True):
                _verify_array_metadata(
                    _metadata_mapping(field_metadata[name], label=f"fields.{name}"),
                    f"fields/{name}",
                    values,
                )

            bundle = _read_mode_bundle(metadata, axes, fields)
            regenerated_json = _canonical_json(_mode_metadata(bundle, axes, *fields))
            if regenerated_json != metadata_json:
                raise ArtifactError("mode HDF5 metadata is not exactly reproducible from its mode")
            arrays = (
                *((f"grid/{name}", axis) for name, axis in zip(_GRID_MEMBERS, axes, strict=True)),
                ("fields/electric", fields[0]),
                ("fields/magnetic", fields[1]),
            )
            content_sha256 = _content_sha256(regenerated_json, arrays)
            if content_sha256 != recorded_content_sha256:
                raise ArtifactError("mode HDF5 logical content digest does not match")
    except ArtifactError:
        raise
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, ContractError) as exc:
        raise ArtifactError(f"invalid mode HDF5 artifact: {exc}") from exc

    if target.stat().st_size != reference.size_bytes or sha256_file(target) != reference.sha256:
        raise ArtifactError("mode HDF5 file changed while it was being read")
    return ModeBundleHDF5Artifact(
        bundle=bundle,
        reference=reference,
        content_sha256=content_sha256,
        logical_data_bytes=_logical_data_bytes(arrays),
    )
