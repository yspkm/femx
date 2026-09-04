"""Topology-bound process-local checkpoints for the collective port operator.

The checkpoint is deliberately narrower than a cloud recovery system.  Every JAX process writes
only its addressable shards, and restore requires the exact source, configuration, transport
layout, process map, device map, shape, dtype, and named sharding observed at save time.  A control
plane may move complete process fragments to durable storage and recreate Spot capacity, but it
must not weaken this numerical-state contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
from jax.sharding import Mesh

from femx.core.errors import ContractError

from .port_collective import PortCollectiveLayout, describe_collective_port_mesh
from .port_collective_runtime import _leading_partition, describe_collective_port_array

PORT_COLLECTIVE_CHECKPOINT_SCHEMA = "femx.jax.port_collective.checkpoint_fragment/v1"
PORT_COLLECTIVE_CHECKPOINT_REPORT_SCHEMA = "femx.jax.port_collective.checkpoint_fragment_report/v1"
PORT_COLLECTIVE_CHECKPOINT_COMPLETE_SCHEMA = (
    "femx.jax.port_collective.checkpoint_fragment_complete/v1"
)

_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ALLOWED_DTYPES = frozenset({"int32", "int64", "float32", "float64", "complex64", "complex128"})
_MAX_JSON_BYTES = 1 << 20
_MAX_NPY_OVERHEAD_BYTES = 1 << 16
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "checkpoint_id",
        "step",
        "source_sha256",
        "config_sha256",
        "layout_sha256",
        "process_index",
        "process_count",
        "mesh_report",
        "arrays",
        "completion_scope",
        "restore_policy",
    }
)
_ARRAY_KEYS = frozenset({"name", "array_report", "shards"})
_SHARD_FILE_KEYS = frozenset(
    {
        "partition_index",
        "process_index",
        "device_id",
        "device_kind",
        "local_shape",
        "logical_bytes",
        "relative_path",
        "file_size_bytes",
        "sha256",
    }
)


def _require_component(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _COMPONENT.fullmatch(value) or value in {".", ".."}:
        raise ContractError(f"{label} must be a canonical path component")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ContractError(f"{label} must be a canonical SHA-256")
    return value


def _require_nonnegative_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{label} must be a nonnegative integer")
    return value


def _canonical_json(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ContractError("checkpoint metadata must be canonical finite JSON") from error
    return (text + "\n").encode("utf-8")


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_bytes(path: Path, value: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _strict_json(path: Path) -> object:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"checkpoint metadata is not a regular file: {path.name}")
    if path.stat().st_size > _MAX_JSON_BYTES:
        raise ContractError(f"checkpoint metadata exceeds {_MAX_JSON_BYTES} bytes: {path.name}")

    def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"checkpoint metadata repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"checkpoint metadata is not valid JSON: {path.name}") from error


def _exact_keys(value: object, expected: frozenset[str], *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ContractError(f"{label} must contain exactly the versioned schema fields")
    return value


def _canonical_local_array(value: object, *, expected_dtype: str, label: str) -> np.ndarray:
    if expected_dtype not in _ALLOWED_DTYPES:
        raise ContractError(f"{label} uses unsupported checkpoint dtype {expected_dtype!r}")
    raw = np.asarray(value)
    if raw.dtype.name != expected_dtype:
        raise ContractError(f"{label} dtype disagrees with its array report")
    if not np.all(np.isfinite(raw)):
        raise ContractError(f"{label} must contain only finite values")
    little_endian = raw.astype(np.dtype(expected_dtype).newbyteorder("<"), copy=False)
    return np.ascontiguousarray(little_endian)


def _checkpoint_container(checkpoint_root: Path, checkpoint_id: str) -> Path:
    root = Path(checkpoint_root)
    if not root.is_absolute():
        raise ContractError("checkpoint root must be absolute")
    _require_component(checkpoint_id, label="checkpoint id")
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise ContractError("checkpoint root must be a real directory")
    root.mkdir(parents=True, exist_ok=True)
    container = root / checkpoint_id
    if container.exists() and (container.is_symlink() or not container.is_dir()):
        raise ContractError("checkpoint container must be a real directory")
    container.mkdir(exist_ok=True)
    return container


def port_collective_checkpoint_fragment_path(
    checkpoint_root: Path,
    checkpoint_id: str,
    process_index: int,
) -> Path:
    """Return the canonical complete-fragment path without creating it."""

    root = Path(checkpoint_root)
    if not root.is_absolute():
        raise ContractError("checkpoint root must be absolute")
    _require_component(checkpoint_id, label="checkpoint id")
    index = _require_nonnegative_integer(process_index, label="checkpoint process index")
    return root / checkpoint_id / f"process-{index:05d}"


@dataclass(frozen=True, slots=True)
class PortCollectiveCheckpointFragment:
    """Identity of one atomically published process-local checkpoint fragment."""

    path: Path
    checkpoint_id: str
    step: int
    source_sha256: str
    config_sha256: str
    layout_sha256: str
    process_index: int
    process_count: int
    manifest_sha256: str
    array_names: tuple[str, ...]

    def __post_init__(self) -> None:
        path = Path(self.path)
        if not path.is_absolute():
            raise ContractError("checkpoint fragment path must be absolute")
        _require_component(self.checkpoint_id, label="checkpoint id")
        _require_nonnegative_integer(self.step, label="checkpoint step")
        for name in ("source_sha256", "config_sha256", "layout_sha256", "manifest_sha256"):
            _require_sha256(getattr(self, name), label=name.replace("_", " "))
        process_index = _require_nonnegative_integer(
            self.process_index, label="checkpoint process index"
        )
        process_count = _require_nonnegative_integer(
            self.process_count, label="checkpoint process count"
        )
        if process_count == 0 or process_index >= process_count:
            raise ContractError("checkpoint process identity is outside the process count")
        names = tuple(
            _require_component(name, label="checkpoint array name") for name in self.array_names
        )
        if not names or names != tuple(sorted(set(names))):
            raise ContractError("checkpoint array names must be nonempty, unique, and sorted")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "array_names", names)

    def canonical_data(self) -> dict[str, object]:
        """Return a path-independent record suitable for an evidence manifest."""

        return {
            "schema_version": PORT_COLLECTIVE_CHECKPOINT_REPORT_SCHEMA,
            "checkpoint_id": self.checkpoint_id,
            "step": self.step,
            "source_sha256": self.source_sha256,
            "config_sha256": self.config_sha256,
            "layout_sha256": self.layout_sha256,
            "process_index": self.process_index,
            "process_count": self.process_count,
            "manifest_sha256": self.manifest_sha256,
            "array_names": list(self.array_names),
            "completion_scope": "one process-local fragment",
            "restore_policy": "exact same topology only; no resharding",
        }


def _fragment_report(
    path: Path, manifest: Mapping[str, object], digest: str
) -> PortCollectiveCheckpointFragment:
    arrays = manifest["arrays"]
    if not isinstance(arrays, list):
        raise ContractError("checkpoint arrays must be a JSON list")
    names: list[str] = []
    for value in arrays:
        entry = _exact_keys(value, _ARRAY_KEYS, label="checkpoint array entry")
        names.append(_require_component(entry["name"], label="checkpoint array name"))
    return PortCollectiveCheckpointFragment(
        path=path,
        checkpoint_id=_require_component(manifest["checkpoint_id"], label="checkpoint id"),
        step=_require_nonnegative_integer(manifest["step"], label="checkpoint step"),
        source_sha256=_require_sha256(manifest["source_sha256"], label="source SHA-256"),
        config_sha256=_require_sha256(manifest["config_sha256"], label="config SHA-256"),
        layout_sha256=_require_sha256(manifest["layout_sha256"], label="layout SHA-256"),
        process_index=_require_nonnegative_integer(
            manifest["process_index"], label="checkpoint process index"
        ),
        process_count=_require_nonnegative_integer(
            manifest["process_count"], label="checkpoint process count"
        ),
        manifest_sha256=digest,
        array_names=tuple(names),
    )


def write_port_collective_checkpoint_fragment(
    checkpoint_root: Path,
    *,
    checkpoint_id: str,
    step: int,
    source_sha256: str,
    config_sha256: str,
    layout: PortCollectiveLayout,
    mesh: Mesh,
    arrays: Mapping[str, jax.Array],
    axis_name: str = "partition",
) -> PortCollectiveCheckpointFragment:
    """Atomically publish this process's addressable shards for a fixed topology."""

    checkpoint_id = _require_component(checkpoint_id, label="checkpoint id")
    step = _require_nonnegative_integer(step, label="checkpoint step")
    source_sha256 = _require_sha256(source_sha256, label="source SHA-256")
    config_sha256 = _require_sha256(config_sha256, label="config SHA-256")
    if not isinstance(arrays, Mapping) or not arrays:
        raise ContractError("checkpoint requires at least one named JAX array")
    raw_arrays = tuple(arrays.items())
    for name, _ in raw_arrays:
        _require_component(name, label="checkpoint array name")
    named_arrays = tuple(sorted(raw_arrays))
    if any(not isinstance(array, jax.Array) for _, array in named_arrays):
        raise ContractError("checkpoint values must be JAX arrays")

    mesh_report = describe_collective_port_mesh(layout, mesh, axis_name=axis_name)
    array_reports = tuple(
        describe_collective_port_array(name, array, mesh, axis_name=axis_name)
        for name, array in named_arrays
    )
    process_indices = {report.process_index for report in array_reports}
    process_counts = {report.process_count for report in array_reports}
    if len(process_indices) != 1 or len(process_counts) != 1:
        raise ContractError("checkpoint arrays disagree on JAX process identity")
    process_index = process_indices.pop()
    process_count = process_counts.pop()
    if process_count != mesh_report.process_count:
        raise ContractError("checkpoint array and Mesh process counts disagree")

    container = _checkpoint_container(Path(checkpoint_root), checkpoint_id)
    target = port_collective_checkpoint_fragment_path(
        Path(checkpoint_root), checkpoint_id, process_index
    )
    incomplete = target.with_name(target.name + ".incomplete")
    if target.exists():
        raise ContractError("checkpoint fragment already exists; overwrite is forbidden")
    if incomplete.exists():
        raise ContractError("checkpoint fragment has a stale incomplete publication")
    incomplete.mkdir()
    arrays_directory = incomplete / "arrays"
    arrays_directory.mkdir()

    manifest_arrays: list[dict[str, object]] = []
    for (name, array), report in zip(named_arrays, array_reports, strict=True):
        by_partition: dict[int, Any] = {}
        shape = tuple(int(value) for value in array.shape)
        for shard in array.addressable_shards:
            if not isinstance(shard.index, tuple):  # pragma: no cover - JAX API invariant
                raise ContractError("checkpoint shard index must be a tuple")
            partition = _leading_partition(tuple(shard.index), shape)
            by_partition[partition] = shard
        if tuple(sorted(by_partition)) != tuple(
            shard.partition_index for shard in report.addressable_shards
        ):
            raise ContractError("checkpoint shard data disagrees with its array report")

        shard_records: list[dict[str, object]] = []
        for shard_report in report.addressable_shards:
            partition = shard_report.partition_index
            local = _canonical_local_array(
                jax.device_get(by_partition[partition].data),
                expected_dtype=report.dtype,
                label=f"checkpoint array {name!r} partition {partition}",
            )
            if (
                tuple(local.shape) != shard_report.local_shape
                or local.nbytes != shard_report.logical_bytes
            ):
                raise ContractError(
                    "checkpoint local shard shape or byte count disagrees with report"
                )
            relative_path = f"arrays/{name}--partition-{partition:05d}.npy"
            file_path = incomplete / relative_path
            with file_path.open("xb") as stream:
                np.save(stream, local, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            shard_records.append(
                {
                    **shard_report.canonical_data(),
                    "relative_path": relative_path,
                    "file_size_bytes": file_path.stat().st_size,
                    "sha256": _sha256_file(file_path),
                }
            )
        manifest_arrays.append(
            {
                "name": name,
                "array_report": report.canonical_data(),
                "shards": shard_records,
            }
        )

    manifest: dict[str, object] = {
        "schema_version": PORT_COLLECTIVE_CHECKPOINT_SCHEMA,
        "checkpoint_id": checkpoint_id,
        "step": step,
        "source_sha256": source_sha256,
        "config_sha256": config_sha256,
        "layout_sha256": layout.digest(),
        "process_index": process_index,
        "process_count": process_count,
        "mesh_report": mesh_report.canonical_data(),
        "arrays": manifest_arrays,
        "completion_scope": "one process-local fragment",
        "restore_policy": "exact same topology only; no resharding",
    }
    manifest_bytes = _canonical_json(manifest)
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    _write_bytes(incomplete / "manifest.json", manifest_bytes)
    marker = {
        "schema_version": PORT_COLLECTIVE_CHECKPOINT_COMPLETE_SCHEMA,
        "manifest_sha256": manifest_digest,
    }
    _write_bytes(incomplete / "COMPLETE", _canonical_json(marker))
    _fsync_directory(arrays_directory)
    _fsync_directory(incomplete)
    os.rename(incomplete, target)
    _fsync_directory(container)
    return _fragment_report(target, manifest, manifest_digest)


def _verified_manifest(fragment_path: Path) -> tuple[dict[str, object], str]:
    path = Path(fragment_path)
    if not path.is_absolute():
        raise ContractError("checkpoint fragment path must be absolute")
    if path.is_symlink() or not path.is_dir():
        raise ContractError("checkpoint fragment must be a complete real directory")
    marker = _exact_keys(
        _strict_json(path / "COMPLETE"),
        frozenset({"schema_version", "manifest_sha256"}),
        label="checkpoint completion marker",
    )
    if marker["schema_version"] != PORT_COLLECTIVE_CHECKPOINT_COMPLETE_SCHEMA:
        raise ContractError("checkpoint completion marker schema is unsupported")
    expected_digest = _require_sha256(
        marker["manifest_sha256"], label="checkpoint manifest SHA-256"
    )
    manifest_path = path / "manifest.json"
    if _sha256_file(manifest_path) != expected_digest:
        raise ContractError("checkpoint manifest SHA-256 does not match COMPLETE")
    manifest = _exact_keys(_strict_json(manifest_path), _MANIFEST_KEYS, label="checkpoint manifest")
    if manifest["schema_version"] != PORT_COLLECTIVE_CHECKPOINT_SCHEMA:
        raise ContractError("checkpoint manifest schema is unsupported")
    return manifest, expected_digest


def restore_port_collective_checkpoint_fragment(
    fragment_path: Path,
    *,
    expected_checkpoint_id: str,
    expected_step: int,
    expected_source_sha256: str,
    expected_config_sha256: str,
    layout: PortCollectiveLayout,
    mesh: Mesh,
    templates: Mapping[str, jax.Array],
    axis_name: str = "partition",
) -> tuple[dict[str, jax.Array], PortCollectiveCheckpointFragment]:
    """Restore addressable shards into caller-provided same-topology array templates."""

    checkpoint_id = _require_component(expected_checkpoint_id, label="checkpoint id")
    step = _require_nonnegative_integer(expected_step, label="checkpoint step")
    source_sha256 = _require_sha256(expected_source_sha256, label="source SHA-256")
    config_sha256 = _require_sha256(expected_config_sha256, label="config SHA-256")
    if not isinstance(templates, Mapping) or not templates:
        raise ContractError("checkpoint restore requires named JAX array templates")
    raw_templates = tuple(templates.items())
    for name, _ in raw_templates:
        _require_component(name, label="checkpoint array name")
    named_templates = tuple(sorted(raw_templates))
    if any(not isinstance(array, jax.Array) for _, array in named_templates):
        raise ContractError("checkpoint templates must be JAX arrays")

    path = Path(fragment_path)
    manifest, manifest_digest = _verified_manifest(path)
    expected_mesh_report = describe_collective_port_mesh(
        layout, mesh, axis_name=axis_name
    ).canonical_data()
    expected_reports = {
        name: describe_collective_port_array(name, template, mesh, axis_name=axis_name)
        for name, template in named_templates
    }
    identity_expectations = {
        "checkpoint_id": checkpoint_id,
        "step": step,
        "source_sha256": source_sha256,
        "config_sha256": config_sha256,
        "layout_sha256": layout.digest(),
        "mesh_report": expected_mesh_report,
        "completion_scope": "one process-local fragment",
        "restore_policy": "exact same topology only; no resharding",
    }
    for key, expected in identity_expectations.items():
        if manifest[key] != expected:
            raise ContractError(f"checkpoint {key.replace('_', ' ')} does not match this run")

    first_report = next(iter(expected_reports.values()))
    if manifest["process_index"] != first_report.process_index:
        raise ContractError("checkpoint process index does not match this process")
    if manifest["process_count"] != first_report.process_count:
        raise ContractError("checkpoint process count does not match this topology")
    if path.name != f"process-{first_report.process_index:05d}":
        raise ContractError("checkpoint fragment directory disagrees with its process index")

    raw_arrays = manifest["arrays"]
    if not isinstance(raw_arrays, list) or len(raw_arrays) != len(named_templates):
        raise ContractError("checkpoint array set does not match restore templates")
    restored_local: dict[str, dict[int, np.ndarray]] = {}
    expected_files = {"manifest.json", "COMPLETE"}
    observed_names: list[str] = []
    for raw_entry in raw_arrays:
        entry = _exact_keys(raw_entry, _ARRAY_KEYS, label="checkpoint array entry")
        name = _require_component(entry["name"], label="checkpoint array name")
        observed_names.append(name)
        if name not in expected_reports:
            raise ContractError("checkpoint contains an unexpected array")
        report = expected_reports[name]
        if entry["array_report"] != report.canonical_data():
            raise ContractError(f"checkpoint array report for {name!r} does not match its template")
        raw_shards = entry["shards"]
        if not isinstance(raw_shards, list) or len(raw_shards) != len(report.addressable_shards):
            raise ContractError(f"checkpoint shard set for {name!r} is incomplete")
        local_by_partition: dict[int, np.ndarray] = {}
        for raw_shard, expected_shard in zip(raw_shards, report.addressable_shards, strict=True):
            shard = _exact_keys(raw_shard, _SHARD_FILE_KEYS, label="checkpoint shard entry")
            expected_metadata = expected_shard.canonical_data()
            if any(shard[key] != value for key, value in expected_metadata.items()):
                raise ContractError("checkpoint shard metadata does not match this topology")
            partition = expected_shard.partition_index
            relative_path = f"arrays/{name}--partition-{partition:05d}.npy"
            if shard["relative_path"] != relative_path:
                raise ContractError("checkpoint shard path is not canonical")
            expected_files.add(relative_path)
            file_path = path / relative_path
            if file_path.is_symlink() or not file_path.is_file():
                raise ContractError("checkpoint shard is not a regular file")
            size = _require_nonnegative_integer(
                shard["file_size_bytes"], label="checkpoint shard file size"
            )
            if size != file_path.stat().st_size:
                raise ContractError("checkpoint shard file size does not match manifest")
            if size > expected_shard.logical_bytes + _MAX_NPY_OVERHEAD_BYTES:
                raise ContractError("checkpoint shard file exceeds its bounded logical size")
            digest = _require_sha256(shard["sha256"], label="checkpoint shard SHA-256")
            if _sha256_file(file_path) != digest:
                raise ContractError("checkpoint shard SHA-256 does not match manifest")
            try:
                loaded = np.load(file_path, allow_pickle=False, mmap_mode="r")
            except (OSError, ValueError) as error:
                raise ContractError(
                    "checkpoint shard is not a valid non-pickle NPY array"
                ) from error
            local = _canonical_local_array(
                loaded,
                expected_dtype=report.dtype,
                label=f"checkpoint array {name!r} partition {partition}",
            )
            if (
                tuple(local.shape) != expected_shard.local_shape
                or local.nbytes != expected_shard.logical_bytes
            ):
                raise ContractError("checkpoint shard payload shape or byte count is invalid")
            local_by_partition[partition] = np.array(local, copy=True, order="C")
        restored_local[name] = local_by_partition

    if observed_names != sorted(expected_reports):
        raise ContractError("checkpoint arrays are not canonical, unique, and complete")
    for filesystem_entry in path.rglob("*"):
        relative = filesystem_entry.relative_to(path).as_posix()
        if filesystem_entry.is_symlink():
            raise ContractError("checkpoint fragment cannot contain symbolic links")
        if filesystem_entry.is_dir() and relative != "arrays":
            raise ContractError("checkpoint fragment contains an unexpected directory")
        if filesystem_entry.is_file() and relative not in expected_files:
            raise ContractError("checkpoint fragment contains an unexpected file")

    restored: dict[str, jax.Array] = {}
    for name, template in named_templates:
        single_device_arrays = []
        shape = tuple(int(value) for value in template.shape)
        for device, index in template.sharding.addressable_devices_indices_map(shape).items():
            if not isinstance(index, tuple):  # pragma: no cover - JAX API invariant
                raise ContractError("checkpoint template shard index must be a tuple")
            partition = _leading_partition(tuple(index), shape)
            single_device_arrays.append(jax.device_put(restored_local[name][partition], device))
        restored[name] = jax.make_array_from_single_device_arrays(
            shape, template.sharding, single_device_arrays
        )

    fragment_report = _fragment_report(path, manifest, manifest_digest)
    return restored, fragment_report
