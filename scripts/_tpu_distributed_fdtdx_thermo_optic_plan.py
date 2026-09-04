"""Immutable controller input for the physical distributed thermo-optic FDTDX gate.

The artifact nests the already verified electrothermal plan and adds the exact physical mesh,
FDTDX cell-center coordinates, canonical P1 sampler, distributed routing operator, material law,
and scene contract.  TPU workers only reconstruct and digest-check these controller-owned inputs;
they never instantiate the dense float64 reference backends.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from femx.interop.fdtdx import (
    DistributedTriangleP1SamplingPlan,
    FDTDXDeviceParameterContract,
    ThermoOpticLaw,
    TriangleP1SamplingPlan,
    build_triangle_p1_sampling_plan,
    prepare_distributed_triangle_p1_sampling_plan,
)
from scripts._distributed_fdtdx_thermo_optic_case import (
    FDTDX_MODULE_SHA256,
    FDTDX_PACKAGE_VERSION,
    FDTDX_SOURCE_DIGEST,
    FDTDX_SOURCE_REVISION,
    device_contract,
    scene_metadata,
    thermo_optic_law,
)
from scripts._tpu_distributed_electrothermal_plan import (
    ARRAYS_FILENAME as ELECTROTHERMAL_ARRAYS_FILENAME,
)
from scripts._tpu_distributed_electrothermal_plan import (
    MANIFEST_FILENAME as ELECTROTHERMAL_MANIFEST_FILENAME,
)
from scripts._tpu_distributed_electrothermal_plan import (
    LoadedDistributedElectrothermalArtifact,
    read_distributed_electrothermal_artifact,
)

ARTIFACT_SCHEMA = "femx.fdtdx.distributed_thermo_optic.tpu_plan/v1"
ARRAYS_FILENAME = "thermo-optic-arrays.npz"
MANIFEST_FILENAME = "manifest.json"
ELECTROTHERMAL_DIRECTORY = "electrothermal"
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_ARRAY_BYTES = 128 * 1024 * 1024

_SAMPLING_ARRAYS = (
    "source_coordinates",
    "source_cells",
    "target_axis_0",
    "target_axis_1",
    "target_axis_2",
    "sampling_target_cell_indices",
    "sampling_barycentric_weights",
)
_TRANSFER_ARRAYS = (
    "transfer_source_cell_ids",
    "transfer_send_source_cell_slots",
    "transfer_send_barycentric_weights",
    "transfer_send_active",
    "transfer_receive_target_local_indices",
    "transfer_receive_active",
)
_ARRAY_DTYPES = {
    "source_coordinates": np.dtype(np.float64),
    "source_cells": np.dtype(np.int64),
    "target_axis_0": np.dtype(np.float64),
    "target_axis_1": np.dtype(np.float64),
    "target_axis_2": np.dtype(np.float64),
    "sampling_target_cell_indices": np.dtype(np.int64),
    "sampling_barycentric_weights": np.dtype(np.float64),
    "transfer_source_cell_ids": np.dtype(np.int64),
    "transfer_send_source_cell_slots": np.dtype(np.int64),
    "transfer_send_barycentric_weights": np.dtype(np.float64),
    "transfer_send_active": np.dtype(np.bool_),
    "transfer_receive_target_local_indices": np.dtype(np.int64),
    "transfer_receive_active": np.dtype(np.bool_),
}


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    return value


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a nonempty trimmed string")
    return value


def _integer(value: object, *, label: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return value


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be a finite number")
    return converted


def _sha256(value: object, *, label: str) -> str:
    result = _text(value, label=label)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{label} must be a canonical lowercase SHA-256")
    return result


def _commit(value: object, *, label: str) -> str:
    result = _text(value, label=label)
    if len(result) != 40 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{label} must be a canonical 40-character Git object ID")
    return result


def _readonly(value: object, *, dtype: np.dtype | type[np.generic], label: str) -> np.ndarray:
    raw = np.asarray(value)
    result = np.array(raw, dtype=dtype, order="C", copy=True)
    if result.dtype.kind in "fc" and np.any(~np.isfinite(result)):
        raise ValueError(f"{label} must contain only finite values")
    result.setflags(write=False)
    return result


def _artifact_arrays(
    sampling: TriangleP1SamplingPlan,
    transfer: DistributedTriangleP1SamplingPlan,
) -> dict[str, np.ndarray]:
    result = {
        "source_coordinates": np.asarray(sampling.source_coordinates, dtype=np.float64),
        "source_cells": np.asarray(sampling.source_cells, dtype=np.int64),
        **{
            f"target_axis_{axis}": np.asarray(values, dtype=np.float64)
            for axis, values in enumerate(sampling.target_coordinates)
        },
        "sampling_target_cell_indices": np.asarray(
            sampling.target_cell_indices,
            dtype=np.int64,
        ),
        "sampling_barycentric_weights": np.asarray(
            sampling.barycentric_weights,
            dtype=np.float64,
        ),
        "transfer_source_cell_ids": np.asarray(transfer.source_cell_ids, dtype=np.int64),
        "transfer_send_source_cell_slots": np.asarray(
            transfer.send_source_cell_slots,
            dtype=np.int64,
        ),
        "transfer_send_barycentric_weights": np.asarray(
            transfer.send_barycentric_weights,
            dtype=np.float64,
        ),
        "transfer_send_active": np.asarray(transfer.send_active, dtype=np.bool_),
        "transfer_receive_target_local_indices": np.asarray(
            transfer.receive_target_local_indices,
            dtype=np.int64,
        ),
        "transfer_receive_active": np.asarray(transfer.receive_active, dtype=np.bool_),
    }
    return {name: np.array(result[name], order="C", copy=True) for name in sorted(result)}


def _sampling_metadata(sampling: TriangleP1SamplingPlan) -> dict[str, object]:
    return {
        "schema_version": sampling.schema_version,
        "source_mesh_sha256": sampling.source_mesh_sha256,
        "target_coordinate_sha256": sampling.target_coordinate_sha256,
        "operator_sha256": sampling.operator_sha256,
        "plane_axes": list(sampling.plane_axes),
        "target_shape_xyz": list(sampling.target_shape),
        "containment_tolerance": sampling.containment_tolerance,
        "maximum_partition_error": sampling.maximum_partition_error,
        "minimum_barycentric_weight": sampling.minimum_barycentric_weight,
    }


def _fdtdx_metadata() -> dict[str, object]:
    return {
        "package_version": FDTDX_PACKAGE_VERSION,
        "source_revision": FDTDX_SOURCE_REVISION,
        "source_digest": FDTDX_SOURCE_DIGEST,
        "module_sha256": dict(FDTDX_MODULE_SHA256),
    }


@dataclass(frozen=True, slots=True)
class LoadedDistributedFDTDXThermoOpticArtifact:
    """Fully verified controller input for one distributed FDTDX objective run."""

    electrothermal: LoadedDistributedElectrothermalArtifact
    sampling: TriangleP1SamplingPlan
    transfer: DistributedTriangleP1SamplingPlan
    law: ThermoOpticLaw
    contract: FDTDXDeviceParameterContract
    scene: Mapping[str, object]
    manifest: Mapping[str, object]
    arrays_sha256: str


def write_distributed_fdtdx_thermo_optic_artifact(
    output_root: Path,
    *,
    electrothermal_root: Path,
    source_commit: str,
    sampling: TriangleP1SamplingPlan,
    transfer: DistributedTriangleP1SamplingPlan,
    law: ThermoOpticLaw,
    contract: FDTDXDeviceParameterContract,
    scene: Mapping[str, object],
) -> Mapping[str, object]:
    """Atomically publish one new controller-owned coupled input directory."""

    root = output_root.resolve(strict=False)
    if not output_root.is_absolute() or root.exists() or output_root.is_symlink():
        raise ValueError("TPU FDTDX thermo-optic artifact root must be a new absolute path")
    source_commit = _commit(source_commit, label="artifact source commit")
    electrothermal = read_distributed_electrothermal_artifact(electrothermal_root)
    nested_commit = _commit(
        electrothermal.manifest.get("source_commit"),
        label="nested electrothermal source commit",
    )
    if source_commit != nested_commit:
        raise ValueError("coupled and nested electrothermal source commits must match")
    if transfer.source_layout_sha256 != electrothermal.plan.layout.digest():
        raise ValueError("thermo-optic transfer must bind the nested electrothermal layout")
    if transfer.partition_count != electrothermal.plan.layout.partition_count:
        raise ValueError("thermo-optic transfer partition count must match electrothermal input")
    if not np.array_equal(
        np.asarray(sampling.source_cells),
        electrothermal.plan.layout.topology.cells,
    ):
        raise ValueError("P1 sampling cells must follow the electrothermal cell order")
    canonical_law = thermo_optic_law()
    if law.canonical_data() != canonical_law.canonical_data() or law.sha256 != canonical_law.sha256:
        raise ValueError("artifact must use the canonical physical-gate thermo-optic law")
    canonical_contract = device_contract(sampling, parameter_dtype="float32")
    if contract.canonical_data() != canonical_contract.canonical_data():
        raise ValueError("artifact must use the canonical float32 FDTDX device contract")
    if transfer.sampling_operator_sha256 != sampling.operator_sha256:
        raise ValueError("distributed transfer must bind the canonical P1 sampling operator")
    time_steps = _integer(scene.get("time_steps"), label="scene time steps", positive=True)
    if dict(scene) != scene_metadata(time_steps=time_steps):
        raise ValueError("artifact must use the canonical physical-gate FDTDX scene")
    _canonical_json(dict(scene))

    source_nested_root = electrothermal_root.resolve(strict=True)
    expected_nested_names = {
        ELECTROTHERMAL_ARRAYS_FILENAME,
        ELECTROTHERMAL_MANIFEST_FILENAME,
    }
    if {path.name for path in source_nested_root.iterdir()} != expected_nested_names:
        raise ValueError("nested electrothermal artifact must contain exactly two canonical files")
    for name in expected_nested_names:
        path = source_nested_root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("nested electrothermal inputs must be regular non-symlink files")

    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        nested_root = temporary / ELECTROTHERMAL_DIRECTORY
        nested_root.mkdir()
        for name in sorted(expected_nested_names):
            shutil.copyfile(source_nested_root / name, nested_root / name)
        arrays_path = temporary / ARRAYS_FILENAME
        np.savez(
            arrays_path,
            **_artifact_arrays(sampling, transfer),  # type: ignore[arg-type]
        )
        arrays_sha256 = _sha256_file(arrays_path)
        nested_manifest_path = nested_root / ELECTROTHERMAL_MANIFEST_FILENAME
        manifest: dict[str, object] = {
            "schema_version": ARTIFACT_SCHEMA,
            "source_commit": source_commit,
            "electrothermal": {
                "path": ELECTROTHERMAL_DIRECTORY,
                "manifest_sha256": _sha256_file(nested_manifest_path),
                "arrays_sha256": electrothermal.arrays_sha256,
                "plan_sha256": electrothermal.plan.digest(),
                "layout_sha256": electrothermal.plan.layout.digest(),
            },
            "sampling": _sampling_metadata(sampling),
            "transfer": dict(transfer.canonical_data()),
            "thermo_optic_law": {
                **dict(law.canonical_data()),
                "sha256": law.sha256,
            },
            "device_contract": dict(contract.canonical_data()),
            "fdtdx": _fdtdx_metadata(),
            "scene": dict(scene),
            "arrays": {
                "path": ARRAYS_FILENAME,
                "sha256": arrays_sha256,
                "byte_count": arrays_path.stat().st_size,
                "pickle_allowed": False,
            },
        }
        (temporary / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, root)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _load_manifest(root: Path) -> Mapping[str, object]:
    path = root / MANIFEST_FILENAME
    if path.is_symlink() or not path.is_file():
        raise ValueError("TPU FDTDX thermo-optic manifest must be a regular non-symlink file")
    size = path.stat().st_size
    if size <= 0 or size > MAX_MANIFEST_BYTES:
        raise ValueError("TPU FDTDX thermo-optic manifest size is outside the admitted range")
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_constant,
    )
    return _mapping(value, label="TPU FDTDX thermo-optic manifest")


def _load_arrays(root: Path, manifest: Mapping[str, object]) -> dict[str, np.ndarray]:
    record = _mapping(manifest.get("arrays"), label="artifact arrays")
    if _text(record.get("path"), label="artifact arrays path") != ARRAYS_FILENAME:
        raise ValueError("artifact arrays path must use the canonical filename")
    if record.get("pickle_allowed") is not False:
        raise ValueError("artifact must explicitly forbid pickle loading")
    path = root / ARRAYS_FILENAME
    if path.is_symlink() or not path.is_file():
        raise ValueError("artifact arrays must be a regular non-symlink file")
    size = path.stat().st_size
    expected_size = _integer(record.get("byte_count"), label="arrays byte count", positive=True)
    if size != expected_size or size > MAX_ARRAY_BYTES:
        raise ValueError("artifact arrays byte count is outside the admitted contract")
    expected_sha256 = _sha256(record.get("sha256"), label="artifact arrays SHA-256")
    if _sha256_file(path) != expected_sha256:
        raise ValueError("artifact arrays SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if set(arrays) != {*_SAMPLING_ARRAYS, *_TRANSFER_ARRAYS}:
        raise ValueError("artifact arrays do not match the exact admitted field set")
    for name, expected_dtype in _ARRAY_DTYPES.items():
        if arrays[name].dtype != expected_dtype:
            raise ValueError(f"artifact array {name!r} must use {expected_dtype}")
    return arrays


def _plane_axes(value: object) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("sampling plane axes must contain exactly two integers")
    result = tuple(_integer(item, label="sampling plane axis") for item in value)
    if len(set(result)) != 2 or any(axis > 2 for axis in result):
        raise ValueError("sampling plane axes must be distinct x/y/z axes")
    return result  # type: ignore[return-value]


def _verify_array(actual: object, expected: object, *, label: str) -> None:
    if not np.array_equal(np.asarray(actual), np.asarray(expected)):
        raise ValueError(f"stored {label} differs from its reconstructed operator")


def read_distributed_fdtdx_thermo_optic_artifact(
    input_root: Path,
) -> LoadedDistributedFDTDXThermoOpticArtifact:
    """Read and independently reconstruct every numerical operator in the coupled input."""

    root = input_root.resolve(strict=True)
    if not root.is_dir() or input_root.is_symlink():
        raise ValueError("TPU FDTDX thermo-optic artifact root must be a non-symlink directory")
    manifest = _load_manifest(root)
    expected_manifest_fields = {
        "schema_version",
        "source_commit",
        "electrothermal",
        "sampling",
        "transfer",
        "thermo_optic_law",
        "device_contract",
        "fdtdx",
        "scene",
        "arrays",
    }
    if (
        set(manifest) != expected_manifest_fields
        or manifest.get("schema_version") != ARTIFACT_SCHEMA
    ):
        raise ValueError("unsupported or non-canonical TPU FDTDX thermo-optic manifest")
    source_commit = _commit(manifest.get("source_commit"), label="artifact source commit")
    arrays = _load_arrays(root, manifest)

    electrothermal_record = _mapping(
        manifest.get("electrothermal"),
        label="nested electrothermal record",
    )
    if set(electrothermal_record) != {
        "path",
        "manifest_sha256",
        "arrays_sha256",
        "plan_sha256",
        "layout_sha256",
    }:
        raise ValueError("nested electrothermal record has non-canonical fields")
    if _text(electrothermal_record.get("path"), label="electrothermal path") != (
        ELECTROTHERMAL_DIRECTORY
    ):
        raise ValueError("nested electrothermal path must use the canonical directory")
    nested_root = root / ELECTROTHERMAL_DIRECTORY
    if nested_root.is_symlink() or not nested_root.is_dir():
        raise ValueError("nested electrothermal path must be a regular directory")
    nested_manifest_path = nested_root / ELECTROTHERMAL_MANIFEST_FILENAME
    if _sha256_file(nested_manifest_path) != _sha256(
        electrothermal_record.get("manifest_sha256"),
        label="nested electrothermal manifest SHA-256",
    ):
        raise ValueError("nested electrothermal manifest SHA-256 mismatch")
    electrothermal = read_distributed_electrothermal_artifact(nested_root)
    if (
        _commit(
            electrothermal.manifest.get("source_commit"),
            label="nested source commit",
        )
        != source_commit
    ):
        raise ValueError("nested electrothermal source commit differs from coupled input")
    expected_nested_identity = (
        electrothermal.arrays_sha256,
        electrothermal.plan.digest(),
        electrothermal.plan.layout.digest(),
    )
    observed_nested_identity = (
        _sha256(electrothermal_record.get("arrays_sha256"), label="nested arrays SHA-256"),
        _sha256(electrothermal_record.get("plan_sha256"), label="nested plan SHA-256"),
        _sha256(electrothermal_record.get("layout_sha256"), label="nested layout SHA-256"),
    )
    if observed_nested_identity != expected_nested_identity:
        raise ValueError("nested electrothermal identity differs from its manifest")

    sampling_record = _mapping(manifest.get("sampling"), label="sampling record")
    plane_axes = _plane_axes(sampling_record.get("plane_axes"))
    sampling = build_triangle_p1_sampling_plan(
        _readonly(arrays["source_coordinates"], dtype=np.float64, label="source coordinates"),
        _readonly(arrays["source_cells"], dtype=np.int64, label="source cells"),
        tuple(
            _readonly(arrays[f"target_axis_{axis}"], dtype=np.float64, label="target axis")
            for axis in range(3)
        ),
        plane_axes=plane_axes,
        containment_tolerance=_number(
            sampling_record.get("containment_tolerance"),
            label="sampling containment tolerance",
        ),
    )
    if _sampling_metadata(sampling) != dict(sampling_record):
        raise ValueError("reconstructed canonical P1 sampler differs from its manifest")
    _verify_array(
        arrays["sampling_target_cell_indices"],
        sampling.target_cell_indices,
        label="sampling cell indices",
    )
    _verify_array(
        arrays["sampling_barycentric_weights"],
        sampling.barycentric_weights,
        label="sampling barycentric weights",
    )
    if not np.array_equal(
        np.asarray(sampling.source_cells),
        electrothermal.plan.layout.topology.cells,
    ):
        raise ValueError("canonical P1 sampler cell order differs from electrothermal input")

    transfer_record = _mapping(manifest.get("transfer"), label="distributed transfer record")
    mesh_axis_name = _text(
        transfer_record.get("mesh_axis_name"),
        label="distributed transfer mesh axis",
    )
    transfer = prepare_distributed_triangle_p1_sampling_plan(
        sampling,
        electrothermal.plan.layout.transport.cell_ids,
        source_layout_sha256=electrothermal.plan.layout.digest(),
        mesh_axis_name=mesh_axis_name,
    )
    if dict(transfer.canonical_data()) != dict(transfer_record):
        raise ValueError("reconstructed distributed transfer differs from its manifest")
    for name, expected in (
        ("transfer_source_cell_ids", transfer.source_cell_ids),
        ("transfer_send_source_cell_slots", transfer.send_source_cell_slots),
        ("transfer_send_barycentric_weights", transfer.send_barycentric_weights),
        ("transfer_send_active", transfer.send_active),
        ("transfer_receive_target_local_indices", transfer.receive_target_local_indices),
        ("transfer_receive_active", transfer.receive_active),
    ):
        _verify_array(arrays[name], expected, label=name.replace("_", " "))

    law = thermo_optic_law()
    law_record = _mapping(manifest.get("thermo_optic_law"), label="thermo-optic law")
    if dict(law_record) != {**dict(law.canonical_data()), "sha256": law.sha256}:
        raise ValueError("thermo-optic law differs from the physical-gate contract")
    contract = device_contract(sampling, parameter_dtype="float32")
    contract_record = _mapping(manifest.get("device_contract"), label="device contract")
    if dict(contract_record) != dict(contract.canonical_data()):
        raise ValueError("FDTDX device contract differs from the physical-gate contract")
    fdtdx_record = _mapping(manifest.get("fdtdx"), label="FDTDX identity")
    if dict(fdtdx_record) != _fdtdx_metadata():
        raise ValueError("FDTDX identity differs from the locked physical-gate source")
    scene = _mapping(manifest.get("scene"), label="FDTDX scene")
    time_steps = _integer(scene.get("time_steps"), label="scene time steps", positive=True)
    if dict(scene) != scene_metadata(time_steps=time_steps):
        raise ValueError("FDTDX scene differs from the canonical physical-gate contract")
    _canonical_json(dict(scene))
    arrays_record = _mapping(manifest.get("arrays"), label="artifact arrays")
    return LoadedDistributedFDTDXThermoOpticArtifact(
        electrothermal=electrothermal,
        sampling=sampling,
        transfer=transfer,
        law=law,
        contract=contract,
        scene=scene,
        manifest=manifest,
        arrays_sha256=_sha256(arrays_record.get("sha256"), label="artifact arrays SHA-256"),
    )


__all__ = [
    "ARRAYS_FILENAME",
    "ARTIFACT_SCHEMA",
    "ELECTROTHERMAL_DIRECTORY",
    "MANIFEST_FILENAME",
    "LoadedDistributedFDTDXThermoOpticArtifact",
    "read_distributed_fdtdx_thermo_optic_artifact",
    "write_distributed_fdtdx_thermo_optic_artifact",
]
