"""Immutable, memory-mapped controller input for the M5d physical TPU run.

This development artifact is not femx's canonical scientific result format.  It keeps the
controller-generated float64 authority separate from float32 TPU execution, stores every large
partition-leading array as an individual ``.npy`` file, and lets each worker transfer only its
addressable slices.  Global topology remains host-side routing metadata and is never presented as
accelerator state.
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
from pathlib import Path, PurePosixPath

import numpy as np

from femx.applications.ring_heater import PublicRingHeaterForwardPlan
from femx.backends.jax.scalar_collective import (
    ScalarH1CollectiveLayout,
    prepare_collective_scalar_h1_layout,
)
from femx.backends.jax.scalar_owned_ghost import prepare_scalar_h1_owned_ghost_topology
from femx.backends.jax.tet4_electrothermal import (
    HostPackedTet4ElectrothermalInputs,
    Tet4ElectrothermalRuntimePlan,
    pack_tet4_electrothermal_inputs_host,
    prepare_tet4_electrothermal_runtime_plan,
)

ARTIFACT_SCHEMA = "femx.public-ring-heater.tpu_forward_input/v1"
MANIFEST_FILENAME = "manifest.json"
ARRAY_DIRECTORY = "arrays"
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_ARRAY_BYTES = 1024 * 1024 * 1024
MAX_TOTAL_ARRAY_BYTES = 4 * 1024 * 1024 * 1024

TOPOLOGY_FIELDS = (
    "current_cells",
    "current_cell_owners",
    "current_free_nodes",
    "thermal_cells",
    "thermal_cell_owners",
    "thermal_free_nodes",
)
PACKED_INPUT_FIELDS = tuple(HostPackedTet4ElectrothermalInputs._fields)
AUTHORITY_FIELDS = ("authority_potential", "authority_temperature_rise")
OBSERVABLE_FIELDS = ("silicon_ring_cell_mask", "tin_heater_cell_mask")
ARRAY_FIELDS = (*TOPOLOGY_FIELDS, *PACKED_INPUT_FIELDS, *AUTHORITY_FIELDS, *OBSERVABLE_FIELDS)

_INTEGER_INPUTS = {
    "current_cell_local_dofs",
    "current_to_thermal_slots",
    "thermal_cell_local_dofs",
}
_MASK_INPUTS = {
    "current_owner_mask",
    "current_cell_mask",
    "thermal_owner_mask",
    "thermal_cell_mask",
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


def _number(value: object, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted) or (positive and converted <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return converted


def _sha256(value: object, *, label: str) -> str:
    result = _text(value, label=label)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return result


def _commit(value: object, *, label: str) -> str:
    result = _text(value, label=label)
    if len(result) != 40 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{label} must be a lowercase 40-character Git object ID")
    return result


def _safe_array_path(name: str) -> str:
    path = PurePosixPath(ARRAY_DIRECTORY, f"{name}.npy")
    if name not in ARRAY_FIELDS or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"invalid TPU input array name {name!r}")
    return path.as_posix()


def _readonly_float64(value: object, *, label: str, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind != "f" or raw.shape != shape:
        raise ValueError(f"{label} must be a real array shaped {shape}")
    result = np.array(raw, dtype=np.float64, order="C", copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain only finite values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class PublicRingHeaterTPUAuthority:
    """One full-node CPU float64 field pair and its admitted witness record."""

    potential: np.ndarray
    temperature: np.ndarray
    record: Mapping[str, object]

    def validate(self, plan: PublicRingHeaterForwardPlan) -> None:
        if not isinstance(plan, PublicRingHeaterForwardPlan):
            raise ValueError("ring-heater TPU authority requires a public forward plan")
        potential = _readonly_float64(
            self.potential,
            label="authority potential",
            shape=(plan.tet4.current_layout.topology.node_count,),
        )
        temperature = _readonly_float64(
            self.temperature,
            label="authority temperature",
            shape=(plan.tet4.thermal_layout.topology.node_count,),
        )
        record = dict(_mapping(self.record, label="authority record"))
        if record.get("schema_version") != "femx.public-ring-heater-forward.cpu-witness/v1":
            raise ValueError("authority record uses an unsupported schema")
        if record.get("status") != "passed" or record.get("profile") != "fine":
            raise ValueError("authority record must be one passing fine-mesh witness")
        numerics = _mapping(record.get("numerics"), label="authority numerics")
        if (
            _sha256(
                numerics.get("potential_sha256_float64"),
                label="authority potential SHA-256",
            )
            != hashlib.sha256(np.ascontiguousarray(potential, dtype="<f8").tobytes()).hexdigest()
        ):
            raise ValueError("authority potential disagrees with its witness hash")
        if (
            _sha256(
                numerics.get("temperature_sha256_float64"),
                label="authority temperature SHA-256",
            )
            != hashlib.sha256(np.ascontiguousarray(temperature, dtype="<f8").tobytes()).hexdigest()
        ):
            raise ValueError("authority temperature disagrees with its witness hash")
        excitation = _mapping(record.get("excitation"), label="authority excitation")
        _number(excitation.get("target_voltage_V"), label="authority target voltage", positive=True)
        _number(excitation.get("target_current_A"), label="authority target current", positive=True)
        _number(
            excitation.get("predicted_joule_power_W"),
            label="authority predicted Joule power",
            positive=True,
        )
        _canonical_json(record)
        object.__setattr__(self, "potential", potential)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "record", record)


@dataclass(frozen=True, slots=True)
class LoadedPublicRingHeaterTPUArtifact:
    """Verified runtime topology plus memory-mapped partition-leading arrays."""

    runtime_plan: Tet4ElectrothermalRuntimePlan
    arrays: Mapping[str, np.ndarray]
    manifest: Mapping[str, object]
    logical_sha256: str


def _pack_owned(layout: ScalarH1CollectiveLayout, full_values: np.ndarray) -> np.ndarray:
    free_nodes = np.asarray(layout.topology.free_nodes, dtype=np.int64)
    free = np.asarray(full_values)[free_nodes]
    extended = np.concatenate((free, np.zeros((1,), dtype=free.dtype)))
    return np.ascontiguousarray(extended[layout.transport.owned_dof_ids], dtype=np.float32)


def _pack_cell_membership(
    layout: ScalarH1CollectiveLayout,
    cell_ids: object,
    *,
    label: str,
) -> np.ndarray:
    raw = np.asarray(cell_ids)
    if raw.dtype.kind not in "iu" or raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{label} must be a nonempty rank-1 integer array")
    canonical = np.asarray(raw, dtype=np.int64)
    if np.any(canonical < 0) or np.any(canonical >= layout.topology.cell_count):
        raise ValueError(f"{label} contains an out-of-range cell")
    if np.unique(canonical).shape[0] != canonical.shape[0]:
        raise ValueError(f"{label} must contain unique cells")
    membership = np.zeros((layout.topology.cell_count + 1,), dtype=np.bool_)
    membership[canonical] = True
    return np.ascontiguousarray(membership[layout.transport.cell_ids])


def _topology_arrays(plan: PublicRingHeaterForwardPlan) -> dict[str, np.ndarray]:
    current = plan.tet4.current_layout.topology
    thermal = plan.tet4.thermal_layout.topology
    return {
        "current_cells": np.asarray(current.cells, dtype=np.int64),
        "current_cell_owners": np.asarray(current.owned_ghost.cell_owners, dtype=np.int64),
        "current_free_nodes": np.asarray(current.free_nodes, dtype=np.int64),
        "thermal_cells": np.asarray(thermal.cells, dtype=np.int64),
        "thermal_cell_owners": np.asarray(thermal.owned_ghost.cell_owners, dtype=np.int64),
        "thermal_free_nodes": np.asarray(thermal.free_nodes, dtype=np.int64),
    }


def _array_role(name: str) -> str:
    if name in TOPOLOGY_FIELDS:
        return "host_runtime_topology"
    if name in PACKED_INPUT_FIELDS:
        return "partitioned_solver_input"
    if name in AUTHORITY_FIELDS:
        return "partitioned_float32_rounded_cpu_authority"
    return "partitioned_observable_membership"


def _storage_data(layout: ScalarH1CollectiveLayout) -> dict[str, object]:
    report = layout.transport.storage_report
    return {
        "cell_capacity": layout.cell_capacity,
        "owned_dof_capacity": layout.owned_dof_capacity,
        "ghost_dof_capacity": layout.transport.ghost_dof_capacity,
        "actual_cell_slots": report.actual_cell_slots,
        "allocated_cell_slots": report.allocated_cell_slots,
        "actual_owned_dof_slots": report.actual_owned_dof_slots,
        "allocated_owned_dof_slots": report.allocated_owned_dof_slots,
        "actual_ghost_dof_slots": report.actual_ghost_dof_slots,
        "allocated_ghost_dof_slots": report.allocated_ghost_dof_slots,
        "halo_link_count": report.halo_link_count,
        "halo_value_count": report.halo_value_count,
    }


def write_public_ring_heater_tpu_artifact(
    output_root: Path,
    plan: PublicRingHeaterForwardPlan,
    authority: PublicRingHeaterTPUAuthority,
    *,
    source_commit: str,
    source_msh_sha256: str,
    partition_owner_sha256: str,
    silicon_ring_cell_ids: object,
    tin_heater_cell_ids: object,
) -> Mapping[str, object]:
    """Publish one new immutable fine-mesh float32 input directory."""

    root = output_root.resolve(strict=False)
    if not output_root.is_absolute() or root.exists() or output_root.is_symlink():
        raise ValueError("ring-heater TPU artifact root must be a new absolute path")
    if not isinstance(plan, PublicRingHeaterForwardPlan):
        raise ValueError("ring-heater TPU artifact requires a public forward plan")
    if not isinstance(authority, PublicRingHeaterTPUAuthority):
        raise ValueError("ring-heater TPU artifact requires a typed CPU authority")
    commit = _commit(source_commit, label="artifact source commit")
    mesh_sha256 = _sha256(source_msh_sha256, label="source MSH SHA-256")
    owner_sha256 = _sha256(partition_owner_sha256, label="partition-owner SHA-256")
    authority.validate(plan)
    runtime_plan = prepare_tet4_electrothermal_runtime_plan(plan.tet4)
    host_inputs = pack_tet4_electrothermal_inputs_host(plan.tet4, value_dtype=np.float32)

    arrays = _topology_arrays(plan)
    arrays.update(
        {
            name: np.asarray(value)
            for name, value in zip(host_inputs._fields, host_inputs, strict=True)
        }
    )
    arrays["authority_potential"] = _pack_owned(
        runtime_plan.current_layout,
        authority.potential,
    )
    arrays["authority_temperature_rise"] = _pack_owned(
        runtime_plan.thermal_layout,
        authority.temperature - runtime_plan.thermal_reference,
    )
    arrays["silicon_ring_cell_mask"] = _pack_cell_membership(
        runtime_plan.thermal_layout,
        silicon_ring_cell_ids,
        label="silicon-ring cells",
    )
    arrays["tin_heater_cell_mask"] = _pack_cell_membership(
        runtime_plan.thermal_layout,
        tin_heater_cell_ids,
        label="TiN-heater cells",
    )
    if set(arrays) != set(ARRAY_FIELDS):
        raise ValueError("ring-heater TPU artifact arrays do not match the exact schema")

    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        array_root = temporary / ARRAY_DIRECTORY
        array_root.mkdir()
        records: dict[str, object] = {}
        total_file_bytes = 0
        for name in ARRAY_FIELDS:
            value = np.ascontiguousarray(arrays[name])
            path = temporary / _safe_array_path(name)
            with path.open("xb") as stream:
                np.save(stream, value, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            size = path.stat().st_size
            if size <= 0 or size > MAX_ARRAY_BYTES:
                raise ValueError(f"ring-heater TPU array {name!r} exceeds its size limit")
            total_file_bytes += size
            records[name] = {
                "path": _safe_array_path(name),
                "sha256": _sha256_file(path),
                "byte_count": size,
                "logical_byte_count": value.nbytes,
                "dtype": value.dtype.name,
                "shape": list(value.shape),
                "role": _array_role(name),
                "pickle_allowed": False,
            }
        if total_file_bytes > MAX_TOTAL_ARRAY_BYTES:
            raise ValueError("ring-heater TPU arrays exceed the aggregate size limit")

        excitation = _mapping(authority.record.get("excitation"), label="authority excitation")
        manifest: dict[str, object] = {
            "schema_version": ARTIFACT_SCHEMA,
            "source_commit": commit,
            "source_msh_sha256": mesh_sha256,
            "partition": {
                "algorithm": "balanced_lexicographic_cell_centroid/v1",
                "primary_axis": "x",
                "partition_count": runtime_plan.thermal_layout.partition_count,
                "owner_sha256": owner_sha256,
                "guaranteed_cell_count_imbalance": 1,
            },
            "model": {
                "forward_schema_version": plan.schema_version,
                "forward_plan_sha256": plan.digest(),
                "reference_sha256": plan.reference.digest(),
                "mesh_report_sha256": plan.mesh_report.digest(),
                "recipe_sha256": plan.mesh_report.recipe_sha256,
                "import_record_sha256": plan.mesh_report.import_record_sha256,
                "node_count": plan.mesh_report.node_count,
                "tetrahedron_count": plan.mesh_report.tetrahedron_count,
                "conductor_node_count": plan.tet4.current_layout.topology.node_count,
                "conductor_tetrahedron_count": plan.tet4.current_layout.topology.cell_count,
                "claim_scope": "uncalibrated public benchmark; not a fabricated-device prediction",
            },
            "runtime_plan": {
                "schema_version": runtime_plan.schema_version,
                "sha256": runtime_plan.digest(),
                "source_plan_sha256": runtime_plan.source_plan_sha256,
                "current_layout_sha256": runtime_plan.current_layout.digest(),
                "thermal_layout_sha256": runtime_plan.thermal_layout.digest(),
                "thermal_reference_K": runtime_plan.thermal_reference,
                "current_storage": _storage_data(runtime_plan.current_layout),
                "thermal_storage": _storage_data(runtime_plan.thermal_layout),
                "host_topology_policy": (
                    "global immutable routing metadata may be memory-mapped on each worker; "
                    "partition-leading numerical arrays transfer only addressable slices"
                ),
            },
            "authority": {
                "record": dict(authority.record),
                "comparison_dtype": "float32-rounded-from-controller-float64",
                "target_voltage_V": _number(
                    excitation.get("target_voltage_V"),
                    label="authority target voltage",
                    positive=True,
                ),
                "target_current_A": _number(
                    excitation.get("target_current_A"),
                    label="authority target current",
                    positive=True,
                ),
                "predicted_joule_power_W": _number(
                    excitation.get("predicted_joule_power_W"),
                    label="authority predicted Joule power",
                    positive=True,
                ),
                "scope": (
                    "single-device CPU float64 same-mesh authority; stored device comparison "
                    "vectors are explicitly rounded to float32"
                ),
            },
            "arrays": records,
            "total_array_file_bytes": total_file_bytes,
        }
        logical_sha256 = hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()
        manifest["logical_sha256"] = logical_sha256
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
        raise ValueError("ring-heater TPU manifest must be a regular non-symlink file")
    size = path.stat().st_size
    if size <= 0 or size > MAX_MANIFEST_BYTES:
        raise ValueError("ring-heater TPU manifest size is outside the admitted range")
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_constant,
    )
    return _mapping(value, label="ring-heater TPU manifest")


def _shape(value: object, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an integer array")
    shape = tuple(_integer(item, label=f"{label} extent") for item in value)
    if not shape or any(extent <= 0 for extent in shape):
        raise ValueError(f"{label} must contain positive extents")
    return shape


def _load_array(
    root: Path,
    name: str,
    value: object,
) -> np.ndarray:
    record = _mapping(value, label=f"array record {name}")
    expected_keys = {
        "path",
        "sha256",
        "byte_count",
        "logical_byte_count",
        "dtype",
        "shape",
        "role",
        "pickle_allowed",
    }
    if set(record) != expected_keys:
        raise ValueError(f"array record {name!r} has unexpected fields")
    relative = _text(record.get("path"), label=f"array {name} path")
    if relative != _safe_array_path(name):
        raise ValueError(f"array {name!r} does not use its canonical path")
    if record.get("pickle_allowed") is not False:
        raise ValueError(f"array {name!r} must explicitly forbid pickle")
    if _text(record.get("role"), label=f"array {name} role") != _array_role(name):
        raise ValueError(f"array {name!r} has the wrong semantic role")
    path = root / PurePosixPath(relative)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"array {name!r} must be a regular non-symlink file")
    byte_count = _integer(record.get("byte_count"), label=f"array {name} bytes", positive=True)
    if byte_count > MAX_ARRAY_BYTES or path.stat().st_size != byte_count:
        raise ValueError(f"array {name!r} byte count is outside the admitted contract")
    if _sha256_file(path) != _sha256(record.get("sha256"), label=f"array {name} SHA-256"):
        raise ValueError(f"array {name!r} SHA-256 mismatch")
    expected_shape = _shape(record.get("shape"), label=f"array {name} shape")
    expected_dtype = _text(record.get("dtype"), label=f"array {name} dtype")
    try:
        loaded = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"array {name!r} is not a valid pickle-free NPY file") from error
    if not isinstance(loaded, np.ndarray) or loaded.dtype.kind == "O":
        raise ValueError(f"array {name!r} must contain a numeric NPY array")
    if loaded.shape != expected_shape or loaded.dtype.name != expected_dtype:
        raise ValueError(f"array {name!r} metadata disagrees with the NPY payload")
    logical_bytes = _integer(
        record.get("logical_byte_count"),
        label=f"array {name} logical bytes",
        positive=True,
    )
    if loaded.nbytes != logical_bytes:
        raise ValueError(f"array {name!r} logical byte count is inconsistent")
    return loaded


def _expected_array_contract(
    runtime_plan: Tet4ElectrothermalRuntimePlan,
) -> dict[str, tuple[tuple[int, ...], str]]:
    current = runtime_plan.current_layout
    thermal = runtime_plan.thermal_layout
    current_cells = (current.partition_count, current.cell_capacity)
    thermal_cells = (thermal.partition_count, thermal.cell_capacity)
    current_owners = (current.partition_count, current.owned_dof_capacity)
    thermal_owners = (thermal.partition_count, thermal.owned_dof_capacity)
    result: dict[str, tuple[tuple[int, ...], str]] = {
        "current_cells": (current.topology.cells.shape, "int64"),
        "current_cell_owners": ((current.topology.cell_count,), "int64"),
        "current_free_nodes": ((current.topology.free_dof_count,), "int64"),
        "thermal_cells": (thermal.topology.cells.shape, "int64"),
        "thermal_cell_owners": ((thermal.topology.cell_count,), "int64"),
        "thermal_free_nodes": ((thermal.topology.free_dof_count,), "int64"),
        "current_cell_local_dofs": ((*current_cells, 4), "int32"),
        "current_owner_mask": (current_owners, "bool"),
        "current_cell_mask": (current_cells, "bool"),
        "current_conduction_stiffness": ((*current_cells, 4, 4), "float32"),
        "current_basis_gradients": ((*current_cells, 4, 3), "float32"),
        "current_cell_volumes": (current_cells, "float32"),
        "current_conductivity": (current_cells, "float32"),
        "current_cell_load": ((*current_cells, 4), "float32"),
        "current_cell_dirichlet_base": ((*current_cells, 4), "float32"),
        "current_cell_dirichlet_scale": ((*current_cells, 4), "float32"),
        "current_to_thermal_slots": (current_cells, "int32"),
        "thermal_cell_local_dofs": ((*thermal_cells, 4), "int32"),
        "thermal_owner_mask": (thermal_owners, "bool"),
        "thermal_cell_mask": (thermal_cells, "bool"),
        "thermal_conduction_stiffness": ((*thermal_cells, 4, 4), "float32"),
        "thermal_robin_matrix": ((*thermal_cells, 4, 4), "float32"),
        "thermal_cell_volumes": (thermal_cells, "float32"),
        "thermal_nonrobin_load": ((*thermal_cells, 4), "float32"),
        "thermal_robin_ambient_load": ((*thermal_cells, 4), "float32"),
        "thermal_cell_dirichlet_shifted": ((*thermal_cells, 4), "float32"),
        "authority_potential": (current_owners, "float32"),
        "authority_temperature_rise": (thermal_owners, "float32"),
        "silicon_ring_cell_mask": (thermal_cells, "bool"),
        "tin_heater_cell_mask": (thermal_cells, "bool"),
    }
    return result


def _validate_loaded_arrays(
    arrays: Mapping[str, np.ndarray],
    runtime_plan: Tet4ElectrothermalRuntimePlan,
) -> None:
    expected = _expected_array_contract(runtime_plan)
    if set(arrays) != set(expected):
        raise ValueError("ring-heater TPU artifact arrays do not match the exact schema")
    for name, (shape, dtype) in expected.items():
        array = arrays[name]
        if array.shape != shape or array.dtype.name != dtype:
            raise ValueError(f"ring-heater TPU array {name!r} violates its runtime contract")
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise ValueError(f"ring-heater TPU array {name!r} contains a non-finite value")
    current = runtime_plan.current_layout
    thermal = runtime_plan.thermal_layout
    exact_arrays = {
        "current_cell_local_dofs": np.asarray(current.transport.cell_local_dofs, dtype=np.int32),
        "current_owner_mask": current.transport.owned_dof_ids < current.topology.free_dof_count,
        "current_cell_mask": current.transport.cell_ids < current.topology.cell_count,
        "thermal_cell_local_dofs": np.asarray(thermal.transport.cell_local_dofs, dtype=np.int32),
        "thermal_owner_mask": thermal.transport.owned_dof_ids < thermal.topology.free_dof_count,
        "thermal_cell_mask": thermal.transport.cell_ids < thermal.topology.cell_count,
    }
    for name, expected_value in exact_arrays.items():
        if not np.array_equal(arrays[name], expected_value):
            raise ValueError(f"ring-heater TPU array {name!r} disagrees with runtime topology")
    for name in OBSERVABLE_FIELDS:
        mask = np.asarray(arrays[name], dtype=np.bool_)
        if not np.any(mask) or np.any(mask & ~np.asarray(arrays["thermal_cell_mask"])):
            raise ValueError(f"ring-heater TPU observable mask {name!r} is invalid")


def read_public_ring_heater_tpu_artifact(
    input_root: Path,
) -> LoadedPublicRingHeaterTPUArtifact:
    """Verify one artifact and reconstruct only the runtime topology."""

    root = input_root.resolve(strict=True)
    if not root.is_dir() or input_root.is_symlink():
        raise ValueError("ring-heater TPU artifact root must be a non-symlink directory")
    manifest = _load_manifest(root)
    if manifest.get("schema_version") != ARTIFACT_SCHEMA:
        raise ValueError("unsupported ring-heater TPU artifact schema")
    _commit(manifest.get("source_commit"), label="artifact source commit")
    _sha256(manifest.get("source_msh_sha256"), label="source MSH SHA-256")
    declared_logical = _sha256(manifest.get("logical_sha256"), label="artifact logical SHA-256")
    logical_payload = dict(manifest)
    logical_payload.pop("logical_sha256")
    observed_logical = hashlib.sha256(_canonical_json(logical_payload).encode("utf-8")).hexdigest()
    if observed_logical != declared_logical:
        raise ValueError("ring-heater TPU artifact logical SHA-256 mismatch")

    raw_records = _mapping(manifest.get("arrays"), label="artifact arrays")
    if set(raw_records) != set(ARRAY_FIELDS):
        raise ValueError("ring-heater TPU artifact array names do not match the schema")
    total_bytes = _integer(
        manifest.get("total_array_file_bytes"),
        label="total array file bytes",
        positive=True,
    )
    if total_bytes > MAX_TOTAL_ARRAY_BYTES:
        raise ValueError("ring-heater TPU artifact exceeds its aggregate size limit")
    arrays = {name: _load_array(root, name, raw_records[name]) for name in ARRAY_FIELDS}
    if sum((root / _safe_array_path(name)).stat().st_size for name in ARRAY_FIELDS) != total_bytes:
        raise ValueError("ring-heater TPU aggregate byte count is inconsistent")

    runtime_record = _mapping(manifest.get("runtime_plan"), label="runtime plan")
    model_record = _mapping(manifest.get("model"), label="model")
    partition_record = _mapping(manifest.get("partition"), label="partition")
    partition_count = _integer(
        partition_record.get("partition_count"),
        label="partition count",
        positive=True,
    )
    if partition_record.get("algorithm") != "balanced_lexicographic_cell_centroid/v1":
        raise ValueError("ring-heater TPU artifact uses an unsupported partition algorithm")
    if partition_record.get("primary_axis") != "x":
        raise ValueError("ring-heater TPU partition must use x as its primary axis")
    if (
        _integer(
            partition_record.get("guaranteed_cell_count_imbalance"),
            label="partition imbalance",
        )
        != 1
    ):
        raise ValueError("ring-heater TPU partition imbalance contract changed")
    _sha256(partition_record.get("owner_sha256"), label="partition-owner SHA-256")

    current_topology = prepare_scalar_h1_owned_ghost_topology(
        arrays["current_cells"],
        arrays["current_cell_owners"],
        node_count=_integer(
            model_record.get("conductor_node_count"),
            label="conductor node count",
            positive=True,
        ),
        free_nodes=arrays["current_free_nodes"],
        partition_count=partition_count,
    )
    thermal_topology = prepare_scalar_h1_owned_ghost_topology(
        arrays["thermal_cells"],
        arrays["thermal_cell_owners"],
        node_count=_integer(model_record.get("node_count"), label="node count", positive=True),
        free_nodes=arrays["thermal_free_nodes"],
        partition_count=partition_count,
    )
    runtime_plan = Tet4ElectrothermalRuntimePlan(
        current_layout=prepare_collective_scalar_h1_layout(current_topology),
        thermal_layout=prepare_collective_scalar_h1_layout(thermal_topology),
        thermal_reference=_number(
            runtime_record.get("thermal_reference_K"),
            label="thermal reference",
        ),
        source_plan_sha256=_sha256(
            runtime_record.get("source_plan_sha256"),
            label="source plan SHA-256",
        ),
    )
    if runtime_record.get("schema_version") != runtime_plan.schema_version:
        raise ValueError("ring-heater TPU runtime-plan schema mismatch")
    identities = {
        "sha256": runtime_plan.digest(),
        "current_layout_sha256": runtime_plan.current_layout.digest(),
        "thermal_layout_sha256": runtime_plan.thermal_layout.digest(),
    }
    for name, observed in identities.items():
        if _sha256(runtime_record.get(name), label=f"runtime-plan {name}") != observed:
            raise ValueError(f"ring-heater TPU runtime-plan {name} mismatch")
    _validate_loaded_arrays(arrays, runtime_plan)
    return LoadedPublicRingHeaterTPUArtifact(
        runtime_plan=runtime_plan,
        arrays=arrays,
        manifest=manifest,
        logical_sha256=declared_logical,
    )


def read_public_ring_heater_cpu_authority(
    record_path: Path,
    state_path: Path,
) -> PublicRingHeaterTPUAuthority:
    """Read the small controller-side CPU witness pair before plan-specific validation."""

    for path, label, maximum in (
        (record_path, "authority record", MAX_MANIFEST_BYTES),
        (state_path, "authority state", 64 * 1024 * 1024),
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{label} must be a regular non-symlink file")
        if path.stat().st_size <= 0 or path.stat().st_size > maximum:
            raise ValueError(f"{label} size is outside the admitted range")
    record = json.loads(
        record_path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_constant,
    )
    with np.load(state_path, allow_pickle=False) as archive:
        if set(archive.files) != {"potential", "temperature"}:
            raise ValueError("authority state must contain exactly potential and temperature")
        potential = np.asarray(archive["potential"])
        temperature = np.asarray(archive["temperature"])
    if potential.dtype.kind == "O" or temperature.dtype.kind == "O":
        raise ValueError("authority state cannot contain object arrays")
    return PublicRingHeaterTPUAuthority(
        potential=potential,
        temperature=temperature,
        record=_mapping(record, label="authority record"),
    )
