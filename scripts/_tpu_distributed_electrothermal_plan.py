"""Strict local-float64 input artifact for the physical TPU electrothermal witness.

The artifact is deliberately a development-run input, not a public femx numerical format.  It
keeps host lowering and the dense authority off the TPU workers.  Workers reconstruct the exact
collective layout, verify both logical digests, and only then lower numerical inputs to float32.
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

from femx.backends.jax.distributed_electrothermal import (
    DISTRIBUTED_ELECTROTHERMAL_SCHEMA,
    DistributedElectrothermalPlan,
    _ScalarAffineFields,
)
from femx.backends.jax.scalar_collective import prepare_collective_scalar_h1_layout
from femx.backends.jax.scalar_owned_ghost import prepare_scalar_h1_owned_ghost_topology
from femx.workflows import CoupledIterationPolicy

ARTIFACT_SCHEMA = "femx.jax.distributed_electrothermal.tpu_plan/v1"
ARRAYS_FILENAME = "arrays.npz"
MANIFEST_FILENAME = "manifest.json"
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_ARRAY_BYTES = 256 * 1024 * 1024

_AFFINE_NAMES = (
    "conductivity_base",
    "conductivity_weights",
    "cell_load_base",
    "cell_load_weights",
    "cell_dirichlet_base",
    "cell_dirichlet_weights",
    "node_dirichlet_base",
    "node_dirichlet_weights",
    "reference_base",
    "reference_weights",
)
_DIRECT_PLAN_ARRAYS = (
    "unit_stiffness",
    "basis_gradients",
    "cell_areas",
    "feedback_reference_base",
    "feedback_reference_weights",
    "feedback_coefficient_base",
    "feedback_coefficient_weights",
    "current_initial",
    "thermal_initial",
    "feedback_initial",
    "current_lower_bounds",
    "current_upper_bounds",
    "thermal_lower_bounds",
    "thermal_upper_bounds",
    "feedback_lower_bounds",
    "feedback_upper_bounds",
)
_AUTHORITY_ARRAYS = (
    "potential",
    "temperature",
    "current_parameter_gradient",
    "thermal_parameter_gradient",
    "feedback_parameter_gradient",
    "temperature_cotangent",
)


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


def _boolean(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _sha256(value: object, *, label: str) -> str:
    result = _text(value, label=label)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{label} must be a canonical lowercase SHA-256")
    return result


def _string_tuple(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    result = tuple(_text(item, label=f"{label} item") for item in value)
    if len(result) != len(set(result)) and label.endswith("names"):
        raise ValueError(f"{label} must be unique")
    return result


def _readonly_float64(value: object, *, label: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind != "f" or raw.dtype.itemsize != 8:
        raise ValueError(f"{label} must use float64")
    result = np.array(raw, dtype=np.float64, order="C", copy=True)
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{label} must contain only finite values")
    result.setflags(write=False)
    return result


def _readonly_int64(value: object, *, label: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu":
        raise ValueError(f"{label} must use an integer dtype")
    result = np.array(raw, dtype=np.int64, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class DistributedElectrothermalAuthority:
    """Dense float64 authority paired with one exact distributed plan."""

    potential: np.ndarray
    temperature: np.ndarray
    current_parameter_gradient: np.ndarray
    thermal_parameter_gradient: np.ndarray
    feedback_parameter_gradient: np.ndarray
    temperature_cotangent: np.ndarray
    objective: float
    forward_converged: bool
    adjoint_converged: bool
    diagnostics: Mapping[str, object]

    def validate(self, plan: DistributedElectrothermalPlan) -> None:
        expected_shapes = {
            "potential": (plan.layout.topology.node_count,),
            "temperature": (plan.layout.topology.node_count,),
            "current_parameter_gradient": (len(plan.current_parameter_names),),
            "thermal_parameter_gradient": (len(plan.thermal_parameter_names),),
            "feedback_parameter_gradient": (len(plan.feedback_parameter_names),),
            "temperature_cotangent": (plan.layout.topology.node_count,),
        }
        for name, shape in expected_shapes.items():
            array = _readonly_float64(getattr(self, name), label=f"authority {name}")
            if array.shape != shape:
                raise ValueError(f"authority {name} must have shape {shape}")
            object.__setattr__(self, name, array)
        if not math.isfinite(float(self.objective)):
            raise ValueError("authority objective must be finite")
        if not isinstance(self.forward_converged, bool) or not isinstance(
            self.adjoint_converged, bool
        ):
            raise ValueError("authority convergence flags must be boolean")
        _canonical_json(dict(self.diagnostics))


@dataclass(frozen=True, slots=True)
class LoadedDistributedElectrothermalArtifact:
    """Verified reconstructed plan, authority, and immutable manifest."""

    plan: DistributedElectrothermalPlan
    authority: DistributedElectrothermalAuthority
    manifest: Mapping[str, object]
    arrays_sha256: str


def _plan_arrays(
    plan: DistributedElectrothermalPlan,
    authority: DistributedElectrothermalAuthority,
) -> dict[str, np.ndarray]:
    topology = plan.layout.topology
    result = {
        "topology_cells": np.asarray(topology.cells, dtype=np.int64),
        "topology_cell_owners": np.asarray(topology.owned_ghost.cell_owners, dtype=np.int64),
        "topology_free_nodes": np.asarray(topology.free_nodes, dtype=np.int64),
    }
    for name in _DIRECT_PLAN_ARRAYS:
        result[f"plan_{name}"] = np.asarray(getattr(plan, name), dtype=np.float64)
    for prefix, fields in (("current", plan.current), ("thermal", plan.thermal)):
        for name in _AFFINE_NAMES:
            result[f"plan_{prefix}_{name}"] = np.asarray(
                getattr(fields, name),
                dtype=np.float64,
            )
    for name in _AUTHORITY_ARRAYS:
        result[f"authority_{name}"] = np.asarray(getattr(authority, name), dtype=np.float64)
    return {name: np.array(result[name], order="C", copy=True) for name in sorted(result)}


def write_distributed_electrothermal_artifact(
    output_root: Path,
    plan: DistributedElectrothermalPlan,
    authority: DistributedElectrothermalAuthority,
    *,
    source_commit: str,
    case_metadata: Mapping[str, object],
) -> Mapping[str, object]:
    """Publish one new immutable plan directory and return its manifest."""

    root = output_root.resolve(strict=False)
    if not output_root.is_absolute() or root.exists() or output_root.is_symlink():
        raise ValueError("TPU electrothermal artifact root must be a new absolute path")
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise ValueError("source commit must be a canonical 40-character Git object ID")
    authority.validate(plan)
    _canonical_json(dict(case_metadata))
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        arrays_path = temporary / ARRAYS_FILENAME
        np.savez(
            arrays_path,
            **_plan_arrays(plan, authority),  # type: ignore[arg-type]
        )
        arrays_sha256 = _sha256_file(arrays_path)
        manifest: dict[str, object] = {
            "schema_version": ARTIFACT_SCHEMA,
            "source_commit": source_commit,
            "plan": {
                "schema_version": plan.schema_version,
                "sha256": plan.digest(),
                "layout_sha256": plan.layout.digest(),
                "partition_count": plan.layout.partition_count,
                "node_count": plan.layout.topology.node_count,
                "cell_count": plan.layout.topology.cell_count,
                "free_dof_count": plan.layout.topology.free_dof_count,
                "current_parameter_names": list(plan.current_parameter_names),
                "current_parameter_units": list(plan.current_parameter_units),
                "thermal_parameter_names": list(plan.thermal_parameter_names),
                "thermal_parameter_units": list(plan.thermal_parameter_units),
                "feedback_parameter_names": list(plan.feedback_parameter_names),
                "feedback_parameter_units": list(plan.feedback_parameter_units),
                "iteration_policy": dict(plan.iteration_policy.canonical_data()),
            },
            "authority": {
                "dtype": "float64",
                "objective": float(authority.objective),
                "forward_converged": authority.forward_converged,
                "adjoint_converged": authority.adjoint_converged,
                "diagnostics": dict(authority.diagnostics),
                "scope": (
                    "controller-generated dense float64 same-discretization authority; "
                    "not measured-device or foundry evidence"
                ),
            },
            "arrays": {
                "path": ARRAYS_FILENAME,
                "sha256": arrays_sha256,
                "byte_count": arrays_path.stat().st_size,
                "pickle_allowed": False,
            },
            "case": dict(case_metadata),
        }
        manifest_path = temporary / MANIFEST_FILENAME
        manifest_path.write_text(
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
        raise ValueError("TPU electrothermal manifest must be a regular non-symlink file")
    size = path.stat().st_size
    if size <= 0 or size > MAX_MANIFEST_BYTES:
        raise ValueError("TPU electrothermal manifest size is outside the admitted range")
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_constant,
    )
    return _mapping(value, label="TPU electrothermal manifest")


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
    if (
        size <= 0
        or size > MAX_ARRAY_BYTES
        or size
        != _integer(
            record.get("byte_count"),
            label="artifact arrays byte count",
            positive=True,
        )
    ):
        raise ValueError("artifact arrays byte count is outside the admitted contract")
    expected_sha256 = _sha256(record.get("sha256"), label="artifact arrays SHA-256")
    if _sha256_file(path) != expected_sha256:
        raise ValueError("artifact arrays SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    expected = {
        "topology_cells",
        "topology_cell_owners",
        "topology_free_nodes",
        *(f"plan_{name}" for name in _DIRECT_PLAN_ARRAYS),
        *(f"plan_{prefix}_{name}" for prefix in ("current", "thermal") for name in _AFFINE_NAMES),
        *(f"authority_{name}" for name in _AUTHORITY_ARRAYS),
    }
    if set(arrays) != expected:
        raise ValueError("artifact arrays do not match the exact admitted field set")
    if any(array.dtype.kind == "O" for array in arrays.values()):
        raise ValueError("artifact arrays cannot contain object dtype values")
    return arrays


def _iteration_policy(value: object) -> CoupledIterationPolicy:
    record = _mapping(value, label="plan iteration policy")
    if record.get("algorithm") != "block_gauss_seidel":
        raise ValueError("plan iteration policy requires block Gauss-Seidel")
    return CoupledIterationPolicy(
        max_iterations=_integer(
            record.get("max_iterations"), label="max iterations", positive=True
        ),
        minimum_iterations=_integer(
            record.get("minimum_iterations"),
            label="minimum iterations",
            positive=True,
        ),
        relative_tolerance=_number(record.get("relative_tolerance"), label="relative tolerance"),
        residual_tolerance=_number(record.get("residual_tolerance"), label="residual tolerance"),
        potential_absolute_tolerance=_number(
            record.get("potential_absolute_tolerance_V"),
            label="potential absolute tolerance",
        ),
        temperature_absolute_tolerance=_number(
            record.get("temperature_absolute_tolerance_K"),
            label="temperature absolute tolerance",
        ),
        potential_relaxation=_number(
            record.get("potential_relaxation"),
            label="potential relaxation",
        ),
        temperature_relaxation=_number(
            record.get("temperature_relaxation"),
            label="temperature relaxation",
        ),
    )


def _affine(arrays: Mapping[str, np.ndarray], prefix: str) -> _ScalarAffineFields:
    return _ScalarAffineFields(
        *(
            _readonly_float64(
                arrays[f"plan_{prefix}_{name}"],
                label=f"plan {prefix} {name}",
            )
            for name in _AFFINE_NAMES
        )
    )


def read_distributed_electrothermal_artifact(
    input_root: Path,
) -> LoadedDistributedElectrothermalArtifact:
    """Read, reconstruct, and digest-check one immutable float64 plan artifact."""

    root = input_root.resolve(strict=True)
    if not root.is_dir() or input_root.is_symlink():
        raise ValueError("TPU electrothermal artifact root must be a non-symlink directory")
    manifest = _load_manifest(root)
    if manifest.get("schema_version") != ARTIFACT_SCHEMA:
        raise ValueError("unsupported TPU electrothermal artifact schema")
    _text(manifest.get("source_commit"), label="artifact source commit")
    plan_record = _mapping(manifest.get("plan"), label="artifact plan")
    if plan_record.get("schema_version") != DISTRIBUTED_ELECTROTHERMAL_SCHEMA:
        raise ValueError("artifact uses an unsupported distributed electrothermal plan schema")
    arrays = _load_arrays(root, manifest)
    partition_count = _integer(
        plan_record.get("partition_count"),
        label="plan partition count",
        positive=True,
    )
    node_count = _integer(plan_record.get("node_count"), label="plan node count", positive=True)
    topology = prepare_scalar_h1_owned_ghost_topology(
        _readonly_int64(arrays["topology_cells"], label="topology cells"),
        _readonly_int64(arrays["topology_cell_owners"], label="topology cell owners"),
        node_count=node_count,
        free_nodes=_readonly_int64(arrays["topology_free_nodes"], label="topology free nodes"),
        partition_count=partition_count,
    )
    layout = prepare_collective_scalar_h1_layout(topology)
    direct = {
        name: _readonly_float64(arrays[f"plan_{name}"], label=f"plan {name}")
        for name in _DIRECT_PLAN_ARRAYS
    }
    plan = DistributedElectrothermalPlan(
        layout=layout,
        unit_stiffness=direct["unit_stiffness"],
        basis_gradients=direct["basis_gradients"],
        cell_areas=direct["cell_areas"],
        current=_affine(arrays, "current"),
        thermal=_affine(arrays, "thermal"),
        feedback_reference_base=direct["feedback_reference_base"],
        feedback_reference_weights=direct["feedback_reference_weights"],
        feedback_coefficient_base=direct["feedback_coefficient_base"],
        feedback_coefficient_weights=direct["feedback_coefficient_weights"],
        current_initial=direct["current_initial"],
        thermal_initial=direct["thermal_initial"],
        feedback_initial=direct["feedback_initial"],
        current_lower_bounds=direct["current_lower_bounds"],
        current_upper_bounds=direct["current_upper_bounds"],
        thermal_lower_bounds=direct["thermal_lower_bounds"],
        thermal_upper_bounds=direct["thermal_upper_bounds"],
        feedback_lower_bounds=direct["feedback_lower_bounds"],
        feedback_upper_bounds=direct["feedback_upper_bounds"],
        current_parameter_names=_string_tuple(
            plan_record.get("current_parameter_names"),
            label="current parameter names",
        ),
        current_parameter_units=_string_tuple(
            plan_record.get("current_parameter_units"),
            label="current parameter units",
        ),
        thermal_parameter_names=_string_tuple(
            plan_record.get("thermal_parameter_names"),
            label="thermal parameter names",
        ),
        thermal_parameter_units=_string_tuple(
            plan_record.get("thermal_parameter_units"),
            label="thermal parameter units",
        ),
        feedback_parameter_names=_string_tuple(
            plan_record.get("feedback_parameter_names"),
            label="feedback parameter names",
        ),
        feedback_parameter_units=_string_tuple(
            plan_record.get("feedback_parameter_units"),
            label="feedback parameter units",
        ),
        iteration_policy=_iteration_policy(plan_record.get("iteration_policy")),
    )
    if plan.digest() != _sha256(plan_record.get("sha256"), label="plan SHA-256"):
        raise ValueError("reconstructed distributed electrothermal plan SHA-256 mismatch")
    if layout.digest() != _sha256(
        plan_record.get("layout_sha256"),
        label="plan layout SHA-256",
    ):
        raise ValueError("reconstructed distributed electrothermal layout SHA-256 mismatch")
    expected_counts = {
        "cell_count": topology.cell_count,
        "free_dof_count": topology.free_dof_count,
    }
    for name, observed in expected_counts.items():
        if _integer(plan_record.get(name), label=f"plan {name}", positive=True) != observed:
            raise ValueError(f"reconstructed plan {name} disagrees with its manifest")
    authority_record = _mapping(manifest.get("authority"), label="artifact authority")
    if authority_record.get("dtype") != "float64":
        raise ValueError("artifact authority must use float64")
    authority = DistributedElectrothermalAuthority(
        **{name: arrays[f"authority_{name}"] for name in _AUTHORITY_ARRAYS},
        objective=_number(authority_record.get("objective"), label="authority objective"),
        forward_converged=_boolean(
            authority_record.get("forward_converged"),
            label="authority forward convergence",
        ),
        adjoint_converged=_boolean(
            authority_record.get("adjoint_converged"),
            label="authority adjoint convergence",
        ),
        diagnostics=_mapping(
            authority_record.get("diagnostics"),
            label="authority diagnostics",
        ),
    )
    authority.validate(plan)
    arrays_record = _mapping(manifest.get("arrays"), label="artifact arrays")
    return LoadedDistributedElectrothermalArtifact(
        plan=plan,
        authority=authority,
        manifest=manifest,
        arrays_sha256=_sha256(arrays_record.get("sha256"), label="artifact arrays SHA-256"),
    )
