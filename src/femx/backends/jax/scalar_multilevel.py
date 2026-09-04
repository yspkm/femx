"""Explicit nested-mesh multilevel preconditioning for collective scalar H1/P1 systems.

The fine operator stays element-local and owner/ghost distributed.  A caller supplies an explicit
nested P1 hierarchy; femx validates every interpolation row on the host and constructs Galerkin
coarse operators from the current fine-cell matrices.  Only the bounded coarse hierarchy is
replicated.  The additive hierarchy is symmetric positive definite when its admitted diagonals
and Galerkin matrices are positive, so it may be used by collective PCG.

The hierarchy and its inverse actions are solver strategy.  They are stopped in reverse mode;
``custom_linear_solve`` continues to define derivatives from the converged fine residual.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import pairwise
from typing import NamedTuple, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from numpy.typing import DTypeLike

from femx.core.errors import ContractError

from .collective import validate_collective_mesh
from .scalar_cg import (
    PackedScalarH1Preconditioner,
    PackedScalarH1PreconditionerFactory,
    ScalarH1CGPolicy,
    ScalarH1CGResult,
    build_packed_collective_scalar_h1_cg,
)
from .scalar_collective import (
    ScalarH1CollectiveLayout,
    build_packed_collective_scalar_h1_rhs_assembly,
    pack_collective_scalar_h1_cell_matrix,
    pack_collective_scalar_h1_cell_vector,
    pack_collective_scalar_h1_owned_mask,
    unpack_collective_scalar_h1_owned_vector,
)

SCALAR_H1_NESTED_PROLONGATION_SCHEMA = "femx.jax.scalar_h1_nested_prolongation/v1"
SCALAR_H1_MULTILEVEL_HIERARCHY_SCHEMA = "femx.jax.scalar_h1_multilevel_hierarchy/v1"
_INTERPOLATION_WIDTH = 3


def _canonical_int_array(
    values: object,
    *,
    label: str,
    rank: int,
    columns: int | None = None,
) -> np.ndarray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{label} must be a regular integer array") from error
    if raw.dtype.kind not in "iu" or raw.ndim != rank:
        raise ContractError(f"{label} must be a rank-{rank} integer array")
    if columns is not None and raw.shape[-1] != columns:
        raise ContractError(f"{label} must have exactly {columns} columns")
    result = np.array(raw, dtype=np.int64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _canonical_float_array(
    values: object,
    *,
    label: str,
    rank: int,
    columns: int | None = None,
) -> np.ndarray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{label} must be a regular real array") from error
    if raw.dtype.kind != "f" or raw.ndim != rank:
        raise ContractError(f"{label} must be a rank-{rank} real floating array")
    if columns is not None and raw.shape[-1] != columns:
        raise ContractError(f"{label} must have exactly {columns} columns")
    result = np.array(raw, dtype=np.float64, order="C", copy=True)
    if not np.all(np.isfinite(result)):
        raise ContractError(f"{label} must contain only finite values")
    result.setflags(write=False)
    return result


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{label} must be a nonnegative integer")
    return value


def _canonical_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{label} must be a canonical SHA-256")
    return value


class _DigestWriter(Protocol):
    def update(self, data: bytes, /) -> object: ...


def _hash_array(
    hasher: _DigestWriter,
    label: str,
    array: np.ndarray,
    *,
    dtype: str,
) -> None:
    canonical = np.asarray(array, dtype=dtype, order="C")
    hasher.update(label.encode("utf-8"))
    hasher.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    hasher.update(canonical.tobytes())


def _source_digest(named_arrays: Sequence[tuple[str, np.ndarray]]) -> str:
    hasher = hashlib.sha256()
    for name, array in named_arrays:
        dtype = "<f8" if array.dtype.kind == "f" else "<i8"
        _hash_array(hasher, name, array, dtype=dtype)
    return hasher.hexdigest()


def _canonical_free_nodes(values: object, *, node_count: int, label: str) -> np.ndarray:
    nodes = _canonical_int_array(values, label=label, rank=1)
    if nodes.size == 0:
        raise ContractError(f"{label} must contain at least one node")
    if np.any(nodes < 0) or np.any(nodes >= node_count):
        raise ContractError(f"{label} contains an out-of-range node")
    if nodes.size > 1 and np.any(np.diff(nodes) <= 0):
        raise ContractError(f"{label} must be strictly increasing")
    return nodes


@dataclass(frozen=True, slots=True)
class ScalarH1NestedProlongation:
    """Canonical sparse P1 interpolation from one coarse free space to a fine space."""

    fine_free_dof_count: int
    coarse_free_dof_count: int
    column_indices: np.ndarray
    weights: np.ndarray
    fine_source_sha256: str
    coarse_space_sha256: str
    coarse_source_sha256: str
    ambiguity_count: int
    containment_tolerance: float
    schema_version: str = SCALAR_H1_NESTED_PROLONGATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SCALAR_H1_NESTED_PROLONGATION_SCHEMA:
            raise ContractError(
                "scalar H1 nested prolongation schema must be "
                f"{SCALAR_H1_NESTED_PROLONGATION_SCHEMA!r}"
            )
        fine_count = _positive_integer(
            self.fine_free_dof_count,
            label="nested prolongation fine free DOF count",
        )
        coarse_count = _positive_integer(
            self.coarse_free_dof_count,
            label="nested prolongation coarse free DOF count",
        )
        if coarse_count >= fine_count:
            raise ContractError("nested prolongation must strictly reduce the free DOF count")
        columns = _canonical_int_array(
            self.column_indices,
            label="nested prolongation column indices",
            rank=2,
            columns=_INTERPOLATION_WIDTH,
        )
        weights = _canonical_float_array(
            self.weights,
            label="nested prolongation weights",
            rank=2,
            columns=_INTERPOLATION_WIDTH,
        )
        expected_shape = (fine_count, _INTERPOLATION_WIDTH)
        if columns.shape != expected_shape or weights.shape != expected_shape:
            raise ContractError("nested prolongation rows must follow the fine free DOF count")
        if np.any(columns < 0) or np.any(columns > coarse_count):
            raise ContractError("nested prolongation contains an out-of-range coarse DOF")
        sentinel = columns == coarse_count
        if np.any(weights[sentinel] != 0.0):
            raise ContractError("nested prolongation sentinel entries must have exact-zero weight")
        if np.any((~sentinel) & (weights <= 0.0)):
            raise ContractError("nested prolongation active entries must have positive weight")
        tolerance = self.containment_tolerance
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            raise ContractError("nested prolongation containment tolerance must be a real scalar")
        tolerance = float(tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0 or tolerance >= 1.0:
            raise ContractError("nested prolongation containment tolerance must lie in (0, 1)")
        if np.any(weights < -tolerance) or np.any(weights > 1.0 + tolerance):
            raise ContractError("nested prolongation weights exceed barycentric bounds")
        active_counts = np.sum(~sentinel, axis=1)
        for row_columns, active_count in zip(columns, active_counts, strict=True):
            active = row_columns[: int(active_count)]
            if np.any(row_columns[int(active_count) :] != coarse_count):
                raise ContractError(
                    "nested prolongation sentinel columns must follow active columns"
                )
            if active.size > 1 and np.any(np.diff(active) <= 0):
                raise ContractError(
                    "nested prolongation active columns must be strictly increasing"
                )
        row_sums = np.sum(weights, axis=1)
        if np.any(row_sums < 0.0) or np.any(row_sums > 1.0 + tolerance):
            raise ContractError("nested prolongation row sums must lie in [0, 1]")
        ambiguity_count = _nonnegative_integer(
            self.ambiguity_count,
            label="nested prolongation ambiguity count",
        )
        object.__setattr__(
            self,
            "fine_source_sha256",
            _canonical_sha256(self.fine_source_sha256, label="fine source SHA-256"),
        )
        object.__setattr__(
            self,
            "coarse_space_sha256",
            _canonical_sha256(self.coarse_space_sha256, label="coarse space SHA-256"),
        )
        object.__setattr__(
            self,
            "coarse_source_sha256",
            _canonical_sha256(self.coarse_source_sha256, label="coarse source SHA-256"),
        )
        object.__setattr__(self, "ambiguity_count", ambiguity_count)
        object.__setattr__(self, "containment_tolerance", tolerance)
        object.__setattr__(self, "column_indices", columns)
        object.__setattr__(self, "weights", weights)

    def dense(self) -> np.ndarray:
        """Return the bounded dense prolongation for inspection and independent tests."""

        result = np.zeros(
            (self.fine_free_dof_count, self.coarse_free_dof_count),
            dtype=np.float64,
        )
        for row in range(self.fine_free_dof_count):
            for column, weight in zip(
                self.column_indices[row],
                self.weights[row],
                strict=True,
            ):
                if column < self.coarse_free_dof_count:
                    result[row, column] += weight
        result.setflags(write=False)
        return result

    def digest(self) -> str:
        metadata = {
            "schema_version": self.schema_version,
            "fine_free_dof_count": self.fine_free_dof_count,
            "coarse_free_dof_count": self.coarse_free_dof_count,
            "fine_source_sha256": self.fine_source_sha256,
            "coarse_space_sha256": self.coarse_space_sha256,
            "coarse_source_sha256": self.coarse_source_sha256,
            "ambiguity_count": self.ambiguity_count,
            "containment_tolerance": self.containment_tolerance,
        }
        hasher = hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        _hash_array(hasher, "column_indices", self.column_indices, dtype="<i8")
        _hash_array(hasher, "weights", self.weights, dtype="<f8")
        return hasher.hexdigest()


def prepare_scalar_h1_nested_prolongation(
    fine_coordinates: object,
    fine_free_nodes: object,
    coarse_coordinates: object,
    coarse_cells: object,
    coarse_free_nodes: object,
    *,
    containment_tolerance: float = 1.0e-12,
) -> ScalarH1NestedProlongation:
    """Evaluate coarse P1 basis functions at every fine free-node coordinate."""

    fine_points = _canonical_float_array(
        fine_coordinates,
        label="nested prolongation fine coordinates",
        rank=2,
        columns=2,
    )
    coarse_points = _canonical_float_array(
        coarse_coordinates,
        label="nested prolongation coarse coordinates",
        rank=2,
        columns=2,
    )
    cells = _canonical_int_array(
        coarse_cells,
        label="nested prolongation coarse cells",
        rank=2,
        columns=3,
    )
    if fine_points.shape[0] == 0 or coarse_points.shape[0] == 0 or cells.shape[0] == 0:
        raise ContractError("nested prolongation meshes must be nonempty")
    if np.any(cells < 0) or np.any(cells >= coarse_points.shape[0]):
        raise ContractError("nested prolongation coarse cells contain an out-of-range node")
    if np.any(np.diff(np.sort(cells, axis=1), axis=1) == 0):
        raise ContractError("nested prolongation coarse cells cannot repeat a node")
    fine_nodes = _canonical_free_nodes(
        fine_free_nodes,
        node_count=fine_points.shape[0],
        label="nested prolongation fine free nodes",
    )
    coarse_nodes = _canonical_free_nodes(
        coarse_free_nodes,
        node_count=coarse_points.shape[0],
        label="nested prolongation coarse free nodes",
    )
    if coarse_nodes.shape[0] >= fine_nodes.shape[0]:
        raise ContractError("nested prolongation must reduce the free-space dimension")
    if isinstance(containment_tolerance, bool) or not isinstance(
        containment_tolerance,
        (int, float),
    ):
        raise ContractError("nested prolongation containment tolerance must be a real scalar")
    tolerance = float(containment_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0 or tolerance >= 1.0:
        raise ContractError("nested prolongation containment tolerance must lie in (0, 1)")

    vertices = coarse_points[cells]
    first_edges = vertices[:, 1] - vertices[:, 0]
    second_edges = vertices[:, 2] - vertices[:, 0]
    determinants = first_edges[:, 0] * second_edges[:, 1] - first_edges[:, 1] * second_edges[:, 0]
    if np.any(np.abs(determinants) <= np.finfo(np.float64).eps):
        raise ContractError("nested prolongation coarse mesh contains a degenerate triangle")
    lower = np.min(vertices, axis=1) - tolerance
    upper = np.max(vertices, axis=1) + tolerance
    coarse_to_reduced = np.full(coarse_points.shape[0], coarse_nodes.shape[0], dtype=np.int64)
    coarse_to_reduced[coarse_nodes] = np.arange(coarse_nodes.shape[0], dtype=np.int64)

    columns = np.full(
        (fine_nodes.shape[0], _INTERPOLATION_WIDTH),
        coarse_nodes.shape[0],
        dtype=np.int64,
    )
    weights = np.zeros((fine_nodes.shape[0], _INTERPOLATION_WIDTH), dtype=np.float64)
    ambiguity_count = 0
    comparison_tolerance = 32.0 * tolerance

    for fine_row, fine_node in enumerate(fine_nodes):
        point = fine_points[fine_node]
        candidates = np.flatnonzero(np.all(point >= lower, axis=1) & np.all(point <= upper, axis=1))
        rows: list[dict[int, float]] = []
        for cell_index in candidates:
            origin = vertices[cell_index, 0]
            matrix = np.column_stack((first_edges[cell_index], second_edges[cell_index]))
            coordinates = np.linalg.solve(matrix, point - origin)
            barycentric = np.asarray(
                (1.0 - coordinates[0] - coordinates[1], coordinates[0], coordinates[1]),
                dtype=np.float64,
            )
            if np.min(barycentric) < -tolerance or np.max(barycentric) > 1.0 + tolerance:
                continue
            barycentric[np.abs(barycentric) <= tolerance] = 0.0
            barycentric[np.abs(barycentric - 1.0) <= tolerance] = 1.0
            row: dict[int, float] = {}
            for local_node, value in enumerate(barycentric):
                coarse_dof = int(coarse_to_reduced[cells[cell_index, local_node]])
                if coarse_dof < coarse_nodes.shape[0] and value != 0.0:
                    row[coarse_dof] = row.get(coarse_dof, 0.0) + float(value)
            rows.append(row)
        if not rows:
            raise ContractError("a fine free node is outside the coarse free interpolation space")
        reference = rows[0]
        if len(rows) > 1:
            ambiguity_count += 1
            keys = set(reference)
            for candidate in rows[1:]:
                keys.update(candidate)
                if any(
                    abs(reference.get(key, 0.0) - candidate.get(key, 0.0)) > comparison_tolerance
                    for key in keys
                ):
                    raise ContractError(
                        "shared-edge coarse triangles disagree on the continuous P1 interpolation"
                    )
        entries = sorted(reference.items())
        for slot, (column, value) in enumerate(entries):
            columns[fine_row, slot] = column
            weights[fine_row, slot] = value

    fine_digest = _source_digest((("coordinates", fine_points), ("free_nodes", fine_nodes)))
    coarse_space_digest = _source_digest(
        (("coordinates", coarse_points), ("free_nodes", coarse_nodes))
    )
    coarse_digest = _source_digest(
        (
            ("coordinates", coarse_points),
            ("cells", cells),
            ("free_nodes", coarse_nodes),
        )
    )
    return ScalarH1NestedProlongation(
        fine_free_dof_count=fine_nodes.shape[0],
        coarse_free_dof_count=coarse_nodes.shape[0],
        column_indices=columns,
        weights=weights,
        fine_source_sha256=fine_digest,
        coarse_space_sha256=coarse_space_digest,
        coarse_source_sha256=coarse_digest,
        ambiguity_count=ambiguity_count,
        containment_tolerance=tolerance,
    )


@dataclass(frozen=True, slots=True)
class ScalarH1MultilevelHierarchy:
    """Nested prolongations bound to one exact scalar collective layout."""

    layout_sha256: str
    prolongations: tuple[ScalarH1NestedProlongation, ...]
    maximum_replicated_dofs: int
    schema_version: str = SCALAR_H1_MULTILEVEL_HIERARCHY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SCALAR_H1_MULTILEVEL_HIERARCHY_SCHEMA:
            raise ContractError(
                "scalar H1 multilevel hierarchy schema must be "
                f"{SCALAR_H1_MULTILEVEL_HIERARCHY_SCHEMA!r}"
            )
        object.__setattr__(
            self,
            "layout_sha256",
            _canonical_sha256(self.layout_sha256, label="multilevel layout SHA-256"),
        )
        prolongations = tuple(self.prolongations)
        if not prolongations or any(
            not isinstance(level, ScalarH1NestedProlongation) for level in prolongations
        ):
            raise ContractError("multilevel hierarchy requires nested prolongation levels")
        for fine, coarse in pairwise(prolongations):
            if fine.coarse_free_dof_count != coarse.fine_free_dof_count:
                raise ContractError("multilevel prolongation dimensions must form one nested chain")
            if fine.coarse_space_sha256 != coarse.fine_source_sha256:
                raise ContractError(
                    "multilevel prolongations must share the exact intermediate space"
                )
        limit = _positive_integer(
            self.maximum_replicated_dofs,
            label="maximum replicated coarse DOFs",
        )
        if prolongations[0].coarse_free_dof_count > limit:
            raise ContractError("first replicated coarse level exceeds the declared DOF limit")
        object.__setattr__(self, "prolongations", prolongations)
        object.__setattr__(self, "maximum_replicated_dofs", limit)

    @property
    def level_dof_counts(self) -> tuple[int, ...]:
        return (
            self.prolongations[0].fine_free_dof_count,
            *(level.coarse_free_dof_count for level in self.prolongations),
        )

    def digest(self) -> str:
        metadata = {
            "schema_version": self.schema_version,
            "layout_sha256": self.layout_sha256,
            "maximum_replicated_dofs": self.maximum_replicated_dofs,
            "level_dof_counts": self.level_dof_counts,
            "prolongation_sha256": tuple(level.digest() for level in self.prolongations),
        }
        return hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def prepare_scalar_h1_multilevel_hierarchy(
    layout: ScalarH1CollectiveLayout,
    prolongations: Sequence[ScalarH1NestedProlongation],
    *,
    maximum_replicated_dofs: int,
) -> ScalarH1MultilevelHierarchy:
    """Bind a caller-supplied nested hierarchy to one exact collective layout."""

    if not isinstance(layout, ScalarH1CollectiveLayout):
        raise ContractError("scalar H1 multilevel preparation requires a collective layout")
    levels = tuple(prolongations)
    if not levels:
        raise ContractError("scalar H1 multilevel preparation requires at least two levels")
    if levels[0].fine_free_dof_count != layout.topology.free_dof_count:
        raise ContractError("first multilevel prolongation must follow the layout free DOFs")
    return ScalarH1MultilevelHierarchy(
        layout_sha256=layout.digest(),
        prolongations=levels,
        maximum_replicated_dofs=maximum_replicated_dofs,
    )


@dataclass(frozen=True, slots=True)
class ScalarH1MultilevelPolicy:
    """Admission policy for the stopped additive Galerkin hierarchy."""

    diagonal_weight: float = 1.0
    minimum_relative_diagonal: float = 1.0e-14
    maximum_relative_symmetry_error: float = 1.0e-10
    maximum_coarse_condition_number: float = 1.0e12

    def __post_init__(self) -> None:
        for name in (
            "diagonal_weight",
            "minimum_relative_diagonal",
            "maximum_relative_symmetry_error",
            "maximum_coarse_condition_number",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ContractError(f"multilevel {name.replace('_', ' ')} must be a real scalar")
            canonical = float(value)
            if not math.isfinite(canonical):
                raise ContractError(f"multilevel {name.replace('_', ' ')} must be finite")
            object.__setattr__(self, name, canonical)
        if self.diagonal_weight <= 0.0:
            raise ContractError("multilevel diagonal weight must be positive")
        if not 0.0 < self.minimum_relative_diagonal < 1.0:
            raise ContractError("multilevel minimum relative diagonal must lie in (0, 1)")
        if not 0.0 <= self.maximum_relative_symmetry_error < 1.0:
            raise ContractError("multilevel maximum relative symmetry error must lie in [0, 1)")
        if self.maximum_coarse_condition_number <= 1.0:
            raise ContractError("multilevel maximum coarse condition number must exceed one")


class PackedScalarH1MultilevelTransfer(NamedTuple):
    """Explicit fine-sharded and coarse-replicated sparse prolongation inputs."""

    owner_columns: jax.Array
    owner_weights: jax.Array
    cell_columns: jax.Array
    cell_weights: jax.Array
    coarse_columns: tuple[jax.Array, ...]
    coarse_weights: tuple[jax.Array, ...]


class HostPackedScalarH1MultilevelTransfer(NamedTuple):
    """Host-owned transfer arrays before callers choose distributed JAX sharding."""

    owner_columns: np.ndarray
    owner_weights: np.ndarray
    cell_columns: np.ndarray
    cell_weights: np.ndarray
    coarse_columns: tuple[np.ndarray, ...]
    coarse_weights: tuple[np.ndarray, ...]


class PackedScalarH1MultilevelState(NamedTuple):
    """Stopped inverse data and numerical setup diagnostics for one current operator."""

    fine_inverse_diagonal: jax.Array
    coarse_matrices: tuple[jax.Array, ...]
    coarse_inverse_diagonals: tuple[jax.Array, ...]
    coarsest_cholesky: jax.Array
    valid: jax.Array
    minimum_relative_diagonal: jax.Array
    maximum_relative_symmetry_error: jax.Array
    maximum_coarse_condition_number: jax.Array


def _packed_rows(
    ids: np.ndarray,
    level: ScalarH1NestedProlongation,
) -> tuple[np.ndarray, np.ndarray]:
    sentinel_columns = np.full(
        (1, _INTERPOLATION_WIDTH),
        level.coarse_free_dof_count,
        dtype=np.int64,
    )
    sentinel_weights = np.zeros((1, _INTERPOLATION_WIDTH), dtype=np.float64)
    columns = np.concatenate((level.column_indices, sentinel_columns), axis=0)[ids]
    weights = np.concatenate((level.weights, sentinel_weights), axis=0)[ids]
    return columns, weights


def pack_scalar_h1_multilevel_transfer_host(
    layout: ScalarH1CollectiveLayout,
    hierarchy: ScalarH1MultilevelHierarchy,
    *,
    value_dtype: DTypeLike,
) -> HostPackedScalarH1MultilevelTransfer:
    """Materialize canonical host arrays without choosing a device or sharding."""

    if not isinstance(layout, ScalarH1CollectiveLayout):
        raise ContractError("multilevel transfer packing requires a scalar collective layout")
    if not isinstance(hierarchy, ScalarH1MultilevelHierarchy):
        raise ContractError("multilevel transfer packing requires a hierarchy")
    if hierarchy.layout_sha256 != layout.digest():
        raise ContractError("multilevel hierarchy does not bind the supplied layout")
    try:
        dtype = np.dtype(value_dtype)
    except TypeError as error:
        raise ContractError("multilevel transfer value dtype must be explicit") from error
    if dtype.kind != "f" or dtype.itemsize not in (4, 8):
        raise ContractError("multilevel transfer values require float32 or float64")

    first = hierarchy.prolongations[0]
    owner_columns, owner_weights = _packed_rows(layout.transport.owned_dof_ids, first)
    canonical_cell_columns, canonical_cell_weights = _packed_rows(
        layout.topology.cell_reduced_dofs,
        first,
    )
    inactive_columns = np.full(
        (1, 3, _INTERPOLATION_WIDTH),
        first.coarse_free_dof_count,
        dtype=np.int64,
    )
    inactive_weights = np.zeros((1, 3, _INTERPOLATION_WIDTH), dtype=np.float64)
    cell_columns = np.concatenate((canonical_cell_columns, inactive_columns), axis=0)[
        layout.transport.cell_ids
    ]
    cell_weights = np.concatenate((canonical_cell_weights, inactive_weights), axis=0)[
        layout.transport.cell_ids
    ]

    def canonical(values: np.ndarray, requested_dtype: DTypeLike) -> np.ndarray:
        result = np.array(values, dtype=requested_dtype, order="C", copy=True)
        result.setflags(write=False)
        return result

    remaining = hierarchy.prolongations[1:]
    return HostPackedScalarH1MultilevelTransfer(
        owner_columns=canonical(owner_columns, np.int32),
        owner_weights=canonical(owner_weights, dtype),
        cell_columns=canonical(cell_columns, np.int32),
        cell_weights=canonical(cell_weights, dtype),
        coarse_columns=tuple(canonical(level.column_indices, np.int32) for level in remaining),
        coarse_weights=tuple(canonical(level.weights, dtype) for level in remaining),
    )


def pack_scalar_h1_multilevel_transfer(
    layout: ScalarH1CollectiveLayout,
    hierarchy: ScalarH1MultilevelHierarchy,
    *,
    value_dtype: DTypeLike,
) -> PackedScalarH1MultilevelTransfer:
    """Materialize transfer inputs as ordinary JAX arrays for local validation."""

    host = pack_scalar_h1_multilevel_transfer_host(
        layout,
        hierarchy,
        value_dtype=value_dtype,
    )
    return PackedScalarH1MultilevelTransfer(
        owner_columns=jnp.asarray(host.owner_columns),
        owner_weights=jnp.asarray(host.owner_weights),
        cell_columns=jnp.asarray(host.cell_columns),
        cell_weights=jnp.asarray(host.cell_weights),
        coarse_columns=tuple(jnp.asarray(values) for values in host.coarse_columns),
        coarse_weights=tuple(jnp.asarray(values) for values in host.coarse_weights),
    )


def _dense_sparse_prolongation(
    columns: jax.Array,
    weights: jax.Array,
    coarse_count: int,
) -> jax.Array:
    rows = columns.shape[0]
    extended = jnp.zeros((rows, coarse_count + 1), dtype=weights.dtype)
    row_ids = jnp.broadcast_to(jnp.arange(rows, dtype=columns.dtype)[:, None], columns.shape)
    return extended.at[row_ids.reshape(-1), columns.reshape(-1)].add(weights.reshape(-1))[
        :, :coarse_count
    ]


def _build_fine_coarse_matrix(
    layout: ScalarH1CollectiveLayout,
    mesh: Mesh,
    *,
    coarse_count: int,
    axis_name: str,
) -> Callable[[jax.Array, jax.Array, jax.Array], jax.Array]:
    validate_collective_mesh(layout.transport, mesh, axis_name)
    cell_spec = P(axis_name, None, None, None)  # type: ignore[no-untyped-call]
    interpolation_spec = P(axis_name, None, None, None)  # type: ignore[no-untyped-call]
    replicated = P()  # type: ignore[no-untyped-call]

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(cell_spec, interpolation_spec, interpolation_spec),
        out_specs=replicated,
        check_vma=True,
    )
    def mapped(
        cell_stiffness: jax.Array,
        cell_columns: jax.Array,
        cell_weights: jax.Array,
    ) -> jax.Array:
        local_columns = cell_columns[0]
        local_weights = cell_weights[0]
        extended = jnp.zeros(
            (layout.cell_capacity, 3, coarse_count + 1),
            dtype=cell_weights.dtype,
        )
        cell_ids = jnp.broadcast_to(
            jnp.arange(layout.cell_capacity, dtype=local_columns.dtype)[:, None, None],
            local_columns.shape,
        )
        local_nodes = jnp.broadcast_to(
            jnp.arange(3, dtype=local_columns.dtype)[None, :, None],
            local_columns.shape,
        )
        local_prolongation = extended.at[
            cell_ids.reshape(-1),
            local_nodes.reshape(-1),
            local_columns.reshape(-1),
        ].add(local_weights.reshape(-1))[:, :, :coarse_count]
        local_matrix = jnp.einsum(
            "cia,cij,cjb->ab",
            local_prolongation,
            cell_stiffness[0],
            local_prolongation,
        )
        return cast(jax.Array, lax.psum(local_matrix, axis_name))  # type: ignore[no-untyped-call]

    return cast(Callable[[jax.Array, jax.Array, jax.Array], jax.Array], mapped)


def _build_fine_restriction(
    layout: ScalarH1CollectiveLayout,
    mesh: Mesh,
    *,
    coarse_count: int,
    axis_name: str,
) -> Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
    validate_collective_mesh(layout.transport, mesh, axis_name)
    owner_spec = P(axis_name, None)  # type: ignore[no-untyped-call]
    interpolation_spec = P(axis_name, None, None)  # type: ignore[no-untyped-call]
    replicated = P()  # type: ignore[no-untyped-call]

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(owner_spec, owner_spec, interpolation_spec, interpolation_spec),
        out_specs=replicated,
        check_vma=True,
    )
    def mapped(
        fine_vector: jax.Array,
        owner_mask: jax.Array,
        owner_columns: jax.Array,
        owner_weights: jax.Array,
    ) -> jax.Array:
        local_values = jnp.where(owner_mask[0], fine_vector[0], 0.0)
        contributions = owner_weights[0] * local_values[:, None]
        restricted = (
            jnp.zeros((coarse_count + 1,), dtype=fine_vector.dtype)
            .at[owner_columns[0].reshape(-1)]
            .add(contributions.reshape(-1))[:coarse_count]
        )
        return cast(jax.Array, lax.psum(restricted, axis_name))  # type: ignore[no-untyped-call]

    return cast(
        Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array],
        mapped,
    )


def _coarse_restrict(
    columns: jax.Array,
    weights: jax.Array,
    coarse_count: int,
    vector: jax.Array,
) -> jax.Array:
    contributions = weights * vector[:, None]
    return (
        jnp.zeros((coarse_count + 1,), dtype=vector.dtype)
        .at[columns.reshape(-1)]
        .add(contributions.reshape(-1))[:coarse_count]
    )


def _coarse_prolong(
    columns: jax.Array,
    weights: jax.Array,
    coarse_count: int,
    vector: jax.Array,
) -> jax.Array:
    extended = jnp.concatenate((vector, jnp.zeros((1,), dtype=vector.dtype)))
    if extended.shape[0] != coarse_count + 1:
        raise ValueError("coarse prolongation vector disagrees with its level")
    return jnp.sum(weights * extended[columns], axis=1)


def _relative_symmetry_error(matrix: jax.Array) -> jax.Array:
    numerator = jnp.linalg.norm(matrix - matrix.T)
    denominator = jnp.linalg.norm(matrix)
    return jnp.where(
        denominator > 0.0,
        numerator / denominator,
        jnp.where(numerator == 0.0, 0.0, jnp.inf),
    )


def _matrix_diagnostics(
    matrix: jax.Array,
    policy: ScalarH1MultilevelPolicy,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    symmetry_error = _relative_symmetry_error(matrix)
    symmetric = 0.5 * (matrix + matrix.T)
    diagonal = jnp.diag(symmetric)
    diagonal_scale = jnp.max(jnp.abs(diagonal))
    minimum_relative_diagonal = jnp.where(
        diagonal_scale > 0.0,
        jnp.min(diagonal) / diagonal_scale,
        -jnp.inf,
    )
    eigenvalues = jnp.linalg.eigvalsh(symmetric)
    minimum_eigenvalue = jnp.min(eigenvalues)
    maximum_eigenvalue = jnp.max(eigenvalues)
    condition = jnp.where(
        minimum_eigenvalue > 0.0,
        maximum_eigenvalue / minimum_eigenvalue,
        jnp.inf,
    )
    valid = (
        jnp.all(jnp.isfinite(symmetric))
        & jnp.isfinite(symmetry_error)
        & (symmetry_error <= policy.maximum_relative_symmetry_error)
        & jnp.isfinite(minimum_relative_diagonal)
        & (minimum_relative_diagonal >= policy.minimum_relative_diagonal)
        & jnp.isfinite(condition)
        & (condition <= policy.maximum_coarse_condition_number)
    )
    return (
        symmetric,
        diagonal,
        minimum_relative_diagonal,
        symmetry_error,
        jnp.where(valid, condition, jnp.inf),
    )


def _validate_transfer_shapes(
    layout: ScalarH1CollectiveLayout,
    hierarchy: ScalarH1MultilevelHierarchy,
    transfer: PackedScalarH1MultilevelTransfer,
    *,
    value_dtype: jnp.dtype,
) -> None:
    first = hierarchy.prolongations[0]
    owner_shape = (layout.partition_count, layout.owned_dof_capacity, _INTERPOLATION_WIDTH)
    cell_shape = (layout.partition_count, layout.cell_capacity, 3, _INTERPOLATION_WIDTH)
    if transfer.owner_columns.shape != owner_shape or transfer.owner_weights.shape != owner_shape:
        raise ValueError("multilevel owner interpolation does not match the layout")
    if transfer.cell_columns.shape != cell_shape or transfer.cell_weights.shape != cell_shape:
        raise ValueError("multilevel cell interpolation does not match the layout")
    for columns, label in (
        (transfer.owner_columns, "multilevel owner columns"),
        (transfer.cell_columns, "multilevel cell columns"),
    ):
        if not jnp.issubdtype(columns.dtype, jnp.integer):
            raise TypeError(f"{label} must use an integer dtype")
    for weights, label in (
        (transfer.owner_weights, "multilevel owner weights"),
        (transfer.cell_weights, "multilevel cell weights"),
    ):
        if weights.dtype != value_dtype:
            raise TypeError(f"{label} must match the scalar operator dtype")
    expected_remaining = len(hierarchy.prolongations) - 1
    if (
        len(transfer.coarse_columns) != expected_remaining
        or len(transfer.coarse_weights) != expected_remaining
    ):
        raise ValueError("multilevel coarse interpolation count disagrees with the hierarchy")
    for level, columns, weights in zip(
        hierarchy.prolongations[1:],
        transfer.coarse_columns,
        transfer.coarse_weights,
        strict=True,
    ):
        expected = (level.fine_free_dof_count, _INTERPOLATION_WIDTH)
        if columns.shape != expected or weights.shape != expected:
            raise ValueError("multilevel coarse interpolation shape disagrees with its level")
        if not jnp.issubdtype(columns.dtype, jnp.integer):
            raise TypeError("multilevel coarse columns must use an integer dtype")
        if weights.dtype != value_dtype:
            raise TypeError("multilevel coarse weights must match the scalar operator dtype")
    if first.fine_free_dof_count != layout.topology.free_dof_count:
        raise ContractError("multilevel fine interpolation identity drifted from the layout")


@dataclass(frozen=True, slots=True)
class PackedScalarH1MultilevelRuntime:
    """JAX setup/apply/factory functions bound to one explicit hierarchy and Mesh."""

    setup: Callable[..., PackedScalarH1MultilevelState]
    apply: Callable[..., jax.Array]
    factory: PackedScalarH1PreconditionerFactory


def build_packed_scalar_h1_multilevel_runtime(
    layout: ScalarH1CollectiveLayout,
    mesh: Mesh,
    hierarchy: ScalarH1MultilevelHierarchy,
    policy: ScalarH1MultilevelPolicy,
    transfer: PackedScalarH1MultilevelTransfer | None = None,
    *,
    axis_name: str = "partition",
) -> PackedScalarH1MultilevelRuntime:
    """Build stopped additive multilevel setup and inverse actions on an explicit Mesh."""

    if not isinstance(layout, ScalarH1CollectiveLayout):
        raise ContractError("multilevel runtime requires a scalar H1 collective layout")
    if not isinstance(hierarchy, ScalarH1MultilevelHierarchy):
        raise ContractError("multilevel runtime requires a scalar H1 hierarchy")
    if hierarchy.layout_sha256 != layout.digest():
        raise ContractError("multilevel runtime hierarchy does not bind the layout")
    if not isinstance(policy, ScalarH1MultilevelPolicy):
        raise ContractError("multilevel runtime requires a ScalarH1MultilevelPolicy")
    if transfer is not None and not isinstance(transfer, PackedScalarH1MultilevelTransfer):
        raise ContractError("multilevel runtime requires explicit packed transfer arrays")
    validate_collective_mesh(layout.transport, mesh, axis_name)

    def resolve_transfer(
        transfer_arguments: tuple[object, ...],
    ) -> PackedScalarH1MultilevelTransfer:
        if transfer is None:
            if len(transfer_arguments) != 1 or not isinstance(
                transfer_arguments[0],
                PackedScalarH1MultilevelTransfer,
            ):
                raise ContractError(
                    "unbound multilevel runtime requires one explicit packed transfer"
                )
            return transfer_arguments[0]
        if transfer_arguments:
            raise ContractError("bound multilevel runtime cannot receive another packed transfer")
        return transfer

    first_coarse_count = hierarchy.prolongations[0].coarse_free_dof_count
    row_assembly = build_packed_collective_scalar_h1_rhs_assembly(
        layout,
        mesh,
        axis_name=axis_name,
    )
    coarse_matrix = _build_fine_coarse_matrix(
        layout,
        mesh,
        coarse_count=first_coarse_count,
        axis_name=axis_name,
    )
    fine_restriction = _build_fine_restriction(
        layout,
        mesh,
        coarse_count=first_coarse_count,
        axis_name=axis_name,
    )

    def setup(
        packed_cell_stiffness: jax.Array,
        packed_cell_local_dofs: jax.Array,
        packed_owner_mask: jax.Array,
        *transfer_arguments: object,
    ) -> PackedScalarH1MultilevelState:
        current_transfer = resolve_transfer(transfer_arguments)
        expected_stiffness = (layout.partition_count, layout.cell_capacity, 3, 3)
        expected_mapping = (layout.partition_count, layout.cell_capacity, 3)
        expected_owner = (layout.partition_count, layout.owned_dof_capacity)
        if packed_cell_stiffness.shape != expected_stiffness:
            raise ValueError("multilevel packed cell stiffness does not match the layout")
        if packed_cell_local_dofs.shape != expected_mapping:
            raise ValueError("multilevel packed cell map does not match the layout")
        if packed_owner_mask.shape != expected_owner:
            raise ValueError("multilevel packed owner mask does not match the layout")
        if not jnp.issubdtype(packed_cell_stiffness.dtype, jnp.floating):
            raise TypeError("multilevel packed cell stiffness must use a real floating dtype")
        if not jnp.issubdtype(packed_cell_local_dofs.dtype, jnp.integer):
            raise TypeError("multilevel packed cell map must use an integer dtype")
        if packed_owner_mask.dtype != jnp.bool_:
            raise TypeError("multilevel packed owner mask must use a boolean dtype")
        _validate_transfer_shapes(
            layout,
            hierarchy,
            current_transfer,
            value_dtype=packed_cell_stiffness.dtype,
        )

        packed_diagonal = jnp.diagonal(packed_cell_stiffness, axis1=-2, axis2=-1)
        fine_diagonal = row_assembly(packed_diagonal, packed_cell_local_dofs)
        active_diagonal = jnp.where(packed_owner_mask, fine_diagonal, jnp.inf)
        fine_scale = jnp.max(jnp.where(packed_owner_mask, jnp.abs(fine_diagonal), 0.0))
        fine_minimum_relative = jnp.where(
            fine_scale > 0.0,
            jnp.min(active_diagonal) / fine_scale,
            -jnp.inf,
        )
        fine_valid = (
            jnp.all(jnp.isfinite(jnp.where(packed_owner_mask, fine_diagonal, 0.0)))
            & jnp.isfinite(fine_minimum_relative)
            & (fine_minimum_relative >= policy.minimum_relative_diagonal)
        )
        fine_inverse = jnp.where(packed_owner_mask, 1.0 / fine_diagonal, 0.0)

        current = coarse_matrix(
            packed_cell_stiffness,
            current_transfer.cell_columns,
            current_transfer.cell_weights,
        )
        matrices: list[jax.Array] = []
        inverse_diagonals: list[jax.Array] = []
        minimum_relative = fine_minimum_relative
        maximum_symmetry = jnp.asarray(0.0, dtype=packed_cell_stiffness.dtype)
        maximum_condition = jnp.asarray(1.0, dtype=packed_cell_stiffness.dtype)
        valid = fine_valid

        for level_index, _level in enumerate(hierarchy.prolongations):
            (
                symmetric,
                diagonal,
                relative_diagonal,
                symmetry_error,
                condition,
            ) = _matrix_diagnostics(current, policy)
            matrices.append(symmetric)
            minimum_relative = jnp.minimum(minimum_relative, relative_diagonal)
            maximum_symmetry = jnp.maximum(maximum_symmetry, symmetry_error)
            maximum_condition = jnp.maximum(maximum_condition, condition)
            level_valid = (
                jnp.isfinite(condition)
                & (relative_diagonal >= policy.minimum_relative_diagonal)
                & (symmetry_error <= policy.maximum_relative_symmetry_error)
                & (condition <= policy.maximum_coarse_condition_number)
            )
            valid = valid & level_valid
            if level_index + 1 == len(hierarchy.prolongations):
                break
            inverse_diagonals.append(1.0 / diagonal)
            columns = current_transfer.coarse_columns[level_index]
            weights = current_transfer.coarse_weights[level_index]
            next_count = hierarchy.prolongations[level_index + 1].coarse_free_dof_count
            prolongation = _dense_sparse_prolongation(columns, weights, next_count)
            current = prolongation.T @ symmetric @ prolongation

        cholesky = jnp.linalg.cholesky(matrices[-1])
        valid = valid & jnp.all(jnp.isfinite(cholesky))
        stopped = PackedScalarH1MultilevelState(
            fine_inverse_diagonal=fine_inverse,
            coarse_matrices=tuple(matrices),
            coarse_inverse_diagonals=tuple(inverse_diagonals),
            coarsest_cholesky=cholesky,
            valid=valid,
            minimum_relative_diagonal=minimum_relative,
            maximum_relative_symmetry_error=maximum_symmetry,
            maximum_coarse_condition_number=maximum_condition,
        )
        return cast(
            PackedScalarH1MultilevelState,
            jax.tree.map(lax.stop_gradient, stopped),
        )

    def apply(
        state: PackedScalarH1MultilevelState,
        residual: jax.Array,
        *transfer_arguments: object,
    ) -> jax.Array:
        current_transfer = resolve_transfer(transfer_arguments)
        expected = (layout.partition_count, layout.owned_dof_capacity)
        if residual.shape != expected:
            raise ValueError("multilevel residual does not match the owner layout")
        if not jnp.issubdtype(residual.dtype, jnp.floating):
            raise TypeError("multilevel residual must use a real floating dtype")
        if residual.dtype != state.fine_inverse_diagonal.dtype:
            raise TypeError("multilevel residual must match the prepared operator dtype")

        def coarse_inverse(level_index: int, level_residual: jax.Array) -> jax.Array:
            if level_index + 1 == len(state.coarse_matrices):
                factor = state.coarsest_cholesky
                first = lax.linalg.triangular_solve(
                    factor,
                    level_residual[:, None],
                    left_side=True,
                    lower=True,
                )
                return lax.linalg.triangular_solve(
                    factor,
                    first,
                    left_side=True,
                    lower=True,
                    transpose_a=True,
                )[:, 0]
            diagonal_part = (
                policy.diagonal_weight
                * state.coarse_inverse_diagonals[level_index]
                * level_residual
            )
            level = hierarchy.prolongations[level_index + 1]
            columns = current_transfer.coarse_columns[level_index]
            weights = current_transfer.coarse_weights[level_index]
            restricted = _coarse_restrict(
                columns,
                weights,
                level.coarse_free_dof_count,
                level_residual,
            )
            corrected = coarse_inverse(level_index + 1, restricted)
            return diagonal_part + _coarse_prolong(
                columns,
                weights,
                level.coarse_free_dof_count,
                corrected,
            )

        fine_diagonal_part = policy.diagonal_weight * state.fine_inverse_diagonal * residual
        restricted = fine_restriction(
            residual,
            pack_collective_scalar_h1_owned_mask(layout),
            current_transfer.owner_columns,
            current_transfer.owner_weights,
        )
        coarse_correction = coarse_inverse(0, restricted)
        extended = jnp.concatenate(
            (coarse_correction, jnp.zeros((1,), dtype=coarse_correction.dtype))
        )
        fine_correction = jnp.sum(
            current_transfer.owner_weights * extended[current_transfer.owner_columns],
            axis=2,
        )
        candidate = jnp.where(
            pack_collective_scalar_h1_owned_mask(layout),
            fine_diagonal_part + fine_correction,
            0.0,
        )
        finite = jnp.all(jnp.isfinite(candidate))
        return jnp.where(state.valid & finite, candidate, jnp.asarray(jnp.nan, candidate.dtype))

    def factory(
        packed_cell_stiffness: jax.Array,
        packed_cell_local_dofs: jax.Array,
        packed_owner_mask: jax.Array,
        *transfer_arguments: object,
    ) -> PackedScalarH1Preconditioner:
        state = setup(
            packed_cell_stiffness,
            packed_cell_local_dofs,
            packed_owner_mask,
            *transfer_arguments,
        )

        def precondition(residual: jax.Array) -> jax.Array:
            return apply(state, residual, *transfer_arguments)

        return precondition

    return PackedScalarH1MultilevelRuntime(setup=setup, apply=apply, factory=factory)


def build_validation_collective_scalar_h1_multilevel_pcg(
    layout: ScalarH1CollectiveLayout,
    mesh: Mesh,
    hierarchy: ScalarH1MultilevelHierarchy,
    multilevel_policy: ScalarH1MultilevelPolicy,
    cg_policy: ScalarH1CGPolicy,
    *,
    value_dtype: DTypeLike,
    axis_name: str = "partition",
) -> Callable[[jax.Array, jax.Array], ScalarH1CGResult]:
    """Build a canonical small-problem multilevel PCG wrapper for validation."""

    transfer = pack_scalar_h1_multilevel_transfer(
        layout,
        hierarchy,
        value_dtype=value_dtype,
    )
    runtime = build_packed_scalar_h1_multilevel_runtime(
        layout,
        mesh,
        hierarchy,
        multilevel_policy,
        transfer,
        axis_name=axis_name,
    )
    solve = build_packed_collective_scalar_h1_cg(
        layout,
        mesh,
        cg_policy,
        axis_name=axis_name,
        preconditioner_factory=runtime.factory,
    )
    rhs_assembly = build_packed_collective_scalar_h1_rhs_assembly(
        layout,
        mesh,
        axis_name=axis_name,
    )
    mapping = jnp.asarray(layout.transport.cell_local_dofs)
    owner_mask = pack_collective_scalar_h1_owned_mask(layout)

    def apply(cell_stiffness: jax.Array, cell_rhs: jax.Array) -> ScalarH1CGResult:
        packed_stiffness = pack_collective_scalar_h1_cell_matrix(layout, cell_stiffness)
        packed_rhs = rhs_assembly(
            pack_collective_scalar_h1_cell_vector(layout, cell_rhs),
            mapping,
        )
        result = solve(packed_stiffness, mapping, owner_mask, packed_rhs)
        return ScalarH1CGResult(
            solution=unpack_collective_scalar_h1_owned_vector(layout, result.solution),
            right_hand_side=unpack_collective_scalar_h1_owned_vector(
                layout,
                result.right_hand_side,
            ),
            iterations=result.iterations,
            rhs_norm=result.rhs_norm,
            recursive_residual_norm=result.recursive_residual_norm,
            recomputed_residual_norm=result.recomputed_residual_norm,
            relative_residual=result.relative_residual,
            backward_error=result.backward_error,
            converged=result.converged,
            breakdown=result.breakdown,
        )

    return apply
