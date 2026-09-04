r"""Matrix-free shifted solves for the Elmer-compatible mixed port pencil.

The locked Elmer eigensolver applies ``inv(A - sigma B) B x`` without inverting the singular
generalized mass.  This module implements that same full mixed-system action from cell-local
6-by-6 blocks.  It does not form a global square matrix or a nested Schur-complement solve.

The mixed scalar and edge equations have very different physical scales.  The iterative solve
therefore uses a stopped-gradient, two-sided row/column equilibration.  Repeated shift-invert
applications may additionally reuse a physics-aware lower block-triangular diagonal
preconditioner: it approximately resolves the scalar constraint before correcting the edge block.
A separate outer ``jax.lax.custom_linear_solve`` gives the reverse pass its own transpose GMRES
tolerance and transposed preconditioner; JAX's public GMRES otherwise reuses primal
right-hand-side tolerances for its internal transpose solve.  Every returned result is admitted by
an independently recomputed equilibrated residual.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import gmres

from femx.core.errors import ContractError

from .port_operator import (
    TrianglePortPencil,
    lossless_port_coefficients,
    triangle_port_local_pencil,
)


def _readonly_int64(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=np.int64)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class PortMatrixFreeTopology:
    """Host-validated reduction from cell-local mixed DOFs to free PEC DOFs.

    ``cell_reduced_dofs`` uses ``free_dof_count`` as one constrained sentinel.  Matrix-free
    gather/scatter kernels append one exact zero entry for that sentinel and discard its residual.
    This avoids a full-size scatter during every operator application.
    """

    cell_reduced_dofs: np.ndarray
    free_dofs: np.ndarray
    full_dof_count: int

    @property
    def free_dof_count(self) -> int:
        """Return the number of unconstrained mixed DOFs."""

        return int(self.free_dofs.shape[0])

    @property
    def constrained_sentinel(self) -> int:
        """Return the local-map value reserved for constrained coefficients."""

        return self.free_dof_count


@dataclass(frozen=True, slots=True)
class MatrixFreePortSolvePolicy:
    """Static GMRES and independent residual-admission policy."""

    relative_tolerance: float = 1.0e-11
    absolute_tolerance: float = 0.0
    restart: int = 80
    maximum_restart_cycles: int = 100
    solve_method: str = "incremental"
    maximum_relative_residual: float = 5.0e-10

    def __post_init__(self) -> None:
        if not math.isfinite(self.relative_tolerance) or self.relative_tolerance <= 0.0:
            raise ValueError("matrix-free relative tolerance must be finite and positive")
        if not math.isfinite(self.absolute_tolerance) or self.absolute_tolerance < 0.0:
            raise ValueError("matrix-free absolute tolerance must be finite and nonnegative")
        if isinstance(self.restart, bool) or not isinstance(self.restart, int) or self.restart <= 0:
            raise ValueError("matrix-free restart must be a positive integer")
        if (
            isinstance(self.maximum_restart_cycles, bool)
            or not isinstance(self.maximum_restart_cycles, int)
            or self.maximum_restart_cycles <= 0
        ):
            raise ValueError("matrix-free maximum restart cycles must be a positive integer")
        if self.solve_method not in {"incremental", "batched"}:
            raise ValueError("matrix-free solve method must be 'incremental' or 'batched'")
        if (
            not math.isfinite(self.maximum_relative_residual)
            or self.maximum_relative_residual <= 0.0
        ):
            raise ValueError("matrix-free maximum residual must be finite and positive")
        if self.maximum_relative_residual < self.relative_tolerance:
            raise ValueError(
                "matrix-free maximum residual must be no smaller than the GMRES tolerance"
            )


DEFAULT_MATRIX_FREE_PORT_SOLVE_POLICY: Final = MatrixFreePortSolvePolicy()


@dataclass(frozen=True, slots=True)
class MatrixFreePortBlockPreconditionerPolicy:
    """Admission policy for the stopped block-triangular diagonal preconditioner."""

    minimum_relative_diagonal: float = 1.0e-14

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.minimum_relative_diagonal)
            or self.minimum_relative_diagonal <= 0.0
            or self.minimum_relative_diagonal >= 1.0
        ):
            raise ValueError(
                "matrix-free preconditioner minimum relative diagonal must lie in (0, 1)"
            )


DEFAULT_MATRIX_FREE_PORT_BLOCK_PRECONDITIONER_POLICY: Final = (
    MatrixFreePortBlockPreconditionerPolicy()
)


class MatrixFreePortPencil(NamedTuple):
    """Cell-local generalized pencil and its reduced gather/scatter map."""

    stiffness: jax.Array
    mass: jax.Array
    cell_reduced_dofs: jax.Array
    free_dof_count: int


class PortShiftEquilibration(NamedTuple):
    """Stopped-gradient two-sided scaling for one shifted mixed system."""

    left_scale: jax.Array
    right_scale: jax.Array
    row_absolute_sums: jax.Array
    column_absolute_sums: jax.Array
    global_normalization: jax.Array
    is_valid: jax.Array


class PreparedMatrixFreePortShift(NamedTuple):
    """Reusable cell-local representation of one equilibrated shifted system."""

    shifted_stiffness: jax.Array
    mass: jax.Array
    cell_reduced_dofs: jax.Array
    shift_per_m2: jax.Array
    equilibration: PortShiftEquilibration
    free_dof_count: int


class MatrixFreePortBlockPreconditioner(NamedTuple):
    """Stopped lower block-triangular approximation of the equilibrated inverse."""

    inverse_scaled_diagonal: jax.Array
    scalar_mask: jax.Array
    edge_mask: jax.Array
    minimum_relative_diagonal: jax.Array
    is_valid: jax.Array


class MatrixFreePortSolveDiagnostics(NamedTuple):
    """Independent numerical-admission evidence for one shifted solve."""

    equilibrated_relative_residual: jax.Array
    equilibrated_rhs_norm: jax.Array
    minimum_row_absolute_sum: jax.Array
    maximum_row_absolute_sum: jax.Array
    minimum_column_absolute_sum: jax.Array
    maximum_column_absolute_sum: jax.Array
    is_valid: jax.Array


class MatrixFreePortSolve(NamedTuple):
    """Admitted physical and equilibrated solutions of one shifted system."""

    solution: jax.Array
    equilibrated_solution: jax.Array
    equilibration: PortShiftEquilibration
    diagnostics: MatrixFreePortSolveDiagnostics


@dataclass(frozen=True, slots=True)
class PortOperatorStorageEstimate:
    """Analytical storage counts, not measured device-memory evidence."""

    matrix_free_value_bytes: int
    matrix_free_index_bytes: int
    matrix_free_total_bytes: int
    dense_pair_bytes: int
    dense_to_matrix_free_ratio: float


def prepare_port_matrix_free_topology(
    cells: object,
    cell_edge_dofs: object,
    free_dofs: object,
    *,
    node_count: int,
    edge_dof_count: int,
) -> PortMatrixFreeTopology:
    """Validate and map canonical full mixed DOFs to reduced matrix-free indices."""

    for label, value in (("node_count", node_count), ("edge_dof_count", edge_dof_count)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ContractError(f"matrix-free {label} must be a positive integer")

    raw_cells = np.asarray(cells)
    raw_edge_dofs = np.asarray(cell_edge_dofs)
    raw_free_dofs = np.asarray(free_dofs)
    if raw_cells.dtype.kind not in "iu" or raw_cells.ndim != 2 or raw_cells.shape[1] != 3:
        raise ContractError("matrix-free triangle cells must be an integer array shaped (cells, 3)")
    if raw_cells.shape[0] == 0:
        raise ContractError("matrix-free topology requires at least one triangle")
    if raw_edge_dofs.dtype.kind not in "iu" or raw_edge_dofs.shape != raw_cells.shape:
        raise ContractError(
            "matrix-free triangle edge DOFs must be an integer array shaped (cells, 3)"
        )
    if np.any(raw_cells < 0) or np.any(raw_cells >= node_count):
        raise ContractError("matrix-free triangle cells contain an out-of-range node")
    if np.any(
        (raw_cells[:, 0] == raw_cells[:, 1])
        | (raw_cells[:, 1] == raw_cells[:, 2])
        | (raw_cells[:, 2] == raw_cells[:, 0])
    ):
        raise ContractError("matrix-free triangle cells contain a repeated node")
    if np.any(raw_edge_dofs < 0) or np.any(raw_edge_dofs >= edge_dof_count):
        raise ContractError("matrix-free triangle cells contain an out-of-range edge DOF")
    if np.any(
        (raw_edge_dofs[:, 0] == raw_edge_dofs[:, 1])
        | (raw_edge_dofs[:, 1] == raw_edge_dofs[:, 2])
        | (raw_edge_dofs[:, 2] == raw_edge_dofs[:, 0])
    ):
        raise ContractError("matrix-free triangle cells contain a repeated edge DOF")
    if raw_free_dofs.dtype.kind not in "iu" or raw_free_dofs.ndim != 1:
        raise ContractError("matrix-free free DOFs must be a rank-one integer array")
    if raw_free_dofs.size == 0:
        raise ContractError("matrix-free topology requires at least one free DOF")

    full_dof_count = node_count + edge_dof_count
    free = np.asarray(raw_free_dofs, dtype=np.int64)
    if np.any(free < 0) or np.any(free >= full_dof_count):
        raise ContractError("matrix-free free DOFs contain an out-of-range index")
    if np.any(np.diff(free) <= 0):
        raise ContractError("matrix-free free DOFs must be unique and strictly increasing")

    local_full_dofs = np.concatenate(
        (
            np.asarray(raw_cells, dtype=np.int64),
            node_count + np.asarray(raw_edge_dofs, dtype=np.int64),
        ),
        axis=1,
    )
    missing = np.setdiff1d(free, np.unique(local_full_dofs), assume_unique=True)
    if missing.size:
        raise ContractError("matrix-free free DOFs include an index absent from all cells")

    sentinel = int(free.shape[0])
    full_to_reduced = np.full(full_dof_count, sentinel, dtype=np.int64)
    full_to_reduced[free] = np.arange(sentinel, dtype=np.int64)
    return PortMatrixFreeTopology(
        cell_reduced_dofs=_readonly_int64(full_to_reduced[local_full_dofs]),
        free_dofs=_readonly_int64(free),
        full_dof_count=full_dof_count,
    )


def build_lossless_matrix_free_port_pencil(
    coordinates: jax.Array,
    cells: jax.Array,
    cell_edge_signs: jax.Array,
    cell_reduced_dofs: jax.Array,
    cell_relative_permittivity: jax.Array,
    cell_relative_permeability: jax.Array,
    frequency_hz: jax.Array,
    *,
    free_dof_count: int,
) -> MatrixFreePortPencil:
    """Build only cell-local Elmer-compatible blocks for a validated reduced topology."""

    if cell_reduced_dofs.shape != (cells.shape[0], 6):
        raise ValueError("matrix-free cell reduced DOFs must be shaped (cells, 6)")
    if (
        isinstance(free_dof_count, bool)
        or not isinstance(free_dof_count, int)
        or free_dof_count <= 0
    ):
        raise ValueError("matrix-free free DOF count must be a positive static integer")
    permittivity, reluctivity = lossless_port_coefficients(
        cell_relative_permittivity,
        cell_relative_permeability,
    )
    local: TrianglePortPencil = triangle_port_local_pencil(
        coordinates,
        cells,
        cell_edge_signs,
        permittivity,
        reluctivity,
        2.0 * jnp.pi * frequency_hz,
    )
    return MatrixFreePortPencil(
        stiffness=local.stiffness,
        mass=local.mass,
        cell_reduced_dofs=cell_reduced_dofs,
        free_dof_count=free_dof_count,
    )


def _safe_reduced_map(
    cell_reduced_dofs: jax.Array,
    *,
    free_dof_count: int,
) -> tuple[jax.Array, jax.Array]:
    mapping_valid = jnp.all((cell_reduced_dofs >= 0) & (cell_reduced_dofs <= free_dof_count))
    safe_mapping = jnp.clip(cell_reduced_dofs, 0, free_dof_count)
    return safe_mapping, mapping_valid


def matrix_free_port_matvec(
    cell_matrix: jax.Array,
    cell_reduced_dofs: jax.Array,
    vector: jax.Array,
) -> jax.Array:
    """Apply one free-DOF principal operator from element-local matrices."""

    if cell_matrix.ndim != 3 or cell_matrix.shape[1:] != (6, 6):
        raise ValueError("matrix-free port cell matrix must be shaped (cells, 6, 6)")
    if cell_reduced_dofs.shape != (*cell_matrix.shape[:1], 6):
        raise ValueError("matrix-free port cell map must be shaped (cells, 6)")
    if vector.ndim != 1 or vector.shape[0] == 0:
        raise ValueError("matrix-free port vector must be a nonempty rank-one array")
    if not jnp.issubdtype(cell_reduced_dofs.dtype, jnp.integer):
        raise TypeError("matrix-free port cell map must use an integer dtype")

    free_dof_count = vector.shape[0]
    safe_mapping, mapping_valid = _safe_reduced_map(
        cell_reduced_dofs,
        free_dof_count=free_dof_count,
    )
    dtype = jnp.result_type(cell_matrix.dtype, vector.dtype)
    extended_vector = jnp.concatenate((vector.astype(dtype), jnp.zeros((1,), dtype=dtype)))
    local_input = extended_vector[safe_mapping]
    local_output = jnp.einsum("cij,cj->ci", cell_matrix.astype(dtype), local_input)
    assembled = (
        jnp.zeros((free_dof_count + 1,), dtype=dtype)
        .at[safe_mapping.reshape(-1)]
        .add(local_output.reshape(-1))
    )
    return jnp.where(
        mapping_valid,
        assembled[:free_dof_count],
        jnp.asarray(jnp.nan, dtype=dtype),
    )


def _equilibrate_shifted_port_matrix(
    shifted_cell_matrix: jax.Array,
    cell_reduced_dofs: jax.Array,
    *,
    free_dof_count: int,
) -> PortShiftEquilibration:
    sentinel = free_dof_count
    safe_mapping, mapping_valid = _safe_reduced_map(
        cell_reduced_dofs,
        free_dof_count=free_dof_count,
    )
    active = safe_mapping < sentinel
    active_entries = active[:, :, None] & active[:, None, :]
    absolute_values = jnp.where(active_entries, jnp.abs(shifted_cell_matrix), 0.0)
    row_local = jnp.sum(absolute_values, axis=2)
    column_local = jnp.sum(absolute_values, axis=1)
    row_absolute_sums = (
        jnp.zeros((sentinel + 1,), dtype=row_local.dtype)
        .at[safe_mapping.reshape(-1)]
        .add(row_local.reshape(-1))[:sentinel]
    )
    column_absolute_sums = (
        jnp.zeros((sentinel + 1,), dtype=column_local.dtype)
        .at[safe_mapping.reshape(-1)]
        .add(column_local.reshape(-1))[:sentinel]
    )

    row_valid = jnp.all(jnp.isfinite(row_absolute_sums) & (row_absolute_sums > 0.0))
    column_valid = jnp.all(jnp.isfinite(column_absolute_sums) & (column_absolute_sums > 0.0))
    safe_rows = jnp.where(row_valid, row_absolute_sums, jnp.ones_like(row_absolute_sums))
    safe_columns = jnp.where(
        column_valid,
        column_absolute_sums,
        jnp.ones_like(column_absolute_sums),
    )
    left_scale = 1.0 / safe_rows
    right_scale = 1.0 / safe_columns

    extended_left = jnp.concatenate((left_scale, jnp.zeros((1,), dtype=left_scale.dtype)))
    extended_right = jnp.concatenate((right_scale, jnp.zeros((1,), dtype=right_scale.dtype)))
    scaled_local = (
        extended_left[safe_mapping][:, :, None]
        * shifted_cell_matrix
        * extended_right[safe_mapping][:, None, :]
    )
    largest_scaled_entry = jnp.max(jnp.where(active_entries, jnp.abs(scaled_local), 0.0))
    normalization_valid = jnp.isfinite(largest_scaled_entry) & (largest_scaled_entry > 0.0)
    safe_largest = jnp.where(normalization_valid, largest_scaled_entry, 1.0)
    global_normalization = 1.0 / safe_largest
    left_scale = left_scale * global_normalization
    valid = mapping_valid & row_valid & column_valid & normalization_valid
    return PortShiftEquilibration(
        left_scale=jax.lax.stop_gradient(left_scale),
        right_scale=jax.lax.stop_gradient(right_scale),
        row_absolute_sums=jax.lax.stop_gradient(row_absolute_sums),
        column_absolute_sums=jax.lax.stop_gradient(column_absolute_sums),
        global_normalization=jax.lax.stop_gradient(global_normalization),
        is_valid=jax.lax.stop_gradient(valid),
    )


def prepare_matrix_free_port_shift(
    pencil: MatrixFreePortPencil,
    shift_per_m2: jax.Array,
) -> PreparedMatrixFreePortShift:
    """Prepare one shifted system once for repeated linear or Arnoldi applications."""

    if shift_per_m2.ndim != 0:
        raise ValueError("matrix-free port shift must be a scalar array")
    shifted = pencil.stiffness - shift_per_m2 * pencil.mass
    equilibration = _equilibrate_shifted_port_matrix(
        shifted,
        pencil.cell_reduced_dofs,
        free_dof_count=pencil.free_dof_count,
    )
    return PreparedMatrixFreePortShift(
        shifted_stiffness=shifted,
        mass=pencil.mass,
        cell_reduced_dofs=pencil.cell_reduced_dofs,
        shift_per_m2=shift_per_m2,
        equilibration=equilibration,
        free_dof_count=pencil.free_dof_count,
    )


def _equilibrated_port_shift_matvec(
    prepared: PreparedMatrixFreePortShift,
    vector: jax.Array,
) -> jax.Array:
    return prepared.equilibration.left_scale * matrix_free_port_matvec(
        prepared.shifted_stiffness,
        prepared.cell_reduced_dofs,
        prepared.equilibration.right_scale * vector,
    )


def prepare_matrix_free_port_block_preconditioner(
    prepared: PreparedMatrixFreePortShift,
    *,
    free_scalar_dof_count: int,
    policy: MatrixFreePortBlockPreconditionerPolicy = (
        DEFAULT_MATRIX_FREE_PORT_BLOCK_PRECONDITIONER_POLICY
    ),
) -> MatrixFreePortBlockPreconditioner:
    r"""Build a reusable lower block-triangular diagonal inverse approximation.

    In equilibrated coordinates ``C = L (A - sigma B) R``, the preconditioner applies

    ``s = inv(diag(C00)) r0`` and ``e = inv(diag(C11)) (r1 - C10 s)``.

    It stores only one inverse diagonal and two semantic masks.  The preconditioner is a stopped
    solver strategy; the admitted residual still belongs to the original equilibrated system.
    """

    if isinstance(free_scalar_dof_count, bool) or not isinstance(free_scalar_dof_count, int):
        raise TypeError("matrix-free preconditioner scalar DOF count must be a static integer")
    if free_scalar_dof_count < 0 or free_scalar_dof_count >= prepared.free_dof_count:
        raise ValueError(
            "matrix-free preconditioner scalar DOF count must leave at least one edge DOF"
        )

    free_dof_count = prepared.free_dof_count
    safe_mapping, mapping_valid = _safe_reduced_map(
        prepared.cell_reduced_dofs,
        free_dof_count=free_dof_count,
    )
    local_diagonal = jnp.diagonal(prepared.shifted_stiffness, axis1=1, axis2=2)
    assembled_diagonal = (
        jnp.zeros((free_dof_count + 1,), dtype=local_diagonal.dtype)
        .at[safe_mapping.reshape(-1)]
        .add(local_diagonal.reshape(-1))[:free_dof_count]
    )
    scaled_diagonal = (
        prepared.equilibration.left_scale * assembled_diagonal * prepared.equilibration.right_scale
    )
    absolute_diagonal = jnp.abs(scaled_diagonal)
    maximum_diagonal = jnp.max(absolute_diagonal)
    maximum_valid = jnp.isfinite(maximum_diagonal) & (maximum_diagonal > 0.0)
    safe_maximum = jnp.where(maximum_valid, maximum_diagonal, 1.0)
    relative_diagonal = absolute_diagonal / safe_maximum
    entries_valid = jnp.all(
        jnp.isfinite(scaled_diagonal)
        & jnp.isfinite(relative_diagonal)
        & (relative_diagonal >= policy.minimum_relative_diagonal)
    )
    valid = mapping_valid & prepared.equilibration.is_valid & maximum_valid & entries_valid
    safe_diagonal = jnp.where(valid, scaled_diagonal, jnp.ones_like(scaled_diagonal))
    indices = jnp.arange(free_dof_count)
    scalar_mask = indices < free_scalar_dof_count
    edge_mask = ~scalar_mask
    minimum_relative_diagonal = jnp.min(relative_diagonal)
    return MatrixFreePortBlockPreconditioner(
        inverse_scaled_diagonal=jax.lax.stop_gradient(1.0 / safe_diagonal),
        scalar_mask=jax.lax.stop_gradient(scalar_mask),
        edge_mask=jax.lax.stop_gradient(edge_mask),
        minimum_relative_diagonal=jax.lax.stop_gradient(minimum_relative_diagonal),
        is_valid=jax.lax.stop_gradient(valid),
    )


def apply_matrix_free_port_block_preconditioner(
    prepared: PreparedMatrixFreePortShift,
    preconditioner: MatrixFreePortBlockPreconditioner,
    vector: jax.Array,
) -> jax.Array:
    """Apply the stopped lower block-triangular preconditioner in equilibrated coordinates."""

    if vector.ndim != 1 or vector.shape[0] != prepared.free_dof_count:
        raise ValueError("matrix-free preconditioner vector does not match the prepared free DOFs")
    if preconditioner.inverse_scaled_diagonal.shape != vector.shape:
        raise ValueError("matrix-free preconditioner diagonal does not match the vector")
    if (
        preconditioner.scalar_mask.shape != vector.shape
        or preconditioner.edge_mask.shape != vector.shape
    ):
        raise ValueError("matrix-free preconditioner masks do not match the vector")

    dtype = jnp.result_type(vector.dtype, preconditioner.inverse_scaled_diagonal.dtype)
    inverse_diagonal = preconditioner.inverse_scaled_diagonal.astype(dtype)
    scalar_rhs = jnp.where(preconditioner.scalar_mask, vector, 0.0)
    scalar_solution = inverse_diagonal * scalar_rhs
    scalar_coupling = _equilibrated_port_shift_matvec(prepared, scalar_solution)
    edge_rhs = jnp.where(preconditioner.edge_mask, vector - scalar_coupling, 0.0)
    edge_solution = inverse_diagonal * edge_rhs
    result = scalar_solution + edge_solution
    return jnp.where(
        preconditioner.is_valid,
        result,
        jnp.asarray(jnp.nan, dtype=result.dtype),
    )


def _relative_residual(residual: jax.Array, right_hand_side: jax.Array) -> jax.Array:
    residual_norm = jnp.linalg.norm(residual)
    right_hand_side_norm = jnp.linalg.norm(right_hand_side)
    return jnp.where(
        right_hand_side_norm > 0.0,
        residual_norm / right_hand_side_norm,
        jnp.where(residual_norm == 0.0, 0.0, jnp.inf),
    )


def _implicit_gmres(
    matvec: Callable[[jax.Array], jax.Array],
    right_hand_side: jax.Array,
    *,
    policy: MatrixFreePortSolvePolicy,
    preconditioner: Callable[[jax.Array], jax.Array] | None = None,
    transpose_preconditioner: Callable[[jax.Array], jax.Array] | None = None,
) -> jax.Array:
    # Public JAX GMRES wraps its own custom_linear_solve, but its transpose solve reuses primal
    # RHS-dependent tolerances.  The outer residual-defined solve below invokes a fresh public
    # GMRES for each primal or transposed RHS, so both directions receive their own tolerance.
    def iterative_solve(operator: Callable[[jax.Array], jax.Array], rhs: jax.Array) -> jax.Array:
        solution, _ = gmres(  # type: ignore[no-untyped-call]
            operator,
            rhs,
            tol=policy.relative_tolerance,
            atol=policy.absolute_tolerance,
            restart=policy.restart,
            maxiter=policy.maximum_restart_cycles,
            M=preconditioner,
            solve_method=policy.solve_method,
        )
        return cast(jax.Array, solution)

    def transpose_iterative_solve(
        operator: Callable[[jax.Array], jax.Array],
        rhs: jax.Array,
    ) -> jax.Array:
        solution, _ = gmres(  # type: ignore[no-untyped-call]
            operator,
            rhs,
            tol=policy.relative_tolerance,
            atol=policy.absolute_tolerance,
            restart=policy.restart,
            maxiter=policy.maximum_restart_cycles,
            M=transpose_preconditioner,
            solve_method=policy.solve_method,
        )
        return cast(jax.Array, solution)

    return cast(
        jax.Array,
        jax.lax.custom_linear_solve(
            matvec,
            right_hand_side,
            solve=iterative_solve,
            transpose_solve=transpose_iterative_solve,
        ),
    )


def solve_prepared_matrix_free_port_shifted(
    prepared: PreparedMatrixFreePortShift,
    right_hand_side: jax.Array,
    *,
    policy: MatrixFreePortSolvePolicy = DEFAULT_MATRIX_FREE_PORT_SOLVE_POLICY,
    preconditioner: MatrixFreePortBlockPreconditioner | None = None,
) -> MatrixFreePortSolve:
    r"""Solve one prepared ``(A - sigma B) x = b`` system matrix-free.

    The returned residual is evaluated independently in the equilibrated coordinates actually
    used by GMRES.  Invalid equilibration, a non-finite result, or a residual above policy masks
    both solution representations with NaNs.  ``policy`` is static when this function is JIT
    compiled.
    """

    if right_hand_side.ndim != 1 or right_hand_side.shape[0] == 0:
        raise ValueError("matrix-free shifted right-hand side must be a nonempty rank-one array")
    if right_hand_side.shape[0] != prepared.free_dof_count:
        raise ValueError("matrix-free shifted right-hand side does not match the bound free DOFs")
    equilibration = prepared.equilibration
    left = equilibration.left_scale
    right = equilibration.right_scale

    def equilibrated_matvec(vector: jax.Array) -> jax.Array:
        return _equilibrated_port_shift_matvec(prepared, vector)

    forward_preconditioner: Callable[[jax.Array], jax.Array] | None = None
    transpose_preconditioner: Callable[[jax.Array], jax.Array] | None = None
    if preconditioner is not None:
        if preconditioner.inverse_scaled_diagonal.shape != right_hand_side.shape:
            raise ValueError("matrix-free preconditioner does not match the prepared free DOFs")

        def forward_preconditioner(vector: jax.Array) -> jax.Array:
            return apply_matrix_free_port_block_preconditioner(
                prepared,
                preconditioner,
                vector,
            )

        def transpose_preconditioner(vector: jax.Array) -> jax.Array:
            zero = jnp.zeros_like(vector)
            return cast(
                jax.Array,
                jax.linear_transpose(forward_preconditioner, zero)(vector)[0],
            )

    equilibrated_rhs = left * right_hand_side

    def zero_solve(_: None) -> jax.Array:
        return jnp.zeros_like(equilibrated_rhs)

    def nonzero_solve(_: None) -> jax.Array:
        return _implicit_gmres(
            equilibrated_matvec,
            equilibrated_rhs,
            policy=policy,
            preconditioner=forward_preconditioner,
            transpose_preconditioner=transpose_preconditioner,
        )

    equilibrated_solution = jax.lax.cond(
        jnp.linalg.norm(equilibrated_rhs) == 0.0,
        zero_solve,
        nonzero_solve,
        operand=None,
    )
    residual = equilibrated_matvec(equilibrated_solution) - equilibrated_rhs
    relative_residual = _relative_residual(residual, equilibrated_rhs)
    physical_solution = right * equilibrated_solution
    finite = (
        jnp.all(jnp.isfinite(equilibrated_solution))
        & jnp.all(jnp.isfinite(physical_solution))
        & jnp.isfinite(relative_residual)
    )
    preconditioner_valid = jnp.asarray(True) if preconditioner is None else preconditioner.is_valid
    valid = (
        equilibration.is_valid
        & preconditioner_valid
        & finite
        & (relative_residual <= policy.maximum_relative_residual)
    )
    valid = jax.lax.stop_gradient(valid)
    invalid_value = jnp.asarray(jnp.nan, dtype=physical_solution.dtype)
    admitted_physical = jnp.where(valid, physical_solution, invalid_value)
    admitted_equilibrated = jnp.where(valid, equilibrated_solution, invalid_value)
    diagnostics = MatrixFreePortSolveDiagnostics(
        equilibrated_relative_residual=jax.lax.stop_gradient(relative_residual),
        equilibrated_rhs_norm=jax.lax.stop_gradient(jnp.linalg.norm(equilibrated_rhs)),
        minimum_row_absolute_sum=jax.lax.stop_gradient(jnp.min(equilibration.row_absolute_sums)),
        maximum_row_absolute_sum=jax.lax.stop_gradient(jnp.max(equilibration.row_absolute_sums)),
        minimum_column_absolute_sum=jax.lax.stop_gradient(
            jnp.min(equilibration.column_absolute_sums)
        ),
        maximum_column_absolute_sum=jax.lax.stop_gradient(
            jnp.max(equilibration.column_absolute_sums)
        ),
        is_valid=valid,
    )
    return MatrixFreePortSolve(
        solution=admitted_physical,
        equilibrated_solution=admitted_equilibrated,
        equilibration=equilibration,
        diagnostics=diagnostics,
    )


def solve_matrix_free_port_shifted(
    pencil: MatrixFreePortPencil,
    right_hand_side: jax.Array,
    shift_per_m2: jax.Array,
    *,
    policy: MatrixFreePortSolvePolicy = DEFAULT_MATRIX_FREE_PORT_SOLVE_POLICY,
) -> MatrixFreePortSolve:
    r"""Prepare and solve ``(A - sigma B) x = b`` without a global square allocation."""

    prepared = prepare_matrix_free_port_shift(pencil, shift_per_m2)
    return solve_prepared_matrix_free_port_shifted(
        prepared,
        right_hand_side,
        policy=policy,
    )


def apply_prepared_matrix_free_port_shift_invert(
    prepared: PreparedMatrixFreePortShift,
    vector: jax.Array,
    *,
    policy: MatrixFreePortSolvePolicy = DEFAULT_MATRIX_FREE_PORT_SOLVE_POLICY,
    preconditioner: MatrixFreePortBlockPreconditioner | None = None,
) -> MatrixFreePortSolve:
    r"""Apply prepared Elmer-style ``inv(A - sigma B) B`` with optional reuse state."""

    right_hand_side = matrix_free_port_matvec(
        prepared.mass,
        prepared.cell_reduced_dofs,
        vector,
    )
    return solve_prepared_matrix_free_port_shifted(
        prepared,
        right_hand_side,
        policy=policy,
        preconditioner=preconditioner,
    )


def apply_matrix_free_port_shift_invert(
    pencil: MatrixFreePortPencil,
    vector: jax.Array,
    shift_per_m2: jax.Array,
    *,
    policy: MatrixFreePortSolvePolicy = DEFAULT_MATRIX_FREE_PORT_SOLVE_POLICY,
) -> MatrixFreePortSolve:
    r"""Apply Elmer's full mixed ``inv(A - sigma B) B`` operator."""

    prepared = prepare_matrix_free_port_shift(pencil, shift_per_m2)
    return apply_prepared_matrix_free_port_shift_invert(
        prepared,
        vector,
        policy=policy,
    )


def estimate_port_operator_storage(
    *,
    cell_count: int,
    free_dof_count: int,
    value_itemsize: int = 8,
    index_itemsize: int = 8,
) -> PortOperatorStorageEstimate:
    """Compare explicit representation bytes without claiming measured runtime memory."""

    for label, value in (
        ("cell_count", cell_count),
        ("free_dof_count", free_dof_count),
        ("value_itemsize", value_itemsize),
        ("index_itemsize", index_itemsize),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"matrix-free storage {label} must be a positive integer")
    matrix_free_value_bytes = 2 * cell_count * 6 * 6 * value_itemsize
    matrix_free_index_bytes = cell_count * 6 * index_itemsize
    matrix_free_total_bytes = matrix_free_value_bytes + matrix_free_index_bytes
    dense_pair_bytes = 2 * free_dof_count * free_dof_count * value_itemsize
    return PortOperatorStorageEstimate(
        matrix_free_value_bytes=matrix_free_value_bytes,
        matrix_free_index_bytes=matrix_free_index_bytes,
        matrix_free_total_bytes=matrix_free_total_bytes,
        dense_pair_bytes=dense_pair_bytes,
        dense_to_matrix_free_ratio=dense_pair_bytes / matrix_free_total_bytes,
    )
