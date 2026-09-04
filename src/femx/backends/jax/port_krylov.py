r"""Matrix-free generalized Arnoldi for the finite Elmer-compatible port spectrum.

Elmer's locked complex eigensolver uses ARPACK mode-3 reverse communication: it repeatedly
applies ``(A - sigma B)^-1 B`` while retaining the singular generalized mass ``B``.  A plain
Euclidean Arnoldi process on the full mixed coefficients is unsuitable here because the scalar
and edge fields have different units and the mass vanishes on the scalar constraint.  This
clean-room JAX kernel instead works in the positive ``B`` inner product after projecting the
starting vector through the shift-invert range.

The implementation stores ``O(N m)`` Krylov vectors and cell-local 6-by-6 operators, never a
global ``N``-by-``N`` matrix.  It is a serial forward-spectrum reference kernel.  Its output is
explicitly stopped from differentiation; residual-defined eigen-adjoint work remains a separate
capability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, NamedTuple, cast

import jax
import jax.numpy as jnp

from .port_matrix_free import (
    DEFAULT_MATRIX_FREE_PORT_BLOCK_PRECONDITIONER_POLICY,
    DEFAULT_MATRIX_FREE_PORT_SOLVE_POLICY,
    MatrixFreePortBlockPreconditionerPolicy,
    MatrixFreePortPencil,
    MatrixFreePortSolvePolicy,
    apply_prepared_matrix_free_port_shift_invert,
    matrix_free_port_matvec,
    prepare_matrix_free_port_block_preconditioner,
    prepare_matrix_free_port_shift,
)


@dataclass(frozen=True, slots=True)
class MatrixFreePortArnoldiPolicy:
    """Static subspace size and independent admission thresholds."""

    krylov_dimension: int = 25
    minimum_mass_norm: float = 1.0e-30
    minimum_subdiagonal: float = 1.0e-13
    minimum_transformed_eigenvalue_magnitude: float = 1.0e-12
    maximum_relative_ritz_residual: float = 1.0e-7
    maximum_generalized_residual: float = 5.0e-9
    maximum_mass_orthogonality_error: float = 1.0e-9

    def __post_init__(self) -> None:
        if (
            isinstance(self.krylov_dimension, bool)
            or not isinstance(self.krylov_dimension, int)
            or self.krylov_dimension <= 1
        ):
            raise ValueError("matrix-free Arnoldi dimension must be an integer greater than one")
        for label, value in (
            ("minimum mass norm", self.minimum_mass_norm),
            ("minimum subdiagonal", self.minimum_subdiagonal),
            (
                "minimum transformed eigenvalue magnitude",
                self.minimum_transformed_eigenvalue_magnitude,
            ),
            ("maximum relative Ritz residual", self.maximum_relative_ritz_residual),
            ("maximum generalized residual", self.maximum_generalized_residual),
            ("maximum mass orthogonality error", self.maximum_mass_orthogonality_error),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"matrix-free Arnoldi {label} must be finite and positive")


DEFAULT_MATRIX_FREE_PORT_ARNOLDI_POLICY: Final = MatrixFreePortArnoldiPolicy()


class MatrixFreePortEigenResiduals(NamedTuple):
    """Componentwise-scaled generalized residuals for each requested Ritz pair."""

    scalar_constraint: jax.Array
    edge_equation: jax.Array
    combined_equation: jax.Array
    maximum_mixed: jax.Array


class MatrixFreePortArnoldiDiagnostics(NamedTuple):
    """Linear, Krylov, orthogonality, and final equation evidence."""

    shift_invert_relative_residuals: jax.Array
    shift_invert_validity: jax.Array
    arnoldi_subdiagonals: jax.Array
    relative_ritz_residuals: jax.Array
    mass_orthogonality_error: jax.Array
    projected_start_mass_norm: jax.Array
    minimum_preconditioner_relative_diagonal: jax.Array
    is_valid: jax.Array


class MatrixFreePortEigenmodes(NamedTuple):
    """Sorted, mass-normalized finite mixed-port Ritz pairs."""

    eigenvalues_per_m2: jax.Array
    propagation_constants_per_m: jax.Array
    scalar_coefficients: jax.Array
    edge_coefficients: jax.Array
    phase_anchor_edge_dofs: jax.Array
    phase_factors: jax.Array
    edge_mass_norms_before_normalization: jax.Array
    residuals: MatrixFreePortEigenResiduals
    diagnostics: MatrixFreePortArnoldiDiagnostics


def _mass_inner(left: jax.Array, mass_right: jax.Array) -> jax.Array:
    return jnp.real(jnp.vdot(left, mass_right))


def _column_norm(values: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.real(jnp.sum(jnp.conj(values) * values, axis=0)))


def _relative_scaled_residual(
    residual: jax.Array,
    row_magnitude_bound: jax.Array,
) -> jax.Array:
    numerator = _column_norm(residual)
    denominator = _column_norm(row_magnitude_bound)
    return jnp.where(
        denominator > 0.0,
        numerator / denominator,
        jnp.where(numerator == 0.0, 0.0, jnp.inf),
    )


def _matrix_free_row_magnitude_bound(
    pencil: MatrixFreePortPencil,
    vector: jax.Array,
    eigenvalue: jax.Array,
) -> jax.Array:
    r"""Scatter ``sum_j (abs(A_ij) + abs(lambda) abs(B_ij)) abs(x_j)`` by row."""

    free_dof_count = vector.shape[0]
    safe_mapping = jnp.clip(pencil.cell_reduced_dofs, 0, free_dof_count)
    mapping_valid = jnp.all(
        (pencil.cell_reduced_dofs >= 0) & (pencil.cell_reduced_dofs <= free_dof_count)
    )
    vector_magnitude = jnp.abs(vector)
    extended_magnitude = jnp.concatenate(
        (vector_magnitude, jnp.zeros((1,), dtype=vector_magnitude.dtype))
    )
    local_input = extended_magnitude[safe_mapping]
    local_matrix_bound = jnp.abs(pencil.stiffness) + jnp.abs(eigenvalue) * jnp.abs(pencil.mass)
    local_bound = jnp.einsum("cij,cj->ci", local_matrix_bound, local_input)
    assembled = (
        jnp.zeros((free_dof_count + 1,), dtype=local_bound.dtype)
        .at[safe_mapping.reshape(-1)]
        .add(local_bound.reshape(-1))
    )
    return jnp.where(mapping_valid, assembled[:free_dof_count], jnp.nan)


def _canonical_propagation_branch(eigenvalues: jax.Array) -> jax.Array:
    propagation_constants = jnp.sqrt(-eigenvalues)
    reverse = (jnp.real(propagation_constants) < 0.0) | (
        (jnp.real(propagation_constants) == 0.0) & (jnp.imag(propagation_constants) < 0.0)
    )
    return jnp.where(reverse, -propagation_constants, propagation_constants)


def _stop_tree(value: MatrixFreePortEigenmodes) -> MatrixFreePortEigenmodes:
    return cast(MatrixFreePortEigenmodes, jax.tree.map(jax.lax.stop_gradient, value))


def solve_matrix_free_port_eigenmodes(
    pencil: MatrixFreePortPencil,
    shift_per_m2: jax.Array,
    initial_vector: jax.Array,
    *,
    free_scalar_dof_count: int,
    mode_count: int,
    arnoldi_policy: MatrixFreePortArnoldiPolicy = DEFAULT_MATRIX_FREE_PORT_ARNOLDI_POLICY,
    linear_policy: MatrixFreePortSolvePolicy = DEFAULT_MATRIX_FREE_PORT_SOLVE_POLICY,
    preconditioner_policy: MatrixFreePortBlockPreconditionerPolicy = (
        DEFAULT_MATRIX_FREE_PORT_BLOCK_PRECONDITIONER_POLICY
    ),
) -> MatrixFreePortEigenmodes:
    r"""Return selected finite modes from a full mixed ``B``-Arnoldi process.

    The dimensionless Krylov operator is

    ``T = abs(sigma) (A - sigma B)^-1 B``.

    Ritz values ``theta`` map back through ``lambda = sigma + abs(sigma) / theta``.  Candidates
    are selected by largest ``abs(theta)``, matching the target-near-shift semantics of Elmer's
    ARPACK mode 3, and are finally ordered by decreasing real propagation constant.
    """

    if initial_vector.ndim != 1 or initial_vector.shape[0] != pencil.free_dof_count:
        raise ValueError("matrix-free Arnoldi initial vector must match the free mixed DOFs")
    if shift_per_m2.ndim != 0:
        raise ValueError("matrix-free Arnoldi shift must be a scalar array")
    if not jnp.issubdtype(pencil.stiffness.dtype, jnp.floating) or not jnp.issubdtype(
        pencil.mass.dtype, jnp.floating
    ):
        raise TypeError("matrix-free Arnoldi v1 requires a real lossless port pencil")
    if not (
        jnp.issubdtype(initial_vector.dtype, jnp.floating)
        or jnp.issubdtype(initial_vector.dtype, jnp.complexfloating)
    ):
        raise TypeError("matrix-free Arnoldi initial vector must be floating or complex")
    if isinstance(free_scalar_dof_count, bool) or not isinstance(free_scalar_dof_count, int):
        raise TypeError("matrix-free Arnoldi scalar DOF count must be a static integer")
    if free_scalar_dof_count < 0 or free_scalar_dof_count >= pencil.free_dof_count:
        raise ValueError("matrix-free Arnoldi scalar DOF count must leave finite edge DOFs")
    edge_dof_count = pencil.free_dof_count - free_scalar_dof_count
    if isinstance(mode_count, bool) or not isinstance(mode_count, int):
        raise TypeError("matrix-free Arnoldi mode count must be a static integer")
    if mode_count <= 0 or mode_count >= arnoldi_policy.krylov_dimension:
        raise ValueError("matrix-free Arnoldi mode count must lie below the Krylov dimension")
    if arnoldi_policy.krylov_dimension >= edge_dof_count:
        raise ValueError("matrix-free Arnoldi dimension must be smaller than the finite spectrum")

    prepared = prepare_matrix_free_port_shift(pencil, shift_per_m2)
    preconditioner = prepare_matrix_free_port_block_preconditioner(
        prepared,
        free_scalar_dof_count=free_scalar_dof_count,
        policy=preconditioner_policy,
    )
    krylov_dimension = arnoldi_policy.krylov_dimension
    free_dof_count = pencil.free_dof_count
    complex_dtype = jnp.result_type(pencil.stiffness.dtype, initial_vector.dtype, jnp.complex64)
    transformed_scale = jnp.abs(shift_per_m2).astype(complex_dtype)

    def mass_action(vector: jax.Array) -> jax.Array:
        return matrix_free_port_matvec(
            prepared.mass,
            prepared.cell_reduced_dofs,
            vector,
        )

    def transformed_action(vector: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        result = apply_prepared_matrix_free_port_shift_invert(
            prepared,
            vector,
            policy=linear_policy,
            preconditioner=preconditioner,
        )
        safe_solution = jnp.where(result.diagnostics.is_valid, result.solution, 0.0)
        return (
            transformed_scale * safe_solution,
            result.diagnostics.equilibrated_relative_residual,
            result.diagnostics.is_valid,
        )

    projected_start, start_linear_residual, start_linear_valid = transformed_action(
        initial_vector.astype(complex_dtype)
    )
    mass_projected_start = mass_action(projected_start)
    start_mass_squared = _mass_inner(projected_start, mass_projected_start)
    start_mass_norm = jnp.sqrt(jnp.maximum(start_mass_squared, 0.0))
    start_valid = (
        start_linear_valid
        & jnp.isfinite(start_mass_norm)
        & (start_mass_norm >= arnoldi_policy.minimum_mass_norm)
    )
    safe_start_mass_norm = jnp.where(start_valid, start_mass_norm, 1.0)
    first_basis = jnp.where(start_valid, projected_start / safe_start_mass_norm, 0.0)
    first_mass_basis = jnp.where(
        start_valid,
        mass_projected_start / safe_start_mass_norm,
        0.0,
    )

    basis = jnp.zeros((free_dof_count, krylov_dimension + 1), dtype=complex_dtype)
    mass_basis = jnp.zeros_like(basis)
    hessenberg = jnp.zeros(
        (krylov_dimension + 1, krylov_dimension),
        dtype=complex_dtype,
    )
    basis = basis.at[:, 0].set(first_basis)
    mass_basis = mass_basis.at[:, 0].set(first_mass_basis)
    linear_residuals = jnp.full((krylov_dimension + 1,), jnp.inf, dtype=jnp.float64)
    linear_validity = jnp.zeros((krylov_dimension + 1,), dtype=jnp.bool_)
    linear_residuals = linear_residuals.at[0].set(start_linear_residual)
    linear_validity = linear_validity.at[0].set(start_linear_valid)

    def arnoldi_step(
        column: int,
        state: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        q_basis, b_basis, h_matrix, solve_residuals, solve_validity, valid_so_far = state
        transformed, solve_residual, solve_valid = transformed_action(q_basis[:, column])
        work = jnp.where(solve_valid, transformed, 0.0)
        mass_work = mass_action(work)
        active_columns = (jnp.arange(krylov_dimension + 1) <= column).astype(complex_dtype)
        h_column = jnp.zeros((krylov_dimension + 1,), dtype=complex_dtype)

        def reorthogonalize(
            _: int,
            orthogonalization_state: tuple[jax.Array, jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            candidate, mass_candidate, accumulated = orthogonalization_state
            projections = (jnp.conj(q_basis.T) @ mass_candidate) * active_columns
            candidate = candidate - q_basis @ projections
            mass_candidate = mass_action(candidate)
            return candidate, mass_candidate, accumulated + projections

        work, mass_work, h_column = jax.lax.fori_loop(
            0,
            2,
            reorthogonalize,
            (work, mass_work, h_column),
        )
        subdiagonal_squared = _mass_inner(work, mass_work)
        subdiagonal = jnp.sqrt(jnp.maximum(subdiagonal_squared, 0.0))
        step_valid = (
            valid_so_far
            & solve_valid
            & jnp.isfinite(subdiagonal)
            & (subdiagonal >= arnoldi_policy.minimum_subdiagonal)
        )
        safe_subdiagonal = jnp.where(step_valid, subdiagonal, 1.0)
        next_basis = jnp.where(step_valid, work / safe_subdiagonal, 0.0)
        next_mass_basis = jnp.where(step_valid, mass_work / safe_subdiagonal, 0.0)
        h_column = h_column.at[column + 1].set(subdiagonal.astype(complex_dtype))
        return (
            q_basis.at[:, column + 1].set(next_basis),
            b_basis.at[:, column + 1].set(next_mass_basis),
            h_matrix.at[:, column].set(h_column),
            solve_residuals.at[column + 1].set(solve_residual),
            solve_validity.at[column + 1].set(solve_valid),
            step_valid,
        )

    basis, mass_basis, hessenberg, linear_residuals, linear_validity, arnoldi_valid = (
        jax.lax.fori_loop(
            0,
            krylov_dimension,
            arnoldi_step,
            (
                basis,
                mass_basis,
                hessenberg,
                linear_residuals,
                linear_validity,
                start_valid,
            ),
        )
    )

    projected_hessenberg = hessenberg[:krylov_dimension, :]
    fallback_hessenberg = jnp.diag(
        jnp.arange(1, krylov_dimension + 1, dtype=jnp.float64).astype(complex_dtype)
    )
    safe_hessenberg = jnp.where(arnoldi_valid, projected_hessenberg, fallback_hessenberg)
    safe_hessenberg = jax.lax.stop_gradient(safe_hessenberg)
    transformed_eigenvalues, hessenberg_eigenvectors = jnp.linalg.eig(safe_hessenberg)
    target = jnp.argsort(-jnp.abs(transformed_eigenvalues))[:mode_count]
    selected_transformed = transformed_eigenvalues[target]
    transformed_valid = jnp.all(
        jnp.isfinite(selected_transformed)
        & (jnp.abs(selected_transformed) >= arnoldi_policy.minimum_transformed_eigenvalue_magnitude)
    )
    safe_transformed = jnp.where(transformed_valid, selected_transformed, 1.0)
    eigenvalues = shift_per_m2.astype(complex_dtype) + transformed_scale / safe_transformed
    propagation_constants = _canonical_propagation_branch(eigenvalues)
    order = jnp.lexsort((jnp.imag(propagation_constants), -jnp.real(propagation_constants)))
    eigenvalues = eigenvalues[order]
    propagation_constants = propagation_constants[order]
    selected_hessenberg_vectors = hessenberg_eigenvectors[:, target][:, order]
    ritz_vectors = basis[:, :krylov_dimension] @ selected_hessenberg_vectors

    mass_ritz_vectors = jax.vmap(mass_action, in_axes=1, out_axes=1)(ritz_vectors)
    mass_norm_squared = jnp.real(jnp.sum(jnp.conj(ritz_vectors) * mass_ritz_vectors, axis=0))
    mass_norms = jnp.sqrt(jnp.maximum(mass_norm_squared, 0.0))
    mass_norms_valid = jnp.all(
        jnp.isfinite(mass_norms) & (mass_norms >= arnoldi_policy.minimum_mass_norm)
    )
    safe_mass_norms = jnp.where(mass_norms_valid, mass_norms, 1.0)
    normalized = ritz_vectors / safe_mass_norms[None, :]
    edge_coefficients = normalized[free_scalar_dof_count:, :]
    anchor_dofs = jnp.argmax(jnp.abs(edge_coefficients), axis=0)
    mode_indices = jnp.arange(mode_count)
    anchors = edge_coefficients[anchor_dofs, mode_indices]
    anchors_valid = jnp.all(jnp.isfinite(anchors) & (jnp.abs(anchors) > 0.0))
    safe_anchors = jnp.where(anchors_valid, anchors, 1.0)
    phase_factors = jnp.conj(safe_anchors) / jnp.abs(safe_anchors)
    normalized = normalized * phase_factors[None, :]
    scalar_coefficients = normalized[:free_scalar_dof_count, :]
    edge_coefficients = normalized[free_scalar_dof_count:, :]

    stiffness_action = jax.vmap(
        lambda vector: matrix_free_port_matvec(
            pencil.stiffness,
            pencil.cell_reduced_dofs,
            vector,
        ),
        in_axes=1,
        out_axes=1,
    )(normalized)
    mass_action_modes = jax.vmap(mass_action, in_axes=1, out_axes=1)(normalized)
    generalized_residual = stiffness_action - mass_action_modes * eigenvalues[None, :]
    scaled_residual = prepared.equilibration.left_scale[:, None] * generalized_residual
    row_magnitude_bounds = jax.vmap(
        lambda vector, eigenvalue: _matrix_free_row_magnitude_bound(
            pencil,
            vector,
            eigenvalue,
        ),
        in_axes=(1, 0),
        out_axes=1,
    )(normalized, eigenvalues)
    scaled_row_magnitude_bounds = prepared.equilibration.left_scale[:, None] * row_magnitude_bounds
    scalar_residuals = _relative_scaled_residual(
        scaled_residual[:free_scalar_dof_count, :],
        scaled_row_magnitude_bounds[:free_scalar_dof_count, :],
    )
    edge_residuals = _relative_scaled_residual(
        scaled_residual[free_scalar_dof_count:, :],
        scaled_row_magnitude_bounds[free_scalar_dof_count:, :],
    )
    combined_residuals = _relative_scaled_residual(
        scaled_residual,
        scaled_row_magnitude_bounds,
    )
    maximum_mixed = jnp.maximum(scalar_residuals, edge_residuals)

    mass_gram = jnp.conj(basis[:, :krylov_dimension].T) @ mass_basis[:, :krylov_dimension]
    mass_orthogonality_error = jnp.linalg.norm(
        mass_gram - jnp.eye(krylov_dimension, dtype=complex_dtype)
    )
    last_subdiagonal = hessenberg[krylov_dimension, krylov_dimension - 1]
    relative_ritz_residuals = jnp.abs(
        last_subdiagonal * selected_hessenberg_vectors[-1, :]
    ) / jnp.maximum(
        jnp.abs(selected_transformed[order]),
        arnoldi_policy.minimum_transformed_eigenvalue_magnitude,
    )
    finite = (
        jnp.all(jnp.isfinite(eigenvalues))
        & jnp.all(jnp.isfinite(propagation_constants))
        & jnp.all(jnp.isfinite(normalized))
        & jnp.all(jnp.isfinite(maximum_mixed))
        & jnp.all(jnp.isfinite(relative_ritz_residuals))
        & jnp.isfinite(mass_orthogonality_error)
        & jnp.isfinite(shift_per_m2)
        & (jnp.abs(shift_per_m2) > 0.0)
    )
    valid = (
        arnoldi_valid
        & transformed_valid
        & mass_norms_valid
        & anchors_valid
        & preconditioner.is_valid
        & finite
        & jnp.all(linear_validity)
        & jnp.all(relative_ritz_residuals <= arnoldi_policy.maximum_relative_ritz_residual)
        & jnp.all(maximum_mixed <= arnoldi_policy.maximum_generalized_residual)
        & (mass_orthogonality_error <= arnoldi_policy.maximum_mass_orthogonality_error)
    )
    valid = jax.lax.stop_gradient(valid)
    invalid_complex = jnp.asarray(jnp.nan + 1.0j * jnp.nan, dtype=complex_dtype)
    invalid_real = jnp.asarray(jnp.nan, dtype=jnp.float64)
    result = MatrixFreePortEigenmodes(
        eigenvalues_per_m2=jnp.where(valid, eigenvalues, invalid_complex),
        propagation_constants_per_m=jnp.where(valid, propagation_constants, invalid_complex),
        scalar_coefficients=jnp.where(valid, scalar_coefficients, invalid_complex),
        edge_coefficients=jnp.where(valid, edge_coefficients, invalid_complex),
        phase_anchor_edge_dofs=jnp.where(valid, anchor_dofs, -1),
        phase_factors=jnp.where(valid, phase_factors, invalid_complex),
        edge_mass_norms_before_normalization=jnp.where(valid, mass_norms, invalid_real),
        residuals=MatrixFreePortEigenResiduals(
            scalar_constraint=jnp.where(valid, scalar_residuals, invalid_real),
            edge_equation=jnp.where(valid, edge_residuals, invalid_real),
            combined_equation=jnp.where(valid, combined_residuals, invalid_real),
            maximum_mixed=jnp.where(valid, maximum_mixed, invalid_real),
        ),
        diagnostics=MatrixFreePortArnoldiDiagnostics(
            shift_invert_relative_residuals=jax.lax.stop_gradient(linear_residuals),
            shift_invert_validity=jax.lax.stop_gradient(linear_validity),
            arnoldi_subdiagonals=jax.lax.stop_gradient(jnp.real(jnp.diag(hessenberg[1:, :]))),
            relative_ritz_residuals=jax.lax.stop_gradient(relative_ritz_residuals),
            mass_orthogonality_error=jax.lax.stop_gradient(mass_orthogonality_error),
            projected_start_mass_norm=jax.lax.stop_gradient(start_mass_norm),
            minimum_preconditioner_relative_diagonal=(preconditioner.minimum_relative_diagonal),
            is_valid=valid,
        ),
    )
    return _stop_tree(result)
