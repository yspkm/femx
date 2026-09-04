r"""Dense native-JAX finite-spectrum solver for the mixed Elmer port pencil.

The locked no-potential ``EMPort`` mass matrix is singular because the scalar block carries no
generalized mass.  This module preserves its finite eigenpairs by eliminating only the invertible
scalar constraint.  It never forms an inverse of the full generalized mass matrix.

The implementation is a serial dense reference kernel.  It establishes algebraic, residual, JIT,
and convergence contracts before sparse shift-invert or distributed eigensolvers are introduced.
It intentionally does not differentiate its eigensolver trace. Separate simple-mode and
Riesz-cluster modules attach gradients to converged residuals and shifted solves.
"""

from __future__ import annotations

from typing import NamedTuple, cast

import jax
import jax.numpy as jnp


class PortSchurReduction(NamedTuple):
    r"""Finite-spectrum edge pencil and exact scalar recovery operator.

    ``scalar_recovery`` is :math:`A_{00}^{-1}A_{01}`.  A finite edge eigenvector ``e`` therefore
    recovers the scalar coefficients as ``s = -scalar_recovery @ e``.
    """

    scalar_stiffness: jax.Array
    scalar_edge_coupling: jax.Array
    edge_scalar_coupling: jax.Array
    edge_stiffness: jax.Array
    edge_mass: jax.Array
    scalar_recovery: jax.Array
    condensed_stiffness: jax.Array


class PortEigenResiduals(NamedTuple):
    """Dimensionless blockwise backward errors for every returned mode."""

    scalar_constraint: jax.Array
    edge_equation: jax.Array
    schur_equation: jax.Array
    maximum_mixed: jax.Array


class DensePortEigenmodes(NamedTuple):
    """Sorted finite eigenpairs of the reduced mixed port pencil.

    Coefficient columns are individually normalized in the positive edge-mass metric.  Their
    common complex phase is fixed by making the largest-magnitude edge coefficient positive real.
    Scalar and edge coefficients remain separate because they represent different physical
    quantities and must not be compared by raw magnitude.
    """

    eigenvalues_per_m2: jax.Array
    propagation_constants_per_m: jax.Array
    scalar_coefficients: jax.Array
    edge_coefficients: jax.Array
    phase_anchor_edge_dofs: jax.Array
    phase_factors: jax.Array
    edge_mass_norms_before_normalization: jax.Array
    residuals: PortEigenResiduals


class PortSubspaceComparison(NamedTuple):
    """Principal-angle evidence in one positive edge-mass metric."""

    singular_values: jax.Array
    principal_angles_rad: jax.Array
    projector_distance: jax.Array


def _validate_square_pencil_layout(
    stiffness: jax.Array,
    mass: jax.Array,
    scalar_dof_count: int,
) -> int:
    if stiffness.ndim != 2 or stiffness.shape[0] != stiffness.shape[1]:
        raise ValueError("port stiffness must be a square rank-two array")
    if mass.shape != stiffness.shape:
        raise ValueError("port mass must have the same square shape as stiffness")
    if isinstance(scalar_dof_count, bool) or not isinstance(scalar_dof_count, int):
        raise TypeError("scalar_dof_count must be a static integer")
    total_dof_count = stiffness.shape[0]
    if scalar_dof_count < 0 or scalar_dof_count >= total_dof_count:
        raise ValueError("scalar_dof_count must leave at least one edge DOF")
    return total_dof_count - scalar_dof_count


def schur_reduce_port_pencil(
    stiffness: jax.Array,
    mass: jax.Array,
    *,
    scalar_dof_count: int,
) -> PortSchurReduction:
    r"""Eliminate the scalar constraint without changing the finite generalized spectrum.

    Given the PEC-reduced Elmer-compatible pencil

    .. math::

       \begin{bmatrix}A_{00}&A_{01}\\A_{10}&A_{11}\end{bmatrix}
       \begin{bmatrix}s\\e\end{bmatrix}
       = \lambda
       \begin{bmatrix}0&0\\0&B_{11}\end{bmatrix}
       \begin{bmatrix}s\\e\end{bmatrix},

    this function constructs ``S = A11 - A10 solve(A00, A01)``.  For a mesh with no free scalar
    DOFs, the reduction is exactly ``S = A11`` and scalar recovery has shape ``(0, edge_dofs)``.
    ``scalar_dof_count`` is a static shape argument when this function is JIT compiled.
    """

    edge_dof_count = _validate_square_pencil_layout(stiffness, mass, scalar_dof_count)
    scalar = slice(0, scalar_dof_count)
    edge = slice(scalar_dof_count, scalar_dof_count + edge_dof_count)
    scalar_stiffness = stiffness[scalar, scalar]
    scalar_edge_coupling = stiffness[scalar, edge]
    edge_scalar_coupling = stiffness[edge, scalar]
    edge_stiffness = stiffness[edge, edge]
    edge_mass = mass[edge, edge]
    if scalar_dof_count == 0:
        scalar_recovery = jnp.zeros(
            (0, edge_dof_count),
            dtype=stiffness.dtype,
        )
    else:
        scalar_recovery = jnp.linalg.solve(
            scalar_stiffness,
            scalar_edge_coupling,
        )
    condensed_stiffness = edge_stiffness - edge_scalar_coupling @ scalar_recovery
    return PortSchurReduction(
        scalar_stiffness=scalar_stiffness,
        scalar_edge_coupling=scalar_edge_coupling,
        edge_scalar_coupling=edge_scalar_coupling,
        edge_stiffness=edge_stiffness,
        edge_mass=edge_mass,
        scalar_recovery=scalar_recovery,
        condensed_stiffness=condensed_stiffness,
    )


def _column_norm(values: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.real(jnp.sum(jnp.conj(values) * values, axis=0)))


def _relative_residual(numerator: jax.Array, denominator: jax.Array) -> jax.Array:
    ratio = numerator / jnp.where(denominator > 0.0, denominator, 1.0)
    return jnp.where(
        denominator > 0.0,
        ratio,
        jnp.where(numerator == 0.0, 0.0, jnp.inf),
    )


def _port_eigen_residuals(
    reduction: PortSchurReduction,
    eigenvalues: jax.Array,
    scalar_coefficients: jax.Array,
    edge_coefficients: jax.Array,
) -> PortEigenResiduals:
    scalar_residual = (
        reduction.scalar_stiffness @ scalar_coefficients
        + reduction.scalar_edge_coupling @ edge_coefficients
    )
    edge_mass_action = reduction.edge_mass @ edge_coefficients
    edge_residual = (
        reduction.edge_scalar_coupling @ scalar_coefficients
        + reduction.edge_stiffness @ edge_coefficients
        - edge_mass_action * eigenvalues[None, :]
    )
    schur_residual = (
        reduction.condensed_stiffness @ edge_coefficients - edge_mass_action * eigenvalues[None, :]
    )

    scalar_norm = _column_norm(scalar_coefficients)
    edge_norm = _column_norm(edge_coefficients)
    scalar_denominator = (
        jnp.linalg.norm(reduction.scalar_stiffness) * scalar_norm
        + jnp.linalg.norm(reduction.scalar_edge_coupling) * edge_norm
    )
    edge_denominator = (
        jnp.linalg.norm(reduction.edge_scalar_coupling) * scalar_norm
        + (
            jnp.linalg.norm(reduction.edge_stiffness)
            + jnp.abs(eigenvalues) * jnp.linalg.norm(reduction.edge_mass)
        )
        * edge_norm
    )
    schur_denominator = (
        jnp.linalg.norm(reduction.condensed_stiffness)
        + jnp.abs(eigenvalues) * jnp.linalg.norm(reduction.edge_mass)
    ) * edge_norm

    scalar_error = _relative_residual(
        _column_norm(scalar_residual),
        scalar_denominator,
    )
    edge_error = _relative_residual(
        _column_norm(edge_residual),
        edge_denominator,
    )
    schur_error = _relative_residual(
        _column_norm(schur_residual),
        schur_denominator,
    )
    return PortEigenResiduals(
        scalar_constraint=scalar_error,
        edge_equation=edge_error,
        schur_equation=schur_error,
        maximum_mixed=jnp.maximum(scalar_error, edge_error),
    )


def solve_dense_port_eigenmodes(
    stiffness: jax.Array,
    mass: jax.Array,
    propagation_scale_per_m: jax.Array,
    *,
    scalar_dof_count: int,
    mode_count: int,
) -> DensePortEigenmodes:
    r"""Solve and sort finite mixed-port eigenpairs with a dense JAX reference kernel.

    The standard eigenproblem is formed as

    ``solve(B11, S) / propagation_scale_per_m**2``.

    The explicit scale is normally Elmer's automatic shift scale
    ``omega * sqrt(max(epsilon) * max(mu))``.  It changes neither eigenvalues nor eigenvectors; it
    keeps the dense standard problem dimensionless.  Modes are ordered by decreasing real
    ``beta = sqrt(-lambda)``.  Increasing imaginary beta is the deterministic secondary key, but
    individual vectors inside a repeated eigenvalue remain non-authoritative.

    This function is JIT compatible when ``scalar_dof_count`` and ``mode_count`` are static.  It
    assumes a validated lossless pencil with invertible ``A00`` and positive-definite ``B11``.  It
    does not provide a reverse-mode derivative; the adjoint modules use residual-defined paths.
    """

    edge_dof_count = _validate_square_pencil_layout(stiffness, mass, scalar_dof_count)
    if isinstance(mode_count, bool) or not isinstance(mode_count, int):
        raise TypeError("mode_count must be a static integer")
    if mode_count <= 0 or mode_count > edge_dof_count:
        raise ValueError("mode_count must be positive and no larger than the finite spectrum")

    reduction = schur_reduce_port_pencil(
        stiffness,
        mass,
        scalar_dof_count=scalar_dof_count,
    )
    scale_squared = propagation_scale_per_m * propagation_scale_per_m
    dimensionless_operator = (
        jnp.linalg.solve(reduction.edge_mass, reduction.condensed_stiffness) / scale_squared
    )
    dimensionless_eigenvalues, raw_edge_coefficients = jnp.linalg.eig(dimensionless_operator)
    eigenvalues = dimensionless_eigenvalues * scale_squared
    propagation_constants = jnp.sqrt(-eigenvalues)
    order = jnp.lexsort(
        (
            jnp.imag(propagation_constants),
            -jnp.real(propagation_constants),
        )
    )[:mode_count]
    eigenvalues = eigenvalues[order]
    propagation_constants = propagation_constants[order]
    raw_edge_coefficients = raw_edge_coefficients[:, order]

    edge_mass_action = reduction.edge_mass @ raw_edge_coefficients
    mass_norm_squared = jnp.real(
        jnp.sum(jnp.conj(raw_edge_coefficients) * edge_mass_action, axis=0)
    )
    mass_norms = jnp.sqrt(mass_norm_squared)
    edge_coefficients = raw_edge_coefficients / mass_norms[None, :]

    anchor_dofs = jnp.argmax(jnp.abs(edge_coefficients), axis=0)
    mode_indices = jnp.arange(mode_count)
    anchors = edge_coefficients[anchor_dofs, mode_indices]
    phase_factors = jnp.conj(anchors) / jnp.abs(anchors)
    edge_coefficients = edge_coefficients * phase_factors[None, :]
    scalar_coefficients = -reduction.scalar_recovery @ edge_coefficients
    residuals = _port_eigen_residuals(
        reduction,
        eigenvalues,
        scalar_coefficients,
        edge_coefficients,
    )
    return DensePortEigenmodes(
        eigenvalues_per_m2=eigenvalues,
        propagation_constants_per_m=propagation_constants,
        scalar_coefficients=scalar_coefficients,
        edge_coefficients=edge_coefficients,
        phase_anchor_edge_dofs=anchor_dofs,
        phase_factors=phase_factors,
        edge_mass_norms_before_normalization=mass_norms,
        residuals=residuals,
    )


def compare_port_mode_subspaces(
    reference_basis: jax.Array,
    candidate_basis: jax.Array,
    edge_mass: jax.Array,
) -> PortSubspaceComparison:
    r"""Compare equally sized mode clusters through mass-weighted principal angles.

    Columnwise phase, permutation, and any nonsingular mixing inside a degenerate cluster do not
    affect the result.  Both bases are independently mass-orthonormalized before the singular
    values of their overlap are computed.
    """

    if reference_basis.ndim != 2 or candidate_basis.ndim != 2:
        raise ValueError("port subspace bases must be rank-two column matrices")
    if reference_basis.shape != candidate_basis.shape:
        raise ValueError("port subspace bases must have the same shape")
    if edge_mass.shape != (reference_basis.shape[0], reference_basis.shape[0]):
        raise ValueError("edge mass shape must match the subspace row dimension")
    if reference_basis.shape[1] == 0:
        raise ValueError("port subspace comparison requires at least one vector")

    def orthonormalize(basis: jax.Array) -> jax.Array:
        gram = jnp.conj(basis.T) @ edge_mass @ basis
        gram = 0.5 * (gram + jnp.conj(gram.T))
        lower = jnp.linalg.cholesky(gram)
        return cast(jax.Array, jnp.linalg.solve(jnp.conj(lower), basis.T).T)

    reference_orthonormal = orthonormalize(reference_basis)
    candidate_orthonormal = orthonormalize(candidate_basis)
    overlap = jnp.conj(reference_orthonormal.T) @ edge_mass @ candidate_orthonormal
    singular_values = jnp.linalg.svd(overlap, compute_uv=False)
    singular_values = jnp.clip(singular_values, 0.0, 1.0)
    principal_angles = jnp.arccos(singular_values)
    minimum_singular_value = jnp.min(singular_values)
    projector_distance = jnp.sqrt(jnp.maximum(0.0, 1.0 - minimum_singular_value**2))
    return PortSubspaceComparison(
        singular_values=singular_values,
        principal_angles_rad=principal_angles,
        projector_distance=projector_distance,
    )
