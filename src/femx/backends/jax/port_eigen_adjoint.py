r"""Residual-defined reverse mode for one simple lossless port eigenpair.

The condensed Elmer-compatible port pencil is generally non-symmetric even for real lossless
materials.  This module therefore differentiates the generalized residual directly instead of
using a Hermitian eigensolver identity or tracing through :func:`jax.numpy.linalg.eig`.

Only one real, forward-propagating, algebraically simple mode is admitted.  The caller supplies a
fixed phase/sign anchor chosen from a validated baseline.  A repeated or insufficiently separated
eigenvalue, an unusable anchor, a complex mode, a non-symmetric mass matrix, or a poor primal
residual produces non-finite outputs and gradients.  That fail-closed behavior is intentional: an
individual eigenvector derivative is not authoritative at a mode crossing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Final, NamedTuple

import jax
import jax.numpy as jnp

_FLOAT64_TINY: Final = 2.225_073_858_507_201_4e-308


@dataclass(frozen=True, slots=True)
class SimplePortEigenpairPolicy:
    """Numerical admission thresholds for an individual port-mode derivative."""

    minimum_relative_eigenvalue_gap: float = 1.0e-6
    maximum_relative_eigenvalue_imaginary_part: float = 1.0e-10
    maximum_relative_eigenvector_imaginary_norm: float = 1.0e-10
    maximum_relative_residual: float = 1.0e-10
    maximum_mass_symmetry_error: float = 1.0e-12
    minimum_phase_anchor_relative_magnitude: float = 1.0e-3

    def __post_init__(self) -> None:
        positive = (
            ("minimum eigenvalue gap", self.minimum_relative_eigenvalue_gap),
            (
                "maximum eigenvalue imaginary part",
                self.maximum_relative_eigenvalue_imaginary_part,
            ),
            (
                "maximum eigenvector imaginary norm",
                self.maximum_relative_eigenvector_imaginary_norm,
            ),
            ("maximum residual", self.maximum_relative_residual),
            ("maximum mass symmetry error", self.maximum_mass_symmetry_error),
            (
                "minimum phase-anchor magnitude",
                self.minimum_phase_anchor_relative_magnitude,
            ),
        )
        for label, value in positive:
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"port eigen-adjoint {label} must be finite and positive")
        if self.minimum_phase_anchor_relative_magnitude > 1.0:
            raise ValueError("port eigen-adjoint minimum phase-anchor magnitude cannot exceed one")


_DEFAULT_SIMPLE_PORT_EIGENPAIR_POLICY = SimplePortEigenpairPolicy()


class SimplePortEigenpair(NamedTuple):
    """One real B-normalized right eigenpair of the dimensionless condensed pencil."""

    dimensionless_eigenvalue: jax.Array
    edge_coefficients: jax.Array


class SimplePortEigenpairDiagnostics(NamedTuple):
    """Stopped-gradient predicates used to admit or reject a simple-mode derivative."""

    relative_eigenvalue_gap: jax.Array
    relative_eigenvalue_imaginary_part: jax.Array
    relative_eigenvector_imaginary_norm: jax.Array
    relative_residual: jax.Array
    mass_symmetry_error: jax.Array
    phase_anchor_relative_magnitude: jax.Array
    mass_norm_squared_before_normalization: jax.Array
    is_valid: jax.Array


class SimplePortEigenpairInspection(NamedTuple):
    """Primal eigenpair plus the evidence controlling differentiability."""

    eigenpair: SimplePortEigenpair
    diagnostics: SimplePortEigenpairDiagnostics


class PhysicalSimplePortEigenpair(NamedTuple):
    """One simple mode in physical propagation units."""

    eigenvalue_per_m2: jax.Array
    propagation_constant_per_m: jax.Array
    edge_coefficients: jax.Array


def _validate_layout(
    dimensionless_stiffness: jax.Array,
    edge_mass: jax.Array,
    *,
    selected_mode_index: int,
    phase_anchor_edge_dof: int,
) -> None:
    if (
        dimensionless_stiffness.ndim != 2
        or dimensionless_stiffness.shape[0] != dimensionless_stiffness.shape[1]
    ):
        raise ValueError("dimensionless port stiffness must be a square rank-two array")
    if edge_mass.shape != dimensionless_stiffness.shape:
        raise ValueError("port edge mass must match the dimensionless stiffness shape")
    edge_dof_count = dimensionless_stiffness.shape[0]
    if isinstance(selected_mode_index, bool) or not isinstance(selected_mode_index, int):
        raise TypeError("selected_mode_index must be a static integer")
    if selected_mode_index < 0 or selected_mode_index >= edge_dof_count:
        raise ValueError("selected_mode_index must lie in the finite edge spectrum")
    if isinstance(phase_anchor_edge_dof, bool) or not isinstance(phase_anchor_edge_dof, int):
        raise TypeError("phase_anchor_edge_dof must be a static integer")
    if phase_anchor_edge_dof < 0 or phase_anchor_edge_dof >= edge_dof_count:
        raise ValueError("phase_anchor_edge_dof must identify one finite edge DOF")


def _solve_and_inspect_dimensionless(
    dimensionless_stiffness: jax.Array,
    edge_mass: jax.Array,
    *,
    selected_mode_index: int,
    phase_anchor_edge_dof: int,
    policy: SimplePortEigenpairPolicy,
) -> SimplePortEigenpairInspection:
    _validate_layout(
        dimensionless_stiffness,
        edge_mass,
        selected_mode_index=selected_mode_index,
        phase_anchor_edge_dof=phase_anchor_edge_dof,
    )
    standard_operator = jnp.linalg.solve(edge_mass, dimensionless_stiffness)
    eigenvalues, raw_edge_coefficients = jnp.linalg.eig(standard_operator)
    propagation_constants = jnp.sqrt(-eigenvalues)
    order = jnp.lexsort(
        (
            jnp.imag(propagation_constants),
            -jnp.real(propagation_constants),
        )
    )
    selected_spectrum_index = order[selected_mode_index]
    selected_eigenvalue = eigenvalues[selected_spectrum_index]
    raw_edge = raw_edge_coefficients[:, selected_spectrum_index]

    anchor = raw_edge[phase_anchor_edge_dof]
    anchor_magnitude = jnp.abs(anchor)
    safe_anchor_magnitude = jnp.where(anchor_magnitude > 0.0, anchor_magnitude, 1.0)
    phase_factor = jnp.conj(anchor) / safe_anchor_magnitude
    phased_edge = raw_edge * phase_factor
    real_edge = jnp.real(phased_edge)

    mass_norm_squared = real_edge @ edge_mass @ real_edge
    safe_mass_norm = jnp.sqrt(jnp.where(mass_norm_squared > 0.0, mass_norm_squared, 1.0))
    normalized_edge = real_edge / safe_mass_norm
    real_eigenvalue = jnp.real(selected_eigenvalue)

    distances = jnp.abs(eigenvalues - selected_eigenvalue)
    distances = distances.at[selected_spectrum_index].set(jnp.inf)
    relative_gap = jnp.min(distances) / jnp.maximum(jnp.abs(selected_eigenvalue), 1.0)
    eigenvalue_imaginary_part = jnp.abs(jnp.imag(selected_eigenvalue)) / jnp.maximum(
        jnp.abs(selected_eigenvalue),
        1.0,
    )
    eigenvector_imaginary_norm = jnp.linalg.norm(jnp.imag(phased_edge)) / jnp.maximum(
        jnp.linalg.norm(phased_edge),
        _FLOAT64_TINY,
    )
    residual = dimensionless_stiffness @ normalized_edge - real_eigenvalue * (
        edge_mass @ normalized_edge
    )
    residual_denominator = (
        jnp.linalg.norm(dimensionless_stiffness)
        + jnp.abs(real_eigenvalue) * jnp.linalg.norm(edge_mass)
    ) * jnp.linalg.norm(normalized_edge)
    relative_residual = jnp.linalg.norm(residual) / jnp.where(
        residual_denominator > 0.0,
        residual_denominator,
        1.0,
    )
    mass_symmetry_error = jnp.linalg.norm(edge_mass - edge_mass.T) / jnp.maximum(
        jnp.linalg.norm(edge_mass),
        _FLOAT64_TINY,
    )
    phase_anchor_relative_magnitude = anchor_magnitude / jnp.maximum(
        jnp.max(jnp.abs(raw_edge)),
        _FLOAT64_TINY,
    )
    finite_inputs = jnp.all(jnp.isfinite(dimensionless_stiffness)) & jnp.all(
        jnp.isfinite(edge_mass)
    )
    finite_eigenpair = (
        jnp.isfinite(real_eigenvalue)
        & jnp.all(jnp.isfinite(normalized_edge))
        & jnp.isfinite(mass_norm_squared)
    )
    is_valid = (
        finite_inputs
        & finite_eigenpair
        & (real_eigenvalue < 0.0)
        & (mass_norm_squared > 0.0)
        & (anchor_magnitude > 0.0)
        & (relative_gap >= policy.minimum_relative_eigenvalue_gap)
        & (eigenvalue_imaginary_part <= policy.maximum_relative_eigenvalue_imaginary_part)
        & (eigenvector_imaginary_norm <= policy.maximum_relative_eigenvector_imaginary_norm)
        & (relative_residual <= policy.maximum_relative_residual)
        & (mass_symmetry_error <= policy.maximum_mass_symmetry_error)
        & (phase_anchor_relative_magnitude >= policy.minimum_phase_anchor_relative_magnitude)
    )
    invalid_scalar = jnp.asarray(jnp.nan, dtype=jnp.float64)
    invalid_edge = jnp.full_like(normalized_edge, jnp.nan)
    pair = SimplePortEigenpair(
        dimensionless_eigenvalue=jnp.where(is_valid, real_eigenvalue, invalid_scalar),
        edge_coefficients=jnp.where(is_valid, normalized_edge, invalid_edge),
    )
    diagnostics = SimplePortEigenpairDiagnostics(
        relative_eigenvalue_gap=jax.lax.stop_gradient(relative_gap),
        relative_eigenvalue_imaginary_part=jax.lax.stop_gradient(eigenvalue_imaginary_part),
        relative_eigenvector_imaginary_norm=jax.lax.stop_gradient(eigenvector_imaginary_norm),
        relative_residual=jax.lax.stop_gradient(relative_residual),
        mass_symmetry_error=jax.lax.stop_gradient(mass_symmetry_error),
        phase_anchor_relative_magnitude=jax.lax.stop_gradient(phase_anchor_relative_magnitude),
        mass_norm_squared_before_normalization=jax.lax.stop_gradient(mass_norm_squared),
        is_valid=jax.lax.stop_gradient(is_valid),
    )
    return SimplePortEigenpairInspection(pair, diagnostics)


@partial(jax.custom_vjp, nondiff_argnums=tuple(range(2, 10)))
def _implicit_simple_dimensionless_eigenpair(
    dimensionless_stiffness: jax.Array,
    edge_mass: jax.Array,
    selected_mode_index: int,
    phase_anchor_edge_dof: int,
    minimum_relative_eigenvalue_gap: float,
    maximum_relative_eigenvalue_imaginary_part: float,
    maximum_relative_eigenvector_imaginary_norm: float,
    maximum_relative_residual: float,
    maximum_mass_symmetry_error: float,
    minimum_phase_anchor_relative_magnitude: float,
) -> SimplePortEigenpair:
    policy = SimplePortEigenpairPolicy(
        minimum_relative_eigenvalue_gap=minimum_relative_eigenvalue_gap,
        maximum_relative_eigenvalue_imaginary_part=(maximum_relative_eigenvalue_imaginary_part),
        maximum_relative_eigenvector_imaginary_norm=(maximum_relative_eigenvector_imaginary_norm),
        maximum_relative_residual=maximum_relative_residual,
        maximum_mass_symmetry_error=maximum_mass_symmetry_error,
        minimum_phase_anchor_relative_magnitude=minimum_phase_anchor_relative_magnitude,
    )
    return _solve_and_inspect_dimensionless(
        dimensionless_stiffness,
        edge_mass,
        selected_mode_index=selected_mode_index,
        phase_anchor_edge_dof=phase_anchor_edge_dof,
        policy=policy,
    ).eigenpair


def _implicit_simple_dimensionless_eigenpair_fwd(
    dimensionless_stiffness: jax.Array,
    edge_mass: jax.Array,
    selected_mode_index: int,
    phase_anchor_edge_dof: int,
    minimum_relative_eigenvalue_gap: float,
    maximum_relative_eigenvalue_imaginary_part: float,
    maximum_relative_eigenvector_imaginary_norm: float,
    maximum_relative_residual: float,
    maximum_mass_symmetry_error: float,
    minimum_phase_anchor_relative_magnitude: float,
) -> tuple[SimplePortEigenpair, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    pair = _implicit_simple_dimensionless_eigenpair(
        dimensionless_stiffness,
        edge_mass,
        selected_mode_index,
        phase_anchor_edge_dof,
        minimum_relative_eigenvalue_gap,
        maximum_relative_eigenvalue_imaginary_part,
        maximum_relative_eigenvector_imaginary_norm,
        maximum_relative_residual,
        maximum_mass_symmetry_error,
        minimum_phase_anchor_relative_magnitude,
    )
    return pair, (
        dimensionless_stiffness,
        edge_mass,
        pair.dimensionless_eigenvalue,
        pair.edge_coefficients,
    )


def _implicit_simple_dimensionless_eigenpair_bwd(
    selected_mode_index: int,
    phase_anchor_edge_dof: int,
    minimum_relative_eigenvalue_gap: float,
    maximum_relative_eigenvalue_imaginary_part: float,
    maximum_relative_eigenvector_imaginary_norm: float,
    maximum_relative_residual: float,
    maximum_mass_symmetry_error: float,
    minimum_phase_anchor_relative_magnitude: float,
    residual: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    cotangent: SimplePortEigenpair,
) -> tuple[jax.Array, jax.Array]:
    del (
        selected_mode_index,
        phase_anchor_edge_dof,
        minimum_relative_eigenvalue_gap,
        maximum_relative_eigenvalue_imaginary_part,
        maximum_relative_eigenvector_imaginary_norm,
        maximum_relative_residual,
        maximum_mass_symmetry_error,
        minimum_phase_anchor_relative_magnitude,
    )
    stiffness, mass, eigenvalue, edge = residual
    eigen_residual_jacobian = stiffness - eigenvalue * mass
    mass_action = mass @ edge
    bordered = jnp.block(
        [
            [eigen_residual_jacobian, -mass_action[:, None]],
            [(edge @ mass)[None, :], jnp.zeros((1, 1), dtype=stiffness.dtype)],
        ]
    )
    right_hand_side = jnp.concatenate(
        (
            jnp.asarray(cotangent.edge_coefficients, dtype=stiffness.dtype),
            jnp.reshape(
                jnp.asarray(cotangent.dimensionless_eigenvalue, dtype=stiffness.dtype),
                (1,),
            ),
        )
    )
    adjoint = jnp.linalg.solve(bordered.T, right_hand_side)
    eigenvector_adjoint = adjoint[:-1]
    normalization_adjoint = adjoint[-1]
    stiffness_cotangent = -jnp.outer(eigenvector_adjoint, edge)
    mass_cotangent = eigenvalue * jnp.outer(
        eigenvector_adjoint, edge
    ) - 0.5 * normalization_adjoint * jnp.outer(edge, edge)
    return stiffness_cotangent, mass_cotangent


_implicit_simple_dimensionless_eigenpair.defvjp(
    _implicit_simple_dimensionless_eigenpair_fwd,
    _implicit_simple_dimensionless_eigenpair_bwd,
)


def inspect_simple_port_eigenpair(
    condensed_stiffness: jax.Array,
    edge_mass: jax.Array,
    propagation_scale_per_m: jax.Array,
    *,
    selected_mode_index: int,
    phase_anchor_edge_dof: int,
    policy: SimplePortEigenpairPolicy = _DEFAULT_SIMPLE_PORT_EIGENPAIR_POLICY,
) -> SimplePortEigenpairInspection:
    """Return the selected primal pair and every simple-mode admission diagnostic."""

    scale = jnp.asarray(propagation_scale_per_m)
    if scale.shape:
        raise ValueError("port propagation scale must be a scalar")
    if scale.dtype != jnp.dtype(jnp.float64):
        raise TypeError("simple port eigen-adjoint requires a float64 propagation scale")
    stiffness = jnp.asarray(condensed_stiffness)
    mass = jnp.asarray(edge_mass)
    if stiffness.dtype != jnp.dtype(jnp.float64):
        raise TypeError("simple port eigen-adjoint requires float64 stiffness")
    if mass.dtype != jnp.dtype(jnp.float64):
        raise TypeError("simple port eigen-adjoint requires float64 edge mass")
    dimensionless_stiffness = stiffness / (scale * scale)
    return _solve_and_inspect_dimensionless(
        dimensionless_stiffness,
        mass,
        selected_mode_index=selected_mode_index,
        phase_anchor_edge_dof=phase_anchor_edge_dof,
        policy=policy,
    )


def solve_simple_port_eigenpair(
    condensed_stiffness: jax.Array,
    edge_mass: jax.Array,
    propagation_scale_per_m: jax.Array,
    *,
    selected_mode_index: int,
    phase_anchor_edge_dof: int,
    policy: SimplePortEigenpairPolicy = _DEFAULT_SIMPLE_PORT_EIGENPAIR_POLICY,
) -> PhysicalSimplePortEigenpair:
    r"""Return one residual-differentiated mode of ``S e = lambda B e``.

    The propagation scale is stopped because it is a conditioning choice, not a physical input.
    Reverse mode solves the transpose of the bordered residual Jacobian

    ``[[S-lambda B, -B e], [e.T B, 0]]``

    and never differentiates the dense eigensolver trace.
    """

    scale = jax.lax.stop_gradient(jnp.asarray(propagation_scale_per_m))
    if scale.shape:
        raise ValueError("port propagation scale must be a scalar")
    if scale.dtype != jnp.dtype(jnp.float64):
        raise TypeError("simple port eigen-adjoint requires a float64 propagation scale")
    stiffness = jnp.asarray(condensed_stiffness)
    mass = jnp.asarray(edge_mass)
    if stiffness.dtype != jnp.dtype(jnp.float64):
        raise TypeError("simple port eigen-adjoint requires float64 stiffness")
    if mass.dtype != jnp.dtype(jnp.float64):
        raise TypeError("simple port eigen-adjoint requires float64 edge mass")
    dimensionless_stiffness = stiffness / (scale * scale)
    pair = _implicit_simple_dimensionless_eigenpair(
        dimensionless_stiffness,
        mass,
        selected_mode_index,
        phase_anchor_edge_dof,
        policy.minimum_relative_eigenvalue_gap,
        policy.maximum_relative_eigenvalue_imaginary_part,
        policy.maximum_relative_eigenvector_imaginary_norm,
        policy.maximum_relative_residual,
        policy.maximum_mass_symmetry_error,
        policy.minimum_phase_anchor_relative_magnitude,
    )
    eigenvalue = pair.dimensionless_eigenvalue * scale * scale
    propagation_constant = jnp.sqrt(-eigenvalue)
    return PhysicalSimplePortEigenpair(
        eigenvalue_per_m2=eigenvalue,
        propagation_constant_per_m=propagation_constant,
        edge_coefficients=pair.edge_coefficients,
    )
