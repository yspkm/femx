r"""Basis-invariant reverse mode for an isolated lossless port-mode cluster.

The condensed Elmer-compatible pencil is generally nonsymmetric.  This module therefore uses a
fixed Riesz contour around an isolated cluster and solves generalized shifted residuals,

``(z B - S_hat) X(z) = B C``,

instead of differentiating individual eigenvectors.  The contour moments define an invariant sum
of propagation constants and a B-orthogonal projector onto the right invariant subspace.  JAX
reverse mode differentiates the shifted linear solves, so every reverse step is an adjoint solve of
the same residual family.

This remains a dense serial float64/complex128 reference kernel.  The contour, its expected rank,
and both probe matrices are fixed baseline data.  A changed eigenvalue count, an insufficiently
clear contour, poor quadrature convergence, a singular probe moment, or a non-real lossless result
fails closed with non-finite outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, NamedTuple, cast

import jax
import jax.numpy as jnp

_FLOAT64_TINY: Final = 2.225_073_858_507_201_4e-308


@dataclass(frozen=True, slots=True)
class PortClusterContour:
    """Fixed contour and admission thresholds for one isolated cluster."""

    center: float
    radius: float
    expected_cluster_size: int
    quadrature_point_count: int = 32
    minimum_relative_contour_clearance: float = 5.0e-2
    maximum_relative_quadrature_error: float = 1.0e-6
    maximum_relative_shifted_residual: float = 1.0e-10
    maximum_mass_symmetry_error: float = 1.0e-12
    maximum_relative_eigenvalue_imaginary_part: float = 1.0e-10
    maximum_relative_moment_imaginary_part: float = 1.0e-10
    maximum_relative_projected_imaginary_norm: float = 1.0e-10
    minimum_probe_moment_singular_value_ratio: float = 1.0e-8
    maximum_projector_idempotency_error: float = 1.0e-8
    maximum_projector_mass_self_adjoint_error: float = 1.0e-8

    def __post_init__(self) -> None:
        if not math.isfinite(self.center):
            raise ValueError("port cluster contour center must be finite")
        if not math.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("port cluster contour radius must be finite and positive")
        if self.center + self.radius >= 0.0:
            raise ValueError("port cluster contour must remain on the negative real half-plane")
        if (
            isinstance(self.expected_cluster_size, bool)
            or not isinstance(self.expected_cluster_size, int)
            or self.expected_cluster_size <= 0
        ):
            raise ValueError("port cluster expected size must be a positive integer")
        if (
            isinstance(self.quadrature_point_count, bool)
            or not isinstance(self.quadrature_point_count, int)
            or self.quadrature_point_count < 8
            or self.quadrature_point_count % 2 != 0
        ):
            raise ValueError("port cluster quadrature count must be an even integer of at least 8")
        thresholds = (
            ("minimum contour clearance", self.minimum_relative_contour_clearance),
            ("maximum quadrature error", self.maximum_relative_quadrature_error),
            ("maximum shifted residual", self.maximum_relative_shifted_residual),
            ("maximum mass symmetry error", self.maximum_mass_symmetry_error),
            (
                "maximum eigenvalue imaginary part",
                self.maximum_relative_eigenvalue_imaginary_part,
            ),
            ("maximum moment imaginary part", self.maximum_relative_moment_imaginary_part),
            (
                "maximum projected imaginary norm",
                self.maximum_relative_projected_imaginary_norm,
            ),
            (
                "minimum probe singular-value ratio",
                self.minimum_probe_moment_singular_value_ratio,
            ),
            ("maximum projector idempotency error", self.maximum_projector_idempotency_error),
            (
                "maximum projector mass-self-adjoint error",
                self.maximum_projector_mass_self_adjoint_error,
            ),
        )
        for label, value in thresholds:
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"port cluster {label} must be finite and positive")


class InvariantPortCluster(NamedTuple):
    """Basis-invariant physical observables for one admitted cluster."""

    dimensionless_eigenvalue_sum: jax.Array
    eigenvalue_sum_per_m2: jax.Array
    propagation_constant_sum_per_m: jax.Array
    mean_propagation_constant_per_m: jax.Array
    reduced_edge_mass_projector: jax.Array


class PortClusterDiagnostics(NamedTuple):
    """Stopped-gradient evidence admitting one contour result."""

    observed_cluster_size: jax.Array
    relative_contour_clearance: jax.Array
    relative_cluster_eigenvalue_imaginary_part: jax.Array
    relative_moment_imaginary_part: jax.Array
    relative_projected_imaginary_norm: jax.Array
    relative_quadrature_error: jax.Array
    maximum_relative_shifted_residual: jax.Array
    mass_symmetry_error: jax.Array
    probe_moment_singular_value_ratio: jax.Array
    projector_idempotency_error: jax.Array
    projector_mass_self_adjoint_error: jax.Array
    is_valid: jax.Array


class InvariantPortClusterInspection(NamedTuple):
    """Cluster observables paired with their admission evidence."""

    cluster: InvariantPortCluster
    diagnostics: PortClusterDiagnostics


class _ContourMoments(NamedTuple):
    projected_probe: jax.Array
    eigenvalue_moment: jax.Array
    propagation_moment: jax.Array
    coarse_projected_probe: jax.Array
    coarse_eigenvalue_moment: jax.Array
    coarse_propagation_moment: jax.Array
    maximum_relative_shifted_residual: jax.Array


def _validate_layout(
    dimensionless_stiffness: jax.Array,
    edge_mass: jax.Array,
    right_probe: jax.Array,
    left_probe: jax.Array,
    contour: PortClusterContour,
) -> None:
    if (
        dimensionless_stiffness.ndim != 2
        or dimensionless_stiffness.shape[0] != dimensionless_stiffness.shape[1]
    ):
        raise ValueError("dimensionless port stiffness must be a square rank-two array")
    if edge_mass.shape != dimensionless_stiffness.shape:
        raise ValueError("port edge mass must match the dimensionless stiffness shape")
    expected_probe_shape = (
        dimensionless_stiffness.shape[0],
        contour.expected_cluster_size,
    )
    if right_probe.shape != expected_probe_shape:
        raise ValueError(f"right cluster probe must have shape {expected_probe_shape}")
    if left_probe.shape != expected_probe_shape:
        raise ValueError(f"left cluster probe must have shape {expected_probe_shape}")


def _relative_error(candidate: jax.Array, reference: jax.Array) -> jax.Array:
    numerator = jnp.linalg.norm(candidate - reference)
    denominator = jnp.maximum(jnp.linalg.norm(reference), _FLOAT64_TINY)
    return cast(jax.Array, numerator / denominator)


def _contour_moments(
    dimensionless_stiffness: jax.Array,
    edge_mass: jax.Array,
    right_probe: jax.Array,
    contour: PortClusterContour,
) -> _ContourMoments:
    complex_dtype = jnp.dtype(jnp.complex128)
    stiffness = dimensionless_stiffness.astype(complex_dtype)
    mass = edge_mass.astype(complex_dtype)
    probe = right_probe.astype(complex_dtype)
    right_hand_side = mass @ probe
    point_count = contour.quadrature_point_count
    angles = (2.0 * jnp.pi / point_count) * jnp.arange(point_count, dtype=jnp.float64)
    radial = contour.radius * jnp.exp(1j * angles)
    points = contour.center + radial
    weights = radial / point_count
    zero_probe = jnp.zeros_like(probe)
    zero_scalar = jnp.asarray(0.0, dtype=jnp.float64)

    def accumulate(
        carry: tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
        inputs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[
        tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
        None,
    ]:
        (
            projected,
            eigenvalue,
            propagation,
            coarse_projected,
            coarse_eigenvalue,
            coarse_propagation,
            maximum_residual,
        ) = carry
        index, point, weight = inputs
        shifted = point * mass - stiffness
        solution = jnp.linalg.solve(shifted, right_hand_side)
        residual = shifted @ solution - right_hand_side
        residual_denominator = jnp.linalg.norm(shifted) * jnp.linalg.norm(
            solution
        ) + jnp.linalg.norm(right_hand_side)
        relative_residual = jnp.linalg.norm(residual) / jnp.maximum(
            residual_denominator,
            _FLOAT64_TINY,
        )
        weighted = weight * solution
        beta = jnp.sqrt(-point)
        coarse_weighted = jnp.where(index % 2 == 0, 2.0 * weighted, zero_probe)
        return (
            (
                projected + weighted,
                eigenvalue + point * weighted,
                propagation + beta * weighted,
                coarse_projected + coarse_weighted,
                coarse_eigenvalue + point * coarse_weighted,
                coarse_propagation + beta * coarse_weighted,
                jnp.maximum(maximum_residual, relative_residual),
            ),
            None,
        )

    initial = (
        zero_probe,
        zero_probe,
        zero_probe,
        zero_probe,
        zero_probe,
        zero_probe,
        zero_scalar,
    )
    indices = jnp.arange(point_count, dtype=jnp.int32)
    final, _ = jax.lax.scan(accumulate, initial, (indices, points, weights))
    return _ContourMoments(*final)


def _small_trace(zero_moment: jax.Array, weighted_moment: jax.Array) -> jax.Array:
    return jnp.trace(jnp.linalg.solve(zero_moment, weighted_moment))


def _mass_orthogonal_projector(
    projected_probe: jax.Array,
    edge_mass: jax.Array,
) -> jax.Array:
    gram = projected_probe.T @ edge_mass @ projected_probe
    gram = 0.5 * (gram + gram.T)
    lower = jnp.linalg.cholesky(gram)
    orthonormal_basis = jnp.linalg.solve(lower, projected_probe.T).T
    return cast(jax.Array, orthonormal_basis @ orthonormal_basis.T @ edge_mass)


def _solve_and_inspect_dimensionless(
    dimensionless_stiffness: jax.Array,
    edge_mass: jax.Array,
    right_probe: jax.Array,
    left_probe: jax.Array,
    *,
    contour: PortClusterContour,
) -> InvariantPortClusterInspection:
    _validate_layout(
        dimensionless_stiffness,
        edge_mass,
        right_probe,
        left_probe,
        contour,
    )
    moments = _contour_moments(
        dimensionless_stiffness,
        edge_mass,
        right_probe,
        contour,
    )
    left = left_probe.astype(jnp.complex128)
    zero_moment = jnp.conj(left.T) @ moments.projected_probe
    eigenvalue_moment = jnp.conj(left.T) @ moments.eigenvalue_moment
    propagation_moment = jnp.conj(left.T) @ moments.propagation_moment
    coarse_zero_moment = jnp.conj(left.T) @ moments.coarse_projected_probe
    coarse_eigenvalue_moment = jnp.conj(left.T) @ moments.coarse_eigenvalue_moment
    coarse_propagation_moment = jnp.conj(left.T) @ moments.coarse_propagation_moment

    eigenvalue_sum_complex = _small_trace(zero_moment, eigenvalue_moment)
    propagation_sum_complex = _small_trace(zero_moment, propagation_moment)
    coarse_eigenvalue_sum = _small_trace(coarse_zero_moment, coarse_eigenvalue_moment)
    coarse_propagation_sum = _small_trace(coarse_zero_moment, coarse_propagation_moment)
    projected_probe = jnp.real(moments.projected_probe)
    projector = _mass_orthogonal_projector(projected_probe, edge_mass)

    standard_operator = jnp.linalg.solve(
        jax.lax.stop_gradient(edge_mass),
        jax.lax.stop_gradient(dimensionless_stiffness),
    )
    spectrum = jnp.linalg.eigvals(standard_operator)
    contour_distances = jnp.abs(spectrum - contour.center)
    inside = contour_distances < contour.radius
    observed_cluster_size = jnp.sum(inside, dtype=jnp.int32)
    relative_contour_clearance = (
        jnp.min(jnp.abs(contour_distances - contour.radius)) / contour.radius
    )
    inside_imaginary = jnp.where(inside, jnp.abs(jnp.imag(spectrum)), 0.0)
    inside_scale = jnp.where(inside, jnp.maximum(jnp.abs(spectrum), 1.0), 1.0)
    relative_cluster_imaginary = jnp.max(inside_imaginary / inside_scale)
    moment_imaginary = jnp.maximum(
        jnp.abs(jnp.imag(eigenvalue_sum_complex))
        / jnp.maximum(jnp.abs(eigenvalue_sum_complex), 1.0),
        jnp.abs(jnp.imag(propagation_sum_complex))
        / jnp.maximum(jnp.abs(propagation_sum_complex), 1.0),
    )
    projected_imaginary = jnp.linalg.norm(jnp.imag(moments.projected_probe)) / jnp.maximum(
        jnp.linalg.norm(moments.projected_probe),
        _FLOAT64_TINY,
    )
    quadrature_error = jnp.maximum(
        _relative_error(moments.coarse_projected_probe, moments.projected_probe),
        jnp.maximum(
            _relative_error(coarse_eigenvalue_sum, eigenvalue_sum_complex),
            _relative_error(coarse_propagation_sum, propagation_sum_complex),
        ),
    )
    mass_symmetry_error = jnp.linalg.norm(edge_mass - edge_mass.T) / jnp.maximum(
        jnp.linalg.norm(edge_mass),
        _FLOAT64_TINY,
    )
    moment_singular_values = jnp.linalg.svd(zero_moment, compute_uv=False)
    probe_ratio = jnp.min(moment_singular_values) / jnp.maximum(
        jnp.max(moment_singular_values),
        _FLOAT64_TINY,
    )
    projector_idempotency = jnp.linalg.norm(projector @ projector - projector) / jnp.maximum(
        jnp.linalg.norm(projector),
        _FLOAT64_TINY,
    )
    projector_mass_self_adjoint = jnp.linalg.norm(
        projector.T @ edge_mass - edge_mass @ projector
    ) / jnp.maximum(
        jnp.linalg.norm(edge_mass @ projector),
        _FLOAT64_TINY,
    )
    mass_cholesky = jnp.linalg.cholesky(0.5 * (edge_mass + edge_mass.T))
    finite_inputs = (
        jnp.all(jnp.isfinite(dimensionless_stiffness))
        & jnp.all(jnp.isfinite(edge_mass))
        & jnp.all(jnp.isfinite(right_probe))
        & jnp.all(jnp.isfinite(left_probe))
    )
    finite_outputs = (
        jnp.isfinite(eigenvalue_sum_complex)
        & jnp.isfinite(propagation_sum_complex)
        & jnp.all(jnp.isfinite(projector))
        & jnp.all(jnp.isfinite(mass_cholesky))
        & jnp.all(jnp.isfinite(spectrum))
    )
    is_valid = (
        finite_inputs
        & finite_outputs
        & (observed_cluster_size == contour.expected_cluster_size)
        & (relative_contour_clearance >= contour.minimum_relative_contour_clearance)
        & (relative_cluster_imaginary <= contour.maximum_relative_eigenvalue_imaginary_part)
        & (moment_imaginary <= contour.maximum_relative_moment_imaginary_part)
        & (projected_imaginary <= contour.maximum_relative_projected_imaginary_norm)
        & (quadrature_error <= contour.maximum_relative_quadrature_error)
        & (moments.maximum_relative_shifted_residual <= contour.maximum_relative_shifted_residual)
        & (mass_symmetry_error <= contour.maximum_mass_symmetry_error)
        & (probe_ratio >= contour.minimum_probe_moment_singular_value_ratio)
        & (projector_idempotency <= contour.maximum_projector_idempotency_error)
        & (projector_mass_self_adjoint <= contour.maximum_projector_mass_self_adjoint_error)
    )
    dimensionless_eigenvalue_sum = jnp.real(eigenvalue_sum_complex)
    dimensionless_propagation_sum = jnp.real(propagation_sum_complex)

    def fail_closed(value: jax.Array) -> jax.Array:
        return cast(
            jax.Array,
            jax.lax.cond(
                is_valid,
                lambda admitted: admitted,
                lambda rejected: jnp.asarray(jnp.nan, dtype=rejected.dtype) * rejected,
                value,
            ),
        )

    cluster = InvariantPortCluster(
        dimensionless_eigenvalue_sum=fail_closed(dimensionless_eigenvalue_sum),
        eigenvalue_sum_per_m2=fail_closed(dimensionless_eigenvalue_sum),
        propagation_constant_sum_per_m=fail_closed(dimensionless_propagation_sum),
        mean_propagation_constant_per_m=fail_closed(
            dimensionless_propagation_sum / contour.expected_cluster_size
        ),
        reduced_edge_mass_projector=fail_closed(projector),
    )
    diagnostics = PortClusterDiagnostics(
        observed_cluster_size=jax.lax.stop_gradient(observed_cluster_size),
        relative_contour_clearance=jax.lax.stop_gradient(relative_contour_clearance),
        relative_cluster_eigenvalue_imaginary_part=jax.lax.stop_gradient(
            relative_cluster_imaginary
        ),
        relative_moment_imaginary_part=jax.lax.stop_gradient(moment_imaginary),
        relative_projected_imaginary_norm=jax.lax.stop_gradient(projected_imaginary),
        relative_quadrature_error=jax.lax.stop_gradient(quadrature_error),
        maximum_relative_shifted_residual=jax.lax.stop_gradient(
            moments.maximum_relative_shifted_residual
        ),
        mass_symmetry_error=jax.lax.stop_gradient(mass_symmetry_error),
        probe_moment_singular_value_ratio=jax.lax.stop_gradient(probe_ratio),
        projector_idempotency_error=jax.lax.stop_gradient(projector_idempotency),
        projector_mass_self_adjoint_error=jax.lax.stop_gradient(projector_mass_self_adjoint),
        is_valid=jax.lax.stop_gradient(is_valid),
    )
    return InvariantPortClusterInspection(cluster=cluster, diagnostics=diagnostics)


def inspect_invariant_port_cluster(
    condensed_stiffness: jax.Array,
    edge_mass: jax.Array,
    propagation_scale_per_m: jax.Array,
    right_probe: jax.Array,
    left_probe: jax.Array,
    *,
    contour: PortClusterContour,
) -> InvariantPortClusterInspection:
    """Return one physical cluster and all stopped admission diagnostics."""

    scale = jax.lax.stop_gradient(jnp.asarray(propagation_scale_per_m))
    if scale.shape:
        raise ValueError("port propagation scale must be a scalar")
    if scale.dtype != jnp.dtype(jnp.float64):
        raise TypeError("port cluster adjoint requires a float64 propagation scale")
    stiffness = jnp.asarray(condensed_stiffness)
    mass = jnp.asarray(edge_mass)
    if stiffness.dtype != jnp.dtype(jnp.float64):
        raise TypeError("port cluster adjoint requires float64 stiffness")
    if mass.dtype != jnp.dtype(jnp.float64):
        raise TypeError("port cluster adjoint requires float64 edge mass")
    right = jnp.asarray(right_probe)
    left = jnp.asarray(left_probe)
    if right.dtype not in (jnp.dtype(jnp.float64), jnp.dtype(jnp.complex128)):
        raise TypeError("right cluster probe requires float64 or complex128")
    if left.dtype not in (jnp.dtype(jnp.float64), jnp.dtype(jnp.complex128)):
        raise TypeError("left cluster probe requires float64 or complex128")
    inspection = _solve_and_inspect_dimensionless(
        stiffness / (scale * scale),
        mass,
        right,
        left,
        contour=contour,
    )
    cluster = inspection.cluster
    physical = InvariantPortCluster(
        dimensionless_eigenvalue_sum=cluster.dimensionless_eigenvalue_sum,
        eigenvalue_sum_per_m2=cluster.dimensionless_eigenvalue_sum * scale * scale,
        propagation_constant_sum_per_m=cluster.propagation_constant_sum_per_m * scale,
        mean_propagation_constant_per_m=cluster.mean_propagation_constant_per_m * scale,
        reduced_edge_mass_projector=cluster.reduced_edge_mass_projector,
    )
    return InvariantPortClusterInspection(physical, inspection.diagnostics)


def solve_invariant_port_cluster(
    condensed_stiffness: jax.Array,
    edge_mass: jax.Array,
    propagation_scale_per_m: jax.Array,
    right_probe: jax.Array,
    left_probe: jax.Array,
    *,
    contour: PortClusterContour,
) -> InvariantPortCluster:
    """Return differentiable basis-invariant observables for an isolated cluster."""

    return inspect_invariant_port_cluster(
        condensed_stiffness,
        edge_mass,
        propagation_scale_per_m,
        right_probe,
        left_probe,
        contour=contour,
    ).cluster
