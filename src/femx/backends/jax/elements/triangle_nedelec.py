r"""Lowest-order first-family Nédélec element on affine triangles.

The reference triangle is ``(0, 0)``, ``(1, 0)``, ``(0, 1)`` with directed local edges
``(0, 1)``, ``(1, 2)``, ``(2, 0)``.  Its basis is the Whitney form

.. math:: N_{ij} = \lambda_i \nabla\lambda_j - \lambda_j \nabla\lambda_i.

The physical basis uses the covariant Piola map ``J^{-T}``; its scalar two-dimensional curl uses
``1 / det(J)``.  Cell signs align local edge moments with femx's smaller-to-larger global-node
orientation.  Numerical preparation must validate nondegeneracy and those signs before calling
these JAX-transformable kernels.
"""

from __future__ import annotations

from typing import Final, NamedTuple

import jax
import jax.numpy as jnp

REFERENCE_TRIANGLE_LOCAL_EDGES: Final = ((0, 1), (1, 2), (2, 0))
_BARYCENTRIC_GRADIENTS: Final = ((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0))
_DEGREE_TWO_POINTS: Final = ((1.0 / 6.0, 1.0 / 6.0), (2.0 / 3.0, 1.0 / 6.0), (1.0 / 6.0, 2.0 / 3.0))
_DEGREE_TWO_WEIGHTS: Final = (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0)


class TriangleNedelec1Evaluation(NamedTuple):
    """Oriented physical basis data at reference evaluation points."""

    basis: jax.Array
    curl: jax.Array
    jacobians: jax.Array
    determinants: jax.Array


class TriangleNedelec1Gram(NamedTuple):
    """Exact affine-cell unit mass and curl-curl Gram matrices."""

    mass: jax.Array
    curl_curl: jax.Array
    determinants: jax.Array


def reference_triangle_nedelec1(
    reference_points: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate the three locally directed reference bases and their scalar curls.

    Returns arrays shaped ``(points, 3, 2)`` and ``(points, 3)``.  The edge moments are one on
    their own directed edge and zero on the other two; every un-oriented reference curl is two.
    """

    r = reference_points[:, 0]
    s = reference_points[:, 1]
    barycentric = jnp.stack((1.0 - r - s, r, s), axis=1)
    gradients = jnp.asarray(_BARYCENTRIC_GRADIENTS, dtype=reference_points.dtype)
    starts = jnp.asarray((0, 1, 2), dtype=jnp.int32)
    ends = jnp.asarray((1, 2, 0), dtype=jnp.int32)
    basis = (
        barycentric[:, starts, None] * gradients[ends][None, :, :]
        - barycentric[:, ends, None] * gradients[starts][None, :, :]
    )
    curl = jnp.full((reference_points.shape[0], 3), 2.0, dtype=reference_points.dtype)
    return basis, curl


def evaluate_triangle_nedelec1(
    coordinates: jax.Array,
    cells: jax.Array,
    cell_edge_signs: jax.Array,
    reference_points: jax.Array,
) -> TriangleNedelec1Evaluation:
    """Apply covariant Piola and canonical edge signs on every affine triangle."""

    points = coordinates[cells]
    first = points[:, 1, :] - points[:, 0, :]
    second = points[:, 2, :] - points[:, 0, :]
    jacobians = jnp.stack((first, second), axis=2)
    determinants = jacobians[:, 0, 0] * jacobians[:, 1, 1] - jacobians[:, 0, 1] * jacobians[:, 1, 0]
    inverse_transpose = (
        jnp.stack(
            (
                jacobians[:, 1, 1],
                -jacobians[:, 1, 0],
                -jacobians[:, 0, 1],
                jacobians[:, 0, 0],
            ),
            axis=1,
        ).reshape((-1, 2, 2))
        / determinants[:, None, None]
    )

    reference_basis, reference_curl = reference_triangle_nedelec1(reference_points)
    signs = cell_edge_signs.astype(coordinates.dtype)
    physical_basis = jnp.einsum(
        "cij,qej->cqei",
        inverse_transpose,
        reference_basis,
    )
    physical_basis = physical_basis * signs[:, None, :, None]
    physical_curl = reference_curl[None, :, :] / determinants[:, None, None] * signs[:, None, :]
    return TriangleNedelec1Evaluation(
        basis=physical_basis,
        curl=physical_curl,
        jacobians=jacobians,
        determinants=determinants,
    )


def triangle_nedelec1_local_gram(
    coordinates: jax.Array,
    cells: jax.Array,
    cell_edge_signs: jax.Array,
) -> TriangleNedelec1Gram:
    """Integrate unit ``L2`` and scalar-curl Gram matrices exactly on affine cells."""

    quadrature_points = jnp.asarray(_DEGREE_TWO_POINTS, dtype=coordinates.dtype)
    quadrature_weights = jnp.asarray(_DEGREE_TWO_WEIGHTS, dtype=coordinates.dtype)
    evaluation = evaluate_triangle_nedelec1(
        coordinates,
        cells,
        cell_edge_signs,
        quadrature_points,
    )
    measure = jnp.abs(evaluation.determinants)
    mass = measure[:, None, None] * jnp.einsum(
        "q,cqia,cqja->cij",
        quadrature_weights,
        evaluation.basis,
        evaluation.basis,
    )
    curl_curl = measure[:, None, None] * jnp.einsum(
        "q,cqi,cqj->cij",
        quadrature_weights,
        evaluation.curl,
        evaluation.curl,
    )
    return TriangleNedelec1Gram(
        mass=mass,
        curl_curl=curl_curl,
        determinants=evaluation.determinants,
    )
