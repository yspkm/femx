import itertools

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from femx.backends.jax.elements.triangle_nedelec import (  # noqa: E402
    REFERENCE_TRIANGLE_LOCAL_EDGES,
    evaluate_triangle_nedelec1,
    reference_triangle_nedelec1,
    triangle_nedelec1_local_gram,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def _canonical_signs(cells: np.ndarray) -> np.ndarray:
    local_edges = cells[:, REFERENCE_TRIANGLE_LOCAL_EDGES]
    return np.where(local_edges[:, :, 0] < local_edges[:, :, 1], 1, -1).astype(np.int8)


def test_reference_basis_has_kronecker_edge_moments_and_constant_curl() -> None:
    vertices = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    gauss = np.asarray((0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)))
    weights = np.asarray((0.5, 0.5))

    moments = []
    for start, end in REFERENCE_TRIANGLE_LOCAL_EDGES:
        tangent = vertices[end] - vertices[start]
        points = vertices[start] + gauss[:, None] * tangent
        basis, curl = jax.jit(reference_triangle_nedelec1)(jnp.asarray(points))
        moments.append(np.sum(weights[:, None] * np.einsum("qed,d->qe", basis, tangent), axis=0))
        np.testing.assert_allclose(curl, 2.0, rtol=0.0, atol=0.0)

    np.testing.assert_allclose(moments, np.eye(3), rtol=0.0, atol=2.0e-15)


def test_reference_basis_contains_the_discrete_gradient_exact_sequence() -> None:
    points = jnp.asarray(((0.2, 0.3), (0.1, 0.7), (1.0 / 3.0, 1.0 / 3.0)))
    basis, curl = reference_triangle_nedelec1(points)
    gradients = np.asarray(((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0)))

    for vertex in range(3):
        nodal_values = np.eye(3)[vertex]
        edge_coefficients = np.asarray(
            [
                nodal_values[end] - nodal_values[start]
                for start, end in REFERENCE_TRIANGLE_LOCAL_EDGES
            ]
        )
        reconstructed = np.einsum("e,qed->qd", edge_coefficients, basis)
        reconstructed_curl = np.einsum("e,qe->q", edge_coefficients, curl)
        np.testing.assert_allclose(
            reconstructed,
            np.broadcast_to(gradients[vertex], reconstructed.shape),
            rtol=0.0,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(reconstructed_curl, 0.0, rtol=0.0, atol=0.0)


def test_covariant_piola_matches_independent_physical_barycentric_construction() -> None:
    coordinates = np.asarray(((0.2, -0.1), (2.0, 0.4), (-0.3, 1.7)))
    cells = np.asarray(((2, 0, 1),), dtype=np.int32)
    signs = _canonical_signs(cells)
    reference_points = np.asarray(((0.2, 0.3), (0.6, 0.1)))

    evaluated = evaluate_triangle_nedelec1(
        jnp.asarray(coordinates),
        jnp.asarray(cells),
        jnp.asarray(signs),
        jnp.asarray(reference_points),
    )

    physical_vertices = coordinates[cells[0]]
    jacobian = np.column_stack(
        (physical_vertices[1] - physical_vertices[0], physical_vertices[2] - physical_vertices[0])
    )
    determinant = np.linalg.det(jacobian)
    reference_gradients = np.asarray(((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0)))
    physical_gradients = reference_gradients @ np.linalg.inv(jacobian)
    barycentric = np.column_stack(
        (
            1.0 - reference_points[:, 0] - reference_points[:, 1],
            reference_points[:, 0],
            reference_points[:, 1],
        )
    )
    direct_basis = np.empty((reference_points.shape[0], 3, 2))
    for edge, (start, end) in enumerate(REFERENCE_TRIANGLE_LOCAL_EDGES):
        direct_basis[:, edge, :] = signs[0, edge] * (
            barycentric[:, start, None] * physical_gradients[end]
            - barycentric[:, end, None] * physical_gradients[start]
        )

    np.testing.assert_allclose(evaluated.jacobians[0], jacobian, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(evaluated.determinants, determinant, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(evaluated.basis[0], direct_basis, rtol=2.0e-15, atol=2.0e-15)
    np.testing.assert_allclose(
        evaluated.curl[0],
        np.broadcast_to(2.0 * signs[0][None, :] / determinant, evaluated.curl[0].shape),
        rtol=2.0e-15,
        atol=2.0e-15,
    )


def test_piola_mapping_reproduces_locked_elmer_equilateral_reference_basis() -> None:
    # Elmer 4f2d7e4 ElemInfo.F90: first-family, BasisDegree=1, triangle CASE(3).
    root_three = np.sqrt(3.0)
    coordinates = jnp.asarray(((-1.0, 0.0), (1.0, 0.0), (0.0, root_three)))
    cells = np.asarray(((0, 1, 2),), dtype=np.int32)
    signs = _canonical_signs(cells)
    reference_points = np.asarray(((0.2, 0.3), (0.6, 0.1)))
    elmer_points = np.column_stack(
        (
            -1.0 + 2.0 * reference_points[:, 0] + reference_points[:, 1],
            root_three * reference_points[:, 1],
        )
    )
    u = elmer_points[:, 0]
    v = elmer_points[:, 1]
    elmer_local_basis = np.stack(
        (
            np.column_stack(((3.0 - root_three * v) / 6.0, u / (2.0 * root_three))),
            np.column_stack((-v / (2.0 * root_three), (1.0 + u) / (2.0 * root_three))),
            np.column_stack((-v / (2.0 * root_three), (-1.0 + u) / (2.0 * root_three))),
        ),
        axis=1,
    )

    evaluated = evaluate_triangle_nedelec1(
        coordinates,
        jnp.asarray(cells),
        jnp.asarray(signs),
        jnp.asarray(reference_points),
    )

    np.testing.assert_allclose(
        evaluated.basis[0],
        elmer_local_basis * signs[0][None, :, None],
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        evaluated.curl[0],
        np.broadcast_to(signs[0] / root_three, evaluated.curl[0].shape),
        rtol=0.0,
        atol=2.0e-15,
    )


def test_canonical_orientation_makes_shared_tangential_trace_single_valued() -> None:
    coordinates = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    first_cell = np.asarray(((0, 1, 2),), dtype=np.int32)
    second_cell = np.asarray(((0, 2, 3),), dtype=np.int32)
    parameters = jnp.asarray((0.2, 0.5, 0.8))

    first = evaluate_triangle_nedelec1(
        coordinates,
        jnp.asarray(first_cell),
        jnp.asarray(_canonical_signs(first_cell)),
        jnp.stack((jnp.zeros_like(parameters), parameters), axis=1),
    )
    second = evaluate_triangle_nedelec1(
        coordinates,
        jnp.asarray(second_cell),
        jnp.asarray(_canonical_signs(second_cell)),
        jnp.stack((parameters, jnp.zeros_like(parameters)), axis=1),
    )
    global_tangent = np.asarray((1.0, 1.0)) / np.sqrt(2.0)
    first_trace = np.einsum("qd,d->q", first.basis[0, :, 2, :], global_tangent)
    second_trace = np.einsum("qd,d->q", second.basis[0, :, 0, :], global_tangent)

    np.testing.assert_allclose(first_trace, second_trace, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(first_trace, 1.0 / np.sqrt(2.0), rtol=0.0, atol=2.0e-15)


def test_reference_local_gram_is_exact_symmetric_and_positive() -> None:
    coordinates = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    cells = np.asarray(((0, 1, 2),), dtype=np.int32)
    signs = _canonical_signs(cells)[0]
    gram = jax.jit(triangle_nedelec1_local_gram)(
        coordinates,
        jnp.asarray(cells),
        jnp.asarray((signs,)),
    )

    raw_mass = np.asarray(
        ((1.0 / 3.0, 0.0, -1.0 / 6.0), (0.0, 1.0 / 6.0, 0.0), (-1.0 / 6.0, 0.0, 1.0 / 3.0))
    )
    orientation = np.diag(signs)
    expected_mass = orientation @ raw_mass @ orientation
    expected_curl_curl = 2.0 * np.outer(signs, signs)

    np.testing.assert_allclose(gram.mass[0], expected_mass, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(gram.curl_curl[0], expected_curl_curl, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(gram.mass[0], gram.mass[0].T, rtol=0.0, atol=0.0)
    assert np.all(np.linalg.eigvalsh(np.asarray(gram.mass[0])) > 0.0)
    assert np.linalg.matrix_rank(np.asarray(gram.curl_curl[0]), tol=1.0e-13) == 1
    np.testing.assert_allclose(gram.determinants, 1.0, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("permutation", tuple(itertools.permutations(range(3))))
def test_local_gram_remains_finite_for_every_node_permutation(
    permutation: tuple[int, int, int],
) -> None:
    coordinates = jnp.asarray(((0.0, 0.0), (1.2, 0.1), (0.2, 0.9)))
    cells = np.asarray((permutation,), dtype=np.int32)
    gram = triangle_nedelec1_local_gram(
        coordinates,
        jnp.asarray(cells),
        jnp.asarray(_canonical_signs(cells)),
    )

    assert np.isfinite(np.asarray(gram.mass)).all()
    assert np.isfinite(np.asarray(gram.curl_curl)).all()
    assert np.sign(float(gram.determinants[0])) == np.sign(
        np.linalg.det(
            np.column_stack(
                (
                    np.asarray(coordinates[permutation[1]] - coordinates[permutation[0]]),
                    np.asarray(coordinates[permutation[2]] - coordinates[permutation[0]]),
                )
            )
        )
    )


def test_local_gram_reverse_mode_matches_central_difference_and_translation_invariance() -> None:
    cells = jnp.asarray(((0, 1, 2),), dtype=jnp.int32)
    signs = jnp.asarray(((1, 1, -1),), dtype=jnp.int8)
    coordinates = jnp.asarray(((0.0, 0.0), (1.2, 0.1), (0.2, 0.9)))

    def objective(points: jax.Array) -> jax.Array:
        gram = triangle_nedelec1_local_gram(points, cells, signs)
        return gram.mass[0, 0, 0] + 0.1 * gram.curl_curl[0, 1, 1]

    gradient = jax.jit(jax.grad(objective))(coordinates)
    step = 1.0e-6
    direction = np.asarray(((0.3, -0.2), (-0.1, 0.4), (0.2, -0.3)))
    plus = objective(coordinates + step * direction)
    minus = objective(coordinates - step * direction)
    central = float((plus - minus) / (2.0 * step))
    reverse = float(jnp.vdot(gradient, jnp.asarray(direction)))

    np.testing.assert_allclose(reverse, central, rtol=2.0e-9, atol=2.0e-10)
    np.testing.assert_allclose(jnp.sum(gradient, axis=0), 0.0, rtol=0.0, atol=2.0e-14)
