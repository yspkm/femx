import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from femx.backends.jax.autodiff import implicit_linear_solve  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def test_implicit_linear_solve_vjp_matches_the_residual_adjoint_formula() -> None:
    matrix = jnp.asarray(((4.0, 1.0), (2.0, 3.0)), dtype=jnp.float64)
    right_hand_side = jnp.asarray((1.0, 5.0), dtype=jnp.float64)
    solution_cotangent = jnp.asarray((0.25, -0.75), dtype=jnp.float64)

    solution, pullback = jax.vjp(implicit_linear_solve, matrix, right_hand_side)
    matrix_cotangent, right_hand_side_cotangent = pullback(solution_cotangent)
    expected_solution = np.linalg.solve(np.asarray(matrix), np.asarray(right_hand_side))
    expected_adjoint = np.linalg.solve(
        np.asarray(matrix).T,
        np.asarray(solution_cotangent),
    )

    np.testing.assert_allclose(solution, expected_solution, rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_allclose(
        right_hand_side_cotangent,
        expected_adjoint,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        matrix_cotangent,
        -np.outer(expected_adjoint, expected_solution),
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_implicit_linear_solve_gradient_is_independent_of_primal_trace() -> None:
    def objective(parameter: jax.Array) -> jax.Array:
        matrix = jnp.asarray(((parameter[0], 0.5), (0.5, 2.0)), dtype=jnp.float64)
        right_hand_side = jnp.asarray((parameter[1], 1.0), dtype=jnp.float64)
        solution = implicit_linear_solve(matrix, right_hand_side)
        return jnp.dot(jnp.asarray((0.3, -0.2)), solution)

    parameters = jnp.asarray((3.0, 1.25), dtype=jnp.float64)
    gradient = jax.jit(jax.grad(objective))(parameters)
    step = 1.0e-6
    finite_difference = np.asarray(
        [
            (
                float(objective(parameters.at[index].add(step)))
                - float(objective(parameters.at[index].add(-step)))
            )
            / (2.0 * step)
            for index in range(2)
        ]
    )

    np.testing.assert_allclose(gradient, finite_difference, rtol=2.0e-9, atol=2.0e-11)
