"""Implicit differentiation primitives owned by the native JAX backend."""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp


@jax.custom_vjp
def implicit_linear_solve(matrix: jax.Array, right_hand_side: jax.Array) -> jax.Array:
    r"""Solve ``A x = b`` with a residual-defined reverse derivative.

    This primitive is intentionally real-valued and two-dimensional for the current reference
    backend. Its reverse rule solves ``A.T lambda = x_bar`` and returns
    ``A_bar = -outer(lambda, x)`` and ``b_bar = lambda``. The derivative therefore depends on the
    converged linear system, not on the implementation trace of the primal solver.
    """

    return cast(jax.Array, jnp.linalg.solve(matrix, right_hand_side))


def _implicit_linear_solve_forward(
    matrix: jax.Array,
    right_hand_side: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    solution = cast(jax.Array, jnp.linalg.solve(matrix, right_hand_side))
    return solution, (matrix, solution)


def _implicit_linear_solve_backward(
    residual: tuple[jax.Array, jax.Array],
    solution_cotangent: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    matrix, solution = residual
    adjoint = jnp.linalg.solve(jnp.swapaxes(matrix, -1, -2), solution_cotangent)
    matrix_cotangent = -jnp.outer(adjoint, solution)
    return matrix_cotangent, adjoint


implicit_linear_solve.defvjp(
    _implicit_linear_solve_forward,
    _implicit_linear_solve_backward,
)
