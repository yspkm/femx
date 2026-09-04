"""Residual-defined distributed conjugate gradients for scalar H1/P1 diffusion.

The solve acts on owner-authoritative fixed-capacity shards.  Pairwise ``ppermute`` operations
apply the matrix, while every Krylov inner product is a global ``psum`` over active owner slots.
The public derivative is the converged linear residual through ``custom_linear_solve``; reverse
mode does not differentiate the iteration trace.  Final admission always recomputes the residual
and may explicitly use either its right-hand-side-relative norm or a representation-aware
componentwise normwise backward error.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from femx.core.errors import ContractError

from .collective import validate_collective_mesh
from .scalar_collective import (
    ScalarH1CollectiveLayout,
    build_packed_collective_scalar_h1_matvec,
    build_packed_collective_scalar_h1_rhs_assembly,
    pack_collective_scalar_h1_cell_matrix,
    pack_collective_scalar_h1_cell_vector,
    pack_collective_scalar_h1_owned_mask,
    unpack_collective_scalar_h1_owned_vector,
)


@dataclass(frozen=True, slots=True)
class ScalarH1CGPolicy:
    """Static convergence policy shared by primal and transpose residual solves."""

    relative_tolerance: float
    absolute_tolerance: float
    max_iterations: int
    backward_error_tolerance: float | None = None

    def __post_init__(self) -> None:
        for name in ("relative_tolerance", "absolute_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ContractError(f"scalar CG {name.replace('_', ' ')} must be a real scalar")
            canonical = float(value)
            if not math.isfinite(canonical):
                raise ContractError(f"scalar CG {name.replace('_', ' ')} must be finite")
            object.__setattr__(self, name, canonical)
        if self.relative_tolerance <= 0.0:
            raise ContractError("scalar CG relative tolerance must be positive")
        if self.absolute_tolerance < 0.0:
            raise ContractError("scalar CG absolute tolerance cannot be negative")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or self.max_iterations <= 0
        ):
            raise ContractError("scalar CG maximum iterations must be a positive integer")
        backward = self.backward_error_tolerance
        if backward is not None:
            if isinstance(backward, bool) or not isinstance(backward, (int, float)):
                raise ContractError("scalar CG backward-error tolerance must be a real scalar")
            canonical = float(backward)
            if not math.isfinite(canonical) or canonical <= 0.0:
                raise ContractError(
                    "scalar CG backward-error tolerance must be finite and positive"
                )
            object.__setattr__(self, "backward_error_tolerance", canonical)


@dataclass(frozen=True, slots=True)
class ScalarH1JacobiPolicy:
    """Fail-closed positive-diagonal policy for symmetric scalar PCG."""

    minimum_relative_diagonal: float = 1.0e-14

    def __post_init__(self) -> None:
        value = self.minimum_relative_diagonal
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError("scalar Jacobi minimum relative diagonal must be a real scalar")
        canonical = float(value)
        if not math.isfinite(canonical):
            raise ContractError("scalar Jacobi minimum relative diagonal must be finite")
        if not 0.0 < canonical < 1.0:
            raise ContractError("scalar Jacobi minimum relative diagonal must lie in (0, 1)")
        object.__setattr__(self, "minimum_relative_diagonal", canonical)


class PackedScalarH1CGResult(NamedTuple):
    """Packed solution and replicated convergence diagnostics from one collective solve."""

    solution: jax.Array
    right_hand_side: jax.Array
    iterations: jax.Array
    rhs_norm: jax.Array
    recursive_residual_norm: jax.Array
    recomputed_residual_norm: jax.Array
    relative_residual: jax.Array
    backward_error: jax.Array
    converged: jax.Array
    breakdown: jax.Array


class ScalarH1CGResult(NamedTuple):
    """Canonical small-problem view of a packed collective CG result."""

    solution: jax.Array
    right_hand_side: jax.Array
    iterations: jax.Array
    rhs_norm: jax.Array
    recursive_residual_norm: jax.Array
    recomputed_residual_norm: jax.Array
    relative_residual: jax.Array
    backward_error: jax.Array
    converged: jax.Array
    breakdown: jax.Array


class _CGState(NamedTuple):
    iteration: jax.Array
    solution: jax.Array
    residual: jax.Array
    direction: jax.Array
    residual_squared: jax.Array
    residual_preconditioned: jax.Array
    breakdown: jax.Array


PackedScalarH1Dot = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]
PackedScalarH1Preconditioner = Callable[[jax.Array], jax.Array]
PackedScalarH1PreconditionerFactory = Callable[..., PackedScalarH1Preconditioner]


def build_packed_scalar_h1_owner_dot(
    layout: ScalarH1CollectiveLayout,
    mesh: Mesh,
    *,
    axis_name: str,
) -> PackedScalarH1Dot:
    """Build an owner-only global dot with an explicit process-local mask input."""

    validate_collective_mesh(layout.transport, mesh, axis_name)
    expected_shape = (layout.partition_count, layout.owned_dof_capacity)
    owner_spec = P(axis_name, None)  # type: ignore[no-untyped-call]
    replicated = P()  # type: ignore[no-untyped-call]

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(owner_spec, owner_spec, owner_spec),
        out_specs=replicated,
        check_vma=True,
    )
    def mapped(first: jax.Array, second: jax.Array, mask: jax.Array) -> jax.Array:
        local_first = jnp.where(mask[0], first[0], 0.0)
        local_second = jnp.where(mask[0], second[0], 0.0)
        local = jnp.vdot(local_first, local_second).real
        return cast(jax.Array, lax.psum(local, axis_name))  # type: ignore[no-untyped-call]

    def apply(first: jax.Array, second: jax.Array, active: jax.Array) -> jax.Array:
        if active.ndim != 2 or active.shape != expected_shape or active.dtype != jnp.bool_:
            raise ContractError("scalar CG active-owner mask disagrees with the collective layout")
        return cast(jax.Array, mapped(first, second, active))

    return apply


_build_packed_scalar_h1_dot = build_packed_scalar_h1_owner_dot


def build_packed_scalar_h1_jacobi_preconditioner_factory(
    layout: ScalarH1CollectiveLayout,
    mesh: Mesh,
    policy: ScalarH1JacobiPolicy,
    *,
    axis_name: str = "partition",
) -> PackedScalarH1PreconditionerFactory:
    """Build a stopped positive-diagonal inverse for the exact packed operator.

    The diagonal is assembled with the same ghost-row reduction as the right-hand side.  Invalid,
    nonpositive, or excessively small active entries make the preconditioner return NaNs, which
    the enclosing PCG reports as a breakdown instead of silently changing solver strategy.
    """

    if not isinstance(policy, ScalarH1JacobiPolicy):
        raise ContractError("scalar Jacobi preconditioner requires a ScalarH1JacobiPolicy")
    validate_collective_mesh(layout.transport, mesh, axis_name)
    row_assembly = build_packed_collective_scalar_h1_rhs_assembly(
        layout,
        mesh,
        axis_name=axis_name,
    )

    def factory(
        packed_cell_stiffness: jax.Array,
        packed_cell_local_dofs: jax.Array,
        packed_owner_mask: jax.Array,
    ) -> PackedScalarH1Preconditioner:
        packed_diagonal = jnp.diagonal(packed_cell_stiffness, axis1=-2, axis2=-1)
        diagonal = row_assembly(packed_diagonal, packed_cell_local_dofs)
        active_diagonal = jnp.where(packed_owner_mask, diagonal, jnp.inf)
        scale = jnp.max(jnp.where(packed_owner_mask, jnp.abs(diagonal), 0.0))
        minimum_relative = jnp.where(
            scale > 0.0,
            jnp.min(active_diagonal) / scale,
            -jnp.inf,
        )
        valid = (
            jnp.all(jnp.isfinite(jnp.where(packed_owner_mask, diagonal, 0.0)))
            & jnp.isfinite(minimum_relative)
            & (minimum_relative >= policy.minimum_relative_diagonal)
        )
        safe_diagonal = jnp.where(packed_owner_mask & valid, diagonal, 1.0)
        inverse = jnp.where(packed_owner_mask, 1.0 / safe_diagonal, 0.0)
        inverse = lax.stop_gradient(inverse)
        valid = lax.stop_gradient(valid)

        def precondition(residual: jax.Array) -> jax.Array:
            candidate = jnp.where(packed_owner_mask, inverse * residual, 0.0)
            finite = jnp.all(jnp.isfinite(candidate))
            return jnp.where(valid & finite, candidate, jnp.asarray(jnp.nan, candidate.dtype))

        return precondition

    return factory


def _cg_iteration(
    operator: Callable[[jax.Array], jax.Array],
    right_hand_side: jax.Array,
    active: jax.Array,
    policy: ScalarH1CGPolicy,
    global_dot: PackedScalarH1Dot,
    preconditioner: PackedScalarH1Preconditioner | None = None,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    zero = jnp.zeros_like(right_hand_side)
    residual = jnp.where(active, right_hand_side, 0.0)
    residual_squared = global_dot(residual, residual, active)
    preconditioned = residual if preconditioner is None else preconditioner(residual)
    preconditioned = jnp.where(active, preconditioned, 0.0)
    residual_preconditioned = global_dot(residual, preconditioned, active)
    rhs_squared = residual_squared
    relative_target_squared = (policy.relative_tolerance**2) * rhs_squared
    absolute_target_squared = jnp.asarray(
        policy.absolute_tolerance**2,
        dtype=rhs_squared.dtype,
    )
    target_squared = jnp.maximum(relative_target_squared, absolute_target_squared)
    initial_breakdown = (
        ~jnp.isfinite(residual_squared)
        | (residual_squared < 0.0)
        | ~jnp.isfinite(residual_preconditioned)
        | (residual_preconditioned < 0.0)
        | ((residual_squared > target_squared) & (residual_preconditioned <= 0.0))
    )
    initial = _CGState(
        iteration=jnp.asarray(0, dtype=jnp.int32),
        solution=zero,
        residual=residual,
        direction=preconditioned,
        residual_squared=residual_squared,
        residual_preconditioned=residual_preconditioned,
        breakdown=initial_breakdown,
    )

    def condition(state: _CGState) -> jax.Array:
        return (
            (state.iteration < policy.max_iterations)
            & (state.residual_squared > target_squared)
            & ~state.breakdown
        )

    def body(state: _CGState) -> _CGState:
        action = operator(state.direction)
        curvature = global_dot(state.direction, action, active)
        valid_curvature = jnp.isfinite(curvature) & (curvature > 0.0)
        safe_curvature = jnp.where(valid_curvature, curvature, 1.0)
        alpha = jnp.where(
            valid_curvature,
            state.residual_preconditioned / safe_curvature,
            0.0,
        )
        candidate_solution = jnp.where(
            active,
            state.solution + alpha * state.direction,
            0.0,
        )
        candidate_residual = jnp.where(active, state.residual - alpha * action, 0.0)
        candidate_squared = global_dot(candidate_residual, candidate_residual, active)
        candidate_preconditioned = (
            candidate_residual if preconditioner is None else preconditioner(candidate_residual)
        )
        candidate_preconditioned = jnp.where(active, candidate_preconditioned, 0.0)
        candidate_residual_preconditioned = global_dot(
            candidate_residual,
            candidate_preconditioned,
            active,
        )
        needs_direction = candidate_squared > target_squared
        valid_residual = (
            jnp.isfinite(candidate_squared)
            & (candidate_squared >= 0.0)
            & jnp.isfinite(candidate_residual_preconditioned)
            & (candidate_residual_preconditioned >= 0.0)
            & (~needs_direction | (candidate_residual_preconditioned > 0.0))
        )
        valid = valid_curvature & valid_residual
        safe_previous = jnp.where(
            state.residual_preconditioned > 0.0,
            state.residual_preconditioned,
            1.0,
        )
        beta = jnp.where(valid, candidate_residual_preconditioned / safe_previous, 0.0)
        candidate_direction = jnp.where(
            active,
            candidate_preconditioned + beta * state.direction,
            0.0,
        )
        return _CGState(
            iteration=state.iteration + 1,
            solution=jnp.where(valid, candidate_solution, state.solution),
            residual=jnp.where(valid, candidate_residual, state.residual),
            direction=jnp.where(valid, candidate_direction, state.direction),
            residual_squared=jnp.where(valid, candidate_squared, state.residual_squared),
            residual_preconditioned=jnp.where(
                valid,
                candidate_residual_preconditioned,
                state.residual_preconditioned,
            ),
            breakdown=state.breakdown | ~valid,
        )

    final = lax.while_loop(condition, body, initial)
    recursive_norm = jnp.sqrt(jnp.maximum(final.residual_squared, 0.0))
    return final.solution, (
        final.iteration,
        recursive_norm,
        final.breakdown,
    )


PackedScalarH1CGSolve = Callable[..., PackedScalarH1CGResult]


def build_packed_collective_scalar_h1_cg(
    layout: ScalarH1CollectiveLayout,
    mesh: Mesh,
    policy: ScalarH1CGPolicy,
    *,
    axis_name: str = "partition",
    preconditioner_factory: PackedScalarH1PreconditionerFactory | None = None,
) -> PackedScalarH1CGSolve:
    """Build global CG or symmetric PCG on owner-authoritative packed arrays.

    A supplied factory is evaluated from the current operator arrays and must return a symmetric
    positive-definite inverse action.  It changes only the iterative strategy: the surrounding
    ``custom_linear_solve`` continues to differentiate the converged residual equation.
    """

    if not isinstance(policy, ScalarH1CGPolicy):
        raise ContractError("scalar collective CG requires a ScalarH1CGPolicy")
    if preconditioner_factory is not None and not callable(preconditioner_factory):
        raise ContractError("scalar collective PCG preconditioner factory must be callable")
    iteration_policy = ScalarH1CGPolicy(
        relative_tolerance=0.25 * policy.relative_tolerance,
        absolute_tolerance=0.25 * policy.absolute_tolerance,
        max_iterations=policy.max_iterations,
    )
    validate_collective_mesh(layout.transport, mesh, axis_name)
    partition_count = layout.partition_count
    cell_capacity = layout.cell_capacity
    cell_dof_count = layout.cell_dof_count
    owned_capacity = layout.owned_dof_capacity
    cell_shape = (partition_count, cell_capacity, cell_dof_count, cell_dof_count)
    map_shape = (partition_count, cell_capacity, cell_dof_count)
    owner_shape = (partition_count, owned_capacity)
    packed_matvec = build_packed_collective_scalar_h1_matvec(
        layout,
        mesh,
        axis_name=axis_name,
    )
    global_dot = build_packed_scalar_h1_owner_dot(
        layout,
        mesh,
        axis_name=axis_name,
    )

    def mapped_solve(
        packed_cell_stiffness: jax.Array,
        packed_cell_local_dofs: jax.Array,
        packed_owner_mask: jax.Array,
        packed_right_hand_side: jax.Array,
        *preconditioner_arguments: object,
    ) -> tuple[jax.Array, ...]:
        right_hand_side = jnp.where(packed_owner_mask, packed_right_hand_side, 0.0)
        if preconditioner_factory is None:
            if preconditioner_arguments:
                raise ContractError("unpreconditioned scalar CG cannot receive strategy arguments")
            preconditioner = None
        else:
            preconditioner = preconditioner_factory(
                packed_cell_stiffness,
                packed_cell_local_dofs,
                packed_owner_mask,
                *preconditioner_arguments,
            )

        def operator(vector: jax.Array) -> jax.Array:
            action = packed_matvec(
                packed_cell_stiffness,
                packed_cell_local_dofs,
                jnp.where(packed_owner_mask, vector, 0.0),
            )
            return jnp.where(packed_owner_mask, action, 0.0)

        def solve(
            linear_operator: Callable[[jax.Array], jax.Array],
            rhs: jax.Array,
        ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
            return _cg_iteration(
                linear_operator,
                rhs,
                packed_owner_mask,
                iteration_policy,
                global_dot,
                preconditioner,
            )

        raw_solution, auxiliary = jax.lax.custom_linear_solve(
            operator,
            right_hand_side,
            solve=solve,
            symmetric=True,
            has_aux=True,
        )
        iterations, recursive_norm, breakdown = auxiliary
        recomputed_residual = operator(raw_solution) - right_hand_side
        residual_squared = global_dot(
            recomputed_residual,
            recomputed_residual,
            packed_owner_mask,
        )
        rhs_squared = global_dot(right_hand_side, right_hand_side, packed_owner_mask)
        solution_squared = global_dot(raw_solution, raw_solution, packed_owner_mask)
        residual_norm = jnp.sqrt(jnp.maximum(residual_squared, 0.0))
        rhs_norm = jnp.sqrt(jnp.maximum(rhs_squared, 0.0))
        relative_residual = jnp.where(
            rhs_norm > 0.0,
            residual_norm / rhs_norm,
            jnp.where(residual_norm == 0.0, 0.0, jnp.inf),
        )
        absolute_action = packed_matvec(
            jnp.abs(packed_cell_stiffness),
            packed_cell_local_dofs,
            jnp.where(packed_owner_mask, jnp.abs(raw_solution), 0.0),
        )
        backward_denominator = jnp.where(
            packed_owner_mask,
            absolute_action + jnp.abs(right_hand_side),
            0.0,
        )
        backward_denominator_squared = global_dot(
            backward_denominator,
            backward_denominator,
            packed_owner_mask,
        )
        backward_denominator_norm = jnp.sqrt(jnp.maximum(backward_denominator_squared, 0.0))
        backward_error = jnp.where(
            backward_denominator_norm > 0.0,
            residual_norm / backward_denominator_norm,
            jnp.where(residual_norm == 0.0, 0.0, jnp.inf),
        )
        target = jnp.maximum(
            jnp.asarray(policy.absolute_tolerance, dtype=rhs_norm.dtype),
            jnp.asarray(policy.relative_tolerance, dtype=rhs_norm.dtype) * rhs_norm,
        )
        finite_solution = jnp.isfinite(solution_squared)
        if policy.backward_error_tolerance is None:
            admitted_residual = residual_norm <= target
        else:
            admitted_residual = backward_error <= policy.backward_error_tolerance
        converged = (
            ~breakdown
            & finite_solution
            & jnp.isfinite(residual_norm)
            & jnp.isfinite(backward_error)
            & admitted_residual
        )
        admitted = jnp.where(packed_owner_mask, raw_solution, 0.0)
        admitted = jnp.where(converged, admitted, jnp.asarray(jnp.nan, admitted.dtype))
        return (
            admitted,
            right_hand_side,
            lax.stop_gradient(iterations),
            lax.stop_gradient(rhs_norm),
            lax.stop_gradient(recursive_norm),
            lax.stop_gradient(residual_norm),
            lax.stop_gradient(relative_residual),
            lax.stop_gradient(backward_error),
            lax.stop_gradient(converged),
            lax.stop_gradient(breakdown),
        )

    def apply(
        packed_cell_stiffness: jax.Array,
        packed_cell_local_dofs: jax.Array,
        packed_owner_mask: jax.Array,
        packed_right_hand_side: jax.Array,
        *preconditioner_arguments: object,
    ) -> PackedScalarH1CGResult:
        if packed_cell_stiffness.ndim != 4 or packed_cell_stiffness.shape != cell_shape:
            raise ValueError("scalar CG packed cell stiffness does not match the layout")
        if packed_cell_local_dofs.ndim != 3 or packed_cell_local_dofs.shape != map_shape:
            raise ValueError("scalar CG packed cell map does not match the layout")
        if packed_owner_mask.ndim != 2 or packed_owner_mask.shape != owner_shape:
            raise ValueError("scalar CG packed owner mask does not match the layout")
        if packed_right_hand_side.ndim != 2 or packed_right_hand_side.shape != owner_shape:
            raise ValueError("scalar CG packed right-hand side does not match the layout")
        if not jnp.issubdtype(packed_cell_stiffness.dtype, jnp.floating):
            raise TypeError("scalar CG packed cell stiffness must use a real floating dtype")
        if not jnp.issubdtype(packed_cell_local_dofs.dtype, jnp.integer):
            raise TypeError("scalar CG packed cell map must use an integer dtype")
        if packed_owner_mask.dtype != jnp.bool_:
            raise TypeError("scalar CG packed owner mask must use a boolean dtype")
        if not jnp.issubdtype(packed_right_hand_side.dtype, jnp.floating):
            raise TypeError("scalar CG packed right-hand side must use a real floating dtype")
        values = mapped_solve(
            packed_cell_stiffness,
            packed_cell_local_dofs,
            packed_owner_mask,
            packed_right_hand_side,
            *preconditioner_arguments,
        )
        return PackedScalarH1CGResult(*values)

    return apply


ValidationScalarH1CGSolve = Callable[[jax.Array, jax.Array], ScalarH1CGResult]


def build_validation_collective_scalar_h1_cg(
    layout: ScalarH1CollectiveLayout,
    mesh: Mesh,
    policy: ScalarH1CGPolicy,
    *,
    axis_name: str = "partition",
) -> ValidationScalarH1CGSolve:
    """Build a canonical wrapper for small RHS/solve/adjoint validation only."""

    assemble_rhs = build_packed_collective_scalar_h1_rhs_assembly(
        layout,
        mesh,
        axis_name=axis_name,
    )
    solve = build_packed_collective_scalar_h1_cg(
        layout,
        mesh,
        policy,
        axis_name=axis_name,
    )
    mapping = jnp.asarray(layout.transport.cell_local_dofs)
    owner_mask = pack_collective_scalar_h1_owned_mask(layout)

    def apply(cell_stiffness: jax.Array, cell_rhs: jax.Array) -> ScalarH1CGResult:
        packed_stiffness = pack_collective_scalar_h1_cell_matrix(layout, cell_stiffness)
        packed_cell_rhs = pack_collective_scalar_h1_cell_vector(layout, cell_rhs)
        packed_rhs = assemble_rhs(packed_cell_rhs, mapping)
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


def assert_scalar_h1_cg_converged(result: ScalarH1CGResult | PackedScalarH1CGResult) -> None:
    """Synchronize a result and fail unless convergence and finiteness are explicit."""

    converged = bool(np.asarray(jax.device_get(result.converged)))
    breakdown = bool(np.asarray(jax.device_get(result.breakdown)))
    finite = bool(np.asarray(jax.device_get(jnp.all(jnp.isfinite(result.solution)))))
    if breakdown:
        raise FloatingPointError("scalar collective CG reported nonpositive curvature or breakdown")
    if not converged or not finite:
        raise RuntimeError("scalar collective CG did not satisfy its recomputed residual policy")
