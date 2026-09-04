"""Residual-defined differentiation shared by scalar H1 JAX reference solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import jax
import jax.numpy as jnp

from femx.backends.jax._parameter_binding import (
    ActiveParameterBinding,
    coefficient_from_vector,
)
from femx.backends.jax.autodiff import implicit_linear_solve
from femx.backends.jax.operators import (
    assemble_scalar_h1_system,
    impose_dirichlet_constraints,
    solve_scalar_h1,
)
from femx.core.errors import ContractError
from femx.core.parameters import (
    ParameterReference,
    ParameterSchema,
    ParameterValues,
)
from femx.physics._scalar import ScalarCoefficient


class ScalarH1Payload(Protocol):
    """Structural payload shared by the supported scalar H1 equations."""

    @property
    def coordinates(self) -> jax.Array: ...

    @property
    def cells(self) -> jax.Array: ...

    @property
    def boundary_facets(self) -> jax.Array: ...

    @property
    def region_cells(self) -> tuple[jax.Array, ...]: ...

    @property
    def region_conductivity(self) -> tuple[ScalarCoefficient, ...]: ...

    @property
    def region_source(self) -> tuple[ScalarCoefficient, ...]: ...

    @property
    def flux_facets(self) -> tuple[jax.Array, ...]: ...

    @property
    def flux_values(self) -> tuple[ScalarCoefficient, ...]: ...

    @property
    def dirichlet_nodes(self) -> jax.Array: ...

    @property
    def dirichlet_values(self) -> tuple[ScalarCoefficient, ...]: ...

    @property
    def free_nodes(self) -> jax.Array: ...

    @property
    def parameter_names(self) -> tuple[str, ...]: ...


class ScalarResolver(Protocol):
    """Resolve and validate one equation-specific scalar coefficient."""

    def __call__(
        self,
        value: ScalarCoefficient,
        parameters: ParameterValues,
        *,
        strictly_positive: bool = False,
    ) -> float: ...


@dataclass(frozen=True, slots=True)
class ScalarH1Vjp:
    """Generic scalar-state VJP plus adjoint linear-solve evidence."""

    state: jax.Array
    state_cotangent: jax.Array
    adjoint: jax.Array
    parameter_gradient: jax.Array
    adjoint_backward_error: jax.Array


@dataclass(frozen=True, slots=True)
class ScalarH1SourceVjp:
    """Scalar-state VJP including an additive cellwise source pullback."""

    state: jax.Array
    state_cotangent: jax.Array
    additive_cell_source: jax.Array
    additive_cell_source_gradient: jax.Array
    adjoint: jax.Array
    parameter_gradient: jax.Array
    adjoint_backward_error: jax.Array


def coefficient_arrays(
    payload: ScalarH1Payload,
    parameter_values: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Expand region, facet, and strong-boundary coefficients onto mesh entities."""

    cell_conductivity = jnp.zeros((payload.cells.shape[0],), dtype=jnp.float64)
    cell_source = jnp.zeros_like(cell_conductivity)
    for ids, conductivity, source in zip(
        payload.region_cells,
        payload.region_conductivity,
        payload.region_source,
        strict=True,
    ):
        cell_conductivity = cell_conductivity.at[ids].set(
            coefficient_from_vector(conductivity, parameter_values, payload.parameter_names)
        )
        cell_source = cell_source.at[ids].set(
            coefficient_from_vector(source, parameter_values, payload.parameter_names)
        )
    facet_load = jnp.zeros((payload.boundary_facets.shape[0],), dtype=jnp.float64)
    for ids, value in zip(payload.flux_facets, payload.flux_values, strict=True):
        facet_load = facet_load.at[ids].set(
            coefficient_from_vector(value, parameter_values, payload.parameter_names)
        )
    dirichlet_values = jnp.stack(
        tuple(
            coefficient_from_vector(value, parameter_values, payload.parameter_names)
            for value in payload.dirichlet_values
        )
    )
    return cell_conductivity, cell_source, facet_load, dirichlet_values


def full_parameter_vector(
    schema: ParameterSchema,
    parameters: ParameterValues,
    payload: ScalarH1Payload,
    resolver: ScalarResolver,
) -> jax.Array:
    """Validate and encode all fixed and active scalar parameters in schema order."""

    if set(parameters.values) != set(schema.names):
        schema.bind(parameters.values)
    for conductivity in payload.region_conductivity:
        resolver(conductivity, parameters, strictly_positive=True)
    values = tuple(
        resolver(ParameterReference(name), parameters) for name in payload.parameter_names
    )
    schema.bind(parameters.values)
    return jnp.asarray(values, dtype=jnp.float64)


@dataclass(frozen=True, slots=True)
class DifferentiableScalarH1:
    """Bound residual-defined scalar H1 state and explicit adjoint VJP."""

    payload: ScalarH1Payload
    binding: ActiveParameterBinding
    state_name: str

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return canonical active parameter names."""

        return self.binding.active_names

    @property
    def parameter_units(self) -> tuple[str, ...]:
        """Return units aligned with :attr:`parameter_names`."""

        return self.binding.active_units

    @property
    def initial_values(self) -> jax.Array:
        """Return the validated initial active vector."""

        return self.binding.initial_values

    def resolved_coefficients(
        self,
        active_parameter_values: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Return active/full vectors and the four entity coefficient arrays."""

        active = self.binding.active_vector(active_parameter_values)
        full = self.binding.full_vector(active)
        conductivity, source, load, dirichlet = coefficient_arrays(self.payload, full)
        return active, full, conductivity, source, load, dirichlet

    def _additive_cell_source(self, values: jax.Array) -> jax.Array:
        """Require an exact float64 P0 source aligned with the bulk-cell ordering."""

        source = jnp.asarray(values)
        expected_shape = self.payload.cells.shape[:1]
        if source.shape != expected_shape:
            raise ContractError(
                f"additive cell source must have shape {expected_shape}, got {source.shape}"
            )
        if source.dtype != jnp.dtype(jnp.float64):
            raise ContractError("additive cell source must use the exact float64 dtype")
        return source

    def _state_cotangent(self, values: jax.Array) -> jax.Array:
        """Require an exact float64 cotangent aligned with the nodal state."""

        cotangent = jnp.asarray(values)
        expected_shape = self.payload.coordinates.shape[:1]
        if cotangent.shape != expected_shape:
            raise ContractError(
                f"{self.state_name} cotangent must have shape "
                f"{expected_shape}, got {cotangent.shape}"
            )
        if cotangent.dtype != jnp.dtype(jnp.float64):
            raise ContractError(f"{self.state_name} cotangent must use the exact float64 dtype")
        return cotangent

    def state(self, active_parameter_values: jax.Array) -> jax.Array:
        """Return the differentiable nodal state for one active vector."""

        additive_source = jnp.zeros(self.payload.cells.shape[:1], dtype=jnp.float64)
        return self.state_with_additive_cell_source(active_parameter_values, additive_source)

    def state_with_additive_cell_source(
        self,
        active_parameter_values: jax.Array,
        additive_cell_source: jax.Array,
    ) -> jax.Array:
        """Return the state after adding an external P0 source to the equation source."""

        active, full, conductivity, source, load, dirichlet = self.resolved_coefficients(
            active_parameter_values
        )
        additive_source = self._additive_cell_source(additive_cell_source)
        state, _ = solve_scalar_h1(
            self.payload.coordinates,
            self.payload.cells,
            conductivity,
            source + additive_source,
            self.payload.boundary_facets,
            load,
            self.payload.dirichlet_nodes,
            dirichlet,
        )
        valid = self.binding.domain_is_valid(active, full, conductivity) & jnp.all(
            jnp.isfinite(additive_source)
        )
        return cast(
            jax.Array,
            jax.lax.cond(
                valid,
                lambda _: state,
                lambda _: (
                    jnp.full_like(state, jnp.nan) * (jnp.sum(active) + jnp.sum(additive_source))
                ),
                operand=None,
            ),
        )

    def vjp(
        self,
        active_parameter_values: jax.Array,
        state_cotangent: jax.Array,
    ) -> ScalarH1Vjp:
        """Apply the residual-defined VJP with one transposed constrained solve."""

        additive_source = jnp.zeros(self.payload.cells.shape[:1], dtype=jnp.float64)
        result = self.vjp_with_additive_cell_source(
            active_parameter_values,
            additive_source,
            state_cotangent,
        )
        return ScalarH1Vjp(
            state=result.state,
            state_cotangent=result.state_cotangent,
            adjoint=result.adjoint,
            parameter_gradient=result.parameter_gradient,
            adjoint_backward_error=result.adjoint_backward_error,
        )

    def vjp_with_additive_cell_source(
        self,
        active_parameter_values: jax.Array,
        additive_cell_source: jax.Array,
        state_cotangent: jax.Array,
    ) -> ScalarH1SourceVjp:
        """Apply a VJP to active parameters and one additive P0 source field."""

        active = self.binding.active_vector(active_parameter_values)
        additive_source = self._additive_cell_source(additive_cell_source)
        cotangent = self._state_cotangent(state_cotangent)

        full = self.binding.full_vector(active)
        conductivity, source, load, dirichlet = coefficient_arrays(self.payload, full)
        unconstrained = assemble_scalar_h1_system(
            self.payload.coordinates,
            self.payload.cells,
            conductivity,
            source + additive_source,
            self.payload.boundary_facets,
            load,
        )
        constrained = impose_dirichlet_constraints(
            unconstrained.stiffness,
            unconstrained.load,
            self.payload.dirichlet_nodes,
            dirichlet,
        )
        state = implicit_linear_solve(constrained.stiffness, constrained.load)
        adjoint = implicit_linear_solve(constrained.stiffness.T, cotangent)
        stopped_state = jax.lax.stop_gradient(state)
        stopped_adjoint = jax.lax.stop_gradient(adjoint)

        def residual(candidate: jax.Array, candidate_additive_source: jax.Array) -> jax.Array:
            candidate_full = self.binding.full_vector(candidate)
            candidate_k, candidate_q, candidate_g, candidate_d = coefficient_arrays(
                self.payload,
                candidate_full,
            )
            candidate_unconstrained = assemble_scalar_h1_system(
                self.payload.coordinates,
                self.payload.cells,
                candidate_k,
                candidate_q + candidate_additive_source,
                self.payload.boundary_facets,
                candidate_g,
            )
            candidate_constrained = impose_dirichlet_constraints(
                candidate_unconstrained.stiffness,
                candidate_unconstrained.load,
                self.payload.dirichlet_nodes,
                candidate_d,
            )
            return candidate_constrained.stiffness @ stopped_state - candidate_constrained.load

        _, residual_pullback = jax.vjp(residual, active, additive_source)
        parameter_residual_gradient, source_residual_gradient = residual_pullback(stopped_adjoint)
        parameter_gradient = -parameter_residual_gradient
        source_gradient = -source_residual_gradient
        adjoint_residual = constrained.stiffness.T @ adjoint - cotangent
        adjoint_scale = jnp.linalg.norm(constrained.stiffness.T) * jnp.linalg.norm(
            adjoint
        ) + jnp.linalg.norm(cotangent)
        adjoint_residual_norm = jnp.linalg.norm(adjoint_residual)
        adjoint_backward_error = jnp.where(
            adjoint_scale > 0.0,
            adjoint_residual_norm / adjoint_scale,
            jnp.where(adjoint_residual_norm == 0.0, 0.0, jnp.inf),
        )
        valid = (
            self.binding.domain_is_valid(active, full, conductivity)
            & jnp.all(jnp.isfinite(additive_source))
            & jnp.all(jnp.isfinite(cotangent))
        )
        return ScalarH1SourceVjp(
            state=jnp.where(valid, state, jnp.full_like(state, jnp.nan)),
            state_cotangent=cotangent,
            additive_cell_source=additive_source,
            additive_cell_source_gradient=jnp.where(
                valid,
                source_gradient,
                jnp.full_like(source_gradient, jnp.nan),
            ),
            adjoint=jnp.where(valid, adjoint, jnp.full_like(adjoint, jnp.nan)),
            parameter_gradient=jnp.where(
                valid,
                parameter_gradient,
                jnp.full_like(parameter_gradient, jnp.nan),
            ),
            adjoint_backward_error=jnp.where(valid, adjoint_backward_error, jnp.nan),
        )
