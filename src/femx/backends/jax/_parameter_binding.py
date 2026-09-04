"""Canonical active-vector binding shared by differentiable JAX backends."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from femx.core.errors import ContractError
from femx.core.parameters import ParameterReference, ParameterRole, ParameterSchema
from femx.physics._scalar import ScalarCoefficient


def coefficient_from_vector(
    coefficient: ScalarCoefficient,
    parameter_values: jax.Array,
    parameter_names: tuple[str, ...],
) -> jax.Array:
    """Resolve a static scalar or one canonical parameter-vector entry."""

    if isinstance(coefficient, ParameterReference):
        return parameter_values[parameter_names.index(coefficient.name)]
    return jnp.asarray(coefficient, dtype=jnp.float64)


@dataclass(frozen=True, slots=True)
class ActiveParameterBinding:
    """Validated canonical active-vector projection over one full parameter vector."""

    base_parameter_values: jax.Array
    active_indices: tuple[int, ...]
    active_names: tuple[str, ...]
    active_units: tuple[str, ...]
    lower_bounds: jax.Array
    upper_bounds: jax.Array
    problem_label: str

    @property
    def initial_values(self) -> jax.Array:
        """Return the validated initial active vector."""

        return self.base_parameter_values[jnp.asarray(self.active_indices, dtype=jnp.int32)]

    def active_vector(self, values: jax.Array) -> jax.Array:
        """Require the exact canonical active-vector shape and float64 dtype."""

        active = jnp.asarray(values)
        expected_shape = (len(self.active_indices),)
        if active.shape != expected_shape:
            raise ContractError(
                f"active {self.problem_label} parameter vector must have shape "
                f"{expected_shape}, got {active.shape}"
            )
        if active.dtype != jnp.dtype(jnp.float64):
            raise ContractError(
                f"active {self.problem_label} parameters must use the exact float64 dtype"
            )
        return active

    def full_vector(self, active: jax.Array) -> jax.Array:
        """Insert active values into the closed-over full parameter vector."""

        return self.base_parameter_values.at[jnp.asarray(self.active_indices, dtype=jnp.int32)].set(
            active
        )

    def domain_is_valid(
        self,
        active: jax.Array,
        full: jax.Array,
        *positive_values: jax.Array,
    ) -> jax.Array:
        """Return the traced finite, bounds, and optional positivity predicate."""

        positive = jnp.asarray(True)
        for values in positive_values:
            positive = positive & jnp.all(values > 0.0)
        return (
            jnp.all(jnp.isfinite(full))
            & jnp.all(active >= self.lower_bounds)
            & jnp.all(active <= self.upper_bounds)
            & positive
        )


def bind_active_parameters(
    schema: ParameterSchema,
    full_values: jax.Array,
    *,
    problem_label: str,
    missing_message: str,
) -> ActiveParameterBinding:
    """Create the canonical DESIGN/CONTROL projection shared by JAX adjoints."""

    active = tuple(
        (index, spec)
        for index, spec in enumerate(schema.specs)
        if spec.role in {ParameterRole.DESIGN, ParameterRole.CONTROL}
    )
    if not active:
        raise ContractError(missing_message)
    return ActiveParameterBinding(
        base_parameter_values=full_values,
        active_indices=tuple(index for index, _spec in active),
        active_names=tuple(spec.name for _index, spec in active),
        active_units=tuple(spec.unit for _index, spec in active),
        lower_bounds=jnp.asarray(
            tuple(
                -jnp.inf if spec.lower_bound is None else spec.lower_bound
                for _index, spec in active
            ),
            dtype=jnp.float64,
        ),
        upper_bounds=jnp.asarray(
            tuple(
                jnp.inf if spec.upper_bound is None else spec.upper_bound for _index, spec in active
            ),
            dtype=jnp.float64,
        ),
        problem_label=problem_label,
    )
