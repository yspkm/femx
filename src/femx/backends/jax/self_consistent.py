"""Self-consistent temperature-dependent electrothermal reference in native JAX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from femx.backends.jax._parameter_binding import (
    ActiveParameterBinding,
    bind_active_parameters,
    coefficient_from_vector,
)
from femx.backends.jax.autodiff import implicit_linear_solve
from femx.backends.jax.operators import (
    AssembledScalarSystem,
    assemble_scalar_h1_system,
    assemble_triangle_p1_cell_nodal_load,
    impose_dirichlet_constraints,
    triangle_p1_geometry,
)
from femx.backends.jax.steady_current import DifferentiableSteadyCurrent
from femx.backends.jax.steady_heat import DifferentiableSteadyHeat
from femx.core.errors import ContractError
from femx.core.parameters import ParameterValues
from femx.physics.steady_current import SteadyCurrent
from femx.workflows.electrothermal import SelfConsistentJouleHeating


class _CoupledSystems(NamedTuple):
    current: AssembledScalarSystem
    heat: AssembledScalarSystem
    cell_nodal_conductivity: jax.Array
    cell_nodal_joule: jax.Array
    valid: jax.Array


class _ForwardSolve(NamedTuple):
    state: jax.Array
    systems: _CoupledSystems
    iterations: jax.Array
    update_error: jax.Array
    current_residual_error: jax.Array
    heat_residual_error: jax.Array
    converged: jax.Array


@dataclass(frozen=True, slots=True)
class SelfConsistentElectrothermalState:
    """One converged or explicitly non-converged coupled reference state."""

    potential: jax.Array
    temperature: jax.Array
    cell_nodal_conductivity: jax.Array
    cell_nodal_joule_heat_density: jax.Array
    iterations: jax.Array
    update_error: jax.Array
    current_residual_error: jax.Array
    heat_residual_error: jax.Array
    converged: jax.Array
    electrical_joule_power: jax.Array
    thermal_joule_load: jax.Array
    transfer_relative_error: jax.Array
    heat_balance_relative_error: jax.Array
    conductivity_unit: str = "S/m"
    joule_heat_density_unit: str = "W/m^3"
    power_unit: str = "W/m"


@dataclass(frozen=True, slots=True)
class SelfConsistentElectrothermalVjpResult:
    """Coupled-residual adjoint and schema-aligned parameter pullbacks."""

    state: SelfConsistentElectrothermalState
    temperature_cotangent: jax.Array
    coupled_adjoint: jax.Array
    current_parameter_gradient: jax.Array
    thermal_parameter_gradient: jax.Array
    feedback_parameter_gradient: jax.Array
    adjoint_backward_error: jax.Array
    current_parameter_names: tuple[str, ...]
    current_parameter_units: tuple[str, ...]
    thermal_parameter_names: tuple[str, ...]
    thermal_parameter_units: tuple[str, ...]
    feedback_parameter_names: tuple[str, ...]
    feedback_parameter_units: tuple[str, ...]


def _relative_shifted_free_residual(
    system: AssembledScalarSystem,
    state: jax.Array,
    free_nodes: jax.Array,
    reference_value: jax.Array,
) -> jax.Array:
    """Measure a constrained residual without contamination by an arbitrary field offset."""

    reference = jnp.full_like(state, reference_value)
    shifted_state = state - reference
    shifted_load = system.load - system.stiffness @ reference
    left = system.stiffness[free_nodes, :] @ shifted_state
    right = shifted_load[free_nodes]
    residual_norm = jnp.linalg.norm(left - right)
    scale = jnp.linalg.norm(left) + jnp.linalg.norm(right)
    return jnp.where(
        scale > 0.0,
        residual_norm / scale,
        jnp.where(residual_norm == 0.0, 0.0, jnp.inf),
    )


@dataclass(frozen=True, slots=True)
class DifferentiableSelfConsistentElectrothermal:
    """Bound nonlinear current/Joule/heat map with a coupled implicit VJP."""

    feedback: SelfConsistentJouleHeating
    current: DifferentiableSteadyCurrent
    thermal: DifferentiableSteadyHeat
    _feedback_binding: ActiveParameterBinding

    @classmethod
    def bind(
        cls,
        feedback: SelfConsistentJouleHeating,
        current: DifferentiableSteadyCurrent,
        thermal: DifferentiableSteadyHeat,
        feedback_parameters: ParameterValues,
    ) -> DifferentiableSelfConsistentElectrothermal:
        """Validate exact problem ownership and bind workflow-owned active coefficients."""

        if current.problem is not feedback.one_way.electrical_problem:
            raise ContractError(
                "bound current map does not belong to the feedback electrical problem"
            )
        if thermal.problem is not feedback.one_way.thermal_problem:
            raise ContractError("bound heat map does not belong to the feedback thermal problem")
        bound = feedback.parameters.bind(feedback_parameters.values)
        values: list[float] = []
        for name in feedback.parameters.names:
            raw = np.asarray(bound[name])
            if raw.shape or raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
                raise ContractError("feedback coefficients must resolve to finite real scalars")
            values.append(float(raw))
        full_values = jnp.asarray(values, dtype=jnp.float64)
        binding = bind_active_parameters(
            feedback.parameters,
            full_values,
            problem_label="electrothermal-feedback",
            missing_message=(
                "differentiable electrothermal feedback requires a DESIGN or CONTROL parameter"
            ),
        )
        return cls(feedback, current, thermal, binding)

    @property
    def initial_current_values(self) -> jax.Array:
        """Return the canonical electrical active vector."""

        return self.current.initial_values

    @property
    def initial_thermal_values(self) -> jax.Array:
        """Return the canonical thermal active vector."""

        return self.thermal.initial_values

    @property
    def initial_feedback_values(self) -> jax.Array:
        """Return the canonical workflow-law active vector."""

        return self._feedback_binding.initial_values

    @property
    def feedback_parameter_names(self) -> tuple[str, ...]:
        """Return workflow-law parameter names in schema order."""

        return self._feedback_binding.active_names

    @property
    def feedback_parameter_units(self) -> tuple[str, ...]:
        """Return units aligned with :attr:`feedback_parameter_names`."""

        return self._feedback_binding.active_units

    def _feedback_vectors(
        self,
        feedback_parameter_values: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        active = self._feedback_binding.active_vector(feedback_parameter_values)
        full = self._feedback_binding.full_vector(active)
        valid = self._feedback_binding.domain_is_valid(
            active,
            full,
            jnp.ones((1,), dtype=jnp.float64),
        )
        return active, full, valid

    def _cell_nodal_conductivity(
        self,
        current_parameter_values: jax.Array,
        feedback_parameter_values: jax.Array,
        temperature: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        current_active, current_full, base_conductivity, _source, _load, _dirichlet = (
            self.current._engine.resolved_coefficients(current_parameter_values)
        )
        _feedback_active, feedback_full, feedback_valid = self._feedback_vectors(
            feedback_parameter_values
        )
        payload = self.current._engine.payload
        local_conductivity = jnp.broadcast_to(
            base_conductivity[:, None],
            payload.cells.shape,
        )
        physics = self.feedback.one_way.electrical_problem.physics
        assert isinstance(physics, SteadyCurrent)
        region_by_tag = {
            region.tag: ids
            for region, ids in zip(physics.regions, payload.region_cells, strict=True)
        }
        for law in self.feedback.conductivity_laws:
            ids = region_by_tag[law.tag]
            reference_temperature = coefficient_from_vector(
                law.reference_temperature,
                feedback_full,
                self.feedback.parameters.names,
            )
            coefficient = coefficient_from_vector(
                law.temperature_coefficient,
                feedback_full,
                self.feedback.parameters.names,
            )
            denominator = 1.0 + coefficient * (
                temperature[payload.cells[ids]] - reference_temperature
            )
            local_conductivity = local_conductivity.at[ids].set(
                base_conductivity[ids, None] / denominator
            )
        current_valid = self.current._engine.binding.domain_is_valid(
            current_active,
            current_full,
            base_conductivity,
        )
        valid = (
            current_valid
            & feedback_valid
            & jnp.all(jnp.isfinite(temperature))
            & jnp.all(jnp.isfinite(local_conductivity))
            & jnp.all(local_conductivity > 0.0)
        )
        return local_conductivity, valid

    def _systems(
        self,
        state: jax.Array,
        current_parameter_values: jax.Array,
        thermal_parameter_values: jax.Array,
        feedback_parameter_values: jax.Array,
    ) -> _CoupledSystems:
        node_count = self.current._engine.payload.coordinates.shape[0]
        potential = state[:node_count]
        temperature = state[node_count:]
        current_payload = self.current._engine.payload
        thermal_payload = self.thermal._engine.payload

        local_conductivity, conductivity_valid = self._cell_nodal_conductivity(
            current_parameter_values,
            feedback_parameter_values,
            temperature,
        )
        (
            _current_active,
            _current_full,
            _base_sigma,
            current_source,
            current_load,
            current_dirichlet,
        ) = self.current._engine.resolved_coefficients(current_parameter_values)
        current_unconstrained = assemble_scalar_h1_system(
            current_payload.coordinates,
            current_payload.cells,
            jnp.mean(local_conductivity, axis=1),
            current_source,
            current_payload.boundary_facets,
            current_load,
        )
        current_system = impose_dirichlet_constraints(
            current_unconstrained.stiffness,
            current_unconstrained.load,
            current_payload.dirichlet_nodes,
            current_dirichlet,
        )
        _areas, basis_gradients = triangle_p1_geometry(
            current_payload.coordinates,
            current_payload.cells,
        )
        potential_gradient = jnp.einsum(
            "ci,cid->cd",
            potential[current_payload.cells],
            basis_gradients,
        )
        electric_norm_squared = jnp.einsum(
            "cd,cd->c",
            potential_gradient,
            potential_gradient,
        )
        local_joule = local_conductivity * electric_norm_squared[:, None]

        thermal_active, thermal_full, heat_k, heat_source, heat_load, heat_dirichlet = (
            self.thermal._engine.resolved_coefficients(thermal_parameter_values)
        )
        heat_unconstrained = assemble_scalar_h1_system(
            thermal_payload.coordinates,
            thermal_payload.cells,
            heat_k,
            heat_source,
            thermal_payload.boundary_facets,
            heat_load,
        )
        joule_load = assemble_triangle_p1_cell_nodal_load(
            thermal_payload.coordinates,
            thermal_payload.cells,
            local_joule,
        )
        heat_system = impose_dirichlet_constraints(
            heat_unconstrained.stiffness,
            heat_unconstrained.load + joule_load,
            thermal_payload.dirichlet_nodes,
            heat_dirichlet,
        )
        thermal_valid = self.thermal._engine.binding.domain_is_valid(
            thermal_active,
            thermal_full,
            heat_k,
        )
        return _CoupledSystems(
            current=current_system,
            heat=heat_system,
            cell_nodal_conductivity=local_conductivity,
            cell_nodal_joule=local_joule,
            valid=conductivity_valid & thermal_valid,
        )

    def _residual(
        self,
        state: jax.Array,
        current_parameter_values: jax.Array,
        thermal_parameter_values: jax.Array,
        feedback_parameter_values: jax.Array,
    ) -> jax.Array:
        node_count = self.current._engine.payload.coordinates.shape[0]
        systems = self._systems(
            state,
            current_parameter_values,
            thermal_parameter_values,
            feedback_parameter_values,
        )
        potential = state[:node_count]
        temperature = state[node_count:]
        return jnp.concatenate(
            (
                systems.current.stiffness @ potential - systems.current.load,
                systems.heat.stiffness @ temperature - systems.heat.load,
            )
        )

    def _solve_forward(
        self,
        current_parameter_values: jax.Array,
        thermal_parameter_values: jax.Array,
        feedback_parameter_values: jax.Array,
    ) -> _ForwardSolve:
        current_values = self.current._engine.binding.active_vector(current_parameter_values)
        thermal_values = self.thermal._engine.binding.active_vector(thermal_parameter_values)
        feedback_values = self._feedback_binding.active_vector(feedback_parameter_values)
        node_count = self.current._engine.payload.coordinates.shape[0]
        _ca, _cf, _ck, _cs, _cl, current_dirichlet = self.current._engine.resolved_coefficients(
            current_values
        )
        _ta, _tf, _tk, _ts, _tl, thermal_dirichlet = self.thermal._engine.resolved_coefficients(
            thermal_values
        )
        potential_reference = current_dirichlet[0]
        temperature_reference = thermal_dirichlet[0]
        initial_temperature = self.thermal.temperature(thermal_values)
        initial_state = jnp.concatenate(
            (jnp.zeros((node_count,), dtype=jnp.float64), initial_temperature)
        )
        initial_systems = self._systems(
            initial_state,
            current_values,
            thermal_values,
            feedback_values,
        )
        initial_potential = implicit_linear_solve(
            initial_systems.current.stiffness,
            initial_systems.current.load,
        )
        initial_state = jnp.concatenate((initial_potential, initial_temperature))
        policy = self.feedback.iteration

        def condition(
            loop: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> jax.Array:
            iteration, _state, update_error, current_error, heat_error = loop
            return (iteration < policy.max_iterations) & (
                (iteration < policy.minimum_iterations)
                | (update_error > 1.0)
                | (current_error > policy.residual_tolerance)
                | (heat_error > policy.residual_tolerance)
            )

        def body(
            loop: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            iteration, old_state, _update, _current_error, _heat_error = loop
            old_potential = old_state[:node_count]
            old_temperature = old_state[node_count:]
            current_systems = self._systems(
                old_state,
                current_values,
                thermal_values,
                feedback_values,
            )
            raw_potential = implicit_linear_solve(
                current_systems.current.stiffness,
                current_systems.current.load,
            )
            potential = old_potential + policy.potential_relaxation * (
                raw_potential - old_potential
            )
            potential_state = jnp.concatenate((potential, old_temperature))
            heat_systems = self._systems(
                potential_state,
                current_values,
                thermal_values,
                feedback_values,
            )
            raw_temperature = implicit_linear_solve(
                heat_systems.heat.stiffness,
                heat_systems.heat.load,
            )
            temperature = old_temperature + policy.temperature_relaxation * (
                raw_temperature - old_temperature
            )
            potential_scale = policy.potential_absolute_tolerance + (
                policy.relative_tolerance * jnp.linalg.norm(potential - potential_reference)
            )
            temperature_scale = policy.temperature_absolute_tolerance + (
                policy.relative_tolerance * jnp.linalg.norm(temperature - temperature_reference)
            )
            potential_error = jnp.linalg.norm(potential - old_potential) / potential_scale
            temperature_error = jnp.linalg.norm(temperature - old_temperature) / temperature_scale
            new_state = jnp.concatenate((potential, temperature))
            new_systems = self._systems(
                new_state,
                current_values,
                thermal_values,
                feedback_values,
            )
            current_error = _relative_shifted_free_residual(
                new_systems.current,
                potential,
                self.current._engine.payload.free_nodes,
                potential_reference,
            )
            heat_error = _relative_shifted_free_residual(
                new_systems.heat,
                temperature,
                self.thermal._engine.payload.free_nodes,
                temperature_reference,
            )
            return (
                iteration + 1,
                new_state,
                jnp.maximum(
                    potential_error,
                    temperature_error,
                ),
                current_error,
                heat_error,
            )

        iterations, state, update_error, current_error, heat_error = jax.lax.while_loop(
            condition,
            body,
            (
                jnp.asarray(0, dtype=jnp.int32),
                initial_state,
                jnp.asarray(jnp.inf),
                jnp.asarray(jnp.inf),
                jnp.asarray(jnp.inf),
            ),
        )
        systems = self._systems(
            state,
            current_values,
            thermal_values,
            feedback_values,
        )
        potential = state[:node_count]
        temperature = state[node_count:]
        current_error = _relative_shifted_free_residual(
            systems.current,
            potential,
            self.current._engine.payload.free_nodes,
            potential_reference,
        )
        heat_error = _relative_shifted_free_residual(
            systems.heat,
            temperature,
            self.thermal._engine.payload.free_nodes,
            temperature_reference,
        )
        converged = (
            systems.valid
            & (iterations >= policy.minimum_iterations)
            & (update_error <= 1.0)
            & (current_error <= policy.residual_tolerance)
            & (heat_error <= policy.residual_tolerance)
        )
        return _ForwardSolve(
            state=state,
            systems=systems,
            iterations=iterations,
            update_error=update_error,
            current_residual_error=current_error,
            heat_residual_error=heat_error,
            converged=converged,
        )

    def _implicit_pullback(
        self,
        state: jax.Array,
        current_parameter_values: jax.Array,
        thermal_parameter_values: jax.Array,
        feedback_parameter_values: jax.Array,
        state_cotangent: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        stopped_state = jax.lax.stop_gradient(state)
        jacobian = jax.jacrev(self._residual, argnums=0)(
            stopped_state,
            current_parameter_values,
            thermal_parameter_values,
            feedback_parameter_values,
        )
        adjoint = implicit_linear_solve(jacobian.T, state_cotangent)
        stopped_adjoint = jax.lax.stop_gradient(adjoint)

        def parameter_residual(
            current_values: jax.Array,
            thermal_values: jax.Array,
            feedback_values: jax.Array,
        ) -> jax.Array:
            return self._residual(
                stopped_state,
                current_values,
                thermal_values,
                feedback_values,
            )

        _, pullback = jax.vjp(
            parameter_residual,
            current_parameter_values,
            thermal_parameter_values,
            feedback_parameter_values,
        )
        current_gradient, thermal_gradient, feedback_gradient = pullback(stopped_adjoint)
        current_gradient = -current_gradient
        thermal_gradient = -thermal_gradient
        feedback_gradient = -feedback_gradient
        adjoint_residual = jacobian.T @ adjoint - state_cotangent
        residual_norm = jnp.linalg.norm(adjoint_residual)
        scale = jnp.linalg.norm(jacobian.T) * jnp.linalg.norm(adjoint) + jnp.linalg.norm(
            state_cotangent
        )
        error = jnp.where(
            scale > 0.0,
            residual_norm / scale,
            jnp.where(residual_norm == 0.0, 0.0, jnp.inf),
        )
        return adjoint, current_gradient, thermal_gradient, feedback_gradient, error

    def _differentiable_state(
        self,
        current_parameter_values: jax.Array,
        thermal_parameter_values: jax.Array,
        feedback_parameter_values: jax.Array,
    ) -> jax.Array:
        def primal(
            current_values: jax.Array,
            thermal_values: jax.Array,
            feedback_values: jax.Array,
        ) -> jax.Array:
            result = self._solve_forward(current_values, thermal_values, feedback_values)
            invalid = jnp.full_like(result.state, jnp.nan) * (
                jnp.sum(current_values) + jnp.sum(thermal_values) + jnp.sum(feedback_values)
            )
            return jnp.where(result.converged, result.state, invalid)

        solve_map = jax.custom_vjp(primal)

        def forward(
            current_values: jax.Array,
            thermal_values: jax.Array,
            feedback_values: jax.Array,
        ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
            state = primal(current_values, thermal_values, feedback_values)
            return state, (state, current_values, thermal_values, feedback_values)

        def backward(
            residual: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
            state_cotangent: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            state, current_values, thermal_values, feedback_values = residual
            _adjoint, current_gradient, thermal_gradient, feedback_gradient, _error = (
                self._implicit_pullback(
                    state,
                    current_values,
                    thermal_values,
                    feedback_values,
                    state_cotangent,
                )
            )
            return current_gradient, thermal_gradient, feedback_gradient

        solve_map.defvjp(forward, backward)
        return solve_map(
            current_parameter_values,
            thermal_parameter_values,
            feedback_parameter_values,
        )

    def coupled_state(
        self,
        current_parameter_values: jax.Array,
        thermal_parameter_values: jax.Array,
        feedback_parameter_values: jax.Array,
    ) -> jax.Array:
        """Return concatenated potential/temperature with a coupled-residual reverse rule."""

        return self._differentiable_state(
            current_parameter_values,
            thermal_parameter_values,
            feedback_parameter_values,
        )

    def potential(
        self,
        current_parameter_values: jax.Array,
        thermal_parameter_values: jax.Array,
        feedback_parameter_values: jax.Array,
    ) -> jax.Array:
        """Return the converged differentiable nodal potential."""

        node_count = self.current._engine.payload.coordinates.shape[0]
        return self.coupled_state(
            current_parameter_values,
            thermal_parameter_values,
            feedback_parameter_values,
        )[:node_count]

    def temperature(
        self,
        current_parameter_values: jax.Array,
        thermal_parameter_values: jax.Array,
        feedback_parameter_values: jax.Array,
    ) -> jax.Array:
        """Return the converged differentiable nodal temperature."""

        node_count = self.current._engine.payload.coordinates.shape[0]
        return self.coupled_state(
            current_parameter_values,
            thermal_parameter_values,
            feedback_parameter_values,
        )[node_count:]

    def solve(
        self,
        current_parameter_values: jax.Array,
        thermal_parameter_values: jax.Array,
        feedback_parameter_values: jax.Array,
    ) -> SelfConsistentElectrothermalState:
        """Return state, convergence, transfer, and energy evidence without hiding failure."""

        result = self._solve_forward(
            current_parameter_values,
            thermal_parameter_values,
            feedback_parameter_values,
        )
        node_count = self.current._engine.payload.coordinates.shape[0]
        potential = result.state[:node_count]
        temperature = result.state[node_count:]
        areas, _ = triangle_p1_geometry(
            self.current._engine.payload.coordinates,
            self.current._engine.payload.cells,
        )
        joule_power = jnp.vdot(areas, jnp.mean(result.systems.cell_nodal_joule, axis=1))
        joule_load = assemble_triangle_p1_cell_nodal_load(
            self.thermal._engine.payload.coordinates,
            self.thermal._engine.payload.cells,
            result.systems.cell_nodal_joule,
        )
        thermal_joule_load = jnp.sum(joule_load)
        transfer_difference = jnp.abs(joule_power - thermal_joule_load)
        transfer_scale = jnp.abs(joule_power) + jnp.abs(thermal_joule_load)
        transfer_error = jnp.where(
            transfer_scale > 0.0,
            transfer_difference / transfer_scale,
            jnp.where(transfer_difference == 0.0, 0.0, jnp.inf),
        )
        _active, _full, heat_k, heat_source, heat_flux, _dirichlet = (
            self.thermal._engine.resolved_coefficients(thermal_parameter_values)
        )
        unconstrained_heat = assemble_scalar_h1_system(
            self.thermal._engine.payload.coordinates,
            self.thermal._engine.payload.cells,
            heat_k,
            heat_source,
            self.thermal._engine.payload.boundary_facets,
            heat_flux,
        )
        total_heat_load = unconstrained_heat.load + joule_load
        heat_residual = unconstrained_heat.stiffness @ temperature - total_heat_load
        reaction = jnp.sum(heat_residual[self.thermal._engine.payload.dirichlet_nodes])
        variational_heat_load = jnp.sum(total_heat_load)
        heat_difference = jnp.abs(variational_heat_load + reaction)
        heat_scale = jnp.abs(variational_heat_load) + jnp.abs(reaction)
        heat_balance_error = jnp.where(
            heat_scale > 0.0,
            heat_difference / heat_scale,
            jnp.where(heat_difference == 0.0, 0.0, jnp.inf),
        )
        return SelfConsistentElectrothermalState(
            potential=potential,
            temperature=temperature,
            cell_nodal_conductivity=result.systems.cell_nodal_conductivity,
            cell_nodal_joule_heat_density=result.systems.cell_nodal_joule,
            iterations=result.iterations,
            update_error=result.update_error,
            current_residual_error=result.current_residual_error,
            heat_residual_error=result.heat_residual_error,
            converged=result.converged,
            electrical_joule_power=joule_power,
            thermal_joule_load=thermal_joule_load,
            transfer_relative_error=transfer_error,
            heat_balance_relative_error=heat_balance_error,
        )

    def vjp(
        self,
        current_parameter_values: jax.Array,
        thermal_parameter_values: jax.Array,
        feedback_parameter_values: jax.Array,
        temperature_cotangent: jax.Array,
    ) -> SelfConsistentElectrothermalVjpResult:
        """Apply one monolithic coupled-residual adjoint at the converged state."""

        result = self._solve_forward(
            current_parameter_values,
            thermal_parameter_values,
            feedback_parameter_values,
        )
        cotangent = jnp.asarray(temperature_cotangent)
        node_count = self.current._engine.payload.coordinates.shape[0]
        expected_shape = (node_count,)
        if cotangent.shape != expected_shape:
            raise ContractError(
                f"self-consistent temperature cotangent must have shape "
                f"{expected_shape}, got {cotangent.shape}"
            )
        if cotangent.dtype != jnp.dtype(jnp.float64):
            raise ContractError(
                "self-consistent temperature cotangent must use the exact float64 dtype"
            )
        state_cotangent = jnp.concatenate((jnp.zeros_like(cotangent), cotangent))
        adjoint, current_gradient, thermal_gradient, feedback_gradient, error = (
            self._implicit_pullback(
                result.state,
                current_parameter_values,
                thermal_parameter_values,
                feedback_parameter_values,
                state_cotangent,
            )
        )
        valid = result.converged & jnp.all(jnp.isfinite(cotangent))
        public_state = self.solve(
            current_parameter_values,
            thermal_parameter_values,
            feedback_parameter_values,
        )
        return SelfConsistentElectrothermalVjpResult(
            state=public_state,
            temperature_cotangent=cotangent,
            coupled_adjoint=jnp.where(valid, adjoint, jnp.full_like(adjoint, jnp.nan)),
            current_parameter_gradient=jnp.where(
                valid,
                current_gradient,
                jnp.full_like(current_gradient, jnp.nan),
            ),
            thermal_parameter_gradient=jnp.where(
                valid,
                thermal_gradient,
                jnp.full_like(thermal_gradient, jnp.nan),
            ),
            feedback_parameter_gradient=jnp.where(
                valid,
                feedback_gradient,
                jnp.full_like(feedback_gradient, jnp.nan),
            ),
            adjoint_backward_error=jnp.where(valid, error, jnp.nan),
            current_parameter_names=self.current.parameter_names,
            current_parameter_units=self.current.parameter_units,
            thermal_parameter_names=self.thermal.parameter_names,
            thermal_parameter_units=self.thermal.parameter_units,
            feedback_parameter_names=self.feedback_parameter_names,
            feedback_parameter_units=self.feedback_parameter_units,
        )
