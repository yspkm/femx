"""Differentiable one-way electrothermal composition for the JAX reference backend."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from femx.backends.jax.operators import assemble_scalar_h1_system, triangle_p1_geometry
from femx.backends.jax.steady_current import (
    DifferentiableSteadyCurrent,
    SteadyCurrentJouleVjpResult,
)
from femx.backends.jax.steady_heat import (
    DifferentiableSteadyHeat,
    SteadyHeatSourceVjpResult,
)
from femx.core.errors import ContractError
from femx.workflows.electrothermal import SameMeshJouleHeating


@dataclass(frozen=True, slots=True)
class JouleTransferBalance:
    """Integrated source and target powers for one identity transfer."""

    electrical_joule_power: jax.Array
    thermal_source_power: jax.Array
    relative_error: jax.Array
    power_unit: str = "W/m"


@dataclass(frozen=True, slots=True)
class ThermalEnergyBalance:
    """Global heat-load and Dirichlet-reaction balance for the coupled state."""

    variational_heat_load: jax.Array
    dirichlet_reaction: jax.Array
    relative_error: jax.Array
    power_unit: str = "W/m"


@dataclass(frozen=True, slots=True)
class ElectrothermalVjpResult:
    """Explicit two-adjoint pullback through current, Joule transfer, and heat."""

    current: SteadyCurrentJouleVjpResult
    thermal: SteadyHeatSourceVjpResult
    transfer: JouleTransferBalance
    thermal_energy: ThermalEnergyBalance


@dataclass(frozen=True, slots=True)
class DifferentiableOneWayElectrothermal:
    """Compose separately bound current and heat residual maps.

    Electrical and thermal active vectors remain separate namespaces. This prevents accidental
    aliasing when both problems contain a parameter with the same name and keeps each gradient
    aligned with its own solver-neutral ``ParameterSchema``.
    """

    coupling: SameMeshJouleHeating
    current: DifferentiableSteadyCurrent
    thermal: DifferentiableSteadyHeat

    def __post_init__(self) -> None:
        if self.current.problem is not self.coupling.electrical_problem:
            raise ContractError(
                "bound current map does not belong to the coupling's electrical problem"
            )
        if self.thermal.problem is not self.coupling.thermal_problem:
            raise ContractError("bound heat map does not belong to the coupling's thermal problem")

    @property
    def initial_current_values(self) -> jax.Array:
        """Return the canonical electrical active vector."""

        return self.current.initial_values

    @property
    def initial_thermal_values(self) -> jax.Array:
        """Return the canonical thermal active vector."""

        return self.thermal.initial_values

    def temperature(
        self,
        current_parameter_values: jax.Array,
        thermal_parameter_values: jax.Array,
    ) -> jax.Array:
        """Return temperature for the one-way Joule-heated system."""

        joule = self.current.joule_heat_density(current_parameter_values)
        return self.thermal.temperature_with_cell_source(thermal_parameter_values, joule)

    def transfer_balance(self, joule_heat_density: jax.Array) -> JouleTransferBalance:
        """Integrate both sides of the same-cell identity transfer independently."""

        current_areas, _ = triangle_p1_geometry(
            self.current._engine.payload.coordinates,
            self.current._engine.payload.cells,
        )
        thermal_areas, _ = triangle_p1_geometry(
            self.thermal._engine.payload.coordinates,
            self.thermal._engine.payload.cells,
        )
        electrical_power = jnp.vdot(current_areas, joule_heat_density)
        thermal_power = jnp.vdot(thermal_areas, joule_heat_density)
        difference = jnp.abs(electrical_power - thermal_power)
        scale = jnp.abs(electrical_power) + jnp.abs(thermal_power)
        relative_error = jnp.where(
            scale > 0.0,
            difference / scale,
            jnp.where(difference == 0.0, 0.0, jnp.inf),
        )
        return JouleTransferBalance(
            electrical_joule_power=electrical_power,
            thermal_source_power=thermal_power,
            relative_error=relative_error,
        )

    def _thermal_energy_balance(
        self,
        thermal_parameter_values: jax.Array,
        joule_heat_density: jax.Array,
        temperature: jax.Array,
    ) -> ThermalEnergyBalance:
        active, full, conductivity, source, load, _dirichlet = (
            self.thermal._engine.resolved_coefficients(thermal_parameter_values)
        )
        del active, full
        system = assemble_scalar_h1_system(
            self.thermal._engine.payload.coordinates,
            self.thermal._engine.payload.cells,
            conductivity,
            source + joule_heat_density,
            self.thermal._engine.payload.boundary_facets,
            load,
        )
        residual = system.stiffness @ temperature - system.load
        variational_heat_load = jnp.sum(system.load)
        dirichlet_reaction = jnp.sum(residual[self.thermal._engine.payload.dirichlet_nodes])
        difference = jnp.abs(variational_heat_load + dirichlet_reaction)
        scale = jnp.abs(variational_heat_load) + jnp.abs(dirichlet_reaction)
        relative_error = jnp.where(
            scale > 0.0,
            difference / scale,
            jnp.where(difference == 0.0, 0.0, jnp.inf),
        )
        return ThermalEnergyBalance(
            variational_heat_load=variational_heat_load,
            dirichlet_reaction=dirichlet_reaction,
            relative_error=relative_error,
        )

    def thermal_energy_balance(
        self,
        current_parameter_values: jax.Array,
        thermal_parameter_values: jax.Array,
    ) -> ThermalEnergyBalance:
        """Return the global steady-heat balance for one coupled forward state."""

        joule = self.current.joule_heat_density(current_parameter_values)
        temperature = self.thermal.temperature_with_cell_source(
            thermal_parameter_values,
            joule,
        )
        return self._thermal_energy_balance(thermal_parameter_values, joule, temperature)

    def vjp(
        self,
        current_parameter_values: jax.Array,
        thermal_parameter_values: jax.Array,
        temperature_cotangent: jax.Array,
    ) -> ElectrothermalVjpResult:
        """Apply the heat adjoint followed by the total Joule/current adjoint."""

        joule = self.current.joule_heat_density(current_parameter_values)
        thermal_result = self.thermal.source_vjp(
            thermal_parameter_values,
            joule,
            temperature_cotangent,
        )
        current_result = self.current.joule_vjp(
            current_parameter_values,
            thermal_result.additive_cell_heat_source_gradient,
        )
        return ElectrothermalVjpResult(
            current=current_result,
            thermal=thermal_result,
            transfer=self.transfer_balance(joule),
            thermal_energy=self._thermal_energy_balance(
                thermal_parameter_values,
                joule,
                thermal_result.temperature,
            ),
        )
