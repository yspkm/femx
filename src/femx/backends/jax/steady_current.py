"""Native-JAX reference backend for steady isotropic electric-current conduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np

from femx.backends._steady_current import (
    CURRENT_DENSITY_UNIT,
    ELECTRIC_FIELD_UNIT,
    JOULE_HEAT_DENSITY_UNIT,
    POTENTIAL_UNIT,
    POWER_PER_DEPTH_UNIT,
    resolve_current_scalar,
    validate_steady_current_problem,
)
from femx.backends.jax._parameter_binding import bind_active_parameters
from femx.backends.jax._scalar_adjoint import (
    DifferentiableScalarH1,
    coefficient_arrays,
    full_parameter_vector,
)
from femx.backends.jax.operators import solve_scalar_h1, triangle_p1_geometry
from femx.backends.protocol import (
    BackendDescriptor,
    PreparedProblem,
    PrepareRequest,
    SolveRequest,
)
from femx.core.capabilities import (
    AnalysisKind,
    CapabilitySet,
    FunctionSpaceFamily,
    GradientMethod,
    ParallelModel,
    ScalarKind,
)
from femx.core.errors import BackendError, CapabilityError, ContractError
from femx.core.parameters import ParameterValues
from femx.core.problem import Problem
from femx.core.solution import (
    ConvergenceReport,
    ConvergenceStatus,
    Field,
    Solution,
)
from femx.mesh import DofLocation, DofMap, FunctionSpace
from femx.physics._scalar import ScalarCoefficient
from femx.physics.steady_current import SteadyCurrent


class _X64Config(Protocol):
    @property
    def x64_enabled(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class PreparedSteadyCurrent:
    """Static topology and coefficient bindings owned by the JAX current backend."""

    coordinates: jax.Array
    cells: jax.Array
    boundary_facets: jax.Array
    region_cells: tuple[jax.Array, ...]
    region_conductivity: tuple[ScalarCoefficient, ...]
    region_source: tuple[ScalarCoefficient, ...]
    flux_facets: tuple[jax.Array, ...]
    flux_values: tuple[ScalarCoefficient, ...]
    dirichlet_nodes: jax.Array
    dirichlet_values: tuple[ScalarCoefficient, ...]
    free_nodes: jax.Array
    parameter_names: tuple[str, ...]


def _derived_cell_fields(
    coordinates: jax.Array,
    cells: jax.Array,
    potential: jax.Array,
    cell_conductivity: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    areas, basis_gradients = triangle_p1_geometry(coordinates, cells)
    potential_gradient = jnp.einsum("ci,cid->cd", potential[cells], basis_gradients)
    electric_field = -potential_gradient
    current_density = cell_conductivity[:, None] * electric_field
    joule_heat_density = jnp.einsum("cd,cd->c", current_density, electric_field)
    return areas, electric_field, current_density, joule_heat_density


@dataclass(frozen=True, slots=True)
class SteadyCurrentVjpResult:
    """One explicit potential-state VJP and its adjoint-solve evidence."""

    potential: jax.Array
    potential_cotangent: jax.Array
    adjoint: jax.Array
    parameter_gradient: jax.Array
    adjoint_backward_error: jax.Array
    parameter_names: tuple[str, ...]
    parameter_units: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SteadyCurrentJouleVjpResult:
    """One Joule-density VJP split across direct material and potential-adjoint paths."""

    potential: jax.Array
    joule_heat_density: jax.Array
    joule_cotangent: jax.Array
    potential_cotangent: jax.Array
    adjoint: jax.Array
    direct_parameter_gradient: jax.Array
    indirect_parameter_gradient: jax.Array
    parameter_gradient: jax.Array
    adjoint_backward_error: jax.Array
    parameter_names: tuple[str, ...]
    parameter_units: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DifferentiableSteadyCurrent:
    """Bound current state and Joule map over canonical active physical parameters."""

    _engine: DifferentiableScalarH1
    _problem: Problem

    @property
    def problem(self) -> Problem:
        """Return the exact solver-neutral problem bound to this state map."""

        return self._problem

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return canonical differentiable parameter names."""

        return self._engine.parameter_names

    @property
    def parameter_units(self) -> tuple[str, ...]:
        """Return units aligned with :attr:`parameter_names`."""

        return self._engine.parameter_units

    @property
    def initial_values(self) -> jax.Array:
        """Return the validated initial active vector."""

        return self._engine.initial_values

    def potential(self, active_parameter_values: jax.Array) -> jax.Array:
        """Return the differentiable nodal potential for one active vector."""

        return self._engine.state(active_parameter_values)

    def joule_heat_density(self, active_parameter_values: jax.Array) -> jax.Array:
        """Return differentiable cellwise Joule heat including direct material dependence."""

        active, full, conductivity, _source, _load, _dirichlet = self._engine.resolved_coefficients(
            active_parameter_values
        )
        potential = self._engine.state(active)
        _areas, _electric, _current, joule = _derived_cell_fields(
            self._engine.payload.coordinates,
            self._engine.payload.cells,
            potential,
            conductivity,
        )
        valid = self._engine.binding.domain_is_valid(active, full, conductivity)
        return jnp.where(valid, joule, jnp.full_like(joule, jnp.nan))

    def vjp(
        self,
        active_parameter_values: jax.Array,
        potential_cotangent: jax.Array,
    ) -> SteadyCurrentVjpResult:
        """Apply the residual-defined potential VJP."""

        generic = self._engine.vjp(active_parameter_values, potential_cotangent)
        return SteadyCurrentVjpResult(
            potential=generic.state,
            potential_cotangent=generic.state_cotangent,
            adjoint=generic.adjoint,
            parameter_gradient=generic.parameter_gradient,
            adjoint_backward_error=generic.adjoint_backward_error,
            parameter_names=self.parameter_names,
            parameter_units=self.parameter_units,
        )

    def joule_vjp(
        self,
        active_parameter_values: jax.Array,
        joule_cotangent: jax.Array,
    ) -> SteadyCurrentJouleVjpResult:
        """Apply the total Joule VJP using one explicit potential adjoint solve."""

        active = self._engine.binding.active_vector(active_parameter_values)
        cotangent = jnp.asarray(joule_cotangent)
        expected_shape = self._engine.payload.cells.shape[:1]
        if cotangent.shape != expected_shape:
            raise ContractError(
                f"Joule-density cotangent must have shape {expected_shape}, got {cotangent.shape}"
            )
        if cotangent.dtype != jnp.dtype(jnp.float64):
            raise ContractError("Joule-density cotangent must use the exact float64 dtype")
        potential = self._engine.state(active)
        stopped_potential = jax.lax.stop_gradient(potential)

        def derived_joule(candidate_potential: jax.Array, candidate: jax.Array) -> jax.Array:
            full = self._engine.binding.full_vector(candidate)
            conductivity, _source, _load, _dirichlet = coefficient_arrays(
                self._engine.payload,
                full,
            )
            _areas, _electric, _current, joule = _derived_cell_fields(
                self._engine.payload.coordinates,
                self._engine.payload.cells,
                candidate_potential,
                conductivity,
            )
            valid = self._engine.binding.domain_is_valid(candidate, full, conductivity)
            return jnp.where(valid, joule, jnp.full_like(joule, jnp.nan))

        joule, pullback = jax.vjp(derived_joule, stopped_potential, active)
        potential_cotangent, direct_gradient = pullback(cotangent)
        state_vjp = self._engine.vjp(active, potential_cotangent)
        total_gradient = direct_gradient + state_vjp.parameter_gradient
        return SteadyCurrentJouleVjpResult(
            potential=state_vjp.state,
            joule_heat_density=joule,
            joule_cotangent=cotangent,
            potential_cotangent=potential_cotangent,
            adjoint=state_vjp.adjoint,
            direct_parameter_gradient=direct_gradient,
            indirect_parameter_gradient=state_vjp.parameter_gradient,
            parameter_gradient=total_gradient,
            adjoint_backward_error=state_vjp.adjoint_backward_error,
            parameter_names=self.parameter_names,
            parameter_units=self.parameter_units,
        )


class JaxSteadyCurrentBackend:
    """Dense serial float64 JAX reference with residual-defined current adjoints."""

    capabilities = CapabilitySet(
        analyses=frozenset({AnalysisKind.STEADY}),
        function_spaces=frozenset({FunctionSpaceFamily.H1}),
        scalar_kinds=frozenset({ScalarKind.REAL}),
        gradients=frozenset({GradientMethod.NONE, GradientMethod.ADJOINT}),
        parallel_models=frozenset({ParallelModel.SERIAL}),
    )

    def __init__(self, *, relative_residual_tolerance: float = 1.0e-10) -> None:
        if relative_residual_tolerance <= 0.0:
            raise ContractError("relative residual tolerance must be positive")
        self._relative_residual_tolerance = float(relative_residual_tolerance)
        self._descriptor = BackendDescriptor(
            name="jax-steady-current",
            version=f"0.2.0+jax-{jax.__version__}",
        )

    @property
    def descriptor(self) -> BackendDescriptor:
        """Return the backend and exact JAX implementation identity."""

        return self._descriptor

    def prepare(self, problem: Problem, request: PrepareRequest) -> PreparedProblem:
        """Validate and lower a serial CPU float64 current-conduction problem."""

        del request
        if jax.default_backend() != "cpu":
            raise BackendError(
                "the initial JAX steady-current backend is validated only on CPU; "
                f"active platform is {jax.default_backend()!r}"
            )
        if not cast(_X64Config, jax.config).x64_enabled:
            raise BackendError(
                "JAX float64 is required for Elmer comparison; set JAX_ENABLE_X64=1 "
                "before importing JAX"
            )
        validated = validate_steady_current_problem(problem)
        payload = PreparedSteadyCurrent(
            coordinates=jnp.asarray(validated.coordinates, dtype=jnp.float64),
            cells=jnp.asarray(validated.cells, dtype=jnp.int32),
            boundary_facets=jnp.asarray(validated.boundary_facets, dtype=jnp.int32),
            region_cells=tuple(jnp.asarray(ids, dtype=jnp.int32) for ids in validated.region_cells),
            region_conductivity=validated.region_conductivity,
            region_source=validated.region_source,
            flux_facets=tuple(jnp.asarray(ids, dtype=jnp.int32) for ids in validated.flux_facets),
            flux_values=validated.flux_values,
            dirichlet_nodes=jnp.asarray(validated.dirichlet_nodes, dtype=jnp.int32),
            dirichlet_values=validated.dirichlet_values,
            free_nodes=jnp.asarray(validated.free_nodes, dtype=jnp.int32),
            parameter_names=problem.parameters.names,
        )
        return PreparedProblem(backend=self.descriptor, problem=problem, payload=payload)

    def bind_differentiable(
        self,
        prepared: PreparedProblem,
        parameters: ParameterValues,
    ) -> DifferentiableSteadyCurrent:
        """Bind fixed values and expose potential and Joule reverse derivatives."""

        if prepared.backend != self.descriptor:
            raise BackendError("prepared JAX backend identity does not match this backend")
        if not isinstance(prepared.payload, PreparedSteadyCurrent):
            raise BackendError("prepared payload is not a JAX steady-current lowering")
        if not isinstance(prepared.problem.physics, SteadyCurrent):
            raise BackendError("prepared problem is not a steady-current specification")
        if prepared.problem.physics.gradient_method is not GradientMethod.ADJOINT:
            raise CapabilityError(
                "differentiable binding requires SteadyCurrent gradient_method=adjoint"
            )
        schema = prepared.problem.parameters
        full_values = full_parameter_vector(
            schema,
            parameters,
            prepared.payload,
            resolve_current_scalar,
        )
        binding = bind_active_parameters(
            schema,
            full_values,
            problem_label="steady-current",
            missing_message=(
                "adjoint steady current requires at least one DESIGN or CONTROL parameter"
            ),
        )
        return DifferentiableSteadyCurrent(
            _engine=DifferentiableScalarH1(
                payload=prepared.payload,
                binding=binding,
                state_name="potential",
            ),
            _problem=prepared.problem,
        )

    def solve(self, prepared: PreparedProblem, request: SolveRequest) -> Solution:
        """Resolve coefficients, solve potential, and derive cellwise electric fields."""

        if prepared.backend != self.descriptor:
            raise BackendError("prepared JAX backend identity does not match this backend")
        if not isinstance(prepared.payload, PreparedSteadyCurrent):
            raise BackendError("prepared payload is not a JAX steady-current lowering")
        payload = prepared.payload
        parameter_vector = full_parameter_vector(
            prepared.problem.parameters,
            request.parameters,
            payload,
            resolve_current_scalar,
        )
        cell_conductivity, cell_source, facet_load, dirichlet_values = coefficient_arrays(
            payload,
            parameter_vector,
        )
        potential, system = solve_scalar_h1(
            payload.coordinates,
            payload.cells,
            cell_conductivity,
            cell_source,
            payload.boundary_facets,
            facet_load,
            payload.dirichlet_nodes,
            dirichlet_values,
        )
        areas, electric_field, current_density, joule_heat_density = _derived_cell_fields(
            payload.coordinates,
            payload.cells,
            potential,
            cell_conductivity,
        )

        residual = system.stiffness @ potential - system.load
        free_residual = residual[payload.free_nodes]
        free_load = system.load[payload.free_nodes]
        free_operator = system.stiffness[payload.free_nodes, :]
        residual_norm = jnp.linalg.norm(free_residual)
        backward_error_scale = jnp.linalg.norm(free_operator) * jnp.linalg.norm(
            potential
        ) + jnp.linalg.norm(free_load)
        relative_residual = jnp.where(
            backward_error_scale > 0.0,
            residual_norm / backward_error_scale,
            jnp.where(residual_norm == 0.0, 0.0, jnp.inf),
        )

        joule_power = jnp.sum(areas * joule_heat_density)
        reaction_power = jnp.vdot(
            potential[payload.dirichlet_nodes],
            residual[payload.dirichlet_nodes],
        )
        variational_input_power = jnp.vdot(potential, system.load) + reaction_power
        energy_difference = jnp.abs(joule_power - variational_input_power)
        energy_scale = jnp.abs(joule_power) + jnp.abs(variational_input_power)
        energy_balance_error = jnp.where(
            energy_scale > 0.0,
            energy_difference / energy_scale,
            jnp.where(energy_difference == 0.0, 0.0, jnp.inf),
        )

        residual_value = float(jax.device_get(relative_residual))
        finite = all(
            bool(np.isfinite(np.asarray(jax.device_get(values))).all())
            for values in (potential, electric_field, current_density, joule_heat_density)
        )
        converged = finite and residual_value <= self._relative_residual_tolerance
        status = ConvergenceStatus.CONVERGED if converged else ConvergenceStatus.NOT_CONVERGED
        h1_space = FunctionSpace(FunctionSpaceFamily.H1, order=1)
        cell_scalar_space = FunctionSpace(
            FunctionSpaceFamily.L2,
            order=0,
            continuity="discontinuous",
        )
        cell_vector_space = FunctionSpace(
            FunctionSpaceFamily.L2,
            order=0,
            value_shape=(2,),
            continuity="discontinuous",
        )
        dof_map = DofMap(
            cell_dofs=payload.cells,
            dof_count=int(payload.coordinates.shape[0]),
            locations=frozenset({DofLocation.VERTEX}),
        )
        return Solution(
            backend_name=self.descriptor.name,
            backend_version=self.descriptor.version,
            fields={
                "potential": Field("potential", potential, POTENTIAL_UNIT, h1_space),
                "electric_field": Field(
                    "electric_field",
                    electric_field,
                    ELECTRIC_FIELD_UNIT,
                    cell_vector_space,
                ),
                "current_density": Field(
                    "current_density",
                    current_density,
                    CURRENT_DENSITY_UNIT,
                    cell_vector_space,
                ),
                "joule_heat_density": Field(
                    "joule_heat_density",
                    joule_heat_density,
                    JOULE_HEAT_DENSITY_UNIT,
                    cell_scalar_space,
                ),
            },
            observables={
                "potential_min_V": float(jax.device_get(jnp.min(potential))),
                "potential_max_V": float(jax.device_get(jnp.max(potential))),
                "joule_power_W_per_m": float(jax.device_get(joule_power)),
                "variational_input_power_W_per_m": float(jax.device_get(variational_input_power)),
                "energy_balance_relative_error": float(jax.device_get(energy_balance_error)),
            },
            convergence=ConvergenceReport(
                status=status,
                iterations=1,
                residual_norm=residual_value,
                tolerance=self._relative_residual_tolerance,
                message="dense float64 direct solve on JAX CPU",
            ),
            metadata={
                "platform": "cpu",
                "precision": "float64",
                "element": "H1 P1 triangle",
                "derived_fields": "cellwise L2 P0 from nodal potential",
                "linear_solver": "jax.numpy.linalg.solve",
                "out_of_plane_convention": "per_unit_depth",
                "current_flux_sign": "positive_variational_rhs",
                "physical_current_density": "J=-sigma*grad(phi)",
                "integrated_power_unit": POWER_PER_DEPTH_UNIT,
                "dof_count": str(dof_map.dof_count),
            },
        )
