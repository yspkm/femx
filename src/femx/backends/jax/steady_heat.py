"""Guarded native-JAX backend for the first steady H1 heat-conduction slice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np

from femx.backends._steady_heat import (
    TEMPERATURE_UNIT,
    resolve_scalar,
    validate_steady_heat_problem,
)
from femx.backends.jax._parameter_binding import bind_active_parameters
from femx.backends.jax._scalar_adjoint import (
    DifferentiableScalarH1,
    coefficient_arrays,
    full_parameter_vector,
)
from femx.backends.jax.operators import solve_steady_heat
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
from femx.physics.steady_heat import SteadyHeat


class _X64Config(Protocol):
    @property
    def x64_enabled(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class PreparedSteadyHeat:
    """Static topology and coefficient bindings owned by the JAX backend."""

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


@dataclass(frozen=True, slots=True)
class SteadyHeatVjpResult:
    """One explicit state-map VJP and its adjoint-solve evidence."""

    temperature: jax.Array
    temperature_cotangent: jax.Array
    adjoint: jax.Array
    parameter_gradient: jax.Array
    adjoint_backward_error: jax.Array
    parameter_names: tuple[str, ...]
    parameter_units: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SteadyHeatSourceVjpResult:
    """Heat-state VJP including the additive cellwise heat-source pullback."""

    temperature: jax.Array
    temperature_cotangent: jax.Array
    additive_cell_heat_source: jax.Array
    additive_cell_heat_source_gradient: jax.Array
    adjoint: jax.Array
    parameter_gradient: jax.Array
    adjoint_backward_error: jax.Array
    parameter_names: tuple[str, ...]
    parameter_units: tuple[str, ...]
    source_unit: str = "W/m^3"


@dataclass(frozen=True, slots=True)
class DifferentiableSteadyHeat:
    """Bound JAX state map from active physical parameters to nodal temperature.

    The canonical active vector contains ``DESIGN`` and ``CONTROL`` parameters in the order of the
    solver-neutral :class:`~femx.core.parameters.ParameterSchema`. Fixed parameters remain closed
    over as concrete values. Values outside declared bounds or the positive-conductivity domain
    produce non-finite state and gradient arrays rather than a silent projection.
    """

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
        """Return the validated initial active vector in canonical order."""

        return self._engine.initial_values

    def temperature(self, active_parameter_values: jax.Array) -> jax.Array:
        """Return the differentiable nodal-temperature state for one active vector."""

        return self._engine.state(active_parameter_values)

    def temperature_with_cell_source(
        self,
        active_parameter_values: jax.Array,
        additive_cell_heat_source: jax.Array,
    ) -> jax.Array:
        """Return temperature after adding one same-mesh P0 heat-density field."""

        return self._engine.state_with_additive_cell_source(
            active_parameter_values,
            additive_cell_heat_source,
        )

    def vjp(
        self,
        active_parameter_values: jax.Array,
        temperature_cotangent: jax.Array,
    ) -> SteadyHeatVjpResult:
        r"""Apply the residual-defined VJP using ``A.T lambda = temperature_cotangent``."""

        generic = self._engine.vjp(active_parameter_values, temperature_cotangent)
        return SteadyHeatVjpResult(
            temperature=generic.state,
            temperature_cotangent=generic.state_cotangent,
            adjoint=generic.adjoint,
            parameter_gradient=generic.parameter_gradient,
            adjoint_backward_error=generic.adjoint_backward_error,
            parameter_names=self.parameter_names,
            parameter_units=self.parameter_units,
        )

    def source_vjp(
        self,
        active_parameter_values: jax.Array,
        additive_cell_heat_source: jax.Array,
        temperature_cotangent: jax.Array,
    ) -> SteadyHeatSourceVjpResult:
        """Pull temperature sensitivity back to heat parameters and the P0 source."""

        generic = self._engine.vjp_with_additive_cell_source(
            active_parameter_values,
            additive_cell_heat_source,
            temperature_cotangent,
        )
        return SteadyHeatSourceVjpResult(
            temperature=generic.state,
            temperature_cotangent=generic.state_cotangent,
            additive_cell_heat_source=generic.additive_cell_source,
            additive_cell_heat_source_gradient=generic.additive_cell_source_gradient,
            adjoint=generic.adjoint,
            parameter_gradient=generic.parameter_gradient,
            adjoint_backward_error=generic.adjoint_backward_error,
            parameter_names=self.parameter_names,
            parameter_units=self.parameter_units,
        )


class JaxSteadyHeatBackend:
    """Dense, serial, float64 JAX reference backend for P1 triangles.

    This reference backend advertises serial CPU execution plus the residual-defined adjoint state
    VJP. Sparse and SPMD implementations remain separate future capabilities.
    """

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
            name="jax-steady-heat",
            version=f"0.2.0+jax-{jax.__version__}",
        )

    @property
    def descriptor(self) -> BackendDescriptor:
        """Return the backend and exact JAX implementation identity."""

        return self._descriptor

    def prepare(self, problem: Problem, request: PrepareRequest) -> PreparedProblem:
        """Validate and lower a serial CPU float64 triangle problem."""

        del request
        if jax.default_backend() != "cpu":
            raise BackendError(
                "the initial JAX steady-heat backend is validated only on CPU; "
                f"active platform is {jax.default_backend()!r}"
            )
        if not cast(_X64Config, jax.config).x64_enabled:
            raise BackendError(
                "JAX float64 is required for Elmer comparison; set JAX_ENABLE_X64=1 "
                "before importing JAX"
            )
        validated = validate_steady_heat_problem(problem)
        payload = PreparedSteadyHeat(
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
    ) -> DifferentiableSteadyHeat:
        """Bind concrete fixed values and expose the canonical active JAX state map."""

        if prepared.backend != self.descriptor:
            raise BackendError("prepared JAX backend identity does not match this backend")
        if not isinstance(prepared.payload, PreparedSteadyHeat):
            raise BackendError("prepared payload is not a JAX steady-heat lowering")
        if not isinstance(prepared.problem.physics, SteadyHeat):
            raise BackendError("prepared problem is not a steady-heat specification")
        if prepared.problem.physics.gradient_method is not GradientMethod.ADJOINT:
            raise CapabilityError(
                "differentiable binding requires SteadyHeat gradient_method=adjoint"
            )
        schema = prepared.problem.parameters
        full_values = full_parameter_vector(
            schema,
            parameters,
            prepared.payload,
            resolve_scalar,
        )
        binding = bind_active_parameters(
            schema,
            full_values,
            problem_label="steady-heat",
            missing_message=(
                "adjoint steady heat requires at least one DESIGN or CONTROL parameter"
            ),
        )
        return DifferentiableSteadyHeat(
            _engine=DifferentiableScalarH1(
                payload=prepared.payload,
                binding=binding,
                state_name="temperature",
            ),
            _problem=prepared.problem,
        )

    def solve(self, prepared: PreparedProblem, request: SolveRequest) -> Solution:
        """Resolve coefficients, run the compiled JAX solve, and report the free residual."""

        if not isinstance(prepared.payload, PreparedSteadyHeat):
            raise BackendError("prepared payload is not a JAX steady-heat lowering")
        payload = prepared.payload
        parameters = request.parameters
        parameter_vector = full_parameter_vector(
            prepared.problem.parameters,
            parameters,
            payload,
            resolve_scalar,
        )
        cell_conductivity, cell_source, facet_heat_load, dirichlet_values = coefficient_arrays(
            payload, parameter_vector
        )

        temperature, system = solve_steady_heat(
            payload.coordinates,
            payload.cells,
            cell_conductivity,
            cell_source,
            payload.boundary_facets,
            facet_heat_load,
            payload.dirichlet_nodes,
            dirichlet_values,
        )
        residual = system.stiffness @ temperature - system.load
        free_residual = residual[payload.free_nodes]
        free_load = system.load[payload.free_nodes]
        free_operator = system.stiffness[payload.free_nodes, :]
        residual_norm = jnp.linalg.norm(free_residual)
        backward_error_scale = jnp.linalg.norm(free_operator) * jnp.linalg.norm(
            temperature
        ) + jnp.linalg.norm(free_load)
        relative_residual = jnp.where(
            backward_error_scale > 0.0,
            residual_norm / backward_error_scale,
            jnp.where(residual_norm == 0.0, 0.0, jnp.inf),
        )
        residual_value = float(jax.device_get(relative_residual))
        finite = bool(np.isfinite(np.asarray(jax.device_get(temperature))).all())
        converged = finite and residual_value <= self._relative_residual_tolerance
        status = ConvergenceStatus.CONVERGED if converged else ConvergenceStatus.NOT_CONVERGED
        function_space = FunctionSpace(FunctionSpaceFamily.H1, order=1)
        dof_map = DofMap(
            cell_dofs=payload.cells,
            dof_count=int(payload.coordinates.shape[0]),
            locations=frozenset({DofLocation.VERTEX}),
        )
        return Solution(
            backend_name=self.descriptor.name,
            backend_version=self.descriptor.version,
            fields={
                "temperature": Field(
                    name="temperature",
                    values=temperature,
                    unit=TEMPERATURE_UNIT,
                    function_space=function_space,
                )
            },
            observables={
                "temperature_min_K": float(jax.device_get(jnp.min(temperature))),
                "temperature_max_K": float(jax.device_get(jnp.max(temperature))),
                "variational_heat_load_W_per_m": float(jax.device_get(jnp.sum(system.load))),
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
                "linear_solver": "jax.numpy.linalg.solve",
                "out_of_plane_convention": "per_unit_depth",
                "heat_flux_sign": "positive_variational_rhs",
                "dof_count": str(dof_map.dof_count),
            },
        )
