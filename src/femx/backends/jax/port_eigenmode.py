"""Guarded native-JAX backend for the lossless mixed port-eigenmode slice."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from itertools import pairwise
from typing import NamedTuple, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np

from femx.backends._port_eigenmode import (
    ELECTRIC_FIELD_UNIT,
    MAGNETIC_FIELD_UNIT,
    PROPAGATION_CONSTANT_UNIT,
    ValidatedPortEigenmode,
    normalize_projected_electromagnetic_mode,
    resolve_port_materials,
    validate_port_eigenmode_problem,
)
from femx.backends.jax._parameter_binding import (
    ActiveParameterBinding,
    bind_active_parameters,
    coefficient_from_vector,
)
from femx.backends.jax.port_cluster_adjoint import (
    PortClusterContour,
    PortClusterDiagnostics,
    inspect_invariant_port_cluster,
)
from femx.backends.jax.port_eigen_adjoint import (
    SimplePortEigenpairDiagnostics,
    SimplePortEigenpairPolicy,
    inspect_simple_port_eigenpair,
    solve_simple_port_eigenpair,
)
from femx.backends.jax.port_eigensolver import (
    DensePortEigenmodes,
    PortSchurReduction,
    schur_reduce_port_pencil,
    solve_dense_port_eigenmodes,
)
from femx.backends.jax.port_operator import (
    assemble_lossless_port_pencil,
    lossless_port_coefficients,
    reduce_port_pencil,
)
from femx.backends.jax.port_projection import (
    NodalPortElectromagneticField,
    expand_reduced_port_coefficients,
    project_port_electromagnetic_fields_to_nodes,
)
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
from femx.core.parameters import ParameterSchema, ParameterValues
from femx.core.problem import Problem
from femx.core.solution import ConvergenceReport, ConvergenceStatus, Field, Solution
from femx.mesh import FunctionSpace
from femx.physics._scalar import ScalarCoefficient
from femx.physics.port_eigenmode import (
    PORT_LONGITUDINAL_POTENTIAL_FIELD,
    PORT_LONGITUDINAL_POTENTIAL_UNIT,
    PORT_TRANSVERSE_ELECTRIC_DOF_UNIT,
    PORT_TRANSVERSE_ELECTRIC_FIELD,
    VACUUM_PERMEABILITY_H_PER_M,
    VACUUM_PERMITTIVITY_F_PER_M,
    VACUUM_SPEED_OF_LIGHT_M_PER_S,
    PortEigenmode,
)

_ADAPTER_VERSION = "0.1.0"


class _X64Config(Protocol):
    @property
    def x64_enabled(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class PreparedJaxPortEigenmode:
    """Validated topology and immutable material bindings for one dense port solve."""

    validated: ValidatedPortEigenmode
    coordinates: jax.Array
    cells: jax.Array
    cell_edge_dofs: jax.Array
    cell_edge_signs: jax.Array
    region_cells: tuple[jax.Array, ...]
    region_relative_permittivity: tuple[ScalarCoefficient, ...]
    region_relative_permeability: tuple[ScalarCoefficient, ...]
    parameter_names: tuple[str, ...]
    free_dofs: np.ndarray
    scalar_dof_count: int
    edge_dof_count: int
    angular_frequency_rad_per_s: jax.Array


class DifferentiablePortMode(NamedTuple):
    """One canonical mixed mode and its material arrays on the differentiable path."""

    eigenvalue_per_m2: jax.Array
    propagation_constant_per_m: jax.Array
    effective_index: jax.Array
    scalar_coefficients: jax.Array
    edge_coefficients: jax.Array
    cell_relative_permittivity: jax.Array
    cell_relative_permeability: jax.Array
    cell_reluctivity_per_henry_m: jax.Array
    is_valid: jax.Array


class DifferentiablePortClusterState(NamedTuple):
    """Invariant cluster observables and material arrays on the differentiable path."""

    dimensionless_eigenvalue_sum: jax.Array
    eigenvalue_sum_per_m2: jax.Array
    propagation_constant_sum_per_m: jax.Array
    mean_propagation_constant_per_m: jax.Array
    mean_effective_index: jax.Array
    reduced_edge_mass_projector: jax.Array
    cell_relative_permittivity: jax.Array
    cell_relative_permeability: jax.Array
    is_valid: jax.Array


def _cell_material_arrays(
    payload: PreparedJaxPortEigenmode,
    parameter_values: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    cell_count = payload.cells.shape[0]
    relative_permittivity = jnp.zeros((cell_count,), dtype=jnp.float64)
    relative_permeability = jnp.zeros((cell_count,), dtype=jnp.float64)
    for cell_ids, epsilon_r, mu_r in zip(
        payload.region_cells,
        payload.region_relative_permittivity,
        payload.region_relative_permeability,
        strict=True,
    ):
        relative_permittivity = relative_permittivity.at[cell_ids].set(
            coefficient_from_vector(epsilon_r, parameter_values, payload.parameter_names)
        )
        relative_permeability = relative_permeability.at[cell_ids].set(
            coefficient_from_vector(mu_r, parameter_values, payload.parameter_names)
        )
    return relative_permittivity, relative_permeability


def _full_port_parameter_vector(
    schema: ParameterSchema,
    payload: PreparedJaxPortEigenmode,
    parameters: ParameterValues,
) -> jax.Array:
    """Validate and encode all port material parameters in canonical schema order."""

    schema.bind(parameters.values)
    resolve_port_materials(payload.validated, parameters)
    values = tuple(float(cast(int | float, parameters[name])) for name in payload.parameter_names)
    return jnp.asarray(values, dtype=jnp.float64)


def _propagation_scale(
    angular_frequency_rad_per_s: jax.Array,
    relative_permittivity: jax.Array,
    relative_permeability: jax.Array,
) -> jax.Array:
    """Return Elmer's automatic beta bound used only to condition the dense pencil."""

    return angular_frequency_rad_per_s * jnp.sqrt(
        VACUUM_PERMITTIVITY_F_PER_M
        * jnp.max(relative_permittivity)
        * VACUUM_PERMEABILITY_H_PER_M
        * jnp.max(relative_permeability)
    )


def _schur_reduction(
    payload: PreparedJaxPortEigenmode,
    cell_relative_permittivity: jax.Array,
    cell_relative_permeability: jax.Array,
) -> PortSchurReduction:
    assembled = assemble_lossless_port_pencil(
        payload.coordinates,
        payload.cells,
        payload.cell_edge_dofs,
        payload.cell_edge_signs,
        cell_relative_permittivity,
        cell_relative_permeability,
        jnp.asarray(payload.validated.frequency_hz, dtype=jnp.float64),
        edge_dof_count=payload.edge_dof_count,
    )
    reduced = reduce_port_pencil(
        assembled.stiffness,
        assembled.mass,
        jnp.asarray(payload.free_dofs, dtype=jnp.int32),
    )
    return schur_reduce_port_pencil(
        reduced.stiffness,
        reduced.mass,
        scalar_dof_count=payload.scalar_dof_count,
    )


@dataclass(frozen=True, slots=True)
class DifferentiablePortEigenmode:
    """Bound simple-mode map from active optical materials to canonical mixed coefficients."""

    _payload: PreparedJaxPortEigenmode
    _binding: ActiveParameterBinding
    _problem: Problem
    _phase_anchor_edge_dof: int
    _policy: SimplePortEigenpairPolicy
    _baseline_diagnostics: SimplePortEigenpairDiagnostics

    @property
    def problem(self) -> Problem:
        """Return the exact solver-neutral problem bound to this mode map."""

        return self._problem

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return canonical differentiable material-parameter names."""

        return self._binding.active_names

    @property
    def parameter_units(self) -> tuple[str, ...]:
        """Return units aligned with :attr:`parameter_names`."""

        return self._binding.active_units

    @property
    def initial_values(self) -> jax.Array:
        """Return the validated initial active material vector."""

        return self._binding.initial_values

    @property
    def phase_anchor_edge_dof(self) -> int:
        """Return the fixed reduced-edge sign anchor selected at the baseline."""

        return self._phase_anchor_edge_dof

    @property
    def baseline_diagnostics(self) -> SimplePortEigenpairDiagnostics:
        """Return the stopped evidence that admitted the baseline simple mode."""

        return self._baseline_diagnostics

    @property
    def angular_frequency_rad_per_s(self) -> jax.Array:
        """Return the fixed angular frequency used by E/H reconstruction."""

        return self._payload.angular_frequency_rad_per_s

    @property
    def target_power_w(self) -> float:
        """Return the requested forward power for a downstream Yee transfer."""

        return self._payload.validated.target_power_w

    def mode(self, active_parameter_values: jax.Array) -> DifferentiablePortMode:
        """Return one residual-differentiated simple mode for an exact active vector."""

        active = self._binding.active_vector(active_parameter_values)
        full = self._binding.full_vector(active)
        relative_permittivity, relative_permeability = _cell_material_arrays(self._payload, full)
        domain_valid = self._binding.domain_is_valid(
            active,
            full,
            relative_permittivity,
            relative_permeability,
        )
        safe_permittivity = jnp.where(domain_valid, relative_permittivity, 1.0)
        safe_permeability = jnp.where(domain_valid, relative_permeability, 1.0)
        reduction = _schur_reduction(
            self._payload,
            safe_permittivity,
            safe_permeability,
        )
        scale = _propagation_scale(
            self._payload.angular_frequency_rad_per_s,
            safe_permittivity,
            safe_permeability,
        )
        pair = solve_simple_port_eigenpair(
            reduction.condensed_stiffness,
            reduction.edge_mass,
            scale,
            selected_mode_index=self._payload.validated.selected_mode_index,
            phase_anchor_edge_dof=self._phase_anchor_edge_dof,
            policy=self._policy,
        )
        reduced_scalar = -reduction.scalar_recovery @ pair.edge_coefficients
        expanded = expand_reduced_port_coefficients(
            reduced_scalar[:, None].astype(jnp.complex128),
            pair.edge_coefficients[:, None].astype(jnp.complex128),
            self._payload.free_dofs,
            node_count=self._payload.coordinates.shape[0],
            edge_dof_count=self._payload.edge_dof_count,
        )
        vacuum_wavenumber = (
            2.0 * math.pi * self._payload.validated.frequency_hz / VACUUM_SPEED_OF_LIGHT_M_PER_S
        )
        _, cell_reluctivity = lossless_port_coefficients(
            safe_permittivity,
            safe_permeability,
        )
        finite_pair = (
            jnp.isfinite(pair.eigenvalue_per_m2)
            & jnp.isfinite(pair.propagation_constant_per_m)
            & jnp.all(jnp.isfinite(expanded.scalar_coefficients))
            & jnp.all(jnp.isfinite(expanded.edge_coefficients))
        )
        valid = domain_valid & finite_pair & (pair.propagation_constant_per_m > 0.0)
        invalid_real = jnp.asarray(jnp.nan, dtype=jnp.float64)
        invalid_complex = jnp.asarray(jnp.nan + 1j * jnp.nan, dtype=jnp.complex128)
        return DifferentiablePortMode(
            eigenvalue_per_m2=jnp.where(valid, pair.eigenvalue_per_m2, invalid_real),
            propagation_constant_per_m=jnp.where(
                valid,
                pair.propagation_constant_per_m,
                invalid_real,
            ),
            effective_index=jnp.where(
                valid,
                pair.propagation_constant_per_m / vacuum_wavenumber,
                invalid_real,
            ),
            scalar_coefficients=jnp.where(
                valid,
                expanded.scalar_coefficients[:, 0],
                invalid_complex,
            ),
            edge_coefficients=jnp.where(
                valid,
                expanded.edge_coefficients[:, 0],
                invalid_complex,
            ),
            cell_relative_permittivity=jnp.where(
                valid,
                relative_permittivity,
                invalid_real,
            ),
            cell_relative_permeability=jnp.where(
                valid,
                relative_permeability,
                invalid_real,
            ),
            cell_reluctivity_per_henry_m=jnp.where(
                valid,
                cell_reluctivity,
                invalid_real,
            ),
            is_valid=jax.lax.stop_gradient(valid),
        )


@dataclass(frozen=True, slots=True)
class DifferentiablePortCluster:
    """Bound Riesz-contour map for one isolated baseline mode cluster."""

    _payload: PreparedJaxPortEigenmode
    _binding: ActiveParameterBinding
    _problem: Problem
    _mode_indices: tuple[int, ...]
    _contour: PortClusterContour
    _right_probe: jax.Array
    _left_probe: jax.Array
    _baseline_diagnostics: PortClusterDiagnostics

    @property
    def problem(self) -> Problem:
        """Return the exact solver-neutral problem bound to this cluster map."""

        return self._problem

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return canonical differentiable material-parameter names."""

        return self._binding.active_names

    @property
    def parameter_units(self) -> tuple[str, ...]:
        """Return units aligned with :attr:`parameter_names`."""

        return self._binding.active_units

    @property
    def initial_values(self) -> jax.Array:
        """Return the validated initial active material vector."""

        return self._binding.initial_values

    @property
    def mode_indices(self) -> tuple[int, ...]:
        """Return the baseline spectrum indices enclosed by the contour."""

        return self._mode_indices

    @property
    def contour(self) -> PortClusterContour:
        """Return the fixed dimensionless Riesz contour."""

        return self._contour

    @property
    def baseline_diagnostics(self) -> PortClusterDiagnostics:
        """Return the stopped evidence that admitted the baseline cluster."""

        return self._baseline_diagnostics

    def cluster(self, active_parameter_values: jax.Array) -> DifferentiablePortClusterState:
        """Evaluate basis-invariant observables for an exact active vector."""

        active = self._binding.active_vector(active_parameter_values)
        full = self._binding.full_vector(active)
        relative_permittivity, relative_permeability = _cell_material_arrays(self._payload, full)
        domain_valid = self._binding.domain_is_valid(
            active,
            full,
            relative_permittivity,
            relative_permeability,
        )
        safe_permittivity = jnp.where(domain_valid, relative_permittivity, 1.0)
        safe_permeability = jnp.where(domain_valid, relative_permeability, 1.0)
        reduction = _schur_reduction(
            self._payload,
            safe_permittivity,
            safe_permeability,
        )
        scale = _propagation_scale(
            self._payload.angular_frequency_rad_per_s,
            safe_permittivity,
            safe_permeability,
        )
        inspection = inspect_invariant_port_cluster(
            reduction.condensed_stiffness,
            reduction.edge_mass,
            scale,
            self._right_probe,
            self._left_probe,
            contour=self._contour,
        )
        cluster = inspection.cluster
        finite_cluster = (
            jnp.isfinite(cluster.dimensionless_eigenvalue_sum)
            & jnp.isfinite(cluster.eigenvalue_sum_per_m2)
            & jnp.isfinite(cluster.propagation_constant_sum_per_m)
            & jnp.isfinite(cluster.mean_propagation_constant_per_m)
            & jnp.all(jnp.isfinite(cluster.reduced_edge_mass_projector))
        )
        valid = domain_valid & inspection.diagnostics.is_valid & finite_cluster
        vacuum_wavenumber = (
            2.0 * math.pi * self._payload.validated.frequency_hz / VACUUM_SPEED_OF_LIGHT_M_PER_S
        )

        def fail_closed(value: jax.Array) -> jax.Array:
            return cast(
                jax.Array,
                jax.lax.cond(
                    valid,
                    lambda admitted: admitted,
                    lambda rejected: jnp.asarray(jnp.nan, dtype=rejected.dtype) * rejected,
                    value,
                ),
            )

        return DifferentiablePortClusterState(
            dimensionless_eigenvalue_sum=fail_closed(cluster.dimensionless_eigenvalue_sum),
            eigenvalue_sum_per_m2=fail_closed(cluster.eigenvalue_sum_per_m2),
            propagation_constant_sum_per_m=fail_closed(cluster.propagation_constant_sum_per_m),
            mean_propagation_constant_per_m=fail_closed(cluster.mean_propagation_constant_per_m),
            mean_effective_index=fail_closed(
                cluster.mean_propagation_constant_per_m / vacuum_wavenumber
            ),
            reduced_edge_mass_projector=fail_closed(cluster.reduced_edge_mass_projector),
            cell_relative_permittivity=fail_closed(relative_permittivity),
            cell_relative_permeability=fail_closed(relative_permeability),
            is_valid=jax.lax.stop_gradient(valid),
        )


def _require_cpu_float64() -> None:
    platform = jax.default_backend()
    if platform != "cpu":
        raise BackendError(
            "the initial JAX port-eigenmode backend is validated only on CPU; "
            f"active platform is {platform!r}"
        )
    if not cast(_X64Config, jax.config).x64_enabled:
        raise BackendError(
            "JAX float64 is required for Elmer port comparison; set JAX_ENABLE_X64=1 "
            "before importing JAX"
        )


class _PortClusterBaseline(NamedTuple):
    contour: PortClusterContour
    right_probe: jax.Array
    left_probe: jax.Array
    diagnostics: PortClusterDiagnostics


def _validated_cluster_mode_indices(
    mode_indices: tuple[int, ...],
    *,
    requested_mode_count: int,
) -> tuple[int, ...]:
    if not isinstance(mode_indices, tuple) or len(mode_indices) < 2:
        raise ContractError("port cluster binding requires a tuple of at least two mode indices")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in mode_indices):
        raise ContractError("port cluster mode indices must be static integers")
    if tuple(sorted(set(mode_indices))) != mode_indices:
        raise ContractError("port cluster mode indices must be strictly increasing and unique")
    if any(right != left + 1 for left, right in pairwise(mode_indices)):
        raise ContractError(
            "port cluster mode indices must be consecutive in the baseline spectrum"
        )
    if mode_indices[0] < 0 or mode_indices[-1] >= requested_mode_count:
        raise ContractError("port cluster mode indices must lie in the requested baseline spectrum")
    return mode_indices


def _build_port_cluster_baseline(
    reduction: PortSchurReduction,
    scale: jax.Array,
    mode_indices: tuple[int, ...],
    *,
    quadrature_point_count: int,
) -> _PortClusterBaseline:
    mass = np.asarray(jax.device_get(reduction.edge_mass), dtype=np.float64)
    stiffness = np.asarray(jax.device_get(reduction.condensed_stiffness), dtype=np.float64)
    scale_value = float(np.asarray(jax.device_get(scale)).item())
    standard_operator = np.linalg.solve(mass, stiffness) / (scale_value * scale_value)
    eigenvalues, raw_vectors = np.linalg.eig(standard_operator)
    propagation_constants = np.sqrt(-eigenvalues.astype(np.complex128))
    order = np.lexsort((propagation_constants.imag, -propagation_constants.real))
    eigenvalues = eigenvalues[order]
    raw_vectors = raw_vectors[:, order]
    selected_indices = np.asarray(mode_indices, dtype=np.int64)
    selected_values = eigenvalues[selected_indices]
    center = float(0.5 * (np.min(selected_values.real) + np.max(selected_values.real)))
    inner_radius = float(np.max(np.abs(selected_values - center)))
    outside_values = np.delete(eigenvalues, selected_indices)
    if outside_values.size == 0:
        raise CapabilityError(
            "automatic port cluster binding requires at least one unselected finite mode"
        )
    outer_radius = float(np.min(np.abs(outside_values - center)))
    if not (
        math.isfinite(center)
        and math.isfinite(inner_radius)
        and math.isfinite(outer_radius)
        and inner_radius < outer_radius
        and outer_radius > 0.0
    ):
        raise CapabilityError("requested port modes do not form an isolated finite cluster")
    contour_radius = (
        0.5 * outer_radius if inner_radius == 0.0 else math.sqrt(inner_radius * outer_radius)
    )
    contour = PortClusterContour(
        center=center,
        radius=contour_radius,
        expected_cluster_size=len(mode_indices),
        quadrature_point_count=quadrature_point_count,
    )

    selected_vectors = raw_vectors[:, selected_indices]
    anchors = selected_vectors[
        np.argmax(np.abs(selected_vectors), axis=0),
        np.arange(len(mode_indices)),
    ]
    if np.any(np.abs(anchors) == 0.0):
        raise CapabilityError("baseline port cluster contains an unusable phase anchor")
    selected_vectors = selected_vectors * (np.conj(anchors) / np.abs(anchors))[None, :]
    imaginary_ratio = np.linalg.norm(selected_vectors.imag) / max(
        np.linalg.norm(selected_vectors),
        np.finfo(np.float64).tiny,
    )
    if imaginary_ratio > contour.maximum_relative_projected_imaginary_norm:
        raise CapabilityError("baseline port cluster does not admit a real lossless right subspace")
    right_probe = selected_vectors.real
    gram = right_probe.T @ mass @ right_probe
    try:
        lower = np.linalg.cholesky(0.5 * (gram + gram.T))
    except np.linalg.LinAlgError as error:
        raise CapabilityError(
            "baseline port cluster probe is not full rank in edge mass"
        ) from error
    right_probe = np.linalg.solve(lower, right_probe.T).T
    left_probe = mass @ right_probe
    inspection = inspect_invariant_port_cluster(
        reduction.condensed_stiffness,
        reduction.edge_mass,
        scale,
        jnp.asarray(right_probe, dtype=jnp.float64),
        jnp.asarray(left_probe, dtype=jnp.float64),
        contour=contour,
    )
    diagnostics = inspection.diagnostics
    if not bool(np.asarray(jax.device_get(diagnostics.is_valid)).item()):
        raise CapabilityError(
            "baseline port cluster failed its contour admission policy: "
            f"observed_size={int(jax.device_get(diagnostics.observed_cluster_size))}, "
            f"relative_clearance={float(jax.device_get(diagnostics.relative_contour_clearance)):.6e}, "
            f"quadrature_error={float(jax.device_get(diagnostics.relative_quadrature_error)):.6e}"
        )
    return _PortClusterBaseline(
        contour=contour,
        right_probe=jnp.asarray(right_probe, dtype=jnp.float64),
        left_probe=jnp.asarray(left_probe, dtype=jnp.float64),
        diagnostics=diagnostics,
    )


@partial(
    jax.jit,
    static_argnames=(
        "free_dofs",
        "scalar_dof_count",
        "edge_dof_count",
        "mode_count",
        "selected_mode_index",
    ),
)
def solve_lossless_port_eigenmode_fields(
    coordinates: jax.Array,
    cells: jax.Array,
    cell_edge_dofs: jax.Array,
    cell_edge_signs: jax.Array,
    cell_relative_permittivity: jax.Array,
    cell_relative_permeability: jax.Array,
    frequency_hz: jax.Array,
    angular_frequency_rad_per_s: jax.Array,
    propagation_scale_per_m: jax.Array,
    *,
    free_dofs: tuple[int, ...],
    scalar_dof_count: int,
    edge_dof_count: int,
    mode_count: int,
    selected_mode_index: int,
) -> tuple[DensePortEigenmodes, NodalPortElectromagneticField]:
    """Assemble, solve, and reconstruct one selected mode in one JAX transform."""

    assembled = assemble_lossless_port_pencil(
        coordinates,
        cells,
        cell_edge_dofs,
        cell_edge_signs,
        cell_relative_permittivity,
        cell_relative_permeability,
        frequency_hz,
        edge_dof_count=edge_dof_count,
    )
    free_dof_array = jnp.asarray(free_dofs, dtype=jnp.int32)
    reduced = reduce_port_pencil(
        assembled.stiffness,
        assembled.mass,
        free_dof_array,
    )
    modes = solve_dense_port_eigenmodes(
        reduced.stiffness,
        reduced.mass,
        propagation_scale_per_m,
        scalar_dof_count=scalar_dof_count,
        mode_count=mode_count,
    )
    _, cell_reluctivity = lossless_port_coefficients(
        cell_relative_permittivity,
        cell_relative_permeability,
    )
    mode_slice = slice(selected_mode_index, selected_mode_index + 1)
    projected = project_port_electromagnetic_fields_to_nodes(
        coordinates,
        cells,
        cell_edge_dofs,
        cell_edge_signs,
        modes.scalar_coefficients[:, mode_slice],
        modes.edge_coefficients[:, mode_slice],
        modes.propagation_constants_per_m[mode_slice],
        cell_reluctivity,
        angular_frequency_rad_per_s,
        free_dofs,
        edge_dof_count=edge_dof_count,
    )
    return modes, projected


class JaxPortEigenmodeBackend:
    """Dense serial float64 reference backend for physical lossless port modes.

    The backend is intentionally guarded to the executed lossless topology.  Its individual-mode
    adjoint is admitted only for a separated real eigenpair with a fixed baseline sign anchor. An
    isolated cluster may expose invariant contour aggregates, but never differentiated individual
    vectors. The backend advertises no accelerator, sparse, or open-boundary capability.
    """

    capabilities = CapabilitySet(
        analyses=frozenset({AnalysisKind.EIGENMODE}),
        function_spaces=frozenset({FunctionSpaceFamily.HCURL, FunctionSpaceFamily.H1}),
        scalar_kinds=frozenset({ScalarKind.COMPLEX}),
        gradients=frozenset({GradientMethod.NONE, GradientMethod.ADJOINT}),
        parallel_models=frozenset({ParallelModel.SERIAL}),
    )

    def __init__(
        self,
        *,
        relative_residual_tolerance: float = 1.0e-10,
        eigen_adjoint_policy: SimplePortEigenpairPolicy | None = None,
    ) -> None:
        if not math.isfinite(relative_residual_tolerance) or relative_residual_tolerance <= 0.0:
            raise ContractError("JAX port relative residual tolerance must be finite and positive")
        self._relative_residual_tolerance = float(relative_residual_tolerance)
        self._eigen_adjoint_policy = (
            SimplePortEigenpairPolicy() if eigen_adjoint_policy is None else eigen_adjoint_policy
        )
        self._descriptor = BackendDescriptor(
            name="jax-port-eigenmode",
            version=f"adapter-{_ADAPTER_VERSION}+jax-{jax.__version__}",
        )

    @property
    def descriptor(self) -> BackendDescriptor:
        """Return the backend and exact JAX implementation identity."""

        return self._descriptor

    def prepare(self, problem: Problem, request: PrepareRequest) -> PreparedProblem:
        """Validate and lower one serial CPU float64 mixed triangle problem."""

        del request
        _require_cpu_float64()
        validated = validate_port_eigenmode_problem(problem)
        node_count = validated.coordinates.shape[0]
        edge_dof_count = validated.edge_nodes.shape[0]
        free_dofs = np.asarray(validated.dof_partition.free_dofs, dtype=np.int64).copy()
        free_dofs.flags.writeable = False
        scalar_dof_count = int(np.count_nonzero(free_dofs < node_count))
        free_edge_dof_count = free_dofs.size - scalar_dof_count
        if validated.eigenmode_count > free_edge_dof_count:
            raise ContractError("requested port mode count exceeds the finite free-edge spectrum")

        angular_frequency = 2.0 * math.pi * validated.frequency_hz
        payload = PreparedJaxPortEigenmode(
            validated=validated,
            coordinates=jnp.asarray(validated.coordinates, dtype=jnp.float64),
            cells=jnp.asarray(validated.cells, dtype=jnp.int32),
            cell_edge_dofs=jnp.asarray(validated.cell_edge_dofs, dtype=jnp.int32),
            cell_edge_signs=jnp.asarray(validated.edge_signs, dtype=jnp.int8),
            region_cells=tuple(jnp.asarray(ids, dtype=jnp.int32) for ids in validated.region_cells),
            region_relative_permittivity=validated.relative_permittivity,
            region_relative_permeability=validated.relative_permeability,
            parameter_names=validated.parameter_names,
            free_dofs=free_dofs,
            scalar_dof_count=scalar_dof_count,
            edge_dof_count=edge_dof_count,
            angular_frequency_rad_per_s=jnp.asarray(angular_frequency, dtype=jnp.float64),
        )
        return PreparedProblem(backend=self.descriptor, problem=problem, payload=payload)

    def bind_differentiable(
        self,
        prepared: PreparedProblem,
        parameters: ParameterValues,
    ) -> DifferentiablePortEigenmode:
        """Bind one baseline and expose the simple-mode residual adjoint map."""

        if prepared.backend != self.descriptor:
            raise BackendError("prepared JAX port identity does not match this backend")
        if not isinstance(prepared.payload, PreparedJaxPortEigenmode):
            raise BackendError("prepared payload is not a JAX port-eigenmode lowering")
        if not isinstance(prepared.problem.physics, PortEigenmode):
            raise BackendError("prepared problem is not a port-eigenmode specification")
        if prepared.problem.physics.gradient_method is not GradientMethod.ADJOINT:
            raise CapabilityError(
                "differentiable binding requires PortEigenmode gradient_method=adjoint"
            )
        _require_cpu_float64()
        payload = prepared.payload
        full_values = _full_port_parameter_vector(
            prepared.problem.parameters,
            payload,
            parameters,
        )
        binding = bind_active_parameters(
            prepared.problem.parameters,
            full_values,
            problem_label="port-eigenmode",
            missing_message=(
                "adjoint port eigenmode requires at least one DESIGN or CONTROL parameter"
            ),
        )
        relative_permittivity, relative_permeability = _cell_material_arrays(payload, full_values)
        reduction = _schur_reduction(payload, relative_permittivity, relative_permeability)
        scale = _propagation_scale(
            payload.angular_frequency_rad_per_s,
            relative_permittivity,
            relative_permeability,
        )
        baseline_modes = solve_dense_port_eigenmodes(
            jnp.block(
                [
                    [reduction.scalar_stiffness, reduction.scalar_edge_coupling],
                    [reduction.edge_scalar_coupling, reduction.edge_stiffness],
                ]
            ),
            jnp.block(
                [
                    [
                        jnp.zeros_like(reduction.scalar_stiffness),
                        jnp.zeros_like(reduction.scalar_edge_coupling),
                    ],
                    [
                        jnp.zeros_like(reduction.edge_scalar_coupling),
                        reduction.edge_mass,
                    ],
                ]
            ),
            scale,
            scalar_dof_count=payload.scalar_dof_count,
            mode_count=payload.validated.eigenmode_count,
        )
        selected = payload.validated.selected_mode_index
        phase_anchor = int(
            np.asarray(jax.device_get(baseline_modes.phase_anchor_edge_dofs[selected])).item()
        )
        inspection = inspect_simple_port_eigenpair(
            reduction.condensed_stiffness,
            reduction.edge_mass,
            scale,
            selected_mode_index=selected,
            phase_anchor_edge_dof=phase_anchor,
            policy=self._eigen_adjoint_policy,
        )
        if not bool(np.asarray(jax.device_get(inspection.diagnostics.is_valid)).item()):
            diagnostics = inspection.diagnostics
            raise CapabilityError(
                "selected port mode is not admissible for an individual eigen-adjoint: "
                f"relative_gap={float(jax.device_get(diagnostics.relative_eigenvalue_gap)):.6e}, "
                f"relative_residual={float(jax.device_get(diagnostics.relative_residual)):.6e}, "
                "use a separated mode or an invariant subspace objective"
            )
        return DifferentiablePortEigenmode(
            _payload=payload,
            _binding=binding,
            _problem=prepared.problem,
            _phase_anchor_edge_dof=phase_anchor,
            _policy=self._eigen_adjoint_policy,
            _baseline_diagnostics=inspection.diagnostics,
        )

    def bind_differentiable_cluster(
        self,
        prepared: PreparedProblem,
        parameters: ParameterValues,
        *,
        mode_indices: tuple[int, ...],
        quadrature_point_count: int = 32,
    ) -> DifferentiablePortCluster:
        """Bind an isolated baseline cluster to invariant contour observables."""

        if prepared.backend != self.descriptor:
            raise BackendError("prepared JAX port identity does not match this backend")
        if not isinstance(prepared.payload, PreparedJaxPortEigenmode):
            raise BackendError("prepared payload is not a JAX port-eigenmode lowering")
        if not isinstance(prepared.problem.physics, PortEigenmode):
            raise BackendError("prepared problem is not a port-eigenmode specification")
        if prepared.problem.physics.gradient_method is not GradientMethod.ADJOINT:
            raise CapabilityError(
                "differentiable cluster binding requires PortEigenmode gradient_method=adjoint"
            )
        _require_cpu_float64()
        payload = prepared.payload
        indices = _validated_cluster_mode_indices(
            mode_indices,
            requested_mode_count=payload.validated.eigenmode_count,
        )
        full_values = _full_port_parameter_vector(
            prepared.problem.parameters,
            payload,
            parameters,
        )
        binding = bind_active_parameters(
            prepared.problem.parameters,
            full_values,
            problem_label="port-eigenmode cluster",
            missing_message=(
                "adjoint port cluster requires at least one DESIGN or CONTROL parameter"
            ),
        )
        relative_permittivity, relative_permeability = _cell_material_arrays(payload, full_values)
        reduction = _schur_reduction(payload, relative_permittivity, relative_permeability)
        scale = _propagation_scale(
            payload.angular_frequency_rad_per_s,
            relative_permittivity,
            relative_permeability,
        )
        baseline = _build_port_cluster_baseline(
            reduction,
            scale,
            indices,
            quadrature_point_count=quadrature_point_count,
        )
        return DifferentiablePortCluster(
            _payload=payload,
            _binding=binding,
            _problem=prepared.problem,
            _mode_indices=indices,
            _contour=baseline.contour,
            _right_probe=baseline.right_probe,
            _left_probe=baseline.left_probe,
            _baseline_diagnostics=baseline.diagnostics,
        )

    def solve(self, prepared: PreparedProblem, request: SolveRequest) -> Solution:
        """Solve the finite spectrum and return jointly power-normalized physical E/H."""

        if prepared.backend != self.descriptor:
            raise BackendError("prepared JAX port identity does not match this backend")
        if not isinstance(prepared.payload, PreparedJaxPortEigenmode):
            raise BackendError("prepared payload is not a JAX port-eigenmode lowering")
        if not isinstance(prepared.problem.physics, PortEigenmode):
            raise BackendError("prepared problem is not a port-eigenmode specification")
        _require_cpu_float64()
        payload = prepared.payload
        parameter_vector = _full_port_parameter_vector(
            prepared.problem.parameters,
            payload,
            request.parameters,
        )
        relative_permittivity, relative_permeability = _cell_material_arrays(
            payload,
            parameter_vector,
        )
        propagation_scale = _propagation_scale(
            payload.angular_frequency_rad_per_s,
            relative_permittivity,
            relative_permeability,
        )

        selected = payload.validated.selected_mode_index
        modes, projected = solve_lossless_port_eigenmode_fields(
            payload.coordinates,
            payload.cells,
            payload.cell_edge_dofs,
            payload.cell_edge_signs,
            relative_permittivity,
            relative_permeability,
            jnp.asarray(payload.validated.frequency_hz, dtype=jnp.float64),
            payload.angular_frequency_rad_per_s,
            propagation_scale,
            free_dofs=tuple(int(value) for value in payload.free_dofs),
            scalar_dof_count=payload.scalar_dof_count,
            edge_dof_count=payload.edge_dof_count,
            mode_count=payload.validated.eigenmode_count,
            selected_mode_index=selected,
        )
        spectrum_arrays = (
            modes.eigenvalues_per_m2,
            modes.propagation_constants_per_m,
            modes.scalar_coefficients,
            modes.edge_coefficients,
            modes.residuals.maximum_mixed,
        )
        if not all(
            np.isfinite(np.asarray(jax.device_get(values))).all() for values in spectrum_arrays
        ):
            raise BackendError("JAX port eigensolve produced non-finite spectrum data")

        selected_beta = complex(
            np.asarray(jax.device_get(modes.propagation_constants_per_m[selected])).item()
        )
        if selected_beta.real <= 0.0:
            raise BackendError("JAX port selected a non-forward propagation constant")
        raw_forward_power = float(
            np.asarray(jax.device_get(projected.raw_forward_power_w[0])).item()
        )
        normalized = normalize_projected_electromagnetic_mode(
            np.asarray(jax.device_get(projected.electric_values[:, 0, :])),
            np.asarray(jax.device_get(projected.magnetic_values[:, 0, :])),
            raw_forward_power_w=raw_forward_power,
            target_forward_power_w=payload.validated.target_power_w,
        )
        power_tolerance = 64.0 * np.finfo(np.float64).eps * payload.validated.target_power_w
        if not math.isclose(
            normalized.normalized_forward_power_w,
            payload.validated.target_power_w,
            rel_tol=64.0 * np.finfo(np.float64).eps,
            abs_tol=power_tolerance,
        ):
            raise BackendError("JAX port normalization did not reach the requested forward power")

        maximum_residual = float(np.max(np.asarray(jax.device_get(modes.residuals.maximum_mixed))))
        selected_residual = float(
            np.asarray(jax.device_get(modes.residuals.maximum_mixed[selected])).item()
        )
        convergence_status = (
            ConvergenceStatus.CONVERGED
            if maximum_residual <= self._relative_residual_tolerance
            else ConvergenceStatus.NOT_CONVERGED
        )
        vacuum_wavenumber = (
            2.0 * math.pi * payload.validated.frequency_hz / VACUUM_SPEED_OF_LIGHT_M_PER_S
        )
        effective_index = selected_beta / vacuum_wavenumber
        selected_eigenvalue = complex(
            np.asarray(jax.device_get(modes.eigenvalues_per_m2[selected])).item()
        )
        coefficient_scale = normalized.phase_factor * normalized.amplitude_scale
        normalized_scalar_coefficients = (
            projected.expanded.scalar_coefficients[:, 0] * coefficient_scale
        )
        normalized_edge_coefficients = (
            projected.expanded.edge_coefficients[:, 0] * coefficient_scale
        )
        vector_space = FunctionSpace(FunctionSpaceFamily.H1, order=1, value_shape=(3,))
        return Solution(
            backend_name=self.descriptor.name,
            backend_version=self.descriptor.version,
            fields={
                "electric_field": Field(
                    name="electric_field",
                    values=jnp.asarray(normalized.electric_field),
                    unit=ELECTRIC_FIELD_UNIT,
                    function_space=vector_space,
                ),
                "magnetic_field": Field(
                    name="magnetic_field",
                    values=jnp.asarray(normalized.magnetic_field),
                    unit=MAGNETIC_FIELD_UNIT,
                    function_space=vector_space,
                ),
                PORT_LONGITUDINAL_POTENTIAL_FIELD: Field(
                    name=PORT_LONGITUDINAL_POTENTIAL_FIELD,
                    values=normalized_scalar_coefficients,
                    unit=PORT_LONGITUDINAL_POTENTIAL_UNIT,
                    function_space=FunctionSpace(FunctionSpaceFamily.H1, order=1),
                ),
                PORT_TRANSVERSE_ELECTRIC_FIELD: Field(
                    name=PORT_TRANSVERSE_ELECTRIC_FIELD,
                    values=normalized_edge_coefficients,
                    unit=PORT_TRANSVERSE_ELECTRIC_DOF_UNIT,
                    function_space=FunctionSpace(
                        FunctionSpaceFamily.HCURL,
                        order=1,
                        value_shape=(2,),
                    ),
                ),
            },
            observables={
                "propagation_constant_rad_per_m": selected_beta,
                "effective_index": effective_index,
                "selected_eigenvalue_per_m2": selected_eigenvalue,
                "vacuum_wavenumber_rad_per_m": vacuum_wavenumber,
                "raw_forward_power_W": raw_forward_power,
                "target_forward_power_W": payload.validated.target_power_w,
                "normalized_forward_power_W": normalized.normalized_forward_power_w,
                "field_amplitude_scale": normalized.amplitude_scale,
                "selected_eigen_residual": selected_residual,
                "maximum_requested_eigen_residual": maximum_residual,
            },
            convergence=ConvergenceReport(
                status=convergence_status,
                iterations=1,
                residual_norm=maximum_residual,
                tolerance=self._relative_residual_tolerance,
                message="dense float64 Schur-reduced mixed port eigensolve on JAX CPU",
            ),
            metadata={
                "platform": "cpu",
                "precision": "float64",
                "element": "first-family first-order Hcurl triangle plus H1 constraint",
                "eigensolver": "jax.numpy.linalg.eig after scalar-constraint Schur elimination",
                "field_representation": "nodal_P1_L2_projection_of_native_mixed_fields",
                "field_components": "x,y,z",
                "field_global_phase": "largest_magnitude_E_component_positive_real",
                "field_power_normalization": "native_FEM_E_cross_conjugate_H_forward_power",
                "field_power_quadrature": "degree_two_three_point_triangle",
                "mixed_coefficient_representation": (
                    "normalized_full_canonical_H1_Pz_then_Hcurl_transverse_E"
                ),
                "magnetic_field_convention": "physical_H",
                "propagation_axis": "+z",
                "propagation_constant_unit": PROPAGATION_CONSTANT_UNIT,
                "fdtdx_mode_bundle_status": (
                    "requires_explicit_hashed_FEM_to_Yee_transfer_plan_and_target_identity"
                ),
                "projected_field_limitation": (
                    "power_authority_is_native_quadrature_not_nodal_projection"
                ),
                "node_count": str(payload.coordinates.shape[0]),
                "edge_dof_count": str(payload.edge_dof_count),
                "free_scalar_dof_count": str(payload.scalar_dof_count),
                "free_edge_dof_count": str(payload.free_dofs.size - payload.scalar_dof_count),
                "phase_anchor_node_zero_based": str(normalized.anchor_node),
                "phase_anchor_component_zero_based": str(normalized.anchor_component),
                "phase_factor_real": format(normalized.phase_factor.real, ".17e"),
                "phase_factor_imag": format(normalized.phase_factor.imag, ".17e"),
            },
        )
