"""Distributed same-mesh electrothermal residual and implicit reverse rule.

The dense self-consistent backend remains the float64 reference authority.  This module lowers the
same P1 current, cell-local Joule, and P1 heat equations to owner-authoritative fixed-capacity
shards.  It deliberately requires one shared free-node identity for the first coupled gate;
different strong-boundary spaces and different meshes remain separate capabilities.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from femx.backends.jax._parameter_binding import coefficient_from_vector
from femx.backends.jax.operators import triangle_p1_diffusion_cell_matrices, triangle_p1_geometry
from femx.backends.jax.scalar_cg import (
    PackedScalarH1CGResult,
    PackedScalarH1PreconditionerFactory,
    ScalarH1CGPolicy,
    build_packed_collective_scalar_h1_cg,
    build_packed_scalar_h1_owner_dot,
)
from femx.backends.jax.scalar_collective import (
    ScalarH1CollectiveLayout,
    build_packed_collective_scalar_h1_cell_gather,
    build_packed_collective_scalar_h1_matvec,
    build_packed_collective_scalar_h1_rhs_assembly,
    pack_collective_scalar_h1_owned_mask,
    prepare_collective_scalar_h1_layout,
    prepare_scalar_h1_boundary_facet_map,
    reconstruct_scalar_h1_state,
    triangle_p1_scalar_cell_load_vectors,
)
from femx.backends.jax.scalar_owned_ghost import prepare_scalar_h1_owned_ghost_topology
from femx.backends.jax.self_consistent import DifferentiableSelfConsistentElectrothermal
from femx.core.errors import ContractError
from femx.physics.steady_current import SteadyCurrent
from femx.workflows.electrothermal import CoupledIterationPolicy

DISTRIBUTED_ELECTROTHERMAL_SCHEMA = "femx.jax.distributed_electrothermal/v1"


def _readonly(values: object, *, dtype: np.dtype | type[np.generic]) -> np.ndarray:
    result = np.array(values, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


class _ScalarAffineFields(NamedTuple):
    conductivity_base: np.ndarray
    conductivity_weights: np.ndarray
    cell_load_base: np.ndarray
    cell_load_weights: np.ndarray
    cell_dirichlet_base: np.ndarray
    cell_dirichlet_weights: np.ndarray
    node_dirichlet_base: np.ndarray
    node_dirichlet_weights: np.ndarray
    reference_base: np.ndarray
    reference_weights: np.ndarray


def _stack_affine(samples: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    base = samples[0]
    weights = np.stack(tuple(sample - base for sample in samples[1:]), axis=-1)
    return base, weights


def _sample_scalar_affine(
    system: DifferentiableSelfConsistentElectrothermal,
    *,
    current: bool,
) -> _ScalarAffineFields:
    engine = system.current._engine if current else system.thermal._engine
    payload = engine.payload
    active_count = len(engine.binding.active_indices)
    boundary_map = prepare_scalar_h1_boundary_facet_map(
        np.asarray(payload.cells),
        np.asarray(payload.boundary_facets),
        node_count=int(payload.coordinates.shape[0]),
    )

    conductivity_samples: list[np.ndarray] = []
    load_samples: list[np.ndarray] = []
    cell_dirichlet_samples: list[np.ndarray] = []
    node_dirichlet_samples: list[np.ndarray] = []
    reference_samples: list[float] = []
    candidates = [np.zeros((active_count,), dtype=np.float64)]
    candidates.extend(np.eye(active_count, dtype=np.float64))
    for candidate in candidates:
        active = jnp.asarray(candidate, dtype=jnp.float64)
        _active, _full, conductivity, source, facet_load, dirichlet = engine.resolved_coefficients(
            active
        )
        local_load = triangle_p1_scalar_cell_load_vectors(
            payload.coordinates,
            payload.cells,
            source,
            payload.boundary_facets,
            facet_load,
            boundary_map,
        )
        nodal_dirichlet = (
            jnp.zeros(
                (payload.coordinates.shape[0],),
                dtype=jnp.float64,
            )
            .at[payload.dirichlet_nodes]
            .set(dirichlet)
        )
        conductivity_samples.append(np.asarray(conductivity, dtype=np.float64))
        load_samples.append(np.asarray(local_load, dtype=np.float64))
        cell_dirichlet_samples.append(np.asarray(nodal_dirichlet[payload.cells], dtype=np.float64))
        node_dirichlet_samples.append(np.asarray(dirichlet, dtype=np.float64))
        reference_samples.append(float(np.asarray(dirichlet[0])))

    conductivity_base, conductivity_weights = _stack_affine(conductivity_samples)
    load_base, load_weights = _stack_affine(load_samples)
    cell_dirichlet_base, cell_dirichlet_weights = _stack_affine(cell_dirichlet_samples)
    node_dirichlet_base, node_dirichlet_weights = _stack_affine(node_dirichlet_samples)
    reference_base = reference_samples[0]
    reference_weights = np.asarray(reference_samples[1:]) - reference_base
    return _ScalarAffineFields(
        conductivity_base=conductivity_base,
        conductivity_weights=conductivity_weights,
        cell_load_base=load_base,
        cell_load_weights=load_weights,
        cell_dirichlet_base=cell_dirichlet_base,
        cell_dirichlet_weights=cell_dirichlet_weights,
        node_dirichlet_base=node_dirichlet_base,
        node_dirichlet_weights=node_dirichlet_weights,
        reference_base=np.asarray(reference_base, dtype=np.float64),
        reference_weights=reference_weights,
    )


def _sample_feedback_affine(
    system: DifferentiableSelfConsistentElectrothermal,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = system.current._engine.payload
    physics = system.feedback.one_way.electrical_problem.physics
    assert isinstance(physics, SteadyCurrent)
    region_by_tag = {
        region.tag: ids for region, ids in zip(physics.regions, payload.region_cells, strict=True)
    }
    active_count = len(system._feedback_binding.active_indices)
    reference_samples: list[np.ndarray] = []
    coefficient_samples: list[np.ndarray] = []
    candidates = [np.zeros((active_count,), dtype=np.float64)]
    candidates.extend(np.eye(active_count, dtype=np.float64))
    for candidate in candidates:
        active = jnp.asarray(candidate, dtype=jnp.float64)
        _active, full, _valid = system._feedback_vectors(active)
        reference = jnp.zeros(payload.cells.shape, dtype=jnp.float64)
        coefficient = jnp.zeros(payload.cells.shape, dtype=jnp.float64)
        for law in system.feedback.conductivity_laws:
            ids = region_by_tag[law.tag]
            law_reference = coefficient_from_vector(
                law.reference_temperature,
                full,
                system.feedback.parameters.names,
            )
            law_coefficient = coefficient_from_vector(
                law.temperature_coefficient,
                full,
                system.feedback.parameters.names,
            )
            reference = reference.at[ids].set(law_reference)
            coefficient = coefficient.at[ids].set(law_coefficient)
        reference_samples.append(np.asarray(reference, dtype=np.float64))
        coefficient_samples.append(np.asarray(coefficient, dtype=np.float64))
    reference_base, reference_weights = _stack_affine(reference_samples)
    coefficient_base, coefficient_weights = _stack_affine(coefficient_samples)
    return reference_base, reference_weights, coefficient_base, coefficient_weights


@dataclass(frozen=True, slots=True)
class DistributedElectrothermalPlan:
    """Host-owned lowering of one exact same-space coupled residual."""

    layout: ScalarH1CollectiveLayout
    unit_stiffness: np.ndarray
    basis_gradients: np.ndarray
    cell_areas: np.ndarray
    current: _ScalarAffineFields
    thermal: _ScalarAffineFields
    feedback_reference_base: np.ndarray
    feedback_reference_weights: np.ndarray
    feedback_coefficient_base: np.ndarray
    feedback_coefficient_weights: np.ndarray
    current_initial: np.ndarray
    thermal_initial: np.ndarray
    feedback_initial: np.ndarray
    current_lower_bounds: np.ndarray
    current_upper_bounds: np.ndarray
    thermal_lower_bounds: np.ndarray
    thermal_upper_bounds: np.ndarray
    feedback_lower_bounds: np.ndarray
    feedback_upper_bounds: np.ndarray
    current_parameter_names: tuple[str, ...]
    current_parameter_units: tuple[str, ...]
    thermal_parameter_names: tuple[str, ...]
    thermal_parameter_units: tuple[str, ...]
    feedback_parameter_names: tuple[str, ...]
    feedback_parameter_units: tuple[str, ...]
    iteration_policy: CoupledIterationPolicy
    schema_version: str = DISTRIBUTED_ELECTROTHERMAL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != DISTRIBUTED_ELECTROTHERMAL_SCHEMA:
            raise ContractError(
                f"distributed electrothermal schema must be {DISTRIBUTED_ELECTROTHERMAL_SCHEMA!r}"
            )
        if not isinstance(self.layout, ScalarH1CollectiveLayout):
            raise ContractError("distributed electrothermal plan requires a scalar H1 layout")

    def digest(self) -> str:
        """Bind topology, coefficient maps, parameter order, and iteration policy."""

        metadata = {
            "schema_version": self.schema_version,
            "layout_sha256": self.layout.digest(),
            "current_parameter_names": self.current_parameter_names,
            "thermal_parameter_names": self.thermal_parameter_names,
            "feedback_parameter_names": self.feedback_parameter_names,
            "iteration_policy": self.iteration_policy.canonical_data(),
        }
        hasher = hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        arrays = (
            self.unit_stiffness,
            self.basis_gradients,
            self.cell_areas,
            *self.current,
            *self.thermal,
            self.feedback_reference_base,
            self.feedback_reference_weights,
            self.feedback_coefficient_base,
            self.feedback_coefficient_weights,
            self.current_initial,
            self.thermal_initial,
            self.feedback_initial,
            self.current_lower_bounds,
            self.current_upper_bounds,
            self.thermal_lower_bounds,
            self.thermal_upper_bounds,
            self.feedback_lower_bounds,
            self.feedback_upper_bounds,
        )
        for value in arrays:
            array = np.asarray(value)
            hasher.update(str(array.dtype).encode("ascii"))
            hasher.update(np.asarray(array.shape, dtype="<i8").tobytes())
            hasher.update(np.ascontiguousarray(array).tobytes())
        return hasher.hexdigest()


def prepare_distributed_electrothermal_plan(
    system: DifferentiableSelfConsistentElectrothermal,
    cell_owners: object,
    *,
    partition_count: int,
) -> DistributedElectrothermalPlan:
    """Lower the dense M2d authority without device discovery or silent space changes."""

    if not isinstance(system, DifferentiableSelfConsistentElectrothermal):
        raise ContractError(
            "distributed electrothermal lowering requires the bound dense reference system"
        )
    current_payload = system.current._engine.payload
    thermal_payload = system.thermal._engine.payload
    for first, second, label in (
        (current_payload.coordinates, thermal_payload.coordinates, "coordinates"),
        (current_payload.cells, thermal_payload.cells, "cell order"),
        (current_payload.free_nodes, thermal_payload.free_nodes, "free-node identity"),
        (current_payload.dirichlet_nodes, thermal_payload.dirichlet_nodes, "Dirichlet nodes"),
    ):
        if not np.array_equal(np.asarray(first), np.asarray(second)):
            raise ContractError(f"distributed electrothermal first gate requires identical {label}")
    topology = prepare_scalar_h1_owned_ghost_topology(
        np.asarray(current_payload.cells),
        cell_owners,
        node_count=int(current_payload.coordinates.shape[0]),
        free_nodes=np.asarray(current_payload.free_nodes),
        partition_count=partition_count,
    )
    layout = prepare_collective_scalar_h1_layout(topology)
    coordinates = current_payload.coordinates
    cells = current_payload.cells
    cell_count = int(cells.shape[0])
    areas, gradients = triangle_p1_geometry(coordinates, cells)
    unit_stiffness = triangle_p1_diffusion_cell_matrices(
        coordinates,
        cells,
        jnp.ones((cell_count,), dtype=jnp.float64),
    )
    current = _sample_scalar_affine(system, current=True)
    thermal = _sample_scalar_affine(system, current=False)
    feedback_reference_base, feedback_reference_weights, coefficient_base, coefficient_weights = (
        _sample_feedback_affine(system)
    )
    bindings = (
        system.current._engine.binding,
        system.thermal._engine.binding,
        system._feedback_binding,
    )

    def affine_fields(fields: _ScalarAffineFields) -> _ScalarAffineFields:
        return _ScalarAffineFields(*(_readonly(value, dtype=np.float64) for value in fields))

    return DistributedElectrothermalPlan(
        layout=layout,
        unit_stiffness=_readonly(unit_stiffness, dtype=np.float64),
        basis_gradients=_readonly(gradients, dtype=np.float64),
        cell_areas=_readonly(areas, dtype=np.float64),
        current=affine_fields(current),
        thermal=affine_fields(thermal),
        feedback_reference_base=_readonly(feedback_reference_base, dtype=np.float64),
        feedback_reference_weights=_readonly(feedback_reference_weights, dtype=np.float64),
        feedback_coefficient_base=_readonly(coefficient_base, dtype=np.float64),
        feedback_coefficient_weights=_readonly(coefficient_weights, dtype=np.float64),
        current_initial=_readonly(system.initial_current_values, dtype=np.float64),
        thermal_initial=_readonly(system.initial_thermal_values, dtype=np.float64),
        feedback_initial=_readonly(system.initial_feedback_values, dtype=np.float64),
        current_lower_bounds=_readonly(bindings[0].lower_bounds, dtype=np.float64),
        current_upper_bounds=_readonly(bindings[0].upper_bounds, dtype=np.float64),
        thermal_lower_bounds=_readonly(bindings[1].lower_bounds, dtype=np.float64),
        thermal_upper_bounds=_readonly(bindings[1].upper_bounds, dtype=np.float64),
        feedback_lower_bounds=_readonly(bindings[2].lower_bounds, dtype=np.float64),
        feedback_upper_bounds=_readonly(bindings[2].upper_bounds, dtype=np.float64),
        current_parameter_names=system.current.parameter_names,
        current_parameter_units=system.current.parameter_units,
        thermal_parameter_names=system.thermal.parameter_names,
        thermal_parameter_units=system.thermal.parameter_units,
        feedback_parameter_names=system.feedback_parameter_names,
        feedback_parameter_units=system.feedback_parameter_units,
        iteration_policy=system.feedback.iteration,
    )


class HostPackedDistributedElectrothermalInputs(NamedTuple):
    """Host arrays before callers choose concrete JAX devices and shardings."""

    cell_local_dofs: np.ndarray
    owner_mask: np.ndarray
    cell_mask: np.ndarray
    unit_stiffness: np.ndarray
    basis_gradients: np.ndarray
    cell_areas: np.ndarray
    current_conductivity_base: np.ndarray
    current_conductivity_weights: np.ndarray
    current_cell_load_base: np.ndarray
    current_cell_load_weights: np.ndarray
    current_dirichlet_base: np.ndarray
    current_dirichlet_weights: np.ndarray
    current_reference_base: np.ndarray
    current_reference_weights: np.ndarray
    thermal_conductivity_base: np.ndarray
    thermal_conductivity_weights: np.ndarray
    thermal_cell_load_base: np.ndarray
    thermal_cell_load_weights: np.ndarray
    thermal_dirichlet_base: np.ndarray
    thermal_dirichlet_weights: np.ndarray
    thermal_reference_base: np.ndarray
    thermal_reference_weights: np.ndarray
    feedback_reference_base: np.ndarray
    feedback_reference_weights: np.ndarray
    feedback_coefficient_base: np.ndarray
    feedback_coefficient_weights: np.ndarray
    current_lower_bounds: np.ndarray
    current_upper_bounds: np.ndarray
    thermal_lower_bounds: np.ndarray
    thermal_upper_bounds: np.ndarray
    feedback_lower_bounds: np.ndarray
    feedback_upper_bounds: np.ndarray


class PackedDistributedElectrothermalInputs(NamedTuple):
    """Explicit JAX inputs; cell-leading arrays are sharded on the partition axis."""

    cell_local_dofs: jax.Array
    owner_mask: jax.Array
    cell_mask: jax.Array
    unit_stiffness: jax.Array
    basis_gradients: jax.Array
    cell_areas: jax.Array
    current_conductivity_base: jax.Array
    current_conductivity_weights: jax.Array
    current_cell_load_base: jax.Array
    current_cell_load_weights: jax.Array
    current_dirichlet_base: jax.Array
    current_dirichlet_weights: jax.Array
    current_reference_base: jax.Array
    current_reference_weights: jax.Array
    thermal_conductivity_base: jax.Array
    thermal_conductivity_weights: jax.Array
    thermal_cell_load_base: jax.Array
    thermal_cell_load_weights: jax.Array
    thermal_dirichlet_base: jax.Array
    thermal_dirichlet_weights: jax.Array
    thermal_reference_base: jax.Array
    thermal_reference_weights: jax.Array
    feedback_reference_base: jax.Array
    feedback_reference_weights: jax.Array
    feedback_coefficient_base: jax.Array
    feedback_coefficient_weights: jax.Array
    current_lower_bounds: jax.Array
    current_upper_bounds: jax.Array
    thermal_lower_bounds: jax.Array
    thermal_upper_bounds: jax.Array
    feedback_lower_bounds: jax.Array
    feedback_upper_bounds: jax.Array


def _pack_cell_array(
    layout: ScalarH1CollectiveLayout,
    values: np.ndarray,
    *,
    dtype: np.dtype,
) -> np.ndarray:
    canonical = np.asarray(values, dtype=dtype)
    sentinel = np.zeros((1, *canonical.shape[1:]), dtype=dtype)
    return _readonly(
        np.concatenate((canonical, sentinel), axis=0)[layout.transport.cell_ids],
        dtype=dtype,
    )


def pack_distributed_electrothermal_inputs_host(
    plan: DistributedElectrothermalPlan,
    *,
    value_dtype: np.dtype | type[np.generic],
) -> HostPackedDistributedElectrothermalInputs:
    """Pack one host plan while keeping every large field partition-leading."""

    if not isinstance(plan, DistributedElectrothermalPlan):
        raise ContractError("distributed electrothermal packing requires a prepared plan")
    dtype = np.dtype(value_dtype)
    if dtype.kind != "f" or dtype.itemsize not in (4, 8):
        raise ContractError("distributed electrothermal values require float32 or float64")
    layout = plan.layout
    cell_mask = layout.transport.cell_ids < layout.topology.cell_count

    def pack(values: np.ndarray) -> np.ndarray:
        return _pack_cell_array(layout, values, dtype=dtype)

    return HostPackedDistributedElectrothermalInputs(
        cell_local_dofs=_readonly(layout.transport.cell_local_dofs, dtype=np.int32),
        owner_mask=_readonly(
            np.asarray(pack_collective_scalar_h1_owned_mask(layout)),
            dtype=np.bool_,
        ),
        cell_mask=_readonly(cell_mask, dtype=np.bool_),
        unit_stiffness=pack(plan.unit_stiffness),
        basis_gradients=pack(plan.basis_gradients),
        cell_areas=pack(plan.cell_areas),
        current_conductivity_base=pack(plan.current.conductivity_base),
        current_conductivity_weights=pack(plan.current.conductivity_weights),
        current_cell_load_base=pack(plan.current.cell_load_base),
        current_cell_load_weights=pack(plan.current.cell_load_weights),
        current_dirichlet_base=pack(plan.current.cell_dirichlet_base),
        current_dirichlet_weights=pack(plan.current.cell_dirichlet_weights),
        current_reference_base=_readonly(plan.current.reference_base, dtype=dtype),
        current_reference_weights=_readonly(plan.current.reference_weights, dtype=dtype),
        thermal_conductivity_base=pack(plan.thermal.conductivity_base),
        thermal_conductivity_weights=pack(plan.thermal.conductivity_weights),
        thermal_cell_load_base=pack(plan.thermal.cell_load_base),
        thermal_cell_load_weights=pack(plan.thermal.cell_load_weights),
        thermal_dirichlet_base=pack(plan.thermal.cell_dirichlet_base),
        thermal_dirichlet_weights=pack(plan.thermal.cell_dirichlet_weights),
        thermal_reference_base=_readonly(plan.thermal.reference_base, dtype=dtype),
        thermal_reference_weights=_readonly(plan.thermal.reference_weights, dtype=dtype),
        feedback_reference_base=pack(plan.feedback_reference_base),
        feedback_reference_weights=pack(plan.feedback_reference_weights),
        feedback_coefficient_base=pack(plan.feedback_coefficient_base),
        feedback_coefficient_weights=pack(plan.feedback_coefficient_weights),
        current_lower_bounds=_readonly(plan.current_lower_bounds, dtype=dtype),
        current_upper_bounds=_readonly(plan.current_upper_bounds, dtype=dtype),
        thermal_lower_bounds=_readonly(plan.thermal_lower_bounds, dtype=dtype),
        thermal_upper_bounds=_readonly(plan.thermal_upper_bounds, dtype=dtype),
        feedback_lower_bounds=_readonly(plan.feedback_lower_bounds, dtype=dtype),
        feedback_upper_bounds=_readonly(plan.feedback_upper_bounds, dtype=dtype),
    )


def pack_distributed_electrothermal_inputs(
    plan: DistributedElectrothermalPlan,
    *,
    value_dtype: np.dtype | type[np.generic],
) -> PackedDistributedElectrothermalInputs:
    """Convert host-packed inputs to ordinary JAX arrays without selecting devices."""

    host = pack_distributed_electrothermal_inputs_host(plan, value_dtype=value_dtype)
    return PackedDistributedElectrothermalInputs(*(jnp.asarray(value) for value in host))


@dataclass(frozen=True, slots=True)
class ElectrothermalAdjointPolicy:
    """Static restarted-GMRES policy for the nonsymmetric coupled transpose."""

    relative_tolerance: float
    absolute_tolerance: float
    restart: int
    max_restarts: int

    def __post_init__(self) -> None:
        for name in ("relative_tolerance", "absolute_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ContractError(f"coupled adjoint {name.replace('_', ' ')} must be real")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ContractError(f"coupled adjoint {name.replace('_', ' ')} must be finite")
        if self.relative_tolerance <= 0.0:
            raise ContractError("coupled adjoint relative tolerance must be positive")
        for name in ("restart", "max_restarts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ContractError(f"coupled adjoint {name.replace('_', ' ')} must be positive")


class PackedElectrothermalVector(NamedTuple):
    """Owner-authoritative free potential and temperature shards."""

    potential: jax.Array
    temperature: jax.Array


class _ResolvedFields(NamedTuple):
    current_conductivity: jax.Array
    current_cell_load: jax.Array
    current_dirichlet: jax.Array
    current_reference: jax.Array
    thermal_conductivity: jax.Array
    thermal_cell_load: jax.Array
    thermal_dirichlet: jax.Array
    thermal_reference: jax.Array
    feedback_reference: jax.Array
    feedback_coefficient: jax.Array
    valid: jax.Array


class _CoupledOperators(NamedTuple):
    current_stiffness: jax.Array
    current_rhs: jax.Array
    heat_stiffness: jax.Array
    heat_rhs: jax.Array
    cell_nodal_conductivity: jax.Array
    cell_nodal_joule: jax.Array
    heat_cell_load: jax.Array
    valid: jax.Array


class PackedDistributedElectrothermalResult(NamedTuple):
    """Forward state, conservation values, and explicit convergence diagnostics."""

    state: PackedElectrothermalVector
    cell_nodal_conductivity: jax.Array
    cell_nodal_joule: jax.Array
    iterations: jax.Array
    update_error: jax.Array
    current_residual_error: jax.Array
    heat_residual_error: jax.Array
    current_linear_iterations: jax.Array
    heat_linear_iterations: jax.Array
    current_linear_recursive_residual: jax.Array
    heat_linear_recursive_residual: jax.Array
    current_linear_recomputed_residual: jax.Array
    heat_linear_recomputed_residual: jax.Array
    current_linear_relative_residual: jax.Array
    heat_linear_relative_residual: jax.Array
    current_linear_backward_error: jax.Array
    heat_linear_backward_error: jax.Array
    current_linear_converged: jax.Array
    heat_linear_converged: jax.Array
    current_linear_breakdown: jax.Array
    heat_linear_breakdown: jax.Array
    electrical_joule_power: jax.Array
    thermal_joule_load: jax.Array
    transfer_relative_error: jax.Array
    converged: jax.Array


class PackedDistributedElectrothermalVjpResult(NamedTuple):
    """Explicit coupled transpose result and three parameter namespaces."""

    forward: PackedDistributedElectrothermalResult
    state_cotangent: PackedElectrothermalVector
    coupled_adjoint: PackedElectrothermalVector
    current_parameter_gradient: jax.Array
    thermal_parameter_gradient: jax.Array
    feedback_parameter_gradient: jax.Array
    adjoint_backward_error: jax.Array
    adjoint_converged: jax.Array


@dataclass(frozen=True, slots=True)
class DistributedElectrothermalRuntime:
    """Functions bound to one host plan, explicit device Mesh, and solver policies."""

    solve: Callable[..., PackedDistributedElectrothermalResult]
    state: Callable[..., PackedElectrothermalVector]
    cell_temperature: Callable[..., jax.Array]
    vjp: Callable[..., PackedDistributedElectrothermalVjpResult]


def _affine(base: jax.Array, weights: jax.Array, active: jax.Array) -> jax.Array:
    return base + jnp.tensordot(weights, active, axes=((-1,), (0,)))


def _relative_difference(numerator: jax.Array, first: jax.Array, second: jax.Array) -> jax.Array:
    scale = first + second
    return jnp.where(
        scale > 0.0,
        numerator / scale,
        jnp.where(numerator == 0.0, 0.0, jnp.inf),
    )


def build_distributed_electrothermal_runtime(
    plan: DistributedElectrothermalPlan,
    mesh: Mesh,
    cg_policy: ScalarH1CGPolicy,
    adjoint_policy: ElectrothermalAdjointPolicy,
    *,
    axis_name: str = "partition",
    linear_preconditioner_factory: PackedScalarH1PreconditionerFactory | None = None,
) -> DistributedElectrothermalRuntime:
    """Build block forward solves and one residual-defined coupled transpose solve.

    An optional symmetric positive-definite scalar preconditioner changes only Krylov strategy.
    The fixed factory path uses it on the right of the coupled transpose so JAX GMRES tests the
    original residual rather than a differently scaled left-preconditioned residual. Forward and
    reverse admission still use freshly recomputed residuals of the unmodified coupled equations.
    """

    if not isinstance(plan, DistributedElectrothermalPlan):
        raise ContractError("distributed electrothermal runtime requires a prepared plan")
    if not isinstance(cg_policy, ScalarH1CGPolicy):
        raise ContractError("distributed electrothermal runtime requires a scalar CG policy")
    if not isinstance(adjoint_policy, ElectrothermalAdjointPolicy):
        raise ContractError("distributed electrothermal runtime requires an adjoint policy")
    layout = plan.layout
    mapping_shape = (layout.partition_count, layout.cell_capacity, 3)
    owner_shape = (layout.partition_count, layout.owned_dof_capacity)
    cell_shape = (layout.partition_count, layout.cell_capacity)
    current_count = len(plan.current_parameter_names)
    thermal_count = len(plan.thermal_parameter_names)
    feedback_count = len(plan.feedback_parameter_names)
    gather = build_packed_collective_scalar_h1_cell_gather(
        layout,
        mesh,
        axis_name=axis_name,
    )
    assemble_rhs = build_packed_collective_scalar_h1_rhs_assembly(
        layout,
        mesh,
        axis_name=axis_name,
    )
    matvec = build_packed_collective_scalar_h1_matvec(layout, mesh, axis_name=axis_name)
    linear_solve = build_packed_collective_scalar_h1_cg(
        layout,
        mesh,
        cg_policy,
        axis_name=axis_name,
        preconditioner_factory=linear_preconditioner_factory,
    )
    global_dot = build_packed_scalar_h1_owner_dot(layout, mesh, axis_name=axis_name)
    cell_spec = P(axis_name, None)  # type: ignore[no-untyped-call]
    replicated = P()  # type: ignore[no-untyped-call]

    @jax.shard_map(
        mesh=mesh,
        in_specs=(cell_spec, cell_spec),
        out_specs=replicated,
        check_vma=True,
    )
    def cell_sum(values: jax.Array, active_cells: jax.Array) -> jax.Array:
        local = jnp.sum(jnp.where(active_cells[0], values[0], 0.0))
        return cast(jax.Array, lax.psum(local, axis_name))  # type: ignore[no-untyped-call]

    def validate_inputs(inputs: PackedDistributedElectrothermalInputs) -> None:
        if not isinstance(inputs, PackedDistributedElectrothermalInputs):
            raise ContractError("distributed electrothermal inputs must use the packed contract")
        if (
            inputs.cell_local_dofs.shape != mapping_shape
            or inputs.owner_mask.shape != owner_shape
            or inputs.cell_mask.shape != cell_shape
        ):
            raise ValueError("distributed electrothermal topology inputs disagree with the plan")
        expected_shapes = (
            (inputs.unit_stiffness, (*cell_shape, 3, 3)),
            (inputs.basis_gradients, (*cell_shape, 3, 2)),
            (inputs.cell_areas, cell_shape),
            (inputs.current_conductivity_base, cell_shape),
            (inputs.current_conductivity_weights, (*cell_shape, current_count)),
            (inputs.current_cell_load_base, (*cell_shape, 3)),
            (inputs.current_cell_load_weights, (*cell_shape, 3, current_count)),
            (inputs.current_dirichlet_base, (*cell_shape, 3)),
            (inputs.current_dirichlet_weights, (*cell_shape, 3, current_count)),
            (inputs.current_reference_base, ()),
            (inputs.current_reference_weights, (current_count,)),
            (inputs.thermal_conductivity_base, cell_shape),
            (inputs.thermal_conductivity_weights, (*cell_shape, thermal_count)),
            (inputs.thermal_cell_load_base, (*cell_shape, 3)),
            (inputs.thermal_cell_load_weights, (*cell_shape, 3, thermal_count)),
            (inputs.thermal_dirichlet_base, (*cell_shape, 3)),
            (inputs.thermal_dirichlet_weights, (*cell_shape, 3, thermal_count)),
            (inputs.thermal_reference_base, ()),
            (inputs.thermal_reference_weights, (thermal_count,)),
            (inputs.feedback_reference_base, (*cell_shape, 3)),
            (inputs.feedback_reference_weights, (*cell_shape, 3, feedback_count)),
            (inputs.feedback_coefficient_base, (*cell_shape, 3)),
            (inputs.feedback_coefficient_weights, (*cell_shape, 3, feedback_count)),
            (inputs.current_lower_bounds, (current_count,)),
            (inputs.current_upper_bounds, (current_count,)),
            (inputs.thermal_lower_bounds, (thermal_count,)),
            (inputs.thermal_upper_bounds, (thermal_count,)),
            (inputs.feedback_lower_bounds, (feedback_count,)),
            (inputs.feedback_upper_bounds, (feedback_count,)),
        )
        if any(value.shape != expected for value, expected in expected_shapes):
            raise ValueError("distributed electrothermal numerical inputs disagree with the plan")
        numeric = tuple(inputs)[3:]
        if not all(jnp.issubdtype(value.dtype, jnp.floating) for value in numeric):
            raise TypeError("distributed electrothermal numerical inputs must be real floating")
        dtype = numeric[0].dtype
        if not all(value.dtype == dtype for value in numeric):
            raise TypeError("distributed electrothermal numerical inputs must share one dtype")
        if not jnp.issubdtype(inputs.cell_local_dofs.dtype, jnp.integer):
            raise TypeError("distributed electrothermal cell map must use an integer dtype")
        if inputs.owner_mask.dtype != jnp.bool_ or inputs.cell_mask.dtype != jnp.bool_:
            raise TypeError("distributed electrothermal activity masks must be boolean")

    def validate_parameters(
        inputs: PackedDistributedElectrothermalInputs,
        current_values: jax.Array,
        thermal_values: jax.Array,
        feedback_values: jax.Array,
    ) -> None:
        numeric_dtype = inputs.unit_stiffness.dtype
        expected = (
            (current_values, (current_count,), "current"),
            (thermal_values, (thermal_count,), "thermal"),
            (feedback_values, (feedback_count,), "feedback"),
        )
        for values, shape, label in expected:
            if values.shape != shape:
                raise ContractError(
                    f"distributed electrothermal {label} parameters must have shape {shape}"
                )
            if values.dtype != numeric_dtype:
                raise ContractError(
                    f"distributed electrothermal {label} parameters must match input dtype"
                )

    def owner_norm(vector: jax.Array, mask: jax.Array) -> jax.Array:
        return jnp.sqrt(jnp.maximum(global_dot(vector, vector, mask), 0.0))

    def state_norm(vector: PackedElectrothermalVector, mask: jax.Array) -> jax.Array:
        squared = global_dot(vector.potential, vector.potential, mask) + global_dot(
            vector.temperature,
            vector.temperature,
            mask,
        )
        return jnp.sqrt(jnp.maximum(squared, 0.0))

    def resolve(
        inputs: PackedDistributedElectrothermalInputs,
        current_values: jax.Array,
        thermal_values: jax.Array,
        feedback_values: jax.Array,
    ) -> _ResolvedFields:
        current_conductivity = _affine(
            inputs.current_conductivity_base,
            inputs.current_conductivity_weights,
            current_values,
        )
        current_cell_load = _affine(
            inputs.current_cell_load_base,
            inputs.current_cell_load_weights,
            current_values,
        )
        current_dirichlet = _affine(
            inputs.current_dirichlet_base,
            inputs.current_dirichlet_weights,
            current_values,
        )
        current_reference = _affine(
            inputs.current_reference_base,
            inputs.current_reference_weights,
            current_values,
        )
        thermal_conductivity = _affine(
            inputs.thermal_conductivity_base,
            inputs.thermal_conductivity_weights,
            thermal_values,
        )
        thermal_cell_load = _affine(
            inputs.thermal_cell_load_base,
            inputs.thermal_cell_load_weights,
            thermal_values,
        )
        thermal_dirichlet = _affine(
            inputs.thermal_dirichlet_base,
            inputs.thermal_dirichlet_weights,
            thermal_values,
        )
        thermal_reference = _affine(
            inputs.thermal_reference_base,
            inputs.thermal_reference_weights,
            thermal_values,
        )
        feedback_reference = _affine(
            inputs.feedback_reference_base,
            inputs.feedback_reference_weights,
            feedback_values,
        )
        feedback_coefficient = _affine(
            inputs.feedback_coefficient_base,
            inputs.feedback_coefficient_weights,
            feedback_values,
        )
        parameter_valid = (
            jnp.all(jnp.isfinite(current_values))
            & jnp.all(current_values >= inputs.current_lower_bounds)
            & jnp.all(current_values <= inputs.current_upper_bounds)
            & jnp.all(jnp.isfinite(thermal_values))
            & jnp.all(thermal_values >= inputs.thermal_lower_bounds)
            & jnp.all(thermal_values <= inputs.thermal_upper_bounds)
            & jnp.all(jnp.isfinite(feedback_values))
            & jnp.all(feedback_values >= inputs.feedback_lower_bounds)
            & jnp.all(feedback_values <= inputs.feedback_upper_bounds)
        )
        fields = (
            current_conductivity,
            current_cell_load,
            current_dirichlet,
            thermal_conductivity,
            thermal_cell_load,
            thermal_dirichlet,
            feedback_reference,
            feedback_coefficient,
        )
        field_valid = jnp.all(
            jnp.stack(
                tuple(jnp.all(jnp.isfinite(value)) for value in fields),
            )
        )
        positive = jnp.all(jnp.where(inputs.cell_mask, current_conductivity > 0.0, True)) & jnp.all(
            jnp.where(inputs.cell_mask, thermal_conductivity > 0.0, True)
        )
        return _ResolvedFields(
            current_conductivity=current_conductivity,
            current_cell_load=current_cell_load,
            current_dirichlet=current_dirichlet,
            current_reference=current_reference,
            thermal_conductivity=thermal_conductivity,
            thermal_cell_load=thermal_cell_load,
            thermal_dirichlet=thermal_dirichlet,
            thermal_reference=thermal_reference,
            feedback_reference=feedback_reference,
            feedback_coefficient=feedback_coefficient,
            valid=parameter_valid & field_valid & positive,
        )

    def reduced_cell_rhs(
        inputs: PackedDistributedElectrothermalInputs,
        stiffness: jax.Array,
        cell_load: jax.Array,
        cell_dirichlet: jax.Array,
    ) -> jax.Array:
        local = cell_load - jnp.einsum("pcij,pcj->pci", stiffness, cell_dirichlet)
        free_rows = inputs.cell_local_dofs < layout.transport.constrained_transport_sentinel
        return jnp.where(free_rows & inputs.cell_mask[:, :, None], local, 0.0)

    def current_operator(
        inputs: PackedDistributedElectrothermalInputs,
        resolved: _ResolvedFields,
        temperature: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        cell_temperature = gather(inputs.cell_local_dofs, temperature) + resolved.thermal_dirichlet
        denominator = 1.0 + resolved.feedback_coefficient * (
            cell_temperature - resolved.feedback_reference
        )
        local_conductivity = resolved.current_conductivity[:, :, None] / denominator
        mean_conductivity = jnp.mean(local_conductivity, axis=2)
        stiffness = inputs.unit_stiffness * mean_conductivity[:, :, None, None]
        cell_rhs = reduced_cell_rhs(
            inputs,
            stiffness,
            resolved.current_cell_load,
            resolved.current_dirichlet,
        )
        rhs = assemble_rhs(cell_rhs, inputs.cell_local_dofs)
        valid = (
            resolved.valid
            & jnp.all(jnp.where(inputs.cell_mask[:, :, None], denominator > 0.0, True))
            & jnp.all(
                jnp.where(inputs.cell_mask[:, :, None], jnp.isfinite(local_conductivity), True)
            )
        )
        return stiffness, rhs, local_conductivity, valid

    def heat_operator(
        inputs: PackedDistributedElectrothermalInputs,
        resolved: _ResolvedFields,
        potential: jax.Array,
        local_conductivity: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        cell_potential = gather(inputs.cell_local_dofs, potential) + resolved.current_dirichlet
        gradient = jnp.einsum("pci,pcid->pcd", cell_potential, inputs.basis_gradients)
        norm_squared = jnp.einsum("pcd,pcd->pc", gradient, gradient)
        cell_nodal_joule = local_conductivity * norm_squared[:, :, None]
        joule_load = (inputs.cell_areas[:, :, None] / 12.0) * (
            jnp.sum(cell_nodal_joule, axis=2, keepdims=True) + cell_nodal_joule
        )
        heat_cell_load = resolved.thermal_cell_load + joule_load
        stiffness = inputs.unit_stiffness * resolved.thermal_conductivity[:, :, None, None]
        cell_rhs = reduced_cell_rhs(
            inputs,
            stiffness,
            heat_cell_load,
            resolved.thermal_dirichlet,
        )
        rhs = assemble_rhs(cell_rhs, inputs.cell_local_dofs)
        valid = jnp.all(
            jnp.where(
                inputs.cell_mask[:, :, None],
                jnp.isfinite(cell_nodal_joule) & (cell_nodal_joule >= 0.0),
                True,
            )
        )
        return stiffness, rhs, cell_nodal_joule, heat_cell_load, valid

    def operators(
        inputs: PackedDistributedElectrothermalInputs,
        state: PackedElectrothermalVector,
        current_values: jax.Array,
        thermal_values: jax.Array,
        feedback_values: jax.Array,
    ) -> _CoupledOperators:
        resolved = resolve(inputs, current_values, thermal_values, feedback_values)
        current_stiffness, current_rhs, conductivity, current_valid = current_operator(
            inputs,
            resolved,
            state.temperature,
        )
        heat_stiffness, heat_rhs, joule, heat_cell_load, heat_valid = heat_operator(
            inputs,
            resolved,
            state.potential,
            conductivity,
        )
        return _CoupledOperators(
            current_stiffness=current_stiffness,
            current_rhs=current_rhs,
            heat_stiffness=heat_stiffness,
            heat_rhs=heat_rhs,
            cell_nodal_conductivity=conductivity,
            cell_nodal_joule=joule,
            heat_cell_load=heat_cell_load,
            valid=current_valid & heat_valid,
        )

    def residual(
        inputs: PackedDistributedElectrothermalInputs,
        state: PackedElectrothermalVector,
        current_values: jax.Array,
        thermal_values: jax.Array,
        feedback_values: jax.Array,
    ) -> PackedElectrothermalVector:
        system = operators(inputs, state, current_values, thermal_values, feedback_values)
        mask = inputs.owner_mask
        current_residual = (
            matvec(
                system.current_stiffness,
                inputs.cell_local_dofs,
                state.potential,
            )
            - system.current_rhs
        )
        heat_residual = (
            matvec(
                system.heat_stiffness,
                inputs.cell_local_dofs,
                state.temperature,
            )
            - system.heat_rhs
        )
        return PackedElectrothermalVector(
            potential=jnp.where(mask, current_residual, 0.0),
            temperature=jnp.where(mask, heat_residual, 0.0),
        )

    def relative_residual(
        inputs: PackedDistributedElectrothermalInputs,
        stiffness: jax.Array,
        cell_load: jax.Array,
        dirichlet: jax.Array,
        reference: jax.Array,
        shifted_state: jax.Array,
    ) -> jax.Array:
        constrained = inputs.cell_local_dofs == layout.transport.constrained_transport_sentinel
        shifted_dirichlet = jnp.where(
            constrained & inputs.cell_mask[:, :, None],
            dirichlet - reference,
            0.0,
        )
        shifted_cell_rhs = reduced_cell_rhs(
            inputs,
            stiffness,
            cell_load,
            shifted_dirichlet,
        )
        shifted_rhs = assemble_rhs(shifted_cell_rhs, inputs.cell_local_dofs)
        action = matvec(stiffness, inputs.cell_local_dofs, shifted_state)
        residual_value = action - shifted_rhs
        return _relative_difference(
            owner_norm(residual_value, inputs.owner_mask),
            owner_norm(action, inputs.owner_mask),
            owner_norm(shifted_rhs, inputs.owner_mask),
        )

    def shifted_linear_solve(
        inputs: PackedDistributedElectrothermalInputs,
        stiffness: jax.Array,
        cell_load: jax.Array,
        dirichlet: jax.Array,
        reference: jax.Array,
    ) -> PackedScalarH1CGResult:
        """Solve for an owner-authoritative reference-relative field.

        Subtracting one constant reference from every nodal value leaves the P1 diffusion
        equation unchanged.  It avoids forming a reduced right-hand side as the difference of
        large absolute-temperature terms in float32.  Restoration happens only at the public
        result boundary so the iteration and convergence diagnostics retain the shifted precision.
        """

        constrained = inputs.cell_local_dofs == layout.transport.constrained_transport_sentinel
        shifted_dirichlet = jnp.where(
            constrained & inputs.cell_mask[:, :, None],
            dirichlet - reference,
            0.0,
        )
        shifted_cell_rhs = reduced_cell_rhs(
            inputs,
            stiffness,
            cell_load,
            shifted_dirichlet,
        )
        shifted_rhs = assemble_rhs(shifted_cell_rhs, inputs.cell_local_dofs)
        return linear_solve(
            stiffness,
            inputs.cell_local_dofs,
            inputs.owner_mask,
            shifted_rhs,
        )

    def restore_state(
        inputs: PackedDistributedElectrothermalInputs,
        shifted: PackedElectrothermalVector,
        resolved: _ResolvedFields,
    ) -> PackedElectrothermalVector:
        return PackedElectrothermalVector(
            jnp.where(inputs.owner_mask, shifted.potential + resolved.current_reference, 0.0),
            jnp.where(inputs.owner_mask, shifted.temperature + resolved.thermal_reference, 0.0),
        )

    def solve_forward(
        inputs: PackedDistributedElectrothermalInputs,
        current_values: jax.Array,
        thermal_values: jax.Array,
        feedback_values: jax.Array,
    ) -> PackedDistributedElectrothermalResult:
        validate_inputs(inputs)
        validate_parameters(inputs, current_values, thermal_values, feedback_values)
        resolved = resolve(inputs, current_values, thermal_values, feedback_values)
        heat_stiffness = inputs.unit_stiffness * resolved.thermal_conductivity[:, :, None, None]
        initial_heat = shifted_linear_solve(
            inputs,
            heat_stiffness,
            resolved.thermal_cell_load,
            resolved.thermal_dirichlet,
            resolved.thermal_reference,
        )
        current_stiffness, _current_rhs, _conductivity, initial_current_valid = current_operator(
            inputs,
            resolved,
            jnp.where(
                inputs.owner_mask,
                initial_heat.solution + resolved.thermal_reference,
                0.0,
            ),
        )
        initial_current = shifted_linear_solve(
            inputs,
            current_stiffness,
            resolved.current_cell_load,
            resolved.current_dirichlet,
            resolved.current_reference,
        )
        initial_state = PackedElectrothermalVector(
            potential=initial_current.solution,
            temperature=initial_heat.solution,
        )
        policy = plan.iteration_policy

        class LoopState(NamedTuple):
            iteration: jax.Array
            state: PackedElectrothermalVector
            update_error: jax.Array
            current_error: jax.Array
            heat_error: jax.Array
            current_linear_iterations: jax.Array
            heat_linear_iterations: jax.Array
            current_linear_recursive_residual: jax.Array
            heat_linear_recursive_residual: jax.Array
            current_linear_recomputed_residual: jax.Array
            heat_linear_recomputed_residual: jax.Array
            current_linear_relative_residual: jax.Array
            heat_linear_relative_residual: jax.Array
            current_linear_backward_error: jax.Array
            heat_linear_backward_error: jax.Array
            current_linear_converged: jax.Array
            heat_linear_converged: jax.Array
            current_linear_breakdown: jax.Array
            heat_linear_breakdown: jax.Array
            valid: jax.Array

        def condition(loop: LoopState) -> jax.Array:
            return (
                loop.valid
                & (loop.iteration < policy.max_iterations)
                & (
                    (loop.iteration < policy.minimum_iterations)
                    | (loop.update_error > 1.0)
                    | (loop.current_error > policy.residual_tolerance)
                    | (loop.heat_error > policy.residual_tolerance)
                )
            )

        def body(loop: LoopState) -> LoopState:
            old = loop.state
            old_absolute = restore_state(inputs, old, resolved)
            current_k, _current_b, conductivity, current_valid = current_operator(
                inputs,
                resolved,
                old_absolute.temperature,
            )
            current_result = shifted_linear_solve(
                inputs,
                current_k,
                resolved.current_cell_load,
                resolved.current_dirichlet,
                resolved.current_reference,
            )
            shifted_potential = old.potential + policy.potential_relaxation * (
                current_result.solution - old.potential
            )
            potential = jnp.where(
                inputs.owner_mask,
                shifted_potential + resolved.current_reference,
                0.0,
            )
            heat_k, _heat_b, _joule, heat_cell_load, heat_valid = heat_operator(
                inputs,
                resolved,
                potential,
                conductivity,
            )
            heat_result = shifted_linear_solve(
                inputs,
                heat_k,
                heat_cell_load,
                resolved.thermal_dirichlet,
                resolved.thermal_reference,
            )
            shifted_temperature = old.temperature + policy.temperature_relaxation * (
                heat_result.solution - old.temperature
            )
            potential_scale = policy.potential_absolute_tolerance + policy.relative_tolerance * (
                owner_norm(shifted_potential, inputs.owner_mask)
            )
            temperature_scale = (
                policy.temperature_absolute_tolerance
                + policy.relative_tolerance * owner_norm(shifted_temperature, inputs.owner_mask)
            )
            update_error = jnp.maximum(
                owner_norm(shifted_potential - old.potential, inputs.owner_mask) / potential_scale,
                owner_norm(shifted_temperature - old.temperature, inputs.owner_mask)
                / temperature_scale,
            )
            new_state = PackedElectrothermalVector(shifted_potential, shifted_temperature)
            temperature = jnp.where(
                inputs.owner_mask,
                shifted_temperature + resolved.thermal_reference,
                0.0,
            )
            current_k, _current_b, conductivity, current_valid = current_operator(
                inputs,
                resolved,
                temperature,
            )
            heat_k, _heat_b, joule, _heat_cell_load, heat_valid = heat_operator(
                inputs,
                resolved,
                potential,
                conductivity,
            )
            current_error = relative_residual(
                inputs,
                current_k,
                resolved.current_cell_load,
                resolved.current_dirichlet,
                resolved.current_reference,
                shifted_potential,
            )
            joule_load = (inputs.cell_areas[:, :, None] / 12.0) * (
                jnp.sum(joule, axis=2, keepdims=True) + joule
            )
            heat_error = relative_residual(
                inputs,
                heat_k,
                resolved.thermal_cell_load + joule_load,
                resolved.thermal_dirichlet,
                resolved.thermal_reference,
                shifted_temperature,
            )
            valid = (
                loop.valid
                & current_valid
                & heat_valid
                & current_result.converged
                & heat_result.converged
            )
            return LoopState(
                iteration=loop.iteration + 1,
                state=new_state,
                update_error=update_error,
                current_error=current_error,
                heat_error=heat_error,
                current_linear_iterations=jnp.maximum(
                    loop.current_linear_iterations,
                    current_result.iterations,
                ),
                heat_linear_iterations=jnp.maximum(
                    loop.heat_linear_iterations,
                    heat_result.iterations,
                ),
                current_linear_recursive_residual=current_result.recursive_residual_norm,
                heat_linear_recursive_residual=heat_result.recursive_residual_norm,
                current_linear_recomputed_residual=current_result.recomputed_residual_norm,
                heat_linear_recomputed_residual=heat_result.recomputed_residual_norm,
                current_linear_relative_residual=current_result.relative_residual,
                heat_linear_relative_residual=heat_result.relative_residual,
                current_linear_backward_error=current_result.backward_error,
                heat_linear_backward_error=heat_result.backward_error,
                current_linear_converged=current_result.converged,
                heat_linear_converged=heat_result.converged,
                current_linear_breakdown=current_result.breakdown,
                heat_linear_breakdown=heat_result.breakdown,
                valid=valid,
            )

        loop = lax.while_loop(
            condition,
            body,
            LoopState(
                iteration=jnp.asarray(0, dtype=jnp.int32),
                state=initial_state,
                update_error=jnp.asarray(jnp.inf, dtype=inputs.unit_stiffness.dtype),
                current_error=jnp.asarray(jnp.inf, dtype=inputs.unit_stiffness.dtype),
                heat_error=jnp.asarray(jnp.inf, dtype=inputs.unit_stiffness.dtype),
                current_linear_iterations=initial_current.iterations,
                heat_linear_iterations=initial_heat.iterations,
                current_linear_recursive_residual=initial_current.recursive_residual_norm,
                heat_linear_recursive_residual=initial_heat.recursive_residual_norm,
                current_linear_recomputed_residual=initial_current.recomputed_residual_norm,
                heat_linear_recomputed_residual=initial_heat.recomputed_residual_norm,
                current_linear_relative_residual=initial_current.relative_residual,
                heat_linear_relative_residual=initial_heat.relative_residual,
                current_linear_backward_error=initial_current.backward_error,
                heat_linear_backward_error=initial_heat.backward_error,
                current_linear_converged=initial_current.converged,
                heat_linear_converged=initial_heat.converged,
                current_linear_breakdown=initial_current.breakdown,
                heat_linear_breakdown=initial_heat.breakdown,
                valid=(
                    resolved.valid
                    & initial_current_valid
                    & initial_current.converged
                    & initial_heat.converged
                ),
            ),
        )
        final_state = restore_state(inputs, loop.state, resolved)
        final = operators(
            inputs,
            final_state,
            current_values,
            thermal_values,
            feedback_values,
        )
        current_error = relative_residual(
            inputs,
            final.current_stiffness,
            resolved.current_cell_load,
            resolved.current_dirichlet,
            resolved.current_reference,
            loop.state.potential,
        )
        heat_error = relative_residual(
            inputs,
            final.heat_stiffness,
            final.heat_cell_load,
            resolved.thermal_dirichlet,
            resolved.thermal_reference,
            loop.state.temperature,
        )
        joule_power = cell_sum(
            inputs.cell_areas * jnp.mean(final.cell_nodal_joule, axis=2),
            inputs.cell_mask,
        )
        thermal_joule_load = cell_sum(
            jnp.sum(
                (inputs.cell_areas[:, :, None] / 12.0)
                * (jnp.sum(final.cell_nodal_joule, axis=2, keepdims=True) + final.cell_nodal_joule),
                axis=2,
            ),
            inputs.cell_mask,
        )
        transfer_difference = jnp.abs(joule_power - thermal_joule_load)
        transfer_error = _relative_difference(
            transfer_difference,
            jnp.abs(joule_power),
            jnp.abs(thermal_joule_load),
        )
        finite_state = jnp.all(
            jnp.stack(
                (
                    jnp.all(jnp.isfinite(final_state.potential)),
                    jnp.all(jnp.isfinite(final_state.temperature)),
                )
            )
        )
        converged = (
            loop.valid
            & final.valid
            & finite_state
            & (loop.iteration >= policy.minimum_iterations)
            & (loop.update_error <= 1.0)
            & (current_error <= policy.residual_tolerance)
            & (heat_error <= policy.residual_tolerance)
        )
        return PackedDistributedElectrothermalResult(
            state=final_state,
            cell_nodal_conductivity=final.cell_nodal_conductivity,
            cell_nodal_joule=final.cell_nodal_joule,
            iterations=loop.iteration,
            update_error=loop.update_error,
            current_residual_error=current_error,
            heat_residual_error=heat_error,
            current_linear_iterations=loop.current_linear_iterations,
            heat_linear_iterations=loop.heat_linear_iterations,
            current_linear_recursive_residual=loop.current_linear_recursive_residual,
            heat_linear_recursive_residual=loop.heat_linear_recursive_residual,
            current_linear_recomputed_residual=loop.current_linear_recomputed_residual,
            heat_linear_recomputed_residual=loop.heat_linear_recomputed_residual,
            current_linear_relative_residual=loop.current_linear_relative_residual,
            heat_linear_relative_residual=loop.heat_linear_relative_residual,
            current_linear_backward_error=loop.current_linear_backward_error,
            heat_linear_backward_error=loop.heat_linear_backward_error,
            current_linear_converged=loop.current_linear_converged,
            heat_linear_converged=loop.heat_linear_converged,
            current_linear_breakdown=loop.current_linear_breakdown,
            heat_linear_breakdown=loop.heat_linear_breakdown,
            electrical_joule_power=joule_power,
            thermal_joule_load=thermal_joule_load,
            transfer_relative_error=transfer_error,
            converged=converged,
        )

    def implicit_pullback(
        inputs: PackedDistributedElectrothermalInputs,
        state: PackedElectrothermalVector,
        current_values: jax.Array,
        thermal_values: jax.Array,
        feedback_values: jax.Array,
        cotangent: PackedElectrothermalVector,
    ) -> tuple[PackedElectrothermalVector, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        stopped_state = jax.tree.map(lax.stop_gradient, state)
        stopped_cotangent = PackedElectrothermalVector(
            jnp.where(inputs.owner_mask, cotangent.potential, 0.0),
            jnp.where(inputs.owner_mask, cotangent.temperature, 0.0),
        )

        def state_residual(candidate: PackedElectrothermalVector) -> PackedElectrothermalVector:
            return residual(
                inputs,
                candidate,
                current_values,
                thermal_values,
                feedback_values,
            )

        _, state_pullback = jax.vjp(state_residual, stopped_state)

        def transpose_action(vector: PackedElectrothermalVector) -> PackedElectrothermalVector:
            return cast(PackedElectrothermalVector, state_pullback(vector)[0])

        diagonal = operators(
            inputs,
            stopped_state,
            current_values,
            thermal_values,
            feedback_values,
        )
        preconditioner_valid = jnp.asarray(True)
        if linear_preconditioner_factory is None:
            current_block_inverse = None
            thermal_block_inverse = None
        else:
            current_block_inverse = linear_preconditioner_factory(
                diagonal.current_stiffness,
                inputs.cell_local_dofs,
                inputs.owner_mask,
            )
            thermal_block_inverse = linear_preconditioner_factory(
                diagonal.heat_stiffness,
                inputs.cell_local_dofs,
                inputs.owner_mask,
            )

        def block_preconditioner(
            vector: PackedElectrothermalVector,
        ) -> PackedElectrothermalVector:
            """Apply the inverse uncoupled scalar blocks to the coupled transpose iterate."""

            if current_block_inverse is not None and thermal_block_inverse is not None:
                return PackedElectrothermalVector(
                    current_block_inverse(vector.potential),
                    thermal_block_inverse(vector.temperature),
                )
            current = linear_solve(
                diagonal.current_stiffness,
                inputs.cell_local_dofs,
                inputs.owner_mask,
                vector.potential,
            )
            thermal = linear_solve(
                diagonal.heat_stiffness,
                inputs.cell_local_dofs,
                inputs.owner_mask,
                vector.temperature,
            )
            return PackedElectrothermalVector(current.solution, thermal.solution)

        if linear_preconditioner_factory is None:
            adjoint, info = jax.scipy.sparse.linalg.gmres(  # type: ignore[no-untyped-call]
                transpose_action,
                stopped_cotangent,
                tol=adjoint_policy.relative_tolerance,
                atol=adjoint_policy.absolute_tolerance,
                restart=adjoint_policy.restart,
                maxiter=adjoint_policy.max_restarts,
                M=block_preconditioner,
                solve_method="batched",
            )
        else:
            # The fail-closed factory wrapper contains finiteness guards. Linearizing it at zero
            # extracts the fixed inverse action required by ordinary right-preconditioned GMRES;
            # a valid linear preconditioner must map zero to finite exact zero.
            zero = jax.tree.map(jnp.zeros_like, stopped_cotangent)
            preconditioner_at_zero, fixed_block_preconditioner = jax.linearize(
                block_preconditioner,
                zero,
            )
            preconditioner_valid = jnp.all(
                jnp.stack(
                    (
                        jnp.all(jnp.isfinite(preconditioner_at_zero.potential)),
                        jnp.all(jnp.isfinite(preconditioner_at_zero.temperature)),
                        jnp.all(preconditioner_at_zero.potential == 0.0),
                        jnp.all(preconditioner_at_zero.temperature == 0.0),
                    )
                )
            )

            def right_preconditioned_action(
                vector: PackedElectrothermalVector,
            ) -> PackedElectrothermalVector:
                return transpose_action(fixed_block_preconditioner(vector))

            coordinate, info = jax.scipy.sparse.linalg.gmres(  # type: ignore[no-untyped-call]
                right_preconditioned_action,
                stopped_cotangent,
                tol=adjoint_policy.relative_tolerance,
                atol=adjoint_policy.absolute_tolerance,
                restart=adjoint_policy.restart,
                maxiter=adjoint_policy.max_restarts,
                solve_method="batched",
            )
            adjoint = fixed_block_preconditioner(coordinate)
        adjoint_residual = jax.tree.map(
            lambda observed, expected: observed - expected,
            transpose_action(adjoint),
            stopped_cotangent,
        )
        residual_norm = state_norm(adjoint_residual, inputs.owner_mask)
        action_norm = state_norm(transpose_action(adjoint), inputs.owner_mask)
        cotangent_norm = state_norm(stopped_cotangent, inputs.owner_mask)
        backward_error = _relative_difference(
            residual_norm,
            action_norm,
            cotangent_norm,
        )
        target = jnp.maximum(
            jnp.asarray(adjoint_policy.absolute_tolerance, dtype=residual_norm.dtype),
            jnp.asarray(adjoint_policy.relative_tolerance, dtype=residual_norm.dtype)
            * cotangent_norm,
        )
        converged = (info == 0) & jnp.isfinite(residual_norm) & (residual_norm <= target)
        stopped_adjoint = jax.tree.map(lax.stop_gradient, adjoint)

        def parameter_residual(
            current_candidate: jax.Array,
            thermal_candidate: jax.Array,
            feedback_candidate: jax.Array,
        ) -> PackedElectrothermalVector:
            return residual(
                inputs,
                stopped_state,
                current_candidate,
                thermal_candidate,
                feedback_candidate,
            )

        _, parameter_pullback = jax.vjp(
            parameter_residual,
            current_values,
            thermal_values,
            feedback_values,
        )
        current_gradient, thermal_gradient, feedback_gradient = parameter_pullback(stopped_adjoint)
        valid = (
            converged
            & preconditioner_valid
            & jnp.all(
                jnp.stack(
                    (
                        jnp.all(jnp.isfinite(adjoint.potential)),
                        jnp.all(jnp.isfinite(adjoint.temperature)),
                    )
                )
            )
        )
        return (
            PackedElectrothermalVector(
                jnp.where(valid, adjoint.potential, jnp.nan),
                jnp.where(valid, adjoint.temperature, jnp.nan),
            ),
            jnp.where(valid, -current_gradient, jnp.nan),
            jnp.where(valid, -thermal_gradient, jnp.nan),
            jnp.where(valid, -feedback_gradient, jnp.nan),
            jnp.where(valid, backward_error, jnp.nan),
            valid,
        )

    def differentiable_state(
        inputs: PackedDistributedElectrothermalInputs,
        current_values: jax.Array,
        thermal_values: jax.Array,
        feedback_values: jax.Array,
    ) -> PackedElectrothermalVector:
        validate_inputs(inputs)
        validate_parameters(inputs, current_values, thermal_values, feedback_values)

        def primal(
            current_candidate: jax.Array,
            thermal_candidate: jax.Array,
            feedback_candidate: jax.Array,
        ) -> PackedElectrothermalVector:
            forward = solve_forward(
                inputs,
                current_candidate,
                thermal_candidate,
                feedback_candidate,
            )
            invalid_potential = jnp.full_like(forward.state.potential, jnp.nan)
            invalid_temperature = jnp.full_like(forward.state.temperature, jnp.nan)
            return PackedElectrothermalVector(
                jnp.where(forward.converged, forward.state.potential, invalid_potential),
                jnp.where(forward.converged, forward.state.temperature, invalid_temperature),
            )

        solve_map = jax.custom_vjp(primal)

        def forward_rule(
            current_candidate: jax.Array,
            thermal_candidate: jax.Array,
            feedback_candidate: jax.Array,
        ) -> tuple[
            PackedElectrothermalVector,
            tuple[PackedElectrothermalVector, jax.Array, jax.Array, jax.Array],
        ]:
            state = primal(current_candidate, thermal_candidate, feedback_candidate)
            return state, (
                state,
                current_candidate,
                thermal_candidate,
                feedback_candidate,
            )

        def backward_rule(
            saved: tuple[PackedElectrothermalVector, jax.Array, jax.Array, jax.Array],
            cotangent: PackedElectrothermalVector,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            state, current_candidate, thermal_candidate, feedback_candidate = saved
            _adjoint, current_gradient, thermal_gradient, feedback_gradient, _error, _valid = (
                implicit_pullback(
                    inputs,
                    state,
                    current_candidate,
                    thermal_candidate,
                    feedback_candidate,
                    cotangent,
                )
            )
            return current_gradient, thermal_gradient, feedback_gradient

        solve_map.defvjp(forward_rule, backward_rule)
        return solve_map(current_values, thermal_values, feedback_values)

    def cell_temperature(
        inputs: PackedDistributedElectrothermalInputs,
        state: PackedElectrothermalVector,
        thermal_values: jax.Array,
    ) -> jax.Array:
        """Return absolute P1 temperatures on each owned cell without a global gather."""

        validate_inputs(inputs)
        if state.potential.shape != owner_shape or state.temperature.shape != owner_shape:
            raise ContractError("distributed electrothermal state must match owner storage")
        if state.potential.dtype != inputs.unit_stiffness.dtype or state.temperature.dtype != (
            inputs.unit_stiffness.dtype
        ):
            raise ContractError("distributed electrothermal state must match input dtype")
        if thermal_values.shape != (thermal_count,):
            raise ContractError(
                f"distributed electrothermal thermal parameters must have shape {(thermal_count,)}"
            )
        if thermal_values.dtype != inputs.unit_stiffness.dtype:
            raise ContractError(
                "distributed electrothermal thermal parameters must match input dtype"
            )
        thermal_dirichlet = _affine(
            inputs.thermal_dirichlet_base,
            inputs.thermal_dirichlet_weights,
            thermal_values,
        )
        gathered = gather(inputs.cell_local_dofs, state.temperature) + thermal_dirichlet
        return jnp.where(inputs.cell_mask[:, :, None], gathered, 0.0)

    def explicit_vjp(
        inputs: PackedDistributedElectrothermalInputs,
        current_values: jax.Array,
        thermal_values: jax.Array,
        feedback_values: jax.Array,
        state_cotangent: PackedElectrothermalVector,
    ) -> PackedDistributedElectrothermalVjpResult:
        validate_inputs(inputs)
        validate_parameters(inputs, current_values, thermal_values, feedback_values)
        if (
            state_cotangent.potential.shape != owner_shape
            or state_cotangent.temperature.shape != owner_shape
        ):
            raise ContractError("distributed electrothermal cotangent must match owner storage")
        if (
            state_cotangent.potential.dtype != inputs.unit_stiffness.dtype
            or state_cotangent.temperature.dtype != inputs.unit_stiffness.dtype
        ):
            raise ContractError("distributed electrothermal cotangent must match input dtype")
        forward = solve_forward(inputs, current_values, thermal_values, feedback_values)
        adjoint, current_gradient, thermal_gradient, feedback_gradient, error, valid = (
            implicit_pullback(
                inputs,
                forward.state,
                current_values,
                thermal_values,
                feedback_values,
                state_cotangent,
            )
        )
        valid = valid & forward.converged
        return PackedDistributedElectrothermalVjpResult(
            forward=forward,
            state_cotangent=state_cotangent,
            coupled_adjoint=PackedElectrothermalVector(
                jnp.where(valid, adjoint.potential, jnp.nan),
                jnp.where(valid, adjoint.temperature, jnp.nan),
            ),
            current_parameter_gradient=jnp.where(valid, current_gradient, jnp.nan),
            thermal_parameter_gradient=jnp.where(valid, thermal_gradient, jnp.nan),
            feedback_parameter_gradient=jnp.where(valid, feedback_gradient, jnp.nan),
            adjoint_backward_error=jnp.where(valid, error, jnp.nan),
            adjoint_converged=valid,
        )

    return DistributedElectrothermalRuntime(
        solve=solve_forward,
        state=differentiable_state,
        cell_temperature=cell_temperature,
        vjp=explicit_vjp,
    )


def reconstruct_distributed_electrothermal_state(
    plan: DistributedElectrothermalPlan,
    packed_state: PackedElectrothermalVector,
    current_values: jax.Array,
    thermal_values: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return canonical full-node fields for bounded validation and artifact writing."""

    from femx.backends.jax.scalar_collective import (  # local to keep this helper explicit
        unpack_collective_scalar_h1_owned_vector,
    )

    current_dirichlet = jnp.asarray(plan.current.node_dirichlet_base) + jnp.tensordot(
        jnp.asarray(plan.current.node_dirichlet_weights),
        current_values,
        axes=((-1,), (0,)),
    )
    thermal_dirichlet = jnp.asarray(plan.thermal.node_dirichlet_base) + jnp.tensordot(
        jnp.asarray(plan.thermal.node_dirichlet_weights),
        thermal_values,
        axes=((-1,), (0,)),
    )
    potential_free = unpack_collective_scalar_h1_owned_vector(
        plan.layout,
        packed_state.potential,
    )
    temperature_free = unpack_collective_scalar_h1_owned_vector(
        plan.layout,
        packed_state.temperature,
    )
    return (
        reconstruct_scalar_h1_state(plan.layout.topology, potential_free, current_dirichlet),
        reconstruct_scalar_h1_state(plan.layout.topology, temperature_free, thermal_dirichlet),
    )
