r"""JAX reconstruction and nodal projection of mixed port eigenvectors.

Topology is prepared on the host and validated before entering the numerical kernel.  The reduced
coefficient ordering is the exact sorted free-DOF ordering produced by
``canonical_mixed_port_dof_partition`` and ``reduce_port_pencil``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from femx.backends.jax.elements.triangle_nedelec import evaluate_triangle_nedelec1
from femx.backends.jax.operators import triangle_p1_geometry

_DEGREE_TWO_POINTS = (
    (1.0 / 6.0, 1.0 / 6.0),
    (2.0 / 3.0, 1.0 / 6.0),
    (1.0 / 6.0, 2.0 / 3.0),
)
_DEGREE_TWO_WEIGHTS = (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0)


class ExpandedPortEigenmodes(NamedTuple):
    """Full canonical nodal and edge coefficient arrays, including exact PEC zeros."""

    scalar_coefficients: jax.Array
    edge_coefficients: jax.Array


class NodalPortElectricField(NamedTuple):
    """Cartesian P1 L2 projection and its exact consistent nodal mass matrix."""

    values: jax.Array
    nodal_mass: jax.Array
    expanded: ExpandedPortEigenmodes


class NodalPortElectromagneticField(NamedTuple):
    """Physical E/H projections plus native signed forward-power evidence."""

    electric_values: jax.Array
    magnetic_values: jax.Array
    nodal_mass: jax.Array
    raw_forward_power_w: jax.Array
    expanded: ExpandedPortEigenmodes


def _validate_projection_topology(
    coordinates: jax.Array,
    cells: jax.Array,
    cell_edge_dofs: jax.Array,
    cell_edge_signs: jax.Array,
) -> None:
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("port projection coordinates must be shaped (nodes, 2)")
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError("port projection cells must be shaped (cells, 3)")
    if cell_edge_dofs.shape != cells.shape or cell_edge_signs.shape != cells.shape:
        raise ValueError("port projection edge topology must match the triangle cells")


def _projection_quadrature(
    coordinates: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    quadrature_points = jnp.asarray(_DEGREE_TWO_POINTS, dtype=coordinates.dtype)
    quadrature_weights = jnp.asarray(_DEGREE_TWO_WEIGHTS, dtype=coordinates.dtype)
    barycentric = jnp.column_stack(
        (
            1.0 - quadrature_points[:, 0] - quadrature_points[:, 1],
            quadrature_points[:, 0],
            quadrature_points[:, 1],
        )
    )
    return quadrature_points, quadrature_weights, barycentric


def _project_quadrature_vectors_to_nodes(
    samples: jax.Array,
    *,
    cells: jax.Array,
    determinants: jax.Array,
    barycentric: jax.Array,
    quadrature_weights: jax.Array,
    node_count: int,
) -> tuple[jax.Array, jax.Array]:
    weighted = (
        jnp.abs(determinants)[:, None, None, None]
        * quadrature_weights[None, :, None, None]
        * samples
    )
    local_rhs = jnp.einsum("qi,cqmd->cimd", barycentric, weighted)
    rhs = jnp.zeros(
        (node_count, samples.shape[2], samples.shape[3]),
        dtype=samples.dtype,
    )
    rhs = rhs.at[cells.reshape(-1)].add(local_rhs.reshape(-1, samples.shape[2], samples.shape[3]))

    area = 0.5 * jnp.abs(determinants)
    reference_mass = (
        jnp.ones((3, 3), dtype=determinants.dtype) + jnp.eye(3, dtype=determinants.dtype)
    ) / 12.0
    local_mass = area[:, None, None] * reference_mass[None, :, :]
    rows = jnp.repeat(cells, 3, axis=1).reshape(-1)
    columns = jnp.tile(cells, (1, 3)).reshape(-1)
    nodal_mass = jnp.zeros((node_count, node_count), dtype=determinants.dtype)
    nodal_mass = nodal_mass.at[rows, columns].add(local_mass.reshape(-1))
    projected = jnp.linalg.solve(
        nodal_mass,
        rhs.reshape(node_count, -1),
    ).reshape(node_count, samples.shape[2], samples.shape[3])
    return projected, nodal_mass


def _validated_free_dofs(
    free_dofs: object,
    *,
    node_count: int,
    edge_dof_count: int,
    free_scalar_count: int,
    free_edge_count: int,
) -> np.ndarray:
    if isinstance(node_count, bool) or not isinstance(node_count, int) or node_count <= 0:
        raise ValueError("port node_count must be a positive static integer")
    if (
        isinstance(edge_dof_count, bool)
        or not isinstance(edge_dof_count, int)
        or edge_dof_count <= 0
    ):
        raise ValueError("port edge_dof_count must be a positive static integer")
    raw = np.asarray(free_dofs)
    if raw.dtype.kind not in "iu" or raw.ndim != 1:
        raise ValueError("port free_dofs must be a rank-one integer host array")
    indices = np.asarray(raw, dtype=np.int64)
    if indices.size != free_scalar_count + free_edge_count:
        raise ValueError("port free_dofs size does not match the reduced coefficient rows")
    if indices.size == 0 or np.any(indices < 0) or np.any(indices >= node_count + edge_dof_count):
        raise ValueError("port free_dofs contain an out-of-range index")
    if np.any(indices[1:] <= indices[:-1]):
        raise ValueError("port free_dofs must be strictly increasing and unique")
    if not (
        np.all(indices[:free_scalar_count] < node_count)
        and np.all(indices[free_scalar_count:] >= node_count)
    ):
        raise ValueError("port free_dofs do not match the nodal-first reduced layout")
    return indices


def expand_reduced_port_coefficients(
    scalar_coefficients: jax.Array,
    edge_coefficients: jax.Array,
    free_dofs: object,
    *,
    node_count: int,
    edge_dof_count: int,
) -> ExpandedPortEigenmodes:
    """Scatter reduced modes once into canonical full mixed space, then split by family.

    ``free_dofs`` is host topology, not a differentiable argument.  Keeping the scalar offset in
    this one operation prevents accidental reuse of full mixed indices against a reduced edge
    array; JAX gathers otherwise permit such mistakes to produce plausible-looking values.
    """

    if scalar_coefficients.ndim != 2 or edge_coefficients.ndim != 2:
        raise ValueError("reduced port coefficients must be rank-two column matrices")
    if scalar_coefficients.shape[1] != edge_coefficients.shape[1]:
        raise ValueError("reduced scalar and edge coefficients must have the same mode count")
    if scalar_coefficients.dtype != edge_coefficients.dtype:
        raise ValueError("reduced scalar and edge coefficients must have the same dtype")
    indices = _validated_free_dofs(
        free_dofs,
        node_count=node_count,
        edge_dof_count=edge_dof_count,
        free_scalar_count=scalar_coefficients.shape[0],
        free_edge_count=edge_coefficients.shape[0],
    )
    reduced = jnp.concatenate((scalar_coefficients, edge_coefficients), axis=0)
    full = jnp.zeros(
        (node_count + edge_dof_count, reduced.shape[1]),
        dtype=reduced.dtype,
    )
    full = full.at[jnp.asarray(indices)].set(reduced)
    return ExpandedPortEigenmodes(
        scalar_coefficients=full[:node_count],
        edge_coefficients=full[node_count:],
    )


def project_port_electric_field_to_nodes(
    coordinates: jax.Array,
    cells: jax.Array,
    cell_edge_dofs: jax.Array,
    cell_edge_signs: jax.Array,
    scalar_coefficients: jax.Array,
    edge_coefficients: jax.Array,
    propagation_constants_per_m: jax.Array,
    free_dofs: object,
    *,
    edge_dof_count: int,
) -> NodalPortElectricField:
    r"""Reconstruct mixed fields and apply Elmer ``EMPortPost``'s P1 L2 projection.

    For each mode the transverse field is evaluated in the first-family Nedelec basis and the
    longitudinal electric component is :math:`E_z=P_z/(i\beta)`.  Degree-two triangle quadrature
    exactly integrates the affine first-order projection used by the locked Elmer reference.
    """

    _validate_projection_topology(coordinates, cells, cell_edge_dofs, cell_edge_signs)
    if propagation_constants_per_m.ndim != 1:
        raise ValueError("port propagation constants must be rank one")
    if propagation_constants_per_m.shape[0] != scalar_coefficients.shape[1]:
        raise ValueError("port propagation constants must match the coefficient mode count")

    node_count = coordinates.shape[0]
    expanded = expand_reduced_port_coefficients(
        scalar_coefficients,
        edge_coefficients,
        free_dofs,
        node_count=node_count,
        edge_dof_count=edge_dof_count,
    )
    quadrature_points, quadrature_weights, barycentric = _projection_quadrature(coordinates)
    evaluation = evaluate_triangle_nedelec1(
        coordinates,
        cells,
        cell_edge_signs,
        quadrature_points,
    )
    local_edge_coefficients = expanded.edge_coefficients[cell_edge_dofs]
    transverse = jnp.einsum(
        "cem,cqed->cqmd",
        local_edge_coefficients,
        evaluation.basis,
    )
    local_scalar_coefficients = expanded.scalar_coefficients[cells]
    longitudinal_potential = jnp.einsum(
        "qi,cim->cqm",
        barycentric,
        local_scalar_coefficients,
    )
    longitudinal_electric = longitudinal_potential / (
        1j * propagation_constants_per_m[None, None, :]
    )
    electric_field = jnp.concatenate(
        (transverse, longitudinal_electric[..., None]),
        axis=3,
    )
    projected, nodal_mass = _project_quadrature_vectors_to_nodes(
        electric_field,
        cells=cells,
        determinants=evaluation.determinants,
        barycentric=barycentric,
        quadrature_weights=quadrature_weights,
        node_count=node_count,
    )
    return NodalPortElectricField(
        values=projected,
        nodal_mass=nodal_mass,
        expanded=expanded,
    )


def project_port_electromagnetic_fields_to_nodes(
    coordinates: jax.Array,
    cells: jax.Array,
    cell_edge_dofs: jax.Array,
    cell_edge_signs: jax.Array,
    scalar_coefficients: jax.Array,
    edge_coefficients: jax.Array,
    propagation_constants_per_m: jax.Array,
    cell_reluctivity_per_henry_m: jax.Array,
    angular_frequency_rad_per_s: jax.Array,
    free_dofs: object,
    *,
    edge_dof_count: int,
) -> NodalPortElectromagneticField:
    r"""Reconstruct physical E/H, integrate power, and apply Elmer's P1 projection.

    The locked no-potential formulation uses :math:`P_z=i\beta E_z` and positive-z dependence
    :math:`\exp(i\beta z)`.  With the physical reluctivity :math:`\nu=1/\mu`, Maxwell's curl
    equation gives

    .. math::

       H_x &= \frac{\nu}{i\omega}(\partial_y E_z-i\beta E_y),\\
       H_y &= \frac{\nu}{i\omega}(i\beta E_x-\partial_x E_z),\\
       H_z &= \frac{\nu}{i\omega}(\partial_x E_y-\partial_y E_x).

    Signed forward power is evaluated on the native mixed fields as
    :math:`\tfrac12\operatorname{Re}\int(E\times H^*)_z\,dA`.  The returned nodal fields are
    visualization/interoperability precursors; the native quadrature integral remains the power
    authority.
    """

    _validate_projection_topology(coordinates, cells, cell_edge_dofs, cell_edge_signs)
    if propagation_constants_per_m.ndim != 1:
        raise ValueError("port propagation constants must be rank one")
    if (
        cell_reluctivity_per_henry_m.ndim != 1
        or cell_reluctivity_per_henry_m.shape[0] != cells.shape[0]
    ):
        raise ValueError("port cell reluctivity must contain one value per triangle")
    if angular_frequency_rad_per_s.ndim != 0:
        raise ValueError("port angular frequency must be a scalar")

    node_count = coordinates.shape[0]
    expanded = expand_reduced_port_coefficients(
        scalar_coefficients,
        edge_coefficients,
        free_dofs,
        node_count=node_count,
        edge_dof_count=edge_dof_count,
    )
    mode_count = expanded.scalar_coefficients.shape[1]
    if propagation_constants_per_m.shape[0] != mode_count:
        raise ValueError("port propagation constants must match the coefficient mode count")

    quadrature_points, quadrature_weights, barycentric = _projection_quadrature(coordinates)
    evaluation = evaluate_triangle_nedelec1(
        coordinates,
        cells,
        cell_edge_signs,
        quadrature_points,
    )
    local_edge_coefficients = expanded.edge_coefficients[cell_edge_dofs]
    transverse = jnp.einsum(
        "cem,cqed->cqmd",
        local_edge_coefficients,
        evaluation.basis,
    )
    local_scalar_coefficients = expanded.scalar_coefficients[cells]
    longitudinal_potential = jnp.einsum(
        "qi,cim->cqm",
        barycentric,
        local_scalar_coefficients,
    )
    inverse_i_beta = 1.0 / (1j * propagation_constants_per_m)
    longitudinal_electric = longitudinal_potential * inverse_i_beta[None, None, :]
    electric = jnp.concatenate((transverse, longitudinal_electric[..., None]), axis=3)

    _, scalar_gradients = triangle_p1_geometry(coordinates, cells)
    longitudinal_potential_gradient = jnp.einsum(
        "cia,cim->cma",
        scalar_gradients,
        local_scalar_coefficients,
    )
    longitudinal_electric_gradient = longitudinal_potential_gradient * inverse_i_beta[None, :, None]
    transverse_curl = jnp.einsum(
        "cem,cqe->cqm",
        local_edge_coefficients,
        evaluation.curl,
    )
    beta = propagation_constants_per_m[None, None, :]
    prefactor = cell_reluctivity_per_henry_m[:, None, None] / (1j * angular_frequency_rad_per_s)
    magnetic_x = prefactor * (
        longitudinal_electric_gradient[:, None, :, 1] - 1j * beta * transverse[..., 1]
    )
    magnetic_y = prefactor * (
        1j * beta * transverse[..., 0] - longitudinal_electric_gradient[:, None, :, 0]
    )
    magnetic_z = prefactor * transverse_curl
    magnetic = jnp.stack((magnetic_x, magnetic_y, magnetic_z), axis=3)

    signed_flux_density = 0.5 * jnp.real(
        electric[..., 0] * jnp.conj(magnetic[..., 1])
        - electric[..., 1] * jnp.conj(magnetic[..., 0])
    )
    quadrature_measure = jnp.abs(evaluation.determinants)[:, None] * quadrature_weights[None, :]
    raw_forward_power = jnp.einsum("cq,cqm->m", quadrature_measure, signed_flux_density)

    stacked = jnp.concatenate((electric, magnetic), axis=3)
    projected, nodal_mass = _project_quadrature_vectors_to_nodes(
        stacked,
        cells=cells,
        determinants=evaluation.determinants,
        barycentric=barycentric,
        quadrature_weights=quadrature_weights,
        node_count=node_count,
    )
    return NodalPortElectromagneticField(
        electric_values=projected[..., :3],
        magnetic_values=projected[..., 3:],
        nodal_mass=nodal_mass,
        raw_forward_power_w=raw_forward_power,
        expanded=expanded,
    )
