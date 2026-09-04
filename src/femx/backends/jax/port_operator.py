r"""Native JAX mixed port-operator kernels matching locked Elmer ``EMPort``.

The v1 local ordering is three scalar P1 nodal DOFs followed by three lowest-order first-family
Nédélec edge DOFs.  Global ordering is all scalar nodes followed by all canonical edges.  These
kernels assemble Elmer's no-potential generalized pencil only; they do not solve or sort modes.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from femx.backends.jax.elements.triangle_nedelec import (
    evaluate_triangle_nedelec1,
    triangle_nedelec1_local_gram,
)
from femx.backends.jax.operators import triangle_p1_geometry
from femx.physics.port_eigenmode import (
    VACUUM_PERMEABILITY_H_PER_M,
    VACUUM_PERMITTIVITY_F_PER_M,
)

_CENTROID = ((1.0 / 3.0, 1.0 / 3.0),)


class TrianglePortPencil(NamedTuple):
    """Cell-local Elmer-compatible stiffness and generalized-mass blocks."""

    stiffness: jax.Array
    mass: jax.Array


class AssembledPortPencil(NamedTuple):
    """Dense global mixed pencil before essential constraints."""

    stiffness: jax.Array
    mass: jax.Array


class ReducedPortPencil(NamedTuple):
    """Free-DOF principal subpencil with its canonical full-space indices."""

    stiffness: jax.Array
    mass: jax.Array
    full_dofs: jax.Array


def lossless_port_coefficients(
    cell_relative_permittivity: jax.Array,
    cell_relative_permeability: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return physical permittivity and reluctivity used by locked Elmer EMPort."""

    permittivity = VACUUM_PERMITTIVITY_F_PER_M * cell_relative_permittivity
    reluctivity = 1.0 / (VACUUM_PERMEABILITY_H_PER_M * cell_relative_permeability)
    return permittivity, reluctivity


def triangle_port_local_pencil(
    coordinates: jax.Array,
    cells: jax.Array,
    cell_edge_signs: jax.Array,
    cell_permittivity: jax.Array,
    cell_reluctivity: jax.Array,
    angular_frequency: jax.Array,
) -> TrianglePortPencil:
    r"""Integrate the locked no-potential ``EMPort`` local generalized pencil exactly.

    ``cell_permittivity`` is physical :math:`\epsilon` in F/m and ``cell_reluctivity`` is
    physical :math:`\nu=1/\mu` in m/H.  Both are piecewise constant in the v1 contract.
    """

    areas, gradients = triangle_p1_geometry(coordinates, cells)
    edge_gram = triangle_nedelec1_local_gram(coordinates, cells, cell_edge_signs)
    centroid = jnp.asarray(_CENTROID, dtype=coordinates.dtype)
    edge_evaluation = evaluate_triangle_nedelec1(
        coordinates,
        cells,
        cell_edge_signs,
        centroid,
    )
    centroid_edge_basis = edge_evaluation.basis[:, 0, :, :]

    scalar_reference_mass = (
        jnp.ones((3, 3), dtype=coordinates.dtype) + jnp.eye(3, dtype=coordinates.dtype)
    ) / 12.0
    scalar_mass = areas[:, None, None] * scalar_reference_mass[None, :, :]
    scalar_edge_coupling = areas[:, None, None] * jnp.einsum(
        "cia,cea->cie",
        gradients,
        centroid_edge_basis,
    )

    epsilon = cell_permittivity[:, None, None]
    nu = cell_reluctivity[:, None, None]
    omega_squared = angular_frequency * angular_frequency
    scalar_scalar = -epsilon * scalar_mass
    scalar_edge = epsilon * scalar_edge_coupling
    edge_scalar = nu * jnp.swapaxes(scalar_edge_coupling, 1, 2)
    edge_edge = nu * edge_gram.curl_curl - omega_squared * epsilon * edge_gram.mass
    edge_mass = nu * edge_gram.mass

    zero_scalar_scalar = jnp.zeros_like(scalar_scalar)
    zero_scalar_edge = jnp.zeros_like(scalar_edge)
    zero_edge_scalar = jnp.zeros_like(edge_scalar)
    stiffness = jnp.concatenate(
        (
            jnp.concatenate((scalar_scalar, scalar_edge), axis=2),
            jnp.concatenate((edge_scalar, edge_edge), axis=2),
        ),
        axis=1,
    )
    mass = jnp.concatenate(
        (
            jnp.concatenate((zero_scalar_scalar, zero_scalar_edge), axis=2),
            jnp.concatenate((zero_edge_scalar, edge_mass), axis=2),
        ),
        axis=1,
    )
    return TrianglePortPencil(stiffness=stiffness, mass=mass)


def assemble_port_pencil(
    coordinates: jax.Array,
    cells: jax.Array,
    cell_edge_dofs: jax.Array,
    cell_edge_signs: jax.Array,
    cell_permittivity: jax.Array,
    cell_reluctivity: jax.Array,
    angular_frequency: jax.Array,
    *,
    edge_dof_count: int,
) -> AssembledPortPencil:
    """Scatter-add the dense nodal-first, edge-second mixed generalized pencil."""

    local = triangle_port_local_pencil(
        coordinates,
        cells,
        cell_edge_signs,
        cell_permittivity,
        cell_reluctivity,
        angular_frequency,
    )
    node_count = coordinates.shape[0]
    total_dof_count = node_count + edge_dof_count
    local_dofs = jnp.concatenate((cells, node_count + cell_edge_dofs), axis=1)
    rows = jnp.repeat(local_dofs, 6, axis=1).reshape(-1)
    columns = jnp.tile(local_dofs, (1, 6)).reshape(-1)
    stiffness = jnp.zeros(
        (total_dof_count, total_dof_count),
        dtype=local.stiffness.dtype,
    )
    mass = jnp.zeros_like(stiffness)
    stiffness = stiffness.at[rows, columns].add(local.stiffness.reshape(-1))
    mass = mass.at[rows, columns].add(local.mass.reshape(-1))
    return AssembledPortPencil(stiffness=stiffness, mass=mass)


def assemble_lossless_port_pencil(
    coordinates: jax.Array,
    cells: jax.Array,
    cell_edge_dofs: jax.Array,
    cell_edge_signs: jax.Array,
    cell_relative_permittivity: jax.Array,
    cell_relative_permeability: jax.Array,
    frequency_hz: jax.Array,
    *,
    edge_dof_count: int,
) -> AssembledPortPencil:
    """Convert explicit relative materials and frequency, then assemble the v1 pencil."""

    permittivity, reluctivity = lossless_port_coefficients(
        cell_relative_permittivity,
        cell_relative_permeability,
    )
    angular_frequency = 2.0 * jnp.pi * frequency_hz
    return assemble_port_pencil(
        coordinates,
        cells,
        cell_edge_dofs,
        cell_edge_signs,
        permittivity,
        reluctivity,
        angular_frequency,
        edge_dof_count=edge_dof_count,
    )


def reduce_port_pencil(
    stiffness: jax.Array,
    mass: jax.Array,
    free_dofs: jax.Array,
) -> ReducedPortPencil:
    """Extract the exact free-DOF principal pencil without artificial eigenvalues."""

    reduced_stiffness = stiffness[free_dofs[:, None], free_dofs[None, :]]
    reduced_mass = mass[free_dofs[:, None], free_dofs[None, :]]
    return ReducedPortPencil(
        stiffness=reduced_stiffness,
        mass=reduced_mass,
        full_dofs=free_dofs,
    )
