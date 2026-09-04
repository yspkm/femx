"""Native JAX P1-triangle operators for steady scalar heat conduction."""

from __future__ import annotations

from typing import NamedTuple, cast

import jax
import jax.numpy as jnp

from femx.backends.jax.autodiff import implicit_linear_solve


class AssembledScalarSystem(NamedTuple):
    """Global scalar-H1 stiffness matrix and variational load vector."""

    stiffness: jax.Array
    load: jax.Array


def triangle_p1_geometry(
    coordinates: jax.Array,
    cells: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return positive cell areas and physical gradients of the three P1 bases."""

    points = coordinates[cells]
    x0, x1, x2 = points[:, 0, :], points[:, 1, :], points[:, 2, :]
    determinant = (x1[:, 0] - x0[:, 0]) * (x2[:, 1] - x0[:, 1]) - (x2[:, 0] - x0[:, 0]) * (
        x1[:, 1] - x0[:, 1]
    )
    twice_area = jnp.abs(determinant)
    gradient_numerators = jnp.stack(
        (
            jnp.stack((x1[:, 1] - x2[:, 1], x2[:, 0] - x1[:, 0]), axis=1),
            jnp.stack((x2[:, 1] - x0[:, 1], x0[:, 0] - x2[:, 0]), axis=1),
            jnp.stack((x0[:, 1] - x1[:, 1], x1[:, 0] - x0[:, 0]), axis=1),
        ),
        axis=1,
    )
    gradients = gradient_numerators / determinant[:, None, None]
    return 0.5 * twice_area, gradients


def triangle_p1_diffusion_cell_matrices(
    coordinates: jax.Array,
    cells: jax.Array,
    cell_diffusion: jax.Array,
) -> jax.Array:
    r"""Return ``k area grad(N_i) dot grad(N_j)`` for every P1 triangle.

    This element-local representation is the common discrete authority for dense assembly and the
    matrix-free heat/current path.  It carries no equation-specific unit interpretation: thermal
    conductivity and electrical conductivity are bound by their respective physics adapters.
    """

    if cell_diffusion.ndim != 1 or cell_diffusion.shape != cells.shape[:1]:
        raise ValueError("P1 cell diffusion must contain one scalar per triangle")
    areas, gradients = triangle_p1_geometry(coordinates, cells)
    return (
        cell_diffusion[:, None, None]
        * areas[:, None, None]
        * jnp.einsum("cid,cjd->cij", gradients, gradients)
    )


def assemble_scalar_h1_system(
    coordinates: jax.Array,
    cells: jax.Array,
    cell_diffusion: jax.Array,
    cell_source: jax.Array,
    boundary_facets: jax.Array,
    facet_load: jax.Array,
) -> AssembledScalarSystem:
    r"""Assemble isotropic scalar ``K`` and ``f`` without essential constraints."""

    areas, _ = triangle_p1_geometry(coordinates, cells)
    local_stiffness = triangle_p1_diffusion_cell_matrices(
        coordinates,
        cells,
        cell_diffusion,
    )
    local_load = jnp.broadcast_to(
        cell_source[:, None] * areas[:, None] / 3.0,
        cells.shape,
    )

    node_count = coordinates.shape[0]
    rows = jnp.repeat(cells, 3, axis=1).reshape(-1)
    columns = jnp.tile(cells, (1, 3)).reshape(-1)
    stiffness = jnp.zeros((node_count, node_count), dtype=coordinates.dtype)
    stiffness = stiffness.at[rows, columns].add(local_stiffness.reshape(-1))
    load = jnp.zeros((node_count,), dtype=coordinates.dtype)
    load = load.at[cells.reshape(-1)].add(local_load.reshape(-1))

    facet_points = coordinates[boundary_facets]
    facet_lengths = jnp.linalg.norm(facet_points[:, 1, :] - facet_points[:, 0, :], axis=1)
    local_facet_load = facet_load * facet_lengths / 2.0
    load = load.at[boundary_facets.reshape(-1)].add(jnp.repeat(local_facet_load, 2))
    return AssembledScalarSystem(stiffness=stiffness, load=load)


def assemble_triangle_p1_cell_nodal_load(
    coordinates: jax.Array,
    cells: jax.Array,
    cell_nodal_source: jax.Array,
) -> jax.Array:
    r"""Assemble ``integral N_i q_h`` for one cell-local P1 source per triangle.

    Cell-local values deliberately permit a material-discontinuous source at shared geometric
    nodes. The consistent triangle mass matrix integrates the product of the P1 test function and
    the linearly interpolated source exactly.
    """

    areas, _ = triangle_p1_geometry(coordinates, cells)
    reference_mass = (jnp.ones((3, 3), dtype=coordinates.dtype) + jnp.eye(3)) / 12.0
    local_load = areas[:, None] * jnp.einsum(
        "ij,cj->ci",
        reference_mass,
        cell_nodal_source,
    )
    load = jnp.zeros((coordinates.shape[0],), dtype=coordinates.dtype)
    return load.at[cells.reshape(-1)].add(local_load.reshape(-1))


def assemble_steady_heat_system(
    coordinates: jax.Array,
    cells: jax.Array,
    cell_conductivity: jax.Array,
    cell_source: jax.Array,
    boundary_facets: jax.Array,
    facet_heat_load: jax.Array,
) -> AssembledScalarSystem:
    r"""Assemble the steady-heat specialization of the scalar H1 operator."""

    return assemble_scalar_h1_system(
        coordinates,
        cells,
        cell_conductivity,
        cell_source,
        boundary_facets,
        facet_heat_load,
    )


def impose_dirichlet_constraints(
    stiffness: jax.Array,
    load: jax.Array,
    nodes: jax.Array,
    values: jax.Array,
) -> AssembledScalarSystem:
    """Apply symmetric strong elimination while retaining parameter differentiability."""

    constrained_load = load - stiffness[:, nodes] @ values
    constrained_stiffness = stiffness.at[:, nodes].set(0.0)
    constrained_stiffness = constrained_stiffness.at[nodes, :].set(0.0)
    constrained_stiffness = constrained_stiffness.at[nodes, nodes].set(1.0)
    constrained_load = constrained_load.at[nodes].set(values)
    return AssembledScalarSystem(constrained_stiffness, constrained_load)


@jax.jit
def solve_scalar_h1(
    coordinates: jax.Array,
    cells: jax.Array,
    cell_diffusion: jax.Array,
    cell_source: jax.Array,
    boundary_facets: jax.Array,
    facet_load: jax.Array,
    dirichlet_nodes: jax.Array,
    dirichlet_values: jax.Array,
) -> tuple[jax.Array, AssembledScalarSystem]:
    """Assemble and solve one dense scalar H1 reference problem in native JAX."""

    unconstrained = assemble_scalar_h1_system(
        coordinates,
        cells,
        cell_diffusion,
        cell_source,
        boundary_facets,
        facet_load,
    )
    constrained = impose_dirichlet_constraints(
        unconstrained.stiffness,
        unconstrained.load,
        dirichlet_nodes,
        dirichlet_values,
    )
    state = implicit_linear_solve(constrained.stiffness, constrained.load)
    return state, unconstrained


@jax.jit
def solve_steady_heat(
    coordinates: jax.Array,
    cells: jax.Array,
    cell_conductivity: jax.Array,
    cell_source: jax.Array,
    boundary_facets: jax.Array,
    facet_heat_load: jax.Array,
    dirichlet_nodes: jax.Array,
    dirichlet_values: jax.Array,
) -> tuple[jax.Array, AssembledScalarSystem]:
    """Solve the steady-heat specialization of the scalar H1 operator."""

    return cast(
        tuple[jax.Array, AssembledScalarSystem],
        solve_scalar_h1(
            coordinates,
            cells,
            cell_conductivity,
            cell_source,
            boundary_facets,
            facet_heat_load,
            dirichlet_nodes,
            dirichlet_values,
        ),
    )
