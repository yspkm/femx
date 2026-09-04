r"""Affine first-order scalar H1 element on tetrahedra.

The reference tetrahedron has vertices ``(0, 0, 0)``, ``(1, 0, 0)``,
``(0, 1, 0)``, and ``(0, 0, 1)``.  Its nodal basis is

.. math::

   (N_0, N_1, N_2, N_3) = (1-r-s-t, r, s, t).

The physical gradients use ``J^{-T}``, and integration uses ``abs(det(J))``.
The signed determinant remains in the returned geometry so mesh orientation is
observable rather than silently repaired.  Numerical preparation must reject
degenerate cells before calling these JAX-transformable kernels.
"""

from __future__ import annotations

from typing import Final, NamedTuple

import jax
import jax.numpy as jnp

_REFERENCE_GRADIENTS: Final = (
    (-1.0, -1.0, -1.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


class TetrahedronP1Geometry(NamedTuple):
    """Affine-map data for a batch of four-node tetrahedra."""

    volumes: jax.Array
    basis_gradients: jax.Array
    jacobians: jax.Array
    determinants: jax.Array


class TetrahedronP1LocalOperators(NamedTuple):
    """Exact unit-diffusion and consistent-mass matrices on affine cells."""

    unit_stiffness: jax.Array
    consistent_mass: jax.Array
    geometry: TetrahedronP1Geometry


def _validate_mesh_arrays(coordinates: jax.Array, cells: jax.Array) -> None:
    if coordinates.ndim != 2 or coordinates.shape[1:] != (3,):
        raise ValueError("Tet4 coordinates must be shaped (nodes, 3)")
    if not jnp.issubdtype(coordinates.dtype, jnp.floating):
        raise TypeError("Tet4 coordinates must use a real floating dtype")
    if cells.ndim != 2 or cells.shape[1:] != (4,):
        raise ValueError("Tet4 cells must be shaped (cells, 4)")
    if cells.shape[0] == 0:
        raise ValueError("Tet4 geometry requires at least one cell")
    if not jnp.issubdtype(cells.dtype, jnp.integer):
        raise TypeError("Tet4 cells must use an integer dtype")


def tetrahedron_p1_geometry(
    coordinates: jax.Array,
    cells: jax.Array,
) -> TetrahedronP1Geometry:
    """Return signed affine maps, positive volumes, and physical P1 gradients."""

    _validate_mesh_arrays(coordinates, cells)
    points = coordinates[cells]
    jacobians = jnp.stack(
        (
            points[:, 1, :] - points[:, 0, :],
            points[:, 2, :] - points[:, 0, :],
            points[:, 3, :] - points[:, 0, :],
        ),
        axis=2,
    )
    determinants = jnp.linalg.det(jacobians)
    reference_gradients = jnp.asarray(_REFERENCE_GRADIENTS, dtype=coordinates.dtype)
    basis_gradients = jnp.einsum(
        "ij,cjk->cik",
        reference_gradients,
        jnp.linalg.inv(jacobians),
    )
    return TetrahedronP1Geometry(
        volumes=jnp.abs(determinants) / 6.0,
        basis_gradients=basis_gradients,
        jacobians=jacobians,
        determinants=determinants,
    )


def tetrahedron_p1_local_operators(
    coordinates: jax.Array,
    cells: jax.Array,
) -> TetrahedronP1LocalOperators:
    """Integrate unit isotropic stiffness and the consistent P1 mass exactly."""

    geometry = tetrahedron_p1_geometry(coordinates, cells)
    unit_stiffness = geometry.volumes[:, None, None] * jnp.einsum(
        "cid,cjd->cij",
        geometry.basis_gradients,
        geometry.basis_gradients,
    )
    reference_mass = (
        jnp.ones((4, 4), dtype=coordinates.dtype) + jnp.eye(4, dtype=coordinates.dtype)
    ) / 20.0
    consistent_mass = geometry.volumes[:, None, None] * reference_mass
    return TetrahedronP1LocalOperators(
        unit_stiffness=unit_stiffness,
        consistent_mass=consistent_mass,
        geometry=geometry,
    )


def _validate_cell_tensor(
    values: jax.Array,
    *,
    cell_count: int,
    coordinate_dtype: jnp.dtype,
    label: str,
) -> None:
    if values.shape not in ((cell_count,), (cell_count, 3, 3)):
        raise ValueError(f"{label} must be shaped (cells,) or (cells, 3, 3)")
    if not jnp.issubdtype(values.dtype, jnp.floating):
        raise TypeError(f"{label} must use a real floating dtype")
    if values.dtype != coordinate_dtype:
        raise TypeError(f"{label} and coordinates must use the same dtype")


def tetrahedron_p1_diffusion_cell_matrices(
    coordinates: jax.Array,
    cells: jax.Array,
    cell_diffusion: jax.Array,
) -> jax.Array:
    r"""Return ``volume * grad(N_i)^T C grad(N_j)`` for every Tet4 cell.

    ``cell_diffusion`` is either one isotropic scalar per cell or one complete
    physical ``3 x 3`` tensor per cell.  Positivity and symmetry belong to the
    prepared physics contract; this local algebra deliberately does not project
    or symmetrize caller data.
    """

    geometry = tetrahedron_p1_geometry(coordinates, cells)
    _validate_cell_tensor(
        cell_diffusion,
        cell_count=cells.shape[0],
        coordinate_dtype=coordinates.dtype,
        label="Tet4 cell diffusion",
    )
    if cell_diffusion.ndim == 1:
        return cell_diffusion[:, None, None] * (
            geometry.volumes[:, None, None]
            * jnp.einsum(
                "cid,cjd->cij",
                geometry.basis_gradients,
                geometry.basis_gradients,
            )
        )
    return geometry.volumes[:, None, None] * jnp.einsum(
        "cid,cde,cje->cij",
        geometry.basis_gradients,
        cell_diffusion,
        geometry.basis_gradients,
    )


def tetrahedron_p1_nodal_diffusion_cell_matrices(
    coordinates: jax.Array,
    cells: jax.Array,
    cell_nodal_diffusion: jax.Array,
) -> jax.Array:
    r"""Integrate a P1-interpolated scalar or tensor diffusion coefficient exactly.

    Affine Tet4 basis gradients are constant, so exact integration reduces the
    nodal coefficient to its cell mean.  This is the cell-local discontinuous
    representation needed to reproduce Elmer's nodal material interpolation
    without identifying values across material interfaces.
    """

    cell_count = cells.shape[0] if cells.ndim == 2 else 0
    if cell_nodal_diffusion.shape not in (
        (cell_count, 4),
        (cell_count, 4, 3, 3),
    ):
        raise ValueError("Tet4 nodal diffusion must be shaped (cells, 4) or (cells, 4, 3, 3)")
    if not jnp.issubdtype(cell_nodal_diffusion.dtype, jnp.floating):
        raise TypeError("Tet4 nodal diffusion must use a real floating dtype")
    if cell_nodal_diffusion.dtype != coordinates.dtype:
        raise TypeError("Tet4 nodal diffusion and coordinates must use the same dtype")
    return tetrahedron_p1_diffusion_cell_matrices(
        coordinates,
        cells,
        jnp.mean(cell_nodal_diffusion, axis=1),
    )


def tetrahedron_p1_cell_nodal_load_vectors(
    coordinates: jax.Array,
    cells: jax.Array,
    cell_nodal_source: jax.Array,
) -> jax.Array:
    r"""Return exact local vectors ``integral N_i q_h dV`` for a P1 source."""

    cell_count = cells.shape[0] if cells.ndim == 2 else 0
    if cell_nodal_source.shape != (cell_count, 4):
        raise ValueError("Tet4 nodal source must be shaped (cells, 4)")
    if not jnp.issubdtype(cell_nodal_source.dtype, jnp.floating):
        raise TypeError("Tet4 nodal source must use a real floating dtype")
    if cell_nodal_source.dtype != coordinates.dtype:
        raise TypeError("Tet4 nodal source and coordinates must use the same dtype")
    operators = tetrahedron_p1_local_operators(coordinates, cells)
    return jnp.einsum("cij,cj->ci", operators.consistent_mass, cell_nodal_source)


def tetrahedron_p1_field_gradient(
    coordinates: jax.Array,
    cells: jax.Array,
    nodal_values: jax.Array,
) -> jax.Array:
    """Evaluate the constant physical gradient of one nodal P1 scalar field."""

    if nodal_values.ndim != 1 or nodal_values.shape != coordinates.shape[:1]:
        raise ValueError("Tet4 nodal field must be shaped (nodes,)")
    if not jnp.issubdtype(nodal_values.dtype, jnp.floating):
        raise TypeError("Tet4 nodal field must use a real floating dtype")
    if nodal_values.dtype != coordinates.dtype:
        raise TypeError("Tet4 nodal field and coordinates must use the same dtype")
    geometry = tetrahedron_p1_geometry(coordinates, cells)
    return jnp.einsum(
        "ci,cid->cd",
        nodal_values[cells],
        geometry.basis_gradients,
    )
