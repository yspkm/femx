from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from tests.support import structured_unit_square_mesh  # noqa: E402

from femx.backends._hcurl import (  # noqa: E402
    canonical_mixed_port_dof_partition,
    canonical_triangle_edge_map,
)
from femx.backends.jax.port_eigen_adjoint import solve_simple_port_eigenpair  # noqa: E402
from femx.backends.jax.port_eigensolver import (  # noqa: E402
    schur_reduce_port_pencil,
    solve_dense_port_eigenmodes,
)
from femx.backends.jax.port_operator import (  # noqa: E402
    assemble_lossless_port_pencil,
    reduce_port_pencil,
)
from femx.physics.port_eigenmode import VACUUM_SPEED_OF_LIGHT_M_PER_S  # noqa: E402

pytestmark = [pytest.mark.scientific, pytest.mark.requires_jax]

_WIDTH_M = 2.0e-6
_HEIGHT_M = 1.0e-6
_FREQUENCY_HZ = 1.0e14
_REFERENCE_RELATIVE_PERMITTIVITY = 2.25


def test_rectangular_port_beta_adjoint_matches_discrete_fd_and_continuum_te10() -> None:
    mesh = structured_unit_square_mesh(6)
    coordinates = np.asarray(mesh.geometry.coordinates).copy()
    coordinates[:, 0] *= _WIDTH_M
    coordinates[:, 1] *= _HEIGHT_M
    cells = np.asarray(mesh.topology.connectivity)
    boundary_facets = np.asarray(mesh.boundary_facets.connectivity)
    local_edges = cells[:, ((0, 1), (1, 2), (2, 0))]
    edge_signs = np.where(local_edges[:, :, 0] < local_edges[:, :, 1], 1, -1).astype(np.int8)
    edge_map = canonical_triangle_edge_map(cells, edge_signs)
    partition = canonical_mixed_port_dof_partition(
        boundary_facets,
        edge_map,
        node_count=coordinates.shape[0],
    )
    free_dofs = jnp.asarray(partition.free_dofs)
    scalar_dof_count = int(np.count_nonzero(partition.free_dofs < coordinates.shape[0]))
    vacuum_wavenumber = 2.0 * math.pi * _FREQUENCY_HZ / VACUUM_SPEED_OF_LIGHT_M_PER_S
    propagation_scale = vacuum_wavenumber * math.sqrt(_REFERENCE_RELATIVE_PERMITTIVITY)

    def reduced_pencil(relative_permittivity: jax.Array):
        assembled = assemble_lossless_port_pencil(
            jnp.asarray(coordinates),
            jnp.asarray(cells),
            jnp.asarray(edge_map.cell_edge_dofs),
            jnp.asarray(edge_map.cell_edge_signs),
            jnp.full((cells.shape[0],), relative_permittivity),
            jnp.ones((cells.shape[0],), dtype=jnp.float64),
            jnp.asarray(_FREQUENCY_HZ),
            edge_dof_count=edge_map.dof_count,
        )
        reduced = reduce_port_pencil(assembled.stiffness, assembled.mass, free_dofs)
        return schur_reduce_port_pencil(
            reduced.stiffness,
            reduced.mass,
            scalar_dof_count=scalar_dof_count,
        )

    reference_reduction = reduced_pencil(jnp.asarray(_REFERENCE_RELATIVE_PERMITTIVITY))
    reference_modes = solve_dense_port_eigenmodes(
        jnp.block(
            [
                [
                    reference_reduction.scalar_stiffness,
                    reference_reduction.scalar_edge_coupling,
                ],
                [
                    reference_reduction.edge_scalar_coupling,
                    reference_reduction.edge_stiffness,
                ],
            ]
        ),
        jnp.block(
            [
                [
                    jnp.zeros_like(reference_reduction.scalar_stiffness),
                    jnp.zeros_like(reference_reduction.scalar_edge_coupling),
                ],
                [
                    jnp.zeros_like(reference_reduction.edge_scalar_coupling),
                    reference_reduction.edge_mass,
                ],
            ]
        ),
        jnp.asarray(propagation_scale),
        scalar_dof_count=scalar_dof_count,
        mode_count=1,
    )
    phase_anchor = int(np.asarray(reference_modes.phase_anchor_edge_dofs[0]))

    def propagation_constant(relative_permittivity: jax.Array) -> jax.Array:
        reduction = reduced_pencil(relative_permittivity)
        return solve_simple_port_eigenpair(
            reduction.condensed_stiffness,
            reduction.edge_mass,
            jnp.asarray(propagation_scale),
            selected_mode_index=0,
            phase_anchor_edge_dof=phase_anchor,
        ).propagation_constant_per_m

    reference = jnp.asarray(_REFERENCE_RELATIVE_PERMITTIVITY, dtype=jnp.float64)
    beta, adjoint_gradient = jax.jit(jax.value_and_grad(propagation_constant))(reference)
    step = 2.0e-4
    central_difference = (
        float(propagation_constant(reference + step))
        - float(propagation_constant(reference - step))
    ) / (2.0 * step)
    exact_beta = math.sqrt(
        vacuum_wavenumber**2 * _REFERENCE_RELATIVE_PERMITTIVITY - (math.pi / _WIDTH_M) ** 2
    )
    exact_gradient = vacuum_wavenumber**2 / (2.0 * exact_beta)

    assert float(adjoint_gradient) == pytest.approx(central_difference, rel=2.0e-8)
    assert float(adjoint_gradient) == pytest.approx(exact_gradient, rel=5.0e-3)
    assert float(beta) == pytest.approx(exact_beta, rel=5.0e-3)
