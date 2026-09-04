from __future__ import annotations

import math

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from tests.support import structured_unit_square_mesh  # noqa: E402

from femx.backends._hcurl import (  # noqa: E402
    canonical_mixed_port_dof_partition,
    canonical_triangle_edge_map,
)
from femx.backends.jax.port_eigensolver import solve_dense_port_eigenmodes  # noqa: E402
from femx.backends.jax.port_operator import (  # noqa: E402
    assemble_lossless_port_pencil,
    reduce_port_pencil,
)
from femx.physics.port_eigenmode import VACUUM_SPEED_OF_LIGHT_M_PER_S  # noqa: E402

pytestmark = [pytest.mark.scientific, pytest.mark.requires_jax]

_WIDTH_M = 2.0e-6
_HEIGHT_M = 1.0e-6
_FREQUENCY_HZ = 1.0e14


def _rectangular_waveguide_modes(intervals: int, *, mode_count: int = 6):
    mesh = structured_unit_square_mesh(intervals)
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
    assembled = assemble_lossless_port_pencil(
        jnp.asarray(coordinates),
        jnp.asarray(cells),
        jnp.asarray(edge_map.cell_edge_dofs),
        jnp.asarray(edge_map.cell_edge_signs),
        jnp.ones(cells.shape[0]),
        jnp.ones(cells.shape[0]),
        jnp.asarray(_FREQUENCY_HZ),
        edge_dof_count=edge_map.dof_count,
    )
    reduced = reduce_port_pencil(
        assembled.stiffness,
        assembled.mass,
        jnp.asarray(partition.free_dofs),
    )
    scalar_dof_count = int(np.count_nonzero(partition.free_dofs < coordinates.shape[0]))
    vacuum_wavenumber = 2.0 * np.pi * _FREQUENCY_HZ / VACUUM_SPEED_OF_LIGHT_M_PER_S
    modes = solve_dense_port_eigenmodes(
        reduced.stiffness,
        reduced.mass,
        jnp.asarray(vacuum_wavenumber),
        scalar_dof_count=scalar_dof_count,
        mode_count=mode_count,
    )
    return reduced, scalar_dof_count, modes


def test_rectangular_pec_te10_beta_converges_without_propagating_spurious_modes() -> None:
    vacuum_wavenumber = 2.0 * np.pi * _FREQUENCY_HZ / VACUUM_SPEED_OF_LIGHT_M_PER_S
    exact_beta = math.sqrt(vacuum_wavenumber**2 - (math.pi / _WIDTH_M) ** 2)
    relative_errors: list[float] = []

    for intervals in (4, 6, 8):
        reduced, scalar_dof_count, modes = _rectangular_waveguide_modes(intervals)
        beta = np.asarray(modes.propagation_constants_per_m)
        relative_errors.append(abs(float(beta[0].real) - exact_beta) / exact_beta)
        assert abs(float(beta[0].imag)) < 1.0e-12 * exact_beta
        propagating = np.count_nonzero(beta.real > 1.0e-10 * vacuum_wavenumber)
        assert propagating == 1
        assert float(np.max(np.asarray(modes.residuals.maximum_mixed))) < 2.0e-14

        edge_mass = np.asarray(reduced.mass)[scalar_dof_count:, scalar_dof_count:]
        edge_coefficients = np.asarray(modes.edge_coefficients)
        mass_norms = np.einsum(
            "im,ij,jm->m",
            edge_coefficients.conj(),
            edge_mass,
            edge_coefficients,
        )
        np.testing.assert_allclose(mass_norms, 1.0, rtol=3.0e-12, atol=3.0e-12)

    assert relative_errors[0] > relative_errors[1] > relative_errors[2]
    observed_orders = (
        math.log(relative_errors[0] / relative_errors[1]) / math.log(6.0 / 4.0),
        math.log(relative_errors[1] / relative_errors[2]) / math.log(8.0 / 6.0),
    )
    assert min(observed_orders) > 1.5
    assert relative_errors[-1] < 4.0e-4


def test_rectangular_port_eigensolve_compiles_as_one_jax_transform() -> None:
    reduced, scalar_dof_count, expected = _rectangular_waveguide_modes(3, mode_count=4)
    vacuum_wavenumber = 2.0 * np.pi * _FREQUENCY_HZ / VACUUM_SPEED_OF_LIGHT_M_PER_S
    compiled = jax.jit(
        solve_dense_port_eigenmodes,
        static_argnames=("scalar_dof_count", "mode_count"),
    )
    observed = compiled(
        reduced.stiffness,
        reduced.mass,
        jnp.asarray(vacuum_wavenumber),
        scalar_dof_count=scalar_dof_count,
        mode_count=4,
    )
    observed.edge_coefficients.block_until_ready()

    np.testing.assert_allclose(
        observed.eigenvalues_per_m2,
        expected.eigenvalues_per_m2,
        rtol=2.0e-14,
        atol=5.0e-2,
    )
    np.testing.assert_allclose(
        observed.residuals.maximum_mixed,
        expected.residuals.maximum_mixed,
        rtol=2.0e-2,
        atol=2.0e-15,
    )
