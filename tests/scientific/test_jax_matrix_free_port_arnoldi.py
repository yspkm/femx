from __future__ import annotations

import math
from functools import partial

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.sparse.linalg import LinearOperator, gmres  # noqa: E402
from tests.support import structured_unit_square_mesh  # noqa: E402

from femx.backends._hcurl import (  # noqa: E402
    canonical_mixed_port_dof_partition,
    canonical_triangle_edge_map,
)
from femx.backends.jax.port_eigensolver import solve_dense_port_eigenmodes  # noqa: E402
from femx.backends.jax.port_krylov import (  # noqa: E402
    MatrixFreePortArnoldiPolicy,
    solve_matrix_free_port_eigenmodes,
)
from femx.backends.jax.port_matrix_free import (  # noqa: E402
    MatrixFreePortSolvePolicy,
    apply_matrix_free_port_block_preconditioner,
    build_lossless_matrix_free_port_pencil,
    prepare_matrix_free_port_block_preconditioner,
    prepare_matrix_free_port_shift,
    prepare_port_matrix_free_topology,
)
from femx.backends.jax.port_operator import (  # noqa: E402
    assemble_lossless_port_pencil,
    reduce_port_pencil,
)
from femx.physics.port_eigenmode import (  # noqa: E402
    VACUUM_PERMEABILITY_H_PER_M,
    VACUUM_PERMITTIVITY_F_PER_M,
)

pytestmark = [pytest.mark.scientific, pytest.mark.requires_jax]


def _physical_port_system(intervals: int):
    mesh = structured_unit_square_mesh(intervals)
    coordinates = np.asarray(mesh.geometry.coordinates) * np.asarray((2.0e-6, 1.0e-6))
    cells = np.asarray(mesh.topology.connectivity)
    facets = np.asarray(mesh.boundary_facets.connectivity)
    local_edges = cells[:, ((0, 1), (1, 2), (2, 0))]
    signs = np.where(local_edges[:, :, 0] < local_edges[:, :, 1], 1, -1).astype(np.int8)
    edge_map = canonical_triangle_edge_map(cells, signs)
    partition = canonical_mixed_port_dof_partition(
        facets,
        edge_map,
        node_count=coordinates.shape[0],
    )
    topology = prepare_port_matrix_free_topology(
        cells,
        edge_map.cell_edge_dofs,
        partition.free_dofs,
        node_count=coordinates.shape[0],
        edge_dof_count=edge_map.dof_count,
    )
    frequency_hz = 100.0e12
    relative_permittivity = np.ones(cells.shape[0])
    relative_permeability = np.ones(cells.shape[0])
    pencil = build_lossless_matrix_free_port_pencil(
        jnp.asarray(coordinates),
        jnp.asarray(cells),
        jnp.asarray(edge_map.cell_edge_signs),
        jnp.asarray(topology.cell_reduced_dofs),
        jnp.asarray(relative_permittivity),
        jnp.asarray(relative_permeability),
        jnp.asarray(frequency_hz),
        free_dof_count=topology.free_dof_count,
    )
    assembled = assemble_lossless_port_pencil(
        jnp.asarray(coordinates),
        jnp.asarray(cells),
        jnp.asarray(edge_map.cell_edge_dofs),
        jnp.asarray(edge_map.cell_edge_signs),
        jnp.asarray(relative_permittivity),
        jnp.asarray(relative_permeability),
        jnp.asarray(frequency_hz),
        edge_dof_count=edge_map.dof_count,
    )
    reduced = reduce_port_pencil(
        assembled.stiffness,
        assembled.mass,
        jnp.asarray(partition.free_dofs),
    )
    scalar_count = int(np.count_nonzero(partition.free_dofs < coordinates.shape[0]))
    propagation_scale = (
        2.0
        * math.pi
        * frequency_hz
        * math.sqrt(VACUUM_PERMITTIVITY_F_PER_M * VACUUM_PERMEABILITY_H_PER_M)
    )
    return pencil, reduced, scalar_count, propagation_scale


def test_generalized_b_arnoldi_matches_dense_spectrum_on_three_refinements() -> None:
    arnoldi_policy = MatrixFreePortArnoldiPolicy(krylov_dimension=31)

    for intervals in (4, 6, 8):
        pencil, reduced, scalar_count, propagation_scale = _physical_port_system(intervals)
        shift = jnp.asarray(-(propagation_scale**2))
        initial = jnp.asarray(
            np.random.default_rng(100 + intervals).normal(size=pencil.free_dof_count)
        )
        linear_policy = MatrixFreePortSolvePolicy(
            relative_tolerance=1.0e-12,
            restart=min(160, pencil.free_dof_count),
            maximum_restart_cycles=100,
            maximum_relative_residual=2.0e-10,
        )
        observed = solve_matrix_free_port_eigenmodes(
            pencil,
            shift,
            initial,
            free_scalar_dof_count=scalar_count,
            mode_count=6,
            arnoldi_policy=arnoldi_policy,
            linear_policy=linear_policy,
        )
        expected = solve_dense_port_eigenmodes(
            reduced.stiffness,
            reduced.mass,
            jnp.asarray(propagation_scale),
            scalar_dof_count=scalar_count,
            mode_count=6,
        )

        assert bool(observed.diagnostics.is_valid)
        relative_beta_error = np.abs(
            np.asarray(observed.propagation_constants_per_m)
            - np.asarray(expected.propagation_constants_per_m)
        ) / np.maximum(np.abs(np.asarray(expected.propagation_constants_per_m)), 1.0)
        assert float(np.max(relative_beta_error)) < 2.0e-9
        assert float(np.max(np.asarray(observed.residuals.maximum_mixed))) < 2.0e-9
        assert float(observed.diagnostics.mass_orthogonality_error) < 1.0e-10
        assert float(np.max(np.asarray(observed.diagnostics.relative_ritz_residuals))) < 1.0e-7

        edge_mass = np.asarray(reduced.mass)[scalar_count:, scalar_count:]
        edge_coefficients = np.asarray(observed.edge_coefficients)
        mass_norms = np.real(
            np.sum(np.conj(edge_coefficients) * (edge_mass @ edge_coefficients), axis=0)
        )
        np.testing.assert_allclose(mass_norms, np.ones(6), rtol=2.0e-12, atol=2.0e-12)


def _scipy_gmres_iterations(
    matrix: np.ndarray,
    right_hand_side: np.ndarray,
    preconditioner: np.ndarray | None,
) -> tuple[int, float]:
    residual_history: list[float] = []
    operator = LinearOperator(matrix.shape, matvec=lambda vector: matrix @ vector)
    preconditioner_operator = None
    if preconditioner is not None:
        preconditioner_operator = LinearOperator(
            matrix.shape,
            matvec=lambda vector: preconditioner @ vector,
        )
    solution, info = gmres(
        operator,
        right_hand_side,
        M=preconditioner_operator,
        restart=min(80, matrix.shape[0]),
        maxiter=100,
        rtol=1.0e-11,
        atol=0.0,
        callback=lambda residual: residual_history.append(float(residual)),
        callback_type="pr_norm",
    )
    assert info == 0
    relative_residual = np.linalg.norm(matrix @ solution - right_hand_side) / np.linalg.norm(
        right_hand_side
    )
    return len(residual_history), float(relative_residual)


def test_block_preconditioner_reduces_refined_explicit_gmres_iterations() -> None:
    iteration_counts: dict[int, tuple[int, int]] = {}

    for intervals in (4, 8):
        pencil, reduced, scalar_count, propagation_scale = _physical_port_system(intervals)
        shift = jnp.asarray(-(propagation_scale**2))
        prepared = prepare_matrix_free_port_shift(pencil, shift)
        preconditioner = prepare_matrix_free_port_block_preconditioner(
            prepared,
            free_scalar_dof_count=scalar_count,
        )
        assert bool(preconditioner.is_valid)

        shifted = np.asarray(reduced.stiffness - shift * reduced.mass)
        left = np.asarray(prepared.equilibration.left_scale)
        right = np.asarray(prepared.equilibration.right_scale)
        equilibrated = left[:, None] * shifted * right[None, :]
        identity = jnp.eye(pencil.free_dof_count)
        explicit_preconditioner = np.asarray(
            jax.vmap(
                partial(
                    apply_matrix_free_port_block_preconditioner,
                    prepared,
                    preconditioner,
                ),
                in_axes=1,
                out_axes=1,
            )(identity)
        )
        right_hand_side = np.random.default_rng(1000 + intervals).normal(size=pencil.free_dof_count)
        unpreconditioned_iterations, unpreconditioned_residual = _scipy_gmres_iterations(
            equilibrated,
            right_hand_side,
            None,
        )
        preconditioned_iterations, preconditioned_residual = _scipy_gmres_iterations(
            equilibrated,
            right_hand_side,
            explicit_preconditioner,
        )
        assert unpreconditioned_residual < 2.0e-10
        assert preconditioned_residual < 2.0e-10
        assert preconditioned_iterations <= unpreconditioned_iterations
        iteration_counts[intervals] = (
            unpreconditioned_iterations,
            preconditioned_iterations,
        )

    coarse_without, coarse_with = iteration_counts[4]
    refined_without, refined_with = iteration_counts[8]
    assert refined_with < 0.5 * refined_without
    assert refined_without - refined_with > coarse_without - coarse_with
