from __future__ import annotations

import math
from dataclasses import replace

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
from femx.backends.jax.port_krylov import (  # noqa: E402
    MatrixFreePortArnoldiPolicy,
    solve_matrix_free_port_eigenmodes,
)
from femx.backends.jax.port_matrix_free import (  # noqa: E402
    MatrixFreePortBlockPreconditionerPolicy,
    MatrixFreePortPencil,
    MatrixFreePortSolvePolicy,
    build_lossless_matrix_free_port_pencil,
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

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def _physical_pencils(intervals: int = 3):
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
    pencil = build_lossless_matrix_free_port_pencil(
        jnp.asarray(coordinates),
        jnp.asarray(cells),
        jnp.asarray(edge_map.cell_edge_signs),
        jnp.asarray(topology.cell_reduced_dofs),
        jnp.ones(cells.shape[0]),
        jnp.ones(cells.shape[0]),
        jnp.asarray(frequency_hz),
        free_dof_count=topology.free_dof_count,
    )
    assembled = assemble_lossless_port_pencil(
        jnp.asarray(coordinates),
        jnp.asarray(cells),
        jnp.asarray(edge_map.cell_edge_dofs),
        jnp.asarray(edge_map.cell_edge_signs),
        jnp.ones(cells.shape[0]),
        jnp.ones(cells.shape[0]),
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


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"krylov_dimension": True}, "dimension"),
        ({"krylov_dimension": 1}, "dimension"),
        ({"minimum_mass_norm": 0.0}, "minimum mass norm"),
        ({"minimum_subdiagonal": float("nan")}, "minimum subdiagonal"),
        (
            {"minimum_transformed_eigenvalue_magnitude": -1.0},
            "minimum transformed eigenvalue magnitude",
        ),
        ({"maximum_relative_ritz_residual": float("inf")}, "maximum relative Ritz"),
        ({"maximum_generalized_residual": 0.0}, "maximum generalized"),
        ({"maximum_mass_orthogonality_error": -1.0}, "maximum mass orthogonality"),
    ),
)
def test_arnoldi_policy_rejects_invalid_values(
    updates: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(MatrixFreePortArnoldiPolicy(), **updates)


def test_matrix_free_b_arnoldi_is_jittable_and_matches_dense_finite_spectrum() -> None:
    pencil, reduced, scalar_count, propagation_scale = _physical_pencils()
    policy = MatrixFreePortArnoldiPolicy(krylov_dimension=17)
    linear_policy = MatrixFreePortSolvePolicy(
        relative_tolerance=1.0e-12,
        restart=pencil.free_dof_count,
        maximum_restart_cycles=20,
        maximum_relative_residual=2.0e-10,
    )
    shift = jnp.asarray(-(propagation_scale**2))
    initial = jnp.asarray(np.random.default_rng(17).normal(size=pencil.free_dof_count))

    solve = jax.jit(
        lambda start: solve_matrix_free_port_eigenmodes(
            pencil,
            shift,
            start,
            free_scalar_dof_count=scalar_count,
            mode_count=3,
            arnoldi_policy=policy,
            linear_policy=linear_policy,
        )
    )
    observed = solve(initial)
    expected = solve_dense_port_eigenmodes(
        reduced.stiffness,
        reduced.mass,
        jnp.asarray(propagation_scale),
        scalar_dof_count=scalar_count,
        mode_count=3,
    )

    assert bool(observed.diagnostics.is_valid)
    relative_beta_error = np.abs(
        np.asarray(observed.propagation_constants_per_m)
        - np.asarray(expected.propagation_constants_per_m)
    ) / np.maximum(np.abs(np.asarray(expected.propagation_constants_per_m)), 1.0)
    assert float(np.max(relative_beta_error)) < 2.0e-9
    assert float(np.max(np.asarray(observed.residuals.maximum_mixed))) < 2.0e-9
    assert float(observed.diagnostics.mass_orthogonality_error) < 1.0e-10
    assert np.all(np.asarray(observed.diagnostics.shift_invert_validity))
    assert float(np.max(np.asarray(observed.diagnostics.shift_invert_relative_residuals))) < 2.0e-10
    anchors = np.asarray(observed.edge_coefficients)[
        np.asarray(observed.phase_anchor_edge_dofs),
        np.arange(3),
    ]
    assert np.max(np.abs(anchors.imag)) < 2.0e-12
    assert np.min(anchors.real) > 0.0

    stopped_gradient = jax.grad(
        lambda scale: jnp.real(
            jnp.sum(
                solve_matrix_free_port_eigenmodes(
                    pencil,
                    shift,
                    initial * scale,
                    free_scalar_dof_count=scalar_count,
                    mode_count=3,
                    arnoldi_policy=policy,
                    linear_policy=linear_policy,
                ).propagation_constants_per_m
            )
        )
    )(jnp.asarray(1.0))
    assert float(stopped_gradient) == 0.0


@pytest.mark.parametrize(
    ("mutation", "error", "message"),
    (
        ("initial_shape", ValueError, "initial vector"),
        ("shift_shape", ValueError, "shift"),
        ("complex_pencil", TypeError, "real lossless"),
        ("integer_initial", TypeError, "floating or complex"),
        ("scalar_type", TypeError, "scalar DOF count"),
        ("scalar_negative", ValueError, "leave finite edge"),
        ("scalar_all", ValueError, "leave finite edge"),
        ("mode_type", TypeError, "mode count"),
        ("mode_zero", ValueError, "below the Krylov"),
        ("mode_dimension", ValueError, "below the Krylov"),
        ("dimension_spectrum", ValueError, "smaller than the finite spectrum"),
    ),
)
def test_matrix_free_b_arnoldi_rejects_invalid_layouts(
    mutation: str,
    error: type[Exception],
    message: str,
) -> None:
    pencil, _, scalar_count, propagation_scale = _physical_pencils()
    shift = jnp.asarray(-(propagation_scale**2))
    initial = jnp.ones(pencil.free_dof_count)
    policy = MatrixFreePortArnoldiPolicy(krylov_dimension=15)
    mode_count: object = 3
    scalar_count_argument: object = scalar_count
    if mutation == "initial_shape":
        initial = jnp.ones((1, pencil.free_dof_count))
    elif mutation == "shift_shape":
        shift = jnp.ones(2)
    elif mutation == "complex_pencil":
        pencil = MatrixFreePortPencil(
            pencil.stiffness.astype(jnp.complex128),
            pencil.mass,
            pencil.cell_reduced_dofs,
            pencil.free_dof_count,
        )
    elif mutation == "integer_initial":
        initial = jnp.ones(pencil.free_dof_count, dtype=jnp.int32)
    elif mutation == "scalar_type":
        scalar_count_argument = True
    elif mutation == "scalar_negative":
        scalar_count_argument = -1
    elif mutation == "scalar_all":
        scalar_count_argument = pencil.free_dof_count
    elif mutation == "mode_type":
        mode_count = True
    elif mutation == "mode_zero":
        mode_count = 0
    elif mutation == "mode_dimension":
        mode_count = policy.krylov_dimension
    elif mutation == "dimension_spectrum":
        edge_count = pencil.free_dof_count - scalar_count
        policy = MatrixFreePortArnoldiPolicy(krylov_dimension=edge_count)

    with pytest.raises(error, match=message):
        solve_matrix_free_port_eigenmodes(
            pencil,
            shift,
            initial,
            free_scalar_dof_count=scalar_count_argument,  # type: ignore[arg-type]
            mode_count=mode_count,  # type: ignore[arg-type]
            arnoldi_policy=policy,
        )


@pytest.mark.parametrize(
    "failure",
    (
        "zero_start",
        "zero_shift",
        "mass_floor",
        "subdiagonal_floor",
        "theta_floor",
        "ritz_threshold",
        "residual_threshold",
        "orthogonality_threshold",
        "preconditioner_threshold",
    ),
)
def test_matrix_free_b_arnoldi_fails_closed_when_admission_fails(failure: str) -> None:
    pencil, _, scalar_count, propagation_scale = _physical_pencils()
    shift = jnp.asarray(-(propagation_scale**2))
    initial = jnp.asarray(np.random.default_rng(21).normal(size=pencil.free_dof_count))
    arnoldi_policy = MatrixFreePortArnoldiPolicy(krylov_dimension=15)
    preconditioner_policy = MatrixFreePortBlockPreconditionerPolicy()
    if failure == "zero_start":
        initial = jnp.zeros_like(initial)
    elif failure == "zero_shift":
        shift = jnp.asarray(0.0)
    elif failure == "mass_floor":
        arnoldi_policy = replace(arnoldi_policy, minimum_mass_norm=1.0e100)
    elif failure == "subdiagonal_floor":
        arnoldi_policy = replace(arnoldi_policy, minimum_subdiagonal=1.0e100)
    elif failure == "theta_floor":
        arnoldi_policy = replace(
            arnoldi_policy,
            minimum_transformed_eigenvalue_magnitude=1.0e100,
        )
    elif failure == "ritz_threshold":
        arnoldi_policy = replace(arnoldi_policy, maximum_relative_ritz_residual=1.0e-30)
    elif failure == "residual_threshold":
        arnoldi_policy = replace(arnoldi_policy, maximum_generalized_residual=1.0e-18)
    elif failure == "orthogonality_threshold":
        arnoldi_policy = replace(
            arnoldi_policy,
            maximum_mass_orthogonality_error=1.0e-18,
        )
    elif failure == "preconditioner_threshold":
        preconditioner_policy = MatrixFreePortBlockPreconditionerPolicy(
            minimum_relative_diagonal=0.99
        )

    observed = solve_matrix_free_port_eigenmodes(
        pencil,
        shift,
        initial,
        free_scalar_dof_count=scalar_count,
        mode_count=3,
        arnoldi_policy=arnoldi_policy,
        preconditioner_policy=preconditioner_policy,
    )
    assert not bool(observed.diagnostics.is_valid)
    assert np.all(np.isnan(np.asarray(observed.propagation_constants_per_m)))
    np.testing.assert_array_equal(observed.phase_anchor_edge_dofs, -np.ones(3, dtype=np.int64))
