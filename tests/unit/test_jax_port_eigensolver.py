from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from femx.backends.jax.port_eigensolver import (  # noqa: E402
    compare_port_mode_subspaces,
    schur_reduce_port_pencil,
    solve_dense_port_eigenmodes,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def _mixed_pencil() -> tuple[jax.Array, jax.Array]:
    scalar_stiffness = np.asarray(((-2.0,),))
    scalar_edge = np.asarray(((1.0, 2.0),))
    edge_scalar = np.asarray(((3.0,), (4.0,)))
    scalar_recovery = np.linalg.solve(scalar_stiffness, scalar_edge)
    condensed = np.diag((-18.0, -12.0))
    edge_stiffness = condensed + edge_scalar @ scalar_recovery
    stiffness = np.block([[scalar_stiffness, scalar_edge], [edge_scalar, edge_stiffness]])
    mass = np.diag((0.0, 2.0, 3.0))
    return jnp.asarray(stiffness), jnp.asarray(mass)


@pytest.mark.parametrize(
    ("stiffness", "mass", "scalar_count", "exception", "message"),
    (
        (jnp.ones((2, 3)), jnp.ones((2, 3)), 0, ValueError, "square rank-two"),
        (jnp.eye(3), jnp.eye(2), 0, ValueError, "same square shape"),
        (jnp.eye(3), jnp.eye(3), True, TypeError, "static integer"),
        (jnp.eye(3), jnp.eye(3), 1.0, TypeError, "static integer"),
        (jnp.eye(3), jnp.eye(3), -1, ValueError, "leave at least one edge"),
        (jnp.eye(3), jnp.eye(3), 3, ValueError, "leave at least one edge"),
    ),
)
def test_schur_reduction_rejects_invalid_static_layout(
    stiffness: jax.Array,
    mass: jax.Array,
    scalar_count: int,
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        schur_reduce_port_pencil(
            stiffness,
            mass,
            scalar_dof_count=scalar_count,
        )


def test_schur_reduction_preserves_finite_mixed_eigenpairs() -> None:
    stiffness, mass = _mixed_pencil()
    reduction = schur_reduce_port_pencil(stiffness, mass, scalar_dof_count=1)

    np.testing.assert_allclose(reduction.scalar_recovery, ((-0.5, -1.0),))
    np.testing.assert_allclose(reduction.condensed_stiffness, np.diag((-18.0, -12.0)))
    np.testing.assert_allclose(reduction.edge_mass, np.diag((2.0, 3.0)))

    modes = solve_dense_port_eigenmodes(
        stiffness,
        mass,
        jnp.asarray(3.0),
        scalar_dof_count=1,
        mode_count=2,
    )
    np.testing.assert_allclose(modes.eigenvalues_per_m2, (-9.0, -4.0), atol=2.0e-14)
    np.testing.assert_allclose(modes.propagation_constants_per_m, (3.0, 2.0), atol=4.0e-15)
    np.testing.assert_allclose(modes.residuals.maximum_mixed, 0.0, atol=2.0e-16)
    np.testing.assert_allclose(modes.residuals.schur_equation, 0.0, atol=2.0e-16)

    edge_mass = np.asarray(reduction.edge_mass)
    coefficients = np.asarray(modes.edge_coefficients)
    mass_norms = np.einsum("im,ij,jm->m", coefficients.conj(), edge_mass, coefficients)
    np.testing.assert_allclose(mass_norms, 1.0, rtol=2.0e-15, atol=2.0e-15)
    anchors = coefficients[np.asarray(modes.phase_anchor_edge_dofs), np.arange(2)]
    assert np.all(anchors.real > 0.0)
    np.testing.assert_allclose(anchors.imag, 0.0, atol=2.0e-15)


def test_edge_only_spectrum_handles_an_empty_scalar_constraint_under_jit() -> None:
    stiffness = jnp.diag(jnp.asarray((-9.0, -4.0, 1.0)))
    mass = jnp.eye(3)
    solve = jax.jit(
        solve_dense_port_eigenmodes,
        static_argnames=("scalar_dof_count", "mode_count"),
    )
    modes = solve(
        stiffness,
        mass,
        jnp.asarray(2.0),
        scalar_dof_count=0,
        mode_count=3,
    )

    np.testing.assert_allclose(modes.eigenvalues_per_m2, (-9.0, -4.0, 1.0))
    np.testing.assert_allclose(
        modes.propagation_constants_per_m,
        (3.0 + 0.0j, 2.0 + 0.0j, 0.0 + 1.0j),
    )
    assert modes.scalar_coefficients.shape == (0, 3)
    np.testing.assert_allclose(modes.residuals.scalar_constraint, 0.0)
    np.testing.assert_allclose(modes.residuals.maximum_mixed, 0.0)


@pytest.mark.parametrize(
    ("mode_count", "exception", "message"),
    (
        (True, TypeError, "static integer"),
        (1.0, TypeError, "static integer"),
        (0, ValueError, "positive"),
        (4, ValueError, "no larger"),
    ),
)
def test_dense_solver_rejects_invalid_static_mode_count(
    mode_count: int,
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        solve_dense_port_eigenmodes(
            jnp.diag(jnp.asarray((-9.0, -4.0, 1.0))),
            jnp.eye(3),
            jnp.asarray(3.0),
            scalar_dof_count=0,
            mode_count=mode_count,
        )


def test_explicit_propagation_scaling_does_not_change_the_physical_spectrum() -> None:
    stiffness, mass = _mixed_pencil()
    reference = solve_dense_port_eigenmodes(
        stiffness,
        mass,
        jnp.asarray(1.0),
        scalar_dof_count=1,
        mode_count=2,
    )
    rescaled = solve_dense_port_eigenmodes(
        stiffness,
        mass,
        jnp.asarray(1.0e7),
        scalar_dof_count=1,
        mode_count=2,
    )
    np.testing.assert_allclose(
        rescaled.eigenvalues_per_m2,
        reference.eigenvalues_per_m2,
        rtol=2.0e-15,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        rescaled.propagation_constants_per_m,
        reference.propagation_constants_per_m,
        rtol=2.0e-15,
        atol=2.0e-15,
    )


def test_mass_weighted_subspace_comparison_is_basis_and_phase_invariant() -> None:
    edge_mass = jnp.diag(jnp.asarray((2.0, 3.0, 5.0)))
    reference = jnp.asarray(
        (
            (1.0 + 1.0j, 0.5 - 0.2j),
            (0.4 - 0.1j, 1.2 + 0.3j),
            (0.7 + 0.2j, -0.3 + 0.8j),
        )
    )
    mixing = jnp.asarray(((1.0j, 0.3), (0.2 - 0.4j, -1.0j)))
    candidate = reference @ mixing

    comparison = jax.jit(compare_port_mode_subspaces)(reference, candidate, edge_mass)
    np.testing.assert_allclose(comparison.singular_values, 1.0, atol=2.0e-15)
    np.testing.assert_allclose(comparison.principal_angles_rad, 0.0, atol=7.0e-8)
    np.testing.assert_allclose(comparison.projector_distance, 0.0, atol=7.0e-8)


def test_mass_weighted_subspace_comparison_detects_an_orthogonal_direction() -> None:
    edge_mass = jnp.eye(3)
    reference = jnp.asarray(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)))
    candidate = jnp.asarray(((1.0, 0.0), (0.0, 0.0), (0.0, 1.0)))
    comparison = compare_port_mode_subspaces(reference, candidate, edge_mass)

    np.testing.assert_allclose(comparison.singular_values, (1.0, 0.0))
    np.testing.assert_allclose(comparison.principal_angles_rad, (0.0, np.pi / 2.0))
    assert float(comparison.projector_distance) == 1.0


@pytest.mark.parametrize(
    ("reference", "candidate", "mass", "message"),
    (
        (jnp.ones(2), jnp.ones(2), jnp.eye(2), "rank-two"),
        (jnp.ones((2, 1)), jnp.ones((2, 2)), jnp.eye(2), "same shape"),
        (jnp.ones((2, 1)), jnp.ones((2, 1)), jnp.eye(3), "mass shape"),
        (jnp.ones((2, 0)), jnp.ones((2, 0)), jnp.eye(2), "at least one"),
    ),
)
def test_subspace_comparison_rejects_invalid_shapes(
    reference: jax.Array,
    candidate: jax.Array,
    mass: jax.Array,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compare_port_mode_subspaces(reference, candidate, mass)
