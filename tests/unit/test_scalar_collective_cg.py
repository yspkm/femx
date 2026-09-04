from __future__ import annotations

from dataclasses import replace

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from femx.backends.jax.operators import (  # noqa: E402
    assemble_scalar_h1_system,
    triangle_p1_diffusion_cell_matrices,
)
from femx.backends.jax.scalar_cg import (  # noqa: E402
    ScalarH1CGPolicy,
    assert_scalar_h1_cg_converged,
    build_validation_collective_scalar_h1_cg,
)
from femx.backends.jax.scalar_collective import (  # noqa: E402
    SCALAR_H1_COLLECTIVE_LAYOUT_SCHEMA,
    build_validation_collective_scalar_h1_rhs_assembly,
    prepare_collective_scalar_h1_layout,
    prepare_scalar_h1_boundary_facet_map,
    reconstruct_scalar_h1_state,
    scalar_h1_reduced_cell_rhs,
    triangle_p1_scalar_cell_load_vectors,
)
from femx.backends.jax.scalar_owned_ghost import (  # noqa: E402
    prepare_scalar_h1_owned_ghost_topology,
)
from femx.core.errors import ContractError  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


COORDINATES = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)))
CELLS = np.asarray(((0, 1, 3), (0, 3, 2)), dtype=np.int64)
FACETS = np.asarray(((0, 1), (1, 3), (3, 2), (2, 0)), dtype=np.int64)
FREE_NODES = np.asarray((1, 2, 3), dtype=np.int64)
DEFAULT_SCALE = jnp.asarray(1.0)


def _topology():
    return prepare_scalar_h1_owned_ghost_topology(
        CELLS,
        np.asarray((0, 0), dtype=np.int64),
        node_count=4,
        free_nodes=FREE_NODES,
        partition_count=1,
    )


def _layout():
    return prepare_collective_scalar_h1_layout(_topology())


def _mesh() -> Mesh:
    return Mesh(np.asarray((jax.devices("cpu")[0],), dtype=object), ("partition",))


def _system(scale: jax.Array = DEFAULT_SCALE):
    coordinates = jnp.asarray(COORDINATES)
    cells = jnp.asarray(CELLS)
    facets = jnp.asarray(FACETS)
    diffusion = scale * jnp.asarray((2.0, 3.0))
    source = jnp.asarray((1.5, 0.7))
    facet_load = jnp.asarray((0.2, -0.1, 0.3, 0.0))
    dirichlet_values = jnp.asarray((2.0,))
    boundary_map = prepare_scalar_h1_boundary_facet_map(CELLS, FACETS, node_count=4)
    stiffness = triangle_p1_diffusion_cell_matrices(coordinates, cells, diffusion)
    cell_load = triangle_p1_scalar_cell_load_vectors(
        coordinates,
        cells,
        source,
        facets,
        facet_load,
        boundary_map,
    )
    cell_rhs = scalar_h1_reduced_cell_rhs(
        stiffness,
        cell_load,
        _topology(),
        dirichlet_values,
    )
    dense = assemble_scalar_h1_system(
        coordinates,
        cells,
        diffusion,
        source,
        facets,
        facet_load,
    )
    expected_rhs = (
        dense.load[jnp.asarray(FREE_NODES)]
        - dense.stiffness[jnp.ix_(jnp.asarray(FREE_NODES), jnp.asarray((0,)))] @ dirichlet_values
    )
    expected_matrix = dense.stiffness[jnp.ix_(jnp.asarray(FREE_NODES), jnp.asarray(FREE_NODES))]
    return stiffness, cell_rhs, expected_matrix, expected_rhs, dirichlet_values


def test_cell_load_reduced_rhs_collective_assembly_and_cg_match_dense_reference() -> None:
    layout = _layout()
    stiffness, cell_rhs, dense_matrix, dense_rhs, dirichlet_values = _system()
    rhs_assembly = jax.jit(build_validation_collective_scalar_h1_rhs_assembly(layout, _mesh()))
    observed_rhs = rhs_assembly(cell_rhs)
    np.testing.assert_allclose(observed_rhs, dense_rhs, rtol=2.0e-15, atol=2.0e-15)

    policy = ScalarH1CGPolicy(1.0e-13, 1.0e-14, 20)
    solve = jax.jit(build_validation_collective_scalar_h1_cg(layout, _mesh(), policy))
    result = solve(stiffness, cell_rhs)
    expected_free = jnp.linalg.solve(dense_matrix, dense_rhs)
    assert_scalar_h1_cg_converged(result)
    assert bool(result.converged)
    assert not bool(result.breakdown)
    assert int(result.iterations) <= 3
    assert float(result.relative_residual) < 1.0e-13
    np.testing.assert_allclose(result.right_hand_side, dense_rhs, rtol=2.0e-15, atol=2.0e-15)
    np.testing.assert_allclose(result.solution, expected_free, rtol=3.0e-14, atol=3.0e-14)
    full = reconstruct_scalar_h1_state(_topology(), result.solution, dirichlet_values)
    np.testing.assert_array_equal(full[jnp.asarray((0,))], dirichlet_values)
    np.testing.assert_allclose(full[jnp.asarray(FREE_NODES)], expected_free)

    assert layout.schema_version == SCALAR_H1_COLLECTIVE_LAYOUT_SCHEMA
    assert layout.transport.storage_report.actual_cell_slots == 2
    assert len(layout.digest()) == 64


def test_scalar_collective_cg_uses_residual_defined_reverse_rule() -> None:
    layout = _layout()
    policy = ScalarH1CGPolicy(1.0e-13, 1.0e-14, 20)
    solve = jax.jit(build_validation_collective_scalar_h1_cg(layout, _mesh(), policy))
    weights = jnp.asarray((0.2, -0.3, 0.7))

    def objective(scale: jax.Array) -> jax.Array:
        stiffness, cell_rhs, _, _, _ = _system(scale)
        return jnp.vdot(weights, solve(stiffness, cell_rhs).solution).real

    value, derivative = jax.jit(jax.value_and_grad(objective))(jnp.asarray(1.1))
    step = 2.0e-5
    finite_difference = (
        objective(jnp.asarray(1.1 + step)) - objective(jnp.asarray(1.1 - step))
    ) / (2.0 * step)
    assert np.isfinite(float(value))
    np.testing.assert_allclose(derivative, finite_difference, rtol=2.0e-8, atol=2.0e-10)


def test_zero_rhs_and_fail_closed_breakdown_and_nonconvergence_are_distinct() -> None:
    layout = _layout()
    stiffness, cell_rhs, _, _, _ = _system()
    zero = jnp.zeros_like(cell_rhs)
    converged = jax.jit(
        build_validation_collective_scalar_h1_cg(
            layout,
            _mesh(),
            ScalarH1CGPolicy(1.0e-12, 0.0, 10),
        )
    )(stiffness, zero)
    assert_scalar_h1_cg_converged(converged)
    assert int(converged.iterations) == 0
    np.testing.assert_array_equal(converged.solution, np.zeros(3))

    breakdown = jax.jit(
        build_validation_collective_scalar_h1_cg(
            layout,
            _mesh(),
            ScalarH1CGPolicy(1.0e-12, 0.0, 10),
        )
    )(-stiffness, cell_rhs)
    assert bool(breakdown.breakdown)
    assert not bool(breakdown.converged)
    with pytest.raises(FloatingPointError, match="breakdown"):
        assert_scalar_h1_cg_converged(breakdown)

    incomplete = jax.jit(
        build_validation_collective_scalar_h1_cg(
            layout,
            _mesh(),
            ScalarH1CGPolicy(1.0e-15, 0.0, 1),
        )
    )(stiffness, cell_rhs)
    assert not bool(incomplete.breakdown)
    assert not bool(incomplete.converged)
    with pytest.raises(RuntimeError, match="residual policy"):
        assert_scalar_h1_cg_converged(incomplete)


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        ((0.0, 0.0, 2), "relative tolerance"),
        ((1.0e-8, -1.0, 2), "absolute tolerance"),
        ((1.0e-8, 0.0, 0), "maximum iterations"),
        ((float("nan"), 0.0, 2), "finite"),
    ),
)
def test_scalar_cg_policy_rejects_ambiguous_values(
    arguments: tuple[float, float, int],
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        ScalarH1CGPolicy(*arguments)


def test_scalar_collective_layout_and_boundary_map_fail_closed_on_identity_drift() -> None:
    layout = _layout()
    with pytest.raises(ContractError, match="schema"):
        replace(layout, schema_version="femx.jax.scalar_h1_collective/v2")
    with pytest.raises(ContractError, match="exact scalar topology"):
        replace(layout, topology=_topology())
    with pytest.raises(ContractError, match="scalar owned/ghost topology"):
        prepare_collective_scalar_h1_layout(object())  # type: ignore[arg-type]

    with pytest.raises(ContractError, match="repeat"):
        prepare_scalar_h1_boundary_facet_map(
            CELLS,
            np.asarray(((0, 1), (1, 0))),
            node_count=4,
        )
    with pytest.raises(ContractError, match="exterior"):
        prepare_scalar_h1_boundary_facet_map(
            CELLS,
            np.asarray(((0, 3),)),
            node_count=4,
        )
