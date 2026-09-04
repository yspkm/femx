from __future__ import annotations

from dataclasses import replace

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from femx.backends.jax.port_matrix_free import (  # noqa: E402
    MatrixFreePortBlockPreconditionerPolicy,
    MatrixFreePortPencil,
    MatrixFreePortSolvePolicy,
    apply_matrix_free_port_block_preconditioner,
    apply_matrix_free_port_shift_invert,
    apply_prepared_matrix_free_port_shift_invert,
    build_lossless_matrix_free_port_pencil,
    estimate_port_operator_storage,
    matrix_free_port_matvec,
    prepare_matrix_free_port_block_preconditioner,
    prepare_matrix_free_port_shift,
    prepare_port_matrix_free_topology,
    solve_matrix_free_port_shifted,
    solve_prepared_matrix_free_port_shifted,
)
from femx.core.errors import ContractError  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def _single_triangle_topology(*, free_dofs: np.ndarray | None = None):
    if free_dofs is None:
        free_dofs = np.arange(6, dtype=np.int64)
    return prepare_port_matrix_free_topology(
        np.asarray(((0, 1, 2),), dtype=np.int32),
        np.asarray(((0, 1, 2),), dtype=np.int32),
        free_dofs,
        node_count=3,
        edge_dof_count=3,
    )


def _synthetic_pencil() -> MatrixFreePortPencil:
    stiffness = jnp.asarray(
        (
            (
                (4.0, 0.2, -0.1, 0.3, 0.0, 0.1),
                (-0.4, 3.5, 0.2, 0.0, -0.2, 0.1),
                (0.1, -0.3, 3.8, 0.2, 0.1, 0.0),
                (0.5, 0.0, -0.2, 5.0, 0.3, -0.1),
                (0.0, -0.1, 0.2, -0.4, 4.5, 0.2),
                (0.2, 0.1, 0.0, 0.1, -0.3, 4.2),
            ),
        )
    )
    mass = jnp.zeros((1, 6, 6), dtype=jnp.float64)
    mass = mass.at[0, 3:, 3:].set(jnp.diag(jnp.asarray((1.0, 1.3, 0.8))))
    topology = _single_triangle_topology()
    return MatrixFreePortPencil(
        stiffness=stiffness,
        mass=mass,
        cell_reduced_dofs=jnp.asarray(topology.cell_reduced_dofs),
        free_dof_count=topology.free_dof_count,
    )


def test_topology_maps_constraints_to_one_sentinel_and_free_dofs_are_read_only() -> None:
    topology = _single_triangle_topology(free_dofs=np.asarray((0, 2, 4), dtype=np.int64))

    np.testing.assert_array_equal(topology.cell_reduced_dofs, ((0, 3, 1, 3, 2, 3),))
    np.testing.assert_array_equal(topology.free_dofs, (0, 2, 4))
    assert topology.full_dof_count == 6
    assert topology.free_dof_count == 3
    assert topology.constrained_sentinel == 3
    assert not topology.cell_reduced_dofs.flags.writeable
    assert not topology.free_dofs.flags.writeable


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"node_count": True}, "node_count"),
        ({"edge_dof_count": 0}, "edge_dof_count"),
        ({"cells": np.asarray(((0.0, 1.0, 2.0),))}, "triangle cells"),
        ({"cells": np.asarray((0, 1, 2), dtype=np.int32)}, "triangle cells"),
        ({"cells": np.empty((0, 3), dtype=np.int32)}, "at least one triangle"),
        ({"cells": np.asarray(((0, 1, 3),), dtype=np.int32)}, "out-of-range node"),
        ({"cells": np.asarray(((0, 1, 1),), dtype=np.int32)}, "repeated node"),
        ({"cell_edge_dofs": np.asarray(((0.0, 1.0, 2.0),))}, "triangle edge DOFs"),
        ({"cell_edge_dofs": np.asarray(((0, 1),), dtype=np.int32)}, "triangle edge DOFs"),
        ({"cell_edge_dofs": np.asarray(((0, 1, 3),), dtype=np.int32)}, "out-of-range edge"),
        ({"cell_edge_dofs": np.asarray(((0, 1, 1),), dtype=np.int32)}, "repeated edge"),
        ({"free_dofs": np.asarray((0.0, 1.0))}, "free DOFs"),
        ({"free_dofs": np.asarray(((0, 1),), dtype=np.int32)}, "free DOFs"),
        ({"free_dofs": np.asarray((), dtype=np.int32)}, "at least one free"),
        ({"free_dofs": np.asarray((0, 6), dtype=np.int32)}, "out-of-range index"),
        ({"free_dofs": np.asarray((0, 0), dtype=np.int32)}, "strictly increasing"),
        ({"free_dofs": np.asarray((1, 0), dtype=np.int32)}, "strictly increasing"),
    ),
)
def test_topology_rejects_invalid_contracts(updates: dict[str, object], message: str) -> None:
    arguments: dict[str, object] = {
        "cells": np.asarray(((0, 1, 2),), dtype=np.int32),
        "cell_edge_dofs": np.asarray(((0, 1, 2),), dtype=np.int32),
        "free_dofs": np.arange(6, dtype=np.int32),
        "node_count": 3,
        "edge_dof_count": 3,
    }
    arguments.update(updates)
    with pytest.raises(ContractError, match=message):
        prepare_port_matrix_free_topology(**arguments)  # type: ignore[arg-type]


def test_topology_rejects_a_nominal_free_dof_absent_from_all_cells() -> None:
    with pytest.raises(ContractError, match="absent from all cells"):
        prepare_port_matrix_free_topology(
            np.asarray(((0, 1, 2),), dtype=np.int32),
            np.asarray(((0, 1, 2),), dtype=np.int32),
            np.asarray((0, 3), dtype=np.int32),
            node_count=4,
            edge_dof_count=3,
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"relative_tolerance": 0.0}, "relative tolerance"),
        ({"relative_tolerance": float("nan")}, "relative tolerance"),
        ({"absolute_tolerance": -1.0}, "absolute tolerance"),
        ({"absolute_tolerance": float("inf")}, "absolute tolerance"),
        ({"restart": True}, "restart"),
        ({"restart": 0}, "restart"),
        ({"maximum_restart_cycles": True}, "maximum restart cycles"),
        ({"maximum_restart_cycles": 0}, "maximum restart cycles"),
        ({"solve_method": "unknown"}, "solve method"),
        ({"maximum_relative_residual": 0.0}, "maximum residual"),
        ({"maximum_relative_residual": float("inf")}, "maximum residual"),
        (
            {"relative_tolerance": 1.0e-5, "maximum_relative_residual": 1.0e-6},
            "no smaller",
        ),
    ),
)
def test_solve_policy_rejects_invalid_values(updates: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replace(MatrixFreePortSolvePolicy(), **updates)


@pytest.mark.parametrize("value", (0.0, 1.0, -1.0, float("nan"), float("inf")))
def test_block_preconditioner_policy_rejects_invalid_thresholds(value: float) -> None:
    with pytest.raises(ValueError, match="minimum relative diagonal"):
        MatrixFreePortBlockPreconditionerPolicy(minimum_relative_diagonal=value)


def test_matrix_free_matvec_matches_dense_and_preserves_complex_values_under_jit() -> None:
    pencil = _synthetic_pencil()
    vector = jnp.asarray((1.0 + 0.2j, -0.4j, 0.3, 0.2j, -0.7, 1.1 - 0.1j))

    observed = jax.jit(matrix_free_port_matvec)(
        pencil.stiffness,
        pencil.cell_reduced_dofs,
        vector,
    )

    expected = np.asarray(pencil.stiffness[0], dtype=np.complex128) @ np.asarray(vector)
    np.testing.assert_allclose(observed, expected, rtol=2.0e-15, atol=2.0e-15)
    assert observed.dtype == jnp.complex128


def test_lossless_builder_rejects_a_cell_map_with_the_wrong_shape() -> None:
    with pytest.raises(ValueError, match="cell reduced DOFs"):
        build_lossless_matrix_free_port_pencil(
            jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
            jnp.asarray(((0, 1, 2),), dtype=jnp.int32),
            jnp.asarray(((1, 1, -1),), dtype=jnp.int8),
            jnp.arange(5, dtype=jnp.int32)[None, :],
            jnp.ones(1),
            jnp.ones(1),
            jnp.asarray(1.0),
            free_dof_count=6,
        )


@pytest.mark.parametrize("free_dof_count", (True, 0))
def test_lossless_builder_requires_a_positive_static_free_count(
    free_dof_count: object,
) -> None:
    with pytest.raises(ValueError, match="free DOF count"):
        build_lossless_matrix_free_port_pencil(
            jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
            jnp.asarray(((0, 1, 2),), dtype=jnp.int32),
            jnp.asarray(((1, 1, -1),), dtype=jnp.int8),
            jnp.arange(6, dtype=jnp.int32)[None, :],
            jnp.ones(1),
            jnp.ones(1),
            jnp.asarray(1.0),
            free_dof_count=free_dof_count,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("cell_matrix", "cell_map", "vector", "error", "message"),
    (
        (jnp.ones((6, 6)), jnp.arange(6)[None, :], jnp.ones(6), ValueError, "cell matrix"),
        (jnp.ones((1, 6, 6)), jnp.arange(5)[None, :], jnp.ones(6), ValueError, "cell map"),
        (jnp.ones((1, 6, 6)), jnp.arange(6)[None, :], jnp.ones((2, 3)), ValueError, "vector"),
        (
            jnp.ones((1, 6, 6)),
            jnp.arange(6, dtype=jnp.float64)[None, :],
            jnp.ones(6),
            TypeError,
            "integer dtype",
        ),
    ),
)
def test_matrix_free_matvec_rejects_invalid_shapes_and_index_dtype(
    cell_matrix: jax.Array,
    cell_map: jax.Array,
    vector: jax.Array,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        matrix_free_port_matvec(cell_matrix, cell_map, vector)


@pytest.mark.parametrize("invalid_index", (-1, 7))
def test_matrix_free_matvec_fails_closed_for_an_out_of_range_reduced_index(
    invalid_index: int,
) -> None:
    mapping = jnp.arange(6, dtype=jnp.int32).at[0].set(invalid_index)[None, :]
    result = matrix_free_port_matvec(jnp.eye(6)[None, :, :], mapping, jnp.ones(6))

    assert np.all(np.isnan(np.asarray(result)))


def test_shifted_solve_and_elmer_shift_invert_action_match_dense_reference() -> None:
    pencil = _synthetic_pencil()
    policy = MatrixFreePortSolvePolicy(
        relative_tolerance=1.0e-13,
        restart=6,
        maximum_restart_cycles=3,
        maximum_relative_residual=1.0e-11,
    )
    shift = jnp.asarray(-0.7)
    right_hand_side = jnp.asarray((0.3, -0.2, 0.8, 1.1, -0.4, 0.6))

    solve = jax.jit(lambda rhs: solve_matrix_free_port_shifted(pencil, rhs, shift, policy=policy))
    observed = solve(right_hand_side)
    dense_shifted = np.asarray(pencil.stiffness[0] - shift * pencil.mass[0])
    expected = np.linalg.solve(dense_shifted, np.asarray(right_hand_side))

    assert bool(observed.diagnostics.is_valid)
    assert float(observed.diagnostics.equilibrated_relative_residual) < 2.0e-14
    np.testing.assert_allclose(observed.solution, expected, rtol=2.0e-13, atol=2.0e-13)

    vector = jnp.asarray((0.2, -0.1, 0.4, 0.8, -0.3, 0.5))
    shift_invert = apply_matrix_free_port_shift_invert(pencil, vector, shift, policy=policy)
    expected_action = np.linalg.solve(
        dense_shifted, np.asarray(pencil.mass[0]) @ np.asarray(vector)
    )
    assert bool(shift_invert.diagnostics.is_valid)
    np.testing.assert_allclose(shift_invert.solution, expected_action, rtol=2.0e-13, atol=2.0e-13)


def test_prepared_block_preconditioned_solve_matches_dense_lower_block_action() -> None:
    pencil = _synthetic_pencil()
    shift = jnp.asarray(-0.7)
    prepared = prepare_matrix_free_port_shift(pencil, shift)
    preconditioner = prepare_matrix_free_port_block_preconditioner(
        prepared,
        free_scalar_dof_count=3,
    )
    vector = jnp.asarray((0.3, -0.2, 0.8, 1.1, -0.4, 0.6))

    observed_preconditioner = jax.jit(
        lambda candidate: apply_matrix_free_port_block_preconditioner(
            prepared,
            preconditioner,
            candidate,
        )
    )(vector)
    dense_shifted = np.asarray(pencil.stiffness[0] - shift * pencil.mass[0])
    left = np.asarray(prepared.equilibration.left_scale)
    right = np.asarray(prepared.equilibration.right_scale)
    equilibrated = left[:, None] * dense_shifted * right[None, :]
    inverse_diagonal = 1.0 / np.diag(equilibrated)
    expected_scalar = inverse_diagonal[:3] * np.asarray(vector)[:3]
    expected_edge = inverse_diagonal[3:] * (
        np.asarray(vector)[3:] - equilibrated[3:, :3] @ expected_scalar
    )
    np.testing.assert_allclose(
        observed_preconditioner,
        np.concatenate((expected_scalar, expected_edge)),
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    assert bool(preconditioner.is_valid)
    assert float(preconditioner.minimum_relative_diagonal) > 0.0

    policy = MatrixFreePortSolvePolicy(
        relative_tolerance=1.0e-13,
        restart=6,
        maximum_restart_cycles=3,
        maximum_relative_residual=1.0e-11,
    )
    solved = solve_prepared_matrix_free_port_shifted(
        prepared,
        vector,
        policy=policy,
        preconditioner=preconditioner,
    )
    expected = np.linalg.solve(dense_shifted, np.asarray(vector))
    assert bool(solved.diagnostics.is_valid)
    np.testing.assert_allclose(solved.solution, expected, rtol=2.0e-13, atol=2.0e-13)

    action = apply_prepared_matrix_free_port_shift_invert(
        prepared,
        vector,
        policy=policy,
        preconditioner=preconditioner,
    )
    expected_action = np.linalg.solve(
        dense_shifted,
        np.asarray(pencil.mass[0]) @ np.asarray(vector),
    )
    assert bool(action.diagnostics.is_valid)
    np.testing.assert_allclose(action.solution, expected_action, rtol=2.0e-13, atol=2.0e-13)


def test_complex_shifted_solve_reverse_mode_matches_central_difference() -> None:
    baseline = _synthetic_pencil()
    perturbation = jnp.arange(36, dtype=jnp.float64).reshape(1, 6, 6) / 500.0
    right_hand_side = jnp.asarray((0.3 + 0.1j, -0.2j, 0.8 - 0.4j, 1.1, -0.4 + 0.2j, 0.6j))
    weights = jnp.asarray((0.2j, 0.4, -0.3j, 0.7 + 0.1j, -0.2, 0.5j))
    shift = jnp.asarray(-0.7)
    policy = MatrixFreePortSolvePolicy(
        relative_tolerance=1.0e-13,
        restart=6,
        maximum_restart_cycles=3,
        maximum_relative_residual=1.0e-11,
    )

    def objective(parameter: jax.Array) -> jax.Array:
        pencil = MatrixFreePortPencil(
            stiffness=baseline.stiffness + parameter * perturbation,
            mass=baseline.mass,
            cell_reduced_dofs=baseline.cell_reduced_dofs,
            free_dof_count=baseline.free_dof_count,
        )
        prepared = prepare_matrix_free_port_shift(pencil, shift)
        preconditioner = prepare_matrix_free_port_block_preconditioner(
            prepared,
            free_scalar_dof_count=3,
        )
        result = solve_prepared_matrix_free_port_shifted(
            prepared,
            right_hand_side,
            policy=policy,
            preconditioner=preconditioner,
        )
        return jnp.real(jnp.vdot(weights, result.solution))

    parameter = jnp.asarray(0.3)
    reverse = jax.jit(jax.grad(objective))(parameter)
    step = 1.0e-5
    central = (objective(parameter + step) - objective(parameter - step)) / (2.0 * step)

    np.testing.assert_allclose(reverse, central, rtol=2.0e-9, atol=2.0e-10)


def test_zero_right_hand_side_returns_an_admitted_exact_zero() -> None:
    result = solve_matrix_free_port_shifted(
        _synthetic_pencil(),
        jnp.zeros(6),
        jnp.asarray(-0.5),
    )

    assert bool(result.diagnostics.is_valid)
    assert float(result.diagnostics.equilibrated_relative_residual) == 0.0
    np.testing.assert_array_equal(result.solution, np.zeros(6))


def test_invalid_equilibration_and_unconverged_solve_fail_closed() -> None:
    topology = _single_triangle_topology()
    zero = jnp.zeros((1, 6, 6))
    invalid_pencil = MatrixFreePortPencil(
        zero,
        zero,
        jnp.asarray(topology.cell_reduced_dofs),
        topology.free_dof_count,
    )
    invalid = solve_matrix_free_port_shifted(
        invalid_pencil,
        jnp.ones(6),
        jnp.asarray(0.0),
        policy=MatrixFreePortSolvePolicy(
            relative_tolerance=1.0e-12,
            restart=1,
            maximum_restart_cycles=1,
            maximum_relative_residual=1.0e-12,
        ),
    )
    assert not bool(invalid.diagnostics.is_valid)
    assert np.all(np.isnan(np.asarray(invalid.solution)))

    strict = solve_matrix_free_port_shifted(
        _synthetic_pencil(),
        jnp.asarray((1.0, -2.0, 0.5, 0.7, -0.1, 0.3)),
        jnp.asarray(-0.2),
        policy=MatrixFreePortSolvePolicy(
            relative_tolerance=1.0e-16,
            restart=1,
            maximum_restart_cycles=1,
            maximum_relative_residual=1.0e-16,
        ),
    )
    assert not bool(strict.diagnostics.is_valid)
    assert np.all(np.isnan(np.asarray(strict.equilibrated_solution)))


def test_shifted_solve_rejects_invalid_right_hand_side_shape() -> None:
    with pytest.raises(ValueError, match="right-hand side"):
        solve_matrix_free_port_shifted(
            _synthetic_pencil(),
            jnp.ones((2, 3)),
            jnp.asarray(-0.2),
        )

    with pytest.raises(ValueError, match="bound free DOFs"):
        solve_matrix_free_port_shifted(
            _synthetic_pencil(),
            jnp.ones(5),
            jnp.asarray(-0.2),
        )


def test_prepared_shift_and_block_preconditioner_reject_invalid_layouts() -> None:
    pencil = _synthetic_pencil()
    with pytest.raises(ValueError, match="shift must be a scalar"):
        prepare_matrix_free_port_shift(pencil, jnp.ones(2))

    prepared = prepare_matrix_free_port_shift(pencil, jnp.asarray(-0.2))
    with pytest.raises(TypeError, match="scalar DOF count"):
        prepare_matrix_free_port_block_preconditioner(
            prepared,
            free_scalar_dof_count=True,  # type: ignore[arg-type]
        )
    for scalar_count in (-1, 6):
        with pytest.raises(ValueError, match="leave at least one edge"):
            prepare_matrix_free_port_block_preconditioner(
                prepared,
                free_scalar_dof_count=scalar_count,
            )

    preconditioner = prepare_matrix_free_port_block_preconditioner(
        prepared,
        free_scalar_dof_count=3,
    )
    with pytest.raises(ValueError, match="vector"):
        apply_matrix_free_port_block_preconditioner(prepared, preconditioner, jnp.ones(5))
    with pytest.raises(ValueError, match="diagonal"):
        apply_matrix_free_port_block_preconditioner(
            prepared,
            preconditioner._replace(inverse_scaled_diagonal=jnp.ones(5)),
            jnp.ones(6),
        )
    with pytest.raises(ValueError, match="masks"):
        apply_matrix_free_port_block_preconditioner(
            prepared,
            preconditioner._replace(scalar_mask=jnp.ones(5, dtype=jnp.bool_)),
            jnp.ones(6),
        )
    with pytest.raises(ValueError, match="preconditioner does not match"):
        solve_prepared_matrix_free_port_shifted(
            prepared,
            jnp.ones(6),
            preconditioner=preconditioner._replace(inverse_scaled_diagonal=jnp.ones(5)),
        )


def test_invalid_block_preconditioner_fails_closed() -> None:
    prepared = prepare_matrix_free_port_shift(_synthetic_pencil(), jnp.asarray(-0.2))
    preconditioner = prepare_matrix_free_port_block_preconditioner(
        prepared,
        free_scalar_dof_count=3,
        policy=MatrixFreePortBlockPreconditionerPolicy(minimum_relative_diagonal=0.99),
    )
    assert not bool(preconditioner.is_valid)
    assert np.all(
        np.isnan(
            np.asarray(
                apply_matrix_free_port_block_preconditioner(
                    prepared,
                    preconditioner,
                    jnp.ones(6),
                )
            )
        )
    )
    result = solve_prepared_matrix_free_port_shifted(
        prepared,
        jnp.zeros(6),
        preconditioner=preconditioner,
    )
    assert not bool(result.diagnostics.is_valid)
    assert np.all(np.isnan(np.asarray(result.solution)))


def test_storage_estimate_is_explicit_and_crosses_dense_storage() -> None:
    small = estimate_port_operator_storage(cell_count=8, free_dof_count=9)
    large = estimate_port_operator_storage(cell_count=512, free_dof_count=961)

    assert small.matrix_free_value_bytes == 2 * 8 * 36 * 8
    assert small.matrix_free_index_bytes == 8 * 6 * 8
    assert small.matrix_free_total_bytes == (
        small.matrix_free_value_bytes + small.matrix_free_index_bytes
    )
    assert small.dense_pair_bytes == 2 * 9 * 9 * 8
    assert small.dense_to_matrix_free_ratio < 1.0
    assert large.dense_to_matrix_free_ratio > 40.0


@pytest.mark.parametrize(
    "updates",
    (
        {"cell_count": 0},
        {"free_dof_count": True},
        {"value_itemsize": -1},
        {"index_itemsize": 0},
    ),
)
def test_storage_estimate_rejects_invalid_counts(updates: dict[str, object]) -> None:
    arguments: dict[str, object] = {"cell_count": 1, "free_dof_count": 1}
    arguments.update(updates)
    with pytest.raises(ValueError, match="positive integer"):
        estimate_port_operator_storage(**arguments)  # type: ignore[arg-type]
