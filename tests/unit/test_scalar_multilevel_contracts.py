from __future__ import annotations

from dataclasses import replace

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from tests.unit.test_scalar_multilevel import (  # noqa: E402
    _cell_operator,
    _device_mesh,
    _hierarchy,
    _layout,
    _mesh_arrays,
    _prolongation,
)

from femx.backends.jax.scalar_collective import (  # noqa: E402
    pack_collective_scalar_h1_cell_matrix,
    pack_collective_scalar_h1_owned_mask,
)
from femx.backends.jax.scalar_multilevel import (  # noqa: E402
    PackedScalarH1MultilevelTransfer,
    ScalarH1MultilevelHierarchy,
    ScalarH1MultilevelPolicy,
    _canonical_float_array,
    _canonical_free_nodes,
    _canonical_int_array,
    _coarse_prolong,
    _matrix_diagnostics,
    build_packed_scalar_h1_multilevel_runtime,
    pack_scalar_h1_multilevel_transfer,
    prepare_scalar_h1_multilevel_hierarchy,
    prepare_scalar_h1_nested_prolongation,
)
from femx.core.errors import ContractError  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


class _BadArray:
    def __array__(self, dtype: object = None, copy: object = None) -> np.ndarray:
        del dtype, copy
        raise ValueError("deliberate conversion failure")


@pytest.mark.parametrize(
    ("function", "value", "keywords", "message"),
    (
        (_canonical_int_array, _BadArray(), {"label": "x", "rank": 1}, "regular"),
        (_canonical_int_array, [1.0], {"label": "x", "rank": 1}, "integer"),
        (_canonical_int_array, [[1, 2]], {"label": "x", "rank": 2, "columns": 3}, "columns"),
        (_canonical_float_array, _BadArray(), {"label": "x", "rank": 1}, "regular"),
        (_canonical_float_array, [1], {"label": "x", "rank": 1}, "floating"),
        (
            _canonical_float_array,
            [[1.0, 2.0]],
            {"label": "x", "rank": 2, "columns": 3},
            "columns",
        ),
        (_canonical_float_array, [np.nan], {"label": "x", "rank": 1}, "finite"),
    ),
)
def test_multilevel_array_canonicalizers_fail_closed(
    function: object,
    value: object,
    keywords: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        function(value, **keywords)  # type: ignore[operator]


def test_multilevel_array_canonicalizers_own_their_read_only_storage() -> None:
    integer_source = np.asarray((1, 2), dtype=np.int64)
    float_source = np.asarray((1.0, 2.0), dtype=np.float64)
    integer_result = _canonical_int_array(integer_source, label="integer", rank=1)
    float_result = _canonical_float_array(float_source, label="float", rank=1)

    assert integer_source.flags.writeable
    assert float_source.flags.writeable
    assert not integer_result.flags.writeable
    assert not float_result.flags.writeable
    assert not np.shares_memory(integer_source, integer_result)
    assert not np.shares_memory(float_source, float_result)


@pytest.mark.parametrize(
    ("nodes", "message"),
    (
        (np.empty(0, dtype=np.int64), "at least one"),
        (np.asarray((0, 3)), "out-of-range"),
        (np.asarray((2, 1)), "strictly increasing"),
    ),
)
def test_free_node_canonicalizer_rejects_invalid_identity(
    nodes: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        _canonical_free_nodes(nodes, node_count=3, label="test nodes")


def _mutated_level(**changes: object):
    return replace(_prolongation(8, 4), **changes)


def test_nested_prolongation_record_rejects_invalid_counts_arrays_and_hashes() -> None:
    level = _prolongation(8, 4)
    with pytest.raises(ContractError, match="positive integer"):
        replace(level, fine_free_dof_count=True)
    with pytest.raises(ContractError, match="strictly reduce"):
        replace(level, coarse_free_dof_count=level.fine_free_dof_count)
    with pytest.raises(ContractError, match="fine free DOF"):
        replace(level, column_indices=level.column_indices[:-1])

    columns = level.column_indices.copy()
    columns[0, 0] = -1
    with pytest.raises(ContractError, match="out-of-range"):
        replace(level, column_indices=columns)

    sentinel_row, sentinel_slot = np.argwhere(level.column_indices == level.coarse_free_dof_count)[
        0
    ]
    weights = level.weights.copy()
    weights[sentinel_row, sentinel_slot] = 0.25
    with pytest.raises(ContractError, match="sentinel entries"):
        replace(level, weights=weights)

    active_row, active_slot = np.argwhere(level.column_indices < level.coarse_free_dof_count)[0]
    weights = level.weights.copy()
    weights[active_row, active_slot] = 0.0
    with pytest.raises(ContractError, match="positive weight"):
        replace(level, weights=weights)
    weights[active_row, active_slot] = -0.5 * level.containment_tolerance
    with pytest.raises(ContractError, match="positive weight"):
        replace(level, weights=weights)

    with pytest.raises(ContractError, match="real scalar"):
        replace(level, containment_tolerance="small")
    with pytest.raises(ContractError, match=r"\(0, 1\)"):
        replace(level, containment_tolerance=0.0)
    weights = level.weights.copy()
    weights[active_row, active_slot] = 2.0
    with pytest.raises(ContractError, match="barycentric"):
        replace(level, weights=weights)

    with pytest.raises(ContractError, match="nonnegative"):
        replace(level, ambiguity_count=-1)
    for field in ("fine_source_sha256", "coarse_space_sha256", "coarse_source_sha256"):
        with pytest.raises(ContractError, match="canonical SHA-256"):
            replace(level, **{field: "NOT-A-DIGEST"})


def test_nested_prolongation_record_rejects_noncanonical_sparse_rows() -> None:
    level = _prolongation(8, 4)
    row = next(
        index
        for index, columns in enumerate(level.column_indices)
        if np.sum(columns < level.coarse_free_dof_count) >= 2
    )
    active_count = int(np.sum(level.column_indices[row] < level.coarse_free_dof_count))

    columns = level.column_indices.copy()
    columns[row, 0], columns[row, active_count] = columns[row, active_count], columns[row, 0]
    weights = level.weights.copy()
    weights[row, 0], weights[row, active_count] = weights[row, active_count], weights[row, 0]
    with pytest.raises(ContractError, match="sentinel columns"):
        replace(level, column_indices=columns, weights=weights)

    columns = level.column_indices.copy()
    columns[row, :2] = columns[row, 1::-1]
    weights = level.weights.copy()
    weights[row, :2] = weights[row, 1::-1]
    with pytest.raises(ContractError, match="strictly increasing"):
        replace(level, column_indices=columns, weights=weights)

    weights = level.weights.copy()
    weights[row, :active_count] = 0.75
    with pytest.raises(ContractError, match="row sums"):
        replace(level, weights=weights)


def test_nested_prolongation_preparation_rejects_invalid_meshes_and_tolerance() -> None:
    fine_coordinates, _, fine_free = _mesh_arrays(4)
    coarse_coordinates, coarse_cells, coarse_free = _mesh_arrays(2)

    def prepare(**changes: object) -> object:
        arguments = {
            "fine_coordinates": fine_coordinates,
            "fine_free_nodes": fine_free,
            "coarse_coordinates": coarse_coordinates,
            "coarse_cells": coarse_cells,
            "coarse_free_nodes": coarse_free,
        }
        arguments.update(changes)
        return prepare_scalar_h1_nested_prolongation(**arguments)

    with pytest.raises(ContractError, match="nonempty"):
        prepare(fine_coordinates=np.empty((0, 2), dtype=np.float64))
    with pytest.raises(ContractError, match="out-of-range"):
        prepare(coarse_cells=np.asarray(((0, 1, 99),), dtype=np.int64))
    with pytest.raises(ContractError, match="repeat"):
        prepare(coarse_cells=np.asarray(((0, 1, 1),), dtype=np.int64))
    with pytest.raises(ContractError, match="reduce"):
        prepare(
            fine_coordinates=coarse_coordinates,
            fine_free_nodes=coarse_free,
        )
    with pytest.raises(ContractError, match="real scalar"):
        prepare(containment_tolerance=True)
    with pytest.raises(ContractError, match=r"\(0, 1\)"):
        prepare(containment_tolerance=float("inf"))
    degenerate = coarse_coordinates.copy()
    degenerate[4] = degenerate[0]
    with pytest.raises(ContractError, match="degenerate"):
        prepare(coarse_coordinates=degenerate)
    shifted = fine_coordinates.copy()
    shifted[fine_free[0]] = (2.0, 2.0)
    with pytest.raises(ContractError, match="outside"):
        prepare(fine_coordinates=shifted)


def test_nested_preparation_rejects_discontinuous_overlap() -> None:
    coarse_coordinates = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 0.0)))
    coarse_cells = np.asarray(((0, 1, 2), (0, 3, 2)), dtype=np.int64)
    fine_coordinates = np.repeat(np.asarray(((0.25, 0.25),)), 3, axis=0)
    with pytest.raises(ContractError, match="disagree"):
        prepare_scalar_h1_nested_prolongation(
            fine_coordinates,
            np.asarray((0, 1, 2)),
            coarse_coordinates,
            coarse_cells,
            np.asarray((1, 3)),
        )


def test_hierarchy_preparation_policy_and_transfer_contracts_fail_closed() -> None:
    layout, hierarchy = _hierarchy()
    level = hierarchy.prolongations[0]
    with pytest.raises(ContractError, match="schema"):
        replace(hierarchy, schema_version="femx.invalid/v2")
    with pytest.raises(ContractError, match="canonical SHA-256"):
        replace(hierarchy, layout_sha256="bad")
    with pytest.raises(ContractError, match="nested prolongation"):
        ScalarH1MultilevelHierarchy(layout.digest(), (), 16)
    with pytest.raises(ContractError, match="nested prolongation"):
        ScalarH1MultilevelHierarchy(layout.digest(), (object(),), 16)  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="dimensions"):
        ScalarH1MultilevelHierarchy(
            layout.digest(),
            (level, _prolongation(6, 2)),
            16,
        )
    with pytest.raises(ContractError, match="positive integer"):
        replace(hierarchy, maximum_replicated_dofs=True)

    with pytest.raises(ContractError, match="collective layout"):
        prepare_scalar_h1_multilevel_hierarchy(object(), (level,), maximum_replicated_dofs=16)  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="at least two levels"):
        prepare_scalar_h1_multilevel_hierarchy(layout, (), maximum_replicated_dofs=16)
    with pytest.raises(ContractError, match="layout free DOFs"):
        prepare_scalar_h1_multilevel_hierarchy(
            _layout(4),
            (level,),
            maximum_replicated_dofs=16,
        )

    for changes, message in (
        ({"diagonal_weight": "one"}, "real scalar"),
        ({"diagonal_weight": float("nan")}, "finite"),
        ({"minimum_relative_diagonal": 1.0}, r"\(0, 1\)"),
        ({"maximum_relative_symmetry_error": 1.0}, r"\[0, 1\)"),
        ({"maximum_coarse_condition_number": 1.0}, "exceed one"),
    ):
        with pytest.raises(ContractError, match=message):
            ScalarH1MultilevelPolicy(**changes)  # type: ignore[arg-type]

    with pytest.raises(ContractError, match="scalar collective layout"):
        pack_scalar_h1_multilevel_transfer(object(), hierarchy, value_dtype=np.float64)  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="requires a hierarchy"):
        pack_scalar_h1_multilevel_transfer(layout, object(), value_dtype=np.float64)  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="does not bind"):
        pack_scalar_h1_multilevel_transfer(
            _layout(4),
            hierarchy,
            value_dtype=np.float64,
        )
    with pytest.raises(ContractError, match="explicit"):
        pack_scalar_h1_multilevel_transfer(layout, hierarchy, value_dtype=object())  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="float32 or float64"):
        pack_scalar_h1_multilevel_transfer(layout, hierarchy, value_dtype=np.int64)


def _runtime_inputs():
    layout, hierarchy = _hierarchy()
    transfer = pack_scalar_h1_multilevel_transfer(layout, hierarchy, value_dtype=np.float64)
    runtime = build_packed_scalar_h1_multilevel_runtime(
        layout,
        _device_mesh(),
        hierarchy,
        ScalarH1MultilevelPolicy(maximum_coarse_condition_number=1.0e8),
        transfer,
    )
    stiffness = pack_collective_scalar_h1_cell_matrix(layout, _cell_operator(8, 10.0))
    mapping = jnp.asarray(layout.transport.cell_local_dofs)
    mask = pack_collective_scalar_h1_owned_mask(layout)
    return layout, hierarchy, transfer, runtime, stiffness, mapping, mask


def test_multilevel_runtime_and_setup_contracts_fail_closed() -> None:
    layout, hierarchy, transfer, runtime, stiffness, mapping, mask = _runtime_inputs()
    policy = ScalarH1MultilevelPolicy()
    with pytest.raises(ContractError, match="collective layout"):
        build_packed_scalar_h1_multilevel_runtime(
            object(),
            _device_mesh(),
            hierarchy,
            policy,
            transfer,  # type: ignore[arg-type]
        )
    with pytest.raises(ContractError, match="hierarchy"):
        build_packed_scalar_h1_multilevel_runtime(
            layout,
            _device_mesh(),
            object(),
            policy,
            transfer,  # type: ignore[arg-type]
        )
    with pytest.raises(ContractError, match="does not bind"):
        build_packed_scalar_h1_multilevel_runtime(
            _layout(4), _device_mesh(), hierarchy, policy, transfer
        )
    with pytest.raises(ContractError, match="Policy"):
        build_packed_scalar_h1_multilevel_runtime(
            layout,
            _device_mesh(),
            hierarchy,
            object(),
            transfer,  # type: ignore[arg-type]
        )
    with pytest.raises(ContractError, match="transfer arrays"):
        build_packed_scalar_h1_multilevel_runtime(
            layout,
            _device_mesh(),
            hierarchy,
            policy,
            object(),  # type: ignore[arg-type]
        )

    for arguments, message in (
        ((stiffness[:, :, :, :2], mapping, mask), "cell stiffness"),
        ((stiffness, mapping[:, :, :2], mask), "cell map"),
        ((stiffness, mapping, mask[:, :-1]), "owner mask"),
        ((stiffness.astype(jnp.complex128), mapping, mask), "cell stiffness"),
        ((stiffness, mapping.astype(jnp.float64), mask), "cell map"),
        ((stiffness, mapping, mask.astype(jnp.int32)), "owner mask"),
    ):
        with pytest.raises((TypeError, ValueError), match=message):
            runtime.setup(*arguments)


def test_multilevel_transfer_and_apply_contracts_fail_closed() -> None:
    layout, hierarchy, transfer, runtime, stiffness, mapping, mask = _runtime_inputs()

    def changed(**changes: object) -> PackedScalarH1MultilevelTransfer:
        return transfer._replace(**changes)

    cases = (
        (changed(owner_columns=transfer.owner_columns[:, :-1]), "owner interpolation"),
        (changed(cell_columns=transfer.cell_columns[:, :-1]), "cell interpolation"),
        (changed(owner_columns=transfer.owner_columns.astype(jnp.float64)), "integer dtype"),
        (changed(cell_columns=transfer.cell_columns.astype(jnp.float64)), "integer dtype"),
        (changed(owner_weights=transfer.owner_weights.astype(jnp.float32)), "operator dtype"),
        (changed(cell_weights=transfer.cell_weights.astype(jnp.float32)), "operator dtype"),
        (changed(coarse_columns=()), "count disagrees"),
        (
            changed(coarse_columns=(transfer.coarse_columns[0][:-1],)),
            "shape disagrees",
        ),
        (
            changed(coarse_columns=(transfer.coarse_columns[0].astype(jnp.float64),)),
            "integer dtype",
        ),
        (
            changed(coarse_weights=(transfer.coarse_weights[0].astype(jnp.float32),)),
            "operator dtype",
        ),
    )
    for invalid, message in cases:
        invalid_runtime = build_packed_scalar_h1_multilevel_runtime(
            layout,
            _device_mesh(),
            hierarchy,
            ScalarH1MultilevelPolicy(),
            invalid,
        )
        with pytest.raises((TypeError, ValueError), match=message):
            invalid_runtime.setup(stiffness, mapping, mask)

    state = runtime.setup(stiffness, mapping, mask)
    residual = jnp.ones((layout.partition_count, layout.owned_dof_capacity))
    with pytest.raises(ValueError, match="residual"):
        runtime.apply(state, residual[:, :-1])
    with pytest.raises(TypeError, match="real floating"):
        runtime.apply(state, residual.astype(jnp.complex128))
    with pytest.raises(TypeError, match="prepared operator dtype"):
        runtime.apply(state, residual.astype(jnp.float32))
    with pytest.raises(ValueError, match="disagrees"):
        _coarse_prolong(jnp.zeros((1, 3), dtype=jnp.int32), jnp.zeros((1, 3)), 2, jnp.ones(1))


def test_invalid_galerkin_setup_returns_nan_instead_of_admitting_inverse() -> None:
    layout, _, _, runtime, stiffness, mapping, mask = _runtime_inputs()
    state = runtime.setup(jnp.zeros_like(stiffness), mapping, mask)
    assert not bool(state.valid)
    result = runtime.apply(state, jnp.ones((layout.partition_count, layout.owned_dof_capacity)))
    assert bool(jnp.all(jnp.isnan(result)))

    diagnostics = _matrix_diagnostics(jnp.zeros((2, 2)), ScalarH1MultilevelPolicy())
    assert np.isinf(float(diagnostics[-1]))
