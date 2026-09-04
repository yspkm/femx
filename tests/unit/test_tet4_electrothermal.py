from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import Array  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from femx.backends.jax.scalar_cg import ScalarH1CGPolicy  # noqa: E402
from femx.backends.jax.scalar_collective import (  # noqa: E402
    prepare_collective_scalar_h1_layout,
)
from femx.backends.jax.scalar_owned_ghost import (  # noqa: E402
    prepare_scalar_h1_owned_ghost_topology,
)
from femx.backends.jax.tet4_electrothermal import (  # noqa: E402
    TET4_ELECTROTHERMAL_RUNTIME_PLAN_SCHEMA,
    TET4_ELECTROTHERMAL_SCHEMA,
    PackedTet4ElectrothermalState,
    Tet4ElectrothermalAdmissionPolicy,
    Tet4ElectrothermalParameters,
    Tet4ElectrothermalPlan,
    Tet4ElectrothermalRuntime,
    Tet4ElectrothermalRuntimePlan,
    _require_anchored_components,
    build_tet4_electrothermal_runtime,
    pack_tet4_electrothermal_inputs,
    pack_tet4_electrothermal_inputs_host,
    prepare_tet4_electrothermal_plan,
    prepare_tet4_electrothermal_runtime_plan,
    reconstruct_tet4_electrothermal_state,
)
from femx.core.errors import ContractError  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def _structured_tet4_mesh(nx: int, ny: int, nz: int) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(
        [
            (x / nx, y / ny, z / nz)
            for x in range(nx + 1)
            for y in range(ny + 1)
            for z in range(nz + 1)
        ],
        dtype=np.float64,
    )

    def node(x: int, y: int, z: int) -> int:
        return (x * (ny + 1) + y) * (nz + 1) + z

    cells: list[tuple[int, int, int, int]] = []
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                a = node(x, y, z)
                b = node(x + 1, y, z)
                c = node(x, y + 1, z)
                d = node(x + 1, y + 1, z)
                e = node(x, y, z + 1)
                f = node(x + 1, y, z + 1)
                g = node(x, y + 1, z + 1)
                h = node(x + 1, y + 1, z + 1)
                cells.extend(
                    (
                        (a, b, d, h),
                        (a, d, c, h),
                        (a, c, g, h),
                        (a, g, e, h),
                        (a, e, f, h),
                        (a, f, b, h),
                    )
                )
    return coordinates, np.asarray(cells, dtype=np.int64)


def _external_faces(cells: np.ndarray) -> np.ndarray:
    local_faces = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    occurrences: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    for cell in cells:
        for local in local_faces:
            face = cast(tuple[int, int, int], tuple(int(cell[index]) for index in local))
            key = cast(tuple[int, int, int], tuple(sorted(face)))
            occurrences.setdefault(key, []).append(face)
    return np.asarray(
        [faces[0] for _key, faces in sorted(occurrences.items()) if len(faces) == 1],
        dtype=np.int64,
    )


def _plan_arguments(
    *,
    intervals: int = 2,
    partition_count: int = 1,
    embedded_current: bool = False,
) -> tuple[tuple[object, object, object, object], dict[str, Any]]:
    coordinates, cells = _structured_tet4_mesh(intervals, intervals, intervals)
    external = _external_faces(cells)
    top = external[np.all(np.isclose(coordinates[external, 2], 1.0), axis=1)]
    if embedded_current:
        current_parent = np.flatnonzero(
            np.all(coordinates[cells, 1] <= 0.5 + 1.0e-14, axis=1)
        ).astype(np.int64)
    else:
        current_parent = np.arange(cells.shape[0], dtype=np.int64)
    current_nodes = np.unique(cells[current_parent])
    current_boundary = current_nodes[
        np.isclose(coordinates[current_nodes, 0], 0.0)
        | np.isclose(coordinates[current_nodes, 0], 1.0)
    ]
    voltage_scale = np.isclose(coordinates[current_boundary, 0], 1.0).astype(np.float64)
    thermal_boundary = np.flatnonzero(np.isclose(coordinates[:, 2], 0.0)).astype(np.int64)
    centroids = np.mean(coordinates[cells], axis=1)
    owners = np.minimum(
        (partition_count * centroids[:, 0]).astype(np.int64),
        partition_count - 1,
    )
    return (
        (coordinates, cells, owners, current_parent),
        {
            "current_conductivity": np.full((current_parent.size,), 2.0),
            "current_cell_source": np.zeros((current_parent.size,)),
            "current_flux_facets": np.empty((0, 3), dtype=np.int64),
            "current_facet_flux": np.empty((0,), dtype=np.float64),
            "current_dirichlet_nodes": current_boundary,
            "current_dirichlet_base": np.zeros((current_boundary.size,)),
            "current_dirichlet_voltage_scale": voltage_scale,
            "thermal_conductivity": np.full((cells.shape[0],), 4.0),
            "thermal_cell_source": np.full((cells.shape[0],), 3.0),
            "thermal_flux_facets": np.empty((0, 3), dtype=np.int64),
            "thermal_facet_flux": np.empty((0,), dtype=np.float64),
            "thermal_robin_facets": top,
            "thermal_robin_transfer": np.full((top.shape[0],), 6.0),
            "thermal_robin_ambient": np.full((top.shape[0],), 300.0),
            "thermal_dirichlet_nodes": thermal_boundary,
            "thermal_dirichlet_values": np.full((thermal_boundary.size,), 300.0),
            "thermal_reference": 300.0,
            "partition_count": partition_count,
        },
    )


def _plan(
    *,
    intervals: int = 2,
    partition_count: int = 1,
    embedded_current: bool = False,
) -> Tet4ElectrothermalPlan:
    arguments, keywords = _plan_arguments(
        intervals=intervals,
        partition_count=partition_count,
        embedded_current=embedded_current,
    )
    return prepare_tet4_electrothermal_plan(*arguments, **keywords)


def _robin_only_thermal_plan() -> Tet4ElectrothermalPlan:
    arguments, keywords = _plan_arguments()
    keywords["thermal_dirichlet_nodes"] = np.empty((0,), dtype=np.int64)
    keywords["thermal_dirichlet_values"] = np.empty((0,), dtype=np.float64)
    return prepare_tet4_electrothermal_plan(*arguments, **keywords)


def _runtime(plan: Tet4ElectrothermalPlan) -> Tet4ElectrothermalRuntime:
    mesh = Mesh(np.asarray((jax.devices("cpu")[0],), dtype=object), ("partition",))
    cg = ScalarH1CGPolicy(2.0e-12, 1.0e-14, 500, backward_error_tolerance=2.0e-12)
    admission = Tet4ElectrothermalAdmissionPolicy(2.0e-10, 2.0e-10, 2.0e-14, 2.0e-10)
    return build_tet4_electrothermal_runtime(plan, mesh, cg, cg, admission)


def test_thermal_robin_boundary_can_anchor_a_connected_domain_without_dirichlet_nodes() -> None:
    plan = _robin_only_thermal_plan()
    assert plan.thermal_layout.topology.constrained_nodes.size == 0
    assert plan.thermal_layout.topology.free_dof_count == plan.thermal_layout.topology.node_count
    assert plan.thermal_dirichlet_shifted.size == 0

    runtime = _runtime(plan)
    inputs = pack_tet4_electrothermal_inputs(plan, value_dtype=np.float64)
    result = runtime.solve(inputs, _parameters())
    assert bool(np.asarray(result.numerically_admitted))
    assert float(np.asarray(result.dirichlet_outward_power)) == pytest.approx(0.0, abs=1.0e-12)
    assert float(np.asarray(result.convection_outward_power)) > 0.0
    assert float(np.asarray(result.thermal_balance_relative_error)) < 2.0e-10


def _triangle_layout():
    cells = np.asarray(((0, 1, 2),), dtype=np.int64)
    topology = prepare_scalar_h1_owned_ghost_topology(
        cells,
        np.asarray((0,), dtype=np.int64),
        node_count=3,
        free_nodes=np.asarray((1, 2), dtype=np.int64),
        partition_count=1,
    )
    return prepare_collective_scalar_h1_layout(topology)


def _parameters(
    voltage: Array | float = 1.0,
    electrical_scale: Array | float = 1.0,
    thermal_scale: Array | float = 1.0,
) -> Tet4ElectrothermalParameters:
    return Tet4ElectrothermalParameters(
        jnp.asarray(voltage, dtype=jnp.float64),
        jnp.asarray(electrical_scale, dtype=jnp.float64),
        jnp.asarray(thermal_scale, dtype=jnp.float64),
    )


def _scatter_cell_matrix(cells: np.ndarray, values: np.ndarray, node_count: int) -> np.ndarray:
    result = np.zeros((node_count, node_count), dtype=np.float64)
    for cell, local in zip(cells, values, strict=True):
        result[np.ix_(cell, cell)] += local
    return result


def _scatter_cell_vector(cells: np.ndarray, values: np.ndarray, node_count: int) -> np.ndarray:
    result = np.zeros((node_count,), dtype=np.float64)
    for cell, local in zip(cells, values, strict=True):
        result[cell] += local
    return result


def test_manufactured_current_joule_heat_solution_and_voltage_reverse_rule() -> None:
    plan = _plan(intervals=3)
    inputs = pack_tet4_electrothermal_inputs(plan, value_dtype=np.float64)
    runtime = _runtime(plan)
    parameters = _parameters()

    result = jax.jit(runtime.solve)(inputs, parameters)
    potential, temperature = reconstruct_tet4_electrothermal_state(
        plan,
        result.state,
        parameters,
    )
    current_coordinates = np.asarray(
        _structured_tet4_mesh(3, 3, 3)[0][plan.current_parent_node_ids]
    )
    np.testing.assert_allclose(potential, current_coordinates[:, 0], rtol=0.0, atol=2.0e-13)
    np.testing.assert_allclose(
        result.current_joule_density[0, : plan.current_layout.topology.cell_count],
        2.0,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert bool(result.numerically_admitted)
    assert float(result.electrical_energy_relative_error) < 2.0e-13
    assert float(result.charge_balance_relative_error) < 2.0e-13
    assert float(result.joule_transfer_relative_error) < 2.0e-15
    assert float(result.thermal_balance_relative_error) < 2.0e-13

    coordinates = _structured_tet4_mesh(3, 3, 3)[0]
    total_source = 5.0
    slope = total_source * (1.0 + 6.0 / 8.0) / 10.0
    expected = 300.0 + slope * coordinates[:, 2] - total_source * coordinates[:, 2] ** 2 / 8.0
    nodal_rms_error = float(np.sqrt(np.mean((np.asarray(temperature) - expected) ** 2)))
    assert nodal_rms_error < 7.0e-3
    packed_temperature = runtime.thermal_cell_temperature(inputs, result.state)
    np.testing.assert_allclose(
        packed_temperature[0, : plan.thermal_layout.topology.cell_count],
        temperature[np.asarray(plan.thermal_layout.topology.cells)],
        rtol=0.0,
        atol=2.0e-12,
    )

    def objective(voltage: Array) -> Array:
        solved = runtime.solve(inputs, _parameters(voltage))
        return jnp.sum(solved.state.temperature_rise * inputs.thermal_owner_mask)

    derivative = jax.jit(jax.grad(objective))(jnp.asarray(1.0))
    step = 2.0e-5
    finite_difference = (
        objective(jnp.asarray(1.0 + step)) - objective(jnp.asarray(1.0 - step))
    ) / (2.0 * step)
    np.testing.assert_allclose(derivative, finite_difference, rtol=3.0e-9, atol=1.0e-9)


def test_runtime_plan_preserves_the_packed_forward_contract() -> None:
    plan = _plan()
    runtime_plan = prepare_tet4_electrothermal_runtime_plan(plan)
    assert runtime_plan.schema_version == TET4_ELECTROTHERMAL_RUNTIME_PLAN_SCHEMA
    assert runtime_plan.source_plan_sha256 == plan.digest()
    assert len(runtime_plan.digest()) == 64

    inputs = pack_tet4_electrothermal_inputs(plan, value_dtype=np.float64)
    parameters = _parameters()
    expected = _runtime(plan).solve(inputs, parameters)
    mesh = Mesh(np.asarray((jax.devices("cpu")[0],), dtype=object), ("partition",))
    cg = ScalarH1CGPolicy(2.0e-12, 1.0e-14, 500, backward_error_tolerance=2.0e-12)
    admission = Tet4ElectrothermalAdmissionPolicy(2.0e-10, 2.0e-10, 2.0e-14, 2.0e-10)
    observed = build_tet4_electrothermal_runtime(
        runtime_plan,
        mesh,
        cg,
        cg,
        admission,
    ).solve(inputs, parameters)
    np.testing.assert_allclose(observed.state.potential, expected.state.potential)
    np.testing.assert_allclose(observed.state.temperature_rise, expected.state.temperature_rise)
    assert bool(observed.numerically_admitted)


def test_runtime_plan_rejects_identity_drift() -> None:
    plan = _plan()
    runtime_plan = prepare_tet4_electrothermal_runtime_plan(plan)
    with pytest.raises(ContractError, match="schema"):
        replace(runtime_plan, schema_version="v2")
    with pytest.raises(ContractError, match="source digest"):
        replace(runtime_plan, source_plan_sha256="bad")
    with pytest.raises(ContractError, match="reference"):
        replace(runtime_plan, thermal_reference=float("nan"))
    with pytest.raises(ContractError, match="full plan"):
        prepare_tet4_electrothermal_runtime_plan(object())  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="equal partition counts"):
        Tet4ElectrothermalRuntimePlan(
            current_layout=plan.current_layout,
            thermal_layout=_plan(partition_count=2).thermal_layout,
            thermal_reference=300.0,
            source_plan_sha256=plan.digest(),
        )


def test_embedded_current_submesh_transfer_and_dense_authority() -> None:
    plan = _plan(intervals=2, embedded_current=True)
    inputs = pack_tet4_electrothermal_inputs(plan, value_dtype=np.float64)
    runtime = _runtime(plan)
    parameters = _parameters(voltage=0.7, electrical_scale=1.2, thermal_scale=0.8)
    result = jax.jit(runtime.solve)(inputs, parameters)
    potential, temperature = reconstruct_tet4_electrothermal_state(plan, result.state, parameters)
    assert bool(result.numerically_admitted)

    current_cells = np.asarray(plan.current_layout.topology.cells)
    current_matrix = _scatter_cell_matrix(
        current_cells,
        1.2 * plan.current_conduction_stiffness,
        plan.current_layout.topology.node_count,
    )
    current_load = _scatter_cell_vector(
        current_cells,
        plan.current_cell_load,
        plan.current_layout.topology.node_count,
    )
    current_free = plan.current_layout.topology.free_nodes
    current_constrained = plan.current_layout.topology.constrained_nodes
    boundary = plan.current_dirichlet_base + 0.7 * plan.current_dirichlet_scale
    dense_potential = np.zeros((plan.current_layout.topology.node_count,), dtype=np.float64)
    dense_potential[current_constrained] = boundary
    dense_potential[current_free] = np.linalg.solve(
        current_matrix[np.ix_(current_free, current_free)],
        current_load[current_free]
        - current_matrix[np.ix_(current_free, current_constrained)] @ boundary,
    )
    np.testing.assert_allclose(potential, dense_potential, rtol=2.0e-13, atol=2.0e-13)

    thermal_joule = np.asarray(
        result.thermal_joule_density[0, : plan.thermal_layout.topology.cell_count]
    )
    expected_joule = np.zeros_like(thermal_joule)
    expected_joule[plan.current_parent_cell_ids] = np.asarray(
        result.current_joule_density[0, : plan.current_layout.topology.cell_count]
    )
    np.testing.assert_allclose(thermal_joule, expected_joule, rtol=0.0, atol=1.0e-14)

    thermal_cells = np.asarray(plan.thermal_layout.topology.cells)
    thermal_matrix = _scatter_cell_matrix(
        thermal_cells,
        0.8 * plan.thermal_conduction_stiffness + plan.thermal_robin_matrix,
        plan.thermal_layout.topology.node_count,
    )
    joule_load = expected_joule[:, None] * plan.thermal_cell_volumes[:, None] / 4.0
    thermal_load = _scatter_cell_vector(
        thermal_cells,
        plan.thermal_nonrobin_load + plan.thermal_robin_ambient_load + joule_load,
        plan.thermal_layout.topology.node_count,
    )
    thermal_free = plan.thermal_layout.topology.free_nodes
    thermal_constrained = plan.thermal_layout.topology.constrained_nodes
    thermal_boundary = plan.thermal_dirichlet_shifted + plan.thermal_reference
    dense_temperature = np.zeros((plan.thermal_layout.topology.node_count,), dtype=np.float64)
    dense_temperature[thermal_constrained] = thermal_boundary
    dense_temperature[thermal_free] = np.linalg.solve(
        thermal_matrix[np.ix_(thermal_free, thermal_free)],
        thermal_load[thermal_free]
        - thermal_matrix[np.ix_(thermal_free, thermal_constrained)] @ thermal_boundary,
    )
    np.testing.assert_allclose(temperature, dense_temperature, rtol=3.0e-13, atol=2.0e-11)


def test_plan_identity_packing_and_basic_contracts() -> None:
    plan = _plan()
    repeated = _plan()
    assert plan.schema_version == TET4_ELECTROTHERMAL_SCHEMA
    assert plan.digest() == repeated.digest()
    assert plan.current_parent_cell_ids.shape[0] == plan.current_layout.topology.cell_count
    assert np.all(
        plan.current_layout.topology.owned_ghost.cell_owners
        == plan.thermal_layout.topology.owned_ghost.cell_owners[plan.current_parent_cell_ids]
    )
    host32 = pack_tet4_electrothermal_inputs_host(plan, value_dtype=np.float32)
    assert host32.current_conduction_stiffness.dtype == np.float32
    assert host32.current_cell_local_dofs.dtype == np.int32
    assert host32.current_owner_mask.dtype == np.bool_
    assert not host32.thermal_conduction_stiffness.flags.writeable
    packed = pack_tet4_electrothermal_inputs(plan, value_dtype=np.float64)
    assert packed.thermal_conduction_stiffness.dtype == jnp.float64

    with pytest.raises(ContractError, match="prepared plan"):
        pack_tet4_electrothermal_inputs_host(object(), value_dtype=np.float64)  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="float32 or float64"):
        pack_tet4_electrothermal_inputs_host(plan, value_dtype=np.int32)
    with pytest.raises(ContractError, match="schema"):
        replace(plan, schema_version="femx.test/wrong")
    with pytest.raises(ContractError, match="transfer slots"):
        replace(plan, current_to_thermal_slots=np.ones_like(plan.current_to_thermal_slots))
    with pytest.raises(ContractError, match="parent identity"):
        replace(plan, current_parent_node_ids=plan.current_parent_node_ids[::-1])

    two_partition = _plan(partition_count=2)
    assert two_partition.current_to_thermal_slots.shape[0] == 2
    with pytest.raises(ContractError, match="identical current and thermal cell owners"):
        replace(
            two_partition,
            current_parent_cell_ids=np.roll(two_partition.current_parent_cell_ids, 1),
        )

    arguments, keywords = _plan_arguments(partition_count=2)
    imbalanced_owners = np.ones(np.asarray(arguments[2]).shape, dtype=np.int64)
    imbalanced_owners[0] = 0
    imbalanced = prepare_tet4_electrothermal_plan(
        arguments[0],
        arguments[1],
        imbalanced_owners,
        arguments[3],
        **keywords,
    )
    assert imbalanced.current_layout.cell_capacity > 1
    assert (
        np.count_nonzero(
            imbalanced.current_layout.transport.cell_ids[0]
            < imbalanced.current_layout.topology.cell_count
        )
        == 1
    )


def test_float32_forward_preserves_bounded_balance_diagnostics() -> None:
    plan = _plan(intervals=2)
    inputs = pack_tet4_electrothermal_inputs(plan, value_dtype=np.float32)
    mesh = Mesh(np.asarray((jax.devices("cpu")[0],), dtype=object), ("partition",))
    cg = ScalarH1CGPolicy(2.0e-5, 1.0e-7, 500, backward_error_tolerance=2.0e-5)
    admission = Tet4ElectrothermalAdmissionPolicy(2.0e-5, 2.0e-5, 2.0e-6, 2.0e-5)
    runtime = build_tet4_electrothermal_runtime(plan, mesh, cg, cg, admission)
    parameters = Tet4ElectrothermalParameters(
        *(jnp.asarray(value, dtype=jnp.float32) for value in (1.0, 1.0, 1.0))
    )

    result = jax.jit(runtime.solve)(inputs, parameters)

    assert bool(result.numerically_admitted)
    assert float(result.current_linear.backward_error) < 1.0e-6
    assert float(result.thermal_linear.backward_error) < 1.0e-6
    assert float(result.thermal_balance_relative_error) < 2.0e-6


def test_preparation_rejects_invalid_mesh_material_and_boundary_contracts() -> None:
    arguments, keywords = _plan_arguments()
    coordinates, cells, owners, parent_cells = arguments
    coordinates = np.asarray(coordinates)
    cells = np.asarray(cells)

    owners_array = np.asarray(owners)
    for changed_arguments, message in (
        ((coordinates[:, :2], cells, owners, parent_cells), "coordinates"),
        ((coordinates.astype(np.int64), cells, owners, parent_cells), "coordinates"),
        (
            (
                coordinates.copy(),
                np.empty((0, 4), dtype=np.int64),
                owners_array[:0],
                parent_cells,
            ),
            "at least one cell",
        ),
        (
            (
                coordinates,
                np.asarray(((0, 1, 2, coordinates.shape[0]),), dtype=np.int64),
                np.asarray((0,)),
                np.asarray((0,)),
            ),
            "out-of-range node",
        ),
    ):
        with pytest.raises(ContractError, match=message):
            prepare_tet4_electrothermal_plan(*changed_arguments, **keywords)

    nonfinite_coordinates = coordinates.copy()
    nonfinite_coordinates[0, 0] = np.nan
    with pytest.raises(ContractError, match="coordinates must be finite"):
        prepare_tet4_electrothermal_plan(
            nonfinite_coordinates,
            cells,
            owners,
            parent_cells,
            **keywords,
        )
    repeated_cell = cells.copy()
    repeated_cell[0, 1] = repeated_cell[0, 0]
    with pytest.raises(ContractError, match="cannot repeat a node"):
        prepare_tet4_electrothermal_plan(
            coordinates,
            repeated_cell,
            owners,
            parent_cells,
            **keywords,
        )
    reversed_cell = cells.copy()
    reversed_cell[0, [1, 2]] = reversed_cell[0, [2, 1]]
    with pytest.raises(ContractError, match="positive orientation"):
        prepare_tet4_electrothermal_plan(
            coordinates,
            reversed_cell,
            owners,
            parent_cells,
            **keywords,
        )
    with pytest.raises(ContractError, match="owners must match"):
        prepare_tet4_electrothermal_plan(
            coordinates,
            cells,
            np.asarray((0,), dtype=np.int64),
            parent_cells,
            **keywords,
        )

    for changed_parent, message in (
        (np.empty((0,), dtype=np.int64), "at least one cell"),
        (np.asarray((cells.shape[0],), dtype=np.int64), "out of range"),
        (np.asarray((1, 0), dtype=np.int64), "strictly increasing"),
    ):
        with pytest.raises(ContractError, match=message):
            prepare_tet4_electrothermal_plan(
                coordinates,
                cells,
                owners,
                changed_parent,
                **keywords,
            )

    for name, value, message in (
        ("current_conductivity", [[1.0], [1.0, 2.0]], "regular real array"),
        ("current_conductivity", np.ones((1,)), "shaped"),
        (
            "current_conductivity",
            np.full(np.asarray(parent_cells).shape, np.nan),
            "must be finite",
        ),
        (
            "current_conductivity",
            np.zeros(np.asarray(parent_cells).shape),
            "must be positive",
        ),
        (
            "thermal_conductivity",
            np.zeros((cells.shape[0],)),
            "must be positive",
        ),
    ):
        changed = dict(keywords)
        changed[name] = value
        with pytest.raises(ContractError, match=message):
            prepare_tet4_electrothermal_plan(*arguments, **changed)

    invalid_face = np.asarray(((0, 1, coordinates.shape[0]),), dtype=np.int64)
    repeated_face = np.asarray(((0, 0, 1),), dtype=np.int64)
    for face, message in (
        (invalid_face, "out-of-range node"),
        (repeated_face, "cannot repeat a node"),
    ):
        changed = dict(keywords)
        changed["thermal_flux_facets"] = face
        changed["thermal_facet_flux"] = np.zeros((1,))
        with pytest.raises(ContractError, match=message):
            prepare_tet4_electrothermal_plan(*arguments, **changed)

    embedded_arguments, embedded_keywords = _plan_arguments(embedded_current=True)
    embedded_coordinates = np.asarray(embedded_arguments[0])
    embedded_cells = np.asarray(embedded_arguments[1])
    outside_node = int(np.flatnonzero(np.isclose(embedded_coordinates[:, 1], 1.0))[0])
    changed = dict(embedded_keywords)
    changed["current_dirichlet_nodes"] = np.asarray(
        (*np.asarray(changed["current_dirichlet_nodes"], dtype=np.int64), outside_node)
    )
    changed["current_dirichlet_base"] = np.zeros(changed["current_dirichlet_nodes"].shape)
    changed["current_dirichlet_voltage_scale"] = np.ones(changed["current_dirichlet_nodes"].shape)
    with pytest.raises(ContractError, match="outside the current domain"):
        prepare_tet4_electrothermal_plan(*embedded_arguments, **changed)

    external = _external_faces(embedded_cells)
    outside_faces = external[np.all(np.isclose(embedded_coordinates[external, 1], 1.0), axis=1)]
    changed = dict(embedded_keywords)
    changed["current_flux_facets"] = outside_faces[:1]
    changed["current_facet_flux"] = np.zeros((1,))
    with pytest.raises(ContractError, match="outside the current domain"):
        prepare_tet4_electrothermal_plan(*embedded_arguments, **changed)

    for nodes, message in (
        (np.empty((0,), dtype=np.int64), "at least one node"),
        (np.asarray((coordinates.shape[0],), dtype=np.int64), "out-of-range node"),
        (np.asarray((0, 0), dtype=np.int64), "must be unique"),
    ):
        changed = dict(keywords)
        changed["current_dirichlet_nodes"] = nodes
        changed["current_dirichlet_base"] = np.zeros(nodes.shape)
        changed["current_dirichlet_voltage_scale"] = np.ones(nodes.shape)
        with pytest.raises(ContractError, match=message):
            prepare_tet4_electrothermal_plan(*arguments, **changed)

    changed = dict(keywords)
    changed["current_dirichlet_voltage_scale"] = np.zeros_like(
        changed["current_dirichlet_voltage_scale"]
    )
    with pytest.raises(ContractError, match="depend on applied voltage"):
        prepare_tet4_electrothermal_plan(*arguments, **changed)
    changed = dict(keywords)
    changed["current_dirichlet_nodes"] = np.arange(coordinates.shape[0], dtype=np.int64)
    changed["current_dirichlet_base"] = np.zeros((coordinates.shape[0],))
    changed["current_dirichlet_voltage_scale"] = np.ones((coordinates.shape[0],))
    with pytest.raises(ContractError, match="requires a free node"):
        prepare_tet4_electrothermal_plan(*arguments, **changed)

    robin_faces = np.asarray(keywords["thermal_robin_facets"])
    changed = dict(keywords)
    changed["thermal_flux_facets"] = robin_faces[:1]
    changed["thermal_facet_flux"] = np.zeros((1,))
    with pytest.raises(ContractError, match="must be disjoint"):
        prepare_tet4_electrothermal_plan(*arguments, **changed)
    changed = dict(keywords)
    changed["thermal_robin_transfer"] = -np.ones(robin_faces.shape[:1])
    with pytest.raises(ContractError, match="cannot be negative"):
        prepare_tet4_electrothermal_plan(*arguments, **changed)
    changed = dict(keywords)
    changed["thermal_dirichlet_nodes"] = np.empty((0,), dtype=np.int64)
    changed["thermal_dirichlet_values"] = np.empty((0,), dtype=np.float64)
    changed["thermal_robin_transfer"] = np.zeros(robin_faces.shape[:1])
    with pytest.raises(ContractError, match="unanchored connected component"):
        prepare_tet4_electrothermal_plan(*arguments, **changed)
    changed = dict(keywords)
    changed["thermal_dirichlet_nodes"] = np.arange(coordinates.shape[0], dtype=np.int64)
    changed["thermal_dirichlet_values"] = np.full((coordinates.shape[0],), 300.0)
    with pytest.raises(ContractError, match="requires a free node"):
        prepare_tet4_electrothermal_plan(*arguments, **changed)
    for reference, message in ((True, "real scalar"), (float("inf"), "finite")):
        changed = dict(keywords)
        changed["thermal_reference"] = reference
        with pytest.raises(ContractError, match=message):
            prepare_tet4_electrothermal_plan(*arguments, **changed)

    with pytest.raises(ContractError, match="unanchored connected component"):
        _require_anchored_components(
            np.asarray(((0, 1, 2, 3), (4, 5, 6, 7)), dtype=np.int64),
            np.asarray((0,), dtype=np.int64),
            node_count=8,
            label="test mesh",
        )


def test_plan_record_rejects_layout_shape_value_and_partition_drift() -> None:
    plan = _plan()
    with pytest.raises(ContractError, match="layout must be scalar H1"):
        replace(plan, current_layout=object())  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="must use Tet4 identity"):
        replace(plan, current_layout=_triangle_layout())
    two_partition = _plan(partition_count=2)
    with pytest.raises(ContractError, match="equal partition counts"):
        replace(plan, current_layout=two_partition.current_layout)
    with pytest.raises(ContractError, match="wrong shape"):
        replace(plan, current_cell_load=plan.current_cell_load[:, :3])
    nonfinite = plan.current_cell_load.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(ContractError, match="must be finite"):
        replace(plan, current_cell_load=nonfinite)
    negative_volume = plan.current_cell_volumes.copy()
    negative_volume[0] = -1.0
    with pytest.raises(ContractError, match="volumes must be positive"):
        replace(plan, current_cell_volumes=negative_volume)
    negative_conductivity = plan.current_conductivity.copy()
    negative_conductivity[0] = -1.0
    with pytest.raises(ContractError, match="conductivity must be positive"):
        replace(plan, current_conductivity=negative_conductivity)
    with pytest.raises(ContractError, match="thermal reference must be finite"):
        replace(plan, thermal_reference=float("inf"))


def test_runtime_rejects_shape_dtype_parameter_and_state_drift() -> None:
    plan = _plan()
    inputs = pack_tet4_electrothermal_inputs(plan, value_dtype=np.float64)
    mesh = Mesh(np.asarray((jax.devices("cpu")[0],), dtype=object), ("partition",))
    cg = ScalarH1CGPolicy(1.0e-10, 0.0, 100)
    admission = Tet4ElectrothermalAdmissionPolicy(1.0e-8, 1.0e-8, 1.0e-8, 1.0e-8)
    with pytest.raises(ContractError, match="prepared plan"):
        build_tet4_electrothermal_runtime(object(), mesh, cg, cg, admission)  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="scalar CG policies"):
        build_tet4_electrothermal_runtime(plan, mesh, object(), cg, admission)  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="admission policy"):
        build_tet4_electrothermal_runtime(plan, mesh, cg, cg, object())  # type: ignore[arg-type]
    runtime = build_tet4_electrothermal_runtime(plan, mesh, cg, cg, admission)
    parameters = _parameters()
    with pytest.raises(ContractError, match="packed contract"):
        runtime.solve(object(), parameters)
    with pytest.raises(ValueError, match="disagree"):
        runtime.solve(
            inputs._replace(current_cell_local_dofs=inputs.current_cell_local_dofs[:, :, :3]),
            parameters,
        )
    with pytest.raises(TypeError, match="integer"):
        runtime.solve(
            inputs._replace(
                current_to_thermal_slots=inputs.current_to_thermal_slots.astype(jnp.float64)
            ),
            parameters,
        )
    with pytest.raises(TypeError, match="boolean"):
        runtime.solve(
            inputs._replace(thermal_cell_mask=inputs.thermal_cell_mask.astype(jnp.int32)),
            parameters,
        )
    with pytest.raises(TypeError, match="real floating"):
        runtime.solve(
            inputs._replace(thermal_cell_volumes=inputs.thermal_cell_volumes.astype(jnp.int32)),
            parameters,
        )
    with pytest.raises(TypeError, match="share one dtype"):
        runtime.solve(
            inputs._replace(thermal_cell_volumes=inputs.thermal_cell_volumes.astype(jnp.float32)),
            parameters,
        )
    with pytest.raises(ContractError, match="typed contract"):
        runtime.solve(inputs, object())
    with pytest.raises(ContractError, match="scalar arrays"):
        runtime.solve(
            inputs, parameters._replace(applied_voltage=jnp.ones((1,), dtype=jnp.float64))
        )
    with pytest.raises(ContractError, match="match the input dtype"):
        runtime.solve(
            inputs,
            parameters._replace(applied_voltage=jnp.asarray(1.0, dtype=jnp.float32)),
        )

    result = runtime.solve(inputs, parameters)
    with pytest.raises(ContractError, match="packed contract"):
        runtime.thermal_cell_temperature(inputs, object())
    with pytest.raises(ValueError, match="thermal owner layout"):
        runtime.thermal_cell_temperature(
            inputs,
            PackedTet4ElectrothermalState(
                result.state.potential,
                result.state.temperature_rise[:, :1],
            ),
        )
    with pytest.raises(TypeError, match="match the input dtype"):
        runtime.thermal_cell_temperature(
            inputs,
            PackedTet4ElectrothermalState(
                result.state.potential,
                result.state.temperature_rise.astype(jnp.float32),
            ),
        )


@pytest.mark.parametrize("value", (True, "small", 0.0, -1.0, float("inf")))
def test_admission_policy_rejects_ambiguous_tolerances(value: object) -> None:
    with pytest.raises(ContractError, match="must be"):
        Tet4ElectrothermalAdmissionPolicy(value, 1.0e-8, 1.0e-8, 1.0e-8)  # type: ignore[arg-type]


def test_reconstruction_rejects_contract_drift() -> None:
    plan = _plan()
    inputs = pack_tet4_electrothermal_inputs(plan, value_dtype=np.float64)
    result = _runtime(plan).solve(inputs, _parameters())
    with pytest.raises(ContractError, match="prepared plan"):
        reconstruct_tet4_electrothermal_state(object(), result.state, _parameters())  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="packed state"):
        reconstruct_tet4_electrothermal_state(plan, object(), _parameters())  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="typed parameters"):
        reconstruct_tet4_electrothermal_state(plan, result.state, object())  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="real scalar"):
        reconstruct_tet4_electrothermal_state(
            plan,
            result.state,
            _parameters()._replace(applied_voltage=jnp.ones((1,), dtype=jnp.float64)),
        )
