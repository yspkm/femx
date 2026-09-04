from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from femx.core.errors import ContractError  # noqa: E402
from femx.interop.fdtdx import (  # noqa: E402
    DISTRIBUTED_THERMO_OPTIC_SCHEMA,
    FDTDXDeviceParameterContract,
    FDTDXFingerprint,
    PackedDistributedThermoOpticInputs,
    ThermoOpticLaw,
    build_distributed_thermo_optic_runtime,
    build_triangle_p1_sampling_plan,
    pack_distributed_thermo_optic_inputs,
    pack_distributed_thermo_optic_inputs_host,
    prepare_distributed_triangle_p1_sampling_plan,
    thermo_optic_parameter_state,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def _sampling_plan(*, x_count: int = 4):
    coordinates = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)),
        dtype=np.float64,
    )
    cells = np.asarray(((0, 1, 2), (1, 3, 2)), dtype=np.int64)
    target = (
        np.linspace(0.2, 0.8, x_count, dtype=np.float64),
        np.asarray((-0.25, 0.25), dtype=np.float64),
        np.asarray((0.2, 0.8), dtype=np.float64),
    )
    return build_triangle_p1_sampling_plan(
        coordinates,
        cells,
        target,
        plane_axes=(0, 2),
    )


def _law() -> ThermoOpticLaw:
    return ThermoOpticLaw(
        material_region="silicon",
        reference_temperature_k=300.0,
        reference_refractive_index=2.0,
        thermo_optic_coefficient_per_k=1.0e-2,
        vacuum_wavelength_m=1.55e-6,
    )


def _contract(plan, *, dtype: str = "float64") -> FDTDXDeviceParameterContract:
    return FDTDXDeviceParameterContract(
        device_name="heated-silicon",
        target_shape=plan.target_shape,
        plane_axes=plan.plane_axes,
        lower_relative_permittivity=3.5,
        upper_relative_permittivity=5.0,
        parameter_dtype=dtype,
        thermo_optic_law_sha256=_law().sha256,
        target_coordinate_sha256=plan.target_coordinate_sha256,
        transfer_operator_sha256=plan.operator_sha256,
        fdtdx=FDTDXFingerprint("0.6.2", "1" * 40, "2" * 64),
    )


def _distributed_plan(*, partition_count: int = 2, x_count: int = 4):
    sampling = _sampling_plan(x_count=x_count)
    if partition_count == 1:
        source_cell_ids = np.asarray(((0, 1),), dtype=np.int64)
    elif partition_count == 2:
        source_cell_ids = np.asarray(((0,), (1,)), dtype=np.int64)
    else:  # pragma: no cover - tests use explicit invalid inputs for other counts
        raise AssertionError("unsupported test partition count")
    return sampling, prepare_distributed_triangle_p1_sampling_plan(
        sampling,
        source_cell_ids,
        source_layout_sha256="3" * 64,
    )


def test_distributed_plan_is_deterministic_hash_bound_and_explicitly_routed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _sampling, first = _distributed_plan()
    _sampling, second = _distributed_plan()

    assert first.schema_version == DISTRIBUTED_THERMO_OPTIC_SCHEMA
    assert first.operator_sha256 == second.operator_sha256
    assert first.partition_count == 2
    assert first.target_shard_shape == (2, 2, 2)
    assert first.transfer_capacity > 0
    assert np.any(first.send_active[0, 1])
    assert np.any(first.send_active[1, 0])
    assert np.array_equal(first.receive_active, np.transpose(first.send_active, (1, 0, 2)))
    assert not np.asarray(first.send_barycentric_weights).flags.writeable

    metadata = first.canonical_data()
    assert metadata["routing_collective"] == "all_to_all"
    assert metadata["global_gather"] is False
    assert metadata["actual_transfer_count"] == 16
    assert metadata["allocated_transfer_slots"] >= 16

    packed = pack_distributed_thermo_optic_inputs_host(first, value_dtype=np.float32)
    assert packed.send_source_cell_slots.dtype == np.int32
    assert packed.send_barycentric_weights.dtype == np.float32
    assert packed.send_active.dtype == np.bool_
    assert not packed.send_source_cell_slots.flags.writeable
    device_inputs = pack_distributed_thermo_optic_inputs(first, value_dtype=np.float64)
    assert device_inputs.send_barycentric_weights.dtype == jnp.float64

    with pytest.raises(ContractError, match="prepared plan"):
        pack_distributed_thermo_optic_inputs_host(object(), value_dtype=np.float64)  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="float32 or float64"):
        pack_distributed_thermo_optic_inputs_host(first, value_dtype=np.int32)
    real_iinfo = np.iinfo
    monkeypatch.setattr(
        np,
        "iinfo",
        lambda dtype: type("TinyIntegerInfo", (), {"max": 0})(),
    )
    with pytest.raises(ContractError, match="int32 addressability"):
        pack_distributed_thermo_optic_inputs_host(first, value_dtype=np.float64)
    monkeypatch.setattr(np, "iinfo", real_iinfo)


def test_single_device_runtime_matches_dense_sampling_and_reverse_mode() -> None:
    sampling, plan = _distributed_plan(partition_count=1)
    contract = _contract(sampling)
    mesh = Mesh(np.asarray((jax.devices("cpu")[0],), dtype=object), ("shard",))
    runtime = build_distributed_thermo_optic_runtime(plan, mesh, _law(), contract)
    inputs = pack_distributed_thermo_optic_inputs(plan, value_dtype=np.float64)
    nodal_temperature = jnp.asarray((301.0, 302.0, 303.0, 304.0), dtype=jnp.float64)
    cells = np.asarray(sampling.source_cells)
    cell_temperature = nodal_temperature[jnp.asarray(cells)][None, ...]

    observed = jax.jit(runtime.state)(inputs, cell_temperature)
    expected = thermo_optic_parameter_state(sampling, nodal_temperature, _law(), contract)
    for actual, reference in zip(observed[:4], expected[:4], strict=True):
        np.testing.assert_allclose(actual, reference, rtol=2.0e-14, atol=2.0e-14)
    assert np.all(np.asarray(observed.valid_cells))
    assert bool(observed.all_valid)

    direction = jnp.linspace(-0.5, 0.5, cell_temperature.size, dtype=jnp.float64).reshape(
        cell_temperature.shape
    )

    def objective(values: jax.Array) -> jax.Array:
        return jnp.mean(runtime.state(inputs, values).parameter)

    derivative = jnp.vdot(jax.grad(objective)(cell_temperature), direction)
    step = 1.0e-4
    finite_difference = (
        objective(cell_temperature + step * direction)
        - objective(cell_temperature - step * direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(derivative, finite_difference, rtol=2.0e-8, atol=2.0e-10)

    invalid = runtime.state(inputs, jnp.full_like(cell_temperature, 1000.0))
    assert not bool(invalid.all_valid)
    assert np.all(np.isnan(np.asarray(invalid.parameter)))


def test_lowering_rejects_source_layout_and_partition_drift() -> None:
    sampling = _sampling_plan()
    source_ids = np.asarray(((0,), (1,)), dtype=np.int64)
    with pytest.raises(ContractError, match="requires a P1 sampling plan"):
        prepare_distributed_triangle_p1_sampling_plan(  # type: ignore[arg-type]
            object(), source_ids, source_layout_sha256="3" * 64
        )
    with pytest.raises(ContractError, match="layout digest"):
        prepare_distributed_triangle_p1_sampling_plan(
            sampling, source_ids, source_layout_sha256="3" * 63
        )
    for changed, message in (
        (np.asarray((0, 1), dtype=np.int64), "rank-two"),
        (source_ids.astype(np.float64), "rank-two"),
        (np.asarray(((0,), (3,)), dtype=np.int64), "invalid id"),
        (np.asarray(((0,), (0,)), dtype=np.int64), "multiple owners"),
        (np.asarray(((0,), (2,)), dtype=np.int64), "omits"),
    ):
        with pytest.raises(ContractError, match=message):
            prepare_distributed_triangle_p1_sampling_plan(
                sampling,
                changed,
                source_layout_sha256="3" * 64,
            )
    with pytest.raises(ContractError, match="x extent"):
        prepare_distributed_triangle_p1_sampling_plan(
            _sampling_plan(x_count=3),
            source_ids,
            source_layout_sha256="3" * 64,
        )


def test_plan_rejects_corrupt_metadata_and_routing_arrays() -> None:
    _sampling, plan = _distributed_plan()
    cases = (
        ({"schema_version": "future"}, "schema"),
        ({"partition_count": 0}, "positive integer"),
        ({"source_cell_count": True}, "positive integer"),
        ({"mesh_axis_name": " shard"}, "axis name"),
        ({"target_sharding_axis": 1}, "shard the FDTDX x axis"),
        ({"target_shape": (4, 0, 2)}, "target shape"),
        ({"target_shape": (3, 2, 2)}, "divide over partitions"),
        ({"target_shard_shape": (1, 2, 2)}, "target shard shape"),
        ({"plane_axes": (0, 0)}, "plane axes"),
        ({"maximum_partition_error": np.nan}, "must be finite"),
        ({"maximum_partition_error": -1.0}, "cannot be negative"),
        ({"source_mesh_sha256": "0" * 63}, "source mesh digest"),
        ({"maximum_partition_error": plan.maximum_partition_error + 1.0e-3}, "partition"),
        ({"minimum_barycentric_weight": plan.minimum_barycentric_weight - 1.0e-3}, "minimum"),
        ({"operator_sha256": "0" * 64}, "operator digest"),
    )
    for changes, message in cases:
        with pytest.raises(ContractError, match=message):
            replace(plan, **changes)

    bad_ids = np.asarray(plan.source_cell_ids).copy()
    bad_ids[1, 0] = 0
    bad_masks = np.asarray(plan.receive_active).copy()
    bad_masks[0, 0, 0] = ~bad_masks[0, 0, 0]
    bad_slots = np.asarray(plan.send_source_cell_slots).copy()
    bad_slots[np.asarray(plan.send_active)] = plan.source_cell_capacity
    bad_inactive_slots = np.asarray(plan.send_source_cell_slots).copy()
    bad_inactive_slots[~np.asarray(plan.send_active)] = 0
    bad_inactive_weights = np.asarray(plan.send_barycentric_weights).copy()
    bad_inactive_weights[~np.asarray(plan.send_active)] = 1.0
    bad_receive = np.asarray(plan.receive_target_local_indices).copy()
    bad_receive[np.asarray(plan.receive_active)] = np.prod(plan.target_shard_shape)
    nan_weights = np.asarray(plan.send_barycentric_weights).copy()
    nan_weights[np.asarray(plan.send_active)] = np.nan
    corruptions = (
        ({"source_cell_ids": np.asarray(plan.source_cell_ids)[:, :0]}, "source-cell table shape"),
        (
            {"send_source_cell_slots": np.asarray(plan.send_source_cell_slots)[:, :, :0]},
            "routing-index shape",
        ),
        (
            {"send_barycentric_weights": np.asarray(plan.send_barycentric_weights)[..., :2]},
            "barycentric-weight shape",
        ),
        ({"send_active": np.asarray(plan.send_active)[:, :, :0]}, "routing-mask shape"),
        ({"source_cell_ids": np.asarray(plan.source_cell_ids, dtype=np.float64)}, "source indices"),
        (
            {
                "receive_target_local_indices": np.asarray(
                    plan.receive_target_local_indices,
                    dtype=np.float64,
                )
            },
            "destination indices",
        ),
        ({"send_barycentric_weights": nan_weights}, "finite real values"),
        ({"send_active": np.asarray(plan.send_active, dtype=np.int32)}, "masks must be boolean"),
        ({"source_cell_ids": bad_ids}, "cover every cell once"),
        ({"source_cell_ids": -np.ones_like(plan.source_cell_ids)}, "invalid id"),
        ({"receive_active": bad_masks}, "masks disagree"),
        ({"send_source_cell_slots": bad_slots}, "active source slot"),
        ({"send_source_cell_slots": bad_inactive_slots}, "inactive source slots"),
        ({"send_barycentric_weights": bad_inactive_weights}, "inactive weights"),
        ({"receive_target_local_indices": bad_receive}, "destination index"),
    )
    for changes, message in corruptions:
        with pytest.raises(ContractError, match=message):
            replace(plan, **changes)

    send_active = np.asarray(plan.send_active).copy()
    receive_active = np.asarray(plan.receive_active).copy()
    source_slots = np.asarray(plan.send_source_cell_slots).copy()
    weights = np.asarray(plan.send_barycentric_weights).copy()
    receive_indices = np.asarray(plan.receive_target_local_indices).copy()
    source, destination, slot = np.argwhere(send_active)[0]
    send_active[source, destination, slot] = False
    receive_active[destination, source, slot] = False
    source_slots[source, destination, slot] = plan.source_cell_capacity
    weights[source, destination, slot] = 0.0
    receive_indices[destination, source, slot] = np.prod(plan.target_shard_shape)
    with pytest.raises(ContractError, match="cover every target cell"):
        replace(
            plan,
            send_source_cell_slots=source_slots,
            send_barycentric_weights=weights,
            send_active=send_active,
            receive_target_local_indices=receive_indices,
            receive_active=receive_active,
        )

    inactive_receive = np.asarray(plan.receive_target_local_indices).copy()
    inactive_receive[~np.asarray(plan.receive_active)] = 0
    with pytest.raises(ContractError, match="inactive destinations"):
        replace(plan, receive_target_local_indices=inactive_receive)

    duplicate_receive = np.asarray(plan.receive_target_local_indices).copy()
    destination = next(
        index
        for index in range(plan.partition_count)
        if np.count_nonzero(plan.receive_active[index]) >= 2
    )
    positions = np.argwhere(plan.receive_active[destination])[:2]
    duplicate_receive[destination, *positions[1]] = duplicate_receive[destination, *positions[0]]
    with pytest.raises(ContractError, match="cover every local cell"):
        replace(plan, receive_target_local_indices=duplicate_receive)


def test_runtime_rejects_semantic_shape_and_dtype_drift() -> None:
    sampling, plan = _distributed_plan(partition_count=1)
    law = _law()
    contract = _contract(sampling)
    mesh = Mesh(np.asarray((jax.devices("cpu")[0],), dtype=object), ("shard",))
    inputs = pack_distributed_thermo_optic_inputs(plan, value_dtype=np.float64)
    temperatures = jnp.full((1, 2, 3), 302.0, dtype=jnp.float64)

    with pytest.raises(ContractError, match="prepared plan"):
        build_distributed_thermo_optic_runtime(object(), mesh, law, contract)  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="explicit JAX Mesh"):
        build_distributed_thermo_optic_runtime(plan, object(), law, contract)
    wrong_axis = Mesh(np.asarray((jax.devices("cpu")[0],), dtype=object), ("wrong",))
    with pytest.raises(ContractError, match="one-dimensional axis"):
        build_distributed_thermo_optic_runtime(plan, wrong_axis, law, contract)
    _sampling, two_partition_plan = _distributed_plan()
    with pytest.raises(ContractError, match="one device per partition"):
        build_distributed_thermo_optic_runtime(two_partition_plan, mesh, law, contract)
    with pytest.raises(ContractError, match="physical law"):
        build_distributed_thermo_optic_runtime(plan, mesh, object(), contract)  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="device contract"):
        build_distributed_thermo_optic_runtime(plan, mesh, law, object())  # type: ignore[arg-type]
    for changed, message in (
        (replace(contract, target_shape=(2, 2, 2)), "geometry"),
        (replace(contract, plane_axes=(1, 2)), "geometry"),
        (replace(contract, thermo_optic_law_sha256="0" * 64), "physical law"),
        (replace(contract, target_coordinate_sha256="0" * 64), "target coordinates"),
        (replace(contract, transfer_operator_sha256="0" * 64), "sampling operator"),
    ):
        with pytest.raises(ContractError, match=message):
            build_distributed_thermo_optic_runtime(plan, mesh, law, changed)

    runtime = build_distributed_thermo_optic_runtime(plan, mesh, law, contract)
    with pytest.raises(ContractError, match="packed contract"):
        runtime.state(object(), temperatures)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="inputs disagree"):
        runtime.state(
            inputs._replace(send_source_cell_slots=inputs.send_source_cell_slots[:, :, :1]),
            temperatures,
        )
    with pytest.raises(ValueError, match="cell temperature"):
        runtime.state(inputs, temperatures[:, :1])
    with pytest.raises(TypeError, match="source slots"):
        runtime.state(
            inputs._replace(
                send_source_cell_slots=inputs.send_source_cell_slots.astype(jnp.float64)
            ),
            temperatures,
        )
    with pytest.raises(TypeError, match="target indices"):
        runtime.state(
            inputs._replace(
                receive_target_local_indices=inputs.receive_target_local_indices.astype(jnp.float64)
            ),
            temperatures,
        )
    with pytest.raises(TypeError, match="activity masks"):
        runtime.state(
            inputs._replace(send_active=inputs.send_active.astype(jnp.int32)), temperatures
        )
    with pytest.raises(TypeError, match="real floating"):
        runtime.state(inputs, temperatures.astype(jnp.int32))
    with pytest.raises(TypeError, match="share one dtype"):
        runtime.state(
            inputs._replace(
                send_barycentric_weights=inputs.send_barycentric_weights.astype(jnp.float32)
            ),
            temperatures,
        )
    float32_contract = replace(contract, parameter_dtype="float32")
    float32_runtime = build_distributed_thermo_optic_runtime(
        plan,
        mesh,
        law,
        float32_contract,
    )
    with pytest.raises(TypeError, match="differs from the contract"):
        float32_runtime.state(inputs, temperatures)


def test_packed_input_type_is_not_satisfied_by_an_untyped_tuple() -> None:
    _sampling, plan = _distributed_plan(partition_count=1)
    packed = pack_distributed_thermo_optic_inputs(plan, value_dtype=np.float64)
    assert isinstance(packed, PackedDistributedThermoOpticInputs)
