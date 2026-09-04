from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec  # noqa: E402
from tests.fdtdx_mode_source_support import (  # noqa: E402
    LOCKED_FDTDX_MODE_SOURCE,
    uniform_mode_bundle,
)

from femx.core.errors import ContractError  # noqa: E402
from femx.interop.fdtdx import (  # noqa: E402
    FDTDXDistributedModeSourceBinding,
    bind_fdtdx_distributed_mode_source,
    build_fdtdx_mode_source_contract,
    lower_mode_source_inputs_for_tpu,
    make_fdtdx_distributed_mode_source,
    make_fdtdx_distributed_mode_source_function,
)
from femx.interop.fdtdx import distributed_mode_source as distributed_module  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


class _Grid:
    def __init__(self, edges: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        self._edges = edges

    def edges(self, axis: int) -> np.ndarray:
        return self._edges[axis]


@dataclass(frozen=True)
class _Source:
    name: str
    grid_shape: tuple[int, int, int]
    propagation_axis: int
    direction: str
    normalize: bool
    allow_device_overlap: bool
    wave_character: Any
    _E: Any
    _H: Any
    _inv_permittivity: Any
    _inv_permeability: Any
    _time_offset_E: Any
    _time_offset_H: Any
    _config: Any
    grid_slice_tuple: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]

    def aset(self, name: str, value: object, *, create_new_ok: bool = False) -> _Source:
        assert create_new_ok
        return replace(self, **{name: value})


@dataclass(frozen=True)
class _Objects:
    object_list: tuple[_Source, ...]

    def __getitem__(self, name: str) -> _Source:
        return self.object_list[self.index(name)]

    def index(self, name: str) -> int:
        return next(index for index, source in enumerate(self.object_list) if source.name == name)

    def aset(self, path: str, value: object) -> _Objects:
        index = int(path.removeprefix("object_list->[").removesuffix("]"))
        updated = list(self.object_list)
        updated[index] = value  # type: ignore[assignment]
        return _Objects(tuple(updated))


def _inputs():
    cell_count_x = max(4, jax.device_count())
    x_edges = np.arange(cell_count_x + 1, dtype=np.float64) * 40.0e-9
    y_edges = np.arange(5, dtype=np.float64) * 45.0e-9
    z_edges = np.asarray((80.0e-9, 120.0e-9), dtype=np.float64)
    canonical = uniform_mode_bundle(
        x_edges=x_edges,
        y_edges=y_edges,
        source_z_edges=z_edges,
        relative_permittivity=2.085136,
    )
    inverse_permittivity = np.full(
        (1, *canonical.electric.grid.shape),
        1.0 / 2.085136,
        dtype=np.float64,
    )
    lowered = lower_mode_source_inputs_for_tpu(
        canonical,
        expected_inverse_permittivity=inverse_permittivity,
        expected_inverse_permeability=np.asarray(1.0, dtype=np.float64),
    )
    contract = build_fdtdx_mode_source_contract(
        lowered.bundle,
        source_name="distributed-port",
        expected_inverse_permittivity=lowered.expected_inverse_permittivity,
        expected_inverse_permeability=lowered.expected_inverse_permeability,
        fdtdx=LOCKED_FDTDX_MODE_SOURCE,
    )
    return lowered.bundle, contract


def _sharding() -> NamedSharding:
    mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("shard",))
    return NamedSharding(mesh, PartitionSpec(None, "shard", None, None))


def _distributed(values: object):
    array = np.asarray(values)
    return jax.make_array_from_process_local_data(
        _sharding(),
        array,
        global_shape=array.shape,
    )


def _coordinates(bundle) -> tuple[jax.Array, jax.Array, jax.Array]:
    centers = []
    for edges in bundle.electric.grid.edge_coordinates:
        edge_array = np.asarray(edges)
        centers.append(0.5 * (edge_array[:-1] + edge_array[1:]))
    return tuple(jnp.asarray(item) for item in np.meshgrid(*centers, indexing="ij"))  # type: ignore[return-value]


def _source_and_objects(bundle, contract):
    callback = make_fdtdx_distributed_mode_source_function(bundle, contract)
    inverse_permittivity = _distributed(contract.expected_inverse_permittivity)
    electric, magnetic = callback(
        coordinates=_coordinates(bundle),
        frequency=contract.frequency_hz,
        propagation_axis=contract.propagation_axis,
        inv_permittivity=inverse_permittivity,
        inv_permeability=np.asarray(1.0, dtype=np.float64),
    )
    shape = electric.shape
    offsets = jnp.zeros(shape, dtype=jnp.float32)
    edges = tuple(np.asarray(item) for item in bundle.electric.grid.edge_coordinates)
    source = _Source(
        name=contract.source_name,
        grid_shape=contract.grid_shape,
        propagation_axis=contract.propagation_axis,
        direction=contract.propagation_direction,
        normalize=False,
        allow_device_overlap=False,
        wave_character=SimpleNamespace(get_frequency=lambda: contract.frequency_hz),
        _E=electric,
        _H=magnetic,
        _inv_permittivity=inverse_permittivity,
        _inv_permeability=np.asarray(1.0, dtype=np.float64),
        _time_offset_E=offsets,
        _time_offset_H=offsets,
        _config=SimpleNamespace(resolved_grid=_Grid(edges)),
        grid_slice_tuple=(
            (0, contract.grid_shape[0]),
            (0, contract.grid_shape[1]),
            (0, contract.grid_shape[2]),
        ),
    )
    return source, _Objects((source,))


def test_distributed_callback_materializes_exact_named_shards() -> None:
    bundle, contract = _inputs()
    callback = make_fdtdx_distributed_mode_source_function(bundle, contract)
    inverse_permittivity = _distributed(contract.expected_inverse_permittivity)

    electric, magnetic = callback(
        coordinates=_coordinates(bundle),
        frequency=contract.frequency_hz,
        propagation_axis=contract.propagation_axis,
        inv_permittivity=inverse_permittivity,
        inv_permeability=np.asarray(1.0, dtype=np.float64),
    )

    assert isinstance(electric.sharding, NamedSharding)
    assert electric.sharding == inverse_permittivity.sharding
    assert magnetic.sharding == inverse_permittivity.sharding
    np.testing.assert_array_equal(np.asarray(electric), bundle.electric.values)
    np.testing.assert_array_equal(np.asarray(magnetic), bundle.magnetic.values)


def test_distributed_callback_fails_closed_on_runtime_drift() -> None:
    bundle, contract = _inputs()
    callback = make_fdtdx_distributed_mode_source_function(bundle, contract)
    common = {
        "coordinates": _coordinates(bundle),
        "frequency": contract.frequency_hz,
        "propagation_axis": contract.propagation_axis,
        "inv_permeability": np.asarray(1.0, dtype=np.float64),
    }

    with pytest.raises(ContractError, match="NamedSharding"):
        callback(inv_permittivity=jnp.asarray(contract.expected_inverse_permittivity), **common)

    changed = np.asarray(contract.expected_inverse_permittivity).copy()
    changed[:, 0] *= np.float32(0.5)
    with pytest.raises(ContractError, match="addressable shard differs"):
        callback(inv_permittivity=_distributed(changed), **common)

    with pytest.raises(ContractError, match="inverse permeability differs"):
        callback(
            inv_permittivity=_distributed(contract.expected_inverse_permittivity),
            **{**common, "inv_permeability": np.asarray(0.5, dtype=np.float64)},
        )

    with pytest.raises(ContractError, match="mesh axis name"):
        make_fdtdx_distributed_mode_source_function(bundle, contract, mesh_axis_name=" ")


def test_binding_shards_time_offsets_and_records_only_a_local_contract() -> None:
    bundle, contract = _inputs()
    _source, objects = _source_and_objects(bundle, contract)

    updated, binding = bind_fdtdx_distributed_mode_source(objects, bundle, contract)

    placed = updated[contract.source_name]
    assert isinstance(placed._time_offset_E.sharding, NamedSharding)
    assert placed._time_offset_E.sharding == placed._E.sharding
    assert placed._time_offset_H.sharding == placed._E.sharding
    assert binding.partition_spec == ("replicated", "shard", "replicated", "replicated")
    assert binding.global_device_count == jax.device_count()
    assert binding.local_device_count == jax.local_device_count()
    assert binding.process_count == jax.process_count()
    assert binding.canonical_data()["physical_evidence"] is False
    assert len(binding.sha256) == 64


def test_distributed_constructor_supplies_the_specialized_callback(monkeypatch) -> None:
    bundle, contract = _inputs()
    captured: dict[str, object] = {}

    def fake_construct(*args, **kwargs):
        captured.update(kwargs)
        return "source"

    monkeypatch.setattr(distributed_module, "_construct_fdtdx_mode_source", fake_construct)
    result = make_fdtdx_distributed_mode_source(
        bundle,
        contract,
        verified_fingerprint=LOCKED_FDTDX_MODE_SOURCE,
    )

    assert result == "source"
    assert callable(captured["mode_function"])
    assert captured["allow_profile_updates"] is False


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"source_name": " "}, "source name"),
        ({"source_contract_sha256": "bad"}, "source-contract digest"),
        ({"partition_spec": ("shard", "replicated", "replicated", "replicated")}, "first spatial"),
        ({"process_index": 2}, "runtime counts"),
        ({"addressable_x_ranges": ()}, "one local x range"),
    ],
)
def test_binding_record_rejects_inconsistent_metadata(changes, message: str) -> None:
    binding = FDTDXDistributedModeSourceBinding(
        source_name="port",
        source_contract_sha256="a" * 64,
        mesh_axis_name="shard",
        partition_spec=("replicated", "shard", "replicated", "replicated"),
        global_shape=(3, 4, 4, 1),
        field_dtype="complex64",
        time_offset_dtype="float32",
        global_device_count=1,
        local_device_count=1,
        process_count=1,
        process_index=0,
        addressable_x_ranges=((0, 4),),
    )

    with pytest.raises(ContractError, match=message):
        replace(binding, **changes)


def test_binding_requires_an_immutable_container() -> None:
    bundle, contract = _inputs()
    with pytest.raises(ContractError, match="immutable ObjectContainer"):
        bind_fdtdx_distributed_mode_source(object(), bundle, contract)
