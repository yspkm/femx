from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from jax.experimental import multihost_utils  # noqa: E402
from scripts import _distributed_fdtdx_thermo_optic_case as case  # noqa: E402
from scripts import run_tpu_distributed_fdtdx_thermo_optic_evidence as runner  # noqa: E402

from femx.backends.jax.distributed_electrothermal import (  # noqa: E402
    PackedDistributedElectrothermalInputs,
)
from femx.interop.fdtdx import PackedDistributedThermoOpticInputs  # noqa: E402
from femx.validation import tpu_distributed_fdtdx_thermo_optic_evidence as admission  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


@dataclass(frozen=True)
class _Device:
    identifier: int
    device_kind: str = "TPU v4"


@dataclass(frozen=True)
class _Mesh:
    devices: np.ndarray
    axis_names: tuple[str, ...] = ("shard",)
    empty: bool = False

    @property
    def size(self) -> int:
        return int(self.devices.size)


def test_runner_and_admission_lock_the_same_physical_case() -> None:
    assert admission.FDTDX_PACKAGE_VERSION == case.FDTDX_PACKAGE_VERSION
    assert admission.FDTDX_SOURCE_REVISION == case.FDTDX_SOURCE_REVISION
    assert admission.FDTDX_SOURCE_DIGEST == case.FDTDX_SOURCE_DIGEST
    assert dict(admission.FDTDX_MODULE_SHA256) == case.FDTDX_MODULE_SHA256
    assert admission.EXPECTED_GRID_SHAPE == case.GRID_SHAPE
    assert admission.EXPECTED_DEVICE_SHAPE == case.DEVICE_SHAPE
    assert admission.GRID_SPACING_M == case.GRID_SPACING_M
    assert (
        admission.TOLERANCES["runtime_coordinate_max_ulp_error"]
        == case.RUNTIME_TARGET_COORDINATE_MAX_ULP_ERROR
    )
    assert (
        admission.TOLERANCES["runtime_coordinate_max_grid_fraction_error"]
        == case.RUNTIME_TARGET_COORDINATE_MAX_GRID_FRACTION_ERROR
    )


def test_runner_report_names_exactly_match_process_set_admission() -> None:
    input_names = {
        f"input-{name.replace('_', '-')}" for name in PackedDistributedElectrothermalInputs._fields
    }
    partitioned = {
        f"input-{name.replace('_', '-')}"
        for name in PackedDistributedElectrothermalInputs._fields
        if name in runner._ELECTROTHERMAL_PARTITIONED_FIELDS
    }
    partitioned.update(
        {
            "authority-cell-temperature",
            "authority-potential",
            "authority-temperature",
            "authority-thermo-optic-parameter",
        }
    )
    partitioned.update(
        f"transfer-{name.replace('_', '-')}" for name in PackedDistributedThermoOpticInputs._fields
    )
    replicated = input_names - partitioned
    replicated.update({"current-parameters", "thermal-parameters", "feedback-parameters"})

    assert tuple(sorted(partitioned)) == admission.PARTITIONED_ARRAY_REPORT_NAMES
    assert tuple(sorted(replicated)) == admission.REPLICATED_ARRAY_REPORT_NAMES


def test_runtime_initializes_before_claim_and_fdtdx_import() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    main_source = source[source.index("def main(") :]
    runtime = main_source.index("jax = _runtime()")
    claim = main_source.index("launch_claim = _claim_worker_entry")
    fdtdx_import = main_source.index("import fdtdx")
    assert runtime < claim < fdtdx_import


def test_coupled_outer_jit_receives_every_globally_sharded_runtime_input() -> None:
    assert case.CoupledRuntimeInputs._fields == (
        "electrothermal",
        "thermo_optic",
        "fdtdx_arrays",
        "fdtdx_objects",
        "fdtdx_parameters",
        "fdtdx_config",
        "fdtdx_key",
    )
    source = Path(runner.__file__).read_text(encoding="utf-8")
    graph = source[
        source.index("    def downstream_phasor(") : source.index(
            "    finite_difference_gradients:"
        )
    ]
    for field in case.CoupledRuntimeInputs._fields:
        assert f"inputs.{field}" in graph
    assert "scene." not in graph
    assert "electrothermal_inputs" not in graph
    assert "transfer_inputs" not in graph


def test_parser_and_environment_contracts_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = runner.build_parser().parse_args(["--input", "inputs/coupled"])
    assert parsed.input == Path("inputs/coupled")
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args([])

    monkeypatch.delenv("FEMX_COUNT", raising=False)
    assert runner._positive_environment_integer("FEMX_COUNT") is None
    assert runner._nonnegative_environment_integer("FEMX_COUNT") is None
    for value in ("0", "-1", "bad"):
        monkeypatch.setenv("FEMX_COUNT", value)
        with pytest.raises(RuntimeError, match="positive integer"):
            runner._positive_environment_integer("FEMX_COUNT")
    for value in ("-1", "bad"):
        monkeypatch.setenv("FEMX_COUNT", value)
        with pytest.raises(RuntimeError, match="nonnegative integer"):
            runner._nonnegative_environment_integer("FEMX_COUNT")

    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    with pytest.raises(RuntimeError, match="must be set before Python starts"):
        runner._runtime()
    monkeypatch.setenv("JAX_PLATFORMS", "tpu,cpu")
    monkeypatch.delenv("JAX_DEFAULT_MATMUL_PRECISION", raising=False)
    with pytest.raises(RuntimeError, match="JAX_DEFAULT_MATMUL_PRECISION=highest"):
        runner._runtime()


def test_manifest_and_worker_claim_are_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = tmp_path / "run"
    metadata = remote / ".phoxla"
    metadata.mkdir(parents=True)
    manifest = {
        "run_id": "run-1",
        "profile": "v4-od-32",
        "source": {"commit": "a" * 40, "digest": "b" * 64},
        "config": {"digest": "c" * 64},
    }
    (metadata / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("PHOXLA_RUN_ID", "run-1")
    monkeypatch.setenv("PHOXLA_PROCESS_INDEX", "3")
    monkeypatch.setenv("PHOXLA_GCLOUD_WORKER_INDEX", "3")

    provenance = runner._manifest_provenance(remote)
    claim = runner._claim_worker_entry(remote, provenance)
    assert claim["process_index"] == 3
    assert claim["worker_index"] == 3
    stored = json.loads(
        (remote / "logs" / "femx-fdtdx-thermo-optic-entry.claim" / "identity.json").read_text(
            encoding="utf-8"
        )
    )
    assert stored == claim
    with pytest.raises(RuntimeError, match="duplicate"):
        runner._claim_worker_entry(remote, provenance)

    bad = tmp_path / "bad"
    (bad / ".phoxla").mkdir(parents=True)
    (bad / ".phoxla" / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid deployed"):
        runner._manifest_provenance(bad)


def test_json_hash_and_atomic_publication_reject_nonfinite_values(tmp_path: Path) -> None:
    value = {"b": [1.0, float("nan")], "a": (float("inf"), -float("inf"), 2.0)}
    assert runner._json_safe(value) == {"a": [None, None, 2.0], "b": [1.0, None]}
    first = runner._canonical_digest({"b": 2, "a": 1})
    second = runner._canonical_digest({"a": 1, "b": 2})
    assert first == second

    output = tmp_path / "record.json"
    with pytest.raises(ValueError):
        runner._atomic_json(output, {"bad": float("nan")})
    assert not output.exists()
    runner._atomic_json(output, runner._json_safe(value))
    assert json.loads(output.read_text(encoding="utf-8")) == runner._json_safe(value)


def test_process_zero_publishes_controller_visible_hlo_and_metrics(tmp_path: Path) -> None:
    payload = {"status": "passed"}
    stablehlo = {name: f"module @{name}" for name in admission.EXECUTABLE_NAMES}

    runner._publish_process_zero_compatibility(
        tmp_path / "nonzero",
        process_index=1,
        process_payload=payload,
        stablehlo_by_name=stablehlo,
    )
    assert not (tmp_path / "nonzero").exists()

    remote = tmp_path / "process-zero"
    runner._publish_process_zero_compatibility(
        remote,
        process_index=0,
        process_payload=payload,
        stablehlo_by_name=stablehlo,
    )
    assert json.loads((remote / "results" / "metrics.json").read_text()) == payload
    for name, text in stablehlo.items():
        assert (remote / "hlo" / f"{name}.stablehlo.mlir").read_text() == text


def test_host_packing_preserves_owned_and_cell_sentinels() -> None:
    layout = SimpleNamespace(
        topology=SimpleNamespace(
            free_nodes=np.asarray((2, 0), dtype=np.int32),
            cells=np.asarray(((0, 1, 2), (1, 2, 3)), dtype=np.int32),
        ),
        transport=SimpleNamespace(
            owned_dof_ids=np.asarray(((0, 2), (1, 2)), dtype=np.int32),
            cell_ids=np.asarray(((0, 2), (1, 2)), dtype=np.int32),
        ),
    )
    plan = SimpleNamespace(layout=layout)
    nodal = np.asarray((10.0, 20.0, 30.0, 40.0))

    np.testing.assert_array_equal(
        runner._host_pack_owned(layout, nodal),
        np.asarray(((30.0, 0.0), (10.0, 0.0))),
    )
    np.testing.assert_array_equal(
        runner._host_pack_cell_temperature(plan, nodal),
        np.asarray(
            (
                ((10.0, 20.0, 30.0), (0.0, 0.0, 0.0)),
                ((20.0, 30.0, 40.0), (0.0, 0.0, 0.0)),
            )
        ),
    )


def test_partition_spec_and_slice_bounds_are_canonical() -> None:
    assert runner._partition_spec(SimpleNamespace(spec=(None, "shard", ("x", "y")))) == [
        None,
        "shard",
        ["x", "y"],
    ]
    assert runner._slice_bounds((slice(None), slice(1, 4)), (2, 5)) == [[0, 2], [1, 4]]
    with pytest.raises(RuntimeError, match="NamedSharding"):
        runner._partition_spec(object())
    with pytest.raises(RuntimeError, match="unsupported"):
        runner._partition_spec(SimpleNamespace(spec=(1,)))
    with pytest.raises(RuntimeError, match="rank differs"):
        runner._slice_bounds((slice(None),), (2, 3))
    with pytest.raises(RuntimeError, match="contiguous"):
        runner._slice_bounds((slice(None, None, 2),), (2,))


def test_critical_array_report_binds_mesh_partitions_and_slices() -> None:
    first = _Device(0)
    second = _Device(1)
    mesh = SimpleNamespace(devices=np.asarray((first, second), dtype=object), size=2)
    shards = (
        SimpleNamespace(
            device=first,
            index=(slice(0, 1), slice(0, 2)),
            data=np.zeros((1, 2), dtype=np.float32),
        ),
        SimpleNamespace(
            device=second,
            index=(slice(1, 2), slice(0, 2)),
            data=np.zeros((1, 2), dtype=np.float32),
        ),
    )
    array = SimpleNamespace(
        shape=(2, 2),
        dtype=np.dtype(np.float32),
        sharding=SimpleNamespace(spec=("shard", None)),
        addressable_shards=shards,
    )
    report = runner._critical_array_report(
        "test-array",
        array,
        mesh,
        process_index=0,
        process_count=1,
    )
    addressable_shards = cast(list[dict[str, object]], report["addressable_shards"])
    assert report["partition_spec"] == ["shard", None]
    assert [item["partition_index"] for item in addressable_shards] == [0, 1]
    assert addressable_shards[1]["index"] == [[1, 2], [0, 2]]

    outside = _Device(2)
    array.addressable_shards = (
        SimpleNamespace(
            device=outside,
            index=(slice(0, 1), slice(0, 2)),
            data=np.zeros((1, 2), dtype=np.float32),
        ),
    )
    with pytest.raises(RuntimeError, match="outside the declared Mesh"):
        runner._critical_array_report(
            "test-array",
            array,
            mesh,
            process_index=0,
            process_count=1,
        )


def test_coordinate_admission_reports_rounding_and_rejects_bad_axes() -> None:
    expected = (
        np.asarray((1.0e-6, 1.0625e-6), dtype=np.float64),
        np.asarray((-3.125e-8, 3.125e-8), dtype=np.float64),
        np.asarray((1.5625e-7, 2.1875e-7), dtype=np.float64),
    )
    actual = tuple(jnp.asarray(axis, dtype=jnp.float32) for axis in expected)
    report = runner._coordinate_admission(jax, actual, expected)
    assert report["maximum_ulp_errors"] == [0, 0, 0]
    assert report["float32_rounding_exact"] == [True, True, True]
    assert report["admitted"] == [True, True, True]

    bad = (actual[0].at[0].set(-1.0), actual[1], actual[2])
    assert runner._coordinate_admission(jax, bad, expected)["admitted"][0] is False  # type: ignore[index]
    with pytest.raises(RuntimeError, match="exactly three axes"):
        runner._coordinate_admission(jax, actual[:2], expected)
    with pytest.raises(RuntimeError, match="shapes differ"):
        runner._coordinate_admission(jax, (actual[0][:-1], *actual[1:]), expected)


def test_coupled_mesh_adopts_the_concrete_fdtdx_device_order() -> None:
    devices = np.asarray((_Device(0), _Device(2), _Device(1), _Device(3)), dtype=object)
    mesh = _Mesh(devices)
    sharding = SimpleNamespace(mesh=mesh)

    assert (
        case.coupled_mesh_from_material_sharding(
            sharding,
            _Mesh,
            axis_name="shard",
            global_device_count=4,
        )
        is mesh
    )
    np.testing.assert_array_equal(mesh.devices, devices)

    for invalid in (
        SimpleNamespace(mesh=None),
        SimpleNamespace(mesh=_Mesh(devices, empty=True)),
        SimpleNamespace(mesh=_Mesh(devices, axis_names=("other",))),
        SimpleNamespace(mesh=_Mesh(devices.reshape((2, 2)))),
    ):
        with pytest.raises(RuntimeError, match="FDTDX material"):
            case.coupled_mesh_from_material_sharding(
                invalid,
                _Mesh,
                axis_name="shard",
                global_device_count=4,
            )


def test_material_difference_reduces_only_two_float32_scalars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = np.asarray(
        (
            ((10.0,), (11.0,)),
            ((12.0,), (13.0,)),
        ),
        dtype=np.float32,
    )
    inverse = np.ones((1, 4, 2, 1), dtype=np.float32)
    inverse[0, 1:3] = 1.0 / expected
    applied = SimpleNamespace(
        shape=inverse.shape,
        addressable_shards=(
            SimpleNamespace(
                index=(slice(0, 1), slice(0, 4), slice(0, 2), slice(0, 1)),
                data=inverse,
            ),
        ),
    )

    def gather(value: Any, *, tiled: bool) -> np.ndarray:
        observed = np.asarray(value)
        assert tiled is False
        assert observed.shape == (2,)
        assert observed.dtype == np.dtype(np.float32)
        return observed[None, :]

    monkeypatch.setattr(multihost_utils, "process_allgather", gather)
    fake_jax = SimpleNamespace(process_count=lambda: 1)
    assert (
        runner._material_relative_difference(
            fake_jax,
            applied,
            expected,
            device_grid_slice=(slice(1, 3), slice(0, 2), slice(0, 1)),
        )
        < 1.0e-7
    )
    with pytest.raises(RuntimeError, match="zero norm"):
        runner._material_relative_difference(
            fake_jax,
            applied,
            np.zeros_like(expected),
            device_grid_slice=(slice(1, 3), slice(0, 2), slice(0, 1)),
        )


def test_stablehlo_memory_and_relative_difference_reports() -> None:
    hlo = (
        "stablehlo.all_to_all stablehlo.collective_permute "
        "stablehlo.all_reduce stablehlo.all_gather f64"
    )
    report = runner._stablehlo_report(hlo)
    assert report["all_to_all_count"] == 1
    assert report["collective_permute_count"] == 1
    assert report["all_reduce_count"] == 1
    assert report["contains_all_gather"] is True
    assert report["contains_float64"] is True

    analysis = SimpleNamespace(
        generated_code_size_in_bytes=10,
        argument_size_in_bytes=100,
        output_size_in_bytes=40,
        alias_size_in_bytes=20,
        temp_size_in_bytes=30,
    )
    memory = runner._memory_report(SimpleNamespace(memory_analysis=lambda: analysis), 1000)
    assert memory["compiler_peak_bytes"] == 150
    assert memory["hbm_fraction"] == pytest.approx(0.15)
    assert runner._relative_difference(jax, np.ones(2), np.ones(2)) == 0.0
    assert runner._relative_difference(jax, np.ones(2), np.zeros(2)) == float("inf")

    with pytest.raises(RuntimeError, match="memory analysis"):
        runner._memory_report(SimpleNamespace(memory_analysis=lambda: None), 1000)
    bad = SimpleNamespace(
        memory_analysis=lambda: SimpleNamespace(
            generated_code_size_in_bytes=True,
            argument_size_in_bytes=100,
            output_size_in_bytes=40,
            alias_size_in_bytes=20,
            temp_size_in_bytes=30,
        )
    )
    with pytest.raises(RuntimeError, match="memory statistic"):
        runner._memory_report(bad, 1000)
