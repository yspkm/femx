from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

jax = pytest.importorskip("jax")
from scripts import build_tpu_distributed_electrothermal_inputs as builder  # noqa: E402
from scripts import run_tpu_distributed_electrothermal_evidence as runner  # noqa: E402
from scripts._tpu_distributed_electrothermal_plan import (  # noqa: E402
    ARRAYS_FILENAME,
    ARTIFACT_SCHEMA,
    MANIFEST_FILENAME,
    read_distributed_electrothermal_artifact,
    write_distributed_electrothermal_artifact,
)

import femx  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def _build_artifact(tmp_path: Path) -> Path:
    root = tmp_path / "inputs"
    manifest = builder.build_inputs(root, intervals=2, partition_count=4)
    assert manifest["schema_version"] == ARTIFACT_SCHEMA
    return root


def test_controller_builder_roundtrips_exact_plan_and_float64_authority(
    tmp_path: Path,
) -> None:
    root = _build_artifact(tmp_path)

    loaded = read_distributed_electrothermal_artifact(root)
    plan_record = loaded.manifest["plan"]
    assert isinstance(plan_record, dict)
    assert plan_record["sha256"] == loaded.plan.digest()
    assert plan_record["layout_sha256"] == loaded.plan.layout.digest()
    assert loaded.plan.layout.partition_count == 4
    assert loaded.plan.layout.topology.node_count == 9
    assert loaded.plan.layout.topology.cell_count == 8
    assert loaded.authority.potential.shape == (9,)
    assert loaded.authority.temperature.shape == (9,)
    assert loaded.authority.current_parameter_gradient.shape == (2,)
    assert loaded.authority.thermal_parameter_gradient.shape == (1,)
    assert loaded.authority.feedback_parameter_gradient.shape == (1,)
    assert loaded.authority.potential.dtype == np.float64
    assert not loaded.authority.potential.flags.writeable
    assert loaded.authority.forward_converged
    assert loaded.authority.adjoint_converged
    assert np.isfinite(loaded.authority.objective)
    assert len(loaded.arrays_sha256) == 64


def test_artifact_writer_refuses_replacement_and_invalid_source_identity(
    tmp_path: Path,
) -> None:
    root = _build_artifact(tmp_path)
    loaded = read_distributed_electrothermal_artifact(root)

    with pytest.raises(ValueError, match="new absolute path"):
        write_distributed_electrothermal_artifact(
            root,
            loaded.plan,
            loaded.authority,
            source_commit="a" * 40,
            case_metadata={"name": "replacement"},
        )
    with pytest.raises(ValueError, match="40-character"):
        write_distributed_electrothermal_artifact(
            tmp_path / "bad-source",
            loaded.plan,
            loaded.authority,
            source_commit="not-a-commit",
            case_metadata={"name": "bad-source"},
        )


def test_artifact_reader_rejects_array_and_manifest_tampering(tmp_path: Path) -> None:
    root = _build_artifact(tmp_path)
    arrays = root / ARRAYS_FILENAME
    arrays.write_bytes(arrays.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match=r"byte count|SHA-256"):
        read_distributed_electrothermal_artifact(root)

    second_root = builder.build_inputs(
        tmp_path / "manifest-tamper",
        intervals=2,
        partition_count=4,
    )
    assert second_root["schema_version"] == ARTIFACT_SCHEMA
    manifest_path = tmp_path / "manifest-tamper" / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["plan"]["partition_count"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=r"partition range|partition count|SHA-256"):
        read_distributed_electrothermal_artifact(tmp_path / "manifest-tamper")


def test_runner_contract_constants_and_environment_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert femx.__file__ is not None
    assert Path(femx.__file__).resolve().is_relative_to(runner.SOURCE_ROOT)
    assert runner.EVIDENCE_SCHEMA.endswith("tpu_evidence/v1")
    assert runner.NUMERICAL_DIAGNOSTIC_SCHEMA.endswith("tpu_numerical_diagnostic/v1")
    assert runner.WORKER_ENTRY_CLAIM_SCHEMA.endswith("worker_entry_claim/v1")
    assert runner.EXECUTION_SAMPLES == 5
    assert runner.REAL_SCALAR_CONTRACT["logical_dtype"] == "float32"
    assert runner.REAL_SCALAR_CONTRACT["precision_fallback"] is False
    monkeypatch.delenv("FEMX_TEST_COUNT", raising=False)
    assert runner._positive_environment_integer("FEMX_TEST_COUNT") is None
    assert runner._nonnegative_environment_integer("FEMX_TEST_COUNT") is None
    monkeypatch.setenv("FEMX_TEST_COUNT", "8")
    runner._require_expected_count("FEMX_TEST_COUNT", 8)
    with pytest.raises(RuntimeError, match="requires 8, observed 4"):
        runner._require_expected_count("FEMX_TEST_COUNT", 4)
    for invalid in ("0", "-1", "bad"):
        monkeypatch.setenv("FEMX_TEST_COUNT", invalid)
        with pytest.raises(RuntimeError, match="positive integer"):
            runner._positive_environment_integer("FEMX_TEST_COUNT")
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    with pytest.raises(RuntimeError, match="must be set before Python starts"):
        runner._runtime()


def test_runner_provenance_and_worker_entry_claim_are_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = tmp_path / "run"
    metadata = remote / ".phoxla"
    metadata.mkdir(parents=True)
    manifest = {
        "run_id": "run-1",
        "profile": "test-profile",
        "source": {"digest": "a" * 64, "commit": "b" * 40},
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
    with pytest.raises(RuntimeError, match="duplicate"):
        runner._claim_worker_entry(remote, provenance)


def test_runner_hlo_memory_and_relative_difference_reports() -> None:
    hlo = "stablehlo.collective_permute stablehlo.all_reduce"
    report = runner._stablehlo_report(hlo)
    assert report["collective_permute_count"] == 1
    assert report["all_reduce_count"] == 1
    assert report["contains_all_gather"] is False
    assert len(str(report["sha256"])) == 64

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


def test_runner_nonfinite_diagnostic_is_json_safe_and_names_each_path(
    tmp_path: Path,
) -> None:
    diagnostic = runner._numerical_diagnostic(
        jax,
        {
            "finite_array": np.array([1.0, 2.0]),
            "bad_array": np.array([np.nan, np.inf, -np.inf]),
        },
        {
            "finite_scalar": 3.0,
            "nan_scalar": float("nan"),
            "positive_inf_scalar": float("inf"),
            "negative_inf_scalar": float("-inf"),
        },
    )

    assert diagnostic["all_finite"] is False
    assert diagnostic["nonfinite_names"] == [
        "bad_array",
        "nan_scalar",
        "negative_inf_scalar",
        "positive_inf_scalar",
    ]
    arrays = diagnostic["arrays"]
    assert arrays["finite_array"]["all_finite"] is True
    assert arrays["bad_array"] == {
        "shape": [3],
        "dtype": "float64",
        "size": 3,
        "finite_count": 0,
        "nan_count": 1,
        "inf_count": 2,
        "all_finite": False,
    }
    scalars = diagnostic["scalars"]
    assert scalars["finite_scalar"] == {
        "finite": True,
        "classification": "finite",
        "value": 3.0,
    }
    assert scalars["nan_scalar"]["classification"] == "nan"
    assert scalars["positive_inf_scalar"]["classification"] == "positive_infinity"
    assert scalars["negative_inf_scalar"]["classification"] == "negative_infinity"
    assert all(scalars[name]["value"] is None for name in scalars if name != "finite_scalar")
    assert runner._json_nonfinite_paths(
        {"outer": [1.0, {"nan": float("nan"), "inf": float("inf")}], "ok": None}
    ) == ["$.outer[1].nan", "$.outer[1].inf"]
    json.dumps(diagnostic, allow_nan=False)

    output = tmp_path / "process"
    remote = tmp_path / "remote"
    runner._write_numerical_diagnostic(output, remote, 0, diagnostic)
    assert (
        json.loads((output / "results" / "numerical-diagnostic.json").read_text(encoding="utf-8"))
        == diagnostic
    )
    assert (
        json.loads((remote / "results" / "numerical-diagnostic.json").read_text(encoding="utf-8"))
        == diagnostic
    )
    second_output = tmp_path / "process-1"
    runner._write_numerical_diagnostic(second_output, remote, 1, diagnostic)
    assert (second_output / "results" / "numerical-diagnostic.json").is_file()


def test_runner_finiteness_reduces_before_host_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jax.numpy as jnp

    array = jnp.asarray([1.0, float("nan"), float("inf")])
    original_asarray = np.asarray

    def guarded_asarray(value: object, *args: object, **kwargs: object) -> np.ndarray:
        if value is array:
            raise AssertionError("the global JAX input must not be materialized on the host")
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(np, "asarray", guarded_asarray)

    assert runner._array_finiteness(jax, array) == {
        "shape": [3],
        "dtype": str(array.dtype),
        "size": 3,
        "finite_count": 1,
        "nan_count": 1,
        "inf_count": 1,
        "all_finite": False,
    }


@pytest.mark.parametrize(
    "compiled",
    [
        SimpleNamespace(memory_analysis=lambda: None),
        SimpleNamespace(
            memory_analysis=lambda: SimpleNamespace(
                generated_code_size_in_bytes=10,
                argument_size_in_bytes=True,
                output_size_in_bytes=40,
                alias_size_in_bytes=20,
                temp_size_in_bytes=30,
            )
        ),
    ],
)
def test_runner_memory_report_rejects_missing_or_invalid_data(compiled: Any) -> None:
    with pytest.raises(RuntimeError, match=r"memory analysis|memory statistic"):
        runner._memory_report(compiled, 1000)
