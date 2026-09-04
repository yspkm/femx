from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

jax = pytest.importorskip("jax")
from scripts import run_tpu_public_ring_heater_forward_evidence as runner  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.requires_jax]


def test_runner_contract_and_environment_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert runner.EVIDENCE_SCHEMA.endswith("tpu_forward_process/v1")
    assert runner.ENTRY_CLAIM_SCHEMA.endswith("tpu_forward_entry_claim/v1")
    assert runner.ONE_SHOT_TIMING_SCHEMA.endswith("tpu_forward_timing/v1")
    assert runner.RUNTIME_SCALAR_CONTRACT["state_dtype"] == "float32"
    assert runner.RUNTIME_SCALAR_CONTRACT["fallback_allowed"] is False

    monkeypatch.delenv("FEMX_TEST_COUNT", raising=False)
    assert runner._positive_environment_integer("FEMX_TEST_COUNT") is None
    assert runner._nonnegative_environment_integer("FEMX_TEST_COUNT") is None
    with pytest.raises(RuntimeError, match="must be set"):
        runner._require_expected_count("FEMX_TEST_COUNT", 8, 8)
    monkeypatch.setenv("FEMX_TEST_COUNT", "8")
    runner._require_expected_count("FEMX_TEST_COUNT", 8, 8)
    with pytest.raises(RuntimeError, match="requires 8, observed 4"):
        runner._require_expected_count("FEMX_TEST_COUNT", 4, 8)
    for invalid in ("0", "-1", "bad"):
        monkeypatch.setenv("FEMX_TEST_COUNT", invalid)
        with pytest.raises(RuntimeError, match="positive integer"):
            runner._positive_environment_integer("FEMX_TEST_COUNT")
    monkeypatch.setenv("FEMX_TEST_COUNT", "-1")
    with pytest.raises(RuntimeError, match="nonnegative integer"):
        runner._nonnegative_environment_integer("FEMX_TEST_COUNT")
    monkeypatch.setenv("FEMX_TEST_COUNT", "bad")
    with pytest.raises(RuntimeError, match="nonnegative integer"):
        runner._nonnegative_environment_integer("FEMX_TEST_COUNT")
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    with pytest.raises(RuntimeError, match="must be set before Python starts"):
        runner._runtime()


def test_manifest_and_worker_entry_claim_are_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = tmp_path / "run"
    metadata = remote / ".phoxla"
    metadata.mkdir(parents=True)
    manifest = {
        "run_id": "run-1",
        "profile": "v4-od-32",
        "source": {"digest": "a" * 64, "commit": "b" * 40},
        "config": {"digest": "c" * 64},
    }
    (metadata / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("PHOXLA_RUN_ID", "run-1")
    monkeypatch.setenv("PHOXLA_PROCESS_INDEX", "3")
    monkeypatch.setenv("PHOXLA_GCLOUD_WORKER_INDEX", "3")

    provenance = runner._manifest_provenance(remote)
    claim = runner._claim_worker_entry(remote, provenance)

    assert provenance["source_commit"] == "b" * 40
    assert claim["process_index"] == 3
    assert claim["worker_index"] == 3
    with pytest.raises(RuntimeError, match="duplicate"):
        runner._claim_worker_entry(remote, provenance)


def test_manifest_and_entry_claim_reject_changed_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = tmp_path / "run"
    (remote / ".phoxla").mkdir(parents=True)
    (remote / ".phoxla" / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid deployed"):
        runner._manifest_provenance(remote)

    provenance = {
        "run_id": "run-1",
        "source_digest": "a" * 64,
        "config_digest": "b" * 64,
    }
    monkeypatch.setenv("PHOXLA_RUN_ID", "other")
    monkeypatch.setenv("PHOXLA_PROCESS_INDEX", "0")
    monkeypatch.setenv("PHOXLA_GCLOUD_WORKER_INDEX", "0")
    with pytest.raises(RuntimeError, match="disagrees"):
        runner._claim_worker_entry(remote, provenance)


def test_runner_memory_and_stablehlo_reports_are_conservative() -> None:
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

    hlo = runner._stablehlo_report(
        "stablehlo.collective_permute stablehlo.all_reduce tensor<1xf64> all_gather"
    )
    assert hlo["collective_permute_count"] == 1
    assert hlo["all_reduce_count"] == 1
    assert hlo["contains_all_gather"] is True
    assert hlo["contains_f64"] is True
    assert len(str(hlo["sha256"])) == 64


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


def test_json_nonfinite_diagnostic_remains_serializable() -> None:
    values = {
        "finite": 2.0,
        "nan": float("nan"),
        "positive": float("inf"),
        "negative": float("-inf"),
    }
    paths = runner._json_nonfinite_paths({"values": list(values.values()), "ok": None})
    assert paths == ["$.values[1]", "$.values[2]", "$.values[3]"]
    diagnostic = runner._diagnostic_payload(
        {"run_id": "run-1"},
        0,
        values,
        paths,
    )
    classes = cast(dict[str, dict[str, object]], diagnostic["scalar_classifications"])
    assert classes["finite"] == {
        "finite": True,
        "classification": "finite",
        "value": 2.0,
    }
    assert classes["nan"]["classification"] == "nan"
    assert classes["positive"]["classification"] == "positive_infinity"
    assert classes["negative"]["classification"] == "negative_infinity"
    json.dumps(diagnostic, allow_nan=False)


def test_atomic_writers_refuse_replacement(tmp_path: Path) -> None:
    record = tmp_path / "evidence.json"
    runner._atomic_json(record, {"status": "passed"})
    assert json.loads(record.read_text()) == {"status": "passed"}
    with pytest.raises(RuntimeError, match="overwrite"):
        runner._atomic_json(record, {"status": "changed"})

    text = tmp_path / "program.mlir"
    runner._atomic_text(text, "module {}")
    assert text.read_text() == "module {}"
    with pytest.raises(RuntimeError, match="overwrite"):
        runner._atomic_text(text, "changed")


def test_scalar_helpers_and_shard_hashes() -> None:
    value = jax.numpy.asarray([1.0, 2.0], dtype=jax.numpy.float32)
    assert runner._host_scalar(jax, value[0]) == 1.0
    assert runner._host_integer(jax, jax.numpy.asarray(3)) == 3
    assert runner._host_boolean(jax, jax.numpy.asarray(True)) is True
    assert runner._relative_error(2.0, 2.0) == 0.0

    hashes = runner._shard_hashes(jax, value.reshape((1, 2)))
    assert hashes == [
        {
            "partition_index": 0,
            "shape": [1, 2],
            "dtype": "float32",
            "sha256": __import__("hashlib")
            .sha256(np.ascontiguousarray([[1.0, 2.0]], dtype="<f4").tobytes())
            .hexdigest(),
            "finite": True,
        }
    ]
