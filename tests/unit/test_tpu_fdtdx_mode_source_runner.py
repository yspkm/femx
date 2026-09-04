from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from scripts import run_tpu_fdtdx_mode_source_evidence as runner

pytestmark = pytest.mark.unit


def test_aot_call_passes_only_the_dynamic_lowering_pytree() -> None:
    captured: dict[str, object] = {}

    def compiled(**kwargs: object) -> str:
        captured.update(kwargs)
        return "state"

    result = runner._run_compiled_fdtdx(
        compiled,
        arrays="arrays",
        objects="objects",
        config="config",
        key="key",
    )

    assert result == "state"
    assert captured == {
        "arrays": "arrays",
        "objects": "objects",
        "config": "config",
        "key": "key",
    }
    assert "show_progress" not in captured
    assert "progress_callback" not in captured


def test_process_zero_publishes_controller_visible_compatibility_files(tmp_path: Path) -> None:
    payload = {"schema_version": runner.EVIDENCE_SCHEMA, "status": "passed"}

    runner._publish_process_zero_compatibility(
        tmp_path,
        process_index=0,
        process_payload=payload,
        stablehlo="module @fdtdx {}\n",
    )

    assert json.loads((tmp_path / "results" / "metrics.json").read_text()) == payload
    assert (
        tmp_path / "hlo" / "fdtdx-time-advance.stablehlo.mlir"
    ).read_text() == "module @fdtdx {}\n"


def test_nonzero_process_does_not_publish_compatibility_files(tmp_path: Path) -> None:
    runner._publish_process_zero_compatibility(
        tmp_path,
        process_index=1,
        process_payload={"status": "passed"},
        stablehlo="module @fdtdx {}\n",
    )

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("value", ["0", "-1", "bad"])
def test_positive_environment_integer_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("FEMX_TEST_COUNT", value)
    with pytest.raises(RuntimeError, match="positive integer"):
        runner._positive_environment_integer("FEMX_TEST_COUNT")


def test_environment_count_helpers_and_expected_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FEMX_TEST_COUNT", raising=False)
    assert runner._positive_environment_integer("FEMX_TEST_COUNT") is None
    assert runner._nonnegative_environment_integer("FEMX_TEST_COUNT") is None
    monkeypatch.setenv("FEMX_TEST_COUNT", "4")
    assert runner._positive_environment_integer("FEMX_TEST_COUNT") == 4
    assert runner._nonnegative_environment_integer("FEMX_TEST_COUNT") == 4
    runner._require_expected_count("FEMX_TEST_COUNT", 4)
    with pytest.raises(RuntimeError, match="requires 4, observed 3"):
        runner._require_expected_count("FEMX_TEST_COUNT", 3)
    monkeypatch.setenv("FEMX_TEST_COUNT", "-1")
    with pytest.raises(RuntimeError, match="nonnegative integer"):
        runner._nonnegative_environment_integer("FEMX_TEST_COUNT")
    monkeypatch.setenv("FEMX_TEST_COUNT", "bad")
    with pytest.raises(RuntimeError, match="nonnegative integer"):
        runner._nonnegative_environment_integer("FEMX_TEST_COUNT")


def test_manifest_and_worker_entry_claim_are_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = tmp_path / "run"
    metadata = remote / ".phoxla"
    metadata.mkdir(parents=True)
    manifest = {
        "run_id": "run-1",
        "profile": "test-profile",
        "source": {"digest": "a" * 64},
        "config": {"digest": "b" * 64},
    }
    (metadata / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("PHOXLA_RUN_ID", "run-1")
    monkeypatch.setenv("PHOXLA_PROCESS_INDEX", "1")
    monkeypatch.setenv("PHOXLA_GCLOUD_WORKER_INDEX", "0")

    provenance = runner._manifest_provenance(remote)
    claim = runner._claim_worker_entry(remote, provenance)

    assert claim["process_index"] == 1
    assert claim["worker_index"] == 0
    stored = json.loads((remote / "logs" / "femx-fdtdx-entry.claim" / "identity.json").read_text())
    assert stored == claim
    with pytest.raises(RuntimeError, match="duplicate"):
        runner._claim_worker_entry(remote, provenance)


def test_manifest_and_claim_reject_missing_or_changed_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = tmp_path / "run"
    (remote / ".phoxla").mkdir(parents=True)
    (remote / ".phoxla" / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid deployed"):
        runner._manifest_provenance(remote)

    provenance: dict[str, object] = {
        "run_id": "run-1",
        "profile": "test-profile",
        "source_digest": "a" * 64,
        "config_digest": "b" * 64,
    }
    monkeypatch.setenv("PHOXLA_RUN_ID", "other")
    monkeypatch.setenv("PHOXLA_PROCESS_INDEX", "0")
    monkeypatch.setenv("PHOXLA_GCLOUD_WORKER_INDEX", "0")
    with pytest.raises(RuntimeError, match="disagrees"):
        runner._claim_worker_entry(remote, provenance)


def test_compiler_memory_report_is_conservative() -> None:
    analysis = SimpleNamespace(
        generated_code_size_in_bytes=10,
        argument_size_in_bytes=100,
        output_size_in_bytes=40,
        alias_size_in_bytes=20,
        temp_size_in_bytes=30,
    )
    compiled = SimpleNamespace(memory_analysis=lambda: analysis)

    report = runner._memory_report(compiled, 1000)

    assert report["compiler_peak_bytes"] == 150
    assert report["hbm_fraction"] == pytest.approx(0.15)


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
def test_compiler_memory_report_rejects_missing_or_invalid_data(compiled: Any) -> None:
    with pytest.raises(RuntimeError, match=r"memory analysis|memory statistic"):
        runner._memory_report(compiled, 1000)
