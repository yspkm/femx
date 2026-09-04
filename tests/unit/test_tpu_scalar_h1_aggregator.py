from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import scripts.aggregate_tpu_scalar_h1_multilevel_evidence as multilevel_aggregator
from scripts.aggregate_tpu_scalar_h1_collective_evidence import (
    _load_process_record,
    _publish,
)

pytestmark = pytest.mark.unit


def test_multilevel_aggregator_direct_script_entrypoint_imports_from_any_cwd(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(project_root / "scripts" / "aggregate_tpu_scalar_h1_multilevel_evidence.py"),
            "--help",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "process_metrics" in completed.stdout


def test_process_record_loader_and_atomic_publication_are_fail_closed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "process.json"
    source.write_text('{"process":0}\n', encoding="utf-8")
    assert _load_process_record(source) == {"process": 0}

    output = tmp_path / "results" / "aggregate.json"
    _publish(output, '{"status":"passed"}')
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "passed"}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _publish(output, "{}")


def test_process_record_loader_rejects_duplicate_keys_and_symlinks(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"process":0,"process":1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _load_process_record(duplicate)

    target = tmp_path / "target.json"
    target.write_text('{"process":0}\n', encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="regular non-symlink"):
        _load_process_record(link)


def test_publication_lock_prevents_competing_writer(tmp_path: Path) -> None:
    output = tmp_path / "aggregate.json"
    lock = tmp_path / ".aggregate.json.publish-lock"
    lock.mkdir()
    with pytest.raises(FileExistsError, match="already locked"):
        _publish(output, "{}")
    assert not output.exists()


class _FakeMultilevelEvidence:
    def canonical_json(self) -> str:
        return '{"runtime":{"process_count":2},"status":"passed"}'

    def canonical_data(self) -> dict[str, object]:
        return {"runtime": {"process_count": 2}, "status": "passed"}

    def digest(self) -> str:
        return "a" * 64


def test_multilevel_aggregator_loads_unique_records_and_publishes_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    process_zero = tmp_path / "process-0.json"
    process_one = tmp_path / "process-1.json"
    process_zero.write_text("{}\n", encoding="utf-8")
    process_one.write_text("{}\n", encoding="utf-8")
    observed: dict[str, Any] = {}

    def load(path: Path) -> dict[str, object]:
        return {"path": path.name}

    def aggregate(records: object) -> _FakeMultilevelEvidence:
        observed["records"] = records
        return _FakeMultilevelEvidence()

    monkeypatch.setattr(multilevel_aggregator, "_load_process_record", load)
    monkeypatch.setattr(
        multilevel_aggregator,
        "aggregate_tpu_scalar_h1_multilevel_process_evidence",
        aggregate,
    )
    output = tmp_path / "results" / "aggregate.json"
    assert (
        multilevel_aggregator.main([str(process_zero), str(process_one), "--output", str(output)])
        == 0
    )
    assert observed["records"] == [
        {"path": "process-0.json"},
        {"path": "process-1.json"},
    ]
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "passed"
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "logical_sha256": "a" * 64,
        "output": str(output.resolve()),
        "process_count": 2,
        "status": "passed",
    }


def test_multilevel_aggregator_rejects_duplicate_resolved_inputs(tmp_path: Path) -> None:
    process = tmp_path / "process.json"
    process.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be unique"):
        multilevel_aggregator.main(
            [str(process), str(process), "--output", str(tmp_path / "aggregate.json")]
        )
