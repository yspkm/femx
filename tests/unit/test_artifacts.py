import json
from datetime import UTC, datetime

import pytest

from femx.artifacts import (
    ArtifactRef,
    ArtifactRole,
    EvidenceStatus,
    ProcessStatus,
    RunManifest,
    sha256_file,
)
from femx.core.errors import ArtifactError
from femx.core.solution import ConvergenceStatus

pytestmark = pytest.mark.unit

ZERO_SHA = "0" * 64


def test_manifest_is_canonical_and_round_trips() -> None:
    artifact = ArtifactRef(
        path="raw/result.vtu",
        role=ArtifactRole.RAW_PROVENANCE,
        media_type="application/vnd.vtk",
        sha256=ZERO_SHA,
        size_bytes=12,
    )
    manifest = RunManifest(
        run_id="run-001",
        created_at=datetime(2026, 8, 30, tzinfo=UTC),
        backend_name="elmer",
        backend_version="26.2",
        problem_digest="1" * 64,
        execution_authorized=True,
        process_status=ProcessStatus.SUCCEEDED,
        convergence_status=ConvergenceStatus.CONVERGED,
        evidence_status=EvidenceStatus.PASSED,
        artifacts=(artifact,),
        metadata={"mesh": {"cells": 12}, "tags": ["heater"]},
    )

    encoded = manifest.canonical_json()
    decoded = RunManifest.from_dict(json.loads(encoded))

    assert decoded == manifest
    assert decoded.digest() == manifest.digest()
    assert encoded.startswith('{"artifacts"')


def test_artifact_path_and_hash_are_validated() -> None:
    with pytest.raises(ArtifactError, match="safe and relative"):
        ArtifactRef("../escape", ArtifactRole.LOG, "text/plain", ZERO_SHA, 0)
    with pytest.raises(ArtifactError, match="invalid SHA"):
        ArtifactRef("run.log", ArtifactRole.LOG, "text/plain", "bad", 0)
    with pytest.raises(ArtifactError, match="media type"):
        ArtifactRef("run.log", ArtifactRole.LOG, "", ZERO_SHA, 0)
    with pytest.raises(ArtifactError, match="negative size"):
        ArtifactRef("run.log", ArtifactRole.LOG, "text/plain", ZERO_SHA, -1)


def test_sha256_file_streams_a_regular_file(tmp_path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"femx")

    assert (
        sha256_file(path, chunk_size=2)
        == "872ab1209b220b95b29791db56ce0a697bce0665fd2b42f0522c674e3b5d919b"
    )
    with pytest.raises(ValueError, match="positive"):
        sha256_file(path, chunk_size=0)
    with pytest.raises(ArtifactError, match="non-file"):
        sha256_file(tmp_path / "missing")


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"run_id": " bad "}, "run_id"),
        ({"created_at": datetime(2026, 8, 30)}, "timezone-aware"),
        ({"backend_name": ""}, "identify the backend"),
        ({"problem_digest": "bad"}, "problem_digest"),
        ({"schema_version": "femx.run/v2"}, "unsupported run schema"),
    ],
)
def test_manifest_rejects_invalid_identity_fields(
    override: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "run_id": "run-001",
        "created_at": datetime(2026, 8, 30, tzinfo=UTC),
        "backend_name": "fake",
        "backend_version": "1",
        "problem_digest": ZERO_SHA,
        "execution_authorized": False,
    }
    values.update(override)
    with pytest.raises(ArtifactError, match=message):
        RunManifest(**values)  # type: ignore[arg-type]


def test_manifest_rejects_duplicate_artifacts_and_non_json_metadata() -> None:
    artifact = ArtifactRef("raw.vtu", ArtifactRole.RAW_PROVENANCE, "model/vtu", ZERO_SHA, 0)
    with pytest.raises(ArtifactError, match="unique"):
        RunManifest(
            "run",
            datetime(2026, 8, 30, tzinfo=UTC),
            "fake",
            "1",
            ZERO_SHA,
            False,
            artifacts=(artifact, artifact),
        )

    with pytest.raises(ArtifactError, match="not JSON-compatible"):
        RunManifest(
            "run",
            datetime(2026, 8, 30, tzinfo=UTC),
            "fake",
            "1",
            ZERO_SHA,
            False,
            metadata={"bad": object()},  # type: ignore[dict-item]
        )

    with pytest.raises(ArtifactError, match="keys must be strings"):
        RunManifest(
            "run",
            datetime(2026, 8, 30, tzinfo=UTC),
            "fake",
            "1",
            ZERO_SHA,
            False,
            metadata={1: "bad"},  # type: ignore[dict-item]
        )
