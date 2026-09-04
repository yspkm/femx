"""Canonical run manifest and checksum implementation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TypeAlias, cast

from femx.core.errors import ArtifactError
from femx.core.solution import ConvergenceStatus

JsonScalar: TypeAlias = bool | int | float | str | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ArtifactRole(StrEnum):
    """Authority/provenance role of one durable file."""

    CANONICAL_NUMERICAL = "canonical_numerical"
    RAW_PROVENANCE = "raw_provenance"
    DESCRIPTOR = "descriptor"
    INPUT = "input"
    LOG = "log"
    REPORT = "report"


class ProcessStatus(StrEnum):
    """External/runtime process status, separate from convergence."""

    NOT_RUN = "not_run"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class EvidenceStatus(StrEnum):
    """Scientific evidence state, separate from process and convergence."""

    NOT_EVALUATED = "not_evaluated"
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A content-addressed artifact located relative to a run root."""

    path: str
    role: ArtifactRole
    media_type: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        parsed = PurePosixPath(self.path)
        if not self.path or parsed.is_absolute() or ".." in parsed.parts:
            raise ArtifactError(f"artifact path must be safe and relative: {self.path!r}")
        if not self.media_type:
            raise ArtifactError(f"artifact {self.path!r} must declare a media type")
        if not _SHA256_PATTERN.fullmatch(self.sha256):
            raise ArtifactError(f"artifact {self.path!r} has an invalid SHA-256")
        if self.size_bytes < 0:
            raise ArtifactError(f"artifact {self.path!r} has a negative size")

    def to_dict(self) -> dict[str, JsonValue]:
        """Return JSON-compatible data in a stable schema."""

        return {
            "path": self.path,
            "role": self.role.value,
            "media_type": self.media_type,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ArtifactRef:
        """Build and validate an artifact reference from decoded JSON."""

        return cls(
            path=str(data["path"]),
            role=ArtifactRole(str(data["role"])),
            media_type=str(data["media_type"]),
            sha256=str(data["sha256"]),
            size_bytes=int(cast(int, data["size_bytes"])),
        )


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Authoritative identity and status axes for one durable run."""

    run_id: str
    created_at: datetime
    backend_name: str
    backend_version: str
    problem_digest: str
    execution_authorized: bool
    process_status: ProcessStatus = ProcessStatus.NOT_RUN
    convergence_status: ConvergenceStatus = ConvergenceStatus.NOT_EVALUATED
    evidence_status: EvidenceStatus = EvidenceStatus.NOT_EVALUATED
    artifacts: tuple[ArtifactRef, ...] = ()
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    schema_version: str = "femx.run/v1"

    def __post_init__(self) -> None:
        if not self.run_id or self.run_id.strip() != self.run_id:
            raise ArtifactError("run_id must be non-empty and trimmed")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ArtifactError("created_at must be timezone-aware")
        if not self.backend_name or not self.backend_version:
            raise ArtifactError("run manifest must identify the backend and version")
        if not _SHA256_PATTERN.fullmatch(self.problem_digest):
            raise ArtifactError("problem_digest must be a lowercase SHA-256")
        paths = tuple(artifact.path for artifact in self.artifacts)
        if len(paths) != len(set(paths)):
            raise ArtifactError("artifact paths must be unique within a run")
        if self.schema_version != "femx.run/v1":
            raise ArtifactError(f"unsupported run schema {self.schema_version!r}")
        normalized_metadata = _normalize_json(dict(self.metadata))
        if not isinstance(normalized_metadata, dict):
            raise ArtifactError("manifest metadata must be a JSON object")
        object.__setattr__(self, "metadata", MappingProxyType(normalized_metadata))

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the versioned, JSON-compatible manifest representation."""

        metadata = cast(dict[str, JsonValue], _normalize_json(dict(self.metadata)))
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat(),
            "backend": {"name": self.backend_name, "version": self.backend_version},
            "problem_digest": self.problem_digest,
            "execution_authorized": self.execution_authorized,
            "status": {
                "process": self.process_status.value,
                "convergence": self.convergence_status.value,
                "evidence": self.evidence_status.value,
            },
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "metadata": metadata,
        }

    def canonical_json(self) -> str:
        """Serialize deterministically for hashing and reproducibility checks."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def digest(self) -> str:
        """Return SHA-256 of the canonical manifest bytes."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RunManifest:
        """Decode and validate a manifest dictionary."""

        backend = cast(Mapping[str, object], data["backend"])
        status = cast(Mapping[str, object], data["status"])
        raw_artifacts = cast(list[Mapping[str, object]], data.get("artifacts", []))
        metadata = cast(Mapping[str, JsonValue], data.get("metadata", {}))
        return cls(
            schema_version=str(data["schema_version"]),
            run_id=str(data["run_id"]),
            created_at=datetime.fromisoformat(str(data["created_at"])),
            backend_name=str(backend["name"]),
            backend_version=str(backend["version"]),
            problem_digest=str(data["problem_digest"]),
            execution_authorized=bool(data["execution_authorized"]),
            process_status=ProcessStatus(str(status["process"])),
            convergence_status=ConvergenceStatus(str(status["convergence"])),
            evidence_status=EvidenceStatus(str(status["evidence"])),
            artifacts=tuple(ArtifactRef.from_dict(item) for item in raw_artifacts),
            metadata=metadata,
        )


def _normalize_json(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ArtifactError("manifest metadata keys must be strings")
            normalized[key] = _normalize_json(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    raise ArtifactError(f"value is not JSON-compatible: {type(value).__name__}")


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash one regular file without reading it all into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not path.is_file():
        raise ArtifactError(f"cannot hash a non-file artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
