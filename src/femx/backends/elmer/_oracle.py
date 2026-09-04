"""Shared fail-closed mechanics for locked external-process Elmer oracles."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from femx.artifacts import sha256_file
from femx.core.errors import BackendError, BackendUnavailableError, ContractError

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SOURCE_STATES = frozenset({"not_checked", "clean", "dirty", "immutable_snapshot"})
_IDENTITY_PATTERN = re.compile(
    r"MAIN:\s+Version:\s+(\S+)\s+\(Rev:\s*([^,)]+)",
    re.IGNORECASE,
)


def file_digest(path: Path) -> str:
    """Hash one retained run artifact."""

    return sha256_file(path)


def installation_digest(path: Path, *, label: str) -> str:
    """Hash one installed oracle file with an availability-specific error."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise BackendUnavailableError(f"cannot read Elmer {label}: {path}") from error
    return digest.hexdigest()


def validate_identity_part(value: str, *, label: str) -> str:
    """Validate a version or revision embedded in a backend descriptor."""

    if not value or value.strip() != value or any(character in value for character in "\r\n"):
        raise ContractError(f"Elmer {label} must be non-empty, trimmed, and single-line")
    return value


def write_text(path: Path, content: str) -> None:
    """Write one deterministic UTF-8/LF input or captured log."""

    try:
        path.write_text(content, encoding="utf-8", newline="\n")
    except OSError as error:
        raise BackendError(f"cannot write Elmer input artifact: {path.name}") from error


def prepare_run_directory(path: Path) -> Path:
    """Create exactly one fresh, absolute, non-symlink Elmer attempt directory."""

    if not path.is_absolute():
        raise ContractError("Elmer run directory must be an absolute path")
    if path.is_symlink():
        raise BackendError("Elmer run directory cannot be a symbolic link")
    try:
        if path.exists():
            if not path.is_dir():
                raise BackendError("Elmer run directory exists but is not a directory")
            if any(path.iterdir()):
                raise BackendError("Elmer run directory must be empty to prevent overwrite")
        else:
            path.mkdir()
        (path / "mesh").mkdir()
    except OSError as error:
        raise BackendError("cannot create the Elmer run directory") from error
    return path


def parse_elmer_identity(stdout: str) -> tuple[str, str]:
    """Read the executable-reported version and revision from captured stdout."""

    match = _IDENTITY_PATTERN.search(stdout)
    if match is None:
        raise BackendError("Elmer output did not report a parseable version and revision")
    return match.group(1), match.group(2).strip()


def parse_steady_change(stdout: str, *, equation_name: str) -> tuple[int, float] | None:
    """Read the last steady relative-change record for one explicitly named equation."""

    pattern = re.compile(
        r"ComputeChange:\s+SS\s+\(ITER=(\d+)\)\s+\(NRM,RELC\):\s+"
        rf"\(\s*(\S+)\s+(\S+)\s*\)\s+::\s+{re.escape(equation_name)}",
        re.IGNORECASE,
    )
    matches = pattern.findall(stdout)
    if not matches:
        return None
    iteration, _norm, relative_change = matches[-1]
    try:
        numeric = float(relative_change.replace("D", "E").replace("d", "e"))
    except ValueError as error:
        raise BackendError("Elmer reported an invalid steady relative change") from error
    if not np.isfinite(numeric) or numeric < 0.0:
        raise BackendError("Elmer reported a non-finite or negative steady relative change")
    return int(iteration), numeric
